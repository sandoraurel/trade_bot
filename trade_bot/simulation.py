from __future__ import annotations

import ast
import datetime as dt
import json
import math
import os
import random
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is already used elsewhere in the repo.
    np = None  # type: ignore

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is already used elsewhere in the repo.
    pd = None  # type: ignore

from .events import (
    BotEvent,
    EVENT_FILL,
    EVENT_MARKET_DATA,
    EVENT_ORDER_ACK,
    EVENT_ORDER_SUBMITTED,
    EVENT_POSITION_UPDATED,
    EVENT_RISK_HALT,
    EVENT_SIGNAL,
)
from .learning import TradeLearningEngine
from .models import Fill
from .risk import RiskManager
from .runtime import build_portfolio_snapshot
from .state import BotState, Position
from .constants import LEARNING_DB_FILE
from .state_store import PersistentLearningStore, SQLiteStateStore, backfill_learning_from_sqlite_artifacts


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for historical simulation")


def timeframe_to_minutes(timeframe: str) -> int:
    raw = str(timeframe or "").strip().lower()
    if raw.endswith("m"):
        return int(raw[:-1])
    if raw.endswith("h"):
        return int(raw[:-1]) * 60
    if raw.endswith("d"):
        return int(raw[:-1]) * 1440
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def timeframe_to_timedelta(timeframe: str) -> dt.timedelta:
    return dt.timedelta(minutes=timeframe_to_minutes(timeframe))


def timeframe_to_rule(timeframe: str) -> str:
    minutes = timeframe_to_minutes(timeframe)
    if minutes % 1440 == 0:
        return f"{minutes // 1440}D"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}min"


def _normalize_timestamp(value: Any) -> "pd.Timestamp":
    _require_pandas()
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _empty_ohlcv_frame() -> "pd.DataFrame":
    _require_pandas()
    frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame.index = pd.DatetimeIndex([], name="timestamp")
    return frame


def _as_utc(now: dt.datetime) -> dt.datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=dt.timezone.utc)
    return now.astimezone(dt.timezone.utc)


def _normalize_ohlcv_frame(frame: "pd.DataFrame") -> "pd.DataFrame":
    _require_pandas()
    working = frame.copy()
    if "timestamp" in working.columns:
        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=False)
        working.set_index("timestamp", inplace=True)
    if working.index.name != "timestamp":
        working.index = pd.to_datetime(working.index, utc=False)
        working.index.name = "timestamp"
    if getattr(working.index, "tz", None) is not None:
        working.index = working.index.tz_convert("UTC").tz_localize(None)
    needed = ["open", "high", "low", "close", "volume"]
    for column in needed:
        if column not in working.columns:
            raise ValueError(f"Historical frame missing column: {column}")
        working[column] = working[column].astype(float)
    working = working[needed]
    working = working.sort_index()
    working = working[~working.index.duplicated(keep="last")]
    return working


def resample_ohlcv(frame: "pd.DataFrame", timeframe: str) -> "pd.DataFrame":
    _require_pandas()
    normalized = _normalize_ohlcv_frame(frame)
    aggregated = normalized.resample(timeframe_to_rule(timeframe), label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    aggregated = aggregated.dropna(subset=["open", "high", "low", "close"])
    aggregated.index.name = "timestamp"
    return aggregated


def chronological_split(
    frame: "pd.DataFrame",
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    embargo_bars: int = 0,
) -> Dict[str, "pd.DataFrame"]:
    normalized = _normalize_ohlcv_frame(frame)
    total = len(normalized)
    if total == 0:
        return {"train": normalized.copy(), "validation": normalized.copy(), "test": normalized.copy()}
    train_end = min(max(int(total * train_ratio), 1), total)
    validation_end = min(max(train_end + int(total * validation_ratio), train_end), total)
    validation_start = min(train_end + max(embargo_bars, 0), total)
    test_start = min(validation_end + max(embargo_bars, 0), total)
    return {
        "train": normalized.iloc[:train_end].copy(),
        "validation": normalized.iloc[validation_start:validation_end].copy(),
        "test": normalized.iloc[test_start:].copy(),
    }


def purged_walk_forward_splits(
    frame: "pd.DataFrame",
    *,
    train_bars: int,
    validation_bars: int,
    test_bars: int,
    embargo_bars: int = 0,
    step_bars: int | None = None,
) -> list[Dict[str, Any]]:
    normalized = _normalize_ohlcv_frame(frame)
    total = len(normalized)
    if total == 0:
        return []
    train_bars = max(int(train_bars), 1)
    validation_bars = max(int(validation_bars), 0)
    test_bars = max(int(test_bars), 1)
    embargo_bars = max(int(embargo_bars), 0)
    step = max(int(step_bars or test_bars), 1)

    windows: list[Dict[str, Any]] = []
    start = 0
    while True:
        train_end = start + train_bars
        validation_start = min(train_end + embargo_bars, total)
        validation_end = min(validation_start + validation_bars, total)
        test_start = min(validation_end + embargo_bars, total)
        test_end = min(test_start + test_bars, total)
        if train_end > total or test_start >= total or test_end <= test_start:
            break
        windows.append(
            {
                "fold": len(windows),
                "train_start": start,
                "train_end": train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "test_start": test_start,
                "test_end": test_end,
                "train": normalized.iloc[start:train_end].copy(),
                "validation": normalized.iloc[validation_start:validation_end].copy(),
                "test": normalized.iloc[test_start:test_end].copy(),
            }
        )
        start += step
    return windows


@dataclass
class SimulatedOrder:
    order_id: str
    symbol: str
    side: str
    strategy: str
    requested_size: float
    remaining_size: float
    requested_price: float
    stop_loss: float
    take_profit: float
    order_type: str
    submitted_at: dt.datetime
    activate_on: dt.datetime
    activate_index: int
    expires_on_index: int
    latency_bars: int
    trace_id: str
    signal: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    filled_size: float = 0.0
    fee_paid: float = 0.0
    fill_count: int = 0
    status: str = "pending"
    position_ref: Optional[Position] = None
    activated_at: dt.datetime | None = None
    last_update_at: dt.datetime | None = None
    queue_ahead_fraction: float = 0.0
    resting_bars: int = 0
    limit_touch_count: int = 0
    stale_cancel_reason: str | None = None
    market_impact_bps: float = 0.0
    adverse_selection_bps: float = 0.0
    fill_history: list[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ClosedTrade:
    symbol: str
    strategy: str
    side: str
    entry_time: dt.datetime
    exit_time: dt.datetime
    entry_price: float
    exit_price: float
    size: float
    gross_pl: float
    pl: float
    fees: float
    holding_minutes: float
    exit_reason: str
    order_type: str
    fill_fraction: float
    latency_bars: int
    slippage_bps: float
    mfe_r: float = 0.0
    mae_r: float = 0.0
    giveback_r: float = 0.0


class HistoricalDataUniverse:
    def __init__(self, datasets: Mapping[tuple[str, str], "pd.DataFrame"], base_timeframe: str):
        _require_pandas()
        self.base_timeframe = base_timeframe
        self.datasets: Dict[tuple[str, str], pd.DataFrame] = {}
        self._timeframe_cache: Dict[str, dt.timedelta] = {}
        for key, frame in datasets.items():
            symbol, timeframe = key
            normalized = _normalize_ohlcv_frame(frame)
            normalized = normalized.copy()
            normalized["bar_end"] = normalized.index + timeframe_to_timedelta(timeframe)
            self.datasets[(symbol, timeframe)] = normalized

    def timeframes(self) -> list[str]:
        seen = {timeframe for _, timeframe in self.datasets}
        return sorted(seen, key=timeframe_to_minutes)

    def symbols(self) -> list[str]:
        return sorted({symbol for symbol, _ in self.datasets})

    def timeline(self, symbols: Sequence[str] | None = None, timeframe: str | None = None) -> list[dt.datetime]:
        timeframe = timeframe or self.base_timeframe
        selected = set(symbols or self.symbols())
        stamps: set[dt.datetime] = set()
        for symbol in selected:
            frame = self.datasets.get((symbol, timeframe))
            if frame is None or frame.empty:
                continue
            for stamp in frame["bar_end"].tolist():
                stamps.add(_normalize_timestamp(stamp).to_pydatetime())
        return sorted(stamps)

    def _resolve_frame(self, symbol: str, timeframe: str) -> "pd.DataFrame | None":
        frame = self.datasets.get((symbol, timeframe))
        if frame is not None:
            return frame
        if len(self.symbols()) == 1:
            only_symbol = self.symbols()[0]
            return self.datasets.get((only_symbol, timeframe)) or self.datasets.get((only_symbol, self.base_timeframe))
        return self.datasets.get((symbol, self.base_timeframe))

    def visible_frame(self, symbol: str, timeframe: str, now: dt.datetime) -> "pd.DataFrame":
        _require_pandas()
        frame = self._resolve_frame(symbol, timeframe)
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "bar_end"])
        boundary = pd.Timestamp(now)
        visible = frame.loc[frame["bar_end"] <= boundary]
        return visible

    def fetch_ohlcv(self, symbol: str, timeframe: str, now: dt.datetime, limit: int = 200) -> list[list[float]]:
        visible = self.visible_frame(symbol, timeframe, now)
        if visible.empty:
            return []
        tail = visible.tail(limit)
        results: list[list[float]] = []
        for ts, row in tail.iterrows():
            results.append(
                [
                    float(int(pd.Timestamp(ts).timestamp() * 1000)),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                ]
            )
        return results

    def bar_for_time(self, symbol: str, timeframe: str, now: dt.datetime) -> "pd.Series | None":
        frame = self._resolve_frame(symbol, timeframe)
        if frame is None or frame.empty:
            return None
        matches = frame.loc[frame["bar_end"] == pd.Timestamp(now)]
        if matches.empty:
            return None
        return matches.iloc[-1]


class HistoricalReplayExchange:
    """
    Sequential, no-lookahead market data adapter.

    Only bars whose end time is <= the current simulated clock are exposed.
    This prevents higher-timeframe leakage when lower-timeframe replay is active.
    """

    def __init__(
        self,
        data: HistoricalDataUniverse | "pd.DataFrame" | Mapping[tuple[str, str], "pd.DataFrame"],
        symbol: str | None = None,
        *,
        base_timeframe: str = "15m",
        config: Any | None = None,
    ) -> None:
        _require_pandas()
        if isinstance(data, HistoricalDataUniverse):
            self.universe = data
            self.symbol = symbol or (data.symbols()[0] if data.symbols() else "")
        elif isinstance(data, Mapping):
            self.universe = HistoricalDataUniverse(data, base_timeframe=base_timeframe)
            self.symbol = symbol or (self.universe.symbols()[0] if self.universe.symbols() else "")
        else:
            if symbol is None:
                raise ValueError("symbol is required when constructing exchange from a single frame")
            self.universe = HistoricalDataUniverse({(symbol, base_timeframe): data}, base_timeframe=base_timeframe)
            self.symbol = symbol
        self.base_timeframe = base_timeframe
        self.config = config
        self.current_time: dt.datetime | None = None
        self._cursor = 0
        self._timeline = self.universe.timeline([self.symbol], timeframe=self.base_timeframe)

    def set_time(self, now: dt.datetime) -> None:
        self.current_time = now

    def set_cursor(self, index: int) -> None:
        if not self._timeline:
            self.current_time = None
            self._cursor = 0
            return
        self._cursor = max(0, min(index, len(self._timeline) - 1))
        self.current_time = self._timeline[self._cursor]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> list[list[float]]:
        if self.current_time is None:
            return []
        return self.universe.fetch_ohlcv(symbol, timeframe, self.current_time, limit=limit)

    def current_bar(self, symbol: str, timeframe: str | None = None) -> "pd.Series | None":
        if self.current_time is None:
            return None
        return self.universe.bar_for_time(symbol, timeframe or self.base_timeframe, self.current_time)

    def estimate_spread_fraction(self, symbol: str) -> float:
        bar = self.current_bar(symbol, self.base_timeframe)
        base_spread = float(getattr(self.config, "backtest_spread_bps", 4.0)) / 10000.0
        if bar is None:
            return base_spread
        close = max(float(bar["close"]), 1e-9)
        bar_range = max(float(bar["high"]) - float(bar["low"]), 0.0) / close
        volatility_markup = min(bar_range * 0.08, base_spread * 4.0)
        return max(base_spread + volatility_markup, base_spread)

    def get_order_book(self, symbol: str) -> Dict[str, float]:
        bar = self.current_bar(symbol, self.base_timeframe)
        if bar is None:
            return {"bid": 100.0, "ask": 100.1}
        mid = float(bar["close"])
        half_spread = mid * self.estimate_spread_fraction(symbol) / 2.0
        return {"bid": mid - half_spread, "ask": mid + half_spread}

    def place_order(self, **_: Any) -> bool:
        return True


class SimulatedExecutionVenue:
    def __init__(self, config: Any, exchange: HistoricalReplayExchange, *, rng: random.Random):
        self.config = config
        self.exchange = exchange
        self.rng = rng
        self.base_timeframe = exchange.base_timeframe
        self.bar_duration = timeframe_to_timedelta(self.base_timeframe)
        self.open_orders: list[SimulatedOrder] = []
        self.last_fill: Fill | None = None
        self.last_execution_report: Dict[str, Any] = {}
        self.touch_escalations: int = 0
        self.touch_escalations_by_strategy: Dict[str, int] = {}

    def serialize_open_orders(self) -> list[Dict[str, Any]]:
        payloads: list[Dict[str, Any]] = []
        for order in self.open_orders:
            payload = asdict(order)
            payload["position_ref"] = None
            if order.position_ref is not None:
                payload["position_ref_id"] = str((getattr(order.position_ref, "metadata", {}) or {}).get("simulation_position_id"))
            payloads.append(payload)
        return payloads

    def restore_open_orders(self, payloads: Sequence[Dict[str, Any]], positions_by_id: Mapping[str, Position]) -> None:
        self.open_orders = []
        for payload in payloads:
            item = dict(payload)
            position_ref_id = str(item.pop("position_ref_id", "") or "")
            item.pop("position_ref", None)
            for key in ("submitted_at", "activate_on", "activated_at", "last_update_at"):
                if item.get(key):
                    item[key] = dt.datetime.fromisoformat(str(item[key]))
            order = SimulatedOrder(**item)
            if position_ref_id:
                order.position_ref = positions_by_id.get(position_ref_id)
            self.open_orders.append(order)

    def submit_order(
        self,
        *,
        signal: Dict[str, Any],
        size: float,
        now: dt.datetime,
        current_index: int,
        trace_id: str,
    ) -> SimulatedOrder:
        order_type = self._order_type_for_signal(signal)
        requested_price = self._requested_price_for_signal(signal, order_type=order_type)
        queue_ahead_fraction = self.rng.uniform(0.12, 0.92) if order_type == "limit" else 0.0
        signal_metadata = dict(signal.get("metadata", {}) or {})
        urgent_limit = order_type == "limit" and bool(signal_metadata.get("urgent_limit_execution", False))
        if order_type == "limit":
            strategy = str(signal.get("strategy", "unknown") or "unknown")
            signal_quality = float(signal.get("signal_quality", 0.0) or 0.0)
            expected_edge_bps = float(signal.get("expected_edge_bps", 0.0) or 0.0)
            if urgent_limit:
                queue_ahead_fraction = min(
                    queue_ahead_fraction,
                    max(float(getattr(self.config, "simulation_execution_urgent_limit_queue_cap", 0.18) or 0.18), 0.0),
                )
                signal_metadata["limit_queue_priority_assist"] = True
            if strategy in {"trend_pullback", "trend_breakout"}:
                quality_floor = float(
                    getattr(
                        self.config,
                        "simulation_breakout_limit_queue_priority_quality_floor" if strategy == "trend_breakout" else "simulation_limit_queue_priority_quality_floor",
                        0.84 if strategy == "trend_breakout" else 0.80,
                    )
                )
                edge_floor = float(
                    getattr(
                        self.config,
                        "simulation_breakout_limit_queue_priority_edge_floor_bps" if strategy == "trend_breakout" else "simulation_limit_queue_priority_edge_floor_bps",
                        30.0 if strategy == "trend_breakout" else 24.0,
                    )
                )
                if signal_quality >= quality_floor and expected_edge_bps >= edge_floor:
                    queue_ahead_fraction = min(queue_ahead_fraction, 0.28)
                    signal_metadata["limit_queue_priority_assist"] = True
                    signal["metadata"] = signal_metadata
            signal["metadata"] = signal_metadata
        latency_floor = max(int(getattr(self.config, "backtest_latency_bars", 1)), 1)
        latency_jitter = max(int(getattr(self.config, "simulation_latency_jitter_bars", 0)), 0)
        latency_bars = max(latency_floor + (self.rng.randint(0, latency_jitter) if latency_jitter else 0), 1)
        if order_type == "limit":
            strategy = str(signal.get("strategy", "unknown") or "unknown")
            signal_quality = float(signal.get("signal_quality", 0.0) or 0.0)
            expected_edge_bps = float(signal.get("expected_edge_bps", 0.0) or 0.0)
            if urgent_limit:
                latency_bars = min(
                    latency_bars,
                    max(int(getattr(self.config, "simulation_execution_urgent_limit_latency_bars", 1) or 1), 1),
                )
                signal_metadata["limit_latency_reduced"] = True
                signal["metadata"] = signal_metadata
            if strategy in {"trend_pullback", "trend_breakout"} and latency_bars > 1:
                quality_floor = float(
                    getattr(
                        self.config,
                        "simulation_breakout_limit_latency_reduction_quality_floor" if strategy == "trend_breakout" else "simulation_limit_latency_reduction_quality_floor",
                        0.85 if strategy == "trend_breakout" else 0.82,
                    )
                )
                edge_floor = float(
                    getattr(
                        self.config,
                        "simulation_breakout_limit_latency_reduction_edge_floor_bps" if strategy == "trend_breakout" else "simulation_limit_latency_reduction_edge_floor_bps",
                        30.0 if strategy == "trend_breakout" else 26.0,
                    )
                )
                if signal_quality >= quality_floor and expected_edge_bps >= edge_floor:
                    latency_bars = max(latency_bars - 1, 1)
                    signal_metadata = dict(signal.get("metadata", {}) or {})
                    signal_metadata["limit_latency_reduced"] = True
                    signal["metadata"] = signal_metadata
        activate_index = current_index + latency_bars
        expires_on_index = activate_index
        if order_type == "limit":
            signal_expiry = self._limit_expiry_bars_for_signal(signal)
            expires_on_index = activate_index + max(signal_expiry, 1)
        order = SimulatedOrder(
            order_id=uuid.uuid4().hex,
            symbol=str(signal.get("symbol", "")),
            side=str(signal.get("side", "long")).lower(),
            strategy=str(signal.get("strategy", "unknown")),
            requested_size=float(size),
            remaining_size=float(size),
            requested_price=requested_price,
            stop_loss=float(signal.get("stop_loss", 0.0) or 0.0),
            take_profit=float(signal.get("take_profit", 0.0) or 0.0),
            order_type=order_type,
            submitted_at=now,
            activate_on=now + (self.bar_duration * latency_bars),
            activate_index=activate_index,
            expires_on_index=expires_on_index,
            latency_bars=latency_bars,
            trace_id=trace_id,
            signal=dict(signal),
            metadata={
                "original_preferred_order_type": str(dict(signal.get("metadata", {}) or {}).get("preferred_order_type", "")),
                "urgent_limit_execution": urgent_limit,
                "execution_urgency_score": float(dict(signal.get("metadata", {}) or {}).get("execution_urgency_score", 0.0) or 0.0),
                "limit_to_market_upgrade": bool(
                    order_type == "market"
                    and str(dict(signal.get("metadata", {}) or {}).get("preferred_order_type", "")).lower() == "limit"
                ),
            },
            queue_ahead_fraction=queue_ahead_fraction,
        )
        self.open_orders.append(order)
        self.last_execution_report = {
            "status": "accepted",
            "order_id": order.order_id,
            "order_type": order_type,
            "latency_bars": latency_bars,
            "requested_size": size,
            "requested_price": order.requested_price,
            "queue_ahead_fraction": order.queue_ahead_fraction,
        }
        return order

    def _order_type_for_signal(self, signal: Dict[str, Any]) -> str:
        metadata = dict(signal.get("metadata", {}) or {})
        preferred = str(metadata.get("preferred_order_type", "")).lower()
        urgency = self._execution_urgency_score(signal)
        metadata["execution_urgency_score"] = urgency
        signal["metadata"] = metadata
        strategy = str(signal.get("strategy", metadata.get("strategy", "unknown")) or "unknown")
        pullback_market_enabled = strategy != "trend_pullback" or bool(getattr(self.config, "simulation_pullback_aggressive_market_enabled", False))
        spread_bps = float(metadata.get("spread_bps", 0.0) or 0.0)
        spread_guard = float(getattr(self.config, "simulation_execution_urgency_spread_guard_bps", 18.0) or 18.0)
        pullback_shape_ok = strategy != "trend_pullback" or self._pullback_aggressive_execution_shape_ok(metadata)
        if pullback_market_enabled and pullback_shape_ok and urgency >= float(getattr(self.config, "simulation_execution_urgency_market_threshold", 0.84) or 0.84) and spread_bps <= spread_guard:
            return "market"
        if preferred == "market":
            return preferred
        if preferred == "limit" and self._should_upgrade_limit_to_market(signal):
            return "market"
        if preferred == "limit":
            if self._should_use_urgent_limit(signal):
                metadata = dict(signal.get("metadata", {}) or {})
                metadata["urgent_limit_execution"] = True
                signal["metadata"] = metadata
            return preferred
        if bool(metadata.get("force_limit_entry", False)):
            return "limit"
        return "market" if bool(signal.get("fast_move", False)) else "limit"

    def _should_use_urgent_limit(self, signal: Dict[str, Any]) -> bool:
        metadata = dict(signal.get("metadata", {}) or {})
        if bool(metadata.get("force_limit_entry", False)):
            return False
        strategy = str(signal.get("strategy", metadata.get("strategy", "unknown")) or "unknown")
        if strategy not in {"trend_pullback", "trend_breakout", "mean_reversion"}:
            return False
        urgency = float(metadata.get("execution_urgency_score", self._execution_urgency_score(signal)) or 0.0)
        if urgency < float(getattr(self.config, "simulation_execution_urgent_limit_threshold", 0.72) or 0.72):
            return False
        spread_bps = float(metadata.get("spread_bps", 0.0) or 0.0)
        if spread_bps > float(getattr(self.config, "simulation_execution_urgency_spread_guard_bps", 18.0) or 18.0):
            return False
        if strategy == "trend_pullback" and not self._pullback_aggressive_execution_shape_ok(metadata):
            return False
        return True

    def _should_upgrade_limit_to_market(self, signal: Dict[str, Any]) -> bool:
        metadata = dict(signal.get("metadata", {}) or {})
        strategy = str(signal.get("strategy", metadata.get("strategy", "unknown")) or "unknown")
        if strategy not in {"trend_pullback", "mean_reversion", "trend_breakout"}:
            return False
        if strategy == "trend_pullback" and not bool(getattr(self.config, "simulation_pullback_aggressive_market_enabled", False)):
            return False
        if bool(metadata.get("force_limit_entry", False)):
            return False
        if float(metadata.get("execution_urgency_score", self._execution_urgency_score(signal)) or 0.0) >= float(
            getattr(self.config, "simulation_execution_urgency_market_threshold", 0.84) or 0.84
        ):
            spread_bps = float(metadata.get("spread_bps", 0.0) or 0.0)
            if spread_bps <= float(getattr(self.config, "simulation_execution_urgency_spread_guard_bps", 18.0) or 18.0):
                return True
        signal_quality = float(signal.get("signal_quality", 0.0) or 0.0)
        expected_edge_bps = float(signal.get("expected_edge_bps", 0.0) or 0.0)
        rr_ratio = float(signal.get("rr_ratio", 0.0) or 0.0)
        quality_floor = float(
            getattr(
                self.config,
                "simulation_breakout_aggressive_entry_quality_floor" if strategy == "trend_breakout" else "simulation_aggressive_entry_quality_floor",
                0.80 if strategy == "trend_breakout" else 0.72,
            )
        )
        edge_floor = float(
            getattr(
                self.config,
                "simulation_breakout_aggressive_entry_edge_floor_bps" if strategy == "trend_breakout" else "simulation_aggressive_entry_edge_floor_bps",
                26.0 if strategy == "trend_breakout" else 18.0,
            )
        )
        rr_floor = float(
            getattr(
                self.config,
                "simulation_breakout_aggressive_entry_rr_floor" if strategy == "trend_breakout" else "simulation_aggressive_entry_rr_floor",
                1.70 if strategy == "trend_breakout" else 1.45,
            )
        )
        max_distance_bps = float(
            getattr(
                self.config,
                "simulation_breakout_aggressive_entry_max_distance_bps" if strategy == "trend_breakout" else "simulation_aggressive_entry_max_distance_bps",
                8.0 if strategy == "trend_breakout" else 14.0,
            )
        )
        if signal_quality < quality_floor:
            return False
        if expected_edge_bps < edge_floor:
            return False
        if rr_ratio < rr_floor:
            return False
        entry_price = float(signal.get("entry_price", 0.0) or 0.0)
        mid_price = float(metadata.get("mid_price", entry_price) or entry_price)
        if entry_price <= 0.0 or mid_price <= 0.0:
            return False
        distance_bps = abs(entry_price - mid_price) / mid_price * 10000.0
        if distance_bps > max_distance_bps:
            return False
        if strategy == "trend_pullback":
            if not self._pullback_aggressive_execution_shape_ok(metadata):
                return False
        liquidity_score = float(metadata.get("liquidity_score", 0.0) or 0.0)
        if strategy == "mean_reversion" and liquidity_score < 0.55:
            return False
        if strategy == "trend_breakout" and liquidity_score < 0.68:
            return False
        rotation_policy = dict(metadata.get("rotation_policy", {}) or {})
        suppressed_family = str(metadata.get("suppressed_family", rotation_policy.get("suppressed_family", "")) or "")
        if strategy == suppressed_family:
            return False
        return True

    def _pullback_aggressive_execution_shape_ok(self, metadata: Dict[str, Any]) -> bool:
        close_location = float(metadata.get("entry_close_location", 0.0) or 0.0)
        body_fraction = float(metadata.get("entry_body_fraction", 0.0) or 0.0)
        min_close_location = float(getattr(self.config, "simulation_pullback_aggressive_min_close_location", 0.52) or 0.52)
        min_body_fraction = float(getattr(self.config, "simulation_pullback_aggressive_min_body_fraction", 0.18) or 0.18)
        return close_location >= min_close_location and body_fraction >= min_body_fraction

    def _requested_price_for_signal(self, signal: Dict[str, Any], *, order_type: str) -> float:
        entry_price = float(signal.get("entry_price", 0.0) or 0.0)
        if order_type != "limit" or entry_price <= 0.0:
            return entry_price
        metadata = dict(signal.get("metadata", {}) or {})
        strategy = str(signal.get("strategy", metadata.get("strategy", "unknown")) or "unknown")
        side = str(signal.get("side", "long")).lower()
        stop_loss = float(signal.get("stop_loss", 0.0) or 0.0)
        offset_bps = 0.0
        if strategy == "trend_pullback":
            offset_bps = float(getattr(self.config, "simulation_pullback_limit_offset_bps", 4.0))
        elif strategy == "mean_reversion":
            offset_bps = float(getattr(self.config, "simulation_mean_reversion_limit_offset_bps", 7.0))
        signal_quality = float(signal.get("signal_quality", 0.0) or 0.0)
        expected_edge_bps = float(signal.get("expected_edge_bps", 0.0) or 0.0)
        if (
            offset_bps > 0.0
            and signal_quality >= float(getattr(self.config, "simulation_high_quality_limit_quality_floor", 0.76))
            and expected_edge_bps >= float(getattr(self.config, "simulation_high_quality_limit_edge_floor_bps", 22.0))
        ):
            offset_bps = max(
                offset_bps - float(getattr(self.config, "simulation_high_quality_limit_offset_tightening_bps", 2.0) or 2.0),
                0.5,
            )
        urgency = float(metadata.get("execution_urgency_score", self._execution_urgency_score(signal)) or 0.0)
        if urgency >= float(getattr(self.config, "simulation_execution_urgency_marketable_limit_threshold", 0.68) or 0.68):
            offset_bps *= max(0.20, 1.0 - urgency)
        if bool(metadata.get("urgent_limit_execution", False)):
            offset_bps *= max(float(getattr(self.config, "simulation_execution_urgent_limit_offset_multiplier", 0.25) or 0.25), 0.0)
        if offset_bps <= 0.0:
            return entry_price
        raw_offset = entry_price * (offset_bps / 10000.0)
        stop_distance = abs(entry_price - stop_loss)
        cap_fraction = float(getattr(self.config, "simulation_limit_offset_stop_distance_cap_fraction", 0.22))
        if stop_distance > 0.0 and cap_fraction > 0.0:
            raw_offset = min(raw_offset, stop_distance * cap_fraction)
        if raw_offset <= 0.0:
            return entry_price
        if side in {"long", "buy"}:
            return max(entry_price - raw_offset, 1e-9)
        if side in {"short", "sell"}:
            return max(entry_price + raw_offset, 1e-9)
        return entry_price

    def _execution_urgency_score(self, signal: Dict[str, Any]) -> float:
        metadata = dict(signal.get("metadata", {}) or {})
        strategy = str(signal.get("strategy", "unknown") or "unknown")
        quality = float(signal.get("signal_quality", 0.0) or 0.0)
        edge = float(signal.get("expected_edge_bps", 0.0) or 0.0)
        liquidity_score = float(metadata.get("liquidity_score", 0.0) or 0.0)
        spread_bps = float(metadata.get("spread_bps", 0.0) or 0.0)
        fast_move = bool(signal.get("fast_move", False))
        rotation_policy = dict(metadata.get("rotation_policy", {}) or {})
        preferred_family = str(metadata.get("preferred_family", rotation_policy.get("preferred_family", "")) or "")
        score = (quality * 0.48) + min(edge / 40.0, 1.0) * 0.22 + min(liquidity_score, 1.0) * 0.18
        if fast_move:
            score += 0.08
        if strategy == "trend_breakout":
            score += float(getattr(self.config, "simulation_execution_urgency_breakout_bonus", 0.08) or 0.08)
        if strategy == "trend_pullback":
            score -= float(getattr(self.config, "simulation_execution_urgency_pullback_penalty", 0.04) or 0.04)
        if strategy == preferred_family:
            score += 0.05
        if spread_bps > float(getattr(self.config, "simulation_execution_urgency_spread_guard_bps", 18.0) or 18.0):
            score -= 0.12
        return max(0.0, min(score, 1.0))

    def _limit_expiry_bars_for_signal(self, signal: Dict[str, Any]) -> int:
        metadata = dict(signal.get("metadata", {}) or {})
        base_expiry = int((metadata.get("order_expiry_bars", 0)) or 0)
        expiry = max(base_expiry or int(getattr(self.config, "simulation_limit_order_expiry_bars", 4)), 1)
        strategy = str(signal.get("strategy", metadata.get("strategy", "unknown")) or "unknown")
        signal_quality = float(signal.get("signal_quality", 0.0) or 0.0)
        quality_floor = float(getattr(self.config, "simulation_expiry_bonus_quality_floor", 0.70))
        if signal_quality < quality_floor:
            return expiry
        if strategy == "trend_pullback":
            expiry += max(int(getattr(self.config, "simulation_pullback_limit_expiry_bonus_bars", 2) or 0), 0)
        elif strategy == "trend_breakout":
            expiry += max(int(getattr(self.config, "simulation_breakout_limit_expiry_bonus_bars", 1) or 0), 0)
        elif strategy == "mean_reversion":
            expiry += max(int(getattr(self.config, "simulation_mean_reversion_limit_expiry_bonus_bars", 1) or 0), 0)
        return expiry

    def process_bar(
        self,
        *,
        now: dt.datetime,
        current_index: int,
        bar_by_symbol: Mapping[str, "pd.Series"],
    ) -> Dict[str, list[Any]]:
        fills: list[tuple[SimulatedOrder, Fill, Dict[str, Any]]] = []
        expired: list[SimulatedOrder] = []
        cancelled: list[SimulatedOrder] = []
        completed: list[SimulatedOrder] = []
        repriced_orders: list[SimulatedOrder] = []
        touch_escalated_orders: list[SimulatedOrder] = []

        for order in list(self.open_orders):
            if order.status not in {"pending", "active", "partially_filled"}:
                continue
            if current_index < order.activate_index:
                continue
            bar = bar_by_symbol.get(order.symbol)
            if bar is None:
                continue
            if order.activated_at is None:
                order.activated_at = now
                order.status = "active"
            order.last_update_at = now
            order.resting_bars += 1
            repriced = self._maybe_reprice_stale_order(order, bar, current_index=current_index)
            if repriced:
                repriced_orders.append(order)
                self.last_execution_report = {
                    "status": "repriced",
                    "order_id": order.order_id,
                    "requested_price": order.requested_price,
                    "expires_on_index": order.expires_on_index,
                    "reprices": int(order.metadata.get("stale_reprices", 0) or 0),
                }
            stale_escalated = self._maybe_stale_escalate_order(order, bar)
            if stale_escalated:
                self._stale_market_escalations += 1
                self._increment_counter(self._stale_market_escalations_by_strategy, str(order.strategy or "unknown"))
                self.last_execution_report = {
                    "status": "stale_escalated",
                    "order_id": order.order_id,
                    "strategy": str(order.strategy or "unknown"),
                    "resting_bars": order.resting_bars,
                }
            cancelled_reason = self._maybe_cancel_stale_order(order, bar)
            if cancelled_reason is not None:
                order.status = "cancelled"
                order.stale_cancel_reason = cancelled_reason
                cancelled.append(order)
                completed.append(order)
                self.last_execution_report = {
                    "status": "cancelled",
                    "order_id": order.order_id,
                    "reason": cancelled_reason,
                    "resting_bars": order.resting_bars,
                }
                continue
            filled, fill_details = self._maybe_fill(order, bar, now)
            if filled is not None:
                order.remaining_size = max(order.remaining_size - filled.size, 0.0)
                order.filled_size += filled.size
                order.fee_paid += filled.fee
                order.fill_count += 1
                order.fill_history.append({"filled_at": now.isoformat(), "size": filled.size, "price": filled.price, **fill_details})
                if order.remaining_size <= 1e-9 or order.order_type == "market":
                    order.status = "filled"
                    completed.append(order)
                else:
                    order.status = "partially_filled"
                fills.append((order, filled, fill_details))
                self.last_fill = filled
                self.last_execution_report = {
                    "status": "filled" if order.status == "filled" else "partial_fill",
                    "order_id": order.order_id,
                    "latency_bars": order.latency_bars,
                    "fill": asdict(filled),
                    "fill_details": fill_details,
                }
            if order.status in {"active", "partially_filled", "pending"} and current_index >= order.expires_on_index:
                order.status = "expired"
                expired.append(order)
                self.last_execution_report = {
                    "status": "expired",
                    "order_id": order.order_id,
                    "latency_bars": order.latency_bars,
                    "filled_size": order.filled_size,
                    "remaining_size": order.remaining_size,
                }
                completed.append(order)

        for order in completed:
            if order in self.open_orders:
                self.open_orders.remove(order)
        return {"fills": fills, "expired": expired, "cancelled": cancelled, "repriced": repriced_orders, "touch_escalated": touch_escalated_orders}

    def _maybe_reprice_stale_order(
        self,
        order: SimulatedOrder,
        bar: "pd.Series",
        *,
        current_index: int,
    ) -> bool:
        if order.order_type != "limit":
            return False
        strategy = str(order.strategy or "")
        cancel_after = max(int(getattr(self.config, "simulation_stale_order_cancel_bars", 0)), 0)
        signal_metadata = dict((order.signal or {}).get("metadata", {}) or {})
        signal_quality = float((order.signal or {}).get("signal_quality", 0.0) or 0.0)
        expected_edge_bps = float((order.signal or {}).get("expected_edge_bps", 0.0) or 0.0)
        early_delta = max(int(getattr(self.config, "simulation_stale_order_reprice_early_bars_delta", 1) or 1), 0)
        early_trigger = max(cancel_after - early_delta, 1)
        qualifies_early = (
            signal_quality >= float(getattr(self.config, "simulation_stale_order_reprice_early_quality_floor", 0.76))
            and expected_edge_bps >= float(getattr(self.config, "simulation_stale_order_reprice_early_edge_floor_bps", 22.0))
        )
        reprice_threshold = early_trigger if qualifies_early else cancel_after
        if cancel_after <= 0 or order.resting_bars < reprice_threshold:
            return False
        max_attempts = max(int(getattr(self.config, "simulation_stale_order_reprice_max_attempts", 1) or 1), 0)
        if strategy in {"trend_pullback", "trend_breakout"}:
            extra_quality_floor = float(
                getattr(
                    self.config,
                    "simulation_breakout_stale_order_extra_reprice_quality_floor" if strategy == "trend_breakout" else "simulation_stale_order_extra_reprice_quality_floor",
                    0.85 if strategy == "trend_breakout" else 0.82,
                )
            )
            extra_edge_floor = float(
                getattr(
                    self.config,
                    "simulation_breakout_stale_order_extra_reprice_edge_floor_bps" if strategy == "trend_breakout" else "simulation_stale_order_extra_reprice_edge_floor_bps",
                    30.0 if strategy == "trend_breakout" else 26.0,
                )
            )
            if signal_quality >= extra_quality_floor and expected_edge_bps >= extra_edge_floor:
                max_attempts += 1
        reprice_attempts = int(order.metadata.get("stale_reprices", 0) or 0)
        if reprice_attempts >= max_attempts:
            return False
        if signal_quality < float(getattr(self.config, "simulation_stale_order_reprice_quality_floor", 0.68)):
            return False
        if expected_edge_bps < float(getattr(self.config, "simulation_stale_order_reprice_edge_floor_bps", 16.0)):
            return False
        close = float(bar["close"])
        price = float(order.requested_price)
        if close <= 0 or price <= 0:
            return False
        distance_bps = abs(close - price) / price * 10000.0
        cancel_distance_bps = max(
            float(signal_metadata.get("stale_cancel_distance_bps", getattr(self.config, "simulation_stale_order_cancel_distance_bps", 18.0))),
            1.0,
        )
        if distance_bps < cancel_distance_bps:
            return False
        offset_bps = max(float(getattr(self.config, "simulation_stale_order_reprice_offset_bps", 6.0) or 6.0), 0.5)
        if order.side in {"long", "buy"}:
            new_price = close * (1.0 - (offset_bps / 10000.0))
            if new_price <= order.requested_price:
                return False
        else:
            new_price = close * (1.0 + (offset_bps / 10000.0))
            if new_price >= order.requested_price:
                return False
        order.requested_price = float(new_price)
        order.queue_ahead_fraction = min(float(order.queue_ahead_fraction), 0.18)
        order.resting_bars = max(cancel_after - 1, 0)
        order.metadata["stale_reprices"] = reprice_attempts + 1
        extension = max(int(getattr(self.config, "simulation_stale_order_reprice_extension_bars", 2) or 2), 0)
        order.expires_on_index = max(order.expires_on_index, current_index + extension)
        return True

    def _maybe_cancel_stale_order(self, order: SimulatedOrder, bar: "pd.Series") -> str | None:
        if order.order_type != "limit":
            return None
        cancel_after = max(int(getattr(self.config, "simulation_stale_order_cancel_bars", 0)), 0)
        if cancel_after <= 0 or order.resting_bars < cancel_after:
            return None
        close = float(bar["close"])
        price = float(order.requested_price)
        if close <= 0 or price <= 0:
            return None
        distance_bps = abs(close - price) / price * 10000.0
        signal_metadata = dict((order.signal or {}).get("metadata", {}) or {})
        cancel_distance_bps = max(
            float(signal_metadata.get("stale_cancel_distance_bps", getattr(self.config, "simulation_stale_order_cancel_distance_bps", 18.0))),
            1.0,
        )
        if distance_bps >= cancel_distance_bps:
            return "stale_limit"
        return None

    def _maybe_stale_escalate_order(self, order: SimulatedOrder, bar: "pd.Series") -> bool:
        if order.order_type != "limit":
            return False
        strategy = str(order.strategy or "")
        if strategy not in {"trend_pullback", "trend_breakout"}:
            return False
        cancel_after = max(int(getattr(self.config, "simulation_stale_order_cancel_bars", 0)), 0)
        if cancel_after <= 0 or order.resting_bars < cancel_after:
            return False
        reprice_attempts = int(order.metadata.get("stale_reprices", 0) or 0)
        max_attempts = max(int(getattr(self.config, "simulation_stale_order_reprice_max_attempts", 1) or 1), 0)
        if reprice_attempts < max_attempts:
            return False
        close = float(bar["close"])
        price = float(order.requested_price)
        if close <= 0 or price <= 0:
            return False
        signal_metadata = dict((order.signal or {}).get("metadata", {}) or {})
        cancel_distance_bps = max(
            float(signal_metadata.get("stale_cancel_distance_bps", getattr(self.config, "simulation_stale_order_cancel_distance_bps", 18.0))),
            1.0,
        )
        distance_bps = abs(close - price) / price * 10000.0
        if distance_bps < cancel_distance_bps:
            return False
        signal_quality = float((order.signal or {}).get("signal_quality", 0.0) or 0.0)
        expected_edge_bps = float((order.signal or {}).get("expected_edge_bps", 0.0) or 0.0)
        quality_floor = float(
            getattr(
                self.config,
                "simulation_breakout_stale_market_escalation_quality_floor" if strategy == "trend_breakout" else "simulation_stale_market_escalation_quality_floor",
                0.84 if strategy == "trend_breakout" else 0.80,
            )
        )
        edge_floor = float(
            getattr(
                self.config,
                "simulation_breakout_stale_market_escalation_edge_floor_bps" if strategy == "trend_breakout" else "simulation_stale_market_escalation_edge_floor_bps",
                30.0 if strategy == "trend_breakout" else 24.0,
            )
        )
        if signal_quality < quality_floor:
            return False
        if expected_edge_bps < edge_floor:
            return False
        order.order_type = "market"
        order.queue_ahead_fraction = 0.0
        order.metadata["stale_escalated_to_market"] = True
        return True

    def _maybe_touch_escalate_order(self, order: SimulatedOrder) -> bool:
        if order.order_type != "limit":
            return False
        strategy = str(order.strategy or "")
        min_touches = max(
            int(
                getattr(
                    self.config,
                    "simulation_breakout_touch_escalation_min_touches" if strategy == "trend_breakout" else "simulation_touch_escalation_min_touches",
                    1 if strategy == "trend_breakout" else 2,
                )
                or (1 if strategy == "trend_breakout" else 2)
            ),
            1,
        )
        if int(order.limit_touch_count or 0) < min_touches:
            return False
        signal_quality = float((order.signal or {}).get("signal_quality", 0.0) or 0.0)
        expected_edge_bps = float((order.signal or {}).get("expected_edge_bps", 0.0) or 0.0)
        quality_floor = float(
            getattr(
                self.config,
                "simulation_breakout_touch_escalation_quality_floor" if strategy == "trend_breakout" else "simulation_touch_escalation_quality_floor",
                0.82 if strategy == "trend_breakout" else 0.74,
            )
        )
        edge_floor = float(
            getattr(
                self.config,
                "simulation_breakout_touch_escalation_edge_floor_bps" if strategy == "trend_breakout" else "simulation_touch_escalation_edge_floor_bps",
                28.0 if strategy == "trend_breakout" else 20.0,
            )
        )
        if signal_quality < quality_floor:
            return False
        if expected_edge_bps < edge_floor:
            return False
        if strategy not in {"trend_pullback", "mean_reversion", "trend_breakout"}:
            return False
        order.order_type = "market"
        order.queue_ahead_fraction = 0.0
        order.metadata["touch_escalated_to_market"] = True
        self.touch_escalations += 1
        self.touch_escalations_by_strategy[strategy] = int(self.touch_escalations_by_strategy.get(strategy, 0)) + 1
        self.last_execution_report = {
            "status": "touch_escalated",
            "order_id": order.order_id,
            "strategy": strategy,
            "limit_touch_count": int(order.limit_touch_count or 0),
        }
        return True

    def _maybe_fill(
        self,
        order: SimulatedOrder,
        bar: "pd.Series",
        now: dt.datetime,
    ) -> tuple[Fill | None, Dict[str, Any]]:
        open_price = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        volume = max(float(bar["volume"]), 0.0)
        spread_fraction = self.exchange.estimate_spread_fraction(order.symbol)
        fee_rate = float(getattr(self.config, "backtest_fee_bps", 10.0)) / 10000.0
        side = "long" if order.side in {"long", "buy"} else "short"

        raw_fill_price: float | None = None
        passive_fill = False
        adverse_selection_bps = 0.0
        queue_before = order.queue_ahead_fraction
        if order.order_type == "market":
            raw_fill_price = open_price
        elif side == "long":
            if open_price <= order.requested_price:
                raw_fill_price = open_price
            elif low <= order.requested_price <= high:
                queue_progress = self._queue_progress(order, bar, close_through=close < order.requested_price)
                order.limit_touch_count += 1
                if queue_progress <= 0:
                    self._maybe_touch_escalate_order(order)
                    order.queue_ahead_fraction = max(order.queue_ahead_fraction * 0.65, 0.0)
                    return None, {"passive_fill": True, "queue_ahead_fraction_start": queue_before, "queue_ahead_fraction_end": order.queue_ahead_fraction}
                raw_fill_price = order.requested_price
                passive_fill = True
                adverse_selection_bps = max((order.requested_price - close) / max(order.requested_price, 1e-9) * 10000.0, 0.0) * 0.10
                order.queue_ahead_fraction = max(order.queue_ahead_fraction - queue_progress, 0.0)
        else:
            if open_price >= order.requested_price:
                raw_fill_price = open_price
            elif low <= order.requested_price <= high:
                queue_progress = self._queue_progress(order, bar, close_through=close > order.requested_price)
                order.limit_touch_count += 1
                if queue_progress <= 0:
                    self._maybe_touch_escalate_order(order)
                    order.queue_ahead_fraction = max(order.queue_ahead_fraction * 0.65, 0.0)
                    return None, {"passive_fill": True, "queue_ahead_fraction_start": queue_before, "queue_ahead_fraction_end": order.queue_ahead_fraction}
                raw_fill_price = order.requested_price
                passive_fill = True
                adverse_selection_bps = max((close - order.requested_price) / max(order.requested_price, 1e-9) * 10000.0, 0.0) * 0.10
                order.queue_ahead_fraction = max(order.queue_ahead_fraction - queue_progress, 0.0)

        if raw_fill_price is None:
            return None, {}

        participation_cap = max(float(getattr(self.config, "simulation_volume_participation_rate", 0.20)), 0.01)
        liquidity_fraction = self._liquidity_fraction(bar, order, passive_fill=passive_fill)
        max_fill_size = max(volume * participation_cap * liquidity_fraction, order.requested_size * 0.1)
        min_fraction = min(max(float(getattr(self.config, "simulated_partial_fill_min_fraction", 0.65)), 0.1), 1.0)
        fill_target = order.remaining_size
        if order.order_type == "limit":
            fill_target *= self.rng.uniform(min_fraction, 1.0)
        actual_size = min(fill_target, max_fill_size, order.remaining_size)
        if actual_size <= 1e-9:
            return None, {}

        bar_range_fraction = max(high - low, 0.0) / max(close, 1e-9)
        size_ratio = actual_size / max(volume, actual_size, 1e-9)
        base_slippage = float(getattr(self.config, "backtest_slippage_bps", 5.0)) / 10000.0
        volatility_weight = float(getattr(self.config, "simulation_slippage_volatility_weight", 0.12))
        impact_weight = float(getattr(self.config, "simulation_volume_impact_weight", 0.10))
        impact_exponent = float(getattr(self.config, "simulation_market_impact_exponent", 0.5))
        impact_weight_extra = float(getattr(self.config, "simulation_market_impact_weight", 0.08))
        market_impact = (size_ratio ** max(impact_exponent, 0.1)) * impact_weight_extra
        slippage = base_slippage + (bar_range_fraction * volatility_weight) + (size_ratio * impact_weight) + market_impact + (adverse_selection_bps / 10000.0)
        if passive_fill:
            # A resting limit order should never receive a worse price than its
            # own limit. Partial fills and queueing already model realism here.
            slippage = 0.0
            market_impact = 0.0
            effective_spread = 0.0
            adjusted_price = raw_fill_price
        else:
            effective_spread = spread_fraction / 2.0
            direction = 1.0 if side == "long" else -1.0
            adjusted_price = raw_fill_price * (1.0 + (direction * (effective_spread + slippage)))
        fee = abs(adjusted_price * actual_size * fee_rate)
        order.market_impact_bps = market_impact * 10000.0
        order.adverse_selection_bps = adverse_selection_bps
        fill = Fill(
            symbol=order.symbol,
            side=side,
            size=actual_size,
            price=adjusted_price,
            fee=fee,
            filled_at=now,
            order_id=order.order_id,
            metadata={
                "latency_ms": int(order.latency_bars * timeframe_to_minutes(self.base_timeframe) * 60 * 1000),
                "fill_fraction": actual_size / max(order.requested_size, 1e-9),
                "spread_bps": spread_fraction * 10000.0,
                "slippage_bps": slippage * 10000.0,
                "market_impact_bps": market_impact * 10000.0,
                "adverse_selection_bps": adverse_selection_bps,
                "passive_fill": passive_fill,
            },
        )
        return fill, {
            "spread_fraction": spread_fraction,
            "slippage_fraction": slippage,
            "market_impact_fraction": market_impact,
            "passive_fill": passive_fill,
            "queue_ahead_fraction_start": queue_before,
            "queue_ahead_fraction_end": order.queue_ahead_fraction,
            "liquidity_fraction": liquidity_fraction,
            "adverse_selection_bps": adverse_selection_bps,
        }

    def _queue_progress(self, order: SimulatedOrder, bar: "pd.Series", *, close_through: bool) -> float:
        volume = max(float(bar["volume"]), 0.0)
        if volume <= 0:
            return 0.0
        bar_range = max(float(bar["high"]) - float(bar["low"]), 0.0)
        close = max(float(bar["close"]), 1e-9)
        touch_strength = min((bar_range / close) * 30.0, 0.65)
        if close_through:
            touch_strength += 0.25
        touch_strength += min(order.resting_bars * 0.04, 0.15)
        queue_decay = float(getattr(self.config, "simulation_queue_decay", 0.55))
        return max(touch_strength * queue_decay, 0.0)

    def _liquidity_fraction(self, bar: "pd.Series", order: SimulatedOrder, *, passive_fill: bool) -> float:
        open_price = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        price = max(float(order.requested_price), 1e-9)
        distance_from_close = abs(close - price) / price
        bar_range = max(high - low, price * 1e-6)
        touch_depth = 1.0 - min(abs(close - price) / bar_range, 1.0)
        base = 0.9 if order.order_type == "market" else 0.55
        if order.order_type == "limit" and bool(order.metadata.get("urgent_limit_execution", False)):
            base += max(float(getattr(self.config, "simulation_execution_urgent_limit_liquidity_bonus", 0.12) or 0.12), 0.0)
        if passive_fill:
            base += 0.15 * touch_depth
        if (order.side in {"long", "buy"} and open_price <= price) or (order.side not in {"long", "buy"} and open_price >= price):
            base += 0.20
        if distance_from_close > 0.01:
            base *= 0.75
        return max(min(base, 1.0), 0.15)


class _SimulationBotAdapter:
    def __init__(self, engine: "HistoricalSimulationEngine"):
        self.config = engine.config
        self.state = engine.state
        self.exec = engine.venue
        self.learning = engine.learning
        self.state_store = engine.state_store
        self.backtest_engine = engine


class HistoricalSimulationEngine:
    """
    Sequential historical replay engine that mirrors the live flow closely enough
    to drive the adaptive learning system without lookahead bias.
    """

    def __init__(
        self,
        config: Any,
        *,
        signal_engine_cls: Any,
        bot_state_cls: type[BotState] = BotState,
        position_cls: type[Position] = Position,
        risk_manager_cls: type[RiskManager] = RiskManager,
        artifact_dir: str | None = None,
        progress_callback: Callable[[Dict[str, Any]], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self.signal_engine_cls = signal_engine_cls
        self.bot_state_cls = bot_state_cls
        self.position_cls = position_cls
        self.risk_manager_cls = risk_manager_cls
        self.artifact_dir = artifact_dir or tempfile.mkdtemp(prefix="trade_bot_sim_")
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.state_store = SQLiteStateStore(os.path.join(self.artifact_dir, "simulation.sqlite3"))
        self.learning_store = PersistentLearningStore(
            self.state_store,
            SQLiteStateStore(os.path.abspath(os.path.join(self.artifact_dir, "..", LEARNING_DB_FILE))),
        )
        self.learning_backfill_summary = backfill_learning_from_sqlite_artifacts(
            os.path.abspath(os.path.join(self.artifact_dir, "..")),
            self.learning_store.global_store,
        )
        self.state = bot_state_cls(balance=float(getattr(config, "starting_balance", 0.0) or 0.0), paper_mode=True)
        self.risk = risk_manager_cls(config, self.state)
        if hasattr(self.risk, "now_provider"):
            self.risk.now_provider = lambda: self._now if self._now is not None else dt.datetime.now()
        self.learning = TradeLearningEngine(config, self.learning_store)
        self.trades: list[Dict[str, Any]] = []
        self.balance_curve: list[float] = []
        self.events: list[Dict[str, Any]] = []
        self._event_counts: Dict[str, int] = {}
        self._raw_signals = 0
        self._skipped_signals = 0
        self._partial_fills = 0
        self._submitted_orders = 0
        self._expired_orders = 0
        self._filled_orders = 0
        self._cancelled_orders = 0
        self._ambiguous_exit_bars = 0
        self._checkpoints_written = 0
        self._now: dt.datetime | None = None
        self.exchange: HistoricalReplayExchange | None = None
        self.venue: SimulatedExecutionVenue | None = None
        self.signals: Any = None
        self._rng = random.Random(int(getattr(config, "simulation_random_seed", 7)))
        self._bot_adapter = _SimulationBotAdapter(self)
        self._simulation_symbol_universe: list[str] = []
        self._base_timeframe = str(getattr(config, "timeframes", {}).get("entry", "15m"))
        self._campaign_days = 0
        self._historical_data_rows: Dict[str, int] = {}
        self._progress_callback = progress_callback
        self._stop_requested = stop_requested
        self._stopped_early = False
        self._stop_reason: str | None = None
        self._resumed_from_checkpoint = False
        self._checkpoint_path = os.path.join(self.artifact_dir, "simulation_checkpoint.json")
        self._artifact_manifest_path = os.path.join(self.artifact_dir, "simulation_artifacts.json")

    def _historical_cache_path(self, symbol: str, timeframe: str, days: int) -> str:
        raw_cache_dir = str(getattr(self.config, "historical_data_cache_dir", "data/historical_cache") or "data/historical_cache")
        cache_dir = raw_cache_dir if os.path.isabs(raw_cache_dir) else os.path.abspath(raw_cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        safe_symbol = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(symbol or "unknown")).strip("_") or "unknown"
        safe_timeframe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(timeframe or "unknown")).strip("_") or "unknown"
        return os.path.join(cache_dir, f"{safe_symbol}_{safe_timeframe}_{int(days)}d.csv")

    def _read_historical_cache(self, symbol: str, timeframe: str, days: int, *, require_fresh: bool) -> "pd.DataFrame":
        _require_pandas()
        if not bool(getattr(self.config, "historical_data_cache_enabled", True)):
            return pd.DataFrame()
        cache_path = self._historical_cache_path(symbol, timeframe, days)
        if not os.path.exists(cache_path):
            return pd.DataFrame()
        if require_fresh:
            max_age_hours = float(getattr(self.config, "historical_data_cache_max_staleness_hours", 168.0) or 168.0)
            age_hours = (dt.datetime.now(dt.timezone.utc).timestamp() - os.path.getmtime(cache_path)) / 3600.0
            if max_age_hours >= 0.0 and age_hours > max_age_hours:
                return pd.DataFrame()
        try:
            frame = pd.read_csv(cache_path, parse_dates=["timestamp"])
        except Exception:
            return pd.DataFrame()
        if frame.empty:
            return frame
        frame.set_index("timestamp", inplace=True)
        return _normalize_ohlcv_frame(frame)

    def _write_historical_cache(self, symbol: str, timeframe: str, days: int, frame: "pd.DataFrame") -> None:
        _require_pandas()
        if not bool(getattr(self.config, "historical_data_cache_enabled", True)) or frame.empty:
            return
        cache_path = self._historical_cache_path(symbol, timeframe, days)
        try:
            output = _normalize_ohlcv_frame(frame).reset_index()
            output.rename(columns={output.columns[0]: "timestamp"}, inplace=True)
            output.to_csv(cache_path, index=False)
        except Exception:
            return

    def load_historical_data(self, symbol: str, timeframe: str, days: int = 180) -> "pd.DataFrame":
        _require_pandas()
        import ccxt

        cached = self._read_historical_cache(symbol, timeframe, days, require_fresh=True)
        if not cached.empty:
            return cached
        exchange = ccxt.binance({"enableRateLimit": True})
        end_ts = pd.Timestamp.now(tz="UTC").tz_localize(None)
        start_ts = end_ts - pd.Timedelta(days=days)
        since = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)
        ohlcv: list[list[Any]] = []
        seen: set[int] = set()
        try:
            while since < end_ms:
                batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                if not batch:
                    break
                rows = [row for row in batch if int(row[0]) not in seen]
                if not rows:
                    break
                seen.update(int(row[0]) for row in rows)
                ohlcv.extend(rows)
                since = int(rows[-1][0]) + 1
                if len(batch) < 1000:
                    break
        except Exception:
            fallback = self._read_historical_cache(symbol, timeframe, days, require_fresh=False)
            if not fallback.empty:
                return fallback
            return _empty_ohlcv_frame()
        frame = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if frame.empty:
            return frame
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms")
        frame.set_index("timestamp", inplace=True)
        frame = _normalize_ohlcv_frame(frame)
        self._write_historical_cache(symbol, timeframe, days, frame)
        return frame

    def build_historical_universe(self, symbols: Sequence[str], timeframe: str, days: int) -> HistoricalDataUniverse:
        _require_pandas()
        required_timeframes = {timeframe}
        required_timeframes.update(str(tf) for tf in getattr(self.config, "timeframes", {}).values())
        datasets: Dict[tuple[str, str], pd.DataFrame] = {}
        self._historical_data_rows = {}
        for symbol in symbols:
            base = self.load_historical_data(symbol, timeframe, days)
            base = _normalize_ohlcv_frame(base)
            datasets[(symbol, timeframe)] = base
            self._historical_data_rows[f"{symbol}:{timeframe}"] = int(len(base))
            for target_tf in required_timeframes:
                if target_tf == timeframe:
                    continue
                target_minutes = timeframe_to_minutes(target_tf)
                base_minutes = timeframe_to_minutes(timeframe)
                if target_minutes >= base_minutes and target_minutes % base_minutes == 0:
                    datasets[(symbol, target_tf)] = resample_ohlcv(base, target_tf)
                else:
                    loaded = self.load_historical_data(symbol, target_tf, days)
                    datasets[(symbol, target_tf)] = _normalize_ohlcv_frame(loaded)
                self._historical_data_rows[f"{symbol}:{target_tf}"] = int(len(datasets[(symbol, target_tf)]))
        return HistoricalDataUniverse(datasets, base_timeframe=timeframe)

    def _reset_run_state(self) -> None:
        self.state.balance = float(getattr(self.config, "starting_balance", self.state.balance) or 0.0)
        self.trades = []
        self.balance_curve = [float(self.state.balance)]
        self.events = []
        self._event_counts = {}
        self._raw_signals = 0
        self._skipped_signals = 0
        self._partial_fills = 0
        self._submitted_orders = 0
        self._expired_orders = 0
        self._filled_orders = 0
        self._cancelled_orders = 0
        self._ambiguous_exit_bars = 0
        self._checkpoints_written = 0
        self._stopped_early = False
        self._stop_reason = None
        self._resumed_from_checkpoint = False
        self._now = None
        self._campaign_timeline_start = None
        self._skip_reason_counts: Dict[str, int] = {}
        self._generation_outcomes: Dict[str, int] = {}
        self._generation_outcomes_by_symbol: Dict[str, Dict[str, int]] = {}
        self._generation_reasons_by_symbol: Dict[str, Dict[str, int]] = {}
        self._replacement_candidates_seen: int = 0
        self._replacement_candidates_selected: int = 0
        self._replacement_candidates_by_strategy: Dict[str, int] = {}
        self._replacement_selected_by_strategy: Dict[str, int] = {}
        self._replacement_rejections_by_reason: Dict[str, int] = {}
        self._replacement_candidates_by_symbol: Dict[str, int] = {}
        self._replacement_cross_symbol_selected: int = 0
        self._replacement_cross_symbol_selected_by_strategy: Dict[str, int] = {}
        self._replacement_near_misses_seen: int = 0
        self._replacement_near_misses_by_strategy: Dict[str, int] = {}
        self._replacement_near_misses_by_symbol: Dict[str, int] = {}
        self._replacement_near_misses_by_reason: Dict[str, int] = {}
        self._replacement_near_misses_by_detail: Dict[str, int] = {}
        self._replacement_near_miss_examples: list[Dict[str, Any]] = []
        self._replacement_submitted: int = 0
        self._replacement_filled: int = 0
        self._replacement_closed: int = 0
        self._replacement_wins: int = 0
        self._replacement_losses: int = 0
        self._replacement_submitted_by_strategy: Dict[str, int] = {}
        self._replacement_filled_by_strategy: Dict[str, int] = {}
        self._replacement_closed_by_strategy: Dict[str, int] = {}
        self._replacement_wins_by_strategy: Dict[str, int] = {}
        self._replacement_losses_by_strategy: Dict[str, int] = {}
        self._replacement_guard_blocks: int = 0
        self._replacement_guard_blocks_by_reason: Dict[str, int] = {}
        self._replacement_submitted_by_day: Dict[str, int] = {}
        self._replacement_submitted_by_symbol_day: Dict[str, Dict[str, int]] = {}
        self._strategy_rejection_reasons_by_symbol: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._proposal_count: int = 0
        self._proposals_by_strategy: Dict[str, int] = {}
        self._frequency_adjustment_counts: Dict[str, int] = {}
        self._frequency_adjustment_counts_by_strategy: Dict[str, Dict[str, int]] = {}
        self._frequency_expansion_allowed_count: int = 0
        self._frequency_expansion_blocked_count: int = 0
        self._frequency_expansion_block_reasons: Dict[str, int] = {}
        self._family_rotation_counts: Dict[str, int] = {}
        self._family_rotation_counts_by_strategy: Dict[str, Dict[str, int]] = {}
        self._family_rotation_recovery_counts: Dict[str, int] = {}
        self._family_rotation_recovery_counts_by_strategy: Dict[str, Dict[str, int]] = {}
        self._learning_evidence_counts: Dict[str, int] = {}
        self._learning_evidence_counts_by_strategy: Dict[str, Dict[str, int]] = {}
        self._learning_asymmetry_counts: Dict[str, int] = {}
        self._learning_asymmetry_counts_by_strategy: Dict[str, Dict[str, int]] = {}
        self._missed_opportunity_relaxations: Dict[str, int] = {}
        self._missed_opportunity_relaxations_by_strategy: Dict[str, Dict[str, int]] = {}
        self._signals_by_strategy: Dict[str, int] = {}
        self._signals_by_regime: Dict[str, int] = {}
        self._signals_by_order_type: Dict[str, int] = {}
        self._limit_to_market_upgrades: int = 0
        self._limit_to_market_upgrades_by_strategy: Dict[str, int] = {}
        self._limit_queue_priority_assists: int = 0
        self._limit_queue_priority_assists_by_strategy: Dict[str, int] = {}
        self._limit_latency_reductions: int = 0
        self._limit_latency_reductions_by_strategy: Dict[str, int] = {}
        self._stale_market_escalations: int = 0
        self._stale_market_escalations_by_strategy: Dict[str, int] = {}
        self._touch_escalations: int = 0
        self._touch_escalations_by_strategy: Dict[str, int] = {}
        self._partial_profit_takes: int = 0
        self._partial_profit_takes_by_strategy: Dict[str, int] = {}
        self._submitted_by_strategy: Dict[str, int] = {}
        self._submitted_by_order_type: Dict[str, int] = {}
        self._repriced_orders: int = 0
        self._repriced_by_strategy: Dict[str, int] = {}
        self._filled_by_strategy: Dict[str, int] = {}
        self._closed_by_strategy: Dict[str, int] = {}
        self._closed_by_order_type: Dict[str, int] = {}
        self._wins_by_strategy: Dict[str, int] = {}
        self._losses_by_strategy: Dict[str, int] = {}
        self._realized_performance_penalty_counts: Dict[str, int] = {}
        self._realized_performance_penalty_counts_by_strategy: Dict[str, Dict[str, int]] = {}
        self._realized_performance_no_trade_blocks: Dict[str, int] = {}
        self._realized_performance_no_trade_blocks_by_strategy: Dict[str, Dict[str, int]] = {}
        self._reentry_cooldowns: Dict[str, Dict[str, Any]] = {}
        self._reentry_cooldown_registrations: Dict[str, int] = {}
        self._reentry_cooldown_registrations_by_strategy: Dict[str, Dict[str, int]] = {}
        self._recent_setup_signatures: Dict[str, Dict[str, Any]] = {}
        self._repeated_setup_blocks: int = 0
        self._repeated_setup_blocks_by_strategy: Dict[str, int] = {}
        self._fresh_setup_blocks: int = 0
        self._fresh_setup_blocks_by_strategy: Dict[str, int] = {}
        self._triple_barrier_labels: list[Dict[str, Any]] = []
        self._triple_barrier_label_counts: Dict[str, int] = {}
        self._triple_barrier_label_counts_by_strategy: Dict[str, Dict[str, int]] = {}
        self._triple_barrier_label_counts_by_symbol: Dict[str, Dict[str, int]] = {}
        self._triple_barrier_bucket_stats: Dict[str, Dict[str, Any]] = {}
        self._pullback_meta_filter_blocks: int = 0
        self._pullback_meta_filter_blocks_by_bucket: Dict[str, int] = {}
        self._pullback_meta_filter_blocks_by_symbol: Dict[str, int] = {}
        self._latest_universe_selection: Dict[str, Any] = {"eligible_symbols": [], "rejected_symbols": {}, "scored_symbols": {}}
        self._raw_signals_by_symbol: Dict[str, int] = {}
        self._submitted_by_symbol: Dict[str, int] = {}
        self._filled_by_symbol: Dict[str, int] = {}
        self._closed_by_symbol: Dict[str, int] = {}
        self._wins_by_symbol: Dict[str, int] = {}
        self._losses_by_symbol: Dict[str, int] = {}
        self._skipped_by_symbol: Dict[str, int] = {}
        self._skip_reasons_by_symbol: Dict[str, Dict[str, int]] = {}
        self._raw_signals_by_strategy_by_symbol: Dict[str, Dict[str, int]] = {}
        self._submitted_by_strategy_by_symbol: Dict[str, Dict[str, int]] = {}
        self._filled_by_strategy_by_symbol: Dict[str, Dict[str, int]] = {}
        self._closed_by_strategy_by_symbol: Dict[str, Dict[str, int]] = {}
        self._wins_by_strategy_by_symbol: Dict[str, Dict[str, int]] = {}
        self._losses_by_strategy_by_symbol: Dict[str, Dict[str, int]] = {}
        self._skip_reasons_by_strategy_by_symbol: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._expected_edge_sum_by_symbol: Dict[str, float] = {}
        self.state.open_positions = {}
        self.state.today_trades_count = 0
        self.state.today_start_date = dt.datetime.now(dt.timezone.utc).date()
        self.state.consecutive_losses = 0
        self.state.reduced_risk_mode = False
        self.state.emergency_mode = False
        self.state.equity_start_of_day = float(self.state.balance)
        self.state.peak_equity = float(self.state.balance)

    @staticmethod
    def _increment_counter(counter: Dict[str, int], key: str, amount: int = 1) -> None:
        counter[key] = int(counter.get(key, 0)) + int(amount)

    @staticmethod
    def _increment_nested_counter(counter: Dict[str, Dict[str, int]], outer_key: str, inner_key: str, amount: int = 1) -> None:
        inner = counter.setdefault(outer_key, {})
        inner[inner_key] = int(inner.get(inner_key, 0)) + int(amount)

    @staticmethod
    def _increment_nested_reason_counter(
        counter: Dict[str, Dict[str, Dict[str, int]]],
        outer_key: str,
        middle_key: str,
        reason: str,
        amount: int = 1,
    ) -> None:
        middle = counter.setdefault(outer_key, {})
        reasons = middle.setdefault(middle_key, {})
        reasons[reason] = int(reasons.get(reason, 0)) + int(amount)

    @staticmethod
    def _serialize_position(position: Position) -> Dict[str, Any]:
        payload = asdict(position)
        for key in ("opened_at", "leverage", "order_id", "status", "fee_paid", "unrealized_pnl", "last_update", "initial_stop_loss", "initial_take_profit", "metadata"):
            value = payload.get(key)
            if isinstance(value, dt.datetime):
                payload[key] = value.isoformat()
        return payload

    def _deserialize_position(self, payload: Dict[str, Any]) -> Position:
        item = dict(payload)
        for key in ("opened_at", "last_update"):
            if item.get(key):
                item[key] = dt.datetime.fromisoformat(str(item[key]))
        return self.position_cls(**item)

    def _positions_by_id(self) -> Dict[str, Position]:
        positions: Dict[str, Position] = {}
        for current in self.state.open_positions.values():
            values = current if isinstance(current, list) else [current]
            for position in values:
                pos_id = str((getattr(position, "metadata", {}) or {}).get("simulation_position_id", ""))
                if pos_id:
                    positions[pos_id] = position
        return positions

    def _persist_checkpoint(self, current_index: int, total_bars: int, timeline: Sequence[dt.datetime], *, reason: str | None = None) -> None:
        if self.venue is None:
            return
        payload = {
            "symbols": list(self._simulation_symbol_universe),
            "timeframe": self._base_timeframe,
            "days": self._campaign_days,
            "timeline_start": timeline[0].isoformat() if timeline else None,
            "timeline_end": timeline[-1].isoformat() if timeline else None,
            "current_index": current_index,
            "current_time": self._now.isoformat() if self._now else None,
            "seed": int(getattr(self.config, "simulation_random_seed", 7)),
            "reason": reason,
            "rng_state": repr(self._rng.getstate()),
            "state": {
                "balance": float(self.state.balance),
                "today_trades_count": int(self.state.today_trades_count),
                "today_start_date": self.state.today_start_date.isoformat() if getattr(self.state, "today_start_date", None) else None,
                "consecutive_losses": int(self.state.consecutive_losses),
                "reduced_risk_mode": bool(self.state.reduced_risk_mode),
                "emergency_mode": bool(self.state.emergency_mode),
                "equity_start_of_day": float(getattr(self.state, "equity_start_of_day", 0.0) or 0.0),
                "peak_equity": float(getattr(self.state, "peak_equity", 0.0) or 0.0),
                "unrealized_pnl": float(getattr(self.state, "unrealized_pnl", 0.0) or 0.0),
                "open_positions": {
                    symbol: [self._serialize_position(pos) for pos in (values if isinstance(values, list) else [values])]
                    for symbol, values in self.state.open_positions.items()
                },
            },
            "run_state": {
                "trades": self.trades,
                "balance_curve": self.balance_curve,
                "event_counts": self._event_counts,
                "raw_signals": self._raw_signals,
                "skipped_signals": self._skipped_signals,
                "partial_fills": self._partial_fills,
                "submitted_orders": self._submitted_orders,
                "expired_orders": self._expired_orders,
                "filled_orders": self._filled_orders,
                "cancelled_orders": self._cancelled_orders,
                "ambiguous_exit_bars": self._ambiguous_exit_bars,
                "open_orders": self.venue.serialize_open_orders(),
            },
        }
        with open(self._checkpoint_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        self._checkpoints_written += 1

    def _restore_checkpoint(self, *, symbols: Sequence[str], timeframe: str, days: int, timeline: Sequence[dt.datetime]) -> int:
        if not bool(getattr(self.config, "simulation_enable_checkpointing", True)):
            return 0
        if not bool(getattr(self.config, "simulation_resume_from_checkpoint", True)):
            return 0
        if not os.path.exists(self._checkpoint_path) or self.venue is None:
            return 0
        with open(self._checkpoint_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if list(payload.get("symbols", [])) != list(symbols):
            return 0
        if str(payload.get("timeframe")) != str(timeframe):
            return 0
        if int(payload.get("days", days) or 0) != int(days):
            return 0
        if str(payload.get("timeline_start")) != (timeline[0].isoformat() if timeline else None):
            return 0
        if str(payload.get("timeline_end")) != (timeline[-1].isoformat() if timeline else None):
            return 0
        state_payload = dict(payload.get("state", {}) or {})
        self.state.balance = float(state_payload.get("balance", self.state.balance) or 0.0)
        self.state.today_trades_count = int(state_payload.get("today_trades_count", 0) or 0)
        if state_payload.get("today_start_date"):
            self.state.today_start_date = dt.date.fromisoformat(str(state_payload["today_start_date"]))
        self.state.consecutive_losses = int(state_payload.get("consecutive_losses", 0) or 0)
        self.state.reduced_risk_mode = bool(state_payload.get("reduced_risk_mode", False))
        self.state.emergency_mode = bool(state_payload.get("emergency_mode", False))
        self.state.equity_start_of_day = float(state_payload.get("equity_start_of_day", self.state.balance) or 0.0)
        self.state.peak_equity = float(state_payload.get("peak_equity", self.state.balance) or 0.0)
        self.state.unrealized_pnl = float(state_payload.get("unrealized_pnl", 0.0) or 0.0)
        open_positions: Dict[str, list[Position]] = {}
        for symbol, items in dict(state_payload.get("open_positions", {}) or {}).items():
            open_positions[symbol] = [self._deserialize_position(item) for item in list(items or [])]
        self.state.open_positions = open_positions
        positions_by_id = self._positions_by_id()
        run_state = dict(payload.get("run_state", {}) or {})
        self.trades = list(run_state.get("trades", []) or [])
        self.balance_curve = list(run_state.get("balance_curve", []) or [float(self.state.balance)])
        self._event_counts = dict(run_state.get("event_counts", {}) or {})
        self._raw_signals = int(run_state.get("raw_signals", 0) or 0)
        self._skipped_signals = int(run_state.get("skipped_signals", 0) or 0)
        self._partial_fills = int(run_state.get("partial_fills", 0) or 0)
        self._submitted_orders = int(run_state.get("submitted_orders", 0) or 0)
        self._expired_orders = int(run_state.get("expired_orders", 0) or 0)
        self._filled_orders = int(run_state.get("filled_orders", 0) or 0)
        self._cancelled_orders = int(run_state.get("cancelled_orders", 0) or 0)
        self._ambiguous_exit_bars = int(run_state.get("ambiguous_exit_bars", 0) or 0)
        self.venue.restore_open_orders(list(run_state.get("open_orders", []) or []), positions_by_id)
        if payload.get("rng_state"):
            self._rng.setstate(ast.literal_eval(str(payload["rng_state"])))
        self._resumed_from_checkpoint = True
        if payload.get("current_time"):
            self._now = dt.datetime.fromisoformat(str(payload["current_time"]))
        return max(int(payload.get("current_index", -1) or -1) + 1, 0)

    def _clear_checkpoint(self) -> None:
        if os.path.exists(self._checkpoint_path):
            os.remove(self._checkpoint_path)

    def _artifact_manifest(self, result: Dict[str, Any] | None = None) -> Dict[str, Any]:
        manifest = {
            "artifact_dir": self.artifact_dir,
            "sqlite_path": getattr(self.state_store, "path", os.path.join(self.artifact_dir, "simulation.sqlite3")),
            "checkpoint_path": self._checkpoint_path,
            "checkpoint_exists": os.path.exists(self._checkpoint_path),
            "manifest_path": self._artifact_manifest_path,
            "resumed_from_checkpoint": self._resumed_from_checkpoint,
            "checkpoints_written": self._checkpoints_written,
            "seed": int(getattr(self.config, "simulation_random_seed", 7)),
            "symbols": list(self._simulation_symbol_universe),
            "timeframe": self._base_timeframe,
            "days": self._campaign_days,
        }
        if result is not None:
            manifest["result_snapshot"] = {
                "num_trades": result.get("num_trades", 0),
                "total_return_pct": result.get("total_return_pct", 0.0),
                "stopped_early": result.get("stopped_early", False),
            }
        return manifest

    def _write_artifact_manifest(self, result: Dict[str, Any] | None = None) -> None:
        with open(self._artifact_manifest_path, "w", encoding="utf-8") as handle:
            json.dump(self._artifact_manifest(result), handle, indent=2, sort_keys=True, default=str)

    def _apply_optional_stress(self, bar_by_symbol: Dict[str, "pd.Series"], current_index: int) -> Dict[str, "pd.Series"]:
        every_n = max(int(getattr(self.config, "simulation_stress_every_n_bars", 0)), 0)
        shock_bps = float(getattr(self.config, "simulation_stress_shock_bps", 0.0) or 0.0)
        if every_n <= 0 or shock_bps <= 0 or current_index <= 0 or current_index % every_n != 0:
            return bar_by_symbol
        shock_fraction = shock_bps / 10000.0
        stressed: Dict[str, pd.Series] = {}
        for symbol, bar in bar_by_symbol.items():
            symbol_seed = sum(ord(ch) for ch in symbol)
            direction = -1.0 if (symbol_seed + current_index) % 2 == 0 else 1.0
            multiplier = 1.0 + (direction * shock_fraction)
            stressed_bar = bar.copy()
            stressed_bar["open"] = float(bar["open"]) * multiplier
            stressed_bar["close"] = float(bar["close"]) * multiplier
            stressed_bar["high"] = max(float(bar["high"]) * max(multiplier, 1.0), float(stressed_bar["open"]), float(stressed_bar["close"]))
            stressed_bar["low"] = min(float(bar["low"]) * min(multiplier, 1.0), float(stressed_bar["open"]), float(stressed_bar["close"]))
            stressed[symbol] = stressed_bar
        return stressed

    def run_backtest(self, symbol: str, timeframe: str = "15m", days: int = 180) -> Dict[str, Any]:
        return self.run_campaign([symbol], timeframe=timeframe, days=days)

    def run_campaign(self, symbols: Sequence[str], *, timeframe: str = "15m", days: int = 180) -> Dict[str, Any]:
        _require_pandas()
        self._simulation_symbol_universe = list(symbols)
        self._base_timeframe = timeframe
        self._campaign_days = int(days)
        universe = self.build_historical_universe(symbols, timeframe, days)
        primary_symbol = symbols[0]
        self.exchange = HistoricalReplayExchange(universe, primary_symbol, base_timeframe=timeframe, config=self.config)
        self.venue = SimulatedExecutionVenue(self.config, self.exchange, rng=self._rng)
        self.signals = self.signal_engine_cls(self.config, self.exchange)
        self.signals.learning_context_provider = self._learning_context_for_signal
        if hasattr(self.signals, "frequency_context_provider"):
            self.signals.frequency_context_provider = self._frequency_context_for_signal
        self._reset_run_state()

        timeline = universe.timeline(symbols, timeframe=timeframe)
        self._campaign_timeline_start = timeline[0] if timeline else None
        warmup = max(int(getattr(self.config, "backtest_warmup_candles", 100)), 20)
        if len(timeline) <= warmup:
            market_data = {
                "datasets": dict(sorted(self._historical_data_rows.items())),
                "datasets_loaded": int(sum(1 for rows in self._historical_data_rows.values() if int(rows or 0) > 0)),
                "datasets_missing": int(sum(1 for rows in self._historical_data_rows.values() if int(rows or 0) <= 0)),
                "total_rows": int(sum(int(rows or 0) for rows in self._historical_data_rows.values())),
                "data_available": bool(any(int(rows or 0) > 0 for rows in self._historical_data_rows.values())),
            }
            return {
                "error": "Not enough data",
                "num_bars": len(timeline),
                "num_trades": 0,
                "raw_signals": 0,
                "total_return_pct": 0.0,
                "win_rate_pct": 0.0,
                "campaign_summary": {
                    "market_data": market_data,
                    "validation_harness": {
                        "signal_to_submission_pct": 0.0,
                        "submission_to_fill_pct": 0.0,
                        "fill_to_close_pct": 0.0,
                        "stop_loss_negative_pl_share_pct": 0.0,
                        "repeated_setup_density_pct": 0.0,
                        "fresh_setup_block_density_pct": 0.0,
                        "repeated_setup_blocks": 0,
                        "fresh_setup_blocks": 0,
                    },
                    "acceptance": {
                        "checks": {"market_data_available": bool(market_data["data_available"])},
                        "passes_all": False,
                    },
                },
            }
        resume_possible = bool(getattr(self.config, "simulation_resume_from_checkpoint", True)) and os.path.exists(self._checkpoint_path)
        if not resume_possible and os.path.exists(self.state_store.path):
            os.remove(self.state_store.path)
            self.state_store = SQLiteStateStore(self.state_store.path)
            self.learning_store = PersistentLearningStore(
                self.state_store,
                SQLiteStateStore(os.path.abspath(os.path.join(self.artifact_dir, "..", LEARNING_DB_FILE))),
            )
            self.learning = TradeLearningEngine(self.config, self.learning_store)
            self._bot_adapter = _SimulationBotAdapter(self)
        restored_index = self._restore_checkpoint(symbols=symbols, timeframe=timeframe, days=days, timeline=timeline) if resume_possible else 0
        if resume_possible and not self._resumed_from_checkpoint and os.path.exists(self.state_store.path):
            self._clear_checkpoint()
            os.remove(self.state_store.path)
            self.state_store = SQLiteStateStore(self.state_store.path)
            self.learning_store = PersistentLearningStore(
                self.state_store,
                SQLiteStateStore(os.path.abspath(os.path.join(self.artifact_dir, "..", LEARNING_DB_FILE))),
            )
            self.learning = TradeLearningEngine(self.config, self.learning_store)
            self._bot_adapter = _SimulationBotAdapter(self)
            restored_index = 0
        start_index = max(restored_index, warmup)

        last_index = 0
        for current_index in range(start_index, len(timeline)):
            now = timeline[current_index]
            if self._should_stop():
                self._stopped_early = True
                self._stop_reason = "stop_requested"
                self._persist_checkpoint(last_index, len(timeline), timeline, reason=self._stop_reason)
                self._write_artifact_manifest()
                break
            last_index = current_index
            self._now = now
            self.exchange.set_time(now)
            self.risk.check_daily_reset()
            self.learning.evaluate_pending_shadow_decisions(self.exchange, _as_utc(now))
            bar_by_symbol = self._apply_optional_stress(self._bars_for_time(symbols, now), current_index)
            self._record_event(EVENT_MARKET_DATA, uuid.uuid4().hex, {"timestamp": now.isoformat(), "symbols": sorted(bar_by_symbol.keys())})
            self._apply_drawdown_controls()
            processed = self.venue.process_bar(now=now, current_index=current_index, bar_by_symbol=bar_by_symbol)
            self._record_reprices(processed.get("repriced", []))
            self._apply_fills(processed["fills"])
            self._expire_orders(processed["expired"])
            self._cancel_orders(processed.get("cancelled", []))
            self._manage_open_positions(bar_by_symbol, now, current_index=current_index)
            self._scan_signals(symbols, current_index)
            self._persist_progress_snapshot(current_index, len(timeline), timeline)
            self._emit_progress(current_index, len(timeline), now)
            self._mark_to_market(bar_by_symbol)

        if not self._stopped_early:
            final_time = self._now or timeline[-1]
            if self.exchange is not None:
                self.exchange.set_time(final_time)
            final_bars = self._bars_for_time(symbols, final_time)
            final_bars = self._apply_optional_stress(final_bars, last_index)
            self._liquidate_positions(final_bars, final_time)
            self.learning.evaluate_pending_shadow_decisions(self.exchange, _as_utc(final_time))
            self._emit_progress(last_index, len(timeline), final_time, force=True)
            self._clear_checkpoint()
        result = self._build_results(symbol=primary_symbol, timeframe=timeframe, days=days, timeline=timeline)
        self._write_artifact_manifest(result)
        return result

    def _bars_for_time(self, symbols: Sequence[str], now: dt.datetime) -> Dict[str, "pd.Series"]:
        assert self.exchange is not None
        bars: Dict[str, pd.Series] = {}
        for symbol in symbols:
            bar = self.exchange.current_bar(symbol, self._base_timeframe)
            if bar is not None:
                bars[symbol] = bar
        return bars

    def _scan_signals(self, symbols: Sequence[str], current_index: int) -> None:
        assert self.exchange is not None
        assert self.venue is not None
        symbols = self._eligible_universe_symbols(symbols)
        candidates: list[Dict[str, Any]] = []
        blocked_replacement_candidates: list[Dict[str, Any]] = []
        for symbol in symbols:
            if self.state.emergency_mode:
                break
            signal = self.signals.generate_signal(symbol)
            generation = dict(getattr(self.signals, "last_generation_diagnostics", {}).get(symbol, {}) or {})
            for adjustment in list(generation.get("frequency_adjustments", []) or []):
                reason = str(dict(adjustment).get("reason", "inactive") or "inactive")
                strategy_name = str(dict(adjustment).get("strategy", "unknown") or "unknown")
                if reason in {"inactive", "neutral"}:
                    continue
                self._increment_counter(self._frequency_adjustment_counts, reason)
                self._increment_nested_counter(self._frequency_adjustment_counts_by_strategy, strategy_name, reason)
            proposal_strategies = [str(item or "unknown") for item in list(generation.get("proposal_strategies", []) or [])]
            if signal and not proposal_strategies:
                proposal_strategies = [str(dict(signal).get("strategy", "unknown") or "unknown")]
            if bool(generation.get("replacement_selected", False)):
                replacement_strategy = str(generation.get("replacement_selected_strategy", dict(signal or {}).get("strategy", "unknown")) or "unknown")
                self._replacement_candidates_selected += 1
                self._increment_counter(self._replacement_selected_by_strategy, replacement_strategy)
            self._proposal_count += len(proposal_strategies)
            for strategy_name in proposal_strategies:
                self._increment_counter(self._proposals_by_strategy, strategy_name)
            if not signal:
                outcome = str(generation.get("outcome", "no_signal") or "no_signal")
                reason = str(generation.get("reason", "no_reason_recorded") or "no_reason_recorded")
                replacement_candidates = list(generation.get("replacement_candidates", []) or [])
                if not replacement_candidates and generation.get("replacement_candidate"):
                    replacement_candidates = [generation.get("replacement_candidate")]
                for candidate in replacement_candidates:
                    candidate_payload = dict(candidate or {})
                    candidate_strategy = str(candidate_payload.get("strategy", "unknown") or "unknown")
                    candidate_reason = str(candidate_payload.get("rejection_reason", reason) or reason)
                    self._replacement_candidates_seen += 1
                    self._increment_counter(self._replacement_candidates_by_strategy, candidate_strategy)
                    self._increment_counter(self._replacement_candidates_by_symbol, symbol)
                    self._increment_counter(self._replacement_rejections_by_reason, candidate_reason)
                    blocked_replacement_candidates.append(candidate_payload)
                self._increment_counter(self._generation_outcomes, outcome)
                by_symbol = self._generation_outcomes_by_symbol.setdefault(symbol, {})
                by_symbol[outcome] = int(by_symbol.get(outcome, 0)) + 1
                reason_counts = self._generation_reasons_by_symbol.setdefault(symbol, {})
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                for strategy_name, strategy_reason in dict(generation.get("strategy_rejection_reasons", {}) or {}).items():
                    self._replacement_near_misses_seen += 1
                    self._increment_counter(self._replacement_near_misses_by_strategy, str(strategy_name))
                    self._increment_counter(self._replacement_near_misses_by_symbol, symbol)
                    self._increment_counter(self._replacement_near_misses_by_reason, str(strategy_reason))
                    reason_bucket = self._strategy_rejection_reasons_by_symbol.setdefault(symbol, {}).setdefault(str(strategy_name), {})
                    reason_bucket[str(strategy_reason)] = int(reason_bucket.get(str(strategy_reason), 0)) + 1
                for candidate in list(generation.get("near_miss_candidates", []) or []):
                    detail = str(dict(candidate or {}).get("rejection_detail", "") or "")
                    if detail:
                        self._increment_counter(self._replacement_near_misses_by_detail, detail)
                    if len(self._replacement_near_miss_examples) >= 12:
                        break
                    self._replacement_near_miss_examples.append(dict(candidate or {}))
                continue
            self._raw_signals += 1
            signal = dict(signal)
            signal["symbol"] = symbol
            triple_barrier_label = self._label_triple_barrier_signal(signal)
            if triple_barrier_label:
                signal_metadata = dict(signal.get("metadata", {}) or {})
                signal_metadata["triple_barrier_label"] = triple_barrier_label
                signal["metadata"] = signal_metadata
                self._record_triple_barrier_label(signal, triple_barrier_label)
            trace_id = uuid.uuid4().hex
            self._record_event(EVENT_SIGNAL, trace_id, {"symbol": symbol, **signal})
            strategy = str(signal.get("strategy", "unknown"))
            self._increment_counter(self._signals_by_strategy, strategy)
            self._increment_counter(self._raw_signals_by_symbol, symbol)
            self._increment_nested_counter(self._raw_signals_by_strategy_by_symbol, symbol, strategy)
            learning_context = dict((signal.get("metadata", {}) or {}).get("learning_context", {}) or {})
            family_rotation = dict(learning_context.get("family_rotation", {}) or {})
            family_rotation_status = str(family_rotation.get("status", "neutral") or "neutral")
            if family_rotation_status != "neutral":
                self._increment_counter(self._family_rotation_counts, family_rotation_status)
                strategy_counts = self._family_rotation_counts_by_strategy.setdefault(strategy, {})
                strategy_counts[family_rotation_status] = int(strategy_counts.get(family_rotation_status, 0)) + 1
            if bool(family_rotation.get("recovery_active", False)):
                self._increment_counter(self._family_rotation_recovery_counts, "recovery_active")
                self._increment_nested_counter(self._family_rotation_recovery_counts_by_strategy, strategy, "recovery_active")
            if bool(learning_context.get("positive_cell_evidence", False)):
                self._increment_counter(self._learning_evidence_counts, "positive_cell_evidence")
                self._increment_nested_counter(self._learning_evidence_counts_by_strategy, strategy, "positive_cell_evidence")
            if bool(learning_context.get("negative_cell_evidence", False)):
                self._increment_counter(self._learning_evidence_counts, "negative_cell_evidence")
                self._increment_nested_counter(self._learning_evidence_counts_by_strategy, strategy, "negative_cell_evidence")
            asymmetric_learning = dict(learning_context.get("asymmetric_learning", {}) or {})
            for action in list(asymmetric_learning.get("actions", []) or []):
                action_name = str(action or "").strip()
                if not action_name:
                    continue
                self._increment_counter(self._learning_asymmetry_counts, action_name)
                self._increment_nested_counter(self._learning_asymmetry_counts_by_strategy, strategy, action_name)
            self._increment_counter(self._signals_by_regime, str(signal.get("regime", "unknown")))
            self._increment_counter(
                self._signals_by_order_type,
                str(dict(signal.get("metadata", {}) or {}).get("preferred_order_type", "market" if bool(signal.get("fast_move", False)) else "limit")),
            )
            realized_penalty = self._realized_performance_penalty_details(signal)
            signal_metadata = dict(signal.get("metadata", {}) or {})
            signal_metadata["realized_performance_penalty"] = realized_penalty
            signal["metadata"] = signal_metadata
            for reason in list(realized_penalty.get("reasons", []) or []):
                self._increment_counter(self._realized_performance_penalty_counts, str(reason))
                self._increment_nested_counter(self._realized_performance_penalty_counts_by_strategy, strategy, str(reason))
            pullback_meta_reason = self._pullback_meta_filter_reason(signal)
            if pullback_meta_reason is not None:
                self._pullback_meta_filter_blocks += 1
                bucket_key = str(dict(signal.get("metadata", {}) or {}).get("triple_barrier_label", {}).get("bucket_key", self._triple_barrier_bucket_key(signal)) or "unknown")
                self._increment_counter(self._pullback_meta_filter_blocks_by_bucket, bucket_key)
                self._increment_counter(self._pullback_meta_filter_blocks_by_symbol, symbol)
                self._record_skipped_signal(signal, pullback_meta_reason, trace_id)
                reason_counts = self._generation_reasons_by_symbol.setdefault(symbol, {})
                reason_counts[pullback_meta_reason] = int(reason_counts.get(pullback_meta_reason, 0)) + 1
                continue
            rescue_quality_reason = self._candidate_flow_rescue_quality_reason(signal)
            if rescue_quality_reason is not None:
                self._record_skipped_signal(signal, rescue_quality_reason, trace_id)
                reason_counts = self._generation_reasons_by_symbol.setdefault(symbol, {})
                reason_counts[rescue_quality_reason] = int(reason_counts.get(rescue_quality_reason, 0)) + 1
                continue
            strategy_probation_reason = self._simulation_strategy_probation_reason(signal)
            if strategy_probation_reason is not None:
                self._record_skipped_signal(signal, strategy_probation_reason, trace_id)
                reason_counts = self._generation_reasons_by_symbol.setdefault(symbol, {})
                reason_counts[strategy_probation_reason] = int(reason_counts.get(strategy_probation_reason, 0)) + 1
                continue
            strategy_symbol_reason = self._strategy_symbol_eligibility_reason(signal)
            if strategy_symbol_reason is not None:
                self._record_skipped_signal(signal, strategy_symbol_reason, trace_id)
                reason_counts = self._generation_reasons_by_symbol.setdefault(symbol, {})
                reason_counts[strategy_symbol_reason] = int(reason_counts.get(strategy_symbol_reason, 0)) + 1
                continue
            fresh_setup_reason = self._fresh_setup_reason(signal, current_index=current_index)
            if fresh_setup_reason is not None:
                self._fresh_setup_blocks += 1
                self._increment_counter(self._fresh_setup_blocks_by_strategy, strategy)
                if fresh_setup_reason == "repeated_setup_density":
                    self._repeated_setup_blocks += 1
                    self._increment_counter(self._repeated_setup_blocks_by_strategy, strategy)
                self._record_skipped_signal(signal, fresh_setup_reason, trace_id)
                reason_counts = self._generation_reasons_by_symbol.setdefault(symbol, {})
                reason_counts[fresh_setup_reason] = int(reason_counts.get(fresh_setup_reason, 0)) + 1
                self._register_setup_signature(signal, current_index=current_index)
                continue
            self._register_setup_signature(signal, current_index=current_index)
            self._expected_edge_sum_by_symbol[symbol] = float(self._expected_edge_sum_by_symbol.get(symbol, 0.0)) + float(signal.get("expected_edge_bps", 0.0) or 0.0)
            self._increment_counter(self._generation_outcomes, "selected")
            by_symbol = self._generation_outcomes_by_symbol.setdefault(symbol, {})
            by_symbol["selected"] = int(by_symbol.get("selected", 0)) + 1
            candidates.append(
                {
                    "symbol": symbol,
                    "signal": signal,
                    "trace_id": trace_id,
                }
            )

        if blocked_replacement_candidates and bool(getattr(self.config, "replacement_cross_symbol_enabled", True)):
            max_replacements = max(int(getattr(self.config, "replacement_cross_symbol_max_per_scan", 1) or 1), 0)
            selected_replacements = 0
            replacement_symbols = {
                str(candidate.get("symbol", "") or "")
                for candidate in blocked_replacement_candidates
            }
            for candidate in sorted(
                candidates,
                key=lambda item: self._candidate_rank_score(item["signal"], provisional_signals=[]),
                reverse=True,
            ):
                if selected_replacements >= max_replacements:
                    break
                if candidate["symbol"] in replacement_symbols:
                    continue
                signal_metadata = dict(candidate["signal"].get("metadata", {}) or {})
                signal_metadata["cross_symbol_replacement"] = {
                    "active": True,
                    "blocked_symbols": sorted(symbol for symbol in replacement_symbols if symbol),
                    "blocked_reasons": sorted(
                        {
                            str(blocked.get("rejection_reason", "unknown") or "unknown")
                            for blocked in blocked_replacement_candidates
                        }
                    ),
                }
                candidate["signal"]["metadata"] = signal_metadata
                selected_replacements += 1
                self._replacement_candidates_selected += 1
                self._replacement_cross_symbol_selected += 1
                strategy_name = str(candidate["signal"].get("strategy", "unknown") or "unknown")
                self._increment_counter(self._replacement_selected_by_strategy, strategy_name)
                self._increment_counter(self._replacement_cross_symbol_selected_by_strategy, strategy_name)

        prioritized_candidates: list[Dict[str, Any]] = []
        remaining_candidates = list(candidates)
        while remaining_candidates:
            best_candidate = max(
                remaining_candidates,
                key=lambda item: self._candidate_rank_score(
                    item["signal"],
                    provisional_signals=[entry["signal"] for entry in prioritized_candidates],
                ),
            )
            best_candidate = dict(best_candidate)
            best_candidate["rank_score"] = self._candidate_rank_score(
                best_candidate["signal"],
                provisional_signals=[entry["signal"] for entry in prioritized_candidates],
            )
            prioritized_candidates.append(best_candidate)
            remaining_candidates.remove(best_candidate if best_candidate in remaining_candidates else next(item for item in remaining_candidates if item["trace_id"] == best_candidate["trace_id"]))

        admitted_candidates: list[Dict[str, Any]] = []
        for candidate in prioritized_candidates:
            symbol = candidate["symbol"]
            signal = candidate["signal"]
            trace_id = candidate["trace_id"]
            duplicate_reason = self._portfolio_duplicate_throttle_reason(signal, provisional_signals=[entry["signal"] for entry in admitted_candidates], candidate_rank_score=float(candidate.get("rank_score", 0.0) or 0.0))
            if duplicate_reason is not None:
                self._record_skipped_signal(signal, duplicate_reason, trace_id)
                continue
            weak_cluster_reason = self._portfolio_persistently_weak_cluster_reason(signal, provisional_signals=[entry["signal"] for entry in admitted_candidates])
            if weak_cluster_reason is not None:
                self._record_skipped_signal(signal, weak_cluster_reason, trace_id)
                continue
            pending_positions = len(self.venue.open_orders)
            capacity_reason = self.risk.entry_capacity_status(symbol, pending_positions=pending_positions)
            if capacity_reason != "ok":
                self._record_skipped_signal(signal, capacity_reason, trace_id)
                continue
            if self.risk.check_daily_loss_limit():
                self.state.emergency_mode = True
                self._record_event(EVENT_RISK_HALT, trace_id, {"reason": "daily_loss_limit"})
                self._record_skipped_signal(signal, "daily_loss_limit", trace_id)
                continue
            side = str(signal.get("side", "long")).lower()
            if getattr(self.config, "trading_mode", "spot") == "spot" and side in {"short", "sell"}:
                self._record_skipped_signal(signal, "spot_short_blocked", trace_id)
                continue
            reentry_cooldown_reason = self._reentry_cooldown_reason(signal)
            if reentry_cooldown_reason is not None:
                self._record_skipped_signal(signal, reentry_cooldown_reason, trace_id)
                continue
            if self._same_side_already_active(symbol, side):
                self._record_skipped_signal(signal, "same_side_exists", trace_id)
                continue
            duplicate_bucket_reason = self._portfolio_duplicate_bucket_throttle_reason(signal, provisional_signals=[entry["signal"] for entry in admitted_candidates], candidate_rank_score=float(candidate.get("rank_score", 0.0) or 0.0))
            if duplicate_bucket_reason is not None:
                self._record_skipped_signal(signal, duplicate_bucket_reason, trace_id)
                continue
            no_trade_reason = self._portfolio_no_trade_reason(signal)
            if no_trade_reason is not None:
                realized_penalty = dict((dict(signal.get("metadata", {}) or {}).get("realized_performance_penalty", {}) or {}))
                if float(realized_penalty.get("no_trade_penalty_bps", 0.0) or 0.0) > 0.0:
                    for reason in list(realized_penalty.get("reasons", []) or []):
                        self._increment_counter(self._realized_performance_no_trade_blocks, str(reason))
                        self._increment_nested_counter(self._realized_performance_no_trade_blocks_by_strategy, str(signal.get("strategy", "unknown") or "unknown"), str(reason))
                self._record_skipped_signal(signal, no_trade_reason, trace_id)
                continue
            replacement_guard_reason = self._replacement_guard_reason(signal)
            if replacement_guard_reason is not None:
                self._replacement_guard_blocks += 1
                self._increment_counter(self._replacement_guard_blocks_by_reason, replacement_guard_reason)
                self._record_skipped_signal(signal, replacement_guard_reason, trace_id)
                continue

            size = self.risk.calc_position_size(
                float(signal.get("entry_price", 0.0) or 0.0),
                float(signal.get("stop_loss", 0.0) or 0.0),
                signal=signal,
            )
            learning_context = dict((signal.get("metadata", {}) or {}).get("learning_context", {}) or {})
            size *= float(learning_context.get("risk_multiplier", 1.0) or 1.0)
            portfolio_decision = self.risk.evaluate_portfolio_risk(
                symbol=symbol,
                strategy=str(signal.get("strategy", "unknown")),
                side=side,
                entry_price=float(signal.get("entry_price", 0.0) or 0.0),
                proposed_size=float(size or 0.0),
            )
            if not portfolio_decision.allowed:
                capped_size = float(portfolio_decision.capped_size or 0.0)
                if capped_size > 0 and portfolio_decision.reason in {"net_exposure_cap", "gross_exposure_cap"}:
                    size = capped_size
                else:
                    self._record_skipped_signal(signal, portfolio_decision.reason, trace_id)
                    self._record_event(EVENT_RISK_HALT, trace_id, {"reason": portfolio_decision.reason, "symbol": symbol})
                    reason_counts = self._generation_reasons_by_symbol.setdefault(symbol, {})
                    reason_counts[f"portfolio_blocked:{portfolio_decision.reason}"] = int(reason_counts.get(f"portfolio_blocked:{portfolio_decision.reason}", 0)) + 1
                    continue
            else:
                size = float(portfolio_decision.capped_size or size)
            if size <= 0:
                self._record_skipped_signal(signal, "invalid_sized_trade", trace_id)
                continue

            order = self.venue.submit_order(signal=signal, size=size, now=self._now, current_index=current_index, trace_id=trace_id)
            self._submitted_orders += 1
            if bool(order.metadata.get("limit_to_market_upgrade", False)):
                self._limit_to_market_upgrades += 1
                self._increment_counter(self._limit_to_market_upgrades_by_strategy, str(signal.get("strategy", "unknown")))
            if bool(dict(signal.get("metadata", {}) or {}).get("limit_queue_priority_assist", False)):
                self._limit_queue_priority_assists += 1
                self._increment_counter(self._limit_queue_priority_assists_by_strategy, str(signal.get("strategy", "unknown")))
            if bool(dict(signal.get("metadata", {}) or {}).get("limit_latency_reduced", False)):
                self._limit_latency_reductions += 1
                self._increment_counter(self._limit_latency_reductions_by_strategy, str(signal.get("strategy", "unknown")))
            self._increment_counter(self._submitted_by_strategy, str(signal.get("strategy", "unknown")))
            self._increment_counter(self._submitted_by_symbol, symbol)
            self._increment_nested_counter(
                self._submitted_by_strategy_by_symbol,
                symbol,
                str(signal.get("strategy", "unknown")),
            )
            self._register_replacement_submission(signal)
            self._increment_counter(self._submitted_by_order_type, order.order_type)
            self._record_event(
                EVENT_ORDER_SUBMITTED,
                trace_id,
                {
                    "order_id": order.order_id,
                    "symbol": symbol,
                    "side": side,
                    "size": size,
                    "order_type": order.order_type,
                    "activate_on": order.activate_on.isoformat(),
                },
            )
            self._record_event(EVENT_ORDER_ACK, trace_id, {"order_id": order.order_id, "status": "accepted"})
            admitted_candidates.append(candidate)

    def _record_reprices(self, orders: Sequence[SimulatedOrder]) -> None:
        for order in orders:
            self._repriced_orders += 1
            self._increment_counter(self._repriced_by_strategy, str(order.strategy or "unknown"))

    def _candidate_rank_score(self, signal: Dict[str, Any], *, provisional_signals: Sequence[Dict[str, Any]] | None = None) -> float:
        metadata = dict(signal.get("metadata", {}) or {})
        confidence = float(signal.get("signal_quality", 0.0) or 0.0)
        expected_edge_bps = float(signal.get("expected_edge_bps", 0.0) or 0.0)
        rr_ratio = float(signal.get("rr_ratio", 0.0) or 0.0)
        holding_minutes = float(signal.get("expected_holding_minutes", 480.0) or 480.0)
        ensemble_score = float(metadata.get("ensemble_score", 0.0) or 0.0)
        cross_sectional_score = float(metadata.get("cross_sectional_score", ensemble_score) or ensemble_score)
        holding_bonus = max(0.0, (480.0 - holding_minutes) / 60.0)
        symbol = str(signal.get("symbol", "") or "")
        strategy = str(signal.get("strategy", "unknown") or "unknown")
        side = str(signal.get("side", "long") or "long").lower()
        exposures = self.risk.portfolio_exposures()
        family = self.risk._family_for_symbol(symbol)
        balance = max(float(exposures.get("balance", getattr(self.state, "balance", 0.0)) or 0.0), 1e-9)
        family_used_fraction = float((dict(exposures.get("by_family", {}) or {}).get(family, 0.0) or 0.0) / balance)
        strategy_used_fraction = float((dict(exposures.get("by_strategy", {}) or {}).get(strategy, 0.0) or 0.0) / balance)

        score = cross_sectional_score + (confidence * 25.0) + (expected_edge_bps * 0.08) + (rr_ratio * 4.0) + holding_bonus
        score -= family_used_fraction * 20.0
        score -= strategy_used_fraction * 12.0
        score -= self._portfolio_crowding_penalty(signal, provisional_signals=provisional_signals)
        score -= self._directional_cluster_crowding_penalty(signal, provisional_signals=provisional_signals)
        score += self._portfolio_diversification_bonus(signal, provisional_signals=provisional_signals)
        score += self._learning_evidence_score_bias(signal)
        score += self._realized_performance_score_bias(signal)

        provisional = list(provisional_signals or [])
        provisional_families = {self.risk._family_for_symbol(str(item.get("symbol", "") or "")) for item in provisional}
        provisional_strategies = {str(item.get("strategy", "unknown") or "unknown") for item in provisional}
        provisional_sides = {str(item.get("side", "long") or "long").lower() for item in provisional}

        if family in provisional_families:
            score -= 8.0
        else:
            score += 3.0
        if strategy in provisional_strategies:
            score -= 4.5
        else:
            score += 1.5
        if provisional and side in provisional_sides:
            score -= 1.25
        repeat_penalty = float((metadata.get("realized_performance_penalty", {}) or {}).get("repeat_setup_penalty_score", 0.0) or 0.0)
        score -= repeat_penalty
        return score

    def _strategy_symbol_eligibility_reason(self, signal: Dict[str, Any]) -> str | None:
        metadata = dict(signal.get("metadata", {}) or {})
        strategy = str(signal.get("strategy", "unknown") or "unknown")
        symbol = str(signal.get("symbol", "") or "")
        strategy_symbol_rollup = self._build_strategy_symbol_rollups().get((symbol, strategy), {})
        strategy_symbol_trades = int(dict(strategy_symbol_rollup or {}).get("trades", 0) or 0)
        strategy_symbol_expectancy = float(dict(strategy_symbol_rollup or {}).get("expectancy", 0.0) or 0.0)
        strategy_symbol_veto_min_trades = max(int(getattr(self.config, "simulation_strategy_symbol_veto_min_trades", 2) or 2), 1)
        bucket = self._symbol_bucket(symbol)
        liquidity_score = float(metadata.get("liquidity_score", 0.0) or 0.0)
        regime_state = str(self._current_universe_regime_state().get("state", "neutral") or "neutral")
        if (
            getattr(self.config, "trading_mode", "spot") == "spot"
            and bool(getattr(self.config, "simulation_spot_high_beta_requires_trend_supportive", True))
            and bucket == "high_beta_alts"
            and regime_state != "trend_supportive"
        ):
            return "spot_high_beta_requires_trend_supportive"
        if strategy == "trend_breakout":
            if (
                strategy_symbol_trades >= strategy_symbol_veto_min_trades
                and strategy_symbol_expectancy <= float(getattr(self.config, "simulation_breakout_symbol_veto_expectancy_floor", -14.0) or -14.0)
            ):
                return "breakout_symbol_probation_veto"
            if liquidity_score > 0.0 and liquidity_score < float(getattr(self.config, "simulation_trend_symbol_min_liquidity_score", 0.72) or 0.72):
                return "trend_symbol_liquidity_gate"
            if (
                bool(getattr(self.config, "simulation_trend_allow_high_beta_only_in_trend_supportive", True))
                and bucket == "high_beta_alts"
                and regime_state not in {"trend_supportive", "neutral"}
            ):
                return "trend_high_beta_state_gate"
        elif strategy == "trend_pullback":
            if (
                strategy_symbol_trades >= strategy_symbol_veto_min_trades
                and strategy_symbol_expectancy <= float(getattr(self.config, "simulation_pullback_symbol_veto_expectancy_floor", -18.0) or -18.0)
            ):
                return "pullback_symbol_probation_veto"
            if getattr(self.config, "trading_mode", "spot") == "spot" and bucket == "high_beta_alts":
                high_beta_pullback_score = float(metadata.get("pullback_score", 0.0) or 0.0)
                high_beta_trend_persistence = float(metadata.get("trend_persistence", 0.0) or 0.0)
                high_beta_quality_escape = (
                    bool(getattr(self.config, "simulation_spot_high_beta_quality_escape_enabled", True))
                    and (
                        (
                            bool(metadata.get("htf_4h_bullish", False))
                            and bool(metadata.get("htf_1h_uptrend", False))
                        )
                        or bool(metadata.get("partial_htf_used", False))
                        or bool(metadata.get("htf_fallback_used", False))
                        or str(metadata.get("strategy_variant", "")) in {"spot_core_local_structure_pullback", "spot_core_micro_reclaim", "spot_major_continuation_pullback"}
                    )
                    and high_beta_pullback_score >= float(getattr(self.config, "simulation_spot_high_beta_quality_escape_min_score", 1.35) or 1.35)
                    and high_beta_trend_persistence >= float(getattr(self.config, "simulation_spot_high_beta_quality_escape_min_trend_persistence", 0.38) or 0.38)
                    and float(signal.get("signal_quality", signal.get("confidence", 0.0)) or 0.0) >= float(getattr(self.config, "simulation_spot_high_beta_quality_escape_min_signal_quality", 0.76) or 0.76)
                    and float(signal.get("expected_edge_bps", 0.0) or 0.0) >= float(getattr(self.config, "simulation_spot_high_beta_quality_escape_min_edge_bps", 35.0) or 35.0)
                    and float(metadata.get("realized_vol_percentile", 1.0) or 1.0) <= float(getattr(self.config, "simulation_spot_high_beta_quality_escape_max_volatility_percentile", 0.72) or 0.72)
                    and float(metadata.get("stretch_from_mean", metadata.get("stretch", 0.0)) or 0.0) <= float(getattr(self.config, "simulation_spot_high_beta_quality_escape_max_stretch", 0.030) or 0.030)
                )
                if (
                    high_beta_pullback_score > 0.0
                    and high_beta_pullback_score < float(getattr(self.config, "simulation_spot_high_beta_pullback_min_score", 1.55) or 1.55)
                    and not high_beta_quality_escape
                ):
                    return "spot_high_beta_pullback_score_gate"
                if (
                    high_beta_trend_persistence > 0.0
                    and high_beta_trend_persistence < float(getattr(self.config, "simulation_spot_high_beta_pullback_min_trend_persistence", 0.50) or 0.50)
                    and not high_beta_quality_escape
                ):
                    return "spot_high_beta_pullback_persistence_gate"
                if high_beta_quality_escape:
                    metadata["high_beta_quality_escape"] = True
                    signal["metadata"] = metadata
            if liquidity_score > 0.0 and liquidity_score < float(getattr(self.config, "simulation_pullback_symbol_min_liquidity_score", 0.76) or 0.76):
                return "pullback_symbol_liquidity_gate"
            if bucket in {"other", "high_beta_alts"} and regime_state not in {"trend_supportive", "neutral"}:
                return "pullback_bucket_quality_gate"
        elif strategy == "mean_reversion":
            if (
                strategy_symbol_trades >= strategy_symbol_veto_min_trades
                and strategy_symbol_expectancy <= float(getattr(self.config, "simulation_mean_reversion_symbol_veto_expectancy_floor", -12.0) or -12.0)
            ):
                return "mean_reversion_symbol_probation_veto"
            if liquidity_score > 0.0 and liquidity_score < float(getattr(self.config, "simulation_mean_reversion_symbol_min_liquidity_score", 0.80) or 0.80):
                return "mean_reversion_symbol_liquidity_gate"
        return None

    def _candidate_flow_rescue_quality_reason(self, signal: Mapping[str, Any]) -> str | None:
        if not bool(getattr(self.config, "candidate_flow_rescue_quality_gate_enabled", True)):
            return None
        if str(signal.get("strategy", "") or "") != "trend_pullback":
            return None
        metadata = dict(signal.get("metadata", {}) or {})
        rescue = dict(metadata.get("candidate_flow_rescue", {}) or {})
        if not bool(rescue.get("active", False)):
            return None
        rank_score = float(rescue.get("rank_score", signal.get("replacement_rank_score", 0.0)) or 0.0)
        min_rank = float(getattr(self.config, "candidate_flow_rescue_quality_gate_min_rank_score", 1.72) or 1.72)
        if rank_score < min_rank:
            return "candidate_flow_rescue_rank_gate"
        pullback_score = float(metadata.get("pullback_score", 0.0) or 0.0)
        if pullback_score < float(getattr(self.config, "candidate_flow_rescue_quality_gate_min_pullback_score", 0.98) or 0.98):
            return "candidate_flow_rescue_pullback_score_gate"
        trend_persistence = float(metadata.get("trend_persistence", 0.0) or 0.0)
        if trend_persistence < float(getattr(self.config, "candidate_flow_rescue_quality_gate_min_trend_persistence", 0.34) or 0.34):
            return "candidate_flow_rescue_persistence_gate"
        directional_efficiency = float(metadata.get("directional_efficiency", metadata.get("trend_efficiency", 0.0)) or 0.0)
        if directional_efficiency < float(getattr(self.config, "candidate_flow_rescue_quality_gate_min_directional_efficiency", 0.08) or 0.08):
            return "candidate_flow_rescue_efficiency_gate"
        realized_vol = float(metadata.get("realized_vol_percentile", 0.0) or 0.0)
        if realized_vol > float(getattr(self.config, "candidate_flow_rescue_quality_gate_max_volatility_percentile", 0.82) or 0.82):
            return "candidate_flow_rescue_volatility_gate"
        label_payload = dict(metadata.get("triple_barrier_label", {}) or {})
        if (
            str(label_payload.get("label", "") or "") == "SL_FIRST"
            and rank_score < float(getattr(self.config, "candidate_flow_rescue_quality_gate_sl_first_min_rank_score", 2.05) or 2.05)
        ):
            return "candidate_flow_rescue_sl_first_gate"
        return None

    def _setup_signature_key(self, signal: Dict[str, Any]) -> str:
        symbol = str(signal.get("symbol", "") or "")
        strategy = str(signal.get("strategy", "unknown") or "unknown")
        side = str(signal.get("side", "long") or "long").lower()
        return f"{symbol}::{strategy}::{side}"

    def _register_setup_signature(self, signal: Dict[str, Any], *, current_index: int) -> None:
        metadata = dict(signal.get("metadata", {}) or {})
        self._recent_setup_signatures[self._setup_signature_key(signal)] = {
            "entry_price": float(signal.get("entry_price", 0.0) or 0.0),
            "regime": str(signal.get("regime", "unknown") or "unknown"),
            "index": int(current_index),
            "setup_score": self._fresh_setup_score(signal),
            "structure_key": str(
                metadata.get("structure_key")
                or metadata.get("setup_structure")
                or metadata.get("pattern")
                or ""
            ),
        }

    def _fresh_setup_score(self, signal: Dict[str, Any]) -> float:
        metadata = dict(signal.get("metadata", {}) or {})
        strategy = str(signal.get("strategy", "unknown") or "unknown")
        if strategy == "trend_breakout":
            family_score = float(metadata.get("breakout_score", 0.0) or 0.0)
        elif strategy == "trend_pullback":
            family_score = float(metadata.get("pullback_score", 0.0) or 0.0)
        elif strategy == "mean_reversion":
            family_score = float(metadata.get("mean_reversion_score", 0.0) or 0.0)
        else:
            family_score = 0.0
        return max(
            family_score,
            float(metadata.get("ensemble_score", 0.0) or 0.0),
            float(metadata.get("cross_sectional_score", 0.0) or 0.0) / 100.0,
            float(signal.get("signal_quality", 0.0) or 0.0),
        )

    def _fresh_setup_reason(self, signal: Dict[str, Any], *, current_index: int) -> str | None:
        quality = float(signal.get("signal_quality", 0.0) or 0.0)
        if quality < float(getattr(self.config, "simulation_fresh_setup_block_quality_floor", 0.55) or 0.55):
            return None
        payload = dict(self._recent_setup_signatures.get(self._setup_signature_key(signal), {}) or {})
        if not payload:
            return None
        bars = max(int(getattr(self.config, "simulation_repeat_setup_suppression_bars", 24) or 24), 1)
        if int(current_index) - int(payload.get("index", -10_000)) > bars:
            return None
        prev_entry = float(payload.get("entry_price", 0.0) or 0.0)
        entry = float(signal.get("entry_price", 0.0) or 0.0)
        if prev_entry <= 0.0 or entry <= 0.0:
            return None
        distance_bps = abs(entry - prev_entry) / prev_entry * 10000.0
        min_shift_bps = float(getattr(self.config, "simulation_fresh_setup_min_entry_shift_bps", 32.0) or 32.0)
        score_improvement = self._fresh_setup_score(signal) - float(payload.get("setup_score", 0.0) or 0.0)
        min_score_improvement = float(getattr(self.config, "simulation_fresh_setup_min_score_improvement", 0.18) or 0.18)
        metadata = dict(signal.get("metadata", {}) or {})
        structure_key = str(
            metadata.get("structure_key")
            or metadata.get("setup_structure")
            or metadata.get("pattern")
            or ""
        )
        prev_structure_key = str(payload.get("structure_key", "") or "")
        structure_changed = bool(structure_key and prev_structure_key and structure_key != prev_structure_key)
        if distance_bps >= min_shift_bps or score_improvement >= min_score_improvement or structure_changed:
            return None
        repeated_reason = self._repeated_setup_reason(signal, current_index=current_index)
        if repeated_reason is not None:
            return repeated_reason
        realized = dict(metadata.get("realized_performance_penalty", {}) or {})
        realized["repeat_setup_penalty_score"] = max(
            float(realized.get("repeat_setup_penalty_score", 0.0) or 0.0),
            float(getattr(self.config, "simulation_repeat_setup_density_penalty_score", 6.0) or 6.0),
        )
        reasons = list(realized.get("reasons", []) or [])
        if "fresh_setup_required" not in reasons:
            reasons.append("fresh_setup_required")
        realized["reasons"] = reasons
        metadata["realized_performance_penalty"] = realized
        signal["metadata"] = metadata
        return "fresh_setup_required"

    def _repeated_setup_reason(self, signal: Dict[str, Any], *, current_index: int) -> str | None:
        quality = float(signal.get("signal_quality", 0.0) or 0.0)
        if quality < float(getattr(self.config, "simulation_repeat_setup_quality_floor", 0.70) or 0.70):
            return None
        payload = dict(self._recent_setup_signatures.get(self._setup_signature_key(signal), {}) or {})
        if not payload:
            return None
        bars = max(int(getattr(self.config, "simulation_repeat_setup_suppression_bars", 24) or 24), 1)
        if int(current_index) - int(payload.get("index", -10_000)) > bars:
            return None
        prev_entry = float(payload.get("entry_price", 0.0) or 0.0)
        entry = float(signal.get("entry_price", 0.0) or 0.0)
        if prev_entry <= 0.0 or entry <= 0.0:
            return None
        tolerance_bps = float(getattr(self.config, "simulation_repeat_setup_entry_tolerance_bps", 20.0) or 20.0)
        distance_bps = abs(entry - prev_entry) / prev_entry * 10000.0
        if distance_bps > tolerance_bps:
            return None
        if str(payload.get("regime", "unknown") or "unknown") != str(signal.get("regime", "unknown") or "unknown"):
            return None
        metadata = dict(signal.get("metadata", {}) or {})
        penalty = float(getattr(self.config, "simulation_repeat_setup_density_penalty_score", 6.0) or 6.0)
        no_trade_penalty = float(getattr(self.config, "simulation_repeat_setup_no_trade_penalty_bps", 3.0) or 3.0)
        realized = dict(metadata.get("realized_performance_penalty", {}) or {})
        realized["repeat_setup_penalty_score"] = max(float(realized.get("repeat_setup_penalty_score", 0.0) or 0.0), penalty)
        realized["no_trade_penalty_bps"] = float(realized.get("no_trade_penalty_bps", 0.0) or 0.0) + no_trade_penalty
        reasons = list(realized.get("reasons", []) or [])
        if "repeated_setup_density" not in reasons:
            reasons.append("repeated_setup_density")
        realized["reasons"] = reasons
        metadata["realized_performance_penalty"] = realized
        signal["metadata"] = metadata
        return "repeated_setup_density"

    def _portfolio_crowding_penalty(self, signal: Dict[str, Any], *, provisional_signals: Sequence[Dict[str, Any]] | None = None) -> float:
        symbol = str(signal.get("symbol", "") or "")
        family = self.risk._family_for_symbol(symbol)
        bucket = self._symbol_bucket(symbol)
        penalty = 0.0
        symbol_penalty = float(getattr(self.config, "simulation_symbol_crowding_penalty", 14.0) or 14.0)
        family_penalty = float(getattr(self.config, "simulation_family_crowding_penalty", 7.5) or 7.5)
        bucket_penalty = float(getattr(self.config, "simulation_bucket_crowding_penalty", 5.0) or 5.0)

        if symbol in self.state.open_positions:
            penalty += symbol_penalty
        for existing_symbol in dict(self.state.open_positions):
            if existing_symbol != symbol and self._symbol_bucket(existing_symbol) == bucket:
                penalty += bucket_penalty
        for provisional in list(provisional_signals or []):
            provisional_symbol = str(provisional.get("symbol", "") or "")
            provisional_family = self.risk._family_for_symbol(provisional_symbol)
            if provisional_symbol == symbol:
                penalty += symbol_penalty
            elif provisional_family == family:
                penalty += family_penalty
            elif self._symbol_bucket(provisional_symbol) == bucket:
                penalty += bucket_penalty
        return penalty

    def _portfolio_diversification_bonus(self, signal: Dict[str, Any], *, provisional_signals: Sequence[Dict[str, Any]] | None = None) -> float:
        symbol = str(signal.get("symbol", "") or "")
        family = self.risk._family_for_symbol(symbol)
        bucket = self._symbol_bucket(symbol)
        exposures = self.risk.portfolio_exposures()
        bonus = 0.0
        diversification_bonus = float(getattr(self.config, "simulation_diversification_bonus", 3.5) or 3.5)
        bucket_bonus = float(getattr(self.config, "simulation_bucket_diversification_bonus", 2.5) or 2.5)
        if symbol not in dict(exposures.get("by_symbol", {}) or {}):
            bonus += diversification_bonus * 0.5
        if family not in dict(exposures.get("by_family", {}) or {}):
            bonus += diversification_bonus
        open_buckets = {self._symbol_bucket(existing_symbol) for existing_symbol in dict(self.state.open_positions)}
        if bucket not in open_buckets:
            bonus += bucket_bonus
        provisional_families = {self.risk._family_for_symbol(str(item.get("symbol", "") or "")) for item in list(provisional_signals or [])}
        provisional_buckets = {self._symbol_bucket(str(item.get("symbol", "") or "")) for item in list(provisional_signals or [])}
        if provisional_families and family not in provisional_families:
            bonus += diversification_bonus * 0.75
        if provisional_buckets and bucket not in provisional_buckets:
            bonus += bucket_bonus * 0.75
        return bonus

    def _symbol_bucket(self, symbol: str) -> str:
        base = str(symbol or "").split("/")[0].upper()
        if base in {"BTC", "ETH"}:
            return "majors"
        if base in {"BNB", "XRP"}:
            return "exchange_beta"
        if base in {"SOL", "AVAX"}:
            return "high_beta_alts"
        if base in {"ADA", "DOT", "LINK", "TON"}:
            return "slower_large_caps"
        return "other"

    def _directional_cluster_crowding_penalty(self, signal: Dict[str, Any], *, provisional_signals: Sequence[Dict[str, Any]] | None = None) -> float:
        symbol = str(signal.get("symbol", "") or "")
        family = self.risk._family_for_symbol(symbol)
        side = str(signal.get("side", "long") or "long").lower()
        normalized_side = "long" if side in {"buy", "long"} else "short"
        penalty = 0.0
        cluster_penalty = float(getattr(self.config, "simulation_directional_cluster_penalty", 6.0) or 6.0)

        current = self.state.open_positions
        for existing_symbol, existing_value in dict(current).items():
            positions = existing_value if isinstance(existing_value, list) else [existing_value]
            for position in positions:
                if self.risk._family_for_symbol(str(existing_symbol or "")) == family and str(getattr(position, "side", "long") or "long").lower() == normalized_side:
                    penalty += cluster_penalty

        if self.venue is not None:
            for order in self.venue.open_orders:
                order_side = str(order.side or "long").lower()
                if self.risk._family_for_symbol(str(order.symbol or "")) == family and order_side in {normalized_side, side}:
                    penalty += cluster_penalty

        for provisional in list(provisional_signals or []):
            provisional_symbol = str(provisional.get("symbol", "") or "")
            provisional_side = str(provisional.get("side", "long") or "long").lower()
            if self.risk._family_for_symbol(provisional_symbol) == family and provisional_side == normalized_side:
                penalty += cluster_penalty
        return penalty

    def _learning_evidence_score_bias(self, signal: Dict[str, Any]) -> float:
        learning_context = dict((signal.get("metadata", {}) or {}).get("learning_context", {}) or {})
        score_weight = float(getattr(self.config, "simulation_learning_score_weight", 1.0) or 1.0)
        negative_penalty = float(getattr(self.config, "simulation_learning_negative_penalty", 2.5) or 2.5)
        bias = float(learning_context.get("score_delta", 0.0) or 0.0) * score_weight
        if bool(learning_context.get("negative_cell_evidence", False)):
            bias -= negative_penalty
        return bias

    def _portfolio_duplicate_throttle_reason(
        self,
        signal: Dict[str, Any],
        *,
        provisional_signals: Sequence[Dict[str, Any]] | None = None,
        candidate_rank_score: float,
    ) -> str | None:
        symbol = str(signal.get("symbol", "") or "")
        strategy = str(signal.get("strategy", "unknown") or "unknown")
        side = str(signal.get("side", "long") or "long").lower()
        family = self.risk._family_for_symbol(symbol)
        score_gap = float(getattr(self.config, "simulation_duplicate_family_throttle_score_gap", 3.0) or 3.0)
        for existing in list(provisional_signals or []):
            existing_symbol = str(existing.get("symbol", "") or "")
            existing_strategy = str(existing.get("strategy", "unknown") or "unknown")
            existing_side = str(existing.get("side", "long") or "long").lower()
            existing_family = self.risk._family_for_symbol(existing_symbol)
            if existing_family != family or existing_side != side or existing_strategy != strategy:
                continue
            existing_score = self._candidate_rank_score(existing, provisional_signals=[])
            if candidate_rank_score <= (existing_score - score_gap):
                return "portfolio_duplicate_family_throttle"
        return None

    def _portfolio_duplicate_bucket_throttle_reason(
        self,
        signal: Dict[str, Any],
        *,
        provisional_signals: Sequence[Dict[str, Any]] | None = None,
        candidate_rank_score: float,
    ) -> str | None:
        symbol = str(signal.get("symbol", "") or "")
        strategy = str(signal.get("strategy", "unknown") or "unknown")
        side = str(signal.get("side", "long") or "long").lower()
        bucket = self._symbol_bucket(symbol)
        score_gap = float(getattr(self.config, "simulation_duplicate_bucket_throttle_score_gap", 2.5) or 2.5)
        for existing in list(provisional_signals or []):
            existing_symbol = str(existing.get("symbol", "") or "")
            existing_strategy = str(existing.get("strategy", "unknown") or "unknown")
            existing_side = str(existing.get("side", "long") or "long").lower()
            existing_bucket = self._symbol_bucket(existing_symbol)
            if existing_bucket != bucket or existing_side != side or existing_strategy != strategy:
                continue
            existing_score = self._candidate_rank_score(existing, provisional_signals=[])
            if candidate_rank_score <= (existing_score - score_gap):
                return "portfolio_duplicate_bucket_throttle"
        return None

    def _portfolio_persistently_weak_cluster_reason(
        self,
        signal: Dict[str, Any],
        *,
        provisional_signals: Sequence[Dict[str, Any]] | None = None,
    ) -> str | None:
        symbol = str(signal.get("symbol", "") or "")
        side = str(signal.get("side", "long") or "long").lower()
        normalized_side = "long" if side in {"buy", "long"} else "short"
        family = self.risk._family_for_symbol(symbol)
        min_trades = max(int(getattr(self.config, "simulation_weak_cluster_min_trades", 2) or 2), 1)
        negative_expectancy_floor = float(getattr(self.config, "simulation_weak_cluster_negative_expectancy_floor", -1.0) or -1.0)

        family_side_trades = [
            trade
            for trade in self.trades
            if isinstance(trade, dict)
            and self.risk._family_for_symbol(str(trade.get("symbol", "") or "")) == family
            and str(trade.get("side", "long") or "long").lower() == normalized_side
        ]
        if len(family_side_trades) < min_trades:
            return None
        expectancy = sum(float(trade.get("pl", 0.0) or 0.0) for trade in family_side_trades) / max(len(family_side_trades), 1)
        if expectancy > negative_expectancy_floor:
            return None

        for provisional in list(provisional_signals or []):
            provisional_symbol = str(provisional.get("symbol", "") or "")
            provisional_side = str(provisional.get("side", "long") or "long").lower()
            if self.risk._family_for_symbol(provisional_symbol) == family and provisional_side == normalized_side:
                return "portfolio_persistently_weak_cluster_throttle"

        for existing_symbol, current in dict(self.state.open_positions).items():
            positions = current if isinstance(current, list) else [current]
            for position in positions:
                if self.risk._family_for_symbol(str(existing_symbol or "")) == family and str(getattr(position, "side", "long") or "long").lower() == normalized_side:
                    return "portfolio_persistently_weak_cluster_throttle"
        if self.venue is not None:
            for order in self.venue.open_orders:
                if self.risk._family_for_symbol(str(order.symbol or "")) == family and str(order.side or "long").lower() in {normalized_side, side}:
                    return "portfolio_persistently_weak_cluster_throttle"
        return None

    def _realized_performance_score_bias(self, signal: Dict[str, Any]) -> float:
        details = self._realized_performance_penalty_details(signal)
        return float(details.get("score_bonus", 0.0) or 0.0) - float(details.get("score_penalty", 0.0) or 0.0)

    def _portfolio_no_trade_reason(self, signal: Dict[str, Any]) -> str | None:
        metadata = dict(signal.get("metadata", {}) or {})
        expected_edge_bps = float(signal.get("expected_edge_bps", 0.0) or 0.0)
        if expected_edge_bps <= 0.0:
            return None
        min_edge_bps = float(getattr(self.config, "min_expected_edge_bps", 8.0) or 8.0)
        spread_bps = float(metadata.get("spread_bps", getattr(self.config, "backtest_spread_bps", 4.0)) or getattr(self.config, "backtest_spread_bps", 4.0))
        buffer_bps = float(getattr(self.config, "simulation_no_trade_buffer_bps", 4.0) or 4.0)
        symbol = str(signal.get("symbol", "") or "")
        strategy = str(signal.get("strategy", "unknown") or "unknown")
        family = self.risk._family_for_symbol(symbol)
        exposures = self.risk.portfolio_exposures()
        balance = max(float(exposures.get("balance", getattr(self.state, "balance", 0.0)) or 0.0), 1e-9)
        family_used_fraction = float((dict(exposures.get("by_family", {}) or {}).get(family, 0.0) or 0.0) / balance)
        strategy_used_fraction = float((dict(exposures.get("by_strategy", {}) or {}).get(strategy, 0.0) or 0.0) / balance)
        crowding_penalty_bps = float(getattr(self.config, "simulation_no_trade_crowding_penalty_bps", 4.0) or 4.0)
        crowding_penalty = (family_used_fraction * crowding_penalty_bps) + (strategy_used_fraction * (crowding_penalty_bps * 0.5))
        directional_cluster_penalty = (
            self._directional_cluster_crowding_penalty(signal) / max(float(getattr(self.config, "simulation_directional_cluster_penalty", 6.0) or 6.0), 1e-9)
        ) * (crowding_penalty_bps * 0.5)
        realized_penalty = self._realized_performance_no_trade_penalty(signal)
        # Treat expected_edge_bps as already partially net-aware; apply a lighter friction buffer here.
        net_edge_bps = expected_edge_bps - (spread_bps * 0.5) - buffer_bps - crowding_penalty - directional_cluster_penalty - realized_penalty
        if net_edge_bps < min_edge_bps:
            missed_adjustment = self.learning.missed_opportunity_gate_adjustment(signal, "portfolio_no_trade_region")
            relax_bps = float(missed_adjustment.get("relax_bps", 0.0) or 0.0)
            if bool(missed_adjustment.get("active", False)) and (net_edge_bps + relax_bps) >= min_edge_bps:
                metadata = dict(signal.get("metadata", {}) or {})
                metadata["missed_opportunity_adjustment"] = missed_adjustment
                signal["metadata"] = metadata
                strategy = str(signal.get("strategy", "unknown") or "unknown")
                reason = str(missed_adjustment.get("reason", "portfolio_no_trade_region") or "portfolio_no_trade_region")
                self._increment_counter(self._missed_opportunity_relaxations, reason)
                self._increment_nested_counter(self._missed_opportunity_relaxations_by_strategy, strategy, reason)
                return None
            return "portfolio_no_trade_region"
        return None

    def _realized_performance_no_trade_penalty(self, signal: Dict[str, Any]) -> float:
        details = self._realized_performance_penalty_details(signal)
        return float(details.get("no_trade_penalty_bps", 0.0) or 0.0)

    def _realized_performance_penalty_details(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        cached = dict((dict(signal.get("metadata", {}) or {}).get("realized_performance_penalty", {}) or {}))
        if cached:
            return cached
        symbol = str(signal.get("symbol", "") or "")
        strategy = str(signal.get("strategy", "unknown") or "unknown")
        min_trades = max(int(getattr(self.config, "simulation_realized_perf_min_trades", 2) or 2), 1)
        negative_expectancy_floor = float(getattr(self.config, "simulation_realized_negative_expectancy_floor", -1.0) or -1.0)
        positive_expectancy_floor = float(getattr(self.config, "simulation_realized_positive_expectancy_floor", 1.0) or 1.0)
        penalty_bps = float(getattr(self.config, "simulation_realized_no_trade_penalty_bps", 3.0) or 3.0)
        score_penalty = 0.0
        score_bonus = 0.0
        no_trade_penalty = 0.0
        reasons: list[str] = []
        symbol_penalty = float(getattr(self.config, "simulation_realized_symbol_penalty_score", 5.0) or 5.0)
        strategy_penalty = float(getattr(self.config, "simulation_realized_strategy_penalty_score", 4.0) or 4.0)
        strategy_symbol_penalty = float(getattr(self.config, "simulation_strategy_symbol_penalty_score", 4.0) or 4.0)
        symbol_bonus = float(getattr(self.config, "simulation_realized_symbol_positive_score", 2.5) or 2.5)
        strategy_bonus = float(getattr(self.config, "simulation_realized_strategy_positive_score", 2.0) or 2.0)
        pullback_symbol_penalty = float(getattr(self.config, "simulation_pullback_realized_symbol_penalty_score", 3.0) or 3.0)
        pullback_no_trade_penalty = float(getattr(self.config, "simulation_pullback_realized_symbol_no_trade_penalty_bps", 2.0) or 2.0)
        strategy_symbol_veto_min_trades = max(int(getattr(self.config, "simulation_strategy_symbol_veto_min_trades", 2) or 2), 1)

        symbol_trades = int(self._closed_by_symbol.get(symbol, 0) or 0)
        if symbol_trades >= min_trades:
            symbol_rollup = self._build_symbol_rollups().get(symbol, {})
            symbol_expectancy = float(dict(symbol_rollup or {}).get("expectancy", 0.0) or 0.0)
            if symbol_expectancy <= negative_expectancy_floor:
                score_penalty += symbol_penalty
                no_trade_penalty += penalty_bps
                reasons.append("symbol_negative_expectancy")
                if strategy == "trend_pullback":
                    score_penalty += pullback_symbol_penalty
                    no_trade_penalty += pullback_no_trade_penalty
                    reasons.append("pullback_symbol_negative_expectancy")
            elif symbol_expectancy >= positive_expectancy_floor:
                score_bonus += symbol_bonus
                reasons.append("symbol_positive_expectancy")

        strategy_symbol_rollup = self._build_strategy_symbol_rollups().get((symbol, strategy), {})
        strategy_symbol_trades = int(dict(strategy_symbol_rollup or {}).get("trades", 0) or 0)
        strategy_symbol_expectancy = float(dict(strategy_symbol_rollup or {}).get("expectancy", 0.0) or 0.0)
        if strategy_symbol_trades >= strategy_symbol_veto_min_trades:
            floor_by_strategy = {
                "trend_pullback": float(getattr(self.config, "simulation_pullback_symbol_veto_expectancy_floor", -18.0) or -18.0),
                "trend_breakout": float(getattr(self.config, "simulation_breakout_symbol_veto_expectancy_floor", -14.0) or -14.0),
                "mean_reversion": float(getattr(self.config, "simulation_mean_reversion_symbol_veto_expectancy_floor", -12.0) or -12.0),
            }
            floor = float(floor_by_strategy.get(strategy, negative_expectancy_floor) or negative_expectancy_floor)
            if strategy_symbol_expectancy <= floor:
                score_penalty += strategy_symbol_penalty
                no_trade_penalty += penalty_bps
                reasons.append("strategy_symbol_negative_expectancy")

        strategy_trades = int(self._closed_by_strategy.get(strategy, 0) or 0)
        if strategy_trades >= min_trades:
            strategy_summary = self._build_realized_performance_summary({}).get("by_strategy", {}).get(strategy, {})
            strategy_expectancy = float(dict(strategy_summary or {}).get("expectancy", 0.0) or 0.0)
            if strategy_expectancy <= negative_expectancy_floor:
                score_penalty += strategy_penalty
                no_trade_penalty += penalty_bps * 0.75
                reasons.append("strategy_negative_expectancy")
            elif strategy_expectancy >= positive_expectancy_floor:
                score_bonus += strategy_bonus
                reasons.append("strategy_positive_expectancy")
        return {
            "score_bonus": score_bonus,
            "score_penalty": score_penalty,
            "no_trade_penalty_bps": no_trade_penalty,
            "reasons": reasons,
        }

    def _build_strategy_symbol_rollups(self) -> Dict[tuple[str, str], Dict[str, Any]]:
        rollups: Dict[tuple[str, str], Dict[str, Any]] = {}
        for trade in self.trades:
            if not isinstance(trade, dict):
                continue
            symbol = str(trade.get("symbol", "") or "")
            strategy = str(trade.get("strategy", "unknown") or "unknown")
            if not symbol:
                continue
            payload = rollups.setdefault(
                (symbol, strategy),
                {"trades": 0, "wins": 0, "losses": 0, "total_pl": 0.0, "expectancy": 0.0},
            )
            payload["trades"] += 1
            pl = float(trade.get("pl", 0.0) or 0.0)
            payload["total_pl"] += pl
            if pl >= 0.0:
                payload["wins"] += 1
            else:
                payload["losses"] += 1
        for payload in rollups.values():
            trades = int(payload.get("trades", 0) or 0)
            payload["expectancy"] = float(payload.get("total_pl", 0.0) or 0.0) / trades if trades else 0.0
            payload["win_rate_pct"] = (int(payload.get("wins", 0) or 0) / trades * 100.0) if trades else 0.0
        return rollups

    def _record_skipped_signal(self, signal: Dict[str, Any], reason: str, trace_id: str) -> None:
        self.learning.record_shadow_decision(
            signal,
            status="skipped",
            reason=reason,
            trace_id=trace_id,
            created_at=self._now.isoformat() if self._now else None,
        )
        self._skipped_signals += 1
        self._increment_counter(self._skip_reason_counts, reason)
        symbol = str(signal.get("symbol", "") or "")
        if symbol:
            self._increment_counter(self._skipped_by_symbol, symbol)
            symbol_reasons = self._skip_reasons_by_symbol.setdefault(symbol, {})
            symbol_reasons[reason] = int(symbol_reasons.get(reason, 0)) + 1
            strategy = str(signal.get("strategy", "unknown") or "unknown")
            self._increment_nested_reason_counter(
                self._skip_reasons_by_strategy_by_symbol,
                symbol,
                strategy,
                reason,
            )

    def _replacement_guard_reason(self, signal: Dict[str, Any]) -> str | None:
        metadata = dict(signal.get("metadata", {}) or {})
        replacement = dict(metadata.get("replacement_opportunity", {}) or {})
        if not bool(replacement.get("active", False)):
            return None
        day_key = self._now.date().isoformat() if self._now is not None else "unknown"
        max_per_day = max(int(getattr(self.config, "replacement_max_per_day", 2) or 0), 0)
        if max_per_day > 0 and int(self._replacement_submitted_by_day.get(day_key, 0) or 0) >= max_per_day:
            return "replacement_daily_cap"
        symbol = str(signal.get("symbol", "") or "")
        max_per_symbol = max(int(getattr(self.config, "replacement_max_per_symbol_per_day", 1) or 0), 0)
        if max_per_symbol > 0 and symbol:
            symbol_counts = dict(self._replacement_submitted_by_symbol_day.get(day_key, {}) or {})
            if int(symbol_counts.get(symbol, 0) or 0) >= max_per_symbol:
                return "replacement_symbol_daily_cap"
        return None

    def _register_replacement_submission(self, signal: Dict[str, Any]) -> None:
        metadata = dict(signal.get("metadata", {}) or {})
        replacement = dict(metadata.get("replacement_opportunity", {}) or {})
        if not bool(replacement.get("active", False)):
            return
        self._replacement_submitted += 1
        self._increment_counter(self._replacement_submitted_by_strategy, str(signal.get("strategy", "unknown") or "unknown"))
        day_key = self._now.date().isoformat() if self._now is not None else "unknown"
        self._replacement_submitted_by_day[day_key] = int(self._replacement_submitted_by_day.get(day_key, 0) or 0) + 1
        symbol = str(signal.get("symbol", "") or "")
        if symbol:
            symbol_counts = self._replacement_submitted_by_symbol_day.setdefault(day_key, {})
            symbol_counts[symbol] = int(symbol_counts.get(symbol, 0) or 0) + 1

    def _eligible_universe_symbols(self, symbols: Sequence[str]) -> list[str]:
        scored = self._build_universe_tradability_snapshot(symbols)
        base_top_n = max(int(getattr(self.config, "simulation_universe_top_n", 0) or 0), 0)
        expansion_gate = self._frequency_expansion_gate()
        top_n = base_top_n
        if bool(expansion_gate.get("allowed", False)):
            top_n += max(int(getattr(self.config, "frequency_expansion_universe_top_n_delta", 2) or 2), 0)
        floor = float(getattr(self.config, "simulation_universe_tradability_floor", 45.0) or 45.0)
        regime_state = self._current_universe_regime_state()
        ranked = sorted(scored.items(), key=lambda item: float(dict(item[1] or {}).get("tradability_score", 0.0) or 0.0), reverse=True)
        eligible: list[str] = []
        rejected: Dict[str, Dict[str, Any]] = {}
        eligible_buckets: Dict[str, int] = {}
        reserved_core_symbols: set[str] = set()
        if getattr(self.config, "trading_mode", "spot") == "spot" and bool(getattr(self.config, "simulation_spot_core_symbol_reservation_enabled", True)):
            core_symbols = {
                str(symbol)
                for symbol in list(getattr(self.config, "simulation_spot_core_symbols", []) or [])
                if str(symbol) in scored
            }
            core_min_count = max(int(getattr(self.config, "simulation_spot_core_symbol_min_count", 2) or 2), 0)
            for symbol, payload in ranked:
                if len(reserved_core_symbols) >= core_min_count:
                    break
                if symbol not in core_symbols:
                    continue
                score = float(dict(payload or {}).get("tradability_score", 0.0) or 0.0)
                if score < floor:
                    continue
                if bool(dict(payload or {}).get("probation_veto_active", False)) or bool(dict(payload or {}).get("realized_veto_active", False)):
                    continue
                reserved_core_symbols.add(symbol)
        for index, (symbol, payload) in enumerate(ranked):
            score = float(dict(payload or {}).get("tradability_score", 0.0) or 0.0)
            bucket = str(dict(payload or {}).get("bucket", "other") or "other")
            bucket_cap = self._universe_bucket_cap(bucket, regime_state=regime_state)
            core_reserved = symbol in reserved_core_symbols
            if bool(dict(payload or {}).get("probation_veto_active", False)):
                rejected[symbol] = {**dict(payload or {}), "reason": "realized_symbol_probation_veto"}
                continue
            if bool(dict(payload or {}).get("realized_veto_active", False)):
                rejected[symbol] = {**dict(payload or {}), "reason": "realized_negative_expectancy_veto"}
                continue
            if score < floor:
                rejected[symbol] = {**dict(payload or {}), "reason": "below_tradability_floor"}
                continue
            if top_n > 0 and len(eligible) >= top_n and not core_reserved:
                rejected[symbol] = {**dict(payload or {}), "reason": "outside_top_n"}
                continue
            if int(eligible_buckets.get(bucket, 0)) >= bucket_cap and not core_reserved:
                rejected[symbol] = {**dict(payload or {}), "reason": "bucket_cap_reached"}
                continue
            eligible.append(symbol)
            eligible_buckets[bucket] = int(eligible_buckets.get(bucket, 0)) + 1
        self._latest_universe_selection = {
            "eligible_symbols": eligible,
            "eligible_bucket_counts": dict(sorted(eligible_buckets.items())),
            "reserved_core_symbols": sorted(reserved_core_symbols),
            "regime_state": dict(sorted(regime_state.items())),
            "frequency_expansion": dict(sorted(expansion_gate.items())),
            "rejected_symbols": {key: dict(sorted(value.items())) for key, value in sorted(rejected.items())},
            "scored_symbols": {key: dict(sorted(value.items())) for key, value in sorted(scored.items())},
        }
        return eligible

    def _current_universe_regime_state(self) -> Dict[str, Any]:
        if self.exchange is None or not self._simulation_symbol_universe:
            return {"state": "neutral", "avg_range": 0.0, "trend_strength": 0.0}
        primary_symbol = str(self._simulation_symbol_universe[0] or "")
        lookback = max(int(getattr(self.config, "simulation_universe_regime_lookback_bars", 20) or 20), 5)
        candles = list(self.exchange.fetch_ohlcv(primary_symbol, self._base_timeframe, limit=lookback) or [])
        if len(candles) < 2:
            return {"state": "neutral", "avg_range": 0.0, "trend_strength": 0.0}
        ranges = []
        closes = []
        for candle in candles:
            high = float(candle[2] or 0.0)
            low = float(candle[3] or 0.0)
            close = max(float(candle[4] or 0.0), 1e-9)
            ranges.append(max(high - low, 0.0) / close)
            closes.append(close)
        avg_range = float(sum(ranges) / len(ranges)) if ranges else 0.0
        trend_strength = abs((closes[-1] / max(closes[0], 1e-9)) - 1.0) if len(closes) >= 2 else 0.0
        high_vol_threshold = float(getattr(self.config, "simulation_universe_regime_high_vol_threshold", 0.02) or 0.02)
        trend_threshold = float(getattr(self.config, "simulation_universe_regime_trend_strength_threshold", 0.015) or 0.015)
        if avg_range >= high_vol_threshold:
            state = "risk_off"
        elif trend_strength >= trend_threshold:
            state = "trend_supportive"
        else:
            state = "neutral"
        return {
            "state": state,
            "avg_range": avg_range,
            "trend_strength": trend_strength,
            "symbol": primary_symbol,
        }

    def _frequency_expansion_gate(self) -> Dict[str, Any]:
        if not bool(getattr(self.config, "frequency_expansion_enabled", True)):
            return {"allowed": False, "reason": "disabled"}
        if self._campaign_timeline_start is None or self._now is None:
            return {"allowed": False, "reason": "frequency_window_unavailable"}
        window_days = max((self._now - self._campaign_timeline_start).total_seconds() / 86400.0, 1.0 / 96.0)
        trades_per_day = len(self.trades) / max(window_days, 1e-9)
        target_min = float(getattr(self.config, "target_trades_per_day_min", 2.0) or 2.0)
        if trades_per_day >= target_min:
            return {
                "allowed": False,
                "reason": "frequency_target_already_met",
                "trades_per_day": trades_per_day,
                "target_trades_per_day_min": target_min,
            }
        closed_trades = len(self.trades)
        min_trades = max(int(getattr(self.config, "frequency_expansion_min_closed_trades", 3) or 3), 1)
        if closed_trades < min_trades:
            return {"allowed": False, "reason": "not_enough_closed_trades", "closed_trades": closed_trades, "required_trades": min_trades}

        negative_trades = [trade for trade in self.trades if float(dict(trade or {}).get("pl", 0.0) or 0.0) < 0.0]
        negative_total_pl = abs(sum(float(trade.get("pl", 0.0) or 0.0) for trade in negative_trades))
        stop_loss_negative_pl = abs(
            sum(
                float(trade.get("pl", 0.0) or 0.0)
                for trade in negative_trades
                if str(trade.get("exit_reason", "UNKNOWN") or "UNKNOWN") == "SL"
            )
        )
        stop_loss_share = (stop_loss_negative_pl / negative_total_pl * 100.0) if negative_total_pl > 0.0 else 0.0
        max_sl_share = float(getattr(self.config, "frequency_expansion_max_stop_loss_negative_pl_share_pct", 55.0) or 55.0)
        if stop_loss_share > max_sl_share:
            return {
                "allowed": False,
                "reason": "stop_loss_damage_too_high",
                "closed_trades": closed_trades,
                "stop_loss_negative_pl_share_pct": stop_loss_share,
                "max_stop_loss_negative_pl_share_pct": max_sl_share,
            }

        by_strategy = self._build_realized_performance_summary(self._build_symbol_rollups()).get("by_strategy", {})
        best_expectancy = max((float(dict(payload or {}).get("expectancy", 0.0) or 0.0) for payload in dict(by_strategy or {}).values()), default=0.0)
        min_expectancy = float(getattr(self.config, "frequency_expansion_min_best_strategy_expectancy", 0.0) or 0.0)
        if best_expectancy < min_expectancy:
            return {
                "allowed": False,
                "reason": "no_positive_strategy_expectancy",
                "closed_trades": closed_trades,
                "best_strategy_expectancy": best_expectancy,
                "required_expectancy": min_expectancy,
            }
        return {
            "allowed": True,
            "reason": "exit_repair_confirmed",
            "closed_trades": closed_trades,
            "trades_per_day": trades_per_day,
            "stop_loss_negative_pl_share_pct": stop_loss_share,
            "best_strategy_expectancy": best_expectancy,
        }

    def _universe_bucket_cap(self, bucket: str, *, regime_state: Dict[str, Any] | None = None) -> int:
        normalized = str(bucket or "other")
        specific = getattr(self.config, f"simulation_universe_bucket_cap_{normalized}", None)
        regime_state = dict(regime_state or {})
        if specific is not None:
            base_cap = max(int(specific or 0), 0)
        else:
            base_cap = max(int(getattr(self.config, "simulation_universe_bucket_cap", 0) or 0), 0)
        state = str(regime_state.get("state", "neutral") or "neutral")
        adjusted = base_cap
        if state == "trend_supportive":
            if normalized == "exchange_beta":
                adjusted += int(getattr(self.config, "simulation_universe_regime_trend_exchange_beta_delta", 1) or 1)
            elif normalized == "high_beta_alts":
                adjusted += int(getattr(self.config, "simulation_universe_regime_trend_high_beta_delta", 1) or 1)
        elif state == "risk_off":
            if normalized == "exchange_beta":
                adjusted += int(getattr(self.config, "simulation_universe_regime_risk_off_exchange_beta_delta", -1) or -1)
            elif normalized == "high_beta_alts":
                adjusted += int(getattr(self.config, "simulation_universe_regime_risk_off_high_beta_delta", -1) or -1)
            elif normalized == "slower_large_caps":
                adjusted += int(getattr(self.config, "simulation_universe_regime_risk_off_slower_large_caps_delta", -1) or -1)
            elif normalized == "other":
                adjusted += int(getattr(self.config, "simulation_universe_regime_risk_off_other_delta", -1) or -1)
        return max(adjusted, 0)

    def _build_universe_tradability_snapshot(self, symbols: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        assert self.exchange is not None
        snapshot: Dict[str, Dict[str, Any]] = {}
        spread_weight = float(getattr(self.config, "simulation_universe_spread_penalty_weight", 1.4) or 1.4)
        vol_weight = float(getattr(self.config, "simulation_universe_volatility_penalty_weight", 60.0) or 60.0)
        volume_weight = float(getattr(self.config, "simulation_universe_volume_score_weight", 4.0) or 4.0)
        realized_min_trades = max(int(getattr(self.config, "simulation_universe_realized_min_trades", 2) or 2), 1)
        negative_expectancy_floor = float(getattr(self.config, "simulation_universe_realized_negative_expectancy_floor", -1.0) or -1.0)
        positive_expectancy_floor = float(getattr(self.config, "simulation_universe_realized_positive_expectancy_floor", 1.0) or 1.0)
        realized_penalty = float(getattr(self.config, "simulation_universe_realized_penalty_score", 10.0) or 10.0)
        realized_boost = float(getattr(self.config, "simulation_universe_realized_boost_score", 5.0) or 5.0)
        realized_veto_min_trades = max(int(getattr(self.config, "simulation_universe_realized_veto_min_trades", 2) or 2), 1)
        realized_veto_floor = float(getattr(self.config, "simulation_universe_realized_veto_expectancy_floor", -8.0) or -8.0)
        probation_veto_min_trades = max(int(getattr(self.config, "simulation_symbol_probation_veto_min_trades", 1) or 1), 1)
        probation_veto_floor = float(getattr(self.config, "simulation_symbol_probation_veto_expectancy_floor", -20.0) or -20.0)
        expectancy_half_life = max(float(getattr(self.config, "simulation_symbol_expectancy_decay_half_life_trades", 6.0) or 6.0), 1.0)
        symbol_rollups = self._build_symbol_rollups()
        strategy_symbol_rollups = self._build_strategy_symbol_rollups()
        for symbol in symbols:
            bar = self.exchange.current_bar(symbol, self._base_timeframe)
            if bar is None:
                snapshot[symbol] = {"tradability_score": 0.0, "reason": "no_bar"}
                continue
            close = max(float(bar.get("close", 0.0) or 0.0), 1e-9)
            high = float(bar.get("high", close) or close)
            low = float(bar.get("low", close) or close)
            volume = max(float(bar.get("volume", 0.0) or 0.0), 0.0)
            spread_bps = float(self.exchange.estimate_spread_fraction(symbol) * 10000.0)
            realized_volatility = max(high - low, 0.0) / close
            volume_score = math.log10(volume + 1.0) * volume_weight
            realized_rollup = dict(symbol_rollups.get(symbol, {}) or {})
            realized_trades = int(realized_rollup.get("trades", 0) or 0)
            realized_expectancy = float(realized_rollup.get("expectancy", 0.0) or 0.0)
            expectancy_memory_weight = 1.0 - (0.5 ** (realized_trades / expectancy_half_life)) if realized_trades > 0 else 0.0
            realized_score_adjustment = 0.0
            realized_adjustment_reason = ""
            realized_veto_active = False
            probation_veto_active = False
            if realized_trades >= realized_min_trades:
                if realized_expectancy <= negative_expectancy_floor:
                    realized_score_adjustment = -realized_penalty * expectancy_memory_weight
                    realized_adjustment_reason = "negative_expectancy"
                elif realized_expectancy >= positive_expectancy_floor:
                    realized_score_adjustment = realized_boost * expectancy_memory_weight
                    realized_adjustment_reason = "positive_expectancy"
            strategy_rotation = self._universe_strategy_rotation_adjustment(symbol, strategy_symbol_rollups)
            strategy_rotation_adjustment = float(strategy_rotation.get("score_adjustment", 0.0) or 0.0)
            strategy_positive_override = bool(strategy_rotation.get("positive_override", False))
            if realized_trades >= probation_veto_min_trades and realized_expectancy <= probation_veto_floor:
                probation_veto_active = True
            if realized_trades >= realized_veto_min_trades and realized_expectancy <= realized_veto_floor:
                realized_veto_active = True
            if strategy_positive_override:
                probation_veto_active = False
                realized_veto_active = False
            tradability_score = max(
                0.0,
                100.0
                + volume_score
                - (spread_bps * spread_weight)
                - (realized_volatility * vol_weight)
                + realized_score_adjustment
                + strategy_rotation_adjustment,
            )
            snapshot[symbol] = {
                "tradability_score": tradability_score,
                "spread_bps": spread_bps,
                "realized_volatility": realized_volatility,
                "volume": volume,
                "bucket": self._symbol_bucket(symbol),
                "realized_trades": realized_trades,
                "realized_expectancy": realized_expectancy,
                "expectancy_memory_weight": expectancy_memory_weight,
                "realized_score_adjustment": realized_score_adjustment,
                "realized_adjustment_reason": realized_adjustment_reason,
                "strategy_rotation_score_adjustment": strategy_rotation_adjustment,
                "strategy_rotation_positive_override": strategy_positive_override,
                "strategy_rotation": dict(strategy_rotation.get("by_strategy", {}) or {}),
                "probation_veto_active": probation_veto_active,
                "realized_veto_active": realized_veto_active,
            }
        return snapshot

    def _universe_strategy_rotation_adjustment(
        self,
        symbol: str,
        strategy_symbol_rollups: Dict[tuple[str, str], Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not bool(getattr(self.config, "simulation_universe_strategy_rotation_enabled", True)):
            return {"score_adjustment": 0.0, "positive_override": False, "by_strategy": {}}
        min_trades = max(int(getattr(self.config, "simulation_universe_strategy_rotation_min_trades", 2) or 2), 1)
        positive_floor = float(getattr(self.config, "simulation_universe_strategy_rotation_positive_floor", 2.0) or 2.0)
        negative_floor = float(getattr(self.config, "simulation_universe_strategy_rotation_negative_floor", -8.0) or -8.0)
        boost_score = float(getattr(self.config, "simulation_universe_strategy_rotation_boost_score", 7.0) or 7.0)
        penalty_score = float(getattr(self.config, "simulation_universe_strategy_rotation_penalty_score", 4.0) or 4.0)
        by_strategy: Dict[str, Dict[str, Any]] = {}
        total_adjustment = 0.0
        positive_override = False
        for (rollup_symbol, strategy), payload in sorted(strategy_symbol_rollups.items()):
            if rollup_symbol != symbol:
                continue
            trades = int(dict(payload or {}).get("trades", 0) or 0)
            expectancy = float(dict(payload or {}).get("expectancy", 0.0) or 0.0)
            if trades < min_trades:
                continue
            memory_weight = 1.0 - (0.5 ** (trades / max(float(getattr(self.config, "simulation_symbol_expectancy_decay_half_life_trades", 6.0) or 6.0), 1.0)))
            adjustment = 0.0
            state = "neutral"
            if expectancy >= positive_floor:
                adjustment = boost_score * memory_weight
                positive_override = True
                state = "positive_rotation"
            elif expectancy <= negative_floor:
                adjustment = -penalty_score * memory_weight
                state = "negative_rotation"
            if state == "neutral":
                continue
            by_strategy[strategy] = {
                "trades": trades,
                "expectancy": expectancy,
                "memory_weight": memory_weight,
                "score_adjustment": adjustment,
                "state": state,
            }
            total_adjustment += adjustment
        return {
            "score_adjustment": total_adjustment,
            "positive_override": positive_override,
            "by_strategy": by_strategy,
        }

    def _simulation_strategy_probation_reason(self, signal: Dict[str, Any]) -> str | None:
        strategy = str(signal.get("strategy", "unknown") or "unknown")
        if strategy != "trend_pullback":
            return None
        if bool(getattr(self.config, "simulation_disable_trend_pullback", False)):
            return "trend_pullback_disabled_in_simulation"
        min_trades = max(int(getattr(self.config, "simulation_pullback_strategy_veto_min_trades", 2) or 2), 1)
        expectancy_floor = float(getattr(self.config, "simulation_pullback_strategy_veto_expectancy_floor", -20.0) or -20.0)
        strategy_trades = int(self._closed_by_strategy.get(strategy, 0) or 0)
        if strategy_trades < min_trades:
            return None
        strategy_summary = self._build_realized_performance_summary({}).get("by_strategy", {}).get(strategy, {})
        strategy_expectancy = float(dict(strategy_summary or {}).get("expectancy", 0.0) or 0.0)
        if strategy_expectancy <= expectancy_floor:
            return "pullback_strategy_probation_veto"
        return None

    def _reentry_cooldown_key(self, symbol: str, strategy: str, side: str) -> str:
        normalized_side = "long" if str(side).lower() in {"buy", "long"} else "short"
        return f"{symbol}::{strategy}::{normalized_side}"

    def _register_reentry_cooldown(self, position: Position, exit_reason: str, now: dt.datetime) -> None:
        strategy = str(getattr(position, "strategy", "unknown") or "unknown").lower()
        side = str(getattr(position, "side", "long") or "long").lower()
        symbol = str(getattr(position, "symbol", "") or "")
        cooldown_bars = 0
        reason = ""
        if exit_reason == "MEAN_REVERSION_RECLAIM_FAIL":
            cooldown_bars = max(int(getattr(self.config, "mean_reversion_reclaim_failure_cooldown_bars", 6) or 6), 0)
            reason = "mean_reversion_reclaim_failure_cooldown"
        elif exit_reason == "SL" and strategy == "trend_breakout":
            metadata = dict(getattr(position, "metadata", {}) or {})
            if bool(metadata.get("volatility_tightened", False)):
                cooldown_bars = max(int(getattr(self.config, "breakout_volatility_exit_cooldown_bars", 8) or 8), 0)
                reason = "breakout_volatility_exit_cooldown"
        if cooldown_bars <= 0 or not symbol:
            return
        self._reentry_cooldowns[self._reentry_cooldown_key(symbol, strategy, side)] = {
            "until": now + dt.timedelta(minutes=timeframe_to_minutes(self._base_timeframe) * cooldown_bars),
            "reason": reason,
        }
        self._increment_counter(self._reentry_cooldown_registrations, reason)
        self._increment_nested_counter(self._reentry_cooldown_registrations_by_strategy, strategy, reason)

    def _reentry_cooldown_reason(self, signal: Dict[str, Any]) -> str | None:
        symbol = str(signal.get("symbol", "") or "")
        strategy = str(signal.get("strategy", "unknown") or "unknown").lower()
        side = str(signal.get("side", "long") or "long").lower()
        cooldown = dict(self._reentry_cooldowns.get(self._reentry_cooldown_key(symbol, strategy, side), {}) or {})
        if not cooldown:
            return None
        until = cooldown.get("until")
        if isinstance(until, dt.datetime) and self._now is not None and self._now < until:
            return str(cooldown.get("reason", "reentry_cooldown") or "reentry_cooldown")
        return None

    def _same_side_already_active(self, symbol: str, side: str) -> bool:
        normalized_side = "long" if side in {"buy", "long"} else "short"
        existing = self.state.open_positions.get(symbol, [])
        positions = existing if isinstance(existing, list) else [existing]
        for position in positions:
            if getattr(position, "side", "") == normalized_side:
                return True
        assert self.venue is not None
        for order in self.venue.open_orders:
            if order.symbol == symbol and order.side in {side, normalized_side}:
                return True
        return False

    def _apply_fills(self, fills: Sequence[tuple[SimulatedOrder, Fill, Dict[str, Any]]]) -> None:
        for order, fill, details in fills:
            self._filled_orders += 1
            self._increment_counter(self._filled_by_strategy, str(order.strategy))
            self._increment_counter(self._filled_by_symbol, order.symbol)
            if bool(dict((order.signal or {}).get("metadata", {}) or {}).get("replacement_opportunity", {}).get("active", False)):
                self._replacement_filled += 1
                self._increment_counter(self._replacement_filled_by_strategy, str(order.strategy))
            self._increment_nested_counter(
                self._filled_by_strategy_by_symbol,
                order.symbol,
                str(order.strategy),
            )
            fill_fraction = float(fill.metadata.get("fill_fraction", 1.0) or 1.0)
            if fill_fraction < 0.999:
                self._partial_fills += 1
            self.state.balance -= float(fill.fee)
            execution_context = {
                "spread_bps": float(fill.metadata.get("spread_bps", 0.0) or 0.0),
                "entry_deviation_bps": self._entry_deviation_bps(order.requested_price, fill.price),
                "fill_fraction": max(order.filled_size / max(order.requested_size, 1e-9), fill_fraction),
                "latency_ms": float(fill.metadata.get("latency_ms", 0.0) or 0.0),
                "slippage_bps": float(fill.metadata.get("slippage_bps", 0.0) or 0.0),
                "market_impact_bps": float(fill.metadata.get("market_impact_bps", 0.0) or 0.0),
                "adverse_selection_bps": float(fill.metadata.get("adverse_selection_bps", 0.0) or 0.0),
                "order_type": order.order_type,
            }
            side = "long" if order.side in {"long", "buy"} else "short"
            if order.position_ref is None:
                position = self.position_cls(
                    symbol=order.symbol,
                    side=side,
                    entry_price=float(fill.price),
                    size=float(fill.size),
                    stop_loss=float(order.stop_loss),
                    take_profit=float(order.take_profit),
                    strategy=str(order.strategy),
                    opened_at=fill.filled_at,
                    initial_stop_loss=float(order.stop_loss),
                    initial_take_profit=float(order.take_profit),
                    fee_paid=float(fill.fee),
                    metadata={},
                )
                order.position_ref = position
                self._attach_position_metadata(position, order.signal, execution_context, order.trace_id)
                self._add_position(position)
            else:
                position = order.position_ref
                total_size = float(position.size) + float(fill.size)
                if total_size > 0:
                    position.entry_price = ((float(position.entry_price) * float(position.size)) + (float(fill.price) * float(fill.size))) / total_size
                position.size = total_size
                position.fee_paid = float(getattr(position, "fee_paid", 0.0) or 0.0) + float(fill.fee)
                self._attach_position_metadata(position, order.signal, execution_context, order.trace_id)
            self._record_event(
                EVENT_FILL,
                order.trace_id,
                {
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "size": fill.size,
                    "price": fill.price,
                    "fee": fill.fee,
                    "fill_fraction": fill_fraction,
                    "passive_fill": bool(details.get("passive_fill", False)),
                    "market_impact_bps": float(fill.metadata.get("market_impact_bps", 0.0) or 0.0),
                },
            )

    def _attach_position_metadata(self, position: Position, signal: Dict[str, Any], execution_context: Dict[str, Any], trace_id: str) -> None:
        decision_context = self.learning.build_trade_context(signal, execution_context=execution_context)
        metadata = dict(getattr(position, "metadata", {}) or {})
        metadata.setdefault("simulation_position_id", uuid.uuid4().hex)
        metadata.setdefault("initial_size", float(getattr(position, "size", 0.0) or 0.0))
        metadata["decision_context"] = decision_context
        metadata["signal_snapshot"] = dict(signal)
        metadata["execution_context"] = execution_context
        metadata["opened_trace_id"] = trace_id
        if "entry_atr" not in metadata and hasattr(self.signals, "compute_atr"):
            try:
                atr = self.signals.compute_atr(
                    str(signal.get("symbol", position.symbol)),
                    timeframe=str(getattr(self.config, "trailing_timeframe", self._base_timeframe)),
                    period=int(getattr(self.config, "trailing_atr_period", 14)),
                )
            except Exception:
                atr = None
            if atr is not None and float(atr) > 0:
                metadata["entry_atr"] = float(atr)
        position.metadata = metadata

    def _add_position(self, position: Position) -> None:
        current = self.state.open_positions.get(position.symbol)
        if current is None:
            self.state.open_positions[position.symbol] = [position]
        elif isinstance(current, list):
            current.append(position)
        else:
            self.state.open_positions[position.symbol] = [current, position]
        self._record_event(
            EVENT_POSITION_UPDATED,
            uuid.uuid4().hex,
            {
                "symbol": position.symbol,
                "side": position.side,
                "size": position.size,
                "entry_price": position.entry_price,
                "strategy": position.strategy,
            },
        )

    def _cancel_orders(self, cancelled: Sequence[SimulatedOrder]) -> None:
        for order in cancelled:
            self._cancelled_orders += 1
            self._record_skipped_signal(order.signal, order.stale_cancel_reason or "order_cancelled", order.trace_id)

    def _expire_orders(self, expired: Sequence[SimulatedOrder]) -> None:
        for order in expired:
            self._expired_orders += 1
            if order.remaining_size > 1e-9:
                self._record_skipped_signal(order.signal, "limit_unfilled", order.trace_id)

    def _manage_open_positions(self, bar_by_symbol: Mapping[str, "pd.Series"], now: dt.datetime, current_index: int | None = None) -> None:
        for symbol in list(self.state.open_positions.keys()):
            current_positions = self.state.open_positions.get(symbol, [])
            positions = list(current_positions if isinstance(current_positions, list) else [current_positions])
            bar = bar_by_symbol.get(symbol)
            if bar is None:
                continue
            remaining: list[Position] = []
            for position in positions:
                self._update_excursions(position, bar)
                self._update_dynamic_risk(position, symbol, bar)
                self._maybe_partial_profit_take(position, bar, current_index=current_index)
                exit_reason, raw_exit_price, exit_details = self._trigger_exit(position, bar, current_index=current_index)
                if raw_exit_price is None:
                    exit_reason, raw_exit_price = self._time_stop_exit(position, bar, now)
                    if raw_exit_price is not None:
                        exit_details["time_stop"] = True
                        exit_details["time_stop_family_multiplier_soft"] = self._time_stop_family_multiplier(position, soft=True)
                        exit_details["time_stop_family_multiplier_hard"] = self._time_stop_family_multiplier(position, soft=False)
                if raw_exit_price is None:
                    position.unrealized_pnl = self._gross_pl(position.side, float(position.entry_price), float(bar["close"]), float(position.size))
                    remaining.append(position)
                    continue
                exit_price, exit_fee = self._apply_exit_costs(position, raw_exit_price, bar)
                gross_pl = self._gross_pl(position.side, float(position.entry_price), exit_price, float(position.size))
                metadata = dict(getattr(position, "metadata", {}) or {})
                partial_gross_pl = float(metadata.get("partial_realized_gross_pl", 0.0) or 0.0)
                partial_exit_fees = float(metadata.get("partial_exit_fees", 0.0) or 0.0)
                total_gross_pl = gross_pl + partial_gross_pl
                total_fees = float(getattr(position, "fee_paid", 0.0) or 0.0) + partial_exit_fees + exit_fee
                total_profit_loss = total_gross_pl - total_fees
                self.state.balance += gross_pl - exit_fee
                self.risk.update_after_trade_result(total_profit_loss)
                self.learning.record_closed_trade(
                    symbol=symbol,
                    position=position,
                    close_price=exit_price,
                    profit_loss=total_profit_loss,
                    exit_reason=exit_reason,
                    closed_at=now.isoformat(),
                )
                self._register_reentry_cooldown(position, exit_reason, now)
                strategy = str(getattr(position, "strategy", "unknown"))
                order_type = str((getattr(position, "metadata", {}) or {}).get("execution_context", {}).get("order_type", "limit"))
                self._increment_counter(self._closed_by_strategy, strategy)
                self._increment_counter(self._closed_by_symbol, symbol)
                self._increment_nested_counter(self._closed_by_strategy_by_symbol, symbol, strategy)
                self._increment_counter(self._closed_by_order_type, order_type)
                signal_snapshot = dict(metadata.get("signal_snapshot", {}) or {})
                replacement_active = bool(dict(signal_snapshot.get("metadata", {}) or {}).get("replacement_opportunity", {}).get("active", False))
                if replacement_active:
                    self._replacement_closed += 1
                    self._increment_counter(self._replacement_closed_by_strategy, strategy)
                if total_profit_loss >= 0:
                    self._increment_counter(self._wins_by_strategy, strategy)
                    self._increment_counter(self._wins_by_symbol, symbol)
                    self._increment_nested_counter(self._wins_by_strategy_by_symbol, symbol, strategy)
                    if replacement_active:
                        self._replacement_wins += 1
                        self._increment_counter(self._replacement_wins_by_strategy, strategy)
                else:
                    self._increment_counter(self._losses_by_strategy, strategy)
                    self._increment_counter(self._losses_by_symbol, symbol)
                    self._increment_nested_counter(self._losses_by_strategy_by_symbol, symbol, strategy)
                    if replacement_active:
                        self._replacement_losses += 1
                        self._increment_counter(self._replacement_losses_by_strategy, strategy)
                self.trades.append(
                    asdict(
                        ClosedTrade(
                            symbol=symbol,
                            strategy=str(getattr(position, "strategy", "unknown")),
                            side=str(getattr(position, "side", "long")),
                            entry_time=getattr(position, "opened_at", now),
                            exit_time=now,
                            entry_price=float(getattr(position, "entry_price", 0.0) or 0.0),
                            exit_price=exit_price,
                            size=float(metadata.get("initial_size", getattr(position, "size", 0.0)) or 0.0),
                            gross_pl=total_gross_pl,
                            pl=total_profit_loss,
                            fees=total_fees,
                            holding_minutes=max((now - getattr(position, "opened_at", now)).total_seconds() / 60.0, 0.0),
                            exit_reason=exit_reason,
                            order_type=str(
                                (getattr(position, "metadata", {}) or {}).get("execution_context", {}).get(
                                    "order_type",
                                    (getattr(position, "metadata", {}) or {}).get("signal_snapshot", {}).get("fast_move", False) and "market" or "limit",
                                )
                            ),
                            fill_fraction=float((getattr(position, "metadata", {}) or {}).get("execution_context", {}).get("fill_fraction", 1.0) or 1.0),
                            latency_bars=max(int(round(float((getattr(position, "metadata", {}) or {}).get("execution_context", {}).get("latency_ms", 0.0) or 0.0) / max(timeframe_to_minutes(self._base_timeframe) * 60 * 1000, 1))), 0),
                            slippage_bps=float((getattr(position, "metadata", {}) or {}).get("execution_context", {}).get("slippage_bps", 0.0) or 0.0),
                            mfe_r=float((getattr(position, "metadata", {}) or {}).get("mfe_r", 0.0) or 0.0),
                            mae_r=float((getattr(position, "metadata", {}) or {}).get("mae_r", 0.0) or 0.0),
                            giveback_r=self._trade_giveback_r(position, total_profit_loss),
                        )
                    )
                )
                self.trades[-1]["exit_details"] = exit_details
            if remaining:
                self.state.open_positions[symbol] = remaining
            else:
                del self.state.open_positions[symbol]

    def _update_dynamic_risk(self, position: Position, symbol: str, bar: "pd.Series") -> None:
        side = str(getattr(position, "side", "long"))
        base_sl = float(getattr(position, "initial_stop_loss", None) or getattr(position, "stop_loss", 0.0) or 0.0)
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        if side == "long":
            risk_dist = entry_price - base_sl
        else:
            risk_dist = base_sl - entry_price
        if risk_dist <= 0:
            return
        if side == "long":
            rr_move = (float(bar["high"]) - entry_price) / risk_dist
            if rr_move >= float(getattr(self.config, "breakeven_rr", 1.0)) and float(position.stop_loss) < entry_price:
                position.stop_loss = entry_price
        else:
            rr_move = (entry_price - float(bar["low"])) / risk_dist
            if rr_move >= float(getattr(self.config, "breakeven_rr", 1.0)) and float(position.stop_loss) > entry_price:
                position.stop_loss = entry_price
        self._apply_profit_protection(position, rr_move, risk_dist)
        self._apply_mean_reversion_profit_capture(position, rr_move, risk_dist)
        self._apply_volatility_tightening(position, symbol, risk_dist)
        trailing_rr, trailing_atr_mult = self._family_trailing_params(position)
        if rr_move >= trailing_rr:
            if not hasattr(self.signals, "compute_atr"):
                return
            atr = self.signals.compute_atr(
                symbol,
                timeframe=str(getattr(self.config, "trailing_timeframe", self._base_timeframe)),
                period=int(getattr(self.config, "trailing_atr_period", 14)),
            )
            if atr is not None and atr > 0:
                if side == "long":
                    new_stop = float(bar["high"]) - (float(atr) * trailing_atr_mult)
                    if new_stop > float(position.stop_loss):
                        position.stop_loss = new_stop
                else:
                    new_stop = float(bar["low"]) + (float(atr) * trailing_atr_mult)
                    if new_stop < float(position.stop_loss):
                        position.stop_loss = new_stop

    def _family_trailing_params(self, position: Position) -> tuple[float, float]:
        strategy = str(getattr(position, "strategy", "unknown") or "unknown").lower()
        if strategy == "trend_breakout":
            rr = float(getattr(self.config, "trailing_breakout_rr", 1.10) or 1.10)
            atr_mult = float(getattr(self.config, "trailing_breakout_atr_mult", 0.85) or 0.85)
        elif strategy == "trend_pullback":
            rr = float(getattr(self.config, "trailing_pullback_rr", 1.80) or 1.80)
            atr_mult = float(getattr(self.config, "trailing_pullback_atr_mult", 1.20) or 1.20)
        elif strategy == "mean_reversion":
            rr = float(getattr(self.config, "trailing_mean_reversion_rr", 2.50) or 2.50)
            atr_mult = float(getattr(self.config, "trailing_mean_reversion_atr_mult", 1.35) or 1.35)
        else:
            rr = float(getattr(self.config, "trailing_rr", 1.5) or 1.5)
            atr_mult = float(getattr(self.config, "trailing_atr_mult", 1.0) or 1.0)
        profile = self._regime_exit_profile(position)
        if profile == "trend_supportive" and bool(dict(getattr(position, "metadata", {}) or {}).get("partial_profit_taken", False)):
            rr *= float(getattr(self.config, "regime_exit_trend_trailing_rr_multiplier", 1.10) or 1.10)
            atr_mult *= float(getattr(self.config, "regime_exit_trend_trailing_atr_multiplier", 1.15) or 1.15)
        elif profile == "risk_off":
            rr *= float(getattr(self.config, "regime_exit_risk_off_trailing_rr_multiplier", 0.70) or 0.70)
            atr_mult *= float(getattr(self.config, "regime_exit_risk_off_trailing_atr_multiplier", 0.65) or 0.65)
        elif profile == "choppy":
            rr *= float(getattr(self.config, "regime_exit_choppy_trailing_rr_multiplier", 0.85) or 0.85)
            atr_mult *= float(getattr(self.config, "regime_exit_choppy_trailing_atr_multiplier", 0.80) or 0.80)
        return rr, atr_mult

    def _regime_exit_profile(self, position: Position) -> str:
        if not bool(getattr(self.config, "regime_exit_profiles_enabled", True)):
            return "neutral"
        metadata = dict(getattr(position, "metadata", {}) or {})
        signal_snapshot = dict(metadata.get("signal_snapshot", {}) or {})
        signal_meta = dict(signal_snapshot.get("metadata", {}) or {})
        raw_state = str(
            signal_meta.get(
                "regime_state",
                signal_snapshot.get("regime_state", signal_snapshot.get("regime", signal_meta.get("regime", ""))),
            )
            or ""
        ).lower()
        if raw_state in {"trend_supportive", "trending", "bullish", "persistent_trend"}:
            return "trend_supportive"
        if raw_state in {"risk_off", "high_volatility", "crash", "bearish"}:
            return "risk_off"
        if raw_state in {"choppy", "neutral", "range", "sideways"}:
            return "choppy"
        try:
            current_state = str(self._current_universe_regime_state().get("state", "neutral") or "neutral").lower()
        except Exception:
            current_state = "neutral"
        if current_state in {"trend_supportive"}:
            return "trend_supportive"
        if current_state in {"risk_off", "high_volatility"}:
            return "risk_off"
        return "choppy" if current_state == "choppy" else "neutral"

    def _regime_adjusted_partial_lock_rr(self, position: Position, lock_rr: float) -> float:
        profile = self._regime_exit_profile(position)
        if profile == "trend_supportive":
            return lock_rr * float(getattr(self.config, "regime_exit_trend_partial_lock_multiplier", 0.85) or 0.85)
        if profile == "risk_off":
            return lock_rr * float(getattr(self.config, "regime_exit_risk_off_partial_lock_multiplier", 1.50) or 1.50)
        if profile == "choppy":
            return lock_rr * float(getattr(self.config, "regime_exit_choppy_partial_lock_multiplier", 1.25) or 1.25)
        return lock_rr

    def _regime_adjusted_close_floor_rr(self, position: Position, close_floor: float) -> float:
        profile = self._regime_exit_profile(position)
        if profile == "risk_off":
            return max(close_floor, float(getattr(self.config, "regime_exit_risk_off_close_floor_rr", -0.06) or -0.06))
        if profile == "choppy":
            return max(close_floor, float(getattr(self.config, "regime_exit_choppy_close_floor_rr", -0.09) or -0.09))
        return close_floor

    def _update_excursions(self, position: Position, bar: "pd.Series") -> None:
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        base_sl = float(getattr(position, "initial_stop_loss", None) or getattr(position, "stop_loss", 0.0) or 0.0)
        risk_dist = abs(entry_price - base_sl)
        if risk_dist <= 0:
            return
        side = str(getattr(position, "side", "long") or "long")
        if side == "long":
            favorable_r = (float(bar["high"]) - entry_price) / risk_dist
            adverse_r = (entry_price - float(bar["low"])) / risk_dist
        else:
            favorable_r = (entry_price - float(bar["low"])) / risk_dist
            adverse_r = (float(bar["high"]) - entry_price) / risk_dist
        metadata = dict(getattr(position, "metadata", {}) or {})
        metadata["mfe_r"] = max(float(metadata.get("mfe_r", 0.0) or 0.0), float(favorable_r or 0.0))
        metadata["mae_r"] = max(float(metadata.get("mae_r", 0.0) or 0.0), float(adverse_r or 0.0))
        position.metadata = metadata

    def _trade_giveback_r(self, position: Position, realized_pl: float) -> float:
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        base_sl = float(getattr(position, "initial_stop_loss", None) or getattr(position, "stop_loss", 0.0) or 0.0)
        risk_dist = abs(entry_price - base_sl)
        size = float(getattr(position, "size", 0.0) or 0.0)
        if risk_dist <= 0 or size <= 0:
            return 0.0
        realized_r = float(realized_pl or 0.0) / (risk_dist * size)
        mfe_r = float((getattr(position, "metadata", {}) or {}).get("mfe_r", 0.0) or 0.0)
        return max(mfe_r - realized_r, 0.0)

    def _apply_profit_protection(self, position: Position, rr_move: float, risk_dist: float) -> None:
        if risk_dist <= 0:
            return
        strategy = str(getattr(position, "strategy", "unknown") or "unknown").lower()
        metadata = dict(getattr(position, "metadata", {}) or {})
        signal_snapshot = dict(metadata.get("signal_snapshot", {}) or {})
        signal_metadata = dict(signal_snapshot.get("metadata", {}) or {})
        high_beta_quality_escape = strategy == "trend_pullback" and bool(signal_metadata.get("high_beta_quality_escape", False))
        spot_core_pullback = strategy == "trend_pullback" and str(signal_metadata.get("symbol_bucket", "") or "") in {"majors", "exchange_beta"}
        if strategy == "trend_breakout":
            trigger_key = "profit_protect_breakout_trigger_rr"
            lock_key = "profit_protect_breakout_lock_rr"
        elif high_beta_quality_escape:
            trigger_key = "high_beta_pullback_profit_protect_trigger_rr"
            lock_key = "high_beta_pullback_profit_protect_lock_rr"
        elif spot_core_pullback:
            trigger_key = "spot_core_pullback_profit_protect_trigger_rr"
            lock_key = "spot_core_pullback_profit_protect_lock_rr"
        elif strategy == "trend_pullback":
            trigger_key = "profit_protect_pullback_trigger_rr"
            lock_key = "profit_protect_pullback_lock_rr"
        elif strategy == "mean_reversion":
            trigger_key = "profit_protect_mean_reversion_trigger_rr"
            lock_key = "profit_protect_mean_reversion_lock_rr"
        else:
            trigger_key = "breakeven_rr"
            lock_key = None
        trigger_rr = float(getattr(self.config, trigger_key, 1.0) or 1.0)
        profile = self._regime_exit_profile(position)
        if profile == "risk_off":
            trigger_rr *= 0.75
        elif profile == "choppy":
            trigger_rr *= 0.85
        elif profile == "trend_supportive" and bool(metadata.get("partial_profit_taken", False)):
            trigger_rr *= 1.10
        if rr_move < trigger_rr:
            return
        lock_rr = float(getattr(self.config, lock_key, 0.0) or 0.0) if lock_key is not None else 0.0
        if lock_rr > 0.0:
            lock_rr = self._regime_adjusted_partial_lock_rr(position, lock_rr)
            metadata["regime_exit_profile"] = profile
            metadata["regime_adjusted_profit_lock_rr"] = float(lock_rr)
            position.metadata = metadata
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        if str(getattr(position, "side", "long") or "long") == "long":
            protected_stop = entry_price + (risk_dist * lock_rr)
            if protected_stop > float(getattr(position, "stop_loss", 0.0) or 0.0):
                position.stop_loss = protected_stop
        else:
            protected_stop = entry_price - (risk_dist * lock_rr)
            if protected_stop < float(getattr(position, "stop_loss", 0.0) or 0.0):
                position.stop_loss = protected_stop

    def _apply_mean_reversion_profit_capture(self, position: Position, rr_move: float, risk_dist: float) -> None:
        strategy = str(getattr(position, "strategy", "unknown") or "unknown").lower()
        if strategy != "mean_reversion" or risk_dist <= 0:
            return
        trigger_rr = float(getattr(self.config, "mean_reversion_profit_capture_trigger_rr", 0.45) or 0.45)
        if rr_move < trigger_rr:
            return
        lock_rr = float(getattr(self.config, "mean_reversion_profit_capture_lock_rr", 0.30) or 0.30)
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        if str(getattr(position, "side", "long") or "long") == "long":
            protected_stop = entry_price + (risk_dist * lock_rr)
            if protected_stop > float(getattr(position, "stop_loss", 0.0) or 0.0):
                position.stop_loss = protected_stop
        else:
            protected_stop = entry_price - (risk_dist * lock_rr)
            if protected_stop < float(getattr(position, "stop_loss", 0.0) or 0.0):
                position.stop_loss = protected_stop

    def _maybe_partial_profit_take(self, position: Position, bar: "pd.Series", current_index: int | None = None) -> None:
        strategy = str(getattr(position, "strategy", "unknown") or "unknown").lower()
        metadata = dict(getattr(position, "metadata", {}) or {})
        signal_snapshot = dict(metadata.get("signal_snapshot", {}) or {})
        signal_metadata = dict(signal_snapshot.get("metadata", {}) or {})
        high_beta_quality_escape = strategy == "trend_pullback" and bool(signal_metadata.get("high_beta_quality_escape", False))
        spot_core_pullback = strategy == "trend_pullback" and str(signal_metadata.get("symbol_bucket", "") or "") in {"majors", "exchange_beta"}
        if strategy not in {"trend_breakout", "mean_reversion"} and not high_beta_quality_escape and not spot_core_pullback:
            return
        if bool(metadata.get("partial_profit_taken", False)):
            return
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        base_sl = float(getattr(position, "initial_stop_loss", None) or getattr(position, "stop_loss", 0.0) or 0.0)
        risk_dist = abs(entry_price - base_sl)
        current_size = float(getattr(position, "size", 0.0) or 0.0)
        if risk_dist <= 0 or current_size <= 0:
            return
        side = str(getattr(position, "side", "long") or "long")
        rr_move = ((float(bar["high"]) - entry_price) / risk_dist) if side == "long" else ((entry_price - float(bar["low"])) / risk_dist)
        if high_beta_quality_escape:
            trigger_rr = float(getattr(self.config, "partial_profit_take_high_beta_pullback_trigger_rr", 0.35) or 0.35)
            fraction = float(getattr(self.config, "partial_profit_take_high_beta_pullback_fraction", 0.45) or 0.45)
            stop_lock_rr = float(getattr(self.config, "partial_profit_take_high_beta_pullback_stop_lock_rr", 0.12) or 0.12)
            take_reason = "high_beta_pullback_partial_profit_take"
        elif spot_core_pullback:
            trigger_rr = float(getattr(self.config, "partial_profit_take_spot_core_pullback_trigger_rr", 0.40) or 0.40)
            fraction = float(getattr(self.config, "partial_profit_take_spot_core_pullback_fraction", 0.35) or 0.35)
            stop_lock_rr = float(getattr(self.config, "partial_profit_take_spot_core_pullback_stop_lock_rr", 0.12) or 0.12)
            take_reason = "spot_core_pullback_partial_profit_take"
        elif strategy == "mean_reversion":
            trigger_rr = float(getattr(self.config, "partial_profit_take_mean_reversion_trigger_rr", 0.70) or 0.70)
            fraction = float(getattr(self.config, "partial_profit_take_mean_reversion_fraction", 0.50) or 0.50)
            stop_lock_rr = 0.0
            take_reason = f"{strategy}_partial_profit_take"
        else:
            trigger_rr = float(getattr(self.config, "partial_profit_take_breakout_trigger_rr", 1.00) or 1.00)
            fraction = float(getattr(self.config, "partial_profit_take_breakout_fraction", 0.35) or 0.35)
            stop_lock_rr = 0.0
            take_reason = f"{strategy}_partial_profit_take"
        profile = self._regime_exit_profile(position)
        if stop_lock_rr > 0.0:
            stop_lock_rr = self._regime_adjusted_partial_lock_rr(position, stop_lock_rr)
            metadata["regime_exit_profile"] = profile
            metadata["regime_adjusted_partial_stop_lock_rr"] = float(stop_lock_rr)
        if rr_move < trigger_rr or fraction <= 0.0:
            return
        closed_size = current_size * min(max(fraction, 0.05), 0.90)
        remaining_size = current_size - closed_size
        if closed_size <= 0.0 or remaining_size <= 1e-9:
            return
        raw_partial_exit = float(bar["close"])
        trigger_price = entry_price + (risk_dist * trigger_rr) if side == "long" else entry_price - (risk_dist * trigger_rr)
        if high_beta_quality_escape or spot_core_pullback:
            raw_partial_exit = max(raw_partial_exit, trigger_price) if side == "long" else min(raw_partial_exit, trigger_price)
        exit_price, exit_fee = self._apply_exit_costs(position, raw_partial_exit, bar)
        partial_gross_pl = self._gross_pl(side, entry_price, exit_price, closed_size)
        self.state.balance += partial_gross_pl - exit_fee
        position.size = remaining_size
        metadata["partial_profit_taken"] = True
        if current_index is not None:
            metadata["partial_profit_bar_index"] = int(current_index)
        metadata["partial_profit_taken_fraction"] = float(closed_size / max(current_size, 1e-9))
        metadata["partial_profit_rr_move"] = float(rr_move)
        metadata["partial_profit_close_price"] = float(exit_price)
        metadata["partial_realized_gross_pl"] = float(metadata.get("partial_realized_gross_pl", 0.0) or 0.0) + partial_gross_pl
        metadata["partial_exit_fees"] = float(metadata.get("partial_exit_fees", 0.0) or 0.0) + exit_fee
        metadata["partial_profit_take_reason"] = take_reason
        if stop_lock_rr > 0.0:
            fee_lock_rr = self._fee_aware_partial_profit_stop_lock_rr(
                position,
                entry_price=entry_price,
                risk_dist=risk_dist,
                remaining_size=remaining_size,
                partial_gross_pl=float(metadata.get("partial_realized_gross_pl", 0.0) or 0.0),
                partial_exit_fees=float(metadata.get("partial_exit_fees", 0.0) or 0.0),
            )
            lock_rr = min(max(stop_lock_rr, fee_lock_rr, 0.0), max(rr_move - 0.05, 0.0))
            if side == "long":
                protected_stop = entry_price + (risk_dist * lock_rr)
                if protected_stop > float(getattr(position, "stop_loss", 0.0) or 0.0):
                    position.stop_loss = protected_stop
                    metadata["partial_profit_stop_lock_rr"] = float(lock_rr)
                    metadata["partial_profit_fee_aware_stop_lock_rr"] = float(fee_lock_rr)
            else:
                protected_stop = entry_price - (risk_dist * lock_rr)
                if protected_stop < float(getattr(position, "stop_loss", 0.0) or 0.0):
                    position.stop_loss = protected_stop
                    metadata["partial_profit_stop_lock_rr"] = float(lock_rr)
                    metadata["partial_profit_fee_aware_stop_lock_rr"] = float(fee_lock_rr)
        position.metadata = metadata
        self._partial_profit_takes += 1
        self._increment_counter(self._partial_profit_takes_by_strategy, strategy)

    def _fee_aware_partial_profit_stop_lock_rr(
        self,
        position: Position,
        *,
        entry_price: float,
        risk_dist: float,
        remaining_size: float,
        partial_gross_pl: float,
        partial_exit_fees: float,
    ) -> float:
        if not bool(getattr(self.config, "partial_profit_fee_aware_stop_lock_enabled", True)):
            return 0.0
        if entry_price <= 0.0 or risk_dist <= 0.0 or remaining_size <= 0.0:
            return 0.0
        fee_rate = float(getattr(self.config, "backtest_fee_bps", 10.0) or 10.0) / 10000.0
        entry_fees = float(getattr(position, "fee_paid", 0.0) or 0.0)
        estimated_exit_fee = abs(entry_price * remaining_size * fee_rate)
        buffer_cost = abs(entry_price * remaining_size * (float(getattr(self.config, "partial_profit_fee_aware_stop_lock_buffer_bps", 2.0) or 2.0) / 10000.0))
        net_cost_to_cover = max(entry_fees + partial_exit_fees + estimated_exit_fee + buffer_cost - partial_gross_pl, 0.0)
        raw_lock_rr = net_cost_to_cover / max(risk_dist * remaining_size, 1e-9)
        max_lock_rr = max(float(getattr(self.config, "partial_profit_fee_aware_stop_lock_max_rr", 0.18) or 0.18), 0.0)
        return min(raw_lock_rr, max_lock_rr) if max_lock_rr > 0.0 else raw_lock_rr

    def _apply_volatility_tightening(self, position: Position, symbol: str, risk_dist: float) -> None:
        if risk_dist <= 0 or not hasattr(self.signals, "compute_atr"):
            return
        metadata = dict(getattr(position, "metadata", {}) or {})
        entry_atr = float(metadata.get("entry_atr", 0.0) or 0.0)
        if entry_atr <= 0:
            return
        current_atr = self.signals.compute_atr(
            symbol,
            timeframe=str(getattr(self.config, "trailing_timeframe", self._base_timeframe)),
            period=int(getattr(self.config, "trailing_atr_period", 14)),
        )
        current_atr = float(current_atr or 0.0)
        if current_atr <= 0:
            return
        expansion_ratio = current_atr / max(entry_atr, 1e-9)
        trigger_ratio = float(getattr(self.config, "volatility_tightening_trigger_ratio", 1.30) or 1.30)
        if expansion_ratio < trigger_ratio:
            return
        metadata["volatility_tightened"] = True
        position.metadata = metadata
        strategy = str(getattr(position, "strategy", "unknown") or "unknown").lower()
        if strategy == "trend_breakout":
            tighten_rr = float(getattr(self.config, "volatility_tightening_breakout_rr", 0.30) or 0.30)
        elif strategy == "trend_pullback":
            tighten_rr = float(getattr(self.config, "volatility_tightening_pullback_rr", 0.12) or 0.12)
        elif strategy == "mean_reversion":
            tighten_rr = float(getattr(self.config, "volatility_tightening_mean_reversion_rr", 0.22) or 0.22)
        else:
            tighten_rr = 0.10
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        if str(getattr(position, "side", "long") or "long") == "long":
            tightened_stop = entry_price + (risk_dist * tighten_rr)
            if tightened_stop > float(getattr(position, "stop_loss", 0.0) or 0.0):
                position.stop_loss = tightened_stop
        else:
            tightened_stop = entry_price - (risk_dist * tighten_rr)
            if tightened_stop < float(getattr(position, "stop_loss", 0.0) or 0.0):
                position.stop_loss = tightened_stop

    def _time_stop_exit(self, position: Position, bar: "pd.Series", now: dt.datetime) -> tuple[str, float | None]:
        metadata = dict(getattr(position, "metadata", {}) or {})
        signal_snapshot = dict(metadata.get("signal_snapshot", {}) or {})
        expected_holding = float(signal_snapshot.get("expected_holding_minutes", 0.0) or 0.0)
        opened_at = getattr(position, "opened_at", None)
        if expected_holding <= 0 or opened_at is None:
            return "OPEN", None
        holding_minutes = max((now - opened_at).total_seconds() / 60.0, 0.0)
        side = str(getattr(position, "side", "long"))
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        base_sl = float(getattr(position, "initial_stop_loss", None) or getattr(position, "stop_loss", 0.0) or 0.0)
        risk_dist = abs(entry_price - base_sl)
        if risk_dist <= 0:
            return "OPEN", None
        current_r = self._gross_pl(side, entry_price, float(bar["close"]), 1.0) / risk_dist
        soft_limit = expected_holding * self._time_stop_family_multiplier(position, soft=True)
        hard_limit = expected_holding * self._time_stop_family_multiplier(position, soft=False)
        if holding_minutes >= hard_limit and current_r < float(getattr(self.config, "time_stop_hard_min_r_multiple", 0.45)):
            return "TIME_HARD", float(bar["close"])
        if holding_minutes >= soft_limit and current_r < float(getattr(self.config, "time_stop_soft_min_r_multiple", 0.15)):
            return "TIME_SOFT", float(bar["close"])
        return "OPEN", None

    def _time_stop_family_multiplier(self, position: Position, *, soft: bool) -> float:
        strategy = str(getattr(position, "strategy", "unknown") or "unknown").lower()
        if strategy == "trend_breakout":
            key = "time_stop_breakout_soft_multiplier" if soft else "time_stop_breakout_hard_multiplier"
        elif strategy == "trend_pullback":
            key = "time_stop_pullback_soft_multiplier" if soft else "time_stop_pullback_hard_multiplier"
        elif strategy == "mean_reversion":
            key = "time_stop_mean_reversion_soft_multiplier" if soft else "time_stop_mean_reversion_hard_multiplier"
        else:
            key = "time_stop_soft_holding_multiplier" if soft else "time_stop_hard_holding_multiplier"
        fallback = 1.35 if soft else 2.25
        return float(getattr(self.config, key, fallback) or fallback)

    def _trigger_exit(self, position: Position, bar: "pd.Series", current_index: int | None = None) -> tuple[str, float | None, Dict[str, Any]]:
        side = str(getattr(position, "side", "long"))
        open_price = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        stop_loss = float(getattr(position, "stop_loss", 0.0) or 0.0)
        take_profit = float(getattr(position, "take_profit", 0.0) or 0.0)
        details = {
            "open": open_price,
            "high": high,
            "low": low,
            "close": float(bar["close"]),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "ambiguous_bar": False,
            "path_assumption": "gap_then_touch",
            "regime_exit_profile": self._regime_exit_profile(position),
        }
        def _stop_reason(raw_price: float) -> str:
            metadata = dict(getattr(position, "metadata", {}) or {})
            if not bool(metadata.get("partial_profit_taken", False)):
                return "SL"
            if float(metadata.get("partial_profit_stop_lock_rr", 0.0) or 0.0) <= 0.0:
                return "SL"
            entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
            protected = (raw_price > entry_price) if side == "long" else (raw_price < entry_price)
            if not protected:
                return "SL"
            details["profit_protected_stop"] = True
            details["partial_profit_stop_lock_rr"] = float(metadata.get("partial_profit_stop_lock_rr", 0.0) or 0.0)
            return "PROFIT_PROTECT_STOP"
        if side == "long":
            if open_price <= stop_loss:
                return _stop_reason(open_price), open_price, details
            if open_price >= take_profit:
                return "TP", open_price, details
            thesis_reason, thesis_price = self._thesis_failure_exit(position, float(bar["close"]))
            if thesis_reason != "OPEN" and str(thesis_reason).startswith("HIGH_BETA_"):
                details["path_assumption"] = "pre_stop_high_beta_thesis_failure"
                return thesis_reason, thesis_price, details
            stop_hit = low <= stop_loss
            tp_hit = high >= take_profit
            if stop_hit and tp_hit:
                details["ambiguous_bar"] = True
                details["path_assumption"] = "conservative_worst_case"
                self._ambiguous_exit_bars += 1
                return _stop_reason(stop_loss), stop_loss, details
            if stop_hit:
                return _stop_reason(stop_loss), stop_loss, details
            if tp_hit:
                return "TP", take_profit, details
            follow_reason, follow_price, follow_details = self._winner_follow_through_exit(position, float(bar["close"]), current_index=current_index)
            if follow_reason != "OPEN":
                details.update(follow_details)
                details["path_assumption"] = "close_follow_through_failure"
                return follow_reason, follow_price, details
            thesis_reason, thesis_price = self._thesis_failure_exit(position, float(bar["close"]))
            if thesis_reason != "OPEN":
                details["path_assumption"] = "close_thesis_failure"
                return thesis_reason, thesis_price, details
            reclaim_reason, reclaim_price = self._mean_reversion_reclaim_failure_exit(position, float(bar["close"]))
            if reclaim_reason != "OPEN":
                details["path_assumption"] = "close_reclaim_failure"
                return reclaim_reason, reclaim_price, details
            return "OPEN", None, details
        if open_price >= stop_loss:
            return _stop_reason(open_price), open_price, details
        if open_price <= take_profit:
            return "TP", open_price, details
        thesis_reason, thesis_price = self._thesis_failure_exit(position, float(bar["close"]))
        if thesis_reason != "OPEN" and str(thesis_reason).startswith("HIGH_BETA_"):
            details["path_assumption"] = "pre_stop_high_beta_thesis_failure"
            return thesis_reason, thesis_price, details
        stop_hit = high >= stop_loss
        tp_hit = low <= take_profit
        if stop_hit and tp_hit:
            details["ambiguous_bar"] = True
            details["path_assumption"] = "conservative_worst_case"
            self._ambiguous_exit_bars += 1
            return _stop_reason(stop_loss), stop_loss, details
        if stop_hit:
            return _stop_reason(stop_loss), stop_loss, details
        if tp_hit:
            return "TP", take_profit, details
        follow_reason, follow_price, follow_details = self._winner_follow_through_exit(position, float(bar["close"]), current_index=current_index)
        if follow_reason != "OPEN":
            details.update(follow_details)
            details["path_assumption"] = "close_follow_through_failure"
            return follow_reason, follow_price, details
        thesis_reason, thesis_price = self._thesis_failure_exit(position, float(bar["close"]))
        if thesis_reason != "OPEN":
            details["path_assumption"] = "close_thesis_failure"
            return thesis_reason, thesis_price, details
        reclaim_reason, reclaim_price = self._mean_reversion_reclaim_failure_exit(position, float(bar["close"]))
        if reclaim_reason != "OPEN":
            details["path_assumption"] = "close_reclaim_failure"
            return reclaim_reason, reclaim_price, details
        return "OPEN", None, details

    def _winner_follow_through_exit(self, position: Position, close_price: float, current_index: int | None = None) -> tuple[str, float | None, Dict[str, Any]]:
        if not bool(getattr(self.config, "winner_follow_through_filter_enabled", True)):
            return "OPEN", None, {}
        metadata = dict(getattr(position, "metadata", {}) or {})
        if not bool(metadata.get("partial_profit_taken", False)):
            return "OPEN", None, {}
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        base_sl = float(getattr(position, "initial_stop_loss", None) or getattr(position, "stop_loss", 0.0) or 0.0)
        risk_dist = abs(entry_price - base_sl)
        if entry_price <= 0.0 or risk_dist <= 0.0 or close_price <= 0.0:
            return "OPEN", None, {}
        partial_index = metadata.get("partial_profit_bar_index")
        confirm_bars = max(int(getattr(self.config, "winner_follow_through_confirm_bars", 2) or 2), 1)
        if current_index is not None and partial_index is not None:
            try:
                bars_since_partial = int(current_index) - int(partial_index)
            except (TypeError, ValueError):
                bars_since_partial = confirm_bars
            if bars_since_partial < confirm_bars:
                return "OPEN", None, {}
        side = str(getattr(position, "side", "long") or "long")
        current_rr = ((close_price - entry_price) / risk_dist) if side == "long" else ((entry_price - close_price) / risk_dist)
        partial_rr = float(metadata.get("partial_profit_rr_move", metadata.get("partial_profit_stop_lock_rr", 0.0)) or 0.0)
        mfe_r = float(metadata.get("mfe_r", 0.0) or 0.0)
        min_mfe = float(getattr(self.config, "winner_follow_through_min_mfe_r", 0.35) or 0.35)
        if mfe_r < min_mfe:
            return "OPEN", None, {}
        min_progress = float(getattr(self.config, "winner_follow_through_min_progress_after_partial_rr", 0.10) or 0.10)
        max_giveback = float(getattr(self.config, "winner_follow_through_max_giveback_rr", 0.22) or 0.22)
        progress_target = partial_rr + min_progress
        giveback_floor = max(float(metadata.get("partial_profit_stop_lock_rr", 0.0) or 0.0), partial_rr - max_giveback)
        if mfe_r < progress_target and current_rr <= giveback_floor:
            return (
                "WINNER_FOLLOW_THROUGH_FAIL",
                close_price,
                {
                    "winner_follow_through_fail": True,
                    "current_rr": float(current_rr),
                    "partial_profit_rr_move": float(partial_rr),
                    "mfe_r": float(mfe_r),
                    "progress_target_rr": float(progress_target),
                    "giveback_floor_rr": float(giveback_floor),
                },
            )
        return "OPEN", None, {}

    def _thesis_failure_exit(self, position: Position, close_price: float) -> tuple[str, float | None]:
        strategy = str(getattr(position, "strategy", "unknown") or "unknown").lower()
        if strategy not in {"trend_breakout", "trend_pullback", "mean_reversion"}:
            return "OPEN", None
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        base_sl = float(getattr(position, "initial_stop_loss", None) or getattr(position, "stop_loss", 0.0) or 0.0)
        risk_dist = abs(entry_price - base_sl)
        if risk_dist <= 0 or close_price <= 0:
            return "OPEN", None
        metadata = dict(getattr(position, "metadata", {}) or {})
        mfe_r = float(metadata.get("mfe_r", 0.0) or 0.0)
        side = str(getattr(position, "side", "long") or "long")
        close_rr = ((close_price - entry_price) / risk_dist) if side == "long" else ((entry_price - close_price) / risk_dist)
        signal_snapshot = dict(metadata.get("signal_snapshot", {}) or {})
        signal_meta = dict(signal_snapshot.get("metadata", {}) or {})
        if strategy == "trend_breakout":
            activation_rr = float(getattr(self.config, "breakout_thesis_fail_activation_rr", 0.10) or 0.10)
            close_floor = float(getattr(self.config, "breakout_thesis_fail_close_rr", -0.20) or -0.20)
            close_floor = self._regime_adjusted_close_floor_rr(position, close_floor)
            structure_buffer_rr = float(getattr(self.config, "breakout_thesis_fail_structure_buffer_rr", 0.08) or 0.08)
            breakout_level = float(signal_meta.get("breakout_level", signal_snapshot.get("breakout_level", 0.0)) or 0.0)
            if mfe_r <= activation_rr and close_rr <= close_floor:
                return "BREAKOUT_THESIS_FAIL", close_price
            if bool(metadata.get("volatility_tightened", False)) and close_rr <= max(close_floor, -0.08):
                return "BREAKOUT_THESIS_FAIL", close_price
            if breakout_level > 0.0:
                failed_level = close_price < (breakout_level - (risk_dist * structure_buffer_rr)) if side == "long" else close_price > (breakout_level + (risk_dist * structure_buffer_rr))
                if failed_level and close_rr <= max(close_floor, -0.05):
                    return "BREAKOUT_STRUCTURE_FAIL", close_price
        elif strategy == "trend_pullback":
            activation_rr = float(getattr(self.config, "pullback_thesis_fail_activation_rr", 0.05) or 0.05)
            close_floor = float(getattr(self.config, "pullback_thesis_fail_close_rr", -0.15) or -0.15)
            close_floor = self._regime_adjusted_close_floor_rr(position, close_floor)
            trend_persistence = float(signal_meta.get("trend_persistence", signal_snapshot.get("trend_persistence", 0.0)) or 0.0)
            min_trend_persistence = float(getattr(self.config, "pullback_thesis_fail_min_trend_persistence", 0.50) or 0.50)
            if bool(signal_meta.get("high_beta_quality_escape", False)):
                high_beta_floor = float(getattr(self.config, "high_beta_pullback_thesis_fail_close_rr", -0.08) or -0.08)
                if close_rr <= high_beta_floor and (mfe_r < 0.55 or trend_persistence < min_trend_persistence):
                    return "HIGH_BETA_PULLBACK_THESIS_FAIL", close_price
            if close_rr <= close_floor and (mfe_r <= activation_rr or (trend_persistence > 0.0 and trend_persistence < min_trend_persistence)):
                return "PULLBACK_THESIS_FAIL", close_price
            reclaim_level = float(signal_meta.get("reclaim_level", signal_snapshot.get("reclaim_level", 0.0)) or 0.0)
            reclaim_buffer_rr = float(getattr(self.config, "pullback_thesis_fail_reclaim_buffer_rr", 0.06) or 0.06)
            if reclaim_level > 0.0:
                failed_reclaim = close_price < (reclaim_level - (risk_dist * reclaim_buffer_rr)) if side == "long" else close_price > (reclaim_level + (risk_dist * reclaim_buffer_rr))
                if failed_reclaim and close_rr <= max(close_floor, -0.05):
                    return "PULLBACK_RECLAIM_FAIL", close_price
            structure_level = float(
                signal_meta.get(
                    "structure_support",
                    signal_snapshot.get("structure_support", signal_meta.get("structure_resistance", signal_snapshot.get("structure_resistance", 0.0))),
                )
                or 0.0
            )
            structure_buffer_rr = float(getattr(self.config, "pullback_thesis_fail_structure_buffer_rr", 0.10) or 0.10)
            if structure_level > 0.0:
                failed_structure = close_price < (structure_level - (risk_dist * structure_buffer_rr)) if side == "long" else close_price > (structure_level + (risk_dist * structure_buffer_rr))
                if failed_structure and close_rr <= max(close_floor, -0.05):
                    return "PULLBACK_STRUCTURE_FAIL", close_price
        else:
            close_floor = float(getattr(self.config, "mean_reversion_thesis_fail_close_rr", -0.12) or -0.12)
            close_floor = self._regime_adjusted_close_floor_rr(position, close_floor)
            if close_rr <= close_floor and mfe_r < 0.20:
                return "MEAN_REVERSION_THESIS_FAIL", close_price
        return "OPEN", None

    def _mean_reversion_reclaim_failure_exit(self, position: Position, close_price: float) -> tuple[str, float | None]:
        strategy = str(getattr(position, "strategy", "unknown") or "unknown").lower()
        if strategy != "mean_reversion":
            return "OPEN", None
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        base_sl = float(getattr(position, "initial_stop_loss", None) or getattr(position, "stop_loss", 0.0) or 0.0)
        risk_dist = abs(entry_price - base_sl)
        if risk_dist <= 0 or close_price <= 0:
            return "OPEN", None
        metadata = dict(getattr(position, "metadata", {}) or {})
        mfe_r = float(metadata.get("mfe_r", 0.0) or 0.0)
        activation_rr = float(getattr(self.config, "mean_reversion_reclaim_failure_activation_rr", 0.25) or 0.25)
        if mfe_r < activation_rr:
            return "OPEN", None
        buffer_rr = float(getattr(self.config, "mean_reversion_reclaim_failure_buffer_rr", 0.05) or 0.05)
        side = str(getattr(position, "side", "long") or "long")
        if side == "long":
            reclaim_floor = entry_price + (risk_dist * buffer_rr)
            if close_price <= reclaim_floor:
                return "MEAN_REVERSION_RECLAIM_FAIL", close_price
        else:
            reclaim_ceiling = entry_price - (risk_dist * buffer_rr)
            if close_price >= reclaim_ceiling:
                return "MEAN_REVERSION_RECLAIM_FAIL", close_price
        return "OPEN", None

    def _apply_exit_costs(self, position: Position, raw_price: float, bar: "pd.Series") -> tuple[float, float]:
        side = str(getattr(position, "side", "long"))
        spread_fraction = self.exchange.estimate_spread_fraction(position.symbol) if self.exchange is not None else float(getattr(self.config, "backtest_spread_bps", 4.0)) / 10000.0
        base_slippage = float(getattr(self.config, "backtest_slippage_bps", 5.0)) / 10000.0
        volatility_slippage = max(float(bar["high"]) - float(bar["low"]), 0.0) / max(float(bar["close"]), 1e-9) * float(getattr(self.config, "simulation_slippage_volatility_weight", 0.12))
        slippage = base_slippage + volatility_slippage
        direction = -1.0 if side == "long" else 1.0
        adjusted = raw_price * (1.0 + (direction * ((spread_fraction / 2.0) + slippage)))
        metadata = dict(getattr(position, "metadata", {}) or {})
        protected_lock_rr = float(metadata.get("partial_profit_stop_lock_rr", 0.0) or 0.0)
        if protected_lock_rr > 0.0:
            entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
            base_sl = float(getattr(position, "initial_stop_loss", None) or getattr(position, "stop_loss", 0.0) or 0.0)
            risk_dist = abs(entry_price - base_sl)
            min_lock_rr = max(protected_lock_rr, 0.0)
            if entry_price > 0.0 and risk_dist > 0.0 and min_lock_rr > 0.0:
                if side == "long" and raw_price > entry_price:
                    adjusted = max(adjusted, entry_price + (risk_dist * min_lock_rr))
                elif side != "long" and raw_price < entry_price:
                    adjusted = min(adjusted, entry_price - (risk_dist * min_lock_rr))
        fee_rate = float(getattr(self.config, "backtest_fee_bps", 10.0)) / 10000.0
        fee = abs(adjusted * float(getattr(position, "size", 0.0) or 0.0) * fee_rate)
        return adjusted, fee

    def _mark_to_market(self, bar_by_symbol: Mapping[str, "pd.Series"]) -> None:
        unrealized = 0.0
        for symbol, current in self.state.open_positions.items():
            close_price = float(bar_by_symbol.get(symbol, {}).get("close", 0.0) if symbol in bar_by_symbol else 0.0)
            positions = current if isinstance(current, list) else [current]
            for position in positions:
                if close_price <= 0:
                    continue
                unrealized += self._gross_pl(position.side, float(position.entry_price), close_price, float(position.size))
        self.state.unrealized_pnl = unrealized
        self.balance_curve.append(float(self.state.balance) + unrealized)
        if float(self.state.balance) > float(getattr(self.state, "peak_equity", 0.0) or 0.0):
            self.state.peak_equity = float(self.state.balance)

    def _liquidate_positions(self, bar_by_symbol: Mapping[str, "pd.Series"], now: dt.datetime) -> None:
        if not self.state.open_positions:
            return
        for symbol in list(self.state.open_positions.keys()):
            positions = self.state.open_positions.get(symbol, [])
            values = positions if isinstance(positions, list) else [positions]
            bar = bar_by_symbol.get(symbol)
            if bar is None:
                continue
            remaining: list[Position] = []
            for position in values:
                exit_price, exit_fee = self._apply_exit_costs(position, float(bar["close"]), bar)
                gross_pl = self._gross_pl(position.side, float(position.entry_price), exit_price, float(position.size))
                total_profit_loss = gross_pl - float(getattr(position, "fee_paid", 0.0) or 0.0) - exit_fee
                self.state.balance += gross_pl - exit_fee
                self.risk.update_after_trade_result(total_profit_loss)
                self.learning.record_closed_trade(
                    symbol=symbol,
                    position=position,
                    close_price=exit_price,
                    profit_loss=total_profit_loss,
                    exit_reason="SESSION_END",
                    closed_at=now.isoformat(),
                )
                self.trades.append(
                    {
                        "symbol": symbol,
                        "strategy": getattr(position, "strategy", "unknown"),
                        "side": getattr(position, "side", "long"),
                        "entry_time": getattr(position, "opened_at", now),
                        "exit_time": now,
                        "entry_price": getattr(position, "entry_price", 0.0),
                        "exit_price": exit_price,
                        "size": getattr(position, "size", 0.0),
                        "gross_pl": gross_pl,
                        "pl": total_profit_loss,
                        "fees": float(getattr(position, "fee_paid", 0.0) or 0.0) + exit_fee,
                        "holding_minutes": max((now - getattr(position, "opened_at", now)).total_seconds() / 60.0, 0.0),
                        "exit_reason": "SESSION_END",
                        "mfe_r": float((getattr(position, "metadata", {}) or {}).get("mfe_r", 0.0) or 0.0),
                        "mae_r": float((getattr(position, "metadata", {}) or {}).get("mae_r", 0.0) or 0.0),
                        "giveback_r": self._trade_giveback_r(position, total_profit_loss),
                    }
                )
            if remaining:
                self.state.open_positions[symbol] = remaining
            else:
                del self.state.open_positions[symbol]

    def _apply_drawdown_controls(self) -> None:
        if float(self.state.peak_equity or 0.0) <= 0:
            self.state.peak_equity = float(self.state.balance)
            return
        drawdown = (float(self.state.balance) - float(self.state.peak_equity)) / max(float(self.state.peak_equity), 1e-9)
        if drawdown <= -0.20:
            self.state.emergency_mode = True
            self.state.reduced_risk_mode = True
        elif drawdown <= -0.10:
            self.state.reduced_risk_mode = True
        elif self.state.consecutive_losses == 0:
            self.state.reduced_risk_mode = False

    def _persist_progress_snapshot(self, current_index: int, total_bars: int, timeline: Sequence[dt.datetime]) -> None:
        interval = max(int(getattr(self.config, "simulation_snapshot_interval_bars", 250)), 1)
        if current_index % interval != 0 and current_index + 1 != total_bars:
            return
        snapshot = {
            "snapshot_key": "simulation_runtime",
            "updated_at": self._now.isoformat() if self._now else dt.datetime.utcnow().isoformat(),
            "progress": {
                "current_bar": current_index,
                "total_bars": total_bars,
                "submitted_orders": self._submitted_orders,
                "filled_orders": self._filled_orders,
                "expired_orders": self._expired_orders,
                "raw_signals": self._raw_signals,
                "cancelled_orders": self._cancelled_orders,
                "ambiguous_exit_bars": self._ambiguous_exit_bars,
                "checkpoint_path": self._checkpoint_path,
            },
            "portfolio": asdict(build_portfolio_snapshot(self._bot_adapter)),
            "learning": self.learning.summary_snapshot(),
            "artifacts": self._artifact_manifest(),
        }
        self.state_store.persist_snapshot(snapshot)
        if bool(getattr(self.config, "simulation_enable_checkpointing", True)):
            checkpoint_interval = max(int(getattr(self.config, "simulation_checkpoint_interval_bars", interval)), 1)
            if current_index % checkpoint_interval == 0 or current_index + 1 == total_bars:
                self._persist_checkpoint(current_index, total_bars, timeline)
                self._write_artifact_manifest()

    def _label_triple_barrier_signal(self, signal: Mapping[str, Any]) -> Dict[str, Any] | None:
        if not bool(getattr(self.config, "triple_barrier_labeling_enabled", True)):
            return None
        if self.exchange is None or getattr(self.exchange, "current_time", None) is None:
            return {"label": "UNAVAILABLE", "reason": "missing_exchange_time"}
        symbol = str(signal.get("symbol", "") or "")
        side = str(signal.get("side", "long") or "long").lower()
        timeframe = str(getattr(self, "_base_timeframe", "") or getattr(self.exchange, "base_timeframe", "15m") or "15m")
        try:
            entry_price = float(signal.get("entry_price", 0.0) or 0.0)
            stop_loss = float(signal.get("stop_loss", 0.0) or 0.0)
            take_profit = float(signal.get("take_profit", 0.0) or 0.0)
        except (TypeError, ValueError):
            return {"label": "UNAVAILABLE", "reason": "invalid_signal_prices"}
        if not symbol or entry_price <= 0.0 or stop_loss <= 0.0 or take_profit <= 0.0:
            return {"label": "UNAVAILABLE", "reason": "missing_signal_prices"}
        if side == "short":
            if not (take_profit < entry_price < stop_loss):
                return {"label": "UNAVAILABLE", "reason": "invalid_short_barriers"}
        elif not (stop_loss < entry_price < take_profit):
            return {"label": "UNAVAILABLE", "reason": "invalid_long_barriers"}

        timeline = self.exchange.universe.timeline([symbol], timeframe=timeframe)
        current_time = _normalize_timestamp(getattr(self.exchange, "current_time")).to_pydatetime()
        try:
            cursor = timeline.index(current_time)
        except ValueError:
            return {"label": "UNAVAILABLE", "reason": "current_time_not_in_symbol_timeline"}
        horizon_bars = max(int(getattr(self.config, "triple_barrier_label_horizon_bars", 12) or 12), 1)
        label = "TIME_EXIT"
        bars_to_label = min(horizon_bars, max(len(timeline) - cursor - 1, 0))
        hit_time: str | None = None
        for offset, future_time in enumerate(timeline[cursor + 1 : cursor + 1 + horizon_bars], start=1):
            bar = self.exchange.universe.bar_for_time(symbol, timeframe, future_time)
            if bar is None:
                continue
            high = float(bar["high"])
            low = float(bar["low"])
            if side == "short":
                hit_tp = low <= take_profit
                hit_sl = high >= stop_loss
            else:
                hit_tp = high >= take_profit
                hit_sl = low <= stop_loss
            if hit_tp and hit_sl:
                label = "AMBIGUOUS"
            elif hit_tp:
                label = "TP_FIRST"
            elif hit_sl:
                label = "SL_FIRST"
            else:
                continue
            bars_to_label = offset
            hit_time = future_time.isoformat()
            break

        bucket_key = self._triple_barrier_bucket_key(signal)
        return {
            "label": label,
            "symbol": symbol,
            "strategy": str(signal.get("strategy", "unknown") or "unknown"),
            "side": side,
            "timeframe": timeframe,
            "horizon_bars": horizon_bars,
            "bars_to_label": int(bars_to_label),
            "hit_time": hit_time,
            "bucket_key": bucket_key,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def _triple_barrier_bucket_key(self, signal: Mapping[str, Any]) -> str:
        metadata = dict(signal.get("metadata", {}) or {})

        def bucket(value: Any, low: float, high: float, prefix: str) -> str:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return f"{prefix}_unknown"
            if numeric < low:
                return f"{prefix}_low"
            if numeric < high:
                return f"{prefix}_mid"
            return f"{prefix}_high"

        strategy = str(signal.get("strategy", "unknown") or "unknown")
        symbol = str(signal.get("symbol", "unknown") or "unknown")
        regime = str(signal.get("regime", metadata.get("regime", "unknown")) or "unknown")
        pullback = bucket(metadata.get("pullback_score", signal.get("pullback_score")), 1.5, 2.5, "pullback")
        volatility = bucket(metadata.get("realized_vol_percentile", metadata.get("volatility_percentile")), 0.33, 0.66, "vol")
        liquidity = bucket(metadata.get("liquidity_score", signal.get("liquidity_score")), 0.50, 0.80, "liq")
        return "|".join([strategy, symbol, regime, pullback, volatility, liquidity])

    def _record_triple_barrier_label(self, signal: Mapping[str, Any], label_payload: Mapping[str, Any]) -> None:
        label = str(label_payload.get("label", "UNAVAILABLE") or "UNAVAILABLE")
        strategy = str(signal.get("strategy", "unknown") or "unknown")
        symbol = str(signal.get("symbol", "unknown") or "unknown")
        bucket_key = str(label_payload.get("bucket_key", self._triple_barrier_bucket_key(signal)) or "unknown")
        self._increment_counter(self._triple_barrier_label_counts, label)
        self._increment_nested_counter(self._triple_barrier_label_counts_by_strategy, strategy, label)
        self._increment_nested_counter(self._triple_barrier_label_counts_by_symbol, symbol, label)
        bucket = self._triple_barrier_bucket_stats.setdefault(
            bucket_key,
            {
                "samples": 0,
                "label_counts": {},
                "strategy": strategy,
                "symbol": symbol,
            },
        )
        bucket["samples"] = int(bucket.get("samples", 0) or 0) + 1
        label_counts = bucket.setdefault("label_counts", {})
        label_counts[label] = int(label_counts.get(label, 0) or 0) + 1
        if len(self._triple_barrier_labels) < 250:
            self._triple_barrier_labels.append(
                {
                    "label": label,
                    "strategy": strategy,
                    "symbol": symbol,
                    "side": str(label_payload.get("side", signal.get("side", "long")) or "long"),
                    "bars_to_label": int(label_payload.get("bars_to_label", 0) or 0),
                    "bucket_key": bucket_key,
                }
            )

    def _pullback_meta_filter_reason(self, signal: Mapping[str, Any]) -> str | None:
        if not bool(getattr(self.config, "pullback_meta_filter_enabled", True)):
            return None
        if str(signal.get("strategy", "") or "") != "trend_pullback":
            return None
        quality = float(signal.get("signal_quality", signal.get("confidence", 0.0)) or 0.0)
        edge_bps = float(signal.get("expected_edge_bps", 0.0) or 0.0)
        if (
            quality >= float(getattr(self.config, "pullback_meta_filter_quality_escape_score", 0.82) or 0.82)
            and edge_bps >= float(getattr(self.config, "pullback_meta_filter_quality_escape_edge_bps", 45.0) or 45.0)
        ):
            return None
        metadata = dict(signal.get("metadata", {}) or {})
        label_payload = dict(metadata.get("triple_barrier_label", {}) or {})
        bucket_key = str(label_payload.get("bucket_key", self._triple_barrier_bucket_key(signal)) or "unknown")
        bucket = dict(self._triple_barrier_bucket_stats.get(bucket_key, {}) or {})
        label_counts = dict(bucket.get("label_counts", {}) or {})

        # The current signal is labelled for diagnostics before filtering. Remove it
        # from the decision sample so the filter only learns from prior bucket evidence.
        current_label = str(label_payload.get("label", "") or "")
        if current_label and current_label in label_counts:
            label_counts[current_label] = max(int(label_counts.get(current_label, 0) or 0) - 1, 0)
        samples = sum(int(value or 0) for value in label_counts.values())
        if samples < max(int(getattr(self.config, "pullback_meta_filter_min_bucket_samples", 3) or 3), 1):
            return None
        sl_first = int(label_counts.get("SL_FIRST", 0) or 0)
        tp_first = int(label_counts.get("TP_FIRST", 0) or 0)
        sl_first_pct = sl_first / samples * 100.0 if samples else 0.0
        tp_first_pct = tp_first / samples * 100.0 if samples else 0.0
        max_sl_pct = float(getattr(self.config, "pullback_meta_filter_max_sl_first_pct", 60.0) or 60.0)
        min_tp_pct = float(getattr(self.config, "pullback_meta_filter_min_tp_first_pct", 20.0) or 20.0)
        if sl_first_pct >= max_sl_pct and tp_first_pct <= min_tp_pct:
            metadata["pullback_meta_filter"] = {
                "bucket_key": bucket_key,
                "samples": int(samples),
                "sl_first_pct": sl_first_pct,
                "tp_first_pct": tp_first_pct,
                "label_counts": dict(sorted(label_counts.items())),
            }
            signal["metadata"] = metadata  # type: ignore[index]
            return "pullback_meta_filter_stop_first_bucket"
        return None

    def _build_triple_barrier_summary(self) -> Dict[str, Any]:
        total = sum(int(value or 0) for value in self._triple_barrier_label_counts.values())
        bucket_stats: Dict[str, Dict[str, Any]] = {}
        min_samples = max(int(getattr(self.config, "triple_barrier_label_min_bucket_samples", 1) or 1), 1)
        for key, payload in sorted(self._triple_barrier_bucket_stats.items()):
            samples = int(payload.get("samples", 0) or 0)
            if samples < min_samples:
                continue
            label_counts = dict(sorted(dict(payload.get("label_counts", {}) or {}).items()))
            bucket_stats[key] = {
                "samples": samples,
                "label_counts": label_counts,
                "tp_first_rate_pct": (float(label_counts.get("TP_FIRST", 0) or 0) / samples * 100.0) if samples else 0.0,
                "sl_first_rate_pct": (float(label_counts.get("SL_FIRST", 0) or 0) / samples * 100.0) if samples else 0.0,
                "time_exit_rate_pct": (float(label_counts.get("TIME_EXIT", 0) or 0) / samples * 100.0) if samples else 0.0,
                "ambiguous_rate_pct": (float(label_counts.get("AMBIGUOUS", 0) or 0) / samples * 100.0) if samples else 0.0,
                "strategy": str(payload.get("strategy", "unknown") or "unknown"),
                "symbol": str(payload.get("symbol", "unknown") or "unknown"),
            }
        return {
            "labels": int(total),
            "label_counts": dict(sorted(self._triple_barrier_label_counts.items())),
            "label_counts_by_strategy": {
                strategy: dict(sorted(counts.items()))
                for strategy, counts in sorted(self._triple_barrier_label_counts_by_strategy.items())
            },
            "label_counts_by_symbol": {
                symbol: dict(sorted(counts.items()))
                for symbol, counts in sorted(self._triple_barrier_label_counts_by_symbol.items())
            },
            "bucket_stats": bucket_stats,
            "examples": list(self._triple_barrier_labels[:12]),
        }

    def _build_results(self, *, symbol: str, timeframe: str, days: int, timeline: Sequence[dt.datetime]) -> Dict[str, Any]:
        metrics = self.compute_metrics()
        learning_summary = self.learning.summary_snapshot()
        pending_shadow = len(self.state_store.load_pending_learning_decisions(limit=500))
        fill_ratios = [
            float(item.get("fill_fraction", 1.0) or 1.0)
            for item in self.trades
            if isinstance(item, dict) and "fill_fraction" in item
        ]
        impact_bps = [
            float(((item.get("exit_details", {}) or {}).get("market_impact_bps", item.get("market_impact_bps", 0.0)) or 0.0))
            for item in self.trades
            if isinstance(item, dict)
        ]
        symbol_rollups = self._build_symbol_rollups()
        campaign_diagnostics = self._build_campaign_diagnostics(symbol_rollups)
        trade_frequency = self._build_trade_frequency_summary(days=days, timeline=timeline)
        validation_harness = self._build_validation_harness_summary(symbol_rollups)
        triple_barrier_summary = self._build_triple_barrier_summary()
        result = {
            **metrics,
            "raw_trades": len(self.trades),
            "total_fees": sum(float(trade.get("fees", 0.0) or 0.0) for trade in self.trades),
            "long_trades": sum(1 for trade in self.trades if trade.get("side") == "long"),
            "short_trades": sum(1 for trade in self.trades if trade.get("side") == "short"),
            "trade_log": self.trades,
            "symbol_rollups": symbol_rollups,
            "raw_signals": self._raw_signals,
            "skipped_signals": self._skipped_signals,
            "submitted_orders": self._submitted_orders,
            "filled_orders": self._filled_orders,
            "expired_orders": self._expired_orders,
            "cancelled_orders": self._cancelled_orders,
            "partial_fills": self._partial_fills,
            "avg_fill_fraction": float(sum(fill_ratios) / len(fill_ratios)) if fill_ratios else 0.0,
            "ambiguous_exit_bars": self._ambiguous_exit_bars,
            "learning_summary": learning_summary,
            "shadow_decisions_pending": pending_shadow,
            "event_counts": dict(self._event_counts),
            "portfolio_snapshot": asdict(build_portfolio_snapshot(self._bot_adapter)),
            "artifact_dir": self.artifact_dir,
            "session_start": timeline[0].isoformat() if timeline else None,
            "session_end": (self._now.isoformat() if self._stopped_early and self._now is not None else (timeline[-1].isoformat() if timeline else None)),
            "symbol": symbol,
            "timeframe": timeframe,
            "days": days,
            "stopped_early": self._stopped_early,
            "stop_reason": self._stop_reason,
            "resumed_from_checkpoint": self._resumed_from_checkpoint,
            "open_orders_remaining": len(self.venue.open_orders) if self.venue is not None else 0,
            "simulation_artifacts": self._artifact_manifest(),
            "execution_realism": {
                "avg_market_impact_bps": float(sum(impact_bps) / len(impact_bps)) if impact_bps else 0.0,
                "queue_model": str(getattr(self.config, "simulation_queue_model", "fractional_queue")),
                "bar_exit_model": "conservative_worst_case_for_ambiguous_bars",
                "stress_enabled": bool(int(getattr(self.config, "simulation_stress_every_n_bars", 0) or 0) > 0 and float(getattr(self.config, "simulation_stress_shock_bps", 0.0) or 0.0) > 0),
            },
            "campaign_summary": {
                "performance": {
                    "num_trades": int(metrics.get("num_trades", 0) or 0),
                    "win_rate_pct": float(metrics.get("win_rate_pct", 0.0) or 0.0),
                    "total_return_pct": float(metrics.get("total_return_pct", 0.0) or 0.0),
                    "profit_factor": float(metrics.get("profit_factor", 0.0) or 0.0),
                    "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0) or 0.0),
                    "expectancy": float(metrics.get("expectancy", 0.0) or 0.0),
                    "avg_holding_minutes": float(metrics.get("avg_holding_minutes", 0.0) or 0.0),
                },
                "trade_flow": {
                    "proposals": int(self._proposal_count),
                    "raw_signals": int(self._raw_signals),
                    "skipped_signals": int(self._skipped_signals),
                    "repeated_setup_blocks": int(self._repeated_setup_blocks),
                    "fresh_setup_blocks": int(self._fresh_setup_blocks),
                    "limit_to_market_upgrades": int(self._limit_to_market_upgrades),
                    "limit_queue_priority_assists": int(self._limit_queue_priority_assists),
                    "limit_latency_reductions": int(self._limit_latency_reductions),
                    "stale_market_escalations": int(self._stale_market_escalations),
                    "touch_escalations": int(getattr(self.venue, "touch_escalations", 0) if self.venue is not None else 0),
                    "partial_profit_takes": int(self._partial_profit_takes),
                    "submitted_orders": int(self._submitted_orders),
                    "repriced_orders": int(self._repriced_orders),
                    "filled_orders": int(self._filled_orders),
                    "closed_trades": len(self.trades),
                    "cancelled_orders": int(self._cancelled_orders),
                    "expired_orders": int(self._expired_orders),
                    "partial_fills": int(self._partial_fills),
                },
                "session": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "days": int(days),
                    "session_start": timeline[0].isoformat() if timeline else None,
                    "session_end": (self._now.isoformat() if self._stopped_early and self._now is not None else (timeline[-1].isoformat() if timeline else None)),
                    "stopped_early": bool(self._stopped_early),
                    "stop_reason": self._stop_reason,
                    "resumed_from_checkpoint": bool(self._resumed_from_checkpoint),
                },
                "universe_selection": dict(self._latest_universe_selection),
                "market_data": {
                    "datasets": dict(sorted(self._historical_data_rows.items())),
                    "datasets_loaded": int(sum(1 for rows in self._historical_data_rows.values() if int(rows or 0) > 0)),
                    "datasets_missing": int(sum(1 for rows in self._historical_data_rows.values() if int(rows or 0) <= 0)),
                    "total_rows": int(sum(int(rows or 0) for rows in self._historical_data_rows.values())),
                    "data_available": bool(any(int(rows or 0) > 0 for rows in self._historical_data_rows.values())),
                },
                "trade_frequency": trade_frequency,
                "acceptance": self._build_campaign_acceptance_summary(metrics=metrics, trade_frequency=trade_frequency, validation_harness=validation_harness),
                "realized_performance": self._build_realized_performance_summary(symbol_rollups),
                "exit_quality": self._build_exit_quality_summary(),
                "validation_harness": validation_harness,
                "triple_barrier": triple_barrier_summary,
            },
            "campaign_diagnostics": campaign_diagnostics,
            "decision_diagnostics": {
                "proposal_count": int(self._proposal_count),
                "proposals_by_strategy": dict(sorted(self._proposals_by_strategy.items())),
                "skip_reasons": dict(sorted(self._skip_reason_counts.items())),
                "generation_outcomes": dict(sorted(self._generation_outcomes.items())),
                "generation_outcomes_by_symbol": {key: dict(sorted(value.items())) for key, value in sorted(self._generation_outcomes_by_symbol.items())},
                "generation_reasons_by_symbol": {key: dict(sorted(value.items())) for key, value in sorted(self._generation_reasons_by_symbol.items())},
                "raw_signals_by_symbol": dict(sorted(self._raw_signals_by_symbol.items())),
                "submitted_by_symbol": dict(sorted(self._submitted_by_symbol.items())),
                "filled_by_symbol": dict(sorted(self._filled_by_symbol.items())),
                "closed_by_symbol": dict(sorted(self._closed_by_symbol.items())),
                "wins_by_symbol": dict(sorted(self._wins_by_symbol.items())),
                "losses_by_symbol": dict(sorted(self._losses_by_symbol.items())),
                "skipped_by_symbol": dict(sorted(self._skipped_by_symbol.items())),
                "skip_reasons_by_symbol": {key: dict(sorted(value.items())) for key, value in sorted(self._skip_reasons_by_symbol.items())},
                "raw_signals_by_strategy_by_symbol": {key: dict(sorted(value.items())) for key, value in sorted(self._raw_signals_by_strategy_by_symbol.items())},
                "submitted_by_strategy_by_symbol": {key: dict(sorted(value.items())) for key, value in sorted(self._submitted_by_strategy_by_symbol.items())},
                "filled_by_strategy_by_symbol": {key: dict(sorted(value.items())) for key, value in sorted(self._filled_by_strategy_by_symbol.items())},
                "closed_by_strategy_by_symbol": {key: dict(sorted(value.items())) for key, value in sorted(self._closed_by_strategy_by_symbol.items())},
                "wins_by_strategy_by_symbol": {key: dict(sorted(value.items())) for key, value in sorted(self._wins_by_strategy_by_symbol.items())},
                "losses_by_strategy_by_symbol": {key: dict(sorted(value.items())) for key, value in sorted(self._losses_by_strategy_by_symbol.items())},
                "skip_reasons_by_strategy_by_symbol": {
                    symbol_key: {
                        strategy_key: dict(sorted(reason_counts.items()))
                        for strategy_key, reason_counts in sorted(strategy_counts.items())
                    }
                    for symbol_key, strategy_counts in sorted(self._skip_reasons_by_strategy_by_symbol.items())
                },
                "family_rotation_counts": dict(sorted(self._family_rotation_counts.items())),
                "family_rotation_counts_by_strategy": {
                    strategy: dict(sorted(reason_counts.items()))
                    for strategy, reason_counts in sorted(self._family_rotation_counts_by_strategy.items())
                },
                "family_rotation_recovery_counts": dict(sorted(self._family_rotation_recovery_counts.items())),
                "family_rotation_recovery_counts_by_strategy": {
                    strategy: dict(sorted(reason_counts.items()))
                    for strategy, reason_counts in sorted(self._family_rotation_recovery_counts_by_strategy.items())
                },
                "learning_evidence_counts": dict(sorted(self._learning_evidence_counts.items())),
                "learning_evidence_counts_by_strategy": {
                    strategy: dict(sorted(reason_counts.items()))
                    for strategy, reason_counts in sorted(self._learning_evidence_counts_by_strategy.items())
                },
                "learning_asymmetry_counts": dict(sorted(self._learning_asymmetry_counts.items())),
                "learning_asymmetry_counts_by_strategy": {
                    strategy: dict(sorted(reason_counts.items()))
                    for strategy, reason_counts in sorted(self._learning_asymmetry_counts_by_strategy.items())
                },
                "missed_opportunity_relaxations": dict(sorted(self._missed_opportunity_relaxations.items())),
                "missed_opportunity_relaxations_by_strategy": {
                    strategy: dict(sorted(reason_counts.items()))
                    for strategy, reason_counts in sorted(self._missed_opportunity_relaxations_by_strategy.items())
                },
                "frequency_expansion_allowed_count": int(self._frequency_expansion_allowed_count),
                "frequency_expansion_blocked_count": int(self._frequency_expansion_blocked_count),
                "frequency_expansion_block_reasons": dict(sorted(self._frequency_expansion_block_reasons.items())),
                "signals_by_strategy": dict(sorted(self._signals_by_strategy.items())),
                "signals_by_regime": dict(sorted(self._signals_by_regime.items())),
                "signals_by_order_type": dict(sorted(self._signals_by_order_type.items())),
                "triple_barrier_label_counts": dict(sorted(self._triple_barrier_label_counts.items())),
                "triple_barrier_label_counts_by_strategy": {
                    strategy: dict(sorted(counts.items()))
                    for strategy, counts in sorted(self._triple_barrier_label_counts_by_strategy.items())
                },
                "triple_barrier_label_counts_by_symbol": {
                    symbol: dict(sorted(counts.items()))
                    for symbol, counts in sorted(self._triple_barrier_label_counts_by_symbol.items())
                },
                "pullback_meta_filter_blocks": int(self._pullback_meta_filter_blocks),
                "pullback_meta_filter_blocks_by_bucket": dict(sorted(self._pullback_meta_filter_blocks_by_bucket.items())),
                "pullback_meta_filter_blocks_by_symbol": dict(sorted(self._pullback_meta_filter_blocks_by_symbol.items())),
                "limit_to_market_upgrades": int(self._limit_to_market_upgrades),
                "limit_to_market_upgrades_by_strategy": dict(sorted(self._limit_to_market_upgrades_by_strategy.items())),
                "repeated_setup_blocks": int(self._repeated_setup_blocks),
                "repeated_setup_blocks_by_strategy": dict(sorted(self._repeated_setup_blocks_by_strategy.items())),
                "fresh_setup_blocks": int(self._fresh_setup_blocks),
                "fresh_setup_blocks_by_strategy": dict(sorted(self._fresh_setup_blocks_by_strategy.items())),
                "replacement_candidates_seen": int(self._replacement_candidates_seen),
                "replacement_candidates_selected": int(self._replacement_candidates_selected),
                "replacement_candidates_by_strategy": dict(sorted(self._replacement_candidates_by_strategy.items())),
                "replacement_selected_by_strategy": dict(sorted(self._replacement_selected_by_strategy.items())),
                "replacement_candidates_by_symbol": dict(sorted(self._replacement_candidates_by_symbol.items())),
                "replacement_rejections_by_reason": dict(sorted(self._replacement_rejections_by_reason.items())),
                "replacement_cross_symbol_selected": int(self._replacement_cross_symbol_selected),
                "replacement_cross_symbol_selected_by_strategy": dict(sorted(self._replacement_cross_symbol_selected_by_strategy.items())),
                "replacement_near_misses_seen": int(self._replacement_near_misses_seen),
                "replacement_near_misses_by_strategy": dict(sorted(self._replacement_near_misses_by_strategy.items())),
                "replacement_near_misses_by_symbol": dict(sorted(self._replacement_near_misses_by_symbol.items())),
                "replacement_near_misses_by_reason": dict(sorted(self._replacement_near_misses_by_reason.items())),
                "replacement_near_misses_by_detail": dict(sorted(self._replacement_near_misses_by_detail.items())),
                "replacement_near_miss_examples": list(self._replacement_near_miss_examples),
                "replacement_submitted": int(self._replacement_submitted),
                "replacement_filled": int(self._replacement_filled),
                "replacement_closed": int(self._replacement_closed),
                "replacement_wins": int(self._replacement_wins),
                "replacement_losses": int(self._replacement_losses),
                "replacement_submitted_by_strategy": dict(sorted(self._replacement_submitted_by_strategy.items())),
                "replacement_filled_by_strategy": dict(sorted(self._replacement_filled_by_strategy.items())),
                "replacement_closed_by_strategy": dict(sorted(self._replacement_closed_by_strategy.items())),
                "replacement_wins_by_strategy": dict(sorted(self._replacement_wins_by_strategy.items())),
                "replacement_losses_by_strategy": dict(sorted(self._replacement_losses_by_strategy.items())),
                "replacement_guard_blocks": int(self._replacement_guard_blocks),
                "replacement_guard_blocks_by_reason": dict(sorted(self._replacement_guard_blocks_by_reason.items())),
                "replacement_submitted_by_day": dict(sorted(self._replacement_submitted_by_day.items())),
                "replacement_submitted_by_symbol_day": {
                    day_key: dict(sorted(symbol_counts.items()))
                    for day_key, symbol_counts in sorted(self._replacement_submitted_by_symbol_day.items())
                },
                "limit_queue_priority_assists": int(self._limit_queue_priority_assists),
                "limit_queue_priority_assists_by_strategy": dict(sorted(self._limit_queue_priority_assists_by_strategy.items())),
                "limit_latency_reductions": int(self._limit_latency_reductions),
                "limit_latency_reductions_by_strategy": dict(sorted(self._limit_latency_reductions_by_strategy.items())),
                "stale_market_escalations": int(self._stale_market_escalations),
                "stale_market_escalations_by_strategy": dict(sorted(self._stale_market_escalations_by_strategy.items())),
                "touch_escalations": int(getattr(self.venue, "touch_escalations", 0) if self.venue is not None else 0),
                "touch_escalations_by_strategy": dict(sorted((getattr(self.venue, "touch_escalations_by_strategy", {}) if self.venue is not None else {}).items())),
                "partial_profit_takes": int(self._partial_profit_takes),
                "partial_profit_takes_by_strategy": dict(sorted(self._partial_profit_takes_by_strategy.items())),
                "submitted_by_strategy": dict(sorted(self._submitted_by_strategy.items())),
                "submitted_by_order_type": dict(sorted(self._submitted_by_order_type.items())),
                "repriced_orders": int(self._repriced_orders),
                "repriced_by_strategy": dict(sorted(self._repriced_by_strategy.items())),
                "filled_by_strategy": dict(sorted(self._filled_by_strategy.items())),
                "closed_by_strategy": dict(sorted(self._closed_by_strategy.items())),
                "closed_by_order_type": dict(sorted(self._closed_by_order_type.items())),
                "wins_by_strategy": dict(sorted(self._wins_by_strategy.items())),
                "losses_by_strategy": dict(sorted(self._losses_by_strategy.items())),
                "realized_performance_penalty_counts": dict(sorted(self._realized_performance_penalty_counts.items())),
                "realized_performance_penalty_counts_by_strategy": {
                    strategy: dict(sorted(reason_counts.items()))
                    for strategy, reason_counts in sorted(self._realized_performance_penalty_counts_by_strategy.items())
                },
                "realized_performance_no_trade_blocks": dict(sorted(self._realized_performance_no_trade_blocks.items())),
                "realized_performance_no_trade_blocks_by_strategy": {
                    strategy: dict(sorted(reason_counts.items()))
                    for strategy, reason_counts in sorted(self._realized_performance_no_trade_blocks_by_strategy.items())
                },
                "reentry_cooldown_registrations": dict(sorted(self._reentry_cooldown_registrations.items())),
                "reentry_cooldown_registrations_by_strategy": {
                    strategy: dict(sorted(reason_counts.items()))
                    for strategy, reason_counts in sorted(self._reentry_cooldown_registrations_by_strategy.items())
                },
            },
        }
        result["simulation_artifacts"]["report_path"] = os.path.join(self.artifact_dir, "report.json")
        return result

    def _build_campaign_diagnostics(self, symbol_rollups: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        by_symbol: Dict[str, Dict[str, Any]] = {}
        symbols = sorted(
            set(symbol_rollups)
            | set(self._generation_outcomes_by_symbol)
            | set(self._generation_reasons_by_symbol)
            | set(self._strategy_rejection_reasons_by_symbol)
            | set(self._raw_signals_by_strategy_by_symbol)
            | set(self._submitted_by_strategy_by_symbol)
            | set(self._filled_by_strategy_by_symbol)
            | set(self._closed_by_strategy_by_symbol)
            | set(self._wins_by_strategy_by_symbol)
            | set(self._losses_by_strategy_by_symbol)
            | set(self._skip_reasons_by_strategy_by_symbol)
        )
        by_strategy_by_symbol: Dict[str, Dict[str, Any]] = {}
        rejection_totals: Dict[str, int] = {}
        for symbol in symbols:
            by_symbol[symbol] = {
                "summary": dict(symbol_rollups.get(symbol, {})),
                "generation_outcomes": dict(sorted((self._generation_outcomes_by_symbol.get(symbol, {}) or {}).items())),
                "generation_reasons": dict(sorted((self._generation_reasons_by_symbol.get(symbol, {}) or {}).items())),
                "strategy_rejection_reasons": {
                    strategy: dict(sorted((reasons or {}).items()))
                    for strategy, reasons in sorted((self._strategy_rejection_reasons_by_symbol.get(symbol, {}) or {}).items())
                },
                "top_rejection_reasons": self._top_items(
                    {
                        **dict((self._generation_reasons_by_symbol.get(symbol, {}) or {})),
                        **{
                            f"{strategy}:{reason}": int(count or 0)
                            for strategy, reasons in dict((self._strategy_rejection_reasons_by_symbol.get(symbol, {}) or {}).items()).items()
                            for reason, count in dict(reasons or {}).items()
                        },
                        **{
                            f"post_selection:{reason}": int(count or 0)
                            for reason, count in dict((self._skip_reasons_by_symbol.get(symbol, {}) or {})).items()
                        },
                    },
                    limit=5,
                ),
            }

            strategies = sorted(
                set((self._raw_signals_by_strategy_by_symbol.get(symbol, {}) or {}))
                | set((self._submitted_by_strategy_by_symbol.get(symbol, {}) or {}))
                | set((self._filled_by_strategy_by_symbol.get(symbol, {}) or {}))
                | set((self._closed_by_strategy_by_symbol.get(symbol, {}) or {}))
                | set((self._wins_by_strategy_by_symbol.get(symbol, {}) or {}))
                | set((self._losses_by_strategy_by_symbol.get(symbol, {}) or {}))
                | set((self._skip_reasons_by_strategy_by_symbol.get(symbol, {}) or {}))
            )
            by_strategy_by_symbol[symbol] = {}
            for strategy in strategies:
                strategy_skip_reasons = dict(
                    sorted(
                        (
                            (
                                (self._skip_reasons_by_strategy_by_symbol.get(symbol, {}) or {}).get(strategy, {})
                                or {}
                            ).items()
                        )
                    )
                )
                by_strategy_by_symbol[symbol][strategy] = {
                    "raw_signals": int(((self._raw_signals_by_strategy_by_symbol.get(symbol, {}) or {}).get(strategy, 0) or 0)),
                    "submitted_orders": int(((self._submitted_by_strategy_by_symbol.get(symbol, {}) or {}).get(strategy, 0) or 0)),
                    "filled_orders": int(((self._filled_by_strategy_by_symbol.get(symbol, {}) or {}).get(strategy, 0) or 0)),
                    "closed_trades": int(((self._closed_by_strategy_by_symbol.get(symbol, {}) or {}).get(strategy, 0) or 0)),
                    "wins": int(((self._wins_by_strategy_by_symbol.get(symbol, {}) or {}).get(strategy, 0) or 0)),
                    "losses": int(((self._losses_by_strategy_by_symbol.get(symbol, {}) or {}).get(strategy, 0) or 0)),
                    "skip_reasons": strategy_skip_reasons,
                }
                for reason, count in strategy_skip_reasons.items():
                    rejection_totals[f"post_selection:{strategy}:{reason}"] = int(rejection_totals.get(f"post_selection:{strategy}:{reason}", 0)) + int(count or 0)

            for reason, count in dict((self._generation_reasons_by_symbol.get(symbol, {}) or {})).items():
                rejection_totals[f"pre_selection:{reason}"] = int(rejection_totals.get(f"pre_selection:{reason}", 0)) + int(count or 0)
            for strategy, reasons in dict((self._strategy_rejection_reasons_by_symbol.get(symbol, {}) or {}).items()).items():
                for reason, count in dict(reasons or {}).items():
                    rejection_totals[f"strategy:{strategy}:{reason}"] = int(rejection_totals.get(f"strategy:{strategy}:{reason}", 0)) + int(count or 0)

        return {
            "primary_summary": {
                "by_symbol": by_symbol,
                "by_strategy": {
                    "proposals": dict(sorted(self._proposals_by_strategy.items())),
                    "signals": dict(sorted(self._signals_by_strategy.items())),
                    "submitted": dict(sorted(self._submitted_by_strategy.items())),
                    "filled": dict(sorted(self._filled_by_strategy.items())),
                    "closed": dict(sorted(self._closed_by_strategy.items())),
                    "wins": dict(sorted(self._wins_by_strategy.items())),
                    "losses": dict(sorted(self._losses_by_strategy.items())),
                    "frequency_adjustments": {
                        strategy: dict(sorted(reason_counts.items()))
                        for strategy, reason_counts in sorted(self._frequency_adjustment_counts_by_strategy.items())
                    },
                    "family_rotation": {
                        strategy: dict(sorted(reason_counts.items()))
                        for strategy, reason_counts in sorted(self._family_rotation_counts_by_strategy.items())
                    },
                    "family_rotation_recovery": {
                        strategy: dict(sorted(reason_counts.items()))
                        for strategy, reason_counts in sorted(self._family_rotation_recovery_counts_by_strategy.items())
                    },
                    "learning_controls": {
                        strategy: {
                            "family_rotation": dict(sorted((self._family_rotation_counts_by_strategy.get(strategy, {}) or {}).items())),
                            "family_rotation_recovery": dict(sorted((self._family_rotation_recovery_counts_by_strategy.get(strategy, {}) or {}).items())),
                            "learning_evidence": dict(sorted((self._learning_evidence_counts_by_strategy.get(strategy, {}) or {}).items())),
                            "learning_asymmetry": dict(sorted((self._learning_asymmetry_counts_by_strategy.get(strategy, {}) or {}).items())),
                            "missed_opportunity_relaxations": dict(sorted((self._missed_opportunity_relaxations_by_strategy.get(strategy, {}) or {}).items())),
                            "reentry_cooldown_registrations": dict(sorted((self._reentry_cooldown_registrations_by_strategy.get(strategy, {}) or {}).items())),
                        }
                        for strategy in sorted(
                            set(self._signals_by_strategy)
                            | set(self._family_rotation_counts_by_strategy)
                            | set(self._family_rotation_recovery_counts_by_strategy)
                            | set(self._learning_evidence_counts_by_strategy)
                            | set(self._learning_asymmetry_counts_by_strategy)
                            | set(self._missed_opportunity_relaxations_by_strategy)
                            | set(self._reentry_cooldown_registrations_by_strategy)
                        )
                    },
                },
                "by_strategy_by_symbol": by_strategy_by_symbol,
                "top_rejection_reasons": self._top_items(rejection_totals, limit=10),
                "family_rotation_actions": dict(sorted(self._family_rotation_counts.items())),
                "family_rotation_recovery_actions": dict(sorted(self._family_rotation_recovery_counts.items())),
                "learning_asymmetry_actions": dict(sorted(self._learning_asymmetry_counts.items())),
                "missed_opportunity_relaxations": dict(sorted(self._missed_opportunity_relaxations.items())),
                "reentry_cooldown_registrations": dict(sorted(self._reentry_cooldown_registrations.items())),
            }
        }

    def _build_trade_frequency_summary(self, *, days: int, timeline: Sequence[dt.datetime]) -> Dict[str, Any]:
        observed_days = 0.0
        if timeline:
            observed_days = max((timeline[-1] - timeline[0]).total_seconds() / 86400.0, 0.0)
        window_days = max(float(days or 0), observed_days, 1.0)

        def per_day(count: int) -> float:
            return float(count) / max(window_days, 1e-9)

        strategies = sorted(
            set(self._proposals_by_strategy)
            | set(self._signals_by_strategy)
            | set(self._filled_by_strategy)
            | set(self._closed_by_strategy)
        )
        return {
            "window_days": window_days,
            "target_band": {
                "preferred_min": float(getattr(self.config, "target_trades_per_day_min", 2.0) or 2.0),
                "preferred_max": float(getattr(self.config, "target_trades_per_day_max", 3.0) or 3.0),
                "soft_floor": float(getattr(self.config, "target_trades_per_day_soft_floor", 1.5) or 1.5),
                "soft_ceiling": float(getattr(self.config, "target_trades_per_day_soft_ceiling", 3.5) or 3.5),
            },
            "global": {
                "proposals": int(self._proposal_count),
                "selected_signals": int(self._raw_signals),
                "fills": int(self._filled_orders),
                "trades": len(self.trades),
                "proposals_per_day": per_day(self._proposal_count),
                "selected_signals_per_day": per_day(self._raw_signals),
                "fills_per_day": per_day(self._filled_orders),
                "trades_per_day": per_day(len(self.trades)),
                "status": self._trade_frequency_status(per_day(len(self.trades))),
                "controller_actions": dict(sorted(self._frequency_adjustment_counts.items())),
                "expansion_gate": self._frequency_expansion_gate(),
            },
            "by_strategy": {
                strategy: {
                    "proposals": int(self._proposals_by_strategy.get(strategy, 0) or 0),
                    "selected_signals": int(self._signals_by_strategy.get(strategy, 0) or 0),
                    "fills": int(self._filled_by_strategy.get(strategy, 0) or 0),
                    "trades": int(self._closed_by_strategy.get(strategy, 0) or 0),
                    "proposals_per_day": per_day(int(self._proposals_by_strategy.get(strategy, 0) or 0)),
                    "selected_signals_per_day": per_day(int(self._signals_by_strategy.get(strategy, 0) or 0)),
                    "fills_per_day": per_day(int(self._filled_by_strategy.get(strategy, 0) or 0)),
                    "trades_per_day": per_day(int(self._closed_by_strategy.get(strategy, 0) or 0)),
                    "status": self._trade_frequency_status(per_day(int(self._closed_by_strategy.get(strategy, 0) or 0))),
                    "controller_actions": dict(sorted((self._frequency_adjustment_counts_by_strategy.get(strategy, {}) or {}).items())),
                }
                for strategy in strategies
            },
        }

    def _build_realized_performance_summary(self, symbol_rollups: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        strategies = sorted(set(self._closed_by_strategy) | set(self._wins_by_strategy) | set(self._losses_by_strategy))
        by_strategy: Dict[str, Dict[str, Any]] = {}
        for strategy in strategies:
            trades = [
                trade
                for trade in self.trades
                if isinstance(trade, dict) and str(trade.get("strategy", "unknown")) == strategy
            ]
            total_trades = len(trades)
            total_pl = sum(float(trade.get("pl", 0.0) or 0.0) for trade in trades)
            wins = int(self._wins_by_strategy.get(strategy, 0) or 0)
            losses = int(self._losses_by_strategy.get(strategy, 0) or 0)
            by_strategy[strategy] = {
                "trades": total_trades,
                "wins": wins,
                "losses": losses,
                "win_rate_pct": (wins / total_trades * 100.0) if total_trades else 0.0,
                "total_pl": total_pl,
                "expectancy": (total_pl / total_trades) if total_trades else 0.0,
            }
        return {
            "by_symbol": {
                symbol: {
                    "trades": int(dict(payload or {}).get("trades", 0) or 0),
                    "total_pl": float(dict(payload or {}).get("total_pl", 0.0) or 0.0),
                    "expectancy": float(dict(payload or {}).get("expectancy", 0.0) or 0.0),
                    "win_rate_pct": float(dict(payload or {}).get("win_rate_pct", 0.0) or 0.0),
                }
                for symbol, payload in sorted(symbol_rollups.items())
            },
            "by_strategy": by_strategy,
        }

    def _build_exit_quality_summary(self) -> Dict[str, Any]:
        by_exit_reason: Dict[str, Dict[str, Any]] = {}
        giveback_by_strategy: Dict[str, Dict[str, Any]] = {}
        for trade in self.trades:
            exit_reason = str(trade.get("exit_reason", "UNKNOWN") or "UNKNOWN")
            strategy = str(trade.get("strategy", "unknown") or "unknown")
            payload = by_exit_reason.setdefault(
                exit_reason,
                {"trades": 0.0, "wins": 0.0, "losses": 0.0, "total_pl": 0.0, "total_mfe_r": 0.0, "total_mae_r": 0.0, "total_giveback_r": 0.0},
            )
            payload["trades"] += 1.0
            payload["total_pl"] += float(trade.get("pl", 0.0) or 0.0)
            payload["total_mfe_r"] += float(trade.get("mfe_r", 0.0) or 0.0)
            payload["total_mae_r"] += float(trade.get("mae_r", 0.0) or 0.0)
            payload["total_giveback_r"] += float(trade.get("giveback_r", 0.0) or 0.0)
            if float(trade.get("pl", 0.0) or 0.0) >= 0.0:
                payload["wins"] += 1.0
            else:
                payload["losses"] += 1.0

            strategy_payload = giveback_by_strategy.setdefault(
                strategy,
                {"trades": 0.0, "total_giveback_r": 0.0, "mfe_above_1r_count": 0.0, "gave_back_below_0_25r_count": 0.0},
            )
            strategy_payload["trades"] += 1.0
            strategy_payload["total_giveback_r"] += float(trade.get("giveback_r", 0.0) or 0.0)
            mfe_r = float(trade.get("mfe_r", 0.0) or 0.0)
            realized_r = max(mfe_r - float(trade.get("giveback_r", 0.0) or 0.0), -999.0)
            if mfe_r >= 1.0:
                strategy_payload["mfe_above_1r_count"] += 1.0
                if realized_r < 0.25:
                    strategy_payload["gave_back_below_0_25r_count"] += 1.0

        for payload in by_exit_reason.values():
            trades = max(float(payload.get("trades", 0.0) or 0.0), 1.0)
            payload["expectancy"] = float(payload.get("total_pl", 0.0) or 0.0) / trades
            payload["win_rate_pct"] = (float(payload.get("wins", 0.0) or 0.0) / trades) * 100.0
            payload["avg_mfe_r"] = float(payload.get("total_mfe_r", 0.0) or 0.0) / trades
            payload["avg_mae_r"] = float(payload.get("total_mae_r", 0.0) or 0.0) / trades
            payload["avg_giveback_r"] = float(payload.get("total_giveback_r", 0.0) or 0.0) / trades
        for payload in giveback_by_strategy.values():
            trades = max(float(payload.get("trades", 0.0) or 0.0), 1.0)
            payload["avg_giveback_r"] = float(payload.get("total_giveback_r", 0.0) or 0.0) / trades
        return {
            "by_exit_reason": {
                key: dict(sorted(value.items()))
                for key, value in sorted(by_exit_reason.items())
            },
            "giveback_by_strategy": {
                key: dict(sorted(value.items()))
                for key, value in sorted(giveback_by_strategy.items())
            },
        }

    def _build_validation_harness_summary(self, symbol_rollups: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        raw_signals = int(self._raw_signals or 0)
        submitted_orders = int(self._submitted_orders or 0)
        filled_orders = int(self._filled_orders or 0)
        total_trades = len(self.trades)
        skip_total = max(int(self._skipped_signals or 0), 0)
        repeated_setup_blocks = int(self._repeated_setup_blocks or 0)
        fresh_setup_blocks = int(self._fresh_setup_blocks or 0)
        triple_barrier_total = sum(int(value or 0) for value in self._triple_barrier_label_counts.values())
        triple_barrier_tp = int(self._triple_barrier_label_counts.get("TP_FIRST", 0) or 0)
        triple_barrier_sl = int(self._triple_barrier_label_counts.get("SL_FIRST", 0) or 0)
        triple_barrier_time = int(self._triple_barrier_label_counts.get("TIME_EXIT", 0) or 0)
        pre_selection_rejections: Dict[str, int] = {}
        for reason_counts in self._generation_reasons_by_symbol.values():
            for reason, count in dict(reason_counts or {}).items():
                self._increment_counter(pre_selection_rejections, str(reason), int(count or 0))
        strategy_rejections: Dict[str, int] = {}
        for strategy_counts in self._strategy_rejection_reasons_by_symbol.values():
            for strategy, reason_counts in dict(strategy_counts or {}).items():
                for reason, count in dict(reason_counts or {}).items():
                    self._increment_counter(strategy_rejections, f"{strategy}:{reason}", int(count or 0))
        post_selection_rejections = dict(self._skip_reason_counts or {})
        candidate_flow_starved = bool(
            self._proposal_count <= 0
            or raw_signals <= 0
            or submitted_orders <= 0
            or filled_orders <= 0
        )

        negative_trades = [trade for trade in self.trades if float(trade.get("pl", 0.0) or 0.0) < 0.0]
        negative_total_pl = abs(sum(float(trade.get("pl", 0.0) or 0.0) for trade in negative_trades))
        stop_loss_negative_pl = abs(
            sum(
                float(trade.get("pl", 0.0) or 0.0)
                for trade in negative_trades
                if str(trade.get("exit_reason", "UNKNOWN") or "UNKNOWN") == "SL"
            )
        )

        return {
            "signal_to_submission_pct": (submitted_orders / raw_signals * 100.0) if raw_signals > 0 else 0.0,
            "submission_to_fill_pct": (filled_orders / submitted_orders * 100.0) if submitted_orders > 0 else 0.0,
            "fill_to_close_pct": (total_trades / filled_orders * 100.0) if filled_orders > 0 else 0.0,
            "expectancy_by_strategy": {
                strategy: float(dict(payload or {}).get("expectancy", 0.0) or 0.0)
                for strategy, payload in sorted(dict(self._build_realized_performance_summary(symbol_rollups).get("by_strategy", {}) or {}).items())
            },
            "expectancy_by_symbol": {
                symbol: float(dict(payload or {}).get("expectancy", 0.0) or 0.0)
                for symbol, payload in sorted(symbol_rollups.items())
            },
            "exit_expectancy_by_reason": {
                reason: float(dict(payload or {}).get("expectancy", 0.0) or 0.0)
                for reason, payload in sorted(dict(self._build_exit_quality_summary().get("by_exit_reason", {}) or {}).items())
            },
            "stop_loss_negative_pl_share_pct": (stop_loss_negative_pl / negative_total_pl * 100.0) if negative_total_pl > 0.0 else 0.0,
            "repeated_setup_blocks": repeated_setup_blocks,
            "fresh_setup_blocks": fresh_setup_blocks,
            "repeated_setup_density_pct": (repeated_setup_blocks / skip_total * 100.0) if skip_total > 0 else 0.0,
            "fresh_setup_block_density_pct": (fresh_setup_blocks / skip_total * 100.0) if skip_total > 0 else 0.0,
            "triple_barrier_labels": int(triple_barrier_total),
            "triple_barrier_tp_first_pct": (triple_barrier_tp / triple_barrier_total * 100.0) if triple_barrier_total > 0 else 0.0,
            "triple_barrier_sl_first_pct": (triple_barrier_sl / triple_barrier_total * 100.0) if triple_barrier_total > 0 else 0.0,
            "triple_barrier_time_exit_pct": (triple_barrier_time / triple_barrier_total * 100.0) if triple_barrier_total > 0 else 0.0,
            "pullback_meta_filter_blocks": int(self._pullback_meta_filter_blocks),
            "candidate_flow": {
                "starved": candidate_flow_starved,
                "proposals": int(self._proposal_count),
                "raw_signals": raw_signals,
                "submitted_orders": submitted_orders,
                "filled_orders": filled_orders,
                "closed_trades": total_trades,
                "generation_outcomes": dict(sorted(self._generation_outcomes.items())),
                "pre_selection_rejections": dict(sorted(pre_selection_rejections.items())),
                "strategy_rejections": dict(sorted(strategy_rejections.items())),
                "post_selection_rejections": dict(sorted(post_selection_rejections.items())),
                "top_blockers": self._top_items(
                    {
                        **{f"pre:{key}": value for key, value in pre_selection_rejections.items()},
                        **{f"strategy:{key}": value for key, value in strategy_rejections.items()},
                        **{f"post:{key}": value for key, value in post_selection_rejections.items()},
                    },
                    limit=8,
                ),
            },
        }

    def _build_campaign_acceptance_summary(self, *, metrics: Dict[str, Any], trade_frequency: Dict[str, Any], validation_harness: Dict[str, Any] | None = None) -> Dict[str, Any]:
        global_frequency = dict(trade_frequency.get("global", {}) or {})
        target_band = dict(trade_frequency.get("target_band", {}) or {})
        validation_harness = dict(validation_harness or {})
        trades_per_day = float(global_frequency.get("trades_per_day", global_frequency.get("closed_trades_per_day", 0.0)) or 0.0)
        preferred_min = float(target_band.get("preferred_min", getattr(self.config, "target_trades_per_day_min", 2.0)) or 0.0)
        preferred_max = float(target_band.get("preferred_max", getattr(self.config, "target_trades_per_day_max", 3.0)) or 0.0)
        win_rate_pct = float(metrics.get("win_rate_pct", 0.0) or 0.0)
        total_return_pct = float(metrics.get("total_return_pct", 0.0) or 0.0)
        profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
        max_drawdown_pct = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
        stop_loss_share = float(validation_harness.get("stop_loss_negative_pl_share_pct", 0.0) or 0.0)
        signal_to_submission = float(validation_harness.get("signal_to_submission_pct", 0.0) or 0.0)
        submission_to_fill = float(validation_harness.get("submission_to_fill_pct", 0.0) or 0.0)
        checks = {
            "trades_per_day_in_target_band": bool(preferred_min <= trades_per_day <= preferred_max),
            "win_rate_meets_floor": bool(win_rate_pct >= float(getattr(self.config, "promotion_min_win_rate_pct", 35.0) or 35.0)),
            "profit_factor_meets_floor": bool(profit_factor >= float(getattr(self.config, "promotion_min_profit_factor", 0.90) or 0.90)),
            "drawdown_within_limit": bool(abs(max_drawdown_pct) <= float(getattr(self.config, "promotion_max_drawdown_pct", 12.0) or 12.0)),
            "positive_return": bool(total_return_pct > 0.0),
            "stop_loss_damage_within_limit": bool(stop_loss_share <= float(getattr(self.config, "promotion_max_stop_loss_negative_pl_share_pct", 55.0) or 55.0)),
            "signal_to_submission_meets_floor": bool(signal_to_submission >= float(getattr(self.config, "promotion_min_signal_to_submission_pct", 18.0) or 18.0)),
            "submission_to_fill_meets_floor": bool(submission_to_fill >= float(getattr(self.config, "promotion_min_submission_to_fill_pct", 45.0) or 45.0)),
        }
        return {
            "objective_hierarchy": [
                "preserve_realism_and_no_lookahead",
                "reach_target_trade_band",
                "improve_win_rate",
                "improve_profit",
            ],
            "checks": checks,
            "values": {
                "trades_per_day": trades_per_day,
                "preferred_min": preferred_min,
                "preferred_max": preferred_max,
                "win_rate_pct": win_rate_pct,
                "total_return_pct": total_return_pct,
                "profit_factor": profit_factor,
                "max_drawdown_pct": max_drawdown_pct,
                "stop_loss_negative_pl_share_pct": stop_loss_share,
                "signal_to_submission_pct": signal_to_submission,
                "submission_to_fill_pct": submission_to_fill,
            },
            "passes_all": bool(all(checks.values())),
        }

    def _trade_frequency_status(self, trades_per_day: float) -> str:
        soft_floor = float(getattr(self.config, "target_trades_per_day_soft_floor", 1.5) or 1.5)
        preferred_min = float(getattr(self.config, "target_trades_per_day_min", 2.0) or 2.0)
        preferred_max = float(getattr(self.config, "target_trades_per_day_max", 3.0) or 3.0)
        soft_ceiling = float(getattr(self.config, "target_trades_per_day_soft_ceiling", 3.5) or 3.5)
        if trades_per_day < soft_floor:
            return "far_below_target"
        if trades_per_day < preferred_min:
            return "below_target"
        if trades_per_day <= preferred_max:
            return "in_target_band"
        if trades_per_day <= soft_ceiling:
            return "above_target"
        return "far_above_target"

    @staticmethod
    def _top_items(counter: Dict[str, int], *, limit: int) -> list[Dict[str, Any]]:
        items = sorted(((str(key), int(value or 0)) for key, value in dict(counter or {}).items()), key=lambda item: item[1], reverse=True)
        return [{"key": key, "count": count} for key, count in items[:limit] if count > 0]

    def _build_symbol_rollups(self) -> Dict[str, Dict[str, Any]]:
        raw_signals_by_symbol = dict(getattr(self, "_raw_signals_by_symbol", {}) or {})
        submitted_by_symbol = dict(getattr(self, "_submitted_by_symbol", {}) or {})
        filled_by_symbol = dict(getattr(self, "_filled_by_symbol", {}) or {})
        closed_by_symbol = dict(getattr(self, "_closed_by_symbol", {}) or {})
        skipped_by_symbol = dict(getattr(self, "_skipped_by_symbol", {}) or {})
        skip_reasons_by_symbol = dict(getattr(self, "_skip_reasons_by_symbol", {}) or {})
        expected_edge_sum_by_symbol = dict(getattr(self, "_expected_edge_sum_by_symbol", {}) or {})
        symbols = sorted(
            set(self._simulation_symbol_universe)
            | set(raw_signals_by_symbol)
            | set(submitted_by_symbol)
            | set(filled_by_symbol)
            | set(closed_by_symbol)
            | set(trade.get("symbol", "") for trade in self.trades if isinstance(trade, dict))
        )
        rollups: Dict[str, Dict[str, Any]] = {}
        for symbol in symbols:
            trades = [trade for trade in self.trades if isinstance(trade, dict) and trade.get("symbol") == symbol]
            wins = sum(1 for trade in trades if float(trade.get("pl", 0.0) or 0.0) >= 0.0)
            losses = sum(1 for trade in trades if float(trade.get("pl", 0.0) or 0.0) < 0.0)
            total_trades = len(trades)
            total_pl = sum(float(trade.get("pl", 0.0) or 0.0) for trade in trades)
            avg_edge = float(expected_edge_sum_by_symbol.get(symbol, 0.0) or 0.0) / max(int(raw_signals_by_symbol.get(symbol, 0) or 0), 1)
            expectancy = total_pl / total_trades if total_trades else 0.0
            rollups[symbol] = {
                "trades": total_trades,
                "raw_signals": int(raw_signals_by_symbol.get(symbol, 0) or 0),
                "fills": int(filled_by_symbol.get(symbol, 0) or 0),
                "submitted_orders": int(submitted_by_symbol.get(symbol, 0) or 0),
                "skipped_signals": int(skipped_by_symbol.get(symbol, 0) or 0),
                "wins": wins,
                "losses": losses,
                "win_rate_pct": (wins / total_trades * 100.0) if total_trades else 0.0,
                "total_pl": total_pl,
                "avg_edge_bps": avg_edge,
                "expectancy": expectancy,
                "skip_reasons": dict(sorted((skip_reasons_by_symbol.get(symbol, {}) or {}).items())),
            }
        return rollups

    def compute_metrics(self) -> Dict[str, float]:
        if not self.balance_curve:
            return {}
        if np is None:  # pragma: no cover
            returns = []
            for prev, curr in zip(self.balance_curve[:-1], self.balance_curve[1:]):
                if prev:
                    returns.append((curr / prev) - 1.0)
            std = float((sum((r - (sum(returns) / len(returns))) ** 2 for r in returns) / len(returns)) ** 0.5) if returns else 0.0
            mean = float(sum(returns) / len(returns)) if returns else 0.0
        else:
            returns = np.array(self.balance_curve, dtype=float)
            returns = np.diff(returns) / np.maximum(returns[:-1], 1e-9)
            mean = float(np.mean(returns)) if len(returns) else 0.0
            std = float(np.std(returns)) if len(returns) else 0.0
        total_return_pct = ((self.balance_curve[-1] / max(self.balance_curve[0], 1e-9)) - 1.0) * 100.0
        wins = [float(trade.get("pl", 0.0) or 0.0) for trade in self.trades if float(trade.get("pl", 0.0) or 0.0) > 0]
        losses = [float(trade.get("pl", 0.0) or 0.0) for trade in self.trades if float(trade.get("pl", 0.0) or 0.0) < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        peak = []
        running_peak = -float("inf")
        for value in self.balance_curve:
            running_peak = max(running_peak, value)
            peak.append(running_peak)
        drawdowns = [((value - top) / top) if top > 0 else 0.0 for value, top in zip(self.balance_curve, peak)]
        max_dd_pct = min(drawdowns) * 100.0 if drawdowns else 0.0
        sharpe = (mean / std * math.sqrt(252 * 24)) if std > 0 else 0.0
        expectancy = float(sum(float(trade.get("pl", 0.0) or 0.0) for trade in self.trades) / len(self.trades)) if self.trades else 0.0
        avg_holding_minutes = float(sum(float(trade.get("holding_minutes", 0.0) or 0.0) for trade in self.trades) / len(self.trades)) if self.trades else 0.0
        return {
            "total_return_pct": total_return_pct,
            "num_trades": len(self.trades),
            "win_rate_pct": (len(wins) / len(self.trades) * 100.0) if self.trades else 0.0,
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd_pct,
            "avg_win": float(sum(wins) / len(wins)) if wins else 0.0,
            "avg_loss": float(sum(losses) / len(losses)) if losses else 0.0,
            "expectancy": expectancy,
            "avg_holding_minutes": avg_holding_minutes,
        }

    def _record_event(self, event_type: str, trace_id: str, payload: Dict[str, Any]) -> None:
        event = BotEvent(event_type=event_type, trace_id=trace_id, payload=payload)
        self.state_store.append_event(event)
        self.events.append({"event_type": event_type, "trace_id": trace_id, "created_at": event.created_at, "payload": payload})
        self._event_counts[event_type] = int(self._event_counts.get(event_type, 0)) + 1

    def _learning_context_for_signal(self, _: str, signal: Dict[str, Any], regime: Any | None = None) -> Dict[str, Any]:
        return self.learning.learning_context_for_signal(signal, regime)

    def _frequency_context_for_signal(self, symbol: str, regime: Any | None = None) -> Dict[str, Any]:
        start = self._campaign_timeline_start
        now = self._now
        if start is None or now is None:
            window_days = 1.0
        else:
            window_days = max((now - start).total_seconds() / 86400.0, 1.0 / 96.0)

        trade_counts = dict(self._closed_by_strategy)
        trades_per_day = len(self.trades) / max(window_days, 1e-9)
        total_strategy_trades = sum(int(value or 0) for value in trade_counts.values())
        trade_share_by_strategy = {
            strategy: (int(count or 0) / max(total_strategy_trades, 1))
            for strategy, count in trade_counts.items()
        }
        dominant_strategy = None
        if trade_counts:
            dominant_strategy = max(trade_counts.items(), key=lambda item: int(item[1] or 0))[0]
        expansion_gate = self._frequency_expansion_gate()
        if bool(expansion_gate.get("allowed", False)):
            self._frequency_expansion_allowed_count += 1
        else:
            self._frequency_expansion_blocked_count += 1
            self._increment_counter(self._frequency_expansion_block_reasons, str(expansion_gate.get("reason", "unknown") or "unknown"))
        return {
            "window_days": window_days,
            "trades_per_day": trades_per_day,
            "status": self._trade_frequency_status(trades_per_day),
            "expansion_allowed": bool(expansion_gate.get("allowed", False)),
            "expansion_gate": expansion_gate,
            "dominant_strategy": dominant_strategy,
            "strategy_trade_share": trade_share_by_strategy,
            "symbol": symbol,
            "regime": getattr(regime, "regime", None) if regime is not None else None,
        }

    @staticmethod
    def _gross_pl(side: str, entry_price: float, exit_price: float, size: float) -> float:
        if side == "short":
            return (entry_price - exit_price) * size
        return (exit_price - entry_price) * size

    @staticmethod
    def _entry_deviation_bps(requested_price: float, fill_price: float) -> float:
        if requested_price <= 0:
            return 0.0
        return abs(fill_price - requested_price) / requested_price * 10000.0

    def _should_stop(self) -> bool:
        return bool(callable(self._stop_requested) and self._stop_requested())

    def _emit_progress(self, current_index: int, total_bars: int, now: dt.datetime, *, force: bool = False) -> None:
        if not callable(self._progress_callback):
            return
        interval = max(int(getattr(self.config, "simulation_snapshot_interval_bars", 250)), 1)
        if not force and current_index % interval != 0:
            return
        payload = {
            "timestamp": now.isoformat(),
            "current_bar": current_index,
            "total_bars": total_bars,
            "progress_fraction": (current_index / max(total_bars - 1, 1)) if total_bars > 0 else 0.0,
            "balance": float(self.state.balance),
            "equity": float(self.state.balance + getattr(self.state, "unrealized_pnl", 0.0)),
            "num_trades": len(self.trades),
            "raw_signals": self._raw_signals,
            "skipped_signals": self._skipped_signals,
            "submitted_orders": self._submitted_orders,
            "filled_orders": self._filled_orders,
            "expired_orders": self._expired_orders,
            "cancelled_orders": self._cancelled_orders,
            "partial_fills": self._partial_fills,
            "ambiguous_exit_bars": self._ambiguous_exit_bars,
            "open_orders": len(self.venue.open_orders) if self.venue is not None else 0,
            "resumed_from_checkpoint": self._resumed_from_checkpoint,
            "stopped_early": self._stopped_early,
            "stop_reason": self._stop_reason,
        }
        self._progress_callback(payload)
