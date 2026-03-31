from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from .models import Signal


@dataclass
class StrategyProposal:
    signal: Signal
    expected_edge_bps: float
    rationale: str


class StrategyBase:
    name = "base"

    def __init__(self, config: Any, exch: Any, helpers: Any):
        self.config = config
        self.exch = exch
        self.helpers = helpers

    def evaluate(self, symbol: str, regime: Any) -> Optional[StrategyProposal]:
        raise NotImplementedError


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) == 0:
        return values
    alpha = 2.0 / (period + 1.0)
    result = np.empty(len(values), dtype=float)
    result[0] = float(values[0])
    for idx in range(1, len(values)):
        result[idx] = (alpha * float(values[idx])) + ((1.0 - alpha) * result[idx - 1])
    return result


def _efficiency_ratio(values: np.ndarray, lookback: int = 10) -> float:
    if len(values) <= lookback:
        return 0.0
    window = values[-(lookback + 1):]
    path = float(np.abs(np.diff(window)).sum())
    if path <= 0:
        return 0.0
    direct = abs(float(window[-1]) - float(window[0]))
    return direct / path


def _candle_body_fraction(candle: list[float]) -> float:
    high = float(candle[2])
    low = float(candle[3])
    open_price = float(candle[1])
    close_price = float(candle[4])
    rng = max(high - low, 1e-9)
    return abs(close_price - open_price) / rng


def _close_location(candle: list[float]) -> float:
    high = float(candle[2])
    low = float(candle[3])
    close_price = float(candle[4])
    rng = max(high - low, 1e-9)
    return (close_price - low) / rng


class TrendBreakoutStrategy(StrategyBase):
    name = "trend_breakout"

    def evaluate(self, symbol: str, regime: Any) -> Optional[StrategyProposal]:
        if regime.regime not in {"trending", "high_volatility"}:
            return None
        candles_15m = self.exch.fetch_ohlcv(symbol, "15m", limit=30)
        lookback = max(int(getattr(self.config, "breakout_swing_lookback", 5)), 3)
        if not candles_15m or len(candles_15m) < max(lookback, 10):
            return None
        prior_candles = candles_15m[:-1]
        if len(prior_candles) < lookback:
            return None
        swing_high, swing_low = self.helpers.get_recent_swing_high_low(prior_candles, lookback=lookback)
        if swing_high is None or swing_low is None:
            return None
        last_close = float(candles_15m[-1][4])
        last_open = float(candles_15m[-1][1])
        atr_15m = self.helpers.compute_atr(symbol, "15m", period=14)
        if atr_15m is None or atr_15m <= 0:
            return None

        direction = regime.metadata.get("trend_direction", "flat")
        volume_impulse = float(regime.metadata.get("volume_impulse", 1.0))
        closes = np.array([float(c[4]) for c in candles_15m], dtype=float)
        ema_fast = _ema(closes, 8)
        ema_slow = _ema(closes, 21)
        trend_efficiency = _efficiency_ratio(closes, lookback=min(12, len(closes) - 1))
        body_fraction = _candle_body_fraction(candles_15m[-1])
        close_location = _close_location(candles_15m[-1])
        micro_trend_up = len(ema_fast) > 1 and ema_fast[-1] > ema_slow[-1] and ema_fast[-1] >= ema_fast[-2]
        micro_trend_down = len(ema_fast) > 1 and ema_fast[-1] < ema_slow[-1] and ema_fast[-1] <= ema_fast[-2]
        stretch = float(regime.metadata.get("stretch_from_mean", 0.0))
        breakout_buffer = atr_15m * 0.12
        if direction == "bearish":
            if not self.helpers.is_4h_bearish(symbol):
                return None
            if not self.helpers.is_1h_downtrend(symbol):
                return None
            if not micro_trend_down:
                return None
            if last_close - breakout_buffer > swing_low:
                return None
            if body_fraction < 0.45 or close_location > 0.35:
                return None
            if trend_efficiency < 0.28 or volume_impulse < 0.95:
                return None
            if stretch < -0.05:
                return None
            entry_price = min(last_close, float(swing_low))
            stop_loss = max(float(swing_high) * 1.001, entry_price + atr_15m * 0.9)
            if stop_loss <= entry_price:
                return None
            sl_distance = stop_loss - entry_price
            rr_ratio = max(float(getattr(self.config, "breakout_rr_ratio", 2.2)), 1.5)
            take_profit = entry_price - (sl_distance * rr_ratio)
            if take_profit >= entry_price or take_profit <= 0:
                return None
            expected_edge_bps = ((entry_price - take_profit) / entry_price) * 10000.0 * 0.42
            confidence = min(0.93, max(0.58, 0.64 + regime.confidence * 0.18 + min(max(volume_impulse - 1.0, 0.0), 0.25)))
            signal = Signal(
                symbol=symbol,
                side="short",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=self.name,
                confidence=confidence,
                timeframe="15m",
                expected_holding_minutes=8 * 60,
                expected_edge_bps=expected_edge_bps,
                fast_move=last_close < float(swing_low),
                is_futures=False,
                regime=regime.regime,
                metadata={
                    "regime_confidence": regime.confidence,
                    "trend_direction": direction,
                    "volume_impulse": volume_impulse,
                    "trend_efficiency": trend_efficiency,
                    "body_fraction": body_fraction,
                },
            )
            return StrategyProposal(signal=signal, expected_edge_bps=expected_edge_bps, rationale="trend continuation breakdown")

        if not self.helpers.is_4h_bullish(symbol):
            return None
        if not self.helpers.is_1h_uptrend(symbol):
            return None
        if not micro_trend_up:
            return None
        if last_close + breakout_buffer < swing_high:
            return None
        if body_fraction < 0.45 or close_location < 0.65:
            return None
        if trend_efficiency < 0.28 or volume_impulse < 0.95:
            return None
        if stretch > 0.05:
            return None
        if last_close <= last_open:
            return None
        entry_price = max(last_close, float(swing_high))
        stop_loss = min(float(swing_low) * 0.999, entry_price - atr_15m * 0.9)
        if stop_loss <= 0 or stop_loss >= entry_price:
            return None
        sl_distance = entry_price - stop_loss
        if sl_distance < atr_15m * 0.35 or sl_distance > atr_15m * 2.2:
            return None
        rr_ratio = max(float(getattr(self.config, "breakout_rr_ratio", 2.2)), 1.5)
        take_profit = entry_price + (sl_distance * rr_ratio)
        if take_profit <= entry_price:
            return None
        expected_edge_bps = ((take_profit - entry_price) / entry_price) * 10000.0 * 0.42
        signal = Signal(
            symbol=symbol,
            side="long",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=self.name,
            confidence=min(0.93, max(0.58, 0.64 + regime.confidence * 0.18 + min(max(volume_impulse - 1.0, 0.0), 0.25))),
            timeframe="15m",
            expected_holding_minutes=8 * 60,
            expected_edge_bps=expected_edge_bps,
            fast_move=last_close > float(swing_high),
            is_futures=False,
            regime=regime.regime,
            metadata={
                "regime_confidence": regime.confidence,
                "trend_direction": direction,
                "volume_impulse": volume_impulse,
                "trend_efficiency": trend_efficiency,
                "body_fraction": body_fraction,
            },
        )
        return StrategyProposal(signal=signal, expected_edge_bps=expected_edge_bps, rationale="trend continuation breakout")


class TrendPullbackStrategy(StrategyBase):
    name = "trend_pullback"

    def evaluate(self, symbol: str, regime: Any) -> Optional[StrategyProposal]:
        if regime.regime not in {"trending", "high_volatility"}:
            return None

        candles_15m = self.exch.fetch_ohlcv(symbol, "15m", limit=80)
        if len(candles_15m) < 40:
            return None

        closes = np.array([float(c[4]) for c in candles_15m], dtype=float)
        highs = np.array([float(c[2]) for c in candles_15m], dtype=float)
        lows = np.array([float(c[3]) for c in candles_15m], dtype=float)
        atr_15m = self.helpers.compute_atr(symbol, "15m", period=14)
        if atr_15m is None or atr_15m <= 0:
            return None

        fast_period = max(int(getattr(self.config, "pullback_ema_fast_period", 8)), 3)
        slow_period = max(int(getattr(self.config, "pullback_ema_slow_period", 21)), fast_period + 1)
        ema_fast = _ema(closes, fast_period)
        ema_slow = _ema(closes, slow_period)
        trend_efficiency = _efficiency_ratio(closes, lookback=min(14, len(closes) - 1))
        volume_impulse = float(regime.metadata.get("volume_impulse", 1.0))
        stretch = float(regime.metadata.get("stretch_from_mean", 0.0))
        direction = regime.metadata.get("trend_direction", "flat")
        pullback_fraction = float(getattr(self.config, "pullback_entry_atr_fraction", 0.35))
        pullback_threshold = atr_15m * pullback_fraction

        last_candle = candles_15m[-1]
        prev_candle = candles_15m[-2]
        last_close = float(last_candle[4])
        prev_close = float(prev_candle[4])
        body_fraction = _candle_body_fraction(last_candle)
        close_location = _close_location(last_candle)
        recent_high = float(np.max(highs[-10:]))
        recent_low = float(np.min(lows[-10:]))

        if direction == "bullish":
            if not self.helpers.is_4h_bullish(symbol) or not self.helpers.is_1h_uptrend(symbol):
                return None
            if ema_fast[-1] <= ema_slow[-1] or ema_fast[-2] <= ema_slow[-2]:
                return None
            if trend_efficiency < 0.05 or volume_impulse < 0.9:
                return None
            if stretch > 0.03:
                return None
            pullback_depth = max(ema_fast[-1] - last_close, 0.0)
            if pullback_depth < pullback_threshold * 0.15 or pullback_depth > atr_15m * 1.75:
                return None
            if last_close < ema_slow[-1]:
                return None
            if last_close <= prev_close:
                return None
            if body_fraction < 0.35 or close_location < 0.35:
                return None
            entry_price = last_close
            stop_loss = min(last_close - atr_15m * float(getattr(self.config, "pullback_stop_atr_mult", 1.15)), recent_low * 0.999)
            if stop_loss <= 0 or stop_loss >= entry_price:
                return None
            risk = entry_price - stop_loss
            if risk < atr_15m * 0.30 or risk > atr_15m * 1.9:
                return None
            take_profit = max(recent_high, entry_price + risk * float(getattr(self.config, "pullback_rr_ratio", 2.0)))
            if take_profit <= entry_price:
                return None
            expected_edge_bps = ((take_profit - entry_price) / entry_price) * 10000.0 * 0.38
            confidence = min(0.88, max(0.56, 0.58 + regime.confidence * 0.16 + min(max(volume_impulse - 1.0, 0.0), 0.12)))
            signal = Signal(
                symbol=symbol,
                side="long",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=self.name,
                confidence=confidence,
                timeframe="15m",
                expected_holding_minutes=6 * 60,
                expected_edge_bps=expected_edge_bps,
                fast_move=False,
                is_futures=False,
                regime=regime.regime,
                metadata={
                    "trend_direction": direction,
                    "pullback_depth": pullback_depth,
                    "trend_efficiency": trend_efficiency,
                    "volume_impulse": volume_impulse,
                },
            )
            return StrategyProposal(signal=signal, expected_edge_bps=expected_edge_bps, rationale="bull trend pullback continuation")

        if direction == "bearish":
            if not self.helpers.is_4h_bearish(symbol) or not self.helpers.is_1h_downtrend(symbol):
                return None
            if ema_fast[-1] >= ema_slow[-1] or ema_fast[-2] >= ema_slow[-2]:
                return None
            if trend_efficiency < 0.05 or volume_impulse < 0.9:
                return None
            if stretch < -0.03:
                return None
            pullback_depth = max(last_close - ema_fast[-1], 0.0)
            if pullback_depth < pullback_threshold * 0.15 or pullback_depth > atr_15m * 1.75:
                return None
            if last_close > ema_slow[-1]:
                return None
            if last_close >= prev_close:
                return None
            if body_fraction < 0.35 or close_location > 0.65:
                return None
            entry_price = last_close
            stop_loss = max(last_close + atr_15m * float(getattr(self.config, "pullback_stop_atr_mult", 1.15)), recent_high * 1.001)
            if stop_loss <= entry_price:
                return None
            risk = stop_loss - entry_price
            if risk < atr_15m * 0.30 or risk > atr_15m * 1.9:
                return None
            take_profit = min(recent_low, entry_price - risk * float(getattr(self.config, "pullback_rr_ratio", 2.0)))
            if take_profit >= entry_price or take_profit <= 0:
                return None
            expected_edge_bps = ((entry_price - take_profit) / entry_price) * 10000.0 * 0.38
            confidence = min(0.88, max(0.56, 0.58 + regime.confidence * 0.16 + min(max(volume_impulse - 1.0, 0.0), 0.12)))
            signal = Signal(
                symbol=symbol,
                side="short",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=self.name,
                confidence=confidence,
                timeframe="15m",
                expected_holding_minutes=6 * 60,
                expected_edge_bps=expected_edge_bps,
                fast_move=False,
                is_futures=False,
                regime=regime.regime,
                metadata={
                    "trend_direction": direction,
                    "pullback_depth": pullback_depth,
                    "trend_efficiency": trend_efficiency,
                    "volume_impulse": volume_impulse,
                },
            )
            return StrategyProposal(signal=signal, expected_edge_bps=expected_edge_bps, rationale="bear trend pullback continuation")

        return None


class MeanReversionStrategy(StrategyBase):
    name = "mean_reversion"

    def evaluate(self, symbol: str, regime: Any) -> Optional[StrategyProposal]:
        if regime.regime not in {"mean_reverting", "choppy"}:
            return None
        if regime.event_risk or regime.unstable or regime.liquidity_score < 0.85:
            return None
        candles = self.exch.fetch_ohlcv(symbol, "15m", limit=100)
        if len(candles) < 50:
            return None
        closes = np.array([float(c[4]) for c in candles], dtype=float)
        rsi = self.helpers.rsi(closes, 14)
        bb_upper, bb_middle, bb_lower = self.helpers.bollinger_bands(closes, 20, 2)
        atr_15m = self.helpers.compute_atr(symbol, "15m", period=14)
        if atr_15m is None or atr_15m <= 0:
            return None
        last_close = float(closes[-1])
        prev_close = float(closes[-2])
        last_rsi = float(rsi[-1])
        prev_rsi = float(rsi[-2])
        bb_mid = float(bb_middle[-1])
        bb_upper_last = float(bb_upper[-1])
        bb_lower_last = float(bb_lower[-1])
        stretch = float(regime.metadata.get("stretch_from_mean", 0.0))
        trend_direction = regime.metadata.get("trend_direction", "flat")
        ema_fast = _ema(closes, 8)
        ema_slow = _ema(closes, 21)
        mean_reversion_quality = _efficiency_ratio(closes, lookback=min(8, len(closes) - 1))
        recent_range = max(float(np.max(closes[-6:]) - np.min(closes[-6:])), 1e-9)
        rebound_strength = (last_close - float(np.min(closes[-6:]))) / recent_range
        fade_strength = (float(np.max(closes[-6:])) - last_close) / recent_range

        if last_rsi < 24 and prev_rsi <= last_rsi and last_close <= bb_lower_last * 0.999:
            return None

        if (
            last_rsi < 28
            and prev_rsi < last_rsi
            and last_close <= bb_lower_last * 1.0005
            and prev_close <= bb_mid
            and stretch < -0.004
            and trend_direction != "bearish"
            and ema_fast[-1] >= ema_fast[-2]
            and rebound_strength >= 0.22
            and mean_reversion_quality <= 0.75
        ):
            sl = min(bb_lower_last * 0.996, last_close - atr_15m * 0.9)
            tp = min(float(closes[-10:].mean()), bb_mid + atr_15m * 0.35)
            if sl <= 0 or sl >= last_close or tp <= last_close:
                return None
            expected_edge_bps = ((tp - last_close) / last_close) * 10000.0 * 0.34
            signal = Signal(
                symbol=symbol,
                side="long",
                entry_price=last_close,
                stop_loss=sl,
                take_profit=tp,
                strategy=self.name,
                confidence=min(0.80, max(0.50, 0.52 + regime.confidence * 0.13 + min((last_rsi < 24) * 0.05, 0.05))),
                timeframe="15m",
                expected_holding_minutes=3 * 60,
                expected_edge_bps=expected_edge_bps,
                fast_move=False,
                is_futures=False,
                regime=regime.regime,
                metadata={"rsi": last_rsi, "stretch_from_mean": stretch, "rebound_strength": rebound_strength},
            )
            return StrategyProposal(signal=signal, expected_edge_bps=expected_edge_bps, rationale="oversold mean reversion rebound")

        if (
            last_rsi > 72
            and prev_rsi > last_rsi
            and last_close >= bb_upper_last * 0.9995
            and prev_close >= bb_mid
            and stretch > 0.004
            and trend_direction != "bullish"
            and ema_fast[-1] <= ema_fast[-2]
            and fade_strength >= 0.22
            and mean_reversion_quality <= 0.75
        ):
            sl = max(bb_upper_last * 1.004, last_close + atr_15m * 0.9)
            tp = max(float(closes[-10:].mean()), bb_mid - atr_15m * 0.35)
            if sl <= last_close or tp >= last_close or tp <= 0:
                return None
            expected_edge_bps = ((last_close - tp) / last_close) * 10000.0 * 0.34
            signal = Signal(
                symbol=symbol,
                side="short",
                entry_price=last_close,
                stop_loss=sl,
                take_profit=tp,
                strategy=self.name,
                confidence=min(0.80, max(0.50, 0.52 + regime.confidence * 0.13 + min((last_rsi > 76) * 0.05, 0.05))),
                timeframe="15m",
                expected_holding_minutes=3 * 60,
                expected_edge_bps=expected_edge_bps,
                fast_move=False,
                is_futures=False,
                regime=regime.regime,
                metadata={"rsi": last_rsi, "stretch_from_mean": stretch, "fade_strength": fade_strength},
            )
            return StrategyProposal(signal=signal, expected_edge_bps=expected_edge_bps, rationale="overbought mean reversion fade")
        return None


def build_strategies(config: Any, exch: Any, helpers: Any) -> List[StrategyBase]:
    strategies: List[StrategyBase] = []
    if getattr(config, "enable_trend_breakout", True):
        strategies.append(TrendBreakoutStrategy(config, exch, helpers))
    if getattr(config, "enable_trend_pullback", True):
        strategies.append(TrendPullbackStrategy(config, exch, helpers))
    if getattr(config, "enable_mean_reversion", True):
        strategies.append(MeanReversionStrategy(config, exch, helpers))
    return strategies
