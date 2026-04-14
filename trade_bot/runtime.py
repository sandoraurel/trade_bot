from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import asdict
from typing import Any, Dict, List

from .events import (
    BotEvent,
    EVENT_RECONCILIATION,
    EVENT_RISK_HALT,
    EVENT_SIGNAL,
    EVENT_STATE_PERSISTED,
)
from .models import PortfolioSnapshot, PositionSnapshot, RiskDecision, StrategyHealth, utc_now
from .readiness import build_readiness_report


def new_trace_id() -> str:
    return uuid.uuid4().hex


def build_portfolio_snapshot(bot: Any) -> PortfolioSnapshot:
    positions: List[PositionSnapshot] = []
    gross_exposure = 0.0
    net_exposure = 0.0

    for symbol, raw in bot.state.open_positions.items():
        values = raw if isinstance(raw, list) else [raw]
        for pos in values:
            size = float(getattr(pos, "size", 0.0) or 0.0)
            entry_price = float(getattr(pos, "entry_price", 0.0) or 0.0)
            notional = size * entry_price
            gross_exposure += notional
            net_exposure += notional if getattr(pos, "side", "long") == "long" else -notional
            positions.append(
                PositionSnapshot(
                    symbol=symbol,
                    side=getattr(pos, "side", "long"),
                    size=size,
                    entry_price=entry_price,
                    stop_loss=float(getattr(pos, "stop_loss", 0.0) or 0.0),
                    take_profit=float(getattr(pos, "take_profit", 0.0) or 0.0),
                    unrealized_pnl=float(getattr(pos, "unrealized_pnl", 0.0) or 0.0),
                    strategy=getattr(pos, "strategy", "unknown"),
                    opened_at=getattr(pos, "opened_at", None),
                )
            )

    return PortfolioSnapshot(
        balance=float(bot.state.balance),
        equity=float(bot.state.balance + getattr(bot.state, "unrealized_pnl", 0.0)),
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        open_positions=positions,
        updated_at=utc_now(),
        metadata={
            "paper_mode": bot.state.paper_mode,
            "reduced_risk_mode": bot.state.reduced_risk_mode,
            "emergency_mode": bot.state.emergency_mode,
            "operating_mode": getattr(bot.config, "operating_mode", "paper"),
        },
    )


def build_risk_decision(bot: Any, signal: Dict[str, Any] | None = None) -> RiskDecision:
    snapshot = build_portfolio_snapshot(bot)
    allowed = True
    reason = "ok"
    controls: Dict[str, Any] = {}

    if bot.state.emergency_mode:
        allowed = False
        reason = "emergency_mode"
    elif bot.state.cooldown_until and bot.state.cooldown_until > dt.datetime.now():
        allowed = False
        reason = "cooldown"
    elif snapshot.gross_exposure >= bot.state.balance * max(float(getattr(bot.config, "default_leverage", 1) or 1), 1.0):
        allowed = False
        reason = "gross_exposure_cap"

    if signal:
        controls["signal_strategy"] = signal.get("strategy")
        controls["signal_quality"] = signal.get("signal_quality")
        controls["expected_edge_bps"] = signal.get("expected_edge_bps", 0.0)
        controls["regime"] = signal.get("regime", "unknown")
        controls["ensemble"] = signal.get("ensemble", {})

    return RiskDecision(
        allowed=allowed,
        reason=reason,
        risk_fraction=bot.risk.current_risk_fraction(),
        max_new_exposure=max(bot.state.balance * max(float(getattr(bot.config, "default_leverage", 1) or 1), 1.0) - snapshot.gross_exposure, 0.0),
        current_gross_exposure=snapshot.gross_exposure,
        current_net_exposure=snapshot.net_exposure,
        regime=controls.get("regime", "unknown"),
        controls=controls,
    )


def build_strategy_health(bot: Any) -> List[StrategyHealth]:
    stats: Dict[str, Dict[str, float]] = {}
    for trade in getattr(getattr(bot, "backtest_engine", None), "trades", []) or []:
        strategy = str(trade.get("strategy", "unknown"))
        bucket = stats.setdefault(strategy, {"samples": 0.0, "wins": 0.0, "pl": 0.0})
        bucket["samples"] += 1
        bucket["pl"] += float(trade.get("pl", 0.0))
        if float(trade.get("pl", 0.0)) > 0:
            bucket["wins"] += 1

    if not stats:
        return [
            StrategyHealth(strategy="trend_breakout", status="baseline", confidence=0.35, notes=["Awaiting tracked results"]),
            StrategyHealth(strategy="mean_reversion", status="baseline", confidence=0.15, notes=["Baseline only; not production trusted"]),
        ]

    result: List[StrategyHealth] = []
    for strategy, bucket in sorted(stats.items()):
        samples = int(bucket["samples"])
        win_rate = (bucket["wins"] / samples * 100.0) if samples else 0.0
        trailing_return_pct = bucket["pl"]
        status = "healthy" if trailing_return_pct > 0 and win_rate >= 45.0 else "degraded"
        result.append(
            StrategyHealth(
                strategy=strategy,
                status=status,
                confidence=0.7 if status == "healthy" else 0.25,
                trailing_return_pct=trailing_return_pct,
                win_rate_pct=win_rate,
                samples=samples,
            )
        )
    return result


def emit_event(bot: Any, event_type: str, trace_id: str, payload: Dict[str, Any]) -> None:
    event = BotEvent(event_type=event_type, trace_id=trace_id, payload=payload)
    if getattr(bot, "decision_logger", None) is not None:
        bot.decision_logger.log(event)
    if getattr(bot, "state_store", None) is not None:
        bot.state_store.append_event(event)


def persist_runtime_snapshot(bot: Any, trace_id: str) -> None:
    if getattr(bot, "state_store", None) is None:
        return
    snapshot = {
        "snapshot_key": "runtime",
        "updated_at": utc_now().isoformat(),
        "portfolio": asdict(build_portfolio_snapshot(bot)),
        "risk": asdict(build_risk_decision(bot)),
        "reconciliation": asdict(getattr(bot, "last_reconciliation_status", None)) if getattr(bot, "last_reconciliation_status", None) else None,
        "strategy_health": [asdict(item) for item in build_strategy_health(bot)],
        "execution": getattr(bot.exec, "last_execution_report", {}),
        "event_risk": bot.news_engine.event_risk_snapshot() if getattr(bot, "news_engine", None) is not None else {},
        "learning": bot.learning.summary_snapshot() if getattr(bot, "learning", None) is not None else {},
        "readiness": asdict(build_readiness_report(bot)) if getattr(bot, "state_store", None) is not None else {},
    }
    bot.state_store.persist_snapshot(snapshot)
    emit_event(bot, EVENT_STATE_PERSISTED, trace_id, {"snapshot_key": "runtime"})


def log_signal(bot: Any, trace_id: str, signal: Dict[str, Any]) -> None:
    emit_event(bot, EVENT_SIGNAL, trace_id, signal)


def log_risk_halt(bot: Any, trace_id: str, decision: RiskDecision) -> None:
    emit_event(bot, EVENT_RISK_HALT, trace_id, asdict(decision))


def log_reconciliation(bot: Any, trace_id: str) -> None:
    if getattr(bot, "last_reconciliation_status", None) is None:
        return
    emit_event(bot, EVENT_RECONCILIATION, trace_id, asdict(bot.last_reconciliation_status))
