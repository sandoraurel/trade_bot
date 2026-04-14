from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PromotionGateResult:
    gate: str
    passed: bool
    observed: Any
    threshold: Any
    detail: str


@dataclass
class ReadinessReport:
    operating_mode: str
    recommended_next_mode: str | None
    ready: bool
    summary: str
    gates: List[PromotionGateResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def _next_mode(mode: str) -> str | None:
    order = ["paper", "shadow", "canary", "capital_limited_live", "full_live"]
    if mode not in order:
        return None
    idx = order.index(mode)
    if idx + 1 >= len(order):
        return None
    return order[idx + 1]


def _runtime_snapshot_summary(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    if not snapshot:
        return {}
    readiness = snapshot.get("readiness", {}) or {}
    metrics = readiness.get("metrics", {}) or {}
    system_health = metrics.get("system_health", {}) or {}
    risk = snapshot.get("risk", {}) or {}
    portfolio = snapshot.get("portfolio", {}) or {}
    return {
        "snapshot_key": snapshot.get("snapshot_key", "runtime"),
        "updated_at": snapshot.get("updated_at"),
        "risk_reason": risk.get("reason", "ok"),
        "risk_allowed": bool(risk.get("allowed", True)),
        "top_blocker": system_health.get("top_blocker", "unknown"),
        "gross_exposure": portfolio.get("gross_exposure", 0.0),
        "open_positions": len(portfolio.get("open_positions", []) or []),
    }


def build_readiness_report(bot: Any) -> ReadinessReport:
    snapshot = bot.state_store.load_snapshot("runtime") if getattr(bot, "state_store", None) else {}
    operating_mode = getattr(bot.config, "operating_mode", "paper")
    next_mode = _next_mode(operating_mode)

    metrics = bot.state_store.load_operational_metrics() if getattr(bot, "state_store", None) else {}
    runtime_hours = float(metrics.get("runtime_hours", 0.0))
    reconciliation_failures = int(metrics.get("reconciliation_failures", 0))
    risk_halts = int(metrics.get("risk_halts", 0))
    closed_trades = int(metrics.get("closed_trades", 0))
    realized_win_rate = float(metrics.get("win_rate_pct", 0.0))
    realized_profit_factor = float(metrics.get("profit_factor", 0.0))
    realized_max_drawdown = float(metrics.get("max_drawdown_pct", 0.0))
    system_health = bot._build_system_health_summary() if hasattr(bot, "_build_system_health_summary") else {}

    gates = [
        PromotionGateResult(
            gate="runtime_burn_in",
            passed=runtime_hours >= float(bot.config.promotion_min_runtime_hours),
            observed=runtime_hours,
            threshold=bot.config.promotion_min_runtime_hours,
            detail="Sustained runtime burn-in hours",
        ),
        PromotionGateResult(
            gate="reconciliation_drift",
            passed=reconciliation_failures <= int(bot.config.promotion_max_reconciliation_drift_events),
            observed=reconciliation_failures,
            threshold=bot.config.promotion_max_reconciliation_drift_events,
            detail="Reconciliation mismatches during burn-in",
        ),
        PromotionGateResult(
            gate="risk_halts",
            passed=risk_halts <= int(bot.config.promotion_max_risk_halts),
            observed=risk_halts,
            threshold=bot.config.promotion_max_risk_halts,
            detail="Risk halts across burn-in",
        ),
        PromotionGateResult(
            gate="closed_trades",
            passed=closed_trades >= int(bot.config.promotion_min_closed_trades),
            observed=closed_trades,
            threshold=bot.config.promotion_min_closed_trades,
            detail="Closed trades observed for statistical minimum",
        ),
        PromotionGateResult(
            gate="win_rate",
            passed=realized_win_rate >= float(bot.config.promotion_min_win_rate_pct),
            observed=realized_win_rate,
            threshold=bot.config.promotion_min_win_rate_pct,
            detail="Observed realized win rate",
        ),
        PromotionGateResult(
            gate="profit_factor",
            passed=realized_profit_factor >= float(bot.config.promotion_min_profit_factor),
            observed=realized_profit_factor,
            threshold=bot.config.promotion_min_profit_factor,
            detail="Observed realized profit factor",
        ),
        PromotionGateResult(
            gate="max_drawdown",
            passed=realized_max_drawdown <= float(bot.config.promotion_max_drawdown_pct),
            observed=realized_max_drawdown,
            threshold=bot.config.promotion_max_drawdown_pct,
            detail="Observed max drawdown percentage",
        ),
    ]

    ready = all(gate.passed for gate in gates)
    summary = (
        f"Ready to promote from {operating_mode} to {next_mode}."
        if ready and next_mode
        else f"Not ready to promote beyond {operating_mode}."
    )
    return ReadinessReport(
        operating_mode=operating_mode,
        recommended_next_mode=next_mode if ready else None,
        ready=ready,
        summary=summary,
        gates=gates,
        metrics={
            "runtime_snapshot": _runtime_snapshot_summary(snapshot),
            "runtime_hours": runtime_hours,
            "reconciliation_failures": reconciliation_failures,
            "risk_halts": risk_halts,
            "closed_trades": closed_trades,
            "win_rate_pct": realized_win_rate,
            "profit_factor": realized_profit_factor,
            "max_drawdown_pct": realized_max_drawdown,
            "system_health": system_health,
        },
    )
