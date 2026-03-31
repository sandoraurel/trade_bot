from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from typing import Any, Callable, Dict, List

from trade_bot.readiness import build_readiness_report
from trade_bot.runtime import build_strategy_health


@dataclass
class ToolContext:
    bot: Any


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, str]
    handler: Callable[[ToolContext, Dict[str, Any]], Dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def call(self, name: str, context: ToolContext, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].handler(context, arguments)


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="get_runtime_snapshot",
            description="Return the bot's live balance, positions, risk flags, and mode.",
            parameters={},
            handler=_get_runtime_snapshot,
        )
    )
    registry.register(
        ToolSpec(
            name="get_symbol_snapshot",
            description="Return latest OHLCV and order book snapshot for one symbol.",
            parameters={"symbol": "Trading pair such as BTC/USDT", "timeframe": "Optional timeframe like 15m"},
            handler=_get_symbol_snapshot,
        )
    )
    registry.register(
        ToolSpec(
            name="list_open_positions",
            description="Return all open positions in a normalized list.",
            parameters={},
            handler=_list_open_positions,
        )
    )
    registry.register(
        ToolSpec(
            name="get_risk_state",
            description="Return the current portfolio risk state and reconciliation status.",
            parameters={},
            handler=_get_risk_state,
        )
    )
    registry.register(
        ToolSpec(
            name="get_event_research",
            description="Return active event-risk state and recent event research records.",
            parameters={},
            handler=_get_event_research,
        )
    )
    registry.register(
        ToolSpec(
            name="get_readiness_report",
            description="Return operating-mode promotion readiness and gate results.",
            parameters={},
            handler=_get_readiness_report,
        )
    )
    return registry


def _normalize_positions(raw_positions: Dict[str, Any]) -> List[Dict[str, Any]]:
    positions: List[Dict[str, Any]] = []
    for symbol, value in raw_positions.items():
        values = value if isinstance(value, list) else [value]
        for pos in values:
            positions.append(
                {
                    "symbol": symbol,
                    "side": getattr(pos, "side", None),
                    "entry_price": getattr(pos, "entry_price", None),
                    "size": getattr(pos, "size", None),
                    "stop_loss": getattr(pos, "stop_loss", None),
                    "take_profit": getattr(pos, "take_profit", None),
                    "opened_at": getattr(pos, "opened_at", None).isoformat() if getattr(pos, "opened_at", None) else None,
                }
            )
    return positions


def _get_runtime_snapshot(context: ToolContext, _arguments: Dict[str, Any]) -> Dict[str, Any]:
    state = context.bot.state
    return {
        "timestamp": dt.datetime.now().isoformat(),
        "paper_mode": state.paper_mode,
        "balance": state.balance,
        "open_positions": _normalize_positions(state.open_positions),
        "today_trades_count": state.today_trades_count,
        "consecutive_losses": state.consecutive_losses,
        "reduced_risk_mode": state.reduced_risk_mode,
        "emergency_mode": state.emergency_mode,
        "cooldown_until": state.cooldown_until.isoformat() if state.cooldown_until else None,
    }


def _get_symbol_snapshot(context: ToolContext, arguments: Dict[str, Any]) -> Dict[str, Any]:
    default_symbol = context.bot.config.symbols[0] if context.bot.config.symbols else "BTC/USDT"
    symbol = arguments.get("symbol", default_symbol)
    timeframe = arguments.get("timeframe", "15m")
    candles = context.bot.exch.fetch_ohlcv(symbol, timeframe, limit=5)
    order_book = context.bot.exch.get_order_book(symbol)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "order_book": order_book,
    }


def _list_open_positions(context: ToolContext, _arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {"positions": _normalize_positions(context.bot.state.open_positions)}


def _get_risk_state(context: ToolContext, _arguments: Dict[str, Any]) -> Dict[str, Any]:
    status = getattr(context.bot, "last_reconciliation_status", None)
    return {
        "balance": context.bot.state.balance,
        "reduced_risk_mode": context.bot.state.reduced_risk_mode,
        "emergency_mode": context.bot.state.emergency_mode,
        "operating_mode": getattr(context.bot.config, "operating_mode", "paper"),
        "cooldown_until": context.bot.state.cooldown_until.isoformat() if context.bot.state.cooldown_until else None,
        "execution_report": getattr(context.bot.exec, "last_execution_report", {}),
        "strategy_health": [item.__dict__ for item in build_strategy_health(context.bot)],
        "reconciliation_status": {
            "ok": getattr(status, "ok", None),
            "positions_match": getattr(status, "positions_match", None),
            "balance_match": getattr(status, "balance_match", None),
            "drift_reasons": getattr(status, "drift_reasons", []),
        },
    }


def _get_event_research(context: ToolContext, _arguments: Dict[str, Any]) -> Dict[str, Any]:
    engine = getattr(context.bot, "news_engine", None)
    if engine is None:
        return {"event_risk": {}, "recent_research_records": []}
    return {
        "event_risk": engine.event_risk_snapshot(),
        "recent_research_records": engine.recent_research_records(limit=10),
    }


def _get_readiness_report(context: ToolContext, _arguments: Dict[str, Any]) -> Dict[str, Any]:
    report = build_readiness_report(context.bot)
    return {
        "operating_mode": report.operating_mode,
        "recommended_next_mode": report.recommended_next_mode,
        "ready": report.ready,
        "summary": report.summary,
        "gates": [gate.__dict__ for gate in report.gates],
        "metrics": report.metrics,
    }
