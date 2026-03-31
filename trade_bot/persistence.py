from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from dataclasses import is_dataclass
from typing import Any, Dict

from .constants import STATE_FILE
from .state import BotState, Position


def save_bot_state(state: BotState, path: str = STATE_FILE) -> None:
    open_positions_list = []
    for value in state.open_positions.values():
        if is_dataclass(value):
            positions = [value]
        elif isinstance(value, list):
            positions = value
        else:
            continue
        for pos in positions:
            open_positions_list.append(
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "size": pos.size,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "is_futures": pos.is_futures,
                    "opened_at": pos.opened_at.isoformat(),
                    "leverage": pos.leverage,
                    "order_id": pos.order_id,
                    "status": pos.status,
                    "fee_paid": pos.fee_paid,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "last_update": pos.last_update.isoformat() if pos.last_update else None,
                    "initial_stop_loss": pos.initial_stop_loss,
                    "initial_take_profit": pos.initial_take_profit,
                    "metadata": getattr(pos, "metadata", {}),
                }
            )

    data = {
        "balance": state.balance,
        "today_trades_count": state.today_trades_count,
        "today_start_date": state.today_start_date.isoformat(),
        "consecutive_losses": state.consecutive_losses,
        "reduced_risk_mode": state.reduced_risk_mode,
        "emergency_mode": state.emergency_mode,
        "equity_start_of_day": state.equity_start_of_day,
        "realized_pl_today": state.realized_pl_today,
        "wins_today": state.wins_today,
        "losses_today": state.losses_today,
        "peak_equity": state.peak_equity,
        "paper_mode": state.paper_mode,
        "cooldown_until": state.cooldown_until.isoformat() if state.cooldown_until else None,
        "last_heartbeat": state.last_heartbeat.isoformat() if state.last_heartbeat else None,
        "lifetime_profit": state.lifetime_profit,
        "lifetime_trades": state.lifetime_trades,
        "best_single_trade": state.best_single_trade,
        "worst_single_trade": state.worst_single_trade,
        "unrealized_pnl": state.unrealized_pnl,
        "last_equity_update": state.last_equity_update.isoformat() if state.last_equity_update else None,
        "multi_position_mode": getattr(state, "multi_position_mode", False),
        "last_news_scan_at": state.last_news_scan_at.isoformat() if state.last_news_scan_at else None,
        "pending_news_commands": state.pending_news_commands,
        "market_regime_alerts": state.market_regime_alerts,
        "data_cooldown_until": getattr(state, "data_cooldown_until", None).isoformat() if getattr(state, "data_cooldown_until", None) else None,
        "execution_cooldown_until": getattr(state, "execution_cooldown_until", None).isoformat() if getattr(state, "execution_cooldown_until", None) else None,
        "health_summary": getattr(state, "health_summary", {}),
        "open_positions": open_positions_list,
    }

    state_dir = os.path.dirname(os.path.abspath(path)) or "."
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=state_dir, delete=False) as tmp:
        json.dump(data, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name
    os.replace(temp_name, path)


def load_bot_state(state: BotState, path: str = STATE_FILE) -> bool:
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    state.balance = data.get("balance", state.balance)
    state.today_trades_count = data.get("today_trades_count", 0)
    tsd = data.get("today_start_date")
    if tsd:
        state.today_start_date = dt.date.fromisoformat(tsd)
    state.consecutive_losses = data.get("consecutive_losses", 0)
    state.reduced_risk_mode = data.get("reduced_risk_mode", False)
    state.emergency_mode = data.get("emergency_mode", False)
    state.equity_start_of_day = data.get("equity_start_of_day", state.balance)
    state.realized_pl_today = data.get("realized_pl_today", 0.0)
    state.wins_today = data.get("wins_today", 0)
    state.losses_today = data.get("losses_today", 0)
    state.peak_equity = data.get("peak_equity", state.balance)
    state.paper_mode = data.get("paper_mode", state.paper_mode)

    _load_datetime(state, "cooldown_until", data)
    _load_datetime(state, "last_heartbeat", data)
    _load_datetime(state, "last_equity_update", data)
    _load_datetime(state, "last_news_scan_at", data)
    _load_datetime(state, "data_cooldown_until", data)
    _load_datetime(state, "execution_cooldown_until", data)

    state.lifetime_profit = data.get("lifetime_profit", 0.0)
    state.lifetime_trades = data.get("lifetime_trades", 0)
    state.best_single_trade = data.get("best_single_trade", 0.0)
    state.worst_single_trade = data.get("worst_single_trade", 0.0)
    state.unrealized_pnl = data.get("unrealized_pnl", 0.0)
    state.multi_position_mode = data.get("multi_position_mode", False)
    state.pending_news_commands = data.get("pending_news_commands", [])
    state.market_regime_alerts = data.get("market_regime_alerts", {})
    state.health_summary = data.get("health_summary", {})

    state.open_positions = {}
    for payload in data.get("open_positions", []):
        try:
            opened_at = _parse_datetime(payload.get("opened_at")) or dt.datetime.now()
            last_update = _parse_datetime(payload.get("last_update"))
            pos = Position(
                symbol=payload["symbol"],
                side=payload["side"],
                entry_price=payload["entry_price"],
                size=payload["size"],
                stop_loss=payload["stop_loss"],
                take_profit=payload["take_profit"],
                is_futures=payload.get("is_futures", False),
                opened_at=opened_at,
                leverage=payload.get("leverage"),
                order_id=payload.get("order_id"),
                status=payload.get("status", "open"),
                fee_paid=payload.get("fee_paid", 0.0),
                unrealized_pnl=payload.get("unrealized_pnl", 0.0),
                last_update=last_update,
                initial_stop_loss=payload.get("initial_stop_loss"),
                initial_take_profit=payload.get("initial_take_profit"),
                metadata=payload.get("metadata", {}),
            )
            state.open_positions.setdefault(pos.symbol, []).append(pos)
        except Exception:
            continue
    return True


def _parse_datetime(raw: Any) -> dt.datetime | None:
    if not raw:
        return None
    try:
        parsed = dt.datetime.fromisoformat(raw)
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except Exception:
        return None


def _load_datetime(state: BotState, attr_name: str, data: Dict[str, Any]) -> None:
    setattr(state, attr_name, _parse_datetime(data.get(attr_name)))
