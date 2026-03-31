from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    strategy: str = "unknown"
    is_futures: bool = False
    opened_at: dt.datetime = field(default_factory=dt.datetime.now)
    leverage: Optional[float] = None
    order_id: Optional[str] = None
    status: str = "open"
    fee_paid: float = 0.0
    unrealized_pnl: float = 0.0
    last_update: Optional[dt.datetime] = None
    initial_stop_loss: Optional[float] = None
    initial_take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BotState:
    balance: float = 0.0
    open_positions: Dict[str, Any] = field(default_factory=dict)
    today_trades_count: int = 0
    today_start_date: dt.date = field(default_factory=lambda: dt.datetime.now().date())
    consecutive_losses: int = 0
    reduced_risk_mode: bool = False
    emergency_mode: bool = False
    last_morning_dashboard_date: Optional[dt.date] = None
    last_evening_dashboard_date: Optional[dt.date] = None
    paper_mode: bool = True
    equity_start_of_day: float = 0.0
    realized_pl_today: float = 0.0
    wins_today: int = 0
    losses_today: int = 0
    peak_equity: float = 0.0
    cooldown_until: Optional[dt.datetime] = None
    last_heartbeat: Optional[dt.datetime] = None
    lifetime_profit: float = 0.0
    lifetime_trades: int = 0
    best_single_trade: float = 0.0
    worst_single_trade: float = 0.0
    unrealized_pnl: float = 0.0
    last_equity_update: Optional[dt.datetime] = None
    multi_position_mode: bool = False
    last_news_scan_at: Optional[dt.datetime] = None
    pending_news_commands: List[Dict[str, Any]] = field(default_factory=list)
    market_regime_alerts: Dict[str, str] = field(default_factory=dict)
    data_cooldown_until: Optional[dt.datetime] = None
    execution_cooldown_until: Optional[dt.datetime] = None
    health_summary: Dict[str, Any] = field(default_factory=dict)
