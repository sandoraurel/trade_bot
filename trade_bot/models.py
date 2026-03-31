from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def normalize_symbol(symbol: str) -> str:
    normalized = (symbol or "").strip().upper().replace("-", "/")
    return normalized


def normalize_side(side: str) -> str:
    raw = (side or "").strip().lower()
    if raw in {"buy", "long"}:
        return "long"
    if raw in {"sell", "short"}:
        return "short"
    return raw


@dataclass
class Signal:
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy: str
    confidence: float
    timeframe: str
    expected_holding_minutes: Optional[int] = None
    expected_edge_bps: float = 0.0
    fast_move: bool = False
    is_futures: bool = False
    regime: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = normalize_symbol(self.symbol)
        self.side = normalize_side(self.side)


@dataclass
class OrderIntent:
    symbol: str
    side: str
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    order_type: str
    strategy: str
    time_in_force: str = "GTC"
    is_futures: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    client_order_id: Optional[str] = None
    trace_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.symbol = normalize_symbol(self.symbol)
        self.side = normalize_side(self.side)


@dataclass
class Fill:
    symbol: str
    side: str
    size: float
    price: float
    fee: float
    filled_at: dt.datetime
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = normalize_symbol(self.symbol)
        self.side = normalize_side(self.side)


@dataclass
class PositionSnapshot:
    symbol: str
    side: str
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    strategy: str = "unknown"
    opened_at: Optional[dt.datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = normalize_symbol(self.symbol)
        self.side = normalize_side(self.side)


@dataclass
class PortfolioSnapshot:
    balance: float
    equity: float
    gross_exposure: float
    net_exposure: float
    open_positions: List[PositionSnapshot]
    updated_at: dt.datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    risk_fraction: float
    max_new_exposure: float
    current_gross_exposure: float
    current_net_exposure: float
    regime: str = "unknown"
    controls: Dict[str, Any] = field(default_factory=dict)
    capped_size: Optional[float] = None


@dataclass
class ReconciliationStatus:
    ok: bool
    checked_at: dt.datetime
    positions_match: bool
    balance_match: bool
    drift_reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyHealth:
    strategy: str
    status: str
    confidence: float
    trailing_return_pct: float = 0.0
    win_rate_pct: float = 0.0
    samples: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    days: int
    total_return_pct: float
    num_trades: int
    win_rate_pct: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    expectancy: float
    total_fees: float
    trade_log: List[Dict[str, Any]] = field(default_factory=list)
    strategy_breakdown: Dict[str, Any] = field(default_factory=dict)
    assumptions: Dict[str, Any] = field(default_factory=dict)
    generated_at: dt.datetime = field(default_factory=utc_now)
