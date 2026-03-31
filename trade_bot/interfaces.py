from __future__ import annotations

from typing import Any, Dict, List, Protocol

from .events import BotEvent
from .models import OrderIntent, PortfolioSnapshot, ReconciliationStatus, RiskDecision, Signal, StrategyHealth


class MarketDataProvider(Protocol):
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> List[List[float]]:
        ...

    def get_order_book(self, symbol: str) -> Dict[str, float]:
        ...


class ExecutionBroker(Protocol):
    def submit_order(self, intent: OrderIntent) -> bool:
        ...

    def reconcile_orders(self) -> Dict[str, Any]:
        ...


class PortfolioRiskEngine(Protocol):
    def evaluate_signal(self, signal: Signal) -> RiskDecision:
        ...

    def build_portfolio_snapshot(self) -> PortfolioSnapshot:
        ...


class Strategy(Protocol):
    name: str

    def evaluate(self, symbol: str) -> Signal | None:
        ...


class RegimeClassifier(Protocol):
    def classify(self, symbol: str) -> Dict[str, Any]:
        ...


class BacktestSimulator(Protocol):
    def run(self, symbol: str, timeframe: str, days: int) -> Dict[str, Any]:
        ...


class StateStore(Protocol):
    def append_event(self, event: BotEvent) -> None:
        ...

    def persist_snapshot(self, snapshot: Dict[str, Any]) -> None:
        ...


class DecisionLogger(Protocol):
    def log(self, event: BotEvent) -> None:
        ...


class Reconciler(Protocol):
    def reconcile(self) -> ReconciliationStatus:
        ...


class StrategyHealthMonitor(Protocol):
    def snapshot(self) -> List[StrategyHealth]:
        ...
