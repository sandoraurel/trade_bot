from __future__ import annotations

import random
import time
import datetime as dt
from dataclasses import asdict
from typing import Any, Dict, Optional

from .config import BotConfig
from .models import Fill, OrderIntent, utc_now
from .state import BotState


class ExecutionEngine:
    def __init__(self, config: BotConfig, state: BotState, exch: Any):
        self.config = config
        self.state = state
        self.exch = exch
        self.last_fill: Optional[Fill] = None
        self.last_execution_report: Dict[str, Any] = {}

    def compute_spread(self, symbol: str) -> float:
        ob = self.exch.get_order_book(symbol)
        bid = ob["bid"]
        ask = ob["ask"]
        mid = (ask + bid) / 2.0
        if mid <= 0:
            return 1.0
        return (ask - bid) / mid

    def validate_trade_plan(
        self,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        size: float,
    ) -> bool:
        if size <= 0 or entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            return False
        normalized_side = (side or "").lower()
        if normalized_side in ("long", "buy"):
            return stop_loss < entry_price < take_profit
        if normalized_side in ("short", "sell"):
            return stop_loss > entry_price > take_profit
        return False

    def _get_mid_price(self, symbol: str) -> Optional[float]:
        try:
            ob = self.exch.get_order_book(symbol)
            bid = ob["bid"]
            ask = ob["ask"]
            mid = (ask + bid) / 2.0
            return mid if mid > 0 else None
        except Exception as exc:
            print(f"[WARN] _get_mid_price failed for {symbol}: {type(exc).__name__}: {exc}")
            return None

    def _check_slippage_for_market(self, symbol: str, entry_price: float) -> bool:
        mid = self._get_mid_price(symbol)
        if mid is None or mid <= 0:
            print(f"[WARN] Could not estimate slippage for {symbol} (no valid mid price). Proceeding with market order.")
            return True
        slippage = abs(mid - entry_price) / mid
        if slippage > self.config.max_slippage_fraction:
            print(
                f"[WARN] Estimated slippage too high on {symbol}: {slippage:.4%} "
                f"(max allowed {self.config.max_slippage_fraction:.4%}). Aborting market order."
            )
            return False
        return True

    def _adjust_order_for_market(self, symbol: str, size: float, price: Optional[float]) -> tuple[float, Optional[float]]:
        market = None
        try:
            markets = getattr(getattr(self.exch, "trade_client", None), "markets", None)
            if markets and symbol in markets:
                market = markets[symbol]
        except Exception as exc:
            print(f"[WARN] _adjust_order_for_market: could not access markets for {symbol}: {exc}")
        if not market:
            return size, price

        limits = market.get("limits", {}) or {}
        precision = market.get("precision", {}) or {}
        amount_min = (limits.get("amount") or {}).get("min")
        cost_min = (limits.get("cost") or {}).get("min")
        amount_prec = precision.get("amount")
        price_prec = precision.get("price")

        if amount_prec is not None and amount_prec >= 0:
            factor = 10 ** amount_prec
            size = int(size * factor) / factor
        if price is not None and price_prec is not None and price_prec >= 0:
            factor = 10 ** price_prec
            price = int(price * factor) / factor
        if amount_min is not None and size < amount_min:
            print(f"[WARN] Adjusted size {size} for {symbol} is below minimum amount {amount_min}. Skipping order.")
            return 0.0, price
        if cost_min is not None and price is not None and size * price < cost_min:
            print(f"[WARN] Notional {size * price:.8f} for {symbol} is below minimum cost {cost_min}. Skipping order.")
            return 0.0, price
        return size, price

    def can_trade_symbol_now(self, symbol: str) -> bool:
        spread = self.compute_spread(symbol)
        if spread > self.config.max_spread_fraction:
            print(f"[INFO] Spread too high on {symbol}: {spread:.4%} (max allowed {self.config.max_spread_fraction:.4%}). Skipping.")
            return False
        return True

    def place_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        is_futures: bool,
        fast_move: bool,
        size: float,
    ) -> bool:
        normalized_side = (side or "").lower()
        if getattr(self.config, "trading_mode", "spot") == "spot" and normalized_side in ("short", "sell"):
            print(f"[WARN] Spot mode does not support opening short exposure on {symbol}.")
            return False
        if not self.validate_trade_plan(side, entry_price, stop_loss, take_profit, size):
            print(
                f"[WARN] Invalid trade plan for {symbol}: side={side}, entry={entry_price}, "
                f"SL={stop_loss}, TP={take_profit}, size={size}"
            )
            return False
        if not self.can_trade_symbol_now(symbol):
            return False
        order_type = "market" if fast_move else "limit"
        price: Optional[float] = None if fast_move else entry_price
        if fast_move and not self._check_slippage_for_market(symbol, entry_price):
            return False
        if size <= 0:
            print("[WARN] Size is zero or negative, cannot trade.")
            return False
        adj_size, adj_price = self._adjust_order_for_market(symbol, size, price)
        if adj_size <= 0:
            print("[WARN] Adjusted size is invalid, aborting trade.")
            return False
        intent = OrderIntent(
            symbol=symbol,
            side=side,
            size=adj_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            order_type=order_type,
            strategy="runtime_signal",
            is_futures=is_futures,
            metadata={"fast_move": fast_move},
        )
        return self.submit_order(intent, requested_price=adj_price)

    def submit_order(self, intent: OrderIntent, requested_price: Optional[float] = None) -> bool:
        retries = max(int(getattr(self.config, "broker_max_retries", 3)), 1)
        delay = max(float(getattr(self.config, "broker_retry_delay_seconds", 0.5)), 0.0)
        self.last_execution_report = {"attempts": 0, "intent": asdict(intent), "requested_price": requested_price}
        for attempt in range(1, retries + 1):
            self.last_execution_report["attempts"] = attempt
            simulated = self._simulate_fill(intent, requested_price=requested_price)
            self.last_fill = simulated
            self.last_execution_report["fill"] = asdict(simulated)
            success = self.exch.place_order(
                symbol=intent.symbol,
                side=intent.side,
                size=simulated.size,
                order_type=intent.order_type,
                price=requested_price,
                stop_loss=intent.stop_loss,
                take_profit=intent.take_profit,
                is_futures=intent.is_futures,
            )
            if success:
                self.last_execution_report["status"] = "success"
                return True
            self.last_execution_report["status"] = "retrying"
            if attempt < retries and delay > 0:
                time.sleep(delay)
        self.last_execution_report["status"] = "failed"
        return False

    def _simulate_fill(self, intent: OrderIntent, requested_price: Optional[float] = None) -> Fill:
        price = requested_price if requested_price is not None else intent.entry_price
        latency_ms = max(int(getattr(self.config, "simulated_order_latency_ms", 150)), 0)
        fill_fraction = 1.0
        if intent.order_type == "market":
            fill_fraction = 1.0
        else:
            min_fraction = float(getattr(self.config, "simulated_partial_fill_min_fraction", 0.65))
            fill_fraction = random.uniform(min_fraction, 1.0)
        size = intent.size * fill_fraction
        fee_rate = float(getattr(self.config, "backtest_fee_bps", 10.0)) / 10000.0
        fee = abs(price * size * fee_rate)
        simulated_at = utc_now() + dt.timedelta(milliseconds=latency_ms)
        return Fill(
            symbol=intent.symbol,
            side=intent.side,
            size=size,
            price=price,
            fee=fee,
            filled_at=simulated_at,
            client_order_id=intent.client_order_id,
            metadata={"latency_ms": latency_ms, "fill_fraction": fill_fraction},
        )
