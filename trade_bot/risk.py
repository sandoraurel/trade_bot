from __future__ import annotations

import datetime as dt
from typing import Any, Dict

from .config import BotConfig
from .models import RiskDecision
from .state import BotState


class RiskManager:
    def __init__(self, config: BotConfig, state: BotState):
        self.config = config
        self.state = state
        self.now_provider = None

    def _now(self) -> dt.datetime:
        if callable(self.now_provider):
            return self.now_provider()
        return dt.datetime.now()

    def _count_open_positions(self) -> int:
        if not self.state.open_positions:
            return 0
        total = 0
        for value in self.state.open_positions.values():
            total += len(value) if isinstance(value, list) else 1
        return total

    def _is_in_cooldown(self) -> bool:
        if self.state.cooldown_until is None:
            return False
        return self._now() < self.state.cooldown_until

    def current_risk_fraction(self) -> float:
        base = self.config.risk_per_trade_max
        if self.state.reduced_risk_mode:
            base = base * self.config.reduced_risk_factor
        operating_mode = getattr(self.config, "operating_mode", "paper")
        if operating_mode == "canary":
            base *= 0.5
        elif operating_mode == "capital_limited_live":
            base *= 0.75
        return max(base, 0.0)

    def can_take_more_trades_today(self) -> bool:
        return self.state.today_trades_count < self.config.max_trades_per_day_max

    def entry_capacity_status(self, symbol: str, pending_positions: int = 0) -> str:
        if self._is_in_cooldown():
            return "cooldown"
        if (self._count_open_positions() + max(int(pending_positions), 0)) >= self.config.max_open_positions:
            return "max_open_positions"
        if not self.can_take_more_trades_today():
            return "daily_trade_limit"
        if not self._check_correlation_limit(symbol):
            print(f"[RISK] Correlation limit hit for {symbol}")
            return "correlation_limit"
        return "ok"

    def can_open_new_position(self, symbol: str, pending_positions: int = 0) -> bool:
        return self.entry_capacity_status(symbol, pending_positions=pending_positions) == "ok"

    def _check_correlation_limit(self, symbol: str) -> bool:
        return self._get_family_risk(symbol.split("/")[0]) < 0.25

    def _get_family_risk(self, coin: str) -> float:
        families = {
            "BTC": ["BTC"],
            "ETH": ["ETH"],
            "BNB": ["BNB"],
            "SOL": ["SOL"],
            "XRP": ["XRP", "ADA"],
            "AVAX": ["AVAX", "DOT"],
            "LINK": ["LINK", "TON"],
        }
        total_risk = 0.0
        for coins in families.values():
            if coin not in coins:
                continue
            for sym_pos in self.state.open_positions.values():
                if isinstance(sym_pos, list):
                    for pos in sym_pos:
                        if getattr(pos, "symbol", "").split("/")[0] in coins:
                            total_risk += self.config.risk_per_trade_max
                else:
                    if getattr(sym_pos, "symbol", "").split("/")[0] in coins:
                        total_risk += self.config.risk_per_trade_max
            break
        return total_risk

    def check_daily_reset(self) -> None:
        today = self._now().date()
        if today != self.state.today_start_date:
            self.state.today_start_date = today
            self.state.today_trades_count = 0
            self.state.consecutive_losses = 0
            self.state.reduced_risk_mode = False
            self.state.equity_start_of_day = self.state.balance
            self.state.realized_pl_today = 0.0
            self.state.wins_today = 0
            self.state.losses_today = 0
            self.state.emergency_mode = False
            self.state.cooldown_until = None

    def check_daily_loss_limit(self) -> bool:
        equity_start = self.state.equity_start_of_day
        current_equity = self.state.balance
        realized_loss = current_equity - equity_start
        max_allowed_loss = equity_start * self.config.daily_loss_limit_fraction * -1
        return realized_loss <= max_allowed_loss

    def update_after_trade_result(self, profit_loss: float) -> None:
        self.state.today_trades_count += 1
        self.state.realized_pl_today += profit_loss
        if profit_loss >= 0:
            self.state.wins_today += 1
        else:
            self.state.losses_today += 1
        if profit_loss < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
        self.state.lifetime_trades += 1
        self.state.lifetime_profit += profit_loss
        if profit_loss > self.state.best_single_trade:
            self.state.best_single_trade = profit_loss
        if profit_loss < self.state.worst_single_trade:
            self.state.worst_single_trade = profit_loss
        if (
            profit_loss < 0
            and self.state.consecutive_losses >= self.config.consecutive_loss_threshold
            and self.config.cooldown_minutes_after_loss_streak > 0
        ):
            self.state.cooldown_until = self._now() + dt.timedelta(minutes=self.config.cooldown_minutes_after_loss_streak)

    def calc_position_size(self, entry_price: float, stop_loss: float, signal: Dict[str, Any] | None = None) -> float:
        risk_amount = self.state.balance * self.current_risk_fraction()
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance <= 0 or risk_amount <= 0:
            return 0.0
        size = risk_amount / sl_distance
        if entry_price <= 0:
            return 0.0

        trading_mode = getattr(self.config, "trading_mode", "spot")
        leverage = max(float(getattr(self.config, "default_leverage", 1) or 1), 1.0)
        if trading_mode == "spot":
            max_size = self.state.balance / entry_price
        else:
            max_size = (self.state.balance * leverage) / entry_price
        metadata = dict((signal or {}).get("metadata", {}) or {})
        strategy = str((signal or {}).get("strategy", metadata.get("strategy", "unknown")))
        order_profile = str(metadata.get("preferred_order_type", "market" if bool((signal or {}).get("fast_move", False)) else "limit")).lower()
        strategy_variant = str(metadata.get("strategy_variant", "base"))
        volatility_percentile = float(metadata.get("realized_vol_percentile", 0.5) or 0.5)
        entry_zscore = abs(float(metadata.get("entry_zscore", 0.0) or 0.0))
        size_multiplier = 1.0
        if volatility_percentile >= 0.88:
            size_multiplier *= 0.76
        elif volatility_percentile >= 0.74:
            size_multiplier *= 0.88
        if strategy == "trend_breakout" and order_profile == "market":
            size_multiplier *= 0.88
        if strategy == "trend_breakout" and strategy_variant == "confirmed_market" and volatility_percentile <= 0.78:
            size_multiplier *= 1.06
        momentum_crash_risk = float(metadata.get("momentum_crash_risk", 0.0) or 0.0)
        if strategy == "trend_breakout" and momentum_crash_risk > 0:
            size_multiplier *= max(0.55, 1.0 - (momentum_crash_risk * 0.35))
        if bool(metadata.get("high_beta_quality_escape", False)):
            size_multiplier *= float(getattr(self.config, "high_beta_quality_escape_risk_multiplier", 0.55) or 0.55)
        if strategy == "mean_reversion" and momentum_crash_risk >= 0.45 and volatility_percentile <= 0.82:
            size_multiplier *= 1.05
        if strategy == "mean_reversion" and volatility_percentile <= 0.58 and entry_zscore >= 1.35:
            size_multiplier *= 1.04
        rotation_policy = metadata.get("rotation_policy", {}) or {}
        preferred_family = str(metadata.get("preferred_family", rotation_policy.get("preferred_family", "")) or "")
        suppressed_family = str(metadata.get("suppressed_family", rotation_policy.get("suppressed_family", "")) or "")
        rotation_confidence = float(metadata.get("rotation_confidence", rotation_policy.get("rotation_confidence", 0.0)) or 0.0)
        if rotation_confidence > 0:
            preferred_multiplier = float(getattr(self.config, "rotation_risk_preferred_multiplier", 1.06) or 1.06)
            suppressed_multiplier = float(getattr(self.config, "rotation_risk_suppressed_multiplier", 0.90) or 0.90)
            scaled_preferred = 1.0 + ((preferred_multiplier - 1.0) * max(0.0, min(rotation_confidence, 1.0)))
            scaled_suppressed = 1.0 - ((1.0 - suppressed_multiplier) * max(0.0, min(rotation_confidence, 1.0)))
            if strategy == preferred_family:
                size_multiplier *= scaled_preferred
            elif strategy == suppressed_family:
                size_multiplier *= scaled_suppressed
        sized = size * size_multiplier
        return min(sized, max_size)

    def _iter_positions(self) -> list[Any]:
        positions: list[Any] = []
        for value in self.state.open_positions.values():
            if isinstance(value, list):
                positions.extend(value)
            else:
                positions.append(value)
        return positions

    def portfolio_exposures(self) -> Dict[str, float]:
        gross = 0.0
        net = 0.0
        by_symbol: Dict[str, float] = {}
        by_strategy: Dict[str, float] = {}
        by_family: Dict[str, float] = {}
        for pos in self._iter_positions():
            symbol = getattr(pos, "symbol", "")
            strategy = getattr(pos, "strategy", "unknown")
            side = getattr(pos, "side", "long")
            notional = float(getattr(pos, "entry_price", 0.0) or 0.0) * float(getattr(pos, "size", 0.0) or 0.0)
            gross += notional
            net += notional if side == "long" else -notional
            by_symbol[symbol] = by_symbol.get(symbol, 0.0) + notional
            by_strategy[strategy] = by_strategy.get(strategy, 0.0) + notional
            family = self._family_for_symbol(symbol)
            by_family[family] = by_family.get(family, 0.0) + notional
        return {
            "gross": gross,
            "net": net,
            "balance": float(self.state.balance),
            "by_symbol": by_symbol,
            "by_strategy": by_strategy,
            "by_family": by_family,
        }

    def _family_for_symbol(self, symbol: str) -> str:
        base = symbol.split("/")[0]
        families = {
            "BTC": "BTC",
            "ETH": "ETH",
            "BNB": "BNB",
            "SOL": "SOL",
            "XRP": "XRP_ADA",
            "ADA": "XRP_ADA",
            "AVAX": "AVAX_DOT",
            "DOT": "AVAX_DOT",
            "LINK": "LINK_TON",
            "TON": "LINK_TON",
        }
        return families.get(base, base or "UNKNOWN")

    def evaluate_portfolio_risk(
        self,
        *,
        symbol: str,
        strategy: str,
        side: str,
        entry_price: float,
        proposed_size: float,
    ) -> RiskDecision:
        exposures = self.portfolio_exposures()
        balance = max(exposures["balance"], 0.0)
        gross = exposures["gross"]
        net = exposures["net"]
        proposed_notional = max(entry_price, 0.0) * max(proposed_size, 0.0)
        gross_after = gross + proposed_notional
        net_after = net + proposed_notional if side in ("long", "buy") else net - proposed_notional
        controls: Dict[str, Any] = {
            "operating_mode": getattr(self.config, "operating_mode", "paper"),
            "symbol": symbol,
            "strategy": strategy,
        }
        if proposed_size <= 0 or proposed_notional <= 0:
            return RiskDecision(False, "invalid_size", self.current_risk_fraction(), 0.0, gross, net, controls=controls, capped_size=0.0)
        if self._is_in_cooldown():
            return RiskDecision(False, "cooldown", self.current_risk_fraction(), 0.0, gross, net, controls=controls, capped_size=0.0)
        if balance <= 0:
            return RiskDecision(False, "no_balance", self.current_risk_fraction(), 0.0, gross, net, controls=controls, capped_size=0.0)

        gross_cap = balance * float(getattr(self.config, "max_gross_exposure_fraction", 1.0))
        net_cap = balance * float(getattr(self.config, "max_net_exposure_fraction", 1.0))
        symbol_cap = balance * float(getattr(self.config, "max_symbol_exposure_fraction", 0.35))
        strategy_cap = balance * float(getattr(self.config, "max_strategy_exposure_fraction", 0.50))
        family_cap = balance * float(getattr(self.config, "max_family_exposure_fraction", 0.45))
        mode_cap_fraction = 1.0
        operating_mode = getattr(self.config, "operating_mode", "paper")
        if operating_mode == "shadow":
            return RiskDecision(False, "shadow_read_only", self.current_risk_fraction(), 0.0, gross, net, controls=controls, capped_size=0.0)
        if operating_mode == "canary":
            mode_cap_fraction = float(getattr(self.config, "canary_max_notional_fraction", 0.10))
        elif operating_mode == "capital_limited_live":
            mode_cap_fraction = float(getattr(self.config, "capital_limited_live_max_notional_fraction", 0.25))

        symbol_used = exposures["by_symbol"].get(symbol, 0.0)
        strategy_used = exposures["by_strategy"].get(strategy, 0.0)
        family_used = exposures["by_family"].get(self._family_for_symbol(symbol), 0.0)
        effective_cap = min(
            gross_cap - gross,
            symbol_cap - symbol_used,
            strategy_cap - strategy_used,
            family_cap - family_used,
            (balance * mode_cap_fraction),
        )
        controls.update(
            {
                "gross_cap": gross_cap,
                "net_cap": net_cap,
                "symbol_cap": symbol_cap,
                "strategy_cap": strategy_cap,
                "family_cap": family_cap,
                "proposed_notional": proposed_notional,
            }
        )
        if effective_cap <= 0:
            return RiskDecision(False, "exposure_cap_reached", self.current_risk_fraction(), 0.0, gross, net, controls=controls, capped_size=0.0)
        if abs(net_after) > net_cap:
            return RiskDecision(False, "net_exposure_cap", self.current_risk_fraction(), effective_cap, gross, net, controls=controls, capped_size=max(effective_cap / entry_price, 0.0))
        if gross_after > gross_cap:
            return RiskDecision(False, "gross_exposure_cap", self.current_risk_fraction(), effective_cap, gross, net, controls=controls, capped_size=max(effective_cap / entry_price, 0.0))

        capped_size = min(proposed_size, effective_cap / entry_price)
        return RiskDecision(True, "ok", self.current_risk_fraction(), effective_cap, gross, net, controls=controls, capped_size=capped_size)
