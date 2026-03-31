from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BotConfig:
    starting_balance: float = 50.0
    risk_per_trade_min: float = 0.02
    risk_per_trade_max: float = 0.03
    max_open_positions: int = 3
    max_trades_per_day_min: int = 3
    max_trades_per_day_max: int = 6
    symbols: List[str] = field(
        default_factory=lambda: [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "AVAX/USDT",
            "DOT/USDT",
            "LINK/USDT",
            "TON/USDT",
        ]
    )
    timeframes: Dict[str, str] = field(
        default_factory=lambda: {
            "structure": "4h",
            "trend": "1h",
            "entry": "15m",
        }
    )
    strict_signals: bool = True
    avoid_chop: bool = True
    late_confirmation: bool = True
    enable_mean_reversion: bool = True
    enable_trend_breakout: bool = True
    enable_trend_pullback: bool = True
    allow_countertrend_in_chop: bool = True
    breakout_swing_lookback: int = 5
    breakout_rr_ratio: float = 2.2
    pullback_ema_fast_period: int = 8
    pullback_ema_slow_period: int = 21
    pullback_entry_atr_fraction: float = 0.35
    pullback_stop_atr_mult: float = 1.15
    pullback_rr_ratio: float = 2.0
    min_signal_quality_score: float = 0.55
    min_reliable_regime_confidence: float = 0.58
    min_reliable_rr_ratio_trend: float = 1.35
    min_reliable_rr_ratio_mean_reversion: float = 1.10
    min_hurst_for_trend_breakout: float = 0.12
    min_hurst_for_trend_pullback: float = 0.10
    max_hurst_for_mean_reversion: float = 0.65
    research_alignment_bonus: float = 10.0
    research_conflict_penalty: float = 14.0
    research_conflict_veto_confidence: float = 0.72
    research_risk_off_veto_confidence: float = 0.75
    max_entry_deviation_from_mid_fraction: float = 0.0045
    max_market_data_age_minutes: int = 90
    max_signal_exceptions_per_cycle: int = 2
    no_signal_cycles_before_alert: int = 30
    max_consecutive_order_failures: int = 3
    data_circuit_breaker_minutes: int = 30
    execution_circuit_breaker_minutes: int = 30
    use_paper_trading: bool = False
    trading_mode: str = "spot"
    operating_mode: str = "paper"
    daily_loss_limit_fraction: float = 0.10
    consecutive_loss_threshold: int = 2
    cooldown_minutes_after_loss_streak: int = 60
    reduced_risk_factor: float = 0.5
    emergency_volatility_threshold: float = 0.08
    max_spread_fraction: float = 0.002
    max_slippage_fraction: float = 0.003
    breakeven_rr: float = 1.0
    trailing_rr: float = 1.5
    trailing_atr_mult: float = 1.0
    trailing_atr_period: int = 14
    trailing_timeframe: str = "15m"
    morning_hour: int = 8
    evening_hour: int = 20
    news_engine_enabled: bool = True
    news_poll_interval_minutes: int = 15
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    default_leverage: int = 5
    symbol_leverage: Dict[str, int] = field(default_factory=dict)
    margin_type: str = "ISOLATED"
    futures_sl_order_type: str = "STOP_MARKET"
    futures_tp_order_type: str = "TAKE_PROFIT_MARKET"
    futures_position_mode: str = "one_way"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    use_testnet_public: bool = True
    backtest_warmup_candles: int = 100
    backtest_fee_bps: float = 10.0
    backtest_slippage_bps: float = 5.0
    backtest_spread_bps: float = 4.0
    backtest_partial_fill_ratio: float = 1.0
    backtest_latency_bars: int = 1
    max_gross_exposure_fraction: float = 1.0
    max_net_exposure_fraction: float = 1.0
    max_symbol_exposure_fraction: float = 0.35
    max_strategy_exposure_fraction: float = 0.50
    max_family_exposure_fraction: float = 0.45
    max_portfolio_var_fraction: float = 0.12
    min_expected_edge_bps: float = 8.0
    spread_shock_halt_fraction: float = 0.006
    slippage_shock_halt_fraction: float = 0.008
    shadow_mode_read_only: bool = True
    canary_max_notional_fraction: float = 0.10
    capital_limited_live_max_notional_fraction: float = 0.25
    broker_max_retries: int = 3
    broker_retry_delay_seconds: float = 0.5
    simulated_order_latency_ms: int = 150
    simulated_partial_fill_min_fraction: float = 0.65
    promotion_min_runtime_hours: float = 24.0
    promotion_max_reconciliation_drift_events: int = 0
    promotion_max_risk_halts: int = 2
    promotion_min_closed_trades: int = 10
    promotion_min_win_rate_pct: float = 35.0
    promotion_min_profit_factor: float = 0.90
    promotion_max_drawdown_pct: float = 12.0

    def validate(self) -> None:
        if self.starting_balance <= 0:
            raise ValueError("starting_balance must be > 0")
        if not (0 < self.risk_per_trade_min < self.risk_per_trade_max < 1):
            raise ValueError("risk_per_trade_min and risk_per_trade_max must be between 0 and 1")
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be > 0")
        if self.max_trades_per_day_max <= 0:
            raise ValueError("max_trades_per_day_max must be > 0")
        if not isinstance(self.symbols, list) or len(self.symbols) == 0:
            raise ValueError("symbols must be a non-empty list")
        if self.daily_loss_limit_fraction <= 0 or self.daily_loss_limit_fraction >= 1:
            raise ValueError("daily_loss_limit_fraction must be between 0 and 1")
        if self.operating_mode not in {"paper", "shadow", "canary", "capital_limited_live", "full_live"}:
            raise ValueError("operating_mode must be one of paper, shadow, canary, capital_limited_live, full_live")
