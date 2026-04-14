import requests
import os
from dotenv import load_dotenv
import signal
import threading
import json
import math
import numpy as np
import os
import sys
from typing import Optional, Dict, Any, List
import argparse
import warnings
from typing import Optional, Dict, Any, Tuple, Sequence
import time
import datetime as dt
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from trade_bot.metrics import MetricsCollector
from trade_bot.ai.assistant import BotOperatorAssistant, ensure_default_knowledge_base
from trade_bot.backtest_reporting import build_backtest_report, save_backtest_report
from trade_bot.bootstrap import ensure_runtime_directories, load_runtime_environment, resolve_runtime_base_dir
from trade_bot.audit import JsonlDecisionLogger
from trade_bot.constants import DATA_DIR, EVENT_LOG_FILE, LEARNING_DB_FILE, LOG_DIR, STATE_DB_FILE, STATE_FILE
from trade_bot.config import BotConfig as CoreBotConfig
from trade_bot.execution import ExecutionEngine as CoreExecutionEngine
from trade_bot.exchange import ExchangeClient as CoreExchangeClient, MockExchange as CoreMockExchange
from trade_bot.learning import TradeLearningEngine
from trade_bot.ensemble import EnsembleAllocator
from trade_bot.news_engine import BinanceNewsEngine
from trade_bot.persistence import _sanitize_jsonish, load_bot_state, save_bot_state
from trade_bot.process_service import write_bot_status
from trade_bot.readiness import build_readiness_report
from trade_bot.reconciliation import BotReconciler
from trade_bot.regime import MarketRegimeEngine
from trade_bot.risk import RiskManager as CoreRiskManager
from trade_bot.runtime import (
    build_portfolio_snapshot,
    build_risk_decision,
    build_strategy_health,
    emit_event,
    log_reconciliation,
    log_risk_halt,
    log_signal,
    new_trace_id,
    persist_runtime_snapshot,
)
from trade_bot.state import BotState as CoreBotState, Position as CorePosition
from trade_bot.state_store import PersistentLearningStore, SQLiteStateStore, backfill_learning_from_sqlite_artifacts
from trade_bot.strategies import build_strategies

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
load_runtime_environment(resolve_runtime_base_dir())

# Import ccxt lazily to avoid import-time hangs in paper mode


# ============================
# MOCK EXCHANGE FOR PAPER TRADING
# ============================
class MockExchange:
    """
    Paper trading exchange that simulates Binance behavior with realistic market data.
    Generates OHLCV data using random walk with trend and volatility components.
    """

    def __init__(self, config: 'BotConfig', state: 'BotState'):
        self.config = config
        self.state = state

        # Market data simulation
        # symbol -> list of OHLCV candles
        self.market_data: Dict[str, List[List[float]]] = {}
        # symbol -> last update timestamp
        self.last_update: Dict[str, dt.datetime] = {}
        self.base_prices = {  # Realistic starting prices
            "BTC/USDT": 45000.0,
            "ETH/USDT": 2800.0,
            "BNB/USDT": 320.0,
            "SOL/USDT": 95.0,
            "XRP/USDT": 0.55,
            "ADA/USDT": 0.45,
            "AVAX/USDT": 35.0,
            "DOT/USDT": 7.20,
            "LINK/USDT": 14.50,
            "TON/USDT": 2.10,
        }

        # Initialize market data for all symbols
        now = dt.datetime.now()
        for symbol in config.symbols:
            self._initialize_symbol_data(symbol, now)

        # Mock order book data
        self.order_books: Dict[str, Dict[str, float]] = {}
        for symbol in config.symbols:
            base_price = self.base_prices.get(
    symbol.split('/')[0] + '/USDT', 100.0)
            self.order_books[symbol] = {
                "bid": base_price * 0.9995,
                "ask": base_price * 1.0005
            }

        print(
            "[MOCK] Paper trading exchange initialized with realistic market simulation.")

    def _initialize_symbol_data(
    self,
    symbol: str,
     start_time: dt.datetime) -> None:
        """Initialize historical OHLCV data for a symbol"""
        base_price = self.base_prices.get(
    symbol.split("/")[0] + "/USDT", 100.0)
        candles: List[List[float]] = []
        current_price: float = float(base_price)
        current_time = start_time - dt.timedelta(hours=200)
        for _ in range(200):
            trend = random.uniform(-0.002, 0.002)
            volatility = random.gauss(0, 0.015)
            open_price = current_price
            high_price = open_price * (1 + abs(random.gauss(0, 0.01)))
            low_price = open_price * (1 - abs(random.gauss(0, 0.01)))
            close_price = open_price * (1 + trend + volatility)
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            candle = [
                float(int(current_time.timestamp() * 1000)),
                round(float(open_price), 4),
                round(float(high_price), 4),
                round(float(low_price), 4),
                round(float(close_price), 4),
                float(random.randint(100000, 1000000))
            ]
            candles.append(candle)
            current_price = close_price
            current_time += dt.timedelta(hours=1)
        self.market_data[symbol] = candles
        self.last_update[symbol] = current_time

    def _generate_new_candle(self, symbol: str,
                             timeframe: str) -> Optional[List[float]]:
        """Generate a new OHLCV candle for the given timeframe"""
        if symbol not in self.market_data:
            return None

        candles: List[List[float]] = self.market_data[symbol]
        if not candles:
            return None

        last_candle: List[float] = candles[-1]
        last_close: float = last_candle[4]
        last_time = dt.datetime.fromtimestamp(last_candle[0] / 1000)

        # Timeframe multipliers
        tf_multipliers = {
            "15m": 15,
            "1h": 60,
            "4h": 240,
        }
        minutes = tf_multipliers.get(timeframe, 60)

        new_time = last_time + dt.timedelta(minutes=minutes)

        # Generate realistic price movement
        trend_strength = random.uniform(-0.005, 0.005)
        volatility = random.gauss(0, 0.012)
        momentum = random.choice([-1, 1]) * random.uniform(0, 0.003)

        change = trend_strength + volatility + momentum
        open_price = last_close
        close_price = last_close * (1 + change)

        # Add some noise to create realistic OHLC
        noise = random.gauss(0, 0.005)
        high_price = max(open_price, close_price) * (1 + abs(noise))
        low_price = min(open_price, close_price) * (1 - abs(noise))

        # Ensure proper OHLC relationships
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        new_candle: List[float] = [
            float(int(new_time.timestamp() * 1000)),
            round(float(open_price), 4),
            round(float(high_price), 4),
            round(float(low_price), 4),
            round(float(close_price), 4),
            float(random.randint(50000, 500000))
        ]

        return new_candle

    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    limit: int = 200) -> List[List[float]]:
        """Return OHLCV data, generating new candles as needed"""
        if symbol not in self.market_data:
            return []

        candles: List[List[float]] = self.market_data[symbol]

        # Generate new candles if needed
        now = dt.datetime.now()
        last_update: dt.datetime = self.last_update.get(
            symbol, now - dt.timedelta(hours=1))

        # Timeframe in minutes
        tf_minutes = {"15m": 15, "1h": 60, "4h": 240}.get(timeframe, 60)

        while (now - last_update).total_seconds() > (tf_minutes * 60):
            new_candle = self._generate_new_candle(symbol, timeframe)
            if new_candle:
                candles.append(new_candle)
                last_update = dt.datetime.fromtimestamp(new_candle[0] / 1000)
                self.last_update[symbol] = last_update

        # Return the most recent 'limit' candles
        return candles[-limit:] if len(candles) >= limit else candles

    def get_order_book(self, symbol: str):
        """Return mock order book data"""
        if symbol not in self.order_books:
            base_price = self.base_prices.get(
    symbol.split('/')[0] + '/USDT', 100.0)
            self.order_books[symbol] = {
                "bid": base_price * 0.9995,
                "ask": base_price * 1.0005
            }

        # Update order book with some small random movement
        current = self.order_books[symbol]
        spread = current["ask"] - current["bid"]
        mid = (current["ask"] + current["bid"]) / 2

        # Small random movement
        movement = random.gauss(0, 0.001)
        new_mid = mid * (1 + movement)

        self.order_books[symbol] = {
            "bid": new_mid - spread / 2,
            "ask": new_mid + spread / 2
        }

        return self.order_books[symbol]

    def place_order(self, symbol: str, side: str, size: float, order_type: str,
                   price: Optional[float] = None, stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None, is_futures: bool = False) -> bool:
        """Simulate order placement in paper mode"""
        print(f"[MOCK ORDER] {order_type.upper()} {side.upper()} {symbol} "
              f"size={size}, price={price}, SL={stop_loss}, TP={take_profit}, futures={is_futures}")
        return True


# ============================
# 1) CONFIG – YOUR RULES
# ============================


@dataclass
class BotConfig:
    starting_balance: float = 50.0
    # Risk
    risk_per_trade_min: float = 0.02  # 2%
    risk_per_trade_max: float = 0.03  # 3%
    max_open_positions: int = 3
    max_trades_per_day_min: int = 3
    max_trades_per_day_max: int = 6
    # Symbols / timeframes
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
    # Behavior
    strict_signals: bool = True
    avoid_chop: bool = True
    late_confirmation: bool = True
    enable_mean_reversion: bool = True
    enable_trend_breakout: bool = True
    allow_countertrend_in_chop: bool = True
    breakout_swing_lookback: int = 5
    breakout_rr_ratio: float = 2.2
    breakout_confirmed_rr_ratio: float = 1.9
    breakout_retest_rr_ratio: float = 1.7
    breakout_confirm_close_atr_fraction: float = 0.05
    breakout_max_stretch: float = 0.045
    breakout_retest_entry_buffer_atr_fraction: float = 0.04
    breakout_retest_expiry_bars: int = 10
    breakout_confirmed_market_rr_ratio: float = 1.55
    breakout_confirmed_market_min_body_fraction: float = 0.58
    breakout_confirmed_market_min_close_location_long: float = 0.72
    breakout_confirmed_market_max_close_location_short: float = 0.28
    breakout_confirmed_market_min_breakout_atr: float = 0.12
    breakout_confirmed_market_max_stretch: float = 0.032
    breakout_confirmed_market_min_continuation_score: float = 1.58
    breakout_confirmed_market_min_breakout_score: float = 1.62
    breakout_confirmed_market_min_trend_persistence: float = 0.44
    breakout_confirmed_market_min_volume_impulse: float = 1.08
    breakout_confirmed_market_min_trend_efficiency: float = 0.34
    breakout_post_shock_drawdown_floor: float = -0.045
    breakout_post_shock_rebound_floor: float = 0.012
    breakout_post_shock_crash_risk_floor: float = 0.60
    breakout_weak_confirmation_penalty_bps: float = 5.0
    pullback_ema_fast_period: int = 8
    pullback_ema_slow_period: int = 21
    pullback_entry_atr_fraction: float = 0.35
    pullback_confirmation_buffer_atr_fraction: float = 0.08
    pullback_shallow_rr_ratio: float = 1.35
    pullback_deep_rr_ratio: float = 1.2
    pullback_max_stretch: float = 0.035
    pullback_shallow_min_pullback_score: float = 1.24
    pullback_shallow_min_trend_persistence: float = 0.38
    pullback_deep_min_continuation_score: float = 1.58
    pullback_shallow_max_volatility_percentile: float = 0.72
    pullback_deep_max_volatility_percentile: float = 0.88
    pullback_deep_min_trend_efficiency: float = 0.16
    pullback_deep_min_volume_impulse: float = 0.92
    pullback_reclaim_body_fraction_min: float = 0.18
    pullback_reclaim_close_location_min: float = 0.24
    pullback_reclaim_close_location_max: float = 0.76
    pullback_reclaim_buffer_atr_fraction: float = 0.04
    min_signal_quality_score: float = 0.55
    min_reliable_regime_confidence: float = 0.54
    min_reliable_rr_ratio_trend: float = 1.35
    min_reliable_rr_ratio_mean_reversion: float = 1.10
    rotation_reliability_quality_delta: float = 0.03
    rotation_reliability_edge_delta_bps: float = 1.5
    rotation_reliability_rr_delta: float = 0.06
    rotation_pullback_reliability_quality_delta: float = 0.12
    rotation_pullback_reliability_edge_delta_bps: float = 9.0
    rotation_pullback_reliability_rr_delta: float = 0.30
    rotation_risk_preferred_multiplier: float = 1.06
    rotation_risk_suppressed_multiplier: float = 0.90
    min_hurst_for_trend_breakout: float = 0.16
    min_hurst_for_trend_pullback: float = 0.18
    max_hurst_for_mean_reversion: float = 0.60
    mean_reversion_entry_zscore: float = 1.10
    mean_reversion_max_efficiency: float = 0.48
    mean_reversion_min_exhaustion_score: float = 0.42
    mean_reversion_min_rebound_strength: float = 0.12
    mean_reversion_min_fade_strength: float = 0.12
    mean_reversion_liquid_relaxed_liquidity_score: float = 0.78
    mean_reversion_liquid_relaxed_efficiency_bonus: float = 0.06
    mean_reversion_liquid_relaxed_exhaustion_delta: float = 0.05
    mean_reversion_liquid_relaxed_rebound_delta: float = 0.03
    mean_reversion_liquid_relaxed_fade_delta: float = 0.03
    mean_reversion_countertrend_continuation_score: float = 1.28
    mean_reversion_countertrend_efficiency_floor: float = 0.34
    mean_reversion_countertrend_trend_strength_floor: float = 0.010
    # 👉 ALAPÉRTELMEZETTEN NEM PAPERMÓD
    use_paper_trading: bool = False
    # Execution mode
    trading_mode: str = "spot"  # "spot", "futures", or "mixed"
    # Safety
    daily_loss_limit_fraction: float = 0.10  # 10% max daily loss
    consecutive_loss_threshold: int = 2
    # minutes to pause trading after hitting loss streak
    cooldown_minutes_after_loss_streak: int = 60
    reduced_risk_factor: float = 0.5  # halve risk when in reduced mode
    emergency_volatility_threshold: float = 0.08  # e.g., 8% candle move
    # Execution
    max_spread_fraction: float = 0.002  # 0.2%
    max_slippage_fraction: float = 0.003  # 0.3%
    # Dynamic stops / partial management (long-only for now)
    breakeven_rr: float = 1.0  # move SL to breakeven after +1R
    trailing_rr: float = 1.5  # start trailing after this R multiple
    trailing_atr_mult: float = 1.0  # trail distance = ATR * mult
    trailing_atr_period: int = 14  # ATR period for trailing
    trailing_timeframe: str = "15m"  # timeframe to compute trailing ATR
    trailing_breakout_rr: float = 1.10
    trailing_pullback_rr: float = 1.80
    trailing_mean_reversion_rr: float = 2.50
    trailing_breakout_atr_mult: float = 0.85
    trailing_pullback_atr_mult: float = 1.20
    trailing_mean_reversion_atr_mult: float = 1.35
    profit_protect_breakout_trigger_rr: float = 0.75
    profit_protect_pullback_trigger_rr: float = 1.10
    profit_protect_mean_reversion_trigger_rr: float = 0.60
    profit_protect_breakout_lock_rr: float = 0.20
    profit_protect_pullback_lock_rr: float = 0.10
    profit_protect_mean_reversion_lock_rr: float = 0.18
    mean_reversion_profit_capture_trigger_rr: float = 0.45
    mean_reversion_profit_capture_lock_rr: float = 0.30
    mean_reversion_reclaim_failure_activation_rr: float = 0.25
    mean_reversion_reclaim_failure_buffer_rr: float = 0.05
    partial_profit_take_breakout_trigger_rr: float = 1.00
    partial_profit_take_breakout_fraction: float = 0.35
    partial_profit_take_mean_reversion_trigger_rr: float = 0.70
    partial_profit_take_mean_reversion_fraction: float = 0.50
    mean_reversion_reclaim_failure_cooldown_bars: int = 6
    breakout_volatility_exit_cooldown_bars: int = 8
    volatility_tightening_trigger_ratio: float = 1.30
    volatility_tightening_breakout_rr: float = 0.30
    volatility_tightening_pullback_rr: float = 0.12
    volatility_tightening_mean_reversion_rr: float = 0.22
    time_stop_soft_holding_multiplier: float = 1.35
    time_stop_soft_min_r_multiple: float = 0.15
    time_stop_hard_holding_multiplier: float = 2.25
    time_stop_hard_min_r_multiple: float = 0.45
    time_stop_breakout_soft_multiplier: float = 0.85
    time_stop_breakout_hard_multiplier: float = 0.80
    time_stop_pullback_soft_multiplier: float = 1.15
    time_stop_pullback_hard_multiplier: float = 1.20
    min_signal_quality_score_pullback: float = 0.62
    min_reliable_rr_ratio_pullback: float = 1.55
    time_stop_mean_reversion_soft_multiplier: float = 0.80
    time_stop_mean_reversion_hard_multiplier: float = 0.75
    # Dashboard times (local)
    morning_hour: int = 8
    evening_hour: int = 20
    news_engine_enabled: bool = True
    news_poll_interval_minutes: int = 15
    # Telegram (optional – if None, no notifications)
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
        # ============================
    # FUTURES-SPECIFIKUS BEÁLLÍTÁSOK
    # ============================

    # Alapértelmezett tőkeáttét minden szimbólumra
    default_leverage: int = 5

    # Szimbólum-specifikus tőkeáttét override
    # Példa: {"BTC/USDT": 10, "ETH/USDT": 7}
    symbol_leverage: Dict[str, int] = field(default_factory=dict)

    # Fedezeti mód: "ISOLATED" vagy "CROSSED"
    margin_type: str = "ISOLATED"

    # Futures SL/TP megbízás típusa
    # "STOP_MARKET" = piaci stop, "STOP" = limit stop
    futures_sl_order_type: str = "STOP_MARKET"
    futures_tp_order_type: str = "TAKE_PROFIT_MARKET"

    # Pozíció mód: "one_way" vagy "hedge"
    # "one_way" = egyirányú (alapértelmezett Binance-n)
    # "hedge"   = fedezeti mód (long + short egyszerre)
    futures_position_mode: str = "one_way"
    # Testnet / API
    # ============================
    # FUTURES-SPECIFIKUS BEÁLLÍTÁSOK
    # ============================
    default_leverage: int = 5
    symbol_leverage: Dict[str, int] = field(default_factory=dict)
    margin_type: str = "ISOLATED"
    futures_sl_order_type: str = "STOP_MARKET"
    futures_tp_order_type: str = "TAKE_PROFIT_MARKET"
    futures_position_mode: str = "one_way"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    # NEW: control whether public data uses Binance testnet (testnet.binance.vision)
    # If True -> public_client.set_sandbox_mode(True)
    # If False -> public_client uses mainnet (api.binance.com)
    use_testnet_public: bool = True
    simulation_random_seed: int = 7
    simulation_latency_jitter_bars: int = 0
    simulation_limit_order_expiry_bars: int = 4
    simulation_stale_order_cancel_bars: int = 3
    simulation_stale_order_cancel_distance_bps: float = 18.0
    simulation_stale_order_reprice_max_attempts: int = 1
    simulation_stale_order_reprice_quality_floor: float = 0.68
    simulation_stale_order_reprice_edge_floor_bps: float = 16.0
    simulation_stale_order_reprice_offset_bps: float = 6.0
    simulation_stale_order_reprice_extension_bars: int = 2
    simulation_stale_order_reprice_early_bars_delta: int = 1
    simulation_stale_order_reprice_early_quality_floor: float = 0.76
    simulation_stale_order_reprice_early_edge_floor_bps: float = 22.0
    simulation_stale_order_extra_reprice_quality_floor: float = 0.76
    simulation_stale_order_extra_reprice_edge_floor_bps: float = 20.0
    simulation_breakout_stale_order_extra_reprice_quality_floor: float = 0.80
    simulation_breakout_stale_order_extra_reprice_edge_floor_bps: float = 24.0
    simulation_limit_latency_reduction_quality_floor: float = 0.76
    simulation_limit_latency_reduction_edge_floor_bps: float = 20.0
    simulation_breakout_limit_latency_reduction_quality_floor: float = 0.80
    simulation_breakout_limit_latency_reduction_edge_floor_bps: float = 24.0
    simulation_limit_queue_priority_quality_floor: float = 0.74
    simulation_limit_queue_priority_edge_floor_bps: float = 18.0
    simulation_breakout_limit_queue_priority_quality_floor: float = 0.78
    simulation_breakout_limit_queue_priority_edge_floor_bps: float = 22.0
    simulation_stale_market_escalation_quality_floor: float = 0.76
    simulation_stale_market_escalation_edge_floor_bps: float = 20.0
    simulation_breakout_stale_market_escalation_quality_floor: float = 0.80
    simulation_breakout_stale_market_escalation_edge_floor_bps: float = 24.0
    simulation_volume_participation_rate: float = 0.20
    simulation_slippage_volatility_weight: float = 0.12
    simulation_volume_impact_weight: float = 0.10
    simulation_market_impact_exponent: float = 0.5
    simulation_market_impact_weight: float = 0.08
    simulation_passive_fill_spread_capture: float = 0.20
    simulation_queue_decay: float = 0.55
    simulation_aggressive_entry_quality_floor: float = 0.72
    simulation_aggressive_entry_edge_floor_bps: float = 18.0
    simulation_aggressive_entry_max_distance_bps: float = 14.0
    simulation_aggressive_entry_rr_floor: float = 1.45
    simulation_breakout_aggressive_entry_quality_floor: float = 0.80
    simulation_breakout_aggressive_entry_edge_floor_bps: float = 26.0
    simulation_breakout_aggressive_entry_max_distance_bps: float = 8.0
    simulation_breakout_aggressive_entry_rr_floor: float = 1.70
    simulation_pullback_limit_offset_bps: float = 4.0
    simulation_mean_reversion_limit_offset_bps: float = 7.0
    simulation_high_quality_limit_offset_tightening_bps: float = 2.0
    simulation_high_quality_limit_quality_floor: float = 0.76
    simulation_high_quality_limit_edge_floor_bps: float = 22.0
    simulation_limit_offset_stop_distance_cap_fraction: float = 0.22
    simulation_pullback_limit_expiry_bonus_bars: int = 2
    simulation_breakout_limit_expiry_bonus_bars: int = 1
    simulation_mean_reversion_limit_expiry_bonus_bars: int = 1
    simulation_expiry_bonus_quality_floor: float = 0.70
    simulation_touch_escalation_min_touches: int = 2
    simulation_touch_escalation_quality_floor: float = 0.74
    simulation_touch_escalation_edge_floor_bps: float = 20.0
    simulation_breakout_touch_escalation_min_touches: int = 1
    simulation_breakout_touch_escalation_quality_floor: float = 0.82
    simulation_breakout_touch_escalation_edge_floor_bps: float = 28.0
    simulation_no_trade_buffer_bps: float = 1.5
    simulation_no_trade_crowding_penalty_bps: float = 4.0
    simulation_symbol_crowding_penalty: float = 14.0
    simulation_family_crowding_penalty: float = 7.5
    simulation_bucket_crowding_penalty: float = 5.0
    simulation_directional_cluster_penalty: float = 6.0
    simulation_diversification_bonus: float = 3.5
    simulation_bucket_diversification_bonus: float = 2.5
    min_expected_edge_bps_pullback: float = 15.0
    simulation_duplicate_family_throttle_score_gap: float = 3.0
    simulation_duplicate_bucket_throttle_score_gap: float = 2.5
    simulation_learning_score_weight: float = 1.0
    simulation_learning_negative_penalty: float = 2.5
    simulation_realized_perf_min_trades: int = 2
    simulation_realized_symbol_penalty_score: float = 5.0
    simulation_realized_strategy_penalty_score: float = 4.0
    simulation_realized_negative_expectancy_floor: float = -1.0
    simulation_realized_positive_expectancy_floor: float = 1.0
    simulation_realized_symbol_positive_score: float = 2.5
    simulation_realized_strategy_positive_score: float = 2.0
    simulation_realized_no_trade_penalty_bps: float = 3.0
    simulation_pullback_realized_symbol_penalty_score: float = 3.0
    simulation_pullback_realized_symbol_no_trade_penalty_bps: float = 2.0
    simulation_weak_cluster_min_trades: int = 2
    simulation_weak_cluster_negative_expectancy_floor: float = -1.0
    simulation_universe_top_n: int = 6
    simulation_universe_tradability_floor: float = 45.0
    simulation_universe_spread_penalty_weight: float = 1.4
    simulation_universe_volatility_penalty_weight: float = 60.0
    simulation_universe_volume_score_weight: float = 4.0
    simulation_universe_bucket_cap: int = 2
    simulation_universe_bucket_cap_majors: int = 2
    simulation_universe_bucket_cap_exchange_beta: int = 2
    simulation_universe_bucket_cap_high_beta_alts: int = 2
    simulation_universe_bucket_cap_slower_large_caps: int = 2
    simulation_universe_bucket_cap_other: int = 2
    simulation_universe_regime_lookback_bars: int = 20
    simulation_universe_regime_high_vol_threshold: float = 0.02
    simulation_universe_regime_trend_strength_threshold: float = 0.015
    simulation_universe_regime_trend_exchange_beta_delta: int = 0
    simulation_universe_regime_trend_high_beta_delta: int = 0
    simulation_universe_regime_risk_off_exchange_beta_delta: int = 0
    simulation_universe_regime_risk_off_high_beta_delta: int = 0
    simulation_universe_regime_risk_off_slower_large_caps_delta: int = 0
    simulation_universe_regime_risk_off_other_delta: int = 0
    simulation_universe_realized_min_trades: int = 2
    simulation_universe_realized_negative_expectancy_floor: float = -1.0
    simulation_universe_realized_positive_expectancy_floor: float = 1.0
    simulation_universe_realized_penalty_score: float = 10.0
    simulation_universe_realized_boost_score: float = 5.0
    simulation_universe_realized_veto_min_trades: int = 2
    simulation_universe_realized_veto_expectancy_floor: float = -8.0
    simulation_symbol_probation_veto_min_trades: int = 1
    simulation_symbol_probation_veto_expectancy_floor: float = -20.0
    simulation_disable_trend_pullback: bool = False
    simulation_pullback_strategy_veto_min_trades: int = 2
    simulation_pullback_strategy_veto_expectancy_floor: float = -20.0
    simulation_snapshot_interval_bars: int = 250
    simulation_checkpoint_interval_bars: int = 250
    simulation_enable_checkpointing: bool = True
    simulation_resume_from_checkpoint: bool = True
    target_trades_per_day_min: float = 2.0
    target_trades_per_day_max: float = 3.0
    target_trades_per_day_soft_floor: float = 1.5
    target_trades_per_day_soft_ceiling: float = 3.5
    simulation_stress_every_n_bars: int = 0
    simulation_stress_shock_bps: float = 0.0
    simulation_queue_model: str = "fractional_queue"
    learning_decay_half_life_days: float = 45.0
    learning_prior_strength: float = 5.0
    learning_min_effective_samples: float = 3.0
    learning_negative_score_cap: float = 9.0
    learning_positive_score_cap: float = 5.0
    learning_confidence_penalty_cap: float = 0.08
    learning_confidence_boost_cap: float = 0.04
    learning_min_risk_multiplier: float = 0.72
    learning_max_risk_multiplier: float = 1.10
    learning_severe_pattern_samples: float = 10.0
    learning_structural_veto_ratio: float = 0.82
    learning_severe_avg_r_multiple: float = -0.85
    learning_drift_ewma_alpha: float = 0.12
    learning_drift_threshold: float = 2.8
    learning_drift_slack: float = 0.15
    learning_drift_cusum_decay: float = 0.92
    learning_calibration_prior_strength: float = 6.0
    learning_calibration_gap_alpha: float = 0.15
    learning_positive_update_min_samples: float = 8.0
    learning_positive_update_max_calibration_gap: float = 0.18
    learning_opportunity_min_samples: float = 6.0
    learning_positive_opportunity_min_samples: float = 6.0
    learning_positive_opportunity_avg_forward_r: float = 0.22
    learning_positive_opportunity_positive_ratio: float = 0.55
    learning_missed_opportunity_min_samples: float = 4.0
    learning_missed_opportunity_avg_forward_r: float = 0.18
    learning_missed_opportunity_positive_ratio: float = 0.56
    learning_positive_prequential_min_samples: float = 6.0
    learning_positive_prequential_avg_r: float = 0.18
    learning_positive_prequential_win_rate: float = 0.52
    learning_positive_prequential_max_brier: float = 0.30
    learning_positive_calibration_slack: float = 0.24
    learning_positive_min_score_delta: float = 1.10
    learning_positive_min_confidence_delta: float = 0.01
    learning_positive_min_risk_multiplier: float = 1.03
    learning_bucket_positive_sample_delta: float = 2.0
    learning_bucket_negative_sample_delta: float = 2.0
    learning_bucket_positive_score_boost: float = 0.50
    learning_bucket_negative_score_penalty: float = 0.75
    learning_bucket_positive_risk_boost: float = 0.03
    learning_bucket_negative_risk_cap: float = 0.84
    learning_family_rotation_window: int = 18
    learning_family_rotation_min_samples: float = 3.0
    learning_family_rotation_negative_avg_r: float = -0.20
    learning_family_rotation_negative_win_rate: float = 0.35
    learning_family_rotation_positive_avg_r: float = 0.15
    learning_family_rotation_positive_win_rate: float = 0.52
    learning_family_rotation_hard_negative_avg_r: float = -0.45
    learning_family_rotation_hard_negative_win_rate: float = 0.28
    learning_family_rotation_hard_positive_avg_r: float = 0.22
    learning_family_rotation_hard_positive_win_rate: float = 0.56
    learning_family_rotation_hard_gap_r: float = 0.35
    learning_family_rotation_hard_gap_win_rate: float = 0.18
    learning_family_rotation_recovery_window: int = 4
    learning_family_rotation_recovery_min_samples: float = 2.0
    learning_family_rotation_recovery_avg_r: float = 0.08
    learning_family_rotation_recovery_win_rate: float = 0.55
    learning_family_rotation_recovery_score_boost: float = 0.45
    learning_family_rotation_recovery_risk_boost: float = 0.02
    learning_family_rotation_penalty_score: float = 1.50
    learning_family_rotation_penalty_risk_cap: float = 0.82
    learning_family_rotation_boost_score: float = 0.75
    learning_family_rotation_boost_risk: float = 0.04
    learning_family_rotation_hard_penalty_score: float = 2.75
    learning_family_rotation_hard_penalty_risk_cap: float = 0.74
    learning_family_rotation_hard_boost_score: float = 1.35
    learning_family_rotation_hard_boost_risk: float = 0.06

    def validate(self):
        """Basic configuration validation"""
        if self.starting_balance <= 0:
            raise ValueError("starting_balance must be > 0")
        if not (0 < self.risk_per_trade_min < self.risk_per_trade_max < 1):
            raise ValueError(
                "risk_per_trade_min and risk_per_trade_max must be between 0 and 1")
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be > 0")
        if self.max_trades_per_day_max <= 0:
            raise ValueError("max_trades_per_day_max must be > 0")
        if not isinstance(self.symbols, list) or len(self.symbols) == 0:
            raise ValueError("symbols must be a non-empty list")
        if self.daily_loss_limit_fraction <= 0 or self.daily_loss_limit_fraction >= 1:
            raise ValueError(
                "daily_loss_limit_fraction must be between 0 and 1")


# ============================
# 2 BOT STATE
# ============================
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


@dataclass
class BotState:
    balance: float = 0.0

    # IMPORTANT: keep interface as-is for backward compatibility
    # Currently: one position per symbol – future multi-position support
    # should be added carefully in other modules when you decide.
    open_positions: Dict[str, Any] = field(default_factory=dict)

    today_trades_count: int = field(default=0)
    today_start_date: dt.date = field(
    default_factory=lambda: dt.datetime.now().date())

    consecutive_losses: int = 0
    reduced_risk_mode: bool = False
    emergency_mode: bool = False

    last_morning_dashboard_date: Optional[dt.date] = None
    last_evening_dashboard_date: Optional[dt.date] = None
    paper_mode: bool = True  # ezt a TradeBot állítja config alapján

    # Napi performance
    equity_start_of_day: float = 0.0
    realized_pl_today: float = 0.0
    wins_today: int = 0
    losses_today: int = 0

    # Globális drawdown
    peak_equity: float = 0.0

    # ----- NEW FIELDS (ALL PASSIVE BY DEFAULT) -----

    # Cooldown / lockout after loss streak – to be used later by
    # RiskManager/TradeBot.
    cooldown_until: Optional[dt.datetime] = None

    # Heartbeat / monitoring – last time we sent a "still alive" message.
    last_heartbeat: Optional[dt.datetime] = None

    # Lifetime statistics (for analytics & dashboards; not used yet).
    lifetime_profit: float = 0.0
    lifetime_trades: int = 0
    best_single_trade: float = 0.0
    worst_single_trade: float = 0.0

    # Unrealized PnL and equity tracking (for more accurate DD and reporting).
    unrealized_pnl: float = 0.0
    last_equity_update: Optional[dt.datetime] = None

    # Placeholder to control behavior of position handling in the future.
    # For now it does nothing, but later you can use it to switch to
    # multiple positions per symbol without changing the schema again.
    multi_position_mode: bool = False
    last_news_scan_at: Optional[dt.datetime] = None
    pending_news_commands: List[Dict[str, Any]] = field(default_factory=list)
    market_regime_alerts: Dict[str, str] = field(default_factory=dict)


# ============================
# 3 EXCHANGE CLIENT (ccxt + Binance)
# ============================
class ExchangeClient:
    """
    Exchange client using ccxt for Binance:
    - public_client: Binance adat (OHLCV, order book)
    - trade_client: Binance sandbox/testnet (order placement, SPOT for now)
    - futures_trade_client: Optional Binance FUTURES TESTNET client (prepared, not used yet)
    """

    def __init__(self, config: BotConfig, state: BotState) -> None:
        self.config = config
        self.state = state
        self.public_client: "Any" = None
        self.trade_client: "Any" = None
        self.futures_trade_client: "Any" = None
        self.last_order: Any = None

        # In paper mode, skip real exchange connection to avoid hanging
        if state.paper_mode:
            print("[INFO] Paper mode: skipping real exchange client initialization.")
            return

        # Lazy import ccxt only when needed for live mode
        import ccxt as ccxt_module

        # ----- PUBLIC CLIENT (DATA ONLY) -----
        self.public_client = ccxt_module.binance(
            {
                "enableRateLimit": True,
                "timeout": 5000,  # 5 second timeout in milliseconds
            }
        )

        # Decide whether public data uses MAINNET or TESTNET
        if getattr(self.config, "use_testnet_public", True):
            # This switches binance URLs to testnet (testnet.binance.vision)
            self.public_client.set_sandbox_mode(True)
            print(
                "[INFO] Public client set to Binance TESTNET (testnet.binance.vision).")
        else:
            print("[INFO] Public client using Binance MAINNET (api.binance.com).")

        # ----- SPOT TRADE CLIENT FOR TESTNET (SANDBOX MODE) -----
        self.trade_client = ccxt_module.binance(
            {
                "apiKey": config.api_key,
                "secret": config.api_secret,
                "enableRateLimit": True,
                "timeout": 5000,  # 5 second timeout in milliseconds
            }
        )
        self.trade_client.set_sandbox_mode(True)
        print("[INFO] Spot trade client set to Binance TESTNET (sandbox mode).")

        # ----- FUTURES TRADE CLIENT PLACEHOLDER (PREPARED, NOT YET USED) -----
        # We prepare a futures client for later use, but DO NOT change behavior yet.
        # This keeps current bot behavior spot-only unless you explicitly
        # integrate futures.
        self.futures_trade_client = None
        trading_mode = getattr(self.config, "trading_mode", "spot")

        if trading_mode in ("futures", "mixed"):
            try:
                self.futures_trade_client = ccxt_module.binance(
                    {
                        "apiKey": config.api_key,
                        "secret": config.api_secret,
                        "enableRateLimit": True,
                        "timeout": 5000,  # 5 second timeout in milliseconds
                        # Tell ccxt we intend to use futures endpoints (for
                        # later wiring)
                        "options": {"defaultType": "future"},
                    }
                )
                self.futures_trade_client.set_sandbox_mode(True)
                print(
                    "[INFO] Futures trade client prepared for Binance FUTURES TESTNET (sandbox mode).")
            except Exception as e:
                # We do not fail hard – we just log and continue with spot-only
                # behavior
                print(f"[WARN] Failed to initialize futures trade client: {type(e).__name__}: {e}")
                self.futures_trade_client = None

        # ----- LOAD MARKETS (BEST-EFFORT) -----
        # This helps ccxt know symbol metadata; failures do not stop the bot.
        try:
            self.public_client.load_markets()
        except Exception as e:
            print(f"[WARN] Failed to load public markets: {type(e).__name__}: {e}")

        try:
            self.trade_client.load_markets()
        except Exception as e:
            print(f"[WARN] Failed to load spot trade markets: {type(e).__name__}: {e}")

        if self.futures_trade_client is not None:
            try:
                self.futures_trade_client.load_markets()
            except Exception as e:
                print(f"[WARN] Failed to load futures trade markets: {type(e).__name__}: {e}")

        # Track the last raw order response for debugging / analytics
        # (optional)
        self.last_order = None

    def safe_fetch(
        self,
        func: callable,
        *args: Any,
        retries: int = 3,
        fallback: Any = None,
        timeout: int = 5,
        **kwargs: Any
    ) -> Any:
        """
        Retry wrapper – tries a few times before giving up and returning fallback.
        Now includes timeout to prevent hanging indefinitely.
        """
        func_name = getattr(func, "__name__", str(func))
        for attempt in range(retries):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                print(
                    f"[WARN] Temporary API error ({attempt + 1}/{retries}) "
                    f"in {func_name}: {type(e).__name__}: {e}"
                )
                time.sleep(0.5)

        print("[ERROR] API failed after retries, using fallback.")
        return fallback

    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    limit: int = 200) -> List[List[float]]:
        """
        OHLCV gyertyák lekérése Binance-ről (public adat).
        In paper mode, returns mock data.
        """
        if self.public_client is None:
            # Paper mode: return mock candles
            return [[0.0, 100.0, 105.0, 95.0, 101.0, 1000.0]
                for _ in range(limit)]

        return self.safe_fetch(
            self.public_client.fetch_ohlcv,
            symbol,
            timeframe=timeframe,
            limit=limit,
            fallback=[],
        ) or []

    def get_order_book(self, symbol: str) -> Dict[str, float]:
        """
        Order book lekérése Binance-ről (spread ellenőrzéshez).
        In paper mode, returns mock data.
        """
        if self.public_client is None:
            # Paper mode: return reasonable mock spread
            return {"bid": 100.0, "ask": 100.1}

        ob: Dict[str, List[List[float]]] = self.safe_fetch(
            self.public_client.fetch_order_book,
            symbol,
            limit=5,
            fallback={"bids": [[100.0, 10.0]], "asks": [[100.1, 10.0]]},
        ) or {}
        try:
            bid: float = ob["bids"][0][0]
            ask: float = ob["asks"][0][0]
            return {"bid": bid, "ask": ask}
        except Exception:
            return {"bid": 100.0, "ask": 100.1}

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        is_futures: bool = False,
    ) -> bool:
        """
        Order küldése:
        - paper_mode=True  → csak szimulál (MockExchange vagy log)
        - paper_mode=False → valós order a TESTNET-re
          * is_futures=False → SPOT trade_client
          * is_futures=True  → FUTURES futures_trade_client
            - Leverage beállítás
            - Margin type beállítás
            - Főmegbízás (market/limit)
            - Stop-Loss mellék-megbízás
            - Take-Profit mellék-megbízás
        """

        # ------------------------------------------------
        # PAPER MODE – nincs valós order, csak napló
        # ------------------------------------------------
        if self.state.paper_mode or self.trade_client is None:
            print(
                f"[PAPER] {order_type.upper()} {side.upper()} {symbol} "
                f"size={size}, price={price}, SL={stop_loss}, TP={take_profit}, "
                f"futures={is_futures}"
            )
            return True

        # ------------------------------------------------
        # FUTURES ÚT
        # ------------------------------------------------
        trading_mode = getattr(self.config, "trading_mode", "spot")

        if is_futures and trading_mode in ("futures", "mixed"):
            return self._place_futures_order(
                symbol=symbol,
                side=side,
                size=size,
                order_type=order_type,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        # ------------------------------------------------
        # SPOT ÚT (eredeti logika megőrizve)
        # ------------------------------------------------
        if is_futures and trading_mode == "spot":
            print(
                "[WARN] place_order called with is_futures=True "
                "but config.trading_mode='spot'. Using SPOT client."
            )

        return self._place_spot_order(
            symbol=symbol,
            side=side,
            size=size,
            order_type=order_type,
            price=price,
        )

    # --------------------------------------------------
    # PRIVÁT SEGÉDMETÓDUS – SPOT
    # --------------------------------------------------
    def _place_spot_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str,
        price: Optional[float] = None,
    ) -> bool:
        """Spot megbízás küldése a TESTNET spot kliensre."""
        if self.trade_client is None:
            print("[ERROR] Spot trade client nem elérhető.")
            return False

        try:
            if order_type == "market":
                order = self.trade_client.create_order(
                    symbol=symbol,
                    type="market",
                    side=side,
                    amount=size,
                )
            else:
                order = self.trade_client.create_order(
                    symbol=symbol,
                    type="limit",
                    side=side,
                    amount=size,
                    price=price,
                )

            self.last_order = order
            print(f"[LIVE-SPOT] Order elküldve: {order}")
            return True

        except Exception as e:
            print(f"[ERROR] Spot order sikertelen: {type(e).__name__}: {e}")
            return False

    # --------------------------------------------------
    # PRIVÁT SEGÉDMETÓDUS – FUTURES
    # --------------------------------------------------
    def _place_futures_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """
        Teljes futures megbízás-folyamat:
          1. Kliens ellenőrzés
          2. Leverage beállítás
          3. Margin type beállítás
          4. Főmegbízás küldése
          5. Stop-Loss megbízás küldése
          6. Take-Profit megbízás küldése
        """

        # 1) Kliens ellenőrzés
        client = self.futures_trade_client
        if client is None:
            print("[ERROR] Futures trade client nem elérhető.")
            return False

        # 2) Leverage meghatározása
        leverage: int = self.config.symbol_leverage.get(
            symbol,
            self.config.default_leverage,
        )

        # 3) Margin type lekérése
        margin_type: str = self.config.margin_type.upper()  # "ISOLATED" vagy "CROSSED"

        # ------- LEVERAGE BEÁLLÍTÁS -------
        leverage_ok = self.safe_fetch(
            client.set_leverage,
            leverage,
            symbol,
            fallback=None,
        )
        if leverage_ok is None:
            print(f"[WARN] Leverage beállítás sikertelen: {symbol} → {leverage}x")
            # Nem állítjuk le – folytatjuk a default leverage-dzsel

        print(f"[FUTURES] Leverage beállítva: {symbol} → {leverage}x")

        # ------- MARGIN TYPE BEÁLLÍTÁS -------
        # A Binance API hibát dob, ha már be van állítva ugyanaz a mód.
        # Ezért elkapjuk és figyelmen kívül hagyjuk azt a specifikus hibát.
        try:
            client.set_margin_mode(
                marginMode=margin_type,
                symbol=symbol,
            )
            print(f"[FUTURES] Margin mód beállítva: {symbol} → {margin_type}")
        except Exception as e:
            error_msg = str(e).lower()
            # Binance kód: "No need to change margin type" → nem hiba
            if "no need" in error_msg or "already" in error_msg or "-4046" in error_msg:
                print(f"[FUTURES] Margin mód már be van állítva ({margin_type}), folytatás.")
            else:
                print(f"[WARN] Margin type beállítás sikertelen: {type(e).__name__}: {e}")
                # Nem kritikus – folytatjuk

        # ------- ELLENTÉTES OLDAL MEGHATÁROZÁSA (SL/TP-hez) -------
        # Ha long pozíciót nyitunk (buy), akkor az SL/TP "sell" oldal és fordítva
        opposite_side: str = "sell" if side.lower() == "buy" else "buy"

        # ------- POZÍCIÓ MÓD PARAMS -------
        # "one_way" módban nincs szükség positionSide megadására
        # "hedge" módban kötelező: "LONG" vagy "SHORT"
        position_mode = getattr(self.config, "futures_position_mode", "one_way")

        base_params: Dict[str, Any] = {}
        if position_mode == "hedge":
            base_params["positionSide"] = "LONG" if side.lower() == "buy" else "SHORT"

        # ------- FŐ MEGBÍZÁS KÜLDÉSE -------
        try:
            if order_type == "market":
                main_order = client.create_order(
                    symbol=symbol,
                    type="market",
                    side=side,
                    amount=size,
                    params=base_params,
                )
            else:
                main_order = client.create_order(
                    symbol=symbol,
                    type="limit",
                    side=side,
                    amount=size,
                    price=price,
                    params={**base_params, "timeInForce": "GTC"},
                )

            self.last_order = main_order
            print(f"[FUTURES] Főmegbízás elküldve: {main_order}")

        except Exception as e:
            print(f"[ERROR] Futures főmegbízás sikertelen: {type(e).__name__}: {e}")
            return False  # Ha a főmegbízás sikertelen, ne küldjük az SL/TP-t sem

        # ------- STOP-LOSS MEGBÍZÁS -------
        if stop_loss is not None:
            sl_type: str = getattr(
                self.config, "futures_sl_order_type", "STOP_MARKET"
            )
            sl_params: Dict[str, Any] = {
                **base_params,
                "stopPrice": stop_loss,
                "reduceOnly": True,        # Csak pozíció zárásra, nem nyitásra!
                "workingType": "MARK_PRICE",  # Mark price alapú trigger (stabilabb)
            }

            sl_ok = self.safe_fetch(
                client.create_order,
                symbol,
                sl_type,          # "STOP_MARKET" vagy "STOP"
                opposite_side,
                size,
                fallback=None,
                params=sl_params,
            )

            if sl_ok is None:
                print(f"[WARN] Stop-Loss megbízás sikertelen: {symbol} SL={stop_loss}")
            else:
                print(f"[FUTURES] Stop-Loss beállítva: {symbol} → {stop_loss}")

        # ------- TAKE-PROFIT MEGBÍZÁS -------
        if take_profit is not None:
            tp_type: str = getattr(
                self.config, "futures_tp_order_type", "TAKE_PROFIT_MARKET"
            )
            tp_params: Dict[str, Any] = {
                **base_params,
                "stopPrice": take_profit,
                "reduceOnly": True,        # Csak pozíció zárásra!
                "workingType": "MARK_PRICE",
            }

            tp_ok = self.safe_fetch(
                client.create_order,
                symbol,
                tp_type,          # "TAKE_PROFIT_MARKET" vagy "TAKE_PROFIT"
                opposite_side,
                size,
                fallback=None,
                params=tp_params,
            )

            if tp_ok is None:
                print(f"[WARN] Take-Profit megbízás sikertelen: {symbol} TP={take_profit}")
            else:
                print(f"[FUTURES] Take-Profit beállítva: {symbol} → {take_profit}")

        return True

        # PAPER MODE: no real orders, just log.
        if self.state.paper_mode or self.trade_client is None:
            print(
                f"[PAPER] {order_type.upper()} {side.upper()} {symbol} "
                f"size={size}, price={price}, SL={stop_loss}, TP={take_profit}, "
                f"futures={is_futures}"
            )
            return True

        # LIVE TESTNET ORDER (SPOT CLIENT ONLY FOR NOW)
        trading_mode = getattr(self.config, "trading_mode", "spot")
        if is_futures and trading_mode == "spot":
            # Explicitly warn about mismatch – behavior stays the same (spot
            # client).
            print(
                "[WARN] place_order called with is_futures=True but config.trading_mode='spot'. "
                "Using SPOT TESTNET trade_client. Futures client is prepared but not yet wired."
            )

        try:
            params = {}

            # Choose client based on mode and availability
            client = self.trade_client
            if (
                is_futures
                and trading_mode in ("futures", "mixed")
                and self.futures_trade_client is not None
            ):
                client = self.futures_trade_client

            if client is None:
                print("[ERROR] No trade client available")
                return False

            if order_type == "market":
                order = client.create_order(
                    symbol=symbol,
                    type="market",
                    side=side,
                    amount=size,
                    params=params,
                )
            else:
                order = client.create_order(
                    symbol=symbol,
                    type="limit",
                    side=side,
                    amount=size,
                    price=price,
                    params=params,
                )
            # Store last order for debugging / analysis
            self.last_order = order

            print(f"[LIVE-TESTNET] Order placed: {order}")
            return True

        except Exception as e:
            print(f"[ERROR] Live order failed: {e}")
            return False


# ============================
# 4) SIGNAL & MARKET LOGIC
# ============================

warnings.filterwarnings('ignore')


try:
    pass
    import pandas as pd
except ImportError:
    pass
    print("WARNING: pandas not installed. Install with: pip install pandas")
    pd = None


class SignalEngine:
    def __init__(self, config: BotConfig, exch: Any):
        self.config = config
        self.exch = exch
        self.regime_engine = MarketRegimeEngine(config, exch)
        self.ensemble = EnsembleAllocator(config)
        self.strategy_modules = build_strategies(config, exch, self)
        self.research_context_provider = None
        self.learning_context_provider = None
        self.frequency_context_provider = None
        # Phase 1: Multi-strategy weights (dynamic later via ML)
        self.strategy_weights = {
            'trend_breakout': 0.35,
            'mean_reversion': 0.30,
            'momentum_scalp': 0.25,
            'regime_filter': 0.10  # Veto power
        }
        # Recent performance tracking (updates after trades)
        self.strategy_performance = {
            'trend_breakout': 1.0,
            'mean_reversion': 1.0,
            'momentum_scalp': 1.0
        }
        self.last_generation_diagnostics: Dict[str, Dict[str, Any]] = {}

    def _store_generation_diagnostics(self, symbol: str, payload: Dict[str, Any]) -> None:
        self.last_generation_diagnostics[symbol] = {
            "symbol": symbol,
            **payload,
        }

    def compute_hurst_exponent(self, symbol: str, timeframe: str = "1h", lookback: int = 100) -> float:
        """
        Hurst exponent becslése Detrended Fluctuation Analysis (DFA) módszerrel.
        """
        candles = self.exch.fetch_ohlcv(symbol, timeframe, limit=lookback * 2) or []
        if len(candles) < lookback:
            return 0.5
        
        prices = np.array([float(c[4]) for c in candles[-lookback:]])
        log_prices = np.log(prices)
        y = log_prices - np.mean(log_prices)
        n = len(y)
        
        min_window = max(10, n // 20)
        max_window = n // 4
        if max_window <= min_window:
            return 0.5
        
        window_sizes = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), num=15, dtype=int))
        fluctuations = []
        
        for w in window_sizes:
            if w < 2 or w >= n:
                continue
            n_segments = n // w
            if n_segments < 2:
                continue
            
            f2 = 0.0
            for i in range(n_segments):
                start = i * w
                end = start + w
                segment = y[start:end]
                x = np.arange(w)
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                detrended = segment - trend
                f2 += np.sum(detrended ** 2)
            
            f2 /= (n_segments * w)
            fluctuations.append(np.sqrt(f2))
        
        if len(fluctuations) < 2:
            return 0.5
        
        log_w = np.log(window_sizes[:len(fluctuations)])
        log_f = np.log(fluctuations)
        H, _ = np.polyfit(log_w, log_f, 1)
        return np.clip(H, 0.0, 1.0)

    def get_market_regime(self, symbol: str) -> str:
        return self.regime_engine.classify(symbol).regime

    def is_choppy_market(self, symbol: str) -> bool:
        "Enhanced chop filter"
        regime = self.get_market_regime(symbol)
        return regime == 'choppy'

    def compute_atr(
        self,
        symbol: str,
        timeframe: str = "15m",
        period: int = 14,
    ) -> Optional[float]:
        """
        Compute a simple ATR-like volatility measure.
        Returns:
            float ATR value, or None if not enough data.
        """
        candles = self.exch.fetch_ohlcv(symbol, timeframe, limit=period + 1)
        if not candles:
            print(
    f"[WARN] compute_atr: no candles for {symbol} on {timeframe}.")
            return None

        if len(candles) < period + 1:
            print(
                f"[WARN] compute_atr: not enough candles for {symbol} on {timeframe} "
                f"(need {period + 1}, got {len(candles)})."
            )
            return None

        trs = []
        prev_close = candles[0][4]

        for c in candles[1:]:
            high = c[2]
            low = c[3]
            close = c[4]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            trs.append(tr)
            prev_close = close

        if not trs:
            return None

        atr = sum(trs) / len(trs)
        return atr

    def _has_higher_highs_lows(self, candles: list, lookback: int = 6) -> bool:
        """
        Check if price structure shows Higher Highs and Higher Lows (uptrend).
        - More lenient: looks for general uptrend direction instead of strict HH/HL
        - Returns True if recent average > earlier average (trend confirmation)
        """
        if not candles or len(candles) < lookback:
            return False

        recent_candles = candles[-lookback:]
        earlier_candles = candles[-(lookback * 2):-lookback]

        if not earlier_candles:
            return False

        # Calculate midpoint (close price as proxy for structure)
        recent_mid = sum(c[4] for c in recent_candles) / len(recent_candles)
        earlier_mid = sum(c[4] for c in earlier_candles) / len(earlier_candles)

        # Uptrend if recent average is higher than earlier
        # Allow 0.1% tolerance to avoid rounding issues
        return recent_mid > earlier_mid * 1.001

    def is_4h_bullish(self, symbol: str) -> bool:
        """
        4H TIMEFRAME FILTER (Structure Confirmation)
        - Must show clear uptrend: Higher Highs AND Higher Lows
        - Looks at 12 candles (6 recent vs 6 earlier)
        - Very strict - only enters when 4H structure is proven bullish
        """
        candles_4h = self.exch.fetch_ohlcv(symbol, "4h", limit=20)
        if not candles_4h or len(candles_4h) < 12:
            return False

        return self._has_higher_highs_lows(candles_4h, lookback=6)

    def is_4h_bearish(self, symbol: str) -> bool:
        candles_4h = self.exch.fetch_ohlcv(symbol, "4h", limit=20)
        if not candles_4h or len(candles_4h) < 12:
            return False
        closes = [c[4] for c in candles_4h]
        recent_mid = sum(closes[-6:]) / 6
        earlier_mid = sum(closes[-12:-6]) / 6
        return recent_mid < earlier_mid * 0.999

    def is_1h_uptrend(self, symbol: str) -> bool:
        """
        1H TIMEFRAME FILTER (Momentum Confirmation)
        - Must show uptrend: Higher Highs AND Higher Lows
        - Price must be above 20-period SMA (additional confirmation)
        - Looks at 8 candles (4 recent vs 4 earlier)
        """
        candles_1h = self.exch.fetch_ohlcv(symbol, "1h", limit=50)
        if not candles_1h or len(candles_1h) < 20:
            return False

        # Check HH/HL structure
        if not self._has_higher_highs_lows(candles_1h, lookback=4):
            return False

        # Price above 20 SMA
        closes_1h = [c[4] for c in candles_1h]
        last_close = closes_1h[-1]
        sma_20 = sum(closes_1h[-20:]) / 20

        return last_close > sma_20

    def is_1h_downtrend(self, symbol: str) -> bool:
        candles_1h = self.exch.fetch_ohlcv(symbol, "1h", limit=50)
        if not candles_1h or len(candles_1h) < 20:
            return False
        closes_1h = [c[4] for c in candles_1h]
        recent_mid = sum(closes_1h[-4:]) / 4
        earlier_mid = sum(closes_1h[-8:-4]) / 4
        sma_20 = sum(closes_1h[-20:]) / 20
        last_close = closes_1h[-1]
        return recent_mid < earlier_mid * 0.999 and last_close < sma_20

    def get_recent_swing_high_low(
    self,
    candles: list,
     lookback: int = 5) -> tuple:
        """
        Find recent swing high and low for breakout entry.
        - Looks at last N candles
        - Returns (swing_high, swing_low)
        """
        if not candles or len(candles) < lookback:
            return None, None

        recent = candles[-lookback:]
        swing_high = max(c[2] for c in recent)
        swing_low = min(c[3] for c in recent)

        return swing_high, swing_low

    def is_atr_spike(
        self,
        symbol: str,
        timeframe: str = "15m",
        period: int = 14,
        spike_mult: float = 3.0,
    ) -> bool:
        """
        Detect extreme volatility spike:
        - Compute ATR-like series and compare last ATR to previous average.
        - If last ATR >= spike_mult * previous ATR average -> spike.
        """
        candles = self.exch.fetch_ohlcv(symbol, timeframe, limit=period + 5)
        if not candles:
            print(
    f"[WARN] is_atr_spike: no candles for {symbol} on {timeframe}.")
            return False

        if len(candles) < period + 2:
            return False

        trs = []
        prev_close = candles[0][4]

        for c in candles[1:]:
            high = c[2]
            low = c[3]
            close = c[4]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            trs.append(tr)
            prev_close = close

        if len(trs) < period + 1:
            return False

        last_atr = sum(trs[-period:]) / period
        prev_atr = sum(trs[-(period + 1): -1]) / period

        if prev_atr <= 0:
            return False

        return last_atr >= prev_atr * spike_mult

    def mean_reversion_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Mean reversion strategy - RSI extremes + BB"""
        candles = self.exch.fetch_ohlcv(symbol, '15m', limit=100)
        if len(candles) < 50:
            return None

        closes = np.array([c[4] for c in candles])
        rsi = self.rsi(closes, 14)
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(closes, 20, 2)

        last_close = closes[-1]
        last_rsi = rsi[-1]
        atr_15m = self.compute_atr(symbol, "15m", period=14)
        if atr_15m is None or atr_15m <= 0:
            return None

        if last_rsi < 25 and last_close <= bb_lower[-1] * 1.001:  # Long
            sl = min(bb_lower[-1] * 0.998, last_close - atr_15m * 0.8)
            tp = bb_middle[-1]
            if sl <= 0 or sl >= last_close or tp <= last_close:
                return None
            return {
                'side': 'long',
                'entry_price': last_close,
                'stop_loss': sl,
                'take_profit': tp,
                'strategy': 'mean_reversion',
                'fast_move': False,
                'is_futures': False,
                'signal_quality': 0.62,
            }
        elif last_rsi > 75 and last_close >= bb_upper[-1] * 0.999:  # Short
            sl = max(bb_upper[-1] * 1.002, last_close + atr_15m * 0.8)
            tp = bb_middle[-1]
            if sl <= last_close or tp >= last_close:
                return None
            return {
                'side': 'short',
                'entry_price': last_close,
                'stop_loss': sl,
                'take_profit': tp,
                'strategy': 'mean_reversion',
                'fast_move': False,
                'is_futures': False,
                'signal_quality': 0.62,
            }
        return None

    def trend_breakout_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        # Trend-friendly setup with slightly looser filters than the original
        # all-or-nothing breakout path.
        if not self.is_4h_bullish(symbol):
            return None
        if not self.is_1h_uptrend(symbol):
            return None

        candles_15m = self.exch.fetch_ohlcv(symbol, "15m", limit=30)
        lookback = max(int(getattr(self.config, "breakout_swing_lookback", 5)), 3)
        if not candles_15m or len(candles_15m) < max(lookback, 10):
            return None

        swing_high, swing_low = self.get_recent_swing_high_low(candles_15m, lookback=lookback)
        if swing_high is None or swing_low is None:
            return None

        last_close = candles_15m[-1][4]
        atr_15m = self.compute_atr(symbol, "15m", period=14)
        if atr_15m is None or atr_15m <= 0:
            return None

        breakout_buffer = atr_15m * 0.15
        if last_close + breakout_buffer < swing_high:
            return None

        entry_price = max(last_close, swing_high)
        stop_loss = min(swing_low * 0.999, entry_price - atr_15m * 0.9)
        if stop_loss <= 0 or stop_loss >= entry_price:
            return None

        sl_distance = entry_price - stop_loss
        if sl_distance < 0.25 * atr_15m or sl_distance > 2.5 * atr_15m:
            return None

        rr_ratio = max(float(getattr(self.config, "breakout_rr_ratio", 2.2)), 1.5)
        take_profit = entry_price + (sl_distance * rr_ratio)
        if take_profit <= entry_price:
            return None

        tp_pct = (take_profit - entry_price) / entry_price
        if tp_pct > 0.12:
            take_profit = entry_price * 1.12

        quality = 0.72 if last_close >= swing_high else 0.58
        return {
            "side": "long",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "fast_move": last_close > swing_high,
            "is_futures": False,
            "confluence_score": "MEDIUM",
            "strategy": "trend_breakout",
            "signal_quality": quality,
        }

    def rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        if np is None:
            return np.array([50.0] * len(prices))
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period) / period, mode='valid')
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate((np.full(period - 1, 50.0), rsi))

    def bollinger_bands(
    self,
    prices: np.ndarray,
    period: int = 20,
     std_mult: float = 2.0) -> tuple:
        sma = np.convolve(prices, np.ones(period) / period, mode='valid')
        std = np.array([np.std(prices[i:i + period])
                       for i in range(len(prices) - period + 1)])
        upper = sma + std * std_mult
        lower = sma - std * std_mult
        return np.concatenate((np.full(period -
    1, prices[:period -
    1]), upper)), sma, np.concatenate((np.full(period -
    1, prices[:period -
     1]), lower))

    def generate_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        regime = self.regime_engine.classify(symbol)
        proposals = []
        hurst = self.compute_hurst_exponent(symbol, timeframe=self.config.timeframes.get("trend", "1h"), lookback=80)
        research_context = self._research_context(symbol)
        for strategy in self.strategy_modules:
            proposal = strategy.evaluate(symbol, regime)
            if proposal is not None:
                signal_stub = {
                    "symbol": proposal.signal.symbol,
                    "side": proposal.signal.side,
                    "entry_price": proposal.signal.entry_price,
                    "stop_loss": proposal.signal.stop_loss,
                    "take_profit": proposal.signal.take_profit,
                    "strategy": proposal.signal.strategy,
                    "signal_quality": proposal.signal.confidence,
                    "expected_edge_bps": proposal.expected_edge_bps,
                    "regime": proposal.signal.regime,
                    "rr_ratio": self._reward_risk_ratio_for_signal(proposal.signal),
                    "hurst_exponent": hurst,
                    "fast_move": proposal.signal.fast_move,
                    "metadata": {
                        **dict(proposal.signal.metadata),
                        **self._microstructure_context(symbol, proposal.signal.entry_price),
                    },
                    "research_context": research_context,
                }
                learning_context = self._learning_context(symbol, signal_stub, regime)
                proposal.signal.metadata["hurst_exponent"] = hurst
                proposal.signal.metadata["research_context"] = research_context
                proposal.signal.metadata.update(self._microstructure_context(symbol, proposal.signal.entry_price))
                proposal.signal.metadata["learning_context"] = learning_context
                proposal.signal.confidence = max(
                    0.0,
                    min(0.99, float(proposal.signal.confidence) + float(learning_context.get("confidence_delta", 0.0) or 0.0)),
                )
                proposals.append(proposal)

        proposal_count = len(proposals)
        if proposal_count <= 0:
            self._store_generation_diagnostics(
                symbol,
                {
                    "outcome": "no_proposal",
                    "reason": "strategy_produced_no_proposal",
                    "proposal_count": 0,
                    "proposal_strategies": [],
                    "regime": regime.regime,
                },
            )
            return None

        decision = self.ensemble.choose(
            symbol,
            regime,
            proposals,
            research_context=research_context,
            frequency_context=self._frequency_context(symbol, regime),
        )
        frequency_adjustments = [
            {
                "strategy": str(item.get("strategy", "unknown")),
                "reason": str(item.get("frequency_reason", "inactive")),
                "score_delta": float(item.get("frequency_score_delta", 0.0) or 0.0),
                "min_net_expectancy_delta": float(item.get("frequency_min_net_expectancy_delta", 0.0) or 0.0),
            }
            for item in list(decision.proposals or [])
            if str(item.get("frequency_reason", "inactive")) not in {"inactive", "neutral"}
        ]
        if decision.signal is None:
            self._store_generation_diagnostics(
                symbol,
                {
                    "outcome": "ensemble_rejected",
                    "reason": "no_ensemble_selection",
                    "proposal_count": proposal_count,
                    "proposal_strategies": [proposal.signal.strategy for proposal in proposals],
                    "ensemble_rejected_reasons": list(decision.rejected_reasons or []),
                    "frequency_adjustments": frequency_adjustments,
                    "regime": regime.regime,
                },
            )
            return None
        signal = decision.signal
        if signal.strategy == "trend_breakout" and hurst < float(getattr(self.config, "min_hurst_for_trend_breakout", 0.12)):
            self._store_generation_diagnostics(
                symbol,
                {
                    "outcome": "reliability_rejected",
                    "reason": "hurst_breakout_blocked",
                    "proposal_count": proposal_count,
                    "proposal_strategies": [proposal.signal.strategy for proposal in proposals],
                    "selected_strategy": signal.strategy,
                    "ensemble_rejected_reasons": list(decision.rejected_reasons or []),
                    "frequency_adjustments": frequency_adjustments,
                    "regime": regime.regime,
                },
            )
            return None
        if signal.strategy == "trend_pullback" and hurst < float(getattr(self.config, "min_hurst_for_trend_pullback", 0.10)):
            self._store_generation_diagnostics(
                symbol,
                {
                    "outcome": "reliability_rejected",
                    "reason": "hurst_pullback_blocked",
                    "proposal_count": proposal_count,
                    "proposal_strategies": [proposal.signal.strategy for proposal in proposals],
                    "selected_strategy": signal.strategy,
                    "ensemble_rejected_reasons": list(decision.rejected_reasons or []),
                    "frequency_adjustments": frequency_adjustments,
                    "regime": regime.regime,
                },
            )
            return None
        if signal.strategy == "mean_reversion" and hurst > float(getattr(self.config, "max_hurst_for_mean_reversion", 0.65)):
            self._store_generation_diagnostics(
                symbol,
                {
                    "outcome": "reliability_rejected",
                    "reason": "hurst_mean_reversion_blocked",
                    "proposal_count": proposal_count,
                    "proposal_strategies": [proposal.signal.strategy for proposal in proposals],
                    "selected_strategy": signal.strategy,
                    "ensemble_rejected_reasons": list(decision.rejected_reasons or []),
                    "frequency_adjustments": frequency_adjustments,
                    "regime": regime.regime,
                },
            )
            return None
        if float(signal.confidence) < float(getattr(self.config, "min_signal_quality_score", 0.55)):
            self._store_generation_diagnostics(
                symbol,
                {
                    "outcome": "reliability_rejected",
                    "reason": "signal_quality_below_threshold",
                    "proposal_count": proposal_count,
                    "proposal_strategies": [proposal.signal.strategy for proposal in proposals],
                    "selected_strategy": signal.strategy,
                    "ensemble_rejected_reasons": list(decision.rejected_reasons or []),
                    "frequency_adjustments": frequency_adjustments,
                    "regime": regime.regime,
                },
            )
            return None
        rr_ratio = 0.0
        risk = abs(float(signal.entry_price) - float(signal.stop_loss))
        reward = abs(float(signal.take_profit) - float(signal.entry_price))
        if risk > 0:
            rr_ratio = reward / risk
        signal_payload = {
            "symbol": symbol,
            "side": signal.side,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "strategy": signal.strategy,
            "fast_move": signal.fast_move,
            "is_futures": signal.is_futures,
            "signal_quality": signal.confidence,
            "expected_edge_bps": signal.expected_edge_bps,
            "expected_holding_minutes": signal.expected_holding_minutes,
            "timeframe": signal.timeframe,
            "regime": signal.regime,
            "rr_ratio": rr_ratio,
            "hurst_exponent": hurst,
            "metadata": signal.metadata,
            "research_context": research_context,
            "ensemble": {
                "selected_strategy": decision.selected_strategy,
                "proposals": decision.proposals,
                "rejected_reasons": decision.rejected_reasons,
                "regime": decision.regime,
            },
        }
        reliability_reason = self._reliability_rejection_reason(symbol, signal_payload, regime, len(proposals))
        if reliability_reason is not None:
            self._store_generation_diagnostics(
                symbol,
                {
                    "outcome": "reliability_rejected",
                    "reason": reliability_reason,
                    "proposal_count": proposal_count,
                    "proposal_strategies": [proposal.signal.strategy for proposal in proposals],
                    "selected_strategy": signal.strategy,
                    "ensemble_rejected_reasons": list(decision.rejected_reasons or []),
                    "frequency_adjustments": frequency_adjustments,
                    "regime": regime.regime,
                },
            )
            return None
        self._store_generation_diagnostics(
            symbol,
            {
                "outcome": "selected",
                "reason": "selected_for_submission",
                "proposal_count": proposal_count,
                "proposal_strategies": [proposal.signal.strategy for proposal in proposals],
                "selected_strategy": signal.strategy,
                "ensemble_rejected_reasons": list(decision.rejected_reasons or []),
                "frequency_adjustments": frequency_adjustments,
                "regime": regime.regime,
            },
        )
        return signal_payload

    def _passes_reliability_checks(
        self,
        symbol: str,
        signal: Dict[str, Any],
        regime: Any,
        proposal_count: int,
    ) -> bool:
        return SignalEngine._reliability_rejection_reason(self, symbol, signal, regime, proposal_count) is None

    def _reliability_rejection_reason(
        self,
        symbol: str,
        signal: Dict[str, Any],
        regime: Any,
        proposal_count: int,
    ) -> Optional[str]:
        side = str(signal.get("side", "")).lower()
        strategy = str(signal.get("strategy", "unknown"))
        entry_price = float(signal.get("entry_price", 0.0) or 0.0)
        stop_loss = float(signal.get("stop_loss", 0.0) or 0.0)
        take_profit = float(signal.get("take_profit", 0.0) or 0.0)
        confidence = float(signal.get("signal_quality", 0.0) or 0.0)
        expected_edge_bps = float(signal.get("expected_edge_bps", 0.0) or 0.0)
        rr_ratio = float(signal.get("rr_ratio", 0.0) or 0.0)
        hurst = float(signal.get("hurst_exponent", 0.5) or 0.5)
        numeric_values = [entry_price, stop_loss, take_profit, confidence, expected_edge_bps, rr_ratio, hurst]
        research_context = signal.get("research_context", {}) or {}
        metadata = dict(signal.get("metadata", {}) or {})
        learning_context = dict(metadata.get("learning_context", {}) or {})
        calibration = dict(learning_context.get("calibration", {}) or {})
        opportunity = dict(learning_context.get("opportunity", {}) or {})
        positive_cell_evidence = bool(learning_context.get("positive_cell_evidence", False))
        order_profile = str(metadata.get("preferred_order_type", "market" if bool(signal.get("fast_move", False)) else "limit")).lower()
        rotation_policy = dict(getattr(regime, "metadata", {}).get("rotation_policy", {}) or {})
        preferred_family = str(rotation_policy.get("preferred_family", getattr(regime, "metadata", {}).get("preferred_family", "")) or "")
        suppressed_family = str(rotation_policy.get("suppressed_family", getattr(regime, "metadata", {}).get("suppressed_family", "")) or "")
        rotation_confidence = float(rotation_policy.get("confidence", getattr(regime, "metadata", {}).get("rotation_confidence", 0.0)) or 0.0)
        quality_delta = float(getattr(self.config, "rotation_reliability_quality_delta", 0.03))
        edge_delta_bps = float(getattr(self.config, "rotation_reliability_edge_delta_bps", 1.5))
        rr_delta = float(getattr(self.config, "rotation_reliability_rr_delta", 0.06))
        quality_floor = float(getattr(self.config, "min_signal_quality_score", 0.55))
        edge_floor = float(getattr(self.config, "min_expected_edge_bps", 8.0))
        trend_rr_floor = float(getattr(self.config, "min_reliable_rr_ratio_trend", 1.35))
        mean_reversion_rr_floor = float(getattr(self.config, "min_reliable_rr_ratio_mean_reversion", 1.10))
        if strategy == "trend_pullback":
            quality_floor = max(quality_floor, float(getattr(self.config, "min_signal_quality_score_pullback", 0.62)))
            edge_floor = max(edge_floor, float(getattr(self.config, "min_expected_edge_bps_pullback", 14.0)))
            trend_rr_floor = max(trend_rr_floor, float(getattr(self.config, "min_reliable_rr_ratio_pullback", 1.55)))
        if rotation_confidence > 0.0:
            if strategy == preferred_family:
                quality_floor = max(quality_floor - (quality_delta * rotation_confidence), 0.0)
                edge_floor = max(edge_floor - (edge_delta_bps * rotation_confidence), 0.0)
                if strategy in {"trend_breakout", "trend_pullback"}:
                    trend_rr_floor = max(trend_rr_floor - (rr_delta * rotation_confidence), 0.0)
                elif strategy == "mean_reversion":
                    mean_reversion_rr_floor = max(mean_reversion_rr_floor - (rr_delta * rotation_confidence), 0.0)
                if strategy == "trend_pullback":
                    quality_floor = max(
                        quality_floor - (float(getattr(self.config, "rotation_pullback_reliability_quality_delta", 0.06)) * rotation_confidence),
                        0.0,
                    )
                    edge_floor = max(
                        edge_floor - (float(getattr(self.config, "rotation_pullback_reliability_edge_delta_bps", 6.0)) * rotation_confidence),
                        0.0,
                    )
                    trend_rr_floor = max(
                        trend_rr_floor - (float(getattr(self.config, "rotation_pullback_reliability_rr_delta", 0.20)) * rotation_confidence),
                        0.0,
                    )
            elif strategy == suppressed_family:
                quality_floor += quality_delta * rotation_confidence
                edge_floor += edge_delta_bps * rotation_confidence
                if strategy in {"trend_breakout", "trend_pullback"}:
                    trend_rr_floor += rr_delta * rotation_confidence
                elif strategy == "mean_reversion":
                    mean_reversion_rr_floor += rr_delta * rotation_confidence

        if proposal_count <= 0:
            return "no_proposals"
        if any(not math.isfinite(value) for value in numeric_values):
            return "non_finite_signal"
        if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            return "invalid_price_triplet"
        if confidence < quality_floor:
            return "signal_quality_below_threshold"
        if float(getattr(regime, "confidence", 0.0)) < float(getattr(self.config, "min_reliable_regime_confidence", 0.58)):
            return "regime_confidence_too_low"
        if expected_edge_bps < edge_floor:
            return "expected_edge_below_threshold"
        if bool(learning_context.get("veto", False)):
            return "learning_veto"
        if (
            float(calibration.get("effective_samples", 0.0) or 0.0) >= float(getattr(self.config, "learning_calibration_gate_min_samples", 6.0))
            and float(calibration.get("calibrated_confidence", confidence) or confidence)
            < (
                float(getattr(self.config, "learning_min_calibrated_confidence", 0.50))
                - (0.03 if positive_cell_evidence else 0.0)
            )
        ):
            return "calibrated_confidence_too_low"
        if (
            float(opportunity.get("samples", 0.0) or 0.0) >= float(getattr(self.config, "learning_negative_opportunity_gate_min_samples", 10.0))
            and float(opportunity.get("avg_forward_r", 0.0) or 0.0) <= float(getattr(self.config, "learning_negative_opportunity_gate_avg_forward_r", -0.20))
            and not positive_cell_evidence
        ):
            return "negative_opportunity_history"

        if side in {"long", "buy"}:
            if not (stop_loss < entry_price < take_profit):
                return "invalid_long_ordering"
        elif side in {"short", "sell"}:
            if not (stop_loss > entry_price > take_profit):
                return "invalid_short_ordering"
        else:
            return "invalid_side"

        if not self._passes_research_checks(side, research_context):
            return "research_conflict"

        if strategy in {"trend_breakout", "trend_pullback"}:
            if rr_ratio < trend_rr_floor:
                return "rr_ratio_too_low_trend"
        elif strategy == "mean_reversion":
            if rr_ratio < mean_reversion_rr_floor:
                return "rr_ratio_too_low_mean_reversion"

        regime_name = str(getattr(regime, "regime", "unknown"))
        if strategy in {"trend_breakout", "trend_pullback"} and regime_name not in {"trending", "high_volatility"}:
            return "regime_mismatch_trend"
        if strategy == "mean_reversion" and regime_name not in {"mean_reverting", "choppy"}:
            return "regime_mismatch_mean_reversion"

        try:
            order_book = self.exch.get_order_book(symbol)
            bid = float(order_book["bid"])
            ask = float(order_book["ask"])
            mid = (bid + ask) / 2.0
            if mid > 0:
                entry_deviation = abs(entry_price - mid) / mid
                max_entry_deviation = float(getattr(self.config, "max_entry_deviation_from_mid_fraction", 0.0045))
                if order_profile == "limit":
                    max_entry_deviation *= 1.20
                if strategy == "trend_pullback":
                    max_entry_deviation *= 1.40
                elif strategy == "mean_reversion":
                    max_entry_deviation *= 1.60
                elif strategy == "trend_breakout" and order_profile == "limit":
                    max_entry_deviation *= 1.15
                max_entry_deviation += min(max(expected_edge_bps - 18.0, 0.0) / 10000.0, 0.0015)
                if entry_deviation > max_entry_deviation:
                    return "entry_too_far_from_mid"
        except Exception:
            pass
        return None

    def _research_context(self, symbol: str) -> Dict[str, Any]:
        if not callable(self.research_context_provider):
            return {}
        try:
            return self.research_context_provider(symbol) or {}
        except Exception:
            return {}

    def _learning_context(self, symbol: str, signal: Dict[str, Any], regime: Any | None = None) -> Dict[str, Any]:
        if not callable(self.learning_context_provider):
            return {}
        try:
            return self.learning_context_provider(symbol, signal, regime) or {}
        except Exception:
            return {}

    def _frequency_context(self, symbol: str, regime: Any | None = None) -> Dict[str, Any]:
        if not callable(self.frequency_context_provider):
            return {}
        try:
            return self.frequency_context_provider(symbol, regime) or {}
        except Exception:
            return {}

    def _passes_research_checks(self, side: str, research_context: Dict[str, Any]) -> bool:
        if not research_context:
            return True
        risk_off = float(research_context.get("risk_off_confidence", 0.0) or 0.0)
        bullish = float(research_context.get("bullish_confidence", 0.0) or 0.0)
        bearish = float(research_context.get("bearish_confidence", 0.0) or 0.0)
        if risk_off >= float(getattr(self.config, "research_risk_off_veto_confidence", 0.75)):
            return False
        if side in {"long", "buy"} and bearish >= float(getattr(self.config, "research_conflict_veto_confidence", 0.72)):
            return False
        if side in {"short", "sell"} and bullish >= float(getattr(self.config, "research_conflict_veto_confidence", 0.72)):
            return False
        return True

    def _microstructure_context(self, symbol: str, entry_price: float) -> Dict[str, Any]:
        try:
            order_book = self.exch.get_order_book(symbol)
            bid = float(order_book["bid"])
            ask = float(order_book["ask"])
            mid = (bid + ask) / 2.0
            if mid <= 0:
                return {}
            spread_bps = ((ask - bid) / mid) * 10000.0
            entry_deviation_bps = abs(float(entry_price or 0.0) - mid) / mid * 10000.0
            return {
                "spread_bps": spread_bps,
                "entry_deviation_bps": entry_deviation_bps,
            }
        except Exception:
            return {}

    @staticmethod
    def _reward_risk_ratio_for_signal(signal: Any) -> float:
        risk = abs(float(signal.entry_price) - float(signal.stop_loss))
        reward = abs(float(signal.take_profit) - float(signal.entry_price))
        if risk <= 0:
            return 0.0
        return reward / risk


# ============================
# 5) RISK MANAGEMENT
# ============================
class RiskManager:
    def __init__(self, config: BotConfig, state: BotState):
        self.config = config
        self.state = state

    # --------- INTERNAL HELPERS (BACKWARDS-COMPATIBLE) ---------
    def _now(self) -> dt.datetime:
        return dt.datetime.now()

    def _count_open_positions(self) -> int:
        if not self.state.open_positions:
            return 0
        total = 0
        for v in self.state.open_positions.values():
            if isinstance(v, list):
                total += len(v)
            else:
                total += 1
        return total

    def _is_in_cooldown(self) -> bool:
        if self.state.cooldown_until is None:
            return False
        return self._now() < self.state.cooldown_until

    # --------- RISK PER TRADE ---------
    def current_risk_fraction(self) -> float:
        # For now:
        #   - Base is risk_per_trade_max.
        #   - If reduced_risk_mode is active -> scale by reduced_risk_factor.
        # NOTE: risk_per_trade_min is not yet actively used to avoid changing
        # behavior.
        base = self.config.risk_per_trade_max
        if self.state.reduced_risk_mode:
            base = base * self.config.reduced_risk_factor

        # Ensure we never go below zero
        if base <= 0:
            return 0.0

        return base

    def can_take_more_trades_today(self) -> bool:
        return self.state.today_trades_count < self.config.max_trades_per_day_max

    def can_open_new_position(self, symbol: str) -> bool:
        """Enhanced: + correlation limits"""
        # Cooldown
        if self._is_in_cooldown():
            return False

        # Max positions
        if self._count_open_positions() >= self.config.max_open_positions:
            return False

        # Daily trades
        if not self.can_take_more_trades_today():
            return False

        # NEW: Correlation family limit (25% risk max per group)
        if not self._check_correlation_limit(symbol):
            print(f"[RISK] Correlation limit hit for {symbol}")
            return False

        return True

    def _check_correlation_limit(self, symbol: str) -> bool:
        "Max 25% portfolio risk per coin family"
        coin = symbol.split('/')[0]
        family_risk = self._get_family_risk(coin)
        return family_risk < 0.25

    def _get_family_risk(self, coin: str) -> float:
        "Calculate risk exposure for coin family"
        families = {
            'BTC': ['BTC'],
            'ETH': ['ETH'],
            'BNB': ['BNB'],
            'SOL': ['SOL'],
            'XRP': ['XRP', 'ADA'],
            'AVAX': ['AVAX', 'DOT'],
            'LINK': ['LINK', 'TON']
        }

        total_risk = 0.0
        for family, coins in families.items():
            if coin in coins:
                for sym_pos in self.state.open_positions.values():
                    if isinstance(sym_pos, list):
                        for pos in sym_pos:
                            if pos.symbol.split('/')[0] in coins:
                                total_risk += self.config.risk_per_trade_max
                    else:
                        pos = sym_pos
                        if getattr(pos, "symbol", "").split('/')[0] in coins:
                            total_risk += self.config.risk_per_trade_max
                break
        return total_risk

    def check_daily_reset(self):
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
            # Clear cooldown at start of a new day
            self.state.cooldown_until = None

    def check_daily_loss_limit(self) -> bool:
        "Returns True if the DAILY LOSS LIMIT is reached or exceeded, meaning trading should be halted for the rest of the day."
        equity_start = self.state.equity_start_of_day
        current_equity = self.state.balance

        realized_loss = current_equity - equity_start
        max_allowed_loss = equity_start * self.config.daily_loss_limit_fraction * -1

        return realized_loss <= max_allowed_loss

    def update_after_trade_result(self, profit_loss: float):
        "Update daily and lifetime stats after a trade closes. Also handles loss-streak based cooldown logic."
        self.state.today_trades_count += 1
        self.state.realized_pl_today += profit_loss

        # Daily win/loss counts
        if profit_loss >= 0:
            self.state.wins_today += 1
        else:
            self.state.losses_today += 1

        # Consecutive loss tracking
        if profit_loss < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        # Lifetime stats (purely informational – no behavior changes)
        self.state.lifetime_trades += 1
        self.state.lifetime_profit += profit_loss
        if profit_loss > self.state.best_single_trade:
            self.state.best_single_trade = profit_loss
        if profit_loss < self.state.worst_single_trade:
            self.state.worst_single_trade = profit_loss

        # Cooldown activation after loss streak
        # Uses: config.consecutive_loss_threshold &
        # config.cooldown_minutes_after_loss_streak
        if (
            profit_loss < 0
            and self.state.consecutive_losses >= self.config.consecutive_loss_threshold
            and self.config.cooldown_minutes_after_loss_streak > 0
        ):
            cooldown_minutes = self.config.cooldown_minutes_after_loss_streak
            self.state.cooldown_until = self._now() + dt.timedelta(minutes=cooldown_minutes)
            # NOTE: logging/notification about cooldown can be added later via
            # Reporter.

    def calc_position_size(
    self,
    entry_price: float,
     stop_loss: float) -> float:
        "Basic position size calculation for spot."
        risk_fraction = self.current_risk_fraction()
        risk_amount = self.state.balance * risk_fraction
        sl_distance = abs(entry_price - stop_loss)

        if sl_distance <= 0 or risk_amount <= 0:
            return 0.0

        return risk_amount / sl_distance


# ============================
# 6) EXECUTION ENGINE
# ============================
from typing import Optional  # ensure this is at top of the file only once


class ExecutionEngine:
    def __init__(self, config: BotConfig, state: BotState,
                 exch: 'ExchangeClient | MockExchange'):
        self.config = config
        self.state = state
        self.exch = exch

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
        if normalized_side == "long":
            return stop_loss < entry_price < take_profit
        if normalized_side == "short":
            return stop_loss > entry_price > take_profit
        if normalized_side == "buy":
            return stop_loss < entry_price < take_profit
        if normalized_side == "sell":
            return stop_loss > entry_price > take_profit
        return False

    # ... [rest of ExecutionEngine methods - truncated for diff]
    # Full methods already in code above, just completing class

class BacktestEngine:
    "Backtesting framework - replays historical data through live signal pipeline."
    # Features:
    # - Realistic fills (slippage, spread, commissions)
    # - Exact signal reproduction
    # - Full metrics (Sharpe, maxDD, Calmar, etc.)


    def __init__(self, config: BotConfig):
        self.config = config
        self.trades: List[Dict] = []
        self.balance_curve: List[float] = []
        self.equity_start = config.starting_balance
        self._total_fees: float = 0.0

    def load_historical_data(
    self,
    symbol: str,
    timeframe: str,
     days: int = 180) -> pd.DataFrame:
        "Load OHLCV from Binance public API"
        import ccxt

        exchange = ccxt.binance({'enableRateLimit': True})
        start_ts = pd.Timestamp.now() - pd.Timedelta(days=days)
        end_ts = pd.Timestamp.now()
        since = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)

        ohlcv: List[List[Any]] = []
        seen_timestamps = set()
        while since < end_ms:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not batch:
                break
            new_rows = [row for row in batch if row[0] not in seen_timestamps]
            if not new_rows:
                break
            ohlcv.extend(new_rows)
            seen_timestamps.update(row[0] for row in new_rows)
            last_ts = int(new_rows[-1][0])
            since = last_ts + 1
            if len(batch) < 1000:
                break

        df = pd.DataFrame(
    ohlcv,
    columns=[
        'timestamp',
        'open',
        'high',
        'low',
        'close',
         'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        print(f"Loaded {len(df)} candles for {symbol} {timeframe} ({days} days)")
        return df

    def run_backtest(
    self,
    symbol: str,
    timeframe: str = '15m',
     days: int = 180) -> Dict:
        "Full backtest pipeline"
        print(f"Starting backtest: {symbol} {timeframe} ({days} days)")

        # Load data
        df = self.load_historical_data(symbol, timeframe, days)
        if df.empty:
            return {'error': 'No data loaded'}

        balance = self.config.starting_balance
        positions: List[Dict[str, Any]] = []
        pending_entries: List[Dict[str, Any]] = []
        self.trades = []
        self.balance_curve = [balance]
        self._total_fees = 0.0

        mock_exch = MockBacktestExchange(df, symbol)
        signal_engine = SignalEngine(self.config, mock_exch)
        warmup = max(int(getattr(self.config, "backtest_warmup_candles", 100)), 20)
        fee_rate = float(getattr(self.config, "backtest_fee_bps", 10.0)) / 10000.0

        for i in range(warmup, len(df)):
            mock_exch.set_cursor(i)
            current_candle = df.iloc[i]
            candle_timestamp = current_candle.name
            active_date = candle_timestamp.date()

            newly_active: List[Dict[str, Any]] = []
            still_pending: List[Dict[str, Any]] = []
            for planned in pending_entries:
                if planned["activate_index"] != i:
                    still_pending.append(planned)
                    continue

                entry_price = self._apply_entry_costs(
                    side=planned["side"],
                    raw_price=float(current_candle["open"]),
                )
                fee_paid = entry_price * planned["size"] * fee_rate
                balance -= fee_paid
                self._total_fees += fee_paid
                newly_active.append(
                    {
                        **planned,
                        "entry_time": candle_timestamp,
                        "entry_price": entry_price,
                        "fee_paid_entry": fee_paid,
                    }
                )
            pending_entries = still_pending
            positions.extend(newly_active)

            remaining_pos = []
            for pos in positions:
                high = float(current_candle["high"])
                low = float(current_candle["low"])
                closed = False
                exit_price = None
                exit_reason = None

                if pos["side"] == "long":
                    if low <= pos["stop_loss"]:
                        exit_price = pos["stop_loss"]
                        exit_reason = "SL"
                        closed = True
                    elif high >= pos["take_profit"]:
                        exit_price = pos["take_profit"]
                        exit_reason = "TP"
                        closed = True
                else:
                    if high >= pos["stop_loss"]:
                        exit_price = pos["stop_loss"]
                        exit_reason = "SL"
                        closed = True
                    elif low <= pos["take_profit"]:
                        exit_price = pos["take_profit"]
                        exit_reason = "TP"
                        closed = True

                if closed:
                    adjusted_exit = self._apply_exit_costs(
                        side=pos["side"],
                        raw_price=float(exit_price),
                    )
                    gross_pl = self._gross_pl(pos["side"], pos["entry_price"], adjusted_exit, pos["size"])
                    exit_fee = adjusted_exit * pos["size"] * fee_rate
                    net_pl = gross_pl - exit_fee
                    balance += net_pl
                    self._total_fees += exit_fee
                    holding_minutes = max(
                        (candle_timestamp - pos["entry_time"]).total_seconds() / 60.0,
                        0.0,
                    )
                    self.trades.append({
                        "symbol": symbol,
                        "strategy": pos.get("strategy", "unknown"),
                        "side": pos["side"],
                        "entry_time": pos["entry_time"],
                        "exit_time": candle_timestamp,
                        "entry_price": pos["entry_price"],
                        "exit_price": adjusted_exit,
                        "size": pos["size"],
                        "gross_pl": gross_pl,
                        "pl": net_pl,
                        "fees": pos.get("fee_paid_entry", 0.0) + exit_fee,
                        "holding_minutes": holding_minutes,
                        "exit_reason": exit_reason,
                    })
                else:
                    remaining_pos.append(pos)
            positions = remaining_pos

            trades_today = sum(1 for trade in self.trades if trade["exit_time"].date() == active_date)
            signal = signal_engine.generate_signal(symbol)
            can_schedule_entry = (
                signal is not None
                and i + 1 < len(df)
                and len(positions) + len(pending_entries) < self.config.max_open_positions
                and trades_today < self.config.max_trades_per_day_max
            )
            if can_schedule_entry:
                signal_side = signal.get("side", "long")
                if signal_side in ("buy", "sell"):
                    signal_side = "long" if signal_side == "buy" else "short"
                if getattr(self.config, "trading_mode", "spot") == "spot" and signal_side == "short":
                    unrealized = 0.0
                    close_price = float(current_candle["close"])
                    for pos in positions:
                        unrealized += self._gross_pl(pos["side"], pos["entry_price"], close_price, pos["size"])
                    self.balance_curve.append(balance + unrealized)
                    continue

                synthetic_state = BotState(balance=balance)
                risk_mgr = RiskManager(self.config, synthetic_state)
                size = risk_mgr.calc_position_size(
                    float(signal["entry_price"]),
                    float(signal["stop_loss"]),
                )
                if size > 0:
                    pending_entries.append(
                        {
                            "activate_index": i + 1,
                            "symbol": symbol,
                            "strategy": signal.get("strategy", "unknown"),
                            "side": signal_side,
                            "stop_loss": float(signal["stop_loss"]),
                            "take_profit": float(signal["take_profit"]),
                            "size": size,
                            "signal_time": candle_timestamp,
                        }
                    )

            unrealized = 0.0
            close_price = float(current_candle["close"])
            for pos in positions:
                unrealized += self._gross_pl(pos["side"], pos["entry_price"], close_price, pos["size"])
            self.balance_curve.append(balance + unrealized)

        metrics = self.compute_metrics()
        return {
            **metrics,
            "raw_trades": len(self.trades),
            "total_fees": self._total_fees,
            "long_trades": sum(1 for trade in self.trades if trade["side"] == "long"),
            "short_trades": sum(1 for trade in self.trades if trade["side"] == "short"),
            "trade_log": self.trades,
        }

    def _apply_entry_costs(self, side: str, raw_price: float) -> float:
        slippage = float(getattr(self.config, "backtest_slippage_bps", 5.0)) / 10000.0
        spread = float(getattr(self.config, "backtest_spread_bps", 4.0)) / 10000.0
        direction = 1.0 if side == "long" else -1.0
        return raw_price * (1.0 + (direction * (slippage + spread / 2.0)))

    def _apply_exit_costs(self, side: str, raw_price: float) -> float:
        slippage = float(getattr(self.config, "backtest_slippage_bps", 5.0)) / 10000.0
        spread = float(getattr(self.config, "backtest_spread_bps", 4.0)) / 10000.0
        direction = -1.0 if side == "long" else 1.0
        return raw_price * (1.0 + (direction * (slippage + spread / 2.0)))

    @staticmethod
    def _gross_pl(side: str, entry_price: float, exit_price: float, size: float) -> float:
        if side == "short":
            return (entry_price - exit_price) * size
        return (exit_price - entry_price) * size

    def compute_metrics(self) -> Dict[str, float]:
        "Compute standard trading metrics"
        if not self.balance_curve:
            return {}

        returns = pd.Series(self.balance_curve).pct_change().dropna()

        if len(returns) == 0:
            return {}

        total_return = (
            self.balance_curve[-1] / self.balance_curve[0] - 1) * 100

        wins = [t['pl'] for t in self.trades if t['pl'] > 0]
        losses = [t['pl'] for t in self.trades if t['pl'] < 0]

        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / \
            gross_loss if gross_loss > 0 else float('inf')

        # Sharpe (annualized)
        sharpe = returns.mean() / returns.std() * \
                              np.sqrt(252 * 24) if returns.std() > 0 else 0

        # Max Drawdown
        peak = np.maximum.accumulate(self.balance_curve)
        drawdown = (self.balance_curve - peak) / peak
        max_dd = drawdown.min() * 100
        expectancy = float(np.mean([t["pl"] for t in self.trades])) if self.trades else 0.0
        avg_holding_minutes = float(np.mean([t.get("holding_minutes", 0.0) for t in self.trades])) if self.trades else 0.0
        by_strategy: Dict[str, int] = {}
        for trade in self.trades:
            strategy = trade.get("strategy", "unknown")
            by_strategy[strategy] = by_strategy.get(strategy, 0) + 1

        return {
            'total_return_pct': total_return,
            'num_trades': len(self.trades),
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'expectancy': expectancy,
            'avg_holding_minutes': avg_holding_minutes,
            'strategy_trade_counts': by_strategy,
        }

class MockBacktestExchange:
    """Mock exchange for backtesting - provides exact historical data on demand"""

    def __init__(self, df: 'pd.DataFrame', symbol: str):  # type: ignore
        self.df = df.reset_index()
        self.symbol = symbol
        self.cursor = len(self.df) - 1

    def set_cursor(self, index: int) -> None:
        self.cursor = max(0, min(index, len(self.df) - 1))

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        if symbol != self.symbol:
            return []
        visible = self.df.iloc[: self.cursor + 1]
        return visible.tail(limit)[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()

    def get_order_book(self, symbol: str):
        "Mock for spread checks"
        if symbol != self.symbol or self.df.empty:
            return {"bid": 100.0, "ask": 100.1}
        close_price = float(self.df.iloc[self.cursor]["close"])
        spread_fraction = float(getattr(self, "spread_fraction", 0.0004))
        half_spread = close_price * spread_fraction / 2.0
        return {"bid": close_price - half_spread, "ask": close_price + half_spread}

class HyperoptEngine:
    "Hyperparameter optimization using backtesting grid search"

    PARAM_GRID = {
        'risk_per_trade_max': [0.01, 0.015, 0.02, 0.025, 0.03],
        'min_rr_ratio': [2.0, 2.5, 3.0, 3.5, 4.0],
        'atr_sl_mult': [0.5, 1.0, 1.5, 2.0],
        'swing_lookback': [3, 5, 7, 10]
    }

    def __init__(self):
        self.best_score = -float('inf')
        self.best_params = {}
        self.results = []

    def optimize(
    self,
    symbol: str = 'BTC/USDT',
    days: int = 90,
     max_combinations: int = None) -> Dict:
        """Full parallel grid search hyperparameter optimization.
        Tests full grid(5 * 5 * 4 * 4=400 combos, approx. 10 minutes) or subset.
        """
        import itertools

        # Full parameter grid
        param_names = list(self.PARAM_GRID.keys())
        full_grid = list(itertools.product(
            *(self.PARAM_GRID[name] for name in param_names)))

        # Limit if requested
        if max_combinations and len(full_grid) > max_combinations:
            import random
            full_grid = random.sample(full_grid, max_combinations)

        print(f"🚀 HYPEROPT: Testing {len(full_grid)} combinations on {symbol} ({days}d)")
        print(f"Grid: {dict(zip(param_names, [len(v) for v in self.PARAM_GRID.values()]))}")

        # Single-threaded sequential for simplicity (parallel had import
        # issues)
        all_results = []
        for i, params in enumerate(full_grid):
            param_dict = dict(zip(param_names, params))
            print(f"  [{i+1}/{len(full_grid)}] Testing {param_dict}")

            # Create test config
            test_config = BotConfig(starting_balance=10000.0, **param_dict)

            # Run backtest
            bt = BacktestEngine(test_config)
            result = bt.run_backtest(symbol, days=days)

            sharpe = result.get('sharpe_ratio', 0)
            all_results.append((param_dict, sharpe, result))

            # Track best
            if sharpe > self.best_score:
                self.best_score = sharpe
                self.best_params = param_dict.copy()

        self.results = all_results

        # Save full results
        summary = {
            'best_sharpe': self.best_score,
            'best_params': self.best_params,
            'total_tested': len(all_results),
            'all_results': [{'params': p, 'sharpe': s} for p, s, _ in all_results]
        }

        with open('hyperopt_results.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save best config
        best_config = BotConfig(starting_balance=10000.0, **self.best_params)
        config_data = {k: v for k, v in vars(
            best_config).items() if not k.startswith('_')}
        with open('best_config.json', 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"\n🎯 BEST RESULTS:")
        print(f"Sharpe: {self.best_score:.3f}")
        print(f"Params: {self.best_params}")
        print(f"Saved: hyperopt_results.json + best_config.json")

        return summary


    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        if symbol != self.symbol:
            return []
        return self.df.tail(limit).reset_index().values.tolist()

    def get_order_book(self, symbol: str):
        # Mock for spread checks
        return {"bid": 100.0, "ask": 100.1}

    def compute_spread(self, symbol: str) -> float:
        ob = self.exch.get_order_book(symbol)
        bid = ob["bid"]
        ask = ob["ask"]
        mid = (ask + bid) / 2.0
        if mid <= 0:
            return 1.0  # absurdly high to block trading
        return (ask - bid) / mid

    def _get_mid_price(self, symbol: str) -> Optional[float]:
        "Helper to compute the mid price using the current order book."
        # Returns None if something goes wrong.
        try:
            ob = self.exch.get_order_book(symbol)
            bid = ob["bid"]
            ask = ob["ask"]
            mid = (ask + bid) / 2.0
            if mid <= 0:
                return None
            return mid
        except Exception as e:
            print(f"[WARN] _get_mid_price failed for {symbol}: {type(e).__name__}: {e}")
            return None

    def _check_slippage_for_market(
    self,
    symbol: str,
     entry_price: float) -> bool:
        "Estimate slippage for a market order."
        # Compare intended entry_price vs current mid price.
        # If relative difference exceeds max_slippage_fraction, block the trade.
        # NOTE: This is an estimate before the order is sent, using the order
        # book mid.
        mid = self._get_mid_price(symbol)
        if mid is None or mid <= 0:
            # If we can't estimate slippage, allow the trade but warn.
            print(
                f"[WARN] Could not estimate slippage for {symbol} "
                f"(no valid mid price). Proceeding with market order."
            )
            return True

        slippage = abs(mid - entry_price) / mid
        if slippage > self.config.max_slippage_fraction:
            print(
                f"[WARN] Estimated slippage too high on {symbol}: {slippage:.4%} "
                f"(max allowed {self.config.max_slippage_fraction:.4%}). Aborting market order."
            )
            return False

        return True

    def _adjust_order_for_market(
        self,
        symbol: str,
        size: float,
        price: Optional[float],
    ) -> tuple[float, Optional[float]]:
        """
        Adjust size and price to match exchange precision and minimums, if metadata is available.

        - Uses trade_client.markets if available.
        - Rounds to 'amount' and 'price' precision.
        - Checks min amount and min notional(cost).
        - If adjustment makes size invalid ( <= 0 or below min), returns (0.0, price).

        This is best - effort and will silently fall back if market metadata is missing.
        """
        market = None
        try:
            if hasattr(self.exch, 'trade_client') and self.exch.trade_client:
                markets = getattr(self.exch.trade_client, "markets", None)
            else:
                markets = None
            if markets and symbol in markets:
                market = markets[symbol]
        except Exception as e:
            print(
    f"[WARN] _adjust_order_for_market: could not access markets for {symbol}: {e}")
            market = None

        if not market:
            # No metadata -> return original values
            return size, price

        limits = market.get("limits", {}) or {}
        precision = market.get("precision", {}) or {}

        amount_min = (limits.get("amount") or {}).get("min")
        cost_min = (limits.get("cost") or {}).get("min")

        amount_prec = precision.get("amount")
        price_prec = precision.get("price")

        # Round size to amount precision
        if amount_prec is not None and amount_prec >= 0:
            factor = 10**amount_prec
            size = int(size * factor) / factor

        # Round price to price precision
        if price is not None and price_prec is not None and price_prec >= 0:
            factor = 10**price_prec
            price = int(price * factor) / factor

        # Enforce minimum amount
        if amount_min is not None and size < amount_min:
            print(
                f"[WARN] Adjusted size {size} for {symbol} is below minimum amount "
                f"{amount_min}. Skipping order."
            )
            return 0.0, price

        # Enforce minimum notional (cost)
        if cost_min is not None and price is not None:
            if size * price < cost_min:
                print(
                    f"[WARN] Notional {size * price:.8f} for {symbol} is below minimum "
                    f"cost {cost_min}. Skipping order."
                )
                return 0.0, price

        return size, price

    def can_trade_symbol_now(self, symbol: str) -> bool:
        spread = self.compute_spread(symbol)
        if spread > self.config.max_spread_fraction:
            print(
                f"[INFO] Spread too high on {symbol}: {spread:.4%} "
                f"(max allowed {self.config.max_spread_fraction:.4%}). Skipping."
            )
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
        """
        Place a trade via ExchangeClient, with:
        - Spread check(can_trade_symbol_now)
        - Optional slippage protection for market orders
        - Size / price adjustment to exchange precision and minimums

        NOTE:
        - Futures routing is still handled by ExchangeClient and currently uses SPOT TESTNET.
        - is_futures flag is passed through unchanged for future extension.
        """
        if not self.validate_trade_plan(side, entry_price, stop_loss, take_profit, size):
            print(
                f"[WARN] Invalid trade plan for {symbol}: side={side}, entry={entry_price}, "
                f"SL={stop_loss}, TP={take_profit}, size={size}"
            )
            return False

        if not self.can_trade_symbol_now(symbol):
            return False

        order_type = "limit"
        price: Optional[float] = entry_price

        if fast_move:
            order_type = "market"
            price = None

            # Slippage estimation only for market orders
            if not self._check_slippage_for_market(symbol, entry_price):
                return False

        if size <= 0:
            print("[WARN] Size is zero or negative, cannot trade.")
            return False

        # Adjust size and price to market constraints (if metadata available)
        adj_size, adj_price = self._adjust_order_for_market(
            symbol, size, price)

        if adj_size <= 0:
            print("[WARN] Adjusted size is invalid, aborting trade.")
            return False

        return self.exch.place_order(
            symbol=symbol,
            side=side,
            size=adj_size,
            order_type=order_type,
            price=adj_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            is_futures=is_futures,
        )


# ============================
# 7) REPORTER + LOG FILES + TELEGRAM
# ============================
class Reporter:
    # Basic log level mapping
    LOG_LEVELS = {
        "DEBUG": 10,
        "INFO": 20,
        "WARN": 30,
        "ERROR": 40,
    }

    def __init__(self, config: BotConfig, state: BotState):
        self.config = config
        self.state = state
        os.makedirs(LOG_DIR, exist_ok=True)

        # Normalize and store numeric log level
        level_name = (
    getattr(
        self.config,
        "log_level",
         "INFO") or "INFO").upper()
        self.log_level = self.LOG_LEVELS.get(
            level_name, self.LOG_LEVELS["INFO"])

    # ---------- INTERNAL HELPERS ----------

    def _should_log(self, level: str) -> bool:
        """Check if a message with this level should be printed / sent, based on config.log_level."""
        lvl = self.LOG_LEVELS.get(level.upper(), self.LOG_LEVELS["INFO"])
        return lvl >= self.log_level

    def _write_log(self, filename: str, text: str):
        try:
            with open(os.path.join(LOG_DIR, filename), "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception as e:
            # Always print this warning, regardless of log_level, because
            # logging failed.
            print(f"[WARN] Failed to write log {filename}: {e}")

    def _count_open_positions(self) -> int:
        """
        Count total open positions across all symbols.
        Compatible with:
        - Dict[str, Position](current structure)
        - Dict[str, List[Position]](future multi - position structure)
        """
        if not self.state.open_positions:
            return 0
        if not self.state.open_positions:
            return 0

        total = 0
        for v in self.state.open_positions.values():
            if isinstance(v, list):
                total += len(v)
            else:
                total += 1
        return total

    # ---------- PUBLIC LOG METHODS ----------

    def log_trade(self, message: str):
        # Trades are always written to file; printing/Telegram depends on
        # caller.
        self._write_log("trades.log", message)

    def log_error(self, message: str):
        # Errors are always written to file; printing/Telegram can use
        # mentor_log with ERROR level.
        self._write_log("errors.log", message)

    def send_telegram(self, message: str):
        """
        Send Telegram message if token + chat_id are configured.
        If either is missing, silently skip.
        """
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            return

        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": message,
            "parse_mode": "HTML",
        }

        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            print(f"[WARN] Telegram send failed: {e}")
            self.log_error(f"Telegram error: {e}")

    def mentor_log(
    self,
    message: str,
    level: str = "INFO",
     send_telegram: bool = True):
        """
        Main logging entry point for human - readable messages.
        - Adds timestamp
        - Writes to mentor.log always
        - Prints & sends Telegram only if log_level threshold is met
        """
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full = f"[MENTOR {now}] {message}"

        # Always write to file
        self._write_log("mentor.log", full)

        # Respect log_level for stdout + Telegram
        if self._should_log(level):
            print(full)
            if send_telegram:
                self.send_telegram(full)

    # ---------- HEARTBEAT & COOLDOWN NOTIFICATIONS ----------

    def heartbeat(self):
        """
        Send periodic heartbeat.
        """
        if not getattr(self.config, "heartbeat_enabled", True):
            return

        interval_min = getattr(self.config, "heartbeat_interval_minutes", 60)
        now = dt.datetime.now()

        if self.state.last_heartbeat is not None:
            delta_sec = (now - self.state.last_heartbeat).total_seconds()
            if delta_sec < interval_min * 60:
                return  # not yet time for next heartbeat

        # Build a compact status snapshot
        open_pos = self._count_open_positions()
        mode = "PAPER" if self.state.paper_mode else "LIVE TESTNET"
        dd_flag = "EMERGENCY" if self.state.emergency_mode else "NORMAL"
        rr_flag = "REDUCED_RISK" if self.state.reduced_risk_mode else "FULL_RISK"

        msg = (
            f"Heartbeat – bot is running.\n"
            f"Mode: <b>{mode}</b>, Drawdown mode: <b>{dd_flag}</b>, Risk mode: <b>{rr_flag}</b>\n"
            f"Balance: <b>{self.state.balance:.2f} USDT</b>, "
            f"Open positions: <b>{open_pos}</b>\n"
        )

        if self.state.cooldown_until:
            remaining = self.state.cooldown_until - now
            if remaining.total_seconds() > 0:
                mins = int(remaining.total_seconds() // 60)
                msg += f"Cooldown active: approx <b>{mins}</b> minutes remaining.\n"
            else:
                msg += "Cooldown scheduled but time already passed (pending reset).\n"

        self.mentor_log(msg, level="INFO", send_telegram=True)
        self.state.last_heartbeat = now

    def notify_cooldown_start(self):
        """
        Optional helper to announce cooldown start.
        Can be called from RiskManager / TradeBot when cooldown is activated.
        """
        if not self.state.cooldown_until:
            return

        now = dt.datetime.now()
        remaining = self.state.cooldown_until - now
        mins = max(int(remaining.total_seconds() // 60), 0)

        msg = (
            f"Loss streak protection activated – entering cooldown for approx "
            f"<b>{mins}</b> minutes. No new trades will be opened during this period."
        )
        self.mentor_log(msg, level="WARN", send_telegram=True)

    def notify_cooldown_end(self):
        """
        Optional helper to announce cooldown end.
        Can be called from TradeBot when cooldown is cleared.
        """
        msg = "Cooldown period ended. Bot is allowed to open new trades again."
        self.mentor_log(msg, level="INFO", send_telegram=True)

    # ---------- DASHBOARDS ----------

    def morning_dashboard(self):
        today = dt.datetime.now().date()
        if self.state.last_morning_dashboard_date == today:
            return

        self.state.last_morning_dashboard_date = today

        open_pos = self._count_open_positions()
        mode = "PAPER" if self.state.paper_mode else "LIVE TESTNET"
        trading_mode = getattr(self.config, "trading_mode", "spot")
        rr_flag = "ON" if self.state.reduced_risk_mode else "OFF"
        dd_flag = "ON" if self.state.emergency_mode else "OFF"

        self.mentor_log(
    "Good morning, Boss. Here's your market prep dashboard.",
     level="INFO")

        self.mentor_log(
            f"Mode: <b>{mode}</b>, Trading mode: <b>{trading_mode}</b>, "
            f"Reduced risk: <b>{rr_flag}</b>, Emergency mode: <b>{dd_flag}</b>",
            level="INFO",
        )

        self.mentor_log(
            f"Balance (test): <b>{self.state.balance:.2f} USDT</b>",
            level="INFO",
        )

        if self.state.equity_start_of_day == 0.0:
            self.state.equity_start_of_day = self.state.balance

        self.mentor_log(
            f"Equity start of day: <b>{self.state.equity_start_of_day:.2f} USDT</b>",
            level="INFO",
        )

        self.mentor_log(
            f"Open positions at start of day: <b>{open_pos}</b>",
            level="INFO",
        )

    def evening_dashboard(self):
        today = dt.datetime.now().date()
        if self.state.last_evening_dashboard_date == today:
            return

        self.state.last_evening_dashboard_date = today

        daily_pl = self.state.balance - self.state.equity_start_of_day
        total_trades = self.state.today_trades_count
        wins = self.state.wins_today
        losses = self.state.losses_today
        winrate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        open_pos = self._count_open_positions()

        self.mentor_log(
            "Good evening, Boss. Here's your performance summary for today.",
            level="INFO",
        )

        self.mentor_log(
            f"Start equity: <b>{self.state.equity_start_of_day:.2f} USDT</b>",
            level="INFO",
        )
        self.mentor_log(
            f"Current balance: <b>{self.state.balance:.2f} USDT</b>",
            level="INFO",
        )
        self.mentor_log(
            f"Realized P/L today: <b>{daily_pl:.2f} USDT</b>",
            level="INFO",
        )
        self.mentor_log(
            f"Trades closed today: <b>{total_trades}</b>",
            level="INFO",
        )
        self.mentor_log(
            f"Wins: <b>{wins}</b>, Losses: <b>{losses}</b>, Winrate: <b>{winrate:.1f}%</b>",
            level="INFO",
        )
        self.mentor_log(
            f"Open positions at end of day: <b>{open_pos}</b>",
            level="INFO",
        )

        # Lifetime stats if being tracked by RiskManager
        lifetime_trades = getattr(self.state, "lifetime_trades", 0)
        lifetime_profit = getattr(self.state, "lifetime_profit", 0.0)
        best_trade = getattr(self.state, "best_single_trade", 0.0)
        worst_trade = getattr(self.state, "worst_single_trade", 0.0)

        self.mentor_log(
            f"Lifetime trades: <b>{lifetime_trades}</b>, "
            f"Lifetime P/L: <b>{lifetime_profit:.2f} USDT</b>",
            level="INFO",
        )
        self.mentor_log(
            f"Best single trade: <b>{best_trade:.2f} USDT</b>, "
            f"Worst single trade: <b>{worst_trade:.2f} USDT</b>",
            level="INFO",
        )

        self.mentor_log(
    "P/L summary: (testnet – no real money).",
     level="INFO")


# ============================
# 8) MAIN LOOP + STATE PERSISTENCE
# ============================
BotConfig = CoreBotConfig
BotState = CoreBotState
Position = CorePosition
MockExchange = CoreMockExchange
ExchangeClient = CoreExchangeClient
RiskManager = CoreRiskManager
ExecutionEngine = CoreExecutionEngine


class TradeBot:
    def __init__(self, config: BotConfig, *, enable_metrics: bool = True):
        self.config = config
        self.base_dir = resolve_runtime_base_dir()
        ensure_runtime_directories(self.base_dir)

        # Set paper_mode correctly based on config FIRST
        paper_mode = config.use_paper_trading

        self.state = BotState(
            balance=config.starting_balance,
            paper_mode=paper_mode,
        )

        # Now initialize exchange AFTER state.paper_mode is set correctly
        if paper_mode:
            print("[PAPER MODE] Using MockExchange")
            self.exch = MockExchange(config, self.state)
        else:
            print("[LIVE TESTNET] Using real ExchangeClient")
            self.exch = ExchangeClient(config, self.state)


        self.risk = RiskManager(config, self.state)
        self.signals = SignalEngine(config, self.exch)  # type: ignore
        self.exec = ExecutionEngine(config, self.state, self.exch)
        self.reporter = Reporter(config, self.state)
        self.state_store = SQLiteStateStore(os.path.join(self.base_dir, STATE_DB_FILE))
        self.learning_store = PersistentLearningStore(
            self.state_store,
            SQLiteStateStore(os.path.join(self.base_dir, LEARNING_DB_FILE)),
        )
        self.learning_backfill_summary = backfill_learning_from_sqlite_artifacts(
            self.base_dir,
            self.learning_store.global_store,
        )
        self.metrics = MetricsCollector(port=8000) if enable_metrics else None
        self.decision_logger = JsonlDecisionLogger(os.path.join(self.base_dir, EVENT_LOG_FILE))
        self.learning = TradeLearningEngine(config, self.learning_store)
        self.reconciler = BotReconciler(self)
        self.last_reconciliation_status = None
        self._shutdown_event = threading.Event()
        self._signal_handlers_registered = False
        self.operator_assistant = BotOperatorAssistant(
            bot=self,
            knowledge_dir=ensure_default_knowledge_base(self.base_dir),
        )
        self.news_engine = BinanceNewsEngine(
            exchange_client=self.exch,
            state_path=os.path.join(self.base_dir, "data", "binance_news_state.json"),
            symbols=self.config.symbols,
            research_store=self.state_store,
        )
        self.signals.research_context_provider = self.news_engine.research_signal_context
        self.signals.learning_context_provider = self._learning_context_for_signal

        # Load previous state (if exists)
        self.load_state()
        self._last_notified_health_blocker = str(getattr(self.state, "health_summary", {}).get("top_blocker", "unknown"))

        # Initialize daily equity and peak equity if needed
        if self.state.equity_start_of_day == 0.0:
            self.state.equity_start_of_day = self.state.balance

        if self.state.peak_equity == 0.0:
            self.state.peak_equity = self.state.balance

    def _refresh_control_status(self, *, status: str = "running") -> None:
        try:
            open_positions = 0
            for value in (self.state.open_positions or {}).values():
                open_positions += len(value) if isinstance(value, list) else 1
            write_bot_status(
                self.base_dir,
                {
                    "status": status,
                    "pid": os.getpid(),
                    "paper": bool(self.state.paper_mode),
                    "trading_mode": getattr(self.config, "trading_mode", "spot"),
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "last_heartbeat": self.state.last_heartbeat.isoformat() if self.state.last_heartbeat else None,
                    "balance": float(self.state.balance),
                    "open_positions": int(open_positions),
                },
            )
        except Exception:
            return

    def answer_operator_question(self, question: str) -> Dict[str, Any]:
        """
        Operator-facing grounded assistant.

        Combines:
        - parametric memory from the model client
        - non-parametric memory from local knowledge files
        - tool outputs from the live bot state and exchange adapters
        """
        response = self.operator_assistant.answer(question)
        return {
            "answer": response.answer,
            "retrieved_context": response.retrieved_context,
            "tool_results": response.tool_results,
            "portfolio_snapshot": _sanitize_jsonish(build_portfolio_snapshot(self)),
            "risk_state": _sanitize_jsonish(build_risk_decision(self)),
            "reconciliation_status": _sanitize_jsonish(self.last_reconciliation_status),
            "strategy_health": _sanitize_jsonish(build_strategy_health(self)),
            "event_risk": _sanitize_jsonish(self.news_engine.event_risk_snapshot()),
            "event_research": _sanitize_jsonish(self.news_engine.recent_research_records(limit=10)),
            "system_health": _sanitize_jsonish(self._build_system_health_summary()),
            "readiness_report": _sanitize_jsonish(build_readiness_report(self)),
        }

    def scan_binance_news(self) -> List[Dict[str, Any]]:
        commands = self.news_engine.scan()
        serialized = [asdict(command) for command in commands]
        self.state.pending_news_commands.extend(serialized)
        self.state.last_news_scan_at = dt.datetime.now()
        if serialized:
            for command in serialized[:5]:
                self.reporter.mentor_log(
                    f"NEWS COMMAND {command['action']} {command['symbol']} "
                    f"confidence={command['confidence']} urgency={command['urgency']} "
                    f"reason={command['rationale']}"
                )
            self.save_state()
        return serialized

    def maybe_run_news_engine(self, now: Optional[dt.datetime] = None) -> List[Dict[str, Any]]:
        if not getattr(self.config, "news_engine_enabled", True):
            return []
        now = now or dt.datetime.now()
        interval_minutes = max(int(getattr(self.config, "news_poll_interval_minutes", 15)), 1)
        if self.state.last_news_scan_at is not None:
            elapsed = (now - self.state.last_news_scan_at).total_seconds()
            if elapsed < interval_minutes * 60:
                return []
        try:
            return self.scan_binance_news()
        except Exception as exc:
            self.reporter.log_error(f"Binance news scan failed: {exc}")
            return []

    # ---------- STATE PERSISTENCE ----------
    def save_state(self):
        trace_id = new_trace_id()
        try:
            save_bot_state(self.state, STATE_FILE)
            persist_runtime_snapshot(self, trace_id)
            self._refresh_control_status(status="running")
        except Exception as e:
            print(f"[WARN] Failed to save state: {e}")

    def load_state(self):
        requested_paper_mode = bool(self.config.use_paper_trading)
        try:
            if load_bot_state(self.state, STATE_FILE):
                print("[INFO] State loaded from file.")
        except Exception as e:
            print(f"[WARN] Failed to load state: {e}")
        self.state.paper_mode = requested_paper_mode
        self.last_reconciliation_status = self.reconciler.reconcile()
        log_reconciliation(self, new_trace_id())
        if getattr(self, "state_store", None) is not None:
            metrics = self.state_store.load_operational_metrics()
            if "runtime_hours" not in metrics:
                self.state_store.set_operational_metric("runtime_hours", 0.0)

    def _cooldown_active(self, attr_name: str, now: Optional[dt.datetime] = None) -> bool:
        now = now or dt.datetime.now()
        until = getattr(self.state, attr_name, None)
        return until is not None and until > now

    def _set_circuit_breaker(self, attr_name: str, minutes: int) -> None:
        if minutes <= 0:
            return
        setattr(self.state, attr_name, dt.datetime.now() + dt.timedelta(minutes=minutes))

    def _hard_halt_reasons(self, now: Optional[dt.datetime] = None) -> List[str]:
        now = now or dt.datetime.now()
        reasons: List[str] = []
        if self.last_reconciliation_status is not None and not bool(getattr(self.last_reconciliation_status, "ok", True)):
            reasons.append("reconciliation_mismatch")
        if self.risk.check_daily_loss_limit():
            reasons.append("daily_loss_limit")
        peak_equity = float(getattr(self.state, "peak_equity", 0.0) or 0.0)
        balance = float(getattr(self.state, "balance", 0.0) or 0.0)
        if peak_equity > 0:
            drawdown = (balance - peak_equity) / peak_equity
            if drawdown <= -0.20:
                reasons.append("max_drawdown")
        if self._cooldown_active("data_cooldown_until", now):
            reasons.append("data_circuit_breaker")
        if self._cooldown_active("execution_cooldown_until", now):
            reasons.append("execution_circuit_breaker")
        return reasons

    def _recover_if_halt_cleared(self, now: Optional[dt.datetime] = None) -> None:
        now = now or dt.datetime.now()
        hard_halts = self._hard_halt_reasons(now)
        if hard_halts:
            return
        if not self.state.emergency_mode:
            return
        self.state.emergency_mode = False
        self.reporter.mentor_log(
            "Emergency mode cleared automatically after halt conditions normalized.",
            level="WARN",
        )

    def _build_system_health_summary(self) -> Dict[str, Any]:
        metrics = self.state_store.load_operational_metrics() if getattr(self, "state_store", None) else {}
        now = dt.datetime.now()
        blockers: List[Dict[str, Any]] = []
        if self.state.emergency_mode:
            blockers.append({"reason": "emergency_mode", "priority": 100})
        if self._cooldown_active("execution_cooldown_until", now):
            blockers.append({"reason": "execution_circuit_breaker", "priority": 95})
        if self._cooldown_active("data_cooldown_until", now):
            blockers.append({"reason": "data_circuit_breaker", "priority": 90})
        if self.state.cooldown_until is not None and self.state.cooldown_until > now:
            blockers.append({"reason": "trade_cooldown", "priority": 80})
        for key, value in self.state.market_regime_alerts.items():
            blockers.append({"reason": f"{key}:{value}", "priority": 60 if key == "SYSTEM" else 40})
        ranked_blockers = [item["reason"] for item in sorted(blockers, key=lambda item: item["priority"], reverse=True)]
        summary = {
            "emergency_mode": self.state.emergency_mode,
            "trade_cooldown_active": self.state.cooldown_until is not None and self.state.cooldown_until > now,
            "data_circuit_breaker_active": self._cooldown_active("data_cooldown_until", now),
            "execution_circuit_breaker_active": self._cooldown_active("execution_cooldown_until", now),
            "market_regime_alerts": dict(self.state.market_regime_alerts),
            "top_blocker": ranked_blockers[0] if ranked_blockers else "healthy",
            "ranked_blockers": ranked_blockers,
            "no_signal_cycles": int(metrics.get("no_signal_cycles", 0)),
            "market_data_health_failures": int(metrics.get("market_data_health_failures", 0)),
            "signal_generation_failures": int(metrics.get("signal_generation_failures", 0)),
            "signal_reliability_rejections": int(metrics.get("signal_reliability_rejections", 0)),
            "consecutive_order_failures": int(metrics.get("consecutive_order_failures", 0)),
        }
        self.state.health_summary = summary
        return summary

    def _learning_context_for_signal(self, symbol: str, signal: Dict[str, Any], regime: Any | None = None) -> Dict[str, Any]:
        try:
            return self.learning.learning_context_for_signal(signal, regime)
        except Exception as exc:
            self.reporter.log_error(f"Learning context failed for {symbol}: {type(exc).__name__}: {exc}")
            return {}

    def _notify_health_transition(self, previous: Dict[str, Any], current: Dict[str, Any]) -> None:
        current_blocker = current.get("top_blocker", "healthy")
        previous_blocker = previous.get("top_blocker", self._last_notified_health_blocker)
        if current_blocker == previous_blocker or current_blocker == self._last_notified_health_blocker:
            return
        self.reporter.mentor_log(
            f"System health changed from <b>{previous_blocker}</b> to <b>{current_blocker}</b>.",
            level="WARN" if current_blocker != "healthy" else "INFO",
            send_telegram=True,
        )
        self._last_notified_health_blocker = str(current_blocker)

    def _assess_market_data_health(self, symbol: str, timeframe: Optional[str] = None) -> tuple[bool, str]:
        timeframe = timeframe or self.config.timeframes.get("entry", "15m")
        candles = self.exch.fetch_ohlcv(symbol, timeframe, limit=3) or []
        if not candles:
            return False, "missing_candles"
        last_candle = candles[-1]
        if len(last_candle) < 6:
            return False, "malformed_candle"
        try:
            candle_ts = float(last_candle[0]) / 1000.0
            high = float(last_candle[2])
            low = float(last_candle[3])
            close = float(last_candle[4])
            volume = float(last_candle[5])
        except Exception:
            return False, "non_numeric_candle"
        if high < low or close <= 0 or volume < 0:
            return False, "invalid_ohlcv"
        age_seconds = max(dt.datetime.now().timestamp() - candle_ts, 0.0)
        max_age_seconds = max(int(getattr(self.config, "max_market_data_age_minutes", 90)), 1) * 60
        if age_seconds > max_age_seconds:
            return False, "stale_market_data"
        return True, "ok"

    def _safe_generate_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        if self._cooldown_active("data_cooldown_until"):
            self.state.market_regime_alerts["SYSTEM"] = "data_circuit_breaker_active"
            return None
        healthy, reason = self._assess_market_data_health(symbol)
        if not healthy:
            self.state.market_regime_alerts[symbol] = reason
            if getattr(self, "state_store", None) is not None:
                self.state_store.increment_operational_metric("market_data_health_failures", 1)
            return None
        try:
            signal = self.signals.generate_signal(symbol)
        except Exception as exc:
            self.reporter.log_error(f"Signal generation failed for {symbol}: {type(exc).__name__}: {exc}")
            self.state.market_regime_alerts[symbol] = "signal_generation_error"
            if getattr(self, "state_store", None) is not None:
                self.state_store.increment_operational_metric("signal_generation_failures", 1)
            return None
        if signal is None:
            return None
        if not self._signal_payload_is_tradeable(signal):
            self.state.market_regime_alerts[symbol] = "untradeable_signal"
            if getattr(self, "state_store", None) is not None:
                self.state_store.increment_operational_metric("signal_reliability_rejections", 1)
            return None
        self.state.market_regime_alerts.pop(symbol, None)
        return signal

    @staticmethod
    def _signal_payload_is_tradeable(signal: Dict[str, Any]) -> bool:
        required = ("side", "entry_price", "stop_loss", "take_profit", "strategy")
        for key in required:
            if key not in signal:
                return False
        try:
            entry_price = float(signal.get("entry_price", 0.0) or 0.0)
            stop_loss = float(signal.get("stop_loss", 0.0) or 0.0)
            take_profit = float(signal.get("take_profit", 0.0) or 0.0)
            rr_ratio = float(signal.get("rr_ratio", 0.0) or 0.0)
            signal_quality = float(signal.get("signal_quality", 0.0) or 0.0)
        except Exception:
            return False
        if not all(math.isfinite(v) for v in (entry_price, stop_loss, take_profit, rr_ratio, signal_quality)):
            return False
        if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            return False
        side = str(signal.get("side", "")).lower()
        if side in {"long", "buy"}:
            return stop_loss < entry_price < take_profit
        if side in {"short", "sell"}:
            return stop_loss > entry_price > take_profit
        return False

    def run_once(self):
        now = dt.datetime.now()
        cycle_trace_id = new_trace_id()
        if getattr(self, "state_store", None) is not None:
            self.state_store.increment_operational_metric("runtime_hours", 1.0 / 60.0)
        signals_generated = 0
        signal_failures = 0
        previous_health = dict(getattr(self.state, "health_summary", {}))

        # Daily reset for counters and equity
        self.risk.check_daily_reset()
        self.last_reconciliation_status = self.reconciler.reconcile()
        log_reconciliation(self, cycle_trace_id)
        if self.metrics is not None:
            self.metrics.set_reconciliation(self.last_reconciliation_status.ok)
        if self.last_reconciliation_status and not self.last_reconciliation_status.ok:
            if getattr(self, "state_store", None) is not None:
                self.state_store.increment_operational_metric("reconciliation_failures", 1)
            self.state.emergency_mode = True
            decision = build_risk_decision(self)
            log_risk_halt(self, cycle_trace_id, decision)
            if self.metrics is not None:
                self.metrics.inc_risk_halt(decision.reason)
            if getattr(self, "state_store", None) is not None:
                self.state_store.increment_operational_metric("risk_halts", 1)
            self.reporter.mentor_log(
                "Reconciliation mismatch detected. Trading halted until operator review."
            )
            self.save_state()
            return

        # Track cooldown transitions for notifications
        in_cooldown_before = self.state.cooldown_until is not None and self.state.cooldown_until > now

        # Peak equity & drawdown checks (based on balance for now)
        if self.state.peak_equity == 0.0 or self.state.balance > self.state.peak_equity:
            self.state.peak_equity = self.state.balance

        if self.state.peak_equity > 0:
            dd = (self.state.balance - self.state.peak_equity) / \
                  self.state.peak_equity

            if dd <= -0.20:
                self.state.emergency_mode = True
                log_risk_halt(self, cycle_trace_id, build_risk_decision(self))
                if self.metrics is not None:
                    self.metrics.inc_risk_halt("max_drawdown")
                if getattr(self, "state_store", None) is not None:
                    self.state_store.increment_operational_metric("risk_halts", 1)
                self.reporter.mentor_log(
                    f"Max drawdown reached ({dd*100:.1f}%). Trading halted (emergency mode)."
                )
                self.save_state()
                return

            elif dd <= -0.15:
                if not self.state.reduced_risk_mode:
                    self.state.reduced_risk_mode = True
                    self.reporter.mentor_log(
                        f"Drawdown level 2 ({dd*100:.1f}%). Reduced risk mode enabled.")

            elif dd <= -0.10:
                if not self.state.reduced_risk_mode:
                    self.state.reduced_risk_mode = True
                    self.reporter.mentor_log(
                        f"Drawdown level 1 ({dd*100:.1f}%). Reduced risk mode enabled.")

            else:
                if self.state.consecutive_losses == 0 and self.state.reduced_risk_mode:
                    self.state.reduced_risk_mode = False
                    self.reporter.mentor_log(
                        "Drawdown recovered. Reduced risk mode disabled.")

        # Daily loss limit
        if self.risk.check_daily_loss_limit():
            self.state.emergency_mode = True
            log_risk_halt(self, cycle_trace_id, build_risk_decision(self))
            if self.metrics is not None:
                self.metrics.inc_risk_halt("daily_loss_limit")
            if getattr(self, "state_store", None) is not None:
                self.state_store.increment_operational_metric("risk_halts", 1)
            self.reporter.mentor_log(
                "Daily loss limit reached. Trading halted for the rest of the day.")
            self.save_state()
            return

        # Multi-coin volatility sweep across the configured symbol universe.
        spiking_symbols = [
            symbol for symbol in self.config.symbols
            if self.signals.is_atr_spike(
                symbol,
                timeframe=self.config.timeframes.get("entry", "15m"),
                period=14,
                spike_mult=3.0,
            )
        ]
        self.state.market_regime_alerts = {
            symbol: "atr_spike" for symbol in spiking_symbols
        }
        if spiking_symbols:
            self.state.emergency_mode = True
            log_risk_halt(self, cycle_trace_id, build_risk_decision(self))
            if self.metrics is not None:
                self.metrics.inc_risk_halt("atr_spike")
            if getattr(self, "state_store", None) is not None:
                self.state_store.increment_operational_metric("risk_halts", 1)
            self.reporter.mentor_log(
                "ATR volatility spike detected on "
                + ", ".join(spiking_symbols[:5])
                + ". Trading halted for the rest of the day."
            )
            self.save_state()
            return

        self._recover_if_halt_cleared(now)

        # Dashboards
        if now.hour == self.config.morning_hour:
            self.reporter.morning_dashboard()

        if now.hour == self.config.evening_hour:
            self.reporter.evening_dashboard()

        # Heartbeat (will send only if interval passed)
        self.reporter.heartbeat()
        self.maybe_run_news_engine(now)
        self.learning.evaluate_pending_shadow_decisions(self.exch, now)
        current_health = self._build_system_health_summary()
        self._notify_health_transition(previous_health, current_health)
        portfolio_snapshot = build_portfolio_snapshot(self)
        if self.metrics is not None:
            self.metrics.set_balance(self.state.balance)
            self.metrics.set_open_positions(len(portfolio_snapshot.open_positions))
            self.metrics.set_exposures(portfolio_snapshot.gross_exposure, portfolio_snapshot.net_exposure)

        # If in emergency mode -> observe only (no repeated spam message)
        if self.state.emergency_mode:
            self.save_state()
            return

        # Manage existing positions (check SL/TP)
        self.manage_open_positions()

        # Cooldown transition notifications
        now_after = dt.datetime.now()
        in_cooldown_after = self.state.cooldown_until is not None and self.state.cooldown_until > now_after

        if not in_cooldown_before and in_cooldown_after:
            self.reporter.notify_cooldown_start()
        elif in_cooldown_before and not in_cooldown_after:
            self.reporter.notify_cooldown_end()

        # Scan symbols for new signals
        from dataclasses import is_dataclass

        for symbol in self.config.symbols:
            # We rely on generate_signal's internal filters
            signal = self._safe_generate_signal(symbol)
            if not signal:
                if self.state.market_regime_alerts.get(symbol) == "signal_generation_error":
                    signal_failures += 1
                continue
            signals_generated += 1
            log_signal(self, cycle_trace_id, {"symbol": symbol, **signal})

            if not self.risk.can_open_new_position(symbol):
                self.learning.record_shadow_decision(signal, status="skipped", reason="risk_capacity", trace_id=cycle_trace_id)
                continue

            risk_decision = build_risk_decision(self, signal)
            if not risk_decision.allowed:
                self.learning.record_shadow_decision(signal, status="skipped", reason=risk_decision.reason, trace_id=cycle_trace_id)
                log_risk_halt(self, cycle_trace_id, risk_decision)
                if self.metrics is not None:
                    self.metrics.inc_risk_halt(risk_decision.reason)
                if getattr(self, "state_store", None) is not None:
                    self.state_store.increment_operational_metric("risk_halts", 1)
                continue

            side = signal.get("side")
            entry_price = signal.get("entry_price")
            stop_loss = signal.get("stop_loss")
            take_profit = signal.get("take_profit")
            metadata = dict(signal.get("metadata", {}) or {})
            fast_move = bool(signal.get("fast_move", False)) and not bool(metadata.get("force_limit_entry", False))
            is_futures = signal.get("is_futures", False)
            normalized_side = (side or "").lower()
            if getattr(self.config, "trading_mode", "spot") == "spot" and normalized_side in ("short", "sell"):
                self.learning.record_shadow_decision(signal, status="skipped", reason="spot_short_blocked", trace_id=cycle_trace_id)
                continue

            # Enforce at most one LONG and one SHORT per symbol (no pyramiding
            # same side)
            existing = self.state.open_positions.get(symbol)
            same_side_exists = False

            if existing is not None:
                if is_dataclass(existing):
                    if existing.side == side:
                        same_side_exists = True
                elif isinstance(existing, list):
                    for pos in existing:
                        if pos.side == side:
                            same_side_exists = True
                            break

            if same_side_exists:
                # A position of the same side already exists for this symbol ->
                # skip new one
                self.learning.record_shadow_decision(signal, status="skipped", reason="same_side_exists", trace_id=cycle_trace_id)
                continue

            size = self.risk.calc_position_size(
                entry_price or 0,
                stop_loss or 0,
                signal=signal,
            )
            learning_context = dict((signal.get("metadata", {}) or {}).get("learning_context", {}) or {})
            size *= float(learning_context.get("risk_multiplier", 1.0) or 1.0)
            portfolio_decision = self.risk.evaluate_portfolio_risk(
                symbol=symbol,
                strategy=signal.get("strategy", "unknown"),
                side=side or "buy",
                entry_price=float(entry_price or 0),
                proposed_size=float(size or 0),
            )
            if not portfolio_decision.allowed:
                self.learning.record_shadow_decision(signal, status="skipped", reason=portfolio_decision.reason, trace_id=cycle_trace_id)
                log_risk_halt(self, cycle_trace_id, portfolio_decision)
                if self.metrics is not None:
                    self.metrics.inc_risk_halt(portfolio_decision.reason)
                if getattr(self, "state_store", None) is not None:
                    self.state_store.increment_operational_metric("risk_halts", 1)
                continue
            size = float(portfolio_decision.capped_size or size)
            if size <= 0 or not entry_price or not stop_loss:
                self.learning.record_shadow_decision(signal, status="skipped", reason="invalid_sized_trade", trace_id=cycle_trace_id)
                continue

            success = self.exec.place_trade(
                symbol=symbol,
                side=side or "buy",
                entry_price=entry_price or 0,
                stop_loss=stop_loss or 0,
                take_profit=take_profit or 0,
                is_futures=is_futures,
                fast_move=fast_move,
                size=size,
            )

            if success:
                if getattr(self, "state_store", None) is not None:
                    self.state_store.set_operational_metric("consecutive_order_failures", 0)
                fill_fraction = float(getattr(getattr(self.exec, "last_fill", None), "metadata", {}).get("fill_fraction", 1.0))
                if self.metrics is not None:
                    self.metrics.inc_order_success()
                    self.metrics.observe_fill_ratio(fill_fraction)
                side_str = (side or "buy").upper()
                msg = (
                    f"Opened {side_str} on {symbol} with entry={entry_price}, "
                    f"SL={stop_loss}, TP={take_profit} "
                    f"(futures={is_futures}, fast_move={fast_move}, size={size})"
                )
                self.reporter.mentor_log(msg)
                self.reporter.log_trade("OPEN " + msg)

                execution_context = {
                    "spread_bps": float(signal.get("metadata", {}).get("spread_bps", 0.0) or 0.0),
                    "entry_deviation_bps": float(signal.get("metadata", {}).get("entry_deviation_bps", 0.0) or 0.0),
                    "fill_fraction": float(getattr(getattr(self.exec, "last_fill", None), "metadata", {}).get("fill_fraction", 1.0) or 1.0),
                    "latency_ms": float(getattr(getattr(self.exec, "last_fill", None), "metadata", {}).get("latency_ms", 0.0) or 0.0),
                    "order_type": "market" if fast_move else "limit",
                }

                # Store as list of positions per symbol for multi-position
                # support
                new_pos = Position(
                    symbol=symbol,
                    side=side or "buy",
                    entry_price=entry_price or 0,
                    size=size,
                    stop_loss=stop_loss or 0,
                    take_profit=take_profit or 0,
                    strategy=signal.get('strategy', 'unknown'),
                    is_futures=is_futures,
                    opened_at=dt.datetime.now(),
                    initial_stop_loss=stop_loss or 0,
                    initial_take_profit=take_profit or 0,
                    metadata={
                        "decision_context": self.learning.build_trade_context(signal, execution_context=execution_context),
                        "signal_snapshot": signal,
                        "execution_context": execution_context,
                        "opened_trace_id": cycle_trace_id,
                    },
                )
                if symbol not in self.state.open_positions:
                    self.state.open_positions[symbol] = [new_pos]
                else:
                    val = self.state.open_positions[symbol]
                    if is_dataclass(val):
                        self.state.open_positions[symbol] = [val, new_pos]
                    elif isinstance(val, list):
                        val.append(new_pos)
                    else:
                        self.state.open_positions[symbol] = [new_pos]

                emit_event(
                    self,
                    "position_updated",
                    cycle_trace_id,
                    {
                        "symbol": symbol,
                        "side": side or "buy",
                        "size": size,
                        "strategy": signal.get("strategy", "unknown"),
                    },
                )
                self.save_state()
            else:
                self.learning.record_shadow_decision(signal, status="skipped", reason="execution_failed", trace_id=cycle_trace_id)
                attempts = int(getattr(self.exec, "last_execution_report", {}).get("attempts", 1))
                for _ in range(max(attempts - 1, 0)):
                    if self.metrics is not None:
                        self.metrics.inc_order_retry()
                if self.metrics is not None:
                    self.metrics.inc_order_failure()
                if getattr(self, "state_store", None) is not None:
                    self.state_store.increment_operational_metric("consecutive_order_failures", 1)
        if getattr(self, "state_store", None) is not None:
            if signals_generated == 0:
                no_signal_cycles = int(self.state_store.load_operational_metrics().get("no_signal_cycles", 0)) + 1
                self.state_store.set_operational_metric("no_signal_cycles", no_signal_cycles)
                alert_threshold = int(getattr(self.config, "no_signal_cycles_before_alert", 30))
                if no_signal_cycles >= alert_threshold:
                    self.state.market_regime_alerts["SYSTEM"] = "extended_no_signal_period"
            else:
                self.state_store.set_operational_metric("no_signal_cycles", 0)
                self.state.market_regime_alerts.pop("SYSTEM", None)
            max_signal_exceptions = int(getattr(self.config, "max_signal_exceptions_per_cycle", 2))
            if signal_failures > max_signal_exceptions:
                self.state.emergency_mode = True
                self.state.market_regime_alerts["SYSTEM"] = "signal_engine_unhealthy"
                self.state_store.increment_operational_metric("risk_halts", 1)
                self._set_circuit_breaker("data_cooldown_until", int(getattr(self.config, "data_circuit_breaker_minutes", 30)))
            consecutive_order_failures = int(self.state_store.load_operational_metrics().get("consecutive_order_failures", 0))
            if consecutive_order_failures >= int(getattr(self.config, "max_consecutive_order_failures", 3)):
                self.state.emergency_mode = True
                self.state.market_regime_alerts["SYSTEM"] = "execution_unhealthy"
                self.state_store.increment_operational_metric("risk_halts", 1)
                self._set_circuit_breaker("execution_cooldown_until", int(getattr(self.config, "execution_circuit_breaker_minutes", 30)))
            market_data_failures = int(self.state_store.load_operational_metrics().get("market_data_health_failures", 0))
            if market_data_failures > 0 and signals_generated == 0:
                self._set_circuit_breaker("data_cooldown_until", int(getattr(self.config, "data_circuit_breaker_minutes", 30)))
            current_health = self._build_system_health_summary()
            self._notify_health_transition(previous_health, current_health)
        self.save_state()

    def render_status_report(self) -> str:
        health = self._build_system_health_summary()
        learning = self.learning.summary_snapshot() if getattr(self, "learning", None) is not None else {}
        drift = dict(learning.get("drift", {}) or {})
        opportunity = dict(learning.get("opportunity", {}) or {})
        lines = [
            f"Balance: {self.state.balance:.2f} USDT",
            f"Emergency mode: {health['emergency_mode']}",
            f"Top blocker: {health['top_blocker']}",
            f"Trade cooldown active: {health['trade_cooldown_active']}",
            f"Data circuit breaker active: {health['data_circuit_breaker_active']}",
            f"Execution circuit breaker active: {health['execution_circuit_breaker_active']}",
            f"No-signal cycles: {health['no_signal_cycles']}",
            f"Signal generation failures: {health['signal_generation_failures']}",
            f"Market data health failures: {health['market_data_health_failures']}",
            f"Consecutive order failures: {health['consecutive_order_failures']}",
            f"Learning recent trades: {int(learning.get('recent_trades', 0) or 0)}",
            f"Learning recent avg R: {float(learning.get('recent_avg_r_multiple', 0.0) or 0.0):.2f}",
            f"Learning drift active: {bool(drift.get('active', False))}",
            f"Learning opportunity avg R: {float(opportunity.get('avg_forward_r', 0.0) or 0.0):.2f}",
        ]
        learning_store_info = self.learning_store.describe() if getattr(self, "learning_store", None) is not None else {}
        global_learning = dict(learning_store_info.get("global", {}) or {})
        if global_learning:
            lines.extend(
                [
                    f"Learning memory path: {global_learning.get('path', 'n/a')}",
                    f"Learning observations stored: {int(global_learning.get('observations', 0) or 0)}",
                    f"Learning patterns stored: {int(global_learning.get('patterns', 0) or 0)}",
                    f"Learning models stored: {int(global_learning.get('models', 0) or 0)}",
                    f"Learning imports completed: {int(global_learning.get('imports', 0) or 0)}",
                ]
            )
        if getattr(self, "learning_backfill_summary", None):
            imported_files = int(self.learning_backfill_summary.get("imported_files", 0) or 0)
            if imported_files > 0:
                lines.append(
                    "Learning backfill: "
                    f"{imported_files} files, "
                    f"{int(self.learning_backfill_summary.get('imported_observations', 0) or 0)} observations, "
                    f"{int(self.learning_backfill_summary.get('imported_patterns', 0) or 0)} patterns, "
                    f"{int(self.learning_backfill_summary.get('imported_models', 0) or 0)} models"
                )
        if health["ranked_blockers"]:
            lines.append("Ranked blockers: " + ", ".join(health["ranked_blockers"]))
        if learning.get("top_attributions"):
            lines.append("Learning attributions: " + ", ".join(learning["top_attributions"]))
        return "\n".join(lines)

    def manage_open_positions(self):
        """
        Check SL / TP on open positions for each symbol.
        Supports multiple positions per symbol and both LONG and SHORT.
        """
        from dataclasses import is_dataclass

        entry_tf = self.config.timeframes.get("entry", "15m")

        for symbol in list(self.state.open_positions.keys()):
            val = self.state.open_positions[symbol]

            # Normalize to list of positions
            if is_dataclass(val):
                positions = [val]
            elif isinstance(val, list):
                positions = list(val)
            else:
                continue

            candles = self.exch.fetch_ohlcv(symbol, entry_tf, limit=1)
            if not candles:
                continue

            _ts, _open, last_high, last_low, last_close, _vol = candles[-1]

            remaining_positions = []

            for pos in positions:
                closed = False
                close_price = None
                exit_reason = "UNKNOWN"

                base_sl = pos.initial_stop_loss if pos.initial_stop_loss else pos.stop_loss
                if pos.side == "long":
                    risk_dist = pos.entry_price - base_sl
                else:
                    risk_dist = base_sl - pos.entry_price
                if risk_dist > 0:
                    if pos.side == "long":
                        rr_move = (last_high - pos.entry_price) / risk_dist
                        if rr_move >= self.config.breakeven_rr and pos.stop_loss < pos.entry_price:
                            pos.stop_loss = pos.entry_price
                    else:
                        rr_move = (pos.entry_price - last_low) / risk_dist
                        if rr_move >= self.config.breakeven_rr and pos.stop_loss > pos.entry_price:
                            pos.stop_loss = pos.entry_price

                    if rr_move >= self.config.trailing_rr:
                        atr = self.signals.compute_atr(
                            symbol,
                            timeframe=self.config.trailing_timeframe,
                            period=self.config.trailing_atr_period,
                        )
                        if atr is not None and atr > 0:
                            if pos.side == "long":
                                new_sl = last_high - atr * self.config.trailing_atr_mult
                                if new_sl > pos.stop_loss:
                                    pos.stop_loss = new_sl
                            else:
                                new_sl = last_low + atr * self.config.trailing_atr_mult
                                if new_sl < pos.stop_loss:
                                    pos.stop_loss = new_sl

                if pos.side == "long":
                    # Long: SL if low <= SL, TP if high >= TP
                    if last_low <= pos.stop_loss:
                        close_price = pos.stop_loss
                        exit_reason = "SL"
                        closed = True
                    elif last_high >= pos.take_profit:
                        close_price = pos.take_profit
                        exit_reason = "TP"
                        closed = True

                elif pos.side == "short":
                    # Short: SL if high >= SL, TP if low <= TP
                    if last_high >= pos.stop_loss:
                        close_price = pos.stop_loss
                        exit_reason = "SL"
                        closed = True
                    elif last_low <= pos.take_profit:
                        close_price = pos.take_profit
                        exit_reason = "TP"
                        closed = True

                if not closed and risk_dist > 0:
                    signal_snapshot = dict((getattr(pos, "metadata", {}) or {}).get("signal_snapshot", {}) or {})
                    expected_holding = float(signal_snapshot.get("expected_holding_minutes", 0.0) or 0.0)
                    if expected_holding > 0 and getattr(pos, "opened_at", None) is not None:
                        holding_minutes = max((dt.datetime.now() - pos.opened_at).total_seconds() / 60.0, 0.0)
                        current_r = ((last_close - pos.entry_price) / risk_dist) if pos.side == "long" else ((pos.entry_price - last_close) / risk_dist)
                        if (
                            holding_minutes >= (expected_holding * self.config.time_stop_hard_holding_multiplier)
                            and current_r < self.config.time_stop_hard_min_r_multiple
                        ):
                            close_price = last_close
                            exit_reason = "TIME_HARD"
                            closed = True
                        elif (
                            holding_minutes >= (expected_holding * self.config.time_stop_soft_holding_multiplier)
                            and current_r < self.config.time_stop_soft_min_r_multiple
                        ):
                            close_price = last_close
                            exit_reason = "TIME_SOFT"
                            closed = True

                if not closed:
                    remaining_positions.append(pos)
                    continue

                # Compute P&L
                entry_price = float(pos.entry_price or 0)
                close_price_val = float(close_price or 0)
                if pos.side == "long":
                    pl = (close_price_val - entry_price) * pos.size
                else:  # short
                    pl = (entry_price - close_price_val) * pos.size

                self.state.balance += pl
                self.risk.update_after_trade_result(pl)
                self.learning.record_closed_trade(
                    symbol=symbol,
                    position=pos,
                    close_price=close_price_val,
                    profit_loss=pl,
                    exit_reason=exit_reason,
                    closed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                )
                if getattr(self, "state_store", None) is not None:
                    self.state_store.increment_operational_metric("closed_trades", 1)
                    wins = float(self.state_store.load_operational_metrics().get("wins", 0.0))
                    losses = float(self.state_store.load_operational_metrics().get("losses", 0.0))
                    gross_profit = float(self.state_store.load_operational_metrics().get("gross_profit", 0.0))
                    gross_loss = float(self.state_store.load_operational_metrics().get("gross_loss", 0.0))
                    if pl >= 0:
                        wins += 1.0
                        gross_profit += pl
                        self.state_store.set_operational_metric("wins", wins)
                        self.state_store.set_operational_metric("gross_profit", gross_profit)
                    else:
                        losses += 1.0
                        gross_loss += abs(pl)
                        self.state_store.set_operational_metric("losses", losses)
                        self.state_store.set_operational_metric("gross_loss", gross_loss)
                    total = wins + losses
                    win_rate = (wins / total * 100.0) if total > 0 else 0.0
                    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit
                    peak_equity = float(self.state_store.load_operational_metrics().get("promotion_peak_equity", self.state.balance))
                    if self.state.balance > peak_equity:
                        peak_equity = self.state.balance
                        self.state_store.set_operational_metric("promotion_peak_equity", peak_equity)
                    max_drawdown_pct = 0.0
                    if peak_equity > 0:
                        max_drawdown_pct = abs(min((self.state.balance - peak_equity) / peak_equity * 100.0, 0.0))
                    self.state_store.set_operational_metric("win_rate_pct", win_rate)
                    self.state_store.set_operational_metric("profit_factor", profit_factor)
                    self.state_store.set_operational_metric("max_drawdown_pct", max_drawdown_pct)

                msg = (
                    f"Closed {pos.side.upper()} on {symbol} at {close_price} "
                    f"for P/L={pl:.2f}. New balance={self.state.balance:.2f}"
                )
                self.reporter.mentor_log(msg)
                self.reporter.log_trade("CLOSE " + msg)

            # Update open_positions for this symbol
            if not remaining_positions:
                del self.state.open_positions[symbol]
            else:
                # Keep list for consistency
                self.state.open_positions[symbol] = remaining_positions
        self.save_state()

    def _request_shutdown(self, signum: int | None = None, frame: Any | None = None) -> None:
        signal_name = None
        if signum is not None:
            try:
                signal_name = signal.Signals(signum).name
            except Exception:
                signal_name = str(signum)
        message = "Shutdown requested. Saving state and stopping loop."
        if signal_name:
            message = f"{message} Signal={signal_name}"
        self.reporter.mentor_log(message, level="WARN", send_telegram=False)
        self._shutdown_event.set()

    def _register_signal_handlers(self) -> None:
        if self._signal_handlers_registered:
            return
        for signame in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, signame, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, self._request_shutdown)
            except ValueError:
                continue
        self._signal_handlers_registered = True

    def run_forever(self, sleep_seconds: int = 60):
        mode = "PAPER MODE" if self.state.paper_mode else "LIVE TESTNET MODE"
        self.reporter.mentor_log(f"Starting passive 24/7 loop in {mode}.")
        self._register_signal_handlers()

        while not self._shutdown_event.is_set():
            try:
                self.run_once()
                self._refresh_control_status(status="running")
            except KeyboardInterrupt:
                self._request_shutdown()
            except Exception as e:
                err_msg = f"Error occurred: {e}"
                self.reporter.log_error(err_msg)
                print(err_msg)
            if self._shutdown_event.wait(max(int(sleep_seconds), 1)):
                break
        self.save_state()
        self.reporter.mentor_log("Runtime loop stopped cleanly.", send_telegram=False)

from trade_bot.simulation import HistoricalReplayExchange as _HistoricalReplayExchange
from trade_bot.simulation import HistoricalSimulationEngine as _HistoricalSimulationEngine


class MockBacktestExchange(_HistoricalReplayExchange):
    pass


class BacktestEngine(_HistoricalSimulationEngine):
    def __init__(self, config: BotConfig, **kwargs: Any):
        super().__init__(
            config,
            signal_engine_cls=SignalEngine,
            bot_state_cls=CoreBotState,
            position_cls=CorePosition,
            risk_manager_cls=CoreRiskManager,
            **kwargs,
        )

    def run_backtest(self, symbol: str, timeframe: str = "15m", days: int = 180) -> Dict[str, Any]:
        self.signal_engine_cls = SignalEngine
        return super().run_backtest(symbol, timeframe=timeframe, days=days)

    def run_campaign(self, symbols: Sequence[str], *, timeframe: str = "15m", days: int = 180) -> Dict[str, Any]:
        self.signal_engine_cls = SignalEngine
        return super().run_campaign(symbols, timeframe=timeframe, days=days)


if __name__ == "__main__":
    from trade_bot.cli import main as cli_main

    cli_main()
