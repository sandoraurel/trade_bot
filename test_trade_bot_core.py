import datetime as dt
import io
import json
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

from trade_bot.backtest_reporting import build_backtest_report, save_backtest_report
from trade_bot.conversation_api import ConversationAPIApp, build_conversation_api
from trade_bot.ensemble import EnsembleAllocator
from trade_bot.health import evaluate_runtime_health
from trade_bot.cli import build_parser, build_runtime_config
from trade_bot.learning import TradeLearningEngine
from trade_bot.main import BacktestEngine, BotConfig, BotState, ExecutionEngine, MockBacktestExchange, Position, RiskManager, SignalEngine, TradeBot
from trade_bot.models import Signal
from trade_bot.regime import RegimeAssessment
from trade_bot.news_engine import BinanceNewsEngine, NewsEvent, TradeCommand
from trade_bot.readiness import build_readiness_report
from trade_bot.strategies import MeanReversionStrategy, StrategyProposal, TrendBreakoutStrategy, TrendPullbackStrategy
from trade_bot.state_store import SQLiteStateStore


def utcnow_naive() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class DummyExchange:
    def __init__(self):
        self.orders = []

    def get_order_book(self, symbol):
        return {"bid": 100.0, "ask": 100.1}

    def place_order(self, **kwargs):
        self.orders.append(kwargs)
        return True

    def fetch_ohlcv(self, symbol, timeframe, limit=1):
        return [[0.0, 100.0, 110.0, 90.0, 105.0, 1000.0] for _ in range(limit)]


class SlippageExchange(DummyExchange):
    def get_order_book(self, symbol):
        return {"bid": 120.0, "ask": 121.0}


class ReporterStub:
    def __init__(self):
        self.messages = []
        self.trades = []
        self.errors = []
        self.cooldown_start_calls = 0
        self.cooldown_end_calls = 0

    def mentor_log(self, message, level="INFO", send_telegram=True):
        self.messages.append((level, message))

    def log_trade(self, message):
        self.trades.append(message)

    def log_error(self, message):
        self.errors.append(message)

    def heartbeat(self):
        return None

    def morning_dashboard(self):
        return None

    def evening_dashboard(self):
        return None

    def notify_cooldown_start(self):
        self.cooldown_start_calls += 1

    def notify_cooldown_end(self):
        self.cooldown_end_calls += 1


class StrategyExchangeStub:
    def __init__(self, candles_by_timeframe=None):
        self.candles_by_timeframe = candles_by_timeframe or {}

    def fetch_ohlcv(self, symbol, timeframe, limit=200):
        candles = self.candles_by_timeframe.get(timeframe, [])
        return candles[-limit:]

    def get_order_book(self, symbol):
        candles = self.candles_by_timeframe.get("15m", [])
        last_close = float(candles[-1][4]) if candles else 100.0
        return {"bid": last_close * 0.9995, "ask": last_close * 1.0005}


class TradeBotCoreTests(unittest.TestCase):
    def test_runtime_health_reports_missing_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            SQLiteStateStore(os.path.join(data_dir, "bot_runtime.sqlite3"))
            health = evaluate_runtime_health(tmpdir, max_snapshot_age_seconds=60)
            self.assertFalse(health.healthy)
            self.assertEqual(health.reason, "missing_runtime_snapshot")

    def test_runtime_health_accepts_fresh_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            store = SQLiteStateStore(os.path.join(data_dir, "bot_runtime.sqlite3"))
            store.persist_snapshot(
                {
                    "snapshot_key": "runtime",
                    "updated_at": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(),
                    "risk": {"allowed": True, "reason": "ok"},
                    "readiness": {"metrics": {"system_health": {"top_blocker": "healthy"}}},
                }
            )
            health = evaluate_runtime_health(tmpdir, max_snapshot_age_seconds=60)
            self.assertTrue(health.healthy)
            self.assertEqual(health.reason, "ok")

    def test_news_engine_anchor_event_ids_are_stable(self):
        engine = BinanceNewsEngine(
            exchange_client=DummyExchange(),
            state_path=os.path.join(tempfile.gettempdir(), "news_state_test.json"),
            symbols=["BNB/USDT"],
        )
        event_id_one = engine._stable_event_id("binance_delistings", "Binance Will Delist BNB", "https://www.binance.com/en/support/announcement/test")
        event_id_two = engine._stable_event_id("binance_delistings", "Binance Will Delist BNB", "https://www.binance.com/en/support/announcement/test")
        self.assertEqual(event_id_one, event_id_two)

    def test_trade_learning_engine_penalizes_repeated_bad_patterns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(BotConfig(), store)
            signal = {
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
                "strategy": "trend_breakout",
                "signal_quality": 0.72,
                "expected_edge_bps": 18.0,
                "regime": "trending",
                "rr_ratio": 2.0,
                "hurst_exponent": 0.3,
                "metadata": {"trend_direction": "bullish", "regime_confidence": 0.8},
                "research_context": {"bullish_confidence": 0.0, "bearish_confidence": 0.0, "risk_off_confidence": 0.0},
            }
            decision_context = learning.build_trade_context(signal)
            position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_breakout",
                initial_stop_loss=98.0,
                metadata={"decision_context": decision_context},
            )
            for _ in range(3):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=position,
                    close_price=98.0,
                    profit_loss=-2.0,
                    exit_reason="SL",
                )

            context = learning.learning_context_for_signal(signal)
            self.assertLess(context["score_delta"], 0.0)
            self.assertLess(context["confidence_delta"], 0.0)
            self.assertGreaterEqual(context["evidence_samples"], 3)

    def test_news_engine_builds_symbol_research_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = BinanceNewsEngine(
                exchange_client=DummyExchange(),
                state_path=os.path.join(tmpdir, "news_state.json"),
                symbols=["BNB/USDT"],
            )
            engine.store.state["commands"] = [
                {
                    "event_id": "e1",
                    "symbol": "BNB/USDT",
                    "side": "buy",
                    "action": "ENTER",
                    "confidence": 0.74,
                    "event_category": "listing",
                    "urgency": "high",
                    "created_at": utcnow_naive().isoformat(),
                    "ttl_minutes": 240,
                },
                {
                    "event_id": "e2",
                    "symbol": "BNB/USDT",
                    "side": "sell",
                    "action": "EXIT_OR_SHORT",
                    "confidence": 0.86,
                    "event_category": "delisting",
                    "urgency": "high",
                    "created_at": utcnow_naive().isoformat(),
                    "ttl_minutes": 240,
                },
            ]
            context = engine.research_signal_context("BNB/USDT")
            self.assertGreater(context["bullish_confidence"], 0.0)
            self.assertGreater(context["bearish_confidence"], 0.0)
            self.assertTrue(context["has_conflict"])

    def test_signal_engine_allows_breakout_when_hurst_clears_configured_floor(self):
        config = BotConfig(min_hurst_for_trend_breakout=0.12)
        engine = SignalEngine(config, DummyExchange())
        regime = RegimeAssessment(
            regime="trending",
            confidence=0.8,
            volatility_ratio=0.01,
            trend_strength=0.02,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={"trend_direction": "bullish"},
        )

        class StubStrategy:
            name = "trend_breakout"

            def evaluate(self, symbol, regime):
                return StrategyProposal(
                    signal=Signal(
                        symbol=symbol,
                        side="long",
                        entry_price=100.0,
                        stop_loss=98.0,
                        take_profit=104.4,
                        strategy="trend_breakout",
                        confidence=0.8,
                        timeframe="15m",
                        expected_edge_bps=18.0,
                        regime=regime.regime,
                    ),
                    expected_edge_bps=18.0,
                    rationale="unit-test breakout",
                )

        engine.strategy_modules = [StubStrategy()]
        engine.regime_engine.classify = mock.Mock(return_value=regime)
        engine.compute_hurst_exponent = mock.Mock(return_value=0.15)

        signal = engine.generate_signal("BTC/USDT")

        self.assertIsNotNone(signal)
        self.assertEqual(signal["strategy"], "trend_breakout")

    def test_signal_engine_vetoes_long_when_research_is_strongly_bearish(self):
        config = BotConfig(min_hurst_for_trend_breakout=0.12, research_conflict_veto_confidence=0.72)
        engine = SignalEngine(config, DummyExchange())
        regime = RegimeAssessment(
            regime="trending",
            confidence=0.8,
            volatility_ratio=0.01,
            trend_strength=0.02,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={"trend_direction": "bullish"},
        )

        class StubStrategy:
            name = "trend_breakout"

            def evaluate(self, symbol, regime):
                return StrategyProposal(
                    signal=Signal(
                        symbol=symbol,
                        side="long",
                        entry_price=100.0,
                        stop_loss=98.0,
                        take_profit=104.4,
                        strategy="trend_breakout",
                        confidence=0.8,
                        timeframe="15m",
                        expected_edge_bps=18.0,
                        regime=regime.regime,
                    ),
                    expected_edge_bps=18.0,
                    rationale="unit-test breakout",
                )

        engine.strategy_modules = [StubStrategy()]
        engine.regime_engine.classify = mock.Mock(return_value=regime)
        engine.compute_hurst_exponent = mock.Mock(return_value=0.2)
        engine.research_context_provider = mock.Mock(
            return_value={
                "bullish_confidence": 0.0,
                "bearish_confidence": 0.86,
                "risk_off_confidence": 0.0,
                "net_bias": -0.86,
                "has_conflict": False,
            }
        )

        signal = engine.generate_signal("BTC/USDT")

        self.assertIsNone(signal)

    def test_signal_engine_learning_context_can_veto_repeat_pattern(self):
        config = BotConfig(min_hurst_for_trend_breakout=0.12)
        engine = SignalEngine(config, DummyExchange())
        regime = RegimeAssessment(
            regime="trending",
            confidence=0.8,
            volatility_ratio=0.01,
            trend_strength=0.02,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={"trend_direction": "bullish"},
        )

        class StubStrategy:
            name = "trend_breakout"

            def evaluate(self, symbol, regime):
                return StrategyProposal(
                    signal=Signal(
                        symbol=symbol,
                        side="long",
                        entry_price=100.0,
                        stop_loss=98.0,
                        take_profit=104.4,
                        strategy="trend_breakout",
                        confidence=0.8,
                        timeframe="15m",
                        expected_edge_bps=18.0,
                        regime=regime.regime,
                    ),
                    expected_edge_bps=18.0,
                    rationale="unit-test breakout",
                )

        engine.strategy_modules = [StubStrategy()]
        engine.regime_engine.classify = mock.Mock(return_value=regime)
        engine.compute_hurst_exponent = mock.Mock(return_value=0.2)
        engine.learning_context_provider = mock.Mock(
            return_value={
                "score_delta": -8.0,
                "confidence_delta": -0.08,
                "veto": True,
                "reasons": ["thesis_invalidated"],
                "evidence_samples": 5,
            }
        )

        signal = engine.generate_signal("BTC/USDT")

        self.assertIsNone(signal)

    def test_health_transition_notifier_emits_only_once_for_same_blocker(self):
        bot = TradeBot(
            BotConfig(
                use_paper_trading=True,
                telegram_bot_token="",
                telegram_chat_id="",
                api_key="",
                api_secret="",
            )
        )
        bot.reporter = ReporterStub()
        bot._last_notified_health_blocker = "unknown"

        bot._notify_health_transition({}, {"top_blocker": "healthy"})
        bot._notify_health_transition({}, {"top_blocker": "healthy"})

        self.assertEqual(len(bot.reporter.messages), 1)

    def test_correlation_family_risk_handles_single_position_objects(self):
        config = BotConfig()
        state = BotState(balance=100.0)
        state.open_positions["ADA/USDT"] = Position(
            symbol="ADA/USDT",
            side="long",
            entry_price=1.0,
            size=1.0,
            stop_loss=0.9,
            take_profit=1.2,
        )
        risk = RiskManager(config, state)
        self.assertGreater(risk._get_family_risk("XRP"), 0.0)

    def test_execution_engine_rejects_invalid_trade_plan(self):
        config = BotConfig()
        state = BotState(balance=100.0)
        engine = ExecutionEngine(config, state, DummyExchange())
        self.assertFalse(engine.validate_trade_plan("buy", 100.0, 101.0, 110.0, 1.0))
        self.assertTrue(engine.validate_trade_plan("buy", 100.0, 95.0, 110.0, 1.0))

    def test_execution_engine_blocks_market_trade_when_slippage_is_too_high(self):
        config = BotConfig(max_slippage_fraction=0.01)
        state = BotState(balance=100.0)
        exch = SlippageExchange()
        engine = ExecutionEngine(config, state, exch)
        success = engine.place_trade(
            symbol="BTC/USDT",
            side="buy",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            is_futures=False,
            fast_move=True,
            size=1.0,
        )
        self.assertFalse(success)
        self.assertEqual(exch.orders, [])

    def test_execution_engine_rejects_short_entry_in_spot_mode(self):
        config = BotConfig(trading_mode="spot")
        state = BotState(balance=100.0)
        exch = DummyExchange()
        engine = ExecutionEngine(config, state, exch)
        success = engine.place_trade(
            symbol="BTC/USDT",
            side="short",
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=95.0,
            is_futures=False,
            fast_move=False,
            size=1.0,
        )
        self.assertFalse(success)
        self.assertEqual(exch.orders, [])

    def test_risk_manager_caps_spot_position_size_to_affordable_notional(self):
        config = BotConfig(trading_mode="spot", risk_per_trade_max=0.03)
        state = BotState(balance=100.0)
        risk = RiskManager(config, state)
        size = risk.calc_position_size(entry_price=100.0, stop_loss=99.0)
        self.assertAlmostEqual(size, 1.0, places=6)

    def test_portfolio_risk_caps_symbol_exposure(self):
        config = BotConfig(trading_mode="spot", max_symbol_exposure_fraction=0.20)
        state = BotState(balance=1000.0)
        state.open_positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.5,
            stop_loss=95.0,
            take_profit=110.0,
            strategy="trend_breakout",
        )
        risk = RiskManager(config, state)
        decision = risk.evaluate_portfolio_risk(
            symbol="BTC/USDT",
            strategy="trend_breakout",
            side="long",
            entry_price=100.0,
            proposed_size=1.0,
        )
        self.assertTrue(decision.allowed)
        self.assertAlmostEqual(decision.capped_size, 0.5, places=6)

    def test_shadow_mode_blocks_new_exposure(self):
        config = BotConfig(trading_mode="spot", operating_mode="shadow")
        state = BotState(balance=1000.0)
        risk = RiskManager(config, state)
        decision = risk.evaluate_portfolio_risk(
            symbol="ETH/USDT",
            strategy="mean_reversion",
            side="long",
            entry_price=100.0,
            proposed_size=1.0,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "shadow_read_only")

    def test_ensemble_prefers_higher_edge_breakout_over_mean_reversion(self):
        allocator = EnsembleAllocator(BotConfig())
        regime = RegimeAssessment(
            regime="trending",
            confidence=0.8,
            volatility_ratio=0.01,
            trend_strength=0.02,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={},
        )
        breakout = StrategyProposal(
            signal=Signal(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=112.0,
                strategy="trend_breakout",
                confidence=0.72,
                timeframe="15m",
                expected_edge_bps=24.0,
                regime="trending",
            ),
            expected_edge_bps=24.0,
            rationale="breakout",
        )
        mr = StrategyProposal(
            signal=Signal(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                stop_loss=98.0,
                take_profit=102.0,
                strategy="mean_reversion",
                confidence=0.66,
                timeframe="15m",
                expected_edge_bps=10.0,
                regime="trending",
            ),
            expected_edge_bps=10.0,
            rationale="mr",
        )
        decision = allocator.choose("BTC/USDT", regime, [mr, breakout])
        self.assertIsNotNone(decision.signal)
        self.assertEqual(decision.selected_strategy, "trend_breakout")

    def test_ensemble_rejects_low_edge_proposals(self):
        allocator = EnsembleAllocator(BotConfig(min_expected_edge_bps=20.0))
        regime = RegimeAssessment(
            regime="mean_reverting",
            confidence=0.7,
            volatility_ratio=0.006,
            trend_strength=0.0,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={},
        )
        proposal = StrategyProposal(
            signal=Signal(
                symbol="ETH/USDT",
                side="long",
                entry_price=100.0,
                stop_loss=98.0,
                take_profit=101.0,
                strategy="mean_reversion",
                confidence=0.7,
                timeframe="15m",
                expected_edge_bps=8.0,
                regime="mean_reverting",
            ),
            expected_edge_bps=8.0,
            rationale="weak edge",
        )
        decision = allocator.choose("ETH/USDT", regime, [proposal])
        self.assertIsNone(decision.signal)
        self.assertTrue(any("edge_below_threshold" in reason for reason in decision.rejected_reasons))

    def test_ensemble_prefers_breakout_when_regime_fit_is_stronger(self):
        allocator = EnsembleAllocator(BotConfig())
        regime = RegimeAssessment(
            regime="trending",
            confidence=0.82,
            volatility_ratio=0.01,
            trend_strength=0.02,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={"trend_direction": "bullish"},
        )
        breakout = StrategyProposal(
            signal=Signal(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                stop_loss=97.0,
                take_profit=109.0,
                strategy="trend_breakout",
                confidence=0.69,
                timeframe="15m",
                expected_edge_bps=18.0,
                regime="trending",
            ),
            expected_edge_bps=18.0,
            rationale="breakout",
        )
        mr = StrategyProposal(
            signal=Signal(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                stop_loss=98.5,
                take_profit=103.0,
                strategy="mean_reversion",
                confidence=0.74,
                timeframe="15m",
                expected_edge_bps=17.0,
                regime="trending",
            ),
            expected_edge_bps=17.0,
            rationale="mr",
        )
        decision = allocator.choose("BTC/USDT", regime, [mr, breakout])
        self.assertEqual(decision.selected_strategy, "trend_breakout")

    def test_ensemble_can_select_trend_pullback(self):
        allocator = EnsembleAllocator(BotConfig())
        regime = RegimeAssessment(
            regime="trending",
            confidence=0.78,
            volatility_ratio=0.01,
            trend_strength=0.018,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={"trend_direction": "bullish"},
        )
        pullback = StrategyProposal(
            signal=Signal(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                stop_loss=97.0,
                take_profit=107.0,
                strategy="trend_pullback",
                confidence=0.73,
                timeframe="15m",
                expected_edge_bps=16.0,
                regime="trending",
            ),
            expected_edge_bps=16.0,
            rationale="pullback",
        )
        decision = allocator.choose("BTC/USDT", regime, [pullback])
        self.assertEqual(decision.selected_strategy, "trend_pullback")

    def test_mean_reversion_strategy_skips_bearish_falling_market(self):
        closes = [100.0] * 90 + [98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 87.0, 86.5, 86.0, 85.8]
        candles_15m = []
        for idx, close in enumerate(closes):
            candles_15m.append([float(idx), close + 0.5, close + 1.0, close - 1.0, close, 1000.0])

        class Helpers:
            def rsi(self, prices, period=14):
                values = [50.0] * len(prices)
                values[-2] = 20.0
                values[-1] = 18.0
                return values

            def bollinger_bands(self, prices, period=20, std_mult=2):
                middle = [95.0] * len(prices)
                upper = [100.0] * len(prices)
                lower = [86.2] * len(prices)
                return upper, middle, lower

            def compute_atr(self, symbol, timeframe="15m", period=14):
                return 2.0

        regime = RegimeAssessment(
            regime="mean_reverting",
            confidence=0.7,
            volatility_ratio=0.01,
            trend_strength=-0.012,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={"stretch_from_mean": -0.01, "trend_direction": "bearish"},
        )
        strategy = MeanReversionStrategy(BotConfig(), StrategyExchangeStub({"15m": candles_15m}), Helpers())
        proposal = strategy.evaluate("BTC/USDT", regime)
        self.assertIsNone(proposal)

    def test_trend_breakout_strategy_can_emit_short_in_bearish_trend(self):
        candles_15m = []
        base = 100.0
        for idx in range(30):
            close = base - (idx * 0.6)
            if idx == 29:
                close -= 1.4
                candles_15m.append([float(idx), close + 1.1, close + 1.3, close - 0.2, close, 1800.0])
            else:
                candles_15m.append([float(idx), close + 0.2, close + 0.4, close - 0.8, close, 1500.0])

        class Helpers:
            def is_4h_bearish(self, symbol):
                return True

            def is_1h_downtrend(self, symbol):
                return True

            def is_4h_bullish(self, symbol):
                return False

            def is_1h_uptrend(self, symbol):
                return False

            def get_recent_swing_high_low(self, candles, lookback=5):
                recent = candles[-lookback:]
                return max(c[2] for c in recent), min(c[3] for c in recent)

            def compute_atr(self, symbol, timeframe="15m", period=14):
                return 1.5

        regime = RegimeAssessment(
            regime="trending",
            confidence=0.8,
            volatility_ratio=0.012,
            trend_strength=-0.02,
            liquidity_score=1.1,
            event_risk=False,
            unstable=False,
            metadata={"trend_direction": "bearish", "volume_impulse": 1.15},
        )
        strategy = TrendBreakoutStrategy(BotConfig(trading_mode="futures"), StrategyExchangeStub({"15m": candles_15m}), Helpers())
        proposal = strategy.evaluate("BTC/USDT", regime)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.signal.side, "short")

    def test_trend_pullback_strategy_can_emit_long_in_bullish_trend(self):
        closes = [
            100.0, 100.4, 100.9, 101.3, 101.8, 102.4, 102.9, 103.5, 104.0, 104.6,
            105.2, 105.9, 106.5, 107.0, 107.6, 108.1, 108.7, 109.2, 109.8, 110.3,
            110.9, 111.4, 112.0, 112.6, 113.2, 113.8, 114.3, 113.9, 113.4, 112.9,
            112.4, 111.9, 112.2, 112.8, 113.5, 114.1, 114.6, 114.2, 113.2, 113.4,
        ]
        candles_15m = []
        for idx, close in enumerate(closes):
            if idx == len(closes) - 1:
                candles_15m.append([float(idx), close - 0.7, close + 0.5, close - 0.3, close, 1800.0])
            else:
                candles_15m.append([float(idx), close - 0.2, close + 0.5, close - 0.5, close, 1500.0])

        class Helpers:
            def is_4h_bullish(self, symbol):
                return True

            def is_1h_uptrend(self, symbol):
                return True

            def is_4h_bearish(self, symbol):
                return False

            def is_1h_downtrend(self, symbol):
                return False

            def compute_atr(self, symbol, timeframe="15m", period=14):
                return 1.2

        regime = RegimeAssessment(
            regime="trending",
            confidence=0.8,
            volatility_ratio=0.011,
            trend_strength=0.02,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={"trend_direction": "bullish", "volume_impulse": 1.05, "stretch_from_mean": 0.01},
        )
        strategy = TrendPullbackStrategy(BotConfig(), StrategyExchangeStub({"15m": candles_15m}), Helpers())
        proposal = strategy.evaluate("BTC/USDT", regime)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.signal.side, "long")

    def test_signal_engine_rejects_signal_with_bad_mid_price_deviation(self):
        config = BotConfig()

        class BadMidExchange(DummyExchange):
            def get_order_book(self, symbol):
                return {"bid": 100.0, "ask": 100.2}

        engine = SignalEngine(config, BadMidExchange())
        regime = RegimeAssessment(
            regime="trending",
            confidence=0.8,
            volatility_ratio=0.01,
            trend_strength=0.02,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={"trend_direction": "bullish"},
        )
        signal = {
            "side": "long",
            "entry_price": 105.0,
            "stop_loss": 100.0,
            "take_profit": 115.0,
            "strategy": "trend_breakout",
            "signal_quality": 0.8,
            "expected_edge_bps": 25.0,
            "rr_ratio": 2.0,
            "hurst_exponent": 0.6,
            "metadata": {},
            "ensemble": {},
        }
        self.assertFalse(engine._passes_reliability_checks("BTC/USDT", signal, regime, proposal_count=1))

    def test_signal_engine_rejects_low_regime_confidence_signal(self):
        engine = SignalEngine(BotConfig(), DummyExchange())
        regime = RegimeAssessment(
            regime="trending",
            confidence=0.4,
            volatility_ratio=0.01,
            trend_strength=0.02,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={"trend_direction": "bullish"},
        )
        signal = {
            "side": "long",
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit": 104.0,
            "strategy": "trend_pullback",
            "signal_quality": 0.75,
            "expected_edge_bps": 18.0,
            "rr_ratio": 2.0,
            "hurst_exponent": 0.6,
            "metadata": {},
            "ensemble": {},
        }
        self.assertFalse(engine._passes_reliability_checks("BTC/USDT", signal, regime, proposal_count=1))

    def test_tradebot_assesses_stale_market_data_as_unhealthy(self):
        bot = TradeBot(
            BotConfig(
                use_paper_trading=True,
                telegram_bot_token="",
                telegram_chat_id="",
                api_key="",
                api_secret="",
            )
        )

        class StaleExchange:
            def fetch_ohlcv(self, symbol, timeframe, limit=3):
                old_ts_ms = (dt.datetime.now().timestamp() - (5 * 3600)) * 1000.0
                return [[old_ts_ms, 100.0, 101.0, 99.0, 100.5, 1000.0]]

        bot.exch = StaleExchange()
        ok, reason = bot._assess_market_data_health("BTC/USDT")
        self.assertFalse(ok)
        self.assertEqual(reason, "stale_market_data")

    def test_tradebot_safe_generate_signal_handles_exceptions(self):
        bot = TradeBot(
            BotConfig(
                use_paper_trading=True,
                telegram_bot_token="",
                telegram_chat_id="",
                api_key="",
                api_secret="",
            )
        )
        bot.reporter = ReporterStub()

        class HealthyExchange(DummyExchange):
            def fetch_ohlcv(self, symbol, timeframe, limit=3):
                ts_ms = dt.datetime.now().timestamp() * 1000.0
                return [[ts_ms, 100.0, 101.0, 99.0, 100.5, 1000.0] for _ in range(limit)]

        class BrokenSignals:
            def generate_signal(self, symbol):
                raise RuntimeError("boom")

        bot.exch = HealthyExchange()
        bot.signals = BrokenSignals()
        signal = bot._safe_generate_signal("BTC/USDT")
        self.assertIsNone(signal)
        self.assertEqual(bot.state.market_regime_alerts.get("BTC/USDT"), "signal_generation_error")

    def test_tradebot_system_health_summary_includes_circuit_breakers(self):
        bot = TradeBot(
            BotConfig(
                use_paper_trading=True,
                telegram_bot_token="",
                telegram_chat_id="",
                api_key="",
                api_secret="",
            )
        )
        bot.state.data_cooldown_until = dt.datetime.now() + dt.timedelta(minutes=5)
        bot.state.execution_cooldown_until = dt.datetime.now() + dt.timedelta(minutes=5)
        summary = bot._build_system_health_summary()
        self.assertTrue(summary["data_circuit_breaker_active"])
        self.assertTrue(summary["execution_circuit_breaker_active"])

    def test_tradebot_operator_response_includes_system_health(self):
        bot = TradeBot(
            BotConfig(
                use_paper_trading=True,
                telegram_bot_token="",
                telegram_chat_id="",
                api_key="",
                api_secret="",
            )
        )
        response = bot.answer_operator_question("why are you not trading?")
        self.assertIn("system_health", response)

    def test_tradebot_status_report_lists_top_blocker(self):
        bot = TradeBot(
            BotConfig(
                use_paper_trading=True,
                telegram_bot_token="",
                telegram_chat_id="",
                api_key="",
                api_secret="",
            )
        )
        bot.state.emergency_mode = True
        report = bot.render_status_report()
        self.assertIn("Top blocker: emergency_mode", report)

    def test_cli_parser_supports_status_mode(self):
        parser = build_parser()
        args = parser.parse_args(["status"])
        self.assertEqual(args.mode, "status")

    def test_cli_live_mode_defaults_to_testnet_execution(self):
        with mock.patch.dict(
            os.environ,
            {"BINANCE_API_KEY": "key", "BINANCE_API_SECRET": "secret"},
            clear=False,
        ):
            config = build_runtime_config(paper_mode=False, trading_mode="spot")
        self.assertFalse(config.use_paper_trading)
        self.assertEqual(config.trading_mode, "spot")
        self.assertEqual(config.api_key, "key")

    def test_cli_live_mode_requires_testnet_credentials(self):
        with mock.patch.dict(
            os.environ,
            {"BINANCE_API_KEY": "", "BINANCE_API_SECRET": ""},
            clear=False,
        ):
            with self.assertRaises(RuntimeError):
                build_runtime_config(paper_mode=False, trading_mode="spot")

    def test_news_engine_records_research_event_to_sqlite_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            engine = BinanceNewsEngine(
                exchange_client=DummyExchange(),
                state_path=os.path.join(tmpdir, "news_state.json"),
                symbols=["BTC/USDT"],
                research_store=store,
            )
            event = NewsEvent(
                event_id="evt-1",
                source="binance_new_listings",
                title="Binance Will List BTC",
                url="https://example.com",
                published_at="2026-01-01T00:00:00",
                category="listing",
                assets=["BTC"],
                summary="Listing",
            )
            command = TradeCommand(
                command_id="cmd-1",
                event_id="evt-1",
                action="ENTER",
                symbol="BTC/USDT",
                side="buy",
                confidence=0.8,
                urgency="high",
                ttl_minutes=60,
                rationale="Listing-driven upside",
                event_category="listing",
                created_at="2026-01-01T00:00:00",
                metadata={"semantic_cluster": "listing_launch", "market_context": {}, "event_labels": {"direction_bias": "bullish"}},
            )
            engine._record_research_event(event, command)
            rows = store.load_recent_research_events(limit=5)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_id"], "evt-1")
            self.assertEqual(rows[0]["status"], "pending_outcome")

    def test_tradebot_operator_response_includes_event_research(self):
        bot = TradeBot(
            BotConfig(
                use_paper_trading=True,
                telegram_bot_token="",
                telegram_chat_id="",
                api_key="",
                api_secret="",
            )
        )
        response = bot.answer_operator_question("show news event risk")
        self.assertIn("event_risk", response)
        self.assertIn("event_research", response)

    def test_readiness_report_blocks_promotion_when_metrics_are_weak(self):
        bot = TradeBot(
            BotConfig(
                use_paper_trading=True,
                operating_mode="paper",
                telegram_bot_token="",
                telegram_chat_id="",
                api_key="",
                api_secret="",
            )
        )
        bot.state_store.set_operational_metric("runtime_hours", 1.0)
        bot.state_store.set_operational_metric("reconciliation_failures", 1)
        bot.state_store.set_operational_metric("risk_halts", 3)
        bot.state_store.set_operational_metric("closed_trades", 2)
        bot.state_store.set_operational_metric("win_rate_pct", 10.0)
        bot.state_store.set_operational_metric("profit_factor", 0.2)
        bot.state_store.set_operational_metric("max_drawdown_pct", 25.0)
        report = build_readiness_report(bot)
        self.assertFalse(report.ready)
        self.assertIsNone(report.recommended_next_mode)

    def test_operator_response_includes_readiness_report(self):
        bot = TradeBot(
            BotConfig(
                use_paper_trading=True,
                operating_mode="paper",
                telegram_bot_token="",
                telegram_chat_id="",
                api_key="",
                api_secret="",
            )
        )
        response = bot.answer_operator_question("are we ready to promote to shadow mode?")
        self.assertIn("readiness_report", response)

    def test_tradebot_save_state_is_atomic_and_loadable(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                bot = TradeBot(
                    BotConfig(
                        use_paper_trading=True,
                        telegram_bot_token="",
                        telegram_chat_id="",
                        api_key="",
                        api_secret="",
                    )
                )
                bot.state.balance = 123.45
                bot.save_state()
                self.assertTrue(os.path.exists("bot_state.json"))

                bot2 = TradeBot(
                    BotConfig(
                        use_paper_trading=True,
                        telegram_bot_token="",
                        telegram_chat_id="",
                        api_key="",
                        api_secret="",
                    )
                )
                self.assertAlmostEqual(bot2.state.balance, 123.45, places=2)
            finally:
                os.chdir(cwd)

    def test_tradebot_state_reload_preserves_open_positions_and_runtime_fields(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                bot = TradeBot(
                    BotConfig(
                        use_paper_trading=True,
                        telegram_bot_token="",
                        telegram_chat_id="",
                        api_key="",
                        api_secret="",
                    )
                )
                bot.state.pending_news_commands = [{"command_id": "c1"}]
                bot.state.market_regime_alerts = {"BTC/USDT": "atr_spike"}
                bot.state.data_cooldown_until = dt.datetime.now() + dt.timedelta(minutes=15)
                bot.state.execution_cooldown_until = dt.datetime.now() + dt.timedelta(minutes=20)
                bot.state.health_summary = {"status": "testing"}
                bot.state.open_positions["BTC/USDT"] = [
                    Position(
                        symbol="BTC/USDT",
                        side="long",
                        entry_price=100.0,
                        size=2.0,
                        stop_loss=95.0,
                        take_profit=110.0,
                    )
                ]
                bot.save_state()

                bot2 = TradeBot(
                    BotConfig(
                        use_paper_trading=True,
                        telegram_bot_token="",
                        telegram_chat_id="",
                        api_key="",
                        api_secret="",
                    )
                )
                self.assertIn("BTC/USDT", bot2.state.open_positions)
                self.assertEqual(len(bot2.state.open_positions["BTC/USDT"]), 1)
                self.assertEqual(bot2.state.pending_news_commands, [{"command_id": "c1"}])
                self.assertEqual(bot2.state.market_regime_alerts, {"BTC/USDT": "atr_spike"})
                self.assertIsNotNone(bot2.state.data_cooldown_until)
                self.assertIsNotNone(bot2.state.execution_cooldown_until)
                self.assertEqual(bot2.state.health_summary, {"status": "testing"})
            finally:
                os.chdir(cwd)

    def test_tradebot_load_state_preserves_runtime_hours_metric(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                bot = TradeBot(
                    BotConfig(
                        use_paper_trading=True,
                        telegram_bot_token="",
                        telegram_chat_id="",
                        api_key="",
                        api_secret="",
                    )
                )
                bot.state_store.set_operational_metric("runtime_hours", 12.5)

                bot2 = TradeBot(
                    BotConfig(
                        use_paper_trading=True,
                        telegram_bot_token="",
                        telegram_chat_id="",
                        api_key="",
                        api_secret="",
                    )
                )
                metrics = bot2.state_store.load_operational_metrics()
                self.assertEqual(metrics.get("runtime_hours"), 12.5)
            finally:
                os.chdir(cwd)

    def test_manage_open_positions_closes_take_profit_and_updates_balance(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                bot = TradeBot(
                    BotConfig(
                        use_paper_trading=True,
                        telegram_bot_token="",
                        telegram_chat_id="",
                        api_key="",
                        api_secret="",
                    )
                )
                bot.reporter = ReporterStub()
                bot.exch = DummyExchange()

                class SignalStub:
                    def compute_atr(self, symbol, timeframe="15m", period=14):
                        return 1.0

                bot.signals = SignalStub()
                bot.state.balance = 100.0
                bot.state.open_positions["BTC/USDT"] = [
                    Position(
                        symbol="BTC/USDT",
                        side="long",
                        entry_price=100.0,
                        size=2.0,
                        stop_loss=95.0,
                        take_profit=110.0,
                        initial_stop_loss=95.0,
                    )
                ]

                bot.manage_open_positions()

                self.assertNotIn("BTC/USDT", bot.state.open_positions)
                self.assertAlmostEqual(bot.state.balance, 118.0, places=2)
                self.assertEqual(bot.state.today_trades_count, 1)
                self.assertTrue(any("CLOSE" in entry for entry in bot.reporter.trades))
            finally:
                os.chdir(cwd)

    def test_run_once_enters_emergency_mode_on_daily_loss_limit(self):
        bot = TradeBot(
            BotConfig(
                use_paper_trading=True,
                telegram_bot_token="",
                telegram_chat_id="",
                api_key="",
                api_secret="",
            )
        )
        bot.reporter = ReporterStub()
        bot.state.balance = 80.0
        bot.state.equity_start_of_day = 100.0
        bot.state.peak_equity = 100.0
        bot.manage_open_positions = mock.Mock()
        bot.maybe_run_news_engine = mock.Mock(return_value=[])

        class NoSpikeSignals:
            def is_atr_spike(self, symbol, timeframe="15m", period=14, spike_mult=3.0):
                return False

            def generate_signal(self, symbol):
                return None

        bot.signals = NoSpikeSignals()
        bot.run_once()

        self.assertTrue(bot.state.emergency_mode)
        bot.manage_open_positions.assert_not_called()
        self.assertTrue(any("Trading halted" in msg for _, msg in bot.reporter.messages))

    def test_scan_binance_news_persists_generated_commands(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                bot = TradeBot(
                    BotConfig(
                        use_paper_trading=True,
                        telegram_bot_token="",
                        telegram_chat_id="",
                        api_key="",
                        api_secret="",
                    )
                )
                bot.reporter = ReporterStub()
                from trade_bot.news_engine import TradeCommand

                bot.news_engine.scan = lambda: [
                    TradeCommand(
                        command_id="c1",
                        event_id="e1",
                        action="watch",
                        symbol="BTC/USDT",
                        side="buy",
                        confidence=0.8,
                        urgency="high",
                        ttl_minutes=60,
                        rationale="listing",
                        event_category="listing",
                        created_at="2025-01-01T00:00:00",
                        metadata={},
                    )
                ]
                commands = bot.scan_binance_news()
                self.assertEqual(len(commands), 1)
                self.assertEqual(bot.state.pending_news_commands[-1]["command_id"], "c1")
                self.assertIsNotNone(bot.state.last_news_scan_at)
            finally:
                os.chdir(cwd)

    def test_backtest_report_is_repeatable_for_same_inputs(self):
        report_one = build_backtest_report(
            symbol="BTC/USDT",
            timeframe="15m",
            days=90,
            starting_balance=10000.0,
            metrics={"total_return_pct": 12.3, "raw_trades": 5},
            assumptions={"fee_bps": 10.0},
        )
        report_two = build_backtest_report(
            symbol="BTC/USDT",
            timeframe="15m",
            days=90,
            starting_balance=10000.0,
            metrics={"total_return_pct": 12.3, "raw_trades": 5},
            assumptions={"fee_bps": 10.0},
        )
        self.assertEqual(report_one, report_two)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "backtest_results.json")
            save_backtest_report(path, report_one)
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        self.assertEqual(loaded, report_one)

    def test_mock_backtest_exchange_only_exposes_history_up_to_cursor(self):
        import pandas as pd

        df = pd.DataFrame(
            [
                {"timestamp": "2025-01-01T00:00:00", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
                {"timestamp": "2025-01-01T01:00:00", "open": 101, "high": 102, "low": 100, "close": 101, "volume": 1},
                {"timestamp": "2025-01-01T02:00:00", "open": 102, "high": 103, "low": 101, "close": 102, "volume": 1},
            ]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        exch = MockBacktestExchange(df, "BTC/USDT")
        exch.set_cursor(1)
        candles = exch.fetch_ohlcv("BTC/USDT", "1h", limit=10)

        self.assertEqual(len(candles), 2)
        self.assertEqual(candles[-1][4], 101)

    def test_backtest_engine_supports_short_signal_paths(self):
        import pandas as pd

        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        closes = [100.0] * 100 + [100.0, 95.0, 95.0] + [95.0] * 27
        rows = []
        for idx, ts in enumerate(timestamps):
            close = closes[idx]
            row = {
                "timestamp": ts,
                "open": close,
                "high": close if idx != 101 else 100.0,
                "low": close if idx != 101 else 94.0,
                "close": close,
                "volume": 1000.0,
            }
            rows.append(row)
        df = pd.DataFrame(rows).set_index("timestamp")

        engine = BacktestEngine(BotConfig(starting_balance=1000.0, trading_mode="futures"))

        class ShortOnlySignals:
            def __init__(self, config, exch):
                self.exch = exch

            def generate_signal(self, symbol):
                candles = self.exch.fetch_ohlcv(symbol, "15m", limit=200)
                if len(candles) == 101:
                    return {
                        "side": "short",
                        "entry_price": 100.0,
                        "stop_loss": 101.0,
                        "take_profit": 95.0,
                        "strategy": "unit_test_short",
                    }
                return None

        with mock.patch("trade_bot.main.SignalEngine", ShortOnlySignals):
            engine.load_historical_data = mock.Mock(return_value=df)
            result = engine.run_backtest("BTC/USDT", timeframe="15m", days=10)

        self.assertEqual(result["num_trades"], 1)
        self.assertEqual(result["short_trades"], 1)

    def test_backtest_engine_ignores_spot_short_signals_without_crashing(self):
        import pandas as pd

        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        rows = []
        for ts in timestamps:
            rows.append(
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
            )
        df = pd.DataFrame(rows).set_index("timestamp")

        engine = BacktestEngine(BotConfig(starting_balance=1000.0, trading_mode="spot"))

        class SpotShortOnlySignals:
            def __init__(self, config, exch):
                self.exch = exch

            def generate_signal(self, symbol):
                candles = self.exch.fetch_ohlcv(symbol, "15m", limit=200)
                if len(candles) == 101:
                    return {
                        "side": "short",
                        "entry_price": 100.0,
                        "stop_loss": 101.0,
                        "take_profit": 95.0,
                        "strategy": "unit_test_spot_short",
                    }
                return None

        with mock.patch("trade_bot.main.SignalEngine", SpotShortOnlySignals):
            engine.load_historical_data = mock.Mock(return_value=df)
            result = engine.run_backtest("BTC/USDT", timeframe="15m", days=10)

        self.assertEqual(result["num_trades"], 0)
        self.assertEqual(result["short_trades"], 0)

    def test_conversation_api_requires_explicit_bootstrap_secrets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "CONVERSATION_API_JWT_SECRET": "",
                "CONVERSATION_API_DEFAULT_TENANT_ID": "",
                "CONVERSATION_API_DEFAULT_CLIENT_ID": "",
                "CONVERSATION_API_DEFAULT_CLIENT_SECRET": "",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                with self.assertRaises(RuntimeError):
                    build_conversation_api(base_dir=tmpdir)

    def test_conversation_api_masks_internal_errors(self):
        class FailingService:
            def issue_access_token(self, client_id, client_secret):
                raise ValueError(f"secret exploded for {client_id}:{client_secret}")

        class DummyStore:
            def log_request(self, **kwargs):
                self.logged = kwargs

        app = ConversationAPIApp(service=FailingService(), store=DummyStore())
        body = b'{"client_id":"alpha","client_secret":"beta"}'
        environ = {
            "PATH_INFO": "/v1/auth/token",
            "REQUEST_METHOD": "POST",
            "CONTENT_LENGTH": str(len(body)),
            "wsgi.input": io.BytesIO(body),
        }
        status_holder = {}

        def start_response(status, headers):
            status_holder["status"] = status
            status_holder["headers"] = headers

        response_body = b"".join(app(environ, start_response))
        payload = response_body.decode("utf-8")

        self.assertEqual(status_holder["status"], "500 Internal Server Error")
        self.assertIn("Internal server error", payload)
        self.assertNotIn("secret exploded", payload)
        self.assertNotIn("alpha", payload)
        self.assertNotIn("beta", payload)


if __name__ == "__main__":
    unittest.main()
