import datetime as dt
import io
import json
import os
import random
import sqlite3
import tempfile
import unittest
from unittest import mock

import pandas as pd

from trade_bot.backtest_reporting import build_backtest_report, build_batch_summary, build_campaign_comparison, save_backtest_report
from trade_bot.conversation_api import ConversationAPIApp, build_conversation_api
from trade_bot.ensemble import EnsembleAllocator
from trade_bot.health import evaluate_runtime_health
from trade_bot.cli import build_parser, build_runtime_config
from trade_bot.learning import TradeLearningEngine
from trade_bot.main import BacktestEngine, BotConfig, BotState, ExecutionEngine, MockBacktestExchange, Position, RiskManager, SignalEngine, TradeBot
from trade_bot.models import Signal
from trade_bot.process_service import read_bot_status, write_bot_status
from trade_bot.regime import MarketRegimeEngine, RegimeAssessment
from trade_bot.news_engine import BinanceNewsEngine, NewsEvent, TradeCommand
from trade_bot.persistence import _sanitize_jsonish, load_bot_state, save_bot_state
from trade_bot.readiness import build_readiness_report
from trade_bot.simulation import HistoricalReplayExchange, HistoricalSimulationEngine, SimulatedExecutionVenue, SimulatedOrder, purged_walk_forward_splits
from trade_bot.simulation_service import read_simulation_batch_status, read_simulation_status, request_simulation_stop, simulation_runtime_dir, write_simulation_batch_status
from trade_bot.strategies import MeanReversionStrategy, StrategyProposal, TrendBreakoutStrategy, TrendPullbackStrategy
from trade_bot.state_store import PersistentLearningStore, SQLiteStateStore, backfill_learning_from_sqlite_artifacts
from trade_bot.telegram_operator import _chat_allowed, _handle_command, _record_command_audit, _record_unauthorized_update, _supported_commands


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


class ShadowExchangeStub(DummyExchange):
    def __init__(self, candles):
        super().__init__()
        self._candles = candles

    def fetch_ohlcv(self, symbol, timeframe, limit=1):
        return self._candles[-limit:]

    def get_order_book(self, symbol):
        last_close = float(self._candles[-1][4])
        return {"bid": last_close * 0.999, "ask": last_close * 1.001}


class QuietSignalEngine:
    def __init__(self, config, exch):
        self.config = config
        self.exch = exch
        self.learning_context_provider = None

    def generate_signal(self, symbol):
        return None


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

    def test_runtime_health_blocks_stale_bot_process_status(self):
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
            write_bot_status(
                tmpdir,
                {
                    "status": "running",
                    "pid": 999999,
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            health = evaluate_runtime_health(tmpdir, max_snapshot_age_seconds=60)
            self.assertFalse(health.healthy)
            self.assertEqual(health.reason, "bot_process_unhealthy:process_not_running")

    def test_runtime_health_blocks_stale_simulation_process_status(self):
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
            write_simulation_batch_status(
                tmpdir,
                {
                    "status": "running",
                    "pid": 999999,
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            health = evaluate_runtime_health(tmpdir, max_snapshot_age_seconds=60)
            self.assertFalse(health.healthy)
            self.assertEqual(health.reason, "simulation_process_unhealthy:process_not_running")

    def test_sanitize_jsonish_replaces_recursive_references(self):
        loop = {}
        loop["self"] = loop
        sanitized = _sanitize_jsonish(loop)
        self.assertEqual(sanitized["self"], "<recursive-ref>")

    def test_save_bot_state_handles_recursive_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "state.json")
            state = BotState(balance=1000.0)
            loop = {}
            loop["self"] = loop
            loop["items"] = [loop]
            state.health_summary = {"loop": loop}
            state.open_positions["BTC/USDT"] = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=95.0,
                take_profit=110.0,
                metadata={"cycle": loop},
            )

            save_bot_state(state, path=state_path)

            with open(state_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertEqual(payload["health_summary"]["loop"]["self"], "<recursive-ref>")
            self.assertEqual(payload["open_positions"][0]["metadata"]["cycle"]["self"], "<recursive-ref>")

            restored = BotState(balance=0.0)
            loaded = load_bot_state(restored, path=state_path)
            self.assertTrue(loaded)
            self.assertEqual(restored.health_summary["loop"]["self"], "<recursive-ref>")
            self.assertEqual(
                restored.open_positions["BTC/USDT"][0].metadata["cycle"]["self"],
                "<recursive-ref>",
            )

    def test_state_store_persist_snapshot_handles_recursive_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            loop = {}
            loop["self"] = loop
            snapshot = {
                "snapshot_key": "runtime",
                "updated_at": dt.datetime.now().isoformat(),
                "payload": {"loop": loop},
            }

            store.persist_snapshot(snapshot)
            loaded = store.load_snapshot("runtime")

            self.assertEqual(loaded["payload"]["loop"]["self"], "<recursive-ref>")

    def test_build_batch_summary_aggregates_completed_runs(self):
        summary = build_batch_summary(
            [
                {
                    "artifact_dir": "run1",
                    "num_trades": 4,
                    "raw_signals": 20,
                    "win_rate_pct": 50.0,
                    "total_return_pct": 1.2,
                    "profit_factor": 1.4,
                    "max_drawdown_pct": -2.0,
                    "decision_diagnostics": {
                        "signals_by_strategy": {"trend_breakout": 3, "mean_reversion": 1},
                        "skip_reasons": {"trend_breakout:negative_net_expectancy": 2},
                    },
                },
                {
                    "artifact_dir": "run2",
                    "num_trades": 6,
                    "raw_signals": 30,
                    "win_rate_pct": 55.0,
                    "total_return_pct": -0.4,
                    "profit_factor": 0.8,
                    "max_drawdown_pct": -3.0,
                    "decision_diagnostics": {
                        "signals_by_strategy": {"trend_breakout": 2, "trend_pullback": 4},
                        "skip_reasons": {"trend_breakout:negative_net_expectancy": 1, "mean_reversion:regime_blocked": 3},
                    },
                },
            ]
        )
        self.assertEqual(summary["num_runs"], 2)
        self.assertAlmostEqual(summary["aggregates"]["num_trades"]["avg"], 5.0)
        self.assertEqual(summary["best_run"]["artifact_dir"], "run1")
        self.assertEqual(summary["worst_run"]["artifact_dir"], "run2")
        self.assertEqual(summary["decision_totals"]["signals_by_strategy"]["trend_breakout"], 5)
        self.assertEqual(summary["decision_totals"]["skip_reasons"]["mean_reversion:regime_blocked"], 3)
        self.assertEqual(summary["comparisons"]["baseline_vs_latest"]["baseline_artifact_dir"], "run1")
        self.assertEqual(summary["comparisons"]["baseline_vs_latest"]["candidate_artifact_dir"], "run2")
        self.assertIn("trades_per_day", summary["comparisons"]["baseline_vs_latest"]["metrics"])
        self.assertEqual(summary["median_run"]["artifact_dir"], "run1")
        self.assertEqual(summary["comparisons"]["baseline_vs_median"]["candidate_artifact_dir"], "run1")
        self.assertAlmostEqual(summary["stability"]["return_range_pct"], 1.6)
        self.assertAlmostEqual(summary["stability"]["best_vs_median_return_gap_pct"], 0.0)
        self.assertGreater(summary["stability"]["return_stddev_pct"], 0.0)

    def test_build_batch_summary_aggregates_symbol_level_decision_totals(self):
        summary = build_batch_summary(
            [
                {
                    "artifact_dir": "run1",
                    "num_trades": 1,
                    "raw_signals": 2,
                    "win_rate_pct": 100.0,
                    "total_return_pct": 0.4,
                    "profit_factor": 2.0,
                    "max_drawdown_pct": -0.2,
                    "decision_diagnostics": {
                        "signals_by_order_type": {"limit": 2},
                        "signals_by_strategy": {"trend_pullback": 2},
                        "submitted_by_strategy": {"trend_pullback": 1},
                        "filled_by_strategy": {"trend_pullback": 1},
                        "closed_by_strategy": {"trend_pullback": 1},
                        "submitted_by_order_type": {"limit": 1},
                        "closed_by_order_type": {"limit": 1},
                        "limit_to_market_upgrades": 1,
                        "limit_to_market_upgrades_by_strategy": {"trend_pullback": 1},
                        "limit_queue_priority_assists": 2,
                        "limit_queue_priority_assists_by_strategy": {"trend_pullback": 2},
                        "limit_latency_reductions": 1,
                        "limit_latency_reductions_by_strategy": {"trend_pullback": 1},
                        "stale_market_escalations": 2,
                        "stale_market_escalations_by_strategy": {"trend_pullback": 2},
                        "repriced_orders": 2,
                        "repriced_by_strategy": {"trend_pullback": 2},
                        "touch_escalations": 1,
                        "touch_escalations_by_strategy": {"trend_pullback": 1},
                        "partial_profit_takes": 2,
                        "partial_profit_takes_by_strategy": {"trend_pullback": 2},
                        "raw_signals_by_symbol": {"BTC/USDT": 1, "ETH/USDT": 1},
                        "filled_by_symbol": {"BTC/USDT": 1},
                        "closed_by_symbol": {"BTC/USDT": 1},
                        "skip_reasons_by_symbol": {"ETH/USDT": {"max_open_positions": 1}},
                    },
                },
                {
                    "artifact_dir": "run2",
                    "num_trades": 2,
                    "raw_signals": 3,
                    "win_rate_pct": 50.0,
                    "total_return_pct": 0.1,
                    "profit_factor": 1.2,
                    "max_drawdown_pct": -0.4,
                    "decision_diagnostics": {
                        "signals_by_order_type": {"market": 1, "limit": 2},
                        "signals_by_strategy": {"trend_breakout": 2},
                        "submitted_by_strategy": {"trend_breakout": 1},
                        "filled_by_strategy": {"trend_breakout": 1},
                        "closed_by_strategy": {"trend_breakout": 1},
                        "submitted_by_order_type": {"market": 1},
                        "closed_by_order_type": {"market": 1},
                        "limit_to_market_upgrades": 2,
                        "limit_to_market_upgrades_by_strategy": {"mean_reversion": 2},
                        "limit_queue_priority_assists": 1,
                        "limit_queue_priority_assists_by_strategy": {"trend_breakout": 1},
                        "limit_latency_reductions": 2,
                        "limit_latency_reductions_by_strategy": {"trend_breakout": 2},
                        "stale_market_escalations": 1,
                        "stale_market_escalations_by_strategy": {"trend_breakout": 1},
                        "repriced_orders": 1,
                        "repriced_by_strategy": {"mean_reversion": 1},
                        "touch_escalations": 2,
                        "touch_escalations_by_strategy": {"mean_reversion": 2},
                        "partial_profit_takes": 1,
                        "partial_profit_takes_by_strategy": {"mean_reversion": 1},
                        "raw_signals_by_symbol": {"BTC/USDT": 1, "SOL/USDT": 2},
                        "filled_by_symbol": {"SOL/USDT": 1},
                        "closed_by_symbol": {"SOL/USDT": 1},
                        "skip_reasons_by_symbol": {"ETH/USDT": {"negative_net_expectancy": 2}},
                    },
                },
            ]
        )
        self.assertEqual(summary["decision_totals"]["raw_signals_by_symbol"]["BTC/USDT"], 2)
        self.assertEqual(summary["decision_totals"]["raw_signals_by_symbol"]["SOL/USDT"], 2)
        self.assertEqual(summary["decision_totals"]["signals_by_order_type"]["limit"], 4)
        self.assertEqual(summary["decision_totals"]["submitted_by_order_type"]["market"], 1)
        self.assertEqual(summary["decision_totals"]["closed_by_order_type"]["market"], 1)
        self.assertEqual(summary["execution_totals"]["limit_to_market_upgrades"], 3)
        self.assertEqual(summary["execution_totals"]["limit_queue_priority_assists"], 3)
        self.assertEqual(summary["execution_totals"]["limit_latency_reductions"], 3)
        self.assertEqual(summary["execution_totals"]["stale_market_escalations"], 3)
        self.assertEqual(summary["execution_totals"]["repriced_orders"], 3)
        self.assertEqual(summary["execution_totals"]["touch_escalations"], 3)
        self.assertEqual(summary["execution_totals"]["partial_profit_takes"], 3)
        self.assertEqual(summary["decision_totals"]["limit_to_market_upgrades_by_strategy"]["mean_reversion"], 2)
        self.assertEqual(summary["decision_totals"]["limit_queue_priority_assists_by_strategy"]["trend_pullback"], 2)
        self.assertEqual(summary["decision_totals"]["limit_queue_priority_assists_by_strategy"]["trend_breakout"], 1)
        self.assertEqual(summary["decision_totals"]["limit_latency_reductions_by_strategy"]["trend_pullback"], 1)
        self.assertEqual(summary["decision_totals"]["limit_latency_reductions_by_strategy"]["trend_breakout"], 2)
        self.assertEqual(summary["decision_totals"]["stale_market_escalations_by_strategy"]["trend_pullback"], 2)
        self.assertEqual(summary["decision_totals"]["stale_market_escalations_by_strategy"]["trend_breakout"], 1)
        self.assertEqual(summary["decision_totals"]["repriced_by_strategy"]["trend_pullback"], 2)
        self.assertEqual(summary["decision_totals"]["touch_escalations_by_strategy"]["mean_reversion"], 2)
        self.assertEqual(summary["decision_totals"]["partial_profit_takes_by_strategy"]["trend_pullback"], 2)
        self.assertEqual(summary["decision_totals"]["partial_profit_takes_by_strategy"]["mean_reversion"], 1)
        self.assertEqual(summary["fill_conversion_by_strategy"]["trend_pullback"]["signals"], 2)
        self.assertEqual(summary["fill_conversion_by_strategy"]["trend_pullback"]["submitted"], 1)
        self.assertEqual(summary["fill_conversion_by_strategy"]["trend_pullback"]["filled"], 1)
        self.assertEqual(summary["fill_conversion_by_strategy"]["trend_pullback"]["closed"], 1)
        self.assertAlmostEqual(summary["fill_conversion_by_strategy"]["trend_pullback"]["submit_rate_pct"], 50.0)
        self.assertAlmostEqual(summary["fill_conversion_by_strategy"]["trend_pullback"]["fill_rate_pct"], 100.0)
        self.assertEqual(summary["fill_conversion_by_strategy"]["trend_breakout"]["signals"], 2)
        self.assertEqual(summary["fill_conversion_by_strategy"]["trend_breakout"]["submitted"], 1)
        self.assertEqual(summary["fill_conversion_by_strategy"]["trend_breakout"]["filled"], 1)
        self.assertEqual(summary["fill_conversion_by_strategy"]["trend_breakout"]["closed"], 1)
        self.assertAlmostEqual(summary["fill_conversion_by_strategy"]["trend_breakout"]["submit_rate_pct"], 50.0)
        self.assertEqual(summary["decision_totals"]["filled_by_symbol"]["BTC/USDT"], 1)
        self.assertEqual(summary["decision_totals"]["filled_by_symbol"]["SOL/USDT"], 1)
        self.assertEqual(summary["skip_reasons_by_symbol"]["ETH/USDT"]["max_open_positions"], 1)
        self.assertEqual(summary["skip_reasons_by_symbol"]["ETH/USDT"]["negative_net_expectancy"], 2)

    def test_build_batch_summary_aggregates_family_rotation_actions(self):
        summary = build_batch_summary(
            [
                {
                    "artifact_dir": "run1",
                    "num_trades": 1,
                    "raw_signals": 2,
                    "win_rate_pct": 50.0,
                    "total_return_pct": 0.5,
                    "decision_diagnostics": {
                        "family_rotation_counts": {"promote_current": 2, "suppress_current": 1, "promote_current_hard": 1},
                        "family_rotation_counts_by_strategy": {
                            "trend_pullback": {"promote_current": 2, "promote_current_hard": 1},
                            "trend_breakout": {"suppress_current": 1},
                        },
                        "family_rotation_recovery_counts": {"recovery_active": 1},
                        "family_rotation_recovery_counts_by_strategy": {
                            "trend_breakout": {"recovery_active": 1},
                        },
                    },
                },
                {
                    "artifact_dir": "run2",
                    "num_trades": 1,
                    "raw_signals": 1,
                    "win_rate_pct": 0.0,
                    "total_return_pct": -0.5,
                    "decision_diagnostics": {
                        "family_rotation_counts": {"suppress_current": 2, "suppress_current_hard": 1},
                        "family_rotation_counts_by_strategy": {
                            "trend_breakout": {"suppress_current": 2, "suppress_current_hard": 1},
                        },
                    },
                },
            ]
        )
        self.assertEqual(summary["decision_totals"]["family_rotation_counts"]["promote_current"], 2)
        self.assertEqual(summary["decision_totals"]["family_rotation_counts"]["suppress_current"], 3)
        self.assertEqual(summary["decision_totals"]["family_rotation_counts"]["promote_current_hard"], 1)
        self.assertEqual(summary["decision_totals"]["family_rotation_counts"]["suppress_current_hard"], 1)
        self.assertEqual(summary["family_rotation_by_strategy"]["trend_pullback"]["promote_current"], 2)
        self.assertEqual(summary["family_rotation_by_strategy"]["trend_breakout"]["suppress_current"], 3)
        self.assertEqual(summary["family_rotation_soft_by_strategy"]["trend_pullback"]["promote_current"], 2)
        self.assertEqual(summary["family_rotation_hard_by_strategy"]["trend_pullback"]["promote_current_hard"], 1)
        self.assertEqual(summary["family_rotation_hard_by_strategy"]["trend_breakout"]["suppress_current_hard"], 1)
        self.assertEqual(summary["decision_totals"]["family_rotation_recovery_counts"]["recovery_active"], 1)
        self.assertEqual(summary["family_rotation_recovery_by_strategy"]["trend_breakout"]["recovery_active"], 1)

    def test_build_batch_summary_aggregates_realized_expectancy_by_symbol_and_strategy(self):
        summary = build_batch_summary(
            [
                {
                    "artifact_dir": "run1",
                    "num_trades": 2,
                    "raw_signals": 4,
                    "win_rate_pct": 50.0,
                    "total_return_pct": 0.5,
                    "campaign_summary": {
                        "realized_performance": {
                            "by_symbol": {
                                "BTC/USDT": {"trades": 1, "wins": 1, "losses": 0, "total_pl": 12.0, "expectancy": 12.0},
                                "ETH/USDT": {"trades": 1, "wins": 0, "losses": 1, "total_pl": -4.0, "expectancy": -4.0},
                            },
                            "by_strategy": {
                                "trend_pullback": {"trades": 1, "wins": 1, "losses": 0, "total_pl": 12.0, "expectancy": 12.0},
                                "trend_breakout": {"trades": 1, "wins": 0, "losses": 1, "total_pl": -4.0, "expectancy": -4.0},
                            },
                        },
                    },
                },
                {
                    "artifact_dir": "run2",
                    "num_trades": 1,
                    "raw_signals": 2,
                    "win_rate_pct": 0.0,
                    "total_return_pct": -0.2,
                    "campaign_summary": {
                        "realized_performance": {
                            "by_symbol": {
                                "BTC/USDT": {"trades": 1, "wins": 0, "losses": 1, "total_pl": -3.0, "expectancy": -3.0},
                            },
                            "by_strategy": {
                                "trend_pullback": {"trades": 1, "wins": 0, "losses": 1, "total_pl": -3.0, "expectancy": -3.0},
                            },
                        },
                    },
                },
            ]
        )
        self.assertEqual(summary["realized_performance"]["by_symbol"]["BTC/USDT"]["trades"], 2.0)
        self.assertEqual(summary["realized_performance"]["by_symbol"]["BTC/USDT"]["total_pl"], 9.0)
        self.assertAlmostEqual(summary["realized_performance"]["by_symbol"]["BTC/USDT"]["expectancy"], 4.5)
        self.assertEqual(summary["realized_performance"]["by_strategy"]["trend_pullback"]["trades"], 2.0)
        self.assertEqual(summary["realized_performance"]["by_strategy"]["trend_pullback"]["total_pl"], 9.0)
        self.assertAlmostEqual(summary["realized_performance"]["by_strategy"]["trend_breakout"]["expectancy"], -4.0)

    def test_build_batch_summary_aggregates_exit_quality(self):
        summary = build_batch_summary(
            [
                {
                    "artifact_dir": "run1",
                    "num_trades": 2,
                    "raw_signals": 4,
                    "campaign_summary": {
                        "exit_quality": {
                            "by_exit_reason": {
                                "TRAILING_STOP": {
                                    "trades": 2,
                                    "wins": 1,
                                    "losses": 1,
                                    "total_pl": 10.0,
                                    "total_mfe_r": 2.4,
                                    "total_mae_r": 1.2,
                                    "total_giveback_r": 1.1,
                                }
                            },
                            "giveback_by_strategy": {
                                "trend_breakout": {
                                    "trades": 2,
                                    "total_giveback_r": 1.1,
                                    "mfe_above_1r_count": 1,
                                    "gave_back_below_0_25r_count": 0,
                                }
                            },
                        },
                    },
                },
                {
                    "artifact_dir": "run2",
                    "num_trades": 1,
                    "raw_signals": 2,
                    "campaign_summary": {
                        "exit_quality": {
                            "by_exit_reason": {
                                "TIME_HARD": {
                                    "trades": 1,
                                    "wins": 0,
                                    "losses": 1,
                                    "total_pl": -3.0,
                                    "total_mfe_r": 1.1,
                                    "total_mae_r": 0.8,
                                    "total_giveback_r": 0.9,
                                }
                            },
                            "giveback_by_strategy": {
                                "trend_pullback": {
                                    "trades": 1,
                                    "total_giveback_r": 0.9,
                                    "mfe_above_1r_count": 1,
                                    "gave_back_below_0_25r_count": 1,
                                }
                            },
                        },
                    },
                },
            ]
        )
        self.assertAlmostEqual(summary["exit_quality"]["by_exit_reason"]["TRAILING_STOP"]["expectancy"], 5.0)
        self.assertAlmostEqual(summary["exit_quality"]["by_exit_reason"]["TRAILING_STOP"]["avg_mfe_r"], 1.2)
        self.assertAlmostEqual(summary["exit_quality"]["by_exit_reason"]["TIME_HARD"]["avg_giveback_r"], 0.9)
        self.assertAlmostEqual(summary["exit_quality"]["giveback_by_strategy"]["trend_breakout"]["avg_giveback_r"], 0.55)
        self.assertEqual(summary["exit_quality"]["giveback_by_strategy"]["trend_pullback"]["gave_back_below_0_25r_count"], 1.0)

    def test_build_batch_summary_aggregates_universe_selection_diagnostics(self):
        summary = build_batch_summary(
            [
                {
                    "artifact_dir": "run1",
                    "num_trades": 1,
                    "raw_signals": 2,
                    "campaign_summary": {
                        "universe_selection": {
                            "eligible_symbols": ["BTC/USDT", "ETH/USDT"],
                            "eligible_bucket_counts": {"majors": 2},
                            "rejected_symbols": {
                                "ADA/USDT": {"reason": "bucket_cap_reached", "bucket": "high_beta_alts"},
                            },
                            "scored_symbols": {
                                "BTC/USDT": {"bucket": "majors", "realized_score_adjustment": -10.0},
                                "ETH/USDT": {"bucket": "majors", "realized_score_adjustment": 5.0},
                                "ADA/USDT": {"bucket": "high_beta_alts"},
                            },
                        }
                    },
                },
                {
                    "artifact_dir": "run2",
                    "num_trades": 1,
                    "raw_signals": 1,
                    "campaign_summary": {
                        "universe_selection": {
                            "eligible_symbols": ["BTC/USDT"],
                            "eligible_bucket_counts": {"majors": 1},
                            "rejected_symbols": {
                                "SOL/USDT": {"reason": "realized_negative_expectancy_veto", "bucket": "high_beta_alts"},
                            },
                            "scored_symbols": {
                                "BTC/USDT": {"bucket": "majors", "realized_score_adjustment": 5.0},
                                "SOL/USDT": {"bucket": "high_beta_alts"},
                            },
                        }
                    },
                },
            ]
        )
        self.assertEqual(summary["universe_selection"]["eligible_symbols"]["BTC/USDT"], 2)
        self.assertEqual(summary["universe_selection"]["eligible_symbols"]["ETH/USDT"], 1)
        self.assertEqual(summary["universe_selection"]["rejected_symbols"]["ADA/USDT:bucket_cap_reached"], 1)
        self.assertEqual(summary["universe_selection"]["rejected_symbols"]["SOL/USDT:realized_negative_expectancy_veto"], 1)
        self.assertEqual(summary["universe_selection"]["eligible_buckets"]["majors"], 3)
        self.assertEqual(summary["universe_selection"]["rejected_buckets"]["high_beta_alts"], 2)
        self.assertEqual(summary["universe_selection"]["bucket_cap_rejections"]["ADA/USDT"], 1)
        self.assertEqual(summary["universe_selection"]["bucket_cap_rejections_by_bucket"]["high_beta_alts"], 1)
        self.assertEqual(summary["universe_selection"]["eligible_bucket_pressure"]["majors"], 3)
        self.assertEqual(summary["universe_selection"]["realized_universe_promotions"]["ETH/USDT"], 1)
        self.assertEqual(summary["universe_selection"]["realized_universe_promotions"]["BTC/USDT"], 1)
        self.assertEqual(summary["universe_selection"]["realized_universe_penalties"]["BTC/USDT"], 1)
        self.assertEqual(summary["universe_selection"]["realized_universe_adjustments_by_bucket"]["majors"], 1)
        self.assertEqual(summary["universe_selection"]["realized_universe_vetoes"]["SOL/USDT"], 1)
        self.assertEqual(summary["universe_selection"]["realized_universe_vetoes_by_bucket"]["high_beta_alts"], 1)

    def test_simulation_campaign_caps_same_bucket_in_eligible_universe(self):
        timestamps = pd.date_range("2025-01-01", periods=21, freq="15min")

        def build_df(volume: float) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "timestamp": ts,
                        "open": 100.0,
                        "high": 100.5,
                        "low": 99.5,
                        "close": 100.0,
                        "volume": volume,
                    }
                    for ts in timestamps
                ]
            ).set_index("timestamp")

        datasets = {
            "BTC/USDT": build_df(5000.0),
            "ETH/USDT": build_df(4500.0),
            "BNB/USDT": build_df(3000.0),
        }

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return datasets[symbol].copy()

        class BucketCrowdingSignals(QuietSignalEngine):
            def generate_signal(self, symbol):
                candles = self.exch.fetch_ohlcv(symbol, "15m", limit=200)
                if len(candles) != 21:
                    return []
                return {
                    "side": "long",
                    "entry_price": 100.0,
                    "stop_loss": 99.0,
                    "take_profit": 103.0,
                    "strategy": "unit_test_bucket_cap",
                    "expected_holding_minutes": 60,
                    "signal_quality": 0.75,
                    "expected_edge_bps": 16.0,
                    "rr_ratio": 3.0,
                    "metadata": {"cross_sectional_score": 90.0},
                }

        engine = LocalSimulationEngine(
            BotConfig(
                starting_balance=1000.0,
                trading_mode="futures",
                max_open_positions=3,
                backtest_warmup_candles=20,
                simulation_universe_top_n=3,
                simulation_universe_tradability_floor=1.0,
                simulation_universe_bucket_cap=1,
                simulation_universe_bucket_cap_majors=1,
            ),
            signal_engine_cls=BucketCrowdingSignals,
        )
        result = engine.run_campaign(["BTC/USDT", "ETH/USDT", "BNB/USDT"], timeframe="15m", days=10)

        universe = result["campaign_summary"]["universe_selection"]

        self.assertEqual(universe["eligible_symbols"], ["BTC/USDT", "BNB/USDT"])
        self.assertEqual(universe["eligible_bucket_counts"]["majors"], 1)
        self.assertEqual(universe["eligible_bucket_counts"]["exchange_beta"], 1)
        self.assertEqual(universe["rejected_symbols"]["ETH/USDT"]["reason"], "bucket_cap_reached")
        self.assertEqual(result["campaign_summary"]["trade_flow"]["raw_signals"], 2)

    def test_simulation_campaign_universe_scoring_uses_realized_symbol_expectancy(self):
        timestamps = pd.date_range("2025-01-01", periods=21, freq="15min")

        def build_df(volume: float) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "timestamp": ts,
                        "open": 100.0,
                        "high": 100.5,
                        "low": 99.5,
                        "close": 100.0,
                        "volume": volume,
                    }
                    for ts in timestamps
                ]
            ).set_index("timestamp")

        datasets = {
            "BTC/USDT": build_df(3000.0),
            "ETH/USDT": build_df(3000.0),
        }

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return datasets[symbol].copy()

        engine = LocalSimulationEngine(
            BotConfig(
                starting_balance=1000.0,
                trading_mode="futures",
                max_open_positions=3,
                backtest_warmup_candles=20,
                simulation_universe_top_n=1,
                simulation_universe_tradability_floor=1.0,
                simulation_universe_realized_min_trades=2,
                simulation_universe_realized_penalty_score=12.0,
                simulation_universe_realized_boost_score=6.0,
            ),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine._simulation_symbol_universe = ["BTC/USDT", "ETH/USDT"]
        engine._base_timeframe = "15m"
        engine.exchange = HistoricalReplayExchange(
            {
                ("BTC/USDT", "15m"): datasets["BTC/USDT"],
                ("ETH/USDT", "15m"): datasets["ETH/USDT"],
            },
            "BTC/USDT",
            base_timeframe="15m",
            config=engine.config,
        )
        engine.exchange.set_time(timestamps[-1].to_pydatetime() + dt.timedelta(minutes=15))
        engine.trades = [
            {"symbol": "BTC/USDT", "pl": -3.0},
            {"symbol": "BTC/USDT", "pl": -3.0},
            {"symbol": "ETH/USDT", "pl": 2.0},
            {"symbol": "ETH/USDT", "pl": 2.0},
        ]
        engine._closed_by_symbol = {"BTC/USDT": 2, "ETH/USDT": 2}

        universe = engine._eligible_universe_symbols(["BTC/USDT", "ETH/USDT"])
        scored = engine._latest_universe_selection["scored_symbols"]

        self.assertEqual(universe, ["ETH/USDT"])
        self.assertEqual(scored["BTC/USDT"]["realized_adjustment_reason"], "negative_expectancy")
        self.assertEqual(scored["ETH/USDT"]["realized_adjustment_reason"], "positive_expectancy")
        self.assertLess(scored["BTC/USDT"]["tradability_score"], scored["ETH/USDT"]["tradability_score"])

    def test_simulation_campaign_universe_vetoes_severely_negative_symbol(self):
        timestamps = pd.date_range("2025-01-01", periods=21, freq="15min")

        def build_df(volume: float) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "timestamp": ts,
                        "open": 100.0,
                        "high": 100.5,
                        "low": 99.5,
                        "close": 100.0,
                        "volume": volume,
                    }
                    for ts in timestamps
                ]
            ).set_index("timestamp")

        datasets = {
            "DOT/USDT": build_df(3000.0),
            "BTC/USDT": build_df(3000.0),
        }

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return datasets[symbol].copy()

        engine = LocalSimulationEngine(
            BotConfig(
                starting_balance=1000.0,
                trading_mode="futures",
                backtest_warmup_candles=20,
                simulation_universe_top_n=2,
                simulation_universe_tradability_floor=1.0,
                simulation_universe_realized_veto_min_trades=2,
                simulation_universe_realized_veto_expectancy_floor=-2.0,
            ),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine._simulation_symbol_universe = ["DOT/USDT", "BTC/USDT"]
        engine._base_timeframe = "15m"
        engine.exchange = HistoricalReplayExchange(
            {
                ("DOT/USDT", "15m"): datasets["DOT/USDT"],
                ("BTC/USDT", "15m"): datasets["BTC/USDT"],
            },
            "BTC/USDT",
            base_timeframe="15m",
            config=engine.config,
        )
        engine.exchange.set_time(timestamps[-1].to_pydatetime() + dt.timedelta(minutes=15))
        engine.trades = [
            {"symbol": "DOT/USDT", "pl": -3.0},
            {"symbol": "DOT/USDT", "pl": -3.0},
            {"symbol": "BTC/USDT", "pl": 1.0},
            {"symbol": "BTC/USDT", "pl": 1.0},
        ]
        engine._closed_by_symbol = {"DOT/USDT": 2, "BTC/USDT": 2}

        universe = engine._eligible_universe_symbols(["DOT/USDT", "BTC/USDT"])
        rejected = engine._latest_universe_selection["rejected_symbols"]

        self.assertEqual(universe, ["BTC/USDT"])
        self.assertTrue(engine._latest_universe_selection["scored_symbols"]["DOT/USDT"]["realized_veto_active"])
        self.assertEqual(rejected["DOT/USDT"]["reason"], "realized_negative_expectancy_veto")

    def test_universe_bucket_cap_tightens_high_beta_alts_in_risk_off_state(self):
        timestamps = pd.date_range("2025-01-01", periods=21, freq="15min")

        def build_df(range_fraction: float, volume: float = 3000.0) -> pd.DataFrame:
            rows = []
            price = 100.0
            for ts in timestamps:
                high = price * (1.0 + range_fraction / 2.0)
                low = price * (1.0 - range_fraction / 2.0)
                rows.append(
                    {
                        "timestamp": ts,
                        "open": price,
                        "high": high,
                        "low": low,
                        "close": price,
                        "volume": volume,
                    }
                )
            return pd.DataFrame(rows).set_index("timestamp")

        datasets = {
            "BTC/USDT": build_df(0.03),
            "SOL/USDT": build_df(0.01),
            "AVAX/USDT": build_df(0.01),
        }

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return datasets[symbol].copy()

        engine = LocalSimulationEngine(
            BotConfig(
                starting_balance=1000.0,
                trading_mode="futures",
                backtest_warmup_candles=20,
                simulation_universe_top_n=3,
                simulation_universe_tradability_floor=1.0,
                simulation_universe_bucket_cap_high_beta_alts=1,
                simulation_universe_regime_high_vol_threshold=0.02,
                simulation_universe_regime_risk_off_high_beta_delta=-1,
            ),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine._simulation_symbol_universe = ["BTC/USDT", "SOL/USDT", "AVAX/USDT"]
        engine._base_timeframe = "15m"
        engine.exchange = HistoricalReplayExchange(
            {(symbol, "15m"): frame for symbol, frame in datasets.items()},
            "BTC/USDT",
            base_timeframe="15m",
            config=engine.config,
        )
        engine.exchange.set_time(timestamps[-1].to_pydatetime() + dt.timedelta(minutes=15))

        universe = engine._eligible_universe_symbols(["BTC/USDT", "SOL/USDT", "AVAX/USDT"])

        self.assertEqual(engine._latest_universe_selection["regime_state"]["state"], "risk_off")
        self.assertEqual(engine._universe_bucket_cap("high_beta_alts", regime_state=engine._latest_universe_selection["regime_state"]), 0)
        self.assertNotIn("SOL/USDT", universe)
        self.assertNotIn("AVAX/USDT", universe)
        self.assertIn("BTC/USDT", universe)

    def test_universe_bucket_cap_relaxes_high_beta_alts_in_trend_supportive_state(self):
        timestamps = pd.date_range("2025-01-01", periods=21, freq="15min")

        def build_trending_df(start: float, step: float, volume: float = 3000.0) -> pd.DataFrame:
            rows = []
            price = start
            for ts in timestamps:
                close = price + step
                high = max(price, close) * 1.002
                low = min(price, close) * 0.998
                rows.append(
                    {
                        "timestamp": ts,
                        "open": price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                    }
                )
                price = close
            return pd.DataFrame(rows).set_index("timestamp")

        datasets = {
            "BTC/USDT": build_trending_df(100.0, 0.2),
            "SOL/USDT": build_trending_df(100.0, 0.1),
            "AVAX/USDT": build_trending_df(100.0, 0.1),
        }

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return datasets[symbol].copy()

        engine = LocalSimulationEngine(
            BotConfig(
                starting_balance=1000.0,
                trading_mode="futures",
                backtest_warmup_candles=20,
                simulation_universe_top_n=3,
                simulation_universe_tradability_floor=1.0,
                simulation_universe_bucket_cap_high_beta_alts=1,
                simulation_universe_regime_high_vol_threshold=0.03,
                simulation_universe_regime_trend_strength_threshold=0.015,
                simulation_universe_regime_trend_high_beta_delta=1,
            ),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine._simulation_symbol_universe = ["BTC/USDT", "SOL/USDT", "AVAX/USDT"]
        engine._base_timeframe = "15m"
        engine.exchange = HistoricalReplayExchange(
            {(symbol, "15m"): frame for symbol, frame in datasets.items()},
            "BTC/USDT",
            base_timeframe="15m",
            config=engine.config,
        )
        engine.exchange.set_time(timestamps[-1].to_pydatetime() + dt.timedelta(minutes=15))

        universe = engine._eligible_universe_symbols(["BTC/USDT", "SOL/USDT", "AVAX/USDT"])

        self.assertEqual(engine._latest_universe_selection["regime_state"]["state"], "trend_supportive")
        self.assertEqual(engine._universe_bucket_cap("high_beta_alts", regime_state=engine._latest_universe_selection["regime_state"]), 2)
        self.assertIn("BTC/USDT", universe)
        self.assertIn("SOL/USDT", universe)
        self.assertIn("AVAX/USDT", universe)

    def test_build_batch_summary_aggregates_campaign_acceptance_totals(self):
        summary = build_batch_summary(
            [
                {
                    "artifact_dir": "run1",
                    "num_trades": 4,
                    "raw_signals": 8,
                    "campaign_summary": {
                        "session": {"days": 30},
                        "acceptance": {
                            "checks": {
                                "trades_per_day_in_target_band": True,
                                "win_rate_meets_floor": True,
                                "profit_factor_meets_floor": True,
                                "drawdown_within_limit": True,
                                "positive_return": True,
                            },
                            "passes_all": True,
                        },
                    },
                },
                {
                    "artifact_dir": "run2",
                    "num_trades": 2,
                    "raw_signals": 5,
                    "campaign_summary": {
                        "session": {"days": 60},
                        "acceptance": {
                            "checks": {
                                "trades_per_day_in_target_band": False,
                                "win_rate_meets_floor": True,
                                "profit_factor_meets_floor": False,
                                "drawdown_within_limit": True,
                                "positive_return": False,
                            },
                            "passes_all": False,
                        },
                    },
                },
            ]
        )
        self.assertEqual(summary["acceptance_totals"]["passes_all"], 1)
        self.assertEqual(summary["acceptance_totals"]["win_rate_meets_floor"], 2)
        self.assertEqual(summary["acceptance_totals"]["drawdown_within_limit"], 2)
        self.assertEqual(summary["acceptance_totals"]["positive_return"], 1)
        self.assertEqual(summary["by_horizon"]["30d"]["num_runs"], 1)
        self.assertEqual(summary["by_horizon"]["60d"]["num_runs"], 1)
        self.assertEqual(summary["by_horizon"]["30d"]["candidate_verdict"]["status"], "promising")
        self.assertEqual(summary["by_horizon"]["60d"]["candidate_verdict"]["status"], "not_ready")
        self.assertEqual(summary["walk_forward"]["iteration_horizon"], "30d")
        self.assertEqual(summary["walk_forward"]["confirmation_horizon"], "60d")
        self.assertAlmostEqual(summary["walk_forward"]["metrics"]["trades_per_day"]["iteration"], 4.0 / 30.0)
        self.assertAlmostEqual(summary["walk_forward"]["metrics"]["trades_per_day"]["confirmation"], 2.0 / 60.0)
        self.assertFalse(summary["walk_forward"]["acceptance"]["passes_all"])

    def test_build_batch_summary_surfaces_failure_leaderboard(self):
        summary = build_batch_summary(
            [
                {
                    "artifact_dir": "run1",
                    "num_trades": 3,
                    "raw_signals": 6,
                    "decision_diagnostics": {
                        "skip_reasons": {
                            "portfolio_no_trade_region": 4,
                            "stale_limit": 2,
                        },
                    },
                    "campaign_summary": {
                        "realized_performance": {
                            "by_symbol": {
                                "BTC/USDT": {"trades": 2, "wins": 0, "losses": 2, "total_pl": -6.0, "expectancy": -3.0},
                                "ETH/USDT": {"trades": 1, "wins": 1, "losses": 0, "total_pl": 4.0, "expectancy": 4.0},
                            },
                            "by_strategy": {
                                "trend_breakout": {"trades": 2, "wins": 0, "losses": 2, "total_pl": -6.0, "expectancy": -3.0},
                                "trend_pullback": {"trades": 1, "wins": 1, "losses": 0, "total_pl": 4.0, "expectancy": 4.0},
                            },
                        },
                    },
                    "symbol_rollups": {
                        "BTC/USDT": {"trades": 2, "avg_edge_bps": 18.0, "expectancy": -3.0},
                        "ETH/USDT": {"trades": 1, "avg_edge_bps": 7.0, "expectancy": 4.0},
                    },
                },
                {
                    "artifact_dir": "run2",
                    "num_trades": 2,
                    "raw_signals": 5,
                    "decision_diagnostics": {
                        "skip_reasons": {
                            "portfolio_no_trade_region": 1,
                            "stale_limit": 3,
                        },
                    },
                    "campaign_summary": {
                        "realized_performance": {
                            "by_symbol": {
                                "ADA/USDT": {"trades": 2, "wins": 0, "losses": 2, "total_pl": -5.0, "expectancy": -2.5},
                            },
                            "by_strategy": {
                                "mean_reversion": {"trades": 2, "wins": 0, "losses": 2, "total_pl": -5.0, "expectancy": -2.5},
                            },
                        },
                    },
                    "symbol_rollups": {
                        "ADA/USDT": {"trades": 2, "avg_edge_bps": 10.0, "expectancy": -2.5},
                    },
                },
            ]
        )
        self.assertEqual(summary["failure_leaderboard"]["top_skip_reasons"]["portfolio_no_trade_region"], 5)
        self.assertAlmostEqual(summary["failure_leaderboard"]["losing_families"]["trend_breakout"]["expectancy"], -3.0)
        self.assertAlmostEqual(summary["failure_leaderboard"]["losing_symbols"]["BTC/USDT"]["expectancy"], -3.0)
        self.assertAlmostEqual(summary["failure_leaderboard"]["expected_vs_realized_edge_divergence"]["BTC/USDT"]["avg_edge_bps"], 18.0)
        self.assertAlmostEqual(summary["failure_leaderboard"]["expected_vs_realized_edge_divergence"]["ADA/USDT"]["realized_expectancy"], -2.5)
        self.assertEqual(summary["candidate_verdict"]["status"], "not_ready")
        self.assertIn("no_run_passed_acceptance", summary["candidate_verdict"]["reasons"])
        self.assertIn("losing_families_present", summary["candidate_verdict"]["reasons"])

    def test_build_batch_summary_aggregates_realized_performance_penalty_diagnostics(self):
        summary = build_batch_summary(
            [
                {
                    "artifact_dir": "run1",
                    "num_trades": 1,
                    "raw_signals": 2,
                    "decision_diagnostics": {
                        "learning_evidence_counts": {"positive_cell_evidence": 2},
                        "learning_evidence_counts_by_strategy": {
                            "trend_pullback": {"positive_cell_evidence": 2},
                        },
                        "learning_asymmetry_counts": {"promote_winner": 2, "promote_family_rotation": 1},
                        "learning_asymmetry_counts_by_strategy": {
                            "trend_pullback": {"promote_winner": 2, "promote_family_rotation": 1},
                        },
                        "missed_opportunity_relaxations": {"portfolio_no_trade_region": 2},
                        "missed_opportunity_relaxations_by_strategy": {
                            "trend_pullback": {"portfolio_no_trade_region": 2},
                        },
                        "realized_performance_penalty_counts": {"symbol_negative_expectancy": 2},
                        "realized_performance_penalty_counts_by_strategy": {
                            "trend_pullback": {"symbol_negative_expectancy": 2},
                        },
                        "realized_performance_no_trade_blocks": {"strategy_negative_expectancy": 1},
                        "realized_performance_no_trade_blocks_by_strategy": {
                            "trend_pullback": {"strategy_negative_expectancy": 1},
                        },
                        "skip_reasons_by_strategy_by_symbol": {
                            "AVAX/USDT": {"trend_pullback": {"portfolio_persistently_weak_cluster_throttle": 2}},
                            "ETH/USDT": {"trend_pullback": {"portfolio_duplicate_bucket_throttle": 1}},
                        },
                    },
                },
                {
                    "artifact_dir": "run2",
                    "num_trades": 1,
                    "raw_signals": 1,
                    "decision_diagnostics": {
                        "learning_evidence_counts": {"negative_cell_evidence": 3},
                        "learning_evidence_counts_by_strategy": {
                            "mean_reversion": {"negative_cell_evidence": 3},
                        },
                        "learning_asymmetry_counts": {"throttle_loser": 3, "throttle_family_rotation": 2},
                        "learning_asymmetry_counts_by_strategy": {
                            "mean_reversion": {"throttle_loser": 3, "throttle_family_rotation": 2},
                        },
                        "missed_opportunity_relaxations": {"portfolio_no_trade_region": 1},
                        "missed_opportunity_relaxations_by_strategy": {
                            "mean_reversion": {"portfolio_no_trade_region": 1},
                        },
                        "realized_performance_penalty_counts": {"strategy_negative_expectancy": 3},
                        "realized_performance_penalty_counts_by_strategy": {
                            "mean_reversion": {"strategy_negative_expectancy": 3},
                        },
                        "realized_performance_no_trade_blocks": {"symbol_negative_expectancy": 2},
                        "realized_performance_no_trade_blocks_by_strategy": {
                            "mean_reversion": {"symbol_negative_expectancy": 2},
                        },
                        "skip_reasons_by_strategy_by_symbol": {
                            "DOT/USDT": {"trend_pullback": {"portfolio_persistently_weak_cluster_throttle": 1}},
                            "BTC/USDT": {"mean_reversion": {"portfolio_duplicate_bucket_throttle": 2}},
                        },
                    },
                },
            ]
        )
        self.assertEqual(summary["decision_totals"]["learning_evidence_counts"]["positive_cell_evidence"], 2)
        self.assertEqual(summary["decision_totals"]["learning_evidence_counts"]["negative_cell_evidence"], 3)
        self.assertEqual(summary["learning_evidence_by_strategy"]["trend_pullback"]["positive_cell_evidence"], 2)
        self.assertEqual(summary["learning_evidence_by_strategy"]["mean_reversion"]["negative_cell_evidence"], 3)
        self.assertEqual(summary["decision_totals"]["learning_asymmetry_counts"]["promote_winner"], 2)
        self.assertEqual(summary["decision_totals"]["learning_asymmetry_counts"]["throttle_loser"], 3)
        self.assertEqual(summary["learning_asymmetry_by_strategy"]["trend_pullback"]["promote_family_rotation"], 1)
        self.assertEqual(summary["learning_asymmetry_by_strategy"]["mean_reversion"]["throttle_family_rotation"], 2)
        self.assertEqual(summary["decision_totals"]["missed_opportunity_relaxations"]["portfolio_no_trade_region"], 3)
        self.assertEqual(summary["missed_opportunity_relaxations_by_strategy"]["trend_pullback"]["portfolio_no_trade_region"], 2)
        self.assertEqual(summary["decision_totals"]["realized_performance_penalty_counts"]["symbol_negative_expectancy"], 2)
        self.assertEqual(summary["decision_totals"]["realized_performance_penalty_counts"]["strategy_negative_expectancy"], 3)
        self.assertEqual(summary["decision_totals"]["realized_performance_no_trade_blocks"]["strategy_negative_expectancy"], 1)
        self.assertEqual(summary["decision_totals"]["realized_performance_no_trade_blocks"]["symbol_negative_expectancy"], 2)
        self.assertEqual(summary["realized_performance_penalty_by_strategy"]["trend_pullback"]["symbol_negative_expectancy"], 2)
        self.assertEqual(summary["realized_performance_no_trade_by_strategy"]["mean_reversion"]["symbol_negative_expectancy"], 2)
        self.assertEqual(summary["duplicate_bucket_throttle_by_strategy"]["trend_pullback"], 1)
        self.assertEqual(summary["duplicate_bucket_throttle_by_strategy"]["mean_reversion"], 2)
        self.assertEqual(summary["weak_cluster_throttle_by_strategy"]["trend_pullback"], 3)

    def test_campaign_diagnostics_include_consolidated_learning_controls(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine._signals_by_strategy["trend_pullback"] = 2
        engine._family_rotation_counts_by_strategy["trend_pullback"] = {"promote_current": 1}
        engine._family_rotation_recovery_counts_by_strategy["trend_pullback"] = {"recovery_active": 1}
        engine._learning_evidence_counts_by_strategy["trend_pullback"] = {"positive_cell_evidence": 2}
        engine._learning_asymmetry_counts_by_strategy["trend_pullback"] = {"promote_winner": 2}
        engine._missed_opportunity_relaxations_by_strategy["trend_pullback"] = {"portfolio_no_trade_region": 1}
        engine._reentry_cooldown_registrations_by_strategy["trend_pullback"] = {"breakout_volatility_exit_cooldown": 1}
        diagnostics = engine._build_campaign_diagnostics({})
        learning_controls = diagnostics["primary_summary"]["by_strategy"]["learning_controls"]["trend_pullback"]
        self.assertEqual(learning_controls["family_rotation"]["promote_current"], 1)
        self.assertEqual(learning_controls["family_rotation_recovery"]["recovery_active"], 1)
        self.assertEqual(learning_controls["learning_evidence"]["positive_cell_evidence"], 2)
        self.assertEqual(learning_controls["learning_asymmetry"]["promote_winner"], 2)
        self.assertEqual(learning_controls["missed_opportunity_relaxations"]["portfolio_no_trade_region"], 1)
        self.assertEqual(learning_controls["reentry_cooldown_registrations"]["breakout_volatility_exit_cooldown"], 1)

    def test_campaign_comparison_tracks_target_metrics_and_mix(self):
        comparison = build_campaign_comparison(
            {
                "num_trades": 4,
                "raw_signals": 12,
                "win_rate_pct": 45.0,
                "total_return_pct": 0.5,
                "profit_factor": 1.1,
                "max_drawdown_pct": -4.0,
                "avg_holding_minutes": 180.0,
                "trade_frequency": {"global": {"trades_per_day": 1.2}},
                "signals_by_strategy": {"trend_breakout": 5, "trend_pullback": 2},
                "raw_signals_by_symbol": {"BTC/USDT": 6, "ETH/USDT": 2},
            },
            {
                "num_trades": 6,
                "raw_signals": 15,
                "win_rate_pct": 48.0,
                "total_return_pct": 1.2,
                "profit_factor": 1.3,
                "max_drawdown_pct": -3.0,
                "avg_holding_minutes": 150.0,
                "trade_frequency": {"global": {"trades_per_day": 2.1}},
                "signals_by_strategy": {"trend_breakout": 2, "trend_pullback": 5, "mean_reversion": 3},
                "raw_signals_by_symbol": {"BTC/USDT": 3, "ETH/USDT": 4, "SOL/USDT": 2},
            },
        )
        self.assertEqual(comparison["metrics"]["trades_per_day"]["baseline"], 1.2)
        self.assertEqual(comparison["metrics"]["trades_per_day"]["candidate"], 2.1)
        self.assertEqual(comparison["metrics"]["strategy_family_mix"]["delta"]["mean_reversion"], 3.0)
        self.assertEqual(comparison["metrics"]["symbol_mix"]["delta"]["SOL/USDT"], 2.0)
        self.assertTrue(comparison["acceptance"]["trades_per_day_not_worse"])
        self.assertTrue(comparison["acceptance"]["strategy_mix_changed"])
        self.assertTrue(comparison["acceptance"]["symbol_mix_changed"])

    def test_telegram_operator_unknown_command_lists_supported_commands(self):
        config = BotConfig(telegram_bot_token="token", telegram_chat_id="123")
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                message = _handle_command(config, tmpdir, "/nope", {})
            finally:
                os.chdir(cwd)
        self.assertIn("/startbot", message)
        self.assertIn("/simlogs", message)

    def test_telegram_operator_startbot_is_idempotent_when_running(self):
        config = BotConfig(telegram_bot_token="token", telegram_chat_id="123")
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("trade_bot.telegram_operator.read_bot_status", return_value={"running": True, "pid": 4321, "status": "running"}):
                message = _handle_command(config, tmpdir, "/startbot", {})
        self.assertIn("already running", message)
        self.assertIn("4321", message)

    def test_telegram_operator_stopbot_is_idempotent_when_stopped(self):
        config = BotConfig(telegram_bot_token="token", telegram_chat_id="123")
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("trade_bot.telegram_operator.read_bot_status", return_value={"running": False, "pid": 0, "status": "stopped"}):
                message = _handle_command(config, tmpdir, "/stopbot", {})
        self.assertIn("already stopped", message)

    def test_telegram_operator_startbot_requires_live_control_flag(self):
        config = BotConfig(telegram_bot_token="token", telegram_chat_id="123")
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {}, clear=False):
                with mock.patch("trade_bot.telegram_operator.read_bot_status", return_value={"running": False, "pid": 0, "status": "stopped"}):
                    message = _handle_command(config, tmpdir, "/startbot", {})
        self.assertIn("Live bot start blocked", message)

    def test_telegram_operator_botlogs_returns_short_operational_summary(self):
        config = BotConfig(telegram_bot_token="token", telegram_chat_id="123")
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout_path = os.path.join(tmpdir, "stdout.log")
            stderr_path = os.path.join(tmpdir, "stderr.log")
            with open(stderr_path, "w", encoding="utf-8") as handle:
                handle.write("[WARN] Example warning\n")
            with mock.patch(
                "trade_bot.telegram_operator.read_bot_status",
                return_value={
                    "running": True,
                    "pid": 9876,
                    "status": "running",
                    "health": {"status": "ok", "reason": "ok"},
                    "last_error": "[WARN] Example warning",
                    "updated_at": "2026-04-04T10:00:00+00:00",
                    "paths": {"stdout": stdout_path, "stderr": stderr_path},
                },
            ):
                with mock.patch("trade_bot.telegram_operator._load_bot_state", return_value={"balance": 120.0, "equity_start_of_day": 100.0, "realized_pl_today": 5.0}):
                    with mock.patch("trade_bot.telegram_operator._operational_metrics", return_value={"win_rate_pct": 55.5}):
                        with mock.patch("trade_bot.telegram_operator._fills_since", return_value=3):
                            message = _handle_command(config, tmpdir, "/botlogs", {})
        self.assertIn("Bot: running", message)
        self.assertIn("PID: 9876", message)
        self.assertIn("Health: ok (ok)", message)
        self.assertIn("Trades since check: 3", message)
        self.assertIn("Last warning/error:", message)

    def test_telegram_operator_startsimulation_is_idempotent_when_running(self):
        config = BotConfig(telegram_bot_token="token", telegram_chat_id="123")
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("trade_bot.telegram_operator.read_simulation_batch_status", return_value={"running": True, "pid": 2468, "completed_runs": 4, "status": "running"}):
                message = _handle_command(config, tmpdir, "/startsimulation", {})
        self.assertIn("already running", message)
        self.assertIn("2468", message)

    def test_telegram_operator_stopsimulation_is_idempotent_when_stopped(self):
        config = BotConfig(telegram_bot_token="token", telegram_chat_id="123")
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("trade_bot.telegram_operator.read_simulation_batch_status", return_value={"running": False, "pid": 0, "status": "stopped"}):
                message = _handle_command(config, tmpdir, "/stopsimulation", {})
        self.assertIn("already stopped", message)

    def test_telegram_operator_simlogs_returns_short_batch_summary(self):
        config = BotConfig(telegram_bot_token="token", telegram_chat_id="123")
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout_path = os.path.join(tmpdir, "batch_stdout.log")
            stderr_path = os.path.join(tmpdir, "batch_stderr.log")
            with open(stderr_path, "w", encoding="utf-8") as handle:
                handle.write("[WARN] Batch warning\n")
            summary_payload = {
                "num_runs": 2,
                "aggregates": {"num_trades": {"avg": 6.5}, "total_return_pct": {"avg": -1.25}},
                "candidate_verdict": {"status": "mixed"},
                "latest_run": {"total_return_pct": -0.5, "num_trades": 7, "win_rate_pct": 42.0},
            }
            with mock.patch(
                "trade_bot.telegram_operator.read_simulation_batch_status",
                return_value={
                    "running": True,
                    "pid": 1357,
                    "last_error": "[WARN] Batch warning",
                    "paths": {"stdout": stdout_path, "stderr": stderr_path},
                },
            ):
                with mock.patch("trade_bot.telegram_operator.load_batch_reports", return_value=[{"artifact_dir": "run_0001_20260404_120000"}]):
                    with mock.patch("trade_bot.telegram_operator.build_batch_summary", return_value=summary_payload):
                        message = _handle_command(config, tmpdir, "/simlogs", {})
        self.assertIn("Simulation batch: running", message)
        self.assertIn("PID: 1357", message)
        self.assertIn("Health: unknown (unknown)", message)
        self.assertIn("Verdict: mixed", message)
        self.assertIn("Latest run:", message)
        self.assertIn("Last warning/error:", message)

    def test_read_bot_status_marks_stale_pid_and_heartbeat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_bot_state(
                BotState(
                    balance=100.0,
                    last_heartbeat=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(hours=3),
                ),
                path=os.path.join(tmpdir, "bot_state.json"),
            )
            write_bot_status(
                tmpdir,
                {
                    "status": "running",
                    "pid": 999999,
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            stale_pid = read_bot_status(tmpdir)
            self.assertEqual(stale_pid["health"]["status"], "stale_pid")

            write_bot_status(
                tmpdir,
                {
                    "status": "running",
                    "pid": os.getpid(),
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            stale_heartbeat = read_bot_status(tmpdir)
            self.assertEqual(stale_heartbeat["health"]["status"], "stale")
            self.assertEqual(stale_heartbeat["health"]["reason"], "heartbeat_stale")

    def test_read_simulation_batch_status_marks_stale_pid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_simulation_batch_status(
                tmpdir,
                {
                    "status": "running",
                    "pid": 999999,
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            status = read_simulation_batch_status(tmpdir)
            self.assertEqual(status["health"]["status"], "stale_pid")

    def test_read_bot_status_includes_last_error_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status = write_bot_status(
                tmpdir,
                {
                    "status": "running",
                    "pid": os.getpid(),
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            stderr_path = status["paths"]["stderr"] if "paths" in status else os.path.join(tmpdir, "data", "bot_control", "stderr.log")
            with open(stderr_path, "w", encoding="utf-8") as handle:
                handle.write("[WARN] Bot warning line\n")
            loaded = read_bot_status(tmpdir)
            self.assertIn("Bot warning line", loaded["last_error"])

    def test_read_simulation_batch_status_includes_last_error_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status = write_simulation_batch_status(
                tmpdir,
                {
                    "status": "running",
                    "pid": os.getpid(),
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            stderr_path = status["paths"]["stderr"] if "paths" in status else os.path.join(tmpdir, "data", "simulation_batch", "batch_stderr.log")
            with open(stderr_path, "w", encoding="utf-8") as handle:
                handle.write("[ERROR] Batch blew up\n")
            loaded = read_simulation_batch_status(tmpdir)
            self.assertIn("Batch blew up", loaded["last_error"])

    def test_telegram_operator_supported_commands_include_expected_entries(self):
        commands = _supported_commands()
        labels = {item["command"] for item in commands}
        self.assertIn("startbot", labels)
        self.assertIn("simlogs", labels)

    def test_telegram_operator_chat_allowed_checks_exact_chat_id(self):
        config = BotConfig(telegram_bot_token="token", telegram_chat_id="123")
        self.assertTrue(_chat_allowed(config, {"message": {"chat": {"id": "123"}}}))
        self.assertFalse(_chat_allowed(config, {"message": {"chat": {"id": "999"}}}))

    def test_telegram_operator_records_unauthorized_update(self):
        state = {}
        _record_unauthorized_update(state, {"update_id": 77, "message": {"chat": {"id": "999"}}})
        self.assertEqual(state["unauthorized_chats"]["999"], 1)
        self.assertEqual(state["last_unauthorized_update_id"], 77)

    def test_telegram_operator_records_command_audit_entries(self):
        state = {}
        _record_command_audit(state, command="/botlogs", response="Bot: running")
        self.assertEqual(len(state["command_audit"]), 1)
        self.assertEqual(state["command_audit"][0]["command"], "/botlogs")
        self.assertIn("Bot: running", state["command_audit"][0]["response"])

    def test_risk_manager_supports_injected_clock_for_cooldown_checks(self):
        state = BotState(balance=1000.0, paper_mode=True)
        manager = RiskManager(BotConfig(), state)
        now = dt.datetime(2025, 1, 1, 12, 0, 0)
        manager.now_provider = lambda: now
        state.cooldown_until = now + dt.timedelta(minutes=5)

        self.assertEqual(manager.entry_capacity_status("BTC/USDT"), "cooldown")

        manager.now_provider = lambda: now + dt.timedelta(minutes=6)
        self.assertNotEqual(manager.entry_capacity_status("BTC/USDT"), "cooldown")

    def test_bot_config_validates_target_trade_frequency_band(self):
        with self.assertRaises(ValueError):
            BotConfig(
                target_trades_per_day_soft_floor=2.1,
                target_trades_per_day_min=2.0,
                target_trades_per_day_max=3.0,
                target_trades_per_day_soft_ceiling=3.5,
            ).validate()

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
            self.assertLess(context["risk_multiplier"], 1.0)
            self.assertFalse(context["veto"])
            self.assertGreaterEqual(context["evidence_samples"], 3)

    def test_trade_learning_engine_treats_execution_issues_as_risk_reduction_more_than_hard_veto(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(BotConfig(), store)
            signal = {
                "symbol": "ETH/USDT",
                "side": "long",
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
                "strategy": "trend_pullback",
                "signal_quality": 0.78,
                "expected_edge_bps": 24.0,
                "regime": "trending",
                "rr_ratio": 2.0,
                "hurst_exponent": 0.35,
                "fast_move": True,
                "metadata": {
                    "trend_direction": "bullish",
                    "regime_confidence": 0.82,
                    "spread_bps": 34.0,
                    "entry_deviation_bps": 18.0,
                },
            }
            decision_context = learning.build_trade_context(
                signal,
                execution_context={"spread_bps": 34.0, "entry_deviation_bps": 18.0, "fill_fraction": 0.68, "latency_ms": 900.0},
            )
            position = Position(
                symbol="ETH/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_pullback",
                initial_stop_loss=98.0,
                metadata={"decision_context": decision_context},
            )
            for _ in range(4):
                learning.record_closed_trade(
                    symbol="ETH/USDT",
                    position=position,
                    close_price=98.5,
                    profit_loss=-1.5,
                    exit_reason="SL",
                )

            context = learning.learning_context_for_signal(signal)
            self.assertLess(context["risk_multiplier"], 1.0)
            self.assertGreaterEqual(context["risk_multiplier"], 0.72)
            self.assertIn("execution_quality_issue", context["dominant_attributions"])
            self.assertFalse(context["veto"])

    def test_trade_learning_engine_recovers_after_recent_wins(self):
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
                "signal_quality": 0.74,
                "expected_edge_bps": 20.0,
                "regime": "trending",
                "rr_ratio": 2.0,
                "hurst_exponent": 0.3,
                "metadata": {"trend_direction": "bullish", "regime_confidence": 0.8},
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
                    closed_at="2025-01-01T00:00:00+00:00",
                )
            for idx in range(3):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=position,
                    close_price=104.0,
                    profit_loss=4.0,
                    exit_reason="TP",
                    closed_at=f"2025-04-0{idx + 1}T00:00:00+00:00",
                )

            context = learning.learning_context_for_signal(signal)
            self.assertGreater(context["score_delta"], -3.0)
            self.assertGreaterEqual(context["risk_multiplier"], 0.9)
            self.assertIn(context["evidence_quality"], {"moderate", "strong"})

    def test_trade_learning_engine_calibrates_confidence_with_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(
                BotConfig(
                    learning_positive_update_min_samples=3.0,
                    learning_positive_update_max_calibration_gap=0.5,
                ),
                store,
            )
            signal = {
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
                "strategy": "trend_breakout",
                "signal_quality": 0.62,
                "expected_edge_bps": 18.0,
                "regime": "trending",
                "rr_ratio": 2.0,
                "hurst_exponent": 0.3,
                "metadata": {"trend_direction": "bullish", "regime_confidence": 0.8},
            }
            position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_breakout",
                initial_stop_loss=98.0,
                metadata={"decision_context": learning.build_trade_context(signal)},
            )
            for _ in range(4):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=position,
                    close_price=104.0,
                    profit_loss=4.0,
                    exit_reason="TP",
                )
            context = learning.learning_context_for_signal(signal)
            self.assertTrue(context["safe_positive_updates"])
            self.assertGreater(context["calibration"]["calibrated_confidence"], 0.62)

    def test_trade_learning_engine_detects_drift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            config = BotConfig(learning_drift_threshold=1.05, learning_drift_slack=0.02)
            learning = TradeLearningEngine(config, store)
            signal = {
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
                "strategy": "trend_breakout",
                "signal_quality": 0.70,
                "expected_edge_bps": 18.0,
                "regime": "trending",
                "rr_ratio": 2.0,
                "hurst_exponent": 0.3,
                "metadata": {"trend_direction": "bullish", "regime_confidence": 0.8},
            }
            position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_breakout",
                initial_stop_loss=98.0,
                metadata={"decision_context": learning.build_trade_context(signal)},
            )
            for _ in range(3):
                learning.record_closed_trade(symbol="BTC/USDT", position=position, close_price=104.0, profit_loss=4.0, exit_reason="TP")
            learning.record_closed_trade(symbol="BTC/USDT", position=position, close_price=97.0, profit_loss=-3.0, exit_reason="SL")
            self.assertTrue(learning.summary_snapshot()["drift"]["active"])

    def test_trade_learning_engine_evaluates_shadow_decisions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(BotConfig(), store)
            signal = {
                "symbol": "SOL/USDT",
                "side": "long",
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
                "strategy": "trend_pullback",
                "signal_quality": 0.68,
                "expected_edge_bps": 16.0,
                "expected_holding_minutes": 60,
                "timeframe": "15m",
                "regime": "trending",
                "rr_ratio": 2.0,
                "hurst_exponent": 0.4,
                "metadata": {"trend_direction": "bullish", "regime_confidence": 0.75},
            }
            learning.record_shadow_decision(signal, status="skipped", reason="risk_capacity")
            pending = store.load_pending_learning_decisions(limit=10)
            self.assertEqual(len(pending), 1)
            pending_item = pending[0]
            pending_item["created_at"] = "2025-01-01T00:00:00+00:00"
            store.append_learning_decision(pending_item["decision_id"], pending_item["created_at"], "pending", pending_item)
            candles = [
                [0.0, 100.0, 101.0, 99.0, 100.5, 1000.0],
                [1.0, 100.5, 103.0, 100.0, 102.5, 1000.0],
                [2.0, 102.5, 105.5, 102.0, 105.0, 1000.0],
            ]
            learning.evaluate_pending_shadow_decisions(ShadowExchangeStub(candles), now=dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc))
            stored = store.load_learning_decision(pending_item["decision_id"])
            self.assertEqual(stored["status"], "evaluated")
            self.assertTrue(stored["outcome"]["evaluated"])
            self.assertGreater(stored["outcome"]["forward_r_multiple"], 0.0)
            self.assertGreaterEqual(learning.summary_snapshot()["opportunity"]["samples"], 1.0)

    def test_trade_learning_engine_tracks_prequential_summary(self):
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
                "signal_quality": 0.66,
                "expected_edge_bps": 18.0,
                "expected_holding_minutes": 60,
                "timeframe": "15m",
                "regime": "trending",
                "rr_ratio": 2.0,
                "hurst_exponent": 0.3,
                "metadata": {"trend_direction": "bullish", "regime_confidence": 0.8},
            }
            position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_breakout",
                initial_stop_loss=98.0,
                metadata={"decision_context": learning.build_trade_context(signal)},
            )
            learning.record_closed_trade(
                symbol="BTC/USDT",
                position=position,
                close_price=104.0,
                profit_loss=4.0,
                exit_reason="TP",
            )
            learning.record_shadow_decision(signal, status="skipped", reason="risk_capacity")
            pending = store.load_pending_learning_decisions(limit=10)[0]
            pending["created_at"] = "2025-01-01T00:00:00+00:00"
            store.append_learning_decision(pending["decision_id"], pending["created_at"], "pending", pending)
            candles = [
                [0.0, 100.0, 101.0, 99.0, 100.5, 1000.0],
                [1.0, 100.5, 103.0, 100.0, 102.0, 1000.0],
                [2.0, 102.0, 104.5, 101.5, 104.0, 1000.0],
            ]
            learning.evaluate_pending_shadow_decisions(ShadowExchangeStub(candles), now=dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc))

            prequential = learning.summary_snapshot()["prequential"]
            self.assertEqual(prequential["status"], "ok")
            self.assertGreaterEqual(prequential["samples"], 2.0)
            self.assertGreaterEqual(prequential["win_rate"], 0.5)

    def test_trade_learning_engine_records_symbol_bucket_specific_learning_cells(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(BotConfig(), store)
            btc_signal = {
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
                "strategy": "trend_pullback",
                "signal_quality": 0.72,
                "expected_edge_bps": 18.0,
                "regime": "trending",
                "rr_ratio": 2.0,
                "hurst_exponent": 0.35,
                "metadata": {"trend_direction": "bullish", "regime_confidence": 0.8},
            }
            btc_position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_pullback",
                initial_stop_loss=98.0,
                metadata={"decision_context": learning.build_trade_context(btc_signal)},
            )
            for _ in range(6):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=btc_position,
                    close_price=98.0,
                    profit_loss=-2.0,
                    exit_reason="SL",
                )
            bucket_payload = store.load_learning_model(
                "learning_prequential::cell::trend_pullback::long::trending::majors::limit"
            )
            generic_payload = store.load_learning_model(
                "learning_prequential::cell::trend_pullback::long::trending::limit"
            )
            unrelated_bucket_payload = store.load_learning_model(
                "learning_prequential::cell::trend_pullback::long::trending::high_beta_alts::limit"
            )
            self.assertGreater(float(bucket_payload.get("count", 0.0) or 0.0), 0.0)
            self.assertGreater(float(generic_payload.get("count", 0.0) or 0.0), 0.0)
            self.assertEqual(unrelated_bucket_payload, {})

    def test_trade_learning_engine_uses_symbol_bucket_specific_positive_cell_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(
                BotConfig(
                    learning_positive_update_min_samples=3.0,
                    learning_positive_update_max_calibration_gap=0.5,
                ),
                store,
            )
            btc_signal = {
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
                "strategy": "trend_breakout",
                "signal_quality": 0.66,
                "expected_edge_bps": 18.0,
                "regime": "trending",
                "rr_ratio": 2.0,
                "hurst_exponent": 0.3,
                "metadata": {"trend_direction": "bullish", "regime_confidence": 0.8},
            }
            btc_position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_breakout",
                initial_stop_loss=98.0,
                metadata={"decision_context": learning.build_trade_context(btc_signal)},
            )
            for _ in range(4):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=btc_position,
                    close_price=104.0,
                    profit_loss=4.0,
                    exit_reason="TP",
                )
            bucket_summary = learning._prequential_summary(
                strategy="trend_breakout",
                symbol_bucket="majors",
                side="long",
                regime="trending",
                order_profile="limit",
            )
            generic_summary = learning._prequential_summary(
                strategy="trend_breakout",
                side="long",
                regime="trending",
                order_profile="limit",
            )
            self.assertGreaterEqual(bucket_summary["samples"], 4.0)
            self.assertGreaterEqual(bucket_summary["avg_r_multiple"], 1.5)
            self.assertGreaterEqual(generic_summary["samples"], 4.0)
            self.assertTrue(bucket_summary["bucket_specific"])

    def test_trade_learning_engine_promotes_bucket_specific_winners_sooner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(
                BotConfig(
                    learning_positive_update_min_samples=3.0,
                    learning_positive_update_max_calibration_gap=0.5,
                    learning_positive_opportunity_min_samples=6.0,
                    learning_positive_prequential_min_samples=6.0,
                    learning_bucket_positive_sample_delta=2.0,
                ),
                store,
            )
            signal = {
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
                "strategy": "trend_breakout",
                "signal_quality": 0.66,
                "expected_edge_bps": 18.0,
                "regime": "trending",
                "rr_ratio": 2.0,
                "hurst_exponent": 0.3,
                "metadata": {"trend_direction": "bullish", "regime_confidence": 0.8},
            }
            position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_breakout",
                initial_stop_loss=98.0,
                metadata={"decision_context": learning.build_trade_context(signal)},
            )
            for _ in range(4):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=position,
                    close_price=104.0,
                    profit_loss=4.0,
                    exit_reason="TP",
                )
            context = learning.learning_context_for_signal(signal)
            self.assertTrue(context["positive_cell_evidence"])
            self.assertGreaterEqual(context["score_delta"], 1.6)
            self.assertGreaterEqual(context["risk_multiplier"], 1.06)

    def test_trade_learning_engine_exposes_explicit_cell_tracking_keys(self):
        config = BotConfig(starting_balance=1000.0, trading_mode="futures")
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(config, store)
            signal = {
                "symbol": "BTC/USDT",
                "strategy": "trend_pullback",
                "side": "long",
                "signal_quality": 0.78,
                "expected_edge_bps": 18.0,
                "rr_ratio": 2.0,
                "regime": "trending",
                "metadata": {"preferred_order_type": "limit"},
            }
            context = learning.build_trade_context(signal)
            self.assertEqual(context["order_type"], "limit")
            self.assertEqual(
                context["learning_cell_keys"]["generic"],
                "cell::trend_pullback::long::trending::limit",
            )
            self.assertEqual(
                context["learning_cell_keys"]["bucket_specific"],
                "cell::trend_pullback::long::trending::majors::limit",
            )

            learning_context = learning.learning_context_for_signal(signal)
            self.assertIn("cell_tracking", learning_context)
            self.assertEqual(
                learning_context["cell_tracking"]["generic_cell_key"],
                "cell::trend_pullback::long::trending::limit",
            )
            self.assertEqual(
                learning_context["cell_tracking"]["bucket_cell_key"],
                "cell::trend_pullback::long::trending::majors::limit",
            )
            self.assertEqual(
                learning_context["cell_tracking"]["active_cell_key"],
                "cell::trend_pullback::long::trending::limit",
            )

    def test_trade_learning_engine_exposes_asymmetric_learning_actions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(BotConfig(learning_family_rotation_min_samples=3.0), store)
            breakout_signal = {
                "symbol": "BTC/USDT",
                "strategy": "trend_breakout",
                "side": "long",
                "signal_quality": 0.78,
                "expected_edge_bps": 18.0,
                "rr_ratio": 2.0,
                "regime": "trending",
                "metadata": {"preferred_order_type": "limit"},
            }
            pullback_signal = dict(breakout_signal)
            pullback_signal["strategy"] = "trend_pullback"
            breakout_position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_breakout",
                initial_stop_loss=98.0,
                metadata={"decision_context": learning.build_trade_context(breakout_signal)},
            )
            pullback_position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_pullback",
                initial_stop_loss=98.0,
                metadata={"decision_context": learning.build_trade_context(pullback_signal)},
            )
            for _ in range(3):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=breakout_position,
                    close_price=98.0,
                    profit_loss=-2.0,
                    exit_reason="SL",
                )
            for _ in range(3):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=pullback_position,
                    close_price=104.0,
                    profit_loss=4.0,
                    exit_reason="TP",
                )
            learning_context = learning.learning_context_for_signal(breakout_signal)
            self.assertIn("asymmetric_learning", learning_context)
            self.assertTrue(learning_context["asymmetric_learning"]["throttle_active"])
            self.assertTrue(
                any(
                    action in {"throttle_family_rotation", "throttle_family_rotation_hard"}
                    for action in learning_context["asymmetric_learning"]["actions"]
                )
            )

    def test_trade_learning_engine_throttles_bucket_specific_losers_sooner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(
                BotConfig(
                    learning_bucket_negative_sample_delta=2.0,
                    learning_cell_gate_min_samples=6.0,
                ),
                store,
            )
            signal = {
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
                "strategy": "trend_pullback",
                "signal_quality": 0.72,
                "expected_edge_bps": 18.0,
                "regime": "trending",
                "rr_ratio": 2.0,
                "hurst_exponent": 0.35,
                "metadata": {"trend_direction": "bullish", "regime_confidence": 0.8},
            }
            position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_pullback",
                initial_stop_loss=98.0,
                metadata={"decision_context": learning.build_trade_context(signal)},
            )
            for _ in range(4):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=position,
                    close_price=98.0,
                    profit_loss=-2.0,
                    exit_reason="SL",
                )
            context = learning.learning_context_for_signal(signal)
            self.assertTrue(context["negative_cell_evidence"])
            self.assertLessEqual(context["risk_multiplier"], 0.84)
            self.assertLessEqual(context["score_delta"], -1.75)

    def test_trade_learning_engine_rotates_away_from_recent_losing_family(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(BotConfig(learning_family_rotation_min_samples=3.0), store)
            breakout_signal = {
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
                "strategy": "trend_breakout",
                "signal_quality": 0.68,
                "expected_edge_bps": 18.0,
                "regime": "trending",
                "rr_ratio": 2.0,
                "hurst_exponent": 0.3,
                "metadata": {"trend_direction": "bullish", "regime_confidence": 0.8},
            }
            pullback_signal = dict(breakout_signal)
            pullback_signal["strategy"] = "trend_pullback"
            breakout_position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_breakout",
                initial_stop_loss=98.0,
                metadata={"decision_context": learning.build_trade_context(breakout_signal)},
            )
            pullback_position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_pullback",
                initial_stop_loss=98.0,
                metadata={"decision_context": learning.build_trade_context(pullback_signal)},
            )
            for _ in range(3):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=breakout_position,
                    close_price=98.5,
                    profit_loss=-1.5,
                    exit_reason="SL",
                )
            for _ in range(3):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=pullback_position,
                    close_price=104.2,
                    profit_loss=4.2,
                    exit_reason="TP",
                )
            breakout_context = learning.learning_context_for_signal(breakout_signal)
            pullback_context = learning.learning_context_for_signal(pullback_signal)
            self.assertIn(breakout_context["family_rotation"]["status"], {"suppress_current", "suppress_current_hard"})
            self.assertIn(pullback_context["family_rotation"]["status"], {"promote_current", "promote_current_hard"})
            self.assertLess(breakout_context["risk_multiplier"], pullback_context["risk_multiplier"])
            self.assertLess(breakout_context["score_delta"], pullback_context["score_delta"])

    def test_trade_learning_engine_hard_rotates_away_from_severely_losing_family(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(BotConfig(learning_family_rotation_min_samples=3.0), store)
            breakout_signal = {
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
                "strategy": "trend_breakout",
                "signal_quality": 0.68,
                "expected_edge_bps": 18.0,
                "regime": "trending",
                "rr_ratio": 2.0,
                "hurst_exponent": 0.3,
                "metadata": {"trend_direction": "bullish", "regime_confidence": 0.8},
            }
            pullback_signal = dict(breakout_signal)
            pullback_signal["strategy"] = "trend_pullback"
            breakout_position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_breakout",
                initial_stop_loss=98.0,
                metadata={"decision_context": learning.build_trade_context(breakout_signal)},
            )
            pullback_position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_pullback",
                initial_stop_loss=98.0,
                metadata={"decision_context": learning.build_trade_context(pullback_signal)},
            )
            for _ in range(3):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=breakout_position,
                    close_price=97.0,
                    profit_loss=-3.0,
                    exit_reason="SL",
                )
            for _ in range(3):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=pullback_position,
                    close_price=105.0,
                    profit_loss=5.0,
                    exit_reason="TP",
                )
            breakout_context = learning.learning_context_for_signal(breakout_signal)
            pullback_context = learning.learning_context_for_signal(pullback_signal)
            self.assertEqual(breakout_context["family_rotation"]["status"], "suppress_current_hard")
            self.assertEqual(pullback_context["family_rotation"]["status"], "promote_current_hard")
            self.assertIn("throttle_family_rotation_hard", breakout_context["asymmetric_learning"]["actions"])
            self.assertIn("promote_family_rotation_hard", pullback_context["asymmetric_learning"]["actions"])
            self.assertLessEqual(breakout_context["risk_multiplier"], 0.74)
            self.assertGreater(pullback_context["score_delta"], breakout_context["score_delta"])

    def test_trade_learning_engine_recovers_from_hard_rotation_after_recent_improvement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(BotConfig(learning_family_rotation_min_samples=3.0), store)
            breakout_signal = {
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 104.0,
                "strategy": "trend_breakout",
                "signal_quality": 0.68,
                "expected_edge_bps": 18.0,
                "regime": "trending",
                "rr_ratio": 2.0,
                "hurst_exponent": 0.3,
                "metadata": {"trend_direction": "bullish", "regime_confidence": 0.8},
            }
            pullback_signal = dict(breakout_signal)
            pullback_signal["strategy"] = "trend_pullback"
            breakout_position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_breakout",
                initial_stop_loss=98.0,
                metadata={"decision_context": learning.build_trade_context(breakout_signal)},
            )
            pullback_position = Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_pullback",
                initial_stop_loss=98.0,
                metadata={"decision_context": learning.build_trade_context(pullback_signal)},
            )
            for _ in range(3):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=breakout_position,
                    close_price=97.0,
                    profit_loss=-3.0,
                    exit_reason="SL",
                )
            for _ in range(3):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=pullback_position,
                    close_price=105.0,
                    profit_loss=5.0,
                    exit_reason="TP",
                )
            for _ in range(2):
                learning.record_closed_trade(
                    symbol="BTC/USDT",
                    position=breakout_position,
                    close_price=104.0,
                    profit_loss=4.0,
                    exit_reason="TP",
                )
            breakout_context = learning.learning_context_for_signal(breakout_signal)
            pullback_context = learning.learning_context_for_signal(pullback_signal)
            self.assertIn(breakout_context["family_rotation"]["status"], {"neutral", "suppress_current"})
            self.assertIn(pullback_context["family_rotation"]["status"], {"neutral", "promote_current"})
            self.assertNotIn("throttle_family_rotation_hard", breakout_context["asymmetric_learning"]["actions"])
            if breakout_context["family_rotation"]["status"] == "suppress_current":
                self.assertIn("recover_family_rotation", breakout_context["asymmetric_learning"]["actions"])

    def test_purged_walk_forward_splits_apply_embargo_without_overlap(self):
        index = pd.date_range("2025-01-01", periods=16, freq="15min")
        frame = pd.DataFrame(
            {
                "open": [100.0 + i for i in range(16)],
                "high": [101.0 + i for i in range(16)],
                "low": [99.0 + i for i in range(16)],
                "close": [100.5 + i for i in range(16)],
                "volume": [1000.0] * 16,
            },
            index=index,
        )
        windows = purged_walk_forward_splits(
            frame,
            train_bars=6,
            validation_bars=2,
            test_bars=3,
            embargo_bars=1,
            step_bars=3,
        )

        self.assertGreaterEqual(len(windows), 2)
        first = windows[0]
        self.assertEqual(len(first["train"]), 6)
        self.assertEqual(len(first["validation"]), 2)
        self.assertEqual(len(first["test"]), 3)
        self.assertLess(first["train"].index[-1], first["validation"].index[0])
        self.assertLess(first["validation"].index[-1], first["test"].index[0])

    def test_simulation_engine_uses_conservative_exit_for_ambiguous_bar(self):
        engine = BacktestEngine(BotConfig())
        position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=102.0,
            strategy="trend_breakout",
        )
        bar = pd.Series({"open": 100.0, "high": 103.0, "low": 97.0, "close": 101.0})

        reason, price, details = engine._trigger_exit(position, bar)

        self.assertEqual(reason, "SL")
        self.assertEqual(price, 98.0)
        self.assertTrue(details["ambiguous_bar"])

    def test_simulation_engine_can_trail_short_positions(self):
        engine = BacktestEngine(BotConfig(trading_mode="futures"))
        engine.signals = mock.Mock()
        engine.signals.compute_atr = mock.Mock(return_value=1.0)
        position = Position(
            symbol="BTC/USDT",
            side="short",
            entry_price=100.0,
            size=1.0,
            stop_loss=102.0,
            take_profit=96.0,
            strategy="trend_breakout",
            initial_stop_loss=102.0,
        )
        bar = pd.Series({"open": 100.0, "high": 100.4, "low": 96.0, "close": 97.0})

        engine._update_dynamic_risk(position, "BTC/USDT", bar)

        self.assertLess(position.stop_loss, 100.0)

    def test_simulation_engine_time_stop_exits_stale_trade(self):
        engine = BacktestEngine(
            BotConfig(
                time_stop_soft_holding_multiplier=1.0,
                time_stop_soft_min_r_multiple=0.2,
                time_stop_hard_holding_multiplier=2.0,
                time_stop_hard_min_r_multiple=0.4,
                time_stop_pullback_soft_multiplier=1.0,
                time_stop_pullback_hard_multiplier=2.0,
            )
        )
        position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=104.0,
            strategy="trend_pullback",
            opened_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=95),
            initial_stop_loss=98.0,
            metadata={"signal_snapshot": {"expected_holding_minutes": 60}},
        )
        bar = pd.Series({"open": 100.0, "high": 100.4, "low": 99.8, "close": 100.1})

        reason, price = engine._time_stop_exit(position, bar, dt.datetime.now(dt.timezone.utc))

        self.assertEqual(reason, "TIME_SOFT")
        self.assertEqual(price, 100.1)

    def test_simulation_engine_time_stop_is_strategy_aware_by_family(self):
        engine = BacktestEngine(
            BotConfig(
                time_stop_soft_holding_multiplier=1.0,
                time_stop_soft_min_r_multiple=0.2,
                time_stop_hard_holding_multiplier=2.0,
                time_stop_hard_min_r_multiple=0.4,
                time_stop_breakout_soft_multiplier=0.50,
                time_stop_pullback_soft_multiplier=1.50,
            )
        )
        now = dt.datetime.now(dt.timezone.utc)
        breakout_position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=104.0,
            strategy="trend_breakout",
            opened_at=now - dt.timedelta(minutes=40),
            initial_stop_loss=98.0,
            metadata={"signal_snapshot": {"expected_holding_minutes": 60}},
        )
        pullback_position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=104.0,
            strategy="trend_pullback",
            opened_at=now - dt.timedelta(minutes=40),
            initial_stop_loss=98.0,
            metadata={"signal_snapshot": {"expected_holding_minutes": 60}},
        )
        bar = pd.Series({"open": 100.0, "high": 100.3, "low": 99.9, "close": 100.1})

        breakout_reason, breakout_price = engine._time_stop_exit(breakout_position, bar, now)
        pullback_reason, pullback_price = engine._time_stop_exit(pullback_position, bar, now)

        self.assertEqual(breakout_reason, "TIME_SOFT")
        self.assertEqual(breakout_price, 100.1)
        self.assertEqual(pullback_reason, "OPEN")
        self.assertIsNone(pullback_price)

    def test_simulation_engine_profit_protection_locks_breakout_earlier(self):
        engine = BacktestEngine(
            BotConfig(
                breakeven_rr=10.0,
                trailing_rr=10.0,
                profit_protect_breakout_trigger_rr=0.5,
                profit_protect_breakout_lock_rr=0.25,
            )
        )
        position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=110.0,
            strategy="trend_breakout",
            initial_stop_loss=98.0,
        )
        bar = pd.Series({"open": 100.0, "high": 101.6, "low": 99.9, "close": 101.2})

        engine._update_dynamic_risk(position, "BTC/USDT", bar)

        self.assertAlmostEqual(position.stop_loss, 100.5)

    def test_simulation_engine_profit_protection_keeps_pullback_looser_before_trigger(self):
        engine = BacktestEngine(
            BotConfig(
                breakeven_rr=10.0,
                trailing_rr=10.0,
                profit_protect_pullback_trigger_rr=1.2,
                profit_protect_pullback_lock_rr=0.10,
            )
        )
        position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=110.0,
            strategy="trend_pullback",
            initial_stop_loss=98.0,
        )
        bar = pd.Series({"open": 100.0, "high": 101.6, "low": 99.9, "close": 101.2})

        engine._update_dynamic_risk(position, "BTC/USDT", bar)

        self.assertAlmostEqual(position.stop_loss, 98.0)

    def test_simulation_engine_mean_reversion_profit_capture_locks_earlier(self):
        engine = BacktestEngine(
            BotConfig(
                breakeven_rr=10.0,
                trailing_rr=10.0,
                profit_protect_mean_reversion_trigger_rr=10.0,
                mean_reversion_profit_capture_trigger_rr=0.40,
                mean_reversion_profit_capture_lock_rr=0.30,
            )
        )
        position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=110.0,
            strategy="mean_reversion",
            initial_stop_loss=98.0,
        )
        bar = pd.Series({"open": 100.0, "high": 101.0, "low": 99.9, "close": 100.7})

        engine._update_dynamic_risk(position, "BTC/USDT", bar)

        self.assertAlmostEqual(position.stop_loss, 100.6)

    def test_simulation_engine_mean_reversion_partial_profit_take_reduces_size(self):
        engine = HistoricalSimulationEngine(
            BotConfig(
                partial_profit_take_mean_reversion_trigger_rr=0.60,
                partial_profit_take_mean_reversion_fraction=0.50,
            ),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=2.0,
            stop_loss=98.0,
            take_profit=110.0,
            strategy="mean_reversion",
            initial_stop_loss=98.0,
        )
        position.metadata = {"initial_size": 2.0}
        bar = pd.Series({"open": 100.0, "high": 101.6, "low": 99.9, "close": 101.0, "volume": 1000.0})

        engine._maybe_partial_profit_take(position, bar)
        expected_exit_price, _ = engine._apply_exit_costs(position, float(bar["close"]), bar)
        expected_partial_gross = engine._gross_pl("long", 100.0, expected_exit_price, 1.0)

        self.assertAlmostEqual(position.size, 1.0)
        self.assertTrue(position.metadata["partial_profit_taken"])
        self.assertAlmostEqual(float(position.metadata["partial_realized_gross_pl"]), expected_partial_gross)

    def test_simulation_engine_attaches_entry_atr_to_position_metadata(self):
        class StubSignals:
            def compute_atr(self, symbol, timeframe="15m", period=14):
                return 1.25

        engine = HistoricalSimulationEngine(BotConfig(), signal_engine_cls=QuietSignalEngine)
        engine.signals = StubSignals()
        position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=104.0,
            strategy="trend_breakout",
            initial_stop_loss=98.0,
        )

        engine._attach_position_metadata(position, {"symbol": "BTC/USDT", "strategy": "trend_breakout"}, {}, "trace")

        self.assertAlmostEqual(float(position.metadata.get("entry_atr", 0.0) or 0.0), 1.25)

    def test_simulation_engine_volatility_tightening_hits_breakout_harder_than_pullback(self):
        class StubSignals:
            def compute_atr(self, symbol, timeframe="15m", period=14):
                return 2.0

        engine = BacktestEngine(
            BotConfig(
                breakeven_rr=10.0,
                trailing_rr=10.0,
                volatility_tightening_trigger_ratio=1.2,
                volatility_tightening_breakout_rr=0.30,
                volatility_tightening_pullback_rr=0.10,
            )
        )
        engine.signals = StubSignals()
        breakout_position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=110.0,
            strategy="trend_breakout",
            initial_stop_loss=98.0,
            metadata={"entry_atr": 1.0},
        )
        pullback_position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=110.0,
            strategy="trend_pullback",
            initial_stop_loss=98.0,
            metadata={"entry_atr": 1.0},
        )
        bar = pd.Series({"open": 100.0, "high": 100.8, "low": 99.9, "close": 100.4})

        engine._update_dynamic_risk(breakout_position, "BTC/USDT", bar)
        engine._update_dynamic_risk(pullback_position, "BTC/USDT", bar)

        self.assertAlmostEqual(breakout_position.stop_loss, 100.6)
        self.assertAlmostEqual(pullback_position.stop_loss, 100.2)

    def test_simulation_engine_builds_exit_quality_summary_from_excursions(self):
        engine = HistoricalSimulationEngine(BotConfig(), signal_engine_cls=QuietSignalEngine)
        engine.trades = [
            {
                "strategy": "trend_breakout",
                "exit_reason": "TRAILING_STOP",
                "pl": 15.0,
                "mfe_r": 1.8,
                "mae_r": 0.4,
                "giveback_r": 0.5,
            },
            {
                "strategy": "trend_breakout",
                "exit_reason": "TRAILING_STOP",
                "pl": -5.0,
                "mfe_r": 0.6,
                "mae_r": 1.1,
                "giveback_r": 0.6,
            },
            {
                "strategy": "trend_pullback",
                "exit_reason": "TIME_HARD",
                "pl": 4.0,
                "mfe_r": 1.2,
                "mae_r": 0.3,
                "giveback_r": 1.05,
            },
        ]

        summary = engine._build_exit_quality_summary()

        self.assertIn("TRAILING_STOP", summary["by_exit_reason"])
        self.assertAlmostEqual(summary["by_exit_reason"]["TRAILING_STOP"]["expectancy"], 5.0)
        self.assertAlmostEqual(summary["by_exit_reason"]["TRAILING_STOP"]["avg_mfe_r"], 1.2)
        self.assertAlmostEqual(summary["by_exit_reason"]["TRAILING_STOP"]["avg_giveback_r"], 0.55)
        self.assertEqual(summary["giveback_by_strategy"]["trend_breakout"]["mfe_above_1r_count"], 1.0)
        self.assertEqual(summary["giveback_by_strategy"]["trend_pullback"]["gave_back_below_0_25r_count"], 1.0)

    def test_simulation_engine_mean_reversion_reclaim_failure_exits_after_snapback_stalls(self):
        engine = HistoricalSimulationEngine(
            BotConfig(
                mean_reversion_reclaim_failure_activation_rr=0.25,
                mean_reversion_reclaim_failure_buffer_rr=0.05,
            ),
            signal_engine_cls=QuietSignalEngine,
        )
        position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=110.0,
            strategy="mean_reversion",
            initial_stop_loss=98.0,
        )
        position.metadata = {"mfe_r": 0.40}
        bar = pd.Series({"open": 100.4, "high": 100.6, "low": 99.9, "close": 100.0})

        reason, price, details = engine._trigger_exit(position, bar)

        self.assertEqual(reason, "MEAN_REVERSION_RECLAIM_FAIL")
        self.assertAlmostEqual(price, 100.0)
        self.assertEqual(details["path_assumption"], "close_reclaim_failure")

    def test_simulation_engine_reentry_cooldown_blocks_mean_reversion_after_reclaim_fail(self):
        engine = HistoricalSimulationEngine(
            BotConfig(mean_reversion_reclaim_failure_cooldown_bars=6),
            signal_engine_cls=QuietSignalEngine,
        )
        now = dt.datetime.now(dt.timezone.utc)
        engine._reset_run_state()
        engine._now = now
        position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=110.0,
            strategy="mean_reversion",
            initial_stop_loss=98.0,
        )

        engine._register_reentry_cooldown(position, "MEAN_REVERSION_RECLAIM_FAIL", now)
        reason = engine._reentry_cooldown_reason({"symbol": "BTC/USDT", "strategy": "mean_reversion", "side": "long"})

        self.assertEqual(reason, "mean_reversion_reclaim_failure_cooldown")

    def test_simulation_engine_reentry_cooldown_blocks_breakout_after_volatility_stopout(self):
        engine = HistoricalSimulationEngine(
            BotConfig(breakout_volatility_exit_cooldown_bars=8),
            signal_engine_cls=QuietSignalEngine,
        )
        now = dt.datetime.now(dt.timezone.utc)
        engine._reset_run_state()
        engine._now = now
        position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=110.0,
            strategy="trend_breakout",
            initial_stop_loss=98.0,
        )
        position.metadata = {"volatility_tightened": True}

        engine._register_reentry_cooldown(position, "SL", now)
        reason = engine._reentry_cooldown_reason({"symbol": "BTC/USDT", "strategy": "trend_breakout", "side": "long"})

        self.assertEqual(reason, "breakout_volatility_exit_cooldown")

    def test_simulation_engine_final_close_includes_prior_partial_profit_take(self):
        engine = HistoricalSimulationEngine(BotConfig(), signal_engine_cls=QuietSignalEngine)
        engine._reset_run_state()
        now = dt.datetime.now(dt.timezone.utc)
        position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=99.0,
            take_profit=105.0,
            strategy="mean_reversion",
            opened_at=now - dt.timedelta(minutes=30),
            initial_stop_loss=99.0,
            fee_paid=0.2,
        )
        position.metadata = {
            "initial_size": 2.0,
            "partial_profit_taken": True,
            "partial_realized_gross_pl": 1.5,
            "partial_exit_fees": 0.1,
            "execution_context": {"order_type": "limit"},
            "signal_snapshot": {"fast_move": False},
        }
        engine.state.open_positions["BTC/USDT"] = [position]
        bar = pd.Series({"open": 100.4, "high": 101.2, "low": 98.8, "close": 99.1, "volume": 1000.0})

        engine._manage_open_positions({"BTC/USDT": bar}, now)
        expected_exit_price, expected_exit_fee = engine._apply_exit_costs(position, float(position.stop_loss), bar)
        expected_remaining_gross = engine._gross_pl("long", 100.0, expected_exit_price, 1.0)

        self.assertEqual(len(engine.trades), 1)
        self.assertAlmostEqual(engine.trades[0]["size"], 2.0)
        self.assertAlmostEqual(engine.trades[0]["gross_pl"], 1.5 + expected_remaining_gross, places=6)
        self.assertAlmostEqual(engine.trades[0]["fees"], 0.2 + 0.1 + expected_exit_fee, places=6)

    def test_simulation_engine_family_trailing_activates_breakout_earlier_than_pullback(self):
        class StubSignals:
            def compute_atr(self, symbol, timeframe="15m", period=14):
                return 1.0

        engine = BacktestEngine(
            BotConfig(
                breakeven_rr=10.0,
                profit_protect_breakout_trigger_rr=10.0,
                profit_protect_pullback_trigger_rr=10.0,
                volatility_tightening_trigger_ratio=10.0,
                trailing_breakout_rr=1.0,
                trailing_pullback_rr=2.0,
                trailing_breakout_atr_mult=0.8,
                trailing_pullback_atr_mult=1.2,
            )
        )
        engine.signals = StubSignals()
        breakout_position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=110.0,
            strategy="trend_breakout",
            initial_stop_loss=98.0,
        )
        pullback_position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=110.0,
            strategy="trend_pullback",
            initial_stop_loss=98.0,
        )
        bar = pd.Series({"open": 100.0, "high": 102.4, "low": 99.9, "close": 101.8})

        engine._update_dynamic_risk(breakout_position, "BTC/USDT", bar)
        engine._update_dynamic_risk(pullback_position, "BTC/USDT", bar)

        self.assertAlmostEqual(breakout_position.stop_loss, 101.6)
        self.assertAlmostEqual(pullback_position.stop_loss, 98.0)

    def test_simulation_engine_family_trailing_uses_looser_pullback_distance_once_active(self):
        class StubSignals:
            def compute_atr(self, symbol, timeframe="15m", period=14):
                return 1.0

        engine = BacktestEngine(
            BotConfig(
                breakeven_rr=10.0,
                profit_protect_breakout_trigger_rr=10.0,
                profit_protect_pullback_trigger_rr=10.0,
                volatility_tightening_trigger_ratio=10.0,
                trailing_breakout_rr=1.0,
                trailing_pullback_rr=1.0,
                trailing_breakout_atr_mult=0.8,
                trailing_pullback_atr_mult=1.2,
            )
        )
        engine.signals = StubSignals()
        breakout_position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=110.0,
            strategy="trend_breakout",
            initial_stop_loss=98.0,
        )
        pullback_position = Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            stop_loss=98.0,
            take_profit=110.0,
            strategy="trend_pullback",
            initial_stop_loss=98.0,
        )
        bar = pd.Series({"open": 100.0, "high": 102.4, "low": 99.9, "close": 101.8})

        engine._update_dynamic_risk(breakout_position, "BTC/USDT", bar)
        engine._update_dynamic_risk(pullback_position, "BTC/USDT", bar)

        self.assertAlmostEqual(breakout_position.stop_loss, 101.6)
        self.assertAlmostEqual(pullback_position.stop_loss, 101.2)

    def test_simulation_engine_writes_checkpoint_and_resumes(self):
        index = pd.date_range("2025-01-01", periods=140, freq="15min")
        frame = pd.DataFrame(
            {
                "open": [100.0 + (i * 0.1) for i in range(140)],
                "high": [100.3 + (i * 0.1) for i in range(140)],
                "low": [99.7 + (i * 0.1) for i in range(140)],
                "close": [100.1 + (i * 0.1) for i in range(140)],
                "volume": [1000.0 + i for i in range(140)],
            },
            index=index,
        )

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return frame.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            stop_counter = {"calls": 0}

            def stop_requested():
                stop_counter["calls"] += 1
                return stop_counter["calls"] > 5

            interrupted = LocalSimulationEngine(
                BotConfig(
                    backtest_warmup_candles=20,
                    simulation_snapshot_interval_bars=2,
                    simulation_checkpoint_interval_bars=2,
                ),
                signal_engine_cls=QuietSignalEngine,
                artifact_dir=tmpdir,
                stop_requested=stop_requested,
            )
            partial = interrupted.run_backtest("BTC/USDT", timeframe="15m", days=30)
            self.assertTrue(partial["stopped_early"])
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "simulation_checkpoint.json")))

            resumed = LocalSimulationEngine(
                BotConfig(
                    backtest_warmup_candles=20,
                    simulation_snapshot_interval_bars=2,
                    simulation_checkpoint_interval_bars=2,
                ),
                signal_engine_cls=QuietSignalEngine,
                artifact_dir=tmpdir,
                stop_requested=lambda: False,
            )
            completed = resumed.run_backtest("BTC/USDT", timeframe="15m", days=30)
            self.assertFalse(completed["stopped_early"])
            self.assertTrue(completed["resumed_from_checkpoint"])
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "simulation_checkpoint.json")))

    def test_simulation_engine_respects_pending_capacity_and_reports_reason(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        class MultiSymbolSignals:
            def __init__(self, config, exch):
                self.exch = exch
                self.learning_context_provider = None

            def generate_signal(self, symbol):
                candles = self.exch.fetch_ohlcv(symbol, "15m", limit=200)
                if len(candles) == 101:
                    return {
                        "side": "long",
                        "entry_price": 100.0,
                        "stop_loss": 99.0,
                        "take_profit": 103.0,
                        "strategy": "unit_test_ranked_entry",
                        "expected_holding_minutes": 60,
                        "signal_quality": 0.75,
                        "expected_edge_bps": 15.0,
                        "rr_ratio": 3.0,
                        "metadata": {"ensemble_score": 90.0 if symbol == "BTC/USDT" else 80.0},
                    }
                return None

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", max_open_positions=1, backtest_warmup_candles=20),
            signal_engine_cls=MultiSymbolSignals,
        )
        result = engine.run_campaign(["BTC/USDT", "ETH/USDT"], timeframe="15m", days=10)

        self.assertEqual(result["submitted_orders"], 1)
        self.assertGreaterEqual(result["decision_diagnostics"]["skip_reasons"].get("max_open_positions", 0), 1)

    def test_candidate_rank_score_penalizes_existing_family_crowding(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine.state.open_positions["BTC/USDT"] = [
            Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=2.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_breakout",
                initial_stop_loss=98.0,
            )
        ]
        btc_signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.72,
            "expected_edge_bps": 18.0,
            "rr_ratio": 1.8,
            "expected_holding_minutes": 240,
            "metadata": {"cross_sectional_score": 80.0},
        }
        eth_signal = dict(btc_signal)
        eth_signal["symbol"] = "ETH/USDT"
        self.assertLess(engine._candidate_rank_score(btc_signal), engine._candidate_rank_score(eth_signal))

    def test_simulation_campaign_prefers_diversified_candidate_when_scores_are_close(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        class MultiSymbolSignals:
            def __init__(self, config, exch):
                self.exch = exch
                self.learning_context_provider = None

            def generate_signal(self, symbol):
                candles = self.exch.fetch_ohlcv(symbol, "15m", limit=200)
                if len(candles) != 101:
                    return None
                if symbol == "BTC/USDT":
                    return {
                        "side": "long",
                        "entry_price": 100.0,
                        "stop_loss": 99.0,
                        "take_profit": 103.0,
                        "strategy": "trend_pullback",
                        "expected_holding_minutes": 60,
                        "signal_quality": 0.75,
                        "expected_edge_bps": 16.0,
                        "rr_ratio": 3.0,
                        "metadata": {"cross_sectional_score": 92.0},
                    }
                if symbol == "ETH/USDT":
                    return {
                        "side": "long",
                        "entry_price": 100.0,
                        "stop_loss": 99.0,
                        "take_profit": 103.0,
                        "strategy": "trend_pullback",
                        "expected_holding_minutes": 60,
                        "signal_quality": 0.75,
                        "expected_edge_bps": 16.0,
                        "rr_ratio": 3.0,
                        "metadata": {"cross_sectional_score": 90.5},
                    }
                return {
                    "side": "long",
                    "entry_price": 100.0,
                    "stop_loss": 99.0,
                    "take_profit": 103.0,
                    "strategy": "trend_pullback",
                    "expected_holding_minutes": 60,
                    "signal_quality": 0.75,
                    "expected_edge_bps": 16.0,
                    "rr_ratio": 3.0,
                    "metadata": {"cross_sectional_score": 90.0},
                }

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", max_open_positions=2, backtest_warmup_candles=20),
            signal_engine_cls=MultiSymbolSignals,
        )
        result = engine.run_campaign(["BTC/USDT", "ETH/USDT", "ADA/USDT"], timeframe="15m", days=10)

        submitted_symbols = set(result["decision_diagnostics"]["submitted_by_symbol"].keys())
        self.assertIn("BTC/USDT", submitted_symbols)
        self.assertIn("ADA/USDT", submitted_symbols)
        self.assertNotIn("ETH/USDT", submitted_symbols)

    def test_simulation_campaign_filters_universe_by_tradability_before_signal_generation(self):
        timestamps = pd.date_range("2025-01-01", periods=21, freq="15min")

        def build_df(volume: float, spread_scale: float = 1.0) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "timestamp": ts,
                        "open": 100.0,
                        "high": 100.5 * spread_scale,
                        "low": 99.5 / max(spread_scale, 1e-9),
                        "close": 100.0,
                        "volume": volume,
                    }
                    for ts in timestamps
                ]
            ).set_index("timestamp")

        datasets = {
            "BTC/USDT": build_df(5000.0, 1.0),
            "ETH/USDT": build_df(200.0, 1.03),
            "ADA/USDT": build_df(50.0, 1.05),
        }

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return datasets[symbol].copy()

        class MultiSymbolSignals:
            def __init__(self, config, exch):
                self.exch = exch
                self.learning_context_provider = None

            def generate_signal(self, symbol):
                candles = self.exch.fetch_ohlcv(symbol, "15m", limit=200)
                if len(candles) != 21:
                    return None
                return {
                    "side": "long",
                    "entry_price": 100.0,
                    "stop_loss": 99.0,
                    "take_profit": 103.0,
                    "strategy": "unit_test_ranked_entry",
                    "expected_holding_minutes": 60,
                    "signal_quality": 0.75,
                    "expected_edge_bps": 16.0,
                    "rr_ratio": 3.0,
                    "metadata": {"cross_sectional_score": 90.0},
                }

        engine = LocalSimulationEngine(
            BotConfig(
                starting_balance=1000.0,
                trading_mode="futures",
                max_open_positions=3,
                backtest_warmup_candles=20,
                simulation_universe_top_n=1,
                simulation_universe_tradability_floor=1.0,
            ),
            signal_engine_cls=MultiSymbolSignals,
        )
        result = engine.run_campaign(["BTC/USDT", "ETH/USDT", "ADA/USDT"], timeframe="15m", days=10)

        universe = result["campaign_summary"]["universe_selection"]

        self.assertEqual(universe["eligible_symbols"], ["BTC/USDT"])
        self.assertEqual(universe["rejected_symbols"]["ETH/USDT"]["reason"], "outside_top_n")
        self.assertEqual(universe["rejected_symbols"]["ADA/USDT"]["reason"], "outside_top_n")
        self.assertIn("BTC/USDT", universe["scored_symbols"])
        self.assertEqual(result["campaign_summary"]["trade_flow"]["raw_signals"], 1)
        self.assertEqual(result["campaign_summary"]["trade_flow"]["proposals"], 1)

    def test_candidate_rank_score_penalizes_same_symbol_and_family_crowding(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        base_signal = {
            "symbol": "AVAX/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.75,
            "expected_edge_bps": 18.0,
            "rr_ratio": 2.0,
            "expected_holding_minutes": 180,
            "metadata": {"cross_sectional_score": 80.0},
        }
        same_symbol = dict(base_signal)
        same_family = dict(base_signal)
        same_family["symbol"] = "DOT/USDT"
        other_family = dict(base_signal)
        other_family["symbol"] = "ETH/USDT"
        provisional = [base_signal]
        self.assertLess(engine._candidate_rank_score(same_symbol, provisional_signals=provisional), engine._candidate_rank_score(other_family, provisional_signals=provisional))
        self.assertLess(engine._candidate_rank_score(same_family, provisional_signals=provisional), engine._candidate_rank_score(other_family, provisional_signals=provisional))

    def test_signal_engine_rejects_weak_pullback_by_pullback_specific_reliability_floor(self):
        engine = SignalEngine(
            BotConfig(
                min_signal_quality_score=0.55,
                min_signal_quality_score_pullback=0.62,
                min_expected_edge_bps=8.0,
                min_expected_edge_bps_pullback=14.0,
                min_reliable_rr_ratio_trend=1.35,
                min_reliable_rr_ratio_pullback=1.55,
            ),
            DummyExchange(),
        )
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
            "entry_price": 100.0,
            "stop_loss": 98.5,
            "take_profit": 102.2,
            "strategy": "trend_pullback",
            "signal_quality": 0.59,
            "expected_edge_bps": 11.0,
            "rr_ratio": 1.46,
            "hurst_exponent": 0.30,
            "metadata": {"preferred_order_type": "limit"},
            "ensemble": {},
        }
        self.assertFalse(engine._passes_reliability_checks("BTC/USDT", signal, regime, proposal_count=1))

    def test_signal_engine_allows_strong_pullback_that_meets_pullback_specific_reliability_floor(self):
        engine = SignalEngine(
            BotConfig(
                min_signal_quality_score=0.55,
                min_signal_quality_score_pullback=0.62,
                min_expected_edge_bps=8.0,
                min_expected_edge_bps_pullback=14.0,
                min_reliable_rr_ratio_trend=1.35,
                min_reliable_rr_ratio_pullback=1.55,
            ),
            DummyExchange(),
        )
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
            "entry_price": 100.0,
            "stop_loss": 98.6,
            "take_profit": 102.4,
            "strategy": "trend_pullback",
            "signal_quality": 0.66,
            "expected_edge_bps": 16.0,
            "rr_ratio": 1.72,
            "hurst_exponent": 0.30,
            "metadata": {"preferred_order_type": "limit"},
            "ensemble": {},
        }
        self.assertTrue(engine._passes_reliability_checks("BTC/USDT", signal, regime, proposal_count=1))

    def test_candidate_rank_score_penalizes_directional_family_cluster_crowding(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine.state.open_positions["AVAX/USDT"] = [
            Position(
                symbol="AVAX/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_pullback",
                initial_stop_loss=98.0,
            )
        ]
        same_direction_family = {
            "symbol": "DOT/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.76,
            "expected_edge_bps": 20.0,
            "rr_ratio": 2.1,
            "expected_holding_minutes": 180,
            "metadata": {"cross_sectional_score": 84.0},
        }
        different_family = dict(same_direction_family)
        different_family["symbol"] = "ETH/USDT"
        self.assertLess(engine._candidate_rank_score(same_direction_family), engine._candidate_rank_score(different_family))

    def test_candidate_rank_score_penalizes_same_bucket_and_rewards_new_bucket(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine.state.open_positions["BTC/USDT"] = [
            Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                size=1.0,
                stop_loss=98.0,
                take_profit=104.0,
                strategy="trend_pullback",
                initial_stop_loss=98.0,
            )
        ]
        same_bucket = {
            "symbol": "ETH/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.76,
            "expected_edge_bps": 20.0,
            "rr_ratio": 2.1,
            "expected_holding_minutes": 180,
            "metadata": {"cross_sectional_score": 84.0},
        }
        new_bucket = dict(same_bucket)
        new_bucket["symbol"] = "BNB/USDT"
        self.assertLess(engine._candidate_rank_score(same_bucket), engine._candidate_rank_score(new_bucket))

    def test_candidate_rank_score_prefers_positive_learning_evidence(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        base = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.76,
            "expected_edge_bps": 20.0,
            "rr_ratio": 2.1,
            "expected_holding_minutes": 180,
            "metadata": {"cross_sectional_score": 84.0},
        }
        with_positive = dict(base)
        with_positive["metadata"] = {
            "cross_sectional_score": 84.0,
            "learning_context": {"score_delta": 2.0, "positive_cell_evidence": True},
        }
        without_learning = dict(base)
        self.assertGreater(engine._candidate_rank_score(with_positive), engine._candidate_rank_score(without_learning))

    def test_candidate_rank_score_penalizes_negative_learning_evidence(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        base = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.76,
            "expected_edge_bps": 20.0,
            "rr_ratio": 2.1,
            "expected_holding_minutes": 180,
            "metadata": {"cross_sectional_score": 84.0},
        }
        with_negative = dict(base)
        with_negative["metadata"] = {
            "cross_sectional_score": 84.0,
            "learning_context": {"score_delta": -1.5, "negative_cell_evidence": True},
        }
        self.assertLess(engine._candidate_rank_score(with_negative), engine._candidate_rank_score(base))

    def test_candidate_rank_score_prefers_positive_realized_symbol_and_strategy_expectancy(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine.trades = [
            {"symbol": "BTC/USDT", "strategy": "trend_pullback", "pl": 2.5},
            {"symbol": "BTC/USDT", "strategy": "trend_pullback", "pl": 1.5},
        ]
        engine._closed_by_symbol["BTC/USDT"] = 2
        engine._closed_by_strategy["trend_pullback"] = 2
        stronger = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.76,
            "expected_edge_bps": 20.0,
            "rr_ratio": 2.1,
            "expected_holding_minutes": 180,
            "metadata": {"cross_sectional_score": 84.0},
        }
        baseline = dict(stronger)
        baseline["symbol"] = "ETH/USDT"
        baseline["strategy"] = "mean_reversion"
        self.assertGreater(engine._candidate_rank_score(stronger), engine._candidate_rank_score(baseline))

    def test_portfolio_duplicate_family_throttle_blocks_weaker_clustered_candidate(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        stronger = {
            "symbol": "AVAX/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.78,
            "expected_edge_bps": 22.0,
            "rr_ratio": 2.2,
            "expected_holding_minutes": 180,
            "metadata": {"cross_sectional_score": 92.0},
        }
        weaker = dict(stronger)
        weaker["symbol"] = "DOT/USDT"
        weaker["metadata"] = {"cross_sectional_score": 82.0}
        reason = engine._portfolio_duplicate_throttle_reason(
            weaker,
            provisional_signals=[stronger],
            candidate_rank_score=engine._candidate_rank_score(weaker, provisional_signals=[stronger]),
        )
        self.assertEqual(reason, "portfolio_duplicate_family_throttle")

    def test_portfolio_duplicate_bucket_throttle_blocks_weaker_same_bucket_candidate(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        stronger = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.78,
            "expected_edge_bps": 22.0,
            "rr_ratio": 2.2,
            "expected_holding_minutes": 180,
            "metadata": {"cross_sectional_score": 92.0},
        }
        weaker = dict(stronger)
        weaker["symbol"] = "ETH/USDT"
        weaker["metadata"] = {"cross_sectional_score": 86.0}
        reason = engine._portfolio_duplicate_bucket_throttle_reason(
            weaker,
            provisional_signals=[stronger],
            candidate_rank_score=engine._candidate_rank_score(weaker, provisional_signals=[stronger]),
        )
        self.assertEqual(reason, "portfolio_duplicate_bucket_throttle")

    def test_portfolio_persistently_weak_cluster_throttle_blocks_repeated_bad_family_side(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine.trades = [
            {"symbol": "AVAX/USDT", "side": "long", "strategy": "trend_pullback", "pl": -2.0},
            {"symbol": "DOT/USDT", "side": "long", "strategy": "trend_pullback", "pl": -3.0},
        ]
        candidate = {
            "symbol": "AVAX/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.78,
            "expected_edge_bps": 22.0,
            "rr_ratio": 2.2,
            "expected_holding_minutes": 180,
            "metadata": {"cross_sectional_score": 92.0},
        }
        provisional = [candidate]
        reason = engine._portfolio_persistently_weak_cluster_reason(candidate, provisional_signals=provisional)
        self.assertEqual(reason, "portfolio_persistently_weak_cluster_throttle")

    def test_portfolio_persistently_weak_cluster_throttle_ignores_unproven_or_healthy_cluster(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine.trades = [
            {"symbol": "AVAX/USDT", "side": "long", "strategy": "trend_pullback", "pl": 2.0},
            {"symbol": "DOT/USDT", "side": "long", "strategy": "trend_pullback", "pl": 1.0},
        ]
        candidate = {
            "symbol": "AVAX/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.78,
            "expected_edge_bps": 22.0,
            "rr_ratio": 2.2,
            "expected_holding_minutes": 180,
            "metadata": {"cross_sectional_score": 92.0},
        }
        reason = engine._portfolio_persistently_weak_cluster_reason(candidate, provisional_signals=[candidate])
        self.assertIsNone(reason)

    def test_portfolio_no_trade_region_blocks_marginal_post_cost_signal(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.74,
            "expected_edge_bps": 18.0,
            "rr_ratio": 2.0,
            "metadata": {"spread_bps": 18.0},
        }
        self.assertEqual(engine._portfolio_no_trade_reason(signal), "portfolio_no_trade_region")

    def test_portfolio_no_trade_region_allows_strong_post_cost_signal(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.80,
            "expected_edge_bps": 34.0,
            "rr_ratio": 2.4,
            "metadata": {"spread_bps": 4.0},
        }
        self.assertIsNone(engine._portfolio_no_trade_reason(signal))

    def test_candidate_rank_score_penalizes_negative_realized_symbol_and_strategy_expectancy(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine.trades = [
            {"symbol": "BTC/USDT", "strategy": "trend_pullback", "pl": -3.0},
            {"symbol": "BTC/USDT", "strategy": "trend_pullback", "pl": -2.0},
        ]
        engine._closed_by_symbol["BTC/USDT"] = 2
        engine._closed_by_strategy["trend_pullback"] = 2
        weak_symbol = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.80,
            "expected_edge_bps": 26.0,
            "rr_ratio": 2.4,
            "metadata": {"cross_sectional_score": 88.0},
        }
        healthier_symbol = dict(weak_symbol)
        healthier_symbol["symbol"] = "ETH/USDT"
        healthier_symbol["strategy"] = "mean_reversion"
        self.assertLess(engine._candidate_rank_score(weak_symbol), engine._candidate_rank_score(healthier_symbol))

    def test_portfolio_no_trade_region_uses_negative_realized_expectancy_penalty(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.79,
            "expected_edge_bps": 12.0,
            "rr_ratio": 2.0,
            "metadata": {"spread_bps": 4.0},
        }
        self.assertIsNone(engine._portfolio_no_trade_reason(signal))
        engine.trades = [
            {"symbol": "BTC/USDT", "strategy": "trend_pullback", "pl": -2.5},
            {"symbol": "BTC/USDT", "strategy": "trend_pullback", "pl": -1.5},
        ]
        engine._closed_by_symbol["BTC/USDT"] = 2
        engine._closed_by_strategy["trend_pullback"] = 2
        self.assertEqual(engine._portfolio_no_trade_reason(signal), "portfolio_no_trade_region")

    def test_pullback_candidate_rank_gets_extra_penalty_for_negative_realized_symbol(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine.trades = [
            {"symbol": "BTC/USDT", "strategy": "trend_pullback", "pl": -3.0},
            {"symbol": "BTC/USDT", "strategy": "trend_pullback", "pl": -2.0},
        ]
        engine._closed_by_symbol["BTC/USDT"] = 2
        engine._closed_by_strategy["trend_pullback"] = 2
        pullback_signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.80,
            "expected_edge_bps": 26.0,
            "rr_ratio": 2.4,
            "metadata": {"cross_sectional_score": 88.0},
        }
        breakout_signal = dict(pullback_signal)
        breakout_signal["strategy"] = "trend_breakout"
        self.assertLess(engine._candidate_rank_score(pullback_signal), engine._candidate_rank_score(breakout_signal))

    def test_pullback_no_trade_penalty_is_stronger_for_negative_realized_symbol(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine.trades = [
            {"symbol": "BTC/USDT", "strategy": "trend_pullback", "pl": -2.5},
            {"symbol": "BTC/USDT", "strategy": "trend_pullback", "pl": -1.5},
        ]
        engine._closed_by_symbol["BTC/USDT"] = 2
        engine._closed_by_strategy["trend_pullback"] = 2
        pullback_signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.79,
            "expected_edge_bps": 16.0,
            "rr_ratio": 2.0,
            "metadata": {"spread_bps": 4.0},
        }
        breakout_signal = dict(pullback_signal)
        breakout_signal["strategy"] = "trend_breakout"
        self.assertGreater(
            engine._realized_performance_penalty_details(pullback_signal)["no_trade_penalty_bps"],
            engine._realized_performance_penalty_details(breakout_signal)["no_trade_penalty_bps"],
        )
        self.assertEqual(engine._portfolio_no_trade_reason(pullback_signal), "portfolio_no_trade_region")
        self.assertIsNone(engine._portfolio_no_trade_reason(breakout_signal))

    def test_symbol_probation_veto_blocks_persistent_loser_before_universe_selection(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(
                starting_balance=1000.0,
                trading_mode="futures",
                backtest_warmup_candles=20,
                simulation_universe_top_n=5,
                simulation_universe_tradability_floor=-1.0,
                simulation_symbol_probation_veto_min_trades=1,
                simulation_symbol_probation_veto_expectancy_floor=-20.0,
            ),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        universe = engine.build_historical_universe(["SOL/USDT", "ETH/USDT"], "15m", 30)
        engine.exchange = HistoricalReplayExchange(universe, "SOL/USDT", base_timeframe="15m", config=engine.config)
        engine.exchange.set_time(universe.timeline(["SOL/USDT", "ETH/USDT"], timeframe="15m")[30])
        engine.trades = [{"symbol": "SOL/USDT", "strategy": "trend_breakout", "pl": -25.0}]
        engine._closed_by_symbol["SOL/USDT"] = 1
        snapshot = engine._build_universe_tradability_snapshot(["SOL/USDT", "ETH/USDT"])
        self.assertTrue(snapshot["SOL/USDT"]["probation_veto_active"])
        self.assertEqual(snapshot["SOL/USDT"]["realized_expectancy"], -25.0)

    def test_simulation_can_disable_trend_pullback_explicitly(self):
        engine = HistoricalSimulationEngine(
            BotConfig(
                starting_balance=1000.0,
                trading_mode="futures",
                simulation_disable_trend_pullback=True,
            ),
            signal_engine_cls=QuietSignalEngine,
        )
        reason = engine._simulation_strategy_probation_reason(
            {
                "symbol": "BTC/USDT",
                "strategy": "trend_pullback",
                "side": "long",
                "metadata": {},
            }
        )
        self.assertEqual(reason, "trend_pullback_disabled_in_simulation")

    def test_pullback_strategy_probation_veto_blocks_when_family_expectancy_collapses(self):
        engine = HistoricalSimulationEngine(
            BotConfig(
                starting_balance=1000.0,
                trading_mode="futures",
                simulation_pullback_strategy_veto_min_trades=2,
                simulation_pullback_strategy_veto_expectancy_floor=-20.0,
            ),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine.trades = [
            {"symbol": "BTC/USDT", "strategy": "trend_pullback", "pl": -22.0},
            {"symbol": "ETH/USDT", "strategy": "trend_pullback", "pl": -24.0},
        ]
        engine._closed_by_strategy["trend_pullback"] = 2
        reason = engine._simulation_strategy_probation_reason(
            {
                "symbol": "BTC/USDT",
                "strategy": "trend_pullback",
                "side": "long",
                "metadata": {},
            }
        )
        self.assertEqual(reason, "pullback_strategy_probation_veto")

    def test_portfolio_no_trade_region_can_relax_for_positive_missed_opportunity_learning(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", backtest_warmup_candles=20),
            signal_engine_cls=QuietSignalEngine,
        )
        engine._reset_run_state()
        engine.learning.state_store.upsert_learning_model(
            "learning_missed::cell::portfolio_no_trade_region::trend_pullback::long::trending::majors::limit",
            {
                "total": 5.0,
                "positive_forward": 4.0,
                "total_forward_r": 1.5,
            },
        )
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.74,
            "expected_edge_bps": 18.0,
            "rr_ratio": 2.0,
            "regime": "trending",
            "metadata": {"spread_bps": 18.0, "preferred_order_type": "limit"},
        }
        self.assertIsNone(engine._portfolio_no_trade_reason(signal))
        self.assertEqual(engine._missed_opportunity_relaxations["portfolio_no_trade_region"], 1)

    def test_trade_learning_engine_exposes_positive_missed_opportunity_adjustment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(BotConfig(), store)
            store.upsert_learning_model(
                "learning_missed::cell::portfolio_no_trade_region::trend_pullback::long::trending::majors::limit",
                {
                    "total": 5.0,
                    "positive_forward": 4.0,
                    "total_forward_r": 1.5,
                },
            )
            signal = {
                "symbol": "BTC/USDT",
                "strategy": "trend_pullback",
                "side": "long",
                "signal_quality": 0.78,
                "expected_edge_bps": 18.0,
                "rr_ratio": 2.0,
                "regime": "trending",
                "metadata": {"preferred_order_type": "limit"},
            }
            adjustment = learning.missed_opportunity_gate_adjustment(signal, "portfolio_no_trade_region")
            self.assertTrue(adjustment["active"])
            self.assertGreater(adjustment["relax_bps"], 0.0)

    def test_simulation_campaign_reports_per_symbol_rollups(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        class MultiSymbolSignals:
            def __init__(self, config, exch):
                self.exch = exch
                self.learning_context_provider = None

            def generate_signal(self, symbol):
                candles = self.exch.fetch_ohlcv(symbol, "15m", limit=200)
                if symbol == "BTC/USDT" and len(candles) == 101:
                    return {
                        "side": "long",
                        "entry_price": 100.0,
                        "stop_loss": 99.0,
                        "take_profit": 103.0,
                        "strategy": "unit_test_ranked_entry",
                        "expected_holding_minutes": 60,
                        "signal_quality": 0.75,
                        "expected_edge_bps": 15.0,
                        "rr_ratio": 3.0,
                        "metadata": {"ensemble_score": 90.0},
                    }
                if symbol == "ETH/USDT" and len(candles) == 101:
                    return {
                        "side": "long",
                        "entry_price": 100.0,
                        "stop_loss": 99.0,
                        "take_profit": 103.0,
                        "strategy": "unit_test_ranked_entry",
                        "expected_holding_minutes": 60,
                        "signal_quality": 0.74,
                        "expected_edge_bps": 14.0,
                        "rr_ratio": 3.0,
                        "metadata": {"ensemble_score": 80.0},
                    }
                return None

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", max_open_positions=1, backtest_warmup_candles=20),
            signal_engine_cls=MultiSymbolSignals,
        )
        result = engine.run_campaign(["BTC/USDT", "ETH/USDT"], timeframe="15m", days=10)

        btc = result["symbol_rollups"]["BTC/USDT"]
        eth = result["symbol_rollups"]["ETH/USDT"]
        self.assertEqual(btc["raw_signals"], 1)
        self.assertEqual(btc["submitted_orders"], 1)
        self.assertEqual(eth["raw_signals"], 1)
        self.assertGreaterEqual(eth["skipped_signals"], 1)
        self.assertIn("max_open_positions", eth["skip_reasons"])
        self.assertEqual(result["decision_diagnostics"]["raw_signals_by_symbol"]["BTC/USDT"], 1)
        self.assertEqual(result["decision_diagnostics"]["raw_signals_by_symbol"]["ETH/USDT"], 1)

    def test_simulation_campaign_separates_generation_outcomes_by_symbol(self):
        timestamps = pd.date_range("2025-01-01", periods=21, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        class OutcomeSignals:
            def __init__(self, config, exch):
                self.exch = exch
                self.learning_context_provider = None
                self.last_generation_diagnostics = {}
                self._fired = set()

            def generate_signal(self, symbol):
                candles = self.exch.fetch_ohlcv(symbol, "15m", limit=200)
                if len(candles) != 21 or symbol in self._fired:
                    return None
                self._fired.add(symbol)
                if symbol == "ETH/USDT":
                    self.last_generation_diagnostics[symbol] = {
                        "outcome": "no_proposal",
                        "reason": "strategy_produced_no_proposal",
                    }
                    return None
                if symbol == "SOL/USDT":
                    self.last_generation_diagnostics[symbol] = {
                        "outcome": "ensemble_rejected",
                        "reason": "no_ensemble_selection",
                    }
                    return None
                if symbol == "XRP/USDT":
                    self.last_generation_diagnostics[symbol] = {
                        "outcome": "reliability_rejected",
                        "reason": "signal_quality_below_threshold",
                    }
                    return None
                self.last_generation_diagnostics[symbol] = {
                    "outcome": "selected",
                    "reason": "selected_for_submission",
                }
                return {
                    "side": "long",
                    "entry_price": 100.0,
                    "stop_loss": 99.0,
                    "take_profit": 103.0,
                    "strategy": "unit_test_ranked_entry",
                    "expected_holding_minutes": 60,
                    "signal_quality": 0.75,
                    "expected_edge_bps": 15.0,
                    "rr_ratio": 3.0,
                    "metadata": {
                        "ensemble_score": 95.0 if symbol == "BTC/USDT" else 85.0,
                    },
                }

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", max_open_positions=1, backtest_warmup_candles=20),
            signal_engine_cls=OutcomeSignals,
        )
        result = engine.run_campaign(["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"], timeframe="15m", days=10)

        diagnostics = result["decision_diagnostics"]
        self.assertEqual(diagnostics["generation_outcomes"]["selected"], 2)
        self.assertEqual(diagnostics["generation_outcomes"]["no_proposal"], 1)
        self.assertEqual(diagnostics["generation_outcomes"]["ensemble_rejected"], 1)
        self.assertEqual(diagnostics["generation_outcomes"]["reliability_rejected"], 1)

        self.assertEqual(diagnostics["generation_outcomes_by_symbol"]["ETH/USDT"]["no_proposal"], 1)
        self.assertEqual(diagnostics["generation_outcomes_by_symbol"]["SOL/USDT"]["ensemble_rejected"], 1)
        self.assertEqual(diagnostics["generation_outcomes_by_symbol"]["XRP/USDT"]["reliability_rejected"], 1)
        self.assertEqual(diagnostics["generation_outcomes_by_symbol"]["BTC/USDT"]["selected"], 1)
        self.assertEqual(diagnostics["generation_outcomes_by_symbol"]["ADA/USDT"]["selected"], 1)

        self.assertEqual(
            diagnostics["generation_reasons_by_symbol"]["ETH/USDT"]["strategy_produced_no_proposal"],
            1,
        )
        self.assertEqual(
            diagnostics["generation_reasons_by_symbol"]["SOL/USDT"]["no_ensemble_selection"],
            1,
        )
        self.assertEqual(
            diagnostics["generation_reasons_by_symbol"]["XRP/USDT"]["signal_quality_below_threshold"],
            1,
        )
        self.assertEqual(diagnostics["skip_reasons_by_symbol"]["ADA/USDT"]["max_open_positions"], 1)

    def test_simulation_campaign_includes_nested_diagnostic_schema(self):
        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        class MultiSymbolSignals:
            def __init__(self, config, exch):
                self.exch = exch
                self.learning_context_provider = None
                self.last_generation_diagnostics = {}

            def generate_signal(self, symbol):
                candles = self.exch.fetch_ohlcv(symbol, "15m", limit=200)
                if symbol == "BTC/USDT" and len(candles) == 101:
                    return {
                        "side": "long",
                        "entry_price": 100.0,
                        "stop_loss": 99.0,
                        "take_profit": 103.0,
                        "strategy": "unit_test_ranked_entry",
                        "expected_holding_minutes": 60,
                        "signal_quality": 0.75,
                        "expected_edge_bps": 15.0,
                        "rr_ratio": 3.0,
                        "metadata": {"ensemble_score": 90.0},
                    }
                if symbol == "ETH/USDT" and len(candles) == 101:
                    return {
                        "side": "long",
                        "entry_price": 100.0,
                        "stop_loss": 99.0,
                        "take_profit": 103.0,
                        "strategy": "unit_test_ranked_entry",
                        "expected_holding_minutes": 60,
                        "signal_quality": 0.74,
                        "expected_edge_bps": 14.0,
                        "rr_ratio": 3.0,
                        "metadata": {"ensemble_score": 80.0},
                    }
                return None

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", max_open_positions=1, backtest_warmup_candles=20),
            signal_engine_cls=MultiSymbolSignals,
        )
        result = engine.run_campaign(["BTC/USDT", "ETH/USDT"], timeframe="15m", days=10)

        campaign_summary = result["campaign_summary"]
        campaign_diagnostics = result["campaign_diagnostics"]["primary_summary"]

        self.assertEqual(campaign_summary["trade_flow"]["proposals"], 2)
        self.assertEqual(campaign_summary["trade_flow"]["raw_signals"], 2)
        self.assertEqual(campaign_summary["trade_flow"]["submitted_orders"], 1)
        self.assertEqual(campaign_summary["session"]["timeframe"], "15m")
        self.assertEqual(campaign_summary["trade_frequency"]["global"]["proposals"], 2)
        self.assertAlmostEqual(campaign_summary["trade_frequency"]["global"]["proposals_per_day"], 0.2)
        self.assertEqual(campaign_summary["trade_frequency"]["global"]["selected_signals"], 2)
        self.assertEqual(campaign_summary["trade_frequency"]["global"]["status"], "far_below_target")
        self.assertEqual(campaign_summary["trade_frequency"]["target_band"]["preferred_min"], 2.0)
        self.assertEqual(campaign_summary["trade_frequency"]["target_band"]["preferred_max"], 3.0)
        self.assertEqual(campaign_summary["trade_frequency"]["target_band"]["soft_floor"], 1.5)
        self.assertEqual(campaign_summary["trade_frequency"]["target_band"]["soft_ceiling"], 3.5)
        self.assertEqual(campaign_summary["trade_frequency"]["by_strategy"]["unit_test_ranked_entry"]["proposals"], 2)
        self.assertEqual(campaign_summary["trade_frequency"]["by_strategy"]["unit_test_ranked_entry"]["selected_signals"], 2)
        self.assertEqual(campaign_summary["trade_frequency"]["by_strategy"]["unit_test_ranked_entry"]["status"], "far_below_target")
        self.assertEqual(campaign_summary["trade_frequency"]["global"]["controller_actions"], {})

        self.assertEqual(campaign_diagnostics["by_symbol"]["BTC/USDT"]["summary"]["submitted_orders"], 1)
        self.assertEqual(campaign_diagnostics["by_symbol"]["ETH/USDT"]["summary"]["skipped_signals"], 1)
        self.assertEqual(campaign_diagnostics["by_strategy"]["proposals"]["unit_test_ranked_entry"], 2)
        self.assertIn("family_rotation_actions", campaign_diagnostics)
        self.assertIn("family_rotation", campaign_diagnostics["by_strategy"])
        self.assertIn("realized_performance", campaign_summary)
        self.assertIn("by_symbol", campaign_summary["realized_performance"])
        self.assertIn("by_strategy", campaign_summary["realized_performance"])
        self.assertIn("exit_quality", campaign_summary)
        self.assertIn("by_exit_reason", campaign_summary["exit_quality"])
        self.assertIn("giveback_by_strategy", campaign_summary["exit_quality"])
        self.assertEqual(
            campaign_diagnostics["by_strategy_by_symbol"]["BTC/USDT"]["unit_test_ranked_entry"]["raw_signals"],
            1,
        )
        self.assertEqual(
            campaign_diagnostics["by_strategy_by_symbol"]["ETH/USDT"]["unit_test_ranked_entry"]["skip_reasons"]["max_open_positions"],
            1,
        )
        self.assertTrue(any(item["key"] == "post_selection:unit_test_ranked_entry:max_open_positions" for item in campaign_diagnostics["top_rejection_reasons"]))

    def test_simulation_campaign_keeps_distinct_symbol_diagnostics_when_only_one_symbol_trades(self):
        timestamps = pd.date_range("2025-01-01", periods=21, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        class LocalSimulationEngine(HistoricalSimulationEngine):
            def load_historical_data(self, symbol, timeframe, days=180):
                return df.copy()

        class DistinctOutcomeSignals:
            def __init__(self, config, exch):
                self.exch = exch
                self.learning_context_provider = None
                self.last_generation_diagnostics = {}
                self._fired = set()

            def generate_signal(self, symbol):
                candles = self.exch.fetch_ohlcv(symbol, "15m", limit=200)
                if len(candles) != 21 or symbol in self._fired:
                    return None
                self._fired.add(symbol)
                if symbol == "ETH/USDT":
                    self.last_generation_diagnostics[symbol] = {
                        "outcome": "no_proposal",
                        "reason": "strategy_produced_no_proposal",
                    }
                    return None
                if symbol == "SOL/USDT":
                    self.last_generation_diagnostics[symbol] = {
                        "outcome": "ensemble_rejected",
                        "reason": "no_ensemble_selection",
                    }
                    return None
                if symbol == "XRP/USDT":
                    self.last_generation_diagnostics[symbol] = {
                        "outcome": "reliability_rejected",
                        "reason": "signal_quality_below_threshold",
                    }
                    return None
                self.last_generation_diagnostics[symbol] = {
                    "outcome": "selected",
                    "reason": "selected_for_submission",
                }
                return {
                    "side": "long",
                    "entry_price": 100.0,
                    "stop_loss": 99.0,
                    "take_profit": 103.0,
                    "strategy": "unit_test_ranked_entry",
                    "expected_holding_minutes": 60,
                    "signal_quality": 0.8,
                    "expected_edge_bps": 18.0,
                    "rr_ratio": 3.0,
                    "metadata": {"ensemble_score": 95.0 if symbol == "BTC/USDT" else 85.0},
                }

        engine = LocalSimulationEngine(
            BotConfig(starting_balance=1000.0, trading_mode="futures", max_open_positions=1, backtest_warmup_candles=20),
            signal_engine_cls=DistinctOutcomeSignals,
        )
        result = engine.run_campaign(["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"], timeframe="15m", days=10)

        rollups = result["symbol_rollups"]
        diagnostics = result["campaign_diagnostics"]["primary_summary"]["by_symbol"]
        strategy_diagnostics = result["campaign_diagnostics"]["primary_summary"]["by_strategy_by_symbol"]

        self.assertEqual(result["submitted_orders"], 1)
        self.assertEqual(rollups["BTC/USDT"]["submitted_orders"], 1)
        self.assertEqual(rollups["ETH/USDT"]["trades"], 0)
        self.assertEqual(rollups["SOL/USDT"]["trades"], 0)
        self.assertEqual(rollups["XRP/USDT"]["trades"], 0)
        self.assertEqual(rollups["ADA/USDT"]["trades"], 0)

        self.assertEqual(diagnostics["ETH/USDT"]["generation_outcomes"]["no_proposal"], 1)
        self.assertEqual(diagnostics["SOL/USDT"]["generation_outcomes"]["ensemble_rejected"], 1)
        self.assertEqual(diagnostics["XRP/USDT"]["generation_outcomes"]["reliability_rejected"], 1)
        self.assertEqual(diagnostics["ADA/USDT"]["summary"]["skipped_signals"], 1)
        self.assertEqual(diagnostics["ADA/USDT"]["top_rejection_reasons"][0]["key"], "post_selection:max_open_positions")

        self.assertEqual(strategy_diagnostics["BTC/USDT"]["unit_test_ranked_entry"]["submitted_orders"], 1)
        self.assertEqual(strategy_diagnostics["ADA/USDT"]["unit_test_ranked_entry"]["skip_reasons"]["max_open_positions"], 1)
        self.assertEqual(strategy_diagnostics["ETH/USDT"], {})

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

    def _build_simulated_venue_for_fill_test(self, bar):
        timestamp = dt.datetime(2025, 1, 1, 0, 0, 0)
        frame = pd.DataFrame([{"timestamp": timestamp, **bar}]).set_index("timestamp")
        config = BotConfig(
            backtest_fee_bps=10.0,
            backtest_slippage_bps=5.0,
            simulated_partial_fill_min_fraction=1.0,
            simulation_latency_jitter_bars=0,
        )
        exchange = HistoricalReplayExchange({("BTC/USDT", "15m"): frame}, "BTC/USDT", base_timeframe="15m", config=config)
        now = timestamp + dt.timedelta(minutes=15)
        exchange.set_time(now)
        venue = SimulatedExecutionVenue(config, exchange, rng=random.Random(1))
        market_bar = exchange.current_bar("BTC/USDT", "15m")
        self.assertIsNotNone(market_bar)
        return venue, now, market_bar

    def test_simulated_passive_short_limit_fill_does_not_cross_below_limit_price(self):
        venue, now, bar = self._build_simulated_venue_for_fill_test(
            {"open": 99.5, "high": 100.5, "low": 99.0, "close": 99.6, "volume": 1000.0}
        )
        order = SimulatedOrder(
            order_id="short-passive",
            symbol="BTC/USDT",
            side="short",
            strategy="trend_pullback",
            requested_size=1.0,
            remaining_size=1.0,
            requested_price=100.0,
            stop_loss=101.0,
            take_profit=98.5,
            order_type="limit",
            submitted_at=now,
            activate_on=now,
            activate_index=0,
            expires_on_index=4,
            latency_bars=0,
            trace_id="trace-short",
            signal={},
            queue_ahead_fraction=0.0,
        )

        fill, details = venue._maybe_fill(order, bar, now)

        self.assertIsNotNone(fill)
        self.assertTrue(details["passive_fill"])
        self.assertGreaterEqual(fill.price, order.requested_price)

    def test_simulated_passive_long_limit_fill_does_not_cross_above_limit_price(self):
        venue, now, bar = self._build_simulated_venue_for_fill_test(
            {"open": 100.5, "high": 101.0, "low": 99.5, "close": 100.4, "volume": 1000.0}
        )
        order = SimulatedOrder(
            order_id="long-passive",
            symbol="BTC/USDT",
            side="long",
            strategy="trend_pullback",
            requested_size=1.0,
            remaining_size=1.0,
            requested_price=100.0,
            stop_loss=99.0,
            take_profit=101.5,
            order_type="limit",
            submitted_at=now,
            activate_on=now,
            activate_index=0,
            expires_on_index=4,
            latency_bars=0,
            trace_id="trace-long",
            signal={},
            queue_ahead_fraction=0.0,
        )

        fill, details = venue._maybe_fill(order, bar, now)

        self.assertIsNotNone(fill)
        self.assertTrue(details["passive_fill"])
        self.assertLessEqual(fill.price, order.requested_price)

    def test_simulated_strong_stale_limit_order_gets_repriced_before_cancel(self):
        venue, now, bar = self._build_simulated_venue_for_fill_test(
            {"open": 102.0, "high": 103.0, "low": 101.8, "close": 102.5, "volume": 1000.0}
        )
        venue.config.simulation_stale_order_cancel_bars = 3
        order = SimulatedOrder(
            order_id="repriced-long",
            symbol="BTC/USDT",
            side="long",
            strategy="trend_pullback",
            requested_size=1.0,
            remaining_size=1.0,
            requested_price=100.0,
            stop_loss=98.0,
            take_profit=106.0,
            order_type="limit",
            submitted_at=now,
            activate_on=now,
            activate_index=0,
            expires_on_index=4,
            latency_bars=0,
            trace_id="trace-reprice",
            signal={"signal_quality": 0.74, "expected_edge_bps": 20.0, "metadata": {"stale_cancel_distance_bps": 18.0}},
            queue_ahead_fraction=0.6,
            resting_bars=3,
        )

        repriced = venue._maybe_reprice_stale_order(order, bar, current_index=3)

        self.assertTrue(repriced)
        self.assertGreater(order.requested_price, 100.0)
        self.assertLessEqual(order.queue_ahead_fraction, 0.18)
        self.assertGreaterEqual(order.expires_on_index, 5)
        self.assertEqual(order.metadata["stale_reprices"], 1)
        self.assertIsNone(venue._maybe_cancel_stale_order(order, bar))

    def test_simulated_top_quality_stale_limit_order_gets_repriced_early(self):
        venue, now, bar = self._build_simulated_venue_for_fill_test(
            {"open": 102.0, "high": 103.0, "low": 101.8, "close": 102.5, "volume": 1000.0}
        )
        venue.config.simulation_stale_order_cancel_bars = 3
        order = SimulatedOrder(
            order_id="repriced-early",
            symbol="BTC/USDT",
            side="long",
            strategy="trend_pullback",
            requested_size=1.0,
            remaining_size=1.0,
            requested_price=100.0,
            stop_loss=98.0,
            take_profit=106.0,
            order_type="limit",
            submitted_at=now,
            activate_on=now,
            activate_index=0,
            expires_on_index=4,
            latency_bars=0,
            trace_id="trace-reprice-early",
            signal={"signal_quality": 0.80, "expected_edge_bps": 24.0, "metadata": {"stale_cancel_distance_bps": 18.0}},
            queue_ahead_fraction=0.6,
            resting_bars=2,
        )

        repriced = venue._maybe_reprice_stale_order(order, bar, current_index=2)

        self.assertTrue(repriced)
        self.assertGreater(order.requested_price, 100.0)
        self.assertEqual(order.metadata["stale_reprices"], 1)

    def test_simulated_top_quality_pullback_limit_gets_second_reprice_attempt(self):
        venue, now, bar = self._build_simulated_venue_for_fill_test(
            {"open": 104.0, "high": 105.0, "low": 103.8, "close": 104.5, "volume": 1000.0}
        )
        venue.config.simulation_stale_order_cancel_bars = 3
        order = SimulatedOrder(
            order_id="repriced-second-pullback",
            symbol="BTC/USDT",
            side="long",
            strategy="trend_pullback",
            requested_size=1.0,
            remaining_size=1.0,
            requested_price=100.0,
            stop_loss=98.0,
            take_profit=108.0,
            order_type="limit",
            submitted_at=now,
            activate_on=now,
            activate_index=0,
            expires_on_index=4,
            latency_bars=0,
            trace_id="trace-reprice-second-pullback",
            signal={"signal_quality": 0.84, "expected_edge_bps": 28.0, "metadata": {"stale_cancel_distance_bps": 18.0}},
            queue_ahead_fraction=0.6,
            resting_bars=3,
            metadata={"stale_reprices": 1},
        )

        repriced = venue._maybe_reprice_stale_order(order, bar, current_index=3)

        self.assertTrue(repriced)
        self.assertEqual(order.metadata["stale_reprices"], 2)

    def test_simulated_top_quality_breakout_limit_gets_second_reprice_attempt(self):
        venue, now, bar = self._build_simulated_venue_for_fill_test(
            {"open": 104.0, "high": 105.0, "low": 103.8, "close": 104.5, "volume": 1000.0}
        )
        venue.config.simulation_stale_order_cancel_bars = 3
        order = SimulatedOrder(
            order_id="repriced-second-breakout",
            symbol="BTC/USDT",
            side="long",
            strategy="trend_breakout",
            requested_size=1.0,
            remaining_size=1.0,
            requested_price=100.0,
            stop_loss=98.5,
            take_profit=108.0,
            order_type="limit",
            submitted_at=now,
            activate_on=now,
            activate_index=0,
            expires_on_index=4,
            latency_bars=0,
            trace_id="trace-reprice-second-breakout",
            signal={"signal_quality": 0.87, "expected_edge_bps": 32.0, "metadata": {"stale_cancel_distance_bps": 18.0}},
            queue_ahead_fraction=0.6,
            resting_bars=3,
            metadata={"stale_reprices": 1},
        )

        repriced = venue._maybe_reprice_stale_order(order, bar, current_index=3)

        self.assertTrue(repriced)
        self.assertEqual(order.metadata["stale_reprices"], 2)

    def test_simulated_weak_stale_limit_order_still_cancels(self):
        venue, now, bar = self._build_simulated_venue_for_fill_test(
            {"open": 102.0, "high": 103.0, "low": 101.8, "close": 102.5, "volume": 1000.0}
        )
        venue.config.simulation_stale_order_cancel_bars = 3
        order = SimulatedOrder(
            order_id="cancel-long",
            symbol="BTC/USDT",
            side="long",
            strategy="trend_pullback",
            requested_size=1.0,
            remaining_size=1.0,
            requested_price=100.0,
            stop_loss=98.0,
            take_profit=106.0,
            order_type="limit",
            submitted_at=now,
            activate_on=now,
            activate_index=0,
            expires_on_index=4,
            latency_bars=0,
            trace_id="trace-cancel",
            signal={"signal_quality": 0.58, "expected_edge_bps": 10.0, "metadata": {"stale_cancel_distance_bps": 18.0}},
            queue_ahead_fraction=0.6,
            resting_bars=3,
        )

        repriced = venue._maybe_reprice_stale_order(order, bar, current_index=3)

        self.assertFalse(repriced)
        self.assertFalse(venue._maybe_stale_escalate_order(order, bar))
        self.assertEqual(venue._maybe_cancel_stale_order(order, bar), "stale_limit")

    def test_simulated_strong_repriced_pullback_limit_escalates_to_market_before_stale_cancel(self):
        venue, now, bar = self._build_simulated_venue_for_fill_test(
            {"open": 102.0, "high": 103.0, "low": 101.8, "close": 102.5, "volume": 1000.0}
        )
        venue.config.simulation_stale_order_cancel_bars = 3
        order = SimulatedOrder(
            order_id="stale-escalate-pullback",
            symbol="BTC/USDT",
            side="long",
            strategy="trend_pullback",
            requested_size=1.0,
            remaining_size=1.0,
            requested_price=100.0,
            stop_loss=98.0,
            take_profit=106.0,
            order_type="limit",
            submitted_at=now,
            activate_on=now,
            activate_index=0,
            expires_on_index=4,
            latency_bars=0,
            trace_id="trace-stale-escalate-pullback",
            signal={"signal_quality": 0.82, "expected_edge_bps": 26.0, "metadata": {"stale_cancel_distance_bps": 18.0}},
            queue_ahead_fraction=0.6,
            resting_bars=3,
            metadata={"stale_reprices": 1},
        )

        self.assertTrue(venue._maybe_stale_escalate_order(order, bar))
        self.assertEqual(order.order_type, "market")
        self.assertTrue(order.metadata["stale_escalated_to_market"])
        self.assertIsNone(venue._maybe_cancel_stale_order(order, bar))

    def test_simulated_strong_repriced_breakout_limit_escalates_to_market_before_stale_cancel(self):
        venue, now, bar = self._build_simulated_venue_for_fill_test(
            {"open": 102.0, "high": 103.0, "low": 101.8, "close": 102.5, "volume": 1000.0}
        )
        venue.config.simulation_stale_order_cancel_bars = 3
        order = SimulatedOrder(
            order_id="stale-escalate-breakout",
            symbol="BTC/USDT",
            side="long",
            strategy="trend_breakout",
            requested_size=1.0,
            remaining_size=1.0,
            requested_price=100.0,
            stop_loss=98.8,
            take_profit=106.0,
            order_type="limit",
            submitted_at=now,
            activate_on=now,
            activate_index=0,
            expires_on_index=4,
            latency_bars=0,
            trace_id="trace-stale-escalate-breakout",
            signal={"signal_quality": 0.86, "expected_edge_bps": 32.0, "metadata": {"stale_cancel_distance_bps": 18.0}},
            queue_ahead_fraction=0.6,
            resting_bars=3,
            metadata={"stale_reprices": 1},
        )

        self.assertTrue(venue._maybe_stale_escalate_order(order, bar))
        self.assertEqual(order.order_type, "market")
        self.assertTrue(order.metadata["stale_escalated_to_market"])
        self.assertIsNone(venue._maybe_cancel_stale_order(order, bar))

    def test_simulated_strong_pullback_limit_upgrades_to_market_entry(self):
        venue, _, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "entry_price": 100.08,
            "stop_loss": 98.5,
            "take_profit": 103.2,
            "signal_quality": 0.76,
            "expected_edge_bps": 21.0,
            "rr_ratio": 1.9,
            "metadata": {
                "preferred_order_type": "limit",
                "mid_price": 100.0,
                "liquidity_score": 0.72,
            },
        }
        self.assertEqual(venue._order_type_for_signal(signal), "market")

    def test_simulated_far_pullback_limit_stays_passive(self):
        venue, _, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "entry_price": 100.40,
            "stop_loss": 98.5,
            "take_profit": 103.2,
            "signal_quality": 0.76,
            "expected_edge_bps": 21.0,
            "rr_ratio": 1.9,
            "metadata": {
                "preferred_order_type": "limit",
                "mid_price": 100.0,
                "liquidity_score": 0.72,
            },
        }
        self.assertEqual(venue._order_type_for_signal(signal), "limit")

    def test_simulated_pullback_limit_requested_price_is_offset_more_passively(self):
        venue, _, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit": 103.0,
            "metadata": {"preferred_order_type": "limit"},
        }
        requested_price = venue._requested_price_for_signal(signal, order_type="limit")
        self.assertLess(requested_price, 100.0)
        self.assertAlmostEqual(requested_price, 99.96, places=6)

    def test_simulated_high_quality_pullback_limit_requested_price_is_less_passive(self):
        venue, _, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit": 103.0,
            "signal_quality": 0.78,
            "expected_edge_bps": 24.0,
            "metadata": {"preferred_order_type": "limit"},
        }
        requested_price = venue._requested_price_for_signal(signal, order_type="limit")
        self.assertAlmostEqual(requested_price, 99.98, places=6)

    def test_simulated_mean_reversion_limit_offset_respects_stop_distance_cap(self):
        venue, _, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "mean_reversion",
            "entry_price": 100.0,
            "stop_loss": 99.8,
            "take_profit": 100.6,
            "metadata": {"preferred_order_type": "limit"},
        }
        requested_price = venue._requested_price_for_signal(signal, order_type="limit")
        self.assertAlmostEqual(requested_price, 99.956, places=6)

    def test_simulated_strong_pullback_limit_gets_longer_expiry_window(self):
        venue, _, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "signal_quality": 0.76,
            "metadata": {"preferred_order_type": "limit", "order_expiry_bars": 4},
        }
        self.assertEqual(venue._limit_expiry_bars_for_signal(signal), 6)

    def test_simulated_strong_breakout_limit_gets_longer_expiry_window(self):
        venue, _, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_breakout",
            "signal_quality": 0.84,
            "metadata": {"preferred_order_type": "limit", "order_expiry_bars": 4},
        }
        self.assertEqual(venue._limit_expiry_bars_for_signal(signal), 5)

    def test_simulated_strong_breakout_limit_upgrades_to_market_entry(self):
        venue, _, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_breakout",
            "entry_price": 100.05,
            "stop_loss": 98.8,
            "take_profit": 103.0,
            "signal_quality": 0.82,
            "expected_edge_bps": 28.0,
            "rr_ratio": 1.9,
            "metadata": {
                "preferred_order_type": "limit",
                "mid_price": 100.0,
                "liquidity_score": 0.75,
            },
        }
        self.assertEqual(venue._order_type_for_signal(signal), "market")

    def test_simulated_high_quality_pullback_limit_gets_better_queue_priority(self):
        venue, now, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        venue.rng.seed(7)
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit": 103.0,
            "signal_quality": 0.82,
            "expected_edge_bps": 26.0,
            "metadata": {"preferred_order_type": "limit"},
        }
        order = venue.submit_order(signal=signal, size=1.0, now=now, current_index=0, trace_id="queue-pullback")
        self.assertLessEqual(order.queue_ahead_fraction, 0.28)

    def test_simulated_high_quality_breakout_limit_gets_better_queue_priority(self):
        venue, now, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        venue.rng.seed(7)
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_breakout",
            "entry_price": 100.4,
            "stop_loss": 98.8,
            "take_profit": 103.0,
            "signal_quality": 0.86,
            "expected_edge_bps": 32.0,
            "rr_ratio": 1.6,
            "metadata": {"preferred_order_type": "limit", "mid_price": 100.0, "liquidity_score": 0.60},
        }
        order = venue.submit_order(signal=signal, size=1.0, now=now, current_index=0, trace_id="queue-breakout")
        self.assertLessEqual(order.queue_ahead_fraction, 0.28)

    def test_simulated_high_quality_pullback_limit_gets_lower_latency(self):
        venue, now, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        venue.config.backtest_latency_bars = 2
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit": 103.0,
            "signal_quality": 0.83,
            "expected_edge_bps": 28.0,
            "metadata": {"preferred_order_type": "limit"},
        }
        order = venue.submit_order(signal=signal, size=1.0, now=now, current_index=0, trace_id="latency-pullback")
        self.assertEqual(order.latency_bars, 1)
        self.assertEqual(order.activate_index, 1)

    def test_simulated_weak_pullback_limit_keeps_base_latency(self):
        venue, now, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        venue.config.backtest_latency_bars = 2
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "trend_pullback",
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit": 103.0,
            "signal_quality": 0.72,
            "expected_edge_bps": 16.0,
            "metadata": {"preferred_order_type": "limit"},
        }
        order = venue.submit_order(signal=signal, size=1.0, now=now, current_index=0, trace_id="latency-weak-pullback")
        self.assertEqual(order.latency_bars, 2)
        self.assertEqual(order.activate_index, 2)

    def test_simulated_weak_limit_keeps_base_expiry_window(self):
        venue, _, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "strategy": "mean_reversion",
            "signal_quality": 0.62,
            "metadata": {"preferred_order_type": "limit", "order_expiry_bars": 3},
        }
        self.assertEqual(venue._limit_expiry_bars_for_signal(signal), 3)

    def test_simulated_strong_touched_limit_escalates_to_market(self):
        venue, now, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        order = SimulatedOrder(
            order_id="touch-escalate",
            symbol="BTC/USDT",
            side="long",
            strategy="trend_pullback",
            requested_size=1.0,
            remaining_size=1.0,
            requested_price=100.0,
            stop_loss=98.0,
            take_profit=103.0,
            order_type="limit",
            submitted_at=now,
            activate_on=now,
            activate_index=0,
            expires_on_index=4,
            latency_bars=0,
            trace_id="trace-touch-escalate",
            signal={"signal_quality": 0.78, "expected_edge_bps": 22.0},
            queue_ahead_fraction=0.4,
            limit_touch_count=2,
        )
        escalated = venue._maybe_touch_escalate_order(order)
        self.assertTrue(escalated)
        self.assertEqual(order.order_type, "market")

    def test_simulated_strong_breakout_touched_limit_escalates_to_market(self):
        venue, now, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        order = SimulatedOrder(
            order_id="touch-escalate-breakout",
            symbol="BTC/USDT",
            side="long",
            strategy="trend_breakout",
            requested_size=1.0,
            remaining_size=1.0,
            requested_price=100.0,
            stop_loss=98.8,
            take_profit=103.0,
            order_type="limit",
            submitted_at=now,
            activate_on=now,
            activate_index=0,
            expires_on_index=4,
            latency_bars=0,
            trace_id="trace-touch-escalate-breakout",
            signal={"signal_quality": 0.84, "expected_edge_bps": 30.0},
            queue_ahead_fraction=0.4,
            limit_touch_count=1,
        )

        escalated = venue._maybe_touch_escalate_order(order)

        self.assertTrue(escalated)
        self.assertEqual(order.order_type, "market")
        self.assertEqual(order.queue_ahead_fraction, 0.0)
        self.assertTrue(order.metadata["touch_escalated_to_market"])

    def test_simulated_weak_touched_limit_does_not_escalate(self):
        venue, now, _ = self._build_simulated_venue_for_fill_test(
            {"open": 100.0, "high": 100.8, "low": 99.6, "close": 100.2, "volume": 1000.0}
        )
        order = SimulatedOrder(
            order_id="touch-no-escalate",
            symbol="BTC/USDT",
            side="long",
            strategy="trend_pullback",
            requested_size=1.0,
            remaining_size=1.0,
            requested_price=100.0,
            stop_loss=98.0,
            take_profit=103.0,
            order_type="limit",
            submitted_at=now,
            activate_on=now,
            activate_index=0,
            expires_on_index=4,
            latency_bars=0,
            trace_id="trace-touch-no-escalate",
            signal={"signal_quality": 0.68, "expected_edge_bps": 14.0},
            queue_ahead_fraction=0.4,
            limit_touch_count=2,
        )
        escalated = venue._maybe_touch_escalate_order(order)
        self.assertFalse(escalated)
        self.assertEqual(order.order_type, "limit")

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

    def test_risk_manager_sizes_up_preferred_rotation_family(self):
        config = BotConfig(
            trading_mode="futures",
            default_leverage=1,
            rotation_risk_preferred_multiplier=1.08,
            rotation_risk_suppressed_multiplier=0.90,
        )
        state = BotState(balance=1000.0)
        risk = RiskManager(config, state)
        baseline = risk.calc_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            signal={
                "strategy": "trend_pullback",
                "metadata": {},
            },
        )
        preferred = risk.calc_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            signal={
                "strategy": "trend_pullback",
                "metadata": {
                    "preferred_family": "trend_pullback",
                    "suppressed_family": "trend_breakout",
                    "rotation_confidence": 0.75,
                },
            },
        )
        self.assertGreater(preferred, baseline)

    def test_risk_manager_sizes_down_suppressed_rotation_family(self):
        config = BotConfig(
            trading_mode="futures",
            default_leverage=1,
            rotation_risk_preferred_multiplier=1.08,
            rotation_risk_suppressed_multiplier=0.90,
        )
        state = BotState(balance=1000.0)
        risk = RiskManager(config, state)
        baseline = risk.calc_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            signal={
                "strategy": "trend_breakout",
                "metadata": {},
            },
        )
        suppressed = risk.calc_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            signal={
                "strategy": "trend_breakout",
                "metadata": {
                    "preferred_family": "trend_pullback",
                    "suppressed_family": "trend_breakout",
                    "rotation_confidence": 0.75,
                },
            },
        )
        self.assertLess(suppressed, baseline)

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

    def test_ensemble_relaxes_non_breakout_under_low_frequency_pressure(self):
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
                take_profit=105.0,
                strategy="trend_pullback",
                confidence=0.72,
                timeframe="15m",
                expected_edge_bps=8.5,
                regime="trending",
            ),
            expected_edge_bps=8.5,
            rationale="pullback",
        )
        decision = allocator.choose(
            "BTC/USDT",
            regime,
            [pullback],
            frequency_context={"status": "far_below_target", "dominant_strategy": None, "strategy_trade_share": {}},
        )
        self.assertEqual(decision.selected_strategy, "trend_pullback")
        self.assertEqual(decision.proposals[0]["frequency_reason"], "below_target_pullback_relax")

    def test_ensemble_tightens_dominant_strategy_when_trade_rate_is_high(self):
        allocator = EnsembleAllocator(BotConfig())
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
        breakout = StrategyProposal(
            signal=Signal(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                stop_loss=96.5,
                take_profit=106.0,
                strategy="trend_breakout",
                confidence=0.75,
                timeframe="15m",
                expected_edge_bps=12.0,
                regime="trending",
            ),
            expected_edge_bps=12.0,
            rationale="breakout",
        )
        pullback = StrategyProposal(
            signal=Signal(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                stop_loss=97.0,
                take_profit=106.0,
                strategy="trend_pullback",
                confidence=0.73,
                timeframe="15m",
                expected_edge_bps=12.0,
                regime="trending",
            ),
            expected_edge_bps=12.0,
            rationale="pullback",
        )
        decision = allocator.choose(
            "BTC/USDT",
            regime,
            [breakout, pullback],
            frequency_context={
                "status": "far_above_target",
                "dominant_strategy": "trend_breakout",
                "strategy_trade_share": {"trend_breakout": 0.8, "trend_pullback": 0.2},
            },
        )
        self.assertEqual(decision.selected_strategy, "trend_pullback")
        breakout_row = next(item for item in decision.proposals if item["strategy"] == "trend_breakout")
        self.assertEqual(breakout_row["frequency_reason"], "over_target_dominant_strategy_tighten")

    def test_ensemble_respects_rotation_preferred_family(self):
        allocator = EnsembleAllocator(BotConfig())
        regime = RegimeAssessment(
            regime="trending",
            confidence=0.82,
            volatility_ratio=0.01,
            trend_strength=0.02,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={
                "trend_direction": "bullish",
                "rotation_policy": {
                    "preferred_family": "trend_pullback",
                    "suppressed_family": "trend_breakout",
                    "confidence": 0.8,
                    "reason": "trend_pullback_bias",
                },
            },
        )
        breakout = StrategyProposal(
            signal=Signal(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                stop_loss=97.0,
                take_profit=106.0,
                strategy="trend_breakout",
                confidence=0.70,
                timeframe="15m",
                expected_edge_bps=14.0,
                regime="trending",
            ),
            expected_edge_bps=14.0,
            rationale="breakout",
        )
        pullback = StrategyProposal(
            signal=Signal(
                symbol="BTC/USDT",
                side="long",
                entry_price=100.0,
                stop_loss=97.0,
                take_profit=106.0,
                strategy="trend_pullback",
                confidence=0.70,
                timeframe="15m",
                expected_edge_bps=14.0,
                regime="trending",
            ),
            expected_edge_bps=14.0,
            rationale="pullback",
        )
        decision = allocator.choose("BTC/USDT", regime, [breakout, pullback])
        self.assertEqual(decision.selected_strategy, "trend_pullback")
        breakout_row = next(item for item in decision.proposals if item["strategy"] == "trend_breakout")
        pullback_row = next(item for item in decision.proposals if item["strategy"] == "trend_pullback")
        self.assertIn("suppressed_family", breakout_row["rotation_reason"])
        self.assertIn("preferred_family", pullback_row["rotation_reason"])

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

    def test_mean_reversion_strategy_blocks_countertrend_long_under_strong_bearish_continuation(self):
        closes = [100.0] * 90 + [98.0, 96.5, 95.0, 93.5, 92.0, 90.5, 89.5, 89.0, 88.8, 88.7]
        candles_15m = []
        for idx, close in enumerate(closes):
            candles_15m.append([float(idx), close + 0.4, close + 0.8, close - 0.9, close, 1200.0])

        class Helpers:
            def rsi(self, prices, period=14):
                values = [50.0] * len(prices)
                values[-2] = 24.0
                values[-1] = 28.0
                return values

            def bollinger_bands(self, prices, period=20, std_mult=2):
                middle = [94.0] * len(prices)
                upper = [99.5] * len(prices)
                lower = [88.6] * len(prices)
                return upper, middle, lower

            def compute_atr(self, symbol, timeframe="15m", period=14):
                return 1.8

        regime = RegimeAssessment(
            regime="mean_reverting",
            confidence=0.72,
            volatility_ratio=0.012,
            trend_strength=-0.014,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={
                "stretch_from_mean": -0.01,
                "trend_direction": "bearish",
                "directional_efficiency": 0.42,
                "continuation_score": 1.36,
                "entry_zscore": -1.18,
                "realized_vol_percentile": 0.74,
                "semivariance_skew": -0.03,
                "mean_reversion_score": 1.08,
                "exhaustion_score": 0.48,
            },
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
                return 2.0

        regime = RegimeAssessment(
            regime="trending",
            confidence=0.8,
            volatility_ratio=0.012,
            trend_strength=-0.02,
            liquidity_score=1.1,
            event_risk=False,
            unstable=False,
            metadata={
                "trend_direction": "bearish",
                "volume_impulse": 1.15,
                "directional_efficiency": 0.45,
                "continuation_score": 1.62,
                "breakout_score": 1.74,
                "trend_persistence": 0.44,
                "entry_zscore": -0.8,
                "stretch_from_mean": -0.01,
                "semivariance_skew": -0.08,
            },
        )
        strategy = TrendBreakoutStrategy(BotConfig(trading_mode="futures"), StrategyExchangeStub({"15m": candles_15m}), Helpers())
        proposal = strategy.evaluate("BTC/USDT", regime)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.signal.side, "short")
        self.assertTrue(proposal.signal.fast_move)
        preferred_order_type = proposal.signal.metadata.get("preferred_order_type")
        self.assertIn(preferred_order_type, {"limit", "market"})
        if preferred_order_type == "limit":
            self.assertTrue(proposal.signal.metadata.get("force_limit_entry"))
            self.assertGreater(proposal.signal.entry_price, candles_15m[-1][4])

    def test_trend_pullback_strategy_can_emit_long_in_bullish_trend(self):
        closes = [
            100.0, 100.4, 100.9, 101.3, 101.8, 102.4, 102.9, 103.5, 104.0, 104.6,
            105.2, 105.9, 106.5, 107.0, 107.6, 108.1, 108.7, 109.2, 109.8, 110.3,
            110.9, 111.4, 112.0, 112.6, 113.2, 113.8, 114.4, 115.0, 115.4, 115.7,
            115.3, 114.9, 114.4, 113.9, 113.4, 113.4, 113.4, 113.4, 113.4, 113.5,
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
            metadata={"trend_direction": "bullish", "volume_impulse": 1.05, "stretch_from_mean": 0.01, "directional_efficiency": 0.25},
        )
        strategy = TrendPullbackStrategy(BotConfig(), StrategyExchangeStub({"15m": candles_15m}), Helpers())
        proposal = strategy.evaluate("BTC/USDT", regime)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.signal.side, "long")

    def test_trend_pullback_strategy_can_emit_long_on_reclaim_hold_confirmation(self):
        closes = [
            100.0, 100.4, 100.9, 101.3, 101.8, 102.4, 102.9, 103.5, 104.0, 104.6,
            105.2, 105.9, 106.5, 107.0, 107.6, 108.1, 108.7, 109.2, 109.8, 110.3,
            110.9, 111.4, 112.0, 112.6, 113.2, 113.8, 114.4, 115.0, 115.4, 115.8,
            115.5, 115.0, 114.6, 114.2, 113.8, 113.6, 113.5, 113.48, 113.46, 113.43,
        ]
        candles_15m = []
        for idx, close in enumerate(closes):
            if idx == len(closes) - 1:
                candles_15m.append([float(idx), close - 0.22, close + 0.26, close - 0.12, close, 1750.0])
            else:
                candles_15m.append([float(idx), close - 0.18, close + 0.42, close - 0.42, close, 1500.0])

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
                return 1.1

        regime = RegimeAssessment(
            regime="trending",
            confidence=0.78,
            volatility_ratio=0.012,
            trend_strength=0.02,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={
                "trend_direction": "bullish",
                "volume_impulse": 1.02,
                "stretch_from_mean": 0.008,
                "directional_efficiency": 0.24,
                "pullback_score": 1.48,
                "trend_persistence": 0.50,
                "entry_zscore": 0.12,
                "realized_vol_percentile": 0.58,
            },
        )
        strategy = TrendPullbackStrategy(BotConfig(), StrategyExchangeStub({"15m": candles_15m}), Helpers())
        proposal = strategy.evaluate("BTC/USDT", regime)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.signal.side, "long")
        self.assertEqual(proposal.signal.metadata.get("confirmation_variant"), "reclaim_hold")

    def test_trend_pullback_strategy_prefers_shallow_profile_in_contained_volatility(self):
        class ShallowHelpers:
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

        class DeepHelpers(ShallowHelpers):
            def compute_atr(self, symbol, timeframe="15m", period=14):
                return 0.70

        shallow_closes = [
            100.0, 100.4, 100.9, 101.3, 101.8, 102.4, 102.9, 103.5, 104.0, 104.6,
            105.2, 105.9, 106.5, 107.0, 107.6, 108.1, 108.7, 109.2, 109.8, 110.3,
            110.9, 111.4, 112.0, 112.6, 113.2, 113.8, 114.4, 115.0, 115.4, 115.8,
            115.3, 114.9, 114.4, 113.9, 113.4, 113.4, 113.4, 113.4, 113.4, 113.5,
        ]
        deep_closes = [
            100.0, 100.4, 100.9, 101.3, 101.8, 102.4, 102.9, 103.5, 104.0, 104.6,
            105.2, 105.9, 106.5, 107.0, 107.6, 108.1, 108.7, 109.2, 109.8, 110.3,
            110.9, 111.4, 112.0, 112.6, 113.2, 113.8, 114.4, 115.0, 115.4, 115.8,
            115.0, 114.1, 113.2, 112.3, 111.7, 111.5, 111.48, 111.50, 111.52, 111.56,
        ]

        def make_candles(closes):
            candles = []
            for idx, close in enumerate(closes):
                if idx == len(closes) - 1:
                    candles.append([float(idx), close - 0.34, close + 0.34, close - 0.12, close, 1850.0])
                else:
                    candles.append([float(idx), close - 0.2, close + 0.5, close - 0.5, close, 1500.0])
            return candles

        strategy = TrendPullbackStrategy(BotConfig(), StrategyExchangeStub({"15m": make_candles(shallow_closes)}), ShallowHelpers())
        shallow_regime = RegimeAssessment(
            regime="trending",
            confidence=0.8,
            volatility_ratio=0.011,
            trend_strength=0.02,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={
                "trend_direction": "bullish",
                "volume_impulse": 1.04,
                "stretch_from_mean": 0.01,
                "directional_efficiency": 0.25,
                "pullback_score": 1.42,
                "trend_persistence": 0.48,
                "entry_zscore": 0.16,
                "realized_vol_percentile": 0.60,
            },
        )
        shallow = strategy.evaluate("BTC/USDT", shallow_regime)

        strategy = TrendPullbackStrategy(BotConfig(), StrategyExchangeStub({"15m": make_candles(deep_closes)}), DeepHelpers())
        deep_regime = RegimeAssessment(
            regime="trending",
            confidence=0.8,
            volatility_ratio=0.014,
            trend_strength=0.02,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={
                "trend_direction": "bullish",
                "volume_impulse": 0.98,
                "stretch_from_mean": 0.006,
                "directional_efficiency": 0.21,
                "continuation_score": 1.72,
                "pullback_score": 1.55,
                "trend_persistence": 0.46,
                "entry_zscore": 0.12,
                "realized_vol_percentile": 0.80,
            },
        )
        deep = strategy.evaluate("BTC/USDT", deep_regime)

        self.assertIsNotNone(shallow)
        self.assertEqual(shallow.signal.metadata.get("profile_preference"), "shallow_preferred")
        if deep is not None:
            self.assertEqual(deep.signal.metadata.get("profile_preference"), "deep_selective")
            self.assertGreater(shallow.signal.confidence, deep.signal.confidence)

    def test_trend_breakout_strategy_can_emit_confirmed_market_long(self):
        candles_15m = []
        base = 100.0
        for idx in range(30):
            close = base + (idx * 0.55)
            if idx == 29:
                close += 1.8
                candles_15m.append([float(idx), close - 1.4, close + 0.05, close - 1.0, close, 2400.0])
            else:
                candles_15m.append([float(idx), close - 0.2, close + 0.4, close - 0.5, close, 1600.0])

        class Helpers:
            def is_4h_bullish(self, symbol):
                return True

            def is_1h_uptrend(self, symbol):
                return True

            def is_4h_bearish(self, symbol):
                return False

            def is_1h_downtrend(self, symbol):
                return False

            def get_recent_swing_high_low(self, candles, lookback=5):
                recent = candles[-lookback:]
                return max(c[2] for c in recent), min(c[3] for c in recent)

            def compute_atr(self, symbol, timeframe="15m", period=14):
                return 2.0

        regime = RegimeAssessment(
            regime="trending",
            confidence=0.82,
            volatility_ratio=0.011,
            trend_strength=0.025,
            liquidity_score=1.05,
            event_risk=False,
            unstable=False,
            metadata={
                "trend_direction": "bullish",
                "volume_impulse": 1.08,
                "directional_efficiency": 0.44,
                "continuation_score": 1.58,
                "breakout_score": 1.62,
                "trend_persistence": 0.52,
                "entry_zscore": 0.92,
                "stretch_from_mean": 0.018,
                "semivariance_skew": 0.08,
            },
        )
        strategy = TrendBreakoutStrategy(BotConfig(trading_mode="futures"), StrategyExchangeStub({"15m": candles_15m}), Helpers())
        proposal = strategy.evaluate("BTC/USDT", regime)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.signal.side, "long")
        self.assertEqual(proposal.signal.metadata.get("strategy_variant"), "confirmed_market")
        self.assertEqual(proposal.signal.metadata.get("preferred_order_type"), "market")

    def test_trend_breakout_strategy_rejects_borderline_confirmed_market_long(self):
        candles_15m = []
        base = 100.0
        for idx in range(30):
            close = base + (idx * 0.55)
            if idx == 29:
                close += 1.6
                candles_15m.append([float(idx), close - 1.35, close + 0.05, close - 1.0, close, 2200.0])
            else:
                candles_15m.append([float(idx), close - 0.2, close + 0.4, close - 0.5, close, 1600.0])

        class Helpers:
            def is_4h_bullish(self, symbol):
                return True

            def is_1h_uptrend(self, symbol):
                return True

            def is_4h_bearish(self, symbol):
                return False

            def is_1h_downtrend(self, symbol):
                return False

            def get_recent_swing_high_low(self, candles, lookback=5):
                recent = candles[-lookback:]
                return max(c[2] for c in recent), min(c[3] for c in recent)

            def compute_atr(self, symbol, timeframe="15m", period=14):
                return 2.0

        regime = RegimeAssessment(
            regime="trending",
            confidence=0.82,
            volatility_ratio=0.011,
            trend_strength=0.025,
            liquidity_score=1.05,
            event_risk=False,
            unstable=False,
            metadata={
                "trend_direction": "bullish",
                "volume_impulse": 1.05,
                "directional_efficiency": 0.33,
                "continuation_score": 1.58,
                "breakout_score": 1.62,
                "trend_persistence": 0.52,
                "entry_zscore": 0.98,
                "stretch_from_mean": 0.018,
                "semivariance_skew": 0.08,
            },
        )
        strategy = TrendBreakoutStrategy(BotConfig(trading_mode="futures"), StrategyExchangeStub({"15m": candles_15m}), Helpers())
        proposal = strategy.evaluate("BTC/USDT", regime)
        self.assertIsNotNone(proposal)
        self.assertNotEqual(proposal.signal.metadata.get("strategy_variant"), "confirmed_market")

    def test_trend_breakout_strategy_blocks_stressed_rebound_breakout(self):
        candles_15m = []
        base = 100.0
        for idx in range(30):
            close = base + (idx * 0.55)
            if idx == 29:
                close += 1.8
                candles_15m.append([float(idx), close - 1.4, close + 0.05, close - 1.0, close, 2400.0])
            else:
                candles_15m.append([float(idx), close - 0.2, close + 0.4, close - 0.5, close, 1600.0])

        class Helpers:
            def is_4h_bullish(self, symbol):
                return True

            def is_1h_uptrend(self, symbol):
                return True

            def is_4h_bearish(self, symbol):
                return False

            def is_1h_downtrend(self, symbol):
                return False

            def get_recent_swing_high_low(self, candles, lookback=5):
                recent = candles[-lookback:]
                return max(c[2] for c in recent), min(c[3] for c in recent)

            def compute_atr(self, symbol, timeframe="15m", period=14):
                return 2.0

        regime = RegimeAssessment(
            regime="trending",
            confidence=0.82,
            volatility_ratio=0.011,
            trend_strength=0.025,
            liquidity_score=1.05,
            event_risk=False,
            unstable=False,
            metadata={
                "trend_direction": "bullish",
                "volume_impulse": 1.12,
                "directional_efficiency": 0.44,
                "continuation_score": 1.62,
                "breakout_score": 1.70,
                "trend_persistence": 0.54,
                "entry_zscore": 0.85,
                "stretch_from_mean": 0.018,
                "semivariance_skew": 0.08,
                "recent_drawdown": -0.06,
                "rebound_from_trough": 0.02,
                "momentum_crash_risk": 0.75,
            },
        )
        strategy = TrendBreakoutStrategy(BotConfig(trading_mode="futures"), StrategyExchangeStub({"15m": candles_15m}), Helpers())
        proposal = strategy.evaluate("BTC/USDT", regime)
        self.assertIsNone(proposal)

    def test_trend_breakout_strategy_applies_weak_confirmation_penalty(self):
        candles_15m = []
        base = 100.0
        for idx in range(30):
            close = base + (idx * 0.55)
            if idx == 29:
                close += 1.4
                candles_15m.append([float(idx), close - 1.2, close + 0.05, close - 1.0, close, 2200.0])
            else:
                candles_15m.append([float(idx), close - 0.2, close + 0.4, close - 0.5, close, 1600.0])

        class Helpers:
            def is_4h_bullish(self, symbol):
                return True

            def is_1h_uptrend(self, symbol):
                return True

            def is_4h_bearish(self, symbol):
                return False

            def is_1h_downtrend(self, symbol):
                return False

            def get_recent_swing_high_low(self, candles, lookback=5):
                recent = candles[-lookback:]
                return max(c[2] for c in recent), min(c[3] for c in recent)

            def compute_atr(self, symbol, timeframe="15m", period=14):
                return 2.0

        strong_regime = RegimeAssessment(
            regime="trending",
            confidence=0.82,
            volatility_ratio=0.011,
            trend_strength=0.025,
            liquidity_score=1.05,
            event_risk=False,
            unstable=False,
            metadata={
                "trend_direction": "bullish",
                "volume_impulse": 1.14,
                "directional_efficiency": 0.44,
                "continuation_score": 1.60,
                "breakout_score": 1.68,
                "trend_persistence": 0.54,
                "entry_zscore": 0.62,
                "stretch_from_mean": 0.010,
                "semivariance_skew": 0.08,
            },
        )
        weak_regime = RegimeAssessment(
            regime="trending",
            confidence=0.82,
            volatility_ratio=0.011,
            trend_strength=0.025,
            liquidity_score=1.05,
            event_risk=False,
            unstable=False,
            metadata={
                "trend_direction": "bullish",
                "volume_impulse": 1.08,
                "directional_efficiency": 0.34,
                "continuation_score": 1.60,
                "breakout_score": 1.68,
                "trend_persistence": 0.54,
                "entry_zscore": 0.90,
                "stretch_from_mean": 0.018,
                "semivariance_skew": 0.08,
            },
        )
        strategy = TrendBreakoutStrategy(BotConfig(trading_mode="futures"), StrategyExchangeStub({"15m": candles_15m}), Helpers())
        strong = strategy.evaluate("BTC/USDT", strong_regime)
        weak = strategy.evaluate("BTC/USDT", weak_regime)
        self.assertIsNotNone(strong)
        self.assertIsNotNone(weak)
        self.assertGreater(float(weak.signal.metadata.get("weak_confirmation_penalty_bps", 0.0) or 0.0), 0.0)
        self.assertGreater(strong.expected_edge_bps, weak.expected_edge_bps)
        self.assertGreater(strong.signal.confidence, weak.signal.confidence)

    def test_mean_reversion_strategy_can_emit_confirmed_reversal_in_chop(self):
        closes = [100.0] * 90 + [99.6, 99.2, 98.9, 98.5, 98.2, 98.0, 97.9, 98.0, 98.1, 98.42]
        candles_15m = []
        for idx, close in enumerate(closes):
            candles_15m.append([float(idx), close + 0.1, close + 0.5, close - 0.5, close, 1200.0])

        class Helpers:
            def rsi(self, prices, period=14):
                values = [50.0] * len(prices)
                values[-2] = 24.0
                values[-1] = 27.0
                return values

            def bollinger_bands(self, prices, period=20, std_mult=2):
                middle = [99.8] * len(prices)
                upper = [101.0] * len(prices)
                lower = [98.40] * len(prices)
                return upper, middle, lower

            def compute_atr(self, symbol, timeframe="15m", period=14):
                return 0.8

        regime = RegimeAssessment(
            regime="mean_reverting",
            confidence=0.72,
            volatility_ratio=0.01,
            trend_strength=0.001,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={
                "stretch_from_mean": -0.008,
                "trend_direction": "flat",
                "directional_efficiency": 0.18,
                "entry_zscore": -1.35,
            },
        )
        strategy = MeanReversionStrategy(BotConfig(), StrategyExchangeStub({"15m": candles_15m}), Helpers())
        proposal = strategy.evaluate("BTC/USDT", regime)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.signal.side, "long")

    def test_mean_reversion_strategy_can_emit_liquid_relaxed_long_in_low_efficiency_chop(self):
        closes = [100.0] * 90 + [99.6, 99.2, 98.9, 98.5, 98.2, 98.0, 97.9, 98.0, 98.05, 98.52]
        candles_15m = []
        for idx, close in enumerate(closes):
            candles_15m.append([float(idx), close + 0.1, close + 0.24, close - 0.24, close, 1800.0])

        class Helpers:
            def rsi(self, prices, period=14):
                values = [50.0] * len(prices)
                values[-2] = 30.0
                values[-1] = 33.0
                return values

            def bollinger_bands(self, prices, period=20, std_mult=2):
                middle = [99.2] * len(prices)
                upper = [100.2] * len(prices)
                lower = [98.16] * len(prices)
                return upper, middle, lower

            def compute_atr(self, symbol, timeframe="15m", period=14):
                return 0.48

        regime = RegimeAssessment(
            regime="choppy",
            confidence=0.70,
            volatility_ratio=0.01,
            trend_strength=0.002,
            liquidity_score=0.92,
            event_risk=False,
            unstable=False,
            metadata={
                "spread": 0.0009,
                "stretch_from_mean": -0.004,
                "trend_direction": "flat",
                "directional_efficiency": 0.24,
                "continuation_score": 1.10,
                "mean_reversion_score": 1.01,
                "exhaustion_score": 0.38,
                "entry_zscore": -1.12,
                "realized_vol_percentile": 0.66,
                "semivariance_skew": -0.04,
            },
        )
        strategy = MeanReversionStrategy(BotConfig(), StrategyExchangeStub({"15m": candles_15m}), Helpers())
        proposal = strategy.evaluate("BTC/USDT", regime)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.signal.side, "long")
        self.assertTrue(proposal.signal.metadata.get("liquid_relaxed"))

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

    def test_market_regime_engine_emits_richer_regime_metadata(self):
        candles_1h = []
        candles_15m = []
        base = 100.0
        for idx in range(60):
            close = base + (idx * 0.35)
            candles_1h.append([float(idx), close - 0.3, close + 0.6, close - 0.5, close, 1200.0])
        for idx in range(60):
            close = base + (idx * 0.08)
            if idx >= 52:
                close += (idx - 51) * 0.18
            candles_15m.append([float(idx), close - 0.15, close + 0.25, close - 0.2, close, 1400.0 if idx >= 52 else 1000.0])

        class ExchangeStub:
            def fetch_ohlcv(self, symbol, timeframe, limit=60):
                if timeframe == "1h":
                    return candles_1h[-limit:]
                return candles_15m[-limit:]

            def get_order_book(self, symbol):
                last_close = float(candles_15m[-1][4])
                return {"bid": last_close * 0.9995, "ask": last_close * 1.0005}

        assessment = MarketRegimeEngine(BotConfig(), ExchangeStub()).classify("BTC/USDT")
        self.assertIn("continuation_score", assessment.metadata)
        self.assertIn("mean_reversion_score", assessment.metadata)
        self.assertIn("realized_vol_percentile", assessment.metadata)
        self.assertIn("semivariance_skew", assessment.metadata)
        self.assertIn("trend_persistence", assessment.metadata)
        self.assertIn("rotation_policy", assessment.metadata)
        self.assertIn("preferred_family", assessment.metadata)
        self.assertIn("suppressed_family", assessment.metadata)
        self.assertIn("rotation_confidence", assessment.metadata)

    def test_market_regime_engine_emits_pullback_biased_rotation_under_crash_risk(self):
        candles_1h = []
        candles_15m = []
        price = 100.0
        for idx in range(60):
            if idx < 40:
                price += 0.55
            elif idx < 54:
                price -= 1.05
            else:
                price += 0.85
            close = round(price, 4)
            candles_1h.append([float(idx), close - 0.3, close + 0.6, close - 0.8, close, 1200.0])
            candles_15m.append([float(idx), close - 0.2, close + 0.45, close - 0.55, close, 1400.0])

        class ExchangeStub:
            def fetch_ohlcv(self, symbol, timeframe, limit=60):
                if timeframe == "1h":
                    return candles_1h[-limit:]
                return candles_15m[-limit:]

            def get_order_book(self, symbol):
                last_close = float(candles_15m[-1][4])
                return {"bid": last_close * 0.9995, "ask": last_close * 1.0005}

        assessment = MarketRegimeEngine(BotConfig(), ExchangeStub()).classify("BTC/USDT")
        self.assertEqual(assessment.metadata.get("preferred_family"), "trend_pullback")
        self.assertEqual(assessment.metadata.get("suppressed_family"), "trend_breakout")
        self.assertGreaterEqual(float(assessment.metadata.get("rotation_confidence", 0.0) or 0.0), 0.6)

    def test_reliability_checks_relax_for_preferred_family(self):
        engine = SignalEngine(BotConfig(), DummyExchange())
        regime = RegimeAssessment(
            regime="trending",
            confidence=0.8,
            volatility_ratio=0.01,
            trend_strength=0.02,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={
                "trend_direction": "bullish",
                "rotation_policy": {
                    "preferred_family": "trend_pullback",
                    "suppressed_family": "trend_breakout",
                    "confidence": 1.0,
                    "reason": "trend_pullback_bias",
                },
            },
        )
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "entry_price": 100.0,
            "stop_loss": 97.0,
            "take_profit": 104.0,
            "strategy": "trend_pullback",
            "signal_quality": 0.53,
            "expected_edge_bps": 6.7,
            "rr_ratio": 1.31,
            "hurst_exponent": 0.4,
            "metadata": {},
            "research_context": {},
        }
        self.assertTrue(engine._passes_reliability_checks("BTC/USDT", signal, regime, proposal_count=1))

    def test_reliability_checks_tighten_for_suppressed_family(self):
        engine = SignalEngine(BotConfig(), DummyExchange())
        regime = RegimeAssessment(
            regime="trending",
            confidence=0.8,
            volatility_ratio=0.01,
            trend_strength=0.02,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={
                "trend_direction": "bullish",
                "rotation_policy": {
                    "preferred_family": "trend_pullback",
                    "suppressed_family": "trend_breakout",
                    "confidence": 1.0,
                    "reason": "trend_pullback_bias",
                },
            },
        )
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "entry_price": 100.0,
            "stop_loss": 97.0,
            "take_profit": 104.0,
            "strategy": "trend_breakout",
            "signal_quality": 0.57,
            "expected_edge_bps": 8.2,
            "rr_ratio": 1.36,
            "hurst_exponent": 0.4,
            "metadata": {},
            "research_context": {},
        }
        self.assertFalse(engine._passes_reliability_checks("BTC/USDT", signal, regime, proposal_count=1))

    def test_trade_learning_engine_flags_negative_cell_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(BotConfig(), store)
            signal = {
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 103.0,
                "strategy": "trend_pullback",
                "signal_quality": 0.68,
                "expected_edge_bps": 16.0,
                "regime": "trending",
                "rr_ratio": 1.5,
                "hurst_exponent": 0.28,
                "metadata": {
                    "trend_direction": "bullish",
                    "regime_confidence": 0.72,
                    "preferred_order_type": "limit",
                },
            }
            store.upsert_learning_model(
                "learning_opportunity::cell::trend_pullback::long::trending::limit",
                {
                    "total": 8.0,
                    "positive_forward": 2.0,
                    "total_forward_r": -2.4,
                },
            )
            store.upsert_learning_model(
                "learning_prequential::cell::trend_pullback::long::trending::limit",
                {
                    "count": 8.0,
                    "wins": 2.0,
                    "sum_brier": 4.0,
                    "sum_confidence": 5.6,
                    "sum_r_multiple": -2.8,
                },
            )
            store.upsert_learning_model(
                "learning_calibration::cell::trend_pullback::long::trending::limit::high",
                {
                    "successes": 2.0,
                    "total": 8.0,
                    "mean_gap": -0.30,
                },
            )

            context = learning.learning_context_for_signal(signal)
            self.assertTrue(context["negative_cell_evidence"])
            self.assertLess(context["risk_multiplier"], 1.0)

    def test_trade_learning_engine_early_vetoes_repeated_structural_losses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            learning = TradeLearningEngine(BotConfig(), store)
            signal = {
                "symbol": "LINK/USDT",
                "side": "short",
                "entry_price": 100.0,
                "stop_loss": 102.0,
                "take_profit": 97.0,
                "strategy": "trend_pullback",
                "signal_quality": 0.74,
                "expected_edge_bps": 18.0,
                "regime": "trending",
                "rr_ratio": 1.5,
                "hurst_exponent": 0.35,
                "metadata": {
                    "trend_direction": "flat",
                    "regime_confidence": 0.55,
                    "preferred_order_type": "limit",
                    "strategy_variant": "shallow_pullback",
                },
            }
            decision_context = learning.build_trade_context(signal)
            position = Position(
                symbol="LINK/USDT",
                side="short",
                entry_price=100.0,
                size=1.0,
                stop_loss=102.0,
                take_profit=97.0,
                strategy="trend_pullback",
                initial_stop_loss=102.0,
                metadata={"decision_context": decision_context},
            )
            for _ in range(3):
                learning.record_closed_trade(
                    symbol="LINK/USDT",
                    position=position,
                    close_price=102.0,
                    profit_loss=-2.0,
                    exit_reason="SL",
                )

            context = learning.learning_context_for_signal(signal)
            self.assertTrue(context["veto"])
            self.assertIn("structural_repeat", context["dominant_attributions"])
            self.assertLess(context["risk_multiplier"], 0.9)

    def test_trade_learning_engine_promotes_positive_cells_from_opportunity_and_prequential_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteStateStore(os.path.join(tmpdir, "runtime.sqlite3"))
            config = BotConfig()
            learning = TradeLearningEngine(config, store)
            observed_at = dt.datetime.now(dt.timezone.utc).isoformat()
            store.upsert_learning_model(
                "learning_opportunity::cell::trend_breakout::long::trending::limit",
                {
                    "updated_at": observed_at,
                    "total": 8.0,
                    "positive_forward": 6.0,
                    "total_forward_r": 2.8,
                },
            )
            store.upsert_learning_model(
                "learning_prequential::cell::trend_breakout::long::trending::limit",
                {
                    "updated_at": observed_at,
                    "count": 8.0,
                    "wins": 5.0,
                    "sum_brier": 1.2,
                    "sum_confidence": 4.6,
                    "sum_r_multiple": 2.4,
                },
            )
            store.upsert_learning_model(
                "learning_calibration::cell::trend_breakout::long::trending::limit::low",
                {
                    "updated_at": observed_at,
                    "successes": 5.0,
                    "total": 8.0,
                    "mean_gap": 0.04,
                },
            )
            signal = {
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "take_profit": 103.6,
                "strategy": "trend_breakout",
                "signal_quality": 0.62,
                "expected_edge_bps": 16.0,
                "regime": "trending",
                "rr_ratio": 1.8,
                "hurst_exponent": 0.36,
                "metadata": {
                    "trend_direction": "bullish",
                    "regime_confidence": 0.78,
                    "preferred_order_type": "limit",
                },
            }

            context = learning.learning_context_for_signal(signal)
            self.assertTrue(context["positive_cell_evidence"])
            self.assertGreaterEqual(context["score_delta"], config.learning_positive_min_score_delta)
            self.assertGreaterEqual(context["confidence_delta"], config.learning_positive_min_confidence_delta)
            self.assertGreaterEqual(context["risk_multiplier"], config.learning_positive_min_risk_multiplier)
            self.assertIn("positive_cell", context["dominant_attributions"])

    def test_persistent_learning_store_keeps_learning_global_but_pending_decisions_local(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_store = SQLiteStateStore(os.path.join(tmpdir, "simulation.sqlite3"))
            global_store = SQLiteStateStore(os.path.join(tmpdir, "bot_learning.sqlite3"))
            store = PersistentLearningStore(local_store, global_store)

            observation = {
                "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "symbol": "BTC/USDT",
                "strategy": "trend_breakout",
                "side": "long",
                "success": True,
                "r_multiple": 1.25,
            }
            store.append_learning_observation(observation)
            self.assertEqual(len(global_store.load_recent_learning_observations(limit=5)), 1)
            self.assertEqual(len(local_store.load_recent_learning_observations(limit=5)), 0)

            payload = {"created_at": observation["observed_at"], "symbol": "BTC/USDT"}
            store.append_learning_decision("decision-1", observation["observed_at"], "pending", payload)
            self.assertEqual(len(store.load_pending_learning_decisions(limit=5)), 1)
            self.assertEqual(len(global_store.load_pending_learning_decisions(limit=5)), 0)

    def test_backfill_learning_imports_historical_sqlite_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            legacy_store = SQLiteStateStore(os.path.join(data_dir, "legacy_campaign.sqlite3"))
            global_store = SQLiteStateStore(os.path.join(data_dir, "bot_learning.sqlite3"))

            observation = {
                "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "symbol": "ETH/USDT",
                "strategy": "mean_reversion",
                "side": "short",
                "success": False,
                "r_multiple": -0.75,
            }
            legacy_store.append_learning_observation(observation)
            legacy_store.upsert_learning_pattern(
                "pattern::test",
                {"updated_at": observation["observed_at"], "effective_samples": 1.0, "total_r_multiple": -0.75},
            )
            legacy_store.upsert_learning_model(
                "learning_prequential::global",
                {"updated_at": observation["observed_at"], "count": 1.0, "wins": 0.0},
            )

            summary_one = backfill_learning_from_sqlite_artifacts(tmpdir, global_store)
            inventory_one = global_store.load_learning_inventory()
            summary_two = backfill_learning_from_sqlite_artifacts(tmpdir, global_store)
            inventory_two = global_store.load_learning_inventory()

            self.assertEqual(summary_one["imported_files"], 1)
            self.assertEqual(summary_one["imported_observations"], 1)
            self.assertEqual(summary_one["imported_patterns"], 1)
            self.assertEqual(summary_one["imported_models"], 1)
            self.assertEqual(summary_two["imported_files"], 0)
            self.assertEqual(inventory_one["observations"], 1)
            self.assertEqual(inventory_one["patterns"], 1)
            self.assertEqual(inventory_one["models"], 1)
            self.assertEqual(inventory_two["observations"], 1)

    def test_reliability_checks_allow_slightly_low_calibration_when_positive_cell_is_strong(self):
        config = BotConfig()
        class _StubBot:
            def __init__(self, config):
                self.config = config
            def _passes_research_checks(self, side, research_context):
                return True
        bot = _StubBot(config)
        regime = RegimeAssessment(
            regime="trending",
            confidence=0.7,
            volatility_ratio=0.01,
            trend_strength=0.03,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={"trend_direction": "bullish"},
        )
        signal = {
            "symbol": "BTC/USDT",
            "side": "long",
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit": 103.0,
            "strategy": "trend_breakout",
            "signal_quality": 0.60,
            "expected_edge_bps": 15.0,
            "rr_ratio": 1.5,
            "hurst_exponent": 0.35,
            "research_context": {},
            "metadata": {
                "proposal_count": 2,
                "preferred_order_type": "limit",
                "learning_context": {
                    "veto": False,
                    "positive_cell_evidence": True,
                    "calibration": {
                        "effective_samples": 8.0,
                        "calibrated_confidence": 0.48,
                    },
                    "opportunity": {
                        "samples": 8.0,
                        "avg_forward_r": 0.30,
                    },
                },
            },
        }

        self.assertTrue(SignalEngine._passes_reliability_checks(bot, "BTC/USDT", signal, regime, 2))

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

    def test_signal_engine_rejects_poorly_calibrated_signal_with_evidence(self):
        engine = SignalEngine(BotConfig(), DummyExchange())
        regime = RegimeAssessment(
            regime="trending",
            confidence=0.8,
            volatility_ratio=0.01,
            trend_strength=0.02,
            liquidity_score=1.0,
            event_risk=False,
            unstable=False,
            metadata={"trend_direction": "bearish"},
        )
        signal = {
            "side": "short",
            "entry_price": 100.0,
            "stop_loss": 102.0,
            "take_profit": 95.0,
            "strategy": "trend_breakout",
            "signal_quality": 0.82,
            "expected_edge_bps": 18.0,
            "rr_ratio": 2.5,
            "hurst_exponent": 0.55,
            "metadata": {
                "learning_context": {
                    "veto": False,
                    "calibration": {
                        "effective_samples": 8.0,
                        "calibrated_confidence": 0.42,
                    },
                    "opportunity": {
                        "samples": 0.0,
                        "avg_forward_r": 0.0,
                    },
                }
            },
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

    def test_cli_parser_supports_simulation_modes(self):
        parser = build_parser()
        args = parser.parse_args(["simulate", "--detach"])
        self.assertEqual(args.mode, "simulate")
        self.assertTrue(args.detach)
        stop_args = parser.parse_args(["simulation-stop"])
        self.assertEqual(stop_args.mode, "simulation-stop")

    def test_simulation_stop_request_updates_runtime_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            simulation_runtime_dir(tmpdir)
            status = request_simulation_stop(tmpdir)
            self.assertEqual(status["status"], "stop_requested")
            loaded = read_simulation_status(tmpdir)
            self.assertTrue(loaded["stop_requested"])

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
        self.assertIsInstance(response["readiness_report"], dict)
        self.assertIn("ready", response["readiness_report"])

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

    def test_tradebot_repeated_save_state_keeps_runtime_snapshot_flat(self):
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
                bot.save_state()
                bot.save_state()
                snapshot = bot.state_store.load_snapshot("runtime")
                runtime_snapshot = (((snapshot.get("readiness", {}) or {}).get("metrics", {}) or {}).get("runtime_snapshot", {}) or {})
                self.assertIn("updated_at", runtime_snapshot)
                self.assertNotIn("readiness", runtime_snapshot)
                self.assertNotIn("portfolio", runtime_snapshot)
            finally:
                os.chdir(cwd)

    def test_tradebot_save_state_refreshes_bot_control_status(self):
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
                bot.state.balance = 321.0
                bot.save_state()
                status = read_bot_status(bot.base_dir)
                self.assertEqual(status.get("status"), "running")
                self.assertEqual(int(status.get("pid", 0) or 0), os.getpid())
                self.assertAlmostEqual(float(status.get("balance", 0.0) or 0.0), 321.0, places=6)
            finally:
                os.chdir(cwd)

    def test_tradebot_refresh_control_status_persists_last_heartbeat(self):
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
                bot.state.last_heartbeat = dt.datetime.now() - dt.timedelta(minutes=5)
                bot._refresh_control_status(status="running")
                status = read_bot_status(bot.base_dir)
                self.assertEqual(status.get("status"), "running")
                self.assertTrue(bool(status.get("last_heartbeat")))
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

    def test_campaign_comparison_flags_acceptance_rules(self):
        comparison = build_campaign_comparison(
            {"num_trades": 5, "win_rate_pct": 40.0, "total_return_pct": 0.3071, "profit_factor": 1.28},
            {"num_trades": 9, "win_rate_pct": 44.0, "total_return_pct": 0.8123, "profit_factor": 1.55},
        )

        self.assertTrue(comparison["acceptance"]["more_trades"])
        self.assertTrue(comparison["acceptance"]["better_profit"])
        self.assertTrue(comparison["acceptance"]["win_rate_not_worse"])
        self.assertTrue(comparison["acceptance"]["passes_all"])

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

    def test_mock_backtest_exchange_hides_incomplete_higher_timeframe_bar(self):
        import pandas as pd

        df_15m = pd.DataFrame(
            [
                {"timestamp": "2025-01-01T09:00:00", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
                {"timestamp": "2025-01-01T09:15:00", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 12},
                {"timestamp": "2025-01-01T09:30:00", "open": 101, "high": 103, "low": 100, "close": 102, "volume": 15},
                {"timestamp": "2025-01-01T09:45:00", "open": 102, "high": 104, "low": 101, "close": 103, "volume": 18},
            ]
        ).set_index("timestamp")
        df_15m.index = pd.to_datetime(df_15m.index)

        df_1h = pd.DataFrame(
            [
                {"timestamp": "2025-01-01T08:00:00", "open": 95, "high": 100, "low": 94, "close": 99, "volume": 40},
                {"timestamp": "2025-01-01T09:00:00", "open": 100, "high": 104, "low": 99, "close": 103, "volume": 55},
            ]
        ).set_index("timestamp")
        df_1h.index = pd.to_datetime(df_1h.index)

        exch = MockBacktestExchange(
            {
                ("BTC/USDT", "15m"): df_15m,
                ("BTC/USDT", "1h"): df_1h,
            },
            "BTC/USDT",
            base_timeframe="15m",
        )
        exch.set_cursor(0)
        candles = exch.fetch_ohlcv("BTC/USDT", "1h", limit=10)

        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[-1][4], 99)

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

    def test_backtest_engine_feeds_learning_for_closed_trades(self):
        import pandas as pd

        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        rows = []
        for idx, ts in enumerate(timestamps):
            close = 100.0
            row = {
                "timestamp": ts,
                "open": close,
                "high": 101.0 if idx != 101 else 105.0,
                "low": 99.0,
                "close": close,
                "volume": 1000.0,
            }
            rows.append(row)
        df = pd.DataFrame(rows).set_index("timestamp")

        engine = BacktestEngine(BotConfig(starting_balance=1000.0, trading_mode="spot"))

        class LongOnlySignals:
            def __init__(self, config, exch):
                self.exch = exch

            def generate_signal(self, symbol):
                candles = self.exch.fetch_ohlcv(symbol, "15m", limit=200)
                if len(candles) == 101:
                    return {
                        "side": "long",
                        "entry_price": 100.0,
                        "stop_loss": 99.0,
                        "take_profit": 104.0,
                        "strategy": "unit_test_long",
                        "expected_holding_minutes": 15,
                        "signal_quality": 0.72,
                        "expected_edge_bps": 16.0,
                        "rr_ratio": 4.0,
                        "metadata": {},
                    }
                return None

        with mock.patch("trade_bot.main.SignalEngine", LongOnlySignals):
            engine.load_historical_data = mock.Mock(return_value=df)
            result = engine.run_backtest("BTC/USDT", timeframe="15m", days=10)

        self.assertEqual(result["num_trades"], 1)
        self.assertGreaterEqual(result["learning_summary"]["recent_trades"], 1)

    def test_backtest_engine_evaluates_shadow_decisions_sequentially(self):
        import pandas as pd

        timestamps = pd.date_range("2025-01-01", periods=130, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 102.0,
                    "low": 98.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        engine = BacktestEngine(BotConfig(starting_balance=1000.0, trading_mode="spot"))

        class SpotShortSignals:
            def __init__(self, config, exch):
                self.exch = exch

            def generate_signal(self, symbol):
                candles = self.exch.fetch_ohlcv(symbol, "15m", limit=200)
                if len(candles) == 101:
                    return {
                        "side": "short",
                        "entry_price": 100.0,
                        "stop_loss": 101.0,
                        "take_profit": 98.0,
                        "strategy": "unit_test_shadow_short",
                        "expected_holding_minutes": 15,
                        "signal_quality": 0.68,
                        "expected_edge_bps": 12.0,
                        "rr_ratio": 2.0,
                        "metadata": {},
                    }
                return None

        with mock.patch("trade_bot.main.SignalEngine", SpotShortSignals):
            engine.load_historical_data = mock.Mock(return_value=df)
            result = engine.run_backtest("BTC/USDT", timeframe="15m", days=10)

        self.assertEqual(result["num_trades"], 0)
        self.assertGreaterEqual(result["skipped_signals"], 1)
        self.assertEqual(result["shadow_decisions_pending"], 0)
        self.assertGreaterEqual(result["learning_summary"]["opportunity"]["samples"], 1)

    def test_backtest_engine_can_stop_early_without_killing_process(self):
        import pandas as pd

        timestamps = pd.date_range("2025-01-01", periods=140, freq="15min")
        df = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
                for ts in timestamps
            ]
        ).set_index("timestamp")

        stop_flag = {"calls": 0}

        def stop_requested():
            stop_flag["calls"] += 1
            return stop_flag["calls"] > 8

        engine = BacktestEngine(
            BotConfig(starting_balance=1000.0, trading_mode="spot"),
            stop_requested=stop_requested,
        )
        engine.load_historical_data = mock.Mock(return_value=df)

        class NoopSignals:
            def __init__(self, config, exch):
                self.exch = exch

            def generate_signal(self, symbol):
                return None

        with mock.patch("trade_bot.main.SignalEngine", NoopSignals):
            result = engine.run_backtest("BTC/USDT", timeframe="15m", days=10)

        self.assertTrue(result["stopped_early"])
        self.assertEqual(result["stop_reason"], "stop_requested")

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
