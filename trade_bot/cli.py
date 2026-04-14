from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

from .bootstrap import getenv_bool, load_runtime_environment, resolve_runtime_base_dir
from .config import BotConfig
from .health import evaluate_runtime_health, render_runtime_health
from .main import BacktestEngine, HyperoptEngine, TradeBot
from .backtest_reporting import build_backtest_report, build_batch_summary, load_batch_reports, save_backtest_report
from .telegram_operator import run_telegram_operator
from .simulation_service import (
    clear_stop_request,
    clear_simulation_batch_stop,
    persist_simulation_report,
    read_simulation_status,
    read_simulation_batch_status,
    request_simulation_stop,
    request_simulation_batch_stop,
    simulation_runtime_dir,
    simulation_batch_dir,
    simulation_batch_paths,
    simulation_batch_stop_requested,
    simulation_stop_requested,
    spawn_detached_simulation,
    spawn_detached_simulation_batch,
    write_simulation_status,
    write_simulation_batch_status,
)


def run_backtest_cli(symbol: str = "BTC/USDT", days: int = 180, timeframe: str = "15m") -> None:
    config = BotConfig(
        starting_balance=10000.0,
        telegram_bot_token="",
        telegram_chat_id="",
        api_key="",
        api_secret="",
    )
    bt = BacktestEngine(config)
    results = bt.run_backtest(symbol, timeframe, days)
    report = build_backtest_report(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        starting_balance=config.starting_balance,
        metrics=results,
        assumptions={
            "warmup_candles": config.backtest_warmup_candles,
            "fee_bps": config.backtest_fee_bps,
            "slippage_bps": config.backtest_slippage_bps,
            "spread_bps": config.backtest_spread_bps,
            "execution_model": "sequential_replay_with_latency_spread_slippage_and_learning_feedback",
        },
    )
    save_backtest_report("backtest_results.json", report)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS:")
    print("=" * 60)
    for key, value in report["metrics"].items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    print("\nFull results saved to backtest_results.json")
    print("=" * 60)


def _simulation_progress_writer(base_dir: str):
    def _writer(progress: dict) -> None:
        write_simulation_status(
            base_dir,
            {
                "status": "running",
                "updated_at": progress.get("timestamp"),
                "progress": progress,
            },
        )

    return _writer


def _render_simulation_status(status: dict, *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(status, indent=2, sort_keys=True, default=str)
    lines = [
        f"Status: {status.get('status', 'unknown')}",
        f"Running: {bool(status.get('running', False))}",
        f"PID: {status.get('pid', 'n/a')}",
        f"Health: {dict(status.get('health', {}) or {}).get('status', 'unknown')} ({dict(status.get('health', {}) or {}).get('reason', 'unknown')})",
        f"Symbol: {status.get('symbol', 'n/a')}",
        f"Timeframe: {status.get('timeframe', 'n/a')}",
        f"Days: {status.get('days', 'n/a')}",
        f"Started At: {status.get('started_at', 'n/a')}",
        f"Updated At: {status.get('updated_at', 'n/a')}",
        f"Stop Requested: {bool(status.get('stop_requested', False))}",
        f"Checkpoint Exists: {bool(status.get('checkpoint_exists', False))}",
    ]
    progress = dict(status.get("progress", {}) or {})
    if progress:
        lines.extend(
            [
                f"Progress: {progress.get('current_bar', 0)}/{progress.get('total_bars', 0)}",
                f"Trades: {progress.get('num_trades', 0)}",
                f"Balance: {float(progress.get('balance', 0.0) or 0.0):.2f}",
                f"Equity: {float(progress.get('equity', 0.0) or 0.0):.2f}",
                f"Open Orders: {progress.get('open_orders', 0)}",
                f"Cancelled Orders: {progress.get('cancelled_orders', 0)}",
                f"Ambiguous Exit Bars: {progress.get('ambiguous_exit_bars', 0)}",
            ]
        )
    report_path = status.get("paths", {}).get("report")
    if report_path:
        lines.append(f"Report Path: {report_path}")
    checkpoint_path = status.get("paths", {}).get("checkpoint")
    if checkpoint_path:
        lines.append(f"Checkpoint Path: {checkpoint_path}")
    if status.get("last_error"):
        lines.append(f"Last Error: {str(status.get('last_error'))[:180]}")
    return "\n".join(lines)


def _render_batch_status(status: dict, *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(status, indent=2, sort_keys=True, default=str)
    lines = [
        f"Status: {status.get('status', 'unknown')}",
        f"Running: {bool(status.get('running', False))}",
        f"PID: {status.get('pid', 'n/a')}",
        f"Health: {dict(status.get('health', {}) or {}).get('status', 'unknown')} ({dict(status.get('health', {}) or {}).get('reason', 'unknown')})",
        f"Days: {status.get('days', 'n/a')}",
        f"Timeframe: {status.get('timeframe', 'n/a')}",
        f"Trading Mode: {status.get('trading_mode', 'n/a')}",
        f"Use Default Universe: {bool(status.get('use_default_universe', False))}",
        f"Completed Runs: {int(status.get('completed_runs', 0) or 0)}",
        f"Started At: {status.get('started_at', 'n/a')}",
        f"Updated At: {status.get('updated_at', 'n/a')}",
        f"Stop Requested: {bool(status.get('stop_requested', False))}",
    ]
    if status.get("current_run_dir"):
        lines.append(f"Current Run Dir: {status.get('current_run_dir')}")
    progress = dict(status.get("progress", {}) or {})
    if progress:
        lines.extend(
            [
                f"Current Bar: {progress.get('current_bar', 0)}",
                f"Total Bars: {progress.get('total_bars', 0)}",
                f"Trades: {progress.get('num_trades', 0)}",
                f"Balance: {float(progress.get('balance', 0.0) or 0.0):.2f}",
                f"Raw Signals: {progress.get('raw_signals', 0)}",
                f"Filled Orders: {progress.get('filled_orders', 0)}",
            ]
        )
    if status.get("last_run_dir"):
        lines.append(f"Last Run Dir: {status.get('last_run_dir')}")
    if status.get("last_error"):
        lines.append(f"Last Error: {str(status.get('last_error'))[:180]}")
    return "\n".join(lines)


def run_simulation_cli(
    *,
    symbol: str,
    days: int,
    timeframe: str,
    trading_mode: str,
    detach: bool,
    worker_mode: bool,
    as_json: bool,
) -> None:
    base_dir = resolve_runtime_base_dir()
    runtime_dir = simulation_runtime_dir(base_dir)
    if detach and not worker_mode:
        status = spawn_detached_simulation(
            base_dir=base_dir,
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            trading_mode=trading_mode,
        )
        print(_render_simulation_status(status, as_json=as_json))
        return

    clear_stop_request(base_dir)
    config = BotConfig(
        starting_balance=10000.0,
        telegram_bot_token="",
        telegram_chat_id="",
        api_key="",
        api_secret="",
        use_paper_trading=True,
        trading_mode=trading_mode,
    )
    write_simulation_status(
        base_dir,
        {
            "status": "starting",
            "pid": os.getpid(),
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "days": days,
            "trading_mode": trading_mode,
            "mode": "worker" if worker_mode else "foreground",
            "runtime_dir": runtime_dir,
        },
    )
    engine = BacktestEngine(
        config,
        artifact_dir=runtime_dir,
        progress_callback=_simulation_progress_writer(base_dir),
        stop_requested=lambda: simulation_stop_requested(base_dir),
    )
    result = engine.run_backtest(symbol, timeframe, days)
    report_path = persist_simulation_report(base_dir, result)
    clear_stop_request(base_dir)
    final_status = write_simulation_status(
        base_dir,
        {
            "status": "stopped" if result.get("stopped_early") else "completed",
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stop_requested": False,
            "progress": {
                "current_bar": result.get("event_counts", {}).get("market_data_update", 0),
                "total_bars": result.get("event_counts", {}).get("market_data_update", 0),
                "num_trades": result.get("num_trades", 0),
                "balance": result.get("portfolio_snapshot", {}).get("balance", 0.0),
                "equity": result.get("portfolio_snapshot", {}).get("equity", 0.0),
            },
            "result_summary": {
                "num_trades": result.get("num_trades", 0),
                "total_return_pct": result.get("total_return_pct", 0.0),
                "win_rate_pct": result.get("win_rate_pct", 0.0),
                "stopped_early": result.get("stopped_early", False),
                "stop_reason": result.get("stop_reason"),
                "resumed_from_checkpoint": result.get("resumed_from_checkpoint", False),
            },
            "report_path": report_path,
        },
    )
    if not worker_mode:
        print(_render_simulation_status(final_status, as_json=as_json))


def run_simulation_batch_cli(
    *,
    symbol: str,
    days: int,
    timeframe: str,
    trading_mode: str,
    detach: bool,
    worker_mode: bool,
    as_json: bool,
    repeat: int,
    use_default_universe: bool,
) -> None:
    base_dir = resolve_runtime_base_dir()
    batch_dir = simulation_batch_dir(base_dir)
    if detach and not worker_mode:
        status = spawn_detached_simulation_batch(
            base_dir=base_dir,
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            trading_mode=trading_mode,
            repeat=repeat,
            use_default_universe=use_default_universe,
        )
        print(_render_batch_status(status, as_json=as_json))
        return

    clear_simulation_batch_stop(base_dir)
    paths = simulation_batch_paths(base_dir)
    os.makedirs(paths["runs_dir"], exist_ok=True)
    write_simulation_batch_status(
        base_dir,
        {
            "status": "running",
            "pid": os.getpid(),
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "days": days,
            "trading_mode": trading_mode,
            "repeat": repeat,
            "use_default_universe": use_default_universe,
            "mode": "worker" if worker_mode else "foreground",
            "runtime_dir": batch_dir,
            "completed_runs": 0,
        },
    )
    run_index = 0
    config = BotConfig(
        starting_balance=10000.0,
        telegram_bot_token="",
        telegram_chat_id="",
        api_key="",
        api_secret="",
        use_paper_trading=True,
        trading_mode=trading_mode,
    )
    reports: list[dict] = []

    def _batch_progress_writer(run_dir: str):
        def _writer(progress: dict) -> None:
            write_simulation_batch_status(
                base_dir,
                {
                    "status": "running",
                    "updated_at": progress.get("timestamp") or dt.datetime.now(dt.timezone.utc).isoformat(),
                    "current_run_dir": run_dir,
                    "current_run_index": run_index,
                    "progress": progress,
                },
            )

        return _writer

    while repeat <= 0 or run_index < repeat:
        if simulation_batch_stop_requested(base_dir):
            break
        run_index += 1
        run_stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{run_index:04d}_{run_stamp}"
        artifact_dir = os.path.join(paths["runs_dir"], run_name)
        write_simulation_batch_status(
            base_dir,
            {
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "current_run_dir": artifact_dir,
                "current_run_index": run_index,
            },
        )
        engine = BacktestEngine(
            config,
            artifact_dir=artifact_dir,
            progress_callback=_batch_progress_writer(artifact_dir),
            stop_requested=lambda: simulation_batch_stop_requested(base_dir),
        )
        if use_default_universe:
            result = engine.run_campaign(config.symbols, timeframe=timeframe, days=days)
        else:
            result = engine.run_backtest(symbol, timeframe, days)
        with open(os.path.join(artifact_dir, "report.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, default=str)
        reports.append(result)
        summary = build_batch_summary(reports)
        with open(paths["summary"], "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, default=str)
        write_simulation_batch_status(
            base_dir,
            {
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "completed_runs": run_index,
                "last_run_dir": artifact_dir,
                "last_result": {
                    "num_trades": result.get("num_trades", 0),
                    "raw_signals": result.get("raw_signals", 0),
                    "win_rate_pct": result.get("win_rate_pct", 0.0),
                    "total_return_pct": result.get("total_return_pct", 0.0),
                },
            },
        )
        if result.get("stopped_early") or simulation_batch_stop_requested(base_dir):
            break
    final_status = write_simulation_batch_status(
        base_dir,
        {
            "status": "stopped" if simulation_batch_stop_requested(base_dir) else "completed",
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stop_requested": simulation_batch_stop_requested(base_dir),
            "current_run_dir": None,
        },
    )
    if not worker_mode:
        print(_render_batch_status(final_status, as_json=as_json))


def render_simulation_batch_summary(base_dir: str, *, as_json: bool = False) -> str:
    paths = simulation_batch_paths(base_dir)
    reports = load_batch_reports(paths["runs_dir"])
    summary = build_batch_summary(reports)
    if as_json:
        return json.dumps(summary, indent=2, sort_keys=True, default=str)
    if summary.get("num_runs", 0) <= 0:
        return "No completed simulation batch runs found."
    aggregates = dict(summary.get("aggregates", {}) or {})
    lines = [
        f"Runs: {summary.get('num_runs', 0)}",
        f"Avg Trades: {float(aggregates.get('num_trades', {}).get('avg', 0.0) or 0.0):.2f}",
        f"Avg Raw Signals: {float(aggregates.get('raw_signals', {}).get('avg', 0.0) or 0.0):.2f}",
        f"Avg Win Rate: {float(aggregates.get('win_rate_pct', {}).get('avg', 0.0) or 0.0):.2f}%",
        f"Avg Return: {float(aggregates.get('total_return_pct', {}).get('avg', 0.0) or 0.0):.4f}%",
        f"Best Return: {float(aggregates.get('total_return_pct', {}).get('max', 0.0) or 0.0):.4f}%",
        f"Worst Return: {float(aggregates.get('total_return_pct', {}).get('min', 0.0) or 0.0):.4f}%",
    ]
    best_run = dict(summary.get("best_run", {}) or {})
    if best_run:
        lines.append(
            "Best Run: "
            f"{best_run.get('artifact_dir', 'n/a')} "
            f"return={float(best_run.get('total_return_pct', 0.0) or 0.0):.4f}% "
            f"trades={int(best_run.get('num_trades', 0) or 0)} "
            f"win_rate={float(best_run.get('win_rate_pct', 0.0) or 0.0):.2f}%"
        )
    median_run = dict(summary.get("median_run", {}) or {})
    if median_run:
        lines.append(
            "Median Run: "
            f"{median_run.get('artifact_dir', 'n/a')} "
            f"return={float(median_run.get('total_return_pct', 0.0) or 0.0):.4f}% "
            f"trades={int(median_run.get('num_trades', 0) or 0)} "
            f"win_rate={float(median_run.get('win_rate_pct', 0.0) or 0.0):.2f}%"
        )
    baseline_vs_latest = dict((dict(summary.get("comparisons", {}) or {}).get("baseline_vs_latest", {}) or {}))
    if baseline_vs_latest:
        acceptance = dict(baseline_vs_latest.get("acceptance", {}) or {})
        metrics = dict(baseline_vs_latest.get("metrics", {}) or {})
        trades_per_day = dict(metrics.get("trades_per_day", {}) or {})
        total_return = dict(metrics.get("total_return_pct", {}) or {})
        win_rate = dict(metrics.get("win_rate_pct", {}) or {})
        lines.append(
            "Baseline vs Latest: "
            f"trades/day {float(trades_per_day.get('baseline', 0.0) or 0.0):.2f}->{float(trades_per_day.get('candidate', 0.0) or 0.0):.2f}, "
            f"return {float(total_return.get('baseline', 0.0) or 0.0):.4f}%->{float(total_return.get('candidate', 0.0) or 0.0):.4f}%, "
            f"win_rate {float(win_rate.get('baseline', 0.0) or 0.0):.2f}%->{float(win_rate.get('candidate', 0.0) or 0.0):.2f}%"
        )
        lines.append(
            "Comparison Flags: "
            f"passes_all={bool(acceptance.get('passes_all', False))}, "
            f"more_trades={bool(acceptance.get('more_trades', False))}, "
            f"trades_per_day_not_worse={bool(acceptance.get('trades_per_day_not_worse', False))}, "
            f"better_profit={bool(acceptance.get('better_profit', False))}, "
            f"win_rate_not_worse={bool(acceptance.get('win_rate_not_worse', False))}, "
            f"drawdown_not_worse={bool(acceptance.get('drawdown_not_worse', False))}"
        )
    baseline_vs_median = dict((dict(summary.get("comparisons", {}) or {}).get("baseline_vs_median", {}) or {}))
    if baseline_vs_median:
        acceptance = dict(baseline_vs_median.get("acceptance", {}) or {})
        metrics = dict(baseline_vs_median.get("metrics", {}) or {})
        trades_per_day = dict(metrics.get("trades_per_day", {}) or {})
        total_return = dict(metrics.get("total_return_pct", {}) or {})
        win_rate = dict(metrics.get("win_rate_pct", {}) or {})
        lines.append(
            "Baseline vs Median: "
            f"trades/day {float(trades_per_day.get('baseline', 0.0) or 0.0):.2f}->{float(trades_per_day.get('candidate', 0.0) or 0.0):.2f}, "
            f"return {float(total_return.get('baseline', 0.0) or 0.0):.4f}%->{float(total_return.get('candidate', 0.0) or 0.0):.4f}%, "
            f"win_rate {float(win_rate.get('baseline', 0.0) or 0.0):.2f}%->{float(win_rate.get('candidate', 0.0) or 0.0):.2f}%"
        )
        lines.append(
            "Median Flags: "
            f"passes_all={bool(acceptance.get('passes_all', False))}, "
            f"more_trades={bool(acceptance.get('more_trades', False))}, "
            f"trades_per_day_not_worse={bool(acceptance.get('trades_per_day_not_worse', False))}, "
            f"better_profit={bool(acceptance.get('better_profit', False))}, "
            f"win_rate_not_worse={bool(acceptance.get('win_rate_not_worse', False))}, "
            f"drawdown_not_worse={bool(acceptance.get('drawdown_not_worse', False))}"
        )
    walk_forward = dict(summary.get("walk_forward", {}) or {})
    if walk_forward:
        metrics = dict(walk_forward.get("metrics", {}) or {})
        acceptance = dict(walk_forward.get("acceptance", {}) or {})
        trades_per_day = dict(metrics.get("trades_per_day", {}) or {})
        total_return = dict(metrics.get("total_return_pct", {}) or {})
        win_rate = dict(metrics.get("win_rate_pct", {}) or {})
        lines.append(
            "Walk-Forward: "
            f"{walk_forward.get('iteration_horizon', 'n/a')}->{walk_forward.get('confirmation_horizon', 'n/a')}, "
            f"trades/day {float(trades_per_day.get('iteration', 0.0) or 0.0):.2f}->{float(trades_per_day.get('confirmation', 0.0) or 0.0):.2f}, "
            f"return {float(total_return.get('iteration', 0.0) or 0.0):.4f}%->{float(total_return.get('confirmation', 0.0) or 0.0):.4f}%, "
            f"win_rate {float(win_rate.get('iteration', 0.0) or 0.0):.2f}%->{float(win_rate.get('confirmation', 0.0) or 0.0):.2f}%"
        )
        lines.append(
            "Walk-Forward Flags: "
            f"passes_all={bool(acceptance.get('passes_all', False))}, "
            f"trades_per_day_not_worse={bool(acceptance.get('confirmation_trades_per_day_not_worse', False))}, "
            f"return_not_worse={bool(acceptance.get('confirmation_return_not_worse', False))}, "
            f"win_rate_not_worse={bool(acceptance.get('confirmation_win_rate_not_worse', False))}"
        )
    candidate_verdict = dict(summary.get("candidate_verdict", {}) or {})
    if candidate_verdict:
        reasons = list(candidate_verdict.get("reasons", []) or [])
        lines.append(
            "Candidate Verdict: "
            f"{candidate_verdict.get('status', 'unknown')}"
            + (f" ({', '.join(str(reason) for reason in reasons[:4])})" if reasons else "")
        )
    decision_totals = dict(summary.get("decision_totals", {}) or {})
    top_signal_sources = sorted(dict(decision_totals.get("signals_by_strategy", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:3]
    top_skip_reasons = sorted(dict(decision_totals.get("skip_reasons", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_signal_symbols = sorted(dict(decision_totals.get("raw_signals_by_symbol", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_filled_symbols = sorted(dict(decision_totals.get("filled_by_symbol", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_closed_symbols = sorted(dict(decision_totals.get("closed_by_symbol", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_signal_order_types = sorted(dict(decision_totals.get("signals_by_order_type", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_submitted_order_types = sorted(dict(decision_totals.get("submitted_by_order_type", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_closed_order_types = sorted(dict(decision_totals.get("closed_by_order_type", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    execution_totals = dict(summary.get("execution_totals", {}) or {})
    acceptance_totals = dict(summary.get("acceptance_totals", {}) or {})
    stability = dict(summary.get("stability", {}) or {})
    failure_leaderboard = dict(summary.get("failure_leaderboard", {}) or {})
    realized_performance = dict(summary.get("realized_performance", {}) or {})
    exit_quality = dict(summary.get("exit_quality", {}) or {})
    fill_conversion_by_strategy = dict(summary.get("fill_conversion_by_strategy", {}) or {})
    universe_selection = dict(summary.get("universe_selection", {}) or {})
    upgraded_by_strategy = sorted(dict(decision_totals.get("limit_to_market_upgrades_by_strategy", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    queue_priority_by_strategy = sorted(dict(decision_totals.get("limit_queue_priority_assists_by_strategy", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    latency_reduced_by_strategy = sorted(dict(decision_totals.get("limit_latency_reductions_by_strategy", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    stale_escalated_by_strategy = sorted(dict(decision_totals.get("stale_market_escalations_by_strategy", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    repriced_by_strategy = sorted(dict(decision_totals.get("repriced_by_strategy", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    touch_escalated_by_strategy = sorted(dict(decision_totals.get("touch_escalations_by_strategy", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    partial_profit_takes_by_strategy = sorted(dict(decision_totals.get("partial_profit_takes_by_strategy", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_family_rotation_actions = sorted(dict(decision_totals.get("family_rotation_counts", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    skip_reasons_by_symbol = dict(summary.get("skip_reasons_by_symbol", {}) or {})
    family_rotation_by_strategy = dict(summary.get("family_rotation_by_strategy", {}) or {})
    family_rotation_soft_by_strategy = dict(summary.get("family_rotation_soft_by_strategy", {}) or {})
    family_rotation_hard_by_strategy = dict(summary.get("family_rotation_hard_by_strategy", {}) or {})
    family_rotation_recovery_by_strategy = dict(summary.get("family_rotation_recovery_by_strategy", {}) or {})
    learning_evidence_by_strategy = dict(summary.get("learning_evidence_by_strategy", {}) or {})
    learning_asymmetry_by_strategy = dict(summary.get("learning_asymmetry_by_strategy", {}) or {})
    missed_opportunity_relaxations_by_strategy = dict(summary.get("missed_opportunity_relaxations_by_strategy", {}) or {})
    duplicate_bucket_throttle_by_strategy = dict(summary.get("duplicate_bucket_throttle_by_strategy", {}) or {})
    weak_cluster_throttle_by_strategy = dict(summary.get("weak_cluster_throttle_by_strategy", {}) or {})
    realized_penalty_by_strategy = dict(summary.get("realized_performance_penalty_by_strategy", {}) or {})
    realized_no_trade_by_strategy = dict(summary.get("realized_performance_no_trade_by_strategy", {}) or {})
    top_symbol_skip_reasons = []
    for symbol, reasons in skip_reasons_by_symbol.items():
        for reason, value in dict(reasons or {}).items():
            top_symbol_skip_reasons.append((f"{symbol}:{reason}", int(value or 0)))
    top_symbol_skip_reasons.sort(key=lambda item: item[1], reverse=True)
    top_symbol_skip_reasons = top_symbol_skip_reasons[:5]
    top_symbol_expectancy = sorted(
        dict(realized_performance.get("by_symbol", {}) or {}).items(),
        key=lambda item: float(dict(item[1] or {}).get("expectancy", 0.0) or 0.0),
    )[:5]
    top_strategy_expectancy = sorted(
        dict(realized_performance.get("by_strategy", {}) or {}).items(),
        key=lambda item: float(dict(item[1] or {}).get("expectancy", 0.0) or 0.0),
    )
    top_exit_reason_expectancy = sorted(
        dict(exit_quality.get("by_exit_reason", {}) or {}).items(),
        key=lambda item: float(dict(item[1] or {}).get("trades", 0.0) or 0.0),
        reverse=True,
    )[:5]
    top_giveback_by_strategy = sorted(
        dict(exit_quality.get("giveback_by_strategy", {}) or {}).items(),
        key=lambda item: float(dict(item[1] or {}).get("avg_giveback_r", 0.0) or 0.0),
        reverse=True,
    )[:5]
    top_fill_conversion_by_strategy = sorted(
        dict(fill_conversion_by_strategy or {}).items(),
        key=lambda item: int(dict(item[1] or {}).get("submitted", 0) or 0),
        reverse=True,
    )[:5]
    top_strategy_rotation = []
    for strategy, reasons in family_rotation_by_strategy.items():
        for reason, value in dict(reasons or {}).items():
            top_strategy_rotation.append((f"{strategy}:{reason}", int(value or 0)))
    top_strategy_rotation.sort(key=lambda item: item[1], reverse=True)
    top_strategy_rotation = top_strategy_rotation[:5]
    top_soft_strategy_rotation = []
    for strategy, reasons in family_rotation_soft_by_strategy.items():
        for reason, value in dict(reasons or {}).items():
            top_soft_strategy_rotation.append((f"{strategy}:{reason}", int(value or 0)))
    top_soft_strategy_rotation.sort(key=lambda item: item[1], reverse=True)
    top_soft_strategy_rotation = top_soft_strategy_rotation[:5]
    top_hard_strategy_rotation = []
    for strategy, reasons in family_rotation_hard_by_strategy.items():
        for reason, value in dict(reasons or {}).items():
            top_hard_strategy_rotation.append((f"{strategy}:{reason}", int(value or 0)))
    top_hard_strategy_rotation.sort(key=lambda item: item[1], reverse=True)
    top_hard_strategy_rotation = top_hard_strategy_rotation[:5]
    top_recovery_strategy_rotation = []
    for strategy, reasons in family_rotation_recovery_by_strategy.items():
        for reason, value in dict(reasons or {}).items():
            top_recovery_strategy_rotation.append((f"{strategy}:{reason}", int(value or 0)))
    top_recovery_strategy_rotation.sort(key=lambda item: item[1], reverse=True)
    top_recovery_strategy_rotation = top_recovery_strategy_rotation[:5]
    top_learning_evidence = []
    for strategy, reasons in learning_evidence_by_strategy.items():
        for reason, value in dict(reasons or {}).items():
            top_learning_evidence.append((f"{strategy}:{reason}", int(value or 0)))
    top_learning_evidence.sort(key=lambda item: item[1], reverse=True)
    top_learning_evidence = top_learning_evidence[:5]
    top_learning_asymmetry = []
    for strategy, reasons in learning_asymmetry_by_strategy.items():
        for reason, value in dict(reasons or {}).items():
            top_learning_asymmetry.append((f"{strategy}:{reason}", int(value or 0)))
    top_learning_asymmetry.sort(key=lambda item: item[1], reverse=True)
    top_learning_asymmetry = top_learning_asymmetry[:5]
    top_missed_opportunity_relaxations = []
    for strategy, reasons in missed_opportunity_relaxations_by_strategy.items():
        for reason, value in dict(reasons or {}).items():
            top_missed_opportunity_relaxations.append((f"{strategy}:{reason}", int(value or 0)))
    top_missed_opportunity_relaxations.sort(key=lambda item: item[1], reverse=True)
    top_missed_opportunity_relaxations = top_missed_opportunity_relaxations[:5]
    reentry_cooldown_by_strategy = dict(summary.get("reentry_cooldown_registrations_by_strategy", {}) or {})
    top_reentry_cooldowns = []
    for strategy, reasons in reentry_cooldown_by_strategy.items():
        for reason, value in dict(reasons or {}).items():
            top_reentry_cooldowns.append((f"{strategy}:{reason}", int(value or 0)))
    top_reentry_cooldowns.sort(key=lambda item: item[1], reverse=True)
    top_reentry_cooldowns = top_reentry_cooldowns[:5]
    top_realized_penalties = []
    for strategy, reasons in realized_penalty_by_strategy.items():
        for reason, value in dict(reasons or {}).items():
            top_realized_penalties.append((f"{strategy}:{reason}", int(value or 0)))
    top_realized_penalties.sort(key=lambda item: item[1], reverse=True)
    top_realized_penalties = top_realized_penalties[:5]
    top_realized_no_trade = []
    for strategy, reasons in realized_no_trade_by_strategy.items():
        for reason, value in dict(reasons or {}).items():
            top_realized_no_trade.append((f"{strategy}:{reason}", int(value or 0)))
    top_realized_no_trade.sort(key=lambda item: item[1], reverse=True)
    top_realized_no_trade = top_realized_no_trade[:5]
    top_duplicate_bucket_throttles = sorted(dict(duplicate_bucket_throttle_by_strategy or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_weak_cluster_throttles = sorted(dict(weak_cluster_throttle_by_strategy or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_acceptance = sorted(dict(acceptance_totals or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_failure_skips = sorted(dict(failure_leaderboard.get("top_skip_reasons", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:3]
    top_losing_families = sorted(
        dict(failure_leaderboard.get("losing_families", {}) or {}).items(),
        key=lambda item: float(dict(item[1] or {}).get("expectancy", 0.0) or 0.0),
    )[:3]
    top_losing_symbols = sorted(
        dict(failure_leaderboard.get("losing_symbols", {}) or {}).items(),
        key=lambda item: float(dict(item[1] or {}).get("expectancy", 0.0) or 0.0),
    )[:3]
    top_edge_divergence = sorted(
        dict(failure_leaderboard.get("expected_vs_realized_edge_divergence", {}) or {}).items(),
        key=lambda item: abs(float(dict(item[1] or {}).get("realized_expectancy", 0.0) or 0.0)),
        reverse=True,
    )[:3]
    top_eligible_symbols = sorted(dict(universe_selection.get("eligible_symbols", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_rejected_symbols = sorted(dict(universe_selection.get("rejected_symbols", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_eligible_buckets = sorted(dict(universe_selection.get("eligible_buckets", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_rejected_buckets = sorted(dict(universe_selection.get("rejected_buckets", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_bucket_cap_rejections = sorted(dict(universe_selection.get("bucket_cap_rejections", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_bucket_cap_rejections_by_bucket = sorted(dict(universe_selection.get("bucket_cap_rejections_by_bucket", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_eligible_bucket_pressure = sorted(dict(universe_selection.get("eligible_bucket_pressure", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_realized_universe_promotions = sorted(dict(universe_selection.get("realized_universe_promotions", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_realized_universe_penalties = sorted(dict(universe_selection.get("realized_universe_penalties", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_realized_universe_adjustments_by_bucket = sorted(dict(universe_selection.get("realized_universe_adjustments_by_bucket", {}) or {}).items(), key=lambda item: abs(item[1]), reverse=True)[:5]
    top_realized_universe_vetoes = sorted(dict(universe_selection.get("realized_universe_vetoes", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    top_realized_universe_vetoes_by_bucket = sorted(dict(universe_selection.get("realized_universe_vetoes_by_bucket", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:5]
    if top_signal_sources:
        lines.append("Signals By Strategy: " + ", ".join(f"{key}={value}" for key, value in top_signal_sources))
    if top_skip_reasons:
        lines.append("Top Skip Reasons: " + ", ".join(f"{key}={value}" for key, value in top_skip_reasons))
    if top_signal_symbols:
        lines.append("Signals By Symbol: " + ", ".join(f"{key}={value}" for key, value in top_signal_symbols))
    if top_filled_symbols:
        lines.append("Filled By Symbol: " + ", ".join(f"{key}={value}" for key, value in top_filled_symbols))
    if top_closed_symbols:
        lines.append("Closed By Symbol: " + ", ".join(f"{key}={value}" for key, value in top_closed_symbols))
    if top_signal_order_types:
        lines.append("Signals By Order Type: " + ", ".join(f"{key}={value}" for key, value in top_signal_order_types))
    if top_submitted_order_types:
        lines.append("Submitted By Order Type: " + ", ".join(f"{key}={value}" for key, value in top_submitted_order_types))
    if top_closed_order_types:
        lines.append("Closed By Order Type: " + ", ".join(f"{key}={value}" for key, value in top_closed_order_types))
    if top_fill_conversion_by_strategy:
        lines.append(
            "Fill Conversion By Strategy: "
            + ", ".join(
                f"{strategy}=signals:{int(dict(payload or {}).get('signals', 0) or 0)}/submitted:{int(dict(payload or {}).get('submitted', 0) or 0)}/filled:{int(dict(payload or {}).get('filled', 0) or 0)}/closed:{int(dict(payload or {}).get('closed', 0) or 0)}"
                for strategy, payload in top_fill_conversion_by_strategy
            )
        )
    if execution_totals:
        lines.append(
            "Execution Adjustments: "
            f"limit_to_market_upgrades={int(execution_totals.get('limit_to_market_upgrades', 0) or 0)}, "
            f"limit_queue_priority_assists={int(execution_totals.get('limit_queue_priority_assists', 0) or 0)}, "
            f"limit_latency_reductions={int(execution_totals.get('limit_latency_reductions', 0) or 0)}, "
            f"stale_market_escalations={int(execution_totals.get('stale_market_escalations', 0) or 0)}, "
            f"repriced_orders={int(execution_totals.get('repriced_orders', 0) or 0)}, "
            f"touch_escalations={int(execution_totals.get('touch_escalations', 0) or 0)}, "
            f"partial_profit_takes={int(execution_totals.get('partial_profit_takes', 0) or 0)}"
        )
    if top_acceptance:
        lines.append("Acceptance Totals: " + ", ".join(f"{key}={value}" for key, value in top_acceptance))
    if stability:
        lines.append(
            "Stability: "
            f"return_range={float(stability.get('return_range_pct', 0.0) or 0.0):.4f}%, "
            f"return_stddev={float(stability.get('return_stddev_pct', 0.0) or 0.0):.4f}%, "
            f"trade_stddev={float(stability.get('trade_stddev', 0.0) or 0.0):.2f}, "
            f"win_rate_stddev={float(stability.get('win_rate_stddev_pct', 0.0) or 0.0):.2f}%, "
            f"best_vs_median={float(stability.get('best_vs_median_return_gap_pct', 0.0) or 0.0):.4f}%"
        )
    if top_failure_skips:
        lines.append("Failure Skip Leaders: " + ", ".join(f"{key}={value}" for key, value in top_failure_skips))
    if top_losing_families:
        lines.append(
            "Losing Families: "
            + ", ".join(
                f"{strategy}={float(dict(payload or {}).get('expectancy', 0.0) or 0.0):.4f}"
                for strategy, payload in top_losing_families
            )
        )
    if top_losing_symbols:
        lines.append(
            "Losing Symbols: "
            + ", ".join(
                f"{symbol}={float(dict(payload or {}).get('expectancy', 0.0) or 0.0):.4f}"
                for symbol, payload in top_losing_symbols
            )
        )
    if top_edge_divergence:
        lines.append(
            "Expected Vs Realized Edge: "
            + ", ".join(
                f"{symbol}=edge_bps:{float(dict(payload or {}).get('avg_edge_bps', 0.0) or 0.0):.2f}/realized:{float(dict(payload or {}).get('realized_expectancy', 0.0) or 0.0):.4f}"
                for symbol, payload in top_edge_divergence
            )
        )
    if top_eligible_symbols:
        lines.append("Eligible Universe Symbols: " + ", ".join(f"{key}={value}" for key, value in top_eligible_symbols))
    if top_rejected_symbols:
        lines.append("Rejected Universe Symbols: " + ", ".join(f"{key}={value}" for key, value in top_rejected_symbols))
    if top_eligible_buckets:
        lines.append("Eligible Universe Buckets: " + ", ".join(f"{key}={value}" for key, value in top_eligible_buckets))
    if top_rejected_buckets:
        lines.append("Rejected Universe Buckets: " + ", ".join(f"{key}={value}" for key, value in top_rejected_buckets))
    if top_bucket_cap_rejections:
        lines.append("Bucket Cap Pressure Symbols: " + ", ".join(f"{key}={value}" for key, value in top_bucket_cap_rejections))
    if top_bucket_cap_rejections_by_bucket:
        lines.append("Bucket Cap Pressure By Bucket: " + ", ".join(f"{key}={value}" for key, value in top_bucket_cap_rejections_by_bucket))
    if top_eligible_bucket_pressure:
        lines.append("Eligible Bucket Pressure: " + ", ".join(f"{key}={value}" for key, value in top_eligible_bucket_pressure))
    if top_realized_universe_promotions:
        lines.append("Realized Universe Promotions: " + ", ".join(f"{key}={value}" for key, value in top_realized_universe_promotions))
    if top_realized_universe_penalties:
        lines.append("Realized Universe Penalties: " + ", ".join(f"{key}={value}" for key, value in top_realized_universe_penalties))
    if top_realized_universe_adjustments_by_bucket:
        lines.append("Realized Universe By Bucket: " + ", ".join(f"{key}={value}" for key, value in top_realized_universe_adjustments_by_bucket))
    if top_realized_universe_vetoes:
        lines.append("Realized Universe Vetoes: " + ", ".join(f"{key}={value}" for key, value in top_realized_universe_vetoes))
    if top_realized_universe_vetoes_by_bucket:
        lines.append("Realized Universe Vetoes By Bucket: " + ", ".join(f"{key}={value}" for key, value in top_realized_universe_vetoes_by_bucket))
    horizon_summaries = dict(summary.get("by_horizon", {}) or {})
    for label, horizon_summary in sorted(horizon_summaries.items()):
        horizon_aggregates = dict(horizon_summary.get("aggregates", {}) or {})
        horizon_verdict = dict(horizon_summary.get("candidate_verdict", {}) or {})
        lines.append(
            f"Horizon {label}: "
            f"runs={int(horizon_summary.get('num_runs', 0) or 0)}, "
            f"avg_return={float(dict(horizon_aggregates.get('total_return_pct', {}) or {}).get('avg', 0.0) or 0.0):.4f}%, "
            f"avg_trades={float(dict(horizon_aggregates.get('num_trades', {}) or {}).get('avg', 0.0) or 0.0):.2f}, "
            f"avg_win_rate={float(dict(horizon_aggregates.get('win_rate_pct', {}) or {}).get('avg', 0.0) or 0.0):.2f}%, "
            f"verdict={str(horizon_verdict.get('status', 'unknown'))}"
        )
    if upgraded_by_strategy:
        lines.append("Limit-To-Market By Strategy: " + ", ".join(f"{key}={value}" for key, value in upgraded_by_strategy))
    if queue_priority_by_strategy:
        lines.append("Queue-Priority Assists By Strategy: " + ", ".join(f"{key}={value}" for key, value in queue_priority_by_strategy))
    if latency_reduced_by_strategy:
        lines.append("Latency-Reduced By Strategy: " + ", ".join(f"{key}={value}" for key, value in latency_reduced_by_strategy))
    if stale_escalated_by_strategy:
        lines.append("Stale-Escalated By Strategy: " + ", ".join(f"{key}={value}" for key, value in stale_escalated_by_strategy))
    if repriced_by_strategy:
        lines.append("Repriced By Strategy: " + ", ".join(f"{key}={value}" for key, value in repriced_by_strategy))
    if touch_escalated_by_strategy:
        lines.append("Touch-Escalated By Strategy: " + ", ".join(f"{key}={value}" for key, value in touch_escalated_by_strategy))
    if partial_profit_takes_by_strategy:
        lines.append("Partial Profit Takes By Strategy: " + ", ".join(f"{key}={value}" for key, value in partial_profit_takes_by_strategy))
    if top_family_rotation_actions:
        lines.append("Family Rotation Actions: " + ", ".join(f"{key}={value}" for key, value in top_family_rotation_actions))
    if top_strategy_rotation:
        lines.append("Family Rotation By Strategy: " + ", ".join(f"{key}={value}" for key, value in top_strategy_rotation))
    if top_soft_strategy_rotation:
        lines.append("Soft Family Rotation By Strategy: " + ", ".join(f"{key}={value}" for key, value in top_soft_strategy_rotation))
    if top_hard_strategy_rotation:
        lines.append("Hard Family Rotation By Strategy: " + ", ".join(f"{key}={value}" for key, value in top_hard_strategy_rotation))
    if top_recovery_strategy_rotation:
        lines.append("Recovery Family Rotation By Strategy: " + ", ".join(f"{key}={value}" for key, value in top_recovery_strategy_rotation))
    if top_learning_evidence:
        lines.append("Learning Evidence By Strategy: " + ", ".join(f"{key}={value}" for key, value in top_learning_evidence))
    if top_learning_asymmetry:
        lines.append("Learning Promotions/Throttles: " + ", ".join(f"{key}={value}" for key, value in top_learning_asymmetry))
    if top_missed_opportunity_relaxations:
        lines.append("Missed Opportunity Relaxations: " + ", ".join(f"{key}={value}" for key, value in top_missed_opportunity_relaxations))
    if top_reentry_cooldowns:
        lines.append("Re-Entry Cooldowns: " + ", ".join(f"{key}={value}" for key, value in top_reentry_cooldowns))
    if top_symbol_skip_reasons:
        lines.append("Top Symbol Skip Reasons: " + ", ".join(f"{key}={value}" for key, value in top_symbol_skip_reasons))
    if top_strategy_expectancy:
        lines.append(
            "Realized Expectancy By Strategy: "
            + ", ".join(
                f"{strategy}={float(dict(payload or {}).get('expectancy', 0.0) or 0.0):.4f}"
                for strategy, payload in top_strategy_expectancy[:5]
            )
        )
    if top_symbol_expectancy:
        lines.append(
            "Worst Symbols By Expectancy: "
            + ", ".join(
                f"{symbol}={float(dict(payload or {}).get('expectancy', 0.0) or 0.0):.4f}"
                for symbol, payload in top_symbol_expectancy
            )
        )
    if top_realized_penalties:
        lines.append("Realized Perf Penalties: " + ", ".join(f"{key}={value}" for key, value in top_realized_penalties))
    if top_realized_no_trade:
        lines.append("Realized Perf No-Trade Blocks: " + ", ".join(f"{key}={value}" for key, value in top_realized_no_trade))
    if top_duplicate_bucket_throttles:
        lines.append("Duplicate Bucket Throttles: " + ", ".join(f"{key}={value}" for key, value in top_duplicate_bucket_throttles))
    if top_weak_cluster_throttles:
        lines.append("Weak Cluster Throttles: " + ", ".join(f"{key}={value}" for key, value in top_weak_cluster_throttles))
    if top_exit_reason_expectancy:
        lines.append(
            "Exit Expectancy By Reason: "
            + ", ".join(
                f"{reason}={float(dict(payload or {}).get('expectancy', 0.0) or 0.0):.4f}"
                for reason, payload in top_exit_reason_expectancy
            )
        )
    if top_giveback_by_strategy:
        lines.append(
            "Winner Giveback By Strategy: "
            + ", ".join(
                f"{strategy}={float(dict(payload or {}).get('avg_giveback_r', 0.0) or 0.0):.2f}R"
                for strategy, payload in top_giveback_by_strategy
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Trading Bot: Live | Backtest | Hyperopt | Status | Healthcheck")
    parser.add_argument(
        "mode",
        choices=["live", "backtest", "hyperopt", "status", "healthcheck", "simulate", "simulation-status", "simulation-stop", "simulate-worker", "simulate-batch", "simulate-batch-worker", "simulation-batch-status", "simulation-batch-stop", "simulation-batch-summary", "telegram-operator"],
        help="Runtime mode",
    )
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=90, help="Backtest/hyperopt days")
    parser.add_argument("--timeframe", default="15m", help="Chart timeframe")
    parser.add_argument("--combinations", type=int, help="Max hyperopt combinations")
    parser.add_argument("--paper", action="store_true", help="Run in paper mode instead of Binance testnet")
    parser.add_argument("--trading-mode", choices=["spot", "futures", "mixed"], default="spot", help="Execution venue for live mode")
    parser.add_argument("--max-snapshot-age-seconds", type=int, default=900, help="Healthcheck staleness threshold")
    parser.add_argument("--json", action="store_true", help="Render machine-readable JSON output where supported")
    parser.add_argument("--detach", action="store_true", help="Run simulation as a detached background worker")
    parser.add_argument("--repeat", type=int, default=0, help="Number of repeated simulations for batch mode; 0 means run until stopped")
    parser.add_argument("--use-default-universe", action="store_true", help="Use the config default multi-symbol universe instead of --symbol")
    return parser


def build_runtime_config(*, paper_mode: bool, trading_mode: str) -> BotConfig:
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not paper_mode and (not api_key or not api_secret):
        raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET are required for Binance testnet live mode")

    return BotConfig(
        starting_balance=float(os.getenv("BOT_STARTING_BALANCE", "50.0")),
        use_paper_trading=paper_mode,
        trading_mode=trading_mode,
        operating_mode=os.getenv("BOT_OPERATING_MODE", "paper"),
        news_engine_enabled=getenv_bool("BOT_NEWS_ENGINE_ENABLED", True),
        news_poll_interval_minutes=int(os.getenv("BOT_NEWS_POLL_INTERVAL_MINUTES", "15")),
        use_testnet_public=getenv_bool("BOT_USE_TESTNET_PUBLIC", True),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        api_key=api_key,
        api_secret=api_secret,
    )


def main(argv: list[str] | None = None) -> None:
    load_runtime_environment(resolve_runtime_base_dir())
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "backtest":
        run_backtest_cli(args.symbol, args.days, args.timeframe)
        return

    if args.mode == "hyperopt":
        hyperopt = HyperoptEngine()
        hyperopt.optimize(args.symbol, args.days, args.combinations)
        return

    if args.mode == "simulate":
        run_simulation_cli(
            symbol=args.symbol,
            days=args.days,
            timeframe=args.timeframe,
            trading_mode=args.trading_mode,
            detach=bool(args.detach),
            worker_mode=False,
            as_json=bool(args.json),
        )
        return

    if args.mode == "simulate-worker":
        run_simulation_cli(
            symbol=args.symbol,
            days=args.days,
            timeframe=args.timeframe,
            trading_mode=args.trading_mode,
            detach=False,
            worker_mode=True,
            as_json=bool(args.json),
        )
        return

    if args.mode == "simulate-batch":
        run_simulation_batch_cli(
            symbol=args.symbol,
            days=args.days,
            timeframe=args.timeframe,
            trading_mode=args.trading_mode,
            detach=bool(args.detach),
            worker_mode=False,
            as_json=bool(args.json),
            repeat=int(args.repeat),
            use_default_universe=bool(args.use_default_universe),
        )
        return

    if args.mode == "simulate-batch-worker":
        run_simulation_batch_cli(
            symbol=args.symbol,
            days=args.days,
            timeframe=args.timeframe,
            trading_mode=args.trading_mode,
            detach=False,
            worker_mode=True,
            as_json=bool(args.json),
            repeat=int(args.repeat),
            use_default_universe=bool(args.use_default_universe),
        )
        return

    if args.mode == "simulation-status":
        print(_render_simulation_status(read_simulation_status(resolve_runtime_base_dir()), as_json=args.json))
        return

    if args.mode == "simulation-stop":
        print(_render_simulation_status(request_simulation_stop(resolve_runtime_base_dir()), as_json=args.json))
        return

    if args.mode == "simulation-batch-status":
        print(_render_batch_status(read_simulation_batch_status(resolve_runtime_base_dir()), as_json=args.json))
        return

    if args.mode == "simulation-batch-stop":
        print(_render_batch_status(request_simulation_batch_stop(resolve_runtime_base_dir()), as_json=args.json))
        return

    if args.mode == "simulation-batch-summary":
        print(render_simulation_batch_summary(resolve_runtime_base_dir(), as_json=args.json))
        return

    if args.mode == "status":
        config = build_runtime_config(paper_mode=True, trading_mode=args.trading_mode)
        config.validate()
        bot = TradeBot(config, enable_metrics=False)
        print(bot.render_status_report())
        return

    if args.mode == "telegram-operator":
        config = build_runtime_config(paper_mode=True, trading_mode=args.trading_mode)
        run_telegram_operator(config)
        return

    if args.mode == "healthcheck":
        health = evaluate_runtime_health(
            resolve_runtime_base_dir(),
            max_snapshot_age_seconds=max(int(args.max_snapshot_age_seconds), 1),
        )
        print(render_runtime_health(health, as_json=args.json))
        raise SystemExit(0 if health.healthy else 1)

    config = build_runtime_config(paper_mode=args.paper, trading_mode=args.trading_mode)
    config.validate()
    bot = TradeBot(config)
    bot.run_forever(int(os.getenv("BOT_LOOP_SLEEP_SECONDS", "60")))


if __name__ == "__main__":
    main()
