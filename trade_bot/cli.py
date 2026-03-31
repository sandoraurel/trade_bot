from __future__ import annotations

import argparse
import os
import sys

from .bootstrap import getenv_bool, load_runtime_environment, resolve_runtime_base_dir
from .config import BotConfig
from .health import evaluate_runtime_health, render_runtime_health
from .main import BacktestEngine, HyperoptEngine, TradeBot
from .backtest_reporting import build_backtest_report, save_backtest_report


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
            "execution_model": "signal_on_close_enter_next_open_with_costs",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Trading Bot: Live | Backtest | Hyperopt | Status | Healthcheck")
    parser.add_argument("mode", choices=["live", "backtest", "hyperopt", "status", "healthcheck"], help="Runtime mode")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=90, help="Backtest/hyperopt days")
    parser.add_argument("--timeframe", default="15m", help="Chart timeframe")
    parser.add_argument("--combinations", type=int, help="Max hyperopt combinations")
    parser.add_argument("--paper", action="store_true", help="Run in paper mode instead of Binance testnet")
    parser.add_argument("--trading-mode", choices=["spot", "futures", "mixed"], default="spot", help="Execution venue for live mode")
    parser.add_argument("--max-snapshot-age-seconds", type=int, default=900, help="Healthcheck staleness threshold")
    parser.add_argument("--json", action="store_true", help="Render machine-readable JSON output where supported")
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

    if args.mode == "status":
        config = build_runtime_config(paper_mode=True, trading_mode=args.trading_mode)
        config.validate()
        bot = TradeBot(config, enable_metrics=False)
        print(bot.render_status_report())
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
