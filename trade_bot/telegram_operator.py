from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import time
from typing import Any, Dict

import requests

from .bootstrap import ensure_runtime_directories, resolve_runtime_base_dir
from .constants import LEARNING_DB_FILE, STATE_DB_FILE, STATE_FILE
from .backtest_reporting import load_batch_reports, build_batch_summary
from .process_service import read_bot_status, spawn_detached_bot, stop_bot_process
from .simulation_service import (
    read_simulation_batch_status,
    read_simulation_status,
    request_simulation_batch_stop,
    request_simulation_stop,
    simulation_batch_paths,
    spawn_detached_simulation_batch,
)

TELEGRAM_OPERATOR_STATE = "data/telegram_operator_state.json"


def _state_path(base_dir: str) -> str:
    return os.path.join(base_dir, TELEGRAM_OPERATOR_STATE)


def load_operator_state(base_dir: str) -> Dict[str, Any]:
    path = _state_path(base_dir)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_operator_state(base_dir: str, state: Dict[str, Any]) -> None:
    ensure_runtime_directories(base_dir)
    path = _state_path(base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True, default=str)


def _telegram_api(config: Any, method: str) -> str:
    return f"https://api.telegram.org/bot{config.telegram_bot_token}/{method}"


def _send_message(config: Any, message: str) -> None:
    if not config.telegram_bot_token or not config.telegram_chat_id:
        return
    requests.post(
        _telegram_api(config, "sendMessage"),
        json={"chat_id": config.telegram_chat_id, "text": message},
        timeout=10,
    )


def _supported_commands() -> list[Dict[str, str]]:
    return [
        {"command": "startbot", "description": "Start the live bot"},
        {"command": "stopbot", "description": "Stop the live bot"},
        {"command": "startsimulation", "description": "Start the simulation batch"},
        {"command": "stopsimulation", "description": "Stop the simulation batch"},
        {"command": "botstatus", "description": "Show live bot status"},
        {"command": "simstatus", "description": "Show simulation status"},
        {"command": "botlogs", "description": "Show short live bot logs"},
        {"command": "simlogs", "description": "Show short simulation logs"},
    ]


def _register_supported_commands(config: Any) -> None:
    if not config.telegram_bot_token:
        return
    requests.post(
        _telegram_api(config, "setMyCommands"),
        json={"commands": _supported_commands()},
        timeout=10,
    )


def _get_updates(config: Any, offset: int | None = None) -> list[Dict[str, Any]]:
    payload: Dict[str, Any] = {"timeout": 25}
    if offset is not None:
        payload["offset"] = offset
    response = requests.get(_telegram_api(config, "getUpdates"), params=payload, timeout=35)
    response.raise_for_status()
    data = response.json()
    return list(data.get("result", []) or [])


def _chat_allowed(config: Any, update: Dict[str, Any]) -> bool:
    expected = str(config.telegram_chat_id or "").strip()
    actual = str((((update.get("message") or {}).get("chat") or {}).get("id")) or "").strip()
    return bool(expected and actual and expected == actual)


def _record_unauthorized_update(operator_state: Dict[str, Any], update: Dict[str, Any]) -> None:
    message = dict(update.get("message", {}) or {})
    chat = dict(message.get("chat", {}) or {})
    actual = str(chat.get("id", "") or "").strip() or "unknown"
    unauthorized = dict(operator_state.get("unauthorized_chats", {}) or {})
    unauthorized[actual] = int(unauthorized.get(actual, 0)) + 1
    operator_state["unauthorized_chats"] = dict(sorted(unauthorized.items()))
    operator_state["last_unauthorized_update_id"] = int(update.get("update_id", 0) or 0)


def _record_command_audit(operator_state: Dict[str, Any], *, command: str, response: str) -> None:
    history = list(operator_state.get("command_audit", []) or [])
    history.append(
        {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": str(command or ""),
            "response": str(response or "")[:200],
        }
    )
    operator_state["command_audit"] = history[-50:]


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_bot_state(base_dir: str) -> Dict[str, Any]:
    return _load_json(os.path.join(base_dir, STATE_FILE))


def _operational_metrics(base_dir: str) -> Dict[str, Any]:
    path = os.path.join(base_dir, STATE_DB_FILE)
    if not os.path.exists(path):
        return {}
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute("SELECT metric_key, metric_value FROM operational_metrics").fetchall()
    finally:
        conn.close()
    result: Dict[str, Any] = {}
    for key, value in rows:
        try:
            result[key] = json.loads(value)
        except Exception:
            result[key] = value
    return result


def _fills_since(base_dir: str, since_iso: str) -> int:
    path = os.path.join(base_dir, STATE_DB_FILE)
    if not os.path.exists(path):
        return 0
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM bot_events WHERE event_type = ? AND created_at >= ?",
            ("fill_received", since_iso),
        ).fetchone()
    finally:
        conn.close()
    return int(row[0] if row else 0)


def _parse_iso(value: str | None) -> dt.datetime:
    if not value:
        return dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=24)
    parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _format_bot_logs(base_dir: str, operator_state: Dict[str, Any]) -> str:
    bot_state = _load_bot_state(base_dir)
    metrics = _operational_metrics(base_dir)
    status = read_bot_status(base_dir)
    now = dt.datetime.now(dt.timezone.utc)
    since = _parse_iso(operator_state.get("last_bot_logs_check"))
    if (now - since) > dt.timedelta(hours=24):
        since = now - dt.timedelta(hours=24)
    fills = _fills_since(base_dir, since.isoformat())
    starting_balance = float(bot_state.get("equity_start_of_day", bot_state.get("balance", 0.0)) or 0.0)
    current_balance = float(bot_state.get("balance", 0.0) or 0.0)
    realized_pl = float(bot_state.get("realized_pl_today", 0.0) or 0.0)
    after_balance = starting_balance + realized_pl
    profit_pct = ((after_balance - starting_balance) / starting_balance * 100.0) if starting_balance > 0 else 0.0
    win_rate = float(metrics.get("win_rate_pct", 0.0) or 0.0)
    last_error = str(status.get("last_error", "") or "")
    operator_state["last_bot_logs_check"] = now.isoformat()
    lines = [
        f"Bot: {'running' if bool(status.get('running', False)) else 'stopped'}",
        f"PID: {status.get('pid', 'n/a')}",
        f"Status: {status.get('status', 'unknown')}",
        f"Health: {dict(status.get('health', {}) or {}).get('status', 'unknown')} ({dict(status.get('health', {}) or {}).get('reason', 'unknown')})",
        f"Trades since check: {fills}",
        f"Balance: {current_balance:.2f}",
        f"Realized PnL: {realized_pl:.2f} ({profit_pct:.4f}%)",
        f"Win rate: {win_rate:.2f}%",
    ]
    updated_at = str(status.get("updated_at", "") or "")
    if updated_at:
        lines.append(f"Last status update: {updated_at}")
    if last_error:
        lines.append(f"Last warning/error: {last_error[:180]}")
    return "\n".join(lines)


def _format_sim_logs(base_dir: str, operator_state: Dict[str, Any]) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    since = _parse_iso(operator_state.get("last_sim_logs_check"))
    if (now - since) > dt.timedelta(hours=24):
        since = now - dt.timedelta(hours=24)
    batch_paths = simulation_batch_paths(base_dir)
    reports = load_batch_reports(batch_paths["runs_dir"])
    recent_reports = []
    for report in reports:
        artifact_dir = str(report.get("artifact_dir", ""))
        run_name = os.path.basename(artifact_dir)
        try:
            run_time = dt.datetime.strptime(run_name.split("_", 2)[-1], "%Y%m%d_%H%M%S").replace(tzinfo=dt.timezone.utc)
        except Exception:
            run_time = now
        if run_time >= since:
            recent_reports.append(report)
    summary = build_batch_summary(recent_reports)
    operator_state["last_sim_logs_check"] = now.isoformat()
    batch_status = read_simulation_batch_status(base_dir)
    last_error = str(batch_status.get("last_error", "") or "")
    if summary.get("num_runs", 0) <= 0:
        current = _load_json(os.path.join(base_dir, "data", "simulation", "report.json"))
        if not current:
            lines = [
                f"Simulation batch: {'running' if bool(batch_status.get('running', False)) else 'stopped'}",
                f"PID: {batch_status.get('pid', 'n/a')}",
                "No completed simulation runs found.",
            ]
            if last_error:
                lines.append(f"Last warning/error: {last_error[:180]}")
            return "\n".join(lines)
        starting_balance = 10000.0
        current_balance = float((current.get("portfolio_snapshot", {}) or {}).get("balance", starting_balance) or starting_balance)
        profit_pct = float(current.get("total_return_pct", 0.0) or 0.0)
        lines = [
            f"Simulation batch: {'running' if bool(batch_status.get('running', False)) else 'stopped'}",
            f"PID: {batch_status.get('pid', 'n/a')}",
            f"Trades: {int(current.get('num_trades', 0) or 0)}",
            f"Balance: {current_balance:.2f}",
            f"Profit %: {profit_pct:.4f}%",
        ]
        if last_error:
            lines.append(f"Last warning/error: {last_error[:180]}")
        return "\n".join(lines)
    aggregates = dict(summary.get("aggregates", {}) or {})
    avg_trades = float(aggregates.get("num_trades", {}).get("avg", 0.0) or 0.0)
    avg_return = float(aggregates.get("total_return_pct", {}).get("avg", 0.0) or 0.0)
    latest = dict(summary.get("latest_run", {}) or {})
    verdict = dict(summary.get("candidate_verdict", {}) or {})
    lines = [
        f"Simulation batch: {'running' if bool(batch_status.get('running', False)) else 'stopped'}",
        f"PID: {batch_status.get('pid', 'n/a')}",
        f"Health: {dict(batch_status.get('health', {}) or {}).get('status', 'unknown')} ({dict(batch_status.get('health', {}) or {}).get('reason', 'unknown')})",
        f"Runs: {int(summary.get('num_runs', 0) or 0)}",
        f"Verdict: {verdict.get('status', 'unknown')}",
        f"Avg trades: {avg_trades:.2f}",
        f"Avg return: {avg_return:.4f}%",
    ]
    if latest:
        lines.append(
            "Latest run: "
            f"return={float(latest.get('total_return_pct', 0.0) or 0.0):.4f}% "
            f"trades={int(latest.get('num_trades', 0) or 0)} "
            f"win_rate={float(latest.get('win_rate_pct', 0.0) or 0.0):.2f}%"
        )
    if last_error:
        lines.append(f"Last warning/error: {last_error[:180]}")
    return "\n".join(lines)


def _handle_command(config: Any, base_dir: str, text: str, operator_state: Dict[str, Any]) -> str:
    command = (text or "").strip().split()[0].lower()
    if command == "/startbot":
        existing = read_bot_status(base_dir)
        if bool(existing.get("running", False)):
            return f"Bot already running. PID={existing.get('pid', 'n/a')} status={existing.get('status', 'unknown')}"
        allow_live = str(os.getenv("TELEGRAM_ALLOW_LIVE_CONTROL", "") or "").strip().lower() in {"1", "true", "yes", "on"}
        if not allow_live:
            return "Live bot start blocked. Set TELEGRAM_ALLOW_LIVE_CONTROL=true to allow /startbot."
        status = spawn_detached_bot(base_dir, paper=False, trading_mode=getattr(config, "trading_mode", "spot"))
        return f"Bot start requested. PID={status.get('pid', 'n/a')} status={status.get('status', 'unknown')}"
    if command == "/stopbot":
        existing = read_bot_status(base_dir)
        if not bool(existing.get("running", False)):
            return f"Bot already stopped. PID={existing.get('pid', 'n/a')} status={existing.get('status', 'unknown')}"
        status = stop_bot_process(base_dir)
        return f"Bot stop requested. Status={status.get('status', 'unknown')}"
    if command == "/startsimulation":
        existing = read_simulation_batch_status(base_dir)
        if bool(existing.get("running", False)):
            return (
                f"Simulation batch already running. PID={existing.get('pid', 'n/a')} "
                f"completed_runs={int(existing.get('completed_runs', 0) or 0)}"
            )
        status = spawn_detached_simulation_batch(
            base_dir=base_dir,
            symbol=getattr(config, "symbols", ["BTC/USDT"])[0],
            timeframe=str(getattr(config, "timeframes", {}).get("entry", "15m")),
            days=int(os.getenv("TELEGRAM_SIM_DAYS", "60")),
            trading_mode="futures",
            repeat=0,
            use_default_universe=True,
        )
        return f"Simulation batch start requested. PID={status.get('pid', 'n/a')} completed_runs={status.get('completed_runs', 0)}"
    if command == "/stopsimulation":
        existing = read_simulation_batch_status(base_dir)
        if not bool(existing.get("running", False)):
            return (
                f"Simulation batch already stopped. PID={existing.get('pid', 'n/a')} "
                f"status={existing.get('status', 'unknown')}"
            )
        batch_status = request_simulation_batch_stop(base_dir)
        sim_status = request_simulation_stop(base_dir)
        return f"Simulation stop requested. Batch={batch_status.get('status', 'unknown')} Single={sim_status.get('status', 'unknown')}"
    if command == "/botstatus":
        status = read_bot_status(base_dir)
        return f"Bot running: {bool(status.get('running', False))} PID={status.get('pid', 'n/a')} status={status.get('status', 'unknown')}"
    if command == "/simstatus":
        batch_status = read_simulation_batch_status(base_dir)
        single_status = read_simulation_status(base_dir)
        return (
            f"Simulation batch running: {bool(batch_status.get('running', False))} "
            f"completed_runs={int(batch_status.get('completed_runs', 0) or 0)}; "
            f"single simulation running: {bool(single_status.get('running', False))}"
        )
    if command == "/botlogs":
        return _format_bot_logs(base_dir, operator_state)
    if command == "/simlogs":
        return _format_sim_logs(base_dir, operator_state)
    return (
        "Unknown command.\n"
        "Supported: /startbot /stopbot /startsimulation /stopsimulation "
        "/botstatus /simstatus /botlogs /simlogs"
    )


def run_telegram_operator(config: Any, *, poll_interval_seconds: int = 3) -> None:
    if not config.telegram_bot_token or not config.telegram_chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required for telegram operator")
    base_dir = resolve_runtime_base_dir()
    state = load_operator_state(base_dir)
    if not bool(state.get("commands_registered", False)):
        try:
            _register_supported_commands(config)
            state["commands_registered"] = True
            state["commands_registered_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            save_operator_state(base_dir, state)
        except Exception:
            pass
    offset = int(state.get("last_update_id", 0) or 0)
    while True:
        updates = _get_updates(config, offset=offset + 1 if offset else None)
        for update in updates:
            offset = max(offset, int(update.get("update_id", 0) or 0))
            if not _chat_allowed(config, update):
                _record_unauthorized_update(state, update)
                state["last_update_id"] = offset
                save_operator_state(base_dir, state)
                continue
            message = update.get("message") or {}
            text = str(message.get("text", "") or "")
            response = _handle_command(config, base_dir, text, state)
            _record_command_audit(state, command=text.strip().split()[0].lower() if text.strip() else "", response=response)
            _send_message(config, response)
            state["last_update_id"] = offset
            save_operator_state(base_dir, state)
        time.sleep(max(poll_interval_seconds, 1))
