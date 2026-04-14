from __future__ import annotations

import datetime as dt
import json
import os
import signal
import subprocess
import sys
import ctypes
from pathlib import Path
from typing import Any, Dict

from .bootstrap import ensure_runtime_directories

BOT_CONTROL_DIRNAME = "bot_control"
BOT_STATUS_FILE = "status.json"
BOT_STDOUT_FILE = "stdout.log"
BOT_STDERR_FILE = "stderr.log"
BOT_STATUS_STALE_SECONDS = 900
BOT_HEARTBEAT_STALE_SECONDS = 7200


def bot_control_dir(base_dir: str) -> str:
    ensure_runtime_directories(base_dir)
    path = Path(base_dir, "data", BOT_CONTROL_DIRNAME)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def bot_control_paths(base_dir: str) -> Dict[str, str]:
    runtime_dir = bot_control_dir(base_dir)
    return {
        "runtime_dir": runtime_dir,
        "status": os.path.join(runtime_dir, BOT_STATUS_FILE),
        "stdout": os.path.join(runtime_dir, BOT_STDOUT_FILE),
        "stderr": os.path.join(runtime_dir, BOT_STDERR_FILE),
    }


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _pid_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return int(exit_code.value) == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _parse_iso(raw: Any) -> dt.datetime | None:
    if not raw:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _tail_lines(path: str, *, max_lines: int = 60) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return [str(line).rstrip() for line in lines[-max_lines:]]


def _last_warning_or_error(paths: Dict[str, str]) -> str:
    merged = _tail_lines(paths.get("stderr", ""), max_lines=80) + _tail_lines(paths.get("stdout", ""), max_lines=80)
    for line in reversed(merged):
        upper = line.upper()
        if "[ERROR]" in upper or "ERROR" in upper or "[WARN]" in upper or "WARN" in upper or "TRACEBACK" in upper:
            return line.strip()
    return ""


def _bot_process_health(base_dir: str, payload: Dict[str, Any], *, running: bool) -> Dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)
    updated_at = _parse_iso(payload.get("updated_at"))
    last_heartbeat = None
    state_path = os.path.join(base_dir, "bot_state.json")
    state_payload = _read_json(state_path)
    if state_payload:
        last_heartbeat = _parse_iso(state_payload.get("last_heartbeat"))

    if payload.get("pid") and not running and payload.get("status") in {"starting", "running"}:
        return {"status": "stale_pid", "reason": "process_not_running"}
    if updated_at is not None and (now - updated_at).total_seconds() > BOT_STATUS_STALE_SECONDS:
        return {"status": "stale", "reason": "status_update_stale"}
    if running and last_heartbeat is not None and (now - last_heartbeat).total_seconds() > BOT_HEARTBEAT_STALE_SECONDS:
        return {"status": "stale", "reason": "heartbeat_stale", "last_heartbeat": last_heartbeat.isoformat()}
    return {
        "status": "ok" if running else "inactive",
        "reason": "ok" if running else str(payload.get("status", "stopped") or "stopped"),
        "last_heartbeat": last_heartbeat.isoformat() if last_heartbeat else None,
    }


def read_bot_status(base_dir: str) -> Dict[str, Any]:
    paths = bot_control_paths(base_dir)
    payload = _read_json(paths["status"])
    pid = int(payload.get("pid", 0) or 0)
    payload["running"] = _pid_is_running(pid) and payload.get("status") in {"starting", "running"}
    payload["health"] = _bot_process_health(base_dir, payload, running=bool(payload["running"]))
    payload["last_error"] = _last_warning_or_error(paths)
    payload["paths"] = paths
    return payload


def write_bot_status(base_dir: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    paths = bot_control_paths(base_dir)
    current = _read_json(paths["status"])
    merged = {**current, **payload}
    merged.setdefault("updated_at", dt.datetime.now(dt.timezone.utc).isoformat())
    _write_json(paths["status"], merged)
    return merged


def spawn_detached_bot(base_dir: str, *, paper: bool = False, trading_mode: str = "spot") -> Dict[str, Any]:
    status = read_bot_status(base_dir)
    if status.get("running"):
        return status
    paths = bot_control_paths(base_dir)
    stdout_handle = open(paths["stdout"], "a", encoding="utf-8")
    stderr_handle = open(paths["stderr"], "a", encoding="utf-8")
    cmd = [sys.executable, "-m", "trade_bot.cli", "live", "--trading-mode", trading_mode]
    if paper:
        cmd.append("--paper")
    process = subprocess.Popen(
        cmd,
        cwd=base_dir,
        stdout=stdout_handle,
        stderr=stderr_handle,
        start_new_session=True,
    )
    return write_bot_status(
        base_dir,
        {
            "status": "starting",
            "pid": process.pid,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "paper": paper,
            "trading_mode": trading_mode,
        },
    )


def stop_bot_process(base_dir: str, force: bool = False) -> Dict[str, Any]:
    status = read_bot_status(base_dir)
    pid = int(status.get("pid", 0) or 0)
    if _pid_is_running(pid):
        os.kill(pid, signal.SIGINT if not force else signal.SIGTERM)
    return write_bot_status(
        base_dir,
        {
            "status": "stopped",
            "stopped_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
