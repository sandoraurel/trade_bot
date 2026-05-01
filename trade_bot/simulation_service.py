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


SIMULATION_DIRNAME = "simulation"
SIMULATION_STATUS_FILE = "status.json"
SIMULATION_STOP_FILE = "stop.request"
SIMULATION_REPORT_FILE = "report.json"
SIMULATION_CHECKPOINT_FILE = "simulation_checkpoint.json"
SIMULATION_ARTIFACTS_FILE = "simulation_artifacts.json"
SIMULATION_STDOUT_FILE = "stdout.log"
SIMULATION_STDERR_FILE = "stderr.log"
BATCH_DIRNAME = "simulation_batch"
BATCH_STATUS_FILE = "batch_status.json"
BATCH_STOP_FILE = "batch_stop.request"
BATCH_SUMMARY_FILE = "batch_summary.json"
BATCH_STDOUT_FILE = "batch_stdout.log"
BATCH_STDERR_FILE = "batch_stderr.log"
VALIDATION_DIRNAME = "simulation_validation"
VALIDATION_SUMMARY_FILE = "validation_summary.json"
SIMULATION_STATUS_STALE_SECONDS = 1800


def simulation_runtime_dir(base_dir: str) -> str:
    ensure_runtime_directories(base_dir)
    path = Path(base_dir, "data", SIMULATION_DIRNAME)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def simulation_batch_dir(base_dir: str) -> str:
    ensure_runtime_directories(base_dir)
    path = Path(base_dir, "data", BATCH_DIRNAME)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def simulation_validation_dir(base_dir: str) -> str:
    ensure_runtime_directories(base_dir)
    path = Path(base_dir, "data", VALIDATION_DIRNAME)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def simulation_paths(base_dir: str) -> Dict[str, str]:
    runtime_dir = simulation_runtime_dir(base_dir)
    return {
        "runtime_dir": runtime_dir,
        "status": os.path.join(runtime_dir, SIMULATION_STATUS_FILE),
        "stop": os.path.join(runtime_dir, SIMULATION_STOP_FILE),
        "report": os.path.join(runtime_dir, SIMULATION_REPORT_FILE),
        "checkpoint": os.path.join(runtime_dir, SIMULATION_CHECKPOINT_FILE),
        "artifacts": os.path.join(runtime_dir, SIMULATION_ARTIFACTS_FILE),
        "stdout": os.path.join(runtime_dir, SIMULATION_STDOUT_FILE),
        "stderr": os.path.join(runtime_dir, SIMULATION_STDERR_FILE),
    }


def simulation_batch_paths(base_dir: str) -> Dict[str, str]:
    batch_dir = simulation_batch_dir(base_dir)
    return {
        "batch_dir": batch_dir,
        "status": os.path.join(batch_dir, BATCH_STATUS_FILE),
        "stop": os.path.join(batch_dir, BATCH_STOP_FILE),
        "summary": os.path.join(batch_dir, BATCH_SUMMARY_FILE),
        "stdout": os.path.join(batch_dir, BATCH_STDOUT_FILE),
        "stderr": os.path.join(batch_dir, BATCH_STDERR_FILE),
        "runs_dir": os.path.join(batch_dir, "runs"),
    }


def simulation_validation_paths(base_dir: str) -> Dict[str, str]:
    validation_dir = simulation_validation_dir(base_dir)
    return {
        "validation_dir": validation_dir,
        "summary": os.path.join(validation_dir, VALIDATION_SUMMARY_FILE),
        "runs_dir": os.path.join(validation_dir, "runs"),
    }


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _detached_popen_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True
    return kwargs


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


def _process_health(payload: Dict[str, Any], *, running: bool) -> Dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)
    updated_at = _parse_iso(payload.get("updated_at"))
    if payload.get("pid") and not running and payload.get("status") in {"starting", "running", "stop_requested"}:
        return {"status": "stale_pid", "reason": "process_not_running"}
    if updated_at is not None and (now - updated_at).total_seconds() > SIMULATION_STATUS_STALE_SECONDS:
        return {"status": "stale", "reason": "status_update_stale"}
    return {"status": "ok" if running else "inactive", "reason": "ok" if running else str(payload.get("status", "stopped") or "stopped")}


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


def read_simulation_status(base_dir: str) -> Dict[str, Any]:
    paths = simulation_paths(base_dir)
    payload = _read_json(paths["status"])
    pid = int(payload.get("pid", 0) or 0)
    payload["running"] = _pid_is_running(pid) and payload.get("status") in {"starting", "running", "stop_requested"}
    payload["stop_requested"] = os.path.exists(paths["stop"])
    payload["checkpoint_exists"] = os.path.exists(paths["checkpoint"])
    payload["health"] = _process_health(payload, running=bool(payload["running"]))
    payload["last_error"] = _last_warning_or_error(paths)
    payload["paths"] = paths
    return payload


def write_simulation_status(base_dir: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    paths = simulation_paths(base_dir)
    current = _read_json(paths["status"])
    merged = {**current, **payload}
    merged.setdefault("updated_at", dt.datetime.now(dt.timezone.utc).isoformat())
    _write_json(paths["status"], merged)
    return merged


def clear_stop_request(base_dir: str) -> None:
    stop_path = simulation_paths(base_dir)["stop"]
    if os.path.exists(stop_path):
        os.remove(stop_path)


def request_simulation_stop(base_dir: str) -> Dict[str, Any]:
    paths = simulation_paths(base_dir)
    status = read_simulation_status(base_dir)
    Path(paths["stop"]).write_text("stop\n", encoding="utf-8")
    status = write_simulation_status(
        base_dir,
        {
            "status": "stop_requested",
            "stop_requested_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    status["running"] = read_simulation_status(base_dir).get("running", False)
    return status


def persist_simulation_report(base_dir: str, result: Dict[str, Any]) -> str:
    path = simulation_paths(base_dir)["report"]
    _write_json(path, result)
    return path


def simulation_stop_requested(base_dir: str) -> bool:
    return os.path.exists(simulation_paths(base_dir)["stop"])


def spawn_detached_simulation(
    *,
    base_dir: str,
    symbol: str,
    timeframe: str,
    days: int,
    trading_mode: str,
) -> Dict[str, Any]:
    paths = simulation_paths(base_dir)
    status = read_simulation_status(base_dir)
    if status.get("running"):
        return status
    clear_stop_request(base_dir)
    stdout_handle = open(paths["stdout"], "a", encoding="utf-8")
    stderr_handle = open(paths["stderr"], "a", encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "trade_bot.cli",
        "simulate-worker",
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--days",
        str(days),
        "--trading-mode",
        trading_mode,
    ]
    process = subprocess.Popen(
        cmd,
        cwd=base_dir,
        stdout=stdout_handle,
        stderr=stderr_handle,
        **_detached_popen_kwargs(),
    )
    return write_simulation_status(
        base_dir,
        {
            "status": "starting",
            "pid": process.pid,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "days": days,
            "trading_mode": trading_mode,
            "mode": "detached",
        },
    )


def stop_simulation_process(base_dir: str, force: bool = False) -> Dict[str, Any]:
    status = request_simulation_stop(base_dir)
    if force and _pid_is_running(int(status.get("pid", 0) or 0)):
        os.kill(int(status["pid"]), signal.SIGINT)
    return read_simulation_status(base_dir)


def read_simulation_batch_status(base_dir: str) -> Dict[str, Any]:
    paths = simulation_batch_paths(base_dir)
    payload = _read_json(paths["status"])
    pid = int(payload.get("pid", 0) or 0)
    payload["running"] = _pid_is_running(pid) and payload.get("status") in {"starting", "running", "stop_requested"}
    payload["stop_requested"] = os.path.exists(paths["stop"])
    payload["health"] = _process_health(payload, running=bool(payload["running"]))
    payload["last_error"] = _last_warning_or_error(paths)
    payload["paths"] = paths
    return payload


def write_simulation_batch_status(base_dir: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    paths = simulation_batch_paths(base_dir)
    current = _read_json(paths["status"])
    merged = {**current, **payload}
    merged.setdefault("updated_at", dt.datetime.now(dt.timezone.utc).isoformat())
    _write_json(paths["status"], merged)
    return merged


def clear_simulation_batch_stop(base_dir: str) -> None:
    stop_path = simulation_batch_paths(base_dir)["stop"]
    if os.path.exists(stop_path):
        os.remove(stop_path)


def request_simulation_batch_stop(base_dir: str) -> Dict[str, Any]:
    paths = simulation_batch_paths(base_dir)
    Path(paths["stop"]).write_text("stop\n", encoding="utf-8")
    write_simulation_batch_status(
        base_dir,
        {
            "status": "stop_requested",
            "stop_requested_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    return read_simulation_batch_status(base_dir)


def simulation_batch_stop_requested(base_dir: str) -> bool:
    return os.path.exists(simulation_batch_paths(base_dir)["stop"])


def spawn_detached_simulation_batch(
    *,
    base_dir: str,
    symbol: str,
    timeframe: str,
    days: int,
    trading_mode: str,
    repeat: int,
    use_default_universe: bool,
) -> Dict[str, Any]:
    paths = simulation_batch_paths(base_dir)
    status = read_simulation_batch_status(base_dir)
    if status.get("running"):
        return status
    clear_simulation_batch_stop(base_dir)
    Path(paths["runs_dir"]).mkdir(parents=True, exist_ok=True)
    stdout_handle = open(paths["stdout"], "a", encoding="utf-8")
    stderr_handle = open(paths["stderr"], "a", encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "trade_bot.cli",
        "simulate-batch-worker",
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--days",
        str(days),
        "--trading-mode",
        trading_mode,
        "--repeat",
        str(repeat),
    ]
    if use_default_universe:
        cmd.append("--use-default-universe")
    process = subprocess.Popen(
        cmd,
        cwd=base_dir,
        stdout=stdout_handle,
        stderr=stderr_handle,
        **_detached_popen_kwargs(),
    )
    return write_simulation_batch_status(
        base_dir,
        {
            "status": "starting",
            "pid": process.pid,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "days": days,
            "trading_mode": trading_mode,
            "repeat": repeat,
            "use_default_universe": use_default_universe,
            "completed_runs": 0,
        },
    )
