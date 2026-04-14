from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .constants import DATA_DIR, LOG_DIR


def resolve_runtime_base_dir() -> str:
    configured = os.getenv("TRADE_BOT_HOME", "").strip()
    if configured:
        candidate = os.path.abspath(configured)
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            return candidate
    return os.getcwd()


def load_runtime_environment(base_dir: str | None = None) -> str:
    resolved = os.path.abspath(base_dir or resolve_runtime_base_dir())
    env_candidates = [
        os.path.join(resolved, ".env"),
        os.path.join(resolved, "trade_bot", ".env"),
    ]
    for candidate in env_candidates:
        if os.path.exists(candidate):
            load_dotenv(candidate, override=False)
            break
    return resolved


def ensure_runtime_directories(base_dir: str) -> None:
    for relative in (LOG_DIR, DATA_DIR):
        Path(base_dir, relative).mkdir(parents=True, exist_ok=True)


def getenv_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
