from __future__ import annotations

import os

LOG_DIR = os.getenv("BOT_LOG_DIR", "logs")
DATA_DIR = os.getenv("BOT_DATA_DIR", "data")
STATE_FILE = os.getenv("BOT_STATE_FILE", "bot_state.json")
EVENT_LOG_FILE = os.getenv("BOT_EVENT_LOG_FILE", f"{LOG_DIR}/decision_events.jsonl")
STATE_DB_FILE = os.getenv("BOT_STATE_DB_FILE", f"{DATA_DIR}/bot_runtime.sqlite3")
