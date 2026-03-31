#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ENV_FILE="$ROOT_DIR/.env"

cd "$ROOT_DIR"

if ! command -v caffeinate >/dev/null 2>&1; then
  echo "caffeinate is required on macOS but was not found." >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtualenv Python not found at $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Environment file not found at $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

export TRADE_BOT_HOME="$ROOT_DIR"
export BOT_STATE_FILE="${BOT_STATE_FILE:-bot_state.json}"

echo "Starting trade bot with caffeinate from $ROOT_DIR"
echo "Stop with Ctrl+C"

exec caffeinate -dimu "$PYTHON_BIN" -m trade_bot.main live
