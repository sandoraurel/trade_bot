#!/usr/bin/env sh
set -eu

BASE_DIR="${TRADE_BOT_HOME:-/app}"
cd "$BASE_DIR"

mkdir -p "$BASE_DIR/data" "$BASE_DIR/logs"

if [ -f "$BASE_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$BASE_DIR/.env"
  set +a
fi

: "${BOT_LOOP_SLEEP_SECONDS:=60}"
export TRADE_BOT_HOME="$BASE_DIR"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

exec python -m trade_bot.main live
