# Trade Bot Deployment

This repo now includes two practical always-on paths:

1. Docker Compose for a VPS or any host with Docker.
2. `systemd` for a Linux machine running the repo directly from a virtualenv.

## 1. Prepare the host

```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
cp .env.example .env
```

Fill in `.env` with Binance testnet credentials. Keep secrets only in `.env`, never in source files.

## 2. Docker Compose path

```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
docker compose up -d --build bot
```

What this gives you:

- automatic restart with `restart: unless-stopped`
- container healthcheck based on runtime snapshot freshness
- persisted `data/`, `logs/`, and `bot_state.json`
- clean `SIGTERM` shutdown with state save
- optional Prometheus and Grafana via:

```bash
docker compose --profile observability up -d
```

Useful commands:

```bash
docker compose ps
docker compose logs -f bot
docker compose exec bot python -m trade_bot.cli status
docker compose exec bot python -m trade_bot.cli healthcheck --json
```

## 3. systemd path

Copy the repo to your Linux host, create a virtualenv, install dependencies, and then install the template service:

```bash
sudo cp deploy/systemd/trade-bot.service /etc/systemd/system/trade-bot.service
sudo systemctl daemon-reload
sudo systemctl enable --now trade-bot
```

Expected layout on the host:

- repo at `/opt/trade_bot`
- virtualenv at `/opt/trade_bot/.venv`
- env file at `/opt/trade_bot/.env`

Useful commands:

```bash
sudo systemctl status trade-bot
journalctl -u trade-bot -f
/opt/trade_bot/.venv/bin/python -m trade_bot.cli healthcheck --json
```

## 4. Operational notes

- The bot still starts with `python -m trade_bot.main live`.
- `TRADE_BOT_HOME` controls where runtime files live.
- `BOT_STATE_FILE` can relocate persisted state; Docker uses `data/bot_state.json` while local runs can keep the existing root-level `bot_state.json`.
- `python -m trade_bot.cli status` remains the human-readable status view.
- `python -m trade_bot.cli healthcheck` is the machine-friendly readiness probe.
- Metrics stay on port `8000` in the live runtime.
- Runtime health is based on the persisted snapshot in `data/bot_runtime.sqlite3`.

## 5. What to watch

- `logs/mentor.log`
- `logs/decision_events.jsonl`
- `data/bot_runtime.sqlite3`
- `bot_state.json`

If healthcheck starts failing, inspect:

1. snapshot freshness
2. reconciliation failures
3. emergency mode or circuit breakers
4. Binance/API connectivity and credentials
