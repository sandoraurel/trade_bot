FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRADE_BOT_HOME=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

COPY trade_bot /app/trade_bot
COPY knowledge /app/knowledge
COPY deploy /app/deploy
COPY prometheus.yml /app/prometheus.yml
COPY AGENTS.md /app/AGENTS.md

RUN mkdir -p /app/data /app/logs \
    && chmod +x /app/deploy/run-live.sh

HEALTHCHECK --interval=60s --timeout=15s --start-period=120s --retries=3 \
  CMD python -m trade_bot.cli healthcheck --max-snapshot-age-seconds 1200 || exit 1

ENTRYPOINT ["/app/deploy/run-live.sh"]
