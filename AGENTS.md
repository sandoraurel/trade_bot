# AGENTS.md

## Repo Overview

This repo currently contains:
- A single runtime package under `trade_bot/`
- One dependency file: `requirements.txt`
- A `unittest`-style test module at `test_trade_bot_core.py`
- No configured lint tool
- No configured type checker
- No `pyproject.toml`, `setup.cfg`, `pytest.ini`, `mypy.ini`, `tox.ini`, or `Makefile`

## Setup

Create and activate a virtual environment, then install the repo dependencies:

```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you want to verify what dependencies are declared:

```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
cat requirements.txt
```

## Run The Bot

Run the live mode:

```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
python -m trade_bot.main live
```

Run backtest mode:

```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
python -m trade_bot.main backtest --symbol BTC/USDT --days 90 --timeframe 15m
```

Run hyperparameter optimization:

```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
python -m trade_bot.main hyperopt --symbol BTC/USDT --days 90 --combinations 20
```

## Tests

The repo includes a basic `unittest` test module.

Run the current tests:

```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
python3 -m unittest -q test_trade_bot_core.py
```

Minimal syntax smoke check for all Python files:

```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
python3 -m py_compile $(find trade_bot -name '*.py' | sort)
```

## Lint / Typecheck

There is no lint tool configured in this repo.
There is no type checker configured in this repo.

Confirm configured tooling files are absent:

```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
find . -maxdepth 2 \( -name 'pyproject.toml' -o -name 'setup.cfg' -o -name 'setup.py' -o -name '.flake8' -o -name 'mypy.ini' -o -name 'pytest.ini' -o -name 'tox.ini' \) | sort
```

Use the syntax smoke check as the only built-in static verification currently available:

```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
python3 -m py_compile $(find trade_bot -name '*.py' | sort)
```

## Coding Conventions

- Keep changes dependency-light unless a new dependency is explicitly required.
- Prefer standard library solutions first.
- Preserve the current runtime entrypoint: `python -m trade_bot.main ...`
- Keep operator-facing answers grounded in retrieved knowledge and tool outputs.
- Add new architecture as isolated modules under `trade_bot/` instead of expanding `trade_bot/main.py` further.
- Use ASCII by default unless a file already contains non-ASCII text.
- Prefer small, testable helpers over long inline logic blocks.
- Keep tool and retrieval code side-effect-light and deterministic where possible.
- Treat `knowledge/` as non-parametric memory and live bot state/tools as runtime truth.
- Do not hardcode secrets; continue reading runtime credentials from environment variables.

## Do

- Do work from the repo root before running commands:
```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
```

- Do verify syntax after Python edits:
```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
python3 -m py_compile $(find trade_bot -name '*.py' | sort)
```

- Do keep new files inside the existing package layout:
```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
find trade_bot -maxdepth 3 -type f | sort
```

- Do inspect the current dependency surface before adding packages:
```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
cat requirements.txt
```

- Do keep retrieval knowledge in the local knowledge base:
```bash
cd /Users/aurel.sandor/.ollama/models/trade_bot/trade_bot
find knowledge -maxdepth 2 -type f | sort
```

## Don't

- Don't invent repo tooling that is not configured here.
- Don't assume `pytest`, `ruff`, `mypy`, `black`, or `make` are available.
- Don't add new logic directly into `trade_bot/main.py` when it can live in a dedicated module.
- Don't replace grounded retrieval/tool behavior with model-only responses.
- Don't commit secrets, API keys, or chat tokens into source files.
- Don't change the CLI contract without updating the `python -m trade_bot.main` entry flow.
- Don't rely on static knowledge files for live runtime truth when a tool can fetch current state.
