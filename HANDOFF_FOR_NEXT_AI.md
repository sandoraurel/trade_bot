# Handoff For Next AI

## Repository Status

This repository is the current upgraded trading bot codebase.

- GitHub repo: `https://github.com/sandoraurel/trade_bot`
- Current uploaded branch: `main`
- Latest uploaded code commit before this handoff: `bb92729 Refresh upgraded trading bot code`
- Local project path used during development: `/Users/aurel.sandor/.ollama/models/trade_bot/trade_bot`
- Unit suite status before this handoff: `323 tests OK`

Important: runtime state, secrets, local simulation databases, historical cache, logs, and `.venv` are intentionally ignored and not uploaded.

## What Is Uploaded

The repository contains the full source code and tests needed for another AI/developer to continue:

- `trade_bot/`: bot source package
- `test_trade_bot_core.py`: main regression/unit test suite
- `requirements.txt`: Python dependencies
- `deploy/`: deployment helpers for local, Windows, systemd, and live run workflows
- `DEPLOYMENT.md`: deployment notes
- `ROADMAP_NEXT_PHASE_EXIT_UNIVERSE_TELEGRAM.md`: previous roadmap/progress context
- `ROADMAP_WINRATE_AND_THROUGHPUT.md`: previous roadmap/progress context
- `knowledge/bot_runtime.md`: runtime notes
- `.env.example`: example environment template

## What Is Not Uploaded

These are intentionally excluded by `.gitignore`:

- `.env`
- `.venv/`
- `data/`
- `logs/`
- `bot_state.json`
- `bot_runtime.sqlite3`
- `*.sqlite3`
- `trade-bot-secrets-and-state*`
- Python caches and macOS `.DS_Store` files

Do not ask the user to upload secrets to GitHub. If runtime state is needed, use sanitized summaries instead.

## Current Goal

The user wants the bot to become significantly better at:

- better trades
- better profitability
- eventually reaching about `2-3 trades/day`

Important working principle: do not chase frequency first. The bot has repeatedly become worse when entries were loosened before profitability improved. The current priority is:

1. Improve expectancy and win rate.
2. Reduce bad exits and winner giveback.
3. Only then expand frequency carefully.

## Latest Validation Snapshot

Latest local validation command used:

```bash
./.venv/bin/python -m trade_bot.cli simulation-validation --validation-windows 7 --repeat 1 --use-default-universe
```

Latest local validation result:

```text
Validation Runs: 1
Window: 7d
Avg Trades: 2.00
Trades/day: 0.286
Avg Return: -0.1921%
Avg Win Rate: 0.00%
Flow Status: flow_recovered_but_unprofitable
Readiness: 67.1/100
Signal-to-submission: 66.67%
Submission-to-fill: 300.00%
SL negative P/L share: 0.00%
Triple-barrier labels: 3
Triple-barrier TP-first: 66.67%
Triple-barrier SL-first: 0.00%
Triple-barrier time-exit: 33.33%
Candidate flow starved: false
```

Latest key blockers:

```text
Win rate: 0.00%
Return still negative: -0.1921%
Trades/day far below goal: 0.286 vs 2.0-3.0
Frequency expansion gate: blocked because not enough closed trades
Main active family: trend_pullback
Trend pullback expectancy: negative
Exit reasons in latest validation: PROFIT_PROTECT_STOP and PULLBACK_THESIS_FAIL
```

Latest candidate-flow blockers:

```text
pre:strategy_produced_no_proposal = 4369
strategy:trend_pullback:regime_not_trend = 1663
strategy:trend_pullback:bearish_higher_timeframe_not_aligned = 892
strategy:trend_pullback:bullish_higher_timeframe_not_aligned = 670
strategy:trend_pullback:bearish_pullback_shape_not_qualified = 407
strategy:trend_pullback:trend_direction_not_actionable = 396
```

## Important Recent Upgrades

The current code includes many strategy-quality and validation upgrades, including:

- Candidate-flow diagnostics in validation reports.
- `Flow Status` in validation CLI output.
- State-family gating and constructive pullback profiles.
- Candidate-flow rescue for proposal-starved periods.
- Single-candidate escape for borderline BTC/ETH pullback candidates.
- Hard veto protection around the escape path so learning vetoes, crash risk, unstable regimes, bad RR, and research conflicts are not bypassed.
- Triple-barrier signal labeling.
- Pullback meta-filter hooks.
- Profit-protect stop handling.
- Thesis-failure exit handling.
- Fee-aware protected stop logic.
- Fresh setup / repeated setup suppression.
- Frequency expansion gate that waits for enough evidence before expanding trade count.

## Most Recent Upgrade Context

The latest work fixed a recurring problem where the bot became too defensive and produced zero trades.

What changed:

- Added a narrow `single_candidate_escape` path in `trade_bot/main.py`.
- It only allows borderline ensemble-rejected pullback candidates for selected major symbols by default: `BTC/USDT`, `ETH/USDT`.
- It still blocks hard vetoes:
  - `learning_veto`
  - `research_conflict`
  - `unstable_regime`
  - `momentum_crash_risk`
  - `rr_too_low`
  - `edge_below_threshold`
- Added more specific ensemble rejection diagnostics instead of collapsing every rejection into `no_ensemble_selection`.
- Added tests for the single-candidate escape and candidate-flow rescue quality gate.

Why it matters:

- Before the final tuning, validation was starved with `0` trades.
- After tuning, flow recovered to `2` trades with no SL-first triple-barrier labels and no stop-loss negative P/L share.
- Profitability is still not fixed.

## Key Commands

Run full unit tests:

```bash
./.venv/bin/python -m unittest -q test_trade_bot_core.py
```

Run validation:

```bash
./.venv/bin/python -m trade_bot.cli simulation-validation --validation-windows 7 --repeat 1 --use-default-universe
```

Show validation summary:

```bash
./.venv/bin/python -m trade_bot.cli simulation-batch-summary
```

Run the bot:

```bash
./.venv/bin/python -m trade_bot
```

On Windows from the project directory, equivalent examples used by the user:

```bat
.\.venv\Scripts\python.exe -m unittest -q test_trade_bot_core.py
.\.venv\Scripts\python.exe -m trade_bot.cli simulation-validation --validation-windows 7 --repeat 1 --use-default-universe
.\.venv\Scripts\python.exe -m trade_bot.cli simulation-batch-summary
.\.venv\Scripts\python.exe -m trade_bot
```

## Recommended Next Work

Do not immediately widen trade frequency. The next best work should target profitability:

1. Analyze the latest losing BTC pullback exits.
2. Reduce `PROFIT_PROTECT_STOP` and `PULLBACK_THESIS_FAIL` losses after partial/winner behavior.
3. Add exit-quality tests around trades with `MFE > 0.25R` that still close negative.
4. Improve pullback entry confirmation so BTC/ETH escapes do not enter before enough follow-through exists.
5. Only after at least one family has non-negative expectancy, reopen frequency expansion beyond BTC/ETH.

Concrete next upgrade idea:

- Add a follow-through confirmation requirement for `single_candidate_escape` entries:
  - require stronger close location, rising short-term volume, or reclaim hold within the last 1-2 bars
  - block entries where MFE historically appears but reverses immediately
  - preserve the current no-starvation behavior by keeping rescue diagnostics visible

Acceptance target for next validation:

```text
Flow remains recovered.
Trades stay above zero.
Return improves from -0.1921%.
Win rate improves above 0%.
Triple-barrier SL-first stays low.
Stop-loss negative P/L share remains low.
No broad frequency expansion yet.
```

## Safety Notes For Next AI

- Do not upload `.env`, SQLite DBs, logs, or `trade-bot-secrets-and-state*`.
- Do not remove `.gitignore` protections.
- Do not loosen all strategy gates just to increase trade count.
- Do not bypass `learning_veto` without a specific test and reason.
- Always run targeted tests and then the full unit suite after changes.
- If validation becomes `candidate_flow_starved`, restore flow before claiming the upgrade is complete.
- If validation gets more trades but worse return, treat that as a failed frequency upgrade.

