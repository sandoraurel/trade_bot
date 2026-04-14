# Next Roadmap: Exit Quality, Universe Selection, Telegram Control, and Reliability

## Summary
Tasks 1-7 are now effectively implemented, so the next phase should stop adding more entry complexity and instead improve what happens after entry, which assets are eligible, and how safely the bot is operated. Research points the same way: crash-aware momentum needs volatility management, crypto reversal only works when liquidity is high, and multi-asset crypto portfolios benefit from decorrelation-aware selection rather than simply trading more names. Relevant anchors: [Momentum Crashes](https://www.aqr.com/insights/research/journal-article/momentum-crashes), [Volatility Managed Portfolios](https://www.nber.org/papers/w22208), [Short-term reversal, momentum, and liquidity effects in cryptocurrency markets](https://www.sciencedirect.com/science/article/pii/S1057521921002349), [A network-based strategy of price correlations for optimal cryptocurrency portfolios](https://arxiv.org/abs/2304.02362), and Telegram’s command model via [Bot commands](https://core.telegram.org/api/bots/commands).

The next plan should be implemented in four workstreams:
1. post-entry risk and exit quality
2. dynamic universe and cluster-aware asset selection
3. Telegram control-plane hardening, including `/botlogs`
4. runtime/state reliability plus final validation discipline

## Key Changes

### 1. Add a post-entry risk and exit engine
- Introduce a dedicated exit-policy layer that evaluates every open trade on each bar using regime, realized volatility, open-trade MFE/MAE, and crash-risk state.
- Add volatility-managed risk scaling after entry:
  - reduce effective risk when realized volatility spikes above a regime-specific ceiling
  - keep breakout exposure smallest under elevated crash risk
  - allow pullback to hold more often than breakout in unstable rebound states
- Add three explicit exit families:
  - `time_stop`: close trades whose thesis has not developed within a family-specific holding window
  - `profit_protect`: tighten stop after a minimum favorable excursion so winners stop round-tripping
  - `volatility_stop`: shrink holding tolerance when ATR/realized volatility expands materially after entry
- Add regime-aware trailing behavior:
  - breakout: fastest trailing and fastest de-risking
  - pullback: moderate trailing, tolerate trend continuation
  - mean reversion: earlier partial-profit / reclaim-failure exits
- Add campaign diagnostics for exits:
  - exits by reason
  - expectancy by exit reason
  - winners given back after MFE threshold
  - average MFE/MAE by family
- Acceptance:
  - low-quality losers should be cut earlier
  - winner giveback should fall
  - breakout drawdowns should shrink without collapsing total opportunity flow

### 2. Add dynamic universe selection before signal ranking
- Add a pre-scan universe filter for every batch/live cycle that scores symbols on:
  - spread
  - liquidity/tradability
  - realized volatility
  - directional efficiency
  - symbol-bucket crowding
  - correlation-cluster crowding
- Build a rolling eligible-universe selector:
  - keep all configured symbols visible in diagnostics
  - only pass the top tradable subset into full ranking when symbols are too crowded or too weak
- Add explicit cluster budgeting:
  - majors
  - exchange-beta
  - high-beta alts
  - slower large caps
  - cap simultaneous exposure per cluster and per side
- Add rolling symbol admission evidence:
  - demote symbols with persistently negative realized expectancy after costs
  - promote symbols with stable positive expectancy and clean fills
- Public interfaces:
  - `campaign_summary.universe_selection`
  - `decision_diagnostics.universe_rejections_by_symbol`
  - batch summary sections for eligible vs rejected symbols
- Acceptance:
  - more symbols can be evaluated than traded
  - trade mix should diversify without low-edge overexpansion
  - weakest symbols should be visibly filtered before entry ranking

### 3. Harden Telegram control and short log reporting
- Keep the existing command set and make it production-safe:
  - `/startbot`
  - `/stopbot`
  - `/startsimulation`
  - `/stopsimulation`
  - `/botstatus`
  - `/simstatus`
  - `/botlogs`
  - `/simlogs`
- Make commands idempotent:
  - starting an already-running bot returns current PID/status instead of spawning duplicates
  - stopping an already-stopped bot returns clean state instead of erroring
- Register the command menu at operator startup using Telegram command metadata so the bot shows the supported commands directly in Telegram.
- Upgrade `/botlogs` to remain short but more useful:
  - bot running/stopped state
  - PID
  - current balance/equity
  - realized PnL for the current reporting window
  - fills count
  - last warning/error summary from stdout/stderr
  - timestamp of last successful heartbeat/update
- Upgrade `/simlogs` similarly:
  - current batch state
  - completed runs
  - candidate verdict
  - latest return / trades / win rate
- Add explicit access/safety controls:
  - only configured chat ID is accepted
  - `/startbot` refuses live start unless `TELEGRAM_ALLOW_LIVE_CONTROL=true`
  - all command actions are written to an operator audit log
- Public interfaces:
  - `data/telegram_operator_state.json` remains the operator state file
  - add command-registration bootstrap on operator startup
  - add short log summary readers over `stdout.log`, `stderr.log`, `batch_stdout.log`, `batch_stderr.log`
- Acceptance:
  - the bot can be started and stopped from Telegram safely
  - `/botlogs` returns short operational info, not raw noisy logs
  - no duplicate process spawns from repeated command taps

### 4. Fix reliability gaps and finish validation discipline
- Fix the existing state-save recursion warning path in core runtime state persistence.
- Add runtime health checks for detached bot/simulation processes:
  - stale PID detection
  - missing heartbeat detection
  - stale stdout/stderr growth without status updates
- Add an operator-safe “last error” summary source for Telegram and CLI.
- Extend validation discipline from reporting into workflow:
  - standard fast run: 30d
  - confirmation run: 60d
  - robustness run: repeated 90d slices before trusting a candidate
- Add explicit acceptance gates for a release candidate:
  - median run must beat baseline on return
  - confirmation horizon cannot degrade materially from iteration horizon
  - candidate verdict cannot be `not_ready`
  - no unresolved runtime reliability warnings
- Acceptance:
  - no more `maximum recursion depth exceeded` save-state warnings
  - final summaries can be trusted operationally, not just analytically

## Public Interfaces / Types
- Add new campaign result sections:
  - `campaign_summary.universe_selection`
  - `campaign_summary.exit_quality`
- Add new batch-summary sections:
  - `eligible_symbols`
  - `universe_rejections`
  - `exit_reason_expectancy`
- Telegram command behavior remains the same by name; behavior is hardened rather than renamed.
- Add env/config gates:
  - `TELEGRAM_ALLOW_LIVE_CONTROL`
  - exit-policy knobs for family-specific time stops, trailing, and volatility de-risking
  - universe-selection knobs for eligible top-N, cluster caps, and tradability floors

## Test Plan
- Exit engine tests:
  - breakout exits faster under post-shock volatility spike
  - pullback survives longer than breakout in the same stable-trend state
  - mean-reversion closes on reclaim failure / time-stop as designed
- Universe selection tests:
  - all configured symbols remain reported
  - weak symbols are filtered before ranking
  - cluster budgets block same-side crowding across correlated names
- Telegram tests:
  - `/startbot` is idempotent
  - `/stopbot` is idempotent
  - unauthorized chat is ignored
  - `/botlogs` returns short status plus last error summary
  - command menu registration is attempted on startup
- Reliability tests:
  - state save no longer recurses
  - stale PID is detected
  - stale heartbeat is surfaced in logs/status
- Validation tests:
  - 30d/60d/90d slices produce consistent batch verdict data
  - release-candidate gate fails when confirmation deteriorates or reliability warnings exist

## Assumptions And Defaults
- Tasks 1-7 remain the current foundation; this plan is the next roadmap layer, not a rewrite.
- Telegram support already exists, so the next work is hardening and operational usefulness, not introducing Telegram from scratch.
- `/botlogs` should stay short and human-readable; it should summarize health and recent issues, not dump full raw logs.
- The next performance gains are expected to come more from exits, universe quality, and operational reliability than from loosening entry rules further.
- Default safety choice: Telegram live start control is disabled unless explicitly enabled by environment/config.
