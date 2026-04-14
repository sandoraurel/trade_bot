# Roadmap To 2-3 Trades/Day, Higher Win Rate, and Broader Multi-Coin Trading

## Summary
We should treat the next phase as a portfolio-system upgrade, not a single-strategy tweak. The current bot is already sequential, adaptive, and realistic enough to trust as a test harness, but the results show three structural problems:

- Too few opportunities survive the regime + strategy + ensemble + reliability stack.
- Too much of the surviving flow is still breakout-driven, so losses cluster when breakout conditions are misread.
- Multi-coin simulation is running, but diagnostics/reporting are BTC-centric and too shallow, which makes it hard to prove whether other symbols are truly undertrading because of logic, ranking, or reporting.

### Research-backed direction
- Crash-aware momentum management: trend/momentum systems need explicit rebound/crash protection, not just confidence scoring.
  Source: Daniel & Moskowitz, Momentum Crashes
- Portfolio no-trade regions after costs: strongest systems do not force trades when post-cost edge is marginal.
  Source: Garleanu-Pedersen style transaction-cost / no-trade-region literature
- Liquidity-conditioned crypto reversal: short-horizon crypto reversal works best when conditioned on liquidity and tradability, not as blanket anti-trend logic.
  Sources: crypto liquidity/reversal literature including Journal of International Financial Markets, Institutions and Money
- Adaptive portfolio construction matters as much as signal generation: the next edge comes from strategy rotation and cross-sectional ranking, not just tighter entry rules.
  Source: adaptive crypto trend/portfolio construction research such as arXiv:2602.11708

## Big Task 1: Fix Multi-Coin Truthfulness And Cross-Symbol Diagnostics
Goal: make it impossible for the bot to look BTC-only unless it is actually BTC-only.

### Subtasks
- Replace BTC-centric campaign reporting with true per-symbol rollups.
  - Report per-symbol:
    - trades
    - raw signals
    - fills
    - PnL
    - win rate
    - skipped signals
    - average edge / expectancy
- Extend batch summaries to include:
  - `signals_by_symbol`
  - `submitted_by_symbol`
  - `filled_by_symbol`
  - `closed_by_symbol`
  - `wins_by_symbol`
  - `losses_by_symbol`
  - `skip_reasons_by_symbol`
- Add explicit no-opportunity vs rejected-opportunity separation.
  - Distinguish:
    - strategy produced no proposal
    - proposal existed but ensemble rejected it
    - ensemble selected it but reliability rejected it
    - reliability passed but portfolio/risk blocked it
- Ensure campaign result schema includes:
  - primary summary
  - per-symbol diagnostics
  - per-strategy-per-symbol diagnostics
  - top rejection reasons
- Add tests that prove a multi-symbol campaign produces distinct symbol-level diagnostics even if only one symbol trades.

### Acceptance
- A completed campaign summary must show whether non-BTC symbols had:
  - no proposals
  - rejected proposals
  - ranked-out proposals
  - actual fills
- Only BTC traded must become an evidence-backed diagnosis, not a reporting ambiguity.

## Big Task 2: Build A Target Frequency Controller For 2-3 Trades/Day
Goal: stop treating trade count as an accidental output and instead make it a managed system target.

### Subtasks
- Add a trade-frequency controller with rolling realized opportunity counts.
  - Track:
    - proposals/day
    - selected signals/day
    - fills/day
    - trades/day
  - Compute these globally and by strategy.
- Define a target band:
  - preferred: `2-3` trades/day
  - soft floor: `1.5`
  - soft ceiling: `3.5`
- Add adaptive admission logic:
  - if rolling trades/day is too low:
    - slightly relax high-quality pullback and mean-reversion acceptance
    - slightly reduce no-trade band for top-ranked non-breakout setups
  - if rolling trades/day is too high:
    - tighten weakest families first, not globally
- Make this controller strategy-aware, not global:
  - prefer restoring `pullback` and `mean_reversion` flow before further loosening breakout
- Add explicit reporting:
  - target trades/day
  - realized trades/day
  - which strategies were loosened or tightened to maintain target band

### Acceptance
- The bot must be able to target `2-3` trades/day in simulation without relying on one strategy family.
- Trade count improvements must come from diversified opportunity flow, not just weaker breakout gating.

## Big Task 3: Rebuild Strategy Family Balance
Goal: convert the bot from mostly breakout into a three-engine system where each family has a clear job.

### Subtasks
- Trend breakout
  - Keep only high-conviction breakout continuation states.
  - Make confirmed-market breakouts the rarest and most expensive-to-admit path.
  - Add stronger post-shock / rebound throttles.
  - Penalize breakouts with low persistence, stretched entries, or insufficient volume confirmation.
- Trend pullback
  - Make pullback the primary continuation engine for reaching target daily trade count.
  - Add clearer shallow vs deep pullback profiles.
  - Prefer pullbacks that occur in aligned higher-timeframe trend and contained volatility.
  - Reduce overdependence on single-candle confirmation.
- Mean reversion
  - Keep it liquidity-conditioned and cost-aware.
  - Allow more exhaustion-reversal participation in non-trending, low-efficiency states.
  - Add stricter block against countertrend reversion into strong continuation regimes.
  - Add explicit distinction between:
    - passive fade near band extremes
    - reclaim/reversal entries after micro-turn confirmation

### Acceptance
- Completed campaigns should show all three strategy families generating proposals.
- Pullback and mean-reversion should materially contribute to trade count.
- Breakout should no longer dominate low-sample losing runs.

## Big Task 4: Add Regime-Conditioned Strategy Rotation
Goal: explicitly decide which strategy family should dominate under which market state.

### Subtasks
- Extend regime metadata with a rotation policy layer:
  - preferred family
  - suppressed family
  - confidence of rotation preference
- Build a small explicit matrix:
  - `trending + low crash risk` -> breakout or pullback favored
  - `trending + elevated crash risk` -> breakout suppressed, pullback favored
  - `choppy / low-efficiency / stretched` -> mean-reversion favored
  - `high_volatility directional` -> smaller breakout / selective pullback
  - `low_liquidity` -> all passive-only or no-trade
- Feed this matrix into:
  - ensemble ranking
  - reliability checks
  - risk sizing
- Add online learning overlays by `strategy x side x regime x symbol bucket`.
  - Let strong cells recover sooner.
  - Suppress degrading cells faster.

### Acceptance
- Campaign diagnostics must show regime-conditioned family mix.
- When breakout underperforms in a regime bucket, pullback/reversion should gain allocation instead of leaving the bot inactive.

## Big Task 5: Upgrade Portfolio Construction And No-Trade Region Logic
Goal: improve profit and win rate by making fewer marginal decisions and better cross-sectional choices.

### Subtasks
- Replace current best local score wins behavior with portfolio-aware opportunity admission.
- Add explicit no-trade region around marginal net expectancy:
  - after spread
  - slippage
  - event risk
  - liquidity penalty
  - crash penalty
  - current portfolio crowding
- Add cross-sectional preference for:
  - symbol diversification
  - lower correlation to current open risk
  - better strategy-family balance
  - stronger realized cell evidence
- Add opportunity throttling for crowded/duplicated trades:
  - same family
  - same directional factor
  - same correlated cluster
- Add realized post-cost expectancy monitoring by symbol and family.

### Acceptance
- The bot should accept more good different trades, not more duplicates of the same setup.
- Win rate and expectancy should improve because marginal opportunities are filtered more intentionally.

## Big Task 6: Make Learning Promote Winners Faster And Kill Losers Earlier
Goal: make the persistent learning layer behave more like an evolving trading brain.

### Subtasks
- Expand learning cells to track:
  - `strategy x side x regime x symbol cluster x order_type`
- Add asymmetric response:
  - losers get throttled sooner after enough negative evidence
  - winners get promoted sooner when positive evidence is stable and cost-adjusted
- Add symbol-cluster learning:
  - majors
  - high-beta alts
  - exchange-beta names
  - slower large-cap names
- Add missed opportunity learning:
  - if skipped high-quality setups repeatedly work, reduce the specific gate responsible
- Add stronger degradation controls:
  - if one family is causing most recent losses, force temporary rotation toward the next-best family

### Acceptance
- Learning should not just reduce trading.
- Learning must visibly support:
  - better family balance
  - faster recovery in good cells
  - fewer repeated mistakes in bad cells

## Big Task 7: Add Validation Discipline Around The Actual Target
Goal: optimize toward the stated target instead of reading one-off run outputs.

### Subtasks
- Define explicit objective hierarchy for tuning:
  1. preserve realism and no-lookahead
  2. reach `2-3` trades/day
  3. improve win rate materially
  4. improve profit materially
- Add campaign acceptance metrics:
  - trades/day
  - total return
  - win rate
  - profit factor
  - drawdown
  - strategy-family mix
  - symbol mix
- Add walk-forward slices:
  - 30d fast iteration
  - 60d confirmation
  - periodic 90d robustness run
- Add comparative reporting:
  - baseline vs candidate
  - best run vs median run
  - stability across repeated runs
- Add failure reason leaderboard after every campaign:
  - most common skip reasons
  - most common losing family
  - most common losing symbol
  - where expected edge diverged from realized edge

### Acceptance
- Every future upgrade can be judged from one summary without manual digging.
- We should be able to tell whether a candidate improved:
  - count
  - quality
  - diversification
  - robustness

## Public Interfaces / Output Changes
- `simulation-batch-summary` should include:
  - top signal sources
  - top skip reasons
  - per-symbol trade/signal stats
  - per-family trade/signal stats
- campaign result JSON should include:
  - `decision_diagnostics.by_symbol`
  - `decision_diagnostics.by_strategy_by_symbol`
  - rolling trade-frequency metrics
  - regime rotation diagnostics
- no live CLI compatibility should be broken:
  - `python -m trade_bot.main live`
  - `python -m trade_bot.cli status`

## Test Plan
- Multi-symbol campaign test proving non-BTC symbols are reported correctly even when they do not trade.
- Batch summary test covering:
  - signal mix
  - skip-reason aggregation
  - per-symbol aggregation
- Strategy tests:
  - breakout blocked under crash-risk regime
  - pullback admitted under continuation-friendly non-breakout regime
  - mean-reversion admitted in liquid low-efficiency stretched state
- Ensemble tests:
  - marginal post-cost trades rejected by no-trade region
  - diversified pullback/reversion beats weak breakout when scores are close
- Learning tests:
  - degrading cell throttles quickly
  - positive cell promotes quickly
  - missed-opportunity evidence reduces the right gate
- Validation tests:
  - campaign comparison includes trades/day and family/symbol mix

## Assumptions And Defaults
- Primary target is best achievable win rate with strong profit, while maintaining roughly `2-3` trades/day.
- We will prefer higher-quality diversified trades over forcing volume from breakout alone.
- Every coin we added should trade perfectly is interpreted as:
  - every configured symbol must be fully evaluated, reported, rankable, and tradable when qualified
  - not every symbol must trade every run
- We will continue using the sequential simulator as source of truth and will not weaken realism for speed.
- Fast iteration should use `30d` targeted campaigns.
- Confirmation should use `60d` multi-symbol campaigns.
- `90d` is reserved for robustness once the candidate is clearly improved.
