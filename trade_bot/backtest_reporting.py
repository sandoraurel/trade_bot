from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Dict


def build_backtest_report(
    *,
    symbol: str,
    timeframe: str,
    days: int,
    starting_balance: float,
    metrics: Dict[str, Any],
    assumptions: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    inputs = {
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days,
        "starting_balance": starting_balance,
    }
    fingerprint = hashlib.sha256(
        json.dumps(inputs, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema_version": 1,
        "run_fingerprint": fingerprint,
        "inputs": inputs,
        "assumptions": assumptions or {},
        "metrics": metrics,
    }


def save_backtest_report(path: str, report: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)


def build_campaign_comparison(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    metrics: list[str] | None = None,
) -> Dict[str, Any]:
    def _value(item: Dict[str, Any], key: str) -> Any:
        if key == "trades_per_day":
            trade_frequency = dict(item.get("trade_frequency", {}) or {})
            global_metrics = dict(trade_frequency.get("global", {}) or {})
            return global_metrics.get("trades_per_day", global_metrics.get("closed_trades_per_day"))
        if key == "strategy_family_mix":
            return dict(item.get("signals_by_strategy", {}) or {})
        if key == "symbol_mix":
            return dict(item.get("raw_signals_by_symbol", {}) or {})
        return item.get(key)

    def _mix_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, float]:
        keys = sorted(set(dict(before or {})) | set(dict(after or {})))
        return {
            str(key): float(dict(after or {}).get(key, 0.0) or 0.0) - float(dict(before or {}).get(key, 0.0) or 0.0)
            for key in keys
        }

    keys = metrics or [
        "num_trades",
        "trades_per_day",
        "raw_signals",
        "win_rate_pct",
        "total_return_pct",
        "profit_factor",
        "max_drawdown_pct",
        "avg_holding_minutes",
        "strategy_family_mix",
        "symbol_mix",
    ]
    comparison: Dict[str, Any] = {"metrics": {}, "acceptance": {}}
    for key in keys:
        before = _value(baseline, key)
        after = _value(candidate, key)
        delta = None
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            delta = after - before
        elif isinstance(before, dict) and isinstance(after, dict):
            delta = _mix_delta(before, after)
        comparison["metrics"][key] = {
            "baseline": before,
            "candidate": after,
            "delta": delta,
        }

    baseline_trades = float(baseline.get("num_trades", 0) or 0.0)
    candidate_trades = float(candidate.get("num_trades", 0) or 0.0)
    baseline_trades_per_day = float(_value(baseline, "trades_per_day") or 0.0)
    candidate_trades_per_day = float(_value(candidate, "trades_per_day") or 0.0)
    baseline_return = float(baseline.get("total_return_pct", 0.0) or 0.0)
    candidate_return = float(candidate.get("total_return_pct", 0.0) or 0.0)
    baseline_win_rate = float(baseline.get("win_rate_pct", 0.0) or 0.0)
    candidate_win_rate = float(candidate.get("win_rate_pct", 0.0) or 0.0)
    baseline_drawdown = float(baseline.get("max_drawdown_pct", 0.0) or 0.0)
    candidate_drawdown = float(candidate.get("max_drawdown_pct", 0.0) or 0.0)
    baseline_strategy_mix = dict(_value(baseline, "strategy_family_mix") or {})
    candidate_strategy_mix = dict(_value(candidate, "strategy_family_mix") or {})
    baseline_symbol_mix = dict(_value(baseline, "symbol_mix") or {})
    candidate_symbol_mix = dict(_value(candidate, "symbol_mix") or {})
    comparison["acceptance"] = {
        "more_trades": candidate_trades > baseline_trades,
        "trades_per_day_not_worse": candidate_trades_per_day >= baseline_trades_per_day,
        "better_profit": candidate_return > baseline_return,
        "win_rate_not_worse": candidate_win_rate >= baseline_win_rate,
        "drawdown_not_worse": candidate_drawdown >= baseline_drawdown,
        "strategy_mix_changed": baseline_strategy_mix != candidate_strategy_mix,
        "symbol_mix_changed": baseline_symbol_mix != candidate_symbol_mix,
        "passes_all": (
            candidate_trades > baseline_trades
            and candidate_trades_per_day >= baseline_trades_per_day
            and candidate_return > baseline_return
            and candidate_win_rate >= baseline_win_rate
            and candidate_drawdown >= baseline_drawdown
        ),
    }
    return comparison


def build_batch_summary(reports: list[Dict[str, Any]], *, include_horizons: bool = True) -> Dict[str, Any]:
    completed = [dict(report or {}) for report in reports if report]
    if not completed:
        return {"num_runs": 0, "status": "empty", "runs": []}
    numeric_keys = [
        "num_trades",
        "raw_signals",
        "win_rate_pct",
        "total_return_pct",
        "profit_factor",
        "max_drawdown_pct",
        "avg_holding_minutes",
    ]
    aggregates: Dict[str, Dict[str, Any]] = {}
    for key in numeric_keys:
        values = [float(item.get(key, 0.0) or 0.0) for item in completed if isinstance(item.get(key, 0.0), (int, float)) and math.isfinite(float(item.get(key, 0.0) or 0.0))]
        if not values:
            continue
        aggregates[key] = {
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1],
        }
    def _stddev(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    decision_totals: Dict[str, Dict[str, int]] = {
        "skip_reasons": {},
        "signals_by_strategy": {},
        "signals_by_order_type": {},
        "submitted_by_strategy": {},
        "submitted_by_order_type": {},
        "filled_by_strategy": {},
        "closed_by_strategy": {},
        "closed_by_order_type": {},
        "raw_signals_by_symbol": {},
        "submitted_by_symbol": {},
        "filled_by_symbol": {},
        "closed_by_symbol": {},
        "wins_by_symbol": {},
        "losses_by_symbol": {},
        "skipped_by_symbol": {},
        "family_rotation_counts": {},
        "family_rotation_recovery_counts": {},
        "learning_evidence_counts": {},
        "learning_asymmetry_counts": {},
        "missed_opportunity_relaxations": {},
        "reentry_cooldown_registrations": {},
        "realized_performance_penalty_counts": {},
        "realized_performance_no_trade_blocks": {},
        "limit_to_market_upgrades_by_strategy": {},
        "limit_queue_priority_assists_by_strategy": {},
        "limit_latency_reductions_by_strategy": {},
        "stale_market_escalations_by_strategy": {},
        "repriced_by_strategy": {},
        "touch_escalations_by_strategy": {},
        "partial_profit_takes_by_strategy": {},
        "replacement_candidates_by_strategy": {},
        "replacement_selected_by_strategy": {},
        "replacement_cross_symbol_selected_by_strategy": {},
        "replacement_submitted_by_strategy": {},
        "replacement_filled_by_strategy": {},
        "replacement_closed_by_strategy": {},
        "replacement_wins_by_strategy": {},
        "replacement_losses_by_strategy": {},
        "replacement_candidates_by_symbol": {},
        "replacement_rejections_by_reason": {},
        "replacement_near_misses_by_strategy": {},
        "replacement_near_misses_by_symbol": {},
        "replacement_near_misses_by_reason": {},
        "replacement_guard_blocks_by_reason": {},
    }
    execution_totals: Dict[str, int] = {
        "limit_to_market_upgrades": 0,
        "limit_queue_priority_assists": 0,
        "limit_latency_reductions": 0,
        "stale_market_escalations": 0,
        "repriced_orders": 0,
        "touch_escalations": 0,
        "partial_profit_takes": 0,
        "replacement_candidates_seen": 0,
        "replacement_candidates_selected": 0,
        "replacement_cross_symbol_selected": 0,
        "replacement_near_misses_seen": 0,
        "replacement_submitted": 0,
        "replacement_filled": 0,
        "replacement_closed": 0,
        "replacement_wins": 0,
        "replacement_losses": 0,
        "replacement_guard_blocks": 0,
    }
    acceptance_totals: Dict[str, int] = {}
    realized_performance: Dict[str, Dict[str, Dict[str, float]]] = {
        "by_symbol": {},
        "by_strategy": {},
    }
    exit_quality: Dict[str, Dict[str, Dict[str, float]]] = {
        "by_exit_reason": {},
        "giveback_by_strategy": {},
    }
    validation_harness: Dict[str, Any] = {
        "signal_to_submission_pct": [],
        "submission_to_fill_pct": [],
        "fill_to_close_pct": [],
        "stop_loss_negative_pl_share_pct": [],
        "repeated_setup_density_pct": [],
        "fresh_setup_block_density_pct": [],
        "triple_barrier_tp_first_pct": [],
        "triple_barrier_sl_first_pct": [],
        "triple_barrier_time_exit_pct": [],
        "repeated_setup_blocks": 0,
        "fresh_setup_blocks": 0,
        "triple_barrier_labels": 0,
        "pullback_meta_filter_blocks": 0,
        "candidate_flow_starved_runs": 0,
        "candidate_flow": {
            "proposals": 0,
            "raw_signals": 0,
            "submitted_orders": 0,
            "filled_orders": 0,
            "closed_trades": 0,
            "generation_outcomes": {},
            "pre_selection_rejections": {},
            "strategy_rejections": {},
            "post_selection_rejections": {},
        },
    }
    triple_barrier: Dict[str, Any] = {
        "label_counts": {},
        "label_counts_by_strategy": {},
        "label_counts_by_symbol": {},
    }
    market_data: Dict[str, Any] = {
        "datasets_loaded": 0,
        "datasets_missing": 0,
        "total_rows": 0,
        "runs_with_data": 0,
        "runs_without_data": 0,
    }
    family_rotation_by_strategy: Dict[str, Dict[str, int]] = {}
    family_rotation_recovery_by_strategy: Dict[str, Dict[str, int]] = {}
    learning_evidence_by_strategy: Dict[str, Dict[str, int]] = {}
    learning_asymmetry_by_strategy: Dict[str, Dict[str, int]] = {}
    missed_opportunity_relaxations_by_strategy: Dict[str, Dict[str, int]] = {}
    reentry_cooldown_registrations_by_strategy: Dict[str, Dict[str, int]] = {}
    family_rotation_soft_by_strategy: Dict[str, Dict[str, int]] = {}
    family_rotation_hard_by_strategy: Dict[str, Dict[str, int]] = {}
    duplicate_bucket_throttle_by_strategy: Dict[str, int] = {}
    weak_cluster_throttle_by_strategy: Dict[str, int] = {}
    realized_performance_penalty_by_strategy: Dict[str, Dict[str, int]] = {}
    realized_performance_no_trade_by_strategy: Dict[str, Dict[str, int]] = {}
    skip_reasons_by_symbol: Dict[str, Dict[str, int]] = {}
    expected_edge_by_symbol: Dict[str, Dict[str, float]] = {}
    universe_selection: Dict[str, Any] = {
        "eligible_symbols": {},
        "reserved_core_symbols": {},
        "rejected_symbols": {},
        "eligible_buckets": {},
        "rejected_buckets": {},
        "bucket_cap_rejections": {},
        "bucket_cap_rejections_by_bucket": {},
        "eligible_bucket_pressure": {},
        "realized_universe_promotions": {},
        "realized_universe_penalties": {},
        "realized_universe_adjustments_by_bucket": {},
        "realized_universe_vetoes": {},
        "realized_universe_vetoes_by_bucket": {},
    }
    for item in completed:
        diagnostics = dict(item.get("decision_diagnostics", {}) or {})
        campaign_summary = dict(item.get("campaign_summary", {}) or {})
        realized_payload = dict(campaign_summary.get("realized_performance", {}) or {})
        exit_quality_payload = dict(campaign_summary.get("exit_quality", {}) or {})
        validation_payload = dict(campaign_summary.get("validation_harness", {}) or {})
        universe_payload = dict(campaign_summary.get("universe_selection", {}) or {})
        acceptance_payload = dict(campaign_summary.get("acceptance", {}) or {})
        market_data_payload = dict(campaign_summary.get("market_data", {}) or {})
        symbol_rollups = dict(item.get("symbol_rollups", {}) or {})
        market_data["datasets_loaded"] += int(market_data_payload.get("datasets_loaded", 0) or 0)
        market_data["datasets_missing"] += int(market_data_payload.get("datasets_missing", 0) or 0)
        market_data["total_rows"] += int(market_data_payload.get("total_rows", 0) or 0)
        if bool(market_data_payload.get("data_available", False)):
            market_data["runs_with_data"] += 1
        elif market_data_payload:
            market_data["runs_without_data"] += 1
        for bucket in decision_totals:
            payload = dict(diagnostics.get(bucket, {}) or {})
            for key, value in payload.items():
                decision_totals[bucket][str(key)] = int(decision_totals[bucket].get(str(key), 0)) + int(value or 0)
        for key in execution_totals:
            execution_totals[key] += int(diagnostics.get(key, 0) or 0)
        validation_harness["repeated_setup_blocks"] += int(
            validation_payload.get("repeated_setup_blocks", diagnostics.get("repeated_setup_blocks", 0)) or 0
        )
        validation_harness["fresh_setup_blocks"] += int(
            validation_payload.get("fresh_setup_blocks", diagnostics.get("fresh_setup_blocks", 0)) or 0
        )
        validation_harness["triple_barrier_labels"] += int(validation_payload.get("triple_barrier_labels", 0) or 0)
        validation_harness["pullback_meta_filter_blocks"] += int(
            validation_payload.get("pullback_meta_filter_blocks", diagnostics.get("pullback_meta_filter_blocks", 0)) or 0
        )
        candidate_flow_payload = dict(validation_payload.get("candidate_flow", {}) or {})
        if bool(candidate_flow_payload.get("starved", False)):
            validation_harness["candidate_flow_starved_runs"] += 1
        candidate_flow_totals = validation_harness["candidate_flow"]
        for key in ("proposals", "raw_signals", "submitted_orders", "filled_orders", "closed_trades"):
            candidate_flow_totals[key] += int(candidate_flow_payload.get(key, 0) or 0)
        for key in ("generation_outcomes", "pre_selection_rejections", "strategy_rejections", "post_selection_rejections"):
            target = candidate_flow_totals[key]
            for reason, value in dict(candidate_flow_payload.get(key, {}) or {}).items():
                target[str(reason)] = int(target.get(str(reason), 0)) + int(value or 0)
        for key in (
            "signal_to_submission_pct",
            "submission_to_fill_pct",
            "fill_to_close_pct",
            "stop_loss_negative_pl_share_pct",
            "repeated_setup_density_pct",
            "fresh_setup_block_density_pct",
            "triple_barrier_tp_first_pct",
            "triple_barrier_sl_first_pct",
            "triple_barrier_time_exit_pct",
        ):
            value = validation_payload.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value or 0.0)):
                validation_harness[key].append(float(value or 0.0))
        triple_payload = dict(campaign_summary.get("triple_barrier", {}) or {})
        for label, value in dict(triple_payload.get("label_counts", {}) or {}).items():
            triple_barrier["label_counts"][str(label)] = int(triple_barrier["label_counts"].get(str(label), 0)) + int(value or 0)
        for strategy, counts in dict(triple_payload.get("label_counts_by_strategy", {}) or {}).items():
            target = triple_barrier["label_counts_by_strategy"].setdefault(str(strategy), {})
            for label, value in dict(counts or {}).items():
                target[str(label)] = int(target.get(str(label), 0)) + int(value or 0)
        for symbol, counts in dict(triple_payload.get("label_counts_by_symbol", {}) or {}).items():
            target = triple_barrier["label_counts_by_symbol"].setdefault(str(symbol), {})
            for label, value in dict(counts or {}).items():
                target[str(label)] = int(target.get(str(label), 0)) + int(value or 0)
        for key, value in dict(acceptance_payload.get("checks", {}) or {}).items():
            if bool(value):
                acceptance_totals[str(key)] = int(acceptance_totals.get(str(key), 0)) + 1
        if bool(acceptance_payload.get("passes_all", False)):
            acceptance_totals["passes_all"] = int(acceptance_totals.get("passes_all", 0)) + 1
        for bucket in ("by_symbol", "by_strategy"):
            payload = dict(realized_payload.get(bucket, {}) or {})
            for entity, metrics in payload.items():
                target = realized_performance[bucket].setdefault(
                    str(entity),
                    {"trades": 0.0, "wins": 0.0, "losses": 0.0, "total_pl": 0.0},
                )
                metric_payload = dict(metrics or {})
                target["trades"] += float(metric_payload.get("trades", 0.0) or 0.0)
                target["wins"] += float(metric_payload.get("wins", 0.0) or 0.0)
                target["losses"] += float(metric_payload.get("losses", 0.0) or 0.0)
                target["total_pl"] += float(metric_payload.get("total_pl", 0.0) or 0.0)
        exit_reason_payload = dict(exit_quality_payload.get("by_exit_reason", {}) or {})
        for exit_reason, metrics in exit_reason_payload.items():
            target = exit_quality["by_exit_reason"].setdefault(
                str(exit_reason),
                {
                    "trades": 0.0,
                    "wins": 0.0,
                    "losses": 0.0,
                    "total_pl": 0.0,
                    "total_mfe_r": 0.0,
                    "total_mae_r": 0.0,
                    "total_giveback_r": 0.0,
                },
            )
            metric_payload = dict(metrics or {})
            target["trades"] += float(metric_payload.get("trades", 0.0) or 0.0)
            target["wins"] += float(metric_payload.get("wins", 0.0) or 0.0)
            target["losses"] += float(metric_payload.get("losses", 0.0) or 0.0)
            target["total_pl"] += float(metric_payload.get("total_pl", 0.0) or 0.0)
            target["total_mfe_r"] += float(metric_payload.get("total_mfe_r", 0.0) or 0.0)
            target["total_mae_r"] += float(metric_payload.get("total_mae_r", 0.0) or 0.0)
            target["total_giveback_r"] += float(metric_payload.get("total_giveback_r", 0.0) or 0.0)
        giveback_payload = dict(exit_quality_payload.get("giveback_by_strategy", {}) or {})
        for strategy, metrics in giveback_payload.items():
            target = exit_quality["giveback_by_strategy"].setdefault(
                str(strategy),
                {
                    "trades": 0.0,
                    "total_giveback_r": 0.0,
                    "mfe_above_1r_count": 0.0,
                    "gave_back_below_0_25r_count": 0.0,
                },
            )
            metric_payload = dict(metrics or {})
            target["trades"] += float(metric_payload.get("trades", 0.0) or 0.0)
            target["total_giveback_r"] += float(metric_payload.get("total_giveback_r", 0.0) or 0.0)
            target["mfe_above_1r_count"] += float(metric_payload.get("mfe_above_1r_count", 0.0) or 0.0)
            target["gave_back_below_0_25r_count"] += float(metric_payload.get("gave_back_below_0_25r_count", 0.0) or 0.0)
        strategy_rotation_payload = dict(diagnostics.get("family_rotation_counts_by_strategy", {}) or {})
        for strategy, reasons in strategy_rotation_payload.items():
            target = family_rotation_by_strategy.setdefault(str(strategy), {})
            for key, value in dict(reasons or {}).items():
                reason_key = str(key)
                target[reason_key] = int(target.get(reason_key, 0)) + int(value or 0)
                severity_target = family_rotation_hard_by_strategy if reason_key.endswith("_hard") else family_rotation_soft_by_strategy
                severity_bucket = severity_target.setdefault(str(strategy), {})
                severity_bucket[reason_key] = int(severity_bucket.get(reason_key, 0)) + int(value or 0)
        strategy_recovery_payload = dict(diagnostics.get("family_rotation_recovery_counts_by_strategy", {}) or {})
        for strategy, reasons in strategy_recovery_payload.items():
            target = family_rotation_recovery_by_strategy.setdefault(str(strategy), {})
            for key, value in dict(reasons or {}).items():
                target[str(key)] = int(target.get(str(key), 0)) + int(value or 0)
        learning_evidence_payload = dict(diagnostics.get("learning_evidence_counts_by_strategy", {}) or {})
        for strategy, reasons in learning_evidence_payload.items():
            target = learning_evidence_by_strategy.setdefault(str(strategy), {})
            for key, value in dict(reasons or {}).items():
                target[str(key)] = int(target.get(str(key), 0)) + int(value or 0)
        learning_asymmetry_payload = dict(diagnostics.get("learning_asymmetry_counts_by_strategy", {}) or {})
        for strategy, reasons in learning_asymmetry_payload.items():
            target = learning_asymmetry_by_strategy.setdefault(str(strategy), {})
            for key, value in dict(reasons or {}).items():
                target[str(key)] = int(target.get(str(key), 0)) + int(value or 0)
        missed_opportunity_payload = dict(diagnostics.get("missed_opportunity_relaxations_by_strategy", {}) or {})
        for strategy, reasons in missed_opportunity_payload.items():
            target = missed_opportunity_relaxations_by_strategy.setdefault(str(strategy), {})
            for key, value in dict(reasons or {}).items():
                target[str(key)] = int(target.get(str(key), 0)) + int(value or 0)
        reentry_cooldown_payload = dict(diagnostics.get("reentry_cooldown_registrations_by_strategy", {}) or {})
        for strategy, reasons in reentry_cooldown_payload.items():
            target = reentry_cooldown_registrations_by_strategy.setdefault(str(strategy), {})
            for key, value in dict(reasons or {}).items():
                target[str(key)] = int(target.get(str(key), 0)) + int(value or 0)
        realized_penalty_payload = dict(diagnostics.get("realized_performance_penalty_counts_by_strategy", {}) or {})
        for strategy, reasons in realized_penalty_payload.items():
            target = realized_performance_penalty_by_strategy.setdefault(str(strategy), {})
            for key, value in dict(reasons or {}).items():
                target[str(key)] = int(target.get(str(key), 0)) + int(value or 0)
        realized_no_trade_payload = dict(diagnostics.get("realized_performance_no_trade_blocks_by_strategy", {}) or {})
        for strategy, reasons in realized_no_trade_payload.items():
            target = realized_performance_no_trade_by_strategy.setdefault(str(strategy), {})
            for key, value in dict(reasons or {}).items():
                target[str(key)] = int(target.get(str(key), 0)) + int(value or 0)
        strategy_skip_payload = dict(diagnostics.get("skip_reasons_by_strategy_by_symbol", {}) or {})
        for symbol_counts in strategy_skip_payload.values():
            for strategy, reasons in dict(symbol_counts or {}).items():
                bucket_count = int(dict(reasons or {}).get("portfolio_duplicate_bucket_throttle", 0) or 0)
                if bucket_count > 0:
                    duplicate_bucket_throttle_by_strategy[str(strategy)] = int(duplicate_bucket_throttle_by_strategy.get(str(strategy), 0)) + bucket_count
                weak_count = int(dict(reasons or {}).get("portfolio_persistently_weak_cluster_throttle", 0) or 0)
                if weak_count > 0:
                    weak_cluster_throttle_by_strategy[str(strategy)] = int(weak_cluster_throttle_by_strategy.get(str(strategy), 0)) + weak_count
        symbol_payload = dict(diagnostics.get("skip_reasons_by_symbol", {}) or {})
        for symbol, reasons in symbol_payload.items():
            target = skip_reasons_by_symbol.setdefault(str(symbol), {})
            for key, value in dict(reasons or {}).items():
                target[str(key)] = int(target.get(str(key), 0)) + int(value or 0)
        for symbol, payload in symbol_rollups.items():
            trades = float(dict(payload or {}).get("trades", 0.0) or 0.0)
            avg_edge_bps = float(dict(payload or {}).get("avg_edge_bps", 0.0) or 0.0)
            target = expected_edge_by_symbol.setdefault(
                str(symbol),
                {"trades": 0.0, "weighted_edge_bps": 0.0},
            )
            target["trades"] += trades
            target["weighted_edge_bps"] += avg_edge_bps * max(trades, 0.0)
        for symbol in list(universe_payload.get("eligible_symbols", []) or []):
            universe_selection["eligible_symbols"][str(symbol)] = int(universe_selection["eligible_symbols"].get(str(symbol), 0)) + 1
            scored_payload = dict(universe_payload.get("scored_symbols", {}) or {}).get(str(symbol), {}) or {}
            bucket = str(dict(scored_payload).get("bucket", "other") or "other")
            universe_selection["eligible_buckets"][bucket] = int(universe_selection["eligible_buckets"].get(bucket, 0)) + 1
            realized_adjustment = float(dict(scored_payload).get("realized_score_adjustment", 0.0) or 0.0)
            if realized_adjustment > 0.0:
                universe_selection["realized_universe_promotions"][str(symbol)] = int(universe_selection["realized_universe_promotions"].get(str(symbol), 0)) + 1
                universe_selection["realized_universe_adjustments_by_bucket"][bucket] = int(universe_selection["realized_universe_adjustments_by_bucket"].get(bucket, 0)) + 1
            elif realized_adjustment < 0.0:
                universe_selection["realized_universe_penalties"][str(symbol)] = int(universe_selection["realized_universe_penalties"].get(str(symbol), 0)) + 1
                universe_selection["realized_universe_adjustments_by_bucket"][bucket] = int(universe_selection["realized_universe_adjustments_by_bucket"].get(bucket, 0)) - 1
        for symbol in list(universe_payload.get("reserved_core_symbols", []) or []):
            universe_selection["reserved_core_symbols"][str(symbol)] = int(universe_selection["reserved_core_symbols"].get(str(symbol), 0)) + 1
        for bucket, value in dict(universe_payload.get("eligible_bucket_counts", {}) or {}).items():
            bucket_key = str(bucket or "other")
            universe_selection["eligible_bucket_pressure"][bucket_key] = int(universe_selection["eligible_bucket_pressure"].get(bucket_key, 0)) + int(value or 0)
        for symbol, payload in dict(universe_payload.get("rejected_symbols", {}) or {}).items():
            reason = str(dict(payload or {}).get("reason", "unknown") or "unknown")
            key = f"{symbol}:{reason}"
            universe_selection["rejected_symbols"][key] = int(universe_selection["rejected_symbols"].get(key, 0)) + 1
            bucket = str(dict(payload or {}).get("bucket", "other") or "other")
            universe_selection["rejected_buckets"][bucket] = int(universe_selection["rejected_buckets"].get(bucket, 0)) + 1
            if reason == "bucket_cap_reached":
                universe_selection["bucket_cap_rejections"][str(symbol)] = int(universe_selection["bucket_cap_rejections"].get(str(symbol), 0)) + 1
                universe_selection["bucket_cap_rejections_by_bucket"][bucket] = int(universe_selection["bucket_cap_rejections_by_bucket"].get(bucket, 0)) + 1
            if reason == "realized_negative_expectancy_veto":
                universe_selection["realized_universe_vetoes"][str(symbol)] = int(universe_selection["realized_universe_vetoes"].get(str(symbol), 0)) + 1
                universe_selection["realized_universe_vetoes_by_bucket"][bucket] = int(universe_selection["realized_universe_vetoes_by_bucket"].get(bucket, 0)) + 1
    best_run = max(completed, key=lambda item: float(item.get("total_return_pct", float("-inf")) or float("-inf")))
    worst_run = min(completed, key=lambda item: float(item.get("total_return_pct", float("inf")) or float("inf")))
    baseline_run = completed[0]
    latest_run = completed[-1]
    median_index = len(completed) // 2
    median_run = sorted(
        completed,
        key=lambda item: float(item.get("total_return_pct", 0.0) or 0.0),
    )[median_index]
    baseline_vs_latest = build_campaign_comparison(baseline_run, latest_run)
    baseline_vs_median = build_campaign_comparison(baseline_run, median_run)
    return_values = [float(item.get("total_return_pct", 0.0) or 0.0) for item in completed]
    trades_values = [float(item.get("num_trades", 0.0) or 0.0) for item in completed]
    win_rate_values = [float(item.get("win_rate_pct", 0.0) or 0.0) for item in completed]
    stability = {
        "return_range_pct": (max(return_values) - min(return_values)) if return_values else 0.0,
        "return_stddev_pct": _stddev(return_values),
        "trade_stddev": _stddev(trades_values),
        "win_rate_stddev_pct": _stddev(win_rate_values),
        "best_vs_median_return_gap_pct": float(best_run.get("total_return_pct", 0.0) or 0.0) - float(median_run.get("total_return_pct", 0.0) or 0.0),
        "latest_vs_median_return_gap_pct": float(latest_run.get("total_return_pct", 0.0) or 0.0) - float(median_run.get("total_return_pct", 0.0) or 0.0),
    }
    for bucket in realized_performance.values():
        for payload in bucket.values():
            trades = float(payload.get("trades", 0.0) or 0.0)
            wins = float(payload.get("wins", 0.0) or 0.0)
            payload["expectancy"] = float(payload.get("total_pl", 0.0) or 0.0) / trades if trades else 0.0
            payload["win_rate_pct"] = (wins / trades * 100.0) if trades else 0.0
    for payload in exit_quality["by_exit_reason"].values():
        trades = float(payload.get("trades", 0.0) or 0.0)
        wins = float(payload.get("wins", 0.0) or 0.0)
        payload["expectancy"] = float(payload.get("total_pl", 0.0) or 0.0) / trades if trades else 0.0
        payload["win_rate_pct"] = (wins / trades * 100.0) if trades else 0.0
        payload["avg_mfe_r"] = float(payload.get("total_mfe_r", 0.0) or 0.0) / trades if trades else 0.0
        payload["avg_mae_r"] = float(payload.get("total_mae_r", 0.0) or 0.0) / trades if trades else 0.0
        payload["avg_giveback_r"] = float(payload.get("total_giveback_r", 0.0) or 0.0) / trades if trades else 0.0
    for payload in exit_quality["giveback_by_strategy"].values():
        trades = float(payload.get("trades", 0.0) or 0.0)
        payload["avg_giveback_r"] = float(payload.get("total_giveback_r", 0.0) or 0.0) / trades if trades else 0.0
    exit_reason_groups: Dict[str, Dict[str, float]] = {}

    def _exit_group(reason: str) -> str:
        reason_upper = str(reason or "UNKNOWN").upper()
        if reason_upper == "SL":
            return "stop_loss"
        if "PROFIT_PROTECT" in reason_upper:
            return "profit_protection"
        if reason_upper == "TP":
            return "take_profit"
        if (
            "THESIS_FAIL" in reason_upper
            or "STRUCTURE_FAIL" in reason_upper
            or "RECLAIM_FAIL" in reason_upper
            or "FOLLOW_THROUGH_FAIL" in reason_upper
        ):
            return "thesis_failure"
        if reason_upper.startswith("TIME_"):
            return "time_stop"
        if "TRAIL" in reason_upper:
            return "trailing"
        return "other"

    for reason, payload in exit_quality["by_exit_reason"].items():
        group = _exit_group(reason)
        target = exit_reason_groups.setdefault(
            group,
            {"trades": 0.0, "wins": 0.0, "losses": 0.0, "total_pl": 0.0, "negative_pl": 0.0},
        )
        trades = float(payload.get("trades", 0.0) or 0.0)
        total_pl = float(payload.get("total_pl", 0.0) or 0.0)
        target["trades"] += trades
        target["wins"] += float(payload.get("wins", 0.0) or 0.0)
        target["losses"] += float(payload.get("losses", 0.0) or 0.0)
        target["total_pl"] += total_pl
        if total_pl < 0.0:
            target["negative_pl"] += abs(total_pl)
    total_negative_exit_pl = sum(float(payload.get("negative_pl", 0.0) or 0.0) for payload in exit_reason_groups.values())
    for payload in exit_reason_groups.values():
        trades = float(payload.get("trades", 0.0) or 0.0)
        wins = float(payload.get("wins", 0.0) or 0.0)
        payload["expectancy"] = float(payload.get("total_pl", 0.0) or 0.0) / trades if trades else 0.0
        payload["win_rate_pct"] = (wins / trades * 100.0) if trades else 0.0
        payload["negative_pl_share_pct"] = (
            float(payload.get("negative_pl", 0.0) or 0.0) / total_negative_exit_pl * 100.0
        ) if total_negative_exit_pl > 0.0 else 0.0
    exit_quality["by_exit_group"] = {
        key: dict(sorted(value.items()))
        for key, value in sorted(exit_reason_groups.items())
    }
    failure_leaderboard: Dict[str, Any] = {
        "top_skip_reasons": dict(sorted(decision_totals["skip_reasons"].items(), key=lambda item: item[1], reverse=True)[:5]),
        "losing_families": {},
        "losing_symbols": {},
        "expected_vs_realized_edge_divergence": {},
    }
    losing_families = {
        strategy: payload
        for strategy, payload in realized_performance["by_strategy"].items()
        if float(dict(payload or {}).get("expectancy", 0.0) or 0.0) < 0.0
    }
    losing_symbols = {
        symbol: payload
        for symbol, payload in realized_performance["by_symbol"].items()
        if float(dict(payload or {}).get("expectancy", 0.0) or 0.0) < 0.0
    }
    failure_leaderboard["losing_families"] = {
        strategy: {
            "expectancy": float(dict(payload or {}).get("expectancy", 0.0) or 0.0),
            "trades": float(dict(payload or {}).get("trades", 0.0) or 0.0),
            "win_rate_pct": float(dict(payload or {}).get("win_rate_pct", 0.0) or 0.0),
        }
        for strategy, payload in sorted(
            losing_families.items(),
            key=lambda item: (float(dict(item[1] or {}).get("expectancy", 0.0) or 0.0), -float(dict(item[1] or {}).get("trades", 0.0) or 0.0)),
        )[:5]
    }
    failure_leaderboard["losing_symbols"] = {
        symbol: {
            "expectancy": float(dict(payload or {}).get("expectancy", 0.0) or 0.0),
            "trades": float(dict(payload or {}).get("trades", 0.0) or 0.0),
            "win_rate_pct": float(dict(payload or {}).get("win_rate_pct", 0.0) or 0.0),
        }
        for symbol, payload in sorted(
            losing_symbols.items(),
            key=lambda item: (float(dict(item[1] or {}).get("expectancy", 0.0) or 0.0), -float(dict(item[1] or {}).get("trades", 0.0) or 0.0)),
        )[:5]
    }
    divergence_rows: Dict[str, Dict[str, float | str]] = {}
    for symbol, edge_payload in expected_edge_by_symbol.items():
        realized_payload = dict(realized_performance["by_symbol"].get(symbol, {}) or {})
        trades = float(edge_payload.get("trades", 0.0) or 0.0)
        if trades <= 0.0:
            continue
        avg_edge_bps = float(edge_payload.get("weighted_edge_bps", 0.0) or 0.0) / trades
        realized_expectancy = float(realized_payload.get("expectancy", 0.0) or 0.0)
        if avg_edge_bps <= 0.0 or realized_expectancy >= 0.0:
            continue
        divergence_rows[symbol] = {
            "avg_edge_bps": avg_edge_bps,
            "realized_expectancy": realized_expectancy,
            "trades": trades,
            "divergence_score": avg_edge_bps * abs(realized_expectancy) * max(trades, 1.0),
            "direction": "positive_expected_negative_realized",
        }
    failure_leaderboard["expected_vs_realized_edge_divergence"] = {
        symbol: {
            "avg_edge_bps": float(dict(payload or {}).get("avg_edge_bps", 0.0) or 0.0),
            "realized_expectancy": float(dict(payload or {}).get("realized_expectancy", 0.0) or 0.0),
            "trades": float(dict(payload or {}).get("trades", 0.0) or 0.0),
            "direction": str(dict(payload or {}).get("direction", "")),
        }
        for symbol, payload in sorted(
            divergence_rows.items(),
            key=lambda item: float(dict(item[1] or {}).get("divergence_score", 0.0) or 0.0),
            reverse=True,
        )[:5]
    }
    by_horizon: Dict[str, Dict[str, Any]] = {}
    walk_forward: Dict[str, Any] = {}
    if include_horizons:
        horizon_groups: Dict[str, list[Dict[str, Any]]] = {}
        for item in completed:
            campaign_summary = dict(item.get("campaign_summary", {}) or {})
            session = dict(campaign_summary.get("session", {}) or {})
            days = int(session.get("days", item.get("days", 0)) or 0)
            if days > 0:
                horizon_groups.setdefault(f"{days}d", []).append(item)
        for label, runs in sorted(horizon_groups.items()):
            by_horizon[label] = build_batch_summary(runs, include_horizons=False)
        horizon_labels = sorted(
            by_horizon.keys(),
            key=lambda label: int(str(label).rstrip("d") or 0),
        )
        if len(horizon_labels) >= 2:
            iteration_label = horizon_labels[0]
            confirmation_label = horizon_labels[-1]
            iteration_summary = dict(by_horizon.get(iteration_label, {}) or {})
            confirmation_summary = dict(by_horizon.get(confirmation_label, {}) or {})
            iteration_aggregates = dict(iteration_summary.get("aggregates", {}) or {})
            confirmation_aggregates = dict(confirmation_summary.get("aggregates", {}) or {})

            def _avg(payload: Dict[str, Any], key: str) -> float:
                return float(dict(payload.get(key, {}) or {}).get("avg", 0.0) or 0.0)

            iteration_return = _avg(iteration_aggregates, "total_return_pct")
            confirmation_return = _avg(confirmation_aggregates, "total_return_pct")
            iteration_trades = _avg(iteration_aggregates, "num_trades")
            confirmation_trades = _avg(confirmation_aggregates, "num_trades")
            iteration_win_rate = _avg(iteration_aggregates, "win_rate_pct")
            confirmation_win_rate = _avg(confirmation_aggregates, "win_rate_pct")
            iteration_label_days = max(int(iteration_label.rstrip("d") or 0), 1)
            confirmation_label_days = max(int(confirmation_label.rstrip("d") or 0), 1)
            iteration_trades_per_day = iteration_trades / iteration_label_days
            confirmation_trades_per_day = confirmation_trades / confirmation_label_days
            walk_forward = {
                "iteration_horizon": iteration_label,
                "confirmation_horizon": confirmation_label,
                "metrics": {
                    "trades_per_day": {
                        "iteration": iteration_trades_per_day,
                        "confirmation": confirmation_trades_per_day,
                        "delta": confirmation_trades_per_day - iteration_trades_per_day,
                    },
                    "total_return_pct": {
                        "iteration": iteration_return,
                        "confirmation": confirmation_return,
                        "delta": confirmation_return - iteration_return,
                    },
                    "win_rate_pct": {
                        "iteration": iteration_win_rate,
                        "confirmation": confirmation_win_rate,
                        "delta": confirmation_win_rate - iteration_win_rate,
                    },
                },
                "acceptance": {
                    "confirmation_trades_per_day_not_worse": confirmation_trades_per_day >= iteration_trades_per_day,
                    "confirmation_return_not_worse": confirmation_return >= iteration_return,
                    "confirmation_win_rate_not_worse": confirmation_win_rate >= iteration_win_rate,
                    "passes_all": (
                        confirmation_trades_per_day >= iteration_trades_per_day
                        and confirmation_return >= iteration_return
                        and confirmation_win_rate >= iteration_win_rate
                    ),
                },
            }
    validation_harness_summary = {
        "signal_to_submission_pct": (sum(validation_harness["signal_to_submission_pct"]) / len(validation_harness["signal_to_submission_pct"])) if validation_harness["signal_to_submission_pct"] else 0.0,
        "submission_to_fill_pct": (sum(validation_harness["submission_to_fill_pct"]) / len(validation_harness["submission_to_fill_pct"])) if validation_harness["submission_to_fill_pct"] else 0.0,
        "fill_to_close_pct": (sum(validation_harness["fill_to_close_pct"]) / len(validation_harness["fill_to_close_pct"])) if validation_harness["fill_to_close_pct"] else 0.0,
        "stop_loss_negative_pl_share_pct": (sum(validation_harness["stop_loss_negative_pl_share_pct"]) / len(validation_harness["stop_loss_negative_pl_share_pct"])) if validation_harness["stop_loss_negative_pl_share_pct"] else 0.0,
        "repeated_setup_density_pct": (sum(validation_harness["repeated_setup_density_pct"]) / len(validation_harness["repeated_setup_density_pct"])) if validation_harness["repeated_setup_density_pct"] else 0.0,
        "fresh_setup_block_density_pct": (sum(validation_harness["fresh_setup_block_density_pct"]) / len(validation_harness["fresh_setup_block_density_pct"])) if validation_harness["fresh_setup_block_density_pct"] else 0.0,
        "triple_barrier_tp_first_pct": (sum(validation_harness["triple_barrier_tp_first_pct"]) / len(validation_harness["triple_barrier_tp_first_pct"])) if validation_harness["triple_barrier_tp_first_pct"] else 0.0,
        "triple_barrier_sl_first_pct": (sum(validation_harness["triple_barrier_sl_first_pct"]) / len(validation_harness["triple_barrier_sl_first_pct"])) if validation_harness["triple_barrier_sl_first_pct"] else 0.0,
        "triple_barrier_time_exit_pct": (sum(validation_harness["triple_barrier_time_exit_pct"]) / len(validation_harness["triple_barrier_time_exit_pct"])) if validation_harness["triple_barrier_time_exit_pct"] else 0.0,
        "repeated_setup_blocks": int(validation_harness["repeated_setup_blocks"] or 0),
        "fresh_setup_blocks": int(validation_harness["fresh_setup_blocks"] or 0),
        "triple_barrier_labels": int(validation_harness["triple_barrier_labels"] or 0),
        "pullback_meta_filter_blocks": int(validation_harness["pullback_meta_filter_blocks"] or 0),
    }
    candidate_flow_totals = dict(validation_harness.get("candidate_flow", {}) or {})
    candidate_flow_blockers = {
        **{
            f"pre:{key}": value
            for key, value in dict(candidate_flow_totals.get("pre_selection_rejections", {}) or {}).items()
        },
        **{
            f"strategy:{key}": value
            for key, value in dict(candidate_flow_totals.get("strategy_rejections", {}) or {}).items()
        },
        **{
            f"post:{key}": value
            for key, value in dict(candidate_flow_totals.get("post_selection_rejections", {}) or {}).items()
        },
    }
    validation_harness_summary["candidate_flow"] = {
        "starved_runs": int(validation_harness.get("candidate_flow_starved_runs", 0) or 0),
        "starved_run_pct": (
            int(validation_harness.get("candidate_flow_starved_runs", 0) or 0) / len(completed) * 100.0
        ) if completed else 0.0,
        "proposals": int(candidate_flow_totals.get("proposals", 0) or 0),
        "raw_signals": int(candidate_flow_totals.get("raw_signals", 0) or 0),
        "submitted_orders": int(candidate_flow_totals.get("submitted_orders", 0) or 0),
        "filled_orders": int(candidate_flow_totals.get("filled_orders", 0) or 0),
        "closed_trades": int(candidate_flow_totals.get("closed_trades", 0) or 0),
        "generation_outcomes": dict(sorted(dict(candidate_flow_totals.get("generation_outcomes", {}) or {}).items())),
        "top_pre_selection_rejections": dict(sorted(dict(candidate_flow_totals.get("pre_selection_rejections", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:8]),
        "top_strategy_rejections": dict(sorted(dict(candidate_flow_totals.get("strategy_rejections", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:8]),
        "top_post_selection_rejections": dict(sorted(dict(candidate_flow_totals.get("post_selection_rejections", {}) or {}).items(), key=lambda item: item[1], reverse=True)[:8]),
        "top_blockers": dict(sorted(candidate_flow_blockers.items(), key=lambda item: item[1], reverse=True)[:8]),
    }
    avg_trades = float(dict(aggregates.get("num_trades", {}) or {}).get("avg", 0.0) or 0.0)
    avg_return = float(dict(aggregates.get("total_return_pct", {}) or {}).get("avg", 0.0) or 0.0)
    avg_win_rate = float(dict(aggregates.get("win_rate_pct", {}) or {}).get("avg", 0.0) or 0.0)
    run_days = [
        float(dict(dict(item.get("campaign_summary", {}) or {}).get("session", {}) or {}).get("days", item.get("days", 0)) or 0.0)
        for item in completed
    ]
    avg_days = sum(day for day in run_days if day > 0.0) / max(sum(1 for day in run_days if day > 0.0), 1)
    avg_trades_per_day = avg_trades / avg_days if avg_days > 0.0 else 0.0
    target_trades_per_day_min = 2.0
    target_stop_loss_share = 55.0
    target_signal_to_submission = 18.0
    target_submission_to_fill = 45.0
    target_win_rate = 35.0
    candidate_flow_summary = dict(validation_harness_summary.get("candidate_flow", {}) or {})
    candidate_flow_starved_runs = int(candidate_flow_summary.get("starved_runs", 0) or 0)
    candidate_flow_starved_pct = float(candidate_flow_summary.get("starved_run_pct", 0.0) or 0.0)
    candidate_flow_raw_signals = int(candidate_flow_summary.get("raw_signals", 0) or 0)
    candidate_flow_closed_trades = int(candidate_flow_summary.get("closed_trades", 0) or 0)
    has_candidate_flow_samples = bool(
        candidate_flow_summary
        and (
            candidate_flow_starved_runs > 0
            or int(candidate_flow_summary.get("proposals", 0) or 0) > 0
            or candidate_flow_raw_signals > 0
            or int(candidate_flow_summary.get("submitted_orders", 0) or 0) > 0
            or int(candidate_flow_summary.get("filled_orders", 0) or 0) > 0
            or candidate_flow_closed_trades > 0
        )
    )
    min_flow_trades_per_day = 0.20
    min_flow_raw_signals_per_run = 1.0
    flow_recovered = (
        has_candidate_flow_samples
        and
        candidate_flow_starved_runs <= 0
        and candidate_flow_raw_signals >= max(len(completed) * min_flow_raw_signals_per_run, 1.0)
        and avg_trades_per_day >= min_flow_trades_per_day
    )
    flow_starved = has_candidate_flow_samples and (candidate_flow_starved_runs > 0 or candidate_flow_raw_signals <= 0)
    readiness_components = {
        "positive_return": 1.0 if avg_return > 0.0 else max(0.0, min(1.0, 1.0 + (avg_return / 2.0))),
        "trade_frequency": max(0.0, min(avg_trades_per_day / target_trades_per_day_min, 1.0)),
        "win_rate": max(0.0, min(avg_win_rate / target_win_rate, 1.0)),
        "stop_loss_damage": max(0.0, min((100.0 - float(validation_harness_summary.get("stop_loss_negative_pl_share_pct", 0.0) or 0.0)) / (100.0 - target_stop_loss_share), 1.0)),
        "signal_to_submission": max(0.0, min(float(validation_harness_summary.get("signal_to_submission_pct", 0.0) or 0.0) / target_signal_to_submission, 1.0)),
        "submission_to_fill": max(0.0, min(float(validation_harness_summary.get("submission_to_fill_pct", 0.0) or 0.0) / target_submission_to_fill, 1.0)),
        "candidate_flow": 0.0 if flow_starved else max(0.0, min(avg_trades_per_day / min_flow_trades_per_day, 1.0)),
    }
    base_readiness_score = (
        readiness_components["positive_return"] * 0.30
        + readiness_components["stop_loss_damage"] * 0.22
        + readiness_components["win_rate"] * 0.18
        + readiness_components["trade_frequency"] * 0.14
        + readiness_components["signal_to_submission"] * 0.08
        + readiness_components["submission_to_fill"] * 0.08
    ) * 100.0
    readiness_penalty = 0.0
    if flow_starved:
        readiness_penalty += 12.0
    if candidate_flow_starved_pct >= 50.0:
        readiness_penalty += 8.0
    readiness_score = max(base_readiness_score - readiness_penalty, 0.0)
    if flow_starved:
        candidate_flow_status = "candidate_flow_starved"
    elif not has_candidate_flow_samples:
        candidate_flow_status = "candidate_flow_not_measured"
    elif avg_return <= 0.0 or avg_win_rate < target_win_rate:
        candidate_flow_status = "flow_recovered_but_unprofitable"
    elif avg_trades_per_day < target_trades_per_day_min:
        candidate_flow_status = "flow_recovered_below_goal_frequency"
    else:
        candidate_flow_status = "flow_healthy"
    validation_targets = {
        "readiness_score": readiness_score,
        "base_readiness_score": base_readiness_score,
        "readiness_penalty": readiness_penalty,
        "readiness_components": readiness_components,
        "candidate_flow_status": candidate_flow_status,
        "flow_recovered": flow_recovered,
        "flow_starved": flow_starved,
        "candidate_flow_measured": has_candidate_flow_samples,
        "min_flow_trades_per_day": min_flow_trades_per_day,
        "min_flow_raw_signals_per_run": min_flow_raw_signals_per_run,
        "candidate_flow_raw_signals": candidate_flow_raw_signals,
        "candidate_flow_closed_trades": candidate_flow_closed_trades,
        "candidate_flow_starved_runs": candidate_flow_starved_runs,
        "avg_trades_per_day": avg_trades_per_day,
        "target_trades_per_day_min": target_trades_per_day_min,
        "target_trades_per_day_max": 3.0,
        "trade_frequency_gap_to_min": max(target_trades_per_day_min - avg_trades_per_day, 0.0),
        "return_gap_to_positive_pct": max(0.0 - avg_return, 0.0),
        "win_rate_gap_to_floor_pct": max(target_win_rate - avg_win_rate, 0.0),
        "stop_loss_share_gap_to_limit_pct": max(float(validation_harness_summary.get("stop_loss_negative_pl_share_pct", 0.0) or 0.0) - target_stop_loss_share, 0.0),
        "signal_to_submission_gap_to_floor_pct": max(target_signal_to_submission - float(validation_harness_summary.get("signal_to_submission_pct", 0.0) or 0.0), 0.0),
        "submission_to_fill_gap_to_floor_pct": max(target_submission_to_fill - float(validation_harness_summary.get("submission_to_fill_pct", 0.0) or 0.0), 0.0),
    }
    blockers = {
        "profitability": validation_targets["return_gap_to_positive_pct"],
        "stop_loss_damage": validation_targets["stop_loss_share_gap_to_limit_pct"],
        "win_rate": validation_targets["win_rate_gap_to_floor_pct"],
        "trade_frequency": validation_targets["trade_frequency_gap_to_min"],
        "signal_conversion": validation_targets["signal_to_submission_gap_to_floor_pct"],
        "fill_conversion": validation_targets["submission_to_fill_gap_to_floor_pct"],
    }
    validation_targets["top_blockers"] = [
        {"name": key, "gap": value}
        for key, value in sorted(blockers.items(), key=lambda item: item[1], reverse=True)
        if value > 0.0
    ][:4]
    has_validation_harness_samples = bool(
        validation_harness["signal_to_submission_pct"]
        or validation_harness["submission_to_fill_pct"]
        or validation_harness["stop_loss_negative_pl_share_pct"]
    )
    candidate_verdict_reasons: list[str] = []
    if int(acceptance_totals.get("passes_all", 0) or 0) <= 0:
        candidate_verdict_reasons.append("no_run_passed_acceptance")
    if float(stability.get("return_stddev_pct", 0.0) or 0.0) > 1.0:
        candidate_verdict_reasons.append("returns_unstable")
    if float(stability.get("best_vs_median_return_gap_pct", 0.0) or 0.0) > 1.0:
        candidate_verdict_reasons.append("best_run_far_above_median")
    if not bool(dict(walk_forward.get("acceptance", {}) or {}).get("passes_all", True)):
        candidate_verdict_reasons.append("confirmation_weaker_than_iteration")
    if dict(failure_leaderboard.get("losing_families", {}) or {}):
        candidate_verdict_reasons.append("losing_families_present")
    if dict(failure_leaderboard.get("expected_vs_realized_edge_divergence", {}) or {}):
        candidate_verdict_reasons.append("edge_realization_divergence_present")
    market_data_unavailable = int(market_data.get("runs_without_data", 0) or 0) > 0 and int(market_data.get("runs_with_data", 0) or 0) <= 0
    if market_data_unavailable:
        candidate_verdict_reasons.append("market_data_unavailable")
    if not market_data_unavailable and has_validation_harness_samples and float(validation_harness_summary.get("stop_loss_negative_pl_share_pct", 0.0) or 0.0) > 55.0:
        candidate_verdict_reasons.append("stop_loss_damage_too_high")
    if not market_data_unavailable and has_validation_harness_samples and float(validation_harness_summary.get("signal_to_submission_pct", 0.0) or 0.0) < 18.0:
        candidate_verdict_reasons.append("signal_to_submission_too_low")
    if not market_data_unavailable and has_validation_harness_samples and float(validation_harness_summary.get("submission_to_fill_pct", 0.0) or 0.0) < 45.0:
        candidate_verdict_reasons.append("submission_to_fill_too_low")
    if not market_data_unavailable and flow_starved:
        candidate_verdict_reasons.append("candidate_flow_starved")
    if not market_data_unavailable and flow_recovered and (avg_return <= 0.0 or avg_win_rate < target_win_rate):
        candidate_verdict_reasons.append("flow_recovered_but_unprofitable")
    if not candidate_verdict_reasons:
        candidate_verdict = {
            "status": "promising",
            "reasons": ["passes_current_validation_checks"],
        }
    elif "no_run_passed_acceptance" in candidate_verdict_reasons:
        candidate_verdict = {
            "status": "not_ready",
            "reasons": candidate_verdict_reasons,
        }
    else:
        candidate_verdict = {
            "status": "mixed",
            "reasons": candidate_verdict_reasons,
        }
    fill_conversion_by_strategy: Dict[str, Dict[str, float]] = {}
    strategy_names = sorted(
        set(decision_totals.get("signals_by_strategy", {}) or {})
        | set(decision_totals.get("submitted_by_strategy", {}) or {})
        | set(decision_totals.get("filled_by_strategy", {}) or {})
        | set(decision_totals.get("closed_by_strategy", {}) or {})
    )
    for strategy in strategy_names:
        signals = int((decision_totals.get("signals_by_strategy", {}) or {}).get(strategy, 0) or 0)
        submitted = int((decision_totals.get("submitted_by_strategy", {}) or {}).get(strategy, 0) or 0)
        filled = int((decision_totals.get("filled_by_strategy", {}) or {}).get(strategy, 0) or 0)
        closed = int((decision_totals.get("closed_by_strategy", {}) or {}).get(strategy, 0) or 0)
        fill_conversion_by_strategy[str(strategy)] = {
            "signals": signals,
            "submitted": submitted,
            "filled": filled,
            "closed": closed,
            "submit_rate_pct": (submitted / signals * 100.0) if signals > 0 else 0.0,
            "fill_rate_pct": (filled / submitted * 100.0) if submitted > 0 else 0.0,
            "close_rate_pct": (closed / submitted * 100.0) if submitted > 0 else 0.0,
        }
    return {
        "num_runs": len(completed),
        "status": "ok",
        "aggregates": aggregates,
        "decision_totals": decision_totals,
        "execution_totals": execution_totals,
        "acceptance_totals": acceptance_totals,
        "stability": stability,
        "failure_leaderboard": failure_leaderboard,
        "realized_performance": realized_performance,
        "exit_quality": exit_quality,
        "family_rotation_by_strategy": family_rotation_by_strategy,
        "family_rotation_soft_by_strategy": family_rotation_soft_by_strategy,
        "family_rotation_hard_by_strategy": family_rotation_hard_by_strategy,
        "family_rotation_recovery_by_strategy": family_rotation_recovery_by_strategy,
        "learning_evidence_by_strategy": learning_evidence_by_strategy,
        "learning_asymmetry_by_strategy": learning_asymmetry_by_strategy,
        "missed_opportunity_relaxations_by_strategy": missed_opportunity_relaxations_by_strategy,
        "reentry_cooldown_registrations_by_strategy": reentry_cooldown_registrations_by_strategy,
        "duplicate_bucket_throttle_by_strategy": duplicate_bucket_throttle_by_strategy,
        "weak_cluster_throttle_by_strategy": weak_cluster_throttle_by_strategy,
        "realized_performance_penalty_by_strategy": realized_performance_penalty_by_strategy,
        "realized_performance_no_trade_by_strategy": realized_performance_no_trade_by_strategy,
        "fill_conversion_by_strategy": fill_conversion_by_strategy,
        "validation_harness": validation_harness_summary,
        "triple_barrier": triple_barrier,
        "validation_targets": validation_targets,
        "market_data": dict(market_data),
        "universe_selection": universe_selection,
        "skip_reasons_by_symbol": skip_reasons_by_symbol,
        "comparisons": {
            "baseline_vs_latest": {
                "baseline_artifact_dir": baseline_run.get("artifact_dir"),
                "candidate_artifact_dir": latest_run.get("artifact_dir"),
                **baseline_vs_latest,
            },
            "baseline_vs_median": {
                "baseline_artifact_dir": baseline_run.get("artifact_dir"),
                "candidate_artifact_dir": median_run.get("artifact_dir"),
                **baseline_vs_median,
            },
        },
        "best_run": {
            "artifact_dir": best_run.get("artifact_dir"),
            "total_return_pct": best_run.get("total_return_pct"),
            "num_trades": best_run.get("num_trades"),
            "win_rate_pct": best_run.get("win_rate_pct"),
        },
        "median_run": {
            "artifact_dir": median_run.get("artifact_dir"),
            "total_return_pct": median_run.get("total_return_pct"),
            "num_trades": median_run.get("num_trades"),
            "win_rate_pct": median_run.get("win_rate_pct"),
        },
        "worst_run": {
            "artifact_dir": worst_run.get("artifact_dir"),
            "total_return_pct": worst_run.get("total_return_pct"),
            "num_trades": worst_run.get("num_trades"),
            "win_rate_pct": worst_run.get("win_rate_pct"),
        },
        "by_horizon": by_horizon,
        "walk_forward": walk_forward,
        "candidate_verdict": candidate_verdict,
        "runs": [
            {
                "artifact_dir": item.get("artifact_dir"),
                "num_trades": item.get("num_trades"),
                "raw_signals": item.get("raw_signals"),
                "win_rate_pct": item.get("win_rate_pct"),
                "total_return_pct": item.get("total_return_pct"),
                "profit_factor": item.get("profit_factor"),
                "max_drawdown_pct": item.get("max_drawdown_pct"),
                "days": int(dict(dict(item.get("campaign_summary", {}) or {}).get("session", {}) or {}).get("days", item.get("days", 0)) or 0),
            }
            for item in completed
        ],
    }


def load_batch_reports(runs_dir: str) -> list[Dict[str, Any]]:
    if not os.path.isdir(runs_dir):
        return []
    reports: list[Dict[str, Any]] = []
    for name in sorted(os.listdir(runs_dir)):
        report_path = os.path.join(runs_dir, name, "report.json")
        if not os.path.exists(report_path):
            continue
        with open(report_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("artifact_dir", os.path.join(runs_dir, name))
        reports.append(payload)
    return reports
