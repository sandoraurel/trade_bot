from __future__ import annotations

import datetime as dt
import json
import math
import uuid
from typing import Any, Dict, List

from .calibration import OnlineConfidenceCalibrator
from .learning_drift import LearningDriftMonitor


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _parse_iso(value: str | None) -> dt.datetime:
    if not value:
        return dt.datetime.now(dt.timezone.utc)
    parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


class TradeLearningEngine:
    """
    Adaptive learning layer with:
    - evidence-weighted hierarchical pattern learning
    - online confidence calibration
    - drift monitoring
    - counterfactual shadow-decision tracking
    - safe-update guardrails for future simulation-driven policy improvement
    """

    def __init__(self, config: Any, state_store: Any):
        self.config = config
        self.state_store = state_store
        self.drift_monitor = LearningDriftMonitor(config, state_store)
        self.calibrator = OnlineConfidenceCalibrator(config, state_store)

    def build_trade_context(
        self,
        signal: Dict[str, Any],
        regime: Any | None = None,
        execution_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        metadata = dict(signal.get("metadata", {}) or {})
        research = dict(signal.get("research_context", {}) or metadata.get("research_context", {}) or {})
        execution = dict(execution_context or metadata.get("execution_context", {}) or {})
        regime_name = str(signal.get("regime") or getattr(regime, "regime", "unknown"))
        regime_metadata = dict(getattr(regime, "metadata", {}) or {})
        regime_confidence = _safe_float(metadata.get("regime_confidence", getattr(regime, "confidence", 0.0)))
        trend_direction = str(metadata.get("trend_direction", regime_metadata.get("trend_direction", "flat")))
        strategy_variant = str(metadata.get("strategy_variant", "base"))
        symbol = str(signal.get("symbol", ""))
        symbol_bucket = self._symbol_bucket(symbol)
        order_profile = str(
            metadata.get(
                "preferred_order_type",
                metadata.get("order_type", "market" if bool(signal.get("fast_move", False)) else "limit"),
            )
        ).lower()
        rr_ratio = _safe_float(signal.get("rr_ratio", 0.0))
        confidence = _safe_float(signal.get("signal_quality", 0.0))
        expected_edge_bps = _safe_float(signal.get("expected_edge_bps", 0.0))
        hurst = _safe_float(signal.get("hurst_exponent", metadata.get("hurst_exponent", 0.5)), 0.5)
        volume_impulse = _safe_float(metadata.get("volume_impulse", regime_metadata.get("volume_impulse", 1.0)), 1.0)
        spread_bps = _safe_float(execution.get("spread_bps", metadata.get("spread_bps", 0.0)))
        entry_deviation_bps = _safe_float(execution.get("entry_deviation_bps", metadata.get("entry_deviation_bps", 0.0)))
        fill_fraction = _safe_float(execution.get("fill_fraction", metadata.get("fill_fraction", 1.0)), 1.0)
        latency_ms = _safe_float(execution.get("latency_ms", metadata.get("latency_ms", 0.0)))
        pattern = self._pattern_features(
            strategy=str(signal.get("strategy", "unknown")),
            symbol_bucket=symbol_bucket,
            side=str(signal.get("side", "flat")),
            regime=regime_name,
            trend_direction=trend_direction,
            strategy_variant=strategy_variant,
            order_profile=order_profile,
            confidence=confidence,
            rr_ratio=rr_ratio,
            expected_edge_bps=expected_edge_bps,
            research=research,
            hurst=hurst,
            regime_confidence=regime_confidence,
            volume_impulse=volume_impulse,
            spread_bps=spread_bps,
            entry_deviation_bps=entry_deviation_bps,
            fill_fraction=fill_fraction,
            latency_ms=latency_ms,
            fast_move=bool(signal.get("fast_move", False)),
        )
        generic_cell_key = f"cell::{str(signal.get('strategy', 'unknown'))}::{str(signal.get('side', 'flat'))}::{regime_name}::{order_profile}"
        bucket_cell_key = f"cell::{str(signal.get('strategy', 'unknown'))}::{str(signal.get('side', 'flat'))}::{regime_name}::{symbol_bucket}::{order_profile}"
        return {
            "captured_at": utcnow_iso(),
            "symbol": symbol,
            "symbol_bucket": symbol_bucket,
            "strategy": str(signal.get("strategy", "unknown")),
            "strategy_variant": strategy_variant,
            "side": str(signal.get("side", "flat")),
            "regime": regime_name,
            "order_profile": order_profile,
            "order_type": order_profile,
            "regime_confidence": regime_confidence,
            "trend_direction": trend_direction,
            "signal_quality": confidence,
            "expected_edge_bps": expected_edge_bps,
            "rr_ratio": rr_ratio,
            "hurst_exponent": hurst,
            "fast_move": bool(signal.get("fast_move", False)),
            "volume_impulse": volume_impulse,
            "expected_holding_minutes": int(signal.get("expected_holding_minutes", metadata.get("expected_holding_minutes", 480)) or 480),
            "timeframe": str(signal.get("timeframe", metadata.get("timeframe", "15m"))),
            "research_context": research,
            "execution_context": {
                "spread_bps": spread_bps,
                "entry_deviation_bps": entry_deviation_bps,
                "fill_fraction": fill_fraction,
                "latency_ms": latency_ms,
                "execution_quality": self._execution_quality_bucket(spread_bps, entry_deviation_bps, fill_fraction),
            },
            "learning_cell_keys": {
                "generic": generic_cell_key,
                "bucket_specific": bucket_cell_key,
            },
            "pattern": pattern,
        }

    def learning_context_for_signal(self, signal: Dict[str, Any], regime: Any | None = None) -> Dict[str, Any]:
        if self.state_store is None:
            return self._neutral_learning_context()
        trade_context = self.build_trade_context(signal, regime)
        pattern = trade_context["pattern"]
        keys = self._pattern_keys(pattern)
        stats = self.state_store.load_learning_patterns(keys)
        aggregate = self._aggregate_stats(stats) if stats else self._empty_aggregate()
        calibration = self.calibrator.predict(
            trade_context["signal_quality"],
            strategy=trade_context["strategy"],
            regime=trade_context["regime"],
            side=trade_context["side"],
            order_profile=trade_context["order_profile"],
        )
        drift = self._drift_summary(
            strategy=trade_context["strategy"],
            symbol_bucket=trade_context.get("symbol_bucket"),
            regime=trade_context["regime"],
            side=trade_context["side"],
            order_profile=trade_context["order_profile"],
        )
        opportunity = self._opportunity_summary(
            strategy=trade_context["strategy"],
            symbol_bucket=trade_context.get("symbol_bucket"),
            side=trade_context["side"],
            regime=trade_context["regime"],
            order_profile=trade_context["order_profile"],
        )
        prequential = self._prequential_summary(
            strategy=trade_context["strategy"],
            symbol_bucket=trade_context.get("symbol_bucket"),
            side=trade_context["side"],
            regime=trade_context["regime"],
            order_profile=trade_context["order_profile"],
        )
        positive_cell_evidence = self._positive_cell_evidence(opportunity, prequential, calibration, drift)
        bucket_specific = bool(opportunity.get("bucket_specific") or prequential.get("bucket_specific"))
        family_rotation = self._family_rotation_summary(trade_context["strategy"])

        score_delta = aggregate["expectancy_score"] + aggregate["win_rate_score"] + aggregate["attribution_score"]
        score_delta += opportunity["score_bias"]
        score_delta += prequential["score_bias"]
        confidence_delta = aggregate["confidence_bonus"] + calibration["confidence_delta"]
        confidence_delta += prequential["confidence_delta"]
        risk_multiplier = 1.0 + (score_delta / 30.0) + aggregate["risk_bias"] + opportunity["risk_bias"] + prequential["risk_bias"]

        safe_positive = self._safe_positive_updates_allowed(aggregate, calibration, drift)
        if not safe_positive and not positive_cell_evidence:
            score_delta = min(score_delta, 0.0)
            confidence_delta = min(confidence_delta, 0.0)
            risk_multiplier = min(risk_multiplier, 1.0)

        drift_penalty = min(float(drift.get("severity", 0.0) or 0.0) * 0.9, 1.8)
        score_delta -= drift_penalty
        confidence_delta -= min(drift_penalty / 40.0, 0.03)
        risk_multiplier -= min(drift_penalty / 15.0, 0.06)

        score_delta = _clamp(
            score_delta,
            -float(getattr(self.config, "learning_negative_score_cap", 9.0)),
            float(getattr(self.config, "learning_positive_score_cap", 5.0)),
        )
        confidence_delta = _clamp(
            confidence_delta,
            -float(getattr(self.config, "learning_confidence_penalty_cap", 0.08)),
            float(getattr(self.config, "learning_confidence_boost_cap", 0.04)),
        )
        risk_multiplier = _clamp(
            risk_multiplier,
            float(getattr(self.config, "learning_min_risk_multiplier", 0.72)),
            float(getattr(self.config, "learning_max_risk_multiplier", 1.10)),
        )

        negative_cell_evidence = bool(
            (
                float(opportunity.get("samples", 0.0) or 0.0) >= (
                    float(getattr(self.config, "learning_cell_gate_min_samples", 6.0))
                    - (float(getattr(self.config, "learning_bucket_negative_sample_delta", 2.0)) if opportunity.get("bucket_specific") else 0.0)
                )
                and float(opportunity.get("avg_forward_r", 0.0) or 0.0) <= float(getattr(self.config, "learning_negative_opportunity_gate_avg_forward_r", -0.20))
            )
            or (
                float(prequential.get("samples", 0.0) or 0.0) >= (
                    float(getattr(self.config, "learning_cell_gate_min_samples", 6.0))
                    - (float(getattr(self.config, "learning_bucket_negative_sample_delta", 2.0)) if prequential.get("bucket_specific") else 0.0)
                )
                and float(prequential.get("avg_r_multiple", 0.0) or 0.0) <= float(getattr(self.config, "learning_prequential_negative_avg_r", -0.25))
                and float(prequential.get("win_rate", 0.0) or 0.0) <= float(getattr(self.config, "learning_prequential_negative_win_rate", 0.34))
            )
        )
        if negative_cell_evidence:
            score_delta = min(
                score_delta,
                -1.0 - (float(getattr(self.config, "learning_bucket_negative_score_penalty", 0.75)) if bucket_specific else 0.0),
            )
            confidence_delta = min(confidence_delta, -0.02)
            risk_multiplier = min(
                risk_multiplier,
                float(getattr(self.config, "learning_bucket_negative_risk_cap", 0.84)) if bucket_specific else 0.88,
            )
        elif positive_cell_evidence:
            score_delta = max(
                score_delta,
                float(getattr(self.config, "learning_positive_min_score_delta", 1.10))
                + (float(getattr(self.config, "learning_bucket_positive_score_boost", 0.50)) if bucket_specific else 0.0),
            )
            confidence_delta = max(
                confidence_delta,
                float(getattr(self.config, "learning_positive_min_confidence_delta", 0.01)),
            )
            risk_multiplier = max(
                risk_multiplier,
                float(getattr(self.config, "learning_positive_min_risk_multiplier", 1.03))
                + (float(getattr(self.config, "learning_bucket_positive_risk_boost", 0.03)) if bucket_specific else 0.0),
            )
            if "positive_cell" not in aggregate["reasons"]:
                aggregate["reasons"] = list(dict.fromkeys([*aggregate["reasons"], "positive_cell"]))
            if "positive_cell" not in aggregate["dominant_attributions"]:
                aggregate["dominant_attributions"] = list(dict.fromkeys([*aggregate["dominant_attributions"], "positive_cell"]))

        if family_rotation.get("status") == "suppress_current_hard":
            score_delta -= float(getattr(self.config, "learning_family_rotation_hard_penalty_score", 2.75))
            risk_multiplier = min(
                risk_multiplier,
                float(getattr(self.config, "learning_family_rotation_hard_penalty_risk_cap", 0.74)),
            )
            confidence_delta = min(confidence_delta, -0.025)
            if "family_rotation_hard_suppressed" not in aggregate["reasons"]:
                aggregate["reasons"] = list(dict.fromkeys([*aggregate["reasons"], "family_rotation_hard_suppressed"]))
            if "family_rotation_hard_suppressed" not in aggregate["dominant_attributions"]:
                aggregate["dominant_attributions"] = list(dict.fromkeys([*aggregate["dominant_attributions"], "family_rotation_hard_suppressed"]))
        elif family_rotation.get("status") == "suppress_current":
            score_delta -= float(getattr(self.config, "learning_family_rotation_penalty_score", 1.50))
            risk_multiplier = min(
                risk_multiplier,
                float(getattr(self.config, "learning_family_rotation_penalty_risk_cap", 0.82)),
            )
            if "family_rotation_suppressed" not in aggregate["reasons"]:
                aggregate["reasons"] = list(dict.fromkeys([*aggregate["reasons"], "family_rotation_suppressed"]))
            if "family_rotation_suppressed" not in aggregate["dominant_attributions"]:
                aggregate["dominant_attributions"] = list(dict.fromkeys([*aggregate["dominant_attributions"], "family_rotation_suppressed"]))
        elif family_rotation.get("status") == "promote_current_hard":
            score_delta += float(getattr(self.config, "learning_family_rotation_hard_boost_score", 1.35))
            risk_multiplier = min(
                float(getattr(self.config, "learning_max_risk_multiplier", 1.10)),
                risk_multiplier + float(getattr(self.config, "learning_family_rotation_hard_boost_risk", 0.06)),
            )
            confidence_delta = max(confidence_delta, 0.015)
            if "family_rotation_hard_preferred" not in aggregate["reasons"]:
                aggregate["reasons"] = list(dict.fromkeys([*aggregate["reasons"], "family_rotation_hard_preferred"]))
            if "family_rotation_hard_preferred" not in aggregate["dominant_attributions"]:
                aggregate["dominant_attributions"] = list(dict.fromkeys([*aggregate["dominant_attributions"], "family_rotation_hard_preferred"]))
        elif family_rotation.get("status") == "promote_current":
            score_delta += float(getattr(self.config, "learning_family_rotation_boost_score", 0.75))
            risk_multiplier = min(
                float(getattr(self.config, "learning_max_risk_multiplier", 1.10)),
                risk_multiplier + float(getattr(self.config, "learning_family_rotation_boost_risk", 0.04)),
            )
            if "family_rotation_preferred" not in aggregate["reasons"]:
                aggregate["reasons"] = list(dict.fromkeys([*aggregate["reasons"], "family_rotation_preferred"]))
            if "family_rotation_preferred" not in aggregate["dominant_attributions"]:
                aggregate["dominant_attributions"] = list(dict.fromkeys([*aggregate["dominant_attributions"], "family_rotation_preferred"]))

        if family_rotation.get("recovery_active", False):
            score_delta += float(getattr(self.config, "learning_family_rotation_recovery_score_boost", 0.45))
            risk_multiplier = min(
                float(getattr(self.config, "learning_max_risk_multiplier", 1.10)),
                risk_multiplier + float(getattr(self.config, "learning_family_rotation_recovery_risk_boost", 0.02)),
            )
            if "family_rotation_recovering" not in aggregate["reasons"]:
                aggregate["reasons"] = list(dict.fromkeys([*aggregate["reasons"], "family_rotation_recovering"]))
            if "family_rotation_recovering" not in aggregate["dominant_attributions"]:
                aggregate["dominant_attributions"] = list(dict.fromkeys([*aggregate["dominant_attributions"], "family_rotation_recovering"]))

        early_structural_veto = bool(
            aggregate["effective_samples"] >= float(getattr(self.config, "learning_early_veto_min_samples", 3.0))
            and aggregate["shrunk_avg_r"] <= float(getattr(self.config, "learning_early_veto_avg_r_multiple", -0.05))
            and aggregate["structural_loss_ratio"] >= float(getattr(self.config, "learning_early_structural_veto_ratio", 0.58))
            and aggregate["matched_key_strength"] >= 0.75
            and aggregate["execution_problem_ratio"] < 0.55
            and not drift.get("active", False)
        )

        veto = bool(
            (
                aggregate["effective_samples"] >= float(getattr(self.config, "learning_severe_pattern_samples", 10.0))
                and aggregate["structural_loss_ratio"] >= float(getattr(self.config, "learning_structural_veto_ratio", 0.82))
                and aggregate["shrunk_avg_r"] <= float(getattr(self.config, "learning_severe_avg_r_multiple", -0.85))
                and aggregate["matched_key_strength"] >= 0.75
                and aggregate["execution_problem_ratio"] < 0.45
                and not drift.get("active", False)
            )
            or (
                negative_cell_evidence
                and float(opportunity.get("samples", 0.0) or 0.0) >= (
                    float(getattr(self.config, "learning_cell_veto_min_samples", 9.0))
                    - (float(getattr(self.config, "learning_bucket_negative_sample_delta", 2.0)) if opportunity.get("bucket_specific") else 0.0)
                )
                and float(prequential.get("samples", 0.0) or 0.0) >= (
                    float(getattr(self.config, "learning_cell_veto_min_samples", 9.0))
                    - (float(getattr(self.config, "learning_bucket_negative_sample_delta", 2.0)) if prequential.get("bucket_specific") else 0.0)
                )
                and float(calibration.get("calibrated_confidence", 0.5) or 0.5) < float(getattr(self.config, "learning_min_calibrated_confidence", 0.50))
            )
            or early_structural_veto
        )
        if early_structural_veto:
            score_delta = min(score_delta, -3.0)
            confidence_delta = min(confidence_delta, -0.03)
            risk_multiplier = min(risk_multiplier, 0.78)
            if "structural_repeat" not in aggregate["reasons"]:
                aggregate["reasons"] = list(dict.fromkeys([*aggregate["reasons"], "structural_repeat"]))
            if "structural_repeat" not in aggregate["dominant_attributions"]:
                aggregate["dominant_attributions"] = list(dict.fromkeys([*aggregate["dominant_attributions"], "structural_repeat"]))
        asymmetry_actions: list[str] = []
        if positive_cell_evidence:
            asymmetry_actions.append("promote_winner")
            if bucket_specific:
                asymmetry_actions.append("promote_winner_bucket_specific")
        if negative_cell_evidence:
            asymmetry_actions.append("throttle_loser")
            if bucket_specific:
                asymmetry_actions.append("throttle_loser_bucket_specific")
        if family_rotation.get("status") == "promote_current_hard":
            asymmetry_actions.append("promote_family_rotation_hard")
        elif family_rotation.get("status") == "promote_current":
            asymmetry_actions.append("promote_family_rotation")
        elif family_rotation.get("status") == "suppress_current_hard":
            asymmetry_actions.append("throttle_family_rotation_hard")
        elif family_rotation.get("status") == "suppress_current":
            asymmetry_actions.append("throttle_family_rotation")
        if family_rotation.get("recovery_active", False):
            asymmetry_actions.append("recover_family_rotation")
        return {
            "score_delta": score_delta,
            "confidence_delta": confidence_delta,
            "risk_multiplier": risk_multiplier,
            "veto": veto,
            "reasons": aggregate["reasons"],
            "dominant_attributions": aggregate["dominant_attributions"],
            "evidence_samples": round(aggregate["effective_samples"], 3),
            "evidence_quality": aggregate["evidence_quality"],
            "pattern": pattern,
            "matched_patterns": aggregate["matched_patterns"],
            "summary": aggregate["summary"],
            "calibration": calibration,
            "drift": drift,
            "opportunity": opportunity,
            "prequential": prequential,
            "negative_cell_evidence": negative_cell_evidence,
            "positive_cell_evidence": positive_cell_evidence,
            "safe_positive_updates": safe_positive,
            "family_rotation": family_rotation,
            "asymmetric_learning": {
                "promotion_active": bool(positive_cell_evidence or family_rotation.get("status") in {"promote_current", "promote_current_hard"}),
                "throttle_active": bool(negative_cell_evidence or family_rotation.get("status") in {"suppress_current", "suppress_current_hard"}),
                "bucket_specific": bucket_specific,
                "actions": asymmetry_actions,
            },
            "cell_tracking": {
                "generic_cell_key": trade_context["learning_cell_keys"]["generic"],
                "bucket_cell_key": trade_context["learning_cell_keys"]["bucket_specific"],
                "active_cell_key": trade_context["learning_cell_keys"]["bucket_specific"] if bucket_specific else trade_context["learning_cell_keys"]["generic"],
                "bucket_specific": bucket_specific,
                "symbol_bucket": trade_context.get("symbol_bucket"),
                "order_type": trade_context["order_profile"],
            },
        }

    def record_closed_trade(
        self,
        *,
        symbol: str,
        position: Any,
        close_price: float,
        profit_loss: float,
        exit_reason: str,
        closed_at: str | None = None,
    ) -> None:
        if self.state_store is None:
            return
        metadata = dict(getattr(position, "metadata", {}) or {})
        trade_context = dict(metadata.get("decision_context", {}) or {})
        if not trade_context:
            trade_context = self.build_trade_context(
                {
                    "symbol": symbol,
                    "strategy": getattr(position, "strategy", "unknown"),
                    "side": getattr(position, "side", "flat"),
                    "signal_quality": 0.0,
                    "expected_edge_bps": 0.0,
                    "rr_ratio": 0.0,
                    "hurst_exponent": 0.5,
                    "fast_move": False,
                    "metadata": {},
                }
            )

        entry_price = _safe_float(getattr(position, "entry_price", 0.0))
        initial_stop_loss = _safe_float(getattr(position, "initial_stop_loss", None) or getattr(position, "stop_loss", 0.0))
        size = _safe_float(getattr(position, "size", 0.0))
        risk_per_unit = abs(entry_price - initial_stop_loss)
        initial_risk = risk_per_unit * size
        r_multiple = (profit_loss / initial_risk) if initial_risk > 0 else 0.0
        observed_at = closed_at or utcnow_iso()
        outcome_labels = self._classify_outcome(trade_context, profit_loss, exit_reason, r_multiple)
        quality_weight = self._observation_quality_weight(trade_context, outcome_labels)
        observation = {
            "observed_at": observed_at,
            "symbol": symbol,
            "strategy": str(trade_context.get("strategy", getattr(position, "strategy", "unknown"))),
            "side": str(getattr(position, "side", "flat")),
            "entry_price": entry_price,
            "close_price": close_price,
            "profit_loss": profit_loss,
            "r_multiple": r_multiple,
            "exit_reason": exit_reason,
            "success": profit_loss > 0,
            "pattern": trade_context.get("pattern", {}),
            "research_context": trade_context.get("research_context", {}),
            "execution_context": trade_context.get("execution_context", {}),
            "outcome_labels": outcome_labels,
            "decision_context": trade_context,
            "quality_weight": quality_weight,
        }
        self.state_store.append_learning_observation(observation)
        self._update_patterns(observation)
        self.calibrator.update(
            trade_context.get("signal_quality", 0.5),
            strategy=trade_context.get("strategy", "unknown"),
            regime=trade_context.get("regime", "unknown"),
            side=trade_context.get("side", "flat"),
            order_profile=trade_context.get("order_profile", "limit"),
            success=bool(observation["success"]),
            observed_at=observed_at,
        )
        self._update_drift_monitors(observation)
        self._update_prequential_models(
            trade_context,
            success=bool(observation["success"]),
            realized_r_multiple=r_multiple,
            observed_at=observed_at,
            source="executed_trade",
        )

    def record_shadow_decision(
        self,
        signal: Dict[str, Any],
        *,
        status: str,
        reason: str,
        trace_id: str | None = None,
        created_at: str | None = None,
    ) -> None:
        if self.state_store is None:
            return
        decision_context = self.build_trade_context(signal)
        decision_id = uuid.uuid4().hex
        created_at = created_at or utcnow_iso()
        payload = {
            "decision_id": decision_id,
            "created_at": created_at,
            "status": status,
            "reason": reason,
            "trace_id": trace_id,
            "symbol": signal.get("symbol"),
            "entry_price": _safe_float(signal.get("entry_price", 0.0)),
            "stop_loss": _safe_float(signal.get("stop_loss", 0.0)),
            "take_profit": _safe_float(signal.get("take_profit", 0.0)),
            "side": signal.get("side"),
            "strategy": signal.get("strategy", "unknown"),
            "expected_holding_minutes": int(signal.get("metadata", {}).get("expected_holding_minutes", signal.get("expected_holding_minutes", 480)) or 480),
            "timeframe": str(signal.get("metadata", {}).get("timeframe", signal.get("timeframe", "15m"))),
            "decision_context": decision_context,
            "signal_snapshot": signal,
        }
        self.state_store.append_learning_decision(decision_id, created_at, "pending", payload)

    def evaluate_pending_shadow_decisions(self, exchange_client: Any, now: dt.datetime | None = None) -> None:
        if self.state_store is None:
            return
        now = now or dt.datetime.now(dt.timezone.utc)
        pending = self.state_store.load_pending_learning_decisions(limit=200)
        for item in pending:
            created_at = _parse_iso(item.get("created_at"))
            horizon_minutes = int(item.get("expected_holding_minutes", 480) or 480)
            if (now - created_at).total_seconds() < horizon_minutes * 60:
                continue
            outcome = self._evaluate_shadow_outcome(exchange_client, item)
            item["status"] = "evaluated"
            item["evaluated_at"] = now.isoformat()
            item["outcome"] = outcome
            self.state_store.append_learning_decision(item["decision_id"], item["created_at"], "evaluated", item)
            self._update_opportunity_models(item, outcome)
            self._update_prequential_models(
                item.get("decision_context", {}),
                success=bool(outcome.get("success")),
                realized_r_multiple=_safe_float(outcome.get("forward_r_multiple", 0.0)),
                observed_at=item["evaluated_at"],
                source="shadow_decision",
            )

    def missed_opportunity_gate_adjustment(self, signal: Dict[str, Any], reason: str) -> Dict[str, Any]:
        if self.state_store is None:
            return {"active": False, "reason": str(reason or "unknown"), "relax_bps": 0.0}
        trade_context = self.build_trade_context(signal)
        summary = self._missed_opportunity_summary(
            reason=str(reason or "unknown"),
            strategy=trade_context.get("strategy"),
            symbol_bucket=trade_context.get("symbol_bucket"),
            side=trade_context.get("side"),
            regime=trade_context.get("regime"),
            order_profile=trade_context.get("order_profile"),
        )
        min_samples = float(getattr(self.config, "learning_missed_opportunity_min_samples", 4.0))
        min_avg_forward_r = float(getattr(self.config, "learning_missed_opportunity_avg_forward_r", 0.18))
        min_positive_ratio = float(getattr(self.config, "learning_missed_opportunity_positive_ratio", 0.56))
        active = bool(
            float(summary.get("samples", 0.0) or 0.0) >= min_samples
            and float(summary.get("avg_forward_r", 0.0) or 0.0) >= min_avg_forward_r
            and float(summary.get("positive_ratio", 0.0) or 0.0) >= min_positive_ratio
        )
        relax_bps = min(
            3.0,
            max(0.0, float(summary.get("avg_forward_r", 0.0) or 0.0) * 4.0),
        ) if active else 0.0
        return {
            "active": active,
            "reason": str(reason or "unknown"),
            "relax_bps": relax_bps,
            "summary": summary,
        }

    def summary_snapshot(self) -> Dict[str, Any]:
        if self.state_store is None:
            return {"status": "disabled"}
        recent = self.state_store.load_recent_learning_observations(limit=40)
        patterns = self.state_store.load_top_learning_patterns(limit=8)
        total = len(recent)
        wins = sum(1 for item in recent if item.get("success"))
        avg_r = sum(_safe_float(item.get("r_multiple", 0.0)) for item in recent) / total if total else 0.0
        top_attributions: Dict[str, int] = {}
        for item in recent:
            for label in item.get("outcome_labels", []):
                top_attributions[label] = int(top_attributions.get(label, 0)) + 1
        ranked_labels = [label for label, _ in sorted(top_attributions.items(), key=lambda row: row[1], reverse=True)[:5]]
        top_patterns = []
        for payload in patterns:
            eff = _safe_float(payload.get("effective_samples", payload.get("samples", 0.0)))
            if eff <= 0:
                continue
            avg_pattern_r = _safe_float(payload.get("total_r_multiple", 0.0)) / eff
            top_patterns.append(
                {
                    "pattern_key": payload.get("pattern_key"),
                    "effective_samples": round(eff, 3),
                    "avg_r_multiple": round(avg_pattern_r, 3),
                    "top_labels": [label for label, _ in sorted(dict(payload.get("outcome_labels", {})).items(), key=lambda row: row[1], reverse=True)[:3]],
                }
            )
        drift = self._drift_summary()
        opportunity = self._opportunity_summary()
        return {
            "status": "ok",
            "recent_trades": total,
            "recent_win_rate": (wins / total) if total else 0.0,
            "recent_avg_r_multiple": avg_r,
            "top_attributions": ranked_labels,
            "top_patterns": top_patterns,
            "drift": drift,
            "opportunity": opportunity,
            "calibration": self.calibrator.summary(limit=6),
            "prequential": self._prequential_summary(),
        }

    def _neutral_learning_context(self) -> Dict[str, Any]:
        return {
            "score_delta": 0.0,
            "confidence_delta": 0.0,
            "risk_multiplier": 1.0,
            "veto": False,
            "reasons": [],
            "dominant_attributions": [],
            "evidence_samples": 0.0,
            "evidence_quality": "insufficient",
            "pattern": {},
            "matched_patterns": [],
            "summary": "No meaningful learning evidence yet.",
            "calibration": {"calibrated_confidence": 0.0, "confidence_delta": 0.0, "calibration_gap": 0.0},
            "drift": {"active": False, "severity": 0.0},
            "opportunity": {"score_bias": 0.0, "risk_bias": 0.0},
            "prequential": {"score_bias": 0.0, "confidence_delta": 0.0, "risk_bias": 0.0},
            "negative_cell_evidence": False,
            "positive_cell_evidence": False,
            "safe_positive_updates": False,
        }

    def _classify_outcome(self, trade_context: Dict[str, Any], profit_loss: float, exit_reason: str, r_multiple: float) -> List[str]:
        labels: List[str] = []
        signal_quality = _safe_float(trade_context.get("signal_quality", 0.0))
        expected_edge_bps = _safe_float(trade_context.get("expected_edge_bps", 0.0))
        rr_ratio = _safe_float(trade_context.get("rr_ratio", 0.0))
        regime_confidence = _safe_float(trade_context.get("regime_confidence", 0.0))
        execution = dict(trade_context.get("execution_context", {}) or {})
        spread_bps = _safe_float(execution.get("spread_bps", 0.0))
        entry_deviation_bps = _safe_float(execution.get("entry_deviation_bps", 0.0))
        fill_fraction = _safe_float(execution.get("fill_fraction", 1.0), 1.0)
        research = dict(trade_context.get("research_context", {}) or {})
        bullish = _safe_float(research.get("bullish_confidence", 0.0))
        bearish = _safe_float(research.get("bearish_confidence", 0.0))
        risk_off = _safe_float(research.get("risk_off_confidence", 0.0))
        side = str(trade_context.get("side", "flat")).lower()

        if profit_loss > 0:
            labels.append("positive_outcome")
            if signal_quality >= 0.72 and expected_edge_bps >= 18.0 and rr_ratio >= 1.6:
                labels.append("strong_setup_quality")
            if regime_confidence >= 0.72:
                labels.append("regime_alignment")
            if spread_bps <= 8.0 and fill_fraction >= 0.9:
                labels.append("clean_execution")
            if r_multiple >= 1.0:
                labels.append("convex_payoff_realized")
            return labels

        if str(exit_reason).upper() == "SL":
            labels.append("thesis_invalidated")
        if spread_bps >= 18.0 or entry_deviation_bps >= 15.0 or fill_fraction < 0.82:
            labels.append("execution_quality_issue")
        if spread_bps >= 25.0 or fill_fraction < 0.72:
            labels.append("liquidity_friction")
        if signal_quality < 0.62 or expected_edge_bps < 12.0 or rr_ratio < 1.2:
            labels.append("poor_setup_quality")
        if regime_confidence < 0.60 or trade_context.get("trend_direction") == "flat":
            labels.append("regime_mismatch")
        if risk_off >= float(getattr(self.config, "research_risk_off_veto_confidence", 0.75)):
            labels.append("ignored_risk_off")
        if side in {"long", "buy"} and bearish > bullish:
            labels.append("research_conflict")
        if side in {"short", "sell"} and bullish > bearish:
            labels.append("research_conflict")
        if signal_quality >= 0.72 and expected_edge_bps >= 18.0 and regime_confidence >= 0.68 and not labels:
            labels.append("market_noise")
        if not labels:
            labels.append("generic_loss_pattern")
        if self._estimate_prior_structural_risk(trade_context) >= 0.65:
            labels.append("structural_repeat")
        return labels

    def _estimate_prior_structural_risk(self, trade_context: Dict[str, Any]) -> float:
        pattern = dict(trade_context.get("pattern", {}) or {})
        if not pattern or self.state_store is None:
            return 0.0
        keys = self._pattern_keys(pattern)
        stats = self.state_store.load_learning_patterns(keys)
        if not stats:
            return 0.0
        weighted_total = 0.0
        weighted_structural = 0.0
        for item in stats:
            effective_samples = _safe_float(item.get("effective_samples", item.get("samples", 0.0)))
            if effective_samples <= 0:
                continue
            weight = _safe_float(item.get("weight", 0.0))
            labels = dict(item.get("outcome_labels", {}) or {})
            structural = (
                _safe_float(labels.get("poor_setup_quality", 0.0))
                + _safe_float(labels.get("regime_mismatch", 0.0))
                + _safe_float(labels.get("structural_repeat", 0.0))
                + _safe_float(labels.get("research_conflict", 0.0))
            )
            weighted_total += effective_samples * weight
            weighted_structural += structural * weight
        if weighted_total <= 0:
            return 0.0
        return _clamp(weighted_structural / weighted_total, 0.0, 1.0)

    def _observation_quality_weight(self, trade_context: Dict[str, Any], outcome_labels: List[str]) -> float:
        execution = dict(trade_context.get("execution_context", {}) or {})
        spread_bps = _safe_float(execution.get("spread_bps", 0.0))
        fill_fraction = _safe_float(execution.get("fill_fraction", 1.0), 1.0)
        weight = 1.0
        if spread_bps >= 25.0:
            weight -= 0.12
        if fill_fraction < 0.75:
            weight -= 0.18
        if "market_noise" in outcome_labels:
            weight -= 0.10
        if "structural_repeat" in outcome_labels:
            weight += 0.08
        if "strong_setup_quality" in outcome_labels:
            weight += 0.05
        return _clamp(weight, 0.55, 1.15)

    def _update_patterns(self, observation: Dict[str, Any]) -> None:
        pattern = dict(observation.get("pattern", {}) or {})
        if not pattern:
            return
        observed_at = str(observation.get("observed_at") or utcnow_iso())
        obs_weight = _safe_float(observation.get("quality_weight", 1.0), 1.0)
        keys = self._pattern_keys(pattern)
        for key, weight in keys.items():
            current = self.state_store.load_learning_pattern(key) or {}
            current = self._decay_payload(current, observed_at)
            effective_samples = _safe_float(current.get("effective_samples", current.get("samples", 0.0)))
            current["samples"] = effective_samples + obs_weight
            current["effective_samples"] = effective_samples + obs_weight
            if observation.get("success"):
                current["wins"] = _safe_float(current.get("wins", 0.0)) + obs_weight
            else:
                current["losses"] = _safe_float(current.get("losses", 0.0)) + obs_weight
            current["total_r_multiple"] = _safe_float(current.get("total_r_multiple", 0.0)) + (_safe_float(observation.get("r_multiple", 0.0)) * obs_weight)
            current["total_profit_loss"] = _safe_float(current.get("total_profit_loss", 0.0)) + (_safe_float(observation.get("profit_loss", 0.0)) * obs_weight)
            if str(observation.get("exit_reason", "")).upper() == "SL":
                current["stop_loss_exits"] = _safe_float(current.get("stop_loss_exits", 0.0)) + obs_weight
            label_counts = dict(current.get("outcome_labels", {}) or {})
            for label in observation.get("outcome_labels", []):
                label_counts[label] = _safe_float(label_counts.get(label, 0.0)) + obs_weight
            current["outcome_labels"] = label_counts
            current["updated_at"] = observed_at
            current["weight"] = weight
            self.state_store.upsert_learning_pattern(key, current)

    def _decay_payload(self, payload: Dict[str, Any], observed_at: str) -> Dict[str, Any]:
        if not payload:
            return {}
        updated_at = payload.get("updated_at")
        if not updated_at:
            return dict(payload)
        elapsed_days = max((_parse_iso(observed_at) - _parse_iso(updated_at)).total_seconds() / 86400.0, 0.0)
        half_life_days = max(float(getattr(self.config, "learning_decay_half_life_days", 45.0)), 1.0)
        if self._drift_summary().get("active", False):
            half_life_days *= 0.6
        decay = 0.5 ** (elapsed_days / half_life_days)
        decayed = dict(payload)
        for key in ("samples", "effective_samples", "wins", "losses", "total_r_multiple", "total_profit_loss", "stop_loss_exits"):
            decayed[key] = _safe_float(decayed.get(key, 0.0)) * decay
        label_counts = dict(decayed.get("outcome_labels", {}) or {})
        decayed["outcome_labels"] = {label: _safe_float(count) * decay for label, count in label_counts.items()}
        return decayed

    def _aggregate_stats(self, stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        prior_strength = max(float(getattr(self.config, "learning_prior_strength", 5.0)), 1.0)
        min_samples = float(getattr(self.config, "learning_min_effective_samples", 3.0))
        weighted_reliability = 0.0
        weighted_expectancy = 0.0
        weighted_win_edge = 0.0
        weighted_stop_rate = 0.0
        weighted_risk_bias = 0.0
        weighted_confidence_bonus = 0.0
        weighted_execution = 0.0
        weighted_structural = 0.0
        strongest_weight = 0.0
        labels: Dict[str, float] = {}
        matched_patterns: List[Dict[str, Any]] = []

        for item in stats:
            effective_samples = _safe_float(item.get("effective_samples", item.get("samples", 0.0)))
            if effective_samples <= 0:
                continue
            weight = _safe_float(item.get("weight", 0.0))
            reliability = effective_samples / (effective_samples + prior_strength)
            avg_r = _safe_float(item.get("total_r_multiple", 0.0)) / effective_samples
            win_rate = _safe_float(item.get("wins", 0.0)) / effective_samples
            stop_rate = _safe_float(item.get("stop_loss_exits", 0.0)) / effective_samples
            shrunk_avg_r = avg_r * reliability
            shrunk_win_edge = (win_rate - 0.5) * reliability
            active_weight = weight * reliability
            if effective_samples < min_samples:
                active_weight *= effective_samples / min_samples

            label_counts = dict(item.get("outcome_labels", {}) or {})
            execution_ratio = (
                _safe_float(label_counts.get("execution_quality_issue", 0.0))
                + _safe_float(label_counts.get("liquidity_friction", 0.0))
            ) / effective_samples
            structural_ratio = (
                _safe_float(label_counts.get("poor_setup_quality", 0.0))
                + _safe_float(label_counts.get("regime_mismatch", 0.0))
                + _safe_float(label_counts.get("research_conflict", 0.0))
                + _safe_float(label_counts.get("structural_repeat", 0.0))
            ) / effective_samples
            positive_ratio = (
                _safe_float(label_counts.get("positive_outcome", 0.0))
                + _safe_float(label_counts.get("strong_setup_quality", 0.0))
            ) / effective_samples

            weighted_reliability += active_weight
            weighted_expectancy += shrunk_avg_r * active_weight
            weighted_win_edge += shrunk_win_edge * active_weight
            weighted_stop_rate += stop_rate * active_weight
            weighted_execution += execution_ratio * active_weight
            weighted_structural += structural_ratio * active_weight
            weighted_risk_bias += ((positive_ratio * 0.06) - (execution_ratio * 0.08) - (structural_ratio * 0.12)) * active_weight
            weighted_confidence_bonus += ((positive_ratio * 0.02) - (structural_ratio * 0.03)) * active_weight
            strongest_weight = max(strongest_weight, weight)

            for label, count in label_counts.items():
                labels[label] = labels.get(label, 0.0) + (_safe_float(count) * active_weight)
            matched_patterns.append(
                {
                    "pattern_key": item.get("pattern_key"),
                    "effective_samples": round(effective_samples, 3),
                    "avg_r_multiple": round(avg_r, 3),
                    "win_rate": round(win_rate, 3),
                    "weight": round(weight, 3),
                }
            )

        if weighted_reliability <= 0:
            return self._empty_aggregate()

        shrunk_avg_r = weighted_expectancy / weighted_reliability
        shrunk_win_edge = weighted_win_edge / weighted_reliability
        stop_rate = weighted_stop_rate / weighted_reliability
        execution_problem_ratio = weighted_execution / weighted_reliability
        structural_loss_ratio = weighted_structural / weighted_reliability
        expectancy_score = shrunk_avg_r * 8.5
        win_rate_score = shrunk_win_edge * 16.0
        attribution_score = 0.0
        if structural_loss_ratio > 0.0:
            attribution_score -= structural_loss_ratio * 5.5
        if execution_problem_ratio > 0.0:
            attribution_score -= execution_problem_ratio * 2.5
        if stop_rate > 0.45:
            attribution_score -= (stop_rate - 0.45) * 5.0
        dominant_attributions = [label for label, _ in sorted(labels.items(), key=lambda row: row[1], reverse=True)[:4]]
        evidence_samples = sum(_safe_float(item.get("effective_samples", item.get("samples", 0.0))) for item in stats)
        evidence_quality = "strong" if evidence_samples >= 12 else "moderate" if evidence_samples >= 5 else "thin"
        return {
            "effective_samples": evidence_samples,
            "shrunk_avg_r": shrunk_avg_r,
            "matched_key_strength": strongest_weight,
            "expectancy_score": expectancy_score,
            "win_rate_score": win_rate_score,
            "attribution_score": attribution_score,
            "risk_bias": weighted_risk_bias / weighted_reliability,
            "confidence_bonus": weighted_confidence_bonus / weighted_reliability,
            "structural_loss_ratio": structural_loss_ratio,
            "execution_problem_ratio": execution_problem_ratio,
            "reasons": dominant_attributions[:3],
            "dominant_attributions": dominant_attributions,
            "evidence_quality": evidence_quality,
            "matched_patterns": matched_patterns[:8],
            "summary": f"Avg learned expectancy {shrunk_avg_r:.2f}R with {evidence_quality} evidence across {evidence_samples:.1f} effective samples.",
        }

    def _empty_aggregate(self) -> Dict[str, Any]:
        return {
            "effective_samples": 0.0,
            "shrunk_avg_r": 0.0,
            "matched_key_strength": 0.0,
            "expectancy_score": 0.0,
            "win_rate_score": 0.0,
            "attribution_score": 0.0,
            "risk_bias": 0.0,
            "confidence_bonus": 0.0,
            "structural_loss_ratio": 0.0,
            "execution_problem_ratio": 0.0,
            "reasons": [],
            "dominant_attributions": [],
            "evidence_quality": "insufficient",
            "matched_patterns": [],
            "summary": "No meaningful learning evidence yet.",
        }

    def _safe_positive_updates_allowed(self, aggregate: Dict[str, Any], calibration: Dict[str, Any], drift: Dict[str, Any]) -> bool:
        return bool(
            aggregate["effective_samples"] >= float(getattr(self.config, "learning_positive_update_min_samples", 8.0))
            and calibration.get("calibration_gap", 1.0) <= float(getattr(self.config, "learning_positive_update_max_calibration_gap", 0.18))
            and not drift.get("active", False)
        )

    def _family_rotation_summary(self, current_strategy: str) -> Dict[str, Any]:
        if self.state_store is None:
            return {"status": "neutral"}
        window = max(int(getattr(self.config, "learning_family_rotation_window", 18) or 18), 4)
        min_samples = float(getattr(self.config, "learning_family_rotation_min_samples", 3.0))
        recent = self.state_store.load_recent_learning_observations(limit=window)
        by_strategy: Dict[str, Dict[str, float]] = {}
        for item in recent:
            strategy = str(item.get("strategy", "unknown"))
            bucket = by_strategy.setdefault(strategy, {"samples": 0.0, "wins": 0.0, "sum_r": 0.0})
            bucket["samples"] += 1.0
            bucket["wins"] += 1.0 if item.get("success") else 0.0
            bucket["sum_r"] += _safe_float(item.get("r_multiple", 0.0))
        ranked: list[Dict[str, Any]] = []
        for strategy, payload in by_strategy.items():
            samples = float(payload.get("samples", 0.0) or 0.0)
            if samples < min_samples:
                continue
            win_rate = float(payload.get("wins", 0.0) or 0.0) / samples
            avg_r = float(payload.get("sum_r", 0.0) or 0.0) / samples
            ranked.append(
                {
                    "strategy": strategy,
                    "samples": samples,
                    "win_rate": win_rate,
                    "avg_r_multiple": avg_r,
                }
            )
        if len(ranked) < 2:
            return {"status": "neutral", "strategies": ranked}
        worst = min(ranked, key=lambda item: (item["avg_r_multiple"], item["win_rate"]))
        best = max(ranked, key=lambda item: (item["avg_r_multiple"], item["win_rate"]))
        worst_is_bad = bool(
            worst["avg_r_multiple"] <= float(getattr(self.config, "learning_family_rotation_negative_avg_r", -0.20))
            and worst["win_rate"] <= float(getattr(self.config, "learning_family_rotation_negative_win_rate", 0.35))
        )
        best_is_good = bool(
            best["avg_r_multiple"] >= float(getattr(self.config, "learning_family_rotation_positive_avg_r", 0.15))
            and best["win_rate"] >= float(getattr(self.config, "learning_family_rotation_positive_win_rate", 0.52))
        )
        hard_gap_r = best["avg_r_multiple"] - worst["avg_r_multiple"]
        hard_gap_win_rate = best["win_rate"] - worst["win_rate"]
        hard_worst_is_bad = bool(
            worst["avg_r_multiple"] <= float(getattr(self.config, "learning_family_rotation_hard_negative_avg_r", -0.45))
            and worst["win_rate"] <= float(getattr(self.config, "learning_family_rotation_hard_negative_win_rate", 0.28))
        )
        hard_best_is_good = bool(
            best["avg_r_multiple"] >= float(getattr(self.config, "learning_family_rotation_hard_positive_avg_r", 0.22))
            and best["win_rate"] >= float(getattr(self.config, "learning_family_rotation_hard_positive_win_rate", 0.56))
        )
        hard_rotation = bool(
            hard_worst_is_bad
            and hard_best_is_good
            and hard_gap_r >= float(getattr(self.config, "learning_family_rotation_hard_gap_r", 0.35))
            and hard_gap_win_rate >= float(getattr(self.config, "learning_family_rotation_hard_gap_win_rate", 0.18))
        )
        recovery_window = max(int(getattr(self.config, "learning_family_rotation_recovery_window", 4) or 4), 2)
        recovery_min_samples = float(getattr(self.config, "learning_family_rotation_recovery_min_samples", 2.0))
        recovery_recent = [
            item for item in self.state_store.load_recent_learning_observations(limit=recovery_window * 3)
            if str(item.get("strategy", "unknown")) == worst["strategy"]
        ][:recovery_window]
        recovery_samples = float(len(recovery_recent))
        recovery_win_rate = (
            sum(1.0 for item in recovery_recent if item.get("success")) / recovery_samples
            if recovery_samples else 0.0
        )
        recovery_avg_r = (
            sum(_safe_float(item.get("r_multiple", 0.0)) for item in recovery_recent) / recovery_samples
            if recovery_samples else 0.0
        )
        recovery_active = bool(
            recovery_samples >= recovery_min_samples
            and recovery_avg_r >= float(getattr(self.config, "learning_family_rotation_recovery_avg_r", 0.08))
            and recovery_win_rate >= float(getattr(self.config, "learning_family_rotation_recovery_win_rate", 0.55))
        )
        status = "neutral"
        if hard_rotation and current_strategy == worst["strategy"]:
            status = "suppress_current_hard"
        elif hard_rotation and current_strategy == best["strategy"]:
            status = "promote_current_hard"
        elif worst_is_bad and current_strategy == worst["strategy"]:
            status = "suppress_current"
        elif worst_is_bad and best_is_good and current_strategy == best["strategy"]:
            status = "promote_current"
        if recovery_active:
            if status == "suppress_current_hard" and current_strategy == worst["strategy"]:
                status = "suppress_current"
            elif status == "promote_current_hard" and current_strategy == best["strategy"]:
                status = "promote_current"
        return {
            "status": status,
            "worst_family": worst["strategy"],
            "best_family": best["strategy"],
            "worst_family_samples": worst["samples"],
            "best_family_samples": best["samples"],
            "worst_family_avg_r_multiple": worst["avg_r_multiple"],
            "best_family_avg_r_multiple": best["avg_r_multiple"],
            "worst_family_win_rate": worst["win_rate"],
            "best_family_win_rate": best["win_rate"],
            "hard_rotation_active": hard_rotation,
            "gap_avg_r_multiple": hard_gap_r,
            "gap_win_rate": hard_gap_win_rate,
            "recovery_active": recovery_active,
            "recovery_samples": recovery_samples,
            "recovery_avg_r_multiple": recovery_avg_r,
            "recovery_win_rate": recovery_win_rate,
            "strategies": ranked,
        }

    def _positive_cell_evidence(
        self,
        opportunity: Dict[str, Any],
        prequential: Dict[str, Any],
        calibration: Dict[str, Any],
        drift: Dict[str, Any],
    ) -> bool:
        if drift.get("active", False):
            return False
        calibration_gap = float(calibration.get("calibration_gap", 1.0) or 1.0)
        if calibration_gap > float(getattr(self.config, "learning_positive_calibration_slack", 0.24)):
            return False
        opportunity_positive = bool(
            float(opportunity.get("samples", 0.0) or 0.0) >= (
                float(getattr(self.config, "learning_positive_opportunity_min_samples", 6.0))
                - (float(getattr(self.config, "learning_bucket_positive_sample_delta", 2.0)) if opportunity.get("bucket_specific") else 0.0)
            )
            and float(opportunity.get("avg_forward_r", 0.0) or 0.0) >= float(getattr(self.config, "learning_positive_opportunity_avg_forward_r", 0.22))
            and float(opportunity.get("positive_ratio", 0.0) or 0.0) >= float(getattr(self.config, "learning_positive_opportunity_positive_ratio", 0.55))
        )
        prequential_positive = bool(
            float(prequential.get("samples", 0.0) or 0.0) >= (
                float(getattr(self.config, "learning_positive_prequential_min_samples", 6.0))
                - (float(getattr(self.config, "learning_bucket_positive_sample_delta", 2.0)) if prequential.get("bucket_specific") else 0.0)
            )
            and float(prequential.get("avg_r_multiple", 0.0) or 0.0) >= float(getattr(self.config, "learning_positive_prequential_avg_r", 0.18))
            and float(prequential.get("win_rate", 0.0) or 0.0) >= float(getattr(self.config, "learning_positive_prequential_win_rate", 0.52))
            and float(prequential.get("brier_score", 1.0) or 1.0) <= float(getattr(self.config, "learning_positive_prequential_max_brier", 0.30))
        )
        return bool(opportunity_positive or prequential_positive)

    def _update_drift_monitors(self, observation: Dict[str, Any]) -> None:
        strategy = str(observation.get("strategy", "unknown"))
        side = str(observation.get("side", "flat"))
        r_multiple = _safe_float(observation.get("r_multiple", 0.0))
        success = 1.0 if observation.get("success") else 0.0
        execution = dict(observation.get("execution_context", {}) or {})
        decision_context = dict(observation.get("decision_context", {}) or {})
        regime = str(decision_context.get("regime", "unknown"))
        order_profile = str(decision_context.get("order_profile", "limit"))
        symbol_bucket = str(decision_context.get("symbol_bucket", self._symbol_bucket(observation.get("symbol", ""))))
        execution_friction = max(_safe_float(execution.get("spread_bps", 0.0)) / 20.0, 0.0) + max(0.9 - _safe_float(execution.get("fill_fraction", 1.0), 1.0), 0.0)
        observed_at = str(observation.get("observed_at") or utcnow_iso())
        self.drift_monitor.update_stream(f"global::r_multiple", r_multiple, observed_at)
        self.drift_monitor.update_stream(f"strategy::{strategy}::r_multiple", r_multiple, observed_at)
        self.drift_monitor.update_stream(f"strategy::{strategy}::success", success, observed_at)
        self.drift_monitor.update_stream(f"strategy::{strategy}::execution_friction", execution_friction, observed_at)
        cell_key = f"cell::{strategy}::{side}::{regime}::{order_profile}"
        self.drift_monitor.update_stream(f"{cell_key}::r_multiple", r_multiple, observed_at)
        self.drift_monitor.update_stream(f"{cell_key}::success", success, observed_at)
        bucket_cell_key = f"cell::{strategy}::{side}::{regime}::{symbol_bucket}::{order_profile}"
        self.drift_monitor.update_stream(f"{bucket_cell_key}::r_multiple", r_multiple, observed_at)
        self.drift_monitor.update_stream(f"{bucket_cell_key}::success", success, observed_at)

    def _drift_summary(
        self,
        strategy: str | None = None,
        symbol_bucket: str | None = None,
        regime: str | None = None,
        side: str | None = None,
        order_profile: str | None = None,
    ) -> Dict[str, Any]:
        streams = [self.drift_monitor.stream_state("global::r_multiple")]
        if strategy:
            streams.append(self.drift_monitor.stream_state(f"strategy::{strategy}::r_multiple"))
            streams.append(self.drift_monitor.stream_state(f"strategy::{strategy}::success"))
        if strategy and side and regime and order_profile:
            cell_key = f"cell::{strategy}::{side}::{regime}::{order_profile}"
            streams.append(self.drift_monitor.stream_state(f"{cell_key}::r_multiple"))
            streams.append(self.drift_monitor.stream_state(f"{cell_key}::success"))
        if strategy and side and regime and order_profile and symbol_bucket:
            bucket_cell_key = f"cell::{strategy}::{side}::{regime}::{symbol_bucket}::{order_profile}"
            streams.append(self.drift_monitor.stream_state(f"{bucket_cell_key}::r_multiple"))
            streams.append(self.drift_monitor.stream_state(f"{bucket_cell_key}::success"))
        severity = 0.0
        active = False
        active_streams = []
        for stream in streams:
            if not stream:
                continue
            stream_severity = _safe_float(stream.get("severity", 0.0))
            severity = max(severity, stream_severity)
            if stream.get("drift_detected"):
                active = True
                active_streams.append(stream.get("stream_key"))
        return {
            "active": active,
            "severity": severity,
            "streams": [item for item in active_streams[:5]],
            "regime": regime or "unknown",
        }

    def _evaluate_shadow_outcome(self, exchange_client: Any, decision: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(decision.get("symbol", ""))
        timeframe = str(decision.get("timeframe", "15m"))
        horizon_minutes = int(decision.get("expected_holding_minutes", 480) or 480)
        bars = max(int(math.ceil(horizon_minutes / max(self._timeframe_minutes(timeframe), 1))) + 2, 3)
        candles = exchange_client.fetch_ohlcv(symbol, timeframe, limit=bars) or []
        if not candles:
            return {"evaluated": False, "forward_r_multiple": 0.0, "success": False}
        highs = [float(candle[2]) for candle in candles]
        lows = [float(candle[3]) for candle in candles]
        last_close = float(candles[-1][4])
        entry = _safe_float(decision.get("entry_price", 0.0))
        stop_loss = _safe_float(decision.get("stop_loss", 0.0))
        take_profit = _safe_float(decision.get("take_profit", 0.0))
        side = str(decision.get("side", "")).lower()
        risk = abs(entry - stop_loss)
        if risk <= 0 or entry <= 0:
            return {"evaluated": False, "forward_r_multiple": 0.0, "success": False}
        if side in {"long", "buy"}:
            forward_r = (last_close - entry) / risk
            max_favorable_r = (max(highs) - entry) / risk
            max_adverse_r = (min(lows) - entry) / risk
            success = max_favorable_r >= ((take_profit - entry) / risk if take_profit > entry else 1.0)
        else:
            forward_r = (entry - last_close) / risk
            max_favorable_r = (entry - min(lows)) / risk
            max_adverse_r = (entry - max(highs)) / risk
            success = max_favorable_r >= ((entry - take_profit) / risk if take_profit < entry else 1.0)
        return {
            "evaluated": True,
            "forward_r_multiple": forward_r,
            "max_favorable_r_multiple": max_favorable_r,
            "max_adverse_r_multiple": max_adverse_r,
            "success": success,
        }

    def _update_opportunity_models(self, decision: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        if self.state_store is None or not outcome.get("evaluated"):
            return
        strategy = str(decision.get("strategy", "unknown"))
        side = str(decision.get("side", "flat"))
        reason = str(decision.get("reason", "unknown"))
        decision_context = dict(decision.get("decision_context", {}) or {})
        regime = str(decision_context.get("regime", "unknown"))
        order_profile = str(decision_context.get("order_profile", "limit"))
        symbol_bucket = str(decision_context.get("symbol_bucket", self._symbol_bucket(decision.get("symbol", ""))))
        observed_at = str(decision.get("evaluated_at", utcnow_iso()))
        for model_key in (
            f"learning_opportunity::global",
            f"learning_opportunity::strategy::{strategy}",
            f"learning_opportunity::cell::{strategy}::{side}::{regime}::{order_profile}",
            f"learning_opportunity::cell::{strategy}::{side}::{regime}::{symbol_bucket}::{order_profile}",
            f"learning_opportunity::reason::{reason}",
        ):
            payload = self.state_store.load_learning_model(model_key) or {}
            payload["total"] = _safe_float(payload.get("total", 0.0)) + 1.0
            payload["positive_forward"] = _safe_float(payload.get("positive_forward", 0.0)) + (1.0 if _safe_float(outcome.get("forward_r_multiple", 0.0)) > 0 else 0.0)
            payload["total_forward_r"] = _safe_float(payload.get("total_forward_r", 0.0)) + _safe_float(outcome.get("forward_r_multiple", 0.0))
            payload["updated_at"] = observed_at
            self.state_store.upsert_learning_model(model_key, payload)
        if str(decision.get("status", "")) == "skipped":
            for model_key in (
                f"learning_missed::reason::{reason}",
                f"learning_missed::strategy::{reason}::{strategy}",
                f"learning_missed::cell::{reason}::{strategy}::{side}::{regime}::{order_profile}",
                f"learning_missed::cell::{reason}::{strategy}::{side}::{regime}::{symbol_bucket}::{order_profile}",
            ):
                payload = self.state_store.load_learning_model(model_key) or {}
                payload["total"] = _safe_float(payload.get("total", 0.0)) + 1.0
                payload["positive_forward"] = _safe_float(payload.get("positive_forward", 0.0)) + (1.0 if _safe_float(outcome.get("forward_r_multiple", 0.0)) > 0 else 0.0)
                payload["total_forward_r"] = _safe_float(payload.get("total_forward_r", 0.0)) + _safe_float(outcome.get("forward_r_multiple", 0.0))
                payload["updated_at"] = observed_at
                self.state_store.upsert_learning_model(model_key, payload)

    def _opportunity_summary(
        self,
        strategy: str | None = None,
        symbol_bucket: str | None = None,
        side: str | None = None,
        regime: str | None = None,
        order_profile: str | None = None,
    ) -> Dict[str, Any]:
        global_payload = self.state_store.load_learning_model("learning_opportunity::global") if self.state_store is not None else {}
        strategy_payload = self.state_store.load_learning_model(f"learning_opportunity::strategy::{strategy}") if (self.state_store is not None and strategy) else {}
        bucket_cell_payload = (
            self.state_store.load_learning_model(f"learning_opportunity::cell::{strategy}::{side}::{regime}::{symbol_bucket}::{order_profile}")
            if (self.state_store is not None and strategy and side and regime and order_profile and symbol_bucket)
            else {}
        )
        cell_payload = (
            self.state_store.load_learning_model(f"learning_opportunity::cell::{strategy}::{side}::{regime}::{order_profile}")
            if (self.state_store is not None and strategy and side and regime and order_profile)
            else {}
        )
        payload = bucket_cell_payload or cell_payload or strategy_payload or global_payload or {}
        total = _safe_float(payload.get("total", 0.0))
        if total <= 0:
            return {"score_bias": 0.0, "risk_bias": 0.0, "avg_forward_r": 0.0, "positive_ratio": 0.0, "samples": 0.0, "bucket_specific": False}
        avg_forward_r = _safe_float(payload.get("total_forward_r", 0.0)) / total
        positive_ratio = _safe_float(payload.get("positive_forward", 0.0)) / total
        score_bias = _clamp(avg_forward_r * 0.9, -0.9, 1.0) if total >= float(getattr(self.config, "learning_opportunity_min_samples", 6.0)) else 0.0
        risk_bias = _clamp((positive_ratio - 0.5) * 0.10, -0.05, 0.05) if total >= float(getattr(self.config, "learning_opportunity_min_samples", 6.0)) else 0.0
        return {
            "score_bias": score_bias,
            "risk_bias": risk_bias,
            "avg_forward_r": avg_forward_r,
            "positive_ratio": positive_ratio,
            "samples": total,
            "bucket_specific": bool(bucket_cell_payload),
        }

    def _missed_opportunity_summary(
        self,
        *,
        reason: str,
        strategy: str | None = None,
        symbol_bucket: str | None = None,
        side: str | None = None,
        regime: str | None = None,
        order_profile: str | None = None,
    ) -> Dict[str, Any]:
        if self.state_store is None:
            return {"avg_forward_r": 0.0, "positive_ratio": 0.0, "samples": 0.0, "bucket_specific": False}
        reason_payload = self.state_store.load_learning_model(f"learning_missed::reason::{reason}") or {}
        strategy_payload = self.state_store.load_learning_model(f"learning_missed::strategy::{reason}::{strategy}") if strategy else {}
        bucket_payload = (
            self.state_store.load_learning_model(f"learning_missed::cell::{reason}::{strategy}::{side}::{regime}::{symbol_bucket}::{order_profile}")
            if (strategy and side and regime and order_profile and symbol_bucket)
            else {}
        ) or {}
        cell_payload = (
            self.state_store.load_learning_model(f"learning_missed::cell::{reason}::{strategy}::{side}::{regime}::{order_profile}")
            if (strategy and side and regime and order_profile)
            else {}
        ) or {}
        payload = bucket_payload or cell_payload or strategy_payload or reason_payload or {}
        total = _safe_float(payload.get("total", 0.0))
        if total <= 0:
            return {"avg_forward_r": 0.0, "positive_ratio": 0.0, "samples": 0.0, "bucket_specific": False}
        return {
            "avg_forward_r": _safe_float(payload.get("total_forward_r", 0.0)) / total,
            "positive_ratio": _safe_float(payload.get("positive_forward", 0.0)) / total,
            "samples": total,
            "bucket_specific": bool(bucket_payload),
        }

    def _update_prequential_models(
        self,
        decision_context: Dict[str, Any],
        *,
        success: bool,
        realized_r_multiple: float,
        observed_at: str,
        source: str,
    ) -> None:
        if self.state_store is None:
            return
        strategy = str(decision_context.get("strategy", "unknown"))
        side = str(decision_context.get("side", "flat"))
        regime = str(decision_context.get("regime", "unknown"))
        order_profile = str(decision_context.get("order_profile", "limit"))
        symbol_bucket = str(decision_context.get("symbol_bucket", self._symbol_bucket(decision_context.get("symbol", ""))))
        raw_confidence = _clamp(_safe_float(decision_context.get("signal_quality", 0.5), 0.5), 0.01, 0.99)
        outcome = 1.0 if success else 0.0
        brier = (raw_confidence - outcome) ** 2
        for model_key in (
            "learning_prequential::global",
            f"learning_prequential::strategy::{strategy}",
            f"learning_prequential::cell::{strategy}::{side}::{regime}::{order_profile}",
            f"learning_prequential::cell::{strategy}::{side}::{regime}::{symbol_bucket}::{order_profile}",
            f"learning_prequential::source::{source}",
        ):
            payload = self.state_store.load_learning_model(model_key) or {}
            payload["count"] = _safe_float(payload.get("count", 0.0)) + 1.0
            payload["wins"] = _safe_float(payload.get("wins", 0.0)) + outcome
            payload["sum_brier"] = _safe_float(payload.get("sum_brier", 0.0)) + brier
            payload["sum_confidence"] = _safe_float(payload.get("sum_confidence", 0.0)) + raw_confidence
            payload["sum_r_multiple"] = _safe_float(payload.get("sum_r_multiple", 0.0)) + realized_r_multiple
            payload["updated_at"] = observed_at
            self.state_store.upsert_learning_model(model_key, payload)

    def _prequential_summary(
        self,
        strategy: str | None = None,
        symbol_bucket: str | None = None,
        side: str | None = None,
        regime: str | None = None,
        order_profile: str | None = None,
    ) -> Dict[str, Any]:
        if self.state_store is None:
            return {"status": "disabled"}
        payload = {}
        if strategy and side and regime and order_profile and symbol_bucket:
            payload = self.state_store.load_learning_model(f"learning_prequential::cell::{strategy}::{side}::{regime}::{symbol_bucket}::{order_profile}") or {}
        if strategy and side and regime and order_profile:
            payload = payload or (self.state_store.load_learning_model(f"learning_prequential::cell::{strategy}::{side}::{regime}::{order_profile}") or {})
        if not payload and strategy:
            payload = self.state_store.load_learning_model(f"learning_prequential::strategy::{strategy}") or {}
        if not payload:
            payload = self.state_store.load_learning_model("learning_prequential::global") or {}
        count = _safe_float(payload.get("count", 0.0))
        if count <= 0:
            return {
                "status": "ok",
                "samples": 0.0,
                "brier_score": 0.0,
                "avg_confidence": 0.0,
                "win_rate": 0.0,
                "avg_r_multiple": 0.0,
                "score_bias": 0.0,
                "confidence_delta": 0.0,
                "risk_bias": 0.0,
                "bucket_specific": False,
            }
        avg_confidence = _safe_float(payload.get("sum_confidence", 0.0)) / count
        win_rate = _safe_float(payload.get("wins", 0.0)) / count
        avg_r_multiple = _safe_float(payload.get("sum_r_multiple", 0.0)) / count
        brier_score = _safe_float(payload.get("sum_brier", 0.0)) / count
        score_bias = 0.0
        confidence_delta = 0.0
        risk_bias = 0.0
        min_samples = float(getattr(self.config, "learning_cell_gate_min_samples", 6.0))
        if count >= min_samples:
            score_bias = _clamp((avg_r_multiple * 0.9) + ((win_rate - 0.5) * 2.5), -1.0, 1.0)
            confidence_delta = _clamp((win_rate - avg_confidence) * 0.08, -0.05, 0.05)
            risk_bias = _clamp((win_rate - 0.5) * 0.10, -0.04, 0.04)
        return {
            "status": "ok",
            "samples": count,
            "brier_score": brier_score,
            "avg_confidence": avg_confidence,
            "win_rate": win_rate,
            "avg_r_multiple": avg_r_multiple,
            "score_bias": score_bias,
            "confidence_delta": confidence_delta,
            "risk_bias": risk_bias,
            "bucket_specific": bool(strategy and side and regime and order_profile and symbol_bucket and payload),
        }

    def _pattern_features(
        self,
        *,
        strategy: str,
        symbol_bucket: str,
        side: str,
        regime: str,
        trend_direction: str,
        strategy_variant: str,
        order_profile: str,
        confidence: float,
        rr_ratio: float,
        expected_edge_bps: float,
        research: Dict[str, Any],
        hurst: float,
        regime_confidence: float,
        volume_impulse: float,
        spread_bps: float,
        entry_deviation_bps: float,
        fill_fraction: float,
        latency_ms: float,
        fast_move: bool,
    ) -> Dict[str, Any]:
        bullish = _safe_float(research.get("bullish_confidence", 0.0))
        bearish = _safe_float(research.get("bearish_confidence", 0.0))
        risk_off = _safe_float(research.get("risk_off_confidence", 0.0))
        if risk_off >= 0.75:
            research_state = "risk_off"
        elif bullish > bearish and bearish > 0:
            research_state = "contested_bullish"
        elif bearish > bullish and bullish > 0:
            research_state = "contested_bearish"
        elif bullish > bearish:
            research_state = "aligned_bullish"
        elif bearish > bullish:
            research_state = "aligned_bearish"
        else:
            research_state = "neutral"
        return {
            "strategy": strategy,
            "symbol_bucket": symbol_bucket,
            "strategy_variant": strategy_variant,
            "side": side,
            "regime": regime,
            "order_profile": order_profile,
            "trend_direction": trend_direction,
            "confidence_bucket": self._bucket(confidence, [(0.58, "low"), (0.74, "mid"), (10.0, "high")]),
            "rr_bucket": self._bucket(rr_ratio, [(1.25, "thin"), (1.9, "good"), (10.0, "strong")]),
            "edge_bucket": self._bucket(expected_edge_bps, [(12.0, "thin"), (25.0, "good"), (10_000.0, "strong")]),
            "hurst_bucket": self._bucket(hurst, [(0.25, "low"), (0.55, "mid"), (1.0, "high")]),
            "regime_conf_bucket": self._bucket(regime_confidence, [(0.58, "weak"), (0.74, "good"), (1.0, "strong")]),
            "volume_bucket": self._bucket(volume_impulse, [(0.95, "soft"), (1.1, "normal"), (10.0, "strong")]),
            "spread_bucket": self._bucket(spread_bps, [(8.0, "tight"), (18.0, "normal"), (10_000.0, "wide")]),
            "entry_efficiency_bucket": self._bucket(entry_deviation_bps, [(4.0, "precise"), (12.0, "acceptable"), (10_000.0, "stretched")]),
            "fill_bucket": self._bucket(fill_fraction, [(0.8, "partial"), (0.93, "good"), (1.1, "full")]),
            "latency_bucket": self._bucket(latency_ms, [(250.0, "low"), (750.0, "normal"), (100_000.0, "high")]),
            "research_state": research_state,
            "fast_move_bucket": "fast" if fast_move else "normal",
        }

    def _execution_quality_bucket(self, spread_bps: float, entry_deviation_bps: float, fill_fraction: float) -> str:
        if spread_bps >= 25.0 or entry_deviation_bps >= 15.0 or fill_fraction < 0.72:
            return "poor"
        if spread_bps >= 10.0 or entry_deviation_bps >= 6.0 or fill_fraction < 0.9:
            return "mixed"
        return "clean"

    @staticmethod
    def _bucket(value: float, levels: List[tuple[float, str]]) -> str:
        for threshold, label in levels:
            if value < threshold:
                return label
        return levels[-1][1]

    def _pattern_keys(self, pattern: Dict[str, Any]) -> Dict[str, float]:
        exact = json.dumps(pattern, sort_keys=True, separators=(",", ":"))
        setup = json.dumps(
            {
                "strategy": pattern.get("strategy"),
                "strategy_variant": pattern.get("strategy_variant"),
                "side": pattern.get("side"),
                "regime": pattern.get("regime"),
                "symbol_bucket": pattern.get("symbol_bucket"),
                "order_profile": pattern.get("order_profile"),
                "confidence_bucket": pattern.get("confidence_bucket"),
                "rr_bucket": pattern.get("rr_bucket"),
                "edge_bucket": pattern.get("edge_bucket"),
                "research_state": pattern.get("research_state"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        strategy_regime = json.dumps(
            {
                "strategy": pattern.get("strategy"),
                "strategy_variant": pattern.get("strategy_variant"),
                "regime": pattern.get("regime"),
                "symbol_bucket": pattern.get("symbol_bucket"),
                "side": pattern.get("side"),
                "order_profile": pattern.get("order_profile"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        regime = json.dumps(
            {
                "strategy": pattern.get("strategy"),
                "side": pattern.get("side"),
                "regime": pattern.get("regime"),
                "trend_direction": pattern.get("trend_direction"),
                "hurst_bucket": pattern.get("hurst_bucket"),
                "regime_conf_bucket": pattern.get("regime_conf_bucket"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        execution = json.dumps(
            {
                "strategy": pattern.get("strategy"),
                "strategy_variant": pattern.get("strategy_variant"),
                "side": pattern.get("side"),
                "spread_bucket": pattern.get("spread_bucket"),
                "entry_efficiency_bucket": pattern.get("entry_efficiency_bucket"),
                "fill_bucket": pattern.get("fill_bucket"),
                "fast_move_bucket": pattern.get("fast_move_bucket"),
                "order_profile": pattern.get("order_profile"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        family = json.dumps(
            {
                "strategy": pattern.get("strategy"),
                "symbol_bucket": pattern.get("symbol_bucket"),
                "side": pattern.get("side"),
                "order_profile": pattern.get("order_profile"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            exact: 1.0,
            setup: 0.84,
            strategy_regime: 0.78,
            regime: 0.70,
            execution: 0.55,
            family: 0.42,
        }

    def _timeframe_minutes(self, timeframe: str) -> int:
        value = timeframe.strip().lower()
        if value.endswith("m"):
            return max(int(value[:-1] or 1), 1)
        if value.endswith("h"):
            return max(int(value[:-1] or 1), 1) * 60
        if value.endswith("d"):
            return max(int(value[:-1] or 1), 1) * 1440
        return 15

    def _symbol_bucket(self, symbol: Any) -> str:
        base = str(symbol or "").split("/")[0].upper()
        if base in {"BTC", "ETH"}:
            return "majors"
        if base in {"BNB", "XRP"}:
            return "exchange_beta"
        if base in {"SOL", "AVAX"}:
            return "high_beta_alts"
        if base in {"ADA", "DOT", "LINK", "TON"}:
            return "slower_large_caps"
        return "other"
