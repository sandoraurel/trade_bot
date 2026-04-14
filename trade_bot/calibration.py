from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict


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


class OnlineConfidenceCalibrator:
    """
    Small online calibrator based on hierarchical beta-binomial buckets.

    It is intentionally explicit and transparent:
    - confidence is bucketed
    - each bucket maintains successes/failures with priors
    - prediction shrinks toward the raw confidence
    """

    def __init__(self, config: Any, state_store: Any):
        self.config = config
        self.state_store = state_store

    def predict(self, raw_confidence: float, *, strategy: str, regime: str, side: str = "flat", order_profile: str = "limit") -> Dict[str, Any]:
        raw_confidence = _clamp(_safe_float(raw_confidence, 0.5), 0.01, 0.99)
        if self.state_store is None:
            return self._neutral(raw_confidence)
        bucket = self._bucket(raw_confidence)
        aggregate_success = 0.0
        aggregate_total = 0.0
        weighted_prob = 0.0
        weighted_gap = 0.0
        total_weight = 0.0
        parts = []
        for key, weight in self._keys(strategy, regime, side, order_profile, bucket).items():
            payload = self.state_store.load_learning_model(key) or {}
            successes = _safe_float(payload.get("successes", 0.0))
            total = _safe_float(payload.get("total", 0.0))
            prior = max(float(getattr(self.config, "learning_calibration_prior_strength", 6.0)), 1.0)
            posterior = (successes + (raw_confidence * prior)) / max(total + prior, 1e-6)
            reliability = total / (total + prior)
            active_weight = weight * max(reliability, 0.2)
            weighted_prob += posterior * active_weight
            weighted_gap += abs(_safe_float(payload.get("mean_gap", 0.0))) * active_weight
            total_weight += active_weight
            aggregate_success += successes * weight
            aggregate_total += total * weight
            parts.append({"key": key, "samples": total, "posterior": posterior, "weight": weight})
        if total_weight <= 0:
            return self._neutral(raw_confidence)
        calibrated = weighted_prob / total_weight
        calibration_gap = weighted_gap / total_weight
        return {
            "raw_confidence": raw_confidence,
            "calibrated_confidence": _clamp(calibrated, 0.01, 0.99),
            "confidence_delta": _clamp(calibrated - raw_confidence, -0.06, 0.06),
            "calibration_gap": calibration_gap,
            "effective_samples": aggregate_total,
            "parts": parts,
        }

    def update(
        self,
        raw_confidence: float,
        *,
        strategy: str,
        regime: str,
        side: str = "flat",
        order_profile: str = "limit",
        success: bool,
        observed_at: str | None = None,
    ) -> None:
        if self.state_store is None:
            return
        raw_confidence = _clamp(_safe_float(raw_confidence, 0.5), 0.01, 0.99)
        bucket = self._bucket(raw_confidence)
        outcome = 1.0 if success else 0.0
        alpha = max(min(float(getattr(self.config, "learning_calibration_gap_alpha", 0.15)), 0.5), 0.01)
        for key in self._keys(strategy, regime, side, order_profile, bucket).keys():
            payload = self.state_store.load_learning_model(key) or {}
            payload["successes"] = _safe_float(payload.get("successes", 0.0)) + outcome
            payload["total"] = _safe_float(payload.get("total", 0.0)) + 1.0
            last_gap = _safe_float(payload.get("mean_gap", 0.0))
            payload["mean_gap"] = ((1.0 - alpha) * last_gap) + (alpha * (outcome - raw_confidence))
            payload["updated_at"] = observed_at or dt.datetime.now(dt.timezone.utc).isoformat()
            self.state_store.upsert_learning_model(key, payload)

    def summary(self, limit: int = 10) -> Dict[str, Any]:
        if self.state_store is None:
            return {"status": "disabled"}
        rows = self.state_store.list_learning_models(prefix="learning_calibration::", limit=limit)
        ranked = []
        for key, payload in rows:
            ranked.append(
                {
                    "key": key,
                    "samples": _safe_float(payload.get("total", 0.0)),
                    "mean_gap": round(_safe_float(payload.get("mean_gap", 0.0)), 4),
                }
            )
        return {"status": "ok", "models": ranked}

    def _neutral(self, raw_confidence: float) -> Dict[str, Any]:
        return {
            "raw_confidence": raw_confidence,
            "calibrated_confidence": raw_confidence,
            "confidence_delta": 0.0,
            "calibration_gap": 0.0,
            "effective_samples": 0.0,
            "parts": [],
        }

    def _bucket(self, raw_confidence: float) -> str:
        if raw_confidence < 0.55:
            return "very_low"
        if raw_confidence < 0.65:
            return "low"
        if raw_confidence < 0.75:
            return "mid"
        if raw_confidence < 0.85:
            return "high"
        return "very_high"

    def _keys(self, strategy: str, regime: str, side: str, order_profile: str, bucket: str) -> Dict[str, float]:
        return {
            f"learning_calibration::global::{bucket}": 0.35,
            f"learning_calibration::strategy::{strategy}::{bucket}": 0.75,
            f"learning_calibration::strategy_regime::{strategy}::{regime}::{bucket}": 1.0,
            f"learning_calibration::cell::{strategy}::{side}::{regime}::{order_profile}::{bucket}": 1.15,
        }
