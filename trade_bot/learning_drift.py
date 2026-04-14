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


class LearningDriftMonitor:
    """
    Lightweight online drift monitor using EWMA residuals plus two-sided CUSUM.

    This is intentionally simple and inspectable:
    - one state blob per stream key
    - no heavy dependencies
    - online updates suitable for live and future sequential simulation
    """

    def __init__(self, config: Any, state_store: Any):
        self.config = config
        self.state_store = state_store

    def update_stream(self, stream_key: str, value: float, observed_at: str | None = None) -> Dict[str, Any]:
        if self.state_store is None:
            return {"stream_key": stream_key, "drift_detected": False, "severity": 0.0, "state": {}}
        model_key = f"learning_drift::{stream_key}"
        state = self.state_store.load_learning_model(model_key) or {}
        alpha = max(min(float(getattr(self.config, "learning_drift_ewma_alpha", 0.12)), 0.5), 0.01)
        threshold = max(float(getattr(self.config, "learning_drift_threshold", 2.8)), 0.5)
        slack = max(float(getattr(self.config, "learning_drift_slack", 0.15)), 0.0)
        decay = max(min(float(getattr(self.config, "learning_drift_cusum_decay", 0.92)), 0.999), 0.5)

        mean = _safe_float(state.get("mean", value), value)
        variance = max(_safe_float(state.get("variance", 1.0), 1.0), 1e-6)
        positive_cusum = _safe_float(state.get("positive_cusum", 0.0))
        negative_cusum = _safe_float(state.get("negative_cusum", 0.0))
        count = int(state.get("count", 0))

        residual = value - mean
        std = math.sqrt(variance)
        normalized = residual / max(std, 1e-6)
        positive_cusum = max(0.0, (positive_cusum * decay) + normalized - slack)
        negative_cusum = max(0.0, (negative_cusum * decay) - normalized - slack)
        drift_detected = positive_cusum >= threshold or negative_cusum >= threshold
        severity = max(positive_cusum, negative_cusum) / threshold if threshold > 0 else 0.0

        mean = ((1.0 - alpha) * mean) + (alpha * value)
        variance = max(((1.0 - alpha) * variance) + (alpha * (residual ** 2)), 1e-6)

        updated_state = {
            "stream_key": stream_key,
            "mean": mean,
            "variance": variance,
            "positive_cusum": positive_cusum,
            "negative_cusum": negative_cusum,
            "last_value": value,
            "last_residual": residual,
            "drift_detected": drift_detected,
            "severity": severity,
            "count": count + 1,
            "updated_at": observed_at or dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        self.state_store.upsert_learning_model(model_key, updated_state)
        return updated_state

    def stream_state(self, stream_key: str) -> Dict[str, Any]:
        if self.state_store is None:
            return {}
        return self.state_store.load_learning_model(f"learning_drift::{stream_key}") or {}
