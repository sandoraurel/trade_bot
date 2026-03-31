from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict

from .constants import STATE_DB_FILE
from .state_store import SQLiteStateStore


@dataclass
class RuntimeHealth:
    healthy: bool
    reason: str
    checked_at: str
    details: Dict[str, Any]


def _parse_iso8601(raw: Any) -> dt.datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def evaluate_runtime_health(base_dir: str, max_snapshot_age_seconds: int = 900) -> RuntimeHealth:
    now_utc = dt.datetime.now(dt.timezone.utc)
    checked_at = now_utc.isoformat()
    db_path = os.path.join(base_dir, STATE_DB_FILE)
    if not os.path.exists(db_path):
        return RuntimeHealth(
            healthy=False,
            reason="missing_runtime_db",
            checked_at=checked_at,
            details={"path": db_path},
        )

    store = SQLiteStateStore(db_path)
    snapshot = store.load_snapshot("runtime")
    if not snapshot:
        return RuntimeHealth(
            healthy=False,
            reason="missing_runtime_snapshot",
            checked_at=checked_at,
            details={"path": db_path},
        )

    updated_at_raw = snapshot.get("updated_at")
    updated_at = _parse_iso8601(updated_at_raw)
    if updated_at is None:
        return RuntimeHealth(
            healthy=False,
            reason="invalid_runtime_snapshot_timestamp",
            checked_at=checked_at,
            details={"updated_at": updated_at_raw},
        )

    age_seconds = max((now_utc.replace(tzinfo=None) - updated_at).total_seconds(), 0.0)
    risk = snapshot.get("risk", {}) or {}
    readiness = snapshot.get("readiness", {}) or {}
    system_health = readiness.get("metrics", {}).get("system_health", {}) or {}
    top_blocker = system_health.get("top_blocker", "unknown")

    unhealthy_reason = ""
    if age_seconds > max_snapshot_age_seconds:
        unhealthy_reason = "stale_runtime_snapshot"
    elif not bool(risk.get("allowed", True)) and str(risk.get("reason", "ok")) in {
        "emergency_mode",
        "cooldown",
        "gross_exposure_cap",
    }:
        unhealthy_reason = f"risk_blocked:{risk.get('reason', 'unknown')}"
    elif top_blocker not in {"healthy", "extended_no_signal_period", "unknown"}:
        unhealthy_reason = f"system_blocker:{top_blocker}"

    return RuntimeHealth(
        healthy=not unhealthy_reason,
        reason=unhealthy_reason or "ok",
        checked_at=checked_at,
        details={
            "snapshot_updated_at": updated_at_raw,
            "snapshot_age_seconds": round(age_seconds, 2),
            "top_blocker": top_blocker,
            "risk_reason": risk.get("reason", "ok"),
            "runtime_db_path": db_path,
        },
    )


def render_runtime_health(health: RuntimeHealth, as_json: bool = False) -> str:
    payload = asdict(health)
    if as_json:
        return json.dumps(payload, indent=2, sort_keys=True)
    return (
        f"healthy={payload['healthy']} "
        f"reason={payload['reason']} "
        f"age_seconds={payload['details'].get('snapshot_age_seconds')} "
        f"top_blocker={payload['details'].get('top_blocker')}"
    )
