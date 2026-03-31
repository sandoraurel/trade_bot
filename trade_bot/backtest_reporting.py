from __future__ import annotations

import hashlib
import json
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
