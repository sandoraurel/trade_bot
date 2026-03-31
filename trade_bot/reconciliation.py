from __future__ import annotations

from typing import Any, List

from .models import ReconciliationStatus, utc_now


class BotReconciler:
    def __init__(self, bot: Any):
        self.bot = bot

    def reconcile(self) -> ReconciliationStatus:
        reasons: List[str] = []
        positions_match = True
        balance_match = True

        if self.bot.state.balance < 0:
            balance_match = False
            reasons.append("negative_balance")

        local_count = 0
        for value in self.bot.state.open_positions.values():
            local_count += len(value) if isinstance(value, list) else 1

        if local_count > self.bot.config.max_open_positions:
            positions_match = False
            reasons.append("open_position_limit_exceeded")

        ok = positions_match and balance_match
        return ReconciliationStatus(
            ok=ok,
            checked_at=utc_now(),
            positions_match=positions_match,
            balance_match=balance_match,
            drift_reasons=reasons,
            metadata={
                "local_open_position_count": local_count,
                "paper_mode": self.bot.state.paper_mode,
            },
        )
