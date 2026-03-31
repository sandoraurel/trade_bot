from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .models import utc_now


@dataclass
class BotEvent:
    event_type: str
    trace_id: str
    created_at: str = field(default_factory=lambda: utc_now().isoformat())
    payload: Dict[str, Any] = field(default_factory=dict)


EVENT_MARKET_DATA = "market_data_update"
EVENT_SIGNAL = "signal_emitted"
EVENT_ORDER_SUBMITTED = "order_submitted"
EVENT_ORDER_ACK = "order_acknowledged"
EVENT_FILL = "fill_received"
EVENT_POSITION_UPDATED = "position_updated"
EVENT_RISK_HALT = "risk_halt"
EVENT_STATE_PERSISTED = "state_persisted"
EVENT_RECONCILIATION = "reconciliation_checked"
