from __future__ import annotations

from dataclasses import dataclass


TERMINAL_ORDER_STATUSES = {"FILLED", "CANCELLED", "CANCELED", "REJECTED", "FAILED", "EXPIRED"}
PENDING_ORDER_STATUSES = {"NEW", "SUBMITTED", "PENDING", "PARTIALLY_FILLED", "PARTIAL_FILLED", "CREATED", "CANCEL_REQUESTED"}


def normalize_order_status(status: object, quantity: int, filled_quantity: int) -> str:
    status_str = str(status or "UNKNOWN").upper().replace(" ", "_")
    if filled_quantity >= quantity > 0:
        return "FILLED"
    if status_str in {"CANCEL_REQUESTED", "CANCELLATION_REQUESTED", "PENDING_CANCEL"}:
        return "CANCEL_REQUESTED"
    if status_str in TERMINAL_ORDER_STATUSES:
        return status_str
    if "CANCEL" in status_str:
        return "CANCELLED" if filled_quantity == 0 else "PARTIALLY_FILLED"
    if "REJECT" in status_str:
        return "REJECTED"
    if "FAIL" in status_str:
        return "FAILED"
    if "PART" in status_str and filled_quantity > 0:
        return "PARTIALLY_FILLED"
    return status_str or "UNKNOWN"


@dataclass
class OrderState:
    client_order_id: str
    symbol: str
    side: str
    quantity: int
    filled_quantity: int
    status: str
    slot_id: int | None = None
    requested_price: float | None = None
    avg_fill_price: float | None = None
    order_type: str | None = None
    reserved_buying_power: float = 0.0


@dataclass
class PositionSlot:
    slot_id: int
    status: str = "AVAILABLE"
    symbol: str | None = None
    client_order_id: str | None = None
    requested_quantity: int = 0
    filled_quantity: int = 0
    reserved_buying_power: float = 0.0
    avg_fill_price: float | None = None
    side: str | None = None
    updated_at: str | None = None
    payload_json: str | None = None
