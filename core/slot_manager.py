from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from .order_state import PENDING_ORDER_STATUSES, TERMINAL_ORDER_STATUSES, PositionSlot, normalize_order_status


class SlotManager:
    def __init__(self, slots: list[PositionSlot], max_positions: int) -> None:
        self.max_positions = max_positions
        slots_by_id = {slot.slot_id: slot for slot in slots}
        self.slots = [slots_by_id.get(slot_id, PositionSlot(slot_id=slot_id)) for slot_id in range(1, max_positions + 1)]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, max_positions: int) -> "SlotManager":
        if frame.empty:
            return cls([], max_positions=max_positions)
        slots: list[PositionSlot] = []
        for row in frame.to_dict(orient="records"):
            slots.append(
                PositionSlot(
                    slot_id=int(row["slot_id"]),
                    status=str(row.get("status") or "AVAILABLE"),
                    symbol=row.get("symbol"),
                    client_order_id=row.get("client_order_id"),
                    requested_quantity=int(row.get("requested_quantity") or 0),
                    filled_quantity=int(row.get("filled_quantity") or 0),
                    reserved_buying_power=float(row.get("reserved_buying_power") or 0.0),
                    avg_fill_price=float(row["avg_fill_price"]) if row.get("avg_fill_price") is not None else None,
                    side=row.get("side"),
                    updated_at=row.get("updated_at"),
                    payload_json=row.get("payload_json"),
                )
            )
        return cls(slots, max_positions=max_positions)

    def to_frame(self, session_date: pd.Timestamp) -> pd.DataFrame:
        rows = []
        for slot in self.slots:
            payload = asdict(slot)
            rows.append(
                {
                    "slot_id": slot.slot_id,
                    "session_date": str(pd.Timestamp(session_date).date()),
                    "status": slot.status,
                    "symbol": slot.symbol,
                    "client_order_id": slot.client_order_id,
                    "requested_quantity": slot.requested_quantity,
                    "filled_quantity": slot.filled_quantity,
                    "reserved_buying_power": slot.reserved_buying_power,
                    "avg_fill_price": slot.avg_fill_price,
                    "side": slot.side,
                    "updated_at": pd.Timestamp.utcnow().isoformat(),
                    "payload_json": json.dumps(payload, ensure_ascii=True, default=str),
                }
            )
        return pd.DataFrame(rows)

    def get_slot(self, slot_id: int) -> PositionSlot:
        return self.slots[slot_id - 1]

    def next_available_slot(self) -> PositionSlot | None:
        for slot in self.slots:
            if slot.status == "AVAILABLE":
                return slot
        return None

    @property
    def available_slots(self) -> int:
        return sum(1 for slot in self.slots if slot.status == "AVAILABLE")

    @property
    def pending_order_slots(self) -> int:
        return sum(1 for slot in self.slots if slot.status in {"BUY_PENDING", "SELL_PENDING"})

    @property
    def partially_filled_slots(self) -> int:
        return sum(1 for slot in self.slots if slot.status == "PARTIALLY_FILLED")

    @property
    def filled_slots(self) -> int:
        return sum(1 for slot in self.slots if slot.status == "FILLED")

    @property
    def reserved_buying_power(self) -> float:
        return float(sum(slot.reserved_buying_power for slot in self.slots if slot.status in {"BUY_PENDING", "PARTIALLY_FILLED"}))

    def available_buying_power_effective(self, current_buying_power: float, opening_buying_power: float) -> float:
        return max(0.0, min(float(current_buying_power), float(opening_buying_power)) - self.reserved_buying_power)

    def contains_symbol(self, symbol: str) -> bool:
        symbol_key = str(symbol).upper()
        return any(str(slot.symbol or "").upper() == symbol_key and slot.status != "AVAILABLE" for slot in self.slots)

    def reserve_for_buy(
        self,
        *,
        slot_id: int,
        symbol: str,
        client_order_id: str,
        quantity: int,
        reserved_buying_power: float,
        side: str,
    ) -> None:
        slot = self.get_slot(slot_id)
        slot.status = "BUY_PENDING"
        slot.symbol = str(symbol).upper()
        slot.client_order_id = client_order_id
        slot.requested_quantity = int(quantity)
        slot.filled_quantity = 0
        slot.reserved_buying_power = float(reserved_buying_power)
        slot.avg_fill_price = None
        slot.side = side.upper()
        slot.updated_at = pd.Timestamp.utcnow().isoformat()

    def mark_sell_pending(self, *, slot_id: int, client_order_id: str, quantity: int) -> None:
        slot = self.get_slot(slot_id)
        slot.status = "SELL_PENDING"
        slot.client_order_id = client_order_id
        slot.requested_quantity = int(quantity)
        slot.side = "SELL"
        slot.updated_at = pd.Timestamp.utcnow().isoformat()

    def release(self, slot_id: int) -> None:
        slot = self.get_slot(slot_id)
        slot.status = "AVAILABLE"
        slot.symbol = None
        slot.client_order_id = None
        slot.requested_quantity = 0
        slot.filled_quantity = 0
        slot.reserved_buying_power = 0.0
        slot.avg_fill_price = None
        slot.side = None
        slot.updated_at = pd.Timestamp.utcnow().isoformat()

    def sync_from_orders_and_positions(self, orders: pd.DataFrame, positions: pd.DataFrame) -> list[dict[str, object]]:
        transitions: list[dict[str, object]] = []
        position_by_symbol = {
            str(row["symbol"]).upper(): row
            for row in positions.to_dict(orient="records")
        }
        orders_by_slot: dict[int, list[dict[str, object]]] = {}
        for row in orders.to_dict(orient="records"):
            try:
                payload = json.loads(row.get("payload_json") or "{}")
            except Exception:
                payload = {}
            slot_id = payload.get("slot_id")
            if slot_id is None:
                continue
            row["slot_id"] = int(slot_id)
            row["normalized_status"] = normalize_order_status(row.get("status"), int(row.get("quantity") or 0), int(row.get("filled_quantity") or 0))
            orders_by_slot.setdefault(int(slot_id), []).append(row)

        for slot in self.slots:
            before = slot.status
            slot_orders = orders_by_slot.get(slot.slot_id, [])
            active_order = slot_orders[-1] if slot_orders else None
            position = position_by_symbol.get(str(slot.symbol or "").upper())

            if active_order is None and position is None:
                if slot.status != "AVAILABLE":
                    self.release(slot.slot_id)
            elif active_order is not None:
                status = str(active_order["normalized_status"])
                side = str(active_order.get("side") or "").upper()
                slot.symbol = str(active_order.get("symbol") or slot.symbol or "").upper() or slot.symbol
                slot.client_order_id = str(active_order.get("client_order_id") or slot.client_order_id or "")
                slot.requested_quantity = int(active_order.get("quantity") or slot.requested_quantity or 0)
                slot.filled_quantity = int(active_order.get("filled_quantity") or slot.filled_quantity or 0)
                slot.side = side or slot.side
                slot.updated_at = pd.Timestamp.utcnow().isoformat()
                if side == "SELL":
                    if status in TERMINAL_ORDER_STATUSES and position is None:
                        self.release(slot.slot_id)
                    else:
                        slot.status = "SELL_PENDING"
                else:
                    if status == "FILLED":
                        slot.status = "FILLED"
                        slot.reserved_buying_power = 0.0
                    elif status == "PARTIALLY_FILLED":
                        slot.status = "PARTIALLY_FILLED"
                    elif status in TERMINAL_ORDER_STATUSES and position is None:
                        self.release(slot.slot_id)
                    elif status in PENDING_ORDER_STATUSES:
                        slot.status = "BUY_PENDING"
            elif position is not None:
                slot.status = "FILLED"
                slot.symbol = str(position.get("symbol") or slot.symbol or "").upper()
                slot.filled_quantity = int(position.get("quantity") or slot.filled_quantity or 0)
                slot.avg_fill_price = float(position["avg_price"]) if position.get("avg_price") is not None else slot.avg_fill_price
                slot.reserved_buying_power = 0.0
                slot.updated_at = pd.Timestamp.utcnow().isoformat()

            if before != slot.status:
                transitions.append(
                    {
                        "slot_id": slot.slot_id,
                        "symbol": slot.symbol,
                        "from_status": before,
                        "to_status": slot.status,
                        "client_order_id": slot.client_order_id,
                    }
                )
        return transitions

