from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SizingDecision:
    quantity: int
    effective_budget: float
    reason: str | None = None


def compute_order_quantity(
    *,
    slot_budget: float,
    effective_buying_power: float,
    expected_price: float,
    fractional_shares_enabled: bool,
) -> SizingDecision:
    if not math.isfinite(expected_price) or expected_price <= 0:
        return SizingDecision(quantity=0, effective_budget=0.0, reason="invalid_expected_price")
    effective_budget = max(0.0, min(float(slot_budget), float(effective_buying_power)))
    if effective_budget < expected_price and not fractional_shares_enabled:
        return SizingDecision(quantity=0, effective_budget=effective_budget, reason="per_slot_budget_below_share_price")
    if fractional_shares_enabled:
        fractional_qty = int((effective_budget / expected_price) * 10_000)
        return SizingDecision(quantity=max(fractional_qty, 0), effective_budget=effective_budget)
    quantity = max(int(math.floor(effective_budget / expected_price)), 0)
    if quantity < 1:
        return SizingDecision(quantity=0, effective_budget=effective_budget, reason="insufficient_buying_power_for_one_share")
    return SizingDecision(quantity=quantity, effective_budget=effective_budget)

