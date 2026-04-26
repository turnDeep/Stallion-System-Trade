from __future__ import annotations

import argparse

from stallion.config import load_settings
from stallion.live_trader import _load_tax_reserve_state, _save_tax_reserve_state
from stallion.storage import SQLiteParquetStore


def _print_state(state: dict[str, object]) -> None:
    print(f"reserved_tax_usd: {float(state.get('reserved_tax_usd', 0.0)):.2f}")
    print(f"realized_profit_usd: {float(state.get('realized_profit_usd', 0.0)):.2f}")
    print(f"events: {int(state.get('events', 0))}")
    if state.get("updated_at"):
        print(f"updated_at: {state['updated_at']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the USD tax reserve used by the live trader.")
    parser.add_argument(
        "action",
        choices=["show", "set", "add", "subtract", "clear"],
        help="show current reserve, set/add/subtract an amount, or clear it after manual FX conversion.",
    )
    parser.add_argument("--amount", type=float, default=0.0, help="USD amount for set/add/subtract.")
    parser.add_argument("--root", default=".", help="Repository root containing .env and data directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.root)
    store = SQLiteParquetStore(settings)
    state = _load_tax_reserve_state(store)

    current = float(state.get("reserved_tax_usd", 0.0))
    if args.action == "show":
        _print_state(state)
        return

    if args.action in {"set", "add", "subtract"} and args.amount < 0:
        raise ValueError("--amount must be non-negative")

    if args.action == "set":
        state["reserved_tax_usd"] = args.amount
    elif args.action == "add":
        state["reserved_tax_usd"] = current + args.amount
    elif args.action == "subtract":
        state["reserved_tax_usd"] = max(0.0, current - args.amount)
    elif args.action == "clear":
        state["reserved_tax_usd"] = 0.0

    _save_tax_reserve_state(store, state)
    _print_state(_load_tax_reserve_state(store))


if __name__ == "__main__":
    main()
