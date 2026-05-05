from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import load_settings
from core.live_trader import _load_tax_reserve_state, _save_tax_reserve_state
from core.storage import SQLiteParquetStore


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
        choices=["show", "set", "add", "subtract", "clear", "archive", "history"],
        help=(
            "show: display current reserve. "
            "set/add/subtract: adjust amount. "
            "clear: zero out (no archive). "
            "archive: save current year to SQLite then zero (run after paying annual tax). "
            "history: show all archived yearly records."
        ),
    )
    parser.add_argument("--amount", type=float, default=0.0, help="USD amount for set/add/subtract.")
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Tax year to archive (e.g. 2024). Defaults to current year. Used with 'archive'.",
    )
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

    if args.action == "history":
        # Read all year-specific archive keys directly from SQLite
        found = False
        for year in range(2020, datetime.now().year + 2):
            raw = store.get_system_state(f"tax_reserve_usd:{year}")
            if raw:
                try:
                    archived = json.loads(raw)
                except Exception:
                    archived = {"reserved_tax_usd": raw}
                reserved = float(archived.get("reserved_tax_usd", 0.0))
                realized = float(archived.get("realized_profit_usd", 0.0))
                archived_at = archived.get("archived_at", "unknown")
                print(f"[{year}] reserved_tax_usd: {reserved:.2f}  realized_profit_usd: {realized:.2f}  archived_at: {archived_at}")
                found = True
        if not found:
            print("No yearly archives found.")
        print("\n[current]")
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
    elif args.action == "archive":
        tax_year = getattr(args, "year", None) or datetime.now().year
        archive_key = f"tax_reserve_usd:{tax_year}"

        # Show current state before prompting
        print(f"\n現在の積立状況:")
        _print_state(state)
        print(f"\n{tax_year}年分として '{archive_key}' にアーカイブし、reserved_tax_usd をゼロにリセットします。")
        print("実際に税金を支払った後にのみ実行してください。")
        confirm = input("続行しますか？ (yes/no): ").strip().lower()
        if confirm != "yes":
            print("キャンセルしました。")
            return

        # Archive to a separate SQLite key (not part of the live state)
        archive_payload = json.dumps({
            "reserved_tax_usd": round(current, 2),
            "realized_profit_usd": round(float(state.get("realized_profit_usd", 0.0)), 2),
            "events": int(state.get("events", 0)),
            "archived_at": datetime.now().isoformat(),
        }, ensure_ascii=True)
        store.put_system_state(archive_key, archive_payload)
        print(f"\n[OK] {tax_year}年分を '{archive_key}' にアーカイブしました。")

        # Reset live state
        state["reserved_tax_usd"] = 0.0
        state["realized_profit_usd"] = 0.0
        state["events"] = 0
        _save_tax_reserve_state(store, state)
        print("[OK] reserved_tax_usd をゼロにリセットしました。\n")
        print("リセット後の状態:")
        _print_state(_load_tax_reserve_state(store))
        return

    _save_tax_reserve_state(store, state)
    _print_state(_load_tax_reserve_state(store))


if __name__ == "__main__":
    main()
