from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import pandas as pd

from .config import load_settings
from .storage import SQLiteParquetStore


LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[1]
STATUS_STALE_OVERRIDES = {
    "running_pipeline": 7200,
    "starting_live_trader": 900,
}


def evaluate_health(max_age_seconds: int | None = None) -> tuple[bool, dict]:
    settings = load_settings(ROOT_DIR)
    store = SQLiteParquetStore(settings)
    stale_limit = max_age_seconds or settings.runtime.watchdog_stale_seconds
    heartbeats = store.load_heartbeats()
    if heartbeats.empty:
        return False, {"reason": "no_heartbeats"}

    now = pd.Timestamp.now(tz="UTC")
    latest = {}
    for row in heartbeats.itertuples(index=False):
        component = str(row.component)
        if component in latest:
            continue
        hb_at = pd.to_datetime(row.heartbeat_at, utc=True, errors="coerce")
        age_seconds = None if pd.isna(hb_at) else float((now - hb_at.tz_convert("UTC")).total_seconds())
        latest[component] = {
            "heartbeat_at": str(row.heartbeat_at),
            "status": str(row.status),
            "age_seconds": age_seconds,
        }

    scheduler = latest.get("master_scheduler")
    scheduler_limit = stale_limit
    if scheduler:
        scheduler_limit = STATUS_STALE_OVERRIDES.get(str(scheduler["status"]), stale_limit)

    if not scheduler or scheduler["age_seconds"] is None or scheduler["age_seconds"] > scheduler_limit:
        return False, {"reason": "scheduler_stale", "heartbeats": latest}
    return True, {"reason": "ok", "heartbeats": latest}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ok, payload = evaluate_health()
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
