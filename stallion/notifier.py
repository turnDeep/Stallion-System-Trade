from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests

from .storage import SQLiteParquetStore


LOGGER = logging.getLogger(__name__)


def emit_alert(
    store: SQLiteParquetStore,
    *,
    level: str,
    component: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    body = payload or {}
    LOGGER.log(
        logging.ERROR if level.upper() in {"ERROR", "CRITICAL"} else logging.WARNING,
        "[%s] %s :: %s",
        component,
        level.upper(),
        message,
    )
    store.append_alert(level=level.upper(), component=component, message=message, payload=body)

    webhook_url = os.getenv("STALLION_ALERT_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return
    try:
        requests.post(
            webhook_url,
            json={
                "text": f"[{component}] {level.upper()} {message}",
                "payload": body,
            },
            timeout=10,
        )
    except Exception:
        LOGGER.exception("Failed to send alert webhook")
