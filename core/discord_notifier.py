from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

from .config import Settings
from .storage import SQLiteParquetStore


LOGGER = logging.getLogger(__name__)
DISCORD_API_BASE = "https://discord.com/api/v10"


@dataclass(frozen=True)
class DiscordProbe:
    token_valid: bool
    bot_id: str | None
    username: str | None
    can_send_messages: bool
    reason: str


class DiscordNotifier:
    def __init__(self, settings: Settings, store: SQLiteParquetStore, session: requests.Session | None = None) -> None:
        self.settings = settings
        self.store = store
        self.session = session or requests.Session()
        self.queue: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, name="discord-notifier", daemon=True)
        self._worker.start()
        self.mode_tag = "[DEMO]" if settings.demo_mode else "[LIVE]"
        self.bot_token = str(settings.credentials.discord_bot_token or "").strip()
        self.channel_id = str(settings.credentials.discord_channel_id or "").strip()

    def probe(self) -> DiscordProbe:
        if not self.bot_token:
            return DiscordProbe(False, None, None, False, "missing_token")
        try:
            response = self.session.get(
                f"{DISCORD_API_BASE}/users/@me",
                headers={"Authorization": f"Bot {self.bot_token}"},
                timeout=15,
            )
            if response.status_code != 200:
                return DiscordProbe(False, None, None, False, f"http_{response.status_code}")
            payload = response.json()
            return DiscordProbe(
                token_valid=True,
                bot_id=str(payload.get("id")),
                username=str(payload.get("username")),
                can_send_messages=bool(self.channel_id),
                reason="ok" if self.channel_id else "missing_channel_id",
            )
        except Exception as exc:
            LOGGER.exception("Discord probe failed")
            return DiscordProbe(False, None, None, False, f"exception:{exc}")

    def notify(self, title: str, lines: list[str], *, level: str = "INFO") -> None:
        content = "\n".join([self.mode_tag, title, *lines])
        payload = {
            "level": level,
            "title": title,
            "content": content,
        }
        self.queue.put(payload)

    def flush(self, timeout: float = 10.0) -> None:
        self.queue.join()

    def close(self) -> None:
        self.queue.put(None)
        self._worker.join(timeout=5)

    def _worker_loop(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            error_text = None
            delivered = False
            try:
                if self.bot_token and self.channel_id:
                    for _ in range(3):
                        try:
                            response = self.session.post(
                                f"{DISCORD_API_BASE}/channels/{self.channel_id}/messages",
                                headers={
                                    "Authorization": f"Bot {self.bot_token}",
                                    "Content-Type": "application/json",
                                },
                                data=json.dumps({"content": item["content"]}),
                                timeout=30,
                            )
                            delivered = response.status_code in {200, 201}
                            if not delivered:
                                error_text = f"http_{response.status_code}:{response.text[:200]}"
                            break
                        except requests.exceptions.ReadTimeout:
                            if _ == 2:
                                raise
                            time.sleep(1.0)
                else:
                    error_text = "discord_disabled_or_missing_channel"
            except Exception as exc:
                LOGGER.exception("Discord send failed")
                error_text = str(exc)
            try:
                self.store.append_discord_notification(
                    level=str(item["level"]),
                    title=str(item["title"]),
                    mode=self.settings.trade_mode,
                    delivered=delivered,
                    channel_id=self.channel_id or None,
                    payload={"content": item["content"]},
                    error_text=error_text,
                )
            finally:
                self.queue.task_done()

