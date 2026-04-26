import datetime
import logging
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import pytz

from core.config import load_settings
from core.discord_notifier import DiscordNotifier
from core.storage import SQLiteParquetStore


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
SCRIPT_DIR = str(REPO_ROOT)
MODEL_FILENAME = "hist_gbm_extended_5m_start.pkl"
STORE = None
NOTIFIER = None
DISCORD_DETAIL_CHUNK_CHARS = 1200
STDIO_CAPTURE_CHAR_LIMIT = 6000
SECRET_ENV_KEYS = (
    "WEBULL_APP_KEY",
    "WEBULL_APP_SECRET",
    "WEBULL_ACCOUNT_ID",
    "FMP_API_KEY",
    "DISCORD_BOT_TOKEN",
    "DISCORD_CHANNEL_ID",
)


@dataclass(frozen=True)
class ScriptExecutionError(RuntimeError):
    script_name: str
    return_code: int
    stdout_tail: str
    stderr_tail: str

    def __str__(self) -> str:
        stream = "stderr" if self.stderr_tail else "stdout"
        detail = self.stderr_tail or self.stdout_tail or "no subprocess output captured"
        first_line = detail.splitlines()[0][:300] if detail else "no subprocess output captured"
        return f"{self.script_name} failed with exit status {self.return_code} ({stream}: {first_line})"


def _tail_text(text: str | None, max_chars: int = STDIO_CAPTURE_CHAR_LIMIT) -> str:
    normalized = str(text or "").replace("\r\n", "\n").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[-max_chars:]


def _redact_sensitive_text(text: str | None) -> str:
    redacted = str(text or "")
    for key in SECRET_ENV_KEYS:
        value = str(os.getenv(key) or "").strip()
        if value:
            redacted = redacted.replace(value, f"[REDACTED:{key}]")
    return redacted


def _chunk_text(text: str | None, max_chars: int = DISCORD_DETAIL_CHUNK_CHARS) -> list[str]:
    normalized = str(text or "").replace("\r\n", "\n").strip()
    if not normalized:
        return []
    chunks: list[str] = []
    remaining = normalized
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        candidate = remaining[:max_chars]
        split_at = candidate.rfind("\n")
        if split_at < max_chars // 2:
            split_at = candidate.rfind(" ")
        if split_at < max_chars // 2:
            split_at = max_chars
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip("\n ")
    return chunks


def _append_alert(level: str, component: str, message: str, payload: dict | None = None) -> None:
    if STORE is None:
        return
    try:
        STORE.append_alert(level=level, component=component, message=message, payload=payload)
    except Exception:
        logger.exception("Failed to append alert to store")


def _notify_detailed_failure(title: str, exc: Exception, *, component: str, script_name: str | None = None) -> None:
    detail_payload: dict[str, object] = {"error_type": type(exc).__name__}
    summary_lines = [f"- error_type: {type(exc).__name__}"]
    if script_name:
        summary_lines.append(f"- script: {script_name}")
        detail_payload["script_name"] = script_name
    if isinstance(exc, ScriptExecutionError):
        summary_lines.append(f"- exit_code: {exc.return_code}")
        detail_payload["exit_code"] = exc.return_code
        summary_lines.append(f"- error: {str(exc)}")
        detail_payload["stdout_tail"] = exc.stdout_tail
        detail_payload["stderr_tail"] = exc.stderr_tail
    else:
        summary_lines.append(f"- error: {str(exc)}")
        trace_tail = _redact_sensitive_text(_tail_text(traceback.format_exc()))
        detail_payload["traceback_tail"] = trace_tail
    logger.error("%s: %s", title, exc)
    _append_alert("ERROR", component, title, detail_payload)
    if NOTIFIER is None:
        return
    NOTIFIER.notify(title, summary_lines, level="ERROR")
    streams: list[tuple[str, str]] = []
    if isinstance(exc, ScriptExecutionError):
        if exc.stderr_tail:
            streams.append(("stderr", exc.stderr_tail))
        if exc.stdout_tail:
            streams.append(("stdout", exc.stdout_tail))
    else:
        trace_tail = detail_payload.get("traceback_tail")
        if isinstance(trace_tail, str) and trace_tail:
            streams.append(("traceback", trace_tail))
    for stream_name, text in streams:
        chunks = _chunk_text(text)
        total = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            NOTIFIER.notify(
                f"{title} DETAIL",
                [
                    f"- source: {stream_name}",
                    f"- chunk: {index}/{total}",
                    "```text",
                    *chunk.splitlines(),
                    "```",
                ],
                level="ERROR",
            )


def run_python_script(script_name: str) -> None:
    logger.info("Running script: %s", script_name)
    script_path = REPO_ROOT / script_name
    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        raise ScriptExecutionError(
            script_name=script_name,
            return_code=int(exc.returncode),
            stdout_tail=_redact_sensitive_text(_tail_text(exc.stdout)),
            stderr_tail=_redact_sensitive_text(_tail_text(exc.stderr)),
        ) from exc


def run_daily_ml_pipeline() -> None:
    if STORE is not None:
        STORE.write_heartbeat("master_scheduler", "running_pipeline", {"job": "nightly_pipeline"})
    logger.info("Starting nightly breakout pipeline...")
    if NOTIFIER is not None:
        NOTIFIER.notify("NIGHTLY PIPELINE START", ["- job: nightly_pipeline"])
    try:
        run_python_script("scripts/nightly_pipeline.py")
        logger.info("Nightly breakout pipeline completed successfully. Shortlist and reports updated.")
        if NOTIFIER is not None:
            NOTIFIER.notify("NIGHTLY PIPELINE COMPLETE", ["- status: success"])
    except Exception as exc:
        _notify_detailed_failure("NIGHTLY PIPELINE FAILED", exc, component="master_scheduler", script_name="scripts/nightly_pipeline.py")


def run_daily_trading_bot() -> None:
    if STORE is not None:
        STORE.write_heartbeat("master_scheduler", "starting_live_trader", {"job": "live_trader"})
    logger.info("Initializing live trading bot...")
    if NOTIFIER is not None:
        NOTIFIER.notify("LIVE TRADER START", ["- job: live_trader"])
    today = datetime.datetime.now(pytz.timezone("America/New_York")).weekday()
    if today >= 5:
        logger.info("Today is a weekend. No trading.")
        return

    try:
        run_python_script("scripts/live_trader.py")
    except Exception as exc:
        _notify_detailed_failure("LIVE TRADER FAILED", exc, component="master_scheduler", script_name="scripts/live_trader.py")


def _sqlite_table_has_rows(sqlite_path: Path, table_name: str) -> bool:
    if not sqlite_path.exists():
        return False
    try:
        with sqlite3.connect(sqlite_path) as connection:
            cursor = connection.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
            return cursor.fetchone() is not None
    except sqlite3.Error:
        return False


def _parquet_has_rows(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return False
    return not frame.empty


def bootstrap_artifacts_ready() -> tuple[bool, list[str]]:
    reasons: list[str] = []
    try:
        settings = load_settings(SCRIPT_DIR)
    except Exception as exc:
        return False, [f"settings unavailable: {exc}"]

    required_files = {
        "sqlite database": settings.paths.sqlite_path,
        "shortlist parquet": settings.paths.watchlist_path,
    }
    for label, path in required_files.items():
        if not path.exists():
            reasons.append(f"missing {label} at {path}")

    if settings.paths.watchlist_path.exists() and not _parquet_has_rows(settings.paths.watchlist_path):
        reasons.append(f"empty shortlist parquet at {settings.paths.watchlist_path}")

    required_tables = ("universe", "nightly_shortlist")
    for table_name in required_tables:
        if not _sqlite_table_has_rows(settings.paths.sqlite_path, table_name):
            reasons.append(f"sqlite table {table_name} is missing or empty")

    return len(reasons) == 0, reasons


def _symbol_preview(symbols: list[str], limit: int = 12) -> str:
    if not symbols:
        return "-"
    preview = ", ".join(symbols[:limit])
    if len(symbols) > limit:
        preview += ", ..."
    return preview


def _check_bars_freshness(store: SQLiteParquetStore | None) -> None:
    if store is None:
        return
    stale_daily_days = 4
    stale_5m_days = 4
    try:
        daily_age = store.get_bars_freshness_days("1d")
        if daily_age == float("inf"):
            logger.warning("FRESHNESS CHECK: bars_1d table is EMPTY - no daily data stored yet.")
        elif daily_age > stale_daily_days:
            logger.warning(
                "FRESHNESS CHECK: bars_1d last updated %.1f days ago (threshold=%d). "
                "Nightly pipeline may not have run recently.",
                daily_age,
                stale_daily_days,
            )
        else:
            logger.info("FRESHNESS CHECK: bars_1d is fresh (%.1f days old).", daily_age)

        intraday_age = store.get_bars_freshness_days("5m")
        if intraday_age == float("inf"):
            logger.warning("FRESHNESS CHECK: bars_5m table is EMPTY - no intraday data stored yet.")
        elif intraday_age > stale_5m_days:
            logger.warning(
                "FRESHNESS CHECK: bars_5m last updated %.1f days ago (threshold=%d). "
                "5m intraday data is STALE - live signals will use outdated conditions.",
                intraday_age,
                stale_5m_days,
            )
        else:
            logger.info("FRESHNESS CHECK: bars_5m is fresh (%.1f days old).", intraday_age)

        universe = store.load_universe()
        symbols = universe["symbol"].dropna().astype(str).str.upper().tolist() if not universe.empty else []
        if not symbols:
            logger.warning("FRESHNESS CHECK: universe is empty, skipping symbol-level bar audit.")
            return

        daily_gap_audit = store.audit_symbol_gaps("1d", symbols, tolerance_days=1.0)
        daily_missing = daily_gap_audit.loc[daily_gap_audit["status"].eq("missing"), "symbol"].tolist()
        daily_stale = daily_gap_audit.loc[daily_gap_audit["status"].eq("stale"), "symbol"].tolist()
        if daily_missing or daily_stale:
            logger.warning(
                "FRESHNESS CHECK: bars_1d symbol audit found %s missing and %s stale symbols. "
                "missing=%s stale=%s",
                len(daily_missing),
                len(daily_stale),
                _symbol_preview(daily_missing),
                _symbol_preview(daily_stale),
            )
        else:
            logger.info("FRESHNESS CHECK: bars_1d symbol audit passed for %s universe symbols.", len(daily_gap_audit))

        intraday_gap_audit = store.audit_symbol_gaps("5m", symbols, tolerance_days=1.0)
        intraday_missing = intraday_gap_audit.loc[intraday_gap_audit["status"].eq("missing"), "symbol"].tolist()
        intraday_stale = intraday_gap_audit.loc[intraday_gap_audit["status"].eq("stale"), "symbol"].tolist()
        if intraday_missing or intraday_stale:
            logger.warning(
                "FRESHNESS CHECK: bars_5m symbol audit found %s missing and %s stale symbols. "
                "missing=%s stale=%s",
                len(intraday_missing),
                len(intraday_stale),
                _symbol_preview(intraday_missing),
                _symbol_preview(intraday_stale),
            )
        else:
            logger.info("FRESHNESS CHECK: bars_5m symbol audit passed for %s universe symbols.", len(intraday_gap_audit))
    except Exception as exc:
        logger.warning("FRESHNESS CHECK: could not read bar timestamps: %s", exc)


def run_startup_pipeline_if_needed() -> None:
    ready, reasons = bootstrap_artifacts_ready()
    if not ready:
        logger.info("Bootstrap artifacts missing or incomplete. Running one-shot nightly pipeline bootstrap now...")
        for reason in reasons:
            logger.info("Bootstrap check: %s", reason)
        run_daily_ml_pipeline()
        return
    logger.info("SQLite + Parquet bootstrap artifacts detected. Skipping startup pipeline bootstrap.")


def main() -> None:
    logger.info("Starting Master Scheduler. Timezone is set to America/New_York.")
    global STORE, NOTIFIER
    try:
        settings = load_settings(SCRIPT_DIR)
        STORE = SQLiteParquetStore(settings)
        NOTIFIER = DiscordNotifier(settings, STORE)
        STORE.write_heartbeat("master_scheduler", "starting", {})
        logger.info("Trading mode resolved to %s", settings.trade_mode)
        broker_mode_lines = [
            f"- mode: {settings.trade_mode}",
            "- bot_running: true",
            f"- discord_enabled: {str(settings.discord_enabled).lower()}",
        ]
        discord_probe = NOTIFIER.probe()
        broker_mode_lines.extend(
            [
                f"- discord_token_valid: {str(discord_probe.token_valid).lower()}",
                f"- discord_can_send: {str(discord_probe.can_send_messages).lower()}",
            ]
        )
        NOTIFIER.notify("SCHEDULER STARTUP", broker_mode_lines)
    except Exception as exc:
        logger.error("Failed to initialize scheduler store: %s", exc)
        STORE = None
        NOTIFIER = None

    run_startup_pipeline_if_needed()
    _check_bars_freshness(STORE)

    # Implementation of native timezone-aware loop to replace 'schedule' library
    # schedule.every().monday.at("17:00").do(run_daily_ml_pipeline) ... etc
    tz_ny = pytz.timezone("America/New_York")
    last_trading_date = None
    last_pipeline_date = None

    logger.info("Scheduler loop started. Timezone is set to America/New_York.")
    logger.info("Monitoring for: 09:25 AM (Trading Bot) and 17:00 (Nightly Pipeline) NY time.")

    while True:
        try:
            if STORE is not None:
                STORE.write_heartbeat("master_scheduler", "idle", {})

            now_ny = datetime.datetime.now(tz_ny)
            current_date = now_ny.strftime("%Y-%m-%d")

            # 1. Trading Bot (09:25 AM NY, Monday-Friday)
            if now_ny.weekday() < 5:
                # Trigger at or after 09:25
                if now_ny.hour > 9 or (now_ny.hour == 9 and now_ny.minute >= 25):
                    if last_trading_date != current_date:
                        logger.info("Triggering Daily Trading Bot (NY Time: %s)", now_ny.strftime("%H:%M:%S"))
                        run_daily_trading_bot()
                        last_trading_date = current_date

            # 2. Nightly ML Pipeline (17:00 / 05:00 PM NY, Monday-Friday)
            if now_ny.weekday() < 5:
                # Trigger at or after 17:00
                if now_ny.hour >= 17:
                    if last_pipeline_date != current_date:
                        logger.info("Triggering Nightly ML Pipeline (NY Time: %s)", now_ny.strftime("%H:%M:%S"))
                        run_daily_ml_pipeline()
                        last_pipeline_date = current_date

            time.sleep(30)
        except Exception as e:
            logger.error("Error in scheduler loop: %s", e)
            time.sleep(60)


if __name__ == "__main__":
    main()
