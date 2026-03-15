import datetime
import logging
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytz
import schedule

from stallion.config import load_settings
from stallion.storage import SQLiteParquetStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "hist_gbm_extended_5m_start.pkl"
STORE = None


def run_python_script(script_name):
    subprocess.run([sys.executable, script_name], check=True, cwd=SCRIPT_DIR)

def run_daily_ml_pipeline():
    if STORE is not None:
        STORE.write_heartbeat("master_scheduler", "running_pipeline", {"job": "nightly_pipeline"})
    logger.info("Starting nightly standard daytrade pipeline...")
    try:
        run_python_script('ml_pipeline_60d.py')
        logger.info("Nightly standard pipeline completed successfully. Shortlist and model artifacts updated.")
    except Exception as e:
        logger.error(f"Error running daily ML pipeline: {e}")

def run_daily_trading_bot():
    if STORE is not None:
        STORE.write_heartbeat("master_scheduler", "starting_live_trader", {"job": "live_trader"})
    logger.info("Initializing live trading bot...")
    # Check if today is a weekday
    today = datetime.datetime.now(pytz.timezone('America/New_York')).weekday()
    if today >= 5: # 5=Saturday, 6=Sunday
        logger.info("Today is a weekend. No trading.")
        return
        
    try:
        run_python_script('webull_live_trader.py')
    except Exception as e:
        logger.error(f"Daily trading bot error: {e}")


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
        "model artifact": settings.paths.model_dir / MODEL_FILENAME,
    }
    for label, path in required_files.items():
        if not path.exists():
            reasons.append(f"missing {label} at {path}")

    if settings.paths.watchlist_path.exists() and not _parquet_has_rows(settings.paths.watchlist_path):
        reasons.append(f"empty shortlist parquet at {settings.paths.watchlist_path}")

    required_tables = ("universe", "daily_features", "nightly_shortlist", "model_registry")
    for table_name in required_tables:
        if not _sqlite_table_has_rows(settings.paths.sqlite_path, table_name):
            reasons.append(f"sqlite table {table_name} is missing or empty")

    return len(reasons) == 0, reasons


def run_startup_pipeline_if_needed():
    ready, reasons = bootstrap_artifacts_ready()
    if not ready:
        logger.info("Bootstrap artifacts missing or incomplete. Running one-shot nightly pipeline bootstrap now...")
        for reason in reasons:
            logger.info("Bootstrap check: %s", reason)
        run_daily_ml_pipeline()
        return

    logger.info("SQLite + Parquet bootstrap artifacts detected. Skipping startup pipeline bootstrap.")

def main():
    logger.info("Starting Master Scheduler. Timezone is set to America/New_York.")
    global STORE
    try:
        STORE = SQLiteParquetStore(load_settings(SCRIPT_DIR))
        STORE.write_heartbeat("master_scheduler", "starting", {})
    except Exception as exc:
        logger.error("Failed to initialize scheduler store: %s", exc)
        STORE = None
    run_startup_pipeline_if_needed()
    
    # Nightly batch: refresh universe, bars, daily features, training panel, model bundle, shortlist.
    schedule.every().monday.at("17:00").do(run_daily_ml_pipeline)
    schedule.every().tuesday.at("17:00").do(run_daily_ml_pipeline)
    schedule.every().wednesday.at("17:00").do(run_daily_ml_pipeline)
    schedule.every().thursday.at("17:00").do(run_daily_ml_pipeline)
    schedule.every().friday.at("17:00").do(run_daily_ml_pipeline)
    
    # Live bot: loads the saved model + shortlist and starts polling before the open.
    schedule.every().monday.at("09:25").do(run_daily_trading_bot)
    schedule.every().tuesday.at("09:25").do(run_daily_trading_bot)
    schedule.every().wednesday.at("09:25").do(run_daily_trading_bot)
    schedule.every().thursday.at("09:25").do(run_daily_trading_bot)
    schedule.every().friday.at("09:25").do(run_daily_trading_bot)
    
    logger.info("Scheduler loops configured. Waiting for next assigned task...")
    
    # Main infinite loop
    while True:
        if STORE is not None:
            STORE.write_heartbeat("master_scheduler", "idle", {})
        schedule.run_pending()
        time.sleep(60) # check every minute

if __name__ == "__main__":
    main()
