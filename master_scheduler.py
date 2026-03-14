import schedule
import time
import datetime
import pytz
import os
import subprocess
import logging
import json
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_PATH = os.path.join(SCRIPT_DIR, "top_10_watchlist.json")


def run_python_script(script_name):
    subprocess.run([sys.executable, script_name], check=True, cwd=SCRIPT_DIR)

def run_daily_ml_pipeline():
    logger.info("Starting nightly standard daytrade pipeline...")
    try:
        run_python_script('ml_pipeline_60d.py')
        logger.info("Nightly standard pipeline completed successfully. Shortlist and model artifacts updated.")
    except Exception as e:
        logger.error(f"Error running daily ML pipeline: {e}")

def run_daily_trading_bot():
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


def watchlist_missing_or_empty():
    if not os.path.exists(WATCHLIST_PATH):
        return True

    try:
        with open(WATCHLIST_PATH, "r", encoding="utf-8") as handle:
            watchlist = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return True

    return not isinstance(watchlist, list) or len(watchlist) == 0


def run_startup_pipeline_if_needed():
    if watchlist_missing_or_empty():
        logger.info("No startup shortlist detected. Running one-shot nightly pipeline bootstrap now...")
        run_daily_ml_pipeline()
        return

    logger.info("Existing watchlist detected. Skipping startup pipeline bootstrap.")

def main():
    logger.info("Starting Master Scheduler. Timezone is set to America/New_York.")
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
        schedule.run_pending()
        time.sleep(60) # check every minute

if __name__ == "__main__":
    main()
