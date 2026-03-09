import schedule
import time
import datetime
import pytz
import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_daily_ml_pipeline():
    logger.info("Starting daily ML optimization pipeline (60-day PCA framework)...")
    try:
        # Run the ML pipeline script to generate the new Top 10 list based on today's close
        subprocess.run(['python', 'ml_pipeline_60d.py'], check=True)
        logger.info("Daily ML pipeline completed successfully. Top 10 watchlist updated for tomorrow.")
    except Exception as e:
        logger.error(f"Error running daily ML pipeline: {e}")

def run_daily_trading_bot():
    logger.info("Initializing daily trading bot...")
    # Check if today is a weekday
    today = datetime.datetime.now(pytz.timezone('America/New_York')).weekday()
    if today >= 5: # 5=Saturday, 6=Sunday
        logger.info("Today is a weekend. No trading.")
        return
        
    try:
        # Run the daily trading execution script
        # This script should connect to Webull, wait for 9:35 AM trigger, execute the SINGLE best trade, and stop out at 15:55.
        subprocess.run(['python', 'webull_live_trader.py'], check=True)
    except Exception as e:
        logger.error(f"Daily trading bot error: {e}")

def main():
    logger.info("Starting Master Scheduler. Timezone is set to America/New_York.")
    
    # Schedule Daily Optimization: Every Mon-Fri at 17:00 (5:00 PM) EST
    schedule.every().monday.at("17:00").do(run_daily_ml_pipeline)
    schedule.every().tuesday.at("17:00").do(run_daily_ml_pipeline)
    schedule.every().wednesday.at("17:00").do(run_daily_ml_pipeline)
    schedule.every().thursday.at("17:00").do(run_daily_ml_pipeline)
    schedule.every().friday.at("17:00").do(run_daily_ml_pipeline)
    
    # Schedule Daily Bot: Every Mon-Fri at 09:25 AM EST (to give it time to boot up before 9:30 open)
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
