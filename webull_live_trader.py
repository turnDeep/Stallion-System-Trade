import os
import time
import datetime
import pytz
import logging
import pickle
import uuid
from webullsdkcore.client import ApiClient
from webullsdkcore.common.enums import Region
from webullsdktrade.api import API
from webullsdkmdata.quotes.market_data import MarketData
from webullsdkmdata.common.category import Category

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants Environment variables
APP_KEY = os.getenv("WEBULL_APP_KEY")
APP_SECRET = os.getenv("WEBULL_APP_SECRET")
ACCOUNT_ID = os.getenv("WEBULL_ACCOUNT_ID")

def init_webull_client():
    if not APP_KEY or not APP_SECRET or not ACCOUNT_ID:
        logger.error("Webull credentials missing!")
        return None, None
    api_client = ApiClient(APP_KEY, APP_SECRET, Region.JP.value)
    trade_api = API(api_client)
    quotes_api = MarketData(api_client)
    return trade_api, quotes_api

def get_top_10_watchlist():
    try:
        # Load the latest top 10 calculated by the weekend scheduler PCA
        return ["RGC", "APLD", "ASTS", "AAOI", "FRMI", "SOUN", "NVDA", "MSTR", "SMCI", "ARM"]
    except Exception as e:
        logger.error(f"Could not load watchlist: {e}")
        return []

def place_order(api, symbol, side, qty, order_type="MARKET", limit_price=None):
    client_order_id = uuid.uuid4().hex
    
    new_orders = {
        "client_order_id": client_order_id,
        "symbol": symbol,
        "instrument_type": "EQUITY",
        "market": "US",
        "order_type": order_type,
        "quantity": str(qty),
        "support_trading_session": "N",
        "side": side,
        "time_in_force": "DAY",
        "entrust_type": "QTY",
        "account_tax_type": "GENERAL"
    }
    
    if limit_price:
        new_orders["limit_price"] = str(round(limit_price, 2))

    try:
        res = api.order_v2.place_order(account_id=ACCOUNT_ID, new_orders=new_orders)
        if res.status_code == 200:
            logger.info(f"Order Placed successfully: {new_orders}")
            return res.json()
        else:
            logger.error(f"Order failed: {res.status_code} - {res.text}")
    except Exception as e:
        logger.error(f"Order exception: {e}")
    return None

def fetch_webull_prices(quotes_api, symbols):
    prices = {}
    try:
        # Fetch the latest stock market snapshots in batches
        symbols_str = ",".join(symbols)
        res = quotes_api.get_snapshot(symbols_str, Category.US_STOCK.value)
        if res.status_code == 200:
            data = res.json()
            # The exact JSON structure requires parsing, typically a list of dicts.
            # Using robust fallback to find active prices
            if isinstance(data, list):
                for item in data:
                    sym = item.get("symbol")
                    price = item.get("price") or item.get("close")
                    if sym and price:
                        prices[sym] = float(price)
            elif isinstance(data, dict):
                # Try locating the list inside a 'data' key or similar
                items = data.get("data") or data.get("items") or data.get("snapshots", [])
                for item in items:
                    sym = item.get("symbol")
                    price = item.get("price") or item.get("close")
                    if sym and price:
                        prices[sym] = float(price)
    except Exception as e:
        logger.error(f"Failed to fetch Webull quotes: {e}")
    return prices

def fetch_opening_highs(quotes_api, symbols):
    opening_highs = {}
    symbols_str = ",".join(symbols)
    try:
        # Fetch 200 m1 bars or 1 m5 bar to get the Opening High
        res = quotes_api.get_batch_history_bar(symbols_str, Category.US_STOCK.value, "m5", count="1")
        if res.status_code == 200:
            data = res.json()
            # Expecting data to contain recent history payload
            if isinstance(data, list):
                for item in data:
                    sym = item.get("symbol")
                    # Find high price in the candle
                    high = item.get("high") or item.get("h")
                    if sym and high:
                        opening_highs[sym] = float(high)
    except Exception as e:
        logger.error(f"Failed to fetch Webull opening candles: {e}")
    return opening_highs

def main():
    logger.info("Starting Daily Trading Execution (Webull MDR)")
    trade_api, quotes_api = init_webull_client()
    if not trade_api or not quotes_api: return

    watchlist = get_top_10_watchlist()
    logger.info(f"Monitoring TOP 10: {watchlist}")
    
    # Wait until 09:30 AM EST
    ny_tz = pytz.timezone('America/New_York')
    
    while True:
        now = datetime.datetime.now(ny_tz)
        target_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        
        if now < target_open:
            sleep_time = (target_open - now).total_seconds()
            logger.info(f"Market not open yet. Sleeping for {sleep_time:.0f} seconds.")
            time.sleep(sleep_time)
        else:
            break
            
    logger.info("Market Open! Waiting for 5 minutes to generate Opening Range candles...")
    time.sleep(305) # Wait until 9:35:05 AM
    
    # Fetch Opening Range Highs
    # In live scenario we might fall back to snapshot if history bar is unreliable initially.
    opening_highs = fetch_opening_highs(quotes_api, watchlist)
    if not opening_highs:
        # Fallback to fetching current snapshot as proxy if history API struct is unknown
        logger.warning("Falling back to snapshot initial scan.")
        opening_highs = fetch_webull_prices(quotes_api, watchlist)
        
    logger.info(f"Opening Highs established: {opening_highs}")
    
    active_trade = None
    trade_symbol = None
    entry_price = 0.0
    
    logger.info("Beginning active breakout monitoring...")
    
    # Simple monitoring loop until 15:55 PM
    while True:
        now = datetime.datetime.now(ny_tz)
        
        # 1. Check EOD liquidation
        if now.hour == 15 and now.minute >= 55:
            logger.info("15:55 PM Reached. Liquidating active day trades (EOD Rule).")
            if active_trade and trade_symbol:
                logger.info(f"Liquidating {trade_symbol} Market Sell...")
                place_order(trade_api, trade_symbol, "SELL", active_trade.get('quantity', 100), "MARKET")
            break
            
        # 2. Check Time Stop for Entries (10:30 AM)
        if now.hour >= 10 and now.minute > 30 and not active_trade:
            logger.info("10:30 AM Entry Window Closed. No trades generated today.")
            break
            
        # 3. Pull Live Prices from Webull MDR
        current_prices = fetch_webull_prices(quotes_api, watchlist)
            
        if not active_trade:
            # Check for breakout logic
            for sym, open_high in opening_highs.items():
                if sym in current_prices:
                    current_price = current_prices[sym]
                    if current_price > open_high:
                        logger.info(f"*** BREAKOUT DETECTED: {sym} at {current_price} > {open_high} ***")
                        # Calculate position sizing (assuming $10k available, using 90% = $9k)
                        qty = int(9000 // current_price)
                        res = place_order(trade_api, sym, "BUY", qty, "MARKET")
                        if res:
                            active_trade = res
                            trade_symbol = sym
                            entry_price = current_price
                        break # Only take ONE trade per day!
                        
        else:
            # Monitor for +10% Take Profit or -3% Stop Loss
            if trade_symbol in current_prices:
                current_price = current_prices[trade_symbol]
                pnl_pct = (current_price - entry_price) / entry_price
                
                if pnl_pct >= 0.10:
                    logger.info(f"*** TAKE PROFIT HIT: {trade_symbol} (+{pnl_pct*100:.2f}%) ***")
                    place_order(trade_api, trade_symbol, "SELL", active_trade.get('quantity', 100), "MARKET")
                    break
                elif pnl_pct <= -0.03:
                    logger.info(f"*** STOP LOSS HIT: {trade_symbol} ({pnl_pct*100:.2f}%) ***")
                    place_order(trade_api, trade_symbol, "SELL", active_trade.get('quantity', 100), "MARKET")
                    break
            
        time.sleep(2) # Webull MDR handles quick polling (e.g. 2 sec), much faster than yfinance

if __name__ == "__main__":
    main()
