from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime

import pandas as pd
import pytz
from webullsdkcore.client import ApiClient
from webullsdkcore.common.region import Region
from webullsdktrade.api import API

from .config import Settings, load_settings
from .features import build_intraday_feature_panel
from .fmp import FMPClient
from .modeling import load_model_bundle, score_candidates
from .storage import SQLiteParquetStore
from .strategy import StandardSystemSpec, select_candidates_for_session


LOGGER = logging.getLogger(__name__)


def init_webull_trade_client(settings: Settings):
    if not settings.credentials.webull_app_key or not settings.credentials.webull_app_secret or not settings.credentials.webull_account_id:
        LOGGER.warning("Webull credentials are not fully configured. Live execution will be skipped.")
        return None
    api_client = ApiClient(settings.credentials.webull_app_key, settings.credentials.webull_app_secret, Region.JP.value)
    return API(api_client)


def place_order(api, account_id: str, symbol: str, side: str, qty: int):
    client_order_id = uuid.uuid4().hex
    new_orders = {
        "client_order_id": client_order_id,
        "symbol": symbol,
        "instrument_type": "EQUITY",
        "market": "US",
        "order_type": "MARKET",
        "quantity": str(qty),
        "support_trading_session": "N",
        "side": side,
        "time_in_force": "DAY",
        "entrust_type": "QTY",
        "account_tax_type": "SPECIFIC",
    }
    return api.order_v2.place_order(account_id=account_id, new_orders=new_orders)


def _ny_now(settings: Settings) -> datetime:
    return datetime.now(pytz.timezone(settings.runtime.market_timezone))


def _within_signal_window(now_ny: datetime, spec: StandardSystemSpec) -> bool:
    minutes_from_open = (now_ny.hour * 60 + now_ny.minute) - (9 * 60 + 30)
    return spec.min_minutes_from_open <= minutes_from_open <= spec.max_minutes_from_open


def _build_latest_bars_from_quotes(quotes: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume", "cumulative_volume"])
    work = quotes.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["volume"] = pd.to_numeric(work.get("volume"), errors="coerce")
    work["ts"] = timestamp
    work["open"] = work["price"]
    work["high"] = work["price"]
    work["low"] = work["price"]
    work["close"] = work["price"]
    work["cumulative_volume"] = work["volume"]
    return work[["symbol", "ts", "open", "high", "low", "close", "volume", "cumulative_volume"]].dropna(subset=["close"])


def run_live_trader(settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    store = SQLiteParquetStore(settings)
    fmp = FMPClient(settings)
    api = init_webull_trade_client(settings)
    spec = StandardSystemSpec(
        min_minutes_from_open=settings.runtime.min_minutes_from_open,
        max_minutes_from_open=settings.runtime.max_minutes_from_open,
        max_positions=settings.runtime.max_positions,
        threshold_floor=settings.runtime.threshold_floor,
        threshold_quantile=settings.runtime.threshold_quantile,
    )

    model_path = settings.paths.model_dir / "hist_gbm_extended_5m_start.pkl"
    model, threshold, _ = load_model_bundle(model_path)

    today = pd.Timestamp(_ny_now(settings).date())
    shortlist = store.load_shortlist(today)
    if shortlist.empty:
        shortlist = store.load_shortlist()
    shortlist = shortlist.head(settings.runtime.monitor_count).copy()
    symbols = shortlist["symbol"].dropna().astype(str).str.upper().tolist()
    if not symbols:
        raise RuntimeError("No shortlist symbols available. Run the nightly pipeline first.")

    LOGGER.info("Loaded %s monitored symbols for live trading", len(symbols))
    executed_symbols: set[str] = set()
    fills_today = 0

    while True:
        now_ny = _ny_now(settings)
        if now_ny.weekday() >= 5:
            LOGGER.info("Weekend detected. Exiting.")
            return
        if now_ny.hour < 9 or (now_ny.hour == 9 and now_ny.minute < 30):
            time.sleep(15)
            continue
        if now_ny.hour >= 16:
            LOGGER.info("Market session complete.")
            return
        if not _within_signal_window(now_ny, spec):
            time.sleep(settings.runtime.quote_poll_seconds)
            continue

        quotes = []
        for start in range(0, len(symbols), settings.runtime.batch_quote_chunk_size):
            chunk = symbols[start : start + settings.runtime.batch_quote_chunk_size]
            frame = fmp.fetch_batch_quotes(chunk)
            if not frame.empty:
                quotes.append(frame)
            time.sleep(0.1)
        quote_frame = pd.concat(quotes, ignore_index=True) if quotes else pd.DataFrame()
        bar_ts = pd.Timestamp(now_ny).floor("5min").tz_localize(None)
        latest_bars = _build_latest_bars_from_quotes(quote_frame, bar_ts)
        if latest_bars.empty:
            time.sleep(settings.runtime.quote_poll_seconds)
            continue

        store.save_bars(latest_bars, timeframe="5m")
        intraday_bars = store.load_bars("5m", symbols=symbols)
        daily_features = store.load_daily_features(today, symbols=symbols)
        if daily_features.empty:
            daily_features = store.load_daily_features(symbols=symbols)
            daily_features = daily_features.sort_values("session_date").groupby("symbol", sort=False).tail(1)
            daily_features["session_date"] = today

        candidate_panel = build_intraday_feature_panel(intraday_bars, daily_features, same_slot_lookback_sessions=settings.runtime.same_slot_lookback_sessions)
        candidate_panel = candidate_panel[candidate_panel["session_date"].eq(today)].copy()
        candidate_panel = candidate_panel[candidate_panel["session_bucket"].eq("open_drive")].copy()
        candidate_panel = candidate_panel[candidate_panel["minutes_from_open"].between(spec.min_minutes_from_open, spec.max_minutes_from_open, inclusive="both")]
        if candidate_panel.empty:
            time.sleep(settings.runtime.quote_poll_seconds)
            continue

        latest_per_symbol = candidate_panel.sort_values(["symbol", "timestamp"]).groupby("symbol", sort=False).tail(1).copy()
        scored = score_candidates(model, latest_per_symbol)
        scored["threshold"] = threshold
        selected = select_candidates_for_session(scored, threshold=threshold, max_positions=spec.max_positions)
        if not selected.empty:
            selected["selected"] = selected["symbol"].apply(lambda item: int(item not in executed_symbols))
            selected["session_date"] = selected["session_date"].astype(str)
            selected["payload_json"] = selected.apply(lambda row: json.dumps(row.to_dict(), ensure_ascii=True, default=str), axis=1)
            store.append_live_signals(selected[["session_date", "timestamp", "symbol", "score", "threshold", "selected", "payload_json"]])

        for row in selected.itertuples(index=False):
            if fills_today >= spec.max_positions:
                break
            if row.symbol in executed_symbols:
                continue
            if row.score < threshold:
                continue
            if api is None:
                LOGGER.info("Signal only mode: would buy %s at %s", row.symbol, row.close)
                executed_symbols.add(row.symbol)
                fills_today += 1
                continue
            quantity = 1
            response = place_order(api, settings.credentials.webull_account_id, row.symbol, "BUY", quantity)
            if response and getattr(response, "status_code", 500) == 200:
                executed_symbols.add(row.symbol)
                fills_today += 1
                store.append_live_fill(
                    {
                        "fill_id": uuid.uuid4().hex,
                        "session_date": str(today.date()),
                        "symbol": row.symbol,
                        "side": "BUY",
                        "timestamp": pd.Timestamp.utcnow().isoformat(),
                        "quantity": quantity,
                        "price": float(row.close),
                        "payload_json": json.dumps({"response": response.json()}, ensure_ascii=True, default=str),
                    }
                )
        time.sleep(settings.runtime.quote_poll_seconds)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_live_trader()


if __name__ == "__main__":
    main()
