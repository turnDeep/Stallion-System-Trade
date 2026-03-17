from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime

import pandas as pd
import pytz

from .bar_aggregator import QuoteBarAggregator
from .broker import WebullBroker
from .config import Settings, load_settings
from .features import build_intraday_feature_panel
from .fmp import FMPClient
from .modeling import load_model_bundle, score_candidates
from .notifier import emit_alert
from .storage import SQLiteParquetStore
from .strategy import StandardSystemSpec, select_candidates_for_session


LOGGER = logging.getLogger(__name__)

TERMINAL_ORDER_STATUSES = {"FILLED", "CANCELLED", "CANCELED", "REJECTED", "FAILED", "EXPIRED"}
PENDING_ORDER_STATUSES = {"NEW", "SUBMITTED", "PENDING", "PARTIALLY_FILLED", "PARTIAL_FILLED", "CREATED"}


def _ny_now(settings: Settings) -> datetime:
    return datetime.now(pytz.timezone(settings.runtime.market_timezone))


def _ny_timestamp(settings: Settings) -> pd.Timestamp:
    return pd.Timestamp(_ny_now(settings))


def _today_ny(settings: Settings) -> pd.Timestamp:
    return pd.Timestamp(_ny_now(settings).date())


def _within_signal_window(now_ny: datetime, spec: StandardSystemSpec) -> bool:
    minutes_from_open = (now_ny.hour * 60 + now_ny.minute) - (9 * 60 + 30)
    return spec.min_minutes_from_open <= minutes_from_open <= spec.max_minutes_from_open


def _after_cutoff(now_ny: datetime, hour: int, minute: int) -> bool:
    return (now_ny.hour, now_ny.minute) >= (hour, minute)


def _normalize_order_status(status: object, quantity: int, filled_quantity: int) -> str:
    status_str = str(status or "UNKNOWN").upper().replace(" ", "_")
    if filled_quantity >= quantity > 0:
        return "FILLED"
    if status_str in TERMINAL_ORDER_STATUSES:
        return status_str
    if "CANCEL" in status_str:
        return "CANCELLED"
    if "REJECT" in status_str:
        return "REJECTED"
    if "FAIL" in status_str:
        return "FAILED"
    if "PART" in status_str and filled_quantity > 0:
        return "PARTIALLY_FILLED"
    return status_str or "UNKNOWN"


def _active_order_symbols(orders: pd.DataFrame) -> set[str]:
    if orders.empty:
        return set()
    active = orders.copy()
    active["normalized_status"] = [
        _normalize_order_status(status, int(quantity or 0), int(filled or 0))
        for status, quantity, filled in zip(active["status"], active["quantity"], active["filled_quantity"])
    ]
    active = active[~active["normalized_status"].isin(TERMINAL_ORDER_STATUSES)]
    return set(active["symbol"].dropna().astype(str).str.upper())


def _executed_buy_symbols(orders: pd.DataFrame) -> set[str]:
    if orders.empty:
        return set()
    work = orders.copy()
    work["normalized_status"] = [
        _normalize_order_status(status, int(quantity or 0), int(filled or 0))
        for status, quantity, filled in zip(work["status"], work["quantity"], work["filled_quantity"])
    ]
    work = work[work["side"].astype(str).str.upper().eq("BUY")]
    work = work[work["normalized_status"].isin({"FILLED", "PARTIALLY_FILLED", "SUBMITTED", "PENDING", "NEW", "CREATED"})]
    return set(work["symbol"].dropna().astype(str).str.upper())


def _derive_positions_from_orders(orders: pd.DataFrame, session_date: pd.Timestamp) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame(columns=["symbol", "session_date", "quantity", "avg_price", "entry_time", "broker_order_id", "status", "payload_json", "updated_at"])

    work = orders.copy()
    work["side_sign"] = work["side"].astype(str).str.upper().map({"BUY": 1, "SELL": -1}).fillna(0)
    work["net_quantity"] = work["filled_quantity"].fillna(0).astype(int) * work["side_sign"].astype(int)
    grouped = work.groupby("symbol", as_index=False).agg(
        quantity=("net_quantity", "sum"),
        broker_order_id=("broker_order_id", "last"),
        updated_at=("updated_at", "last"),
        placed_at=("placed_at", "last"),
        payload_json=("payload_json", "last"),
    )
    grouped = grouped[grouped["quantity"] > 0].copy()
    if grouped.empty:
        return pd.DataFrame(columns=["symbol", "session_date", "quantity", "avg_price", "entry_time", "broker_order_id", "status", "payload_json", "updated_at"])
    grouped["session_date"] = str(pd.Timestamp(session_date).date())
    grouped["avg_price"] = None
    grouped["entry_time"] = grouped["placed_at"].fillna(pd.Timestamp.utcnow().isoformat())
    grouped["status"] = "OPEN"
    return grouped[["symbol", "session_date", "quantity", "avg_price", "entry_time", "broker_order_id", "status", "payload_json", "updated_at"]]


def _load_or_fetch_opening_buying_power(store: SQLiteParquetStore, broker: WebullBroker, session_date: pd.Timestamp) -> float:
    state_key = f"opening_buying_power:{pd.Timestamp(session_date).date()}"
    cached = store.get_system_state(state_key)
    if cached:
        return float(cached)
    opening_buying_power = broker.get_account_buying_power()
    store.put_system_state(state_key, str(opening_buying_power))
    return float(opening_buying_power)


def _compute_order_quantity(slot_budget: float, expected_fill_price: float) -> int:
    if not math.isfinite(slot_budget) or slot_budget <= 0:
        return 0
    if not math.isfinite(expected_fill_price) or expected_fill_price <= 0:
        return 0
    return max(int(math.floor(slot_budget / expected_fill_price)), 0)


def _build_quote_snapshot_frame(quotes: pd.DataFrame, observed_at_utc: pd.Timestamp) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame(columns=["symbol", "ts", "price", "cumulative_volume", "payload_json"])
    work = quotes.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["cumulative_volume"] = pd.to_numeric(work.get("volume"), errors="coerce").fillna(0.0)
    work["ts"] = observed_at_utc
    work["payload_json"] = work.apply(lambda row: json.dumps(row.to_dict(), ensure_ascii=True, default=str), axis=1)
    return work[["symbol", "ts", "price", "cumulative_volume", "payload_json"]].dropna(subset=["price"])


def _reconcile_orders_and_positions(
    store: SQLiteParquetStore,
    broker: WebullBroker,
    session_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    order_history = broker.get_order_history_df(lookback_days=2, page_size=200)
    if not order_history.empty:
        order_history["session_date"] = str(pd.Timestamp(session_date).date())
        order_history["requested_price"] = None
        order_history["broker_order_id"] = order_history["order_id"]
        order_history["updated_at"] = pd.Timestamp.utcnow().isoformat()
        order_history["placed_at"] = order_history["place_time_at"].fillna(pd.Timestamp.utcnow().isoformat())
        for row in order_history.to_dict(orient="records"):
            row["status"] = _normalize_order_status(row.get("status"), int(row.get("quantity") or 0), int(row.get("filled_quantity") or 0))
            store.upsert_live_order(row)

    stored_orders = store.load_live_orders(session_date=session_date)
    broker_positions = broker.get_positions_df()
    if broker_positions.empty:
        position_frame = _derive_positions_from_orders(stored_orders, session_date)
    else:
        position_frame = broker_positions.copy()
        position_frame["session_date"] = str(pd.Timestamp(session_date).date())
        position_frame["entry_time"] = pd.Timestamp.utcnow().isoformat()
        position_frame["broker_order_id"] = None
        position_frame["status"] = "OPEN"
        position_frame["updated_at"] = pd.Timestamp.utcnow().isoformat()
    store.replace_open_positions(position_frame)
    return stored_orders, position_frame


def _cancel_stale_orders(
    store: SQLiteParquetStore,
    broker: WebullBroker,
    session_date: pd.Timestamp,
    settings: Settings,
) -> None:
    orders = store.load_live_orders(session_date=session_date)
    if orders.empty:
        return
    now_utc = pd.Timestamp.now(tz="UTC")
    for row in orders.itertuples(index=False):
        quantity = int(getattr(row, "quantity", 0) or 0)
        filled_quantity = int(getattr(row, "filled_quantity", 0) or 0)
        status = _normalize_order_status(getattr(row, "status", None), quantity, filled_quantity)
        if status in TERMINAL_ORDER_STATUSES:
            continue
        placed_at = pd.to_datetime(getattr(row, "placed_at", None), utc=True, errors="coerce")
        if pd.isna(placed_at):
            continue
        age_seconds = (now_utc - placed_at.tz_convert("UTC")).total_seconds()
        if age_seconds < settings.runtime.order_cancel_after_seconds:
            continue
        try:
            result = broker.cancel_order(client_order_id=str(row.client_order_id))
            store.upsert_live_order(
                {
                    "client_order_id": str(row.client_order_id),
                    "session_date": str(pd.Timestamp(session_date).date()),
                    "symbol": str(row.symbol),
                    "side": str(row.side),
                    "quantity": quantity,
                    "filled_quantity": filled_quantity,
                    "requested_price": getattr(row, "requested_price", None),
                    "status": "CANCELLED",
                    "broker_order_id": getattr(row, "broker_order_id", None),
                    "placed_at": str(row.placed_at),
                    "updated_at": pd.Timestamp.utcnow().isoformat(),
                    "payload_json": json.dumps(result, ensure_ascii=True, default=str),
                }
            )
        except Exception as exc:
            emit_alert(
                store,
                level="ERROR",
                component="live_trader",
                message=f"Failed to cancel stale order {row.client_order_id}",
                payload={"error": str(exc), "symbol": str(row.symbol)},
            )


def _load_shortlist_symbols(store: SQLiteParquetStore, settings: Settings, session_date: pd.Timestamp) -> list[str]:
    shortlist = store.load_shortlist(session_date)
    if shortlist.empty:
        shortlist = store.load_shortlist()
    shortlist = shortlist.head(settings.runtime.monitor_count).copy()
    symbols = shortlist["symbol"].dropna().astype(str).str.upper().tolist()
    if not symbols:
        raise RuntimeError("No shortlist symbols available. Run the nightly pipeline first.")
    return symbols


def _load_daily_feature_slice(store: SQLiteParquetStore, session_date: pd.Timestamp, symbols: list[str]) -> pd.DataFrame:
    daily_features = store.load_daily_features(session_date, symbols=symbols)
    if daily_features.empty:
        daily_features = store.load_daily_features(symbols=symbols)
        if not daily_features.empty:
            daily_features = daily_features.sort_values("session_date").groupby("symbol", sort=False).tail(1)
            daily_features["session_date"] = session_date
    return daily_features


def _close_positions(
    store: SQLiteParquetStore,
    broker: WebullBroker,
    positions: pd.DataFrame,
    session_date: pd.Timestamp,
) -> None:
    if positions.empty:
        return
    existing_orders = store.load_live_orders(session_date=session_date)
    active_symbols = _active_order_symbols(existing_orders[existing_orders["side"].astype(str).str.upper().eq("SELL")])
    for row in positions.itertuples(index=False):
        symbol = str(row.symbol).upper()
        quantity = int(getattr(row, "quantity", 0) or 0)
        if quantity <= 0 or symbol in active_symbols:
            continue
        result = broker.place_market_order(symbol=symbol, side="SELL", quantity=quantity)
        status = "SUBMITTED" if int(result.get("status_code") or 500) == 200 else "ERROR"
        store.upsert_live_order(
            {
                "client_order_id": result["client_order_id"],
                "session_date": str(pd.Timestamp(session_date).date()),
                "symbol": symbol,
                "side": "SELL",
                "quantity": quantity,
                "filled_quantity": 0,
                "requested_price": None,
                "status": status,
                "broker_order_id": None,
                "placed_at": pd.Timestamp.utcnow().isoformat(),
                "updated_at": pd.Timestamp.utcnow().isoformat(),
                "payload_json": json.dumps(result, ensure_ascii=True, default=str),
            }
        )


def run_live_trader(settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    store = SQLiteParquetStore(settings)
    fmp = FMPClient(settings)
    broker = WebullBroker(settings)
    probe = broker.probe()
    store.put_system_state("broker_probe", json.dumps(probe.__dict__, ensure_ascii=True))

    spec = StandardSystemSpec(
        min_minutes_from_open=settings.runtime.min_minutes_from_open,
        max_minutes_from_open=settings.runtime.max_minutes_from_open,
        max_positions=settings.runtime.max_positions,
        threshold_floor=settings.runtime.threshold_floor,
        threshold_quantile=settings.runtime.threshold_quantile,
    )

    model_path = settings.paths.model_dir / "hist_gbm_extended_5m_start.pkl"
    model, threshold, _ = load_model_bundle(model_path)

    today = _today_ny(settings)
    symbols = _load_shortlist_symbols(store, settings, today)
    aggregator = QuoteBarAggregator(session_timezone=settings.runtime.market_timezone)
    stored_snapshots = store.load_quote_snapshots(session_date=today, symbols=symbols)
    bootstrap_bars = aggregator.bootstrap_from_snapshots(stored_snapshots)
    if not bootstrap_bars.empty:
        store.save_bars(bootstrap_bars, timeframe="5m")

    LOGGER.info("Loaded %s monitored symbols for live trading", len(symbols))
    last_broker_sync_at = pd.Timestamp("1970-01-01", tz="UTC")
    flattened_today = False
    opening_buying_power: float | None = None

    while True:
        now_ny = _ny_now(settings)
        now_utc = pd.Timestamp.now(tz="UTC")
        store.write_heartbeat(
            "live_trader",
            "running",
            {
                "now_ny": now_ny.isoformat(),
                "threshold": threshold,
                "flattened_today": flattened_today,
            },
        )

        if now_ny.weekday() >= 5:
            LOGGER.info("Weekend detected. Exiting live trader.")
            return

        if now_ny.hour < 9 or (now_ny.hour == 9 and now_ny.minute < 30):
            time.sleep(15)
            continue

        if opening_buying_power is None and _after_cutoff(now_ny, 9, 30):
            opening_buying_power = _load_or_fetch_opening_buying_power(store, broker, today)

        if (now_utc - last_broker_sync_at).total_seconds() >= settings.runtime.broker_sync_seconds:
            live_orders, open_positions = _reconcile_orders_and_positions(store, broker, today)
            _cancel_stale_orders(store, broker, today, settings)
            last_broker_sync_at = now_utc
        else:
            live_orders = store.load_live_orders(session_date=today)
            open_positions = store.load_open_positions()

        if _after_cutoff(now_ny, settings.runtime.flatten_positions_hour, settings.runtime.flatten_positions_minute):
            if not flattened_today:
                _close_positions(store, broker, open_positions, today)
                flattened_today = True
            if _after_cutoff(now_ny, settings.runtime.shutdown_hour, settings.runtime.shutdown_minute):
                LOGGER.info("Shutdown cutoff reached. Exiting live trader.")
                return
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
        snapshot_frame = _build_quote_snapshot_frame(quote_frame, now_utc)
        if snapshot_frame.empty:
            time.sleep(settings.runtime.quote_poll_seconds)
            continue

        store.append_quote_snapshots(snapshot_frame)
        finalized = aggregator.ingest_quotes(snapshot_frame[["symbol", "price", "cumulative_volume"]], observed_at_utc=now_utc)
        flushed = aggregator.flush_completed(now_utc - pd.Timedelta(seconds=1))
        finalized = pd.concat([finalized, flushed], ignore_index=True) if not flushed.empty else finalized
        if not finalized.empty:
            store.save_bars(finalized, timeframe="5m")

        if not _within_signal_window(now_ny, spec):
            time.sleep(settings.runtime.quote_poll_seconds)
            continue
        if _after_cutoff(now_ny, settings.runtime.no_new_orders_after_hour, settings.runtime.no_new_orders_after_minute):
            time.sleep(settings.runtime.quote_poll_seconds)
            continue
        if finalized.empty:
            time.sleep(settings.runtime.quote_poll_seconds)
            continue
        if opening_buying_power is None:
            time.sleep(settings.runtime.quote_poll_seconds)
            continue

        intraday_bars = store.load_bars("5m", symbols=symbols)
        daily_features = _load_daily_feature_slice(store, today, symbols)
        candidate_panel = build_intraday_feature_panel(
            intraday_bars,
            daily_features,
            same_slot_lookback_sessions=settings.runtime.same_slot_lookback_sessions,
        )
        candidate_panel = candidate_panel[candidate_panel["session_date"].eq(today)].copy()
        candidate_panel = candidate_panel[candidate_panel["session_bucket"].eq("open_drive")].copy()
        candidate_panel = candidate_panel[candidate_panel["minutes_from_open"].between(spec.min_minutes_from_open, spec.max_minutes_from_open, inclusive="both")]
        if candidate_panel.empty:
            time.sleep(settings.runtime.quote_poll_seconds)
            continue

        latest_timestamp = pd.to_datetime(finalized["ts"], utc=True, errors="coerce").max()
        latest_timestamp_ny = latest_timestamp.tz_convert(settings.runtime.market_timezone).tz_localize(None) if pd.notna(latest_timestamp) else None
        latest_per_symbol = candidate_panel.sort_values(["symbol", "timestamp"]).groupby("symbol", sort=False).tail(1).copy()
        if latest_timestamp_ny is not None:
            latest_per_symbol = latest_per_symbol[latest_per_symbol["timestamp"].eq(latest_timestamp_ny)].copy()
        if latest_per_symbol.empty:
            time.sleep(settings.runtime.quote_poll_seconds)
            continue

        scored = score_candidates(model, latest_per_symbol)
        scored["threshold"] = threshold
        selected = select_candidates_for_session(scored, threshold=threshold, max_positions=spec.max_positions)
        if not selected.empty:
            selected["selected"] = 1
            selected["session_date"] = selected["session_date"].astype(str)
            selected["payload_json"] = selected.apply(lambda row: json.dumps(row.to_dict(), ensure_ascii=True, default=str), axis=1)
            store.append_live_signals(selected[["session_date", "timestamp", "symbol", "score", "threshold", "selected", "payload_json"]])

        live_orders = store.load_live_orders(session_date=today)
        open_positions = store.load_open_positions()
        active_symbols = _active_order_symbols(live_orders) | set(open_positions.get("symbol", pd.Series(dtype="object")).astype(str).str.upper())
        executed_symbols = _executed_buy_symbols(live_orders)
        slot_budget = opening_buying_power / spec.max_positions
        open_count = len(active_symbols)

        for row in selected.itertuples(index=False):
            symbol = str(row.symbol).upper()
            if open_count >= spec.max_positions:
                break
            if symbol in active_symbols or symbol in executed_symbols:
                continue
            quantity = _compute_order_quantity(slot_budget, float(row.close))
            if quantity < 1:
                continue
            result = broker.place_market_order(symbol=symbol, side="BUY", quantity=quantity)
            status = "SUBMITTED" if int(result.get("status_code") or 500) == 200 else "ERROR"
            store.upsert_live_order(
                {
                    "client_order_id": result["client_order_id"],
                    "session_date": str(today.date()),
                    "symbol": symbol,
                    "side": "BUY",
                    "quantity": quantity,
                    "filled_quantity": 0,
                    "requested_price": float(row.close),
                    "status": status,
                    "broker_order_id": None,
                    "placed_at": pd.Timestamp.utcnow().isoformat(),
                    "updated_at": pd.Timestamp.utcnow().isoformat(),
                    "payload_json": json.dumps(result, ensure_ascii=True, default=str),
                }
            )
            if status != "SUBMITTED":
                emit_alert(
                    store,
                    level="ERROR",
                    component="live_trader",
                    message=f"Buy order failed for {symbol}",
                    payload={"response": result},
                )
                continue
            active_symbols.add(symbol)
            executed_symbols.add(symbol)
            open_count += 1

        time.sleep(settings.runtime.quote_poll_seconds)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_live_trader()


if __name__ == "__main__":
    main()
