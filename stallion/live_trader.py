from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

import pandas as pd
import pytz

from .bar_aggregator import QuoteBarAggregator
from .broker import create_broker
from .buying_power_manager import compute_order_quantity
from .config import Settings, load_settings
from .discord_notifier import DiscordNotifier
from .features import build_intraday_feature_panel
from .fmp import FMPClient
from .modeling import load_model_bundle, score_candidates
from .notifier import emit_alert
from .order_state import TERMINAL_ORDER_STATUSES, normalize_order_status
from .slot_manager import SlotManager
from .storage import SQLiteParquetStore
from .strategy import StandardSystemSpec, select_candidates_for_session


LOGGER = logging.getLogger(__name__)


def _ny_now(settings: Settings) -> datetime:
    return datetime.now(pytz.timezone(settings.runtime.market_timezone))


def _today_ny(settings: Settings) -> pd.Timestamp:
    return pd.Timestamp(_ny_now(settings).date())


def _within_signal_window(now_ny: datetime, spec: StandardSystemSpec) -> bool:
    minutes_from_open = (now_ny.hour * 60 + now_ny.minute) - (9 * 60 + 30)
    return spec.min_minutes_from_open <= minutes_from_open <= spec.max_minutes_from_open


def _after_cutoff(now_ny: datetime, hour: int, minute: int) -> bool:
    return (now_ny.hour, now_ny.minute) >= (hour, minute)


def _payload_dict(raw_payload: object) -> dict[str, Any]:
    if isinstance(raw_payload, dict):
        return dict(raw_payload)
    try:
        return json.loads(raw_payload or "{}")
    except Exception:
        return {}


def _extract_slot_id(row: dict[str, Any]) -> int | None:
    value = _payload_dict(row.get("payload_json")).get("slot_id")
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _extract_avg_fill_price(row: dict[str, Any]) -> float | None:
    payload = _payload_dict(row.get("payload_json"))
    for key in ("avg_fill_price", "filled_avg_price", "average_price", "avg_price", "deal_price"):
        value = payload.get(key)
        if value not in {None, ""}:
            try:
                return float(value)
            except Exception:
                continue
    try:
        return float(row["requested_price"]) if row.get("requested_price") is not None else None
    except Exception:
        return None


def _resolve_close_quantity(row: dict[str, Any]) -> int:
    available_quantity = row.get("available_quantity")
    quantity = row.get("quantity")
    available_int = int(available_quantity or 0)
    quantity_int = int(quantity or 0)
    if available_quantity is not None and available_int > 0:
        return available_int
    return max(quantity_int, 0)


def _build_quote_snapshot_frame(quotes: pd.DataFrame, observed_at_utc: pd.Timestamp) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame(columns=["symbol", "ts", "price", "cumulative_volume", "payload_json"])
    work = quotes.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    volume_series = work["volume"] if "volume" in work.columns else pd.Series(0.0, index=work.index, dtype="float64")
    work["cumulative_volume"] = pd.to_numeric(volume_series, errors="coerce").fillna(0.0)
    work["ts"] = observed_at_utc
    work["payload_json"] = work.apply(lambda row: json.dumps(row.to_dict(), ensure_ascii=True, default=str), axis=1)
    return work[["symbol", "ts", "price", "cumulative_volume", "payload_json"]].dropna(subset=["price"])


def _load_or_fetch_opening_buying_power(store: SQLiteParquetStore, broker, session_date: pd.Timestamp) -> float:
    state_key = f"opening_buying_power:{pd.Timestamp(session_date).date()}"
    cached = store.get_system_state(state_key)
    if cached:
        return float(cached)
    opening_buying_power = broker.get_account_buying_power()
    store.put_system_state(state_key, str(opening_buying_power))
    return float(opening_buying_power)


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
            daily_features["session_date"] = pd.Timestamp(session_date)
    return daily_features


def _summarize_signal_reason(row: pd.Series) -> str:
    candidates = {
        "daily_buy_pressure_prev": row.get("daily_buy_pressure_prev"),
        "daily_rrs_prev": row.get("daily_rrs_prev"),
        "daily_rs_score_prev": row.get("daily_rs_score_prev"),
        "close_vs_vwap_15": row.get("close_vs_vwap_15"),
        "volume_spike_5m": row.get("volume_spike_5m"),
        "intraday_range_expansion_vs_atr": row.get("intraday_range_expansion_vs_atr"),
        "rs_x_intraday_rvol": row.get("rs_x_intraday_rvol"),
    }
    ranked = []
    for key, value in candidates.items():
        try:
            numeric = float(value)
        except Exception:
            continue
        if pd.isna(numeric):
            continue
        ranked.append((abs(numeric), key, numeric))
    ranked.sort(reverse=True)
    top = [f"{name}={value:.3f}" for _, name, value in ranked[:3]]
    return ", ".join(top) if top else "model_score_above_threshold"


def _log_order_transition(
    store: SQLiteParquetStore,
    *,
    session_date: pd.Timestamp,
    row: dict[str, Any],
    previous_status: str | None,
    new_status: str,
) -> None:
    store.append_order_state_event(
        client_order_id=str(row.get("client_order_id") or ""),
        session_date=session_date,
        symbol=row.get("symbol"),
        slot_id=_extract_slot_id(row),
        event_type="order_status_transition",
        from_status=previous_status,
        to_status=new_status,
        payload={
            "quantity": row.get("quantity"),
            "filled_quantity": row.get("filled_quantity"),
            "requested_price": row.get("requested_price"),
        },
    )


def _append_fill_if_needed(
    store: SQLiteParquetStore,
    *,
    session_date: pd.Timestamp,
    row: dict[str, Any],
    previous_filled_quantity: int,
) -> int:
    current_filled = int(row.get("filled_quantity") or 0)
    delta = max(0, current_filled - previous_filled_quantity)
    if delta <= 0:
        return 0
    fill_price = _extract_avg_fill_price(row)
    if fill_price is None:
        return 0
    fill_id = f"{row.get('client_order_id')}:{current_filled}"
    store.append_live_fill(
        {
            "fill_id": fill_id,
            "session_date": str(pd.Timestamp(session_date).date()),
            "symbol": str(row.get("symbol") or "").upper(),
            "side": str(row.get("side") or "").upper(),
            "timestamp": row.get("updated_at") or pd.Timestamp.utcnow().isoformat(),
            "quantity": delta,
            "price": fill_price,
            "payload_json": json.dumps(row, ensure_ascii=True, default=str),
        }
    )
    return delta


def _replace_demo_positions_from_slots(store: SQLiteParquetStore, slot_manager: SlotManager, session_date: pd.Timestamp) -> None:
    rows = []
    for slot in slot_manager.slots:
        if slot.status in {"FILLED", "PARTIALLY_FILLED", "SELL_PENDING"} and slot.symbol and slot.filled_quantity > 0:
            rows.append(
                {
                    "symbol": slot.symbol,
                    "session_date": str(pd.Timestamp(session_date).date()),
                    "quantity": slot.filled_quantity,
                    "avg_price": slot.avg_fill_price,
                    "entry_time": slot.updated_at or pd.Timestamp.utcnow().isoformat(),
                    "broker_order_id": slot.client_order_id,
                    "status": "OPEN",
                    "payload_json": json.dumps({"slot_id": slot.slot_id}, ensure_ascii=True),
                    "updated_at": pd.Timestamp.utcnow().isoformat(),
                }
            )
    store.replace_open_positions(pd.DataFrame(rows))


def _build_pre_market_lines(*, settings: Settings, buying_power: float, threshold: float) -> list[str]:
    return [
        "- bot_running: true",
        f"- mode: {settings.trade_mode}",
        f"- buying_power: ${buying_power:,.2f}",
        f"- threshold: {threshold:.3f}",
    ]


def _build_order_submitted_lines(
    *,
    symbol: str,
    quantity: int,
    expected_price: float,
    score: float,
    threshold: float,
    slot_id: int,
    signal_reason: str,
) -> list[str]:
    return [
        f"- symbol: {symbol}",
        f"- qty: {quantity}",
        f"- expected_price: {expected_price:.2f}",
        f"- score: {score:.3f}",
        f"- threshold: {threshold:.3f}",
        f"- slot_id: {slot_id}",
        f"- reason: {signal_reason}",
    ]


def _build_fill_lines(
    *,
    symbol: str,
    qty_filled: int,
    avg_fill_price: float | None,
    filled_at: str | None,
    partial_fill: bool,
    remaining_qty: int,
    slot_id: int | None,
) -> list[str]:
    return [
        f"- symbol: {symbol}",
        f"- qty_filled: {qty_filled}",
        f"- avg_fill_price: {avg_fill_price:.2f}" if avg_fill_price is not None else "- avg_fill_price: unknown",
        f"- filled_at: {filled_at}",
        f"- partial_fill: {str(partial_fill).lower()}",
        f"- remaining_qty: {remaining_qty}",
        f"- slot_id: {slot_id}",
    ]


def _build_close_summary_lines(summary: dict[str, Any]) -> list[str]:
    return [
        f"- all_positions_closed: {str(summary['all_positions_closed']).lower()}",
        f"- remaining_positions: {summary['remaining_positions']}",
        f"- today_pnl: ${summary['today_pnl']:,.2f}",
        f"- cumulative_pnl_since_deploy: ${summary['cumulative_pnl']:,.2f}",
        f"- fills_today: {summary['fills_today']}",
        f"- wins_today: {summary['wins_today']}",
        f"- losses_today: {summary['losses_today']}",
        f"- canceled_orders_today: {summary['canceled_orders_today']}",
        f"- failed_orders_today: {summary['failed_orders_today']}",
        f"- max_drawdown_today: ${summary['max_drawdown']:,.2f}",
    ]


def _simulate_demo_fill(
    store: SQLiteParquetStore,
    slot_manager: SlotManager,
    *,
    session_date: pd.Timestamp,
    order_row: dict[str, Any],
    fill_price: float,
    notifier: DiscordNotifier,
) -> None:
    filled = int(order_row.get("quantity") or 0)
    order_row = dict(order_row)
    order_row["filled_quantity"] = filled
    order_row["status"] = "FILLED"
    order_row["updated_at"] = pd.Timestamp.utcnow().isoformat()
    payload = _payload_dict(order_row.get("payload_json"))
    payload["avg_fill_price"] = fill_price
    order_row["payload_json"] = json.dumps(payload, ensure_ascii=True, default=str)
    store.upsert_live_order(order_row)
    _append_fill_if_needed(store, session_date=session_date, row=order_row, previous_filled_quantity=0)
    slot_id = _extract_slot_id(order_row)
    if slot_id is not None:
        slot = slot_manager.get_slot(slot_id)
        slot.status = "FILLED" if str(order_row.get("side")).upper() == "BUY" else "AVAILABLE"
        slot.filled_quantity = filled if str(order_row.get("side")).upper() == "BUY" else 0
        slot.avg_fill_price = fill_price
        slot.reserved_buying_power = 0.0
        if str(order_row.get("side")).upper() == "SELL":
            slot_manager.release(slot_id)
    _replace_demo_positions_from_slots(store, slot_manager, session_date)
    notifier.notify(
        "BUY FILLED" if str(order_row.get("side")).upper() == "BUY" else "SELL FILLED",
        _build_fill_lines(
            symbol=str(order_row.get("symbol") or ""),
            qty_filled=filled,
            avg_fill_price=fill_price,
            filled_at=str(order_row.get("updated_at") or ""),
            partial_fill=False,
            remaining_qty=0,
            slot_id=slot_id,
        ),
    )


def _handle_broker_order_updates(
    store: SQLiteParquetStore,
    session_date: pd.Timestamp,
    previous_orders: pd.DataFrame,
    new_orders: list[dict[str, Any]],
    notifier: DiscordNotifier,
) -> None:
    previous_by_id = {str(row["client_order_id"]): row for row in previous_orders.to_dict(orient="records")}
    for row in new_orders:
        client_order_id = str(row.get("client_order_id") or "")
        previous = previous_by_id.get(client_order_id, {})
        previous_status = str(previous.get("status") or "") or None
        previous_filled = int(previous.get("filled_quantity") or 0)
        new_status = str(row.get("status") or "")
        if previous_status != new_status:
            _log_order_transition(store, session_date=session_date, row=row, previous_status=previous_status, new_status=new_status)
            if new_status in {"CANCELLED", "REJECTED", "FAILED", "EXPIRED"}:
                notifier.notify(
                    "ORDER STATUS UPDATE",
                    [
                        f"- symbol: {row.get('symbol')}",
                        f"- client_order_id: {client_order_id}",
                        f"- status: {new_status}",
                        f"- slot_id: {_extract_slot_id(row)}",
                    ],
                    level="WARNING",
                )
        delta_fill = _append_fill_if_needed(store, session_date=session_date, row=row, previous_filled_quantity=previous_filled)
        if delta_fill > 0:
            fill_price = _extract_avg_fill_price(row)
            remaining = max(int(row.get("quantity") or 0) - int(row.get("filled_quantity") or 0), 0)
            partial = remaining > 0
            notifier.notify(
                "BUY FILLED" if str(row.get("side") or "").upper() == "BUY" else "SELL FILLED",
                _build_fill_lines(
                    symbol=str(row.get("symbol") or ""),
                    qty_filled=delta_fill,
                    avg_fill_price=fill_price,
                    filled_at=str(row.get("updated_at") or ""),
                    partial_fill=partial,
                    remaining_qty=remaining,
                    slot_id=_extract_slot_id(row),
                ),
            )


def _reconcile_orders_and_positions(
    store: SQLiteParquetStore,
    broker,
    session_date: pd.Timestamp,
    slot_manager: SlotManager,
    notifier: DiscordNotifier,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_orders = store.load_live_orders(session_date=session_date)
    existing_by_id = {str(row["client_order_id"]): row for row in existing_orders.to_dict(orient="records")}
    reconciled_rows: list[dict[str, Any]] = []

    if not broker.is_demo:
        order_history = broker.get_order_history_df(lookback_days=2, page_size=100)
        for row in order_history.to_dict(orient="records"):
            client_order_id = str(row.get("client_order_id") or "")
            existing = existing_by_id.get(client_order_id, {})
            payload = _payload_dict(existing.get("payload_json"))
            payload.update(_payload_dict(row.get("payload_json")))
            normalized_status = normalize_order_status(row.get("status"), int(row.get("quantity") or 0), int(row.get("filled_quantity") or 0))
            reconciled = {
                "client_order_id": client_order_id,
                "session_date": str(pd.Timestamp(session_date).date()),
                "symbol": str(row.get("symbol") or existing.get("symbol") or "").upper(),
                "side": str(row.get("side") or existing.get("side") or "").upper(),
                "quantity": int(row.get("quantity") or existing.get("quantity") or 0),
                "filled_quantity": int(row.get("filled_quantity") or 0),
                "requested_price": existing.get("requested_price"),
                "status": normalized_status,
                "broker_order_id": row.get("order_id") or existing.get("broker_order_id"),
                "placed_at": row.get("place_time_at") or existing.get("placed_at") or pd.Timestamp.utcnow().isoformat(),
                "updated_at": row.get("filled_time_at") or pd.Timestamp.utcnow().isoformat(),
                "payload_json": json.dumps(payload, ensure_ascii=True, default=str),
            }
            reconciled_rows.append(reconciled)
            store.upsert_live_order(reconciled)
        _handle_broker_order_updates(store, session_date, existing_orders, reconciled_rows, notifier)
        live_orders = store.load_live_orders(session_date=session_date)
        open_positions = broker.get_positions_df()
        open_positions = open_positions.copy()
        open_positions["session_date"] = str(pd.Timestamp(session_date).date())
        open_positions["entry_time"] = pd.Timestamp.now(tz="UTC").isoformat()
        open_positions["broker_order_id"] = None
        open_positions["status"] = "OPEN"
        open_positions["updated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        if "payload_json" not in open_positions.columns:
            if open_positions.empty:
                open_positions["payload_json"] = pd.Series(dtype="object")
            else:
                open_positions["payload_json"] = open_positions.apply(lambda item: json.dumps(item.to_dict(), ensure_ascii=True, default=str), axis=1)
        store.replace_open_positions(open_positions)
    else:
        live_orders = existing_orders
        open_positions = store.load_open_positions()

    slot_manager.sync_from_orders_and_positions(live_orders, open_positions)
    store.replace_slot_states(slot_manager.to_frame(session_date), session_date)
    return live_orders, open_positions


def _cancel_stale_orders(
    store: SQLiteParquetStore,
    broker,
    session_date: pd.Timestamp,
    settings: Settings,
    notifier: DiscordNotifier,
) -> None:
    orders = store.load_live_orders(session_date=session_date)
    if orders.empty:
        return
    now_utc = pd.Timestamp.now(tz="UTC")
    for row in orders.to_dict(orient="records"):
        quantity = int(row.get("quantity") or 0)
        filled_quantity = int(row.get("filled_quantity") or 0)
        status = normalize_order_status(row.get("status"), quantity, filled_quantity)
        if status in TERMINAL_ORDER_STATUSES or status == "CANCEL_REQUESTED":
            continue
        placed_at = pd.to_datetime(row.get("placed_at"), utc=True, errors="coerce")
        if pd.isna(placed_at):
            continue
        age_seconds = float((now_utc - placed_at).total_seconds())
        if age_seconds < settings.runtime.order_cancel_after_seconds:
            continue
        result = broker.cancel_order(client_order_id=str(row["client_order_id"]))
        payload = _payload_dict(row.get("payload_json"))
        payload["cancel_requested_at"] = pd.Timestamp.utcnow().isoformat()
        row["status"] = "CANCEL_REQUESTED"
        row["updated_at"] = pd.Timestamp.utcnow().isoformat()
        row["payload_json"] = json.dumps(payload, ensure_ascii=True, default=str)
        store.upsert_live_order(row)
        store.append_order_state_event(
            client_order_id=str(row.get("client_order_id") or ""),
            session_date=session_date,
            symbol=row.get("symbol"),
            slot_id=_extract_slot_id(row),
            event_type="cancel_requested",
            from_status=status,
            to_status="CANCEL_REQUESTED",
            payload=result,
        )
        notifier.notify(
            "ORDER CANCEL REQUESTED",
            [
                f"- symbol: {row.get('symbol')}",
                f"- client_order_id: {row.get('client_order_id')}",
                f"- age_seconds: {age_seconds:.0f}",
                f"- slot_id: {_extract_slot_id(row)}",
            ],
            level="WARNING",
        )


def _compute_close_summary(store: SQLiteParquetStore, session_date: pd.Timestamp, settings: Settings) -> dict[str, Any]:
    fills = store.load_live_fills(session_date)
    orders = store.load_live_orders(session_date=session_date)
    summaries = store.load_daily_trade_summaries()

    symbol_pnl: list[float] = []
    today_pnl = 0.0
    if not fills.empty:
        work = fills.copy()
        work["side"] = work["side"].astype(str).str.upper()
        work["quantity"] = pd.to_numeric(work["quantity"], errors="coerce").fillna(0).astype(int)
        work["price"] = pd.to_numeric(work["price"], errors="coerce").fillna(0.0)
        for _, frame in work.groupby("symbol", sort=False):
            buy_notional = float((frame.loc[frame["side"].eq("BUY"), "quantity"] * frame.loc[frame["side"].eq("BUY"), "price"]).sum())
            sell_notional = float((frame.loc[frame["side"].eq("SELL"), "quantity"] * frame.loc[frame["side"].eq("SELL"), "price"]).sum())
            symbol_realized = (sell_notional * (1.0 - settings.costs.commission_rate_one_way)) - (buy_notional * (1.0 + settings.costs.commission_rate_one_way))
            if buy_notional or sell_notional:
                symbol_pnl.append(symbol_realized)
                today_pnl += symbol_realized
    previous_cumulative = 0.0
    if not summaries.empty:
        previous = summaries.loc[summaries["session_date"] < str(pd.Timestamp(session_date).date())]
        if not previous.empty:
            previous_cumulative = float(previous["today_pnl"].sum())
    cumulative_pnl = previous_cumulative + today_pnl
    pnl_curve = pd.Series((summaries["today_pnl"].tolist() if not summaries.empty else []) + [today_pnl], dtype="float64").cumsum()
    max_drawdown = float((pnl_curve - pnl_curve.cummax()).min()) if not pnl_curve.empty else 0.0
    canceled_orders = int(orders["status"].astype(str).str.upper().str.contains("CANCEL").sum()) if not orders.empty else 0
    failed_orders = int(orders["status"].astype(str).str.upper().isin({"FAILED", "REJECTED"}).sum()) if not orders.empty else 0
    remaining_positions = int(len(store.load_open_positions()))
    return {
        "today_pnl": today_pnl,
        "cumulative_pnl": cumulative_pnl,
        "fills_today": int(len(fills)),
        "wins_today": int(sum(1 for value in symbol_pnl if value > 0)),
        "losses_today": int(sum(1 for value in symbol_pnl if value < 0)),
        "canceled_orders_today": canceled_orders,
        "failed_orders_today": failed_orders,
        "remaining_positions": remaining_positions,
        "all_positions_closed": remaining_positions == 0,
        "max_drawdown": max_drawdown,
    }


def _submit_order_with_fallback(
    broker,
    *,
    symbol: str,
    side: str,
    quantity: int,
    expected_price: float,
    settings: Settings,
    notifier: DiscordNotifier,
) -> dict[str, Any]:
    result = broker.place_market_order(symbol=symbol, side=side, quantity=quantity)
    if int(result.get("status_code") or 500) == 200:
        return result
    retry_count = max(0, int(settings.runtime.marketable_limit_retry_count))
    for retry_index in range(1, retry_count + 1):
        direction = 1.0 if side.upper() == "BUY" else -1.0
        limit_price = expected_price * (1.0 + direction * (settings.runtime.marketable_limit_bps / 10_000.0))
        notifier.notify(
            "ORDER FALLBACK",
            [
                f"- symbol: {symbol}",
                f"- original_type: market",
                f"- fallback_type: marketable_limit",
                f"- retry_count: {retry_index}",
                f"- limit_price: {limit_price:.4f}",
            ],
            level="WARNING",
        )
        fallback = broker.place_marketable_limit_order(symbol=symbol, side=side, quantity=quantity, limit_price=limit_price)
        if int(fallback.get("status_code") or 500) == 200:
            return fallback
        result = fallback
    return result


def _close_positions(
    store: SQLiteParquetStore,
    broker,
    slot_manager: SlotManager,
    session_date: pd.Timestamp,
    settings: Settings,
    notifier: DiscordNotifier,
) -> None:
    positions = store.load_open_positions()
    if positions.empty:
        return
    existing_orders = store.load_live_orders(session_date=session_date)
    active_sell_symbols = {
        str(row["symbol"]).upper()
        for row in existing_orders.to_dict(orient="records")
        if str(row.get("side") or "").upper() == "SELL" and normalize_order_status(row.get("status"), int(row.get("quantity") or 0), int(row.get("filled_quantity") or 0)) not in TERMINAL_ORDER_STATUSES
    }
    for row in positions.to_dict(orient="records"):
        symbol = str(row.get("symbol") or "").upper()
        quantity = _resolve_close_quantity(row)
        if quantity <= 0 or symbol in active_sell_symbols:
            continue
        slot_id = _payload_dict(row.get("payload_json")).get("slot_id")
        result = _submit_order_with_fallback(
            broker,
            symbol=symbol,
            side="SELL",
            quantity=quantity,
            expected_price=float(row.get("avg_price") or 0.0) or 0.01,
            settings=settings,
            notifier=notifier,
        )
        if int(result.get("status_code") or 500) != 200:
            emit_alert(store, level="ERROR", component="live_trader", message=f"Close order failed for {symbol}", payload=result, discord=notifier)
            continue
        payload = {"slot_id": slot_id, "order_type": result.get("order_type", "MARKET"), "reserved_buying_power": 0.0}
        order_row = {
            "client_order_id": result["client_order_id"],
            "session_date": str(pd.Timestamp(session_date).date()),
            "symbol": symbol,
            "side": "SELL",
            "quantity": quantity,
            "filled_quantity": 0,
            "requested_price": row.get("avg_price"),
            "status": "SUBMITTED",
            "broker_order_id": None,
            "placed_at": pd.Timestamp.utcnow().isoformat(),
            "updated_at": pd.Timestamp.utcnow().isoformat(),
            "payload_json": json.dumps(payload, ensure_ascii=True, default=str),
        }
        store.upsert_live_order(order_row)
        if slot_id:
            slot_manager.mark_sell_pending(slot_id=int(slot_id), client_order_id=result["client_order_id"], quantity=quantity)
        notifier.notify("SELL ORDER SUBMITTED", [f"- symbol: {symbol}", f"- qty: {quantity}", f"- slot_id: {slot_id}"])
        if broker.is_demo:
            _simulate_demo_fill(store, slot_manager, session_date=session_date, order_row=order_row, fill_price=float(row.get("avg_price") or 0.0), notifier=notifier)


def run_live_trader(settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    store = SQLiteParquetStore(settings)
    notifier = DiscordNotifier(settings, store)
    discord_probe = notifier.probe()
    broker = create_broker(settings)
    broker_probe = broker.probe()
    store.put_system_state("broker_probe", json.dumps(broker_probe.__dict__, ensure_ascii=True))
    LOGGER.info("Trading mode resolved to %s", settings.trade_mode)
    notifier.notify(
        "BOT STARTUP",
        [
            f"- mode: {settings.trade_mode}",
            f"- bot_running: true",
            f"- broker_region: {broker_probe.region}",
            f"- discord_token_valid: {str(discord_probe.token_valid).lower()}",
            f"- discord_can_send: {str(discord_probe.can_send_messages).lower()}",
        ],
    )

    spec = StandardSystemSpec(
        min_minutes_from_open=settings.runtime.min_minutes_from_open,
        max_minutes_from_open=settings.runtime.max_minutes_from_open,
        max_positions=settings.runtime.max_positions,
        threshold_floor=settings.runtime.threshold_floor,
        threshold_quantile=settings.runtime.threshold_quantile,
    )
    model_path = settings.paths.model_dir / "hist_gbm_extended_5m_start.pkl"
    model, threshold, _ = load_model_bundle(model_path)
    fmp = FMPClient(settings)

    today = _today_ny(settings)
    slot_manager = SlotManager.from_frame(store.load_slot_states(today), max_positions=settings.runtime.max_positions)
    symbols = _load_shortlist_symbols(store, settings, today)
    aggregator = QuoteBarAggregator(session_timezone=settings.runtime.market_timezone)
    stored_snapshots = store.load_quote_snapshots(session_date=today, symbols=symbols)
    bootstrap_bars = aggregator.bootstrap_from_snapshots(stored_snapshots)
    if not bootstrap_bars.empty:
        store.save_bars(bootstrap_bars, timeframe="5m")

    opening_buying_power: float | None = None
    premarket_notified = False
    flattened_today = False
    last_broker_sync_at = pd.Timestamp("1970-01-01", tz="UTC")

    while True:
        now_ny = _ny_now(settings)
        now_utc = pd.Timestamp.now(tz="UTC")
        store.write_heartbeat("live_trader", "running", {"now_ny": now_ny.isoformat(), "threshold": threshold, "mode": settings.trade_mode})

        if now_ny.weekday() >= 5:
            notifier.notify("LIVE TRADER EXIT", ["- reason: weekend"])
            notifier.flush()
            return

        if not premarket_notified and _after_cutoff(now_ny, settings.runtime.pre_market_notification_hour, settings.runtime.pre_market_notification_minute):
            try:
                current_buying_power = broker.get_account_buying_power()
            except Exception:
                current_buying_power = settings.runtime.demo_starting_buying_power if settings.demo_mode else 0.0
            notifier.notify(
                "Pre-market status",
                _build_pre_market_lines(settings=settings, buying_power=current_buying_power, threshold=threshold),
            )
            premarket_notified = True

        if now_ny.hour < 9 or (now_ny.hour == 9 and now_ny.minute < 30):
            time.sleep(15)
            continue

        if opening_buying_power is None:
            opening_buying_power = _load_or_fetch_opening_buying_power(store, broker, today)

        if (now_utc - last_broker_sync_at).total_seconds() >= settings.runtime.broker_sync_seconds:
            _reconcile_orders_and_positions(store, broker, today, slot_manager, notifier)
            _cancel_stale_orders(store, broker, today, settings, notifier)
            last_broker_sync_at = now_utc
        else:
            live_orders = store.load_live_orders(session_date=today)
            open_positions = store.load_open_positions()
            slot_manager.sync_from_orders_and_positions(live_orders, open_positions)
            store.replace_slot_states(slot_manager.to_frame(today), today)

        if _after_cutoff(now_ny, settings.runtime.flatten_positions_hour, settings.runtime.flatten_positions_minute):
            if not flattened_today:
                _close_positions(store, broker, slot_manager, today, settings, notifier)
                flattened_today = True
            if _after_cutoff(now_ny, settings.runtime.shutdown_hour, settings.runtime.shutdown_minute):
                summary = _compute_close_summary(store, today, settings)
                store.save_daily_trade_summary(session_date=today, mode=settings.trade_mode, payload=summary, **summary)
                notifier.notify(
                    "MARKET CLOSE SUMMARY",
                    _build_close_summary_lines(summary),
                    level="WARNING" if not summary["all_positions_closed"] else "INFO",
                )
                notifier.flush()
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

        if not _within_signal_window(now_ny, spec) or _after_cutoff(now_ny, settings.runtime.no_new_orders_after_hour, settings.runtime.no_new_orders_after_minute):
            time.sleep(settings.runtime.quote_poll_seconds)
            continue
        if finalized.empty or opening_buying_power is None or slot_manager.available_slots <= 0:
            time.sleep(settings.runtime.quote_poll_seconds)
            continue

        intraday_bars = store.load_bars("5m", symbols=symbols)
        daily_features = _load_daily_feature_slice(store, today, symbols)
        candidate_panel = build_intraday_feature_panel(intraday_bars, daily_features, same_slot_lookback_sessions=settings.runtime.same_slot_lookback_sessions)
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

        try:
            current_buying_power = broker.get_account_buying_power()
        except Exception as exc:
            emit_alert(store, level="ERROR", component="live_trader", message="Failed to fetch buying power", payload={"error": str(exc)}, discord=notifier)
            current_buying_power = 0.0
        slot_budget = opening_buying_power / spec.max_positions

        for row in selected.itertuples(index=False):
            slot = slot_manager.next_available_slot()
            if slot is None:
                break
            symbol = str(row.symbol).upper()
            if slot_manager.contains_symbol(symbol):
                continue
            sizing = compute_order_quantity(
                slot_budget=slot_budget,
                effective_buying_power=slot_manager.available_buying_power_effective(current_buying_power, opening_buying_power),
                expected_price=float(row.close),
                fractional_shares_enabled=settings.runtime.fractional_shares_enabled,
            )
            if sizing.quantity < 1:
                notifier.notify(
                    "ORDER SKIPPED",
                    [
                        f"- symbol: {symbol}",
                        f"- expected_price: {float(row.close):.2f}",
                        f"- effective_budget: ${sizing.effective_budget:,.2f}",
                        f"- reason: {sizing.reason}",
                    ],
                    level="WARNING",
                )
                continue

            result = _submit_order_with_fallback(
                broker,
                symbol=symbol,
                side="BUY",
                quantity=sizing.quantity,
                expected_price=float(row.close),
                settings=settings,
                notifier=notifier,
            )
            if int(result.get("status_code") or 500) != 200:
                notifier.notify(
                    "ORDER FAILED",
                    [
                        f"- symbol: {symbol}",
                        f"- qty: {sizing.quantity}",
                        f"- expected_price: {float(row.close):.2f}",
                        f"- slot_id: {slot.slot_id}",
                    ],
                    level="ERROR",
                )
                continue

            reserved_buying_power = float(sizing.quantity) * float(row.close)
            slot_manager.reserve_for_buy(slot_id=slot.slot_id, symbol=symbol, client_order_id=result["client_order_id"], quantity=sizing.quantity, reserved_buying_power=reserved_buying_power, side="BUY")
            payload = {
                "slot_id": slot.slot_id,
                "score": float(row.score),
                "threshold": float(threshold),
                "signal_reason": _summarize_signal_reason(pd.Series(row._asdict())),
                "reserved_buying_power": reserved_buying_power,
                "order_type": result.get("order_type", "MARKET"),
                "limit_price": result.get("limit_price"),
            }
            order_row = {
                "client_order_id": result["client_order_id"],
                "session_date": str(today.date()),
                "symbol": symbol,
                "side": "BUY",
                "quantity": sizing.quantity,
                "filled_quantity": 0,
                "requested_price": float(row.close),
                "status": "SUBMITTED",
                "broker_order_id": None,
                "placed_at": pd.Timestamp.utcnow().isoformat(),
                "updated_at": pd.Timestamp.utcnow().isoformat(),
                "payload_json": json.dumps(payload, ensure_ascii=True, default=str),
            }
            store.upsert_live_order(order_row)
            store.replace_slot_states(slot_manager.to_frame(today), today)
            notifier.notify(
                "BUY ORDER SUBMITTED",
                _build_order_submitted_lines(
                    symbol=symbol,
                    quantity=sizing.quantity,
                    expected_price=float(row.close),
                    score=float(row.score),
                    threshold=float(threshold),
                    slot_id=slot.slot_id,
                    signal_reason=str(payload["signal_reason"]),
                ),
            )
            if broker.is_demo:
                _simulate_demo_fill(store, slot_manager, session_date=today, order_row=order_row, fill_price=float(row.close), notifier=notifier)
                store.replace_slot_states(slot_manager.to_frame(today), today)

        time.sleep(settings.runtime.quote_poll_seconds)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_live_trader()


if __name__ == "__main__":
    main()
