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
from .breakout_bridge import (
    BreakoutConfig,
    BreakoutPositionState,
    build_breakout_signal_report,
    build_position_state_from_signal,
    evaluate_exit_action,
    prepare_exit_daily_frame,
    select_breakout_candidates,
)
from .config import Settings, load_settings
from .discord_notifier import DiscordNotifier
from .fmp import FMPClient
from .notifier import emit_alert
from .storage import SQLiteParquetStore
from signals.industry_priority import add_industry_composite_priority, choose_replacement_index, is_a_plus_candidate


LOGGER = logging.getLogger(__name__)


def _ny_now(settings: Settings) -> datetime:
    return datetime.now(pytz.timezone(settings.runtime.market_timezone))


def _today_ny(settings: Settings) -> pd.Timestamp:
    return pd.Timestamp(_ny_now(settings).date())


def _within_signal_window(now_ny: datetime, settings: Settings) -> bool:
    minutes_from_open = (now_ny.hour * 60 + now_ny.minute) - (9 * 60 + 30)
    return settings.runtime.min_minutes_from_open <= minutes_from_open <= settings.runtime.max_minutes_from_open


def _after_time(now_ny: datetime, hour: int, minute: int) -> bool:
    return (now_ny.hour, now_ny.minute) >= (hour, minute)


def _payload_dict(raw_payload: object) -> dict[str, Any]:
    if isinstance(raw_payload, dict):
        return dict(raw_payload)
    try:
        return json.loads(raw_payload or "{}")
    except Exception:
        return {}


TAX_RATE = 0.20315  # Japan capital gains tax: income tax 15.315% + local tax 5%.
TAX_RESERVE_STATE_KEY = "tax_reserve_usd"


def _load_tax_reserve_state(store: SQLiteParquetStore) -> dict[str, Any]:
    raw = store.get_system_state(TAX_RESERVE_STATE_KEY)
    if not raw:
        return {"reserved_tax_usd": 0.0, "realized_profit_usd": 0.0, "events": 0}
    try:
        payload = json.loads(raw)
        return {
            "reserved_tax_usd": float(payload.get("reserved_tax_usd", 0.0)),
            "realized_profit_usd": float(payload.get("realized_profit_usd", 0.0)),
            "events": int(payload.get("events", 0)),
        }
    except Exception:
        # Backward compatibility if the state was ever stored as a plain number.
        try:
            return {"reserved_tax_usd": float(raw), "realized_profit_usd": 0.0, "events": 0}
        except Exception:
            return {"reserved_tax_usd": 0.0, "realized_profit_usd": 0.0, "events": 0}


def _save_tax_reserve_state(store: SQLiteParquetStore, state: dict[str, Any]) -> None:
    payload = {
        "reserved_tax_usd": round(float(state.get("reserved_tax_usd", 0.0)), 2),
        "realized_profit_usd": round(float(state.get("realized_profit_usd", 0.0)), 2),
        "events": int(state.get("events", 0)),
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }
    store.put_system_state(TAX_RESERVE_STATE_KEY, json.dumps(payload, ensure_ascii=True, default=str))


def _reserved_tax_usd(store: SQLiteParquetStore) -> float:
    return max(0.0, float(_load_tax_reserve_state(store).get("reserved_tax_usd", 0.0)))


def _tax_adjusted_buying_power(store: SQLiteParquetStore, raw_buying_power: float) -> float:
    return max(0.0, float(raw_buying_power) - _reserved_tax_usd(store))


def _reserve_tax_if_profitable(
    store: SQLiteParquetStore,
    notifier: DiscordNotifier,
    *,
    symbol: str,
    entry_price: float,
    exit_price: float,
    shares: int,
    reason: str,
) -> float:
    """Reserve estimated Japan tax on profitable sells and notify Discord."""
    profit_usd = (exit_price - entry_price) * shares
    if profit_usd <= 0:
        return 0.0
    tax_usd = profit_usd * TAX_RATE
    state = _load_tax_reserve_state(store)
    state["reserved_tax_usd"] = float(state.get("reserved_tax_usd", 0.0)) + tax_usd
    state["realized_profit_usd"] = float(state.get("realized_profit_usd", 0.0)) + profit_usd
    state["events"] = int(state.get("events", 0)) + 1
    _save_tax_reserve_state(store, state)

    total_reserved_tax = float(state["reserved_tax_usd"])
    notifier.notify(
        "Tax Reserve Alert",
        [
            f"Profit of ${profit_usd:.2f} USD was realized, so ${tax_usd:.2f} USD was reserved for tax.",
            f"- symbol: {symbol}",
            f"- reason: {reason}",
            f"- shares: {shares}",
            f"- entry: ${entry_price:.2f}",
            f"- exit: ${exit_price:.2f}",
            f"- tax_rate: {TAX_RATE:.3%}",
            f"- total_reserved_tax_usd: ${total_reserved_tax:.2f}",
            "- Webull charges Japanese tax in JPY; manually convert the reserved USD to JPY when needed.",
        ],
        level="WARNING",
    )
    return float(tax_usd)


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


def _load_monitor_symbols(
    store: SQLiteParquetStore,
    settings: Settings,
    session_date: pd.Timestamp,
    *,
    extra_symbols: list[str] | None = None,
) -> list[str]:
    """Build monitor symbols from the shortlist plus currently open positions."""
    shortlist = store.load_shortlist(session_date)
    if shortlist.empty:
        shortlist = store.load_shortlist()
    shortlist = shortlist.head(settings.runtime.monitor_count).copy()
    base = set(shortlist["symbol"].dropna().astype(str).str.upper().tolist())

    # Always monitor open positions so hard stops and EOD exits still run.
    if extra_symbols:
        for sym in extra_symbols:
            sym_upper = str(sym).strip().upper()
            if sym_upper:
                base.add(sym_upper)

    symbols = sorted(base)
    if not symbols:
        raise RuntimeError("No shortlist symbols available. Run the nightly pipeline first.")
    return symbols


def _load_or_fetch_opening_buying_power(store: SQLiteParquetStore, broker, session_date: pd.Timestamp) -> float:
    state_key = f"opening_buying_power:{pd.Timestamp(session_date).date()}"
    cached = store.get_system_state(state_key)
    if cached:
        return float(cached)
    opening_buying_power = broker.get_account_buying_power()
    store.put_system_state(state_key, str(opening_buying_power))
    return float(opening_buying_power)


def _open_positions_frame(store: SQLiteParquetStore) -> pd.DataFrame:
    positions = store.load_open_positions()
    if positions.empty:
        return pd.DataFrame(columns=["symbol", "quantity", "payload_json"])
    positions["symbol"] = positions["symbol"].astype(str).str.upper()
    return positions


def _position_state_from_row(row: dict[str, Any]) -> BreakoutPositionState | None:
    payload = _payload_dict(row.get("payload_json"))
    try:
        entry_price = float(payload["entry_price"])
        initial_stop = float(payload["initial_stop"])
        pivot_level = float(payload["pivot_level"])
        breakout_day_low = float(payload["breakout_day_low"])
        initial_shares = int(payload["initial_shares"])
    except Exception:
        return None
    entry_bar_time = pd.to_datetime(payload.get("trigger_time"), errors="coerce")
    return BreakoutPositionState(
        symbol=str(row.get("symbol") or "").upper(),
        entry_date=pd.to_datetime(payload.get("entry_date")),
        entry_price=entry_price,
        initial_stop=initial_stop,
        pivot_level=pivot_level,
        breakout_day_low=breakout_day_low,
        initial_shares=initial_shares,
        shares=int(row.get("quantity") or 0),
        initial_risk_per_share=max(entry_price - initial_stop, 0.0),
        entry_bar_time=None if pd.isna(entry_bar_time) else entry_bar_time,
        pending_dma21_grace=bool(payload.get("pending_dma21_grace", False)),
        partial_profit_taken=bool(payload.get("partial_profit_taken", False)),
        reduced_on_dma21=bool(payload.get("reduced_on_dma21", False)),
        entry_source=str(payload.get("entry_source", "standard_breakout")),
        entry_lane=str(payload.get("entry_lane", "none")),
        same_day_priority_score=pd.to_numeric(payload.get("same_day_priority_score"), errors="coerce"),
        industry_a_plus_candidate=bool(payload.get("industry_a_plus_candidate", False)),
    )


def _replace_position_rows(store: SQLiteParquetStore, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    store.replace_open_positions(frame)


def _upsert_demo_position(
    store: SQLiteParquetStore,
    *,
    state: BreakoutPositionState,
    session_date: pd.Timestamp,
) -> None:
    positions = _open_positions_frame(store)
    remaining = positions.loc[positions["symbol"].ne(state.symbol)].to_dict(orient="records")
    if state.shares > 0:
        remaining.append(
            {
                "symbol": state.symbol,
                "session_date": str(pd.Timestamp(session_date).date()),
                "quantity": state.shares,
                "avg_price": state.entry_price,
                "entry_time": pd.Timestamp.utcnow().isoformat(),
                "broker_order_id": None,
                "status": "OPEN",
                "payload_json": json.dumps(
                    {
                        "strategy_type": "qullamaggie_breakout",
                        "entry_date": str(state.entry_date.date()),
                        "entry_price": state.entry_price,
                        "initial_stop": state.initial_stop,
                        "pivot_level": state.pivot_level,
                        "breakout_day_low": state.breakout_day_low,
                        "initial_shares": state.initial_shares,
                        "trigger_time": None if state.entry_bar_time is None else str(state.entry_bar_time),
                        "pending_dma21_grace": state.pending_dma21_grace,
                        "partial_profit_taken": state.partial_profit_taken,
                        "reduced_on_dma21": state.reduced_on_dma21,
                        "entry_source": state.entry_source,
                        "entry_lane": state.entry_lane,
                        "same_day_priority_score": state.same_day_priority_score,
                        "industry_a_plus_candidate": state.industry_a_plus_candidate,
                    },
                    ensure_ascii=True,
                ),
                "updated_at": pd.Timestamp.utcnow().isoformat(),
            }
        )
    _replace_position_rows(store, remaining)


def _priority_position_summaries(
    positions: pd.DataFrame,
    latest_quotes: pd.DataFrame,
) -> list[dict[str, Any]]:
    if positions.empty:
        return []
    quote_map = {}
    if latest_quotes is not None and not latest_quotes.empty:
        quote_map = {
            str(row.symbol).upper(): float(row.price)
            for row in latest_quotes.itertuples(index=False)
            if pd.notna(getattr(row, "price", None))
        }

    summaries: list[dict[str, Any]] = []
    for idx, row in enumerate(positions.to_dict(orient="records")):
        payload = _payload_dict(row.get("payload_json"))
        symbol = str(row.get("symbol") or "").upper()
        entry_price = float(payload.get("entry_price") or row.get("avg_price") or 0.0)
        last_price = quote_map.get(symbol, entry_price)
        current_gain = (last_price / entry_price - 1.0) if entry_price > 0 else 0.0
        summaries.append(
            {
                "index": idx,
                "symbol": symbol,
                "row": row,
                "quantity": int(row.get("quantity") or 0),
                "last_price": float(last_price),
                "current_gain": float(current_gain),
                "priority_score": float(payload.get("same_day_priority_score") or payload.get("priority_score_within_source") or 0.0),
                "a_plus_candidate": bool(payload.get("industry_a_plus_candidate", False)),
            }
        )
    return summaries


def _submit_order(
    store: SQLiteParquetStore,
    broker,
    notifier: DiscordNotifier,
    *,
    session_date: pd.Timestamp,
    symbol: str,
    side: str,
    quantity: int,
    price_hint: float,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    result = broker.place_market_order(symbol=symbol, side=side, quantity=quantity)
    if int(result.get("status_code") or 500) != 200:
        emit_alert(store, level="ERROR", component="live_trader", message=f"{side} order failed for {symbol}", payload=result, discord=notifier)
        return None

    order_row = {
        "client_order_id": result["client_order_id"],
        "session_date": str(pd.Timestamp(session_date).date()),
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "filled_quantity": quantity if broker.is_demo else 0,
        "requested_price": price_hint,
        "status": "FILLED" if broker.is_demo else "SUBMITTED",
        "broker_order_id": result.get("order_id"),
        "placed_at": pd.Timestamp.utcnow().isoformat(),
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "payload_json": json.dumps(payload, ensure_ascii=True, default=str),
    }
    store.upsert_live_order(order_row)
    if broker.is_demo:
        store.append_live_fill(
            {
                "fill_id": f"{result['client_order_id']}:filled",
                "session_date": str(pd.Timestamp(session_date).date()),
                "symbol": symbol,
                "side": side,
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "quantity": quantity,
                "price": price_hint,
                "payload_json": json.dumps(order_row, ensure_ascii=True, default=str),
            }
        )
    return order_row


def _evaluate_intraday_hard_stops(
    store: SQLiteParquetStore,
    broker,
    notifier: DiscordNotifier,
    *,
    session_date: pd.Timestamp,
    latest_quotes: pd.DataFrame,
) -> None:
    if latest_quotes.empty:
        return
    positions = _open_positions_frame(store)
    if positions.empty:
        return
    quote_map = {
        str(row.symbol).upper(): float(row.price)
        for row in latest_quotes.itertuples(index=False)
        if pd.notna(row.price)
    }
    updated_rows: list[dict[str, Any]] = []
    for row in positions.to_dict(orient="records"):
        state = _position_state_from_row(row)
        if state is None or state.shares <= 0:
            continue
        last_price = quote_map.get(state.symbol)
        if last_price is not None and last_price <= state.initial_stop:
            payload = _payload_dict(row.get("payload_json"))
            payload["exit_reason"] = "intraday_hard_stop"
            submitted = _submit_order(
                store,
                broker,
                notifier,
                session_date=session_date,
                symbol=state.symbol,
                side="SELL",
                quantity=state.shares,
                price_hint=state.initial_stop,
                payload=payload,
            )
            if submitted is not None:
                notifier.notify("SELL ORDER SUBMITTED", [f"- symbol: {state.symbol}", f"- qty: {state.shares}", "- reason: intraday_hard_stop"])
                _reserve_tax_if_profitable(
                    store,
                    notifier,
                    symbol=state.symbol,
                    entry_price=state.entry_price,
                    exit_price=state.initial_stop,
                    shares=state.shares,
                    reason="intraday_hard_stop",
                )
            continue
        updated_rows.append(row)
    _replace_position_rows(store, updated_rows)


def _evaluate_end_of_day_exits(
    store: SQLiteParquetStore,
    broker,
    notifier: DiscordNotifier,
    *,
    session_date: pd.Timestamp,
    daily_bars: pd.DataFrame,
    cfg: BreakoutConfig,
) -> None:
    positions = _open_positions_frame(store)
    if positions.empty:
        return
    exit_frame = prepare_exit_daily_frame(daily_bars, session_timezone=cfg.session_timezone)
    latest_rows = exit_frame.sort_values("date").groupby("symbol", sort=False).tail(1)
    latest_map = {str(row.symbol).upper(): row for row in latest_rows.itertuples(index=False)}

    survivors: list[dict[str, Any]] = []
    for row in positions.to_dict(orient="records"):
        state = _position_state_from_row(row)
        if state is None or state.shares <= 0:
            continue
        latest = latest_map.get(state.symbol)
        if latest is None:
            survivors.append(row)
            continue
        action = evaluate_exit_action(state, row, cfg=cfg)
        payload = _payload_dict(row.get("payload_json"))
        payload["pending_dma21_grace"] = bool(action.get("pending_dma21_grace", False))
        payload["partial_profit_taken"] = bool(action.get("partial_profit_taken", state.partial_profit_taken))
        payload["reduced_on_dma21"] = bool(action.get("reduced_on_dma21", state.reduced_on_dma21))

        if action.get("action") == "reduce":
            target_remaining = int(action.get("target_remaining_shares", state.shares))
            shares_to_sell = max(0, state.shares - target_remaining)
            if shares_to_sell > 0:
                exit_price_reduce = float(action.get("price", latest.close))
                submitted_reduce = _submit_order(
                    store,
                    broker,
                    notifier,
                    session_date=session_date,
                    symbol=state.symbol,
                    side="SELL",
                    quantity=shares_to_sell,
                    price_hint=exit_price_reduce,
                    payload={**payload, "exit_reason": action.get("reason")},
                )
                if submitted_reduce is not None:
                    _reserve_tax_if_profitable(
                        store,
                        notifier,
                        symbol=state.symbol,
                        entry_price=state.entry_price,
                        exit_price=exit_price_reduce,
                        shares=shares_to_sell,
                        reason=str(action.get("reason", "reduce")),
                    )
                state.shares = target_remaining
            state.pending_dma21_grace = bool(action.get("pending_dma21_grace", False))
            state.partial_profit_taken = bool(action.get("partial_profit_taken", state.partial_profit_taken))
            state.reduced_on_dma21 = bool(action.get("reduced_on_dma21", state.reduced_on_dma21))
            _upsert_demo_position(store, state=state, session_date=session_date)
            continue

        if action.get("action") == "exit_all":
            exit_price_all = float(action.get("price", latest.close))
            submitted_all = _submit_order(
                store,
                broker,
                notifier,
                session_date=session_date,
                symbol=state.symbol,
                side="SELL",
                quantity=state.shares,
                price_hint=exit_price_all,
                payload={**payload, "exit_reason": action.get("reason")},
            )
            if submitted_all is not None:
                _reserve_tax_if_profitable(
                    store,
                    notifier,
                    symbol=state.symbol,
                    entry_price=state.entry_price,
                    exit_price=exit_price_all,
                    shares=state.shares,
                    reason=str(action.get("reason", "exit_all")),
                )
            continue

        payload_json = json.dumps(payload, ensure_ascii=True, default=str)
        row["payload_json"] = payload_json
        survivors.append(row)

    _replace_position_rows(store, survivors)


def run_live_trader(settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    store = SQLiteParquetStore(settings)
    notifier = DiscordNotifier(settings, store)
    broker = create_broker(settings)
    fmp = FMPClient(settings)
    cfg = BreakoutConfig.from_settings(settings)

    notifier.notify("BOT STARTUP", [f"- mode: {settings.trade_mode}", "- strategy: qullamaggie_breakout"])

    today = _today_ny(settings)
    # 謖√■雜翫＠蟒ｺ邇峨ｒ shortlist 縺ｫ蜆ｪ蜈医・繝ｼ繧ｸ縺励※逶｣隕門ｯｾ雎｡繧堤｢ｺ螳壹☆繧・
    open_positions_at_start = _open_positions_frame(store)
    carried_symbols = (
        open_positions_at_start["symbol"].dropna().astype(str).str.upper().tolist()
        if not open_positions_at_start.empty
        else []
    )
    symbols = _load_monitor_symbols(store, settings, today, extra_symbols=carried_symbols)
    if carried_symbols:
        LOGGER.info(
            "Monitor symbols: %d total (shortlist + %d carried open positions: %s)",
            len(symbols),
            len(carried_symbols),
            carried_symbols,
        )
    else:
        LOGGER.info("Monitor symbols: %d (shortlist only, no carried positions)", len(symbols))
    aggregator = QuoteBarAggregator(session_timezone=settings.runtime.market_timezone)
    opening_buying_power = None
    eod_exit_done = False  # ensure EOD exits run only once per session

    while True:
        now_ny = _ny_now(settings)
        now_utc = pd.Timestamp.now(tz="UTC")
        store.write_heartbeat(
            "live_trader",
            "running",
            {
                "now_ny": now_ny.isoformat(),
                "strategy": "qullamaggie_breakout",
                "reserved_tax_usd": _reserved_tax_usd(store),
            },
        )

        if now_ny.weekday() >= 5:
            notifier.flush()
            return

        if now_ny.hour < 9 or (now_ny.hour == 9 and now_ny.minute < 30):
            time.sleep(15)
            continue

        if opening_buying_power is None:
            opening_buying_power = _load_or_fetch_opening_buying_power(store, broker, today)

        quotes = []
        for start in range(0, len(symbols), settings.runtime.batch_quote_chunk_size):
            chunk = symbols[start : start + settings.runtime.batch_quote_chunk_size]
            frame = fmp.fetch_batch_quotes(chunk)
            if not frame.empty:
                quotes.append(frame)
            time.sleep(0.1)
        quote_frame = pd.concat(quotes, ignore_index=True) if quotes else pd.DataFrame()
        snapshot_frame = _build_quote_snapshot_frame(quote_frame, now_utc)
        if not snapshot_frame.empty:
            store.append_quote_snapshots(snapshot_frame)
            finalized = aggregator.ingest_quotes(snapshot_frame[["symbol", "price", "cumulative_volume"]], observed_at_utc=now_utc)
            flushed = aggregator.flush_completed(now_utc - pd.Timedelta(seconds=1))
            finalized = pd.concat([finalized, flushed], ignore_index=True) if not flushed.empty else finalized
            if not finalized.empty:
                store.save_bars(finalized, timeframe="5m")
        else:
            finalized = pd.DataFrame()

        _evaluate_intraday_hard_stops(store, broker, notifier, session_date=today, latest_quotes=quote_frame)

        # ---------------------------------------------------------------
        # EOD Exit at 15:55 (5 min before close) using live quote prices
        # ---------------------------------------------------------------
        if (
            not eod_exit_done
            and _after_time(now_ny, settings.runtime.eod_exit_hour, settings.runtime.eod_exit_minute)
        ):
            LOGGER.info("EOD EXIT triggered at %s (using live quote prices as proxy close)", now_ny.strftime("%H:%M"))
            daily_bars_for_exit = store.load_bars("1d", symbols=symbols)
            if not quote_frame.empty and not daily_bars_for_exit.empty:
                # Patch today's close in daily_bars with the current live quote price
                today_str = str(today.date())
                quote_price_map = {
                    str(row.symbol).upper(): float(row.price)
                    for row in quote_frame.itertuples(index=False)
                    if pd.notna(getattr(row, "price", None))
                }
                daily_bars_for_exit["ts"] = pd.to_datetime(daily_bars_for_exit["ts"], utc=True, errors="coerce")
                today_mask = daily_bars_for_exit["ts"].dt.normalize().dt.tz_localize(None).dt.strftime("%Y-%m-%d") == today_str
                for sym, live_price in quote_price_map.items():
                    sym_mask = daily_bars_for_exit["symbol"].eq(sym)
                    daily_bars_for_exit.loc[sym_mask & today_mask, "close"] = live_price
                LOGGER.info(
                    "Patched today's close for %d symbols using live quote prices for EOD exit evaluation.",
                    len(quote_price_map),
                )
            _evaluate_end_of_day_exits(
                store, broker, notifier,
                session_date=today,
                daily_bars=daily_bars_for_exit,
                cfg=cfg,
            )
            eod_exit_done = True
            notifier.notify(
                "EOD EXIT SCAN COMPLETE",
                [f"- time: {now_ny.strftime('%H:%M')} NY", "- using: live quote prices as proxy close"],
            )

        if _after_time(now_ny, settings.runtime.shutdown_hour, settings.runtime.shutdown_minute):
            notifier.flush()
            return

        if not _within_signal_window(now_ny, settings) or _after_time(now_ny, settings.runtime.no_new_orders_after_hour, settings.runtime.no_new_orders_after_minute):
            time.sleep(settings.runtime.quote_poll_seconds)
            continue

        if finalized.empty:
            time.sleep(settings.runtime.quote_poll_seconds)
            continue

        open_positions = _open_positions_frame(store)
        daily_bars = store.load_bars("1d", symbols=symbols)
        intraday_bars = store.load_bars("5m", symbols=symbols)
        signal_report = build_breakout_signal_report(daily_bars, intraday_bars, cfg=cfg)
        if cfg.use_industry_composite_priority:
            try:
                full_daily_for_priority = store.load_bars("1d")
                signal_report = add_industry_composite_priority(signal_report, full_daily_for_priority, store.load_universe())
            except Exception as exc:
                LOGGER.warning("Industry composite priority unavailable; using base ranking. error=%s", exc)
        selection_limit = cfg.max_positions
        if cfg.enable_a_plus_replacement:
            selection_limit = max(cfg.max_positions * 3, cfg.max_positions)
        selected = select_breakout_candidates(signal_report, session_date=today, max_positions=selection_limit)
        if selected.empty:
            time.sleep(settings.runtime.quote_poll_seconds)
            continue

        exit_frame = prepare_exit_daily_frame(daily_bars, session_timezone=cfg.session_timezone)
        latest_exit = exit_frame.sort_values("date").groupby("symbol", sort=False).tail(1)
        selected = selected.merge(
            latest_exit[["symbol", "adr20_pct", "dma10", "dma21", "hold_score", "tight_low_volume_day"]],
            on="symbol",
            how="left",
        )
        selected["payload_json"] = selected.apply(lambda row: json.dumps(row.to_dict(), ensure_ascii=True, default=str), axis=1)
        store.append_live_signals(
            selected.assign(
                session_date=str(today.date()),
                timestamp=selected["trigger_time"].astype(str),
                score=selected["leader_score"],
                threshold=0.0,
                selected=1,
            )[["session_date", "timestamp", "symbol", "score", "threshold", "selected", "payload_json"]]
        )

        raw_buying_power = broker.get_account_buying_power()
        current_buying_power = _tax_adjusted_buying_power(store, raw_buying_power)
        effective_opening_buying_power = _tax_adjusted_buying_power(store, float(opening_buying_power))
        held_symbols = set(open_positions["symbol"].tolist())
        for row in selected.itertuples(index=False):
            if row.symbol in held_symbols:
                continue

            desired_slot_cash = float(effective_opening_buying_power) * cfg.max_alloc_pct_per_trade
            precheck_state = build_position_state_from_signal(
                row,
                equity=effective_opening_buying_power,
                cash=max(float(current_buying_power), desired_slot_cash),
                cfg=cfg,
            )
            if precheck_state is None:
                continue
            needs_replacement = len(held_symbols) >= cfg.max_positions or current_buying_power < desired_slot_cash
            if needs_replacement and cfg.enable_a_plus_replacement:
                position_summaries = _priority_position_summaries(_open_positions_frame(store), quote_frame)
                replace_idx = choose_replacement_index(position_summaries, row)
                if replace_idx is None:
                    continue
                replacement = position_summaries[replace_idx]
                if replacement["quantity"] <= 0:
                    continue
                replacement_payload = _payload_dict(replacement["row"].get("payload_json"))
                replacement_payload["exit_reason"] = "priority_replacement_exit"
                submitted_replacement = _submit_order(
                    store,
                    broker,
                    notifier,
                    session_date=today,
                    symbol=replacement["symbol"],
                    side="SELL",
                    quantity=int(replacement["quantity"]),
                    price_hint=float(replacement["last_price"]),
                    payload=replacement_payload,
                )
                if submitted_replacement is None:
                    continue
                notifier.notify(
                    "PRIORITY REPLACEMENT SELL",
                    [
                        f"- sold: {replacement['symbol']}",
                        f"- incoming: {row.symbol}",
                        f"- gain: {replacement['current_gain']:.2%}",
                        f"- reason: A+ industry composite candidate",
                    ],
                    level="WARNING",
                )
                reserved_tax = _reserve_tax_if_profitable(
                    store,
                    notifier,
                    symbol=replacement["symbol"],
                    entry_price=float(replacement_payload.get("entry_price") or replacement["last_price"]),
                    exit_price=float(replacement["last_price"]),
                    shares=int(replacement["quantity"]),
                    reason="priority_replacement_exit",
                )
                remaining_rows = [
                    p["row"]
                    for idx, p in enumerate(position_summaries)
                    if idx != replace_idx
                ]
                _replace_position_rows(store, remaining_rows)
                held_symbols = {str(p.get("symbol") or "").upper() for p in remaining_rows}
                current_buying_power += max(
                    0.0,
                    float(replacement["last_price"]) * int(replacement["quantity"]) - reserved_tax,
                )
            elif len(held_symbols) >= cfg.max_positions:
                continue

            state = build_position_state_from_signal(row, equity=effective_opening_buying_power, cash=current_buying_power, cfg=cfg)
            if state is None:
                continue

            payload = {
                "strategy_type": "qullamaggie_breakout",
                "entry_date": str(state.entry_date.date()),
                "entry_price": state.entry_price,
                "initial_stop": state.initial_stop,
                "pivot_level": state.pivot_level,
                "breakout_day_low": state.breakout_day_low,
                "initial_shares": state.initial_shares,
                "trigger_time": None if state.entry_bar_time is None else str(state.entry_bar_time),
                "leader_score": getattr(row, "leader_score", None),
                "setup_score_pre": getattr(row, "setup_score_pre", None),
                "trigger_score": getattr(row, "trigger_score", None),
                "cum_vol_ratio_at_trigger": getattr(row, "cum_vol_ratio_at_trigger", None),
                "move_from_open_at_trigger": getattr(row, "move_from_open_at_trigger", None),
                "entry_source": getattr(row, "entry_source", "standard_breakout"),
                "entry_lane": getattr(row, "entry_lane", "none"),
                "priority_score_within_source": getattr(row, "priority_score_within_source", None),
                "same_day_priority_score": getattr(row, "same_day_priority_score", None),
                "prior_runup_pre": getattr(row, "prior_runup_pre", None),
                "industry_a_plus_candidate": bool(is_a_plus_candidate(row)),
                "priority_a_plus_candidate": getattr(row, "priority_a_plus_candidate", None),
                "priority_leader95": getattr(row, "priority_leader95", None),
                "priority_leader98": getattr(row, "priority_leader98", None),
                "priority_volume_thrust": getattr(row, "priority_volume_thrust", None),
                "priority_trigger_score_norm": getattr(row, "priority_trigger_score_norm", None),
                "priority_prior_runup": getattr(row, "priority_prior_runup", None),
                "priority_move_thrust": getattr(row, "priority_move_thrust", None),
                "priority_industry_rs": getattr(row, "priority_industry_rs", None),
                "priority_setup_sweet": getattr(row, "priority_setup_sweet", None),
            }
            submitted = _submit_order(
                store,
                broker,
                notifier,
                session_date=today,
                symbol=state.symbol,
                side="BUY",
                quantity=state.shares,
                price_hint=state.entry_price,
                payload=payload,
            )
            if submitted is None:
                continue

            notifier.notify(
                "BUY ORDER SUBMITTED",
                [
                    f"- symbol: {state.symbol}",
                    f"- qty: {state.shares}",
                    f"- entry_price: {state.entry_price:.2f}",
                    f"- leader_score: {getattr(row, 'leader_score', float('nan')):.2f}",
                ],
            )
            if broker.is_demo:
                _upsert_demo_position(store, state=state, session_date=today)
            held_symbols.add(state.symbol)
            current_buying_power = max(0.0, current_buying_power - state.shares * state.entry_price)

        time.sleep(settings.runtime.quote_poll_seconds)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_live_trader()


if __name__ == "__main__":
    main()
