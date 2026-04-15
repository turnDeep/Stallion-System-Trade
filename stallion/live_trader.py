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


TAX_RATE = 0.20315  # 所得税15.315% + 住民税5%


def _notify_tax_if_profitable(
    notifier: DiscordNotifier,
    *,
    symbol: str,
    entry_price: float,
    exit_price: float,
    shares: int,
    reason: str,
) -> None:
    """利益が出た場合のみ、税金換金アラートをDiscordへ送信する。"""
    profit_usd = (exit_price - entry_price) * shares
    if profit_usd <= 0:
        return  # 損失・プラマイゼロ → 通知不要
    tax_usd = profit_usd * TAX_RATE
    notifier.notify(
        "⚠️ 税金換金アラート",
        [
            f"- symbol:     {symbol}",
            f"- reason:     {reason}",
            f"- shares:     {shares} 株",
            f"- entry:      ${entry_price:.2f}",
            f"- exit:       ${exit_price:.2f}",
            f"- 利益:        ${profit_usd:.2f} USD",
            f"- 税金概算:    ${tax_usd:.2f} USD (20.315%)",
            "- 👉 Webull アプリで上記金額を USD → JPY に換金してください",
        ],
        level="WARNING",
    )


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
    """shortlist ∪ extra_symbols (open positions) で監視対象を構築する。"""
    shortlist = store.load_shortlist(session_date)
    if shortlist.empty:
        shortlist = store.load_shortlist()
    shortlist = shortlist.head(settings.runtime.monitor_count).copy()
    base = set(shortlist["symbol"].dropna().astype(str).str.upper().tolist())

    # 持ち越し建玉を必ず含める（shortlistに入っていなくても EOD exit が正しく動くよう）
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
        entry_source=str(payload.get("entry_source", "standard_breakout")),
        entry_lane=str(payload.get("entry_lane", "none")),
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
                        "entry_source": state.entry_source,
                        "entry_lane": state.entry_lane,
                    },
                    ensure_ascii=True,
                ),
                "updated_at": pd.Timestamp.utcnow().isoformat(),
            }
        )
    _replace_position_rows(store, remaining)


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
                _notify_tax_if_profitable(
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
        action = evaluate_exit_action(state, pd.Series(latest._asdict()), cfg=cfg)
        payload = _payload_dict(row.get("payload_json"))
        payload["pending_dma21_grace"] = bool(action.get("pending_dma21_grace", False))

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
                    _notify_tax_if_profitable(
                        notifier,
                        symbol=state.symbol,
                        entry_price=state.entry_price,
                        exit_price=exit_price_reduce,
                        shares=shares_to_sell,
                        reason=str(action.get("reason", "reduce")),
                    )
                state.shares = target_remaining
            # アクションの結果として state が変わった場合（partial_profit_takenなど）も常に更新をかける
            state.pending_dma21_grace = bool(action.get("pending_dma21_grace", False))
            state.partial_profit_taken = bool(action.get("partial_profit_taken", state.partial_profit_taken))
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
                _notify_tax_if_profitable(
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
    # 持ち越し建玉を shortlist に優先マージして監視対象を確定する
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
        store.write_heartbeat("live_trader", "running", {"now_ny": now_ny.isoformat(), "strategy": "qullamaggie_breakout"})

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
        if len(open_positions) >= cfg.max_positions:
            time.sleep(settings.runtime.quote_poll_seconds)
            continue

        daily_bars = store.load_bars("1d", symbols=symbols)
        intraday_bars = store.load_bars("5m", symbols=symbols)
        signal_report = build_breakout_signal_report(daily_bars, intraday_bars, cfg=cfg)
        selected = select_breakout_candidates(signal_report, session_date=today, max_positions=cfg.max_positions)
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

        current_buying_power = broker.get_account_buying_power()
        held_symbols = set(open_positions["symbol"].tolist())
        for row in selected.itertuples(index=False):
            if len(held_symbols) >= cfg.max_positions:
                break
            if row.symbol in held_symbols:
                continue

            state = build_position_state_from_signal(row, equity=opening_buying_power, cash=current_buying_power, cfg=cfg)
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
                "entry_source": getattr(row, "entry_source", "standard_breakout"),
                "entry_lane": getattr(row, "entry_lane", "none"),
                "priority_score_within_source": getattr(row, "priority_score_within_source", None),
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

        time.sleep(settings.runtime.quote_poll_seconds)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_live_trader()


if __name__ == "__main__":
    main()
