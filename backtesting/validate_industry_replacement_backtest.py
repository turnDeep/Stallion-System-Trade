#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signals.industry_priority import add_industry_composite_priority, choose_replacement_index, sort_by_industry_priority
from backtesting.qullamaggie_breakout_backtest import prepare_daily


INITIAL_EQUITY = 100_000.0
MAX_SLOTS = 5
SLOT_ALLOC_PCT = 0.25


def _num(row: pd.Series, *cols: str, default: float = np.nan) -> float:
    for col in cols:
        if col in row.index and pd.notna(row[col]):
            value = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(value):
                return float(value)
    return default


def _equity_factor(entry_price: float, realized: float, remaining: float, mark_price: float) -> float:
    return 1.0 + realized + remaining * (mark_price / entry_price - 1.0)


def load_validation_inputs(
    signals_path: Path,
    daily_path: Path,
    universe_path: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    events = pd.read_csv(signals_path)
    events["date"] = pd.to_datetime(events["date"]).dt.normalize()
    universe = pd.read_parquet(universe_path)
    daily = pd.read_parquet(daily_path)
    if "date" not in daily.columns and "ts" in daily.columns:
        daily = daily.rename(columns={"ts": "date"})
    daily.columns = [str(c).lower() for c in daily.columns]
    daily["symbol"] = daily["symbol"].astype(str).str.upper()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()

    events = add_industry_composite_priority(events, daily, universe)
    events = sort_by_industry_priority(events)

    symbols = set(events["symbol"].astype(str).str.upper().unique())
    daily_prepared = prepare_daily(daily.loc[daily["symbol"].isin(symbols)].copy())
    daily_prepared["dma50"] = (
        daily_prepared.groupby("symbol", sort=False)["close"]
        .rolling(50, min_periods=50)
        .mean()
        .reset_index(level=0, drop=True)
    )
    daily_by_symbol = {
        symbol: sub.sort_values("date", kind="mergesort").reset_index(drop=True)
        for symbol, sub in daily_prepared.groupby("symbol", sort=False)
    }
    return events, daily_by_symbol


def simulate_super_winner_trade(
    event: pd.Series,
    daily_by_symbol: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any], pd.DataFrame]:
    symbol = str(event["symbol"]).upper()
    entry_date = pd.Timestamp(event["date"]).normalize()
    daily = daily_by_symbol.get(symbol)
    if daily is None or daily.empty:
        return {"symbol": symbol, "entry_date": entry_date, "exit_reason": "missing_daily", "trade_return_pct": np.nan}, pd.DataFrame()

    rows = daily.loc[daily["date"] >= entry_date].reset_index(drop=True)
    if rows.empty:
        return {"symbol": symbol, "entry_date": entry_date, "exit_reason": "missing_entry", "trade_return_pct": np.nan}, pd.DataFrame()

    day0 = rows.iloc[0]
    entry_price = _num(event, "trigger_close", "close", default=float(day0["close"]))
    pivot = _num(event, "effective_pivot_level", "pivot_high", "zigzag_line_value", default=float(day0["low"]))
    initial_stop = min(float(day0["low"]), pivot) * 0.999
    risk_per_share = entry_price - initial_stop
    if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(risk_per_share) or risk_per_share <= 0:
        return {"symbol": symbol, "entry_date": entry_date, "exit_reason": "invalid_entry", "trade_return_pct": np.nan}, pd.DataFrame()

    leader_score = _num(event, "leader_score", default=np.nan)
    realized = 0.0
    remaining = 1.0
    partial_taken = False
    super_winner = False
    super_winner_date = pd.NaT
    reduced_21dma = False
    consecutive_21dma_breaks = 0
    highest_close = float(day0["close"])
    exit_reason = "open_end"
    exit_date = pd.NaT
    marks: list[dict[str, Any]] = []

    def add_mark(date: pd.Timestamp, factor: float, reason: str | None = None) -> None:
        marks.append(
            {
                "date": pd.Timestamp(date).normalize(),
                "equity_factor": factor,
                "reason": reason,
                "super_winner": super_winner,
                "remaining": remaining,
            }
        )

    if float(day0["close"]) < pivot:
        ret = float(day0["close"]) / entry_price - 1.0
        add_mark(entry_date, 1.0 + ret, "day0_pivot_fail")
        return (
            {
                "symbol": symbol,
                "entry_date": entry_date,
                "exit_date": entry_date,
                "entry_source": event.get("entry_source"),
                "exit_reason": "day0_pivot_fail",
                "trade_return_pct": ret,
                "partial_taken": False,
                "super_winner": False,
                "super_winner_date": pd.NaT,
            },
            pd.DataFrame(marks),
        )

    add_mark(entry_date, _equity_factor(entry_price, realized, remaining, float(day0["close"])))

    for session_idx in range(1, len(rows)):
        row = rows.iloc[session_idx]
        cur_date = pd.Timestamp(row["date"]).normalize()
        days_since_entry = int((cur_date - entry_date).days)
        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])
        highest_close = max(highest_close, close)
        hold_score = float(row["hold_score"]) if pd.notna(row.get("hold_score", np.nan)) else np.nan
        dma10 = float(row["dma10"]) if pd.notna(row.get("dma10", np.nan)) else np.nan
        dma21 = float(row["dma21"]) if pd.notna(row.get("dma21", np.nan)) else np.nan
        dma50 = float(row["dma50"]) if pd.notna(row.get("dma50", np.nan)) else np.nan
        atr20 = float(row["atr20"]) if pd.notna(row.get("atr20", np.nan)) else np.nan

        if low <= initial_stop and remaining > 0:
            realized += remaining * (initial_stop / entry_price - 1.0)
            remaining = 0.0
            exit_reason = "hard_stop_lod"
            exit_date = cur_date
            add_mark(cur_date, 1.0 + realized, exit_reason)
            break

        if days_since_entry <= 1 and close < pivot and remaining > 0:
            realized += remaining * (close / entry_price - 1.0)
            remaining = 0.0
            exit_reason = "pivot_fail_exit_all"
            exit_date = cur_date
            add_mark(cur_date, 1.0 + realized, exit_reason)
            break

        if not partial_taken and remaining > 0:
            tp_price = min(entry_price * 1.10, entry_price + 1.75 * risk_per_share)
            if high >= tp_price:
                sell_frac = min(remaining, 0.33)
                realized += sell_frac * (tp_price / entry_price - 1.0)
                remaining -= sell_frac
                partial_taken = True

        close_gain = close / entry_price - 1.0
        peak_gain = highest_close / entry_price - 1.0
        leader_fast_track = pd.notna(leader_score) and leader_score >= 98.0 and close_gain >= 0.35
        if (
            not super_winner
            and remaining > 0
            and (
                peak_gain >= 1.0
                or (close_gain >= 0.50 and pd.notna(dma21) and close > dma21)
                or (leader_fast_track and pd.notna(dma21) and close > dma21)
            )
        ):
            super_winner = True
            super_winner_date = cur_date

        dma_allowed = session_idx >= 10

        if super_winner and dma_allowed and remaining > 0:
            if pd.notna(dma21) and close < dma21:
                consecutive_21dma_breaks += 1
            else:
                consecutive_21dma_breaks = 0

            if consecutive_21dma_breaks >= 2 and pd.notna(hold_score) and hold_score < 40.0 and not reduced_21dma:
                target = 0.50
                sell_frac = max(0.0, remaining - target)
                realized += sell_frac * (close / entry_price - 1.0)
                remaining -= sell_frac
                reduced_21dma = True

            active_atr_mult = 3.0
            if peak_gain >= 2.0:
                active_atr_mult = 2.0
            if peak_gain >= 3.0:
                active_atr_mult = 1.5

            hit_50dma = pd.notna(dma50) and close < dma50
            hit_atr = pd.notna(atr20) and close < highest_close - active_atr_mult * atr20
            if hit_50dma or hit_atr:
                realized += remaining * (close / entry_price - 1.0)
                remaining = 0.0
                exit_reason = "super_winner_50dma" if hit_50dma else f"super_winner_{active_atr_mult:g}atr"
                exit_date = cur_date
                add_mark(cur_date, 1.0 + realized, exit_reason)
                break

            add_mark(cur_date, _equity_factor(entry_price, realized, remaining, close))
            continue

        if dma_allowed and remaining > 0:
            if pd.notna(dma21) and close < dma21:
                if pd.notna(hold_score) and hold_score < 45.0:
                    target = 0.15
                    sell_frac = max(0.0, remaining - target)
                    realized += sell_frac * (close / entry_price - 1.0)
                    remaining -= sell_frac
                    reduced_21dma = True
                elif reduced_21dma and pd.notna(hold_score) and hold_score < 40.0:
                    realized += remaining * (close / entry_price - 1.0)
                    remaining = 0.0
                    exit_reason = "dma21_exit_after_reduce"
                    exit_date = cur_date
                    add_mark(cur_date, 1.0 + realized, exit_reason)
                    break

            if pd.notna(dma10) and close < dma10:
                if pd.notna(hold_score) and hold_score < 50.0:
                    realized += remaining * (close / entry_price - 1.0)
                    remaining = 0.0
                    exit_reason = "dma10_holdscore_exit_all"
                    exit_date = cur_date
                    add_mark(cur_date, 1.0 + realized, exit_reason)
                    break
                if pd.notna(hold_score) and hold_score < 55.0:
                    target = 0.25
                    sell_frac = max(0.0, remaining - target)
                    realized += sell_frac * (close / entry_price - 1.0)
                    remaining -= sell_frac

        add_mark(cur_date, _equity_factor(entry_price, realized, remaining, close))

    if remaining > 0:
        last = rows.iloc[-1]
        exit_date = pd.Timestamp(last["date"]).normalize()
        realized += remaining * (float(last["close"]) / entry_price - 1.0)
        add_mark(exit_date, 1.0 + realized, exit_reason)

    return (
        {
            "symbol": symbol,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_source": event.get("entry_source"),
            "exit_reason": exit_reason,
            "trade_return_pct": realized,
            "partial_taken": partial_taken,
            "super_winner": super_winner,
            "super_winner_date": super_winner_date,
        },
        pd.DataFrame(marks).drop_duplicates("date", keep="last"),
    )


def simulate_events(
    events: pd.DataFrame,
    daily_by_symbol: dict[str, pd.DataFrame],
) -> list[tuple[int, pd.Series, dict[str, Any], pd.DataFrame]]:
    simulations = []
    for event_id, event in events.reset_index(drop=True).iterrows():
        trade, path = simulate_super_winner_trade(event, daily_by_symbol)
        trade["event_id"] = event_id
        if not path.empty:
            path = path.copy()
            path["symbol"] = event["symbol"]
            path["event_id"] = event_id
        simulations.append((event_id, event, trade, path))
    return simulations


def run_portfolio_with_replacement(
    simulations: list[tuple[int, pd.Series, dict[str, Any], pd.DataFrame]],
    commission_each_side: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    signal_dates = {pd.Timestamp(event["date"]).normalize() for _, event, _, _ in simulations}
    path_dates: set[pd.Timestamp] = set()
    for _, _, _, path in simulations:
        if not path.empty:
            path_dates.update(pd.to_datetime(path["date"]).dt.normalize().tolist())
    all_dates = sorted(signal_dates | {pd.Timestamp(d).normalize() for d in path_dates})

    sims_by_date: dict[pd.Timestamp, list[tuple[int, pd.Series, dict[str, Any], pd.DataFrame]]] = {}
    for item in simulations:
        _, event, _, _ = item
        sims_by_date.setdefault(pd.Timestamp(event["date"]).normalize(), []).append(item)

    cash = INITIAL_EQUITY
    open_positions: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    total_commission = 0.0

    for cur_date in all_dates:
        survivors = []
        for pos in open_positions:
            factor = pos["factor_by_date"].get(cur_date)
            if factor is not None:
                pos["last_factor"] = factor
            if cur_date >= pos["exit_date"]:
                gross_exit_value = pos["allocation"] * pos["last_factor"]
                exit_commission = gross_exit_value * commission_each_side
                cash += gross_exit_value - exit_commission
                total_commission += exit_commission
                trade = dict(pos["trade"])
                trade["entry_commission"] = pos["entry_commission"]
                trade["exit_commission"] = exit_commission
                trade["total_commission"] = pos["entry_commission"] + exit_commission
                trade["replaced"] = False
                trades.append(trade)
            else:
                survivors.append(pos)
        open_positions = survivors
        held_symbols = {pos["symbol"] for pos in open_positions}

        for _, event, trade, path in sims_by_date.get(cur_date, []):
            symbol = str(event["symbol"]).upper()
            if symbol in held_symbols or path.empty or pd.isna(trade.get("trade_return_pct")):
                continue

            current_position_value = sum(pos["allocation"] * pos["last_factor"] for pos in open_positions)
            current_equity = cash + current_position_value
            desired_allocation = current_equity * SLOT_ALLOC_PCT

            if len(open_positions) >= MAX_SLOTS or cash < desired_allocation:
                replaceable = [
                    {
                        "priority_score": pos["priority_score"],
                        "current_gain": pos["last_factor"] - 1.0,
                        "a_plus_candidate": pos["a_plus_candidate"],
                    }
                    for pos in open_positions
                ]
                replace_idx = choose_replacement_index(replaceable, event)
                if replace_idx is not None:
                    pos = open_positions.pop(replace_idx)
                    gross_exit_value = pos["allocation"] * pos["last_factor"]
                    exit_commission = gross_exit_value * commission_each_side
                    cash += gross_exit_value - exit_commission
                    total_commission += exit_commission
                    replaced_trade = dict(pos["trade"])
                    replaced_trade["exit_date"] = cur_date
                    replaced_trade["exit_reason"] = "priority_replacement_exit"
                    replaced_trade["trade_return_pct"] = pos["last_factor"] - 1.0
                    replaced_trade["entry_commission"] = pos["entry_commission"]
                    replaced_trade["exit_commission"] = exit_commission
                    replaced_trade["total_commission"] = pos["entry_commission"] + exit_commission
                    replaced_trade["replaced"] = True
                    trades.append(replaced_trade)
                    held_symbols = {p["symbol"] for p in open_positions}

            if len(open_positions) >= MAX_SLOTS:
                continue

            current_position_value = sum(pos["allocation"] * pos["last_factor"] for pos in open_positions)
            current_equity = cash + current_position_value
            gross_allocation = min(cash, current_equity * SLOT_ALLOC_PCT)
            if gross_allocation <= 0:
                continue
            entry_commission = gross_allocation * commission_each_side
            allocation = gross_allocation - entry_commission
            if allocation <= 0:
                continue

            factor_by_date = {
                pd.Timestamp(row.date).normalize(): float(row.equity_factor)
                for row in path.itertuples(index=False)
            }
            entry_factor = factor_by_date.get(cur_date, 1.0)
            exit_date = pd.Timestamp(path["date"].max()).normalize()
            cash -= gross_allocation
            total_commission += entry_commission

            if exit_date <= cur_date:
                gross_exit_value = allocation * entry_factor
                exit_commission = gross_exit_value * commission_each_side
                cash += gross_exit_value - exit_commission
                total_commission += exit_commission
                closed_trade = dict(trade)
                closed_trade["entry_commission"] = entry_commission
                closed_trade["exit_commission"] = exit_commission
                closed_trade["total_commission"] = entry_commission + exit_commission
                closed_trade["replaced"] = False
                trades.append(closed_trade)
            else:
                open_positions.append(
                    {
                        "symbol": symbol,
                        "allocation": allocation,
                        "factor_by_date": factor_by_date,
                        "last_factor": entry_factor,
                        "exit_date": exit_date,
                        "trade": trade,
                        "entry_commission": entry_commission,
                        "priority_score": float(event.get("same_day_priority_score", 0.0)),
                        "a_plus_candidate": bool(event.get("industry_a_plus_candidate", False)),
                    }
                )
                held_symbols.add(symbol)

        equity = cash + sum(pos["allocation"] * pos["last_factor"] for pos in open_positions)
        rows.append(
            {
                "date": cur_date,
                "equity": equity,
                "cash": cash,
                "open_positions": len(open_positions),
                "total_commission": total_commission,
            }
        )

    curve = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    curve["return_pct"] = curve["equity"] / INITIAL_EQUITY - 1.0
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0
    return curve, pd.DataFrame(trades), total_commission


def summarize(curve: pd.DataFrame, trades: pd.DataFrame, commission: float, total_commission: float) -> dict[str, object]:
    eras = trades[
        trades["symbol"].astype(str).eq("ERAS")
        & pd.to_datetime(trades["entry_date"]).dt.normalize().eq(pd.Timestamp("2026-01-07"))
    ]
    return {
        "commission_each_side": commission,
        "end_equity": float(curve["equity"].iloc[-1]),
        "total_return": float(curve["equity"].iloc[-1] / INITIAL_EQUITY - 1.0),
        "max_drawdown": float(curve["drawdown"].min()),
        "max_drawdown_date": curve.loc[curve["drawdown"].idxmin(), "date"],
        "trades": int(len(trades)),
        "win_rate": float((trades["trade_return_pct"] > 0).mean()) if not trades.empty else np.nan,
        "replacements": int(trades.get("replaced", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "eras_20260107_entered": bool(not eras.empty),
        "eras_20260107_return": float(eras["trade_return_pct"].iloc[0]) if not eras.empty else np.nan,
        "total_commission": float(total_commission),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate industry composite + replacement backtest")
    parser.add_argument("--signals", required=True, type=Path, help="Selected signal CSV generated from the 3000-symbol daily+5m dataset")
    parser.add_argument("--daily", required=True, type=Path, help="Long daily OHLCV parquet")
    parser.add_argument("--universe", required=True, type=Path, help="Universe parquet with industry and market cap")
    parser.add_argument("--outdir", default=Path("reports/industry_replacement_validation"), type=Path)
    parser.add_argument("--commission", action="append", type=float, default=[0.0, 0.002], help="One-way commission rates to test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    events, daily_by_symbol = load_validation_inputs(args.signals, args.daily, args.universe)
    events.to_csv(args.outdir / "selected_signals_with_industry_priority.csv", index=False)
    simulations = simulate_events(events, daily_by_symbol)

    rows = []
    for commission in args.commission:
        curve, trades, total_commission = run_portfolio_with_replacement(simulations, commission)
        suffix = "no_fee" if commission == 0 else f"fee_{commission:g}".replace(".", "p")
        curve.to_csv(args.outdir / f"equity_{suffix}.csv", index=False)
        trades.to_csv(args.outdir / f"trades_{suffix}.csv", index=False)
        trades.sort_values("trade_return_pct", ascending=False).head(30).to_csv(args.outdir / f"top30_trades_{suffix}.csv", index=False)
        rows.append(summarize(curve, trades, commission, total_commission))

    summary = pd.DataFrame(rows)
    summary.to_csv(args.outdir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.outdir}")


if __name__ == "__main__":
    main()
