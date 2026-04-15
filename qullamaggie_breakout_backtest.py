#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# config
# ============================================================

@dataclass
class BacktestConfig:
    initial_equity: float = 100_000.0
    max_positions: int = 5

    risk_pct_per_trade: float = 1.0
    max_alloc_pct_per_trade: float = 0.25

    commission_rate: float = 0.00132
    slippage_bps: float = 5.0

    ep_gap_exclusion_pct: float = 0.10

    stop_buffer_bps: float = 10.0
    atr_limit_mult: float = 1.0
    adr_limit_mult: float = 1.0

    fast_fail_days: int = 1
    day0_day1_pivot_fail_exit_all: bool = True

    tp_partial_r: float = 1.75
    tp_partial_pct: float = 0.10

    dma10_reduce_to_frac: float = 0.25
    dma10_reduce_threshold: float = 55.0
    dma10_exit_threshold: float = 50.0

    dma21_reduce_to_frac: float = 0.10
    dma21_reduce_threshold: float = 45.0
    dma21_exit_after_reduce_threshold: float = 40.0
    use_dma21_tight_low_volume_grace: bool = True

    require_breakout_signal: bool = True

    use_intraday_trigger_time: bool = True
    entry_at: str = "trigger_close"

    allow_reentry_same_symbol: bool = False


@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_bar_time: pd.Timestamp | None
    entry_price: float
    initial_stop: float
    pivot_level: float
    breakout_day_low: float

    initial_shares: int
    shares: int
    initial_risk_per_share: float

    pending_dma21_grace: bool = False
    partial_profit_taken: bool = False
    reduced_on_dma21: bool = False
    entry_source: str = "standard_breakout"
    entry_lane: str = "none"


@dataclass
class Fill:
    date: pd.Timestamp
    symbol: str
    side: str
    shares: int
    price: float
    reason: str
    bar_time: pd.Timestamp | None
    gross_cash_flow: float
    commission: float
    net_cash_flow: float


# ============================================================
# I/O helpers
# ============================================================

def load_table(path: str | Path) -> Any:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)

    raise ValueError(f"Unsupported file type: {path}")


def normalize_intraday_input(obj: Any) -> pd.DataFrame:
    if isinstance(obj, dict):
        frames = []
        for symbol, df_sym in obj.items():
            if df_sym is None or len(df_sym) == 0:
                continue
            tmp = df_sym.copy()
            if "symbol" not in tmp.columns:
                tmp["symbol"] = symbol
            frames.append(tmp)
        if not frames:
            raise ValueError("intraday object is an empty dict")
        intra = pd.concat(frames, ignore_index=True)
    elif isinstance(obj, pd.DataFrame):
        intra = obj.copy()
    else:
        raise TypeError("intraday input must be dict[symbol->DataFrame] or DataFrame")

    required = {"symbol", "datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(intra.columns)
    if missing:
        raise ValueError(f"intraday missing columns: {sorted(missing)}")

    intra["datetime"] = pd.to_datetime(intra["datetime"])
    for c in ["open", "high", "low", "close", "volume"]:
        intra[c] = pd.to_numeric(intra[c], errors="coerce").astype(float)

    intra["session_date"] = intra["datetime"].dt.normalize()
    intra = intra.sort_values(["symbol", "datetime"], kind="mergesort").reset_index(drop=True)
    return intra


# ============================================================
# feature prep
# ============================================================

def _clip01(s: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(np.clip(np.asarray(s, dtype=float), 0.0, 1.0))


def prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily missing columns: {sorted(missing)}")

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    g = df.groupby("symbol", sort=False)

    df["prev_close"] = g["close"].shift(1)

    tr = np.maximum.reduce([
        (df["high"] - df["low"]).to_numpy(),
        (df["high"] - df["prev_close"]).abs().fillna(0.0).to_numpy(),
        (df["low"] - df["prev_close"]).abs().fillna(0.0).to_numpy(),
    ])
    df["tr"] = tr

    df["atr20"] = g["tr"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    df["dma10"] = g["close"].rolling(10, min_periods=10).mean().reset_index(level=0, drop=True)
    df["dma21"] = g["close"].rolling(21, min_periods=21).mean().reset_index(level=0, drop=True)

    daily_range = df["high"] - df["low"]
    df["adr20_pct"] = (
        (daily_range / df["close"].replace(0, np.nan))
        .groupby(df["symbol"], sort=False)
        .rolling(20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["avg_volume_20"] = g["volume"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["median_range_10"] = (
        df["range_pct"]
        .groupby(df["symbol"], sort=False)
        .rolling(10, min_periods=10)
        .median()
        .reset_index(level=0, drop=True)
    )

    df["row_num"] = g.cumcount()

    # ---------- hold_score features ----------
    df["dma10_lag3"] = g["dma10"].shift(3)
    df["dma21_lag5"] = g["dma21"].shift(5)

    ma_regime_exit = (
        (df["close"] >= df["dma10"])
        & (df["close"] >= df["dma21"])
        & (df["dma10"] > df["dma10_lag3"])
        & (df["dma21"] > df["dma21_lag5"])
    ).astype(float)

    # short-window surf for exits
    above_dma10_5d = (
        (df["close"] >= df["dma10"]).astype(float)
        .groupby(df["symbol"], sort=False)
        .rolling(5, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    dist_dma10_atr5 = (
        ((df["close"] - df["dma10"]).abs() / df["atr20"].replace(0, np.nan))
        .groupby(df["symbol"], sort=False)
        .rolling(5, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    deep_below_dma10_5d = (
        (df["close"] < df["dma10"] * 0.985).astype(float)
        .groupby(df["symbol"], sort=False)
        .rolling(5, min_periods=5)
        .sum()
        .reset_index(level=0, drop=True)
    )

    surf10_exit = (
        0.40 * _clip01(above_dma10_5d / 0.80)
        + 0.35 * _clip01(1.0 - (dist_dma10_atr5 / 1.00))
        + 0.25 * _clip01(1.0 - (deep_below_dma10_5d / 1.0))
    )

    above_dma21_8d = (
        (df["close"] >= df["dma21"]).astype(float)
        .groupby(df["symbol"], sort=False)
        .rolling(8, min_periods=8)
        .mean()
        .reset_index(level=0, drop=True)
    )
    dist_dma21_atr8 = (
        ((df["close"] - df["dma21"]).abs() / df["atr20"].replace(0, np.nan))
        .groupby(df["symbol"], sort=False)
        .rolling(8, min_periods=8)
        .mean()
        .reset_index(level=0, drop=True)
    )
    deep_below_dma21_8d = (
        (df["close"] < df["dma21"] * 0.98).astype(float)
        .groupby(df["symbol"], sort=False)
        .rolling(8, min_periods=8)
        .sum()
        .reset_index(level=0, drop=True)
    )

    surf21_exit = (
        0.40 * _clip01(above_dma21_8d / 0.75)
        + 0.35 * _clip01(1.0 - (dist_dma21_atr8 / 1.20))
        + 0.25 * _clip01(1.0 - (deep_below_dma21_8d / 1.5))
    )

    df["surf_quality_exit"] = np.fmax(surf10_exit, surf21_exit)

    rolling_high_10 = (
        df.groupby("symbol", sort=False)["high"]
        .rolling(10, min_periods=10)
        .max()
        .reset_index(level=0, drop=True)
    )
    drawdown_from_high10 = ((rolling_high_10 - df["close"]) / rolling_high_10.replace(0, np.nan)).clip(lower=0)
    shallow_pullback_exit = _clip01(1.0 - (drawdown_from_high10 / 0.10))

    slope10 = (df["dma10"] / df["dma10_lag3"].replace(0, np.nan) - 1.0)
    slope21 = (df["dma21"] / df["dma21_lag5"].replace(0, np.nan) - 1.0)
    slope_quality_exit = (
        0.60 * _clip01(slope10 / 0.03)
        + 0.40 * _clip01(slope21 / 0.05)
    )

    tight_low_volume_day = (
        (df["range_pct"] < df["median_range_10"])
        & (df["volume"] < df["avg_volume_20"])
        & (df["close"] <= df["prev_close"])
    ).astype(float)

    close_pos_in_bar = (
        (df["close"] - df["low"])
        / (df["high"] - df["low"]).replace(0, np.nan)
    )
    distribution_day = (
        (df["close"] < df["prev_close"])
        & (df["volume"] > 1.2 * df["avg_volume_20"])
        & (close_pos_in_bar < 0.35)
        & (df["range_pct"] > df["median_range_10"] * 1.05)
    ).astype(float)
    anti_distribution_score = 1.0 - distribution_day

    df["hold_score"] = (
        25.0 * ma_regime_exit
        + 25.0 * df["surf_quality_exit"].fillna(0.0)
        + 15.0 * shallow_pullback_exit.fillna(0.0)
        + 15.0 * slope_quality_exit.fillna(0.0)
        + 10.0 * tight_low_volume_day.fillna(0.0)
        + 10.0 * pd.Series(anti_distribution_score, index=df.index).fillna(0.0)
    )

    df["tight_low_volume_day"] = tight_low_volume_day.astype(bool)
    return df


def prepare_signals(signals: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "breakout_signal"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"signals missing columns: {sorted(missing)}")

    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    sig["breakout_signal"] = sig["breakout_signal"].fillna(False).astype(bool)

    if "effective_pivot_level" in sig.columns:
        sig["pivot_high"] = pd.to_numeric(sig["effective_pivot_level"], errors="coerce")
    elif "pivot_high" in sig.columns:
        sig["pivot_high"] = pd.to_numeric(sig["pivot_high"], errors="coerce")
    else:
        sig["pivot_high"] = np.nan

    if "trigger_time" in sig.columns:
        sig["trigger_time"] = pd.to_datetime(sig["trigger_time"], errors="coerce")

    if "entry_source" not in sig.columns:
        sig["entry_source"] = "standard_breakout"

    if "entry_priority_bucket" not in sig.columns:
        sig["entry_priority_bucket"] = np.select(
            [
                sig["entry_source"].eq("standard_breakout"),
                sig["entry_source"].eq("tight_reversal"),
                sig["entry_source"].eq("power_continuation"),
            ],
            [0, 1, 2],
            default=99,
        )

    if "priority_score_within_source" not in sig.columns:
        sig["priority_score_within_source"] = np.where(
            sig["entry_source"].eq("standard_breakout"),
            pd.to_numeric(sig.get("rs_rating"), errors="coerce"),
            pd.to_numeric(sig.get("trigger_score"), errors="coerce"),
        )

    if "entry_stop_policy" not in sig.columns:
        sig["entry_stop_policy"] = "respect_stop_limit"

    if "leader_score" not in sig.columns:
        sig["leader_score"] = np.nan

    # 同一 symbol/date に standard + zigzag が共存しても、
    # 優先順位ルールで 1 行に潰して backtest merge を安定化させる。
    sig = (
        sig.sort_values(
            [
                "date",
                "symbol",
                "entry_priority_bucket",
                "priority_score_within_source",
                "trigger_time",
                "leader_score",
            ],
            ascending=[True, True, True, False, True, False],
            kind="mergesort",
        )
        .groupby(["symbol", "date"], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )

    return sig.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)


# ============================================================
# execution helpers
# ============================================================

def apply_slippage(price: float, side: str, slippage_bps: float) -> float:
    slip = slippage_bps / 10000.0
    if side.lower() == "buy":
        return price * (1.0 + slip)
    if side.lower() == "sell":
        return price * (1.0 - slip)
    raise ValueError(f"Unknown side: {side}")


def commission_cost(notional: float, commission_rate: float) -> float:
    return abs(notional) * commission_rate


def make_fill(
    date: pd.Timestamp,
    symbol: str,
    side: str,
    shares: int,
    price: float,
    reason: str,
    bar_time: pd.Timestamp | None,
    commission_rate: float,
) -> Fill:
    gross = -shares * price if side.lower() == "buy" else shares * price
    comm = commission_cost(shares * price, commission_rate)
    net = gross - comm
    return Fill(
        date=date,
        symbol=symbol,
        side=side,
        shares=shares,
        price=price,
        reason=reason,
        bar_time=bar_time,
        gross_cash_flow=gross,
        commission=comm,
        net_cash_flow=net,
    )


def sell_down_to_target(
    pos: Position,
    target_remaining_shares: int,
    date: pd.Timestamp,
    price: float,
    reason: str,
    bar_time: pd.Timestamp | None,
    fills: list[Fill],
    cfg: BacktestConfig,
    cash: float,
) -> float:
    target_remaining_shares = max(0, min(target_remaining_shares, pos.shares))
    shares_to_sell = pos.shares - target_remaining_shares
    if shares_to_sell <= 0:
        return cash

    exec_price = apply_slippage(price, "sell", cfg.slippage_bps)
    fill = make_fill(
        date=date,
        symbol=pos.symbol,
        side="sell",
        shares=shares_to_sell,
        price=exec_price,
        reason=reason,
        bar_time=bar_time,
        commission_rate=cfg.commission_rate,
    )
    fills.append(fill)
    cash += fill.net_cash_flow
    pos.shares -= shares_to_sell
    return cash


# ============================================================
# backtest engine
# ============================================================

def run_backtest(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    intraday: pd.DataFrame | None,
    cfg: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Position]]:
    daily = prepare_daily(daily)
    signals = prepare_signals(signals)

    if intraday is not None:
        intraday = normalize_intraday_input(intraday)

    df = daily.merge(
        signals,
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_sig"),
    )
    df["breakout_signal"] = df["breakout_signal"].fillna(False).astype(bool)

    daily_lookup = {(r.symbol, r.date): r for r in df.itertuples(index=False)}
    intra_groups = {}
    if intraday is not None:
        intra_groups = {
            (sym, d): g.reset_index(drop=True)
            for (sym, d), g in intraday.groupby(["symbol", "session_date"], sort=False)
        }

    all_days = sorted(df["date"].drop_duplicates().tolist())

    open_positions: dict[str, Position] = {}
    fills: list[Fill] = []
    equity_rows: list[dict[str, Any]] = []

    cash = cfg.initial_equity
    closed_symbols_once: set[str] = set()

    def equity_on_date(cur_date: pd.Timestamp) -> float:
        total = cash
        for sym, pos in open_positions.items():
            row = daily_lookup.get((sym, cur_date))
            if row is not None and not pd.isna(row.close):
                total += pos.shares * float(row.close)
            else:
                total += pos.shares * pos.entry_price
        return total

    for cur_date in all_days:
        # --------------------------------------------------------
        # exits
        # --------------------------------------------------------
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            row = daily_lookup.get((sym, cur_date))
            if row is None:
                continue

            days_since_entry = int((cur_date - pos.entry_date).days)

            if cur_date == pos.entry_date:
                continue

            row_low = float(row.low)
            row_high = float(row.high)
            row_close = float(row.close)
            row_dma10 = float(row.dma10) if not pd.isna(row.dma10) else np.nan
            row_dma21 = float(row.dma21) if not pd.isna(row.dma21) else np.nan
            row_hold_score = float(row.hold_score) if not pd.isna(row.hold_score) else np.nan
            row_tight_low_volume = bool(row.tight_low_volume_day) if pd.notna(row.tight_low_volume_day) else False

            if row_low <= pos.initial_stop and pos.shares > 0:
                cash = sell_down_to_target(
                    pos=pos,
                    target_remaining_shares=0,
                    date=cur_date,
                    price=pos.initial_stop,
                    reason="hard_stop_lod",
                    bar_time=None,
                    fills=fills,
                    cfg=cfg,
                    cash=cash,
                )
                if pos.shares == 0:
                    del open_positions[sym]
                    closed_symbols_once.add(sym)
                continue

            if cfg.day0_day1_pivot_fail_exit_all and days_since_entry <= cfg.fast_fail_days and row_close < pos.pivot_level and pos.shares > 0:
                cash = sell_down_to_target(
                    pos=pos,
                    target_remaining_shares=0,
                    date=cur_date,
                    price=row_close,
                    reason="pivot_fail_exit_all",
                    bar_time=None,
                    fills=fills,
                    cfg=cfg,
                    cash=cash,
                )
                if pos.shares == 0:
                    del open_positions[sym]
                    closed_symbols_once.add(sym)
                continue

            if not pos.partial_profit_taken and pos.shares > 0:
                tp_price_r = pos.entry_price + cfg.tp_partial_r * pos.initial_risk_per_share
                tp_price_pct = pos.entry_price * (1.0 + cfg.tp_partial_pct)
                tp_hit_price = min(tp_price_r, tp_price_pct)
                if row_high >= tp_hit_price:
                    target_remaining = math.ceil(pos.initial_shares * 0.50)
                    cash = sell_down_to_target(
                        pos=pos,
                        target_remaining_shares=target_remaining,
                        date=cur_date,
                        price=tp_hit_price,
                        reason="partial_tp_1st",
                        bar_time=None,
                        fills=fills,
                        cfg=cfg,
                        cash=cash,
                    )
                    pos.partial_profit_taken = True

            if pos.shares == 0:
                del open_positions[sym]
                closed_symbols_once.add(sym)
                continue

            if not pd.isna(row_dma21) and row_close < row_dma21 and pos.shares > 0:
                if (
                    cfg.use_dma21_tight_low_volume_grace
                    and row_tight_low_volume
                    and not pos.pending_dma21_grace
                ):
                    pos.pending_dma21_grace = True
                    continue

                if not pos.reduced_on_dma21:
                    if row_hold_score < cfg.dma21_reduce_threshold:
                        target_remaining = math.ceil(pos.initial_shares * cfg.dma21_reduce_to_frac)
                        cash = sell_down_to_target(
                            pos=pos,
                            target_remaining_shares=target_remaining,
                            date=cur_date,
                            price=row_close,
                            reason="dma21_reduce_to_small_runner",
                            bar_time=None,
                            fills=fills,
                            cfg=cfg,
                            cash=cash,
                        )
                        pos.reduced_on_dma21 = True
                        pos.pending_dma21_grace = False
                        if pos.shares == 0:
                            del open_positions[sym]
                            closed_symbols_once.add(sym)
                        continue
                    else:
                        pos.pending_dma21_grace = False
                else:
                    if row_hold_score < cfg.dma21_exit_after_reduce_threshold:
                        cash = sell_down_to_target(
                            pos=pos,
                            target_remaining_shares=0,
                            date=cur_date,
                            price=row_close,
                            reason="dma21_exit_after_reduce",
                            bar_time=None,
                            fills=fills,
                            cfg=cfg,
                            cash=cash,
                        )
                        if pos.shares == 0:
                            del open_positions[sym]
                            closed_symbols_once.add(sym)
                        continue
                    else:
                        pos.pending_dma21_grace = False
            else:
                pos.pending_dma21_grace = False

            if pos.shares == 0:
                del open_positions[sym]
                closed_symbols_once.add(sym)
                continue

            if not pd.isna(row_dma10) and row_close < row_dma10 and pos.shares > 0:
                if row_hold_score < cfg.dma10_exit_threshold:
                    cash = sell_down_to_target(
                        pos=pos,
                        target_remaining_shares=0,
                        date=cur_date,
                        price=row_close,
                        reason="dma10_holdscore_exit_all",
                        bar_time=None,
                        fills=fills,
                        cfg=cfg,
                        cash=cash,
                    )
                elif row_hold_score < cfg.dma10_reduce_threshold:
                    target_remaining = math.ceil(pos.initial_shares * cfg.dma10_reduce_to_frac)
                    if pos.shares > target_remaining:
                        cash = sell_down_to_target(
                            pos=pos,
                            target_remaining_shares=target_remaining,
                            date=cur_date,
                            price=row_close,
                            reason="dma10_holdscore_reduce",
                            bar_time=None,
                            fills=fills,
                            cfg=cfg,
                            cash=cash,
                        )

                if pos.shares == 0:
                    del open_positions[sym]
                    closed_symbols_once.add(sym)
                    continue

        # --------------------------------------------------------
        # entries
        # --------------------------------------------------------
        todays = df.loc[df["date"].eq(cur_date) & df["breakout_signal"]].copy()
        if not todays.empty:
            # 優先順位付け（同時に発火した場合の枠割り当ての厳選）
            sort_cols = []
            sort_asc = []
            if "entry_priority_bucket" in todays.columns:
                sort_cols.append("entry_priority_bucket")
                sort_asc.append(True)
            if "priority_score_within_source" in todays.columns:
                sort_cols.append("priority_score_within_source")
                sort_asc.append(False)
            if "leader_score" in todays.columns:
                sort_cols.append("leader_score")
                sort_asc.append(False)
            if "trigger_time" in todays.columns:
                sort_cols.append("trigger_time")
                sort_asc.append(True)
            sort_cols.append("symbol")
            sort_asc.append(True)
            
            todays = todays.sort_values(sort_cols, ascending=sort_asc, kind="mergesort")
            
            for row in todays.itertuples(index=False):
                if cfg.require_breakout_signal and not bool(row.breakout_signal):
                    continue
                if row.symbol in open_positions:
                    continue
                if (not cfg.allow_reentry_same_symbol) and (row.symbol in closed_symbols_once):
                    continue
                if len(open_positions) >= cfg.max_positions:
                    continue

                # EP-like exclude
                if pd.notna(row.prev_close) and row.prev_close > 0:
                    gap_pct = row.open / row.prev_close - 1.0
                    if gap_pct >= cfg.ep_gap_exclusion_pct:
                        continue

                pivot_level = float(row.pivot_high)
                breakout_day_low = float(row.low)

                entry_price = None
                entry_bar_time = None

                if cfg.use_intraday_trigger_time and intraday is not None and hasattr(row, "trigger_time") and pd.notna(row.trigger_time):
                    intra_day = intra_groups.get((row.symbol, cur_date))
                    if intra_day is not None and len(intra_day) > 0:
                        trig_ts = pd.Timestamp(row.trigger_time)
                        match = intra_day.loc[intra_day["datetime"] >= trig_ts]
                        if len(match) > 0:
                            entry_bar = match.iloc[0]
                            entry_bar_time = pd.Timestamp(entry_bar["datetime"])
                            if cfg.entry_at == "trigger_close":
                                entry_price = float(entry_bar["close"])

                if entry_price is None:
                    entry_price = float(row.close)
                    entry_bar_time = None

                if not np.isfinite(entry_price) or entry_price <= 0:
                    continue

                stop_buffer = cfg.stop_buffer_bps / 10000.0
                initial_stop = min(breakout_day_low, pivot_level) * (1.0 - stop_buffer)
                risk_per_share = entry_price - initial_stop

                if not np.isfinite(risk_per_share) or risk_per_share <= 0:
                    continue

                # stop limit check (standard only or based on policy)
                entry_stop_policy = str(getattr(row, "entry_stop_policy", "respect_stop_limit"))
                if entry_stop_policy == "respect_stop_limit":
                    atr_limit = float(row.atr20) * cfg.atr_limit_mult if pd.notna(row.atr20) else np.nan
                    adr_limit = float(row.adr20_pct) * entry_price * cfg.adr_limit_mult if pd.notna(row.adr20_pct) else np.nan
                    stop_limit = np.nanmin([atr_limit, adr_limit]) if not (pd.isna(atr_limit) and pd.isna(adr_limit)) else np.nan
                    if pd.notna(stop_limit) and risk_per_share > stop_limit:
                        continue

                equity = equity_on_date(cur_date)
                risk_budget = equity * cfg.risk_pct_per_trade
                alloc_budget = equity * cfg.max_alloc_pct_per_trade

                shares_by_risk = math.floor(risk_budget / risk_per_share)
                shares_by_alloc = math.floor(alloc_budget / entry_price)
                shares_by_cash = math.floor(
                    cash / (entry_price * (1 + cfg.commission_rate + cfg.slippage_bps / 10000.0))
                )
                shares = min(shares_by_risk, shares_by_alloc, shares_by_cash)

                if shares <= 0:
                    continue

                exec_price = apply_slippage(entry_price, "buy", cfg.slippage_bps)
                buy_fill = make_fill(
                    date=cur_date,
                    symbol=row.symbol,
                    side="buy",
                    shares=shares,
                    price=exec_price,
                    reason="entry",
                    bar_time=entry_bar_time,
                    commission_rate=cfg.commission_rate,
                )
                fills.append(buy_fill)
                cash += buy_fill.net_cash_flow

                pos = Position(
                    symbol=row.symbol,
                    entry_date=cur_date,
                    entry_bar_time=entry_bar_time,
                    entry_price=exec_price,
                    initial_stop=initial_stop,
                    pivot_level=pivot_level,
                    breakout_day_low=breakout_day_low,
                    initial_shares=shares,
                    shares=shares,
                    initial_risk_per_share=exec_price - initial_stop,
                    entry_source=str(getattr(row, "entry_source", "standard_breakout")),
                    entry_lane=str(getattr(row, "entry_lane", "none")),
                )
                open_positions[row.symbol] = pos

                # same-day intraday LOD stop
                intra_day = intra_groups.get((row.symbol, cur_date))
                if intra_day is not None and entry_bar_time is not None and len(intra_day) > 0:
                    after_entry = intra_day.loc[intra_day["datetime"] > entry_bar_time]
                    if len(after_entry) > 0:
                        stop_hit = after_entry.loc[after_entry["low"] <= pos.initial_stop]
                        if len(stop_hit) > 0 and pos.shares > 0:
                            stop_bar = stop_hit.iloc[0]
                            cash = sell_down_to_target(
                                pos=pos,
                                target_remaining_shares=0,
                                date=cur_date,
                                price=pos.initial_stop,
                                reason="same_day_lod_stop",
                                bar_time=pd.Timestamp(stop_bar["datetime"]),
                                fills=fills,
                                cfg=cfg,
                                cash=cash,
                            )

                if pos.shares == 0:
                    del open_positions[row.symbol]
                    closed_symbols_once.add(row.symbol)
                    continue

                # day0 pivot failure
                if cfg.day0_day1_pivot_fail_exit_all and float(row.close) < pos.pivot_level and pos.shares > 0:
                    cash = sell_down_to_target(
                        pos=pos,
                        target_remaining_shares=0,
                        date=cur_date,
                        price=float(row.close),
                        reason="day0_close_below_pivot_exit_all",
                        bar_time=None,
                        fills=fills,
                        cfg=cfg,
                        cash=cash,
                    )

                if pos.shares == 0:
                    del open_positions[row.symbol]
                    closed_symbols_once.add(row.symbol)

        equity_rows.append({
            "date": cur_date,
            "cash": cash,
            "positions_value": sum(
                open_positions[sym].shares * float(daily_lookup[(sym, cur_date)].close)
                for sym in open_positions
                if (sym, cur_date) in daily_lookup
            ),
            "equity": equity_on_date(cur_date),
            "open_positions": len(open_positions),
        })

    equity_curve = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
    fills_df = pd.DataFrame([asdict(f) for f in fills])
    stats = summarize_backtest(equity_curve, fills_df)
    return equity_curve, fills_df, stats, open_positions


# ============================================================
# stats
# ============================================================

def summarize_backtest(equity_curve: pd.DataFrame, fills_df: pd.DataFrame) -> dict[str, float]:
    stats: dict[str, float] = {}
    if equity_curve.empty:
        return stats

    eq = equity_curve["equity"].astype(float)
    ret = eq.pct_change().fillna(0.0)

    total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
    rolling_peak = eq.cummax()
    drawdown = eq / rolling_peak - 1.0
    max_drawdown = drawdown.min()

    ann_factor = 252.0
    mean_ret = ret.mean()
    std_ret = ret.std(ddof=0)

    sharpe = np.nan
    if std_ret > 0:
        sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor)

    downside = ret[ret < 0]
    downside_std = downside.std(ddof=0)
    sortino = np.nan
    if downside_std > 0:
        sortino = (mean_ret / downside_std) * np.sqrt(ann_factor)

    stats["start_equity"] = float(eq.iloc[0])
    stats["end_equity"] = float(eq.iloc[-1])
    stats["total_return"] = float(total_return)
    stats["max_drawdown"] = float(max_drawdown)
    stats["sharpe"] = float(sharpe) if pd.notna(sharpe) else np.nan
    stats["sortino"] = float(sortino) if pd.notna(sortino) else np.nan

    if not fills_df.empty:
        round_trips = build_round_trips(fills_df)
        stats["num_round_trips"] = float(len(round_trips))
        if not round_trips.empty:
            wins = (round_trips["pnl"] > 0).mean()
            stats["win_rate"] = float(wins)
            stats["avg_pnl"] = float(round_trips["pnl"].mean())
            stats["median_pnl"] = float(round_trips["pnl"].median())
            stats["profit_factor"] = compute_profit_factor(round_trips["pnl"])
        else:
            stats["num_round_trips"] = 0.0
            stats["win_rate"] = np.nan
            stats["avg_pnl"] = np.nan
            stats["median_pnl"] = np.nan
            stats["profit_factor"] = np.nan
    else:
        stats["num_round_trips"] = 0.0
        stats["win_rate"] = np.nan
        stats["avg_pnl"] = np.nan
        stats["median_pnl"] = np.nan
        stats["profit_factor"] = np.nan

    return stats


def build_round_trips(fills_df: pd.DataFrame) -> pd.DataFrame:
    if fills_df.empty:
        return pd.DataFrame(columns=["symbol", "entry_date", "exit_date", "shares", "pnl"])

    fills = fills_df.sort_values(["symbol", "date", "bar_time"], kind="mergesort").copy()
    rows = []

    for sym, g in fills.groupby("symbol", sort=False):
        open_lots: list[tuple[pd.Timestamp, int, float]] = []

        for r in g.itertuples(index=False):
            if r.side == "buy":
                open_lots.append((r.date, int(r.shares), float(r.price)))
            else:
                shares_to_match = int(r.shares)
                sell_price = float(r.price)

                while shares_to_match > 0 and open_lots:
                    entry_date, lot_shares, lot_price = open_lots[0]
                    matched = min(shares_to_match, lot_shares)
                    pnl = matched * (sell_price - lot_price)

                    rows.append({
                        "symbol": sym,
                        "entry_date": entry_date,
                        "exit_date": r.date,
                        "shares": matched,
                        "pnl": pnl,
                    })

                    shares_to_match -= matched
                    lot_shares -= matched

                    if lot_shares == 0:
                        open_lots.pop(0)
                    else:
                        open_lots[0] = (entry_date, lot_shares, lot_price)

    return pd.DataFrame(rows)


def compute_profit_factor(pnl: pd.Series) -> float:
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses <= 0:
        return np.nan
    return float(gains / losses)
