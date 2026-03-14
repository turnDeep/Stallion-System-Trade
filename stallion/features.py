from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .strategy import STANDARD_FEATURE_COLUMNS, session_bucket_from_minutes


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def _anchored_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, lookback: int = 63) -> pd.Series:
    typical = (high + low + close) / 3.0
    result = pd.Series(index=close.index, dtype="float64")
    for end in range(len(close)):
        start = max(0, end - lookback + 1)
        window_high = high.iloc[start : end + 1]
        if window_high.empty:
            result.iloc[end] = np.nan
            continue
        anchor_label = window_high.idxmax()
        anchor_loc = close.index.get_loc(anchor_label)
        pv = (typical.iloc[anchor_loc : end + 1] * volume.iloc[anchor_loc : end + 1]).sum()
        vv = volume.iloc[anchor_loc : end + 1].sum()
        result.iloc[end] = pv / vv if vv else np.nan
    return result


def build_daily_feature_history(daily_bars: pd.DataFrame, universe: pd.DataFrame, spy_symbol: str = "SPY") -> pd.DataFrame:
    if daily_bars.empty or universe.empty:
        return pd.DataFrame(columns=["session_date", "symbol", *STANDARD_FEATURE_COLUMNS, "prev_day_high", "prev_day_atr14"])

    work = daily_bars.copy()
    work["session_date"] = pd.to_datetime(work["ts"], utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.normalize()
    work = work.dropna(subset=["session_date"]).sort_values(["symbol", "session_date"]).reset_index(drop=True)
    work["adj_close"] = work["adj_close"].fillna(work["close"])
    metadata = universe[["symbol", "sector", "industry"]].drop_duplicates(subset=["symbol"]).copy()
    work = work.merge(metadata, on="symbol", how="left")

    spy = work.loc[work["symbol"] == spy_symbol, ["session_date", "adj_close"]].rename(columns={"adj_close": "spy_adj_close"})
    spy = spy.drop_duplicates(subset=["session_date"]).set_index("session_date")

    feature_frames: list[pd.DataFrame] = []
    for symbol, frame in work.groupby("symbol", sort=False):
        frame = frame.sort_values("session_date").copy()
        frame["atr14"] = _true_range(frame["high"], frame["low"], frame["adj_close"]).rolling(14, min_periods=14).mean()
        frame["atr_pct"] = _safe_div(frame["atr14"], frame["adj_close"])
        frame["adr_pct_20"] = ((frame["high"] / frame["low"].replace(0, np.nan)) - 1.0).rolling(20, min_periods=10).mean()
        frame["sma50"] = frame["adj_close"].rolling(50, min_periods=50).mean()
        money_flow_multiplier = ((frame["adj_close"] - frame["low"]) - (frame["high"] - frame["adj_close"])) / (frame["high"] - frame["low"]).replace(0, np.nan)
        frame["buy_pressure_20"] = money_flow_multiplier.rolling(20, min_periods=10).mean()
        frame["anchored_vwap_63"] = _anchored_vwap(frame["high"], frame["low"], frame["adj_close"], frame["volume"], lookback=63)
        frame["distance_to_avwap_63"] = (frame["adj_close"] / frame["anchored_vwap_63"]) - 1.0
        spy_close = spy["spy_adj_close"].reindex(frame["session_date"])
        frame["rs_21"] = frame["adj_close"].pct_change(21) - spy_close.pct_change(21).to_numpy()
        frame["rs_63"] = frame["adj_close"].pct_change(63) - spy_close.pct_change(63).to_numpy()
        frame["rs_126"] = frame["adj_close"].pct_change(126) - spy_close.pct_change(126).to_numpy()
        frame["rs_252"] = frame["adj_close"].pct_change(252) - spy_close.pct_change(252).to_numpy()
        frame["daily_rs_score"] = (
            0.15 * frame["rs_21"].fillna(0.0)
            + 0.30 * frame["rs_63"].fillna(0.0)
            + 0.30 * frame["rs_126"].fillna(0.0)
            + 0.25 * frame["rs_252"].fillna(0.0)
        )
        frame["daily_rrs"] = frame["daily_rs_score"] / frame["atr_pct"].replace(0, np.nan)
        frame["prev_day_close_vs_sma50"] = (frame["adj_close"] / frame["sma50"]) - 1.0
        frame["prev_day_high"] = frame["high"].shift(1)
        frame["prev_day_atr14"] = frame["atr14"].shift(1)
        feature_frames.append(frame)

    features = pd.concat(feature_frames, ignore_index=True)

    sector_daily = (
        features.groupby(["session_date", "sector"], observed=True)["buy_pressure_20"]
        .mean()
        .reset_index(name="sector_buy_pressure")
        .sort_values(["sector", "session_date"])
    )
    sector_daily["sector_buy_pressure_prev"] = sector_daily.groupby("sector", observed=True)["sector_buy_pressure"].shift(1)

    industry_bp_daily = (
        features.groupby(["session_date", "industry"], observed=True)["buy_pressure_20"]
        .mean()
        .reset_index(name="industry_buy_pressure")
        .sort_values(["industry", "session_date"])
    )
    industry_bp_daily["industry_buy_pressure_prev"] = industry_bp_daily.groupby("industry", observed=True)["industry_buy_pressure"].shift(1)

    industry_rs_daily = (
        features.groupby(["session_date", "industry"], observed=True)["daily_rs_score"]
        .mean()
        .reset_index(name="industry_rs")
        .sort_values(["industry", "session_date"])
    )
    industry_rs_daily["industry_rs_prev"] = industry_rs_daily.groupby("industry", observed=True)["industry_rs"].shift(1)

    features = features.merge(
        sector_daily[["session_date", "sector", "sector_buy_pressure_prev"]],
        on=["session_date", "sector"],
        how="left",
    )
    features = features.merge(
        industry_bp_daily[["session_date", "industry", "industry_buy_pressure_prev"]],
        on=["session_date", "industry"],
        how="left",
    )
    features = features.merge(
        industry_rs_daily[["session_date", "industry", "industry_rs_prev"]],
        on=["session_date", "industry"],
        how="left",
    )

    features["daily_buy_pressure_prev"] = features.groupby("symbol", sort=False)["buy_pressure_20"].shift(1)
    features["prev_day_adr_pct"] = features.groupby("symbol", sort=False)["adr_pct_20"].shift(1)
    features["daily_rrs_prev"] = features.groupby("symbol", sort=False)["daily_rrs"].shift(1)
    features["daily_rs_score_prev"] = features.groupby("symbol", sort=False)["daily_rs_score"].shift(1)
    features["distance_to_avwap_63_prev"] = features.groupby("symbol", sort=False)["distance_to_avwap_63"].shift(1)
    features["prev_day_close_vs_sma50"] = features.groupby("symbol", sort=False)["prev_day_close_vs_sma50"].shift(1)

    output_columns = [
        "session_date",
        "symbol",
        "daily_buy_pressure_prev",
        "prev_day_adr_pct",
        "industry_buy_pressure_prev",
        "sector_buy_pressure_prev",
        "daily_rrs_prev",
        "daily_rs_score_prev",
        "distance_to_avwap_63_prev",
        "industry_rs_prev",
        "prev_day_close_vs_sma50",
        "prev_day_high",
        "prev_day_atr14",
        "sector",
        "industry",
    ]
    result = features[output_columns].dropna(subset=["session_date"]).copy()
    result["session_date"] = pd.to_datetime(result["session_date"]).dt.normalize()
    return result


def build_intraday_feature_panel(
    intraday_bars: pd.DataFrame,
    daily_features: pd.DataFrame,
    same_slot_lookback_sessions: int = 20,
) -> pd.DataFrame:
    if intraday_bars.empty or daily_features.empty:
        return pd.DataFrame(columns=["timestamp", "session_date", "symbol", "minutes_from_open", "session_bucket", *STANDARD_FEATURE_COLUMNS])

    work = intraday_bars.copy()
    work["timestamp"] = pd.to_datetime(work["ts"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    work = work.dropna(subset=["timestamp"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    work["session_date"] = work["timestamp"].dt.normalize()
    open_minutes = work["timestamp"].dt.hour * 60 + work["timestamp"].dt.minute
    work["minutes_from_open"] = open_minutes - (9 * 60 + 30)
    work = work[work["minutes_from_open"] >= 0].copy()
    work["slot_index"] = (work["minutes_from_open"] // 5).astype("int16")
    work["session_bucket"] = session_bucket_from_minutes(work["minutes_from_open"])

    same_slot_rows: list[pd.DataFrame] = []
    session_rows: list[pd.DataFrame] = []
    for symbol, frame in work.groupby("symbol", sort=False):
        frame = frame.sort_values("timestamp").copy()
        date_group = frame.groupby("session_date", sort=False)
        frame["session_high_so_far"] = date_group["high"].cummax()
        frame["session_low_so_far"] = date_group["low"].cummin()
        frame["session_close"] = date_group["close"].transform("last")
        frame["next_open"] = date_group["open"].shift(-1)
        frame["next_timestamp"] = date_group["timestamp"].shift(-1)
        typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
        frame["session_pv"] = typical * frame["volume"]
        frame["cum_pv"] = date_group["session_pv"].cumsum()
        frame["cum_vol"] = date_group["volume"].cumsum()
        frame["session_vwap"] = _safe_div(frame["cum_pv"], frame["cum_vol"])
        boundary_mask = (frame["minutes_from_open"] % 15 == 0) & (frame["minutes_from_open"] >= 15)
        frame["close_vs_vwap_15"] = np.nan
        frame.loc[boundary_mask, "close_vs_vwap_15"] = (frame.loc[boundary_mask, "close"] / frame.loc[boundary_mask, "session_vwap"]) - 1.0
        frame["close_vs_vwap_15"] = date_group["close_vs_vwap_15"].ffill()

        fifteen = frame.loc[boundary_mask, ["timestamp", "close"]].copy()
        fifteen["EMA_8_15"] = _ema(fifteen["close"], 8)
        frame = frame.merge(fifteen[["timestamp", "EMA_8_15"]], on="timestamp", how="left")
        frame["EMA_8_15"] = frame["EMA_8_15"].ffill()

        per_slot = frame[["symbol", "session_date", "slot_index", "volume"]].copy()
        per_slot["same_slot_avg_vol_20d"] = (
            per_slot.groupby(["symbol", "slot_index"], sort=False)["volume"]
            .transform(lambda s: s.shift(1).rolling(same_slot_lookback_sessions, min_periods=5).mean())
        )
        frame = frame.merge(
            per_slot[["symbol", "session_date", "slot_index", "same_slot_avg_vol_20d"]],
            on=["symbol", "session_date", "slot_index"],
            how="left",
        )
        frame["intraday_rvol"] = frame["volume"] / frame["same_slot_avg_vol_20d"].replace(0, np.nan)
        frame["volume_spike_5m"] = frame["intraday_rvol"] - 1.0
        same_slot_rows.append(frame)

    panel = pd.concat(same_slot_rows, ignore_index=True)
    daily = daily_features.copy()
    daily["session_date"] = pd.to_datetime(daily["session_date"]).dt.normalize()
    panel = panel.merge(daily, on=["symbol", "session_date"], how="left")
    panel["distance_to_prev_day_high"] = (panel["close"] / panel["prev_day_high"]) - 1.0
    panel["rs_x_intraday_rvol"] = panel["daily_rs_score_prev"] * panel["intraday_rvol"]
    panel["intraday_range_expansion_vs_atr"] = (panel["session_high_so_far"] - panel["session_low_so_far"]) / panel["prev_day_atr14"].replace(0, np.nan)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=False)

    return panel[
        [
            "timestamp",
            "session_date",
            "symbol",
            "minutes_from_open",
            "session_bucket",
            *STANDARD_FEATURE_COLUMNS,
            "next_open",
            "next_timestamp",
            "session_close",
            "close",
        ]
    ].copy()


def build_training_labels(feature_panel: pd.DataFrame, commission_rate_one_way: float, slippage_bps_per_side: float, spread_bps_round_trip: float, adverse_fill_floor: float, adverse_fill_cap: float) -> pd.DataFrame:
    if feature_panel.empty:
        return feature_panel.copy()

    work = feature_panel.copy()
    slippage_side = slippage_bps_per_side / 10_000.0
    spread_half = (spread_bps_round_trip / 2.0) / 10_000.0
    base_entry_markup = slippage_side + spread_half
    base_exit_markdown = slippage_side + spread_half

    next_bar_range_pct = ((work["session_close"] - work["next_open"]).abs() / work["next_open"].replace(0, np.nan)).fillna(0.0)
    adverse_fill_penalty = (0.0005 + (0.25 * next_bar_range_pct)).clip(lower=adverse_fill_floor, upper=adverse_fill_cap)
    entry_fill = work["next_open"] * (1.0 + base_entry_markup + adverse_fill_penalty)
    exit_fill = work["session_close"] * (1.0 - base_exit_markdown)
    work["net_return_stress_exec"] = ((exit_fill * (1.0 - commission_rate_one_way)) / (entry_fill * (1.0 + commission_rate_one_way))) - 1.0
    work["label_stress_exec"] = (work["net_return_stress_exec"] > 0).astype("int8")
    return work
