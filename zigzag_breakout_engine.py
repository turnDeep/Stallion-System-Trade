from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from breakout_signal_engine import compute_breakout_scores_with_diag


SESSION_TZ = "America/New_York"


@dataclass(frozen=True)
class ZigZagBreakoutConfig:
    session_timezone: str = SESSION_TZ
    zigzag_order: int = 3
    pivot_confirm_bars: int = 1
    leader_min: float = 90.0
    setup_min: float | None = None
    setup_max: float | None = None
    trigger_min: float | None = None
    line_min_gap: int = 5
    line_max_gap: int = 40
    max_line_slope_pct_per_bar: float = 0.03
    breakout_dist_ref: float = 0.008
    reversal_move_ref: float = 0.06
    reversal_from_low_ref: float = 0.15
    pullback_depth_ref: float = 0.15
    trigger_selection_window_bars: int = 6
    trigger_weight_breakout: float = 45.0
    trigger_weight_volume: float = 5.0
    trigger_weight_price_expansion: float = 15.0
    trigger_weight_reversal: float = 25.0
    trigger_weight_hold: float = 10.0


def _normalize_daily_input(
    daily_df: pd.DataFrame | dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if isinstance(daily_df, dict):
        frames = []
        for sym, df_sym in daily_df.items():
            if df_sym is None or len(df_sym) == 0:
                continue
            x = df_sym.copy().reset_index()
            x.columns = [str(c).lower() for c in x.columns]
            if "symbol" not in x.columns:
                x["symbol"] = sym
            if "date" not in x.columns:
                first = str(x.columns[0]).lower()
                x = x.rename(columns={first: "date"})
            frames.append(x)
        if not frames:
            raise ValueError("daily input dict is empty")
        daily = pd.concat(frames, ignore_index=True)
    else:
        daily = daily_df.copy()
        daily.columns = [str(c).lower() for c in daily.columns]

    rename_map = {
        "adj close": "adj_close",
    }
    daily = daily.rename(columns=rename_map)
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily input missing columns: {sorted(missing)}")
    return daily


def _normalize_intraday_input(
    intraday_df: pd.DataFrame | dict[str, pd.DataFrame],
    *,
    session_timezone: str = SESSION_TZ,
) -> pd.DataFrame:
    if isinstance(intraday_df, dict):
        frames = []
        for sym, df_sym in intraday_df.items():
            if df_sym is None or len(df_sym) == 0:
                continue
            x = df_sym.copy()
            if "symbol" not in x.columns:
                x["symbol"] = sym
            x = x.reset_index()
            x.columns = [str(c).lower() for c in x.columns]
            if "datetime" not in x.columns:
                first = str(x.columns[0]).lower()
                x = x.rename(columns={first: "datetime"})
            frames.append(x)
        if not frames:
            raise ValueError("intraday input dict is empty")
        intra = pd.concat(frames, ignore_index=True)
    else:
        intra = intraday_df.copy()
        intra.columns = [str(c).lower() for c in intra.columns]

    required = {"symbol", "datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(intra.columns)
    if missing:
        raise ValueError(f"intraday input missing columns: {sorted(missing)}")

    intra["datetime"] = pd.to_datetime(intra["datetime"], errors="coerce")
    if getattr(intra["datetime"].dt, "tz", None) is not None:
        intra["datetime"] = intra["datetime"].dt.tz_convert(session_timezone).dt.tz_localize(None)
    for c in ["open", "high", "low", "close", "volume"]:
        intra[c] = pd.to_numeric(intra[c], errors="coerce").astype("float64")
    intra = intra.sort_values(["symbol", "datetime"], kind="mergesort").reset_index(drop=True)
    return intra


def _add_same_time_volume_features(
    intra: pd.DataFrame,
    lookback_sessions: int = 20,
    min_periods: int = 5,
) -> pd.DataFrame:
    work = intra.copy()
    work["_orig_idx"] = np.arange(len(work))

    tmp = work.sort_values(
        ["symbol", "slot_index", "session_date", "datetime"],
        kind="mergesort",
    ).copy()

    tmp["avg_bar_vol_same_slot"] = (
        tmp.groupby(["symbol", "slot_index"], sort=False)["volume"]
        .transform(lambda s: s.shift(1).rolling(lookback_sessions, min_periods=min_periods).mean())
    )

    tmp["avg_cum_vol_same_slot"] = (
        tmp.groupby(["symbol", "slot_index"], sort=False)["cum_volume_session"]
        .transform(lambda s: s.shift(1).rolling(lookback_sessions, min_periods=min_periods).mean())
    )

    tmp = tmp.sort_values("_orig_idx", kind="mergesort")
    work["avg_bar_vol_same_slot"] = tmp["avg_bar_vol_same_slot"].to_numpy()
    work["avg_cum_vol_same_slot"] = tmp["avg_cum_vol_same_slot"].to_numpy()
    work["bar_vol_ratio"] = work["volume"] / work["avg_bar_vol_same_slot"].replace(0, np.nan)
    work["cum_vol_ratio"] = work["cum_volume_session"] / work["avg_cum_vol_same_slot"].replace(0, np.nan)
    return work.drop(columns="_orig_idx")


def _clip01(x: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(np.clip(np.asarray(x, dtype="float64"), 0.0, 1.0))


def _build_confirmed_pivots(sub: pd.DataFrame, order: int) -> list[dict[str, Any]]:
    highs = sub["high"].to_numpy()
    lows = sub["low"].to_numpy()
    dates = sub["date"].to_numpy()

    high_idxs = argrelextrema(highs, np.greater, order=order)[0]
    low_idxs = argrelextrema(lows, np.less, order=order)[0]

    candidates: list[dict[str, Any]] = []
    for idx in high_idxs:
        candidates.append({"idx": int(idx), "date": dates[idx], "price": float(highs[idx]), "type": "high"})
    for idx in low_idxs:
        candidates.append({"idx": int(idx), "date": dates[idx], "price": float(lows[idx]), "type": "low"})
    candidates.sort(key=lambda x: x["idx"])

    if not candidates:
        return []

    stack = [candidates[0]]
    for current in candidates[1:]:
        last = stack[-1]
        if last["type"] == current["type"]:
            if last["type"] == "high" and current["price"] > last["price"]:
                stack[-1] = current
            elif last["type"] == "low" and current["price"] < last["price"]:
                stack[-1] = current
        else:
            stack.append(current)

    out: list[dict[str, Any]] = []
    n = len(sub)
    for pivot in stack:
        confirm_idx = pivot["idx"] + order
        if confirm_idx >= n:
            continue
        out.append(
            {
                **pivot,
                "confirm_idx": confirm_idx,
                "confirm_date": sub.iloc[confirm_idx]["date"],
            }
        )
    return out


def _compute_zigzag_setup_daily(
    base: pd.DataFrame,
    *,
    zigzag_order: int,
    pivot_confirm_bars: int,
    line_min_gap: int,
    line_max_gap: int,
    max_line_slope_pct_per_bar: float,
) -> pd.DataFrame:
    out = base.copy()
    extra_cols = [
        "zigzag_high1_date",
        "zigzag_high1_price",
        "zigzag_high2_date",
        "zigzag_high2_price",
        "zigzag_high_gap_bars",
        "zigzag_age_bars",
        "zigzag_line_value",
        "zigzag_line_prev",
        "zigzag_line_slope",
        "zigzag_line_valid",
        "zigzag_setup_raw",
        "setup_pts_line_valid",
        "setup_pts_proximity",
        "setup_pts_spacing",
        "setup_pts_age",
        "setup_pts_slope",
        "setup_pts_tightness",
        "setup_pts_dryup",
    ]
    for col in extra_cols:
        if col.endswith("_date"):
            out[col] = pd.NaT
        elif col == "zigzag_line_valid":
            out[col] = False
        else:
            out[col] = np.nan

    for symbol, idx in out.groupby("symbol", sort=False).groups.items():
        locs = np.asarray(idx)
        sub = out.loc[locs, ["date", "open", "high", "low", "close", "volume", "vol20"]].reset_index(drop=True)
        pivots = _build_confirmed_pivots(sub, zigzag_order)
        high_pivots = [p for p in pivots if p["type"] == "high"]

        if len(high_pivots) < 2:
            continue

        for i in range(len(sub)):
            confirmed_highs = [
                p
                for p in high_pivots
                if (p["idx"] + pivot_confirm_bars) <= i and p["idx"] < i
            ]
            if len(confirmed_highs) < 2:
                continue

            h1 = confirmed_highs[-2]
            h2 = confirmed_highs[-1]
            gap = h2["idx"] - h1["idx"]
            if gap < line_min_gap or gap > line_max_gap:
                continue
            if not (h2["price"] < h1["price"]):
                continue

            slope = (h2["price"] - h1["price"]) / gap
            slope_pct = abs(slope) / max(abs(h2["price"]), 1e-9)
            if slope >= 0 or slope_pct > max_line_slope_pct_per_bar:
                continue

            age = i - h2["idx"]
            line_value = h2["price"] + slope * (i - h2["idx"])
            line_prev = h2["price"] + slope * (i - 1 - h2["idx"])

            recent_start = max(h2["idx"] + 1, i - 5 + 1)
            recent_slice = sub.iloc[recent_start : i + 1].copy()
            if recent_slice.empty:
                continue

            recent_range_pct = (recent_slice["high"].max() - recent_slice["low"].min()) / max(line_value, 1e-9)
            recent_vol_ratio = recent_slice["volume"].tail(min(3, len(recent_slice))).mean() / max(float(sub.iloc[i]["vol20"]) if pd.notna(sub.iloc[i]["vol20"]) else np.nan, 1e-9)
            if not np.isfinite(recent_vol_ratio):
                recent_vol_ratio = np.nan

            close_now = float(sub.iloc[i]["close"])
            proximity = abs((line_value - close_now) / max(line_value, 1e-9))

            line_valid_score = 1.0
            proximity_score = float(np.clip(1.0 - (proximity / 0.06), 0.0, 1.0))
            spacing_score = float(np.clip(1.0 - abs(gap - 12.0) / 18.0, 0.0, 1.0))
            age_score = float(np.clip(1.0 - abs(age - 8.0) / 12.0, 0.0, 1.0))
            slope_score = float(np.clip(1.0 - (slope_pct / max_line_slope_pct_per_bar), 0.0, 1.0))
            tightness_score = float(np.clip(1.0 - (recent_range_pct / 0.20), 0.0, 1.0))
            dryup_score = float(np.clip(1.0 - ((recent_vol_ratio - 0.60) / 0.70), 0.0, 1.0)) if np.isfinite(recent_vol_ratio) else 0.0

            setup_score_raw = (
                20.0 * line_valid_score
                + 20.0 * proximity_score
                + 15.0 * spacing_score
                + 15.0 * age_score
                + 10.0 * slope_score
                + 10.0 * tightness_score
                + 10.0 * dryup_score
            )

            row_idx = locs[i]
            out.at[row_idx, "zigzag_high1_date"] = pd.Timestamp(h1["date"])
            out.at[row_idx, "zigzag_high1_price"] = h1["price"]
            out.at[row_idx, "zigzag_high2_date"] = pd.Timestamp(h2["date"])
            out.at[row_idx, "zigzag_high2_price"] = h2["price"]
            out.at[row_idx, "zigzag_high_gap_bars"] = gap
            out.at[row_idx, "zigzag_age_bars"] = age
            out.at[row_idx, "zigzag_line_value"] = line_value
            out.at[row_idx, "zigzag_line_prev"] = line_prev
            out.at[row_idx, "zigzag_line_slope"] = slope
            out.at[row_idx, "zigzag_line_valid"] = True
            out.at[row_idx, "zigzag_setup_raw"] = setup_score_raw
            out.at[row_idx, "setup_pts_line_valid"] = 20.0 * line_valid_score
            out.at[row_idx, "setup_pts_proximity"] = 20.0 * proximity_score
            out.at[row_idx, "setup_pts_spacing"] = 15.0 * spacing_score
            out.at[row_idx, "setup_pts_age"] = 15.0 * age_score
            out.at[row_idx, "setup_pts_slope"] = 10.0 * slope_score
            out.at[row_idx, "setup_pts_tightness"] = 10.0 * tightness_score
            out.at[row_idx, "setup_pts_dryup"] = 10.0 * dryup_score

    out["setup_score_pre"] = out.groupby("symbol", sort=False)["zigzag_setup_raw"].shift(1)
    out["zigzag_line_prevday"] = out.groupby("symbol", sort=False)["zigzag_line_value"].shift(1)
    out["zigzag_line_valid_pre"] = out.groupby("symbol", sort=False)["zigzag_line_valid"].shift(1).fillna(False)
    return out


def compute_zigzag_breakout_scores(
    daily_df: pd.DataFrame | dict[str, pd.DataFrame],
    *,
    cfg: ZigZagBreakoutConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or ZigZagBreakoutConfig()
    daily = _normalize_daily_input(daily_df)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    if getattr(daily["date"].dt, "tz", None) is not None:
        daily["date"] = daily["date"].dt.tz_localize(None)
    daily["date"] = daily["date"].dt.normalize()

    base = compute_breakout_scores_with_diag(daily)
    keep_cols = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "prev_close",
        "atr20",
        "vol20",
        "history_ok",
        "leader_pass",
        "leader_score",
        "rs_pct_21",
        "rs_pct_63",
        "rs_pct_126",
        "rs_rating",
    ]
    base = base[keep_cols].copy()
    out = _compute_zigzag_setup_daily(
        base,
        zigzag_order=cfg.zigzag_order,
        pivot_confirm_bars=cfg.pivot_confirm_bars,
        line_min_gap=cfg.line_min_gap,
        line_max_gap=cfg.line_max_gap,
        max_line_slope_pct_per_bar=cfg.max_line_slope_pct_per_bar,
    )
    g = out.groupby("symbol", sort=False)
    out["recent_low_5_pre"] = (
        g["low"].shift(1).rolling(5, min_periods=3).min().reset_index(level=0, drop=True)
    )
    out["recent_high_5_pre"] = (
        g["high"].shift(1).rolling(5, min_periods=3).max().reset_index(level=0, drop=True)
    )
    out["recent_low_10_pre"] = (
        g["low"].shift(1).rolling(10, min_periods=5).min().reset_index(level=0, drop=True)
    )
    out["pullback_depth_5_pre"] = (
        (out["recent_high_5_pre"] - out["prev_close"]) / out["recent_high_5_pre"].replace(0, np.nan)
    ).clip(lower=0)
    out["leader_pass_min"] = out["leader_score"] >= cfg.leader_min
    out["setup_candidate"] = out["history_ok"] & out["leader_pass_min"] & out["zigzag_line_valid_pre"]
    return out


def _compute_intraday_first_breakouts(
    intraday_df: pd.DataFrame,
    daily_scored: pd.DataFrame,
    cfg: ZigZagBreakoutConfig,
) -> pd.DataFrame:
    if intraday_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "trigger_time",
                "trigger_close",
                "trigger_score",
                "breakout_type",
                "cum_vol_ratio_at_trigger",
                "bar_vol_ratio_at_trigger",
                "move_from_open_at_trigger",
                "dist_above_line_at_trigger",
            ]
        )

    setup_candidates = daily_scored.loc[daily_scored["setup_candidate"]].copy()
    if setup_candidates.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "trigger_time",
                "trigger_close",
                "trigger_score",
                "breakout_type",
                "cum_vol_ratio_at_trigger",
                "bar_vol_ratio_at_trigger",
                "move_from_open_at_trigger",
                "dist_above_line_at_trigger",
            ]
        )

    intra = intraday_df.copy()
    intra["datetime"] = pd.to_datetime(intra["datetime"])
    intra["session_date"] = intra["datetime"].dt.normalize()
    candidate_symbols = setup_candidates["symbol"].dropna().astype(str).unique().tolist()
    intra = intra.loc[intra["symbol"].astype(str).isin(candidate_symbols)].copy()
    if intra.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "trigger_time",
                "trigger_close",
                "trigger_score",
                "breakout_type",
                "cum_vol_ratio_at_trigger",
                "bar_vol_ratio_at_trigger",
                "move_from_open_at_trigger",
                "dist_above_line_at_trigger",
            ]
        )

    intra = intra.sort_values(["symbol", "datetime"], kind="mergesort").reset_index(drop=True)
    intra["slot_index"] = intra.groupby(["symbol", "session_date"], sort=False).cumcount()
    intra["cum_volume_session"] = intra.groupby(["symbol", "session_date"], sort=False)["volume"].cumsum()
    intra["session_open"] = intra.groupby(["symbol", "session_date"], sort=False)["open"].transform("first")
    intra["prev_bar_close_session"] = intra.groupby(["symbol", "session_date"], sort=False)["close"].shift(1)
    intra = _add_same_time_volume_features(intra)
    candidate_dates = (
        setup_candidates.loc[:, ["symbol", "date"]]
        .rename(columns={"date": "session_date"})
        .drop_duplicates()
    )
    intra = intra.merge(candidate_dates, on=["symbol", "session_date"], how="inner")

    ctx_cols = [
        "symbol",
        "date",
        "leader_score",
        "setup_score_pre",
        "zigzag_line_value",
        "zigzag_line_prevday",
        "zigzag_line_valid_pre",
        "atr20",
        "prev_close",
        "recent_low_5_pre",
        "recent_high_5_pre",
        "recent_low_10_pre",
        "pullback_depth_5_pre",
    ]
    daily_ctx = daily_scored[ctx_cols].rename(columns={"date": "session_date"})
    intra = intra.merge(daily_ctx, on=["symbol", "session_date"], how="left", validate="many_to_one")

    line_today = intra["zigzag_line_value"]
    prev_close_session = intra["prev_bar_close_session"]
    breakout_close_confirmed = (
        intra["zigzag_line_valid_pre"].fillna(False)
        & line_today.notna()
        & (intra["close"] >= line_today)
        & (prev_close_session.isna() | (prev_close_session < line_today))
    )

    intra["breakout_any_intraday"] = breakout_close_confirmed
    intra["trigger_close"] = intra["close"]
    dist_above_line = ((intra["trigger_close"] / line_today.replace(0, np.nan)) - 1.0).clip(lower=0)
    close_strength_component = np.clip(dist_above_line / cfg.breakout_dist_ref, 0, 1)
    breakout_component = 0.55 * breakout_close_confirmed.astype(float) + 0.45 * close_strength_component
    breakout_component = pd.Series(breakout_component, index=intra.index).fillna(0.0)

    volume_component = (
        0.60 * np.clip(intra["cum_vol_ratio"] / 2.0, 0, 1)
        + 0.40 * np.clip(intra["bar_vol_ratio"] / 2.0, 0, 1)
    )
    volume_component = pd.Series(volume_component, index=intra.index).fillna(0.0)

    atr20_pct_daily = intra["atr20"] / intra["prev_close"].replace(0, np.nan)
    intraday_bar_range_pct = (intra["high"] - intra["low"]) / intra["close"].replace(0, np.nan)
    range_expand = np.clip((intraday_bar_range_pct / atr20_pct_daily.replace(0, np.nan)) / 0.35, 0, 1)
    move_from_open = (intra["close"] / intra["session_open"].replace(0, np.nan)) - 1.0
    move_from_open_component = np.clip(move_from_open / 0.03, 0, 1)
    price_expansion = 0.50 * range_expand + 0.50 * move_from_open_component
    price_expansion = pd.Series(price_expansion, index=intra.index).fillna(0.0)

    reversal_from_prev_close = np.clip(
        move_from_open.where(intra["slot_index"] == 0, (intra["close"] / intra["prev_close"].replace(0, np.nan)) - 1.0)
        / cfg.reversal_move_ref,
        0,
        1,
    )
    reversal_from_low5 = np.clip(
        (intra["close"] / intra["recent_low_5_pre"].replace(0, np.nan) - 1.0) / cfg.reversal_from_low_ref,
        0,
        1,
    )
    pullback_depth_component = np.clip(intra["pullback_depth_5_pre"] / cfg.pullback_depth_ref, 0, 1)
    reversal_component = (
        0.40 * pd.Series(reversal_from_prev_close, index=intra.index).fillna(0.0)
        + 0.40 * pd.Series(reversal_from_low5, index=intra.index).fillna(0.0)
        + 0.20 * pd.Series(pullback_depth_component, index=intra.index).fillna(0.0)
    )

    pos_in_bar = (intra["close"] - intra["low"]) / (intra["high"] - intra["low"]).replace(0, np.nan)
    hold_component = np.clip((pos_in_bar - 0.50) / 0.50, 0, 1)
    not_too_extended = np.clip(1.0 - (dist_above_line / 0.06), 0, 1)
    hold_component = pd.Series(hold_component, index=intra.index).fillna(0.0)
    not_too_extended = pd.Series(not_too_extended, index=intra.index).fillna(0.0)
    hold_quality = 0.70 * hold_component + 0.30 * not_too_extended

    intra["trigger_score_intraday"] = (
        cfg.trigger_weight_breakout * breakout_component
        + cfg.trigger_weight_volume * volume_component
        + cfg.trigger_weight_price_expansion * price_expansion
        + cfg.trigger_weight_reversal * reversal_component
        + cfg.trigger_weight_hold * hold_quality
    )
    intra["trigger_pts_breakout"] = cfg.trigger_weight_breakout * breakout_component
    intra["trigger_pts_volume"] = cfg.trigger_weight_volume * volume_component
    intra["trigger_pts_price_expansion"] = cfg.trigger_weight_price_expansion * price_expansion
    intra["trigger_pts_reversal"] = cfg.trigger_weight_reversal * reversal_component
    intra["trigger_pts_hold"] = cfg.trigger_weight_hold * hold_quality
    intra["dist_above_line_at_trigger"] = dist_above_line
    intra["move_from_open_at_trigger"] = move_from_open

    trig = intra.loc[intra["breakout_any_intraday"]].copy()
    if trig.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "trigger_time",
                "trigger_close",
                "trigger_score",
                "breakout_type",
                "cum_vol_ratio_at_trigger",
                "bar_vol_ratio_at_trigger",
                "move_from_open_at_trigger",
                "dist_above_line_at_trigger",
                "trigger_pts_breakout",
                "trigger_pts_volume",
                "trigger_pts_price_expansion",
                "trigger_pts_reversal",
                "trigger_pts_hold",
            ]
        )

    trig = trig.sort_values(["symbol", "session_date", "datetime"], kind="mergesort")
    trig["first_breakout_slot"] = trig.groupby(["symbol", "session_date"], sort=False)["slot_index"].transform("min")
    trig = trig.loc[trig["slot_index"] <= (trig["first_breakout_slot"] + cfg.trigger_selection_window_bars)].copy()
    trig = trig.sort_values(
        ["symbol", "session_date", "trigger_score_intraday", "datetime"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    first_trig = (
        trig.groupby(["symbol", "session_date"], as_index=False)
        .first()
        .rename(
            columns={
                "session_date": "date",
                "datetime": "trigger_time",
                "trigger_score_intraday": "trigger_score",
            }
        )
    )
    first_trig["date"] = pd.to_datetime(first_trig["date"]).dt.normalize()
    first_trig["breakout_type"] = "zigzag_diagonal"
    keep_cols = [
        "symbol",
        "date",
        "trigger_time",
        "trigger_close",
        "trigger_score",
        "breakout_type",
        "cum_vol_ratio",
        "bar_vol_ratio",
        "move_from_open_at_trigger",
        "dist_above_line_at_trigger",
        "trigger_pts_breakout",
        "trigger_pts_volume",
        "trigger_pts_price_expansion",
        "trigger_pts_reversal",
        "trigger_pts_hold",
    ]
    first_trig = first_trig[keep_cols].rename(
        columns={
            "cum_vol_ratio": "cum_vol_ratio_at_trigger",
            "bar_vol_ratio": "bar_vol_ratio_at_trigger",
        }
    )
    return first_trig


def build_zigzag_breakout_signal_report(
    daily_df: pd.DataFrame | dict[str, pd.DataFrame],
    intraday_df: pd.DataFrame | dict[str, pd.DataFrame],
    *,
    cfg: ZigZagBreakoutConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or ZigZagBreakoutConfig()
    daily = _normalize_daily_input(daily_df)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    if getattr(daily["date"].dt, "tz", None) is not None:
        daily["date"] = daily["date"].dt.tz_localize(None)
    daily["date"] = daily["date"].dt.normalize()

    daily_scored = compute_zigzag_breakout_scores(daily, cfg=cfg)
    candidate_symbols = set(
        daily_scored.loc[daily_scored["setup_candidate"], "symbol"].dropna().astype(str).unique().tolist()
    )
    if isinstance(intraday_df, dict):
        intraday_input: pd.DataFrame | dict[str, pd.DataFrame] = {
            sym: df_sym for sym, df_sym in intraday_df.items() if sym in candidate_symbols
        }
    else:
        intraday_input = intraday_df.loc[intraday_df["symbol"].astype(str).isin(candidate_symbols)].copy()
    return finalize_zigzag_breakout_signal_report(daily_scored, intraday_input, cfg=cfg)


def finalize_zigzag_breakout_signal_report(
    daily_scored: pd.DataFrame,
    intraday_df: pd.DataFrame | dict[str, pd.DataFrame],
    *,
    cfg: ZigZagBreakoutConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or ZigZagBreakoutConfig()
    intraday = _normalize_intraday_input(intraday_df, session_timezone=cfg.session_timezone)
    first_breakouts = _compute_intraday_first_breakouts(intraday, daily_scored, cfg)
    report = daily_scored.loc[daily_scored["setup_candidate"]].merge(
        first_breakouts,
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )

    report["breakout_type"] = report["breakout_type"].fillna("none")
    report["broke_out"] = report["breakout_type"].ne("none")
    pass_mask = report["history_ok"] & (report["leader_score"] >= cfg.leader_min) & report["broke_out"]
    if cfg.setup_min is not None:
        pass_mask &= report["setup_score_pre"] >= cfg.setup_min
    if cfg.setup_max is not None:
        pass_mask &= report["setup_score_pre"] <= cfg.setup_max
    if cfg.trigger_min is not None:
        pass_mask &= pd.to_numeric(report["trigger_score"], errors="coerce") >= cfg.trigger_min
    report["breakout_signal"] = pass_mask
    report["trigger_time_ny"] = pd.to_datetime(report["trigger_time"], errors="coerce").dt.strftime("%H:%M")

    sort_cols = ["date", "broke_out", "breakout_signal", "trigger_time", "leader_score", "symbol"]
    sort_asc = [True, False, False, True, False, True]
    return report.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
