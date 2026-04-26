from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from pathlib import Path

# Handle numba
try:
    from numba import njit
except Exception:
    njit = None

def _diag_resistance_from_swings_np_impl(
    high: np.ndarray,
    symbol_codes: np.ndarray,
    swing_left: int = 2,
    swing_right: int = 2,
    diag_max_age: int = 15,
    diag_min_gap: int = 2,
    diag_max_gap: int = 15,
    max_slope_pct_per_bar: float = 0.01,
    allow_rising_diag: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(high)
    diag_res = np.full(n, np.nan, dtype=np.float64)
    diag_res_prev = np.full(n, np.nan, dtype=np.float64)
    diag_slope = np.full(n, np.nan, dtype=np.float64)
    diag_valid = np.zeros(n, dtype=np.bool_)
    swing_hi = np.zeros(n, dtype=np.bool_)
    anchor_gap = np.full(n, np.nan, dtype=np.float64)

    start = 0
    while start < n:
        end = start + 1
        sym = symbol_codes[start]
        while end < n and symbol_codes[end] == sym:
            end += 1

        h = high[start:end]
        m = end - start
        sh = np.zeros(m, dtype=np.bool_)

        for i in range(swing_left, m - swing_right):
            v = h[i]
            ok = True
            for k in range(1, swing_left + 1):
                if not (v > h[i - k]):
                    ok = False
                    break
            if ok:
                for k in range(1, swing_right + 1):
                    if not (v >= h[i + k]):
                        ok = False
                        break
            sh[i] = ok
        swing_hi[start:end] = sh

        prev_idx = -1
        prev_px = np.nan
        last_idx = -1
        last_px = np.nan

        for i in range(m):
            if sh[i]:
                prev_idx, prev_px = last_idx, last_px
                last_idx, last_px = i, h[i]

            if prev_idx >= 0 and last_idx >= 0 and i >= last_idx:
                gap = last_idx - prev_idx
                age = i - last_idx
                if diag_min_gap <= gap <= diag_max_gap and age <= diag_max_age:
                    slope = (last_px - prev_px) / gap
                    denom = abs(last_px) if abs(last_px) > 1e-12 else 1.0
                    slope_pct = abs(slope) / denom
                    slope_ok = slope_pct <= max_slope_pct_per_bar
                    if not allow_rising_diag and slope > 0:
                        slope_ok = False

                    if slope_ok:
                        diag_res[start + i] = last_px + slope * (i - last_idx)
                        diag_res_prev[start + i] = last_px + slope * (i - 1 - last_idx)
                        diag_slope[start + i] = slope
                        diag_valid[start + i] = True
                        anchor_gap[start + i] = gap
        start = end
    return diag_res, diag_res_prev, diag_slope, diag_valid, swing_hi, anchor_gap

if njit is not None:
    _diag_resistance_from_swings_np = njit(cache=True)(_diag_resistance_from_swings_np_impl)
else:
    _diag_resistance_from_swings_np = _diag_resistance_from_swings_np_impl

def compute_breakout_scores_with_diag(
    df: pd.DataFrame,
    pivot_window: int = 20,
    min_history: int = 126,
    leader_weights: tuple[float, float, float] = (0.45, 0.35, 0.20),
    total_weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
    normal_setup_min: float = 58.0,
    normal_trigger_min: float = 70.0,
    override_setup_min: float = 48.0,
    override_trigger_min: float = 75.0,
    leader_rank_min: float = 80.0,
    leader_fallback_roc21_min: float = 0.15,
    leader_fallback_roc63_min: float = 0.30,
    leader_fallback_roc126_min: float = 0.50,
    swing_left: int = 2,
    swing_right: int = 2,
    diag_max_age: int = 15,
    diag_min_gap: int = 2,
    diag_max_gap: int = 15,
    max_slope_pct_per_bar: float = 0.01,
    allow_rising_diag: bool = True,
    near_resistance_floor: float = 0.04,
    near_resistance_atr_mult: float = 1.75,
) -> pd.DataFrame:
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    g = out.groupby("symbol", sort=False)

    def gshift(col: str, n: int = 1) -> pd.Series:
        return g[col].shift(n)

    def groll_mean(col: str, w: int, minp: int | None = None) -> pd.Series:
        return g[col].rolling(w, min_periods=minp or w).mean().reset_index(level=0, drop=True)

    def by_symbol_shift(series: pd.Series, n: int = 1) -> pd.Series:
        return series.groupby(out["symbol"], sort=False).shift(n)

    out["prev_close"] = gshift("close", 1)
    out["prev_high"] = gshift("high", 1)
    out["rolling_high_3_pre"] = (
        out["prev_high"]
        .groupby(out["symbol"], sort=False)
        .rolling(3, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )

    tr = np.maximum.reduce([
        (out["high"] - out["low"]).to_numpy(),
        (out["high"] - out["prev_close"]).abs().fillna(0.0).to_numpy(),
        (out["low"] - out["prev_close"]).abs().fillna(0.0).to_numpy(),
    ])
    out["tr"] = tr

    out["sma10"] = groll_mean("close", 10)
    out["sma20"] = groll_mean("close", 20)
    out["atr5"] = groll_mean("tr", 5)
    out["atr20"] = groll_mean("tr", 20)
    range_pct = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["adr20_pct"] = (
        range_pct.groupby(out["symbol"], sort=False)
        .rolling(20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out["vol5"] = groll_mean("volume", 5)
    out["vol20"] = groll_mean("volume", 20)

    out["range5"] = (
        g["high"].rolling(5, min_periods=5).max().reset_index(level=0, drop=True)
        - g["low"].rolling(5, min_periods=5).min().reset_index(level=0, drop=True)
    )
    out["range20"] = (
        g["high"].rolling(20, min_periods=20).max().reset_index(level=0, drop=True)
        - g["low"].rolling(20, min_periods=20).min().reset_index(level=0, drop=True)
    )

    out["sma10_lag3"] = gshift("sma10", 3)
    out["sma20_lag5"] = gshift("sma20", 5)

    above10 = (out["close"] >= out["sma10"]).astype("float64")
    out["close_above_sma10_10d"] = (
        above10.groupby(out["symbol"], sort=False)
        .rolling(10, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out["dist_sma10_atr10"] = (
        ((out["close"] - out["sma10"]).abs() / out["atr20"].replace(0, np.nan))
        .groupby(out["symbol"], sort=False)
        .rolling(10, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out["deep_below_sma10_10d"] = (
        (out["close"] < out["sma10"] * 0.985).astype("float64")
        .groupby(out["symbol"], sort=False)
        .rolling(10, min_periods=10)
        .sum()
        .reset_index(level=0, drop=True)
    )

    above20 = (out["close"] >= out["sma20"]).astype("float64")
    out["close_above_sma20_15d"] = (
        above20.groupby(out["symbol"], sort=False)
        .rolling(15, min_periods=15)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out["dist_sma20_atr15"] = (
        ((out["close"] - out["sma20"]).abs() / out["atr20"].replace(0, np.nan))
        .groupby(out["symbol"], sort=False)
        .rolling(15, min_periods=15)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out["deep_below_sma20_15d"] = (
        (out["close"] < out["sma20"] * 0.98).astype("float64")
        .groupby(out["symbol"], sort=False)
        .rolling(15, min_periods=15)
        .sum()
        .reset_index(level=0, drop=True)
    )

    out["pivot_high"] = (
        g["high"].shift(1).rolling(pivot_window, min_periods=pivot_window).max().reset_index(level=0, drop=True)
    )
    out["high10"] = (
        g["high"].shift(1).rolling(10, min_periods=10).max().reset_index(level=0, drop=True)
    )

    symbol_codes = pd.factorize(out["symbol"], sort=False)[0]
    diag_res, diag_res_prev, diag_slope, diag_valid, swing_hi, anchor_gap = _diag_resistance_from_swings_np(
        out["high"].to_numpy(),
        symbol_codes,
        swing_left=swing_left,
        swing_right=swing_right,
        diag_max_age=diag_max_age,
        diag_min_gap=diag_min_gap,
        diag_max_gap=diag_max_gap,
        max_slope_pct_per_bar=max_slope_pct_per_bar,
        allow_rising_diag=allow_rising_diag,
    )

    out["diag_resistance"] = diag_res
    out["diag_resistance_prev"] = diag_res_prev
    out["diag_slope"] = diag_slope
    out["diag_valid"] = diag_valid
    out["is_swing_high"] = swing_hi
    out["diag_anchor_gap"] = anchor_gap

    diag_res_s = pd.Series(diag_res, index=out.index)
    diag_res_prev_s = pd.Series(diag_res_prev, index=out.index)

    out["roc_21"] = g["close"].pct_change(21)
    out["roc_63"] = g["close"].pct_change(63)
    out["roc_126"] = g["close"].pct_change(126)

    # Requested RS method: use cross-sectional RS percentiles for each horizon,
    # then blend 21/63/126-day percentiles into a single RS rating.
    out["rs_pct_21"] = out.groupby("date")["roc_21"].rank(pct=True, method="average") * 100.0
    out["rs_pct_63"] = out.groupby("date")["roc_63"].rank(pct=True, method="average") * 100.0
    out["rs_pct_126"] = out.groupby("date")["roc_126"].rank(pct=True, method="average") * 100.0
    out["rank_roc_21"] = out["rs_pct_21"]
    out["rank_roc_63"] = out["rs_pct_63"]
    out["rank_roc_126"] = out["rs_pct_126"]

    w21, w63, w126 = leader_weights
    out["rs_rating"] = (
        w21 * out["rs_pct_21"].fillna(0.0)
        + w63 * out["rs_pct_63"].fillna(0.0)
        + w126 * out["rs_pct_126"].fillna(0.0)
    )
    out["leader_score"] = out["rs_rating"]
    out["leader_pass_rank"] = out["leader_score"] >= leader_rank_min
    out["leader_pass_fallback"] = (
        (out["roc_21"] >= leader_fallback_roc21_min)
        & (out["roc_63"] >= leader_fallback_roc63_min)
        & (out["roc_126"] >= leader_fallback_roc126_min)
    )
    out["leader_pass"] = out["leader_pass_rank"] | out["leader_pass_fallback"]

    ma_regime_raw = (
        (out["close"] >= out["sma10"])
        & (out["close"] >= out["sma20"])
        & (out["sma10"] > out["sma10_lag3"])
        & (out["sma20"] > out["sma20_lag5"])
    ).astype(float)

    surf10_raw = (
        0.40 * np.clip(out["close_above_sma10_10d"] / 0.70, 0, 1)
        + 0.35 * np.clip(1 - (out["dist_sma10_atr10"] / 0.80), 0, 1)
        + 0.25 * np.clip(1 - (out["deep_below_sma10_10d"] / 2.0), 0, 1)
    )
    surf20_raw = (
        0.40 * np.clip(out["close_above_sma20_15d"] / 0.73, 0, 1)
        + 0.35 * np.clip(1 - (out["dist_sma20_atr15"] / 1.00), 0, 1)
        + 0.25 * np.clip(1 - (out["deep_below_sma20_15d"] / 2.0), 0, 1)
    )
    surf_quality_raw = np.maximum(surf10_raw, surf20_raw)

    atr_ratio_raw = out["atr5"] / out["atr20"].replace(0, np.nan)
    range_ratio_raw = out["range5"] / out["range20"].replace(0, np.nan)
    vol_ratio_raw = out["vol5"] / out["vol20"].replace(0, np.nan)

    dist_to_hpivot_raw = ((out["pivot_high"] - out["close"]) / out["pivot_high"].replace(0, np.nan)).clip(lower=0)
    dist_to_dpivot_raw = ((diag_res_s - out["close"]) / diag_res_s.replace(0, np.nan)).clip(lower=0).where(out["diag_valid"], np.nan)
    dist_to_any_res_raw = pd.concat([dist_to_hpivot_raw, dist_to_dpivot_raw], axis=1).min(axis=1, skipna=True)
    drawdown_10_raw = ((out["high10"] - out["close"]) / out["high10"].replace(0, np.nan)).clip(lower=0)
    contraction_raw = 0.50 * np.clip(1 - ((atr_ratio_raw - 0.60) / 0.40), 0, 1) + 0.50 * np.clip(1 - ((range_ratio_raw - 0.60) / 0.40), 0, 1)
    dryup_raw = np.clip(1 - ((vol_ratio_raw - 0.50) / 0.50), 0, 1)
    near_resistance_raw = np.clip(1 - (dist_to_any_res_raw / 0.03), 0, 1)
    shallow_dd_raw = np.clip(1 - (drawdown_10_raw / 0.08), 0, 1)
    diag_quality_raw = np.where(out["diag_valid"], np.clip(1 - ((np.abs(out["diag_slope"]) / diag_res_s.abs().replace(0, np.nan)) / max_slope_pct_per_bar), 0, 1), 0.0)

    # Weights
    out["setup_score_raw"] = (
        18.0 * ma_regime_raw.fillna(0.0)
        + 24.0 * pd.Series(surf_quality_raw, index=out.index).fillna(0.0)
        + 12.0 * contraction_raw.fillna(0.0)
        +  6.0 * dryup_raw.fillna(0.0)
        + 20.0 * near_resistance_raw.fillna(0.0)
        + 12.0 * shallow_dd_raw.fillna(0.0)
        +  8.0 * diag_quality_raw
    )

    ma_regime_pre = by_symbol_shift(pd.Series(ma_regime_raw, index=out.index), 1)
    surf10_pre = by_symbol_shift(pd.Series(surf10_raw, index=out.index), 1)
    surf20_pre = by_symbol_shift(pd.Series(surf20_raw, index=out.index), 1)
    surf_quality_pre = np.maximum(surf10_pre, surf20_pre)
    contraction_pre = by_symbol_shift(pd.Series(contraction_raw, index=out.index), 1)
    dryup_pre = by_symbol_shift(pd.Series(dryup_raw, index=out.index), 1)
    diag_quality_pre = by_symbol_shift(pd.Series(diag_quality_raw, index=out.index), 1)

    approach_arr = np.maximum.reduce([
        out["prev_close"].fillna(-1).to_numpy(),
        out["prev_high"].fillna(-1).to_numpy(),
        out["rolling_high_3_pre"].fillna(-1).to_numpy(),
    ])
    out["approach_price_pre"] = pd.Series(approach_arr, index=out.index).replace(-1, np.nan)

    atr20_pre = g["atr20"].shift(1)
    atr20_pct_pre = atr20_pre / out["prev_close"].replace(0, np.nan)
    prox_band_pre = np.maximum(near_resistance_floor, near_resistance_atr_mult * atr20_pct_pre.fillna(0.0))
    
    dist_to_hpivot_pre = ((out["pivot_high"] - out["approach_price_pre"]) / out["pivot_high"].replace(0, np.nan)).clip(lower=0)
    dist_to_dpivot_pre = ((diag_res_prev_s - out["approach_price_pre"]) / diag_res_prev_s.replace(0, np.nan)).clip(lower=0).where(diag_res_prev_s.notna(), np.nan)
    dist_to_any_res_pre = pd.concat([dist_to_hpivot_pre, dist_to_dpivot_pre], axis=1).min(axis=1, skipna=True)
    near_resistance_pre = np.clip(1 - (dist_to_any_res_pre / pd.Series(prox_band_pre, index=out.index)), 0, 1)

    drawdown_10_pre = ((out["high10"] - out["prev_close"]) / out["high10"].replace(0, np.nan)).clip(lower=0)
    shallow_dd_pre = np.clip(1 - (drawdown_10_pre / 0.08), 0, 1)

    out["setup_score_pre"] = (
        18.0 * ma_regime_pre.fillna(0.0)
        + 24.0 * pd.Series(surf_quality_pre, index=out.index).fillna(0.0)
        + 12.0 * contraction_pre.fillna(0.0)
        +  6.0 * dryup_pre.fillna(0.0)
        + 20.0 * pd.Series(near_resistance_pre, index=out.index).fillna(0.0)
        + 12.0 * pd.Series(shallow_dd_pre, index=out.index).fillna(0.0)
        +  8.0 * diag_quality_pre.fillna(0.0)
    )
    out["setup_score"] = out["setup_score_pre"]

    breakout_horizontal = (out["close"] > out["pivot_high"]) & (out["prev_close"] <= out["pivot_high"])
    breakout_diagonal = out["diag_valid"] & (out["close"] > out["diag_resistance"]) & (out["prev_close"] <= out["diag_resistance_prev"])
    breakout_today = (breakout_horizontal | breakout_diagonal).astype(float)

    dist_above_h = ((out["close"] / out["pivot_high"].replace(0, np.nan)) - 1).clip(lower=0)
    dist_above_d = ((out["close"] / diag_res_s.replace(0, np.nan)) - 1).clip(lower=0).where(out["diag_valid"], np.nan)
    dist_above_any = pd.concat([dist_above_h, dist_above_d], axis=1).max(axis=1, skipna=True)

    breakout_strength = np.clip(dist_above_any / 0.02, 0, 1)
    breakout_component = 0.70 * breakout_today + 0.30 * breakout_strength
    vol_expand = np.clip((out["volume"] / out["vol20"].replace(0, np.nan)) / 1.50, 0, 1)
    bar_range_pct = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    atr20_pct = out["atr20"] / out["close"].replace(0, np.nan)
    range_expand = np.clip((bar_range_pct / atr20_pct.replace(0, np.nan)) / 1.20, 0, 1)
    day_return = (out["close"] / out["open"].replace(0, np.nan)) - 1
    day_return_component = np.clip(day_return / 0.03, 0, 1)
    price_expansion = 0.50 * range_expand + 0.50 * day_return_component
    risk_frac = (out["close"] - out["low"]) / out["close"].replace(0, np.nan)
    risk_component = np.clip(1 - (risk_frac / atr20_pct.replace(0, np.nan)), 0, 1)
    pos_in_bar = ((out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan))
    hold_component = np.clip((pos_in_bar - 0.50) / 0.50, 0, 1)
    not_too_extended = np.clip(1 - (dist_above_any / 0.03), 0, 1)

    out["trigger_score"] = (
        35.0 * breakout_component
        + 25.0 * vol_expand
        + 15.0 * price_expansion
        + 15.0 * risk_component
        + 10.0 * (0.70 * hold_component + 0.30 * not_too_extended)
    )

    out["breakout_type"] = np.select(
        [ breakout_horizontal & breakout_diagonal, breakout_horizontal, breakout_diagonal ],
        ["both", "horizontal", "diagonal"], default="none"
    )

    history_count = g.cumcount() + 1
    out["history_ok"] = history_count >= min_history
    out["normal_pass"] = (out["setup_score_pre"] >= normal_setup_min) & (out["trigger_score"] >= normal_trigger_min)
    out["strong_trigger_override"] = (out["setup_score_pre"] >= override_setup_min) & (out["trigger_score"] >= override_trigger_min) & (out["breakout_type"] != "none")

    out["breakout_signal"] = out["history_ok"] & out["leader_pass"] & (out["normal_pass"] | out["strong_trigger_override"])

    a, b, c = total_weights
    out["total_score"] = a * out["leader_score"].fillna(0.0) + b * out["setup_score_pre"].fillna(0.0) + c * out["trigger_score"].fillna(0.0)
    return out

def main():
    daily_history = pd.read_pickle(r"C:\Users\plane\BreakOut\analysis_outputs\russell3000_full_dataset\daily_history.pkl")
    targets = [
        ("AXTI", "2026-02-20"),
        ("TNGX", "2026-03-05"),
        ("ERAS", "2026-01-07"),
        ("BW", "2026-03-04"),
        ("FSLY", "2026-02-12"),
        ("AAOI", "2026-02-27")
    ]
    all_rows = []
    for sym, date in targets:
        df = daily_history.get(sym)
        if df is not None:
            df = df.copy()
            df.index.name = "date"
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df["symbol"] = sym
            all_rows.append(df)
    
    if not all_rows:
        return

    full_df = pd.concat(all_rows, ignore_index=True)
    scored = compute_breakout_scores_with_diag(full_df)
    
    print("\nVerification Results (V4 Code):")
    print("-" * 150)
    print(f"{'Symbol':<6} | {'Date':<10} | {'Setup_Pre':<10} | {'Trigger':<7} | {'Type':<12} | {'Sgnl':<5} | {'Leader':<6} | {'Normal':<6} | {'Ovrd':<6} | {'Reason'}")
    print("-" * 150)
    
    for sym, date in targets:
        row = scored[(scored["symbol"] == sym) & (scored["date"].astype(str).str.contains(date))]
        if not row.empty:
            r = row.iloc[0]
            reason = "PASS" if r["breakout_signal"] else "FAIL"
            if not r["breakout_signal"]:
                fails = []
                if not r["leader_pass"]: fails.append(f"Leader")
                if not (r["normal_pass"] or r["strong_trigger_override"]): 
                    fails.append(f"Setup/Trig({r['setup_score_pre']:.1f}/{r['trigger_score']:.1f})")
                if r["breakout_type"] == "none": fails.append("NotBreakout")
                reason = "FAIL: " + ", ".join(fails)
            
            print(f"{sym:<6} | {date:<10} | {r['setup_score_pre']:>10.1f} | {r['trigger_score']:>7.1f} | {r['breakout_type']:<12} | {str(r['breakout_signal']):<5} | {str(r['leader_pass']):<6} | {str(r['normal_pass']):<6} | {str(r['strong_trigger_override']):<6} | {reason}")
        else:
            print(f"{sym:<6} | {date:<10} | {'N/A':>10} | {'N/A':>7} | {'N/A':<12} | {'N/A':<5} | Data missing.")

if __name__ == "__main__":
    main()
