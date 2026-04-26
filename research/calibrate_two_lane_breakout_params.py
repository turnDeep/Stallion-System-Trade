#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signals.breakout_signal_engine import compute_breakout_scores_with_diag
from signals.breakout_signal_report import (
    DEFAULT_CALIBRATED_PARAMS,
    _compute_intraday_first_breakouts as _compute_standard_first_breakouts,
)
from signals.zigzag_breakout_engine import (
    ZigZagBreakoutConfig,
    _compute_intraday_first_breakouts as _compute_zigzag_first_breakouts,
    compute_zigzag_breakout_scores,
)
from signals.zigzag_entry_engine import ZigZagEntryConfig, apply_zigzag_entry_engine


ANALYSIS_ROOT = REPO_ROOT / "analysis_outputs"
FULL_DATASET_DIR = ANALYSIS_ROOT / "russell3000_full_dataset"
DAILY_10Y_DIR = ANALYSIS_ROOT / "russell3000_daily_10y_dataset"
OUT_DIR = ANALYSIS_ROOT / "two_lane_calibration"
CONFIGS_DIR = REPO_ROOT / "configs"
SESSION_TZ = "America/New_York"


def _log(message: str) -> None:
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _read_sessions() -> pd.DatetimeIndex:
    sessions = pd.read_csv(FULL_DATASET_DIR / "market_sessions_341.csv")
    return pd.DatetimeIndex(pd.to_datetime(sessions["session_date"]).dt.normalize())


def _expand_dates_with_lookback(
    dates: Iterable[pd.Timestamp],
    sessions: pd.DatetimeIndex,
    lookback_sessions: int = 25,
) -> set[pd.Timestamp]:
    session_pos = {d: i for i, d in enumerate(sessions)}
    expanded: set[pd.Timestamp] = set()
    for raw_date in dates:
        d = pd.Timestamp(raw_date).normalize()
        pos = session_pos.get(d)
        if pos is None:
            continue
        start = max(0, pos - lookback_sessions)
        expanded.update(pd.Timestamp(x).normalize() for x in sessions[start : pos + 1])
    return expanded


def _load_daily_history() -> pd.DataFrame:
    daily_path = DAILY_10Y_DIR / "daily_history.parquet"
    universe_path = FULL_DATASET_DIR / "universe.parquet"
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing daily history: {daily_path}")
    if not universe_path.exists():
        raise FileNotFoundError(f"Missing universe: {universe_path}")

    universe = pd.read_parquet(universe_path, columns=["symbol"])
    symbols = set(universe["symbol"].astype(str))
    daily = pd.read_parquet(
        daily_path,
        columns=["symbol", "date", "open", "high", "low", "close", "volume"],
    )
    daily = daily.loc[daily["symbol"].astype(str).isin(symbols)].copy()
    daily["symbol"] = daily["symbol"].astype(str)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    return daily.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)


def _load_intraday_symbol(symbol: str, wanted_dates: set[pd.Timestamp]) -> pd.DataFrame:
    path = FULL_DATASET_DIR / "intraday_5m_by_symbol" / f"{symbol}.parquet"
    if not path.exists() or not wanted_dates:
        return pd.DataFrame(columns=["symbol", "datetime", "session_date", "open", "high", "low", "close", "volume"])

    intra = pd.read_parquet(
        path,
        columns=["symbol", "ts", "open", "high", "low", "close", "volume"],
    )
    intra = intra.rename(columns={"ts": "datetime"})
    intra["datetime"] = pd.to_datetime(intra["datetime"], utc=True, errors="coerce")
    intra["datetime"] = intra["datetime"].dt.tz_convert(SESSION_TZ).dt.tz_localize(None)
    intra["session_date"] = intra["datetime"].dt.normalize()
    intra = intra.loc[intra["session_date"].isin(wanted_dates)].copy()
    if intra.empty:
        return pd.DataFrame(columns=["symbol", "datetime", "session_date", "open", "high", "low", "close", "volume"])
    return intra[["symbol", "datetime", "session_date", "open", "high", "low", "close", "volume"]]


def _candidate_date_map(frame: pd.DataFrame) -> dict[str, set[pd.Timestamp]]:
    mapping: dict[str, set[pd.Timestamp]] = {}
    if frame.empty:
        return mapping
    work = frame[["symbol", "date"]].copy()
    work["symbol"] = work["symbol"].astype(str)
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    for symbol, sub in work.groupby("symbol", sort=False):
        mapping[str(symbol)] = set(pd.Timestamp(x).normalize() for x in sub["date"].tolist())
    return mapping


def _merge_candidate_maps(*maps: dict[str, set[pd.Timestamp]]) -> dict[str, set[pd.Timestamp]]:
    merged: dict[str, set[pd.Timestamp]] = {}
    for mapping in maps:
        for symbol, dates in mapping.items():
            merged.setdefault(symbol, set()).update(dates)
    return merged


def _compute_first_breakouts_chunked(
    *,
    standard_scored: pd.DataFrame,
    zigzag_scored: pd.DataFrame,
    standard_candidates: pd.DataFrame,
    zigzag_candidates: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_std = OUT_DIR / "standard_first_breakouts.parquet"
    cache_zig = OUT_DIR / "zigzag_first_breakouts.parquet"
    if not force and cache_std.exists() and cache_zig.exists():
        _log("Loading cached first-breakout tables")
        return pd.read_parquet(cache_std), pd.read_parquet(cache_zig)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    std_map = _candidate_date_map(standard_candidates)
    zig_map = _candidate_date_map(zigzag_candidates)
    union_map = _merge_candidate_maps(std_map, zig_map)

    standard_groups = standard_scored.groupby("symbol", sort=False)
    zigzag_groups = zigzag_scored.groupby("symbol", sort=False)
    standard_symbols = set(map(str, standard_groups.groups.keys()))
    zigzag_symbols = set(map(str, zigzag_groups.groups.keys()))

    std_rows: list[pd.DataFrame] = []
    zig_rows: list[pd.DataFrame] = []
    zig_cfg = ZigZagBreakoutConfig(leader_min=80.0)
    symbols = sorted(union_map)
    started = time.time()

    for idx, symbol in enumerate(symbols, 1):
        expanded_dates = _expand_dates_with_lookback(union_map[symbol], sessions)
        intra = _load_intraday_symbol(symbol, expanded_dates)
        if not intra.empty:
            std_dates = std_map.get(symbol, set())
            if std_dates and symbol in standard_symbols:
                first = _compute_standard_first_breakouts(intra, standard_groups.get_group(symbol))
                if not first.empty:
                    first["date"] = pd.to_datetime(first["date"]).dt.normalize()
                    std_rows.append(first.loc[first["date"].isin(std_dates)].copy())

            zig_dates = zig_map.get(symbol, set())
            if zig_dates and symbol in zigzag_symbols:
                first = _compute_zigzag_first_breakouts(intra, zigzag_groups.get_group(symbol), zig_cfg)
                if not first.empty:
                    first["date"] = pd.to_datetime(first["date"]).dt.normalize()
                    zig_rows.append(first.loc[first["date"].isin(zig_dates)].copy())

        if idx % 100 == 0 or idx == len(symbols):
            elapsed = max(time.time() - started, 1.0)
            rate = idx / elapsed
            remaining = (len(symbols) - idx) / rate if rate > 0 else math.nan
            _log(
                f"5m breakout scan {idx}/{len(symbols)} "
                f"({idx / len(symbols):.1%}), ETA {remaining / 60:.1f} min"
            )

    std_first = pd.concat(std_rows, ignore_index=True) if std_rows else pd.DataFrame()
    zig_first = pd.concat(zig_rows, ignore_index=True) if zig_rows else pd.DataFrame()
    std_first.to_parquet(cache_std, index=False)
    zig_first.to_parquet(cache_zig, index=False)
    return std_first, zig_first


def _add_forward_outcomes(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    daily_work = daily[["symbol", "date", "close", "high", "low"]].copy()
    daily_work = daily_work.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    g = daily_work.groupby("symbol", sort=False)
    for horizon in [5, 10, 20]:
        daily_work[f"close_fwd_{horizon}d"] = g["close"].shift(-horizon)
    daily_work["entry_day_close"] = daily_work["close"]
    daily_work["entry_day_low"] = daily_work["low"]

    out = events.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.merge(
        daily_work.drop(columns=["close", "high", "low"]),
        on=["symbol", "date"],
        how="left",
        validate="many_to_one",
    )
    entry_price = pd.to_numeric(out["trigger_close"], errors="coerce").fillna(pd.to_numeric(out["entry_day_close"], errors="coerce"))
    out["entry_price_proxy"] = entry_price
    for horizon in [5, 10, 20]:
        out[f"return_{horizon}d"] = pd.to_numeric(out[f"close_fwd_{horizon}d"], errors="coerce") / entry_price - 1.0
    out["same_day_mae_pct"] = pd.to_numeric(out["entry_day_low"], errors="coerce") / entry_price - 1.0
    out["success_20d"] = out["return_20d"] > 0.0
    out["strong_success_20d"] = out["return_20d"] >= 0.05
    return out


def _summarize_mask(events: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    sample = events.loc[mask & events["return_20d"].notna()].copy()
    if sample.empty:
        return {
            "count": 0.0,
            "avg_return_20d": np.nan,
            "median_return_20d": np.nan,
            "win_rate_20d": np.nan,
            "avg_mae_pct": np.nan,
            "score": -np.inf,
        }
    avg_return = float(sample["return_20d"].mean())
    median_return = float(sample["return_20d"].median())
    win_rate = float((sample["return_20d"] > 0).mean())
    avg_mae = float(sample["same_day_mae_pct"].mean())
    score = avg_return + 0.35 * median_return + 0.03 * win_rate + 0.15 * avg_mae
    return {
        "count": float(len(sample)),
        "avg_return_20d": avg_return,
        "median_return_20d": median_return,
        "win_rate_20d": win_rate,
        "avg_mae_pct": avg_mae,
        "score": float(score),
    }


def _sweep_standard(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    rows = []
    leader_values = [80.0, 85.0, 89.0, 90.0, 94.0]
    setup_values = [48.0, 52.0, 55.0, 58.0, 61.0, 65.0, 70.0, 71.0]
    trigger_values = [68.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0]
    cumvol_values = [0.0, 1.0, 1.20, 1.35, 1.50]
    barvol_values = [0.0, 1.0, 1.10]

    leader = pd.to_numeric(events["leader_score"], errors="coerce")
    setup = pd.to_numeric(events["setup_score_pre"], errors="coerce")
    trigger = pd.to_numeric(events["trigger_score"], errors="coerce")
    cumvol = pd.to_numeric(events["cum_vol_ratio_at_trigger"], errors="coerce").fillna(0.0)
    barvol = pd.to_numeric(events["bar_vol_ratio_at_trigger"], errors="coerce").fillna(0.0)

    for leader_min in leader_values:
        for setup_min in setup_values:
            for trigger_min in trigger_values:
                for cumvol_min in cumvol_values:
                    for barvol_min in barvol_values:
                        mask = (
                            (leader >= leader_min)
                            & (setup >= setup_min)
                            & (trigger >= trigger_min)
                            & (cumvol >= cumvol_min)
                            & (barvol >= barvol_min)
                        )
                        summary = _summarize_mask(events, mask)
                        if summary["count"] < 20:
                            continue
                        rows.append(
                            {
                                "leader_min": leader_min,
                                "setup_min": setup_min,
                                "trigger_min": trigger_min,
                                "cum_vol_ratio_min": cumvol_min,
                                "bar_vol_ratio_min": barvol_min,
                                **summary,
                            }
                        )
    return pd.DataFrame(rows).sort_values(["score", "count"], ascending=[False, False]).reset_index(drop=True)


def _sweep_tight(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    rows = []
    leader_values = [80.0, 85.0, 89.0, 90.0, 94.0]
    setup_values = [50.0, 55.0, 57.0, 60.0, 62.0, 65.0]
    trigger_values = [62.0, 65.0, 68.0, 69.0, 72.0, 75.0]
    cumvol_values = [0.0, 1.0, 1.20, 1.35]
    dist_values = [0.50, 0.65, 0.80, 1.00, 1.20]
    gap_values = [0.40, 0.55, 0.70, 0.90, 1.10]

    leader = pd.to_numeric(events["leader_score"], errors="coerce")
    setup = pd.to_numeric(events["setup_score_pre"], errors="coerce")
    trigger = pd.to_numeric(events["trigger_score"], errors="coerce")
    cumvol = pd.to_numeric(events["cum_vol_ratio_at_trigger"], errors="coerce").fillna(0.0)
    dist_norm = pd.to_numeric(events["entry_dist_norm"], errors="coerce")
    gap_norm = pd.to_numeric(events["positive_gap_norm"], errors="coerce")

    for leader_min in leader_values:
        for setup_min in setup_values:
            for trigger_min in trigger_values:
                for cumvol_min in cumvol_values:
                    for dist_max in dist_values:
                        for gap_max in gap_values:
                            mask = (
                                (leader >= leader_min)
                                & (setup >= setup_min)
                                & (trigger >= trigger_min)
                                & (cumvol >= cumvol_min)
                                & (dist_norm <= dist_max)
                                & (gap_norm <= gap_max)
                            )
                            summary = _summarize_mask(events, mask)
                            if summary["count"] < 10:
                                continue
                            rows.append(
                                {
                                    "leader_min": leader_min,
                                    "setup_min": setup_min,
                                    "trigger_min": trigger_min,
                                    "cum_vol_ratio_min": cumvol_min,
                                    "entry_dist_norm_max": dist_max,
                                    "gap_norm_max": gap_max,
                                    **summary,
                                }
                            )
    return pd.DataFrame(rows).sort_values(["score", "count"], ascending=[False, False]).reset_index(drop=True)


def _quantile_params(events: pd.DataFrame) -> dict[str, float]:
    success = events.loc[events["strong_success_20d"] & events["return_20d"].notna()].copy()
    if len(success) < 10:
        success = events.loc[events["success_20d"] & events["return_20d"].notna()].copy()
    if success.empty:
        return {}

    params = {
        "leader_min": float(success["leader_score"].quantile(0.30)),
        "setup_min": float(success["setup_score_pre"].quantile(0.30)),
        "trigger_min": float(success["trigger_score"].quantile(0.30)),
        "cum_vol_ratio_min": float(success["cum_vol_ratio_at_trigger"].fillna(0.0).quantile(0.30)),
        "bar_vol_ratio_min": float(success["bar_vol_ratio_at_trigger"].fillna(0.0).quantile(0.30))
        if "bar_vol_ratio_at_trigger" in success.columns
        else 0.0,
    }
    if "entry_dist_norm" in success.columns:
        params["entry_dist_norm_max"] = float(success["entry_dist_norm"].quantile(0.80))
    if "positive_gap_norm" in success.columns:
        params["gap_norm_max"] = float(success["positive_gap_norm"].quantile(0.80))
    return params


def _build_calibrated_json(std_best: pd.Series | None, tight_best: pd.Series | None) -> dict[str, object]:
    params = json.loads(json.dumps(DEFAULT_CALIBRATED_PARAMS))

    leader_values = []
    if std_best is not None:
        leader_values.append(float(std_best["leader_min"]))
        params["daily_calibration"]["standard_setup_min"] = float(std_best["setup_min"])
        params["intraday_calibration"]["standard_breakout"].update(
            {
                "trigger_min": float(std_best["trigger_min"]),
                "cum_vol_ratio_min": float(std_best["cum_vol_ratio_min"]),
                "bar_vol_ratio_min": float(std_best["bar_vol_ratio_min"]),
            }
        )
    if tight_best is not None:
        leader_values.append(float(tight_best["leader_min"]))
        params["daily_calibration"]["tight_setup_min"] = float(tight_best["setup_min"])
        params["intraday_calibration"]["tight_reversal"].update(
            {
                "trigger_min": float(tight_best["trigger_min"]),
                "cum_vol_ratio_min": float(tight_best["cum_vol_ratio_min"]),
                "entry_dist_norm_max": float(tight_best["entry_dist_norm_max"]),
                "gap_norm_max": float(tight_best["gap_norm_max"]),
            }
        )
    if leader_values:
        params["daily_calibration"]["leader_min"] = float(min(leader_values))
    params["daily_calibration"]["use_setup_max"] = False
    params["metadata"] = {
        "generated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
        "source": "calibrate_two_lane_breakout_params.py",
        "selection_note": "Grid search on 341-session FMP 5m triggers, using 10y FMP daily scores and 20-session forward returns.",
    }
    return params


def _select_best(sweep: pd.DataFrame, *, min_count: int, min_leader: float | None = None) -> pd.Series | None:
    if sweep.empty:
        return None
    work = sweep.loc[sweep["count"] >= min_count].copy()
    if min_leader is not None:
        leader_filtered = work.loc[work["leader_min"] >= min_leader].copy()
        if not leader_filtered.empty:
            work = leader_filtered
    if work.empty:
        return sweep.iloc[0]
    return work.sort_values(["score", "count"], ascending=[False, False]).iloc[0]


def run(force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions = _read_sessions()
    session_min = sessions.min()
    session_max = sessions.max()

    _log("Loading 10y daily history")
    daily = _load_daily_history()
    _log(f"Daily rows: {len(daily):,}, symbols: {daily['symbol'].nunique():,}")

    score_start = session_min - pd.Timedelta(days=360)
    score_end = session_max + pd.Timedelta(days=45)
    daily_for_scoring = daily.loc[(daily["date"] >= score_start) & (daily["date"] <= score_end)].copy()
    _log(
        "Daily scoring window: "
        f"{score_start.date()} -> {score_end.date()} "
        f"({len(daily_for_scoring):,} rows)"
    )

    cache_std_scored = OUT_DIR / "standard_daily_scored_360d.parquet"
    cache_zig_scored = OUT_DIR / "zigzag_daily_scored_360d.parquet"
    if not force and cache_std_scored.exists():
        _log("Loading cached standard daily score table")
        standard_scored = pd.read_parquet(cache_std_scored)
    else:
        _log("Computing standard daily scores")
        standard_scored = compute_breakout_scores_with_diag(daily_for_scoring)
        standard_scored.to_parquet(cache_std_scored, index=False)

    if not force and cache_zig_scored.exists():
        _log("Loading cached zigzag daily score table")
        zigzag_scored = pd.read_parquet(cache_zig_scored)
    else:
        _log("Computing zigzag daily scores")
        zigzag_scored = compute_zigzag_breakout_scores(daily_for_scoring, cfg=ZigZagBreakoutConfig(leader_min=80.0))
        zigzag_scored.to_parquet(cache_zig_scored, index=False)

    standard_scored["date"] = pd.to_datetime(standard_scored["date"]).dt.normalize()
    zigzag_scored["date"] = pd.to_datetime(zigzag_scored["date"]).dt.normalize()

    std_scoped = standard_scored.loc[
        (standard_scored["date"] >= session_min) & (standard_scored["date"] <= session_max)
    ].copy()
    standard_candidates = std_scoped.loc[
        std_scoped["history_ok"].fillna(False).astype(bool)
        & (pd.to_numeric(std_scoped["leader_score"], errors="coerce") >= 80.0)
        & (pd.to_numeric(std_scoped["setup_score_pre"], errors="coerce") >= 48.0)
    ].copy()
    standard_candidates["setup_candidate"] = True

    zigzag_candidates = zigzag_scored.loc[
        (zigzag_scored["date"] >= session_min)
        & (zigzag_scored["date"] <= session_max)
        & zigzag_scored["setup_candidate"].fillna(False).astype(bool)
    ].copy()

    _log(f"Standard broad candidates: {len(standard_candidates):,}")
    _log(f"Zigzag broad candidates: {len(zigzag_candidates):,}")

    std_first, zig_first = _compute_first_breakouts_chunked(
        standard_scored=standard_scored,
        zigzag_scored=zigzag_scored,
        standard_candidates=standard_candidates,
        zigzag_candidates=zigzag_candidates,
        sessions=sessions,
        force=force,
    )
    _log(f"Standard first breakouts: {len(std_first):,}")
    _log(f"Zigzag first breakouts: {len(zig_first):,}")

    std_first_cols = [c for c in std_first.columns if c not in {"symbol", "date"}]
    std_candidates_for_merge = standard_candidates.drop(
        columns=[c for c in std_first_cols if c in standard_candidates.columns],
        errors="ignore",
    )
    std_report = std_candidates_for_merge.merge(
        std_first,
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )
    std_report["broke_out"] = std_report["breakout_type"].fillna("none").ne("none")
    std_events = std_report.loc[std_report["broke_out"]].copy()
    std_events["entry_source"] = "standard_breakout"

    zig_first_cols = [c for c in zig_first.columns if c not in {"symbol", "date"}]
    zig_candidates_for_merge = zigzag_candidates.drop(
        columns=[c for c in zig_first_cols if c in zigzag_candidates.columns],
        errors="ignore",
    )
    zig_report = zig_candidates_for_merge.merge(
        zig_first,
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )
    zig_report["breakout_type"] = zig_report["breakout_type"].fillna("none")
    zig_report["broke_out"] = zig_report["breakout_type"].ne("none")
    broad_entry_cfg = ZigZagEntryConfig(
        leader_min=0.0,
        setup_min=0.0,
        setup_max=None,
        trigger_min=0.0,
        tight_max_entry_dist_norm=999.0,
        tight_max_positive_gap_norm=999.0,
    )
    zig_report = apply_zigzag_entry_engine(zig_report, broad_entry_cfg)
    zig_events = zig_report.loc[zig_report["broke_out"]].copy()
    zig_events["entry_source"] = "tight_reversal"

    std_events = _add_forward_outcomes(std_events, daily)
    zig_events = _add_forward_outcomes(zig_events, daily)

    std_events.to_csv(OUT_DIR / "standard_events_with_outcomes.csv", index=False)
    zig_events.to_csv(OUT_DIR / "tight_reversal_events_with_outcomes.csv", index=False)

    _log("Sweeping standard breakout parameters")
    std_sweep = _sweep_standard(std_events)
    std_sweep.to_csv(OUT_DIR / "standard_param_sweep.csv", index=False)

    _log("Sweeping tight reversal parameters")
    tight_sweep = _sweep_tight(zig_events)
    tight_sweep.to_csv(OUT_DIR / "tight_reversal_param_sweep.csv", index=False)

    std_best = _select_best(std_sweep, min_count=500)
    standard_leader_floor = None if std_best is None else float(std_best["leader_min"])
    tight_best = _select_best(tight_sweep, min_count=100, min_leader=standard_leader_floor)

    calibrated = _build_calibrated_json(std_best, tight_best)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "calibrated_params.json").write_text(json.dumps(calibrated, indent=2), encoding="utf-8")
    (CONFIGS_DIR / "calibrated_params.json").write_text(json.dumps(calibrated, indent=2), encoding="utf-8")

    quantile_report = {
        "standard_quantile_params": _quantile_params(std_events),
        "tight_reversal_quantile_params": _quantile_params(zig_events),
        "standard_best_grid": {} if std_best is None else std_best.to_dict(),
        "tight_reversal_best_grid": {} if tight_best is None else tight_best.to_dict(),
        "standard_event_count": int(len(std_events)),
        "tight_reversal_event_count": int(len(zig_events)),
    }
    (OUT_DIR / "calibration_report.json").write_text(json.dumps(quantile_report, indent=2, default=str), encoding="utf-8")

    _log("Calibration complete")
    if std_best is not None:
        _log(f"Best standard: {std_best.to_dict()}")
    if tight_best is not None:
        _log(f"Best tight: {tight_best.to_dict()}")
    _log(f"Wrote {CONFIGS_DIR / 'calibrated_params.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate two-lane breakout parameters from FMP 10y daily + 341-session 5m data.")
    parser.add_argument("--force", action="store_true", help="Recompute cached daily scores and first breakout tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
