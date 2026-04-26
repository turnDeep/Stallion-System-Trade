from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ZigZagEntryConfig:
    leader_min: float = 89.0
    setup_min: float = 57.0
    setup_max: float | None = None
    trigger_min: float = 69.0
    cum_vol_ratio_min: float = 0.0
    bar_vol_ratio_min: float = 0.0
    tight_max_entry_dist_norm: float = 0.80
    tight_max_positive_gap_norm: float = 0.70
    tight_override_max_entry_dist_norm: float = 0.35
    tight_override_min_setup: float = 62.0
    tight_override_min_trigger: float = 80.0


def _clip01(values: pd.Series | np.ndarray) -> pd.Series:
    index = values.index if isinstance(values, pd.Series) else None
    return pd.Series(np.clip(np.asarray(values, dtype="float64"), 0.0, 1.0), index=index)


def _numeric_column(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def apply_zigzag_entry_engine(
    report: pd.DataFrame,
    cfg: ZigZagEntryConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or ZigZagEntryConfig()
    out = report.copy()

    leader = _numeric_column(out, "leader_score")
    setup = _numeric_column(out, "setup_score_pre")
    trigger = _numeric_column(out, "trigger_score")
    trigger_close = _numeric_column(out, "trigger_close")
    line_value = _numeric_column(out, "zigzag_line_value")
    prev_close = _numeric_column(out, "prev_close")
    open_px = _numeric_column(out, "open")
    adr20_pct = _numeric_column(out, "adr20_pct")
    cum_vol_ratio = _numeric_column(out, "cum_vol_ratio_at_trigger", 0.0)
    bar_vol_ratio = _numeric_column(out, "bar_vol_ratio_at_trigger", 0.0)
    pts_hold = _numeric_column(out, "trigger_pts_hold", 0.0)

    out["entry_dist_above_line_pct"] = (trigger_close / line_value.replace(0, np.nan)) - 1.0
    out["gap_pct"] = (open_px / prev_close.replace(0, np.nan)) - 1.0
    out["positive_gap_pct"] = out["gap_pct"].clip(lower=0.0)
    out["entry_dist_norm"] = out["entry_dist_above_line_pct"] / adr20_pct.replace(0, np.nan)
    out["positive_gap_norm"] = out["positive_gap_pct"] / adr20_pct.replace(0, np.nan)

    setup_pass = setup >= cfg.setup_min
    if cfg.setup_max is not None:
        setup_pass &= setup <= cfg.setup_max

    base_signal = (
        out["history_ok"].fillna(False).astype(bool)
        & out["broke_out"].fillna(False).astype(bool)
        & (leader >= cfg.leader_min)
        & setup_pass
        & (trigger >= cfg.trigger_min)
        & (cum_vol_ratio.fillna(0.0) >= cfg.cum_vol_ratio_min)
        & (bar_vol_ratio.fillna(0.0) >= cfg.bar_vol_ratio_min)
    )

    tight_lane = (
        base_signal
        & (out["entry_dist_norm"] <= cfg.tight_max_entry_dist_norm)
        & (out["positive_gap_norm"] <= cfg.tight_max_positive_gap_norm)
    )

    tight_override = (
        tight_lane
        & (out["entry_dist_norm"] <= cfg.tight_override_max_entry_dist_norm)
        & (setup >= cfg.tight_override_min_setup)
        & (trigger >= cfg.tight_override_min_trigger)
    )

    leader_norm = _clip01((leader - cfg.leader_min) / 12.0)
    trigger_norm = _clip01((trigger - cfg.trigger_min) / 16.0)
    setup_norm = _clip01((setup - cfg.setup_min) / 18.0)
    entry_proximity_bonus = _clip01(1.0 - (out["entry_dist_norm"] / cfg.tight_max_entry_dist_norm))
    gap_bonus = _clip01(1.0 - (out["positive_gap_norm"] / cfg.tight_max_positive_gap_norm))
    hold_norm = _clip01(pts_hold / 10.0)

    out["lane_tight_score"] = (
        15.0 * leader_norm
        + 30.0 * setup_norm
        + 15.0 * trigger_norm
        + 25.0 * entry_proximity_bonus
        + 15.0 * gap_bonus
    )
    out["same_day_priority_score"] = np.where(
        tight_lane,
        out["lane_tight_score"] + 5.0 * hold_norm,
        np.nan,
    )

    out["entry_base_signal"] = base_signal
    out["entry_lane_tight"] = tight_lane.fillna(False)
    out["entry_lane"] = np.where(out["entry_lane_tight"], "tight_reversal", "none")
    out["entry_stop_policy"] = np.select(
        [tight_override, out["entry_lane_tight"]],
        ["ignore_stop_limit", "respect_stop_limit"],
        default="none",
    )
    out["entry_signal"] = out["entry_lane_tight"]

    out["entry_filter_reason"] = np.select(
        [
            ~base_signal,
            base_signal & ~tight_lane & (out["entry_dist_norm"] > cfg.tight_max_entry_dist_norm),
            base_signal & ~tight_lane & (out["positive_gap_norm"] > cfg.tight_max_positive_gap_norm),
            base_signal & ~tight_lane,
        ],
        [
            "failed_base_signal",
            "entry_distance_norm_too_high",
            "gap_norm_too_high",
            "failed_tight_lane_shape",
        ],
        default="passed_entry_engine",
    )

    sort_cols = [
        "date",
        "entry_signal",
        "same_day_priority_score",
        "leader_score",
        "trigger_time",
        "symbol",
    ]
    sort_asc = [True, False, False, False, True, True]
    return out.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
