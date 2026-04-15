from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ZigZagEntryConfig:
    leader_min: float = 89.0
    setup_min: float = 57.0
    setup_max: float = 67.0
    trigger_min: float = 69.0
    tight_max_entry_dist_above_line_pct: float = 0.10
    tight_max_positive_gap_pct: float = 0.07
    tight_override_max_entry_dist_above_line_pct: float = 0.04
    tight_override_min_setup: float = 62.0
    tight_override_min_trigger: float = 80.0
    power_min_entry_dist_above_line_pct: float = 0.15
    power_max_entry_dist_above_line_pct: float = 1.25
    power_max_positive_gap_pct: float = 0.09
    power_setup_max: float = 60.9
    power_min_breakout_pts: float = 42.0
    power_min_price_expansion_pts: float = 6.0
    power_min_reversal_pts: float = 6.0
    power_low_gap_max_pct: float = 0.02
    power_leader_override_min: float = 95.0
    power_gap_tolerant_max_pct: float = 0.07
    power_gap_tolerant_min_trigger: float = 69.5
    power_gap_tolerant_min_price_expansion_pts: float = 7.0
    power_gap_tolerant_min_reversal_pts: float = 10.0


def _clip01(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(np.clip(np.asarray(values, dtype="float64"), 0.0, 1.0))


def apply_zigzag_entry_engine(
    report: pd.DataFrame,
    cfg: ZigZagEntryConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or ZigZagEntryConfig()
    out = report.copy()

    leader = pd.to_numeric(out["leader_score"], errors="coerce")
    setup = pd.to_numeric(out["setup_score_pre"], errors="coerce")
    trigger = pd.to_numeric(out["trigger_score"], errors="coerce")
    trigger_close = pd.to_numeric(out["trigger_close"], errors="coerce")
    line_value = pd.to_numeric(out["zigzag_line_value"], errors="coerce")
    prev_close = pd.to_numeric(out["prev_close"], errors="coerce")
    open_px = pd.to_numeric(out["open"], errors="coerce")
    pts_breakout = pd.to_numeric(out["trigger_pts_breakout"], errors="coerce")
    pts_price_expansion = pd.to_numeric(out["trigger_pts_price_expansion"], errors="coerce")
    pts_reversal = pd.to_numeric(out["trigger_pts_reversal"], errors="coerce")
    pts_hold = pd.to_numeric(out["trigger_pts_hold"], errors="coerce")

    out["entry_dist_above_line_pct"] = (trigger_close / line_value.replace(0, np.nan)) - 1.0
    out["gap_pct"] = (open_px / prev_close.replace(0, np.nan)) - 1.0
    out["positive_gap_pct"] = out["gap_pct"].clip(lower=0.0)

    base_signal = (
        out["history_ok"].fillna(False).astype(bool)
        & out["broke_out"].fillna(False).astype(bool)
        & (leader >= cfg.leader_min)
        & setup.between(cfg.setup_min, cfg.setup_max)
        & (trigger >= cfg.trigger_min)
    )

    tight_lane = (
        base_signal
        & (out["entry_dist_above_line_pct"] <= cfg.tight_max_entry_dist_above_line_pct)
        & (out["positive_gap_pct"] <= cfg.tight_max_positive_gap_pct)
    )
    tight_override = (
        tight_lane
        & (out["entry_dist_above_line_pct"] <= cfg.tight_override_max_entry_dist_above_line_pct)
        & (setup >= cfg.tight_override_min_setup)
        & (trigger >= cfg.tight_override_min_trigger)
    )

    power_lane = (
        base_signal
        & out["entry_dist_above_line_pct"].between(
            cfg.power_min_entry_dist_above_line_pct,
            cfg.power_max_entry_dist_above_line_pct,
            inclusive="both",
        )
        & (out["positive_gap_pct"] <= cfg.power_max_positive_gap_pct)
        & (setup <= cfg.power_setup_max)
        & (pts_breakout >= cfg.power_min_breakout_pts)
        & (pts_price_expansion >= cfg.power_min_price_expansion_pts)
        & (pts_reversal >= cfg.power_min_reversal_pts)
        & (
            (out["positive_gap_pct"] <= cfg.power_low_gap_max_pct)
            | (leader >= cfg.power_leader_override_min)
            | (
                (out["positive_gap_pct"] <= cfg.power_gap_tolerant_max_pct)
                & (trigger >= cfg.power_gap_tolerant_min_trigger)
                & (pts_price_expansion >= cfg.power_gap_tolerant_min_price_expansion_pts)
                & (pts_reversal >= cfg.power_gap_tolerant_min_reversal_pts)
            )
        )
    )

    leader_norm = _clip01((leader - cfg.leader_min) / 12.0)
    trigger_norm = _clip01((trigger - cfg.trigger_min) / 16.0)
    setup_high_norm = _clip01((setup - cfg.setup_min) / max(cfg.setup_max - cfg.setup_min, 1e-9))
    gap_bonus = _clip01(1.0 - (out["positive_gap_pct"] / cfg.power_max_positive_gap_pct))
    tight_proximity_bonus = _clip01(
        1.0 - (out["entry_dist_above_line_pct"] / cfg.tight_max_entry_dist_above_line_pct)
    )
    continuation_bonus = _clip01(
        (out["entry_dist_above_line_pct"] - cfg.power_min_entry_dist_above_line_pct)
        / max(cfg.power_max_entry_dist_above_line_pct - cfg.power_min_entry_dist_above_line_pct, 1e-9)
    )
    reversal_norm = _clip01(pts_reversal / 20.0)
    price_exp_norm = _clip01(pts_price_expansion / 15.0)
    hold_norm = _clip01(pts_hold / 10.0)

    out["lane_tight_score"] = (
        15.0 * leader_norm
        + 35.0 * setup_high_norm
        + 15.0 * trigger_norm
        + 20.0 * tight_proximity_bonus
        + 15.0 * gap_bonus
    )
    out["lane_power_score"] = (
        10.0 * leader_norm
        + 10.0 * trigger_norm
        + 10.0 * reversal_norm
        + 15.0 * price_exp_norm
        + 35.0 * gap_bonus
        + 20.0 * continuation_bonus
    )
    out["same_day_priority_score"] = np.where(
        tight_lane,
        out["lane_tight_score"] + 5.0 * hold_norm,
        np.where(
            power_lane,
            out["lane_power_score"] + 5.0 * hold_norm,
            np.nan,
        ),
    )

    out["entry_base_signal"] = base_signal
    out["entry_lane_tight"] = tight_lane.fillna(False)
    out["entry_lane_power"] = power_lane.fillna(False)
    out["entry_lane"] = np.select(
        [out["entry_lane_tight"], out["entry_lane_power"]],
        ["tight_reversal", "power_continuation"],
        default="none",
    )
    out["entry_stop_policy"] = np.select(
        [tight_override, out["entry_lane_power"], out["entry_lane_tight"]],
        ["ignore_stop_limit", "ignore_stop_limit", "respect_stop_limit"],
        default="none",
    )
    out["entry_signal"] = out["entry_lane_tight"] | out["entry_lane_power"]

    out["entry_filter_reason"] = np.select(
        [
            ~base_signal,
            base_signal & ~tight_lane & ~power_lane & (out["entry_dist_above_line_pct"] <= cfg.tight_max_entry_dist_above_line_pct),
            base_signal & ~tight_lane & ~power_lane & (out["entry_dist_above_line_pct"] > cfg.tight_max_entry_dist_above_line_pct) & (out["entry_dist_above_line_pct"] < cfg.power_min_entry_dist_above_line_pct),
            base_signal & ~tight_lane & ~power_lane & (out["entry_dist_above_line_pct"] > cfg.power_max_entry_dist_above_line_pct),
            base_signal & ~tight_lane & ~power_lane & (out["positive_gap_pct"] > cfg.power_max_positive_gap_pct),
            base_signal & ~tight_lane & ~power_lane & (setup > cfg.power_setup_max),
            base_signal & ~tight_lane & ~power_lane,
        ],
        [
            "failed_base_signal",
            "tight_lane_gap_or_stop_only",
            "between_lanes",
            "too_far_above_line",
            "gap_too_large",
            "power_lane_setup_too_high",
            "failed_lane_shape",
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
