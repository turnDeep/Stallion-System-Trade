from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndustryPriorityConfig:
    core_leader_weight: float = 25.0
    core_setup_weight: float = 20.0
    core_trigger_weight: float = 18.0
    core_volume_weight: float = 15.0
    core_prior_runup_weight: float = 12.0
    core_move_weight: float = 10.0
    leader95_bonus: float = 20.0

    leader_min_for_norm: float = 60.0
    leader_max_for_norm: float = 100.0
    setup_sweet_center: float = 72.0
    setup_sweet_half_width: float = 10.0
    trigger_min_for_norm: float = 50.0
    trigger_max_for_norm: float = 90.0
    volume_norm_cap: float = 5.0
    prior_runup_norm_cap: float = 1.20
    move_norm_floor: float = -0.02
    move_norm_cap: float = 0.06

    a_plus_leader_min: float = 95.0
    a_plus_setup_min: float = 68.0
    a_plus_trigger_min: float = 77.0
    a_plus_cum_vol_ratio_min: float = 3.0
    a_plus_move_from_open_min: float = -0.05
    replacement_score_margin: float = 18.0
    strong_winner_gain: float = 0.50
    protected_a_plus_gain: float = 0.10


def _clip01(values: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float).clip(0.0, 1.0).fillna(0.0)
    return pd.Series(np.clip(np.asarray(values, dtype="float64"), 0.0, 1.0))


def _num(item: Mapping[str, Any], key: str, default: float = np.nan) -> float:
    value = pd.to_numeric(item.get(key, default), errors="coerce")
    return float(value) if pd.notna(value) else default


def _numeric_column(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def normalize_universe(universe: pd.DataFrame | None) -> pd.DataFrame:
    if universe is None or universe.empty:
        return pd.DataFrame(columns=["symbol", "market_cap", "sector", "industry", "rank_market_cap"])
    out = universe.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    for column in ["market_cap", "rank_market_cap"]:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ["sector", "industry"]:
        if column not in out.columns:
            out[column] = np.nan
    return out[["symbol", "market_cap", "sector", "industry", "rank_market_cap"]].drop_duplicates("symbol")


def build_industry_rs(daily: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    if daily.empty or universe.empty:
        return pd.DataFrame(columns=["date", "industry", "industry_momentum", "industry_rs_pct"])

    bars = daily.copy()
    if "date" not in bars.columns and "ts" in bars.columns:
        bars = bars.rename(columns={"ts": "date"})
    bars["symbol"] = bars["symbol"].astype(str).str.upper()
    bars["date"] = pd.to_datetime(bars["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    bars = bars.dropna(subset=["symbol", "date", "close"])

    uni = normalize_universe(universe)[["symbol", "industry"]].dropna(subset=["industry"])
    bars = bars.merge(uni, on="symbol", how="inner")
    if bars.empty:
        return pd.DataFrame(columns=["date", "industry", "industry_momentum", "industry_rs_pct"])

    bars = bars.sort_values(["symbol", "date"], kind="mergesort")
    grouped = bars.groupby("symbol", sort=False)
    bars["roc_21"] = bars["close"] / grouped["close"].shift(21).replace(0, np.nan) - 1.0
    bars["roc_63"] = bars["close"] / grouped["close"].shift(63).replace(0, np.nan) - 1.0
    bars["roc_126"] = bars["close"] / grouped["close"].shift(126).replace(0, np.nan) - 1.0
    bars["industry_momentum"] = 0.25 * bars["roc_21"] + 0.35 * bars["roc_63"] + 0.40 * bars["roc_126"]

    industry_daily = (
        bars.groupby(["date", "industry"], as_index=False)["industry_momentum"]
        .median()
        .dropna(subset=["industry_momentum"])
    )
    industry_daily["industry_momentum_eod"] = industry_daily["industry_momentum"]
    industry_daily["industry_rs_pct_eod"] = industry_daily.groupby("date")["industry_momentum"].rank(pct=True)
    industry_daily = industry_daily.sort_values(["industry", "date"], kind="mergesort")
    # Same-day entries cannot know the close-based industry rank for that day.
    industry_daily["industry_momentum"] = industry_daily.groupby("industry", sort=False)["industry_momentum_eod"].shift(1)
    industry_daily["industry_rs_pct"] = industry_daily.groupby("industry", sort=False)["industry_rs_pct_eod"].shift(1)
    return industry_daily[
        [
            "date",
            "industry",
            "industry_momentum",
            "industry_rs_pct",
            "industry_momentum_eod",
            "industry_rs_pct_eod",
        ]
    ]


def build_prior_runup(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["symbol", "date", "prior_runup_pre"])

    bars = daily.copy()
    if "date" not in bars.columns and "ts" in bars.columns:
        bars = bars.rename(columns={"ts": "date"})
    bars["symbol"] = bars["symbol"].astype(str).str.upper()
    bars["date"] = pd.to_datetime(bars["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    bars = bars.dropna(subset=["symbol", "date", "close"]).sort_values(["symbol", "date"], kind="mergesort")
    grouped = bars.groupby("symbol", sort=False)
    prior21 = bars["close"] / grouped["close"].shift(21).replace(0, np.nan) - 1.0
    prior63 = bars["close"] / grouped["close"].shift(63).replace(0, np.nan) - 1.0
    bars["prior_runup_eod"] = pd.concat([prior21, prior63], axis=1).max(axis=1)
    bars["prior_runup_pre"] = grouped["prior_runup_eod"].shift(1)
    return bars[["symbol", "date", "prior_runup_pre"]]


def add_industry_composite_priority(
    report: pd.DataFrame,
    daily: pd.DataFrame,
    universe: pd.DataFrame | None,
    cfg: IndustryPriorityConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or IndustryPriorityConfig()
    if report.empty:
        return report.copy()

    out = report.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    uni = normalize_universe(universe)
    if not uni.empty:
        out = out.drop(columns=[c for c in ["market_cap", "sector", "industry", "rank_market_cap"] if c in out.columns])
        out = out.merge(uni, on="symbol", how="left")
    else:
        for column in ["market_cap", "sector", "industry", "rank_market_cap"]:
            if column not in out.columns:
                out[column] = np.nan

    industry_rs = build_industry_rs(daily, uni)
    if not industry_rs.empty:
        out = out.drop(
            columns=[
                c
                for c in [
                    "industry_momentum",
                    "industry_rs_pct",
                    "industry_momentum_eod",
                    "industry_rs_pct_eod",
                ]
                if c in out.columns
            ]
        )
        out = out.merge(industry_rs, on=["date", "industry"], how="left")
    elif "industry_rs_pct" not in out.columns:
        out["industry_rs_pct"] = np.nan

    prior_runup = build_prior_runup(daily)
    if not prior_runup.empty:
        out = out.drop(columns=[c for c in ["prior_runup_pre"] if c in out.columns])
        out = out.merge(prior_runup, on=["symbol", "date"], how="left")
    elif "prior_runup_pre" not in out.columns:
        out["prior_runup_pre"] = np.nan

    leader = _numeric_column(out, "leader_score")
    setup = _numeric_column(out, "setup_score_pre")
    trigger = _numeric_column(out, "trigger_score")
    cumvol = _numeric_column(out, "cum_vol_ratio_at_trigger")
    barvol = _numeric_column(out, "bar_vol_ratio_at_trigger")
    move = _numeric_column(out, "move_from_open_at_trigger").fillna(0.0)
    prior = _numeric_column(out, "prior_runup_pre").fillna(0.0)
    industry_rs_pct = _numeric_column(out, "industry_rs_pct").fillna(0.5)
    market_cap = _numeric_column(out, "market_cap")

    cap_rank = market_cap.rank(pct=True, ascending=True)
    small_cap_score = (1.0 - cap_rank).fillna(0.5)
    leader98 = (leader >= 98.0).astype(float)
    leader95 = (leader >= 95.0).astype(float)
    leader_score_norm = _clip01((leader - cfg.leader_min_for_norm) / (cfg.leader_max_for_norm - cfg.leader_min_for_norm))
    trigger_norm = _clip01((trigger.fillna(60.0) - cfg.trigger_min_for_norm) / (cfg.trigger_max_for_norm - cfg.trigger_min_for_norm))
    volume_thrust = _clip01(np.log1p(cumvol.fillna(0.75)) / np.log1p(cfg.volume_norm_cap))
    bar_volume_thrust = _clip01(np.log1p(barvol.fillna(0.75)) / np.log1p(cfg.volume_norm_cap))
    prior_runup_score = _clip01(prior / cfg.prior_runup_norm_cap)
    move_thrust = _clip01((move - cfg.move_norm_floor) / (cfg.move_norm_cap - cfg.move_norm_floor))
    setup_sweet = _clip01(1.0 - (setup - cfg.setup_sweet_center).abs() / cfg.setup_sweet_half_width)
    standard_bonus = out.get("entry_source", pd.Series("", index=out.index)).eq("standard_breakout").astype(float)
    a_plus_candidate = (
        (leader >= cfg.a_plus_leader_min)
        & (setup >= cfg.a_plus_setup_min)
        & (trigger >= cfg.a_plus_trigger_min)
        & (cumvol >= cfg.a_plus_cum_vol_ratio_min)
        & (move >= cfg.a_plus_move_from_open_min)
    ).fillna(False)

    out["priority_leader98"] = leader98
    out["priority_leader95"] = leader95
    out["priority_leader_score_norm"] = leader_score_norm
    out["priority_volume_thrust"] = volume_thrust
    out["priority_bar_volume_thrust"] = bar_volume_thrust
    out["priority_trigger_score_norm"] = trigger_norm
    out["priority_prior_runup"] = prior_runup_score
    out["priority_move_thrust"] = move_thrust
    out["priority_setup_sweet"] = setup_sweet
    out["priority_industry_rs"] = industry_rs_pct
    out["priority_small_cap"] = small_cap_score
    out["priority_standard_bonus"] = standard_bonus
    out["priority_a_plus_candidate"] = a_plus_candidate.astype(float)
    out["same_day_priority_score"] = (
        cfg.core_leader_weight * leader_score_norm
        + cfg.core_setup_weight * setup_sweet
        + cfg.core_trigger_weight * trigger_norm
        + cfg.core_volume_weight * volume_thrust
        + cfg.core_prior_runup_weight * prior_runup_score
        + cfg.core_move_weight * move_thrust
        + cfg.leader95_bonus * leader95
    )
    out["industry_a_plus_candidate"] = a_plus_candidate.astype(bool)
    return out


def is_a_plus_candidate(row: Mapping[str, Any] | pd.Series | Any, cfg: IndustryPriorityConfig | None = None) -> bool:
    cfg = cfg or IndustryPriorityConfig()
    if isinstance(row, pd.Series):
        item = row.to_dict()
    elif isinstance(row, Mapping):
        item = dict(row)
    elif hasattr(row, "_asdict"):
        item = dict(row._asdict())
    else:
        item = dict(row)

    leader = _num(item, "leader_score", 0.0)
    setup = _num(item, "setup_score_pre", 0.0)
    trigger = _num(item, "trigger_score", 0.0)
    cumvol = _num(item, "cum_vol_ratio_at_trigger", 0.0)
    move = _num(item, "move_from_open_at_trigger", 0.0)
    return bool(
        leader >= cfg.a_plus_leader_min
        and setup >= cfg.a_plus_setup_min
        and trigger >= cfg.a_plus_trigger_min
        and cumvol >= cfg.a_plus_cum_vol_ratio_min
        and move >= cfg.a_plus_move_from_open_min
    )


def replacement_score(priority_score: float, current_gain: float) -> float:
    return float(priority_score) + 40.0 * max(float(current_gain), 0.0) - 20.0 * max(-float(current_gain), 0.0)


def choose_replacement_index(
    positions: list[dict[str, Any]],
    candidate: Mapping[str, Any] | pd.Series | Any,
    *,
    cfg: IndustryPriorityConfig | None = None,
) -> int | None:
    cfg = cfg or IndustryPriorityConfig()
    if not is_a_plus_candidate(candidate, cfg=cfg):
        return None
    if isinstance(candidate, pd.Series):
        item = candidate.to_dict()
    elif isinstance(candidate, Mapping):
        item = dict(candidate)
    elif hasattr(candidate, "_asdict"):
        item = dict(candidate._asdict())
    else:
        item = dict(candidate)
    candidate_score = _num(item, "same_day_priority_score", 0.0)

    replaceable: list[tuple[float, int]] = []
    for idx, pos in enumerate(positions):
        current_gain = float(pos.get("current_gain", 0.0))
        if bool(pos.get("a_plus_candidate", False)) and current_gain > cfg.protected_a_plus_gain:
            continue
        if current_gain >= cfg.strong_winner_gain:
            continue
        score = replacement_score(float(pos.get("priority_score", 0.0)), current_gain)
        replaceable.append((score, idx))

    if not replaceable:
        return None
    weakest_score, weakest_idx = min(replaceable, key=lambda item: item[0])
    if candidate_score >= weakest_score + cfg.replacement_score_margin:
        return weakest_idx
    return None


def sort_by_industry_priority(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "same_day_priority_score" not in events.columns:
        return events.copy().reset_index(drop=True)
    work = events.copy()
    work["trigger_time"] = pd.to_datetime(work.get("trigger_time"), errors="coerce")
    return (
        work.sort_values(
            ["date", "same_day_priority_score", "trigger_time", "symbol"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
