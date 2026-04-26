from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndustryPriorityConfig:
    leader98_bonus: float = 24.0
    leader_score_weight: float = 16.0
    volume_thrust_weight: float = 20.0
    bar_volume_thrust_weight: float = 10.0
    move_thrust_weight: float = 18.0
    industry_rs_weight: float = 14.0
    setup_sweet_weight: float = 8.0
    small_cap_weight: float = 5.0
    standard_bonus: float = 3.0

    leader_min_for_norm: float = 94.0
    setup_sweet_center: float = 74.0
    setup_sweet_half_width: float = 8.0
    a_plus_min_industry_rs: float = 0.70
    a_plus_min_setup_sweet: float = 0.65
    a_plus_min_volume_thrust: float = 0.75
    a_plus_min_move_thrust: float = 0.50
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
    industry_daily["industry_rs_pct"] = industry_daily.groupby("date")["industry_momentum"].rank(pct=True)
    return industry_daily[["date", "industry", "industry_momentum", "industry_rs_pct"]]


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
        out = out.drop(columns=[c for c in ["industry_momentum", "industry_rs_pct"] if c in out.columns])
        out = out.merge(industry_rs, on=["date", "industry"], how="left")
    elif "industry_rs_pct" not in out.columns:
        out["industry_rs_pct"] = np.nan

    leader = pd.to_numeric(out.get("leader_score"), errors="coerce")
    setup = pd.to_numeric(out.get("setup_score_pre"), errors="coerce")
    cumvol = pd.to_numeric(out.get("cum_vol_ratio_at_trigger"), errors="coerce").fillna(0.0)
    barvol = pd.to_numeric(out.get("bar_vol_ratio_at_trigger"), errors="coerce").fillna(0.0)
    move = pd.to_numeric(out.get("move_from_open_at_trigger"), errors="coerce").fillna(0.0)
    industry_rs_pct = pd.to_numeric(out.get("industry_rs_pct"), errors="coerce").fillna(0.5)
    market_cap = pd.to_numeric(out.get("market_cap"), errors="coerce")

    cap_rank = market_cap.rank(pct=True, ascending=True)
    small_cap_score = (1.0 - cap_rank).fillna(0.5)
    leader98 = (leader >= 98.0).astype(float)
    leader_score_norm = _clip01((leader - cfg.leader_min_for_norm) / 6.0)
    volume_thrust = _clip01(np.log1p(cumvol) / np.log1p(5.0))
    bar_volume_thrust = _clip01(np.log1p(barvol) / np.log1p(5.0))
    move_thrust = _clip01((move - 0.02) / 0.08)
    setup_sweet = _clip01(1.0 - (setup - cfg.setup_sweet_center).abs() / cfg.setup_sweet_half_width)
    standard_bonus = out.get("entry_source", pd.Series("", index=out.index)).eq("standard_breakout").astype(float)

    out["priority_leader98"] = leader98
    out["priority_leader_score_norm"] = leader_score_norm
    out["priority_volume_thrust"] = volume_thrust
    out["priority_bar_volume_thrust"] = bar_volume_thrust
    out["priority_move_thrust"] = move_thrust
    out["priority_setup_sweet"] = setup_sweet
    out["priority_industry_rs"] = industry_rs_pct
    out["priority_small_cap"] = small_cap_score
    out["priority_standard_bonus"] = standard_bonus
    out["same_day_priority_score"] = (
        cfg.leader98_bonus * leader98
        + cfg.leader_score_weight * leader_score_norm
        + cfg.volume_thrust_weight * volume_thrust
        + cfg.bar_volume_thrust_weight * bar_volume_thrust
        + cfg.move_thrust_weight * move_thrust
        + cfg.industry_rs_weight * industry_rs_pct
        + cfg.setup_sweet_weight * setup_sweet
        + cfg.small_cap_weight * small_cap_score
        + cfg.standard_bonus * standard_bonus
    )
    out["industry_a_plus_candidate"] = out.apply(lambda row: is_a_plus_candidate(row, cfg=cfg), axis=1)
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

    leader98 = _num(item, "priority_leader98", 0.0) >= 1.0 or _num(item, "leader_score", 0.0) >= 98.0
    industry_rs = _num(item, "priority_industry_rs", _num(item, "industry_rs_pct", 0.5))
    setup_sweet = _num(item, "priority_setup_sweet", 0.0)
    volume = _num(item, "priority_volume_thrust", 0.0)
    move = _num(item, "priority_move_thrust", 0.0)
    return bool(
        leader98
        and industry_rs >= cfg.a_plus_min_industry_rs
        and setup_sweet >= cfg.a_plus_min_setup_sweet
        and (volume >= cfg.a_plus_min_volume_thrust or move >= cfg.a_plus_min_move_thrust)
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
