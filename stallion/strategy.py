from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


STANDARD_FEATURE_COLUMNS = [
    "daily_buy_pressure_prev",
    "prev_day_adr_pct",
    "industry_buy_pressure_prev",
    "EMA_8_15",
    "distance_to_prev_day_high",
    "close_vs_vwap_15",
    "sector_buy_pressure_prev",
    "daily_rrs_prev",
    "daily_rs_score_prev",
    "distance_to_avwap_63_prev",
    "volume_spike_5m",
    "industry_rs_prev",
    "same_slot_avg_vol_20d",
    "rs_x_intraday_rvol",
    "intraday_range_expansion_vs_atr",
    "prev_day_close_vs_sma50",
]


@dataclass(frozen=True)
class StandardSystemSpec:
    min_minutes_from_open: int = 5
    max_minutes_from_open: int = 90
    max_positions: int = 4
    threshold_floor: float = 0.55
    threshold_quantile: float = 0.90
    max_train_rows: int = 1_200_000


def compute_threshold(train_scores: Iterable[float], floor: float = 0.55, quantile: float = 0.90) -> float:
    series = pd.Series(train_scores, dtype="float64").dropna()
    if series.empty:
        return floor
    return float(max(floor, np.nanquantile(series.to_numpy(), quantile)))


def session_bucket_from_minutes(minutes_from_open: pd.Series) -> pd.Series:
    minutes = pd.Series(minutes_from_open, copy=False)
    bucket = pd.Series(index=minutes.index, dtype="object")
    bucket.loc[minutes.between(0, 90, inclusive="both")] = "open_drive"
    bucket.loc[minutes.between(95, 300, inclusive="both")] = "midday"
    bucket.loc[minutes > 300] = "power_hour"
    return bucket.fillna("off_hours")


def candidate_sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["timestamp", "score"], ascending=[True, False]).reset_index(drop=True)


def select_candidates_for_session(
    frame: pd.DataFrame,
    threshold: float,
    max_positions: int = 4,
) -> pd.DataFrame:
    work = candidate_sort_frame(frame)
    chosen_rows = []
    seen_symbols: set[str] = set()
    for row in work.itertuples(index=False):
        if len(chosen_rows) >= max_positions:
            break
        if not np.isfinite(row.score) or row.score < threshold:
            continue
        symbol = str(row.symbol)
        if symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)
        chosen_rows.append(row._asdict())
    return pd.DataFrame(chosen_rows)
