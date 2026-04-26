from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .breakout_bridge import select_breakout_candidates


STANDARD_FEATURE_COLUMNS: list[str] = []


@dataclass(frozen=True)
class StandardSystemSpec:
    min_minutes_from_open: int = 5
    max_minutes_from_open: int = 90
    max_positions: int = 5
    max_train_rows: int = 1_200_000


def compute_threshold(*args, **kwargs) -> float:
    return 0.0


def session_bucket_from_minutes(minutes_from_open: pd.Series) -> pd.Series:
    minutes = pd.Series(minutes_from_open, copy=False)
    bucket = pd.Series(index=minutes.index, dtype="object")
    bucket.loc[minutes.between(0, 90, inclusive="both")] = "open_drive"
    bucket.loc[minutes.between(95, 300, inclusive="both")] = "midday"
    bucket.loc[minutes > 300] = "power_hour"
    return bucket.fillna("off_hours")


def candidate_sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "leader_score" in work.columns:
        sort_cols.append("leader_score")
        ascending.append(False)
    if "trigger_time" in work.columns:
        work["trigger_time"] = pd.to_datetime(work["trigger_time"], errors="coerce")
        sort_cols.append("trigger_time")
        ascending.append(True)
    sort_cols.append("symbol")
    ascending.append(True)
    return work.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)


def select_candidates_for_session(
    frame: pd.DataFrame,
    threshold: float = 0.0,
    max_positions: int = 5,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    session_date = pd.to_datetime(frame["date"]).max() if "date" in frame.columns else None
    return select_breakout_candidates(frame, session_date=session_date, max_positions=max_positions)
