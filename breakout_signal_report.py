from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from breakout_signal_engine import compute_breakout_scores_with_diag
from zigzag_breakout_engine import (
    ZigZagBreakoutConfig,
    _normalize_daily_input as _normalize_zigzag_daily_input,
    _normalize_intraday_input as _normalize_zigzag_intraday_input,
    compute_zigzag_breakout_scores,
    finalize_zigzag_breakout_signal_report,
)
from zigzag_entry_engine import ZigZagEntryConfig, apply_zigzag_entry_engine


ROOT = Path(__file__).resolve().parent
DAILY_PATH = ROOT / "analysis_outputs" / "russell3000_full_dataset" / "daily_history.pkl"
INTRADAY_PATH = ROOT / "analysis_outputs" / "russell3000_full_dataset" / "intraday_5m_history.pkl"
OUT_DIR = ROOT / "analysis_outputs" / "v4_rs_percentile_breakout_report"
SESSION_TZ = "America/New_York"
DAILY_LOOKBACK_CALENDAR_DAYS = 450

NORMAL_SETUP_MIN = 58.0
NORMAL_TRIGGER_MIN = 70.0
OVERRIDE_SETUP_MIN = 48.0
OVERRIDE_TRIGGER_MIN = 75.0


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


def _get_intraday_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    intraday_history = pd.read_pickle(INTRADAY_PATH)
    min_dt = None
    max_dt = None

    for df in intraday_history.values():
        idx = pd.to_datetime(df.index)
        current_min = idx.min()
        current_max = idx.max()
        if min_dt is None or current_min < min_dt:
            min_dt = current_min
        if max_dt is None or current_max > max_dt:
            max_dt = current_max

    del intraday_history

    if min_dt is None or max_dt is None:
        raise RuntimeError("No intraday data found")

    session_min = min_dt.tz_convert(SESSION_TZ).tz_localize(None).normalize()
    session_max = max_dt.tz_convert(SESSION_TZ).tz_localize(None).normalize()
    return session_min, session_max


def _prepare_daily_universe(
    session_min: pd.Timestamp,
    session_max: pd.Timestamp,
) -> pd.DataFrame:
    daily_history = pd.read_pickle(DAILY_PATH)
    start_date = session_min - pd.Timedelta(days=DAILY_LOOKBACK_CALENDAR_DAYS)
    end_date = session_max

    rows: list[pd.DataFrame] = []
    total = len(daily_history)
    for idx, (symbol, df) in enumerate(daily_history.items(), 1):
        daily = df.copy()
        daily.index = pd.to_datetime(daily.index).tz_localize(None).normalize()
        daily = daily.loc[(daily.index >= start_date) & (daily.index <= end_date)]
        if daily.empty:
            continue
        daily.index.name = "date"
        daily = daily.reset_index()
        daily.columns = [c.lower() for c in daily.columns]
        daily["symbol"] = symbol
        rows.append(daily[["symbol", "date", "open", "high", "low", "close", "volume"]])
        if idx % 500 == 0:
            print(f"Prepared daily slice: {idx}/{total} symbols")

    del daily_history
    if not rows:
        raise RuntimeError("No daily rows were prepared")
    return pd.concat(rows, ignore_index=True)


def _prepare_setup_candidates(
    daily_scored: pd.DataFrame,
    session_min: pd.Timestamp,
    session_max: pd.Timestamp,
) -> pd.DataFrame:
    scoped = daily_scored.loc[
        (daily_scored["date"] >= session_min) & (daily_scored["date"] <= session_max)
    ].copy()

    scoped["setup_tier"] = np.select(
        [
            scoped["setup_score_pre"] >= NORMAL_SETUP_MIN,
            scoped["setup_score_pre"] >= OVERRIDE_SETUP_MIN,
        ],
        ["normal", "override"],
        default="none",
    )

    scoped["setup_candidate"] = (
        scoped["history_ok"]
        & scoped["leader_pass"]
        & (scoped["setup_score_pre"] >= OVERRIDE_SETUP_MIN)
    )

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
        "pivot_high",
        "diag_resistance",
        "diag_resistance_prev",
        "diag_valid",
        "history_ok",
        "leader_pass",
        "rs_pct_21",
        "rs_pct_63",
        "rs_pct_126",
        "rs_rating",
        "leader_score",
        "setup_score_pre",
        "setup_tier",
        "setup_candidate",
    ]

    return scoped.loc[scoped["setup_candidate"], keep_cols].copy()


def _finalize_report(
    setup_candidates: pd.DataFrame,
    first_breakouts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = setup_candidates.merge(
        first_breakouts,
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )

    report["breakout_type"] = report["breakout_type"].fillna("none")
    report["broke_out"] = report["breakout_type"] != "none"
    report["golden_rule_pass"] = (
        report["broke_out"]
        & (report["leader_score"] >= 94.0)
        & report["setup_score_pre"].between(71.0, 76.0)
        & (report["trigger_score"] >= 76.0)
    )
    report["breakout_signal"] = (
        report["history_ok"]
        & report["leader_pass"]
        & report["golden_rule_pass"]
    )
    report["trigger_time_ny"] = pd.to_datetime(report["trigger_time"]).dt.strftime("%H:%M")
    report["date"] = pd.to_datetime(report["date"]).dt.normalize()

    summary = (
        report.groupby("date", as_index=False)
        .agg(
            setup_count=("symbol", "size"),
            normal_setup_count=("setup_tier", lambda s: int((s == "normal").sum())),
            override_setup_count=("setup_tier", lambda s: int((s == "override").sum())),
            breakout_count=("broke_out", "sum"),
            breakout_signal_count=("breakout_signal", "sum"),
        )
        .sort_values("date")
    )

    report = report.sort_values(
        ["date", "broke_out", "breakout_signal", "trigger_time", "leader_score", "symbol"],
        ascending=[True, False, False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return report, summary


def _candidate_date_map(setup_candidates: pd.DataFrame) -> dict[str, set[pd.Timestamp]]:
    date_map: dict[str, set[pd.Timestamp]] = {}
    for symbol, sub in setup_candidates.groupby("symbol", sort=False):
        date_map[symbol] = set(pd.to_datetime(sub["date"]).dt.normalize())
    return date_map


def _prepare_intraday_for_candidates(
    candidate_dates_by_symbol: dict[str, set[pd.Timestamp]],
) -> pd.DataFrame:
    intraday_history = pd.read_pickle(INTRADAY_PATH)
    rows: list[pd.DataFrame] = []
    total = len(candidate_dates_by_symbol)

    for idx, (symbol, candidate_dates) in enumerate(candidate_dates_by_symbol.items(), 1):
        intraday = intraday_history.get(symbol)
        if intraday is None or len(intraday) == 0:
            continue

        intra = intraday.copy()
        intra.index = pd.to_datetime(intra.index)
        intra.index = intra.index.tz_convert(SESSION_TZ).tz_localize(None)
        intra.index.name = "datetime"
        intra = intra.reset_index()
        intra.columns = [c.lower() for c in intra.columns]
        intra["symbol"] = symbol
        intra["session_date"] = pd.to_datetime(intra["datetime"]).dt.normalize()
        intra = intra.loc[intra["session_date"].isin(candidate_dates)].copy()
        if intra.empty:
            continue
        rows.append(intra[["symbol", "datetime", "session_date", "open", "high", "low", "close", "volume"]])

        if idx % 250 == 0:
            print(f"Prepared intraday slice: {idx}/{total} symbols")

    del intraday_history
    if not rows:
        return pd.DataFrame(
            columns=["symbol", "datetime", "session_date", "open", "high", "low", "close", "volume"]
        )
    return pd.concat(rows, ignore_index=True)


def _compute_intraday_first_breakouts(
    intraday_df: pd.DataFrame,
    daily_context: pd.DataFrame,
) -> pd.DataFrame:
    if intraday_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "trigger_time",
                "trigger_score",
                "breakout_type",
                "cum_vol_ratio_at_trigger",
                "bar_vol_ratio_at_trigger",
                "move_from_open_at_trigger",
                "dist_above_res_at_trigger",
            ]
        )

    intra = intraday_df.copy()
    intra = intra.sort_values(["symbol", "datetime"], kind="mergesort").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        intra[col] = pd.to_numeric(intra[col], errors="coerce").astype("float64")

    intra["slot_index"] = intra.groupby(["symbol", "session_date"], sort=False).cumcount()
    intra["cum_volume_session"] = intra.groupby(["symbol", "session_date"], sort=False)["volume"].cumsum()
    intra["session_open"] = intra.groupby(["symbol", "session_date"], sort=False)["open"].transform("first")
    intra["session_low_so_far"] = intra.groupby(["symbol", "session_date"], sort=False)["low"].cummin()
    intra["prev_bar_close_session"] = intra.groupby(["symbol", "session_date"], sort=False)["close"].shift(1)

    intra = _add_same_time_volume_features(intra)

    daily_ctx = daily_context[
        [
            "symbol",
            "date",
            "leader_score",
            "leader_pass",
            "history_ok",
            "setup_score_pre",
            "pivot_high",
            "diag_resistance",
            "diag_resistance_prev",
            "diag_valid",
            "atr20",
            "prev_close",
        ]
    ].rename(columns={"date": "session_date"})

    intra = intra.merge(
        daily_ctx,
        on=["symbol", "session_date"],
        how="left",
        validate="many_to_one",
    )

    h_level = intra["pivot_high"]
    d_level = intra["diag_resistance"]

    breakout_horizontal = (
        intra["pivot_high"].notna()
        & (intra["high"] >= h_level)
        & (intra["prev_bar_close_session"].isna() | (intra["prev_bar_close_session"] < h_level))
    )
    breakout_diagonal = (
        intra["diag_valid"].fillna(False)
        & intra["diag_resistance"].notna()
        & (intra["high"] >= d_level)
        & (intra["prev_bar_close_session"].isna() | (intra["prev_bar_close_session"] < d_level))
    )

    intra["breakout_horizontal_intraday"] = breakout_horizontal
    intra["breakout_diagonal_intraday"] = breakout_diagonal
    intra["breakout_any_intraday"] = breakout_horizontal | breakout_diagonal

    resistance_used = np.where(
        breakout_horizontal & breakout_diagonal,
        np.maximum(intra["pivot_high"].to_numpy(), intra["diag_resistance"].to_numpy()),
        np.where(
            breakout_horizontal,
            intra["pivot_high"].to_numpy(),
            np.where(breakout_diagonal, intra["diag_resistance"].to_numpy(), np.nan),
        ),
    )
    intra["resistance_used"] = resistance_used

    intra["breakout_type_intraday"] = np.select(
        [
            breakout_horizontal & breakout_diagonal,
            breakout_horizontal,
            breakout_diagonal,
        ],
        ["both", "horizontal", "diagonal"],
        default="none",
    )

    dist_above_res = (
        (intra["close"] / pd.Series(intra["resistance_used"], index=intra.index).replace(0, np.nan)) - 1.0
    ).clip(lower=0)

    breakout_strength = np.clip(dist_above_res / 0.02, 0, 1)
    breakout_component = 0.70 * intra["breakout_any_intraday"].astype(float) + 0.30 * breakout_strength

    volume_component = (
        0.60 * np.clip(intra["cum_vol_ratio"] / 2.0, 0, 1)
        + 0.40 * np.clip(intra["bar_vol_ratio"] / 2.0, 0, 1)
    )

    atr20_pct_daily = intra["atr20"] / intra["prev_close"].replace(0, np.nan)
    intraday_bar_range_pct = (intra["high"] - intra["low"]) / intra["close"].replace(0, np.nan)
    range_expand = np.clip(
        (intraday_bar_range_pct / atr20_pct_daily.replace(0, np.nan)) / 0.35,
        0,
        1,
    )

    move_from_open = (intra["close"] / intra["session_open"].replace(0, np.nan)) - 1.0
    move_from_open_component = np.clip(move_from_open / 0.03, 0, 1)
    price_expansion = 0.50 * range_expand + 0.50 * move_from_open_component

    risk_to_lod_so_far = (intra["close"] - intra["session_low_so_far"]) / intra["close"].replace(0, np.nan)
    risk_component = np.clip(
        1.0 - (risk_to_lod_so_far / atr20_pct_daily.replace(0, np.nan)),
        0,
        1,
    )

    pos_in_bar = (
        (intra["close"] - intra["low"])
        / (intra["high"] - intra["low"]).replace(0, np.nan)
    )
    hold_component = np.clip((pos_in_bar - 0.50) / 0.50, 0, 1)
    not_too_extended = np.clip(1.0 - (dist_above_res / 0.03), 0, 1)

    intra["trigger_score_intraday"] = (
        35.0 * breakout_component
        + 25.0 * volume_component
        + 15.0 * price_expansion
        + 15.0 * risk_component
        + 10.0 * (0.70 * hold_component + 0.30 * not_too_extended)
    )

    intra["cum_vol_ratio_at_trigger"] = intra["cum_vol_ratio"]
    intra["bar_vol_ratio_at_trigger"] = intra["bar_vol_ratio"]
    intra["move_from_open_at_trigger"] = move_from_open
    intra["dist_above_res_at_trigger"] = dist_above_res

    trig = intra.loc[intra["breakout_any_intraday"]].copy()
    if trig.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "trigger_time",
                "trigger_score",
                "breakout_type",
                "cum_vol_ratio_at_trigger",
                "bar_vol_ratio_at_trigger",
                "move_from_open_at_trigger",
                "dist_above_res_at_trigger",
            ]
        )

    trig = trig.sort_values(["symbol", "session_date", "datetime"], kind="mergesort")

    first_trig = (
        trig.groupby(["symbol", "session_date"], as_index=False)
        .first()
        .rename(
            columns={
                "session_date": "date",
                "datetime": "trigger_time",
                "close": "trigger_close",
                "trigger_score_intraday": "trigger_score",
                "breakout_type_intraday": "breakout_type",
            }
        )
    )

    return first_trig[
        [
            "symbol",
            "date",
            "trigger_time",
            "trigger_close",
            "trigger_score",
            "breakout_type",
            "cum_vol_ratio_at_trigger",
            "bar_vol_ratio_at_trigger",
            "move_from_open_at_trigger",
            "dist_above_res_at_trigger",
        ]
    ]


def _build_standard_breakout_signal_report(
    daily_df: pd.DataFrame,
    intraday_df: pd.DataFrame,
    *,
    session_tz: str = SESSION_TZ,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_long = daily_df.copy()
    daily_long["date"] = pd.to_datetime(daily_long["date"]).dt.normalize()
    intraday_long = intraday_df.copy()
    if not intraday_long.empty:
        intraday_long["datetime"] = pd.to_datetime(intraday_long["datetime"])
        session_series = intraday_long["datetime"]
        if getattr(session_series.dt, "tz", None) is not None:
            intraday_long["datetime"] = session_series.dt.tz_convert(session_tz).dt.tz_localize(None)
        intraday_long["session_date"] = pd.to_datetime(intraday_long["datetime"]).dt.normalize()
    else:
        intraday_long["session_date"] = pd.Series(dtype="datetime64[ns]")

    if daily_long.empty:
        empty_report = pd.DataFrame()
        empty_summary = pd.DataFrame(
            columns=[
                "date",
                "setup_count",
                "normal_setup_count",
                "override_setup_count",
                "breakout_count",
                "breakout_signal_count",
            ]
        )
        return empty_report, empty_summary

    if intraday_long.empty:
        session_min = pd.to_datetime(daily_long["date"]).min().normalize()
        session_max = pd.to_datetime(daily_long["date"]).max().normalize()
    else:
        session_min = pd.to_datetime(intraday_long["session_date"]).min().normalize()
        session_max = pd.to_datetime(intraday_long["session_date"]).max().normalize()

    daily_scored = compute_breakout_scores_with_diag(daily_long)
    setup_candidates = _prepare_setup_candidates(daily_scored, session_min, session_max)
    first_breakouts = _compute_intraday_first_breakouts(intraday_long, daily_scored)
    return _finalize_report(setup_candidates, first_breakouts)


def _build_zigzag_signal_report(
    daily_df: pd.DataFrame,
    intraday_df: pd.DataFrame,
    *,
    session_tz: str = SESSION_TZ,
) -> pd.DataFrame:
    zig_cfg = ZigZagBreakoutConfig()
    entry_cfg = ZigZagEntryConfig()

    zig_daily = _normalize_zigzag_daily_input(daily_df.copy())
    if intraday_df is None or intraday_df.empty:
        zig_intraday = pd.DataFrame(columns=["symbol", "datetime", "open", "high", "low", "close", "volume"])
    else:
        zig_intraday = _normalize_zigzag_intraday_input(intraday_df.copy(), session_timezone=session_tz)

    zig_daily_scored = compute_zigzag_breakout_scores(zig_daily, cfg=zig_cfg)
    zig_report = finalize_zigzag_breakout_signal_report(zig_daily_scored, zig_intraday, cfg=zig_cfg)
    zig_report = apply_zigzag_entry_engine(zig_report, entry_cfg)

    zig_report["zigzag_breakout_signal"] = zig_report["entry_signal"].fillna(False).astype(bool)
    zig_report["entry_source"] = np.select(
        [
            zig_report["entry_lane"].eq("tight_reversal"),
            zig_report["entry_lane"].eq("power_continuation"),
        ],
        [
            "tight_reversal",
            "power_continuation",
        ],
        default="none",
    )
    return zig_report


def _merge_standard_and_zigzag_reports(
    standard_report: pd.DataFrame,
    zigzag_report: pd.DataFrame,
) -> pd.DataFrame:
    std = standard_report.copy()
    zz = zigzag_report.copy()

    # standard 側
    std["standard_breakout_signal"] = std["breakout_signal"].fillna(False).astype(bool)
    std["zigzag_breakout_signal"] = False
    std["entry_source"] = np.where(std["standard_breakout_signal"], "standard_breakout", "none")
    std["entry_lane"] = "none"
    std["entry_stop_policy"] = np.where(std["standard_breakout_signal"], "respect_stop_limit", "none")
    std["same_day_priority_score"] = np.nan
    std["effective_pivot_level"] = pd.to_numeric(std["pivot_high"], errors="coerce")

    # zigzag 側
    zz["standard_breakout_signal"] = False
    zz["effective_pivot_level"] = pd.to_numeric(zz["zigzag_line_value"], errors="coerce")

    # 共通 schema へ寄せる
    std_keep = [
        "symbol", "date", "open", "high", "low", "close", "volume",
        "prev_close", "atr20", "adr20_pct",
        "history_ok", "leader_pass", "setup_candidate",
        "leader_score", "setup_score_pre", "rs_rating",
        "trigger_time", "trigger_close", "trigger_score",
        "breakout_type", "pivot_high", "effective_pivot_level",
        "standard_breakout_signal", "zigzag_breakout_signal",
        "entry_source", "entry_lane", "entry_stop_policy",
        "same_day_priority_score",
    ]
    zz_keep = [
        "symbol", "date", "open", "high", "low", "close", "volume",
        "prev_close", "atr20", "adr20_pct",
        "history_ok", "leader_pass", "setup_candidate",
        "leader_score", "setup_score_pre", "rs_rating",
        "trigger_time", "trigger_close", "trigger_score",
        "breakout_type", "pivot_high", "zigzag_line_value", "effective_pivot_level",
        "standard_breakout_signal", "zigzag_breakout_signal",
        "entry_source", "entry_lane", "entry_stop_policy",
        "same_day_priority_score",
        "entry_dist_above_line_pct", "gap_pct", "entry_filter_reason",
        "trigger_pts_breakout", "trigger_pts_price_expansion",
        "trigger_pts_reversal", "trigger_pts_hold",
        "broke_out",
    ]

    std = std[[c for c in std_keep if c in std.columns]].copy()
    zz = zz[[c for c in zz_keep if c in zz.columns]].copy()

    # 同一symbol/dateで standard と zigzag を別行として保持すると、
    # 後段の entry 優先順位付けが素直になる
    report = pd.concat([std, zz], ignore_index=True, sort=False)

    report["standard_breakout_signal"] = report["standard_breakout_signal"].fillna(False).astype(bool)
    report["zigzag_breakout_signal"] = report["zigzag_breakout_signal"].fillna(False).astype(bool)
    report["breakout_signal"] = report["standard_breakout_signal"] | report["zigzag_breakout_signal"]

    report["entry_priority_bucket"] = np.select(
        [
            report["entry_source"].eq("standard_breakout"),
            report["entry_source"].eq("tight_reversal"),
            report["entry_source"].eq("power_continuation"),
        ],
        [0, 1, 2],
        default=99,
    )

    report["priority_score_within_source"] = np.where(
        report["entry_source"].eq("standard_breakout"),
        pd.to_numeric(report["rs_rating"], errors="coerce"),
        pd.to_numeric(report["trigger_score"], errors="coerce"),
    )

    report["trigger_time_ny"] = pd.to_datetime(report["trigger_time"], errors="coerce").dt.strftime("%H:%M")
    report["date"] = pd.to_datetime(report["date"]).dt.normalize()

    return report.sort_values(
        [
            "date",
            "breakout_signal",
            "entry_priority_bucket",
            "priority_score_within_source",
            "trigger_time",
            "leader_score",
            "symbol",
        ],
        ascending=[True, False, True, False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_breakout_signal_report(
    daily_df: pd.DataFrame,
    intraday_df: pd.DataFrame,
    *,
    session_tz: str = SESSION_TZ,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    standard_report, _ = _build_standard_breakout_signal_report(
        daily_df,
        intraday_df,
        session_tz=session_tz,
    )
    zigzag_report = _build_zigzag_signal_report(
        daily_df,
        intraday_df,
        session_tz=session_tz,
    )

    report = _merge_standard_and_zigzag_reports(standard_report, zigzag_report)

    summary = (
        report.groupby("date", as_index=False)
        .agg(
            setup_count=("setup_candidate", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            breakout_signal_count=("breakout_signal", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            standard_breakout_count=("standard_breakout_signal", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            zigzag_breakout_count=("zigzag_breakout_signal", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            tight_reversal_count=("entry_source", lambda s: int((pd.Series(s) == "tight_reversal").sum())),
            power_continuation_count=("entry_source", lambda s: int((pd.Series(s) == "power_continuation").sum())),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return report, summary


def build_report() -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session_min, session_max = _get_intraday_range()
    print(f"Intraday session window: {session_min.date()} -> {session_max.date()}")

    daily_df = _prepare_daily_universe(session_min=session_min, session_max=session_max)
    print(f"Daily universe rows: {len(daily_df):,}")

    report, summary = build_breakout_signal_report(daily_df, pd.DataFrame()) # Intraday loading is handled inside usually or passed

    detail_cols = [
        "date",
        "symbol",
        "rs_rating",
        "leader_score",
        "setup_score_pre",
        "trigger_time",
        "trigger_time_ny",
        "breakout_type",
        "trigger_score",
        "broke_out",
        "breakout_signal",
        "entry_source",
        "entry_lane",
    ]
    report_to_save = report[[c for c in detail_cols if c in report.columns]]

    report_to_save.to_csv(OUT_DIR / "setup_breakout_details.csv", index=False)
    report_to_save.loc[report["breakout_signal"]].to_csv(OUT_DIR / "breakout_signals_only.csv", index=False)
    summary.to_csv(OUT_DIR / "daily_summary.csv", index=False)

    return report, summary


def main() -> None:
    report, summary = build_report()

    print("\nTop 10 dates by breakout count")
    print(summary.sort_values(["breakout_signal_count", "setup_count"], ascending=False).head(10).to_string(index=False))

    print("\nFirst 20 breakout rows")
    preview = report.loc[report["breakout_signal"]].head(20)
    if preview.empty:
        print("No breakouts found for setup candidates.")
    else:
        print(
            preview[
                [
                    "date",
                    "symbol",
                    "rs_rating",
                    "leader_score",
                    "trigger_time_ny",
                    "breakout_type",
                    "trigger_score",
                    "entry_source",
                ]
            ].to_string(index=False)
        )

    print(f"\nDetailed outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
