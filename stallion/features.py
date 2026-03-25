from __future__ import annotations

import gc
import logging
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np
import pandas as pd

from .strategy import STANDARD_FEATURE_COLUMNS, session_bucket_from_minutes


LOGGER = logging.getLogger(__name__)
MARKET_TIMEZONE = "America/New_York"


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def _anchored_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, lookback: int = 63) -> pd.Series:
    typical = (high + low + close) / 3.0
    result = pd.Series(index=close.index, dtype="float64")
    for end in range(len(close)):
        start = max(0, end - lookback + 1)
        window_high = high.iloc[start : end + 1]
        if window_high.empty:
            result.iloc[end] = np.nan
            continue
        if window_high.dropna().empty:
            result.iloc[end] = np.nan
            continue
        anchor_label = window_high.idxmax()
        anchor_loc = close.index.get_loc(anchor_label)
        pv = (typical.iloc[anchor_loc : end + 1] * volume.iloc[anchor_loc : end + 1]).sum()
        vv = volume.iloc[anchor_loc : end + 1].sum()
        result.iloc[end] = pv / vv if vv else np.nan
    return result


def _normalize_session_dates(values) -> pd.Series:
    series = pd.to_datetime(pd.Series(values), errors="coerce")
    if getattr(series.dt, "tz", None) is not None:
        series = series.dt.tz_localize(None)
    return series.dt.normalize()


def _daily_bar_session_dates(values) -> pd.Series:
    series = pd.to_datetime(pd.Series(values), errors="coerce")
    if getattr(series.dt, "tz", None) is not None:
        series = series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series.dt.normalize()


def _ensure_market_timezone(values, market_timezone: str = MARKET_TIMEZONE) -> pd.Series:
    series = pd.to_datetime(pd.Series(values), errors="coerce")
    if getattr(series.dt, "tz", None) is None:
        return series.dt.tz_localize(market_timezone, ambiguous="NaT", nonexistent="shift_forward")
    return series.dt.tz_convert(market_timezone)


def build_daily_tradeability_flags(
    daily_bars: pd.DataFrame,
    *,
    min_price: float,
    min_daily_volume: float,
    min_dollar_volume: float,
) -> pd.DataFrame:
    if daily_bars.empty:
        return pd.DataFrame(columns=["session_date", "symbol", "close", "volume", "dollar_volume", "is_eligible"])

    work = daily_bars.copy()
    work["session_date"] = _daily_bar_session_dates(work["ts"])
    work = work.dropna(subset=["session_date", "symbol"]).copy()
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce")
    work["dollar_volume"] = work["close"] * work["volume"]
    work["is_eligible"] = (
        work["close"].ge(float(min_price))
        & work["volume"].ge(float(min_daily_volume))
        & work["dollar_volume"].ge(float(min_dollar_volume))
    ).astype("int8")
    return work[["session_date", "symbol", "close", "volume", "dollar_volume", "is_eligible"]].copy()


def build_daily_feature_history(daily_bars: pd.DataFrame, universe: pd.DataFrame, spy_symbol: str = "SPY") -> pd.DataFrame:
    if daily_bars.empty or universe.empty:
        return pd.DataFrame(columns=["session_date", "symbol", *STANDARD_FEATURE_COLUMNS, "prev_day_high", "prev_day_atr14"])

    work = daily_bars.copy()
    work["session_date"] = _daily_bar_session_dates(work["ts"])
    work = work.dropna(subset=["session_date"]).sort_values(["symbol", "session_date"]).reset_index(drop=True)
    work["adj_close"] = work["adj_close"].fillna(work["close"])
    metadata = universe[["symbol", "sector", "industry"]].drop_duplicates(subset=["symbol"]).copy()
    work = work.merge(metadata, on="symbol", how="left")

    feature_frames: list[pd.DataFrame] = []
    for symbol, frame in work.groupby("symbol", sort=False):
        frame = frame.sort_values("session_date").copy()
        frame["atr14"] = _true_range(frame["high"], frame["low"], frame["adj_close"]).rolling(14, min_periods=14).mean()
        frame["atr_pct"] = _safe_div(frame["atr14"], frame["adj_close"])
        frame["adr_pct_20"] = ((frame["high"] / frame["low"].replace(0, np.nan)) - 1.0).rolling(20, min_periods=10).mean()
        frame["sma50"] = frame["adj_close"].rolling(50, min_periods=50).mean()
        money_flow_multiplier = ((frame["adj_close"] - frame["low"]) - (frame["high"] - frame["adj_close"])) / (frame["high"] - frame["low"]).replace(0, np.nan)
        frame["buy_pressure_20"] = money_flow_multiplier.rolling(20, min_periods=10).mean()
        frame["anchored_vwap_63"] = _anchored_vwap(frame["high"], frame["low"], frame["adj_close"], frame["volume"], lookback=63)
        frame["distance_to_avwap_63"] = (frame["adj_close"] / frame["anchored_vwap_63"]) - 1.0
        frame["roc_21"] = ((frame["adj_close"] / frame["adj_close"].shift(21)) - 1.0) * 100.0
        frame["roc_63"] = ((frame["adj_close"] / frame["adj_close"].shift(63)) - 1.0) * 100.0
        frame["roc_126"] = ((frame["adj_close"] / frame["adj_close"].shift(126)) - 1.0) * 100.0
        frame["roc_252"] = ((frame["adj_close"] / frame["adj_close"].shift(252)) - 1.0) * 100.0
        frame["daily_rs_score"] = (
            0.40 * frame["roc_21"].fillna(0.0)
            + 0.20 * frame["roc_63"].fillna(0.0)
            + 0.20 * frame["roc_126"].fillna(0.0)
            + 0.20 * frame["roc_252"].fillna(0.0)
        )
        frame["daily_rrs"] = frame["daily_rs_score"] / frame["atr_pct"].replace(0, np.nan)
        frame["prev_day_close_vs_sma50"] = (frame["adj_close"] / frame["sma50"]) - 1.0
        frame["prev_day_high"] = frame["high"].shift(1)
        frame["prev_day_atr14"] = frame["atr14"].shift(1)
        feature_frames.append(frame)

    features = pd.concat(feature_frames, ignore_index=True)

    sector_daily = (
        features.groupby(["session_date", "sector"], observed=True)["buy_pressure_20"]
        .mean()
        .reset_index(name="sector_buy_pressure")
        .sort_values(["sector", "session_date"])
    )
    sector_daily["sector_buy_pressure_prev"] = sector_daily.groupby("sector", observed=True)["sector_buy_pressure"].shift(1)

    industry_bp_daily = (
        features.groupby(["session_date", "industry"], observed=True)["buy_pressure_20"]
        .mean()
        .reset_index(name="industry_buy_pressure")
        .sort_values(["industry", "session_date"])
    )
    industry_bp_daily["industry_buy_pressure_prev"] = industry_bp_daily.groupby("industry", observed=True)["industry_buy_pressure"].shift(1)

    industry_rs_daily = (
        features.groupby(["session_date", "industry"], observed=True)["daily_rs_score"]
        .mean()
        .reset_index(name="industry_rs")
        .sort_values(["industry", "session_date"])
    )
    industry_rs_daily["industry_rs_prev"] = industry_rs_daily.groupby("industry", observed=True)["industry_rs"].shift(1)

    features = features.merge(
        sector_daily[["session_date", "sector", "sector_buy_pressure"]],
        on=["session_date", "sector"],
        how="left",
    )
    features = features.merge(
        sector_daily[["session_date", "sector", "sector_buy_pressure_prev"]],
        on=["session_date", "sector"],
        how="left",
    )
    features = features.merge(
        industry_bp_daily[["session_date", "industry", "industry_buy_pressure"]],
        on=["session_date", "industry"],
        how="left",
    )
    features = features.merge(
        industry_bp_daily[["session_date", "industry", "industry_buy_pressure_prev"]],
        on=["session_date", "industry"],
        how="left",
    )
    features = features.merge(
        industry_rs_daily[["session_date", "industry", "industry_rs"]],
        on=["session_date", "industry"],
        how="left",
    )
    features = features.merge(
        industry_rs_daily[["session_date", "industry", "industry_rs_prev"]],
        on=["session_date", "industry"],
        how="left",
    )

    features["daily_buy_pressure_eod"] = features["buy_pressure_20"]
    features["adr_pct_20_eod"] = features["adr_pct_20"]
    features["daily_rrs_eod"] = features["daily_rrs"]
    features["daily_rs_score_eod"] = features["daily_rs_score"]
    features["sector_buy_pressure_eod"] = features["sector_buy_pressure"]
    features["industry_buy_pressure_eod"] = features["industry_buy_pressure"]
    features["industry_rs_eod"] = features["industry_rs"]

    features["daily_buy_pressure_prev"] = features.groupby("symbol", sort=False)["buy_pressure_20"].shift(1)
    features["prev_day_adr_pct"] = features.groupby("symbol", sort=False)["adr_pct_20"].shift(1)
    features["daily_rrs_prev"] = features.groupby("symbol", sort=False)["daily_rrs"].shift(1)
    features["daily_rs_score_prev"] = features.groupby("symbol", sort=False)["daily_rs_score"].shift(1)
    features["distance_to_avwap_63_prev"] = features.groupby("symbol", sort=False)["distance_to_avwap_63"].shift(1)
    features["prev_day_close_vs_sma50"] = features.groupby("symbol", sort=False)["prev_day_close_vs_sma50"].shift(1)

    output_columns = [
        "session_date",
        "symbol",
        "daily_buy_pressure_eod",
        "adr_pct_20_eod",
        "industry_buy_pressure_eod",
        "sector_buy_pressure_eod",
        "daily_rrs_eod",
        "daily_rs_score_eod",
        "industry_rs_eod",
        "daily_buy_pressure_prev",
        "prev_day_adr_pct",
        "industry_buy_pressure_prev",
        "sector_buy_pressure_prev",
        "daily_rrs_prev",
        "daily_rs_score_prev",
        "distance_to_avwap_63_prev",
        "industry_rs_prev",
        "prev_day_close_vs_sma50",
        "prev_day_high",
        "prev_day_atr14",
        "sector",
        "industry",
    ]
    result = features[output_columns].dropna(subset=["session_date"]).copy()
    result["session_date"] = pd.to_datetime(result["session_date"]).dt.normalize()
    return result


def build_intraday_feature_panel(
    intraday_bars: pd.DataFrame,
    daily_features: pd.DataFrame,
    same_slot_lookback_sessions: int = 20,
    symbol_chunk_size: int = 100,
    allowed_session_buckets: tuple[str, ...] | None = None,
    min_minutes_from_open: int | None = None,
    max_minutes_from_open: int | None = None,
    parquet_spill_dir: str | Path | None = None,
) -> pd.DataFrame:
    if intraday_bars.empty or daily_features.empty:
        return pd.DataFrame(columns=["timestamp", "session_date", "symbol", "minutes_from_open", "session_bucket", *STANDARD_FEATURE_COLUMNS])

    work = intraday_bars.copy()
    keep_columns = ["symbol", "ts", "open", "high", "low", "close", "volume"]
    work = work[[column for column in keep_columns if column in work.columns]].copy()
    work["timestamp"] = pd.to_datetime(work["ts"], utc=True, errors="coerce").dt.tz_convert(MARKET_TIMEZONE)
    work = work.dropna(subset=["timestamp"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    work["session_date"] = _normalize_session_dates(work["timestamp"])
    open_minutes = work["timestamp"].dt.hour * 60 + work["timestamp"].dt.minute
    work["minutes_from_open"] = open_minutes - (9 * 60 + 30)
    work = work[work["minutes_from_open"] >= 0].copy()
    work["slot_index"] = (work["minutes_from_open"] // 5).astype("int16")
    work["session_bucket"] = session_bucket_from_minutes(work["minutes_from_open"])

    daily_lookup_columns = [
        "symbol",
        "session_date",
        "daily_buy_pressure_prev",
        "prev_day_adr_pct",
        "industry_buy_pressure_prev",
        "sector_buy_pressure_prev",
        "daily_rrs_prev",
        "daily_rs_score_prev",
        "distance_to_avwap_63_prev",
        "industry_rs_prev",
        "prev_day_close_vs_sma50",
        "prev_day_high",
        "prev_day_atr14",
    ]
    daily = daily_features[daily_lookup_columns].copy()
    daily["session_date"] = _normalize_session_dates(daily["session_date"])

    def _downcast_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        work_frame = frame.copy()
        for column in work_frame.select_dtypes(include=["float64"]).columns:
            work_frame[column] = pd.to_numeric(work_frame[column], downcast="float")
        for column in work_frame.select_dtypes(include=["int64", "int32"]).columns:
            work_frame[column] = pd.to_numeric(work_frame[column], downcast="integer")
        return work_frame

    def _build_symbol_frame(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values("timestamp").copy()
        date_group = frame.groupby("session_date", sort=False)
        frame["session_high_so_far"] = date_group["high"].cummax()
        frame["session_low_so_far"] = date_group["low"].cummin()
        frame["session_close"] = date_group["close"].transform("last")
        frame["next_open"] = date_group["open"].shift(-1)
        frame["next_timestamp"] = date_group["timestamp"].shift(-1)
        typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
        frame["session_pv"] = typical * frame["volume"]
        frame["cum_pv"] = date_group["session_pv"].cumsum()
        frame["cum_vol"] = date_group["volume"].cumsum()
        frame["session_vwap"] = _safe_div(frame["cum_pv"], frame["cum_vol"])
        boundary_mask = (frame["minutes_from_open"] % 15 == 0) & (frame["minutes_from_open"] >= 15)
        frame["close_vs_vwap_15"] = np.nan
        frame.loc[boundary_mask, "close_vs_vwap_15"] = (frame.loc[boundary_mask, "close"] / frame.loc[boundary_mask, "session_vwap"]) - 1.0
        frame["close_vs_vwap_15"] = frame.groupby("session_date", sort=False)["close_vs_vwap_15"].ffill()

        fifteen = frame.loc[boundary_mask, ["timestamp", "close"]].copy()
        fifteen["EMA_8_15"] = _ema(fifteen["close"], 8)
        frame = frame.merge(fifteen[["timestamp", "EMA_8_15"]], on="timestamp", how="left")
        # Preserve the pre-chunk semantics: carry the last 15m EMA forward across session boundaries
        # until the first 15m boundary of the new session is available.
        frame["EMA_8_15"] = frame["EMA_8_15"].ffill()

        frame["same_slot_avg_vol_20d"] = (
            frame.groupby("slot_index", sort=False)["volume"]
            .transform(lambda s: s.shift(1).rolling(same_slot_lookback_sessions, min_periods=5).mean())
        )
        frame["intraday_rvol"] = frame["volume"] / frame["same_slot_avg_vol_20d"].replace(0, np.nan)
        frame["volume_spike_5m"] = frame["intraday_rvol"] - 1.0
        if allowed_session_buckets:
            frame = frame.loc[frame["session_bucket"].isin(allowed_session_buckets)].copy()
        if min_minutes_from_open is not None:
            frame = frame.loc[frame["minutes_from_open"] >= min_minutes_from_open].copy()
        if max_minutes_from_open is not None:
            frame = frame.loc[frame["minutes_from_open"] <= max_minutes_from_open].copy()
        return frame[
            [
                "timestamp",
                "session_date",
                "symbol",
                "minutes_from_open",
                "session_bucket",
                "EMA_8_15",
                "close_vs_vwap_15",
                "same_slot_avg_vol_20d",
                "intraday_rvol",
                "volume_spike_5m",
                "session_high_so_far",
                "session_low_so_far",
                "next_open",
                "next_timestamp",
                "session_close",
                "close",
            ]
        ].copy()

    def _finalize_chunk(chunk_frames: list[pd.DataFrame], chunk_symbols: list[str]) -> pd.DataFrame:
        if not chunk_frames:
            return pd.DataFrame()
        panel = pd.concat(chunk_frames, ignore_index=True)
        daily_chunk = daily.loc[daily["symbol"].isin(chunk_symbols)].copy()
        panel = panel.merge(daily_chunk, on=["symbol", "session_date"], how="left")
        panel["distance_to_prev_day_high"] = (panel["close"] / panel["prev_day_high"]) - 1.0
        panel["rs_x_intraday_rvol"] = panel["daily_rs_score_prev"] * panel["intraday_rvol"]
        panel["intraday_range_expansion_vs_atr"] = (panel["session_high_so_far"] - panel["session_low_so_far"]) / panel["prev_day_atr14"].replace(0, np.nan)
        panel["timestamp"] = _ensure_market_timezone(panel["timestamp"], market_timezone=MARKET_TIMEZONE)
        panel["next_timestamp"] = _ensure_market_timezone(panel["next_timestamp"], market_timezone=MARKET_TIMEZONE)
        panel = panel[
            [
                "timestamp",
                "session_date",
                "symbol",
                "minutes_from_open",
                "session_bucket",
                *STANDARD_FEATURE_COLUMNS,
                "next_open",
                "next_timestamp",
                "session_close",
                "close",
            ]
        ].copy()
        return _downcast_frame(panel)

    spill_paths: list[Path] = []
    collected_chunks: list[pd.DataFrame] = []
    chunk_frames: list[pd.DataFrame] = []
    chunk_symbols: list[str] = []
    spill_root = Path(parquet_spill_dir) if parquet_spill_dir is not None else None
    if spill_root is not None:
        spill_root.mkdir(parents=True, exist_ok=True)

    chunk_index = 0
    for symbol, symbol_frame in work.groupby("symbol", sort=False, observed=True):
        processed = _build_symbol_frame(symbol_frame)
        if not processed.empty:
            chunk_frames.append(processed)
            chunk_symbols.append(str(symbol))
        if len(chunk_symbols) < max(1, int(symbol_chunk_size)):
            continue
        finalized = _finalize_chunk(chunk_frames, chunk_symbols)
        if not finalized.empty:
            if spill_root is not None:
                path = spill_root / f"intraday_feature_chunk_{chunk_index:04d}.parquet"
                finalized.to_parquet(path, index=False)
                spill_paths.append(path)
            else:
                collected_chunks.append(finalized)
        chunk_index += 1
        chunk_frames.clear()
        chunk_symbols.clear()
        gc.collect()
        if chunk_index % 5 == 0:
            LOGGER.info("Built %s intraday feature chunks", chunk_index)

    if chunk_symbols:
        finalized = _finalize_chunk(chunk_frames, chunk_symbols)
        if not finalized.empty:
            if spill_root is not None:
                path = spill_root / f"intraday_feature_chunk_{chunk_index:04d}.parquet"
                finalized.to_parquet(path, index=False)
                spill_paths.append(path)
            else:
                collected_chunks.append(finalized)
        gc.collect()

    if spill_paths:
        return pd.concat((pd.read_parquet(path) for path in spill_paths), ignore_index=True)
    if collected_chunks:
        return pd.concat(collected_chunks, ignore_index=True)
    return pd.DataFrame(columns=["timestamp", "session_date", "symbol", "minutes_from_open", "session_bucket", *STANDARD_FEATURE_COLUMNS])


def build_training_labels(feature_panel: pd.DataFrame, commission_rate_one_way: float, slippage_bps_per_side: float, spread_bps_round_trip: float, adverse_fill_floor: float, adverse_fill_cap: float) -> pd.DataFrame:
    if feature_panel.empty:
        return feature_panel.copy()

    work = feature_panel.copy()
    slippage_side = slippage_bps_per_side / 10_000.0
    spread_half = (spread_bps_round_trip / 2.0) / 10_000.0
    base_entry_markup = slippage_side + spread_half
    base_exit_markdown = slippage_side + spread_half

    next_bar_range_pct = ((work["session_close"] - work["next_open"]).abs() / work["next_open"].replace(0, np.nan)).fillna(0.0)
    adverse_fill_penalty = (0.0005 + (0.25 * next_bar_range_pct)).clip(lower=adverse_fill_floor, upper=adverse_fill_cap)
    entry_fill = work["next_open"] * (1.0 + base_entry_markup + adverse_fill_penalty)
    exit_fill = work["session_close"] * (1.0 - base_exit_markdown)
    work["net_return_stress_exec"] = ((exit_fill * (1.0 - commission_rate_one_way)) / (entry_fill * (1.0 + commission_rate_one_way))) - 1.0
    work["label_stress_exec"] = (work["net_return_stress_exec"] > 0).astype("int8")
    return work


def build_stage2_labeled_panel(
    intraday_bars: pd.DataFrame,
    daily_features: pd.DataFrame,
    same_slot_lookback_sessions: int,
    min_minutes_from_open: int,
    max_minutes_from_open: int,
    commission_rate_one_way: float,
    slippage_bps_per_side: float,
    spread_bps_round_trip: float,
    adverse_fill_floor: float,
    adverse_fill_cap: float,
    symbol_chunk_size: int = 100,
    spill_parent_dir: str | Path | None = None,
) -> pd.DataFrame:
    empty_columns = [
        "timestamp",
        "session_date",
        "symbol",
        "minutes_from_open",
        "session_bucket",
        *STANDARD_FEATURE_COLUMNS,
        "next_open",
        "next_timestamp",
        "session_close",
        "close",
        "net_return_stress_exec",
        "label_stress_exec",
    ]
    if intraday_bars.empty or daily_features.empty:
        return pd.DataFrame(columns=empty_columns)

    temp_root = Path(spill_parent_dir) if spill_parent_dir is not None else None
    if temp_root is not None:
        temp_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=temp_root) as temp_dir_name:
        feature_spill_dir = Path(temp_dir_name) / "feature_chunks"
        label_spill_dir = Path(temp_dir_name) / "label_chunks"
        feature_spill_dir.mkdir(parents=True, exist_ok=True)
        label_spill_dir.mkdir(parents=True, exist_ok=True)

        _ = build_intraday_feature_panel(
            intraday_bars,
            daily_features,
            same_slot_lookback_sessions=same_slot_lookback_sessions,
            symbol_chunk_size=symbol_chunk_size,
            allowed_session_buckets=("open_drive",),
            min_minutes_from_open=min_minutes_from_open,
            max_minutes_from_open=max_minutes_from_open,
            parquet_spill_dir=feature_spill_dir,
        )

        label_paths: list[Path] = []
        for chunk_index, feature_path in enumerate(sorted(feature_spill_dir.glob("intraday_feature_chunk_*.parquet"))):
            feature_chunk = pd.read_parquet(feature_path)
            labeled_chunk = build_training_labels(
                feature_chunk,
                commission_rate_one_way=commission_rate_one_way,
                slippage_bps_per_side=slippage_bps_per_side,
                spread_bps_round_trip=spread_bps_round_trip,
                adverse_fill_floor=adverse_fill_floor,
                adverse_fill_cap=adverse_fill_cap,
            ).dropna(subset=["next_open", "session_close"])
            if labeled_chunk.empty:
                continue
            label_path = label_spill_dir / f"labeled_chunk_{chunk_index:04d}.parquet"
            for column in labeled_chunk.select_dtypes(include=["float64"]).columns:
                labeled_chunk[column] = pd.to_numeric(labeled_chunk[column], downcast="float")
            for column in labeled_chunk.select_dtypes(include=["int64", "int32"]).columns:
                labeled_chunk[column] = pd.to_numeric(labeled_chunk[column], downcast="integer")
            labeled_chunk.to_parquet(label_path, index=False)
            label_paths.append(label_path)
            del feature_chunk, labeled_chunk
            gc.collect()
            if (chunk_index + 1) % 5 == 0:
                LOGGER.info("Built %s labeled intraday chunks", chunk_index + 1)

        if not label_paths:
            return pd.DataFrame(columns=empty_columns)
        return pd.concat((pd.read_parquet(path) for path in label_paths), ignore_index=True)
