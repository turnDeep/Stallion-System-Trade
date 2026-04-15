from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .breakout_bridge import BreakoutConfig, build_breakout_signal_report
from .config import Settings, load_settings
from .fmp import FMPClient, download_yfinance_bars
from .storage import SQLiteParquetStore


LOGGER = logging.getLogger(__name__)


def _build_daily_summary(report: pd.DataFrame) -> pd.DataFrame:
    if report.empty:
        return pd.DataFrame()
    
    # breakout_signal_report.py 側の summary を優先
    # が、念の為ここでも集計ロジックを更新
    return (
        report.groupby("date", as_index=False)
        .agg(
            setup_count=("setup_candidate", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            breakout_signal_count=("breakout_signal", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            standard_breakout_count=("standard_breakout_signal", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum()) if "standard_breakout_signal" in report.columns else 0),
            zigzag_breakout_count=("zigzag_breakout_signal", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum()) if "zigzag_breakout_signal" in report.columns else 0),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def _symbol_preview(symbols: list[str], limit: int = 12) -> str:
    if not symbols:
        return "-"
    preview = ", ".join(symbols[:limit])
    if len(symbols) > limit:
        preview += ", ..."
    return preview


def _expected_latest_ts(
    store: SQLiteParquetStore,
    timeframe: str,
    symbols: list[str],
) -> pd.Timestamp | None:
    latest_by_symbol = store.get_latest_timestamps_by_symbol(timeframe, symbols)
    if latest_by_symbol.empty:
        return None
    latest_ts = latest_by_symbol["latest_ts"].max()
    return None if pd.isna(latest_ts) else latest_ts


def _repair_symbol_gaps(
    *,
    store: SQLiteParquetStore,
    timeframe: str,
    symbols: list[str],
    interval: str,
    bootstrap_period: str,
    stale_tolerance_days: float,
    overlap_days: int,
    max_lookback_days: int | None = None,
) -> pd.DataFrame:
    audit = store.audit_symbol_gaps(timeframe, symbols, tolerance_days=stale_tolerance_days)
    missing_symbols = audit.loc[audit["status"].eq("missing"), "symbol"].tolist()
    stale_rows = audit.loc[audit["status"].eq("stale") & audit["latest_ts"].notna()].copy()
    repaired_frames: list[pd.DataFrame] = []

    if stale_rows.empty and not missing_symbols:
        LOGGER.info("%s symbol audit passed for %s symbols", timeframe, len(audit))
        return audit

    stale_symbols = stale_rows["symbol"].tolist()
    if stale_symbols:
        stale_start = stale_rows["latest_ts"].min() - pd.Timedelta(days=overlap_days)
        if max_lookback_days is not None:
            min_allowed = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max_lookback_days)
            stale_start = max(stale_start, min_allowed)
        stale_start_text = stale_start.strftime("%Y-%m-%d")
        LOGGER.warning(
            "%s symbol audit found %s stale symbols. Repairing from %s. preview=%s",
            timeframe,
            len(stale_symbols),
            stale_start_text,
            _symbol_preview(stale_symbols),
        )
        repaired_frames.append(
            download_yfinance_bars(stale_symbols, interval=interval, start=stale_start_text)
        )

    if missing_symbols:
        LOGGER.warning(
            "%s symbol audit found %s missing symbols. Repairing with bootstrap window %s. preview=%s",
            timeframe,
            len(missing_symbols),
            bootstrap_period,
            _symbol_preview(missing_symbols),
        )
        repaired_frames.append(
            download_yfinance_bars(missing_symbols, period=bootstrap_period, interval=interval)
        )

    if repaired_frames:
        repaired = pd.concat(repaired_frames, ignore_index=True)
        if timeframe == "5m" and not repaired.empty:
            repaired["cumulative_volume"] = repaired["volume"]
        store.save_bars(repaired, timeframe=timeframe)

    final_audit = store.audit_symbol_gaps(timeframe, symbols, tolerance_days=stale_tolerance_days)
    remaining = final_audit.loc[final_audit["status"].ne("fresh")].copy()
    if not remaining.empty:
        remaining_missing = remaining.loc[remaining["status"].eq("missing"), "symbol"].tolist()
        remaining_stale = remaining.loc[remaining["status"].eq("stale"), "symbol"].tolist()
        LOGGER.warning(
            "%s symbol audit still has %s missing and %s stale symbols after repair. missing=%s stale=%s",
            timeframe,
            len(remaining_missing),
            len(remaining_stale),
            _symbol_preview(remaining_missing),
            _symbol_preview(remaining_stale),
        )
    else:
        LOGGER.info("%s symbol repair complete for %s symbols", timeframe, len(final_audit))
    return final_audit


def run_nightly_pipeline(settings: Settings | None = None) -> dict[str, Path]:
    settings = settings or load_settings()
    store = SQLiteParquetStore(settings)
    fmp = FMPClient(settings)
    cfg = BreakoutConfig.from_settings(settings)

    LOGGER.info("Fetching top %s universe from FMP", settings.runtime.top_n_universe)
    universe = fmp.fetch_top_universe(settings.runtime.top_n_universe)
    store.save_universe(universe)

    symbols = universe["symbol"].tolist()
    benchmark_symbols = ["SPY"]
    daily_symbols = symbols + benchmark_symbols

    LOGGER.info("Downloading daily bars via yfinance for %s symbols", len(daily_symbols))
    daily_latest_ts = _expected_latest_ts(store, "1d", daily_symbols)
    if daily_latest_ts is not None:
        daily_start = (daily_latest_ts - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        LOGGER.info("Differential daily fetch: start=%s (latest_ts=%s)", daily_start, daily_latest_ts.date())
        daily_bars = download_yfinance_bars(daily_symbols, interval="1d", start=daily_start)
    else:
        LOGGER.info("No existing daily bars found - performing full 2y bootstrap download.")
        daily_bars = download_yfinance_bars(daily_symbols, period="2y", interval="1d")
    store.save_bars(daily_bars, timeframe="1d")
    daily_gap_audit = _repair_symbol_gaps(
        store=store,
        timeframe="1d",
        symbols=daily_symbols,
        interval="1d",
        bootstrap_period="2y",
        stale_tolerance_days=1.0,
        overlap_days=2,
        max_lookback_days=730,
    )

    intraday_limit_days = 59
    LOGGER.info("Downloading 5m bars via yfinance for %s symbols", len(symbols))
    intraday_latest_ts = _expected_latest_ts(store, "5m", symbols)
    if intraday_latest_ts is not None:
        gap_days = (pd.Timestamp.utcnow() - intraday_latest_ts.tz_convert("UTC")).total_seconds() / 86400.0
        fetch_days = min(int(gap_days) + 2, intraday_limit_days)
        if gap_days > intraday_limit_days:
            LOGGER.warning(
                "5m data gap of %.1f days exceeds Yahoo 60-day limit - capping fetch at %d days. "
                "Permanent hole of about %.0f days exists.",
                gap_days,
                intraday_limit_days,
                gap_days - intraday_limit_days,
            )
        intraday_start = (pd.Timestamp.utcnow() - pd.Timedelta(days=fetch_days)).strftime("%Y-%m-%d")
        LOGGER.info(
            "Differential 5m fetch: start=%s (gap=%.1f days, fetch_days=%d)",
            intraday_start,
            gap_days,
            fetch_days,
        )
        intraday_bars = download_yfinance_bars(symbols, interval="5m", start=intraday_start)
    else:
        LOGGER.info("No existing 5m bars found - performing full %d-day bootstrap download.", intraday_limit_days)
        intraday_bars = download_yfinance_bars(symbols, period=f"{intraday_limit_days}d", interval="5m")
    intraday_bars["cumulative_volume"] = intraday_bars["volume"]
    store.save_bars(intraday_bars, timeframe="5m")
    intraday_gap_audit = _repair_symbol_gaps(
        store=store,
        timeframe="5m",
        symbols=symbols,
        interval="5m",
        bootstrap_period=f"{intraday_limit_days}d",
        stale_tolerance_days=1.0,
        overlap_days=2,
        max_lookback_days=intraday_limit_days,
    )

    LOGGER.info("Reloading full daily history from SQLite for signal scoring...")
    full_daily_bars = store.load_bars("1d")
    full_daily_bars["ts"] = pd.to_datetime(full_daily_bars["ts"], utc=True, errors="coerce")
    full_daily_bars = full_daily_bars.rename(columns={"ts": "date"})
    full_daily_bars["date"] = full_daily_bars["date"].dt.normalize()

    LOGGER.info("Reloading full 5m intraday history from SQLite for signal scoring...")
    full_intraday_bars = store.load_bars("5m")
    full_intraday_bars["ts"] = pd.to_datetime(full_intraday_bars["ts"], utc=True, errors="coerce")

    daily_signal_bars = full_daily_bars.loc[full_daily_bars["symbol"].ne("SPY")].copy()
    report, summary = build_breakout_signal_report(daily_signal_bars, full_intraday_bars, cfg=cfg)

    latest_date = pd.to_datetime(report["date"]).max().normalize() if not report.empty else pd.Timestamp.utcnow().normalize()
    
    # shortlist 生成: セットアップ候補 or 当日組シグナル発生銘柄
    shortlist_pool = report.loc[report["date"].eq(latest_date)].copy()
    mask = shortlist_pool["setup_candidate"].fillna(False) | shortlist_pool["breakout_signal"].fillna(False)
    
    sort_cols = []
    ascending = []
    if "entry_priority_bucket" in shortlist_pool.columns:
        sort_cols.append("entry_priority_bucket")
        ascending.append(True)
    if "priority_score_within_source" in shortlist_pool.columns:
        sort_cols.append("priority_score_within_source")
        ascending.append(False)
    
    sort_cols.extend(["leader_score", "setup_score_pre", "symbol"])
    ascending.extend([False, False, True])

    shortlist = (
        shortlist_pool.loc[mask]
        .sort_values(sort_cols, ascending=ascending, kind="mergesort")
        .head(settings.runtime.shortlist_count)
        .copy()
    )
    shortlist["session_date"] = latest_date
    shortlist["next_session_date"] = latest_date + pd.offsets.BDay(1)

    shortlist_session_key = latest_date + pd.offsets.BDay(1)
    if not shortlist.empty:
        shortlist_session_key = pd.to_datetime(shortlist["next_session_date"]).iloc[0].normalize()
        store.save_shortlist(shortlist_session_key, shortlist)
        shortlist.to_parquet(settings.paths.watchlist_path, index=False)
    else:
        settings.paths.watchlist_path.parent.mkdir(parents=True, exist_ok=True)
        shortlist.to_parquet(settings.paths.watchlist_path, index=False)

    report_path = settings.paths.reports_dir / "breakout_signal_report.parquet"
    summary_path = settings.paths.reports_dir / "breakout_signal_daily_summary.csv"
    nightly_report_path = settings.paths.reports_dir / "nightly_pipeline_report.json"

    report.to_parquet(report_path, index=False)
    summary.to_csv(summary_path, index=False)

    nightly_report = {
        "session_date": str(latest_date.date()),
        "universe_count": int(len(universe)),
        "daily_rows_fetched": int(len(daily_bars)),
        "intraday_rows_fetched": int(len(intraday_bars)),
        "daily_rows_total": int(len(full_daily_bars)),
        "intraday_rows_total": int(len(full_intraday_bars)),
        "report_rows": int(len(report)),
        "breakout_signal_count": int(report["breakout_signal"].fillna(False).sum()) if not report.empty else 0,
        "shortlist_count": int(len(shortlist)),
        "daily_symbol_missing_count": int(daily_gap_audit["status"].eq("missing").sum()),
        "daily_symbol_stale_count": int(daily_gap_audit["status"].eq("stale").sum()),
        "intraday_symbol_missing_count": int(intraday_gap_audit["status"].eq("missing").sum()),
        "intraday_symbol_stale_count": int(intraday_gap_audit["status"].eq("stale").sum()),
        "watchlist_path": str(settings.paths.watchlist_path),
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "shortlist_session_key": str(pd.Timestamp(shortlist_session_key).date()),
    }
    nightly_report_path.write_text(json.dumps(nightly_report, indent=2), encoding="utf-8")

    LOGGER.info("Nightly breakout pipeline complete for %s", latest_date.date())
    return {
        "report_path": nightly_report_path,
        "watchlist_path": settings.paths.watchlist_path,
        "signal_report_path": report_path,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    outputs = run_nightly_pipeline()
    print(f"Nightly breakout pipeline complete: {outputs['report_path']}")


if __name__ == "__main__":
    main()
