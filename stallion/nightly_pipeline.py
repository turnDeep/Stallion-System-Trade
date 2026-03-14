from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .config import Settings, load_settings
from .features import build_daily_feature_history, build_intraday_feature_panel, build_training_labels
from .fmp import FMPClient, download_yfinance_bars
from .modeling import fit_hist_gbm, save_model_bundle
from .storage import SQLiteParquetStore
from .strategy import StandardSystemSpec


LOGGER = logging.getLogger(__name__)


def _latest_session_date(feature_frame: pd.DataFrame) -> pd.Timestamp:
    if feature_frame.empty:
        raise ValueError("No daily feature rows available.")
    return pd.to_datetime(feature_frame["session_date"]).max().normalize()


def _build_shortlist(daily_features: pd.DataFrame, session_date: pd.Timestamp, shortlist_count: int) -> pd.DataFrame:
    latest = daily_features.loc[pd.to_datetime(daily_features["session_date"]).eq(session_date)].copy()
    if latest.empty:
        return latest
    latest["shortlist_score"] = (
        latest["daily_buy_pressure_prev"].rank(pct=True).fillna(0.0) * 0.20
        + latest["daily_rs_score_prev"].rank(pct=True).fillna(0.0) * 0.25
        + latest["daily_rrs_prev"].rank(pct=True).fillna(0.0) * 0.20
        + latest["prev_day_adr_pct"].rank(pct=True).fillna(0.0) * 0.10
        + latest["industry_buy_pressure_prev"].rank(pct=True).fillna(0.0) * 0.10
        + latest["sector_buy_pressure_prev"].rank(pct=True).fillna(0.0) * 0.05
        + latest["industry_rs_prev"].rank(pct=True).fillna(0.0) * 0.10
    )
    latest = latest.sort_values(["shortlist_score", "daily_rs_score_prev"], ascending=[False, False]).head(shortlist_count).copy()
    latest["rank_order"] = range(1, len(latest) + 1)
    return latest


def run_nightly_pipeline(settings: Settings | None = None) -> dict[str, Path]:
    settings = settings or load_settings()
    store = SQLiteParquetStore(settings)
    fmp = FMPClient(settings)
    spec = StandardSystemSpec(
        min_minutes_from_open=settings.runtime.min_minutes_from_open,
        max_minutes_from_open=settings.runtime.max_minutes_from_open,
        max_positions=settings.runtime.max_positions,
        threshold_floor=settings.runtime.threshold_floor,
        threshold_quantile=settings.runtime.threshold_quantile,
    )

    LOGGER.info("Fetching top %s universe from FMP", settings.runtime.top_n_universe)
    universe = fmp.fetch_top_universe(settings.runtime.top_n_universe)
    store.save_universe(universe)

    symbols = universe["symbol"].tolist()
    benchmark_symbols = ["SPY"]
    LOGGER.info("Downloading daily bars via yfinance for %s symbols", len(symbols) + len(benchmark_symbols))
    daily_bars = download_yfinance_bars(symbols + benchmark_symbols, period="2y", interval="1d")
    store.save_bars(daily_bars, timeframe="1d")

    LOGGER.info("Downloading 5m bars via yfinance for %s symbols", len(symbols))
    intraday_bars = download_yfinance_bars(symbols, period=f"{min(60, settings.runtime.intraday_history_sessions)}d", interval="5m")
    intraday_bars["cumulative_volume"] = intraday_bars["volume"]
    store.save_bars(intraday_bars, timeframe="5m")

    LOGGER.info("Building daily feature history")
    daily_features = build_daily_feature_history(daily_bars, universe, spy_symbol="SPY")
    store.save_daily_features(daily_features)

    LOGGER.info("Building intraday training panel")
    intraday_features = build_intraday_feature_panel(intraday_bars, daily_features, same_slot_lookback_sessions=settings.runtime.same_slot_lookback_sessions)
    intraday_features = intraday_features[intraday_features["session_bucket"].eq("open_drive")].copy()
    intraday_features = intraday_features[intraday_features["minutes_from_open"].between(spec.min_minutes_from_open, spec.max_minutes_from_open, inclusive="both")].copy()
    labeled = build_training_labels(
        intraday_features,
        commission_rate_one_way=settings.costs.commission_rate_one_way,
        slippage_bps_per_side=settings.costs.slippage_bps_per_side,
        spread_bps_round_trip=settings.costs.spread_bps_round_trip,
        adverse_fill_floor=settings.costs.extra_adverse_fill_floor,
        adverse_fill_cap=settings.costs.extra_adverse_fill_cap,
    )

    if len(labeled) > spec.max_train_rows:
        labeled = labeled.tail(spec.max_train_rows).copy()

    LOGGER.info("Training HistGradientBoostingClassifier on %s rows", len(labeled))
    model, threshold = fit_hist_gbm(labeled, spec)
    model_path = settings.paths.model_dir / "hist_gbm_extended_5m_start.pkl"
    bundle = save_model_bundle(model, threshold, model_path)
    store.save_model_registry(
        model_name=bundle.model_name,
        created_at=bundle.created_at,
        threshold=bundle.threshold,
        artifact_path=bundle.artifact_path,
        metadata={"feature_count": len(bundle.feature_columns), "training_rows": len(labeled)},
    )

    latest_date = _latest_session_date(daily_features)
    shortlist = _build_shortlist(daily_features, latest_date, settings.runtime.shortlist_count)
    store.save_shortlist(latest_date, shortlist)
    shortlist.to_parquet(settings.paths.watchlist_path, index=False)

    report_path = settings.paths.reports_dir / "nightly_pipeline_report.json"
    report = {
        "session_date": str(latest_date.date()),
        "universe_count": int(len(universe)),
        "daily_rows": int(len(daily_bars)),
        "intraday_rows": int(len(intraday_bars)),
        "training_rows": int(len(labeled)),
        "shortlist_count": int(len(shortlist)),
        "threshold": threshold,
        "model_path": str(model_path),
        "watchlist_path": str(settings.paths.watchlist_path),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    LOGGER.info("Nightly pipeline complete for %s", latest_date.date())
    return {
        "report_path": report_path,
        "model_path": model_path,
        "watchlist_path": settings.paths.watchlist_path,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    outputs = run_nightly_pipeline()
    print(f"Nightly pipeline complete: {outputs['report_path']}")


if __name__ == "__main__":
    main()
