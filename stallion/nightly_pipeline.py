from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .config import Settings, load_settings
from .features import build_daily_feature_history
from .fmp import FMPClient, download_yfinance_bars
from .modeling import fit_hist_gbm, save_model_bundle
from .storage import SQLiteParquetStore
from .strategy import StandardSystemSpec
from .watchlist_model import (
    build_next_session_watchlist,
    build_stage2_intraday_panel,
    build_watchlist_training_panel,
    evaluate_watchlist_model_cv,
    make_watchlist_labels,
    make_watchlist_model_spec,
    save_watchlist_model,
    score_watchlist_universe,
    train_watchlist_model,
    write_watchlist_reports,
)


LOGGER = logging.getLogger(__name__)


def _latest_session_date(feature_frame: pd.DataFrame) -> pd.Timestamp:
    if feature_frame.empty:
        raise ValueError("No daily feature rows available.")
    return pd.to_datetime(feature_frame["session_date"]).max().normalize()


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

    intraday_period_days = min(60, max(settings.runtime.intraday_history_sessions, settings.runtime.training_sessions))
    LOGGER.info("Downloading 5m bars via yfinance for %s symbols", len(symbols))
    intraday_bars = download_yfinance_bars(symbols, period=f"{intraday_period_days}d", interval="5m")
    intraday_bars["cumulative_volume"] = intraday_bars["volume"]
    store.save_bars(intraday_bars, timeframe="5m")

    LOGGER.info("Building daily feature history")
    daily_features = build_daily_feature_history(daily_bars, universe, spy_symbol="SPY")
    store.save_daily_features(daily_features)

    LOGGER.info("Building stage-2 intraday training panel")
    labeled = build_stage2_intraday_panel(intraday_bars, daily_features, settings)

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

    LOGGER.info("Building watchlist training dataset")
    watchlist_spec = make_watchlist_model_spec(settings)
    watchlist_labels = make_watchlist_labels(daily_features, daily_bars, labeled)
    watchlist_training_panel = build_watchlist_training_panel(daily_features, watchlist_labels)
    store.write_parquet_snapshot(watchlist_training_panel, "artifacts/watchlist_training_panel/latest.parquet")
    if watchlist_training_panel.empty:
        raise RuntimeError("Watchlist training panel is empty; cannot build nightly shortlist model.")

    latest_date = _latest_session_date(daily_features)
    watchlist_model_path = settings.paths.model_dir / "watchlist_logreg_top400.pkl"
    LOGGER.info("Running watchlist OOS comparison")
    watchlist_outputs = evaluate_watchlist_model_cv(
        watchlist_training_panel,
        daily_features,
        labeled,
        settings,
        watchlist_spec,
    )
    watchlist_report_paths: dict[str, Path] = write_watchlist_reports(settings.paths.reports_dir / "watchlist_model", watchlist_outputs, watchlist_spec)
    watchlist_model, watchlist_bundle = train_watchlist_model(watchlist_training_panel, watchlist_spec)
    watchlist_bundle = save_watchlist_model(watchlist_model, watchlist_bundle, watchlist_model_path)
    store.save_model_registry(
        model_name=watchlist_bundle.model_name,
        created_at=watchlist_bundle.created_at,
        threshold=0.0,
        artifact_path=watchlist_bundle.artifact_path,
        metadata={
            **watchlist_bundle.metadata,
            "feature_count": len(watchlist_bundle.feature_columns),
            "label_mode": watchlist_bundle.label_mode,
            "report_path": str(watchlist_report_paths.get("watchlist_model_report.md", "")),
        },
    )
    latest_watchlist_frame = (
        daily_features.loc[pd.to_datetime(daily_features["session_date"]).eq(latest_date), ["session_date", "symbol", *watchlist_spec.feature_columns]]
        .rename(columns={"session_date": "feature_date"})
        .copy()
    )
    latest_watchlist_frame["feature_date"] = pd.to_datetime(latest_watchlist_frame["feature_date"]).dt.normalize()
    scored_watchlist = score_watchlist_universe(watchlist_model, watchlist_bundle, latest_watchlist_frame)
    shortlist = build_next_session_watchlist(scored_watchlist, settings.runtime.shortlist_count)
    shortlist["session_date"] = latest_date
    shortlist["next_session_date"] = latest_date + pd.offsets.BDay(1)

    shortlist_session_key = latest_date
    if "next_session_date" in shortlist.columns and shortlist["next_session_date"].notna().any():
        shortlist_session_key = pd.to_datetime(shortlist["next_session_date"]).dropna().min().normalize()
    store.save_shortlist(shortlist_session_key, shortlist)
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
        "watchlist_model_path": str(watchlist_model_path),
        "watchlist_label_mode": watchlist_spec.label_mode,
        "watchlist_path": str(settings.paths.watchlist_path),
        "shortlist_session_key": str(pd.Timestamp(shortlist_session_key).date()),
        "watchlist_report_paths": {key: str(path) for key, path in watchlist_report_paths.items()},
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
