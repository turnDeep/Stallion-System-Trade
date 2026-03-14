from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import load_settings
from .features import build_intraday_feature_panel, build_training_labels
from .modeling import fit_hist_gbm, score_candidates
from .storage import SQLiteParquetStore
from .strategy import StandardSystemSpec, select_candidates_for_session


LOGGER = logging.getLogger(__name__)


def run_backtest_report() -> Path:
    settings = load_settings()
    store = SQLiteParquetStore(settings)
    spec = StandardSystemSpec(
        min_minutes_from_open=settings.runtime.min_minutes_from_open,
        max_minutes_from_open=settings.runtime.max_minutes_from_open,
        max_positions=settings.runtime.max_positions,
        threshold_floor=settings.runtime.threshold_floor,
        threshold_quantile=settings.runtime.threshold_quantile,
    )

    intraday = store.load_bars("5m")
    daily_features = store.load_daily_features()
    if intraday.empty or daily_features.empty:
        raise RuntimeError("No local SQLite history available. Run the nightly pipeline first.")

    panel = build_intraday_feature_panel(intraday, daily_features, same_slot_lookback_sessions=settings.runtime.same_slot_lookback_sessions)
    panel = panel[panel["session_bucket"].eq("open_drive")].copy()
    panel = panel[panel["minutes_from_open"].between(spec.min_minutes_from_open, spec.max_minutes_from_open, inclusive="both")].copy()
    labeled = build_training_labels(
        panel,
        commission_rate_one_way=settings.costs.commission_rate_one_way,
        slippage_bps_per_side=settings.costs.slippage_bps_per_side,
        spread_bps_round_trip=settings.costs.spread_bps_round_trip,
        adverse_fill_floor=settings.costs.extra_adverse_fill_floor,
        adverse_fill_cap=settings.costs.extra_adverse_fill_cap,
    ).dropna(subset=["next_open", "session_close"])
    if labeled.empty:
        raise RuntimeError("No labeled panel rows available for backtest.")

    split_idx = int(len(labeled) * 0.7)
    train = labeled.iloc[:split_idx].copy()
    test = labeled.iloc[split_idx:].copy()
    model, threshold = fit_hist_gbm(train, spec)
    scored = score_candidates(model, test)
    scored["threshold"] = threshold

    trade_rows = []
    for session_date, frame in scored.groupby("session_date", sort=True):
        chosen = select_candidates_for_session(frame, threshold=threshold, max_positions=spec.max_positions)
        if chosen.empty:
            continue
        chosen["trade_return"] = chosen["net_return_stress_exec"]
        chosen["session_date"] = pd.to_datetime(chosen["session_date"])
        trade_rows.append(chosen)

    trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    report_path = settings.paths.reports_dir / "backtest_report.md"
    if trades.empty:
        report_path.write_text("# Backtest Report\n\nNo trades executed.\n", encoding="utf-8")
        return report_path

    summary = {
        "trade_count": int(len(trades)),
        "win_rate": float((trades["trade_return"] > 0).mean()),
        "avg_trade_return": float(trades["trade_return"].mean()),
        "total_return": float(trades["trade_return"].sum()),
    }
    lines = [
        "# Backtest Report",
        "",
        "## Summary",
        "",
        f"- trade_count: {summary['trade_count']}",
        f"- win_rate: {summary['win_rate']:.2%}",
        f"- avg_trade_return: {summary['avg_trade_return']:.4%}",
        f"- total_return: {summary['total_return']:.4%}",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    trades.to_parquet(settings.paths.reports_dir / "backtest_trades.parquet", index=False)
    return report_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report_path = run_backtest_report()
    print(f"Backtest report ready: {report_path}")


if __name__ == "__main__":
    main()
