from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from .breakout_bridge import (
    BreakoutConfig,
    normalize_daily_bars,
    normalize_intraday_bars,
    run_breakout_backtest,
    run_breakout_backtest_from_inputs,
)
from .config import load_settings
from .storage import SQLiteParquetStore


LOGGER = logging.getLogger(__name__)


def _flatten_daily_history(path: Path) -> pd.DataFrame:
    history = pd.read_pickle(path)
    if not isinstance(history, dict):
        return normalize_daily_bars(history)
    frames: list[pd.DataFrame] = []
    for symbol, frame in history.items():
        if frame is None or len(frame) == 0:
            continue
        work = frame.copy()
        work.index = pd.to_datetime(work.index).tz_localize(None).normalize()
        work.index.name = "date"
        work = work.reset_index()
        work.columns = [str(col).lower() for col in work.columns]
        work["symbol"] = symbol
        frames.append(work[["symbol", "date", "open", "high", "low", "close", "volume"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _flatten_intraday_history(path: Path) -> pd.DataFrame:
    history = pd.read_pickle(path)
    if not isinstance(history, dict):
        return normalize_intraday_bars(history)
    frames: list[pd.DataFrame] = []
    for symbol, frame in history.items():
        if frame is None or len(frame) == 0:
            continue
        work = frame.copy()
        work.index = pd.to_datetime(work.index)
        if getattr(work.index, "tz", None) is not None:
            work.index = work.index.tz_convert("America/New_York").tz_localize(None)
        work.index.name = "datetime"
        work = work.reset_index()
        work.columns = [str(col).lower() for col in work.columns]
        work["symbol"] = symbol
        frames.append(work[["symbol", "datetime", "open", "high", "low", "close", "volume"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_backtest_pickles() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    daily_pickle = os.getenv("CORE_BACKTEST_DAILY_PICKLE", "").strip()
    intraday_pickle = os.getenv("CORE_BACKTEST_INTRADAY_PICKLE", "").strip()
    if not daily_pickle or not intraday_pickle:
        return None

    daily_path = Path(daily_pickle)
    intraday_path = Path(intraday_pickle)
    if not daily_path.exists() or not intraday_path.exists():
        raise FileNotFoundError("CORE_BACKTEST_DAILY_PICKLE / CORE_BACKTEST_INTRADAY_PICKLE paths must exist")

    daily = _flatten_daily_history(daily_path)
    intraday = _flatten_intraday_history(intraday_path)
    return daily, intraday


def run_backtest_report() -> Path:
    settings = load_settings()
    store = SQLiteParquetStore(settings)
    signals_parquet = os.getenv("CORE_BACKTEST_SIGNALS_PARQUET", "").strip()
    daily_parquet = os.getenv("CORE_BACKTEST_DAILY_PARQUET", "").strip()
    intraday_parquet = os.getenv("CORE_BACKTEST_INTRADAY_PARQUET", "").strip()
    direct_inputs = _load_backtest_pickles()
    if signals_parquet:
        if not daily_parquet or not intraday_parquet:
            raise ValueError("CORE_BACKTEST_DAILY_PARQUET and CORE_BACKTEST_INTRADAY_PARQUET are required with CORE_BACKTEST_SIGNALS_PARQUET.")
        LOGGER.info("Running breakout backtest from prepared signal inputs")
        daily_bars = pd.read_parquet(daily_parquet)
        intraday_bars = pd.read_parquet(intraday_parquet)
        signals = pd.read_parquet(signals_parquet) if signals_parquet.endswith(".parquet") else pd.read_csv(signals_parquet)
    else:
        if direct_inputs is not None:
            LOGGER.info("Running breakout backtest directly from frozen pickles")
            daily_bars, intraday_bars = direct_inputs
        else:
            daily_bars = store.load_bars("1d")
            intraday_bars = store.load_bars("5m")
        signals = None
    if daily_bars.empty or intraday_bars.empty:
        raise RuntimeError("No local 1d/5m history available. Run the nightly pipeline first or set CORE_BACKTEST_* environment variables.")

    cfg = BreakoutConfig.from_settings(settings)
    daily_bars = daily_bars.loc[daily_bars["symbol"].ne("SPY")].copy() if "symbol" in daily_bars.columns else daily_bars
    if signals is not None:
        equity_curve, fills_df, stats = run_breakout_backtest_from_inputs(daily_bars, intraday_bars, signals, cfg=cfg)
        report = pd.DataFrame()
    else:
        equity_curve, fills_df, stats, report = run_breakout_backtest(daily_bars, intraday_bars, cfg=cfg)

    reports_dir = settings.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    equity_path = reports_dir / "breakout_equity_curve.csv"
    fills_path = reports_dir / "breakout_fills.csv"
    summary_path = reports_dir / "breakout_summary.csv"
    report_path = reports_dir / "breakout_signal_report.parquet"
    markdown_path = reports_dir / "backtest_report.md"

    equity_curve.to_csv(equity_path, index=False)
    fills_df.to_csv(fills_path, index=False)
    pd.DataFrame([stats]).to_csv(summary_path, index=False)
    if not report.empty:
        report.to_parquet(report_path, index=False)

    lines = [
        "# Backtest Report",
        "",
        "## Summary",
        "",
        f"- end_equity: {stats.get('end_equity', float('nan')):,.2f}",
        f"- total_return: {stats.get('total_return', float('nan')):.4%}",
        f"- max_drawdown: {stats.get('max_drawdown', float('nan')):.4%}",
        f"- sharpe: {stats.get('sharpe', float('nan')):.4f}",
        f"- sortino: {stats.get('sortino', float('nan')):.4f}",
        f"- win_rate: {stats.get('win_rate', float('nan')):.2%}",
        f"- profit_factor: {stats.get('profit_factor', float('nan')):.4f}",
        f"- signal_count: {int(report['breakout_signal'].fillna(False).sum()) if not report.empty and 'breakout_signal' in report.columns else len(signals) if signals is not None else 0}",
        "",
        "## Outputs",
        "",
        f"- equity_curve: {equity_path}",
        f"- fills: {fills_path}",
        f"- summary: {summary_path}",
        f"- signal_report: {report_path}",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report_path = run_backtest_report()
    print(f"Backtest report ready: {report_path}")


if __name__ == "__main__":
    main()
