from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import pandas as pd

from stallion.breakout_bridge import (
    BreakoutConfig,
    run_breakout_backtest_from_inputs,
    signals_from_report,
)
from stallion.config import load_settings
from stallion.storage import SQLiteParquetStore


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
SIGNAL_REPORT_PATH = REPORTS_DIR / "breakout_signal_report.parquet"
SUMMARY_PATH = REPORTS_DIR / "fresh_variant_summary.csv"
REASON_PATH = REPORTS_DIR / "fresh_variant_exit_reason_summary.csv"
MARKDOWN_PATH = REPORTS_DIR / "fresh_variant_report.md"


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, BreakoutConfig]:
    settings = load_settings(ROOT)
    store = SQLiteParquetStore(settings)
    daily = store.load_bars("1d")
    intraday = store.load_bars("5m")
    report = pd.read_parquet(SIGNAL_REPORT_PATH)
    if "symbol" in daily.columns:
        daily = daily.loc[daily["symbol"].ne("SPY")].copy()
    cfg = BreakoutConfig.from_settings(settings)
    return daily, intraday, report, cfg


def _time_cutoff_filter(signals: pd.DataFrame, cutoff_hhmm: str) -> pd.DataFrame:
    work = signals.copy()
    work["trigger_time"] = pd.to_datetime(work["trigger_time"], errors="coerce")
    cutoff = pd.Timestamp(f"2000-01-01 {cutoff_hhmm}").time()
    keep = work["trigger_time"].dt.time >= cutoff
    return work.loc[keep].copy()


def _trigger_score_filter(signals: pd.DataFrame, min_trigger_score: float) -> pd.DataFrame:
    work = signals.copy()
    work["trigger_score"] = pd.to_numeric(work["trigger_score"], errors="coerce")
    return work.loc[work["trigger_score"] >= min_trigger_score].copy()


def _open_trade_count(fills_df: pd.DataFrame) -> int:
    buy_counts = fills_df.loc[fills_df["side"].eq("buy"), "symbol"].value_counts()
    sell_counts = fills_df.loc[fills_df["side"].eq("sell"), "symbol"].value_counts()
    remaining = buy_counts.subtract(sell_counts, fill_value=0)
    return int((remaining > 0).sum())


def _reason_summary(name: str, fills_df: pd.DataFrame) -> pd.DataFrame:
    sells = fills_df.loc[fills_df["side"].eq("sell")].copy()
    if sells.empty:
        return pd.DataFrame(columns=["variant", "reason", "count"])
    summary = (
        sells.groupby("reason", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "reason"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    summary.insert(0, "variant", name)
    return summary


def _run_variant(
    name: str,
    *,
    base_signals: pd.DataFrame,
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    cfg: BreakoutConfig,
    signal_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    cfg_overrides: dict | None = None,
    notes: str = "",
) -> tuple[dict, pd.DataFrame]:
    signals = base_signals.copy()
    signals = signals.loc[signals["breakout_signal"].fillna(False)].copy()
    if signal_filter is not None:
        signals = signal_filter(signals)

    variant_cfg = replace(cfg, **(cfg_overrides or {}))
    equity_curve, fills_df, stats = run_breakout_backtest_from_inputs(
        daily_bars=daily,
        intraday_bars=intraday,
        signals=signals,
        cfg=variant_cfg,
    )

    row = {
        "variant": name,
        "notes": notes,
        "signal_count": int(len(signals)),
        "buy_count": int((fills_df["side"] == "buy").sum()),
        "sell_count": int((fills_df["side"] == "sell").sum()),
        "open_symbol_count": _open_trade_count(fills_df),
        "end_equity": float(stats.get("end_equity", float("nan"))),
        "total_return": float(stats.get("total_return", float("nan"))),
        "max_drawdown": float(stats.get("max_drawdown", float("nan"))),
        "sharpe": float(stats.get("sharpe", float("nan"))),
        "sortino": float(stats.get("sortino", float("nan"))),
        "num_round_trips": float(stats.get("num_round_trips", float("nan"))),
        "win_rate": float(stats.get("win_rate", float("nan"))),
        "avg_pnl": float(stats.get("avg_pnl", float("nan"))),
        "median_pnl": float(stats.get("median_pnl", float("nan"))),
        "profit_factor": float(stats.get("profit_factor", float("nan"))),
    }
    return row, _reason_summary(name, fills_df)


def main() -> None:
    daily, intraday, report, cfg = _load_inputs()
    base_signals = signals_from_report(report)

    variants: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame] | None, dict | None, str]] = [
        ("baseline", None, None, "Current fresh-period configuration"),
        (
            "time_0945_only",
            lambda s: _time_cutoff_filter(s, "09:45"),
            None,
            "Trigger after 09:45 NY only",
        ),
        (
            "fastfail2_only",
            None,
            {"fast_fail_days": 2},
            "Only extend pivot-fail exit window to 2 days",
        ),
        (
            "A_0945_fastfail2",
            lambda s: _time_cutoff_filter(s, "09:45"),
            {"fast_fail_days": 2},
            "Trigger after 09:45 NY only, plus fast_fail_days=2",
        ),
        (
            "epgap5_only",
            None,
            {"ep_gap_exclusion_pct": 0.05},
            "Tighten EP gap exclusion from 10% to 5%",
        ),
        (
            "B_stop25",
            None,
            {"stop_buffer_bps": 25.0},
            "Wider stop buffer only",
        ),
        (
            "AplusB_0945_fastfail2_stop25",
            lambda s: _time_cutoff_filter(s, "09:45"),
            {"fast_fail_days": 2, "stop_buffer_bps": 25.0},
            "A plus wider stop buffer",
        ),
        (
            "C_trigger78",
            lambda s: _trigger_score_filter(s, 78.0),
            None,
            "Require trigger_score >= 78",
        ),
        (
            "trigger78_epgap5",
            lambda s: _trigger_score_filter(s, 78.0),
            {"ep_gap_exclusion_pct": 0.05},
            "Require trigger_score >= 78 and EP gap <= 5%",
        ),
        (
            "trigger78_epgap5_stop25",
            lambda s: _trigger_score_filter(s, 78.0),
            {"ep_gap_exclusion_pct": 0.05, "stop_buffer_bps": 25.0},
            "trigger_score >= 78, EP gap <= 5%, stop buffer 25 bps",
        ),
        (
            "ABC_0945_fastfail2_stop25_trigger78",
            lambda s: _trigger_score_filter(_time_cutoff_filter(s, "09:45"), 78.0),
            {"fast_fail_days": 2, "stop_buffer_bps": 25.0},
            "A plus B plus trigger_score >= 78",
        ),
    ]

    summary_rows: list[dict] = []
    reason_frames: list[pd.DataFrame] = []
    for name, signal_filter, cfg_overrides, notes in variants:
        row, reason_df = _run_variant(
            name,
            base_signals=base_signals,
            daily=daily,
            intraday=intraday,
            cfg=cfg,
            signal_filter=signal_filter,
            cfg_overrides=cfg_overrides,
            notes=notes,
        )
        summary_rows.append(row)
        reason_frames.append(reason_df)

    summary = pd.DataFrame(summary_rows)
    summary["return_pct"] = summary["total_return"] * 100.0
    summary["max_drawdown_pct"] = summary["max_drawdown"] * 100.0
    summary = summary[
        [
            "variant",
            "notes",
            "signal_count",
            "buy_count",
            "sell_count",
            "open_symbol_count",
            "end_equity",
            "total_return",
            "return_pct",
            "max_drawdown",
            "max_drawdown_pct",
            "sharpe",
            "sortino",
            "win_rate",
            "profit_factor",
            "avg_pnl",
            "median_pnl",
            "num_round_trips",
        ]
    ]
    summary.to_csv(SUMMARY_PATH, index=False)

    reason_summary = pd.concat(reason_frames, ignore_index=True) if reason_frames else pd.DataFrame()
    reason_summary.to_csv(REASON_PATH, index=False)

    markdown_lines = [
        "# Fresh Variant Report",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Exit Reasons",
        "",
        reason_summary.to_markdown(index=False) if not reason_summary.empty else "No sell reasons.",
        "",
    ]
    MARKDOWN_PATH.write_text("\n".join(markdown_lines), encoding="utf-8")

    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {REASON_PATH}")
    print(f"Wrote {MARKDOWN_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
