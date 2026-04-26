# Auto-Swing-Trade-Bot

English | [日本語](./README.ja.md)

Auto-Swing-Trade-Bot is a Qullamaggie-style breakout swing trading system for U.S. equities.

- Universe: top 3000 U.S. stocks by market cap from FMP
- Signal engine: daily setup + intraday 5-minute breakout confirmation
- Portfolio style: swing positions with overnight holds
- Storage: SQLite + Parquet snapshots plus daily archive files
- Runtime: Webull-backed live mode or automatic demo mode fallback

## Core Flow

1. Run the nightly breakout pipeline to refresh universe, daily bars, 5-minute bars, signal report, and next-session shortlist.
2. Run the live trader during market hours to monitor shortlist symbols, detect breakouts, place entries, and manage open swing positions.
3. Let the scheduler orchestrate nightly refresh and market-hours execution.

## Main Commands

Nightly breakout pipeline:

```bash
python nightly_breakout_pipeline.py
```

Live trader:

```bash
python breakout_live_trader.py
```

Scheduler:

```bash
python master_scheduler.py
```

Backtester:

```bash
python backtester.py
```

Docker:

```bash
docker compose up -d --build
```

## Main Files

| File | Role |
|---|---|
| `nightly_breakout_pipeline.py` | top-level nightly pipeline entrypoint |
| `breakout_live_trader.py` | top-level live trader entrypoint |
| `master_scheduler.py` | scheduler and bootstrap loop |
| `breakout_signal_engine.py` | daily breakout scoring engine |
| `breakout_signal_report.py` | setup/breakout reporting and golden-rule filtering |
| `qullamaggie_breakout_backtest.py` | swing backtest and exit engine |
| `core/nightly_pipeline.py` | nightly refresh, repair fetch, signal report generation |
| `core/live_trader.py` | live polling, entries, hard stops, end-of-day exit checks |
| `core/breakout_bridge.py` | glue layer between signals, sizing, exits, and backtests |
| `core/storage.py` | SQLite + Parquet operational store |
| `core/fmp.py` | FMP universe download and yfinance bar retrieval |

## Notes

- The system no longer uses the old 60-day ML pipeline naming, threshold-based classifiers, or same-day flatten logic.
- 5-minute data from Yahoo remains source-limited, so the repository keeps appending and auditing local history over time.
- Demo mode is used automatically when the required live broker credentials are not present.
