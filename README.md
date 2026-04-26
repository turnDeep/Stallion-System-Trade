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
python scripts/nightly_pipeline.py
```

Live trader:

```bash
python scripts/live_trader.py
```

Scheduler:

```bash
python scripts/scheduler.py
```

Backtester:

```bash
python scripts/backtest.py
```

Tax reserve manager:

```bash
python scripts/manage_tax_reserve.py show
```

Docker:

```bash
docker compose up -d --build
```

## Directory Layout

| Directory | Role |
|---|---|
| `core/` | production runtime: storage, broker, scheduler-facing pipelines, live trader, watchdog |
| `signals/` | standard breakout, zigzag breakout, entry lane, and industry-priority signal engines |
| `backtesting/` | reusable swing backtest and portfolio validation code |
| `research/` | calibration and exploratory analysis scripts |
| `scripts/` | CLI entrypoints used by operators, Docker, and the scheduler |
| `configs/` | calibrated strategy parameters |

## Notes

- Root-level thin wrappers were removed; use `scripts/` commands instead.
- The old duplicated `backtester.py` entrypoint is now `scripts/backtest.py`; the implementation remains in `core/backtester.py`.
- A stale legacy `optimizer.py` was removed because it referenced a non-existent `run_backtest` export and was not used by the current swing breakout system.
- The system no longer uses the old 60-day ML pipeline naming, threshold-based classifiers, or same-day flatten logic.
- 5-minute data from Yahoo remains source-limited, so the repository keeps appending and auditing local history over time.
- Demo mode is used automatically when the required live broker credentials are not present.