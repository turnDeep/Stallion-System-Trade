# Auto-Swing-Trade-Bot

English | [日本語](./README.ja.md)

Auto-Swing-Trade-Bot is a fully automated, **Qullamaggie-style breakout swing trading system** for U.S. equities. It handles everything from nightly universe screening to live intraday execution, allowing for hands-off swing trading.

## 🌟 System Overview

This system is designed to identify and trade explosive multi-day or multi-week moves (swing trades) by entering precisely when a stock breaks out of a daily consolidation pattern.

- **Universe:** Top 3000 U.S. stocks by market cap, fetched via Financial Modeling Prep (FMP).
- **Trading Strategy (Base Compact):** Focuses on stocks with strong momentum and structural breakouts. Entries are triggered based on high volume impact (`cum_vol_ratio >= 1.5`) and a clear intraday confirmation (`trigger_close >= pivot * 1.01`). Extensive backtesting has shown that true high-conviction breakouts organically achieve high relative strength; thus, `leader_score >= 60` is utilized strictly as a fail-safe / safety net rather than a primary selection metric.
- **Portfolio Management:** A highly concentrated 3-slot architecture (33.3% allocation each) designed to aggressively capture multi-bagger runners. It includes a tax reserve manager to safeguard capital.
- **Dynamic Exit Logic:** A multi-tiered runner management system tailored for the leader lifecycle:
  - Takes +20% partial profit to lock in initial gains.
  - Trails the remaining core position using progressive moving averages (10, 21, 50 DMAs).
  - Promotes positions that achieve a +200% peak gain to a "super winner prior-day-low stop", riding massive trends until structural breakdown.
- **Runtime Environment:** Runs via Docker. Connects to Webull for live trading, with an automatic fallback to "Demo Mode" if API credentials are not provided.
- **Data Storage:** Uses a robust mix of SQLite, Parquet snapshots, and daily archive files for persistent historical data, circumventing API rate limits.

## ⚙️ Core Workflow

1. **Nightly Pipeline:** Runs after market close. It updates the stock universe, fetches daily and 5-minute bars, calculates signal scores, and generates a "shortlist" of top candidates for the next session.
2. **Live Trader:** Runs during market hours. It actively monitors the shortlisted symbols, detects real-time breakouts, places orders, and manages stop-losses for open swing positions.
3. **Scheduler:** The master process that orchestrates both the nightly refresh and the live trading seamlessly.

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
python scripts/manage_tax_reserve.py archive --year 2024
python scripts/manage_tax_reserve.py history
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

- The system now operates strictly on the "Base Compact" strategy configuration. Over-optimized features like complex industry-priority caps or A+ replacement logic have been disabled in favor of raw volume impact ranking (`cum_vol_ratio_at_trigger`) to ensure robustness.
- Root-level thin wrappers were removed; use `scripts/` commands instead.
- The old duplicated `backtester.py` entrypoint is now `scripts/backtest.py`; the implementation remains in `core/backtester.py`.
- A stale legacy `optimizer.py` was removed because it referenced a non-existent `run_backtest` export and was not used by the current swing breakout system.
- The system no longer uses the old 60-day ML pipeline naming, threshold-based classifiers, or same-day flatten logic.
- 5-minute data from Yahoo remains source-limited, so the repository keeps appending and auditing local history over time.
- Demo mode is used automatically when the required live broker credentials are not present.