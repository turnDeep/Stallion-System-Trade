# Stallion-System-Trade

Stallion-System-Trade is now a **live-trading scaffold for the standard intraday Russell-3000 model**:

- night-before refresh of the **top 3000 U.S. stocks by market cap**
- daily context computed on **252+ trading days of split-adjusted daily bars**
- intraday monitoring on **5-minute bars**
- 15-minute context derived from the 5-minute stream
- model scoring with the **16-feature hist_gbm_extended 5m_start logic**
- **next-bar open entry**, **same-day close exit**, **max 4 positions per session**
- data persisted to **SQLite + Parquet**, not pickle

## Standard Logic

The production system is aligned to the standard logic used in the current research codebase.

### Trading window

- Signal window: **5 to 90 minutes after the U.S. market open**
- Earliest practical fill: **next 5-minute bar open**
- No overnight holding
- One trade per symbol per day
- Maximum concurrent positions: **4**

### Model

- Model: `HistGradientBoostingClassifier`
- Feature set: **16 features**
- Threshold per training run:

```text
threshold = max(0.55, 90th percentile of train_scores)
```

### Entry selection

The system does **not** pick the daily top-4 names by score in hindsight.

It works like this:

1. score all eligible candidates in real time
2. keep only names with `score >= threshold`
3. process candidates in **timestamp order**
4. if several names appear at the same timestamp, sort by **score descending**
5. fill until **4 positions** are used

### Exit

- operational assumption: **same-day close**
- backtest/live accounting uses:
  - commission: `0.2%` per side
  - slippage: `5 bps` per side
  - spread: `5 bps` round trip

## 16 Live Features

The deployed feature set is:

1. `daily_buy_pressure_prev`
2. `prev_day_adr_pct`
3. `industry_buy_pressure_prev`
4. `EMA_8_15`
5. `distance_to_prev_day_high`
6. `close_vs_vwap_15`
7. `sector_buy_pressure_prev`
8. `daily_rrs_prev`
9. `daily_rs_score_prev`
10. `distance_to_avwap_63_prev`
11. `volume_spike_5m`
12. `industry_rs_prev`
13. `same_slot_avg_vol_20d`
14. `rs_x_intraday_rvol`
15. `intraday_range_expansion_vs_atr`
16. `prev_day_close_vs_sma50`

## Data Requirements

### Night-before full universe data

- top 3000 symbols by market cap
- `symbol`, `exchange`, `sector`, `industry`, `market_cap`
- split-adjusted daily OHLCV
- `SPY` daily OHLCV

### Live intraday data

- current-session 5-minute OHLCV for the monitored shortlist
- 15-minute context derived from the 5-minute stream

### Historical local retention

- daily bars: **300 to 400 sessions**
- 5-minute bars: **20 to 40 sessions**
- same-slot volume history: **20 sessions**

## Storage Layout

The project now uses:

- **SQLite**
  - universe metadata
  - daily bars
  - 5-minute bars
  - daily feature rows
  - shortlist rows
  - model registry
  - live signals / fills
- **Parquet**
  - raw daily snapshots
  - raw intraday snapshots
  - daily feature snapshots
  - nightly shortlist snapshots

Important:

- pickle is no longer the primary production data format
- Parquet is used for reproducible snapshots and research reuse
- SQLite is used as the operational store

## Main Files

| File | Role |
|---|---|
| `ml_pipeline_60d.py` | Nightly pipeline entrypoint |
| `webull_live_trader.py` | Live execution entrypoint |
| `backtester.py` | Event-driven backtest entrypoint |
| `master_scheduler.py` | Daily scheduler |
| `stallion/config.py` | Runtime and path settings |
| `stallion/storage.py` | SQLite + Parquet persistence |
| `stallion/features.py` | Daily and intraday feature construction |
| `stallion/modeling.py` | HistGBM training / scoring / thresholding |
| `stallion/nightly_pipeline.py` | Universe refresh, feature build, model fit, shortlist build |
| `stallion/live_trader.py` | Real-time polling, scoring, selection, order routing |

## How It Runs

### Nightly pipeline

```bash
python ml_pipeline_60d.py
```

What it does:

1. refresh the top-3000 universe from FMP
2. update local daily and 5-minute history
3. compute daily feature history
4. build the intraday training panel
5. train the HistGBM model
6. save the model artifact and threshold
7. build the next-session shortlist

### Live trader

```bash
python webull_live_trader.py
```

What it does:

1. load the saved model and nightly shortlist
2. poll FMP batch quotes for the monitored symbols
3. aggregate snapshots into the operational 5-minute store
4. rebuild current intraday features
5. score candidates
6. select up to 4 names in real time
7. route orders to Webull

### Scheduler

```bash
python master_scheduler.py
```

Default schedule:

- `17:00 America/New_York`: nightly pipeline
- `09:25 America/New_York`: live trader bootstrap

## Environment Variables

Create `.env` from `.env.example`.

```env
FMP_API_KEY=
WEBULL_APP_KEY=
WEBULL_APP_SECRET=
WEBULL_ACCOUNT_ID=
```

## Install

```bash
git clone https://github.com/turnDeep/Stallion-System-Trade.git
cd Stallion-System-Trade
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Docker

```bash
docker compose up -d
```

The container still uses `master_scheduler.py` as the entrypoint.

## Notes

- This repository is now oriented around the **standard daytrade live architecture**, not the old ORB Top-10 system.
- The current live engine is intentionally simple and transparent.
- If you want stricter market-regime blocking, crash-day filters, or a websocket collector, those can be layered on top of this base cleanly.
