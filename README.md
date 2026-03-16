# Stallion-System-Trade

Language / 言語: **English** | [日本語](./README.ja.md)

Stallion-System-Trade is now a **two-stage live-trading scaffold for the standard intraday Russell-3000 model**:

- night-before refresh of the **top 3000 U.S. stocks by market cap**
- stage-1 nightly **watchlist model** on 7 daily features to select the next-session top 400 shortlist
- daily context computed on **252+ trading days of split-adjusted daily bars**
- intraday monitoring on **5-minute bars**
- 15-minute context derived from the 5-minute stream
- model scoring with the **16-feature hist_gbm_extended 5m_start logic**
- **next-bar open entry**, **same-day close exit**, **max 4 positions per session**
- data persisted to **SQLite + Parquet**, not pickle
- startup / recovery state persisted for **quotes, bars, orders, positions, heartbeats, and alerts**

## Standard Logic

The production system is aligned to the standard logic used in the current research codebase.

### Trading window

- Signal window: **5 to 90 minutes after the U.S. market open**
- Earliest practical fill: **next 5-minute bar open**
- No overnight holding
- One trade per symbol per day
- Maximum concurrent positions: **4**

### Model

- Stage 1 shortlist model: `LogisticRegression` on **7 daily features**
- Stage 2 execution model: `HistGradientBoostingClassifier` on **16 intraday features**
- Stage 2 threshold per training run:

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

### Shortlist construction

- runtime shortlist generation is now **always** the learned stage-1 watchlist model
- the old hand-weighted shortlist is retained only inside the nightly OOS comparison report
- the live trader and backtester consume the learned top-400 shortlist artifact

### Exit

- operational assumption: **same-day close**
- live trader now stops taking new entries after **15:55 America/New_York**
- live trader now sends **automatic SELL market orders** to flatten open positions starting at **15:58 America/New_York**
- backtest/live accounting uses:
  - commission: `0.2%` per side
  - slippage: `5 bps` per side
  - spread: `5 bps` round trip

### Position sizing

- the system reads the **opening account equity** from Webull
- the day budget is split into **4 equal slots**
- each order size is:

```text
slot_budget = opening_equity / 4
quantity = floor(slot_budget / expected_fill_price)
```

- orders are sent as **integer-share market orders**

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

## 7 Watchlist Features

1. `daily_buy_pressure_prev`
2. `daily_rs_score_prev`
3. `daily_rrs_prev`
4. `prev_day_adr_pct`
5. `industry_buy_pressure_prev`
6. `sector_buy_pressure_prev`
7. `industry_rs_prev`

## Data Requirements

### Night-before full universe data

- top 3000 symbols by market cap
- `symbol`, `exchange`, `sector`, `industry`, `market_cap`
- split-adjusted daily OHLCV
- `SPY` daily OHLCV

### Live intraday data

- current-session quote snapshots for the monitored shortlist
- production 5-minute OHLCV is **aggregated from the quote stream**
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
  - quote snapshots
  - daily feature rows
  - shortlist rows
  - model registry
  - live signals / fills
  - live orders
  - open positions
  - heartbeats
  - alerts
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
6. train the nightly watchlist model and write OOS comparison reports against the old hand-crafted shortlist
7. save the watchlist artifact plus the stage-2 HistGBM artifact
8. build the next-session shortlist

### Live trader

```bash
python webull_live_trader.py
```

What it does:

1. load the saved model and nightly shortlist
2. poll FMP batch quotes for the monitored symbols
3. persist raw quote snapshots to SQLite
4. aggregate snapshots into production 5-minute OHLCV bars
5. rebuild current intraday features
6. score candidates
7. select up to 4 names in real time
8. size orders from opening account equity
9. route orders to Webull
10. reconcile order history / positions and auto-flatten before the close

### Scheduler

```bash
python master_scheduler.py
```

Default schedule:

- startup bootstrap: runs the nightly pipeline once only if the SQLite + Parquet artifacts are missing or incomplete
- `17:00 America/New_York`: nightly pipeline
- `09:25 America/New_York`: live trader bootstrap

## Broker Compatibility

- the current implementation is wired for **Webull Japan** (`Region.JP`)
- the repository assumes the provided account supports:
  - account list
  - account balance
  - account positions
  - order placement / cancellation
- before using live trading, verify the APIs succeed with your account and credentials

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

The container uses `master_scheduler.py` as the entrypoint and now includes a Docker healthcheck via `python -m stallion.watchdog`.

## Production Notes

- this repository now includes:
  - startup bootstrap checks
  - quote snapshot persistence
  - production 5-minute bar aggregation
  - opening-equity slot sizing
  - live order / position state tables
  - restart reconciliation against broker state
  - stale-order cancellation
  - automatic pre-close flattening
  - heartbeat / alert storage
- it still assumes your host stays online; WSL2 + Docker Desktop on a sleeping Windows machine is **not** true 24/365 infrastructure
- for real unattended operation, use an always-on host or VPS

## Notes

- The current live engine is intentionally simple and transparent.
- If you want stricter market-regime blocking, crash-day filters, or a websocket collector, those can be layered on top of this base cleanly.
