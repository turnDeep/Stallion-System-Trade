import json
import os
import pickle

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


DATA_PKL = "russell3000_60d_5min.pkl"
LEGACY_DATA_PKL = "russell3000_5min.pkl"
MIN_ADR = 0.05
MIN_PRICE = 5.0
MIN_DOLLAR_ADV = 5_000_000.0
TOP_N = 10
YF_BATCH_SIZE = 100
MAX_DOWNLOAD_PERIOD = "60d"


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def normalize_intraday_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = df.copy()
    df.columns = [str(col).lower() for col in df.columns]
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return pd.DataFrame(columns=required)

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    else:
        df.index = pd.to_datetime(df.index)

    df = df[required].dropna(how="any")
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def merge_intraday_frames(existing: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return normalize_intraday_frame(new_data)
    if new_data is None or new_data.empty:
        return normalize_intraday_frame(existing)

    merged = pd.concat([normalize_intraday_frame(existing), normalize_intraday_frame(new_data)])
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


def load_pickle(path: str) -> dict[str, pd.DataFrame]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {symbol: normalize_intraday_frame(df) for symbol, df in data.items()}


def save_pickle(path: str, data: dict[str, pd.DataFrame]) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_or_bootstrap_intraday_store() -> dict[str, pd.DataFrame]:
    if os.path.exists(DATA_PKL):
        print(f"Loading existing {DATA_PKL}...")
        return load_pickle(DATA_PKL)

    if os.path.exists(LEGACY_DATA_PKL):
        print(f"Bootstrapping {DATA_PKL} from {LEGACY_DATA_PKL}...")
        data = load_pickle(LEGACY_DATA_PKL)
        save_pickle(DATA_PKL, data)
        return data

    raise FileNotFoundError(
        f"Neither {DATA_PKL} nor {LEGACY_DATA_PKL} was found. Please provide a universe data file first."
    )


def latest_trade_timestamp(data_60d: dict[str, pd.DataFrame]) -> pd.Timestamp | None:
    timestamps = [df.index.max() for df in data_60d.values() if not df.empty]
    if not timestamps:
        return None
    return pd.Timestamp(max(timestamps))


def choose_download_period(last_timestamp: pd.Timestamp | None) -> str:
    if last_timestamp is None:
        return MAX_DOWNLOAD_PERIOD

    now_ny = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    days_stale = (now_ny.normalize() - last_timestamp.normalize()).days
    if days_stale <= 7:
        return "5d"
    if days_stale <= 31:
        return "1mo"
    return MAX_DOWNLOAD_PERIOD


def fetch_yfinance_batch(symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
    if not symbols:
        return {}

    fetched: dict[str, pd.DataFrame] = {}
    batch_tickers = " ".join(symbols)
    df_batch = yf.download(
        batch_tickers,
        period=period,
        interval="5m",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
        prepost=False,
    )

    if df_batch.empty:
        return fetched

    if len(symbols) == 1 and not isinstance(df_batch.columns, pd.MultiIndex):
        fetched[symbols[0]] = normalize_intraday_frame(df_batch)
        return fetched

    if not isinstance(df_batch.columns, pd.MultiIndex):
        return fetched

    first_level = set(df_batch.columns.get_level_values(0))
    for sym in symbols:
        if sym not in first_level:
            continue
        df_sym = df_batch[sym].dropna(how="all")
        normalized = normalize_intraday_frame(df_sym)
        if not normalized.empty:
            fetched[sym] = normalized
    return fetched


def update_intraday_store(data_60d: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    symbols = sorted(data_60d.keys())
    if not symbols:
        return data_60d

    last_timestamp = latest_trade_timestamp(data_60d)
    period = choose_download_period(last_timestamp)
    if last_timestamp is not None:
        now_ny = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
        days_stale = (now_ny.normalize() - last_timestamp.normalize()).days
        if days_stale > 60:
            print(
                "Warning: the local store is more than 60 days stale. "
                "yfinance 5m data may not fully backfill the missing gap."
            )
    print(
        f"Updating {DATA_PKL} for {len(symbols)} symbols "
        f"(last timestamp: {last_timestamp}, yfinance period={period})..."
    )

    updated_symbols = 0
    for start in tqdm(range(0, len(symbols), YF_BATCH_SIZE), desc="Updating intraday store"):
        batch = symbols[start : start + YF_BATCH_SIZE]
        fetched = fetch_yfinance_batch(batch, period=period)
        for sym, df_new in fetched.items():
            merged = merge_intraday_frames(data_60d.get(sym), df_new)
            if len(merged) > len(data_60d.get(sym, pd.DataFrame())):
                updated_symbols += 1
            data_60d[sym] = merged

    save_pickle(DATA_PKL, data_60d)
    print(f"Intraday store update complete. Symbols extended: {updated_symbols}")
    return data_60d


def all_trade_dates(data_60d: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    for df in data_60d.values():
        dates.update(pd.Timestamp(x).normalize() for x in df.index)
    return sorted(dates)


def build_daily_feature_frame(df_train: pd.DataFrame) -> pd.DataFrame:
    if df_train.empty:
        return pd.DataFrame()

    df_train = df_train.sort_index().copy()
    df_train["session_date"] = pd.Index(df_train.index.date)

    daily = df_train.groupby("session_date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        opening_volume=("volume", "first"),
    )
    if daily.empty:
        return daily

    daily.index = pd.to_datetime(daily.index)
    daily.sort_index(inplace=True)

    daily["prev_close"] = daily["close"].shift(1)
    daily["range_pct"] = (daily["high"] - daily["low"]) / daily["low"].replace(0.0, np.nan)

    true_range = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - daily["prev_close"]).abs(),
            (daily["low"] - daily["prev_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    daily["atr_pct"] = true_range / daily["prev_close"].replace(0.0, np.nan)

    daily["dollar_volume"] = daily["close"] * daily["volume"]
    opening_volume_avg20 = daily["opening_volume"].rolling(20).mean().shift(1)
    daily["opening_rvol20"] = daily["opening_volume"] / opening_volume_avg20.replace(0.0, np.nan)
    daily["abs_gap_pct"] = ((daily["open"] - daily["prev_close"]) / daily["prev_close"]).abs()

    return daily.dropna(subset=["prev_close"])


def calculate_ex_ante_features(df_train: pd.DataFrame) -> dict[str, float] | None:
    daily = build_daily_feature_frame(df_train)
    if len(daily) < 20:
        return None

    opening_rvol = daily["opening_rvol20"].dropna()
    if opening_rvol.empty:
        return None

    return {
        "ADR": float(daily["range_pct"].mean()),
        "ATRPct": float(daily["atr_pct"].mean()),
        "DollarADV": float(daily["dollar_volume"].mean()),
        "OpeningRVOL": float(opening_rvol.mean()),
        "AbsGap": float(daily["abs_gap_pct"].dropna().mean()),
        "LastClose": float(daily["close"].iloc[-1]),
    }


def generate_watchlist(
    data_60d: dict[str, pd.DataFrame],
    train_dates: list[pd.Timestamp],
    top_n: int = TOP_N,
) -> pd.DataFrame:
    stats_list = []
    start = str(train_dates[0].date())
    end = str(train_dates[-1].date())

    for sym, df in data_60d.items():
        df_train = df.loc[start:end]
        features = calculate_ex_ante_features(df_train)
        if not features:
            continue
        if features["ADR"] < MIN_ADR:
            continue
        if features["LastClose"] < MIN_PRICE:
            continue
        if features["DollarADV"] < MIN_DOLLAR_ADV:
            continue

        features["Symbol"] = sym
        stats_list.append(features)

    if not stats_list:
        return pd.DataFrame()

    df_stats = pd.DataFrame(stats_list)
    df_stats["LogDollarADV"] = np.log1p(df_stats["DollarADV"])

    for col in ["ADR", "ATRPct", "LogDollarADV", "OpeningRVOL", "AbsGap"]:
        df_stats[f"{col}_Z"] = zscore(df_stats[col].astype(float))

    df_stats["Composite_Score"] = (
        (df_stats["ADR_Z"] * 0.30)
        + (df_stats["ATRPct_Z"] * 0.25)
        + (df_stats["LogDollarADV_Z"] * 0.20)
        + (df_stats["OpeningRVOL_Z"] * 0.15)
        + (df_stats["AbsGap_Z"] * 0.10)
    )
    df_stats.sort_values("Composite_Score", ascending=False, inplace=True)
    df_stats["Rank"] = np.arange(1, len(df_stats) + 1)

    return df_stats.head(top_n).reset_index(drop=True)


def main() -> None:
    print("Loading universe symbols...")
    data_60d = load_or_bootstrap_intraday_store()
    print(f"Loaded {len(data_60d)} symbols before update.")

    data_60d = update_intraday_store(data_60d)
    print(f"Loaded {len(data_60d)} symbols after update.")

    all_dates = all_trade_dates(data_60d)
    if len(all_dates) < 20:
        print("Not enough trading days for ex-ante watchlist generation.")
        return

    train_dates = all_dates[-min(40, len(all_dates)) :]

    print("\n--- EX-ANTE WATCHLIST GENERATION ---")
    print(f"Training on the most recent {len(train_dates)} days ({train_dates[0].date()} to {train_dates[-1].date()})")

    top_10_df = generate_watchlist(data_60d, train_dates, top_n=TOP_N)
    if top_10_df.empty:
        print("No stocks passed the filters.")
        return

    print("\n==================================================")
    print(f"STALLION TOP {len(top_10_df)} WATCHLIST FOR NEXT SESSION")
    print("==================================================")
    print(
        top_10_df[
            ["Symbol", "ADR", "ATRPct", "DollarADV", "OpeningRVOL", "AbsGap", "Composite_Score"]
        ].to_string(index=False)
    )
    print("==================================================")

    top_symbols = top_10_df["Symbol"].tolist()
    with open("top_10_watchlist.json", "w") as f:
        json.dump(top_symbols, f)

    print(f"\nSaved Top 10 watchlist for the next session: {top_symbols}")
    print("Ready for live trading execution.")


if __name__ == "__main__":
    main()
