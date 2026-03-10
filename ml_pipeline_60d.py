import json
import os
import pickle

import numpy as np
import pandas as pd


MIN_ADR = 0.05
MIN_PRICE = 5.0
MIN_DOLLAR_ADV = 5_000_000.0
TOP_N = 10


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


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
    """
    Calculates ex-ante features over the training window.

    Features:
    - ADR
    - ATRPct
    - DollarADV
    - OpeningRVOL
    - AbsGap
    """
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
    """
    Generates a scored watchlist based on Z-scores of ex-ante features.
    """
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
    data_pkl = "russell3000_60d_5min.pkl"

    if not os.path.exists(data_pkl):
        print(f"Error: {data_pkl} not found. Please run the data collector.")
        return

    with open(data_pkl, "rb") as f:
        data_60d = pickle.load(f)

    print(f"Loaded {len(data_60d)} symbols with data.")

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
