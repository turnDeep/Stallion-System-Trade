from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy import BreakoutStrategy


COMMISSION_RATE = 0.00132


@dataclass
class StrategyConfig:
    initial_equity: float = 100000.0
    baseline_train_days: int = 40
    improved_feature_lookback: int = 20
    embargo_days: int = 1
    baseline_top_n: int = 10
    baseline_allocation: float = 0.90
    improved_candidate_pool: int = 8
    improved_max_positions: int = 3
    improved_risk_budget: float = 0.009
    improved_max_weight: float = 0.30
    improved_total_weight_cap: float = 0.90
    improved_fixed_stop_pct: float = 0.03
    improved_entry_buffer_atr_frac: float = 0.03
    improved_stop_atr_frac: float = 0.12
    improved_min_price: float = 5.0
    improved_min_adv_usd: float = 5_000_000.0
    improved_min_atr_pct: float = 0.02
    improved_min_opening_rvol: float = 1.5
    improved_min_gap_pct: float = 0.0
    slippage_bps_each_side: float = 5.0
    dynamic_break_even_r: float = 0.75
    dynamic_trail_start_r: float = 1.50
    dynamic_trail_offset_r: float = 0.75

    def get_hash(self) -> str:
        s = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


@dataclass
class TradeCandidate:
    symbol: str
    date: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    gross_return_pct: float
    stop_price: float | None = None
    exit_reason: str | None = None
    score: float | None = None
    weight: float | None = None
    strategy: str | None = None


@dataclass(frozen=True)
class RuleWatchlistTemplate:
    name: str
    min_adv_usd: float
    min_atr_pct: float
    min_opening_rvol: float
    gap_min: float
    gap_max: float
    watchlist_size: int
    max_positions: int
    weight: float
    w_rvol: float = 0.50
    w_gap: float = -0.30
    w_atr: float = 0.20
    w_adv: float = 0.10


def zscore(values: pd.Series) -> np.ndarray:
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return np.zeros(len(values), dtype=float)
    return ((values - values.mean()) / std).to_numpy(dtype=float)


def objective_score(metrics: dict[str, Any]) -> float:
    return metrics["sharpe"] + (metrics["total_return_pct"] / 10.0) - (abs(metrics["max_drawdown_pct"]) / 20.0)


def pca_first_component_scores(df_stats: pd.DataFrame) -> np.ndarray:
    if len(df_stats) < 2:
        return np.zeros(len(df_stats))
    features = df_stats[["Trades", "WinRate", "AvgWin", "TotalPnL", "ADR"]].fillna(0.0)
    means = features.mean(axis=0)
    stds = features.std(axis=0).replace(0.0, 1.0)
    scaled = (features - means) / stds
    matrix = scaled.to_numpy(dtype=float)
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    scores = matrix @ vt[0]
    corrs = []
    for column in ("ADR", "WinRate"):
        corr = np.corrcoef(df_stats[column].to_numpy(dtype=float), scores)[0, 1]
        if not np.isnan(corr):
            corrs.append(corr)
    if corrs and sum(corrs) < 0:
        scores = -scores
    score_mean = scores.mean()
    score_std = scores.std()
    if score_std == 0:
        return np.zeros(len(scores))
    return (scores - score_mean) / score_std


def apply_trading_costs(
    entry_price: float,
    exit_price: float,
    commission_rate: float,
    slippage_bps_each_side: float,
) -> float:
    slippage_multiplier = slippage_bps_each_side / 10_000.0
    entry_fill = entry_price * (1.0 + slippage_multiplier)
    exit_fill = exit_price * (1.0 - slippage_multiplier)
    gross = (exit_fill - entry_fill) / entry_fill
    return gross - (commission_rate * 2.0)


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def compute_metrics(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    trade_log: pd.DataFrame,
    initial_equity: float,
) -> dict[str, Any]:
    if equity_curve.empty:
        raise ValueError("Equity curve is empty.")
    total_return = equity_curve.iloc[-1] / initial_equity - 1.0
    periods = len(daily_returns)
    annualized_return = (1.0 + total_return) ** (252 / periods) - 1.0 if periods > 0 else 0.0
    return_std = daily_returns.std(ddof=0)
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std(ddof=0) if not downside.empty else 0.0
    sharpe = float(math.sqrt(252) * daily_returns.mean() / return_std) if return_std > 0 else 0.0
    sortino = (
        float(math.sqrt(252) * daily_returns.mean() / downside_std) if downside_std and downside_std > 0 else 0.0
    )
    mdd = max_drawdown(equity_curve)
    calmar = float(annualized_return / abs(mdd)) if mdd < 0 else 0.0

    avg_gross_exposure = 0.0
    if not trade_log.empty:
        avg_gross_exposure = float(trade_log.groupby("date")["weight"].sum().mean() * 100.0)

    metrics = {
        "days": int(periods),
        "trade_count": int(len(trade_log)),
        "final_equity": float(equity_curve.iloc[-1]),
        "total_return_pct": float(total_return * 100.0),
        "annualized_return_pct": float(annualized_return * 100.0),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown_pct": float(mdd * 100.0),
        "avg_daily_return_pct": float(daily_returns.mean() * 100.0),
        "avg_daily_capital_deployed_pct": avg_gross_exposure,
    }
    if not trade_log.empty:
        metrics.update(
            {
                "win_rate_pct": float((trade_log["portfolio_return"] > 0).mean() * 100.0),
                "avg_trade_return_pct": float(trade_log["portfolio_return"].mean() * 100.0),
                "avg_weight_pct": float(trade_log["weight"].mean() * 100.0),
            }
        )
    else:
        metrics.update(
            {
                "win_rate_pct": 0.0,
                "avg_trade_return_pct": 0.0,
                "avg_weight_pct": 0.0,
            }
        )
    return metrics


def daily_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for trade_date, df_day in df.groupby(df.index.date):
        df_day = df_day.sort_index()
        first_bar = df_day.iloc[0]
        grouped.append(
            {
                "date": pd.Timestamp(trade_date),
                "daily_open": df_day["open"].iloc[0],
                "daily_high": df_day["high"].max(),
                "daily_low": df_day["low"].min(),
                "daily_close": df_day["close"].iloc[-1],
                "daily_volume": df_day["volume"].sum(),
                "first_open": first_bar["open"],
                "first_high": first_bar["high"],
                "first_low": first_bar["low"],
                "first_close": first_bar["close"],
                "first_volume": first_bar["volume"],
            }
        )
    daily = pd.DataFrame(grouped).set_index("date").sort_index()
    daily["dollar_volume"] = daily["daily_close"] * daily["daily_volume"]
    daily["daily_adr_pct"] = (daily["daily_high"] - daily["daily_low"]) / daily["daily_low"].replace(0.0, np.nan)
    daily["prev_close"] = daily["daily_close"].shift(1)
    true_range = pd.concat(
        [
            daily["daily_high"] - daily["daily_low"],
            (daily["daily_high"] - daily["prev_close"]).abs(),
            (daily["daily_low"] - daily["prev_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    daily["atr20_abs"] = true_range.rolling(20).mean().shift(1)
    daily["adv20_dollar"] = daily["dollar_volume"].rolling(20).mean().shift(1)
    daily["atr20_pct"] = (daily["atr20_abs"] / daily["prev_close"]).replace([np.inf, -np.inf], np.nan)
    daily["opening_rvol20"] = daily["first_volume"] / daily["first_volume"].rolling(20).mean().shift(1)
    daily["gap_pct"] = (daily["first_open"] - daily["prev_close"]) / daily["prev_close"]
    daily["opening_drive_pct"] = (daily["first_close"] - daily["first_open"]) / daily["first_open"]
    daily["opening_range_abs"] = daily["first_high"] - daily["first_low"]
    daily["opening_range_pct"] = daily["opening_range_abs"] / daily["first_open"]
    return daily


def precompute_baseline_daily_summary(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    baseline_strategy = BreakoutStrategy(
        entry_start_time="09:35:00",
        entry_end_time="10:30:00",
        take_profit_pct=0.10,
        stop_loss_pct=0.03,
        min_volume_ratio=1.0,
        use_historical_rvol=False,
    )
    summary: dict[str, pd.DataFrame] = {}
    for symbol, df in data.items():
        daily = daily_aggregate(df)
        candidate_rows = []
        for trade_date, df_day in df.groupby(df.index.date):
            df_day = df_day.sort_index()
            trade = baseline_strategy.generate_daily_signals_vectorized(df_day.copy())
            row = {
                "date": pd.Timestamp(trade_date),
                "baseline_has_trade": 0,
                "baseline_pnl_pct": 0.0,
                "baseline_pnl_net_pct": 0.0,
                "baseline_win_net": 0.0,
                "baseline_entry_time": pd.NaT,
                "baseline_exit_time": pd.NaT,
                "baseline_entry_price": np.nan,
                "baseline_exit_price": np.nan,
            }
            if trade:
                net_pnl = trade["pnl_pct"] - (COMMISSION_RATE * 2.0)
                row.update(
                    {
                        "baseline_has_trade": 1,
                        "baseline_pnl_pct": float(trade["pnl_pct"]),
                        "baseline_pnl_net_pct": float(net_pnl),
                        "baseline_win_net": float(net_pnl > 0),
                        "baseline_entry_time": trade["entry_time"],
                        "baseline_exit_time": trade["exit_time"],
                        "baseline_entry_price": float(trade["entry_price"]),
                        "baseline_exit_price": float(trade["exit_price"]),
                    }
                )
            candidate_rows.append(row)
        baseline_daily = pd.DataFrame(candidate_rows).set_index("date").sort_index()
        summary[symbol] = daily.join(baseline_daily, how="left")
    return summary


def rolling_window(df: pd.DataFrame, end_idx: int, window: int) -> pd.DataFrame:
    start_idx = max(0, end_idx - window)
    return df.iloc[start_idx:end_idx]


def baseline_rank_for_day(
    symbol_daily: dict[str, pd.DataFrame],
    trade_date: pd.Timestamp,
    config: StrategyConfig,
) -> list[tuple[str, float]]:
    base_stats = []
    daily_scores = {}
    for symbol, daily in symbol_daily.items():
        if trade_date not in daily.index:
            continue
        row_idx = daily.index.get_loc(trade_date)
        if isinstance(row_idx, slice):
            continue
        train_end = row_idx - config.embargo_days
        if train_end < config.baseline_train_days:
            continue
        metrics_by_period = {}
        for window in (40, 20, 10, 5):
            hist = rolling_window(daily, train_end, window)
            if len(hist) < window:
                metrics_by_period = {}
                break
            trades = float(hist["baseline_has_trade"].sum())
            wins = hist.loc[hist["baseline_pnl_net_pct"] > 0.0, "baseline_pnl_pct"]
            metrics_by_period[str(window)] = {
                "Trades": trades,
                "WinRate": float(hist["baseline_win_net"].sum() / trades) if trades > 0 else 0.0,
                "AvgWin": float(wins.mean()) if not wins.empty else 0.0,
                "TotalPnL": float(hist["baseline_pnl_net_pct"].sum()),
                "ADR": float(hist["daily_adr_pct"].mean()),
            }
        if not metrics_by_period:
            continue
        forty = metrics_by_period["40"]
        if forty["ADR"] < 0.06 or forty["WinRate"] < 0.45 or forty["Trades"] < 5:
            continue
        daily_scores[symbol] = metrics_by_period
        base_stats.append({"Symbol": symbol, **forty})
    if not base_stats:
        return []
    passed = pd.DataFrame(base_stats)
    weights = {"5": 0.40, "10": 0.30, "20": 0.20, "40": 0.10}
    final_scores = np.zeros(len(passed), dtype=float)
    for period, weight in weights.items():
        period_rows = []
        for symbol in passed["Symbol"]:
            stats = daily_scores[symbol][period]
            period_rows.append({"Symbol": symbol, **stats})
        period_df = pd.DataFrame(period_rows)
        final_scores += pca_first_component_scores(period_df) * weight
    passed["score"] = final_scores
    passed.sort_values("score", ascending=False, inplace=True)
    top = passed.head(config.baseline_top_n)
    return list(zip(top["Symbol"], top["score"]))


def improved_rank_for_day(
    symbol_daily: dict[str, pd.DataFrame],
    trade_date: pd.Timestamp,
    config: StrategyConfig,
) -> pd.DataFrame:
    rows = []
    for symbol, daily in symbol_daily.items():
        if trade_date not in daily.index:
            continue
        row = daily.loc[trade_date]
        if row.isna().any():
            continue
        if row["prev_close"] < config.improved_min_price:
            continue
        if row["adv20_dollar"] < config.improved_min_adv_usd:
            continue
        if row["atr20_pct"] < config.improved_min_atr_pct:
            continue
        if row["opening_rvol20"] < config.improved_min_opening_rvol:
            continue
        if row["gap_pct"] < config.improved_min_gap_pct:
            continue
        if row["opening_drive_pct"] <= 0:
            continue
        rows.append(
            {
                "symbol": symbol,
                "atr20_pct": row["atr20_pct"],
                "opening_rvol20": row["opening_rvol20"],
                "gap_pct": row["gap_pct"],
                "opening_drive_pct": row["opening_drive_pct"],
                "opening_range_pct": row["opening_range_pct"],
                "adv20_dollar": row["adv20_dollar"],
                "atr20_abs": row["atr20_abs"],
                "opening_range_abs": row["opening_range_abs"],
            }
        )
    if not rows:
        return pd.DataFrame()
    ranked = pd.DataFrame(rows)
    features = {
        "opening_rvol20": 0.40,
        "atr20_pct": 0.25,
        "gap_pct": 0.15,
        "opening_drive_pct": 0.10,
        "opening_range_pct": 0.10,
    }
    score = np.zeros(len(ranked), dtype=float)
    for column, weight in features.items():
        values = ranked[column].astype(float)
        std = values.std(ddof=0)
        if std == 0:
            continue
        z = (values - values.mean()) / std
        score += z.to_numpy() * weight
    ranked["score"] = score
    ranked.sort_values("score", ascending=False, inplace=True)
    return ranked.head(config.improved_candidate_pool).reset_index(drop=True)


def baseline_candidate_for_day(
    symbol: str,
    daily_row: pd.Series,
    trade_date: pd.Timestamp,
) -> TradeCandidate | None:
    if not daily_row["baseline_has_trade"]:
        return None
    return TradeCandidate(
        symbol=symbol,
        date=trade_date,
        entry_time=pd.Timestamp(daily_row["baseline_entry_time"]),
        exit_time=pd.Timestamp(daily_row["baseline_exit_time"]),
        entry_price=float(daily_row["baseline_entry_price"]),
        exit_price=float(daily_row["baseline_exit_price"]),
        gross_return_pct=float(daily_row["baseline_pnl_pct"]),
        exit_reason="Original Fixed Exit",
        strategy="baseline",
    )


def generate_reversal_candidate(
    symbol: str,
    df_day: pd.DataFrame,
    feature_row: pd.Series,
    trade_date: pd.Timestamp,
    config: StrategyConfig,
    score: float,
) -> TradeCandidate | None:
    """Gap down reversal variant: gap down, lower open drive, then break above opening range."""
    if len(df_day) < 4:
        return None
    atr_abs = float(feature_row["atr20_abs"])
    if not np.isfinite(atr_abs) or atr_abs <= 0:
        return None

    opening_bar = df_day.iloc[0]
    opening_range_high = float(opening_bar["high"])

    # We want a reversal, so trigger is opening range high plus some buffer
    trigger = float(opening_range_high + max(0.02, atr_abs * config.improved_entry_buffer_atr_frac))

    hours = df_day.index.hour
    minutes = df_day.index.minute
    time_in_minutes = hours * 60 + minutes

    valid = (time_in_minutes >= 9 * 60 + 35) & (time_in_minutes < 10 * 60 + 30)
    confirm_mask = valid & (df_day["close"] > trigger)
    if not confirm_mask.any():
        return None

    signal_time = confirm_mask.idxmax()
    signal_pos = df_day.index.get_loc(signal_time)
    if signal_pos >= len(df_day) - 1:
        return None

    entry_bar = df_day.iloc[signal_pos + 1]
    entry_time = df_day.index[signal_pos + 1]
    entry_price = float(max(trigger, entry_bar["open"]))

    initial_stop = float(max(opening_bar["low"] - 0.025, entry_price - atr_abs * config.improved_stop_atr_frac))
    if initial_stop >= entry_price:
        return None

    current_stop = initial_stop
    risk_r = entry_price - initial_stop
    highest_high = entry_price
    exit_time = df_day.index[-1]
    exit_price = float(df_day.iloc[-1]["close"])
    exit_reason = "EOD"

    for bar in df_day.iloc[signal_pos + 1 :].itertuples():
        if bar.open <= current_stop:
            exit_time = bar.Index
            exit_price = float(bar.open)
            exit_reason = "Stop"
            break
        elif bar.low <= current_stop:
            exit_time = bar.Index
            exit_price = current_stop
            exit_reason = "Stop"
            break

        highest_high = max(highest_high, float(bar.high))
        if highest_high >= entry_price + risk_r:
            current_stop = max(current_stop, entry_price)
        if highest_high >= entry_price + 2.0 * risk_r:
            current_stop = max(current_stop, highest_high - risk_r)

    return TradeCandidate(
        symbol=symbol,
        date=trade_date,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_return_pct=(exit_price - entry_price) / entry_price,
        stop_price=initial_stop,
        exit_reason=exit_reason,
        score=score,
        strategy="improved_reversal",
    )


def improved_candidate_for_day(
    symbol: str,
    df_day: pd.DataFrame,
    feature_row: pd.Series,
    trade_date: pd.Timestamp,
    config: StrategyConfig,
    score: float,
) -> TradeCandidate | None:
    if len(df_day) < 4:
        return None
    atr_abs = float(feature_row["atr20_abs"])
    if not np.isfinite(atr_abs) or atr_abs <= 0:
        return None
    opening_bar = df_day.iloc[0]
    trigger = float(opening_bar["high"] + max(0.05, atr_abs * config.improved_entry_buffer_atr_frac))

    hours = df_day.index.hour
    minutes = df_day.index.minute
    time_in_minutes = hours * 60 + minutes
    valid = (time_in_minutes >= 9 * 60 + 35) & (time_in_minutes < 10 * 60 + 30)

    confirm_mask = valid & (df_day["close"] > trigger)
    if not confirm_mask.any():
        return None
    signal_time = confirm_mask.idxmax()
    signal_pos = df_day.index.get_loc(signal_time)
    if signal_pos >= len(df_day) - 1:
        return None
    entry_bar = df_day.iloc[signal_pos + 1]
    entry_time = df_day.index[signal_pos + 1]
    entry_price = float(max(trigger, entry_bar["open"]))
    initial_stop = float(max(opening_bar["low"] - 0.025, entry_price - atr_abs * config.improved_stop_atr_frac))
    if initial_stop >= entry_price:
        return None
    current_stop = initial_stop
    risk_r = entry_price - initial_stop
    highest_high = entry_price
    exit_time = df_day.index[-1]
    exit_price = float(df_day.iloc[-1]["close"])
    exit_reason = "EOD"

    for bar in df_day.iloc[signal_pos + 1 :].itertuples():
        if bar.open <= current_stop:
            exit_time = bar.Index
            exit_price = float(bar.open)
            exit_reason = "Stop"
            break
        elif bar.low <= current_stop:
            exit_time = bar.Index
            exit_price = current_stop
            exit_reason = "Stop"
            break

        highest_high = max(highest_high, float(bar.high))
        if highest_high >= entry_price + risk_r:
            current_stop = max(current_stop, entry_price)
        if highest_high >= entry_price + 2.0 * risk_r:
            current_stop = max(current_stop, highest_high - risk_r)
    return TradeCandidate(
        symbol=symbol,
        date=trade_date,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_return_pct=(exit_price - entry_price) / entry_price,
        stop_price=initial_stop,
        exit_reason=exit_reason,
        score=score,
        strategy="improved",
    )


def rule_watchlist_templates() -> list[RuleWatchlistTemplate]:
    return [
        RuleWatchlistTemplate(
            name="gap_up_continuation",
            min_adv_usd=5_000_000.0,
            min_atr_pct=0.02,
            min_opening_rvol=0.0,
            gap_min=0.0,
            gap_max=0.03,
            watchlist_size=20,
            max_positions=2,
            weight=0.45,
        ),
        RuleWatchlistTemplate(
            name="gap_up_high_rvol",
            min_adv_usd=5_000_000.0,
            min_atr_pct=0.02,
            min_opening_rvol=1.0,
            gap_min=0.0,
            gap_max=0.03,
            watchlist_size=15,
            max_positions=2,
            weight=0.45,
            w_rvol=0.60,
            w_gap=-0.20,
            w_atr=0.10,
            w_adv=0.10,
        ),
        RuleWatchlistTemplate(
            name="gap_up_liquid_small_gap",
            min_adv_usd=10_000_000.0,
            min_atr_pct=0.018,
            min_opening_rvol=0.50,
            gap_min=0.0,
            gap_max=0.02,
            watchlist_size=18,
            max_positions=3,
            weight=0.30,
            w_rvol=0.35,
            w_gap=-0.15,
            w_atr=0.15,
            w_adv=0.35,
        ),
        RuleWatchlistTemplate(
            name="gap_up_atr_expansion",
            min_adv_usd=7_500_000.0,
            min_atr_pct=0.03,
            min_opening_rvol=0.80,
            gap_min=0.005,
            gap_max=0.05,
            watchlist_size=12,
            max_positions=2,
            weight=0.40,
            w_rvol=0.35,
            w_gap=-0.10,
            w_atr=0.35,
            w_adv=0.20,
        ),
        RuleWatchlistTemplate(
            name="liquid_open_drive",
            min_adv_usd=15_000_000.0,
            min_atr_pct=0.018,
            min_opening_rvol=0.20,
            gap_min=-0.005,
            gap_max=0.025,
            watchlist_size=20,
            max_positions=3,
            weight=0.30,
            w_rvol=0.25,
            w_gap=-0.05,
            w_atr=0.20,
            w_adv=0.50,
        ),
        RuleWatchlistTemplate(
            name="high_atr_any_gap",
            min_adv_usd=7_500_000.0,
            min_atr_pct=0.035,
            min_opening_rvol=0.30,
            gap_min=-0.01,
            gap_max=0.04,
            watchlist_size=15,
            max_positions=2,
            weight=0.40,
            w_rvol=0.20,
            w_gap=-0.05,
            w_atr=0.50,
            w_adv=0.25,
        ),
        RuleWatchlistTemplate(
            name="gap_down_reversal",
            min_adv_usd=5_000_000.0,
            min_atr_pct=0.02,
            min_opening_rvol=0.0,
            gap_min=-0.05,
            gap_max=-0.002,
            watchlist_size=10,
            max_positions=2,
            weight=0.45,
            w_rvol=0.45,
            w_gap=0.20,
            w_atr=0.20,
            w_adv=0.15,
        ),
        RuleWatchlistTemplate(
            name="deep_gap_down_reversal",
            min_adv_usd=7_500_000.0,
            min_atr_pct=0.025,
            min_opening_rvol=0.80,
            gap_min=-0.08,
            gap_max=-0.015,
            watchlist_size=10,
            max_positions=2,
            weight=0.40,
            w_rvol=0.35,
            w_gap=0.35,
            w_atr=0.15,
            w_adv=0.15,
        ),
    ]


def apply_rule_score(ranked: pd.DataFrame, template: RuleWatchlistTemplate) -> pd.DataFrame:
    score = np.zeros(len(ranked), dtype=float)
    for column, weight in (
        ("opening_rvol20", template.w_rvol),
        ("gap_pct", template.w_gap),
        ("atr20_pct", template.w_atr),
        ("adv20_dollar", template.w_adv),
    ):
        score += zscore(ranked[column].astype(float)) * weight
    ranked = ranked.copy()
    ranked["score"] = score
    return ranked


def rule_rank_for_day(
    symbol_daily: dict[str, pd.DataFrame],
    trade_date: pd.Timestamp,
    template: RuleWatchlistTemplate,
) -> pd.DataFrame:
    rows = []
    for symbol, daily in symbol_daily.items():
        if trade_date not in daily.index:
            continue
        row = daily.loc[trade_date]
        required = ["prev_close", "adv20_dollar", "atr20_pct", "opening_rvol20", "gap_pct"]
        if row[required].isna().any():
            continue
        if row["prev_close"] < 5.0:
            continue
        if row["adv20_dollar"] < template.min_adv_usd:
            continue
        if row["atr20_pct"] < template.min_atr_pct:
            continue
        if row["opening_rvol20"] < template.min_opening_rvol:
            continue
        if row["gap_pct"] < template.gap_min or row["gap_pct"] > template.gap_max:
            continue
        rows.append(
            {
                "symbol": symbol,
                "adv20_dollar": float(row["adv20_dollar"]),
                "opening_rvol20": float(row["opening_rvol20"]),
                "gap_pct": float(row["gap_pct"]),
                "atr20_pct": float(row["atr20_pct"]),
                "baseline_has_trade": int(row["baseline_has_trade"]),
                "entry_time": row["baseline_entry_time"],
                "exit_time": row["baseline_exit_time"],
                "entry_price": float(row["baseline_entry_price"]) if pd.notna(row["baseline_entry_price"]) else np.nan,
                "exit_price": float(row["baseline_exit_price"]) if pd.notna(row["baseline_exit_price"]) else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame()
    ranked = apply_rule_score(pd.DataFrame(rows), template)
    ranked.sort_values(["score", "symbol"], ascending=[False, True], inplace=True)
    return ranked.head(template.watchlist_size).reset_index(drop=True)


def hybrid_rank_for_day(
    symbol_daily: dict[str, pd.DataFrame],
    trade_date: pd.Timestamp,
    config: StrategyConfig,
    template: RuleWatchlistTemplate,
    pca_ranked: list[tuple[str, float]] | None = None,
) -> pd.DataFrame:
    if pca_ranked is None:
        pca_ranked = baseline_rank_for_day(symbol_daily, trade_date, config)
    rows = []
    for symbol, pca_score in pca_ranked:
        daily = symbol_daily[symbol]
        if trade_date not in daily.index:
            continue
        row = daily.loc[trade_date]
        required = ["adv20_dollar", "opening_rvol20", "gap_pct", "atr20_pct"]
        if row[required].isna().any():
            continue
        if row["prev_close"] < 5.0:
            continue
        if row["adv20_dollar"] < template.min_adv_usd:
            continue
        if row["atr20_pct"] < template.min_atr_pct:
            continue
        if row["opening_rvol20"] < template.min_opening_rvol:
            continue
        if row["gap_pct"] < template.gap_min or row["gap_pct"] > template.gap_max:
            continue
        rows.append(
            {
                "symbol": symbol,
                "pca_score": float(pca_score),
                "adv20_dollar": float(row["adv20_dollar"]),
                "opening_rvol20": float(row["opening_rvol20"]),
                "gap_pct": float(row["gap_pct"]),
                "atr20_pct": float(row["atr20_pct"]),
                "baseline_has_trade": int(row["baseline_has_trade"]),
                "entry_time": row["baseline_entry_time"],
                "entry_price": float(row["baseline_entry_price"]) if pd.notna(row["baseline_entry_price"]) else np.nan,
                "exit_time": row["baseline_exit_time"],
                "exit_price": float(row["baseline_exit_price"]) if pd.notna(row["baseline_exit_price"]) else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame()
    ranked = apply_rule_score(pd.DataFrame(rows), template)
    rule_score = ranked["score"].to_numpy(dtype=float)
    pca_values = ranked["pca_score"].astype(float)
    pca_std = pca_values.std(ddof=0)
    if pca_std == 0 or np.isnan(pca_std):
        ranked["hybrid_score"] = rule_score
    else:
        pca_z = (pca_values - pca_values.mean()) / pca_std
        ranked["hybrid_score"] = (pca_z * 0.5) + (rule_score * 0.5)
    ranked.sort_values(["hybrid_score", "symbol"], ascending=[False, True], inplace=True)
    return ranked.reset_index(drop=True)


def precompute_baseline_rankings(
    symbol_daily: dict[str, pd.DataFrame],
    trade_dates: list[pd.Timestamp],
    config: StrategyConfig,
) -> dict[pd.Timestamp, list[tuple[str, float]]]:
    return {trade_date: baseline_rank_for_day(symbol_daily, trade_date, config) for trade_date in trade_dates}


def precompute_baseline_candidates(
    symbol_daily: dict[str, pd.DataFrame],
    trade_dates: list[pd.Timestamp],
    ranking_cache: dict[pd.Timestamp, list[tuple[str, float]]],
) -> dict[pd.Timestamp, list[TradeCandidate]]:
    candidate_cache: dict[pd.Timestamp, list[TradeCandidate]] = {}
    for trade_date in trade_dates:
        day_candidates = []
        for symbol, score in ranking_cache.get(trade_date, []):
            candidate = baseline_candidate_for_day(symbol, symbol_daily[symbol].loc[trade_date], trade_date)
            if candidate:
                candidate.score = score
                day_candidates.append(candidate)
        day_candidates.sort(key=lambda trade: (trade.entry_time, -(trade.score or 0.0), trade.symbol))
        candidate_cache[trade_date] = day_candidates
    return candidate_cache


def simulate_baseline(
    baseline_candidates: dict[pd.Timestamp, list[TradeCandidate]],
    trade_dates: list[pd.Timestamp],
    config: StrategyConfig,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    equity = config.initial_equity
    equity_curve = []
    daily_returns = []
    trade_rows = []
    for trade_date in trade_dates:
        candidates = baseline_candidates.get(trade_date, [])
        portfolio_return = 0.0
        if candidates:
            chosen = candidates[0]
            net_return = apply_trading_costs(
                chosen.entry_price,
                chosen.exit_price,
                COMMISSION_RATE,
                config.slippage_bps_each_side,
            )
            portfolio_return = config.baseline_allocation * net_return
            trade_rows.append(
                {
                    **asdict(chosen),
                    "net_trade_return": net_return,
                    "portfolio_return": portfolio_return,
                    "weight": config.baseline_allocation,
                }
            )
        equity *= 1.0 + portfolio_return
        equity_curve.append({"date": trade_date, "equity": equity})
        daily_returns.append({"date": trade_date, "daily_return": portfolio_return})
    equity_series = pd.DataFrame(equity_curve).set_index("date")["equity"]
    return_series = pd.DataFrame(daily_returns).set_index("date")["daily_return"]
    trade_log = pd.DataFrame(trade_rows)
    return equity_series, return_series, trade_log


def simulate_improved(
    intraday_data: dict[str, pd.DataFrame],
    symbol_daily: dict[str, pd.DataFrame],
    trade_dates: list[pd.Timestamp],
    config: StrategyConfig,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    equity = config.initial_equity
    equity_curve = []
    daily_returns = []
    trade_rows = []

    fixed_weight = min(config.improved_max_weight, config.improved_risk_budget / config.improved_fixed_stop_pct)

    for trade_date in trade_dates:
        # Rank daily symbols
        ranked_symbols_df = improved_rank_for_day(symbol_daily, trade_date, config)

        candidates = []
        for row in ranked_symbols_df.itertuples():
            symbol = row.symbol
            score = row.score
            feature_row = symbol_daily[symbol].loc[trade_date]

            # Find intraday day df
            try:
                df_day = intraday_data[symbol].loc[str(trade_date.date())].sort_index()
                candidate = improved_candidate_for_day(symbol, df_day, feature_row, trade_date, config, score)
                if candidate:
                    candidates.append(candidate)
            except KeyError:
                continue

        # Sort candidates chronologically to take earliest breakouts
        candidates.sort(key=lambda x: x.entry_time)
        selected = candidates[: config.improved_max_positions]

        for trade in selected:
            trade.weight = fixed_weight

        total_weight = sum(trade.weight or 0.0 for trade in selected)
        scale = 1.0
        if total_weight > config.improved_total_weight_cap > 0:
            scale = config.improved_total_weight_cap / total_weight

        portfolio_return = 0.0
        for trade in selected:
            final_weight = (trade.weight or 0.0) * scale
            trade.weight = final_weight
            net_return = apply_trading_costs(
                trade.entry_price,
                trade.exit_price,
                COMMISSION_RATE,
                config.slippage_bps_each_side,
            )
            contribution = final_weight * net_return
            portfolio_return += contribution
            trade_rows.append(
                {
                    **asdict(trade),
                    "net_trade_return": net_return,
                    "portfolio_return": contribution,
                }
            )

        equity *= 1.0 + portfolio_return
        equity_curve.append({"date": trade_date, "equity": equity})
        daily_returns.append({"date": trade_date, "daily_return": portfolio_return})

    equity_series = pd.DataFrame(equity_curve).set_index("date")["equity"]
    return_series = pd.DataFrame(daily_returns).set_index("date")["daily_return"]
    trade_log = pd.DataFrame(trade_rows)
    return equity_series, return_series, trade_log


def simulate_rule_watchlist(
    intraday_data: dict[str, pd.DataFrame],
    symbol_daily: dict[str, pd.DataFrame],
    trade_dates: list[pd.Timestamp],
    template: RuleWatchlistTemplate,
    config: StrategyConfig,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    equity = config.initial_equity
    equity_curve = []
    daily_returns = []
    trade_rows = []

    is_reversal = "reversal" in template.name.lower()

    for trade_date in trade_dates:
        ranked = rule_rank_for_day(symbol_daily, trade_date, template)
        if ranked.empty:
            equity_curve.append({"date": trade_date, "equity": equity})
            daily_returns.append({"date": trade_date, "daily_return": 0.0})
            continue

        candidates = []
        for row in ranked.itertuples():
            symbol = row.symbol
            score = row.score
            feature_row = symbol_daily[symbol].loc[trade_date]

            try:
                df_day = intraday_data[symbol].loc[str(trade_date.date())].sort_index()
                if is_reversal:
                    candidate = generate_reversal_candidate(symbol, df_day, feature_row, trade_date, config, score)
                else:
                    candidate = improved_candidate_for_day(symbol, df_day, feature_row, trade_date, config, score)

                if candidate:
                    candidates.append(candidate)
            except KeyError:
                continue

        # Sort chronologically, then by score
        candidates.sort(key=lambda x: (x.entry_time, -(x.score or 0.0)))
        selected = candidates[: template.max_positions]

        # Calculate weight scaling
        weights = [template.weight] * len(selected)
        total_weight = sum(weights)
        scale = config.improved_total_weight_cap / total_weight if total_weight > config.improved_total_weight_cap else 1.0

        portfolio_return = 0.0
        for trade in selected:
            final_weight = template.weight * scale
            trade.weight = final_weight
            trade.strategy = "rule_watchlist"

            net_return = apply_trading_costs(
                trade.entry_price,
                trade.exit_price,
                COMMISSION_RATE,
                config.slippage_bps_each_side,
            )
            contribution = final_weight * net_return
            portfolio_return += contribution
            trade_rows.append(
                {
                    **asdict(trade),
                    "net_trade_return": net_return,
                    "portfolio_return": contribution,
                }
            )

        equity *= 1.0 + portfolio_return
        equity_curve.append({"date": trade_date, "equity": equity})
        daily_returns.append({"date": trade_date, "daily_return": portfolio_return})

    equity_series = pd.DataFrame(equity_curve).set_index("date")["equity"]
    return_series = pd.DataFrame(daily_returns).set_index("date")["daily_return"]
    trade_log = pd.DataFrame(trade_rows)
    return equity_series, return_series, trade_log


def dynamic_exit_from_baseline_entry(
    intraday_data: dict[str, pd.DataFrame],
    trade: TradeCandidate,
    config: StrategyConfig,
) -> tuple[float, pd.Timestamp, str]:
    df_day = intraday_data[trade.symbol].loc[str(trade.date.date())].sort_index()
    try:
        entry_pos = df_day.index.get_loc(trade.entry_time)
    except KeyError:
        return trade.exit_price, trade.exit_time, "Fallback"
    if isinstance(entry_pos, slice):
        entry_pos = entry_pos.start
    risk_abs = trade.entry_price * config.improved_fixed_stop_pct
    current_stop = trade.entry_price - risk_abs
    highest_high = trade.entry_price
    exit_time = df_day.index[-1]
    exit_price = float(df_day.iloc[-1]["close"])
    exit_reason = "EOD"

    for bar in df_day.iloc[entry_pos + 1 :].itertuples():
        if bar.open <= current_stop:
            exit_time = bar.Index
            exit_price = float(bar.open)
            exit_reason = "Dynamic Stop"
            break
        elif bar.low <= current_stop:
            exit_time = bar.Index
            exit_price = current_stop
            exit_reason = "Dynamic Stop"
            break

        highest_high = max(highest_high, float(bar.high))
        if highest_high >= trade.entry_price + (config.dynamic_break_even_r * risk_abs):
            current_stop = max(current_stop, trade.entry_price)
        if highest_high >= trade.entry_price + (config.dynamic_trail_start_r * risk_abs):
            current_stop = max(current_stop, highest_high - (config.dynamic_trail_offset_r * risk_abs))
    return exit_price, exit_time, exit_reason


def simulate_hybrid_watchlist(
    symbol_daily: dict[str, pd.DataFrame],
    trade_dates: list[pd.Timestamp],
    config: StrategyConfig,
    template: RuleWatchlistTemplate,
    ranking_cache: dict[pd.Timestamp, list[tuple[str, float]]],
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    equity = config.initial_equity
    equity_curve = []
    daily_returns = []
    trade_rows = []
    fixed_weight = min(config.improved_max_weight, config.improved_risk_budget / config.improved_fixed_stop_pct)

    for trade_date in trade_dates:
        ranked = hybrid_rank_for_day(symbol_daily, trade_date, config, template, ranking_cache.get(trade_date))
        if ranked.empty:
            equity_curve.append({"date": trade_date, "equity": equity})
            daily_returns.append({"date": trade_date, "daily_return": 0.0})
            continue

        triggered = ranked[ranked["baseline_has_trade"] == 1].copy()
        triggered.sort_values(["hybrid_score", "entry_time", "symbol"], ascending=[False, True, True], inplace=True)

        selected = triggered.head(config.improved_max_positions)
        weights = [fixed_weight] * len(selected)
        total_weight = sum(weights)
        scale = config.improved_total_weight_cap / total_weight if total_weight > config.improved_total_weight_cap else 1.0

        portfolio_return = 0.0
        for row in selected.itertuples(index=False):
            net_return = apply_trading_costs(
                float(row.entry_price),
                float(row.exit_price),
                COMMISSION_RATE,
                config.slippage_bps_each_side,
            )
            final_weight = fixed_weight * scale
            contribution = final_weight * net_return
            portfolio_return += contribution
            trade_rows.append(
                {
                    "symbol": row.symbol,
                    "date": trade_date,
                    "entry_time": row.entry_time,
                    "exit_time": row.exit_time,
                    "entry_price": float(row.entry_price),
                    "exit_price": float(row.exit_price),
                    "gross_return_pct": float((row.exit_price - row.entry_price) / row.entry_price),
                    "exit_reason": "Original Fixed Exit",
                    "score": float(row.hybrid_score),
                    "weight": final_weight,
                    "strategy": "hybrid_watchlist",
                    "net_trade_return": net_return,
                    "portfolio_return": contribution,
                }
            )
        equity *= 1.0 + portfolio_return
        equity_curve.append({"date": trade_date, "equity": equity})
        daily_returns.append({"date": trade_date, "daily_return": portfolio_return})

    equity_series = pd.DataFrame(equity_curve).set_index("date")["equity"]
    return_series = pd.DataFrame(daily_returns).set_index("date")["daily_return"]
    trade_log = pd.DataFrame(trade_rows)
    return equity_series, return_series, trade_log


def simulate_phase3_dynamic_exit(
    intraday_data: dict[str, pd.DataFrame],
    baseline_candidates: dict[pd.Timestamp, list[TradeCandidate]],
    trade_dates: list[pd.Timestamp],
    config: StrategyConfig,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    equity = config.initial_equity
    equity_curve = []
    daily_returns = []
    trade_rows = []
    fixed_weight = min(config.improved_max_weight, config.improved_risk_budget / config.improved_fixed_stop_pct)

    for trade_date in trade_dates:
        portfolio_return = 0.0
        selected = baseline_candidates.get(trade_date, [])[: config.improved_max_positions]
        weights = [fixed_weight] * len(selected)
        total_weight = sum(weights)
        scale = config.improved_total_weight_cap / total_weight if total_weight > config.improved_total_weight_cap else 1.0

        for trade in selected:
            exit_price, exit_time, exit_reason = dynamic_exit_from_baseline_entry(intraday_data, trade, config)
            net_return = apply_trading_costs(
                trade.entry_price,
                exit_price,
                COMMISSION_RATE,
                config.slippage_bps_each_side,
            )
            final_weight = fixed_weight * scale
            contribution = final_weight * net_return
            portfolio_return += contribution
            trade_rows.append(
                {
                    **asdict(trade),
                    "exit_time": exit_time,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "net_trade_return": net_return,
                    "portfolio_return": contribution,
                    "weight": final_weight,
                    "strategy": "phase3_dynamic_exit",
                }
            )
        equity *= 1.0 + portfolio_return
        equity_curve.append({"date": trade_date, "equity": equity})
        daily_returns.append({"date": trade_date, "daily_return": portfolio_return})
    equity_series = pd.DataFrame(equity_curve).set_index("date")["equity"]
    return_series = pd.DataFrame(daily_returns).set_index("date")["daily_return"]
    trade_log = pd.DataFrame(trade_rows)
    return equity_series, return_series, trade_log


def select_rule_watchlist_template(
    intraday_data: dict[str, pd.DataFrame],
    symbol_daily: dict[str, pd.DataFrame],
    selection_dates: list[pd.Timestamp],
    config: StrategyConfig,
) -> tuple[RuleWatchlistTemplate, dict[str, Any]]:
    best_template = None
    best_metrics = None
    best_objective = -float("inf")
    for template in rule_watchlist_templates():
        equity, returns, trade_log = simulate_rule_watchlist(intraday_data, symbol_daily, selection_dates, template, config)
        metrics = compute_metrics(equity, returns, trade_log, config.initial_equity)
        objective = objective_score(metrics)
        if objective > best_objective:
            best_objective = objective
            best_template = template
            best_metrics = metrics
    if best_template is None or best_metrics is None:
        raise RuntimeError("No rule-based watchlist template could be selected.")
    return best_template, {"objective": best_objective, **best_metrics}


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "final_equity_delta": candidate["final_equity"] - baseline["final_equity"],
        "total_return_pct_delta": candidate["total_return_pct"] - baseline["total_return_pct"],
        "sharpe_delta": candidate["sharpe"] - baseline["sharpe"],
        "max_drawdown_pct_delta": candidate["max_drawdown_pct"] - baseline["max_drawdown_pct"],
    }


def rank_variants(metric_map: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_total_return = sorted(
        ((name, metrics["total_return_pct"]) for name, metrics in metric_map.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    by_sharpe = sorted(
        ((name, metrics["sharpe"]) for name, metrics in metric_map.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    by_drawdown = sorted(
        ((name, metrics["max_drawdown_pct"]) for name, metrics in metric_map.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "by_total_return_pct": [{"strategy": name, "value": value} for name, value in by_total_return],
        "by_sharpe": [{"strategy": name, "value": value} for name, value in by_sharpe],
        "by_max_drawdown_pct": [{"strategy": name, "value": value} for name, value in by_drawdown],
    }


def load_data(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    return {symbol: df.sort_index() for symbol, df in data.items() if not df.empty}


def run_research(data_path: Path, output_dir: Path, config: StrategyConfig) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(data_path)
    cache_path = output_dir / f"symbol_daily_cache_{config.get_hash()}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            symbol_daily = pickle.load(handle)
    else:
        symbol_daily = precompute_baseline_daily_summary(data)
        with cache_path.open("wb") as handle:
            pickle.dump(symbol_daily, handle)

    # Fix 6-2: Use common dates intersection
    all_dates = sorted(set().union(*[set(df.index) for df in symbol_daily.values()]))

    selection_dates = all_dates[: config.baseline_train_days]
    trade_dates = all_dates[config.baseline_train_days + config.embargo_days :]
    baseline_ranking_cache = precompute_baseline_rankings(symbol_daily, trade_dates, config)
    baseline_candidate_cache = precompute_baseline_candidates(symbol_daily, trade_dates, baseline_ranking_cache)

    baseline_equity, baseline_returns, baseline_log = simulate_baseline(
        baseline_candidate_cache, trade_dates, config
    )
    improved_equity, improved_returns, improved_log = simulate_improved(
        intraday_data=data,
        symbol_daily=symbol_daily,
        trade_dates=trade_dates,
        config=config,
    )
    selected_template, selection_metrics = select_rule_watchlist_template(
        intraday_data=data,
        symbol_daily=symbol_daily,
        selection_dates=selection_dates,
        config=config,
    )
    rule_equity, rule_returns, rule_log = simulate_rule_watchlist(
        intraday_data=data,
        symbol_daily=symbol_daily,
        trade_dates=trade_dates,
        template=selected_template,
        config=config,
    )
    hybrid_equity, hybrid_returns, hybrid_log = simulate_hybrid_watchlist(
        symbol_daily=symbol_daily,
        trade_dates=trade_dates,
        config=config,
        template=selected_template,
        ranking_cache=baseline_ranking_cache,
    )
    dynamic_equity, dynamic_returns, dynamic_log = simulate_phase3_dynamic_exit(
        intraday_data=data,
        baseline_candidates=baseline_candidate_cache,
        trade_dates=trade_dates,
        config=config,
    )

    baseline_metrics = compute_metrics(baseline_equity, baseline_returns, baseline_log, config.initial_equity)
    improved_metrics = compute_metrics(improved_equity, improved_returns, improved_log, config.initial_equity)
    rule_metrics = compute_metrics(rule_equity, rule_returns, rule_log, config.initial_equity)
    hybrid_metrics = compute_metrics(hybrid_equity, hybrid_returns, hybrid_log, config.initial_equity)
    dynamic_metrics = compute_metrics(dynamic_equity, dynamic_returns, dynamic_log, config.initial_equity)

    metric_map = {
        "baseline": baseline_metrics,
        "phase1_improved": improved_metrics,
        "phase2_rule_watchlist": rule_metrics,
        "phase2_hybrid_watchlist": hybrid_metrics,
        "phase3_dynamic_exit": dynamic_metrics,
    }
    comparison = {
        "config": asdict(config),
        "evaluation_window": {
            "start": str(trade_dates[0].date()) if trade_dates else None,
            "end": str(trade_dates[-1].date()) if trade_dates else None,
            "days": len(trade_dates),
        },
        "rule_selection_window": {
            "start": str(selection_dates[0].date()) if selection_dates else None,
            "end": str(selection_dates[-1].date()) if selection_dates else None,
            "days": len(selection_dates),
        },
        "baseline": baseline_metrics,
        "improved": improved_metrics,
        "phase2_rule_watchlist": rule_metrics,
        "phase2_hybrid_watchlist": hybrid_metrics,
        "phase3_dynamic_exit": dynamic_metrics,
        "selected_rule_template": asdict(selected_template),
        "rule_selection_metrics": selection_metrics,
        "phase3_dynamic_parameters": {
            "initial_stop_pct": config.improved_fixed_stop_pct,
            "break_even_r": config.dynamic_break_even_r,
            "trail_start_r": config.dynamic_trail_start_r,
            "trail_offset_r": config.dynamic_trail_offset_r,
        },
        "variant_ranking": rank_variants(metric_map),
        "improvement": metric_delta(improved_metrics, baseline_metrics),
        "phase2_vs_baseline": metric_delta(rule_metrics, baseline_metrics),
        "hybrid_vs_baseline": metric_delta(hybrid_metrics, baseline_metrics),
        "phase3_vs_baseline": metric_delta(dynamic_metrics, baseline_metrics),
        "phase2_vs_phase1": metric_delta(rule_metrics, improved_metrics),
        "hybrid_vs_phase1": metric_delta(hybrid_metrics, improved_metrics),
        "phase3_vs_phase1": metric_delta(dynamic_metrics, improved_metrics),
        "notes": {
            "baseline": "Rolling 40-day original-style PCA watchlist, 1-day embargo, one earliest breakout per day, 90% allocation.",
            "improved": "Full improved execution: actually uses improved rule ranking & entry criteria (ATR stop, breakeven, traitling stop) for phase 1.",
            "phase2_rule_watchlist": "The PCA watchlist is replaced by a rule-based daily watchlist. Reversal templates use specialized logic.",
            "phase2_hybrid_watchlist": "The original PCA top-10 candidate pool reassessed with ATR/ADV/opening RVOL/gap rule scores.",
            "phase3_dynamic_exit": "Phase1 PCA base candidate selection with 3% initial stop, break-even at 0.75R, trailing 1.5R.",
        },
    }

    baseline_log.to_csv(output_dir / "baseline_trade_log.csv", index=False)
    improved_log.to_csv(output_dir / "improved_trade_log.csv", index=False)
    rule_log.to_csv(output_dir / "phase2_rule_watchlist_trade_log.csv", index=False)
    hybrid_log.to_csv(output_dir / "phase2_hybrid_watchlist_trade_log.csv", index=False)
    dynamic_log.to_csv(output_dir / "phase3_dynamic_exit_trade_log.csv", index=False)
    pd.DataFrame(
        {
            "baseline_equity": baseline_equity,
            "improved_equity": improved_equity,
            "phase2_rule_equity": rule_equity,
            "phase2_hybrid_equity": hybrid_equity,
            "phase3_dynamic_equity": dynamic_equity,
        }
    ).to_csv(
        output_dir / "equity_curves.csv"
    )
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=False, indent=2)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the original Stallion ORB logic with an improved ORB design.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("russell3000_60d_5min.pkl"),
        help="Path to the pickled dict[symbol, DataFrame] of 5-minute bars.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory where the comparison artifacts should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = run_research(args.data_path, args.output_dir, StrategyConfig())
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()