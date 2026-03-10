import pandas as pd

from strategy import DynamicExitStrategy, check_entry_condition


COMMISSION_RATE = 0.002
ROUND_TRIP_COST = COMMISSION_RATE * 2.0


def net_pnl_pct(entry_price: float, exit_price: float) -> float:
    gross = (exit_price - entry_price) / entry_price
    return gross - ROUND_TRIP_COST


def run_backtest(df, strategy_params=None):
    """
    Runs the backtest on the full historical dataset using the shared DynamicExitStrategy.

    The reported `pnl_pct` is net of the assumed 0.2% per-side transaction cost.
    """
    if strategy_params is None:
        strategy_params = {}

    df = df.copy()
    df["date_only"] = df.index.date
    trades = []

    for current_date, df_day in df.groupby("date_only"):
        df_day = df_day.sort_index()
        if len(df_day) < 2:
            continue

        opening_high = df_day.iloc[0]["high"]
        active_strategy = None
        entry_price = 0.0
        entry_time = None

        for idx, row in df_day.iterrows():
            current_time = idx

            if not active_strategy:
                if check_entry_condition(row["high"], opening_high, current_time):
                    entry_price = max(opening_high, row["open"])
                    entry_time = current_time
                    active_strategy = DynamicExitStrategy(entry_price=entry_price, **strategy_params)
                continue

            exit_triggered, exit_reason, execute_price = active_strategy.check_exit_condition(row["low"], current_time)
            if exit_triggered:
                execute_price = min(execute_price, row["open"])
                gross_pnl = (execute_price - entry_price) / entry_price
                trades.append(
                    {
                        "date": current_date,
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "exit_time": current_time,
                        "exit_price": execute_price,
                        "exit_reason": exit_reason,
                        "gross_pnl_pct": gross_pnl,
                        "pnl_pct": gross_pnl - ROUND_TRIP_COST,
                    }
                )
                break

            exit_triggered, exit_reason, _ = active_strategy.check_exit_condition(row["high"], current_time)
            if exit_triggered:
                gross_pnl = (row["close"] - entry_price) / entry_price
                trades.append(
                    {
                        "date": current_date,
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "exit_time": current_time,
                        "exit_price": row["close"],
                        "exit_reason": exit_reason,
                        "gross_pnl_pct": gross_pnl,
                        "pnl_pct": gross_pnl - ROUND_TRIP_COST,
                    }
                )
                break

    df_trades = pd.DataFrame(trades)
    if df_trades.empty:
        return 0.0, df_trades

    total_pnl = df_trades["pnl_pct"].sum()
    return total_pnl, df_trades


def print_backtest_stats(df_trades):
    if df_trades.empty:
        print("No trades executed.")
        return

    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades["pnl_pct"] > 0])
    losing_trades = len(df_trades[df_trades["pnl_pct"] <= 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_pnl = df_trades["pnl_pct"].sum() * 100

    avg_win = df_trades[df_trades["pnl_pct"] > 0]["pnl_pct"].mean() * 100 if winning_trades > 0 else 0
    avg_loss = df_trades[df_trades["pnl_pct"] <= 0]["pnl_pct"].mean() * 100 if losing_trades > 0 else 0

    print("-" * 30)
    print("BACKTEST RESULTS (NET OF COSTS)")
    print("-" * 30)
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate:     {win_rate:.2%}")
    print(f"Total PnL:    {total_pnl:.2f}%")
    print(f"Avg Win:      {avg_win:.2f}%")
    print(f"Avg Loss:     {avg_loss:.2f}%")
    print(f"Round Trip Cost Assumption: {ROUND_TRIP_COST * 100:.2f}%")
    print("-" * 30)
