import pandas as pd
from strategy import BreakoutStrategy

def run_backtest(df, strategy_params):
    """
    Runs the backtest on the full historical dataset.
    df: Intraday DataFrame with datetime index.
    strategy_params: dictionary of parameters for BreakoutStrategy.
    """
    strategy = BreakoutStrategy(**strategy_params)
    
    # Group data by date
    # Create a column for date only
    df['date_only'] = df.index.date
    
    trades = []
    
    for current_date, df_day in df.groupby('date_only'):
        df_day = df_day.sort_index()
        trade = strategy.generate_daily_signals_vectorized(df_day)
        if trade:
            trade['date'] = current_date
            trades.append(trade)
            
    df_trades = pd.DataFrame(trades)
    
    if df_trades.empty:
        return 0.0, df_trades
        
    total_pnl = df_trades['pnl_pct'].sum()
    return total_pnl, df_trades

def print_backtest_stats(df_trades):
    if df_trades.empty:
        print("No trades executed.")
        return
        
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades['pnl_pct'] > 0])
    losing_trades = len(df_trades[df_trades['pnl_pct'] <= 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_pnl = df_trades['pnl_pct'].sum() * 100 # percentage
    
    avg_win = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].mean() * 100 if winning_trades > 0 else 0
    avg_loss = df_trades[df_trades['pnl_pct'] <= 0]['pnl_pct'].mean() * 100 if losing_trades > 0 else 0
    
    print("-" * 30)
    print("BACKTEST RESULTS")
    print("-" * 30)
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate:     {win_rate:.2%}")
    print(f"Total PnL:    {total_pnl:.2f}%")
    print(f"Avg Win:      {avg_win:.2f}%")
    print(f"Avg Loss:     {avg_loss:.2f}%")
    print("-" * 30)
