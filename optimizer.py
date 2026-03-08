import itertools
from backtester import run_backtest

def optimize_strategy(df, param_grid, verbose=True):
    """
    Runs a grid search over the strategy parameters to maximize total PnL (PDCA cycle base).
    param_grid: dictionary mapping param name to a list of values to test.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    best_pnl = -float('inf')
    best_params = None
    best_trades = None
    
    if verbose:
        print(f"Starting PDCA Optimization... Total combinations: {len(combinations)}")
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        total_pnl, df_trades = run_backtest(df, params)
        
        if total_pnl > best_pnl:
            best_pnl = total_pnl
            best_params = params
            best_trades = df_trades
            
    return best_params, best_pnl, best_trades
