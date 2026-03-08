import os
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from backtester import run_backtest
from optimizer import optimize_strategy

def calculate_adr(df):
    daily = df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    if len(daily) < 2: return 0
    daily['adr'] = (daily['high'] - daily['low']) / daily['low']
    return daily['adr'].mean()

def run_portfolio_sim(top_100_symbols, trades_dict):
    all_events = []
    trade_id_counter = 0
    for sym in top_100_symbols:
        trades = trades_dict.get(sym)
        if trades is not None and not trades.empty:
            for t in trades.itertuples():
                all_events.append({'time': t.entry_time, 'type': 'ENTRY', 'id': trade_id_counter, 'symbol': sym, 'pnl_pct': t.pnl_pct})
                all_events.append({'time': t.exit_time, 'type': 'EXIT', 'id': trade_id_counter, 'symbol': sym, 'pnl_pct': t.pnl_pct})
                trade_id_counter += 1
                
    all_events.sort(key=lambda x: (x['time'], 0 if x['type'] == 'EXIT' else 1))
    
    STARTING_CAPITAL = 10000.0
    RISK_PER_TRADE = 0.10
    capital = STARTING_CAPITAL
    cash = STARTING_CAPITAL
    active_trades = {}
    
    for ev in all_events:
        if ev['type'] == 'ENTRY':
            trade_size = capital * RISK_PER_TRADE
            actual_investment = min(trade_size, cash)
            if actual_investment >= 10:
                cash -= actual_investment
                active_trades[ev['id']] = actual_investment
        elif ev['type'] == 'EXIT':
            trade_id = ev['id']
            if trade_id in active_trades:
                invested = active_trades.pop(trade_id)
                profit = invested * ev['pnl_pct']
                cash += (invested + profit)
                capital += profit
                
    return capital, trade_id_counter

def main():
    print("Loading universe symbols...")
    with open('russell3000_5min.pkl', 'rb') as f:
        old_data = pickle.load(f)
    symbols = list(old_data.keys())
    
    output_pkl = 'russell3000_60d_5min.pkl'
    data_60d = {}
    
    if os.path.exists(output_pkl):
        print(f"Loading existing {output_pkl}...")
        with open(output_pkl, 'rb') as f:
            data_60d = pickle.load(f)
    else:
        print(f"Downloading 60 days of 5m data for {len(symbols)} symbols from yfinance...")
        batch_size = 100
        for i in tqdm(range(0, len(symbols), batch_size), desc="yfinance bulk downloading"):
            batch = symbols[i:i+batch_size]
            batch_str = " ".join(batch)
            df_batch = yf.download(batch_str, period='60d', interval='5m', group_by='ticker', progress=False, threads=10)
            
            if len(batch) == 1:
                df_batch.columns = [c.lower() for c in df_batch.columns]
                try:
                    df_batch.index = df_batch.index.tz_convert('America/New_York').tz_localize(None)
                except:
                    pass
                data_60d[batch[0]] = df_batch
            else:
                for sym in batch:
                    if sym in df_batch.columns.levels[0]:
                        df_sym = df_batch[sym].dropna(how='all').copy()
                        if not df_sym.empty:
                            df_sym.columns = [c.lower() for c in df_sym.columns]
                            try:
                                df_sym.index = df_sym.index.tz_convert('America/New_York').tz_localize(None)
                            except:
                                pass
                            data_60d[sym] = df_sym
                            
        with open(output_pkl, 'wb') as f:
            pickle.dump(data_60d, f)
            
    print(f"Loaded {len(data_60d)} symbols with data.")
    
    sample_sym = None
    for sym, df_sym in data_60d.items():
        if len(df_sym) > 3000:
            sample_sym = sym
            break
            
    if not sample_sym: 
        print("Could not find a symbol with enough data.")
        return
        
    all_dates = sorted(list(set(data_60d[sample_sym].index.date)))
    print(f"Total trading days fetched: {len(all_dates)} (Min: {all_dates[0]}, Max: {all_dates[-1]})")
    
    n_days = len(all_dates)
    if n_days < 20: 
        print("Not enough days to do train/val/test splits!")
        return
        
    test_days_len = 5
    val_days_len = 5
    train_days_len = n_days - test_days_len - val_days_len
    if train_days_len <= 0: train_days_len = 5
    
    train1_dates = all_dates[:train_days_len]
    val_dates = all_dates[train_days_len:train_days_len+val_days_len]
    
    train2_dates = all_dates[val_days_len:train_days_len+val_days_len] 
    test_dates = all_dates[train_days_len+val_days_len:]
    
    print(f"\n--- DATA SPLITS ---")
    print(f"Train1 (Selection for Val): {len(train1_dates)} days ({train1_dates[0]} to {train1_dates[-1]})")
    print(f"Validation (Hyperparam tuning): {len(val_dates)} days ({val_dates[0]} to {val_dates[-1]})")
    print(f"Train2 (Selection for Test): {len(train2_dates)} days ({train2_dates[0]} to {train2_dates[-1]})")
    print(f"Test (Final OOS): {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")
    
    param_grid = {
        "entry_start_time": ["09:35:00"], 
        "entry_end_time": ["10:30:00"], 
        "take_profit_pct": [0.10], 
        "stop_loss_pct": [0.03],  
        "min_volume_ratio": [1.0], 
        "use_historical_rvol": [False]
    }
    
    def evaluate_period(dates, desc):
        res = []
        for sym, df in tqdm(data_60d.items(), desc=desc):
            per_df = df.loc[str(dates[0]):str(dates[-1])]
            if len(per_df) < 50: continue
            
            adr = calculate_adr(per_df)
            best_params, best_pnl, best_trades = optimize_strategy(per_df, param_grid, verbose=False)
            if best_trades is not None and not best_trades.empty:
                res.append({
                    'Symbol': sym, 'ADR_pct': adr, 'Total_Trades': len(best_trades),
                    'Win_Rate': len(best_trades[best_trades['pnl_pct'] > 0]) / len(best_trades),
                    'Total_PnL': best_pnl, 'Params': best_params
                })
        return pd.DataFrame(res)
        
    def collect_trades(dates, desc):
        trades_dict = {}
        core_params = {
            "entry_start_time": "09:35:00", "entry_end_time": "10:30:00", 
            "take_profit_pct": 0.10, "stop_loss_pct": 0.03,  
            "min_volume_ratio": 1.0, "use_historical_rvol": False
        }
        for sym, df in tqdm(data_60d.items(), desc=desc):
            per_df = df.loc[str(dates[0]):str(dates[-1])]
            if len(per_df) > 0:
                _, trades = run_backtest(per_df, core_params)
                trades_dict[sym] = trades
        return trades_dict
        
    print("\n--- Precomputing Metrics ---")
    train1_df = evaluate_period(train1_dates, "Evaluating Train1")
    val_trades_dict = collect_trades(val_dates, "Collecting Val Trades")
    
    train2_df = evaluate_period(train2_dates, "Evaluating Train2")
    test_trades_dict = collect_trades(test_dates, "Collecting Test Trades")
    
    print("\n--- PHASE 2: VALIDATION (Grid Search) ---")
    adr_thresholds = [0.03, 0.04, 0.05, 0.06]
    trade_thresholds = [5, 10, 15, 20, 25, 30] 
    win_rate_thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
    
    results = []
    
    for adr_th in adr_thresholds:
        for tr_th in trade_thresholds:
            for wr_th in win_rate_thresholds:
                valid = train1_df[(train1_df['ADR_pct'] >= adr_th) & 
                                     (train1_df['Total_Trades'] >= tr_th) & 
                                     (train1_df['Win_Rate'] >= wr_th) & 
                                     (train1_df['Total_PnL'] > 0)]
                top_100 = valid.sort_values('Total_PnL', ascending=False).head(100)
                
                if len(top_100) == 0: continue
                    
                top_symbols = top_100['Symbol'].tolist()
                final_cap, _ = run_portfolio_sim(top_symbols, val_trades_dict)
                multiplier = final_cap / 10000.0
                
                results.append({
                    'ADR_Min': adr_th, 'Trades_Min': tr_th, 'WinRate_Min': wr_th,
                    'Selected_Stocks': len(top_100), 'Val_Return(%)': (multiplier-1)*100
                })
                
    res_df = pd.DataFrame(results).sort_values('Val_Return(%)', ascending=False)
    print("\nTop 5 Hyperparameter Combinations on Validation Set:")
    print(res_df.head(5).to_string(index=False))
    
    best_params = res_df.iloc[0]
    opt_adr = best_params['ADR_Min']
    opt_tr = best_params['Trades_Min']
    opt_wr = best_params['WinRate_Min']
    
    print(f"\n=> GOLDEN PARAMETERS SELECTED FROM VALIDATION:")
    print(f"   ADR >= {opt_adr*100:.0f}%, Trades >= {opt_tr}, WinRate >= {opt_wr*100:.0f}%")
    
    print("\n--- PHASE 3: FINAL OUT-OF-SAMPLE TEST ---")
    print(f"Applying Golden Parameters to Train2 ({len(train2_dates)} days) to select Test stocks...")
    
    valid_test = train2_df[(train2_df['ADR_pct'] >= opt_adr) & 
                           (train2_df['Total_Trades'] >= opt_tr) & 
                           (train2_df['Win_Rate'] >= opt_wr) & 
                           (train2_df['Total_PnL'] > 0)]
    top_100_test = valid_test.sort_values('Total_PnL', ascending=False).head(100)
    
    print(f"Selected {len(top_100_test)} stocks for Final Test.")
    
    top_symbols_test = top_100_test['Symbol'].tolist()
    final_test_cap, test_trades = run_portfolio_sim(top_symbols_test, test_trades_dict)
    
    print("\n==================================================")
    print(f"--- FINAL OOS TEST PORTFOLIO RESULTS ({len(test_dates)} DAYS) ---")
    print(f"Starting Capital: $10,000.00")
    print(f"Final Capital:    ${final_test_cap:,.2f}")
    if final_test_cap > 0:
        multiplier = final_test_cap / 10000.0
        print(f"\nFinal Test Return: {multiplier:.3f}x ({(multiplier-1)*100:.2f}%)")
        print(f"Total OOS Trades Executed: {test_trades}")
    print("==================================================")

if __name__ == '__main__':
    main()
