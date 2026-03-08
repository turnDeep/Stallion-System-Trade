import numpy as np
import pandas as pd

class BreakoutStrategy:
    def __init__(self, entry_start_time="09:35:00", entry_end_time="11:00:00", 
                 take_profit_pct=0.03, stop_loss_pct=0.01, min_volume_ratio=1.0, use_historical_rvol=False):
        """
        Vectorized Breakout / Momentum strategy comparing Daily RVOL vs Historical Time-of-Day RVOL.
        """
        self.entry_start_time = pd.to_timedelta(entry_start_time)
        self.entry_end_time = pd.to_timedelta(entry_end_time)
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.min_volume_ratio = min_volume_ratio
        self.use_historical_rvol = use_historical_rvol

    def generate_daily_signals_vectorized(self, df_day):
        """
        df_day should be intraday data for a single day, sorted chronologically.
        Vectorized approach for extreme speed.
        Returns: trade_info dict if a trade occurs, else None
        """
        if len(df_day) < 2:
            return None

        # 1. Opening Range High
        opening_range_high = df_day['high'].iloc[0]
        
        # 2. Time filtering
        time_only = pd.to_timedelta(df_day.index.strftime('%H:%M:%S'))
        valid_entry_times = (time_only >= self.entry_start_time) & (time_only <= self.entry_end_time)
        
        # 3. RVOL Filter Comparison
        if self.use_historical_rvol:
            # Type B: True Historical Time-of-day RVOL (e.g. 20-day average at exactly 10:30)
            if 'volume' in df_day.columns and 'cum_vol_avg_20d' in df_day.columns:
                volume_surge = df_day['volume'] > (df_day['cum_vol_avg_20d'] * self.min_volume_ratio)
            else:
                volume_surge = pd.Series(True, index=df_day.index)
        else:
            # Type A: Daily Cumulative RVOL (Average of the 5-min bars seen *so far today*)
            if 'volume' in df_day.columns:
                cum_vol_avg = df_day['volume'].expanding().mean().shift(1)
                volume_surge = df_day['volume'] > (cum_vol_avg * self.min_volume_ratio)
            else:
                volume_surge = pd.Series(True, index=df_day.index)
        
        # 4. Find the first bar that breaks the ORH during valid times with a volume surge
        breakout_mask = valid_entry_times & (df_day['high'] > opening_range_high) & volume_surge
        
        if not breakout_mask.any():
            return None # No breakout today
            
        # 4. Entry point
        entry_idx = breakout_mask.idxmax() # First occurrence of True
        entry_row_idx = df_day.index.get_loc(entry_idx)
        
        # If breakout happens at the very end of the day, no time to trade
        if entry_row_idx >= len(df_day) - 1:
            return None
            
        entry_row = df_day.loc[entry_idx]
        entry_price = max(opening_range_high, entry_row['open'])
        
        # 5. Extract forward data (after entry bar) to calculate exits
        forward_df = df_day.iloc[entry_row_idx+1:].copy()
        
        if forward_df.empty:
            return None
            
        tp_price = entry_price * (1 + self.take_profit_pct)
        sl_price = entry_price * (1 - self.stop_loss_pct)
        
        # Vectorized Exit Condition checks
        # Prioritize SL if both hit in same 5min candle (conservative)
        hit_sl = forward_df['low'] <= sl_price
        hit_tp = forward_df['high'] >= tp_price
        
        # Find the first index where SL or TP is hit
        sl_idx = hit_sl.idxmax() if hit_sl.any() else None
        tp_idx = hit_tp.idxmax() if hit_tp.any() else None
        
        # Determine which exit triggered first temporally
        exit_reason = "EOD Close"
        exit_time = forward_df.index[-1]
        exit_price = forward_df.iloc[-1]['close']
        
        if sl_idx is not None and tp_idx is not None:
            if sl_idx <= tp_idx: # SL hit first or at same time
                exit_time, exit_price, exit_reason = sl_idx, sl_price, "Stop Loss"
            else:
                exit_time, exit_price, exit_reason = tp_idx, tp_price, "Take Profit"
        elif sl_idx is not None:
            exit_time, exit_price, exit_reason = sl_idx, sl_price, "Stop Loss"
        elif tp_idx is not None:
            exit_time, exit_price, exit_reason = tp_idx, tp_price, "Take Profit"
        
        # Enforce 15:55 EOD Close boundary (if exit time is later than EOD)
        # Note: In 5min data, 15:55:00 is usually the last bar of regular session
        eod_time = df_day.index[0].replace(hour=15, minute=55, second=0)
        if exit_time > eod_time:
            # Fallback to EOD close at 15:55
            try:
                eod_idx = df_day.index.get_loc(eod_time, method='pad')
                exit_time = df_day.index[eod_idx]
                exit_price = df_day.iloc[eod_idx]['close']
                exit_reason = "EOD Close"
            except KeyError:
                pass # Use standard end of day
        
        return {
            'entry_time': entry_idx,
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': (exit_price - entry_price) / entry_price
        }
