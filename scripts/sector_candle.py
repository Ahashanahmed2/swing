import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# কনফিগারেশন
INPUT_CSV = './csv/mongodb.csv'
OUTPUT_WEEKLY_DIR = './csv/sector/weekly/'

os.makedirs(OUTPUT_WEEKLY_DIR, exist_ok=True)

def calculate_rsi_series(close_prices, period=14):
    """Calculate RSI for a series"""
    valid_prices = close_prices.dropna().reset_index(drop=True)
    
    if len(valid_prices) < period + 1:
        return [None] * len(close_prices)
    
    deltas = valid_prices.diff()
    gains = deltas.copy()
    losses = deltas.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    avg_gain = gains.iloc[1:period+1].mean()
    avg_loss = losses.iloc[1:period+1].mean()
    
    rsi_values = [None] * period
    
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    rsi_values.append(round(rsi, 2))
    
    for i in range(period + 1, len(valid_prices)):
        avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_values.append(round(rsi, 2))
    
    return rsi_values

def calculate_weighted_price(day_df, price_col, weight_col='marketCap'):
    """Market Cap Weighted price"""
    valid = day_df.dropna(subset=[weight_col, price_col])
    if len(valid) == 0 or valid[weight_col].sum() == 0:
        return np.nan
    total_mcap = valid[weight_col].sum()
    return (valid[weight_col] * valid[price_col]).sum() / total_mcap

def calculate_sector_index_weekly():
    """
    প্লাটফর্মের মতো সেক্টর ইনডেক্স ক্যান্ডেল তৈরি
    ✅ Market Cap Weighted Index Method
    ✅ Open = আগের সপ্তাহের Close (continuity)
    ✅ High/Low = Actual traded High/Low
    ✅ Index Scaling
    """
    print("=" * 60)
    print("সেক্টর ইনডেক্স উইকলি ক্যান্ডেল (Market Cap Weighted)")
    print("   ✅ Continuity Open | Actual High/Low | Scaled")
    print("=" * 60)
    
    # 1. ডাটা লোড
    print("\n1. ডাটা লোড হচ্ছে...")
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df['sector'] = df['sector'].fillna('Unknown').apply(lambda x: str(x).strip())
    
    print(f"   মোট রো: {len(df):,}")
    print(f"   সিম্বল: {df['symbol'].nunique():,}")
    print(f"   সেক্টর: {df['sector'].nunique():,}")
    
    # 2. DSE উইক স্টার্ট (Sunday)
    def get_dse_week_start(date):
        weekday = date.dayofweek
        if weekday == 6:
            return date
        else:
            return date - timedelta(days=weekday + 1)
    
    df['week_start'] = df['date'].apply(get_dse_week_start)
    
    # 3. সেক্টর ভিত্তিক Weekly ক্যান্ডেল
    print("\n2. উইকলি ক্যান্ডেল তৈরি হচ্ছে...")
    
    unique_sectors = df['sector'].unique()
    total_sectors = len(unique_sectors)
    
    for idx, sector in enumerate(unique_sectors, 1):
        sector_df = df[df['sector'] == sector].copy()
        symbols = sector_df['symbol'].unique()
        
        print(f"\n[{idx}/{total_sectors}] {sector} ({len(symbols)} symbols)")
        
        all_weeks = sorted(sector_df['week_start'].unique())
        sector_weekly = []
        prev_close = None  # আগের সপ্তাহের Close
        
        for week_start in all_weeks:
            week_data = sector_df[sector_df['week_start'] == week_start]
            
            if len(week_data) == 0:
                continue
            
            week_dates = sorted(week_data['date'].unique())
            first_day = week_dates[0]
            last_day = week_dates[-1]
            trading_days = len(week_dates)
            
            # ============ MARKET CAP WEIGHTED INDEX ============
            
            # ✅ Close: শেষ দিনের Market Cap Weighted Close
            last_day_data = week_data[week_data['date'] == last_day]
            weighted_close = calculate_weighted_price(last_day_data, 'close', 'marketCap')
            
            # ✅ Open: আগের সপ্তাহের Close (continuity)
            # প্রথম সপ্তাহের জন্য প্রথম দিনের Open
            if prev_close is not None:
                weighted_open = prev_close
            else:
                first_day_data = week_data[week_data['date'] == first_day]
                weighted_open = calculate_weighted_price(first_day_data, 'open', 'marketCap')
            
            # ✅ High: Actual traded High values (max of all daily highs)
            week_highs = []
            for date in week_dates:
                day_data = week_data[week_data['date'] == date]
                valid_high = day_data.dropna(subset=['high'])
                if len(valid_high) > 0:
                    week_highs.append(valid_high['high'].max())
            
            # ✅ Low: Actual traded Low values (min of all daily lows)
            week_lows = []
            for date in week_dates:
                day_data = week_data[week_data['date'] == date]
                valid_low = day_data.dropna(subset=['low'])
                if len(valid_low) > 0:
                    week_lows.append(valid_low['low'].min())
            
            week_high = max(week_highs) if week_highs else np.nan
            week_low = min(week_lows) if week_lows else np.nan
            
            # Volume, Value, Trades - Sum
            total_volume = week_data['volume'].sum()
            total_value = week_data['value'].sum() if 'value' in week_data.columns else 0
            total_trades = week_data['trades'].sum() if 'trades' in week_data.columns else 0
            
            sector_weekly.append({
                'sector': sector,
                'week_start': week_start,
                'week_end_date': last_day,
                'open': round(weighted_open, 2) if not np.isnan(weighted_open) else None,
                'high': round(week_high, 2) if not np.isnan(week_high) else None,
                'low': round(week_low, 2) if not np.isnan(week_low) else None,
                'close': round(weighted_close, 2) if not np.isnan(weighted_close) else None,
                'volume': int(total_volume) if not np.isnan(total_volume) else 0,
                'value': round(total_value, 2) if not np.isnan(total_value) else 0,
                'trades': int(total_trades) if not np.isnan(total_trades) else 0,
                'trading_days': trading_days,
                'symbols_count': len(symbols)
            })
            
            # Update prev_close for next week
            prev_close = weighted_close
        
        # DataFrame এ কনভার্ট
        weekly_df = pd.DataFrame(sector_weekly)
        
        if len(weekly_df) > 0:
            weekly_df = weekly_df.sort_values('week_start')
            
            # ✅ Scale to 100 for first week
            first_close = weekly_df['close'].iloc[0]
            if pd.notna(first_close) and first_close > 0:
                scale_factor = 100.0 / first_close
                weekly_df['open'] = (weekly_df['open'] * scale_factor).round(2)
                weekly_df['high'] = (weekly_df['high'] * scale_factor).round(2)
                weekly_df['low'] = (weekly_df['low'] * scale_factor).round(2)
                weekly_df['close'] = (weekly_df['close'] * scale_factor).round(2)
            
            # Change calculate
            weekly_df['change'] = weekly_df['close'].diff().round(2)
            
            # RSI calculate
            close_prices = weekly_df['close']
            rsi_values = calculate_rsi_series(close_prices, period=14)
            weekly_df['rsi'] = rsi_values
            
            # Save
            filename = f"{sector.replace(' ', '_').replace('/', '_').replace('&', 'and').lower()}_weekly.csv"
            filepath = os.path.join(OUTPUT_WEEKLY_DIR, filename)
            
            columns = ['sector', 'week_start', 'week_end_date', 'open', 'high', 'low', 'close', 
                      'volume', 'value', 'trades', 'change', 'rsi', 'trading_days', 'symbols_count']
            
            weekly_df[columns].to_csv(filepath, index=False)
            
            # Latest candle info
            latest = weekly_df.iloc[-1]
            rsi_val = f"{latest['rsi']:.2f}" if pd.notna(latest['rsi']) else 'N/A'
            print(f"  ✓ Saved: {filename}")
            print(f"     Latest: O:{latest['open']:.2f} H:{latest['high']:.2f} L:{latest['low']:.2f} C:{latest['close']:.2f}")
            print(f"     Ch:{latest['change']:+.2f} | RSI:{rsi_val} | Vol:{latest['volume']:,.0f}")
    
    print(f"\n✅ Done! Files saved in: {OUTPUT_WEEKLY_DIR}")

# Run
calculate_sector_index_weekly()