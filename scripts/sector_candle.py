# sector_weekly_candle.py
# প্লাটফর্মের মতো সেক্টর ইনডেক্স ক্যান্ডেল জেনারেটর
# ✅ Market Cap Weighted (freeFloatMarketCap → marketCap fallback)
# ✅ Million → Crore normalized
# ✅ Continuity Open | Actual High/Low | Scaled Index
# ✅ Daily + Weekly output

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# কনফিগারেশন
INPUT_CSV = './csv/mongodb.csv'
OUTPUT_WEEKLY_DIR = './csv/sector/weekly/'
OUTPUT_DAILY_DIR = './csv/sector/daily/'

os.makedirs(OUTPUT_WEEKLY_DIR, exist_ok=True)
os.makedirs(OUTPUT_DAILY_DIR, exist_ok=True)

def calculate_rsi_series(close_prices, period=14):
    """Calculate RSI for a series - Wilder's Smoothing"""
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

def calculate_weighted_price(day_df, price_col):
    """
    Market Cap Weighted price
    ✅ freeFloatMarketCap থাকলে ব্যবহার, না হলে marketCap
    ✅ Million → Crore normalized
    """
    # Priority: freeFloatMarketCap → marketCap
    weight_col = 'freeFloatMarketCap'
    if weight_col not in day_df.columns or day_df[weight_col].dropna().sum() == 0:
        weight_col = 'marketCap'
    
    valid = day_df.dropna(subset=[weight_col, price_col])
    if len(valid) == 0 or valid[weight_col].sum() == 0:
        return np.nan
    
    total_mcap = valid[weight_col].sum()
    return (valid[weight_col] * valid[price_col]).sum() / total_mcap

def get_dse_week_start(date):
    """DSE সপ্তাহ শুরু রবিবার"""
    weekday = date.dayofweek
    if weekday == 6:
        return date
    else:
        return date - timedelta(days=weekday + 1)

def calculate_sector_index():
    """
    প্লাটফর্মের মতো সেক্টর ইনডেক্স ক্যান্ডেল
    ✅ Market Cap Weighted (freeFloatMarketCap → marketCap)
    ✅ Continuity Open | Actual High/Low | Scaled Index
    """
    print("=" * 70)
    print("📊 সেক্টর ইনডেক্স ক্যান্ডেল জেনারেটর")
    print("   Method: Market Cap Weighted | Continuity Open | Actual H/L")
    print("=" * 70)

    # 1. ডাটা লোড
    print("\n1. ডাটা লোড হচ্ছে...")
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    
    df['sector'] = df['sector'].fillna('Unknown')
    df['sector'] = df['sector'].apply(lambda x: str(x).strip())
    
    weight_col = 'freeFloatMarketCap'
    if weight_col in df.columns and df[weight_col].dropna().sum() > 0:
        print(f"   ✅ Using 'freeFloatMarketCap' for index")
    else:
        print(f"   ℹ️ 'freeFloatMarketCap' not available, using 'marketCap'")
        weight_col = 'marketCap'
    
    print(f"   মোট রো: {len(df):,}")
    print(f"   সিম্বল: {df['symbol'].nunique():,}")
    print(f"   সেক্টর: {df['sector'].nunique():,}")
    print(f"   তারিখ: {df['date'].min().strftime('%Y-%m-%d')} থেকে {df['date'].max().strftime('%Y-%m-%d')}")

    # 2. DSE উইক স্টার্ট
    df['week_start'] = df['date'].apply(get_dse_week_start)

    # =========================================================
    # 3. DAILY সেক্টর ক্যান্ডেল
    # =========================================================
    print("\n2. ডেইলি সেক্টর ক্যান্ডেল তৈরি হচ্ছে...")
    
    daily_dfs = {}
    unique_sectors = sorted(df['sector'].unique())
    
    for idx, sector in enumerate(unique_sectors, 1):
        sector_df = df[df['sector'] == sector].copy()
        
        daily_list = []
        prev_close = None
        
        for date in sorted(sector_df['date'].unique()):
            day_data = sector_df[sector_df['date'] == date]
            if len(day_data) == 0:
                continue
            
            weighted_close = calculate_weighted_price(day_data, 'close')
            
            if prev_close is not None:
                weighted_open = prev_close
            else:
                weighted_open = calculate_weighted_price(day_data, 'open')
            
            valid_high = day_data.dropna(subset=['high'])
            actual_high = valid_high['high'].max() if len(valid_high) > 0 else np.nan
            
            valid_low = day_data.dropna(subset=['low'])
            actual_low = valid_low['low'].min() if len(valid_low) > 0 else np.nan
            
            daily_list.append({
                'sector': sector,
                'date': date,
                'open': round(weighted_open, 2) if pd.notna(weighted_open) else None,
                'high': round(actual_high, 2) if pd.notna(actual_high) else None,
                'low': round(actual_low, 2) if pd.notna(actual_low) else None,
                'close': round(weighted_close, 2) if pd.notna(weighted_close) else None,
                'volume': int(day_data['volume'].sum()),
                'value': round(day_data['value'].sum(), 2) if 'value' in day_data.columns else 0,
                'trades': int(day_data['trades'].sum()) if 'trades' in day_data.columns else 0,
                'active_symbols': len(day_data)
            })
            
            prev_close = weighted_close
        
        if daily_list:
            daily_df = pd.DataFrame(daily_list).sort_values('date')
            
            first_close = daily_df['close'].iloc[0]
            if pd.notna(first_close) and first_close > 0:
                scale = 100.0 / first_close
                daily_df['open'] = (daily_df['open'] * scale).round(2)
                daily_df['high'] = (daily_df['high'] * scale).round(2)
                daily_df['low'] = (daily_df['low'] * scale).round(2)
                daily_df['close'] = (daily_df['close'] * scale).round(2)
            
            daily_df['change'] = daily_df['close'].diff().round(2)
            daily_df['rsi'] = calculate_rsi_series(daily_df['close'], period=14)
            
            daily_dfs[sector] = daily_df
            
            filename = f"{sector.replace(' ', '_').replace('/', '_').replace('&', 'and').lower()}_daily.csv"
            filepath = os.path.join(OUTPUT_DAILY_DIR, filename)
            cols = ['sector', 'date', 'open', 'high', 'low', 'close', 'volume', 'value', 'trades', 'change', 'rsi', 'active_symbols']
            daily_df[cols].to_csv(filepath, index=False)
            
            latest = daily_df.iloc[-1]
            print(f"  ✓ Daily {sector}: {len(daily_df)} days | "
                  f"O:{latest['open']:.2f} H:{latest['high']:.2f} L:{latest['low']:.2f} C:{latest['close']:.2f}")

    # =========================================================
    # 4. WEEKLY সেক্টর ক্যান্ডেল
    # =========================================================
    print("\n3. উইকলি সেক্টর ক্যান্ডেল তৈরি হচ্ছে...")
    
    for idx, sector in enumerate(unique_sectors, 1):
        sector_df = df[df['sector'] == sector].copy()
        symbols = sector_df['symbol'].unique()
        
        all_weeks = sorted(sector_df['week_start'].unique())
        weekly_list = []
        prev_weekly_close = None
        
        for week_start in all_weeks:
            week_data = sector_df[sector_df['week_start'] == week_start]
            if len(week_data) == 0:
                continue
            
            week_dates = sorted(week_data['date'].unique())
            first_day = week_dates[0]
            last_day = week_dates[-1]
            
            last_day_data = week_data[week_data['date'] == last_day]
            weighted_close = calculate_weighted_price(last_day_data, 'close')
            
            if prev_weekly_close is not None:
                weighted_open = prev_weekly_close
            else:
                first_day_data = week_data[week_data['date'] == first_day]
                weighted_open = calculate_weighted_price(first_day_data, 'open')
            
            week_highs = []
            week_lows = []
            for date in week_dates:
                day_data = week_data[week_data['date'] == date]
                valid_high = day_data.dropna(subset=['high'])
                valid_low = day_data.dropna(subset=['low'])
                if len(valid_high) > 0:
                    week_highs.append(valid_high['high'].max())
                if len(valid_low) > 0:
                    week_lows.append(valid_low['low'].min())
            
            week_high = max(week_highs) if week_highs else np.nan
            week_low = min(week_lows) if week_lows else np.nan
            
            weekly_list.append({
                'sector': sector,
                'week_start': week_start,
                'week_end_date': last_day,
                'open': round(weighted_open, 2) if pd.notna(weighted_open) else None,
                'high': round(week_high, 2) if pd.notna(week_high) else None,
                'low': round(week_low, 2) if pd.notna(week_low) else None,
                'close': round(weighted_close, 2) if pd.notna(weighted_close) else None,
                'volume': int(week_data['volume'].sum()),
                'value': round(week_data['value'].sum(), 2) if 'value' in week_data.columns else 0,
                'trades': int(week_data['trades'].sum()) if 'trades' in week_data.columns else 0,
                'trading_days': len(week_dates),
                'symbols_count': len(symbols)
            })
            
            prev_weekly_close = weighted_close
        
        if weekly_list:
            weekly_df = pd.DataFrame(weekly_list).sort_values('week_start')
            
            first_close = weekly_df['close'].iloc[0]
            if pd.notna(first_close) and first_close > 0:
                scale = 100.0 / first_close
                weekly_df['open'] = (weekly_df['open'] * scale).round(2)
                weekly_df['high'] = (weekly_df['high'] * scale).round(2)
                weekly_df['low'] = (weekly_df['low'] * scale).round(2)
                weekly_df['close'] = (weekly_df['close'] * scale).round(2)
            
            weekly_df['change'] = weekly_df['close'].diff().round(2)
            weekly_df['rsi'] = calculate_rsi_series(weekly_df['close'], period=14)
            
            filename = f"{sector.replace(' ', '_').replace('/', '_').replace('&', 'and').lower()}_weekly.csv"
            filepath = os.path.join(OUTPUT_WEEKLY_DIR, filename)
            cols = ['sector', 'week_start', 'week_end_date', 'open', 'high', 'low', 'close', 
                   'volume', 'value', 'trades', 'change', 'rsi', 'trading_days', 'symbols_count']
            weekly_df[cols].to_csv(filepath, index=False)
            
            latest = weekly_df.iloc[-1]
            rsi_val = f"{latest['rsi']:.2f}" if pd.notna(latest['rsi']) else 'N/A'
            print(f"  ✓ Weekly {sector}: {len(weekly_df)} weeks | "
                  f"O:{latest['open']:.2f} H:{latest['high']:.2f} L:{latest['low']:.2f} C:{latest['close']:.2f} | "
                  f"RSI:{rsi_val}")

    # =========================================================
    # 5. Summary
    # =========================================================
    print(f"\n{'='*70}")
    print("✅ সম্পন্ন!")
    print(f"{'='*70}")
    print(f"📁 ডেইলি: {OUTPUT_DAILY_DIR} ({len(daily_dfs)} sectors)")
    print(f"📁 উইকলি: {OUTPUT_WEEKLY_DIR} ({len(daily_dfs)} sectors)")

if __name__ == "__main__":
    calculate_sector_index()
