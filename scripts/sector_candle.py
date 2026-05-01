# sector_weekly_candle.py
# প্লাটফর্মের মতো সেক্টর ইনডেক্স ক্যান্ডেল জেনারেটর (Price Weighted + RSI 14)
# সম্পূর্ণ ডাটা থেকে জেনারেট করে

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

def calculate_rsi(close_prices, period=14):
    """RSI (Relative Strength Index) - Wilder's Smoothing"""
    prices = close_prices.dropna().values
    
    if len(prices) < period + 1:
        # কম ডাটা থাকলে None ফিল
        return [None] * len(close_prices)
    
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    rsi = np.full(len(prices), None)
    
    # First average gain and loss (Simple average for first period)
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    
    # First RSI value
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = round(100.0 - (100.0 / (1.0 + rs)), 2)
    
    # Wilder's Smoothing for rest
    for i in range(period + 1, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = round(100.0 - (100.0 / (1.0 + rs)), 2)
    
    return rsi.tolist()

def get_dse_week_start(date):
    """DSE সপ্তাহ শুরু রবিবার"""
    weekday = date.dayofweek  # Monday=0, Sunday=6
    if weekday == 6:  # Sunday
        return date
    else:
        return date - timedelta(days=weekday + 1)

def calculate_sector_index():
    """
    🎯 প্লাটফর্মের মতো সেক্টর ইনডেক্স ক্যান্ডেল
    Method: Equal Weighted (সকল স্টকের সমান ওজন)
    RSI: 14 period (Wilder's Smoothing)
    """
    print("=" * 70)
    print("📊 সেক্টর ইনডেক্স ক্যান্ডেল জেনারেটর")
    print("   Method: Equal Weighted | RSI: 14 (Wilder)")
    print("=" * 70)
    
    # 1. ডাটা লোড
    print("\n1. ডাটা লোড হচ্ছে...")
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df['sector'] = df['sector'].fillna('Unknown').apply(lambda x: str(x).strip())
    
    print(f"   মোট রো: {len(df):,}")
    print(f"   সিম্বল: {df['symbol'].nunique():,}")
    print(f"   সেক্টর: {df['sector'].nunique():,}")
    print(f"   তারিখ: {df['date'].min().strftime('%Y-%m-%d')} থেকে {df['date'].max().strftime('%Y-%m-%d')}")
    
    total_days = (df['date'].max() - df['date'].min()).days
    print(f"   মোট দিন: {total_days}+")
    
    # 2. DSE উইক স্টার্ট
    df['week_start'] = df['date'].apply(get_dse_week_start)
    
    # 3. ডেইলি সেক্টর ক্যান্ডেল (Equal Weighted)
    print("\n2. ডেইলি সেক্টর ক্যান্ডেল তৈরি হচ্ছে...")
    
    daily_dfs = {}
    
    for sector, sector_df in df.groupby('sector'):
        daily_list = []
        
        for date, day_df in sector_df.groupby('date'):
            valid_open = day_df.dropna(subset=['open'])
            valid_high = day_df.dropna(subset=['high'])
            valid_low = day_df.dropna(subset=['low'])
            valid_close = day_df.dropna(subset=['close'])
            
            if len(valid_close) == 0:
                continue
            
            daily_list.append({
                'sector': sector,
                'date': date,
                'open': round(valid_open['open'].mean(), 2),
                'high': round(valid_high['high'].max(), 2),
                'low': round(valid_low['low'].min(), 2),
                'close': round(valid_close['close'].mean(), 2),
                'volume': int(day_df['volume'].sum()),
                'value': round(day_df['value'].sum(), 2),
                'trades': int(day_df['trades'].sum()),
                'active_symbols': len(valid_close)
            })
        
        if daily_list:
            daily_df = pd.DataFrame(daily_list).sort_values('date')
            daily_df['change'] = daily_df['close'].diff().round(2)
            
            # RSI for daily (14 period)
            daily_df['rsi'] = calculate_rsi(daily_df['close'], period=14)
            
            daily_dfs[sector] = daily_df
            
            # Save daily
            filename = f"{sector.replace(' ', '_').replace('/', '_').replace('&', 'and').lower()}_daily.csv"
            filepath = os.path.join(OUTPUT_DAILY_DIR, filename)
            cols = ['sector', 'date', 'open', 'high', 'low', 'close', 'volume', 'value', 'trades', 'change', 'rsi', 'active_symbols']
            daily_df[cols].to_csv(filepath, index=False)
            
            print(f"  ✓ Daily {sector}: {len(daily_df)} trading days")
    
    # 4. উইকলি সেক্টর ক্যান্ডেল (Equal Weighted + RSI 14)
    print("\n3. উইকলি সেক্টর ক্যান্ডেল তৈরি হচ্ছে (DSE Calendar + RSI 14)...")
    
    for sector, daily_df in daily_dfs.items():
        daily_df['week_start'] = daily_df['date'].apply(get_dse_week_start)
        
        weekly_list = []
        
        for week_start, week_df in daily_df.groupby('week_start'):
            if len(week_df) == 0:
                continue
            
            week_dates = sorted(week_df['date'].unique())
            first_day_data = week_df[week_df['date'] == week_dates[0]]
            last_day_data = week_df[week_df['date'] == week_dates[-1]]
            
            weekly_list.append({
                'sector': sector,
                'week_start': week_start,
                'week_end_date': week_dates[-1],
                'open': round(first_day_data['open'].iloc[0], 2),
                'high': round(week_df['high'].max(), 2),
                'low': round(week_df['low'].min(), 2),
                'close': round(last_day_data['close'].iloc[-1], 2),
                'volume': int(week_df['volume'].sum()),
                'value': round(week_df['value'].sum(), 2),
                'trades': int(week_df['trades'].sum()),
                'trading_days': len(week_dates),
                'symbols_count': int(daily_df['active_symbols'].iloc[-1]) if len(daily_df) > 0 else 0
            })
        
        if weekly_list:
            weekly_df = pd.DataFrame(weekly_list).sort_values('week_start')
            weekly_df['change'] = weekly_df['close'].diff().round(2)
            
            # RSI for weekly (14 period - Wilder's Smoothing)
            weekly_df['rsi'] = calculate_rsi(weekly_df['close'], period=14)
            
            # Save weekly
            filename = f"{sector.replace(' ', '_').replace('/', '_').replace('&', 'and').lower()}_weekly.csv"
            filepath = os.path.join(OUTPUT_WEEKLY_DIR, filename)
            cols = ['sector', 'week_start', 'week_end_date', 'open', 'high', 'low', 'close', 
                   'volume', 'value', 'trades', 'change', 'rsi', 'trading_days', 'symbols_count']
            weekly_df[cols].to_csv(filepath, index=False)
            
            # Latest candle info
            latest = weekly_df.iloc[-1]
            rsi_val = f"{latest['rsi']:.2f}" if pd.notna(latest['rsi']) else 'N/A'
            
            # RSI status
            rsi_status = ""
            if pd.notna(latest['rsi']):
                if latest['rsi'] > 70:
                    rsi_status = "⚠️ Overbought"
                elif latest['rsi'] < 30:
                    rsi_status = "★ Oversold"
                else:
                    rsi_status = "Neutral"
            
            print(f"  ✓ {sector}: {len(weekly_df)} weeks | "
                  f"O:{latest['open']:.2f} H:{latest['high']:.2f} L:{latest['low']:.2f} C:{latest['close']:.2f} | "
                  f"Ch:{latest['change']:+.2f} | RSI:{rsi_val} {rsi_status}")
    
    # 5. Summary
    print(f"\n{'='*70}")
    print("✅ সম্পন্ন!")
    print(f"{'='*70}")
    print(f"📁 ডেইলি: {OUTPUT_DAILY_DIR}")
    print(f"📁 উইকলি: {OUTPUT_WEEKLY_DIR}")
    print(f"\n📋 ডেইলি CSV কলাম:")
    print(f"   sector, date, open, high, low, close, volume, value, trades, change, rsi, active_symbols")
    print(f"\n📋 উইকলি CSV কলাম:")
    print(f"   sector, week_start, week_end_date, open, high, low, close, volume, value, trades, change, rsi, trading_days, symbols_count")
    
    # RSI Summary
    print(f"\n📊 RSI Summary (Weekly, 14 period):")
    for sector in sorted(daily_dfs.keys()):
        filepath = os.path.join(OUTPUT_WEEKLY_DIR, f"{sector.replace(' ', '_').replace('/', '_').replace('&', 'and').lower()}_weekly.csv")
        if os.path.exists(filepath):
            wdf = pd.read_csv(filepath)
            valid_rsi = wdf['rsi'].dropna()
            if len(valid_rsi) > 0:
                last_rsi = valid_rsi.iloc[-1]
                signal = "🔴" if last_rsi > 70 else "🟢" if last_rsi < 30 else "🟡"
                print(f"   {signal} {sector:<25}: RSI {last_rsi:.2f} ({len(wdf)} weeks, {len(valid_rsi)} RSI values)")

if __name__ == "__main__":
    # পুরনো ট্র্যাকার ডিলিট (সম্পূর্ণ রি-জেনারেট)
    tracker = './csv/sector/processed_dates.txt'
    if os.path.exists(tracker):
        os.remove(tracker)
        print("🗑️ পুরনো ট্র্যাকার ডিলিট করা হয়েছে - সম্পূর্ণ ডাটা রি-জেনারেট হবে\n")
    
    calculate_sector_index()
