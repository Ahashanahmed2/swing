# প্লাটফর্মের মতো সেক্টর ইনডেক্স ক্যান্ডেল জেনারেটর
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# কনফিগারেশন
INPUT_CSV = './csv/mongodb.csv'
OUTPUT_WEEKLY_DIR = './csv/sector/weekly_v2/'  # নতুন ভার্সন

os.makedirs(OUTPUT_WEEKLY_DIR, exist_ok=True)

def calculate_sector_index_weekly():
    """
    প্লাটফর্মের মতো সেক্টর ইনডেক্স ক্যান্ডেল তৈরি
    Market Cap Weighted Index Method
    """
    print("=" * 60)
    print("সেক্টর ইনডেক্স উইকলি ক্যান্ডেল (Market Cap Weighted)")
    print("=" * 60)
    
    # 1. ডাটা লোড
    print("\n1. ডাটা লোড হচ্ছে...")
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    
    # Sector clean
    df['sector'] = df['sector'].fillna('Unknown')
    df['sector'] = df['sector'].apply(lambda x: str(x).strip())
    
    print(f"   মোট রো: {len(df):,}")
    print(f"   সিম্বল: {df['symbol'].nunique():,}")
    print(f"   সেক্টর: {df['sector'].nunique():,}")
    
    # 2. DSE উইক স্টার্ট বের করুন (Sunday start)
    def get_dse_week_start(date):
        """রবিবার থেকে শুরু DSE উইক"""
        weekday = date.dayofweek  # Monday=0, Sunday=6
        # Sunday=6 -> 0, Monday=0 -> 1, ... Thursday=4 -> 5
        if weekday == 6:  # Sunday
            return date
        else:
            return date - timedelta(days=weekday + 1)
    
    df['week_start'] = df['date'].apply(get_dse_week_start)
    
    # 3. প্রতিটি স্টকের weekly মার্কেট ক্যাপ বের করুন
    print("\n2. স্টক লেভেল উইকলি ক্যান্ডেল + Market Cap Weight...")
    
    unique_sectors = df['sector'].unique()
    total_sectors = len(unique_sectors)
    
    for idx, sector in enumerate(unique_sectors, 1):
        sector_df = df[df['sector'] == sector].copy()
        symbols = sector_df['symbol'].unique()
        
        print(f"\n[{idx}/{total_sectors}] {sector} ({len(symbols)} symbols)")
        
        # Weekly সেক্টর ইনডেক্স বানান
        sector_weekly = []
        
        all_weeks = sorted(sector_df['week_start'].unique())
        
        for week_start in all_weeks:
            week_data = sector_df[sector_df['week_start'] == week_start]
            
            if len(week_data) == 0:
                continue
            
            # Get trading days
            week_dates = sorted(week_data['date'].unique())
            first_day = week_dates[0]
            last_day = week_dates[-1]
            trading_days = len(week_dates)
            
            # ============= MARKET CAP WEIGHTED INDEX =============
            # প্লাটফর্মের মতো: Market Cap দিয়ে ওয়েটেড এভারেজ
            
            # প্রথম দিনের ডেটা (Open calculation)
            first_day_data = week_data[week_data['date'] == first_day]
            
            # শেষ দিনের ডেটা (Close calculation)
            last_day_data = week_data[week_data['date'] == last_day]
            
            # Market Cap Weighted Open
            # Open = ∑(marketCap_i × open_i) / ∑(marketCap_i)
            valid_open = first_day_data.dropna(subset=['marketCap', 'open'])
            if len(valid_open) > 0:
                total_mcap_open = valid_open['marketCap'].sum()
                weighted_open = (valid_open['marketCap'] * valid_open['open']).sum() / total_mcap_open
            else:
                weighted_open = np.nan
            
            # Market Cap Weighted Close
            valid_close = last_day_data.dropna(subset=['marketCap', 'close'])
            if len(valid_close) > 0:
                total_mcap_close = valid_close['marketCap'].sum()
                weighted_close = (valid_close['marketCap'] * valid_close['close']).sum() / total_mcap_close
            else:
                weighted_close = np.nan
            
            # High এবং Low: Market Cap Weighted নয়, বরং Index Method
            # High = Index এর সর্বোচ্চ পয়েন্ট
            # Low = Index এর সর্বনিম্ন পয়েন্ট
            
            # প্রতিটি দিনের জন্য ইনডেক্স ভ্যালু ক্যালকুলেট
            daily_index_values = []
            
            for date in week_dates:
                day_data = week_data[week_data['date'] == date]
                valid = day_data.dropna(subset=['marketCap', 'close'])
                
                if len(valid) > 0:
                    total_mcap = valid['marketCap'].sum()
                    weighted_price = (valid['marketCap'] * valid['close']).sum() / total_mcap
                    daily_index_values.append(weighted_price)
            
            if daily_index_values:
                week_high = max(daily_index_values)  # Weekly High Point
                week_low = min(daily_index_values)   # Weekly Low Point
            else:
                week_high = np.nan
                week_low = np.nan
            
            # Volume and Value - Sum (not weighted)
            total_volume = week_data['volume'].sum()
            total_value = week_data['value'].sum()
            total_trades = week_data['trades'].sum()
            
            # Change calculation
            week_end_date = last_day
            
            sector_weekly.append({
                'sector': sector,
                'week_start': week_start,
                'week_end_date': week_end_date,
                'open': round(weighted_open, 2),
                'high': round(week_high, 2),
                'low': round(week_low, 2),
                'close': round(weighted_close, 2),
                'volume': total_volume,
                'value': total_value,
                'trades': total_trades,
                'trading_days': trading_days,
                'symbols_count': len(symbols)
            })
        
        # DataFrame এ কনভার্ট
        weekly_df = pd.DataFrame(sector_weekly)
        
        if len(weekly_df) > 0:
            # Change calculate
            weekly_df = weekly_df.sort_values('week_start')
            weekly_df['change'] = weekly_df['close'].diff().round(2)
            
            # RSI calculate
            weekly_df['rsi'] = calculate_rsi_series(weekly_df['close'], period=14)
            
            # Save
            filename = f"{sector.replace(' ', '_').replace('/', '_').replace('&', 'and').lower()}_weekly_v2.csv"
            filepath = os.path.join(OUTPUT_WEEKLY_DIR, filename)
            
            columns = ['sector', 'week_start', 'week_end_date', 'open', 'high', 'low', 'close', 
                      'volume', 'value', 'trades', 'change', 'rsi', 'trading_days', 'symbols_count']
            
            weekly_df[columns].to_csv(filepath, index=False)
            
            # Latest candle info
            latest = weekly_df.iloc[-1]
            print(f"  ✓ Latest Week: {latest['week_start'].strftime('%Y-%m-%d')} | "
                  f"O:{latest['open']:.2f} H:{latest['high']:.2f} L:{latest['low']:.2f} C:{latest['close']:.2f} | "
                  f"Change:{latest['change']:+.2f} | Vol:{latest['volume']:,.0f}")
    
    print(f"\n✅ Done! Files saved in: {OUTPUT_WEEKLY_DIR}")

def calculate_rsi_series(close_prices, period=14):
    """Calculate RSI for a series"""
    if len(close_prices) < period + 1:
        return [None] * len(close_prices)
    
    deltas = close_prices.diff()
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
    
    for i in range(period + 1, len(close_prices)):
        avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_values.append(round(rsi, 2))
    
    return rsi_values

# Run
calculate_sector_index_weekly()