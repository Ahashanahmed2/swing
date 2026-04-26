# sector_candle.csv
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import glob

# কনফিগারেশন
INPUT_CSV = './csv/mongodb.csv'
OUTPUT_DAILY_DIR = './csv/sector/daily/'
OUTPUT_WEEKLY_DIR = './csv/sector/weekly/'
PROCESSED_TRACKER = './csv/sector/processed_dates.txt'

# আউটপুট ডিরেক্টরি তৈরি
os.makedirs(OUTPUT_DAILY_DIR, exist_ok=True)
os.makedirs(OUTPUT_WEEKLY_DIR, exist_ok=True)

def load_processed_dates():
    """কোন তারিখ পর্যন্ত প্রসেস করা হয়েছে ট্র্যাক করে"""
    if os.path.exists(PROCESSED_TRACKER):
        with open(PROCESSED_TRACKER, 'r') as f:
            dates = f.read().strip().split(',')
            return set(pd.to_datetime(dates))
    return set()

def save_processed_dates(dates):
    """প্রসেস করা তারিখগুলো সেভ করে"""
    with open(PROCESSED_TRACKER, 'w') as f:
        f.write(','.join(d.strftime('%Y-%m-%d') for d in sorted(dates)))

def load_existing_sector_data(sector, period='daily'):
    """আগে থেকে থাকা সেক্টর ডাটা লোড করে"""
    safe_name = sector.replace(' ', '_').replace('/', '_').replace('&', 'and').lower()
    filepath = os.path.join(OUTPUT_DAILY_DIR if period == 'daily' else OUTPUT_WEEKLY_DIR, 
                            f"{safe_name}_{period}.csv")
    if os.path.exists(filepath):
        return pd.read_csv(filepath, parse_dates=['date', 'week_end_date'] if period == 'weekly' else ['date'])
    return pd.DataFrame()

def save_sector_data(sector, df, period='daily'):
    """সেক্টর ডাটা সেভ করে (existing data-তে নতুন data অ্যাপেন্ড/মার্জ করে)"""
    safe_name = sector.replace(' ', '_').replace('/', '_').replace('&', 'and').lower()
    filepath = os.path.join(OUTPUT_DAILY_DIR if period == 'daily' else OUTPUT_WEEKLY_DIR, 
                            f"{safe_name}_{period}.csv")
    
    existing_df = load_existing_sector_data(sector, period)
    
    if len(existing_df) > 0:
        # Combine existing and new data
        combined = pd.concat([existing_df, df], ignore_index=True)
        
        if period == 'daily':
            # Remove duplicate dates, keep last entry (newest)
            combined = combined.drop_duplicates(subset=['date'], keep='last')
        else:  # weekly
            combined = combined.drop_duplicates(subset=['date'], keep='last')
        
        # Sort by date
        combined = combined.sort_values('date').reset_index(drop=True)
    else:
        combined = df.sort_values('date').reset_index(drop=True)
    
    # Select appropriate columns
    if period == 'daily':
        columns = ['sector', 'date', 'open', 'close', 'high', 'low', 'volume', 'value', 'trades', 'change', 'active_symbols']
    else:
        columns = ['sector', 'date', 'week_end_date', 'dse_year', 'dse_week_num', 'trading_days', 'active_symbols',
                  'open', 'close', 'high', 'low', 'volume', 'value', 'trades', 'change']
    
    available_columns = [col for col in columns if col in combined.columns]
    combined[available_columns].to_csv(filepath, index=False)
    
    return combined

def aggregate_daily_sector(df_new, processed_dates):
    """নতুন ডাটা থেকে ডেইলি সেক্টর ক্যান্ডেল তৈরি"""
    # শুধু নতুন তারিখের ডাটা নিন
    new_dates = set(df_new['date'].unique()) - processed_dates
    
    if not new_dates:
        print("  -> নতুন কোনো ডেইলি ডাটা নেই")
        return {}, processed_dates
    
    df_new_dates = df_new[df_new['date'].isin(new_dates)]
    
    sector_candles = {}
    unique_sectors = df_new_dates['sector'].unique()
    
    for sector in unique_sectors:
        sector_df = df_new_dates[df_new_dates['sector'] == sector]
        
        # প্রতিটি তারিখের জন্য aggregation
        for date in sorted(sector_df['date'].unique()):
            day_data = sector_df[sector_df['date'] == date]
            active_symbols = day_data['symbol'].nunique()
            
            candle = {
                'sector': sector,
                'date': date,
                'open': day_data['open'].iloc[0],  # প্রথম active symbol-এর open
                'high': day_data['high'].max(),
                'low': day_data['low'].min(),
                'close': day_data['close'].iloc[-1],  # শেষ active symbol-এর close
                'volume': day_data['volume'].sum(),
                'value': day_data['value'].sum(),
                'trades': day_data['trades'].sum(),
                'active_symbols': active_symbols
            }
            
            if sector not in sector_candles:
                sector_candles[sector] = []
            sector_candles[sector].append(candle)
    
    # Change calculate করুন (আগের ডাটার সাথে মিলিয়ে)
    for sector in list(sector_candles.keys()):
        sector_df = pd.DataFrame(sector_candles[sector])
        sector_df = sector_df.sort_values('date')
        
        # আগের close value নিন existing data থেকে
        existing = load_existing_sector_data(sector, 'daily')
        
        if len(existing) > 0:
            last_close = existing['close'].iloc[-1]
            all_dates = list(existing['date']) + list(sector_df['date'])
            all_closes = list(existing['close']) + list(sector_df['close'])
        else:
            last_close = None
            all_dates = list(sector_df['date'])
            all_closes = list(sector_df['close'])
        
        # Change calculate
        changes = []
        for i, close_val in enumerate(all_closes):
            if i == 0:
                if last_close is not None:
                    prev_close = last_close
                else:
                    changes.append(0)
                    continue
            else:
                prev_close = all_closes[i-1]
            changes.append(close_val - prev_close)
        
        # শুধু নতুন এন্ট্রিগুলোর change assign
        sector_df['change'] = changes[-len(sector_df):] if len(changes) >= len(sector_df) else changes + [0] * (len(sector_df) - len(changes))
        sector_candles[sector] = sector_df.to_dict('records')
    
    # প্রতিটি সেক্টর সেভ করুন
    for sector in sector_candles:
        sector_df = pd.DataFrame(sector_candles[sector])
        save_sector_data(sector, sector_df, 'daily')
        print(f"  ✓ {sector}: {len(sector_df)} নতুন ডেইলি রো যোগ হয়েছে")
    
    # প্রসেস করা তারিখ আপডেট
    processed_dates.update(new_dates)
    save_processed_dates(processed_dates)
    
    return sector_candles, processed_dates

def aggregate_weekly_sector(df_new, processed_weekly_dates):
    """নতুন ডাটা থেকে DSE উইকলি সেক্টর ক্যান্ডেল তৈরি"""
    # শুধু new weeks identify
    df_new['weekday'] = (df_new['date'].dt.dayofweek + 1) % 7
    
    def get_dse_week_start(date):
        weekday = (date.dayofweek + 1) % 7
        if weekday == 0:  # রবিবার
            return date
        elif weekday <= 4:  # সোম-বৃহস্পতি
            return date - timedelta(days=weekday)
        else:  # শুক্র-শনি
            return date - timedelta(days=weekday) + timedelta(days=7)
    
    df_new['week_start'] = df_new['date'].apply(get_dse_week_start)
    new_weeks = set(df_new['week_start'].unique()) - processed_weekly_dates
    
    if not new_weeks:
        print("  -> নতুন কোনো উইকলি ডাটা নেই")
        return {}, processed_weekly_dates
    
    # সকল সেক্টরের aggregations
    sector_weekly_candles = {}
    
    for sector in df_new['sector'].unique():
        sector_data = df_new[df_new['sector'] == sector]
        
        for week_start in sorted(new_weeks):
            week_data = sector_data[sector_data['week_start'] == week_start]
            
            if len(week_data) == 0:
                continue
            
            # DSE ট্রেডিং ডে count (রবি-বৃহস্পতি)
            trading_days = len(week_data[week_data['weekday'].between(0, 4)])
            active_symbols = week_data['symbol'].nunique()
            
            candle = {
                'sector': sector,
                'date': week_start,
                'week_end_date': week_start + timedelta(days=4),
                'dse_year': week_start.year,
                'dse_week_num': int(week_start.strftime('%U')) + 1,
                'trading_days': trading_days,
                'active_symbols': active_symbols,
                'open': week_data[week_data['date'] == week_data['date'].min()]['open'].iloc[0],
                'high': week_data['high'].max(),
                'low': week_data['low'].min(),
                'close': week_data[week_data['date'] == week_data['date'].max()]['close'].iloc[-1],
                'volume': week_data['volume'].sum(),
                'value': week_data['value'].sum(),
                'trades': week_data['trades'].sum()
            }
            
            if sector not in sector_weekly_candles:
                sector_weekly_candles[sector] = []
            sector_weekly_candles[sector].append(candle)
    
    # Calculate changes for weekly
    for sector in sector_weekly_candles:
        sector_df = pd.DataFrame(sector_weekly_candles[sector]).sort_values('date')
        
        existing = load_existing_sector_data(sector, 'weekly')
        
        if len(existing) > 0:
            last_close = existing['close'].iloc[-1]
            all_closes = list(existing['close']) + list(sector_df['close'])
        else:
            last_close = None
            all_closes = list(sector_df['close'])
        
        changes = []
        for i, close_val in enumerate(all_closes):
            if i == 0:
                changes.append(0)
            else:
                changes.append(close_val - all_closes[i-1])
        
        sector_df['change'] = changes[-len(sector_df):] if len(changes) >= len(sector_df) else changes + [0] * (len(sector_df) - len(changes))
        
        save_sector_data(sector, sector_df, 'weekly')
        print(f"  ✓ {sector}: {len(sector_df)} নতুন উইকলি রো যোগ হয়েছে")
    
    processed_weekly_dates.update(new_weeks)
    
    return sector_weekly_candles, processed_weekly_dates

# ==== MAIN EXECUTION ====
print("=" * 60)
print("সেক্টর ক্যান্ডেল জেনারেটর (ইনক্রিমেন্টাল মোড)")
print("=" * 60)

# ডাটা লোড
print("\n1. ডাটা লোড হচ্ছে...")
df = pd.read_csv(INPUT_CSV)
df['date'] = pd.to_datetime(df['date'])
print(f"   মোট রো: {len(df)}")
print(f"   সিম্বল: {df['symbol'].nunique()}")
print(f"   সেক্টর: {df['sector'].nunique()}")
print(f"   তারিখ রেঞ্জ: {df['date'].min()} থেকে {df['date'].max()}")

# প্রসেসড ডাটা ট্র্যাক
processed_dates = load_processed_dates()
processed_weekly_dates = set()  # Track করে processed weekly start dates

# ডেইলি সেক্টর ক্যান্ডেল
print("\n2. ডেইলি সেক্টর ক্যান্ডেল প্রসেসিং...")
daily_candles, processed_dates = aggregate_daily_sector(df, processed_dates)

# উইকলি সেক্টর ক্যান্ডেল
print("\n3. উইকলি সেক্টর ক্যান্ডেল প্রসেসিং (DSE Calendar)...")
weekly_candles, processed_weekly_dates = aggregate_weekly_sector(df, processed_weekly_dates)

# Statistics
print("\n" + "=" * 60)
print("প্রসেসিং সম্পন্ন!")
print("=" * 60)
print(f"নতুন প্রসেসড ডেইলি তারিখ: {len(processed_dates)}")
print(f"নতুন প্রসেসড উইকলি: {len(processed_weekly_dates)}")

# উদাহরণ দেখান
print("\n4. উদাহরণ আউটপুট (প্রথম সেক্টর):")
for sector in sorted(df['sector'].unique())[:3]:
    daily = load_existing_sector_data(sector, 'daily')
    weekly = load_existing_sector_data(sector, 'weekly')
    print(f"\n  {sector}:")
    print(f"    ডেইলি: {len(daily)} রো")
    if len(daily) > 0:
        print(f"    শেষ ডেইলি: {daily['date'].iloc[-1].strftime('%Y-%m-%d')} | Close: {daily['close'].iloc[-1]}")
    print(f"    উইকলি: {len(weekly)} রো")
    if len(weekly) > 0:
        print(f"    শেষ উইকলি: {weekly['date'].iloc[-1].strftime('%Y-%m-%d')} | Close: {weekly['close'].iloc[-1]}")