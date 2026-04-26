import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# কনফিগারেশন
INPUT_CSV = './csv/mongodb.csv'
OUTPUT_DAILY_DIR = './csv/sector/daily/'
OUTPUT_WEEKLY_DIR = './csv/sector/weekly/'
PROCESSED_TRACKER = './csv/sector/processed_dates.txt'

# আউটপুট ডিরেক্টরি তৈরি
os.makedirs(OUTPUT_DAILY_DIR, exist_ok=True)
os.makedirs(OUTPUT_WEEKLY_DIR, exist_ok=True)

def clean_sector_name(sector):
    """সেক্টর নাম ক্লিন এবং সেফ করে"""
    if pd.isna(sector) or sector == '' or sector is None:
        return 'Unknown'
    return str(sector)

def safe_filename(sector):
    """ফাইল নামের জন্য সেফ স্ট্রিং তৈরি"""
    name = clean_sector_name(sector)
    return name.replace(' ', '_').replace('/', '_').replace('&', 'and').replace('(', '').replace(')', '').lower()

def calculate_rsi(close_prices, period=14):
    """RSI (Relative Strength Index) হিসাব করে"""
    if len(close_prices) < period + 1:
        return [None] * len(close_prices)
    
    deltas = close_prices.diff()
    
    gains = deltas.copy()
    losses = deltas.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    avg_gain = gains.iloc[:period].mean()
    avg_loss = losses.iloc[:period].mean()
    
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

def calculate_rsi_with_history(existing_closes, new_closes, period=14):
    """হিস্ট্রি ডাটা সহ RSI হিসাব করে"""
    all_closes = pd.concat([existing_closes, new_closes])
    all_rsi = calculate_rsi(all_closes, period)
    return all_rsi[-len(new_closes):]

def load_processed_dates():
    """কোন তারিখ পর্যন্ত প্রসেস করা হয়েছে ট্র্যাক করে"""
    if os.path.exists(PROCESSED_TRACKER):
        with open(PROCESSED_TRACKER, 'r') as f:
            dates = f.read().strip().split(',')
            return set(pd.to_datetime([d for d in dates if d]))
    return set()

def save_processed_dates(dates):
    """প্রসেস করা তারিখগুলো সেভ করে"""
    with open(PROCESSED_TRACKER, 'w') as f:
        f.write(','.join(d.strftime('%Y-%m-%d') for d in sorted(dates)))

def load_existing_sector_data(sector, period='daily'):
    """আগে থেকে থাকা সেক্টর ডাটা লোড করে"""
    filename = f"{safe_filename(sector)}_{period}.csv"
    filepath = os.path.join(OUTPUT_DAILY_DIR if period == 'daily' else OUTPUT_WEEKLY_DIR, filename)
    
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            if len(df) == 0:
                return pd.DataFrame()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            if 'week_end_date' in df.columns:
                df['week_end_date'] = pd.to_datetime(df['week_end_date'])
            return df
        except Exception as e:
            print(f"  ⚠ Error loading {filepath}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def save_sector_data(sector, df, period='daily'):
    """সেক্টর ডাটা সেভ করে"""
    filename = f"{safe_filename(sector)}_{period}.csv"
    filepath = os.path.join(OUTPUT_DAILY_DIR if period == 'daily' else OUTPUT_WEEKLY_DIR, filename)
    
    existing_df = load_existing_sector_data(sector, period)
    
    if len(existing_df) > 0:
        combined = pd.concat([existing_df, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep='last')
        combined = combined.sort_values('date').reset_index(drop=True)
    else:
        combined = df.sort_values('date').reset_index(drop=True)
    
    # Select appropriate columns
    if period == 'daily':
        columns = ['sector', 'date', 'open', 'close', 'high', 'low', 'volume', 'value', 'trades', 'change', 'rsi']
    else:
        columns = ['sector', 'date', 'week_end_date', 'dse_year', 'dse_week_num', 'trading_days', 
                  'open', 'close', 'high', 'low', 'volume', 'value', 'trades', 'change', 'rsi']
    
    available_columns = [col for col in columns if col in combined.columns]
    combined[available_columns].to_csv(filepath, index=False)
    
    return combined

def aggregate_daily_sector(df_new, processed_dates):
    """নতুন ডাটা থেকে ডেইলি সেক্টর ক্যান্ডেল তৈরি"""
    new_dates = set(df_new['date'].unique()) - processed_dates
    
    if not new_dates:
        print("  -> নতুন কোনো ডেইলি ডাটা নেই")
        return processed_dates
    
    df_new_dates = df_new[df_new['date'].isin(new_dates)]
    
    unique_sectors = df_new_dates['sector'].dropna().unique()
    
    for sector in unique_sectors:
        if pd.isna(sector) or sector == '':
            continue
            
        sector_df = df_new_dates[df_new_dates['sector'] == sector]
        
        # Date wise aggregate
        daily_agg = sector_df.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'value': 'sum',
            'trades': 'sum'
        }).reset_index()
        
        daily_agg['sector'] = sector
        daily_agg = daily_agg.sort_values('date')
        
        # RSI হিসাব
        existing = load_existing_sector_data(sector, 'daily')
        
        if len(existing) > 0 and 'close' in existing.columns:
            historical_closes = existing['close'].tail(14)
            new_closes = daily_agg['close']
            rsi_values = calculate_rsi_with_history(historical_closes, new_closes, period=14)
        else:
            rsi_values = calculate_rsi(daily_agg['close'], period=14)
        
        daily_agg['rsi'] = rsi_values
        
        # Change হিসাব
        if len(existing) > 0 and 'close' in existing.columns:
            all_closes = pd.concat([existing['close'], daily_agg['close']])
            changes = all_closes.diff().iloc[-len(daily_agg):].fillna(0)
        else:
            changes = daily_agg['close'].diff().fillna(0)
        
        daily_agg['change'] = changes.values
        
        # সেভ করুন
        save_sector_data(sector, daily_agg, 'daily')
        
        # RSI স্ট্যাটিসটিক্স
        valid_rsi = daily_agg['rsi'].dropna()
        rsi_info = ""
        if len(valid_rsi) > 0:
            rsi_info = f" | RSI: {valid_rsi.iloc[-1]:.1f}"
        
        print(f"  ✓ {sector}: {len(daily_agg)} নতুন ডেইলি রো{rsi_info}")
    
    processed_dates.update(new_dates)
    save_processed_dates(processed_dates)
    
    return processed_dates

def aggregate_weekly_sector(df_new, processed_weekly_dates):
    """নতুন ডাটা থেকে DSE উইকলি সেক্টর ক্যান্ডেল তৈরি"""
    df_new = df_new.copy()
    df_new['weekday'] = (df_new['date'].dt.dayofweek + 1) % 7
    
    def get_dse_week_start(date):
        weekday = (date.dayofweek + 1) % 7
        if weekday == 0:
            return date
        elif weekday <= 4:
            return date - timedelta(days=weekday)
        else:
            return date - timedelta(days=weekday) + timedelta(days=7)
    
    df_new['week_start'] = df_new['date'].apply(get_dse_week_start)
    new_weeks = set(df_new['week_start'].unique()) - processed_weekly_dates
    
    if not new_weeks:
        print("  -> নতুন কোনো উইকলি ডাটা নেই")
        return processed_weekly_dates
    
    unique_sectors = df_new['sector'].dropna().unique()
    
    for sector in unique_sectors:
        if pd.isna(sector) or sector == '':
            continue
            
        sector_data = df_new[df_new['sector'] == sector]
        
        weekly_list = []
        for week_start in sorted(new_weeks):
            week_data = sector_data[sector_data['week_start'] == week_start]
            
            if len(week_data) == 0:
                continue
            
            trading_days = len(week_data[week_data['weekday'].between(0, 4)])
            
            candle = {
                'sector': sector,
                'date': week_start,
                'week_end_date': week_start + timedelta(days=4),
                'dse_year': week_start.year,
                'dse_week_num': int(week_start.strftime('%U')) + 1,
                'trading_days': trading_days,
                'open': week_data[week_data['date'] == week_data['date'].min()]['open'].iloc[0],
                'high': week_data['high'].max(),
                'low': week_data['low'].min(),
                'close': week_data[week_data['date'] == week_data['date'].max()]['close'].iloc[-1],
                'volume': week_data['volume'].sum(),
                'value': week_data['value'].sum(),
                'trades': week_data['trades'].sum()
            }
            weekly_list.append(candle)
        
        if weekly_list:
            new_weekly_df = pd.DataFrame(weekly_list).sort_values('date')
            
            # Weekly RSI
            existing = load_existing_sector_data(sector, 'weekly')
            
            if len(existing) > 0 and 'close' in existing.columns:
                historical_closes = existing['close'].tail(14)
                new_closes = new_weekly_df['close']
                rsi_values = calculate_rsi_with_history(historical_closes, new_closes, period=14)
            else:
                rsi_values = calculate_rsi(new_weekly_df['close'], period=14)
            
            new_weekly_df['rsi'] = rsi_values
            
            # Weekly change
            if len(existing) > 0 and 'close' in existing.columns:
                all_closes = pd.concat([existing['close'], new_weekly_df['close']])
                changes = all_closes.diff().iloc[-len(new_weekly_df):].fillna(0)
            else:
                changes = new_weekly_df['close'].diff().fillna(0)
            
            new_weekly_df['change'] = changes.values
            
            # সেভ করুন
            save_sector_data(sector, new_weekly_df, 'weekly')
            
            # RSI স্ট্যাটাস
            valid_rsi = new_weekly_df['rsi'].dropna()
            rsi_info = ""
            if len(valid_rsi) > 0:
                last_rsi = valid_rsi.iloc[-1]
                if last_rsi > 70:
                    rsi_info = f" | RSI: {last_rsi:.1f} ⚠"
                elif last_rsi < 30:
                    rsi_info = f" | RSI: {last_rsi:.1f} ★"
                else:
                    rsi_info = f" | RSI: {last_rsi:.1f}"
            
            print(f"  ✓ {sector}: {len(new_weekly_df)} নতুন উইকলি রো{rsi_info}")
    
    processed_weekly_dates.update(new_weeks)
    
    return processed_weekly_dates

# ==== MAIN EXECUTION ====
print("=" * 60)
print("সেক্টর ক্যান্ডেল জেনারেটর (RSI + ইনক্রিমেন্টাল)")
print("=" * 60)

# ডাটা লোড
print("\n1. ডাটা লোড হচ্ছে...")
df = pd.read_csv(INPUT_CSV)
df['date'] = pd.to_datetime(df['date'])

# sector NaN fix - ফাঁকা sector গুলো 'Unknown' হিসেবে সেট
df['sector'] = df['sector'].apply(clean_sector_name)

print(f"   মোট রো: {len(df):,}")
print(f"   সিম্বল: {df['symbol'].nunique():,}")
print(f"   সেক্টর: {df['sector'].nunique():,}")
print(f"   তারিখ: {df['date'].min().strftime('%Y-%m-%d')} থেকে {df['date'].max().strftime('%Y-%m-%d')}")

# Sector distribution
print("\n   সেক্টর ডিস্ট্রিবিউশন:")
for sector, count in df['sector'].value_counts().items():
    symbols = df[df['sector'] == sector]['symbol'].nunique()
    print(f"     {sector}: {count:,} rows, {symbols} symbols")

# প্রসেসড ডাটা ট্র্যাক
processed_dates = load_processed_dates()
processed_weekly_dates = set()

print(f"\n   পূর্বে প্রসেসড দিন: {len(processed_dates)}")

# ডেইলি সেক্টর ক্যান্ডেল
print("\n2. ডেইলি সেক্টর ক্যান্ডেল (RSI 14)...")
processed_dates = aggregate_daily_sector(df, processed_dates)

# উইকলি সেক্টর ক্যান্ডেল
print("\n3. উইকলি সেক্টর ক্যান্ডেল (DSE Calendar + RSI 14)...")
processed_weekly_dates = aggregate_weekly_sector(df, processed_weekly_dates)

# Statistics
print("\n" + "=" * 60)
print("প্রসেসিং সম্পন্ন!")
print("=" * 60)

# সেক্টর ওভারভিউ
print("\n4. সেক্টর RSI ওভারভিউ:")
print("-" * 65)
print(f"{'সেক্টর':<20} {'ডেইলি':>6} {'উইকলি':>6} {'RSI(D)':>8} {'RSI(W)':>8} {'স্ট্যাটাস':>12}")
print("-" * 65)

overbought = []
oversold = []

for sector in sorted(df['sector'].unique()):
    daily = load_existing_sector_data(sector, 'daily')
    weekly = load_existing_sector_data(sector, 'weekly')
    
    d_count = len(daily) if len(daily) > 0 else 0
    w_count = len(weekly) if len(weekly) > 0 else 0
    
    daily_rsi = None
    weekly_rsi = None
    
    if d_count > 0 and 'rsi' in daily.columns:
        valid = daily['rsi'].dropna()
        if len(valid) > 0:
            daily_rsi = valid.iloc[-1]
    
    if w_count > 0 and 'rsi' in weekly.columns:
        valid = weekly['rsi'].dropna()
        if len(valid) > 0:
            weekly_rsi = valid.iloc[-1]
    
    status = "Neutral"
    if weekly_rsi and weekly_rsi > 70:
        status = "⚠ Overbought"
        overbought.append((sector, weekly_rsi))
    elif weekly_rsi and weekly_rsi < 30:
        status = "★ Oversold"
        oversold.append((sector, weekly_rsi))
    
    daily_rsi_str = f"{daily_rsi:.1f}" if daily_rsi else "N/A"
    weekly_rsi_str = f"{weekly_rsi:.1f}" if weekly_rsi else "N/A"
    
    print(f"{sector:<20} {d_count:>6} {w_count:>6} {daily_rsi_str:>8} {weekly_rsi_str:>8} {status:>12}")

if overbought:
    print(f"\n⚠ Overbought সেক্টর (RSI > 70):")
    for s, r in sorted(overbought, key=lambda x: x[1], reverse=True):
        print(f"   {s}: {r:.1f}")

if oversold:
    print(f"\n★ Oversold সেক্টর (RSI < 30):")
    for s, r in sorted(oversold, key=lambda x: x[1]):
        print(f"   {s}: {r:.1f}")

print(f"\n📁 আউটপুট:")
print(f"   ডেইলি: {OUTPUT_DAILY_DIR}")
print(f"   উইকলি: {OUTPUT_WEEKLY_DIR}")

print("\n📋 CSV কলাম:")
print("   ডেইলি: sector, date, open, close, high, low, volume, value, trades, change, rsi")
print("   উইকলি: sector, date, week_end_date, dse_year, dse_week_num, trading_days, open, close, high, low, volume, value, trades, change, rsi")