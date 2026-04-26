import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# ফাইল পাথ এবং আউটপুট ডিরেক্টরি সেটআপ
input_file = './csv/mongodb.csv'
output_daily_dir = './csv/sector/daily/'
output_weekly_dir = './csv/sector/weekly/'

# আউটপুট ডিরেক্টরি তৈরি
os.makedirs(output_daily_dir, exist_ok=True)
os.makedirs(output_weekly_dir, exist_ok=True)

# ডাটা লোড
print("CSV ফাইল লোড হচ্ছে...")
df = pd.read_csv(input_file)
df['date'] = pd.to_datetime(df['date'])

# প্রতিটি সিম্বলের লেটেস্ট রো বের করা
latest_rows = df.sort_values('date').groupby('symbol').last().reset_index()

# সেক্টর-ভিত্তিক aggregation ফাংশন
def aggregate_sector_candles(grouped_df):
    """
    সেক্টর অনুযায়ী গ্রুপ করা ডেটা থেকে ক্যান্ডেল তৈরি করে
    """
    agg_dict = {
        'open': 'first',      # প্রথম symbol এর open
        'high': 'max',        # সর্বোচ্চ high
        'low': 'min',         # সর্বনিম্ন low
        'close': 'last',      # শেষ symbol এর close
        'volume': 'sum',      # মোট ভলিউম
        'value': 'sum',       # মোট ভ্যালু
        'trades': 'sum',      # মোট ট্রেড
    }
    
    # Group by sector and date
    sector_candles = grouped_df.groupby(['sector', 'date']).agg(agg_dict).reset_index()
    
    # Change calculation (close - previous close)
    sector_candles['change'] = sector_candles.groupby('sector')['close'].diff()
    
    # প্রথম রো-এর জন্য change 0 সেট করা
    sector_candles['change'] = sector_candles['change'].fillna(0)
    
    return sector_candles

def create_dse_weekly_candles(daily_sector_df):
    """
    DSE ক্যালেন্ডার অনুযায়ী উইকলি ক্যান্ডেল তৈরি করে
    সপ্তাহ: রবিবার (0) থেকে বৃহস্পতিবার (4)
    শুক্রবার (5) ও শনিবার (6) মার্কেট বন্ধ
    """
    daily_df = daily_sector_df.copy()
    
    # DSE সপ্তাহ শনাক্ত করার জন্য কাস্টম logic
    # date-কে DSE week অনুযায়ী গ্রুপ করা
    
    # প্রথমে সপ্তাহের দিন বের করি (0=রবি, 1=সোম, ..., 4=বৃহস্পতি, 5=শুক্র, 6=শনি)
    daily_df['weekday'] = (daily_df['date'].dt.dayofweek + 1) % 7  # DSE: 0=রবি, 1=সোম, ..., 4=বৃহস্পতি, 5=শুক্র, 6=শনি
    
    # DSE উইক আইডেন্টিফায়ার তৈরি - প্রতিটি সপ্তাহ রবিবার শুরু হয়
    # রবিবারকে week_start হিসেবে ধরে সপ্তাহ গ্রুপিং
    
    def get_dse_week_info(date_series):
        """DSE সপ্তাহের start date এবং week number বের করা"""
        week_starts = []
        week_numbers = []
        
        for date in date_series:
            weekday = (date.dayofweek + 1) % 7  # DSE weekday
            
            if weekday == 0:  # রবিবার
                week_start = date
            elif weekday <= 4:  # সোম-বৃহস্পতি
                week_start = date - timedelta(days=weekday)
            else:  # শুক্র-শনি (মার্কেট বন্ধের দিন, এটা হওয়ার কথা না)
                week_start = date - timedelta(days=weekday) + timedelta(days=7)
            
            week_starts.append(week_start)
            
            # ISO year-week এর মতো DSE year-week তৈরি
            year = week_start.year
            # বছরের প্রথম DSE সপ্তাহ হিসাব
            year_start = datetime(year, 1, 1)
            year_start_weekday = (year_start.dayofweek + 1) % 7
            first_sunday = year_start - timedelta(days=year_start_weekday)
            if year_start_weekday != 0:
                first_sunday += timedelta(days=7)
            
            week_num = int((week_start - first_sunday).days / 7) + 1
            if week_num < 1:
                week_num = 1
            
            week_numbers.append(week_num)
        
        return week_starts, week_numbers
    
    daily_df['dse_week_start'], daily_df['dse_week_num'] = get_dse_week_info(daily_df['date'])
    daily_df['dse_year'] = daily_df['dse_week_start'].dt.year
    
    # উইকলি aggregation
    weekly_agg = {
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'value': 'sum',
        'trades': 'sum',
    }
    
    weekly_candles = daily_df.groupby(['sector', 'dse_year', 'dse_week_start', 'dse_week_num']).agg(weekly_agg).reset_index()
    
    # তারিখ হিসেবে week_start ব্যবহার
    weekly_candles['date'] = weekly_candles['dse_week_start']
    
    # Change calculation
    weekly_candles['change'] = weekly_candles.groupby('sector')['close'].diff()
    weekly_candles['change'] = weekly_candles['change'].fillna(0)
    
    # সপ্তাহের শেষ দিন যোগ (বৃহস্পতিবার)
    weekly_candles['week_end_date'] = weekly_candles['date'] + timedelta(days=4)
    
    # Select and order relevant columns
    columns_order = ['sector', 'date', 'open', 'close', 'high', 'low', 'volume', 'value', 'trades', 'change', 
                    'week_end_date', 'dse_year', 'dse_week_num']
    weekly_candles = weekly_candles[columns_order]
    
    # Sort by date
    weekly_candles = weekly_candles.sort_values(['sector', 'date'])
    
    # সপ্তাহের মধ্যে কত দিনের ট্রেডিং হয়েছে তা যোগ করা
    trading_days = daily_df[daily_df['weekday'].between(0, 4)].groupby(['sector', 'dse_week_start']).size().reset_index(name='trading_days')
    weekly_candles = weekly_candles.merge(trading_days, left_on=['sector', 'date'], right_on=['sector', 'dse_week_start'], how='left')
    
    # Additional useful info
    weekly_candles['trading_days'] = weekly_candles['trading_days'].fillna(0).astype(int)
    
    # Reorder columns including trading_days
    final_columns = ['sector', 'date', 'week_end_date', 'dse_year', 'dse_week_num', 'trading_days', 
                    'open', 'close', 'high', 'low', 'volume', 'value', 'trades', 'change']
    weekly_candles = weekly_candles[final_columns]
    
    return weekly_candles

# ডেইলি সেক্টর ক্যান্ডেল তৈরি
print("ডেইলি সেক্টর ক্যান্ডেল তৈরি হচ্ছে...")
sector_daily_candles = aggregate_sector_candles(df)

# উইকলি সেক্টর ক্যান্ডেল তৈরি (DSE calendar অনুযায়ী)
print("DSE উইকলি সেক্টর ক্যান্ডেল তৈরি হচ্ছে...")
sector_weekly_candles = create_dse_weekly_candles(sector_daily_candles)

# DSE সপ্তাহের ভ্যালিডেশন দেখানো
print("\n=== DSE উইকলি উদাহরণ (সর্বশেষ 5 সপ্তাহ) ===")
weekly_example = sector_weekly_candles[sector_weekly_candles['sector'] == sector_weekly_candles['sector'].iloc[0]]
if len(weekly_example) > 0:
    print(weekly_example[['sector', 'date', 'week_end_date', 'trading_days', 'open', 'close']].tail())
    print(f"সপ্তাহিক দিন: {weekly_example['date'].dt.day_name().unique()} (এই দিনগুলো সপ্তাহ শুরু)")

# প্রতিটি সেক্টরের জন্য আলাদা CSV ফাইল তৈরি
unique_sectors = df['sector'].unique()
print(f"\nমোট {len(unique_sectors)}টি সেক্টর পাওয়া গেছে")

for sector in unique_sectors:
    # ফাইল নাম তৈরি (স্পেস এবং স্পেশাল ক্যারেক্টার হ্যান্ডলিং)
    safe_sector_name = sector.replace(' ', '_').replace('/', '_').replace('&', 'and').lower()
    
    # ডেইলি ডেটা ফিল্টার এবং সেভ
    sector_daily_data = sector_daily_candles[sector_daily_candles['sector'] == sector]
    if len(sector_daily_data) > 0:
        daily_filename = f"{safe_sector_name}_daily.csv"
        daily_filepath = os.path.join(output_daily_dir, daily_filename)
        
        # ডেইলির জন্য কলাম সিলেক্ট করা
        daily_columns = ['sector', 'date', 'open', 'close', 'high', 'low', 'volume', 'value', 'trades', 'change']
        sector_daily_data[daily_columns].to_csv(daily_filepath, index=False)
        print(f"✓ ডেইলি: {daily_filepath} ({len(sector_daily_data)} ট্রেডিং দিন)")
    
    # উইকলি ডেটা ফিল্টার এবং সেভ
    sector_weekly_data = sector_weekly_candles[sector_weekly_candles['sector'] == sector]
    if len(sector_weekly_data) > 0:
        weekly_filename = f"{safe_sector_name}_weekly.csv"
        weekly_filepath = os.path.join(output_weekly_dir, weekly_filename)
        sector_weekly_data.to_csv(weekly_filepath, index=False)
        print(f"✓ উইকলি: {weekly_filepath} ({len(sector_weekly_data)} সপ্তাহ)")

# সামারি তৈরি
print("\n" + "="*50)
print("প্রসেসিং সম্পন্ন")
print("="*50)
print(f"ইনপুট ফাইল: {input_file}")
print(f"মোট সিম্বল: {df['symbol'].nunique()}")
print(f"মোট সেক্টর: {len(unique_sectors)}")
print(f"মোট ডেইলি ডেটা পয়েন্ট: {len(df)}")
print(f"মোট ডেইলি সেক্টর রো: {len(sector_daily_candles)}")
print(f"মোট উইকলি সেক্টর রো: {len(sector_weekly_candles)}")
print(f"\nআউটপুট ডিরেক্টরি:")
print(f"  ডেইলি: {output_daily_dir}")
print(f"  উইকলি: {output_weekly_dir}")

# সেক্টর সামারি
print("\n=== সেক্টর সামারি (DSE উইকলি) ===")
for sector in sorted(unique_sectors):
    daily_count = len(sector_daily_candles[sector_daily_candles['sector'] == sector])
    weekly_count = len(sector_weekly_candles[sector_weekly_candles['sector'] == sector])
    if daily_count > 0:
        avg_trading_days = sector_weekly_candles[sector_weekly_candles['sector'] == sector]['trading_days'].mean()
        print(f"{sector:20s}: {daily_count:4d} দিন | {weekly_count:3d} সপ্তাহ | গড় ট্রেডিং দিন/সপ্তাহ: {avg_trading_days:.1f}")

# DSE সপ্তাহের স্ট্রাকচার ভেরিফাই করা
print("\n=== DSE সপ্তাহ ভেরিফিকেশন ===")
if len(sector_weekly_candles) > 0:
    sample_week = sector_weekly_candles.iloc[0]
    week_start = pd.to_datetime(sample_week['date'])
    week_end = pd.to_datetime(sample_week['week_end_date'])
    print(f"উদাহরণ সপ্তাহ: {week_start.strftime('%A, %d %B %Y')} থেকে {week_end.strftime('%A, %d %B %Y')}")
    print(f"ট্রেডিং দিন: রবি, সোম, মঙ্গল, বুধ, বৃহস্পতি")
    print(f"বন্ধ: শুক্র, শনি")