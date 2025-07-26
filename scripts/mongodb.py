from pymongo import MongoClient
import pandas as pd
import numpy as np
import ta
import os
import sys
from dotenv import load_dotenv
from hf_uploader import download_from_hf_or_run_script

# Step 1: Download CSV from HF if needed
if download_from_hf_or_run_script():
    print(f"HF data download success")
download_from_hf_or_run_script()
# 📁 CSV ফাইল path

# Step 2: CSV Load

csv_path = './csv/mongodb.csv'
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

if os.path.exists(csv_path):
    print("✅ CSV file exists. Checking for new data in MongoDB...")
    try:
        csv_df = pd.read_csv(csv_path)
        csv_df['date'] = pd.to_datetime(csv_df['date'])  # normalize বাদ
        csv_last_date = csv_df['date'].max()
        print(f"🕒 Last date in CSV: {csv_last_date}")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        csv_last_date = None
else:
    print("🆕 CSV file not found. Downloading full data from MongoDB...")
    csv_df = pd.DataFrame()
    csv_last_date = None

# Step 3: Connect MongoDB
load_dotenv()
client = MongoClient(os.getenv("MONGO_URL"))
collection = client["candleData"]["candledatas"]

latest_doc = collection.find_one({}, sort=[('date', -1)])
if not latest_doc:
    print("❌ MongoDB is empty.")
    sys.exit(1)

mongo_last_date = pd.to_datetime(latest_doc['date'])
print(f"🧾 Latest date in MongoDB: {mongo_last_date}")

if csv_last_date and csv_last_date >= mongo_last_date:
    print(f"✅ No new data in MongoDB after {csv_last_date}. Exiting to prevent re-download.")
    sys.exit(1)

# যখন MongoDB তে date ফিল্ডটি string
query = {"date": {"$gt": csv_last_date.strftime('%Y-%m-%d')}} if csv_last_date else {}

print("📥 Downloading data from MongoDB...")
data = list(collection.find(query, {'_id': 0, '__v': 0}))
client.close()

if not data:
    print("✅ No new data found in MongoDB. Exiting.")
    sys.exit(0)

df = pd.DataFrame(data)

# Step 4: Data Cleaning
df.rename(columns={
    'open_price': 'open', 'close_price': 'close',
    'high_price': 'high', 'low_price': 'low',
    'vol': 'volume', 'val': 'value', 'trade_count': 'trades',
    'chg': 'change', 'mcap': 'marketCap'
}, inplace=True)

df['date'] = pd.to_datetime(df['date'], errors='coerce')
numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'value', 'trades', 'change', 'marketCap']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

if 'collectedAt' in df.columns:
    df.drop(columns=['collectedAt'], inplace=True)

# Merge if CSV exists
if not csv_df.empty:
    df = pd.concat([csv_df, df], ignore_index=True).drop_duplicates(['symbol', 'date'], keep='last')
    df.sort_values(['symbol', 'date'], inplace=True)
    print("💾 Appended new data to existing CSV.")

# Step 5: Indicators
def apply_indicators(group):
    bb = ta.volatility.BollingerBands(close=group['close'], window=20)
    group['bb_upper'] = bb.bollinger_hband()
    group['bb_middle'] = bb.bollinger_mavg()
    group['bb_lower'] = bb.bollinger_lband()

    macd = ta.trend.MACD(close=group['close'])
    group['macd'] = macd.macd()
    group['macd_signal'] = macd.macd_signal()
    group['macd_hist'] = macd.macd_diff()

    group['rsi'] = ta.momentum.RSIIndicator(close=group['close']).rsi()
    return group

df = df.groupby('symbol', group_keys=False).apply(apply_indicators)

# Step 6: ZigZag
def compute_zigzag(close, threshold=0.05):
    zz = [np.nan] * len(close)
    last_pivot = close.iloc[0]
    for i in range(1, len(close)):
        change = abs(close.iloc[i] - last_pivot) / last_pivot
        if change >= threshold:
            zz[i] = close.iloc[i]
            last_pivot = close.iloc[i]
    return zz

df['zigzag'] = df.groupby('symbol')['close'].transform(compute_zigzag)

# Step 7: Patterns
def detect_patterns(df):
    def hammer(row):
        return (row['close'] > row['open']) and ((row['low'] < row['open'] - (row['high'] - row['low']) * 0.6))

    def bullish_engulfing(prev, curr):
        return (prev['close'] < prev['open']) and (curr['close'] > curr['open']) and (curr['close'] > prev['open']) and (curr['open'] < prev['close'])

    def morning_star(df, i):
     if i < 2:
        return False
     window = df.iloc[i-2:i+1]
     if len(window) < 3:
        return False
     a, b, c = window.iloc[0], window.iloc[1], window.iloc[2]
     return (
        a['close'] < a['open'] and
        abs(b['close'] - b['open']) < (b['high'] - b['low']) * 0.1 and
        c['close'] > c['open']
    )

    def doji(row):
        return abs(row['close'] - row['open']) <= (row['high'] - row['low']) * 0.1

    def piercing_line(prev, curr):
        return (prev['close'] < prev['open']) and (curr['open'] < prev['low']) and (curr['close'] > prev['close'] + (prev['open'] - prev['close']) / 2)

    def three_white_soldiers(df, i):
        if i < 2:
         return False
        window = df.iloc[i-2:i+1]
        if len(window) < 3:
            return False
        a, b, c = window.iloc[0], window.iloc[1], window.iloc[2]
        return all([
        a['close'] > a['open'],
        b['close'] > b['open'],
        c['close'] > c['open'],
        a['close'] < b['close'] < c['close']
        ])

    pattern_data = {k: [] for k in ['Hammer', 'BullishEngulfing', 'MorningStar', 'Doji', 'PiercingLine', 'ThreeWhiteSoldiers']}
    for i in range(len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1] if i > 0 else row
        pattern_data['Hammer'].append(hammer(row))
        pattern_data['BullishEngulfing'].append(bullish_engulfing(prev, row))
        pattern_data['MorningStar'].append(morning_star(df, i))
        pattern_data['Doji'].append(doji(row))
        pattern_data['PiercingLine'].append(piercing_line(prev, row))
        pattern_data['ThreeWhiteSoldiers'].append(three_white_soldiers(df, i))

    for k, v in pattern_data.items():
        df[k] = v
    return df

df = df.groupby('symbol', group_keys=False).apply(detect_patterns)

# Step 8: Save
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✅ {csv_path} updated with new data and patterns.")
