from pymongo import MongoClient
import pandas as pd
import numpy as np
import ta
import os
import sys
from dotenv import load_dotenv
from hf_uploader import download_from_hf_or_run_script


download_from_hf_or_run_script()
# 📁 CSV ফাইল path
csv_path = './csv/mongodb.csv'
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
# ✅ ফাইল আছে কি না চেক
if os.path.exists(csv_path):
    print("✅ CSV file exists. Checking for new data in MongoDB...")

    try:
        csv_df = pd.read_csv(csv_path)
        csv_df['date'] = pd.to_datetime(csv_df['date'])
        csv_last_date = csv_df['date'].max()
        print(f"🕒 Last date in CSV: {csv_last_date}")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        csv_last_date = None
else:
    print("🆕 CSV file not found. Downloading full data from MongoDB...")
    csv_last_date = None

# 🔌 MongoDB সংযোগ
load_dotenv()
mongourl = os.getenv("MONGO_URL")
client = MongoClient(mongourl)
db = client["candleData"]
collection = db["candledatas"]

# 🕒 MongoDB এর সর্বশেষ তারিখ বের করো
latest_doc = collection.find_one({}, sort=[('date', -1)])
if not latest_doc:
    print("❌ MongoDB is empty.")
    sys.exit(1)
mongo_last_date = pd.to_datetime(latest_doc['date'])

# 🧪 যদি নতুন ডেটা না থাকে
if csv_last_date and csv_last_date >= mongo_last_date:
    print(f"✅ No new data in MongoDB after {csv_last_date}. Exiting to prevent re-download.")
    sys.exit(1)

# 📥 ডেটা লোড (নতুন বা সব)
if csv_last_date:
    query = {"date": {"$gt": csv_last_date}}
    print(f"🔄 Downloading new data from MongoDB after {csv_last_date}...")
else:
    query = {}
    print("📥 Downloading full data from MongoDB...")

data = list(collection.find(query, {'_id': 0, '__v': 0}))
client.close()

if not data:
    print("✅ No new data found in MongoDB. Exiting.")
    sys.exit(1)

df = pd.DataFrame(data)

# 🔁 Column rename
df.rename(columns={
    'open_price': 'open',
    'close_price': 'close',
    'high_price': 'high',
    'low_price': 'low',
    'vol': 'volume',
    'val': 'value',
    'trade_count': 'trades',
    'chg': 'change',
    'mcap': 'marketCap'
}, inplace=True)

# 🧼 Cleanup
df['date'] = pd.to_datetime(df['date'], errors='coerce')
numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'value', 'trades', 'change', 'marketCap']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

if 'collectedAt' in df.columns:
    df.drop(columns=['collectedAt'], inplace=True)
    print("🗑️ 'collectedAt' column removed.")

# 🧷 পুরানো CSV এর সাথে যুক্ত করা
if csv_last_date:
    old_df = csv_df
    combined_df = pd.concat([old_df, df], ignore_index=True)
    combined_df.drop_duplicates(subset=['symbol', 'date'], keep='last', inplace=True)
    combined_df.sort_values(by=['symbol', 'date'], inplace=True)
    df = combined_df
    print("💾 Appended new data to existing CSV.")

# ========== 📈 Technical Indicators ==========
def apply_indicators(group):
    bb = ta.volatility.BollingerBands(close=group['close'], window=20, window_dev=2)
    group['bb_upper'] = bb.bollinger_hband()
    group['bb_middle'] = bb.bollinger_mavg()
    group['bb_lower'] = bb.bollinger_lband()

    macd = ta.trend.MACD(close=group['close'])
    group['macd'] = macd.macd()
    group['macd_signal'] = macd.macd_signal()
    group['macd_hist'] = macd.macd_diff()

    group['rsi'] = ta.momentum.RSIIndicator(close=group['close']).rsi()
    return group

df = df.groupby('symbol', group_keys=False).apply(apply_indicators).reset_index(drop=True)

# ========== 🔺 ZigZag ==========
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

# ========== 📉 Candlestick Patterns ==========
def detect_patterns(df):
    def hammer(row):
        return abs(row['close'] - row['open']) < (row['high'] - row['low']) * 0.25 and \
               min(row['open'], row['close']) - row['low'] > (row['high'] - row['low']) * 0.6

    def bullish_engulfing(prev, curr):
        return prev['close'] < prev['open'] and curr['close'] > curr['open'] and \
               curr['open'] < prev['close'] and curr['close'] > prev['open']

    def morning_star(df, i):
        if i < 2: return False
        first, second, third = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        small_body = abs(second['close'] - second['open']) < (second['high'] - second['low']) * 0.1
        return first['close'] < first['open'] and small_body and \
               third['close'] > third['open'] and \
               third['close'] > (first['open'] + first['close']) / 2

    def doji(row): return abs(row['open'] - row['close']) <= (row['high'] - row['low']) * 0.05

    def piercing_line(prev, curr): return prev['close'] < prev['open'] and \
                                        curr['open'] < prev['low'] and \
                                        curr['close'] > ((prev['open'] + prev['close']) / 2) and \
                                        curr['close'] < prev['open']

    def three_white_soldiers(df, i):
        if i < 2: return False
        first, second, third = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        return all([
            first['close'] > first['open'],
            second['close'] > second['open'],
            third['close'] > third['open'],
            first['close'] < second['open'] < second['close'] < third['open'] < third['close']
        ])

    pattern_data = {
        'Hammer': [], 'BullishEngulfing': [], 'MorningStar': [],
        'Doji': [], 'PiercingLine': [], 'ThreeWhiteSoldiers': []
    }

    for i in range(len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1] if i > 0 else row

        pattern_data['Hammer'].append(hammer(row))
        pattern_data['BullishEngulfing'].append(bullish_engulfing(prev, row))
        pattern_data['MorningStar'].append(morning_star(df, i))
        pattern_data['Doji'].append(doji(row))
        pattern_data['PiercingLine'].append(piercing_line(prev, row))
        pattern_data['ThreeWhiteSoldiers'].append(three_white_soldiers(df, i))

    for key in pattern_data:
        df[key] = pattern_data[key]

    return df

df = df.groupby('symbol', group_keys=False).apply(detect_patterns).reset_index(drop=True)

# 💾 Save Final Data
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✅ {csv_path} updated with new data and patterns.")
