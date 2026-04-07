from pymongo import MongoClient
import pandas as pd
import numpy as np
import ta
import os
import sys
from dotenv import load_dotenv
import requests
from io import StringIO

# =========================
# STEP 1: LOAD EXISTING CSV
# =========================

print("="*60)
print("STEP 1: Loading existing CSV from local storage...")
print("="*60)

csv_path = './csv/mongodb.csv'
os.makedirs('./csv', exist_ok=True)

# Load existing CSV if it exists
if os.path.exists(csv_path):
    df_existing = pd.read_csv(csv_path, encoding='utf-8-sig')
    df_existing['date'] = pd.to_datetime(df_existing['date'])
    print(f"✅ Loaded existing CSV:")
    print(f"   Total rows: {len(df_existing):,}")
    print(f"   Unique symbols: {df_existing['symbol'].nunique():,}")
    print(f"   Date range: {df_existing['date'].min().date()} to {df_existing['date'].max().date()}")
    existing_last_date = df_existing['date'].max()
    print(f"   Last date in CSV: {existing_last_date.date()}")
else:
    df_existing = pd.DataFrame()
    existing_last_date = None
    print("⚠️ No existing CSV found. Will create new from MongoDB.")

# =========================
# STEP 2: CHECK FOR NEW DATA IN MONGODB
# =========================
print("\n" + "="*60)
print("STEP 2: Checking MongoDB for new data...")
print("="*60)

load_dotenv()
client = MongoClient(os.getenv("MONGO_URL"))
collection = client["candleData"]["candledatas"]

# Query MongoDB for data after CSV's last date
if existing_last_date:
    query = {"date": {"$gt": existing_last_date.strftime('%Y-%m-%d')}}
    print(f"🔍 Looking for data after {existing_last_date.date()}...")
else:
    query = {}  # Get all if no existing CSV
    print("🔍 No existing CSV, fetching all data from MongoDB...")

data = list(collection.find(query, {'_id': 0, '__v': 0}))
client.close()

if not data:
    print("✅ No new data found in MongoDB. Using existing CSV only.")
    df_mongo = pd.DataFrame()
else:
    print(f"✅ Found {len(data)} new rows in MongoDB")
    df_mongo = pd.DataFrame(data)

    # Rename columns to match CSV format
    df_mongo.rename(columns={
        'open_price': 'open', 'close_price': 'close',
        'high_price': 'high', 'low_price': 'low',
        'vol': 'volume', 'val': 'value', 'trade_count': 'trades',
        'chg': 'change', 'mcap': 'marketCap'
    }, inplace=True)

    df_mongo['date'] = pd.to_datetime(df_mongo['date'], errors='coerce')

    # Clean numeric columns
    numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'value', 'trades', 'change', 'marketCap']
    for col in numeric_cols:
        if col in df_mongo.columns:
            df_mongo[col] = pd.to_numeric(df_mongo[col], errors='coerce')

    if 'collectedAt' in df_mongo.columns:
        df_mongo.drop(columns=['collectedAt'], inplace=True)

    print(f"   New data range: {df_mongo['date'].min().date()} to {df_mongo['date'].max().date()}")
    print(f"   New symbols: {df_mongo['symbol'].nunique()}")

# =========================
# STEP 3: MERGE EXISTING CSV AND MONGODB DATA
# =========================
print("\n" + "="*60)
print("STEP 3: Merging data...")
print("="*60)

if df_existing.empty and df_mongo.empty:
    print("❌ No data available from either source!")
    sys.exit(1)

elif df_existing.empty:
    df = df_mongo
    print("Using only MongoDB data")

elif df_mongo.empty:
    df = df_existing
    print("Using only existing CSV data (no new data found)")

else:
    # Merge both
    df = pd.concat([df_existing, df_mongo], ignore_index=True)
    df = df.drop_duplicates(['symbol', 'date'], keep='last')
    df = df.sort_values(['symbol', 'date'])
    print(f"✅ Merged: {len(df_existing)} (existing) + {len(df_mongo)} (Mongo) = {len(df)} total rows")
    print(f"   Unique symbols: {df['symbol'].nunique()}")

# =========================
# STEP 4: CALCULATE INDICATORS
# =========================
print("\n" + "="*60)
print("STEP 4: Calculating technical indicators...")
print("="*60)

def apply_indicators(group):
    if len(group) < 35:
        group['bb_upper'] = np.nan
        group['bb_middle'] = np.nan
        group['bb_lower'] = np.nan
        group['macd'] = np.nan
        group['macd_signal'] = np.nan
        group['macd_hist'] = np.nan
        group['rsi'] = np.nan
        group['atr'] = np.nan
        group['ema_200'] = np.nan
        return group

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=group['close'], window=20)
    group['bb_upper'] = bb.bollinger_hband()
    group['bb_middle'] = bb.bollinger_mavg()
    group['bb_lower'] = bb.bollinger_lband()

    # MACD
    macd = ta.trend.MACD(close=group['close'])
    group['macd'] = macd.macd()
    group['macd_signal'] = macd.macd_signal()
    group['macd_hist'] = macd.macd_diff()

    # RSI
    group['rsi'] = ta.momentum.RSIIndicator(close=group['close']).rsi()

    # ATR
    atr = ta.volatility.AverageTrueRange(
        high=group['high'], low=group['low'], close=group['close'], window=14
    )
    group['atr'] = atr.average_true_range()

    # EMA 200
    group['ema_200'] = ta.trend.EMAIndicator(close=group['close'], window=200).ema_indicator()

    return group

print("🔄 Calculating indicators by symbol...")
df = df.groupby('symbol', group_keys=False).apply(apply_indicators)

# =========================
# STEP 5: ZIGZAG
# =========================
print("🔄 Calculating ZigZag...")

def compute_zigzag(close, threshold=0.05):
    zz = [np.nan] * len(close)
    if len(close) == 0:
        return zz
    last_pivot = close.iloc[0]
    for i in range(1, len(close)):
        change = abs(close.iloc[i] - last_pivot) / last_pivot
        if change >= threshold:
            zz[i] = close.iloc[i]
            last_pivot = close.iloc[i]
    return zz

df['zigzag'] = df.groupby('symbol')['close'].transform(compute_zigzag)

# =========================
# STEP 6: PATTERN DETECTION
# =========================
print("🔄 Detecting candlestick patterns...")

def detect_patterns(group):
    def hammer(row):
        body = abs(row['close'] - row['open'])
        lower_shadow = min(row['open'], row['close']) - row['low']
        return (row['close'] > row['open']) and (lower_shadow > body * 2)

    def bullish_engulfing(prev, curr):
        return (prev['close'] < prev['open'] and
                curr['close'] > curr['open'] and
                curr['close'] > prev['open'] and
                curr['open'] < prev['close'])

    def morning_star(df, i):
        if i < 2:
            return False
        a, b, c = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        return (a['close'] < a['open'] and
                abs(b['close'] - b['open']) < (b['high'] - b['low']) * 0.1 and
                c['close'] > c['open'])

    def doji(row):
        return abs(row['close'] - row['open']) <= (row['high'] - row['low']) * 0.1

    def piercing_line(prev, curr):
        return (prev['close'] < prev['open'] and
                curr['open'] < prev['low'] and
                curr['close'] > prev['close'] + (prev['open'] - prev['close']) / 2)

    def three_white_soldiers(df, i):
        if i < 2:
            return False
        a, b, c = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        return (a['close'] > a['open'] and
                b['close'] > b['open'] and
                c['close'] > c['open'] and
                a['close'] < b['close'] < c['close'])

    patterns = ['Hammer', 'BullishEngulfing', 'MorningStar', 'Doji', 'PiercingLine', 'ThreeWhiteSoldiers']
    for p in patterns:
        group[p] = False

    for i in range(len(group)):
        row = group.iloc[i]
        prev = group.iloc[i-1] if i > 0 else row

        group.loc[group.index[i], 'Hammer'] = hammer(row)
        group.loc[group.index[i], 'BullishEngulfing'] = bullish_engulfing(prev, row)
        group.loc[group.index[i], 'MorningStar'] = morning_star(group, i)
        group.loc[group.index[i], 'Doji'] = doji(row)
        group.loc[group.index[i], 'PiercingLine'] = piercing_line(prev, row)
        group.loc[group.index[i], 'ThreeWhiteSoldiers'] = three_white_soldiers(group, i)

    return group

df = df.groupby('symbol', group_keys=False).apply(detect_patterns)

# =========================
# STEP 7: SAVE FINAL CSV
# =========================
print("\n" + "="*60)
print("STEP 7: Saving final CSV...")
print("="*60)

df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✅ Saved to {csv_path}")
print(f"   Total rows: {len(df):,}")
print(f"   Unique symbols: {df['symbol'].nunique():,}")
print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# =========================
# SUMMARY
# =========================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"📥 Existing CSV: {len(df_existing):,} rows" if not df_existing.empty else "📥 Existing CSV: None")
print(f"📥 MongoDB New Data: {len(df_mongo):,} rows" if not df_mongo.empty else "📥 MongoDB New Data: None")
print(f"📊 Final CSV: {len(df):,} rows, {df['symbol'].nunique()} symbols")
print("="*60)
print("✅ mongodb.py completed!")