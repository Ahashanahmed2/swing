import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------
df = pd.read_csv("./csv/mongodb.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol', 'date'])

# ---------------------------------------------------------
# 5-day average volume & value
# ---------------------------------------------------------
df['Avolume'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['Avalue'] = df.groupby('symbol')['value'].transform(lambda x: x.rolling(5, min_periods=1).mean())
lastrows = df.groupby('symbol').tail(1).copy().reset_index(drop=True)
latestdf = lastrows

# ---------------------------------------------------------
# Turnover Ratio as percentage
# ---------------------------------------------------------
latestdf['TR'] = ((latestdf['value'] / latestdf['marketCap']) * 100).round(2)

# ---------------------------------------------------------
# ✅ ENHANCED: Liquidity Score (0.0 to 1.0) + Rating
# ---------------------------------------------------------
def liquidity_rating_adjusted(row):
    price = row['close']
    mcap_cr = row['marketCap'] * 0.1  # Million → Crore
    value_cr = row['value'] * 0.1     # Million → Crore
    vol = row['Avolume']
    symbol = row['symbol']

    bucket = None
    thresholds = []

    # DSE-adjusted thresholds
    if price <= 5:
        bucket = "1-5"
        thresholds = [(80, 200_000, 0.8, "Excellent", 1.0),
                      (200, 120_000, 0.5, "Good", 0.7),
                      (400, 70_000, 0.3, "Moderate", 0.4),
                      (np.inf, 30_000, 0.1, "Poor", 0.1)]
    elif price <= 10:
        bucket = "5-10"
        thresholds = [(100, 150_000, 0.6, "Excellent", 1.0),
                      (250, 90_000, 0.4, "Good", 0.7),
                      (500, 50_000, 0.2, "Moderate", 0.4),
                      (np.inf, 25_000, 0.08, "Poor", 0.1)]
    elif price <= 20:
        bucket = "10-20"
        thresholds = [(150, 100_000, 0.5, "Excellent", 1.0),
                      (300, 60_000, 0.3, "Good", 0.7),
                      (600, 30_000, 0.15, "Moderate", 0.4),
                      (np.inf, 15_000, 0.07, "Poor", 0.1)]
    elif price <= 40:
        bucket = "20-40"
        thresholds = [(200, 70_000, 0.4, "Excellent", 1.0),
                      (400, 40_000, 0.2, "Good", 0.7),
                      (800, 20_000, 0.1, "Moderate", 0.4),
                      (np.inf, 12_000, 0.05, "Poor", 0.1)]
    elif price <= 60:
        bucket = "40-60"
        thresholds = [(300, 50_000, 0.3, "Excellent", 1.0),
                      (600, 30_000, 0.18, "Good", 0.7),
                      (1000, 15_000, 0.1, "Moderate", 0.4),
                      (np.inf, 10_000, 0.05, "Poor", 0.1)]
    elif price <= 80:
        bucket = "60-80"
        thresholds = [(400, 40_000, 0.3, "Excellent", 1.0),
                      (800, 25_000, 0.18, "Good", 0.7),
                      (1200, 13_000, 0.1, "Moderate", 0.4),
                      (np.inf, 9_000, 0.05, "Poor", 0.1)]
    elif price <= 120:
        bucket = "80-120"
        thresholds = [(500, 30_000, 0.25, "Excellent", 1.0),
                      (900, 18_000, 0.15, "Good", 0.7),
                      (1500, 10_000, 0.1, "Moderate", 0.4),
                      (np.inf, 7_000, 0.05, "Poor", 0.1)]
    elif price <= 200:
        bucket = "120-200"
        thresholds = [(600, 22_000, 0.2, "Excellent", 1.0),
                      (1000, 15_000, 0.12, "Good", 0.7),
                      (1800, 8_000, 0.08, "Moderate", 0.4),
                      (np.inf, 6_000, 0.04, "Poor", 0.1)]
    else:
        bucket = ">200"
        thresholds = [(700, 15_000, 0.2, "Excellent", 1.0),
                      (1200, 10_000, 0.12, "Good", 0.7),
                      (2000, 6_000, 0.08, "Moderate", 0.4),
                      (np.inf, 4_000, 0.03, "Poor", 0.1)]

    # Determine rating & numeric score
    rating = "Avoid"
    score = 0.0
    for mcapth, volth, val_th, r, s in thresholds:
        if mcap_cr < mcapth and vol >= volth and value_cr >= val_th:
            rating = r
            score = s
            break

    return pd.Series([rating, score, bucket])

# ---------------------------------------------------------
# Apply function → get BOTH rating AND score
# ---------------------------------------------------------
latestdf[['liquidity_rating', 'liquidity_score', 'price_bucket']] = latestdf.apply(liquidity_rating_adjusted, axis=1)

# ---------------------------------------------------------
# ✅ INTEGRATE WITH YOUR SYSTEM:
#    → liquidity.csv uses original column names
#    → liquidity_system.csv ready for trade_stock.csv merge
# ---------------------------------------------------------
os.makedirs("./csv", exist_ok=True)
os.makedirs("./output/ai_signal", exist_ok=True)

# Final output (for human) — ✅ ORIGINAL COLUMN NAMES
finaldf = latestdf[['symbol', 'close', 'Avolume', 'value', 'TR', 'liquidity_rating', 'liquidity_score', 'price_bucket']].copy()
finaldf.columns = ['symbol', 'price', 'Avolume', 'Avalue', 'TR(%)', 'liquidity_rating', 'liquidity_score', 'price_bucket']
finaldf = finaldf.sort_values('liquidity_score', ascending=False).reset_index(drop=True)
finaldf.insert(0, 'No', range(1, len(finaldf)+1))

finaldf.to_csv("./csv/liquidity.csv", index=False)
print("✅ liquidity.csv saved (with original column names)")

# ✅ SYSTEM INTELLIGENCE FILE (for trade_stock.csv merge)
system_liquidity = latestdf[['symbol', 'liquidity_rating', 'liquidity_score', 'price_bucket']].copy()
system_liquidity.to_csv("./csv/liquidity_system.csv", index=False)
print("✅ liquidity_system.csv saved (for system integration)")

# ✅ Update trade_stock.csv (if exists)
trade_stock_path = "./csv/trade_stock.csv"
if os.path.exists(trade_stock_path):
    try:
        trade_df = pd.read_csv(trade_stock_path)
        trade_df = trade_df.merge(system_liquidity, on='symbol', how='left')
        trade_df.to_csv(trade_stock_path, index=False)
        print(f"🔄 Updated {trade_stock_path} with liquidity columns")
    except Exception as e:
        print(f"⚠️ Failed to update trade_stock.csv: {e}")

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print(f"\n📊 Total stocks processed: {len(latestdf)}")
print("\n📈 Liquidity Rating Distribution:")
print(latestdf['liquidity_rating'].value_counts())
print(f"\n🔢 Avg Liquidity Score: {latestdf['liquidity_score'].mean():.2f}")

# 🔔 Alert if too many "Avoid"
avoid_count = (latestdf['liquidity_rating'] == 'Avoid').sum()
if avoid_count > len(latestdf) * 0.3:
    print(f"\n❗ Warning: {avoid_count} stocks ({avoid_count/len(latestdf)*100:.1f}%) are 'Avoid' — consider liquidity filter in signals.")
