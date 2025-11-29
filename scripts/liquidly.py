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
# 5-day average volume
# ---------------------------------------------------------
df['Avolume'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
lastrows = df.groupby('symbol').tail(1).copy().reset_index(drop=True)

latestdf = lastrows

# ---------------------------------------------------------
# Turnover Ratio as percentage
# ---------------------------------------------------------
latestdf['TR'] = ((latestdf['value'] / latestdf['marketCap']) * 100).round(2)

# ---------------------------------------------------------
# Liquidity Rating (DSE adjusted)
# ---------------------------------------------------------
def liquidityrating_adjusted(row):
    price = row['close']
    mcap_cr = row['marketCap'] * 0.1  # Million -> Crore
    value_cr = row['value'] * 0.1     # Million -> Crore
    vol = row['Avolume']
    symbol = row['symbol']

    bucket = None
    thresholds = []

    # ---------------------------------------------------------
    # Price buckets with thresholds
    # ---------------------------------------------------------
    if price <= 5:
        bucket = "1-5"
        thresholds = [
            (80, 200_000, 0.8, "Excellent"),
            (200, 120_000, 0.5, "Good"),
            (400, 70_000, 0.3, "Moderate"),
            (np.inf, 30_000, 0.1, "Poor")
        ]
    elif price <= 10:
        bucket = "5-10"
        thresholds = [
            (100, 150_000, 0.6, "Excellent"),
            (250, 90_000, 0.4, "Good"),
            (500, 50_000, 0.2, "Moderate"),
            (np.inf, 25_000, 0.08, "Poor")
        ]
    elif price <= 20:
        bucket = "10-20"
        thresholds = [
            (150, 100_000, 0.5, "Excellent"),
            (300, 60_000, 0.3, "Good"),
            (600, 30_000, 0.15, "Moderate"),
            (np.inf, 15_000, 0.07, "Poor")
        ]
    elif price <= 40:
        bucket = "20-40"
        thresholds = [
            (200, 70_000, 0.4, "Excellent"),
            (400, 40_000, 0.2, "Good"),
            (800, 20_000, 0.1, "Moderate"),
            (np.inf, 12_000, 0.05, "Poor")
        ]
    elif price <= 60:
        bucket = "40-60"
        thresholds = [
            (300, 50_000, 0.3, "Excellent"),
            (600, 30_000, 0.18, "Good"),
            (1000, 15_000, 0.1, "Moderate"),
            (np.inf, 10_000, 0.05, "Poor")
        ]
    elif price <= 80:
        bucket = "60-80"
        thresholds = [
            (400, 40_000, 0.3, "Excellent"),
            (800, 25_000, 0.18, "Good"),
            (1200, 13_000, 0.1, "Moderate"),
            (np.inf, 9_000, 0.05, "Poor")
        ]
    elif price <= 120:
        bucket = "80-120"
        thresholds = [
            (500, 30_000, 0.25, "Excellent"),
            (900, 18_000, 0.15, "Good"),
            (1500, 10_000, 0.1, "Moderate"),
            (np.inf, 7_000, 0.05, "Poor")
        ]
    elif price <= 200:
        bucket = "120-200"
        thresholds = [
            (600, 22_000, 0.2, "Excellent"),
            (1000, 15_000, 0.12, "Good"),
            (1800, 8_000, 0.08, "Moderate"),
            (np.inf, 6_000, 0.04, "Poor")
        ]
    else:
        bucket = ">200"
        thresholds = [
            (700, 15_000, 0.2, "Excellent"),
            (1200, 10_000, 0.12, "Good"),
            (2000, 6_000, 0.08, "Moderate"),
            (np.inf, 4_000, 0.03, "Poor")
        ]

    # ---------------------------------------------------------
    # Determine rating
    # ---------------------------------------------------------
    rating = "Avoid"
    for mcapth, volth, val_th, r in thresholds:
        if mcap_cr < mcapth and vol >= volth and value_cr >= val_th:
            rating = r
            break

    # Debug print
    print(f"Symbol: {symbol}, Price: {price}, Bucket: {bucket}, "
          f"MCAPcr: {mcap_cr:.2f}, AVolume: {vol:.0f}, Valuecr: {value_cr:.2f}, Rating: {rating}")

    return rating

# ---------------------------------------------------------
# Apply function
# ---------------------------------------------------------
latestdf['liquidity_rating'] = latestdf.apply(liquidityrating_adjusted, axis=1)

# ---------------------------------------------------------
# Final Output
# ---------------------------------------------------------
latestdf['No'] = range(1, len(latestdf) + 1)
latestdf['price'] = latestdf['close']

finaldf = latestdf[['No','date','symbol','price','Avolume','TR','liquidity_rating']]

# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------
os.makedirs("./csv", exist_ok=True)
os.makedirs("./output/ai_signal", exist_ok=True)

finaldf.to_csv("./csv/liquidity.csv", index=False)
finaldf.to_csv("./output/ai_signal/liquidity.csv", index=False)

print("✅ Final liquidity.csv generated successfully!")
print(f"📊 Total stocks processed: {len(finaldf)}")
print(f"📈 Rating distribution:")
print(finaldf['liquidityrating'].value_counts())
