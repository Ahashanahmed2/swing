import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------
df = pd.read_csv("./csv/mongodb.csv")

# Ensure date is parsed
df['date'] = pd.to_datetime(df['date'])

# Sort by symbol + date
df = df.sort_values(['symbol', 'date'])

# ---------------------------------------------------------
# Get last row + last 5-day avg volume (FAST)
# ---------------------------------------------------------
# last row per symbol
last_rows = df.groupby('symbol').tail(1)

# fast rolling avg (5 days)
df['Avolume'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(5).mean())

# take Avolume at last row
Avol = df.groupby('symbol').tail(1)['Avolume']

latest_df = last_rows.copy()
latest_df['Avolume'] = Avol.values

# ---------------------------------------------------------
# Turnover Ratio
# ---------------------------------------------------------
latest_df['TR'] = (latest_df['value'] / latest_df['marketCap']).round(2)

# ---------------------------------------------------------
# Convert MarketCap (million → crore)
# 1 million = 0.1 crore
# ---------------------------------------------------------
latest_df['mcap_crore'] = (latest_df['marketCap'] * 0.1).round(2)

# ---------------------------------------------------------
# Granular Liquidity Rating (Optimized & Vectorized)
# New finer price bands: 40-60, 60-80, 80-120, 120-160, 160-200
# ---------------------------------------------------------
def liquidity_rating_granular(price, mcap, vol, val):
    # convert value (BDT) to কোটি (1 কোটি = 10,000,000)
    vcr = val / 1e7 if val is not None and not np.isnan(val) else 0.0

    # basic guards
    if price is None or np.isnan(price) or mcap is None or np.isnan(mcap) or np.isnan(vol) or vol < 0:
        return "Avoid"
    try:
        price = float(price)
        mcap = float(mcap)
        vol = float(vol)
    except Exception:
        return "Avoid"

    # small price bands first (keep existing logic for low prices)
    if 1 <= price <= 5:
        if mcap < 100 and vol >= 1_500_000 and vcr >= 5: return "Excellent"
        if 100 <= mcap < 300 and vol >= 800_000 and vcr >= 3: return "Good"
        if 300 <= mcap < 600 and vol >= 400_000 and vcr >= 2: return "Moderate"
        if mcap >= 600 and vol >= 200_000 and vcr >= 1: return "Poor"
        return "Avoid"

    if 5 < price <= 10:
        if mcap < 100 and vol >= 1_000_000 and vcr >= 4: return "Excellent"
        if 100 <= mcap < 300 and vol >= 600_000 and vcr >= 3: return "Good"
        if 300 <= mcap < 600 and vol >= 300_000 and vcr >= 1.5: return "Moderate"
        if mcap >= 600 and vol >= 100_000 and vcr >= 1: return "Poor"
        return "Avoid"

    if 10 < price <= 20:
        if mcap < 100 and vol >= 800_000 and vcr >= 4: return "Excellent"
        if 100 <= mcap < 300 and vol >= 500_000 and vcr >= 2: return "Good"
        if 300 <= mcap < 600 and vol >= 200_000 and vcr >= 1: return "Moderate"
        if mcap >= 600 and vol >= 100_000 and vcr >= 0.5: return "Poor"
        return "Avoid"

    if 20 < price <= 40:
        if mcap < 150 and vol >= 400_000 and vcr >= 3: return "Excellent"
        if 150 <= mcap < 300 and vol >= 300_000 and vcr >= 2: return "Good"
        if 300 <= mcap < 600 and vol >= 100_000 and vcr >= 1: return "Moderate"
        if mcap >= 600 and vol >= 80_000 and vcr >= 0.5: return "Poor"
        return "Avoid"

    # ---------- New finer bands ----------
    # 40-60
    if 40 < price <= 60:
        if mcap < 220 and vol >= 225_000 and vcr >= 1.8: return "Excellent"
        if 220 <= mcap < 420 and vol >= 180_000 and vcr >= 1.2: return "Good"
        if 420 <= mcap < 800 and vol >= 90_000 and vcr >= 0.9: return "Moderate"
        if mcap >= 800 and vol >= 45_000 and vcr >= 0.5: return "Poor"
        return "Avoid"

    # 60-80
    if 60 < price <= 80:
        if mcap < 220 and vol >= 200_000 and vcr >= 1.6: return "Excellent"
        if 220 <= mcap < 420 and vol >= 160_000 and vcr >= 1.0: return "Good"
        if 420 <= mcap < 800 and vol >= 80_000 and vcr >= 0.8: return "Moderate"
        if mcap >= 800 and vol >= 40_000 and vcr >= 0.4: return "Poor"
        return "Avoid"

    # 80-120
    if 80 < price <= 120:
        if mcap < 280 and vol >= 170_000 and vcr >= 1.6: return "Excellent"
        if 280 <= mcap < 600 and vol >= 120_000 and vcr >= 1.1: return "Good"
        if 600 <= mcap < 1000 and vol >= 70_000 and vcr >= 0.9: return "Moderate"
        if mcap >= 1000 and vol >= 35_000 and vcr >= 0.45: return "Poor"
        return "Avoid"

    # 120-160
    if 120 < price <= 160:
        if mcap < 350 and vol >= 140_000 and vcr >= 1.4: return "Excellent"
        if 350 <= mcap < 800 and vol >= 90_000 and vcr >= 1.0: return "Good"
        if 800 <= mcap < 1300 and vol >= 60_000 and vcr >= 0.8: return "Moderate"
        if mcap >= 1300 and vol >= 30_000 and vcr >= 0.4: return "Poor"
        return "Avoid"

    # 160-200
    if 160 < price <= 200:
        if mcap < 420 and vol >= 120_000 and vcr >= 1.2: return "Excellent"
        if 420 <= mcap < 1000 and vol >= 80_000 and vcr >= 0.9: return "Good"
        if 1000 <= mcap < 1500 and vol >= 50_000 and vcr >= 0.7: return "Moderate"
        if mcap >= 1500 and vol >= 25_000 and vcr >= 0.35: return "Poor"
        return "Avoid"

    # rest larger bands remain Avoid (or you can expand further)
    return "Avoid"

# vectorize for speed
vec_rating = np.vectorize(liquidity_rating_granular, otypes=[object])

latest_df['liquidity_rating'] = vec_rating(
    latest_df['close'].fillna(0).values,
    latest_df['mcap_crore'].fillna(0).values,
    latest_df['volume'].fillna(0).values,
    latest_df['value'].fillna(0).values
)

# ---------------------------------------------------------
# Final Output
# ---------------------------------------------------------
latest_df['No'] = range(1, len(latest_df) + 1)
latest_df['price'] = latest_df['close']
latest_df['value_traded'] = latest_df['value']

final_df = latest_df[['No', 'date', 'symbol', 'price', 'Avolume',
                      'TR', 'mcap_crore', 'volume', 'value_traded',
                      'liquidity_rating']]

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
os.makedirs("./csv", exist_ok=True)
os.makedirs("./output/ai_signal", exist_ok=True)

final_df.to_csv("./csv/liquidity.csv", index=False)
final_df.to_csv("./output/ai_signal/liquidity.csv", index=False)

print("✅ Granular liquidity.csv generated!")
print(final_df.head())