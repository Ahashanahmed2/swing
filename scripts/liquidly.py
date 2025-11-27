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
# Get last row + 5-day avg volume
# ---------------------------------------------------------
last_rows = df.groupby('symbol').tail(1).copy()
df['Avolume'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(5).mean())
Avol = df.groupby('symbol').tail(1)['Avolume']
last_rows['Avolume'] = Avol.values
latest_df = last_rows

# ---------------------------------------------------------
# Turnover Ratio as percentage
# ---------------------------------------------------------
latest_df['TR'] = ((latest_df['value'] / latest_df['marketCap']) * 100).round(2)

# ---------------------------------------------------------
# Liquidity Rating with debug
# ---------------------------------------------------------
def liquidity_rating_debug(price, mcap, vol, value):
    value_cr = value * 0.1  # million → crore
    bucket = None
    thresholds = []

    # --- define price buckets ---
    if price <= 5:
        bucket = "1-5"
        thresholds = [(100, 1_500_000, 5, "Excellent"),
                      (300, 800_000, 3, "Good"),
                      (600, 400_000, 2, "Moderate"),
                      (np.inf, 200_000, 1, "Poor")]
    elif price <= 10:
        bucket = "5-10"
        thresholds = [(100, 1_000_000, 4, "Excellent"),
                      (300, 600_000, 3, "Good"),
                      (600, 300_000, 1.5, "Moderate"),
                      (np.inf, 100_000, 1, "Poor")]
    elif price <= 20:
        bucket = "10-20"
        thresholds = [(100, 800_000, 4, "Excellent"),
                      (300, 500_000, 2, "Good"),
                      (600, 200_000, 1, "Moderate"),
                      (np.inf, 100_000, 0.5, "Poor")]
    elif price <= 40:
        bucket = "20-40"
        thresholds = [(150, 400_000, 3, "Excellent"),
                      (300, 300_000, 2, "Good"),
                      (600, 100_000, 1, "Moderate"),
                      (np.inf, 80_000, 0.5, "Poor")]
    elif price <= 60:
        bucket = "40-60"
        thresholds = [(200, 250_000, 2.5, "Excellent"),
                      (400, 200_000, 2, "Good"),
                      (800, 100_000, 1, "Moderate"),
                      (np.inf, 50_000, 0.5, "Poor")]
    elif price <= 80:
        bucket = "60-80"
        thresholds = [(300, 200_000, 2, "Excellent"),
                      (600, 150_000, 1.5, "Good"),
                      (1000, 100_000, 1, "Moderate"),
                      (np.inf, 50_000, 0.5, "Poor")]
    elif price <= 120:
        bucket = "80-120"
        thresholds = [(300, 150_000, 2, "Excellent"),
                      (600, 100_000, 1.5, "Good"),
                      (1000, 70_000, 1, "Moderate"),
                      (np.inf, 40_000, 0.5, "Poor")]
    elif price <= 200:
        bucket = "120-200"
        thresholds = [(400, 100_000, 2, "Excellent"),
                      (800, 80_000, 1.5, "Good"),
                      (1200, 60_000, 1, "Moderate"),
                      (np.inf, 30_000, 0.5, "Poor")]
    else:
        bucket = ">200"
        thresholds = [(500, 80_000, 2, "Excellent"),
                      (1000, 60_000, 1.5, "Good"),
                      (1500, 40_000, 1, "Moderate"),
                      (np.inf, 20_000, 0.5, "Poor")]

    rating = "Avoid"
    for mcap_th, vol_th, val_th, r in thresholds:
        if mcap < mcap_th and vol >= vol_th and value_cr >= val_th:
            rating = r
            break

    # Debug print
    print(f"Symbol: {symbol}, Price: {price}, Bucket: {bucket}, "
          f"MCAP: {mcap}, Volume: {vol}, Value_cr: {value_cr}, Rating: {rating}")

    return rating

# Apply with debug
latest_df['liquidity_rating'] = latest_df.apply(
    lambda r: liquidity_rating_debug(r['close'], r['marketCap'], r['volume'], r['value']),
    axis=1
)

# ---------------------------------------------------------
# Final Output (only necessary columns)
# ---------------------------------------------------------
latest_df['No'] = range(1, len(latest_df) + 1)
latest_df['price'] = latest_df['close']

final_df = latest_df[['No','date','symbol','price','Avolume','TR','liquidity_rating']]

# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------
os.makedirs("./csv", exist_ok=True)
os.makedirs("./output/ai_signal", exist_ok=True)

final_df.to_csv("./csv/liquidity.csv", index=False)
final_df.to_csv("./output/ai_signal/liquidity.csv", index=False)

print("✅ Final liquidity.csv generated with debug prints!")