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
# ---------------------------------------------------------
# Get last row + 5-day avg volume
# ---------------------------------------------------------
# ✅ সরাসরি একসাথে নিন — ইনডেক্স সেফ
df['Avolume'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
last_rows = df.groupby('symbol').tail(1).copy().reset_index(drop=True)  # ✅ reset_index দিন

latest_df = last_rows

# ---------------------------------------------------------
# Turnover Ratio as percentage
# ---------------------------------------------------------
latest_df['TR'] = ((latest_df['value'] / latest_df['marketCap']) * 100).round(2)

# ---------------------------------------------------------
# Liquidity Rating (with ADJUSTED THRESHOLDS FOR DSE)
# ---------------------------------------------------------
def liquidity_rating_adjusted(row):
    price = row['close']
    
    # ⭐ পরিবর্তন ১: marketCap এবং value কে কোটি টাকায় (Crores) রূপান্তর 
    # ধরে নেওয়া হচ্ছে মূল ডেটাফ্রেমের marketCap এবং value মিলিয়নে আছে।
    mcap_cr = row['marketCap'] * 0.1  # Million -> Crore
    value_cr = row['value'] * 0.1    # Million -> Crore
    
    # ⭐ পরিবর্তন ২: ৫ দিনের গড় ভলিউম (Avolume) ব্যবহার করা 
    vol = row['Avolume'] 
    
    symbol = row['symbol']

    bucket = None
    thresholds = []

    # ---------------------------------------------------------
    # ADJUSTED THRESHOLDS FOR DSE (BANGLADESH) - এখন MCAP ও VALUE কোটি টাকায়
    # (MCAP_TH, VOL_TH, VAL_CR_TH, RATING)
    # ---------------------------------------------------------
    
    # [MCAP_CR_TH, VOL_TH, VAL_CR_TH, RATING]
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
    # Rating Select
    # ---------------------------------------------------------
    rating = "Avoid"
    # mcap_th (কোটি), vol_th (শেয়ার), val_th (কোটি)
    for mcap_th, vol_th, val_th, r in thresholds:
        if mcap_cr < mcap_th and vol >= vol_th and value_cr >= val_th:
            rating = r
            break

    # Debug Print (আপডেট করা ভ্যারিয়েবল ব্যবহার)
    print(f"Symbol: {symbol}, Price: {price}, Bucket: {bucket}, "
          f"MCAP_cr: {mcap_cr:.2f}, AVolume: {vol:.0f}, Value_cr: {value_cr:.2f}, Rating: {rating}")

    return rating

# Apply (আপডেট করা ফাংশন ব্যবহার)
latest_df['liquidity_rating'] = latest_df.apply(liquidity_rating_adjusted, axis=1)

# ---------------------------------------------------------
# Final Output
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

print("✅ Final liquidity.csv generated successfully!")
print(f"📊 Total stocks processed: {len(final_df)}")
print(f"📈 Rating distribution:")
print(final_df['liquidity_rating'].value_counts())
