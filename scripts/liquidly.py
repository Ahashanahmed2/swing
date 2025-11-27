# ---------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 1. Load & pre-process
# ---------------------------------------------------------
df = pd.read_csv("./csv/mongodb.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol', 'date'])

# 5-day avg volume (vectorized)
df['Avolume'] = df.groupby('symbol')['volume'].transform(
                    lambda s: s.rolling(5, min_periods=1).mean())
latest_df = df.groupby('symbol').tail(1).copy()          # last row per symbol

# ---------------------------------------------------------
# 2. Derived fields
# ---------------------------------------------------------
latest_df['TR'] = ((latest_df['value'] / latest_df['marketCap']) * 100).round(2)
latest_df['mcap_crore'] = (latest_df['marketCap'] * 0.1).round(2)
latest_df['value_cr'] = (latest_df['value'] * 0.1).round(2)

# volume ratio vs 5-day avg
latest_df['vol_ratio'] = (latest_df['volume'] / latest_df['Avolume']).replace([np.inf, -np.inf], np.nan)
latest_df['vol_pct'] = (latest_df.groupby('symbol', group_keys=False)['vol_ratio']
                          .rank(pct=True, method='average')
                          .fillna(0))

# ---------------------------------------------------------
# 3. Micro-buckets
# ---------------------------------------------------------
price_bin = (latest_df['close'] // 5).clip(0, 39).astype(int)          # 0-4.99→0, 5-9.99→1 … 195-199.99→39
log_mcap = np.log10(latest_df['mcap_crore'].clip(lower=1))
mcap_bin = (log_mcap // 0.5).clip(0, 9).astype(int)                    # 0-0.49→0 … 4.5-4.99→9
vol_bin = (latest_df['vol_pct'] // 0.1).clip(0, 9).astype(int)         # decile 0-9

# ---------------------------------------------------------
# 4. 3-D lookup table (score 1-5)
# shape: (price_bin, mcap_bin, vol_bin)
# খালি ১-৫ র‍্যানডম না দিয়া, নিচের টেবিলটা আপনার পুরনা threshold-গুলোকে
# মাইক্রো-লেভেলে ম্যাপ করে বানানো।  এখানে ডেমো হিসেবে সিমপ্লিফাইড মান দিলাম:
lookup = np.array([   # axis-0: price_bin (40)
    # for each price_bin 40-rows, inside every row 10×10 table
    [[5,5,5,4,4,3,3,2,2,1],
     [5,5,4,4,3,3,2,2,1,1],
     … 9 more mcap rows …],
    … 39 more price slices …
])

# ডেমো তৈরি (আসলে আপনার পুরনা threshold অনুযায়ী এটা বানাবেন)
lookup = np.random.randint(1, 6, size=(40, 10, 10))   # <--- replace with real table
score = lookup[price_bin, mcap_bin, vol_bin]

# ---------------------------------------------------------
# 5. Score → label
# ---------------------------------------------------------
label_map = {5: "Excellent", 4: "Good", 3: "Moderate", 2: "Poor", 1: "Avoid"}
latest_df['liq_score'] = score
latest_df['liquidity_rating'] = score.map(label_map)

# ---------------------------------------------------------
# 6. Final output
# ---------------------------------------------------------
latest_df['No'] = range(1, len(latest_df) + 1)
final_df = latest_df[['No', 'date', 'symbol', 'close', 'Avolume',
                      'TR', 'liq_score', 'liquidity_rating']]
final_df.rename(columns={'close': 'price'}, inplace=True)

# ---------------------------------------------------------
# 7. Save
# ---------------------------------------------------------
os.makedirs("./csv", exist_ok=True)
os.makedirs("./output/ai_signal", exist_ok=True)
final_df.to_csv("./csv/liquidity.csv", index=False)
final_df.to_csv("./output/ai_signal/liquidity.csv", index=False)

print("✅ Ultra-granular liquidity.csv generated!")
print(final_df.head())