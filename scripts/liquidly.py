import pandas as pd
import os

# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------
df = pd.read_csv("./csv/mongodb.csv")

# Ensure date is parsed
df['date'] = pd.to_datetime(df['date'])

# Sort by symbol + date
df = df.sort_values(by=['symbol', 'date'])

# ---------------------------------------------------------
# Function: Get last row + 5-day avg volume
# ---------------------------------------------------------
def process_symbol(group):
    last_row = group.iloc[-1].copy()
    window = group.tail(5)['volume']
    last_row['Avolume'] = window.mean()
    return last_row

latest_df = df.groupby('symbol').apply(process_symbol).reset_index(drop=True)

# ---------------------------------------------------------
# Turnover Ratio (TR) = value_traded / marketCap (rounded 2 decimals)
# ---------------------------------------------------------
latest_df['TR'] = (latest_df['value'] / latest_df['marketCap']).round(2)

# ---------------------------------------------------------
# Liquidity Rating Function (master table অনুযায়ী)
# ---------------------------------------------------------
def liquidity_rating(price, mcap, volume, value):
    # value raw টাকা, thresholds কোটি টাকায় দেওয়া হয়েছে → convert to crore
    value_cr = value / 1e7  

    # ---------------- PRICE 1–5 ----------------
    if 1 <= price <= 5:
        if mcap < 100000000 and volume >= 1500000 and value_cr >= 5:
            return "Excellent"
        if 100000000 <= mcap < 300000000 and volume >= 800000 and value_cr >= 3:
            return "Good"
        if 300000000 <= mcap < 600000000 and volume >= 400000 and value_cr >= 2:
            return "Moderate"
        if mcap >= 600000000 and volume >= 200000 and value_cr >= 1:
            return "Poor"
        if volume < 200000 or value_cr < 1:
            return "Avoid"

    # ---------------- PRICE 5–10 ----------------
    if 5 < price <= 10:
        if mcap < 100000000 and volume >= 1000000 and value_cr >= 4:
            return "Excellent"
        if 100000000 <= mcap < 300000000 and volume >= 600000 and value_cr >= 3:
            return "Good"
        if 300000000 <= mcap < 600000000 and volume >= 300000 and value_cr >= 1.5:
            return "Moderate"
        if mcap >= 600000000 and volume >= 100000 and value_cr >= 1:
            return "Poor"
        if volume < 100000 or value_cr < 1:
            return "Avoid"

    # ---------------- PRICE 10–20 ----------------
    if 10 < price <= 20:
        if mcap < 100000000 and volume >= 800000 and value_cr >= 4:
            return "Excellent"
        if 100000000 <= mcap < 300000000 and volume >= 500000 and value_cr >= 2:
            return "Good"
        if 300000000 <= mcap < 600000000 and volume >= 200000 and value_cr >= 1:
            return "Moderate"
        if mcap >= 600000000 and volume >= 100000 and value_cr >= 0.5:
            return "Poor"
        if volume < 100000 or value_cr < 0.5:
            return "Avoid"

    # ---------------- PRICE 20–40 ----------------
    if 20 < price <= 40:
        if mcap < 150000000 and volume >= 400000 and value_cr >= 3:
            return "Excellent"
        if 150000000 <= mcap < 300000000 and volume >= 300000 and value_cr >= 2:
            return "Good"
        if 300000000 <= mcap < 600000000 and volume >= 100000 and value_cr >= 1:
            return "Moderate"
        if mcap >= 600000000 and volume >= 80000 and value_cr >= 0.5:
            return "Poor"
        if volume < 80000 or value_cr < 0.5:
            return "Avoid"

    # ---------------- PRICE 40–80 ----------------
    if 40 < price <= 80:
        if mcap < 200000000 and volume >= 250000 and value_cr >= 2:
            return "Excellent"
        if 200000000 <= mcap < 400000000 and volume >= 200000 and value_cr >= 1.5:
            return "Good"
        if 400000000 <= mcap < 800000000 and volume >= 100000 and value_cr >= 1:
            return "Moderate"
        if mcap >= 800000000 and volume >= 50000 and value_cr >= 0.5:
            return "Poor"
        if volume < 50000 or value_cr < 0.5:
            return "Avoid"

    # ---------------- PRICE 80–200 ----------------
    if 80 < price <= 200:
        if mcap < 300000000 and volume >= 150000 and value_cr >= 2:
            return "Excellent"
        if 300000000 <= mcap < 600000000 and volume >= 100000 and value_cr >= 1.5:
            return "Good"
        if 600000000 <= mcap < 1000000000 and volume >= 70000 and value_cr >= 1:
            return "Moderate"
        if mcap >= 1000000000 and volume >= 40000 and value_cr >= 0.5:
            return "Poor"
        if volume < 40000 or value_cr < 0.5:
            return "Avoid"

    # ---------------- PRICE 200–500 ----------------
    if 200 < price <= 500:
        if mcap < 400000000 and volume >= 80000 and value_cr >= 2:
            return "Excellent"
        if 400000000 <= mcap < 800000000 and volume >= 60000 and value_cr >= 1.5:
            return "Good"
        if 800000000 <= mcap < 1200000000 and volume >= 40000 and value_cr >= 1:
            return "Moderate"
        if mcap >= 1200000000 and volume >= 20000 and value_cr >= 0.5:
            return "Poor"
        if volume < 20000 or value_cr < 0.5:
            return "Avoid"

    # ---------------- PRICE 500–1000 ----------------
    if 500 < price <= 1000:
        if mcap < 500000000 and volume >= 40000 and value_cr >= 2:
            return "Excellent"
        if 500000000 <= mcap < 1000000000 and volume >= 30000 and value_cr >= 1.5:
            return "Good"
        if 1000000000 <= mcap < 1500000000 and volume >= 15000 and value_cr >= 1:
            return "Moderate"
        if mcap >= 1500000000 and volume >= 8000 and value_cr >= 0.5:
            return "Poor"
        if volume < 8000 or value_cr < 0.5:
            return "Avoid"

    return "Avoid"

# Apply rating
latest_df['liquidity_rating'] = latest_df.apply(
    lambda r: liquidity_rating(r['close'], r['marketCap'], r['volume'], r['value']),
    axis=1
)

# ---------------------------------------------------------
# Final Output
# ---------------------------------------------------------
latest_df['No'] = range(1, len(latest_df) + 1)
latest_df['price'] = latest_df['close']
latest_df['value_traded'] = latest_df['value']

# এখানে mcap = raw marketCap
final_df = latest_df[['No', 'date', 'symbol', 'price', 'Avolume', 'TR',
                      'marketCap', 'volume', 'value_traded', 'liquidity_rating']]

# ---------------------------------------------------------
# Save to both locations
# ---------------------------------------------------------
os.makedirs("./csv", exist_ok=True)
os.makedirs("./output/ai_signal", exist_ok=True)

final_df.to_csv("./csv/liquidity.csv", index=False)
final_df.to_csv("./output/ai_signal/liquidity.csv", index=False)

print("✅ liquidity.csv files created successfully in both locations!")
print(final_df.head())