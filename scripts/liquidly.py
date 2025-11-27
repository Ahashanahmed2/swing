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
# Market Cap কে কোটি হিসাবে রূপান্তর করা
# ---------------------------------------------------------
def mcap_in_crore(mcap_million):
    # CSV থেকে আসা marketCap মিলিয়ন এ আছে → কোটি তে রূপান্তর
    return round(mcap_million / 10, 2)

latest_df['mcap_crore'] = latest_df['marketCap'].apply(mcap_in_crore)

# ---------------------------------------------------------
# Liquidity Rating Function (সব Price Range বসানো হয়েছে)
# ---------------------------------------------------------
def liquidity_rating(price, mcap_crore, volume, value):
    value_cr = value / 1e7  # কোটি টাকায় কনভার্ট

    # ---------------- PRICE 1–5 ----------------
    if 1 <= price <= 5:
        if mcap_crore < 100 and volume >= 1500000 and value_cr >= 5:
            return "Excellent"
        if 100 <= mcap_crore < 300 and volume >= 800000 and value_cr >= 3:
            return "Good"
        if 300 <= mcap_crore < 600 and volume >= 400000 and value_cr >= 2:
            return "Moderate"
        if mcap_crore >= 600 and volume >= 200000 and value_cr >= 1:
            return "Poor"
        if volume < 200000 or value_cr < 1:
            return "Avoid"

    # ---------------- PRICE 5–10 ----------------
    if 5 < price <= 10:
        if mcap_crore < 100 and volume >= 1000000 and value_cr >= 4:
            return "Excellent"
        if 100 <= mcap_crore < 300 and volume >= 600000 and value_cr >= 3:
            return "Good"
        if 300 <= mcap_crore < 600 and volume >= 300000 and value_cr >= 1.5:
            return "Moderate"
        if mcap_crore >= 600 and volume >= 100000 and value_cr >= 1:
            return "Poor"
        if volume < 100000 or value_cr < 1:
            return "Avoid"

    # ---------------- PRICE 10–20 ----------------
    if 10 < price <= 20:
        if mcap_crore < 100 and volume >= 800000 and value_cr >= 4:
            return "Excellent"
        if 100 <= mcap_crore < 300 and volume >= 500000 and value_cr >= 2:
            return "Good"
        if 300 <= mcap_crore < 600 and volume >= 200000 and value_cr >= 1:
            return "Moderate"
        if mcap_crore >= 600 and volume >= 100000 and value_cr >= 0.5:
            return "Poor"
        if volume < 100000 or value_cr < 0.5:
            return "Avoid"

    # ---------------- PRICE 20–40 ----------------
    if 20 < price <= 40:
        if mcap_crore < 150 and volume >= 400000 and value_cr >= 3:
            return "Excellent"
        if 150 <= mcap_crore < 300 and volume >= 300000 and value_cr >= 2:
            return "Good"
        if 300 <= mcap_crore < 600 and volume >= 100000 and value_cr >= 1:
            return "Moderate"
        if mcap_crore >= 600 and volume >= 80000 and value_cr >= 0.5:
            return "Poor"
        if volume < 80000 or value_cr < 0.5:
            return "Avoid"

    # ---------------- PRICE 40–80 ----------------
    if 40 < price <= 80:
        if mcap_crore < 200 and volume >= 250000 and value_cr >= 2:
            return "Excellent"
        if 200 <= mcap_crore < 400 and volume >= 200000 and value_cr >= 1.5:
            return "Good"
        if 400 <= mcap_crore < 800 and volume >= 100000 and value_cr >= 1:
            return "Moderate"
        if mcap_crore >= 800 and volume >= 50000 and value_cr >= 0.5:
            return "Poor"
        if volume < 50000 or value_cr < 0.5:
            return "Avoid"

    # ---------------- PRICE 80–200 ----------------
    if 80 < price <= 200:
        if mcap_crore < 300 and volume >= 150000 and value_cr >= 2:
            return "Excellent"
        if 300 <= mcap_crore < 600 and volume >= 100000 and value_cr >= 1.5:
            return "Good"
        if 600 <= mcap_crore < 1000 and volume >= 70000 and value_cr >= 1:
            return "Moderate"
        if mcap_crore >= 1000 and volume >= 40000 and value_cr >= 0.5:
            return "Poor"
        if volume < 40000 or value_cr < 0.5:
            return "Avoid"

    # ---------------- PRICE 200–500 ----------------
    if 200 < price <= 500:
        if mcap_crore < 400 and volume >= 80000 and value_cr >= 2:
            return "Excellent"
        if 400 <= mcap_crore < 800 and volume >= 60000 and value_cr >= 1.5:
            return "Good"
        if 800 <= mcap_crore < 1200 and volume >= 40000 and value_cr >= 1:
            return "Moderate"
        if mcap_crore >= 1200 and volume >= 20000 and value_cr >= 0.5:
            return "Poor"
        if volume < 20000 or value_cr < 0.5:
            return "Avoid"

    # ---------------- PRICE 500–1000 ----------------
    if 500 < price <= 1000:
        if mcap_crore < 500 and volume >= 40000 and value_cr >= 2:
            return "Excellent"
        if 500 <= mcap_crore < 1000 and volume >= 30000 and value_cr >= 1.5:
            return "Good"
        if 1000 <= mcap_crore < 1500 and volume >= 15000 and value_cr >= 1:
            return "Moderate"
        if mcap_crore >= 1500 and volume >= 8000 and value_cr >= 0.5:
            return "Poor"
        if volume < 8000 or value_cr < 0.5:
            return "Avoid"

    return "Avoid"

# ---------------------------------------------------------
# Apply rating
# ---------------------------------------------------------
latest_df['liquidity_rating'] = latest_df.apply(
    lambda r: liquidity_rating(r['close'], r['mcap_crore'], r['volume'], r['value']),
    axis=1
)

# ---------------------------------------------------------
# Final Output
# ---------------------------------------------------------
latest_df['No'] = range(1, len(latest_df) + 1)
latest_df['price'] = latest_df['close']
latest_df['value_traded'] = latest_df['value']

final_df = latest_df[['No', 'date', 'symbol', 'price', 'Avolume', 'TR',
                      'mcap_crore', 'volume', 'value_traded', 'liquidity_rating']]

# ---------------------------------------------------------
# Save to both locations
# ---------------------------------------------------------
os.makedirs("./csv", exist_ok=True)
os.makedirs("./output/ai_signal", exist_ok=True)

final_df.to_csv("./csv/liquidity.csv", index=False)
final_df.to_csv("./output/ai_signal/liquidity.csv", index=False)

print("✅ liquidity.csv files created successfully in both locations!")
print(final_df.head())