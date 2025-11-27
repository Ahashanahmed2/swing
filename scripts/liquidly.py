
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
# Market Cap Range Label
# ---------------------------------------------------------
def mcap_label(mcap):
    cr = 10000000  # 1 crore = 10,000,000
    if mcap < 100 * cr:
        return "<100cr"
    elif mcap < 150 * cr:
        return "<150cr"
    elif mcap < 200 * cr:
        return "<200cr"
    elif mcap < 300 * cr:
        return "100–300cr"
    elif mcap < 400 * cr:
        return "200–400cr"
    elif mcap < 500 * cr:
        return "<500cr"
    elif mcap < 600 * cr:
        return "300–600cr"
    elif mcap < 800 * cr:
        return "400–800cr"
    elif mcap < 1000 * cr:
        return "600–1000cr"
    elif mcap < 1200 * cr:
        return "800–1200cr"
    elif mcap < 1500 * cr:
        return "1000–1500cr"
    else:
        return "1500+"

latest_df['mcap'] = latest_df['marketCap'].apply(mcap_label)

# ---------------------------------------------------------
# Liquidity Rating Function (master table অনুযায়ী)
# ---------------------------------------------------------
def liquidity_rating(price, mcap, volume, value):
    # value raw টাকা, thresholds কোটি টাকায় দেওয়া হয়েছে → convert to crore
    value_cr = value / 1e7  

    # ---------------- PRICE 1–5 ----------------
    if 1 <= price <= 5:
        if mcap == "<100cr" and volume >= 1500000 and value_cr >= 5:
            return "Excellent"
        if mcap == "100–300cr" and volume >= 800000 and value_cr >= 3:
            return "Good"
        if mcap == "300–600cr" and volume >= 400000 and value_cr >= 2:
            return "Moderate"
        if "600" in mcap or "+" in mcap and volume >= 200000 and value_cr >= 1:
            return "Poor"
        if volume < 200000 or value_cr < 1:
            return "Avoid"

    # ---------------- PRICE 5–10 ----------------
    if 5 < price <= 10:
        if mcap == "<100cr" and volume >= 1000000 and value_cr >= 4:
            return "Excellent"
        if mcap == "100–300cr" and volume >= 600000 and value_cr >= 3:
            return "Good"
        if mcap == "300–600cr" and volume >= 300000 and value_cr >= 1.5:
            return "Moderate"
        if "600" in mcap or "+" in mcap and volume >= 100000 and value_cr >= 1:
            return "Poor"
        if volume < 100000 or value_cr < 1:
            return "Avoid"

    # ---------------- PRICE 10–20 ----------------
    if 10 < price <= 20:
        if mcap == "<100cr" and volume >= 800000 and value_cr >= 4:
            return "Excellent"
        if mcap == "100–300cr" and volume >= 500000 and value_cr >= 2:
            return "Good"
        if mcap == "300–600cr" and volume >= 200000 and value_cr >= 1:
            return "Moderate"
        if "600" in mcap or "+" in mcap and volume >= 100000 and value_cr >= 0.5:
            return "Poor"
        if volume < 100000 or value_cr < 0.5:
            return "Avoid"

    # ---------------- PRICE 20–40 ----------------
    if 20 < price <= 40:
        if mcap == "<150cr" and volume >= 400000 and value_cr >= 3:
            return "Excellent"
        if mcap == "150–300cr" and volume >= 300000 and value_cr >= 2:
            return "Good"
        if mcap == "300–600cr" and volume >= 100000 and value_cr >= 1:
            return "Moderate"
        if "600" in mcap or "+" in mcap and volume >= 80000 and value_cr >= 0.5:
            return "Poor"
        if volume < 80000 or value_cr < 0.5:
            return "Avoid"

    # ---------------- PRICE 40–80 ----------------
    if 40 < price <= 80:
        if mcap == "<200cr" and volume >= 250000 and value_cr >= 2:
            return "Excellent"
        if mcap == "200–400cr" and volume >= 200000 and value_cr >= 1.5:
            return "Good"
        if mcap == "400–800cr" and volume >= 100000 and value_cr >= 1:
            return "Moderate"
        if "800" in mcap or "+" in mcap and volume >= 50000 and value_cr >= 0.5:
            return "Poor"
        if volume < 50000 or value_cr < 0.5:
            return "Avoid"

    # ---------------- PRICE 80–200 ----------------
    if 80 < price <= 200:
        if mcap == "<300cr" and volume >= 150000 and value_cr >= 2:
            return "Excellent"
        if mcap == "300–600cr" and volume >= 100000 and value_cr >= 1.5:
            return "Good"
        if mcap == "600–1000cr" and volume >= 70000 and value_cr >= 1:
            return "Moderate"
        if "1000" in mcap or "+" in mcap and volume >= 40000 and value_cr >= 0.5:
            return "Poor"
        if volume < 40000 or value_cr < 0.5:
            return "Avoid"

    # ---------------- PRICE 200–500 ----------------
    if 200 < price <= 500:
        if mcap == "<400cr" and volume >= 80000 and value_cr >= 2:
            return "Excellent"
        if mcap == "400–800cr" and volume >= 60000 and value_cr >= 1.5:
            return "Good"
        if mcap == "800–1200cr" and volume >= 40000 and value_cr >= 1:
            return "Moderate"
        if "1200" in mcap or "+" in mcap and volume >= 20000 and value_cr >= 0.5:
            return "Poor"
        if volume < 20000 or value_cr < 0.5:
            return "Avoid"

    # ---------------- PRICE 500–1000 ----------------
    if 500 < price <= 1000:
        if mcap == "<500cr" and volume >= 40000 and value_cr >= 2:
            return "Excellent"
        if mcap == "500–1000cr" and volume >= 30000 and value_cr >= 1.5:
            return "Good"
        if mcap == "1000–1500cr" and volume >= 15000 and value_cr >= 1:
            return "Moderate"
        if "1500" in mcap or "+" in mcap and volume >= 8000 and value_cr >= 0.5:
            return "Poor"
        if volume < 8000 or value_cr < 0.5:
            return "Avoid"

    return "Avoid"

# Apply rating
latest_df['liquidity_rating'] = latest_df.apply(
    lambda r: liquidity_rating(r['close'], r['mcap'], r['volume'], r['value']),
    axis=1
)

# ---------------------------------------------------------
# Final Output
# ---------------------------------------------------------
latest_df['No'] = range(1, len(latest_df) + 1)
latest_df['price'] = latest_df['close']
latest_df['value_traded'] = latest_df['value']

final_df = latest_df[['No', 'date', 'symbol', 'price', 'Avolume', '