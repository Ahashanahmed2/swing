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
# 5-day Average Volume (Avolume)
# ---------------------------------------------------------
df['Avolume'] = df.groupby('symbol')['volume'].rolling(5).mean().reset_index(0, drop=True)

# ---------------------------------------------------------
# Turnover Ratio (TR) = value_traded / marketCap
# ---------------------------------------------------------
df['TR'] = df['value'] / df['marketCap']

# ---------------------------------------------------------
# Market Cap Range Label
# ---------------------------------------------------------
def mcap_label(mcap):
    cr = 1e7  # 1 crore = 1e7
    if mcap < 100 * cr:
        return "<100cr"
    elif mcap < 300 * cr:
        return "100–300cr"
    elif mcap < 600 * cr:
        return "300–600cr"
    elif mcap < 1000 * cr:
        return "600–1000cr"
    elif mcap < 1500 * cr:
        return "1000–1500cr"
    else:
        return "1500+"

df['mcap'] = df['marketCap'].apply(mcap_label)

# ---------------------------------------------------------
# Liquidity Rating Function
# ---------------------------------------------------------
def liquidity_rating(price, mcap, volume, value):
    value_cr = value / 1e7  # convert to crore

    # ---------------- PRICE 1–5 ----------------
    if 1 <= price <= 5:
        if mcap == "<100cr" and volume >= 1500000 and value_cr >= 5:
            return "Excellent"
        if mcap == "100–300cr" and volume >= 800000 and value_cr >= 3:
            return "Good"
        if mcap == "300–600cr" and volume >= 400000 and value_cr >= 2:
            return "Moderate"
        if mcap == "600–1000cr" and volume >= 200000 and value_cr >= 1:
            return "Poor"
        return "Avoid"

    # ---------------- PRICE 5–10 ----------------
    if 5 < price <= 10:
        if mcap == "<100cr" and volume >= 1000000 and value_cr >= 4:
            return "Excellent"
        if mcap == "100–300cr" and volume >= 600000 and value_cr >= 3:
            return "Good"
        if mcap == "300–600cr" and volume >= 300000 and value_cr >= 1.5:
            return "Moderate"
        if mcap == "600–1000cr" and volume >= 100000 and value_cr >= 1:
            return "Poor"
        return "Avoid"

    # ---------------- PRICE 10–20 ----------------
    if 10 < price <= 20:
        if mcap == "<100cr" and volume >= 800000 and value_cr >= 4:
            return "Excellent"
        if mcap == "100–300cr" and volume >= 500000 and value_cr >= 2:
            return "Good"
        if mcap == "300–600cr" and volume >= 200000 and value_cr >= 1:
            return "Moderate"
        if mcap == "600–1000cr" and volume >= 100000 and value_cr >= 0.5:
            return "Poor"
        return "Avoid"

    # ---------------- PRICE 20–40 ----------------
    if 20 < price <= 40:
        if mcap == "<150cr" and volume >= 400000 and value_cr >= 3:
            return "Excellent"
        if mcap == "150–300cr" and volume >= 300000 and value_cr >= 2:
            return "Good"
        if mcap == "300–600cr" and volume >= 100000 and value_cr >= 1:
            return "Moderate"
        if mcap == "600–1000cr" and volume >= 80000 and value_cr >= 0.5:
            return "Poor"
        return "Avoid"

    # ---------------- PRICE 40–80 ----------------
    if 40 < price <= 80:
        if mcap == "<200cr" and volume >= 250000 and value_cr >= 2:
            return "Excellent"
        if mcap == "200–400cr" and volume >= 200000 and value_cr >= 1.5:
            return "Good"
        if mcap == "400–800cr" and volume >= 100000 and value_cr >= 1:
            return "Moderate"
        if mcap == "800–1000cr" and volume >= 50000 and value_cr >= 0.5:
            return "Poor"
        return "Avoid"

    # ---------------- PRICE 80–200 ----------------
    if 80 < price <= 200:
        if mcap == "<300cr" and volume >= 150000 and value_cr >= 2:
            return "Excellent"
        if mcap == "300–600cr" and volume >= 100000 and value_cr >= 1.5:
            return "Good"
        if mcap == "600–1000cr" and volume >= 70000 and value_cr >= 1:
            return "Moderate"
        if mcap == "1000–1500cr" and volume >= 40000 and value_cr >= 0.5:
            return "Poor"
        return "Avoid"

    # ---------------- PRICE 200–500 ----------------
    if 200 < price <= 500:
        if mcap == "<400cr" and volume >= 80000 and value_cr >= 2:
            return "Excellent"
        if mcap == "400–800cr" and volume >= 60000 and value_cr >= 1.5:
            return "Good"
        if mcap == "800–1200cr" and volume >= 40000 and value_cr >= 1:
            return "Moderate"
        if mcap == "1200–1500cr" and volume >= 20000 and value_cr >= 0.5:
            return "Poor"
        return "Avoid"

    # ---------------- PRICE 500–1000 ----------------
    if 500 < price <= 1000:
        if mcap == "<500cr" and volume >= 40000 and value_cr >= 2:
            return "Excellent"
        if mcap == "500–1000cr" and volume >= 30000 and value_cr >= 1.5:
            return "Good"
        if mcap == "1000–1500cr" and volume >= 15000 and value_cr >= 1:
            return "Moderate"
        if mcap == "1500+" and volume >= 8000 and value_cr >= 0.5:
            return "Poor"
        return "Avoid"

    return "Avoid"


# Apply rating
df['liquidity_rating'] = df.apply(
    lambda r: liquidity_rating(r['close'], r['mcap'], r['volume'], r['value']),
    axis=1
)

# ---------------------------------------------------------
# Final Output
# ---------------------------------------------------------
df['No'] = range(1, len(df) + 1)
df['price'] = df['close']
df['value_traded'] = df['value']

final_df = df[['No', 'date', 'symbol', 'price', 'Avolume', 'TR',
               'mcap', 'volume', 'value_traded', 'liquidity_rating']]

# ---------------------------------------------------------
# Save to both locations
# ---------------------------------------------------------
os.makedirs("./csv", exist_ok=True)
os.makedirs("./output/ai_signal", exist_ok=True)

final_df.to_csv("./csv/liquidity.csv", index=False)
final_df.to_csv("./output/ai_signal/liquidity.csv", index=False)

print("✅ liquidity.csv files created successfully in both locations!")
print(final_df.head())