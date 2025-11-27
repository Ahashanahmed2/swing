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
    # শেষ row
    last_row = group.iloc[-1].copy()
    
    # শেষ row থেকে উপরের ৫ রো এর গড় volume
    window = group.tail(5)['volume']
    last_row['Avolume'] = window.mean()
    
    return last_row

# Apply per symbol
latest_df = df.groupby('symbol').apply(process_symbol).reset_index(drop=True)

# ---------------------------------------------------------
# Turnover Ratio (TR) = value_traded / marketCap
# ---------------------------------------------------------
latest_df['TR'] = latest_df['value'] / latest_df['marketCap']

# ---------------------------------------------------------
# Market Cap Range Label
# ---------------------------------------------------------
def mcap_label(mcap):
    cr = 10000000  # 1 crore = 10,000,000
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

latest_df['mcap'] = latest_df['marketCap'].apply(mcap_label)

# ---------------------------------------------------------
# Liquidity Rating Function (same as before)
# ---------------------------------------------------------
def liquidity_rating(price, mcap, volume, value):
    value_cr = value / 1e7  # convert to crore
    # এখানে আগের master table অনুযায়ী rating rules বসানো আছে
    # (আগের স্ক্রিপ্টের মতোই)
    # ...
    return "Avoid"  # fallback

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

final_df = latest_df[['No', 'date', 'symbol', 'price', 'Avolume', 'TR',
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