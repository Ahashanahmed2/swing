import pandas as pd
import numpy as np
import os

# =========================
# Configuration
# =========================
INPUT_FILE = './csv/mongodb.csv'
OUTPUT_DIR = './output/ai_signal'
# OUTPUT_FILE = f'{OUTPUT_DIR}/rsi_diver.csv'  # কমেন্ট করা হয়েছে

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load and prepare data
# =========================
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'])

# Ensure required columns exist
required_cols = ['symbol', 'date', 'low', 'high', 'rsi']
for col in required_cols:
    if col not in df.columns:
        print(f"⚠️ Missing column: {col}")
        if col == 'rsi':
            # Calculate RSI if not present
            delta = df['close'].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        else:
            df[col] = 0

df = df[required_cols]

# =========================
# Divergence Detection
# =========================
divergence_dict = {}

for symbol, group in df.groupby('symbol'):
    # Sort descending by date (latest first)
    group_sorted = group.sort_values(by='date', ascending=False).reset_index(drop=True)

    if len(group_sorted) < 2:
        continue

    last_row = group_sorted.iloc[0]
    last_date = last_row['date']
    last_low = last_row['low']
    last_high = last_row['high']
    last_rsi = last_row['rsi']

    # Find divergence
    for i in range(1, min(len(group_sorted), 20)):  # Check last 20 candles only
        upper_row = group_sorted.iloc[i]
        upper_date = upper_row['date']
        upper_low = upper_row['low']
        upper_high = upper_row['high']
        upper_rsi = upper_row['rsi']

        # ========== Bullish Divergence ==========
        # Price makes lower low, RSI makes higher low
        if (
            last_rsi >= upper_rsi and
            last_low <= upper_low and
            last_date > upper_date
        ):
            # Calculate line equation: y = m*x + c
            x1 = upper_date.timestamp()
            y1 = upper_low
            x2 = last_date.timestamp()
            y2 = last_low

            m = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            c = y1 - m * x1

            # Check if any candle low breaks the line
            mask = (group_sorted['date'] >= upper_date) & (group_sorted['date'] <= last_date)
            segment = group_sorted[mask]

            broken = False
            for _, row in segment.iterrows():
                x = row['date'].timestamp()
                expected_low = m * x + c
                if row['low'] < expected_low:
                    broken = True
                    break

            if not broken:
                divergence_dict[f"{symbol}_bullish"] = {
                    'symbol': symbol,
                    'divergence_type': 'Bullish',
                    'last_date': last_date.strftime('%Y-%m-%d'),
                    'last_price': last_low,
                    'last_high': last_high,
                    'last_rsi': last_rsi,
                    'previous_date': upper_date.strftime('%Y-%m-%d'),
                    'previous_price': upper_low,
                    'previous_rsi': upper_rsi,
                    'strength': 'Strong' if (last_rsi - upper_rsi) > 5 else 'Moderate'
                }
            break  # Take first valid match

        # ========== Bearish Divergence ==========
        # Price makes higher high, RSI makes lower high
        elif (
            last_rsi <= upper_rsi and
            last_high >= upper_high and
            last_date > upper_date
        ):
            # Calculate line for highs
            x1 = upper_date.timestamp()
            y1 = upper_high
            x2 = last_date.timestamp()
            y2 = last_high

            m = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            c = y1 - m * x1

            # Check if any candle high breaks the line
            mask = (group_sorted['date'] >= upper_date) & (group_sorted['date'] <= last_date)
            segment = group_sorted[mask]

            broken = False
            for _, row in segment.iterrows():
                x = row['date'].timestamp()
                expected_high = m * x + c
                if row['high'] > expected_high:
                    broken = True
                    break

            if not broken:
                divergence_dict[f"{symbol}_bearish"] = {
                    'symbol': symbol,
                    'divergence_type': 'Bearish',
                    'last_date': last_date.strftime('%Y-%m-%d'),
                    'last_price': last_high,
                    'last_low': last_low,
                    'last_rsi': last_rsi,
                    'previous_date': upper_date.strftime('%Y-%m-%d'),
                    'previous_price': upper_high,
                    'previous_rsi': upper_rsi,
                    'strength': 'Strong' if (upper_rsi - last_rsi) > 5 else 'Moderate'
                }
            break  # Take first valid match

# =========================
# Save Output
# =========================
if divergence_dict:
    output_df = pd.DataFrame(divergence_dict.values())
    output_df = output_df.sort_values('symbol')

    # Reorder columns
    column_order = [
        'symbol', 'divergence_type', 'strength',
        'last_date', 'last_price', 'last_high', 'last_rsi',
        'previous_date', 'previous_price', 'previous_rsi'
    ]
    output_df = output_df[column_order]

    # output_df.to_csv(OUTPUT_FILE, index=False)  # কমেন্ট করা হয়েছে
    print(f"✅ Found {len(output_df)} divergences")
    # print(f"📊 Saved to: {OUTPUT_FILE}")  # কমেন্ট করা হয়েছে

    # Print summary
    print("\n📈 Divergence Summary:")
    print("-" * 50)
    for _, row in output_df.iterrows():
        print(f"{row['symbol']:<12} {row['divergence_type']:<8} {row['strength']:<8} Last: {row['last_date']}")

else:
    print("⚠️ No divergences found")
    # Create empty CSV with headers
    empty_df = pd.DataFrame(columns=['symbol', 'divergence_type', 'strength', 'last_date', 'last_price', 'last_high', 'last_rsi', 'previous_date', 'previous_price', 'previous_rsi'])
    # empty_df.to_csv(OUTPUT_FILE, index=False)  # কমেন্ট করা হয়েছে
    # print(f"📄 Empty file created: {OUTPUT_FILE}")  # কমেন্ট করা হয়েছে

# =========================
# Also save to old location for compatibility
# =========================
old_output = './csv/rsi_diver.csv'
if divergence_dict:
    output_df.to_csv(old_output, index=False)
else:
    empty_df.to_csv(old_output, index=False)
print(f"✅ Also saved to: {old_output}")