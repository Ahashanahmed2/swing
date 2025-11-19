import pandas as pd
import os

# Paths
rsi_path = './csv/rsi_diver_retest.csv'
mongo_path = './csv/mongodb.csv'
output_path = './output/ai_signal/short_buy.csv'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Read data
rsi_df = pd.read_csv(rsi_path)
mongo_df = pd.read_csv(mongo_path)

# Group mongodb data by symbol
mongo_groups = mongo_df.groupby('symbol')

# Prepare output
output_rows = []

# Iterate through RSI symbols
for _, rsi_row in rsi_df.iterrows():
    symbol = rsi_row['symbol']
    last_high = rsi_row['last row high']

    # Check if symbol exists in mongodb
    if symbol not in mongo_groups.groups:
        continue

    # Get last row of that symbol group
    symbol_group = mongo_groups.get_group(symbol)
    last_row = symbol_group.iloc[-1]

    # Compare close with last row high
    if last_row['close'] > last_high:
        output_rows.append({
            # MongoDB last row info
            'symbol': symbol,
            'date': last_row['date'],
            'close': last_row['close'],
            'low': last_row['low'],
            'high': last_row['high'],
            
            # RSI retest info
            'last_row_date': rsi_row['last row date'],
            'last_row_low': rsi_row['last row low'],
            'last_row_high': rsi_row['last row high'],
            'last_row_rsi': rsi_row['last row rsi'],
            'second_row_date': rsi_row['second row date'],
            'second_row_low': rsi_row['second row low'],
            'second_row_rsi': rsi_row['second row rsi']
        })

# Create DataFrame
output_df = pd.DataFrame(output_rows)

# Convert second_row_date to datetime for proper sorting
output_df['second_row_date'] = pd.to_datetime(output_df['second_row_date'], errors='coerce')

# Sort by symbol first, then by second_row_date (older first)
output_df = output_df.sort_values(by=['symbol', 'second_row_date'], ascending=[True, True])

# Save to CSV
output_df.to_csv(output_path, index=False)

print(f"Saved {len(output_rows)} signals to {output_path} (sorted by symbol and second_row_date)")