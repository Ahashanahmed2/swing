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
            'symbol': symbol,
            'date': last_row['date'],
            'close': last_row['close'],
            'low': last_row['low'],
            'high': last_row['high']
        })

# Save output
output_df = pd.DataFrame(output_rows)
output_df.to_csv(output_path, index=False)

print(f"Saved {len(output_rows)} signals to {output_path}")