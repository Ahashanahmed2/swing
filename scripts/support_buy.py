import pandas as pd
import os

# File paths
sr_path = './csv/support_resistance.csv'
mongo_path = './csv/mongodb.csv'
output_path = './output/ai_signal/support_resistant.csv'

# Ensure output directory exists
os.makedirs('./output/ai_signal', exist_ok=True)

# Load data
sr_df = pd.read_csv(sr_path)
mongo_df = pd.read_csv(mongo_path)

# Convert date columns
sr_df['current_date'] = pd.to_datetime(sr_df['current_date'])
mongo_df['date'] = pd.to_datetime(mongo_df['date'])

# Sort mongodb for safety
mongo_df = mongo_df.sort_values(['symbol', 'date']).reset_index(drop=True)

results = []

# Filter only support type
sr_df = sr_df[sr_df['type'] == 'support']

for _, row in sr_df.iterrows():
    symbol = row['symbol']
    current_date = row['current_date']
    current_low = row['current_low']

    # Find matching row in mongodb
    df_symbol = mongo_df[mongo_df['symbol'] == symbol].reset_index(drop=True)

    match_idx = df_symbol[df_symbol['date'] == current_date].index

    if len(match_idx) == 0:
        continue

    idx = match_idx[0]

    # Check next row exists
    if idx + 1 >= len(df_symbol):
        continue

    next_row = df_symbol.iloc[idx + 1]

    # Condition: next low > current low
    if next_row['low'] > current_low:
        results.append({
            'type': row['type'],
            'symbol': row['symbol'],
            'level_date': row['level_date'],
            'level_price': row['level_price'],
            'gap_days': row['gap_days'],
            'strength': row['strength']
        })

# Save output
output_df = pd.DataFrame(results)
output_df.to_csv(output_path, index=False)

print(f"✅ Done! Saved to {output_path}")