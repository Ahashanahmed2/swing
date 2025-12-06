import pandas as pd
import os

# Paths
rsi_path = './csv/rsi_diver_retest.csv'
mongo_path = './csv/mongodb.csv'
output_path1 = './output/ai_signal/short_buy.csv'
output_path2 = './csv/short_buy.csv'

# Ensure output directories exist
os.makedirs(os.path.dirname(output_path1), exist_ok=True)
os.makedirs(os.path.dirname(output_path2), exist_ok=True)

# -------------------------------
# আগের short_buy.csv ডিলিট করা
# -------------------------------
if os.path.exists(output_path1):
    os.remove(output_path1)

if os.path.exists(output_path2):
    os.remove(output_path2)

# Read data
rsi_df = pd.read_csv(rsi_path)
mongo_df = pd.read_csv(mongo_path)

# কলাম নাম fix
rsi_df.columns = rsi_df.columns.str.replace(" ", "_")

# Group mongodb data by symbol
mongo_groups = mongo_df.groupby('symbol')

output_rows = []

# Iterate through RSI symbols
for _, rsi_row in rsi_df.iterrows():
    symbol = rsi_row['symbol']
    last_high = rsi_row['last_row_high']
    last_row_date = rsi_row['last_row_date']
    last_row_low = rsi_row['last_row_low']
    last_row_rsi = rsi_row['last_row_rsi']
    second_row_date = rsi_row['second_row_date']
    second_row_low = rsi_row['second_row_low']
    second_row_rsi = rsi_row['second_row_rsi']

    if symbol not in mongo_groups.groups:
        continue

    symbol_group = mongo_groups.get_group(symbol).sort_values(by='date')

    last_row = symbol_group.iloc[-1]

    if last_row['close'] > last_high:

        prev_rows = symbol_group[symbol_group['date'] < last_row_date]
        if prev_rows.empty:
            continue

        prev_row = prev_rows.iloc[-1]

        if (last_row_low > prev_row['low']) and (last_row['close'] > prev_row['high']):

            pre_candle = None
            for i in range(len(prev_rows)-1, -1, -1):
                row = prev_rows.iloc[i]
                if row['low'] <= last_high <= row['high'] or row['low'] <= last_row_low <= row['high']:
                    pre_candle = row
                    break

            if pre_candle is None:
                continue

            pre_candle_date = pre_candle['date']

            between_rows = symbol_group[(symbol_group['date'] > pre_candle_date) & (symbol_group['date'] < last_row_date)]
            if between_rows.empty:
                continue

            low_candle = between_rows.loc[between_rows['low'].idxmin()]
            low_candle_date = low_candle['date']
            SL = low_candle['low']

            candles_between = symbol_group[(symbol_group['date'] > low_candle_date) & (symbol_group['date'] < last_row_date)]
            candle_count = len(candles_between)

            output_rows.append({
                'symbol': symbol,
                'date': last_row['date'],
                'close': last_row['close'],
                'low': last_row['low'],
                'high': last_row['high'],
                'last_row_date': last_row_date,
                'last_row_low': last_row_low,
                'last_row_high': last_high,
                'last_row_rsi': last_row_rsi,
                'second_row_date': second_row_date,
                'second_row_low': second_row_low,
                'second_row_rsi': second_row_rsi,
                'pre_candle_date': pre_candle_date,
                'low_candle_date': low_candle_date,
                'candle_count': candle_count + 1,
                'SL': SL
            })

# Create DataFrame
output_df = pd.DataFrame(output_rows)

if not output_df.empty:
    output_df = output_df.sort_values(by=['SL', 'candle_count'], ascending=[True, True]).reset_index(drop=True)
    output_df.insert(0, 'id', range(1, len(output_df) + 1))

# Save outputs
output_df.to_csv(output_path1, index=False)
output_df.to_csv(output_path2, index=False)

print(f"✅ Saved {len(output_rows)} signals to {output_path1} and {output_path2} (sorted by SL and candle_count)")