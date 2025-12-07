import pandas as pd
import os

# Paths
rsi_path = './csv/rsi_diver_retest.csv'
mongo_path = './csv/mongodb.csv'
output_path1 = './output/ai_signal/short_buy.csv'
output_path2 = './csv/short_buy.csv'

os.makedirs(os.path.dirname(output_path1), exist_ok=True)
os.makedirs(os.path.dirname(output_path2), exist_ok=True)

# Delete old files
for path in [output_path1, output_path2]:
    if os.path.exists(path):
        os.remove(path)

# Read data
rsi_df = pd.read_csv(rsi_path)
mongo_df = pd.read_csv(mongo_path)

# Clean column names
rsi_df.columns = rsi_df.columns.str.replace(" ", "_")

# Ensure MongoDB date is datetime
mongo_df['date'] = pd.to_datetime(mongo_df['date'], errors='coerce')
mongo_df = mongo_df.dropna(subset=['date'])
mongo_groups = mongo_df.groupby('symbol')

output_rows = []

for _, rsi_row in rsi_df.iterrows():
    symbol = str(rsi_row['symbol']).strip().upper()
    last_high = rsi_row['last_row_high']
    last_row_date = pd.to_datetime(rsi_row['last_row_date'], errors='coerce')
    last_row_low = rsi_row['last_row_low']

    if pd.isna(last_row_date) or symbol not in mongo_groups.groups:
        continue

    symbol_group = mongo_groups.get_group(symbol).sort_values('date')
    last_row = symbol_group.iloc[-1]

    if not (last_row['close'] > last_high):
        continue

    prev_rows = symbol_group[symbol_group['date'] < last_row_date]
    if prev_rows.empty:
        continue
    prev_row = prev_rows.iloc[-1]

    if not (last_row_low > prev_row['low'] and last_row['close'] > prev_row['high']):
        continue

    pre_candle = None
    for i in range(len(prev_rows)-1, -1, -1):
        row = prev_rows.iloc[i]
        if (row['low'] <= last_high <= row['high']) or (row['low'] <= last_row_low <= row['high']):
            pre_candle = row
            break
    if pre_candle is None:
        continue

    between_rows = symbol_group[
        (symbol_group['date'] > pre_candle['date']) & 
        (symbol_group['date'] < last_row_date)
    ]
    if between_rows.empty:
        continue

    low_candle = between_rows.loc[between_rows['low'].idxmin()]
    SL_price = low_candle['low']
    buy_price = last_row['close']

    output_rows.append({
        'symbol': symbol,
        'date': last_row['date'].date(),
        'buy': buy_price,
        'SL': SL_price,
    })

# Build standardized DataFrame
if output_rows:
    df = pd.DataFrame(output_rows)
    df = df.sort_values(['SL', 'symbol'], ascending=[True, True]).reset_index(drop=True)
    df.insert(0, 'No', range(1, len(df) + 1))
    df['buy'] = pd.to_numeric(df['buy'], errors='coerce')
    df['SL'] = pd.to_numeric(df['SL'], errors='coerce')
else:
    df = pd.DataFrame(columns=['No', 'symbol', 'date', 'buy', 'SL'])

# Save
df.to_csv(output_path1, index=False)
df.to_csv(output_path2, index=False)

print(f"✅ short_buy.csv saved ({len(df)} rows)")