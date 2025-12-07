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

    # Skip if critical data missing
    if pd.isna(last_row_date) or symbol not in mongo_groups.groups:
        continue

    symbol_group = mongo_groups.get_group(symbol).sort_values('date')
    last_row = symbol_group.iloc[-1]

    # Condition 1: breakout above last_high
    if not (last_row['close'] > last_high):
        continue

    # Condition 2: get previous row before last_row_date
    prev_rows = symbol_group[symbol_group['date'] < last_row_date]
    if prev_rows.empty:
        continue
    prev_row = prev_rows.iloc[-1]

    # Condition 3: higher low + breakout over prior high
    if not (last_row_low > prev_row['low'] and last_row['close'] > prev_row['high']):
        continue

    # Find pre-candle (touching last_high or last_row_low zone)
    pre_candle = None
    for i in range(len(prev_rows)-1, -1, -1):
        row = prev_rows.iloc[i]
        if (row['low'] <= last_high <= row['high']) or (row['low'] <= last_row_low <= row['high']):
            pre_candle = row
            break
    if pre_candle is None:
        continue

    # Find lowest low between pre_candle and last_row_date → SL price
    between_rows = symbol_group[
        (symbol_group['date'] > pre_candle['date']) & 
        (symbol_group['date'] < last_row_date)
    ]
    if between_rows.empty:
        continue

    low_candle = between_rows.loc[between_rows['low'].idxmin()]
    SL_price = low_candle['low']
    buy_price = last_row['close']  # entry = close of breakout day

    output_rows.append({
        'symbol': symbol,
        'date': last_row['date'].date(),      # date-only (YYYY-MM-DD)
        'last_row_close': buy_price,          # → will be used as 'buy'
        'SL': SL_price,                       # → price, not %
    })

# Create minimal DataFrame
if output_rows:
    output_df = pd.DataFrame(output_rows)
    output_df = output_df.sort_values(['SL', 'symbol'], ascending=[True, True]).reset_index(drop=True)
    output_df.insert(0, 'id', range(1, len(output_df) + 1))

    # Ensure numeric
    output_df['last_row_close'] = pd.to_numeric(output_df['last_row_close'], errors='coerce')
    output_df['SL'] = pd.to_numeric(output_df['SL'], errors='coerce')
else:
    # Empty but valid 5-col structure
    output_df = pd.DataFrame(columns=['id', 'symbol', 'date', 'last_row_close', 'SL'])

# Save (both locations)
output_df.to_csv(output_path1, index=False)
output_df.to_csv(output_path2, index=False)

print(f"✅ Successfully saved {len(output_df)} rows to short_buy.csv")
print("📁 Columns: id, symbol, date, last_row_close (→ buy), SL (price)")
if not output_df.empty:
    print("\n📋 Sample:")
    print(output_df.head(3)[['id','symbol','date','last_row_close','SL']].to_string(index=False))