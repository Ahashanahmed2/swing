import pandas as pd
import os

# Paths
gape_file = "./csv/gape.csv"
mongodb_file = "./csv/mongodb.csv"
output_file1 = "./output/ai_signal/gape_buy.csv"
output_file2 = "./csv/gape_buy.csv"

# Ensure dirs
os.makedirs(os.path.dirname(output_file1), exist_ok=True)
os.makedirs(os.path.dirname(output_file2), exist_ok=True)

# Remove old files
for f in [output_file1, output_file2]:
    if os.path.exists(f):
        os.remove(f)

# Load and clean
gape_df = pd.read_csv(gape_file)
mongodb_df = pd.read_csv(mongodb_file)

# Convert dates
gape_df['last_row_date'] = pd.to_datetime(gape_df['last_row_date'], errors='coerce')
mongodb_df['date'] = pd.to_datetime(mongodb_df['date'], errors='coerce')
mongodb_df = mongodb_df.dropna(subset=['date'])

results = []

for _, r in gape_df.iterrows():
    symbol = str(r['symbol']).strip().upper()
    date = r['last_row_date']
    last_row_close = r['last_row_close']
    last_row_high = r['last_row_high']
    last_row_low = r['last_row_low']

    if pd.isna(date) or pd.isna(last_row_close):
        continue

    # MongoDB data for symbol
    sym_data = mongodb_df[mongodb_df['symbol'] == symbol].sort_values('date')
    if sym_data.empty:
        continue

    # Rows before signal date
    prev_rows = sym_data[sym_data['date'] < date]
    if prev_rows.empty:
        continue

    prev_row = prev_rows.iloc[-1]

    # Signal condition
    if not (last_row_low > prev_row['low'] and last_row_close > prev_row['high']):
        continue

    # Find pre-candle (zone touch)
    pre_candle = None
    for i in range(len(prev_rows)-1, -1, -1):
        row = prev_rows.iloc[i]
        if (row['low'] <= last_row_high <= row['high']) or (row['low'] <= last_row_low <= row['high']):
            pre_candle = row
            break
    if pre_candle is None:
        continue

    # Candles between pre_candle and signal date
    between = sym_data[
        (sym_data['date'] > pre_candle['date']) & 
        (sym_data['date'] < date)
    ]
    if between.empty:
        continue

    # SL = lowest low in between
    low_candle = between.loc[between['low'].idxmin()]
    SL_price = low_candle['low']

    # ✅ CRITICAL: rename 'last_row_close' → 'buy'
    results.append({
        'symbol': symbol,
        'date': date.date(),    # date-only (YYYY-MM-DD)
        'buy': last_row_close,  # ← renamed! (used as buy price)
        'SL': SL_price,         # price, not %
    })

# Build DataFrame
if results:
    df = pd.DataFrame(results)
    df = df.sort_values(['SL', 'symbol'], ascending=[True, True]).reset_index(drop=True)
    df.insert(0, 'row_id', range(1, len(df) + 1))

    # Ensure numeric
    df['buy'] = pd.to_numeric(df['buy'], errors='coerce')
    df['SL'] = pd.to_numeric(df['SL'], errors='coerce')
else:
    df = pd.DataFrame(columns=['row_id', 'symbol', 'date', 'buy', 'SL'])

# Save
df.to_csv(output_file1, index=False)
df.to_csv(output_file2, index=False)

print(f"✅ Successfully saved {len(df)} rows to gape_buy.csv")
print("📁 Columns: row_id, symbol, date, buy, SL")
if not df.empty:
    print("\n📋 Sample:")
    print(df.head(3)[['row_id','symbol','date','buy','SL']].to_string(index=False))