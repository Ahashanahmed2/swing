import pandas as pd
import os

gape_file = "./csv/gape.csv"
mongodb_file = "./csv/mongodb.csv"
output_file1 = "./output/ai_signal/gape_buy.csv"
output_file2 = "./csv/gape_buy.csv"

os.makedirs(os.path.dirname(output_file1), exist_ok=True)
os.makedirs(os.path.dirname(output_file2), exist_ok=True)

# Delete old files
for f in [output_file1, output_file2]:
    if os.path.exists(f):
        os.remove(f)

# Load data
gape_df = pd.read_csv(gape_file)
mongodb_df = pd.read_csv(mongodb_file)

# Parse dates
gape_df['last_row_date'] = pd.to_datetime(gape_df['last_row_date'], errors='coerce')
mongodb_df['date'] = pd.to_datetime(mongodb_df['date'], errors='coerce')
mongodb_df = mongodb_df.dropna(subset=['date'])

# Group MongoDB by symbol for efficiency
mongo_groups = mongodb_df.groupby('symbol')

results = []

for _, r in gape_df.iterrows():
    symbol = str(r['symbol']).strip().upper()
    date = r['last_row_date']
    last_row_close = r['last_row_close']
    last_row_high = r['last_row_high']
    last_row_low = r['last_row_low']

    if pd.isna(date) or pd.isna(last_row_close) or symbol not in mongo_groups.groups:
        continue

    sym_data = mongo_groups.get_group(symbol).sort_values('date').reset_index(drop=True)

    # Get rows before trigger date
    prev_rows = sym_data[sym_data['date'] < date]
    if prev_rows.empty:
        continue
    prev_row = prev_rows.iloc[-1]

    # Entry condition
    if not (last_row_low > prev_row['low'] and last_row_close > prev_row['high']):
        continue

    # Find pre_candle (overlap with last_high or last_low)
    pre_candle = None
    for i in range(len(prev_rows)-1, -1, -1):
        row = prev_rows.iloc[i]
        if (row['low'] <= last_row_high <= row['high']) or (row['low'] <= last_row_low <= row['high']):
            pre_candle = row
            break
    if pre_candle is None:
        continue

    # Rows between pre_candle and trigger
    between = sym_data[
        (sym_data['date'] > pre_candle['date']) & 
        (sym_data['date'] < date)
    ]
    if between.empty:
        continue

    # SL = lowest low in between segment
    low_candle = between.loc[between['low'].idxmin()]
    SL_price = low_candle['low']
    buy_price = last_row_close

    # 🔑 Find tp: scan BACKWARD from low_candle
    tp_price = None
    try:
        low_idx = sym_data[sym_data['date'] == low_candle['date']].index[0]
    except IndexError:
        # Fallback: closest match by time
        low_idx = (abs(sym_data['date'] - low_candle['date'])).idxmin()

    # Scan backward: i = index of candidate 's' (need sb = i-2, sa = i-1, s = i)
    for i in range(low_idx - 1, 1, -1):
        try:
            sb = sym_data.iloc[i - 2]
            sa = sym_data.iloc[i - 1]
            s  = sym_data.iloc[i]
        except IndexError:
            break

        # Ensure chronological order
        if not (sb['date'] < sa['date'] < s['date'] < low_candle['date']):
            continue

        # ✅ tp condition: s.high > sa.high and sa.high >= sb.high
        if (s['high'] > sa['high']) and (sa['high'] >= sb['high']):
            tp_price = s['high']
            break

    if tp_price is None:
        continue  # skip if no valid tp

    # Append for later RRR calculation
    results.append({
        'symbol': symbol,
        'date': date.date(),
        'buy': buy_price,
        'SL': SL_price,
        'tp': tp_price,
    })

# Build final DataFrame
if results:
    df = pd.DataFrame(results)
    df['buy'] = pd.to_numeric(df['buy'], errors='coerce')
    df['SL'] = pd.to_numeric(df['SL'], errors='coerce')
    df['tp'] = pd.to_numeric(df['tp'], errors='coerce')

    # Compute diff & RRR
    df['diff'] = (df['buy'] - df['SL']).round(4)
    df['RRR'] = ((df['tp'] - df['buy']) / (df['buy'] - df['SL'])).round(2)

    # ✅ Filter: only keep valid, positive RRR signals
    df = df[
        (df['buy'] > df['SL']) &
        (df['tp'] > df['buy']) &
        (df['RRR'] > 0)
    ].reset_index(drop=True)

    if len(df) > 0:
        # Sort: highest RRR first → then smallest risk (diff) first
        df = df.sort_values(['RRR', 'diff'], ascending=[False, True]).reset_index(drop=True)
        df.insert(0, 'No', range(1, len(df) + 1))
        # Final column order
        df = df[['No', 'symbol', 'date', 'buy', 'SL', 'tp', 'diff', 'RRR']]
    else:
        df = pd.DataFrame(columns=['No', 'symbol', 'date', 'buy', 'SL', 'tp', 'diff', 'RRR'])
else:
    df = pd.DataFrame(columns=['No', 'symbol', 'date', 'buy', 'SL', 'tp', 'diff', 'RRR'])

# Save
df.to_csv(output_file1, index=False)
df.to_csv(output_file2, index=False)

print(f"✅ gape_buy.csv saved with {len(df)} signals:")
if len(df) > 0:
    print(f"   📈 Max RRR: {df['RRR'].max():.2f} | Avg RRR: {df['RRR'].mean():.2f}")
    print(f"   📉 Min diff: {df['diff'].min():.4f}")