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

    symbol_group = mongo_groups.get_group(symbol).sort_values('date').reset_index(drop=True)
    
    last_row_candidates = symbol_group[symbol_group['date'] == last_row_date]
    if last_row_candidates.empty:
        continue
    last_row = last_row_candidates.iloc[-1]

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

    # 🔑 Find tp: backward from low_candle
    tp_price = None
    try:
        low_idx = symbol_group[symbol_group['date'] == low_candle['date']].index[0]
    except IndexError:
        low_idx = (abs(symbol_group['date'] - low_candle['date'])).idxmin()

    for i in range(low_idx - 1, 1, -1):
        try:
            sb = symbol_group.iloc[i - 2]
            sa = symbol_group.iloc[i - 1]
            s  = symbol_group.iloc[i]
        except IndexError:
            break

        if not (sb['date'] < sa['date'] < s['date'] < low_candle['date']):
            continue

        if (s['high'] > sa['high']) and (sa['high'] >= sb['high']):
            tp_price = s['high']
            break

    # Skip if tp not found
    if tp_price is None:
        continue

    # Append raw values (RRR will be computed after DataFrame creation)
    output_rows.append({
        'symbol': symbol,
        'date': last_row['date'].date(),
        'buy': buy_price,
        'SL': SL_price,
        'tp': tp_price,
    })

# Build DataFrame
if output_rows:
    df = pd.DataFrame(output_rows)
    df['buy'] = pd.to_numeric(df['buy'], errors='coerce')
    df['SL'] = pd.to_numeric(df['SL'], errors='coerce')
    df['tp'] = pd.to_numeric(df['tp'], errors='coerce')
    
    # ✅ Calculate RRR
    df['RRR'] = (df['tp'] - df['buy']) / (df['buy'] - df['SL'])
    
    # ✅ Filter: keep only valid, positive RRR (tp > buy > SL)
    df = df[
        (df['buy'] > df['SL']) & 
        (df['tp'] > df['buy']) & 
        (df['RRR'] > 0)
    ].reset_index(drop=True)
    
    if len(df) == 0:
        df = pd.DataFrame(columns=['No', 'symbol', 'date', 'buy', 'SL', 'tp', 'RRR'])
    else:
        # Sort: highest RRR first → then smallest risk (buy-SL) first
        df = df.sort_values(['RRR', 'buy', 'SL'], ascending=[False, True, False]).reset_index(drop=True)
        df.insert(0, 'No', range(1, len(df) + 1))
        df = df[['No', 'symbol', 'date', 'buy', 'SL', 'tp', 'RRR']]
else:
    df = pd.DataFrame(columns=['No', 'symbol', 'date', 'buy', 'SL', 'tp', 'RRR'])

# Save
df.to_csv(output_path1, index=False)
df.to_csv(output_path2, index=False)

print(f"✅ short_buy.csv saved with 'tp' and 'RRR' columns ({len(df)} rows)")
if len(df) > 0:
    print(f"📈 Best RRR: {df['RRR'].max():.2f} | Worst RRR: {df['RRR'].min():.2f}")