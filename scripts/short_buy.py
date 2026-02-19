import pandas as pd
import os

# Paths
rsi_path = './csv/rsi_diver_retest.csv'
mongo_path = './csv/mongodb.csv'
output_path2 = './csv/short_buy.csv'

os.makedirs(os.path.dirname(output_path2), exist_ok=True)

# Delete old file
if os.path.exists(output_path2):
    os.remove(output_path2)

# Read data
rsi_df = pd.read_csv(rsi_path)
mongo_df = pd.read_csv(mongo_path)

# Clean column names (spaces to underscores)
rsi_df.columns = rsi_df.columns.str.replace(" ", "_")

# Ensure MongoDB date is datetime
mongo_df['date'] = pd.to_datetime(mongo_df['date'], errors='coerce')
mongo_df = mongo_df.dropna(subset=['date'])
mongo_groups = mongo_df.groupby('symbol')

output_rows = []

print("\n" + "="*80)
print("PROCESSING SIGNALS:")
print("="*80)

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
    
    # diff ক্যালকুলেশন (buy - SL)
    diff = round(buy_price - SL_price, 4)

    # Find tp (optional - you can remove if not needed)
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

    if tp_price is None:
        continue

    # symbol, date, buy, sl, diff স্টোর করছি
    output_rows.append({
        'symbol': symbol,
        'date': last_row['date'].date(),
        'buy': buy_price,
        'sl': SL_price,
        'diff': diff
    })
    
    print(f"✅ Signal: {symbol} | Buy={buy_price:.2f} | SL={SL_price:.2f} | Diff={diff:.4f}")

# DataFrame তৈরি করা
if output_rows:
    df = pd.DataFrame(output_rows)
    
    # diff অনুযায়ী সর্ট (ছোট diff প্রথমে)
    df = df.sort_values('diff', ascending=True).reset_index(drop=True)
    
    # প্রথম কলাম হিসেবে সিরিয়াল নাম্বার যোগ করা
    df.insert(0, 'no', range(1, len(df) + 1))
    
    # কলামের অর্ডার ঠিক করা: no, symbol, date, buy, sl, diff
    df = df[['no', 'symbol', 'date', 'buy', 'sl', 'diff']]
    
    print(f"\n{'='*80}")
    print(f"✅ TOTAL SIGNALS: {len(df)}")
    print(f"{'='*80}")
    print(df.to_string())
    
else:
    print(f"\n❌ No signals generated")
    df = pd.DataFrame(columns=['no', 'symbol', 'date', 'buy', 'sl', 'diff'])

# Save to CSV
df.to_csv(output_path2, index=False)
print(f"\n✅ Saved to {output_path2}")
print(f"📁 File contains columns: {', '.join(df.columns)}")