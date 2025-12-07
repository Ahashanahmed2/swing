import pandas as pd
import os

gape_file = "./csv/gape.csv"
mongodb_file = "./csv/mongodb.csv"
output_file1 = "./output/ai_signal/gape_buy.csv"
output_file2 = "./csv/gape_buy.csv"

os.makedirs(os.path.dirname(output_file1), exist_ok=True)
os.makedirs(os.path.dirname(output_file2), exist_ok=True)

for f in [output_file1, output_file2]:
    if os.path.exists(f):
        os.remove(f)

gape_df = pd.read_csv(gape_file)
mongodb_df = pd.read_csv(mongodb_file)

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

    sym_data = mongodb_df[mongodb_df['symbol'] == symbol].sort_values('date')
    if sym_data.empty:
        continue

    prev_rows = sym_data[sym_data['date'] < date]
    if prev_rows.empty:
        continue

    prev_row = prev_rows.iloc[-1]

    if not (last_row_low > prev_row['low'] and last_row_close > prev_row['high']):
        continue

    pre_candle = None
    for i in range(len(prev_rows)-1, -1, -1):
        row = prev_rows.iloc[i]
        if (row['low'] <= last_row_high <= row['high']) or (row['low'] <= last_row_low <= row['high']):
            pre_candle = row
            break
    if pre_candle is None:
        continue

    between = sym_data[
        (sym_data['date'] > pre_candle['date']) & 
        (sym_data['date'] < date)
    ]
    if between.empty:
        continue

    low_candle = between.loc[between['low'].idxmin()]
    SL_price = low_candle['low']

    results.append({
        'symbol': symbol,
        'date': date.date(),
        'buy': last_row_close,
        'SL': SL_price,
    })

if results:
    df = pd.DataFrame(results)
    df = df.sort_values(['SL', 'symbol'], ascending=[True, True]).reset_index(drop=True)
    df.insert(0, 'No', range(1, len(df) + 1))
    df['buy'] = pd.to_numeric(df['buy'], errors='coerce')
    df['SL'] = pd.to_numeric(df['SL'], errors='coerce')
else:
    df = pd.DataFrame(columns=['No', 'symbol', 'date', 'buy', 'SL'])

df.to_csv(output_file1, index=False)
df.to_csv(output_file2, index=False)

print(f"✅ gape_buy.csv saved ({len(df)} rows)")