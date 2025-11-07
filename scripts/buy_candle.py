import pandas as pd
import os

# ---------- 1. Read source files ----------
up_candle   = pd.read_csv('./csv/up_candle.csv')
mongo_df    = pd.read_csv('./csv/mongodb.csv')

# date columns to datetime
up_candle['date'] = pd.to_datetime(up_candle['date'])
mongo_df['date']  = pd.to_datetime(mongo_df['date'])

buy_rows = []

# ---------- 2. Loop every up-candle row ----------
for _, uc_row in up_candle.iterrows():
    symbol = uc_row['symbol']
    base_date = uc_row['date']

    # symbol group from mongodb
    sym_df = mongo_df[mongo_df['symbol'] == symbol].sort_values('date').reset_index(drop=True)

    # slice: only rows after base_date
    future = sym_df[sym_df['date'] > base_date].reset_index(drop=True)
    if future.empty:
        continue

    # ---------- 3. Check condition ----------
    for i in range(len(future) - 1):
        if future.loc[i + 1, 'close'] > future.loc[i, 'high']:
            # take the LAST row of this symbol
            last_row = sym_df.iloc[-1]
            buy_rows.append({
                'symbol': last_row['symbol'],
                'date': last_row['date'].strftime('%Y-%m-%d'),
                'close': last_row['close'],
                'low': last_row['low'],
                'high': last_row['high']
            })
            break  # only first trigger per symbol

# ---------- 4. Save outputs ----------
os.makedirs('./output/ai_signal', exist_ok=True)

csv_path     = './csv/buy_stock.csv'
ai_path      = './output/ai_signal/buy_stock.csv'

def save_unique(file_path, new_rows):
    if os.path.exists(file_path):
        old = pd.read_csv(file_path)
        old['date'] = pd.to_datetime(old['date'])
    else:
        old = pd.DataFrame()

    new_df = pd.DataFrame(new_rows)
    new_df['date'] = pd.to_datetime(new_df['date'])

    # drop duplicates (symbol+date)
    combined = pd.concat([old, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['symbol', 'date'])
    combined.to_csv(file_path, index=False)

if buy_rows:
    save_unique(csv_path, buy_rows)
    save_unique(ai_path, buy_rows)
    print("✅ buy_stock.csv saved to both ./csv/ & ./output/ai_signal/")
else:
    print("⚠️ No buy condition matched.")
