import pandas as pd
import os

RSI30_PATH = "./csv/rsi_30.csv"
MONGO_PATH = "./csv/mongodb.csv"
BUY_PATH_OUTPUT = "./output/ai_signal/rsi_30_buy.csv"
BUY_PATH_CSV = "./csv/rsi_30_buy.csv"

os.makedirs("./output/ai_signal/", exist_ok=True)
os.makedirs("./csv/", exist_ok=True)

for path in [BUY_PATH_OUTPUT, BUY_PATH_CSV]:
    if os.path.exists(path):
        os.remove(path)

# --- [mongodb & rsi_30.csv generation: same as before] ---
mongodb = pd.read_csv(MONGO_PATH)
required_cols = {'symbol', 'date', 'low', 'high', 'close', 'rsi'}
if not required_cols.issubset(mongodb.columns):
    raise ValueError(f"Missing cols in mongodb.csv: {required_cols - set(mongodb.columns)}")

mongodb['date'] = pd.to_datetime(mongodb['date'])
mongodb = mongodb.sort_values(['symbol', 'date']).reset_index(drop=True)

latest_mongo = mongodb.groupby('symbol').tail(1)
rsi_under_30 = latest_mongo[(latest_mongo['rsi'] < 30) & (latest_mongo['rsi'].notna())].copy()
if len(rsi_under_30) > 0:
    rsi30_auto = rsi_under_30[['symbol', 'date', 'low', 'high', 'rsi']].copy()
    rsi30_auto.insert(0, 'sl', range(1, len(rsi30_auto)+1))
else:
    rsi30_auto = pd.DataFrame(columns=['sl', 'symbol', 'date', 'low', 'high', 'rsi'])
rsi30_auto.to_csv(RSI30_PATH, index=False)

rsi30 = pd.read_csv(RSI30_PATH)
rsi30['date'] = pd.to_datetime(rsi30['date'])

buy_rows = []
rsi30_final = rsi30.copy()

if len(rsi30) > 0:
    for _, row in rsi30.iterrows():
        symbol = row['symbol']
        base_date = row['date']
        base_low = row['low']
        base_high = row['high']
        base_rsi = row['rsi']
        sl = row['sl']

        md = mongodb[mongodb['symbol'] == symbol].copy()
        if md.empty or base_date not in md['date'].values:
            continue

        start_idx = md[md['date'] == base_date].index[0]

        for i in range(start_idx + 1, len(md)):
            r = md.loc[i]
            m_rsi, m_close, m_date = r['rsi'], r['close'], r['date']

            # DELETE / EXIT
            if (m_rsi > base_rsi and m_close < base_low) or (m_rsi > 30):
                rsi30_final = rsi30_final[rsi30_final['sl'] != sl]
                break

            # BUY
            if m_close > base_high:
                buy_rows.append([m_date.date(), symbol, m_close, base_low])
                break

# Save standardized BUY
if buy_rows:
    df = pd.DataFrame(buy_rows, columns=['date', 'symbol', 'buy', 'SL'])
    df['diff'] = df['buy'] - df['SL']
    df = df.sort_values('diff').reset_index(drop=True)
    df.insert(0, 'No', range(1, len(df)+1))
    df = df[['No', 'symbol', 'date', 'buy', 'SL']].copy()
    df['buy'] = pd.to_numeric(df['buy'], errors='coerce')
    df['SL'] = pd.to_numeric(df['SL'], errors='coerce')
else:
    df = pd.DataFrame(columns=['No', 'symbol', 'date', 'buy', 'SL'])

df.to_csv(BUY_PATH_OUTPUT, index=False)
df.to_csv(BUY_PATH_CSV, index=False)

# Update rsi_30.csv
rsi30_final = rsi30_final.reindex(columns=['sl','symbol','date','low','high','rsi']) if len(rsi30_final) else pd.DataFrame(columns=['sl','symbol','date','low','high','rsi'])
if len(rsi30_final): rsi30_final['sl'] = range(1, len(rsi30_final)+1)
rsi30_final.to_csv(RSI30_PATH, index=False)

print(f"✅ rsi_30_buy.csv saved ({len(df)} rows)")