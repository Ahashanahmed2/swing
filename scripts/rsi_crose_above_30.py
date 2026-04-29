import pandas as pd
import os
import numpy as np

# ---------------------------------------------------------
# Path
# ---------------------------------------------------------
RSI30_PATH = "./csv/rsi_30.csv"
MONGO_PATH = "./csv/mongodb.csv"
BUY_PATH_CSV = "./csv/rsi_30_buy.csv"

os.makedirs("./csv/", exist_ok=True)

# ---------------------------------------------------------
# Helper: Print log
# ---------------------------------------------------------
def log(msg):
    print("🔹", msg)

# ---------------------------------------------------------
# Load mongodb.csv
# ---------------------------------------------------------
log("Loading mongodb.csv...")
if not os.path.exists(MONGO_PATH):
    raise FileNotFoundError(f"❌ Required file not found: {MONGO_PATH}")

mongodb = pd.read_csv(MONGO_PATH)

required_cols = {'symbol', 'date', 'low', 'high', 'close', 'rsi'}
if not required_cols.issubset(mongodb.columns):
    missing = required_cols - set(mongodb.columns)
    raise ValueError(f"❌ Missing required columns in mongodb.csv: {missing}")

mongodb['date'] = pd.to_datetime(mongodb['date'])
mongodb = mongodb.sort_values(['symbol', 'date']).reset_index(drop=True)

# ---------------------------------------------------------
# 🔁 AUTO-GENERATE rsi_30.csv: Keep ONLY symbols with latest RSI < 30
# ---------------------------------------------------------
log("🔍 Finding latest record per symbol with RSI < 30...")
latest_mongo = mongodb.sort_values('date').groupby('symbol').tail(1).reset_index(drop=True)

rsi_under_30 = latest_mongo[
    (latest_mongo['rsi'] < 30) &
    (latest_mongo['rsi'].notna())
].copy()

log(f"✅ Found {len(rsi_under_30)} symbols with latest RSI < 30")

if len(rsi_under_30) > 0:
    rsi30_auto = rsi_under_30[['symbol', 'date', 'low', 'high', 'rsi']].copy()
    rsi30_auto = rsi30_auto.sort_values('symbol').reset_index(drop=True)
    rsi30_auto.insert(0, 'sl', range(1, len(rsi30_auto) + 1))
else:
    rsi30_auto = pd.DataFrame(columns=['sl', 'symbol', 'date', 'low', 'high', 'rsi'])

rsi30_auto.to_csv(RSI30_PATH, index=False)
log(f"💾 Updated rsi_30.csv with {len(rsi30_auto)} active symbols")

# ---------------------------------------------------------
# Load rsi_30.csv
# ---------------------------------------------------------
log("Loading rsi_30.csv (auto-generated)...")
rsi30 = pd.read_csv(RSI30_PATH)
rsi30['date'] = pd.to_datetime(rsi30['date'])

log(f"mongodb rows: {len(mongodb)}")
log(f"rsi_30 rows loaded: {len(rsi30)}")

rsi30_final = rsi30.copy()

# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------
if len(rsi30) == 0:
    log("ℹ No symbols with RSI < 30 → skipping processing.")
else:
    for idx, row in rsi30.iterrows():
        symbol = row['symbol']
        base_date = row['date']
        base_low = row['low']
        base_rsi = row['rsi']
        sl = row['sl']

        log(f"Processing: {symbol} | Base Date: {base_date.strftime('%Y-%m-%d')} | RSI={base_rsi:.2f}")

        md = mongodb[mongodb['symbol'] == symbol].copy()
        if md.empty:
            log(f"  ⚠ No mongodb data for {symbol}")
            continue

        if base_date not in md['date'].values:
            log(f"  ⚠ Base date {base_date.date()} not found in mongodb for {symbol} → skipping")
            continue

        start_idx = md[md['date'] == base_date].index[0]

        for i in range(start_idx + 1, len(md)):
            r = md.loc[i]
            m_rsi = r['rsi']
            m_close = r['close']

            # DELETE rule
            if m_rsi > base_rsi and m_close < base_low:
                log(f"  ❌ DELETE {symbol}: RSI↑({m_rsi:.2f}) & close({m_close}) < base_low({base_low})")
                rsi30_final = rsi30_final[rsi30_final['sl'] != sl]
                break

            # EXIT rule
            if m_rsi > 30:
                log(f"  ❌ EXIT {symbol}: RSI={m_rsi:.2f} > 30")
                rsi30_final = rsi30_final[rsi30_final['sl'] != sl]
                break

# ---------------------------------------------------------
# ✅ SAVE BUY SIGNALS (empty)
# ---------------------------------------------------------
buy_df = pd.DataFrame(columns=['No', 'date', 'symbol', 'buy', 'SL'])

# -------------------------------
# DELETE OLD FILE & SAVE
# -------------------------------
if os.path.exists(BUY_PATH_CSV):
    os.remove(BUY_PATH_CSV)

buy_df.to_csv(BUY_PATH_CSV, index=False)

log("ℹ BUY signals: empty file created.")

# ---------------------------------------------------------
# Save updated rsi_30.csv
# ---------------------------------------------------------
rsi30_final = rsi30_final.reset_index(drop=True)
COLUMNS = ['sl', 'symbol', 'date', 'low', 'high', 'rsi']
if len(rsi30_final) == 0:
    rsi30_final = pd.DataFrame(columns=COLUMNS)
else:
    rsi30_final = rsi30_final.reindex(columns=COLUMNS)
    rsi30_final['sl'] = range(1, len(rsi30_final) + 1)

rsi30_final.to_csv(RSI30_PATH, index=False)
log(f"💾 rsi_30.csv updated — remaining symbols: {len(rsi30_final)}")

log("✅ Script completed successfully!")
