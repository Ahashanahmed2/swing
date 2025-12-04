import pandas as pd
import os

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
RSI30_PATH = "./csv/rsi_30.csv"
MONGO_PATH = "./csv/mongodb.csv"
BUY_PATH = "./output/ai_signal/rsi_30_buy.csv"

os.makedirs("./output/ai_signal/", exist_ok=True)

# ---------------------------------------------------------
# Helper: Print log
# ---------------------------------------------------------
def log(msg):
    print("🔹", msg)

# ---------------------------------------------------------
# Load mongodb.csv (must exist)
# ---------------------------------------------------------
log("Loading mongodb.csv...")
mongodb = pd.read_csv(MONGO_PATH)
mongodb['date'] = pd.to_datetime(mongodb['date'])
mongodb = mongodb.sort_values(['symbol', 'date']).reset_index(drop=True)

# ---------------------------------------------------------
# Load rsi_30.csv (SAFE: no error if missing)
# ---------------------------------------------------------
if os.path.exists(RSI30_PATH):
    log("Loading rsi_30.csv...")
    rsi30 = pd.read_csv(RSI30_PATH)
    rsi30['date'] = pd.to_datetime(rsi30['date'])
else:
    log("⚠ rsi_30.csv not found → starting with empty list.")
    rsi30 = pd.DataFrame(columns=['sl', 'symbol', 'date', 'low', 'high', 'rsi'])

log(f"mongodb rows: {len(mongodb)}")
log(f"rsi_30 rows loaded: {len(rsi30)}")

# ---------------------------------------------------------
# Prepare containers
# ---------------------------------------------------------
buy_rows = []
rsi30_final = rsi30.copy()

# If rsi_30 is empty → nothing to process
if len(rsi30) == 0:
    log("ℹ No symbols in rsi_30 → skipping processing.")
else:
    # ---------------------------------------------------------
    # Process each symbol
    # ---------------------------------------------------------
    for idx, row in rsi30.iterrows():

        symbol = row['symbol']
        base_date = row['date']
        base_low = row['low']
        base_high = row['high']
        base_rsi = row['rsi']

        log(f"Processing: {symbol} (RSI={base_rsi})")

        md = mongodb[mongodb['symbol'] == symbol]

        if md.empty:
            log(f"  ⚠ No mongodb data for {symbol}")
            continue

        if base_date not in md['date'].values:
            log(f"  ⚠ Date mismatch for {symbol}")
            continue

        start_idx = md[md['date'] == base_date].index[0]

        for i in range(start_idx + 1, len(md)):
            r = md.loc[i]
            m_rsi = r['rsi']
            m_close = r['close']
            m_date = r['date']

            # --------------------------
            # RULE–A: Delete Condition
            # --------------------------
            if m_rsi > base_rsi and m_close < base_low:
                log(f"  ❌ DELETE {symbol}: rsi↑ & close dropped < low")
                rsi30_final = rsi30_final[rsi30_final['symbol'] != symbol]
                break

            # --------------------------
            # RULE–C: BUY Condition
            # --------------------------
            if m_close > base_high:
                log(f"  ✅ BUY SIGNAL {symbol} @ close={m_close}")
                buy_rows.append([
                    m_date.strftime("%Y-%m-%d"),
                    symbol,
                    m_close,
                    base_low
                ])

            # --------------------------
            # RULE–D: RSI Exit Condition
            # --------------------------
            if m_rsi > 30:
                log(f"  ❌ EXIT {symbol}: RSI>30")
                rsi30_final = rsi30_final[rsi30_final['symbol'] != symbol]
                break

# ---------------------------------------------------------
# Save BUY signals
# ---------------------------------------------------------
if buy_rows:
    buy_df = pd.DataFrame(buy_rows, columns=['date', 'symbol', 'close', 'SL'])
    buy_df.insert(0, 'No', range(1, len(buy_df) + 1))

    buy_df.to_csv(BUY_PATH, index=False)
    log(f"💾 BUY signals saved to: {BUY_PATH}")
else:
    log("ℹ No BUY signals today.")

# ---------------------------------------------------------
# Save updated rsi_30.csv
# ---------------------------------------------------------
rsi30_final = rsi30_final.reset_index(drop=True)
if len(rsi30_final) > 0:
    rsi30_final.insert(0, "sl", rsi30_final.index + 1)

rsi30_final.to_csv(RSI30_PATH, index=False)

log("💾 rsi_30.csv updated successfully!")
log(f"Final symbols count: {len(rsi30_final)}")