import pandas as pd
import os

# ---------------------------------------------------------
# Path
# ---------------------------------------------------------

RSI30_PATH = "./csv/rsi_30.csv"
MONGO_PATH = "./csv/mongodb.csv"
BUY_PATH_OUTPUT = "./output/ai_signal/rsi_30_buy.csv"
BUY_PATH_CSV = "./csv/rsi_30_buy.csv"

os.makedirs("./output/ai_signal/", exist_ok=True)
os.makedirs("./csv/", exist_ok=True)

# ---------------------------------------------------------
# Helper: Print log
# ---------------------------------------------------------

def log(msg):
    print("🔹", msg)

# ---------------------------------------------------------
# Load mongodb.csv (must exist)
# ---------------------------------------------------------

log("Loading mongodb.csv...")
if not os.path.exists(MONGO_PATH):
    raise FileNotFoundError(f"❌ Required file not found: {MONGO_PATH}")

mongodb = pd.read_csv(MONGO_PATH)

# Ensure required columns exist
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

buy_rows = []
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
        base_high = row['high']
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
            m_date = r['date']

            # DELETE rule
            if m_rsi > base_rsi and m_close < base_low:
                log(f"  ❌ DELETE {symbol}: RSI↑({m_rsi:.2f}) & close({m_close}) < base_low({base_low})")
                rsi30_final = rsi30_final[rsi30_final['sl'] != sl]
                break

            # BUY rule
            if m_close > base_high:
                log(f"  ✅ BUY SIGNAL {symbol} @ {m_date.strftime('%Y-%m-%d')}")
                buy_rows.append({
                    'date': m_date,
                    'symbol': symbol,
                    'buy': m_close,
                    'SL': base_low,
                    'base_date': base_date  # 🔑 store for tp search
                })

            # EXIT rule
            if m_rsi > 30:
                log(f"  ❌ EXIT {symbol}: RSI={m_rsi:.2f} > 30")
                rsi30_final = rsi30_final[rsi30_final['sl'] != sl]
                break

# ---------------------------------------------------------
# SAVE BUY SIGNALS (with tp, RRR, diff & sorted)
# ---------------------------------------------------------

if buy_rows:
    buy_df = pd.DataFrame(buy_rows)

    tp_list = []
    diff_list = []
    rrr_list = []

    for _, row in buy_df.iterrows():
        symbol = row['symbol']
        buy_date = row['date']
        buy_price = row['buy']
        SL_price = row['SL']
        base_date = row['base_date']

        # Get symbol data
        sym_data = mongodb[mongodb['symbol'] == symbol].sort_values('date').reset_index(drop=True)

        # Find index of base_date (SL source)
        try:
            sl_idx = sym_data[sym_data['date'] == base_date].index[0]
        except IndexError:
            sl_idx = (abs(sym_data['date'] - base_date)).idxmin()

        # 🔍 Search BACKWARD from sl_idx for tp
        tp = None
        for i in range(sl_idx - 1, 1, -1):
            try:
                sb = sym_data.iloc[i - 2]
                sa = sym_data.iloc[i - 1]
                s  = sym_data.iloc[i]
            except IndexError:
                break

            if not (sb['date'] < sa['date'] < s['date'] < base_date):
                continue

            if (s['high'] > sa['high']) and (sa['high'] >= sb['high']):
                tp = s['high']
                break

        diff = buy_price - SL_price
        diff_list.append(round(diff, 4))

        if tp is not None and buy_price > SL_price and tp > buy_price:
            rrr = (tp - buy_price) / diff
            tp_list.append(round(tp, 4))
            rrr_list.append(round(rrr, 2))
        else:
            tp_list.append(None)
            rrr_list.append(None)

    # Add columns
    buy_df['tp'] = tp_list
    buy_df['diff'] = diff_list
    buy_df['RRR'] = rrr_list

    # ✅ Filter valid signals
    buy_df = buy_df.dropna(subset=['tp', 'RRR']).reset_index(drop=True)
    buy_df = buy_df[buy_df['RRR'] > 0].reset_index(drop=True)

    if len(buy_df) > 0:
        # Sort: RRR ↓ → diff ↑
        buy_df = buy_df.sort_values(['RRR', 'diff'], ascending=[False, True]).reset_index(drop=True)
        buy_df.insert(0, 'No', range(1, len(buy_df) + 1))

        # Format & final columns
        buy_df['date'] = buy_df['date'].dt.strftime("%Y-%m-%d")
        buy_df = buy_df[['No', 'date', 'symbol', 'buy', 'SL', 'tp', 'diff', 'RRR']]
    else:
        buy_df = pd.DataFrame(columns=['No', 'date', 'symbol', 'buy', 'SL', 'tp', 'diff', 'RRR'])

else:
    buy_df = pd.DataFrame(columns=['No', 'date', 'symbol', 'buy', 'SL', 'tp', 'diff', 'RRR'])

# -------------------------------
# DELETE OLD FILES
# -------------------------------
for f in [BUY_PATH_CSV, BUY_PATH_OUTPUT]:
    if os.path.exists(f):
        os.remove(f)

# Save
buy_df.to_csv(BUY_PATH_OUTPUT, index=False)
buy_df.to_csv(BUY_PATH_CSV, index=False)

if len(buy_df) > 0:
    log(f"✅ {len(buy_df)} BUY signals saved (with tp & RRR).")
    log(f"📈 Max RRR: {buy_df['RRR'].max():.2f} | Avg RRR: {buy_df['RRR'].mean():.2f}")
else:
    log("ℹ No valid BUY signals (tp/RRR missing or invalid).")

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