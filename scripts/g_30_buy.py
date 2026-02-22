import pandas as pd
import os
import json
import numpy as np

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
                    'SL': base_low
                })

            # EXIT rule
            if m_rsi > 30:
                log(f"  ❌ EXIT {symbol}: RSI={m_rsi:.2f} > 30")
                rsi30_final = rsi30_final[rsi30_final['sl'] != sl]
                break

# ---------------------------------------------------------
# ✅ SAVE BUY SIGNALS (শুধু বেসিক তথ্য)
# ---------------------------------------------------------
if buy_rows:
    buy_df = pd.DataFrame(buy_rows)

    # Filter valid signals (buy > SL)
    buy_df = buy_df[buy_df['buy'] > buy_df['SL']].reset_index(drop=True)

    if len(buy_df) > 0:
        # Sort by date (newest first)
        buy_df = buy_df.sort_values(['date'], ascending=[False]).reset_index(drop=True)
        buy_df.insert(0, 'No', range(1, len(buy_df) + 1))

        # Format date
        buy_df['date'] = buy_df['date'].dt.strftime("%Y-%m-%d")

        # ✅ Final columns (শুধু বেসিক তথ্য)
        buy_df = buy_df[[
            'No', 'date', 'symbol', 'buy', 'SL'
        ]]
    else:
        buy_df = pd.DataFrame(columns=[
            'No', 'date', 'symbol', 'buy', 'SL'
        ])
else:
    buy_df = pd.DataFrame(columns=[
        'No', 'date', 'symbol', 'buy', 'SL'
    ])

# -------------------------------
# Load existing data and merge (remove duplicates)
# -------------------------------
# Check both possible output paths for existing data
existing_df = pd.DataFrame()
if os.path.exists(BUY_PATH_CSV):
    existing_df = pd.read_csv(BUY_PATH_CSV)
    log(f"📂 Existing file loaded from {BUY_PATH_CSV} with {len(existing_df)} signals")
elif os.path.exists(BUY_PATH_OUTPUT):
    existing_df = pd.read_csv(BUY_PATH_OUTPUT)
    log(f"📂 Existing file loaded from {BUY_PATH_OUTPUT} with {len(existing_df)} signals")

# DELETE OLD FILES
for f in [BUY_PATH_CSV, BUY_PATH_OUTPUT]:
    if os.path.exists(f):
        os.remove(f)

# Merge with existing data
if not existing_df.empty and not buy_df.empty:
    # Check if existing file has the correct format
    required_cols = ['symbol', 'date', 'buy', 'SL']
    if not all(col in existing_df.columns for col in required_cols):
        # If old format, use new data
        log("🔄 Existing file has old format - creating new file with updated format")
        final_df = buy_df.copy()
    else:
        # Get existing symbols
        existing_symbols = set(existing_df['symbol'].unique())
        
        # Filter out symbols that already exist
        new_symbols_df = buy_df[~buy_df['symbol'].isin(existing_symbols)]
        
        if not new_symbols_df.empty:
            # Adjust 'No' column for new symbols
            new_symbols_df = new_symbols_df.reset_index(drop=True)
            new_symbols_df.insert(0, 'No', range(1, len(new_symbols_df) + 1))
            
            # Combine existing and new data
            final_df = pd.concat([existing_df, new_symbols_df], ignore_index=True)
            log(f"➕ Added {len(new_symbols_df)} new symbols to existing {len(existing_df)} signals")
        else:
            final_df = existing_df
            log("⏭️ No new symbols to add")
        
elif not buy_df.empty:
    final_df = buy_df
    log(f"🆕 Created new file with {len(buy_df)} signals")
else:
    final_df = existing_df if not existing_df.empty else pd.DataFrame()
    log("⚠️ No signals found")

# Save final DataFrame to both locations
if not final_df.empty:
    # Ensure correct column order
    column_order = ['No', 'date', 'symbol', 'buy', 'SL']
    
    # Make sure all columns exist
    for col in column_order:
        if col not in final_df.columns:
            if col == 'No':
                final_df.insert(0, 'No', range(1, len(final_df) + 1))
            else:
                final_df[col] = None
    
    final_df = final_df[column_order]
    
    # Save to both locations
    final_df.to_csv(BUY_PATH_OUTPUT, index=False)
    final_df.to_csv(BUY_PATH_CSV, index=False)
    
    log(f"✅ {len(final_df)} BUY signals saved (basic info only).")
    if len(final_df) > 0:
        log(f"📈 Latest signal: {final_df.iloc[0]['symbol']} - Buy: {final_df.iloc[0]['buy']:.2f}, SL: {final_df.iloc[0]['SL']:.2f}")
    
    # Show newly added symbols if any
    if not existing_df.empty and not buy_df.empty and 'new_symbols_df' in locals() and not new_symbols_df.empty:
        new_added = set(new_symbols_df['symbol'].unique())
        if new_added:
            log(f"   🆕 New symbols added: {', '.join(sorted(new_added))}")
else:
    # Save empty DataFrame with correct columns to both locations
    empty_df = pd.DataFrame(columns=['No', 'date', 'symbol', 'buy', 'SL'])
    empty_df.to_csv(BUY_PATH_OUTPUT, index=False)
    empty_df.to_csv(BUY_PATH_CSV, index=False)
    log("⚠️ No valid BUY signals - empty file saved")

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