import pandas as pd
import os
import json
import numpy as np

# ---------------------------------------------------------
# 🔧 Load config (only 2 params needed)
# ---------------------------------------------------------
CONFIG_PATH = "./config.json"
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    TOTAL_CAPITAL = float(config.get("total_capital", 500000))
    RISK_PERCENT = float(config.get("risk_percent", 0.01))
    print(f"✅ Config: capital={TOTAL_CAPITAL:,.0f} BDT, risk={RISK_PERCENT*100:.1f}% per trade")
except Exception as e:
    print(f"⚠️ Config load failed → using defaults: 5,00,000 BDT, 1% risk")
    TOTAL_CAPITAL = 500000
    RISK_PERCENT = 0.01

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
        low_idx = (abs(sym_data['date'] - low_candle['date'])).idxmin()

    # Scan backward
    for i in range(low_idx - 1, 1, -1):
        try:
            sb = sym_data.iloc[i - 2]
            sa = sym_data.iloc[i - 1]
            s  = sym_data.iloc[i]
        except IndexError:
            break

        if not (sb['date'] < sa['date'] < s['date'] < low_candle['date']):
            continue

        if (s['high'] > sa['high']) and (sa['high'] >= sb['high']):
            tp_price = s['high']
            break

    if tp_price is None:
        continue

    # ✅ Calculate position size (DSE-compliant: 1 share allowed)
    risk_per_trade = TOTAL_CAPITAL * RISK_PERCENT
    risk_per_share = buy_price - SL_price
    
    if risk_per_share <= 0:
        continue  # invalid SL
    
    position_size = int(risk_per_trade / risk_per_share)  # floor division
    position_size = max(1, position_size)  # min 1 share (DSE allows it!)
    
    exposure_bdt = position_size * buy_price
    actual_risk_bdt = position_size * risk_per_share

    # Append
    results.append({
        'symbol': symbol,
        'date': date.date(),
        'buy': buy_price,
        'SL': SL_price,
        'tp': tp_price,
        'position_size': position_size,      # ✅ NEW
        'exposure_bdt': round(exposure_bdt, 2),  # ✅ NEW
        'actual_risk_bdt': round(actual_risk_bdt, 2)  # ✅ NEW
    })

# Build final DataFrame
if results:
    df = pd.DataFrame(results)
    df['buy'] = pd.to_numeric(df['buy'], errors='coerce')
    df['SL'] = pd.to_numeric(df['SL'], errors='coerce')
    df['tp'] = pd.to_numeric(df['tp'], errors='coerce')
    df['position_size'] = df['position_size'].astype(int)
    df['exposure_bdt'] = pd.to_numeric(df['exposure_bdt'], errors='coerce')
    df['actual_risk_bdt'] = pd.to_numeric(df['actual_risk_bdt'], errors='coerce')

    # Compute diff & RRR
    df['diff'] = (df['buy'] - df['SL']).round(4)
    df['RRR'] = ((df['tp'] - df['buy']) / (df['buy'] - df['SL'])).round(2)

    # Filter valid signals
    df = df[
        (df['buy'] > df['SL']) &
        (df['tp'] > df['buy']) &
        (df['RRR'] > 0)
    ].reset_index(drop=True)

    if len(df) > 0:
        # Sort: highest RRR first → then smallest risk (diff) first
        df = df.sort_values(['RRR', 'diff'], ascending=[False, True]).reset_index(drop=True)
        df.insert(0, 'No', range(1, len(df) + 1))
        
        # ✅ Final column order (with position info)
        df = df[[
            'No', 'symbol', 'date', 'buy', 'SL', 'tp',
            'position_size', 'exposure_bdt', 'actual_risk_bdt',
            'diff', 'RRR'
        ]]
    else:
        df = pd.DataFrame(columns=[
            'No', 'symbol', 'date', 'buy', 'SL', 'tp',
            'position_size', 'exposure_bdt', 'actual_risk_bdt',
            'diff', 'RRR'
        ])
else:
    df = pd.DataFrame(columns=[
        'No', 'symbol', 'date', 'buy', 'SL', 'tp',
        'position_size', 'exposure_bdt', 'actual_risk_bdt',
        'diff', 'RRR'
    ])

# Save
df.to_csv(output_file1, index=False)
df.to_csv(output_file2, index=False)

print(f"✅ gape_buy.csv saved with {len(df)} signals:")
if len(df) > 0:
    print(f"   📈 Max RRR: {df['RRR'].max():.2f} | Avg RRR: {df['RRR'].mean():.2f}")
    print(f"   📉 Min diff: {df['diff'].min():.4f}")
    print(f"   💰 Avg position: {df['position_size'].mean():.0f} shares | Max: {df['position_size'].max()}")
    print(f"   🎯 Avg risk/trade: {df['actual_risk_bdt'].mean():,.0f} BDT (target: {TOTAL_CAPITAL*RISK_PERCENT:,.0f})")