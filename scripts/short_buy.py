import pandas as pd
import os
import json
import numpy as np

# ---------------------------------------------------------
# 🔧 Load config.json (only 2 params)
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

# Paths
rsi_path = './csv/rsi_diver_retest.csv'
mongo_path = './csv/mongodb.csv'
# ⚠️ Note: saving as 
output_path2 = './csv/short_buy.csv'


os.makedirs(os.path.dirname(output_path2), exist_ok=True)

# Delete old files
for path in [output_path2]:
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

    if tp_price is None:
        continue

    # ✅ DSE-COMPLIANT POSITION SIZING (1 share allowed)
    risk_per_trade = TOTAL_CAPITAL * RISK_PERCENT
    risk_per_share = buy_price - SL_price  # since buy > SL for long

    if risk_per_share <= 0:
        continue

    position_size = int(risk_per_trade / risk_per_share)  # floor
    position_size = max(1, position_size)  # min 1 share (DSE allows it!)

    exposure_bdt = position_size * buy_price
    actual_risk_bdt = position_size * risk_per_share

    # Append with position info
    output_rows.append({
        'symbol': symbol,
        'date': last_row['date'].date(),
        'buy': buy_price,
        'SL': SL_price,
        'tp': tp_price,
        'position_size': position_size,
        'exposure_bdt': round(exposure_bdt, 2),
        'actual_risk_bdt': round(actual_risk_bdt, 2)
    })

# Build DataFrame
if output_rows:
    df = pd.DataFrame(output_rows)
    df['buy'] = pd.to_numeric(df['buy'], errors='coerce')
    df['SL'] = pd.to_numeric(df['SL'], errors='coerce')
    df['tp'] = pd.to_numeric(df['tp'], errors='coerce')
    df['position_size'] = df['position_size'].astype(int)
    df['exposure_bdt'] = pd.to_numeric(df['exposure_bdt'], errors='coerce')
    df['actual_risk_bdt'] = pd.to_numeric(df['actual_risk_bdt'], errors='coerce')

    # Compute RRR & diff
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
        
        # ✅ Final column order (with position sizing)
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

df.to_csv(output_path2, index=False)

print(f"✅ short_buy.csv saved with {len(df)} signals (DSE-compliant position sizing)")
if len(df) > 0:
    print(f"   📈 Max RRR: {df['RRR'].max():.2f} | Avg RRR: {df['RRR'].mean():.2f}")
    print(f"   📉 Min diff: {df['diff'].min():.4f}")
    print(f"   💰 Avg position: {df['position_size'].mean():.0f} shares | Max: {df['position_size'].max()}")
    print(f"   🎯 Avg actual risk: {df['actual_risk_bdt'].mean():,.0f} BDT (target: {TOTAL_CAPITAL*RISK_PERCENT:,.0f})")