import pandas as pd
import os
from swing_point import identify_swing_points
from hf_uploader import download_from_hf_or_run_script

# Step 1: Download CSV from HF if needed
if download_from_hf_or_run_script():
    print("HF data download success")

# Output path for orderblocks
orderblock_path = './csv/orderblock/'
os.makedirs(orderblock_path, exist_ok=True)

# Read master CSV
mongodb_data = pd.read_csv('./csv/mongodb.csv')

# Group by symbol
symbol_group = mongodb_data.groupby('symbol')

# Processing loop
for symbol, df in symbol_group:
    df = df.reset_index(drop=True)
    swing_lows, swing_highs = identify_swing_points(df)

    orderblock_rows = []

    # Swing High Orderblocks
    for candle_idx, confirm_idx in swing_highs:
        candle = df.iloc[candle_idx]
        confirm = df.iloc[confirm_idx]
        orderblock_rows.append({
            'symbol': symbol,
            'date': candle['date'],  # ✅ candle date as orderblock date
            'orderblock low': candle['low'],
            'orderblock high': candle['high'],
            'fvg low': confirm['low'],
            'fvg high': confirm['high'],
        })

    # Swing Low Orderblocks
    for candle_idx, confirm_idx in swing_lows:
        candle = df.iloc[candle_idx]
        confirm = df.iloc[confirm_idx]
        orderblock_rows.append({
            'symbol': symbol,
            'date': candle['date'],  # ✅ candle date as orderblock date
            'orderblock low': candle['low'],
            'orderblock high': candle['high'],
            'fvg low': confirm['low'],
            'fvg high': confirm['high'],
        })

    # Save to CSV
    if orderblock_rows:
        orderblock_df = pd.DataFrame(orderblock_rows)
        file_path = f'{orderblock_path}{symbol}.csv'

        if os.path.exists(file_path):
            existing = pd.read_csv(file_path)
            combined = pd.concat([existing, orderblock_df])
            combined = combined.drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
        else:
            combined = orderblock_df

        combined = combined.sort_values(by='date').reset_index(drop=True)
        combined.to_csv(file_path, index=False)
