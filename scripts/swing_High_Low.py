import pandas as pd
import os
from swing_point import identify_swing_points
from hf_uploader import download_from_hf_or_run_script

# Step 1: Download CSV from HF if needed
if download_from_hf_or_run_script():
    print(f"HF data download success")

# 📁 CSV ফাইল path

# Output paths
low_candle_path = './csv/swing/swing_low/low_candle/'
low_confirm_path = './csv/swing/swing_low/low_confirm/'
high_candle_path = './csv/swing/swing_high/high_candle/'
high_confirm_path = './csv/swing/swing_high/high_confirm/'

# Ensure directories exist
for path in [low_candle_path, low_confirm_path, high_candle_path, high_confirm_path]:
    os.makedirs(path, exist_ok=True)

# Read master CSV
mongodb_data = pd.read_csv('./csv/mongodb.csv')

# Group by symbol
symbol_group = mongodb_data.groupby('symbol')

# Processing loop
for symbol, df in symbol_group:
    df = df.reset_index(drop=True)  # Ensure integer index for iloc
    swing_lows, swing_highs = identify_swing_points(df)

    # ---- Swing Highs ----
    if swing_highs:
        high_candle_rows = [df.iloc[candle_idx] for candle_idx, _ in swing_highs]
        high_confirm_rows = [df.iloc[confirm_idx] for _, confirm_idx in swing_highs]

        new_high_candle_df = pd.DataFrame(high_candle_rows)
        new_high_confirm_df = pd.DataFrame(high_confirm_rows)

        candle_file = f'{high_candle_path}{symbol}.csv'
        confirm_file = f'{high_confirm_path}{symbol}.csv'

        for file_path, new_df in [(candle_file, new_high_candle_df), (confirm_file, new_high_confirm_df)]:
            if os.path.exists(file_path):
                existing = pd.read_csv(file_path)
                combined = pd.concat([existing, new_df])
                combined = combined.drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
            else:
                combined = new_df
            combined.to_csv(file_path, index=False)

    # ---- Swing Lows ----
    if swing_lows:
        low_candle_rows = [df.iloc[candle_idx] for candle_idx, _ in swing_lows]
        low_confirm_rows = [df.iloc[confirm_idx] for _, confirm_idx in swing_lows]

        new_low_candle_df = pd.DataFrame(low_candle_rows)
        new_low_confirm_df = pd.DataFrame(low_confirm_rows)

        candle_file = f'{low_candle_path}{symbol}.csv'
        confirm_file = f'{low_confirm_path}{symbol}.csv'

        for file_path, new_df in [(candle_file, new_low_candle_df), (confirm_file, new_low_confirm_df)]:
            if os.path.exists(file_path):
                existing = pd.read_csv(file_path)
                combined = pd.concat([existing, new_df])
                combined = combined.drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
            else:
                combined = new_df
            combined.to_csv(file_path, index=False)
