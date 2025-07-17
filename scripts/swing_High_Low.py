import pandas as pd
from collections import defaultdict
import os
from swing_point import identify_swing_points  # ধরে নেওয়া হয়েছে এটি ঠিকমতো কাজ করে
from hf_uploader import download_from_hf_or_run_script


download_from_hf_or_run_script()
# Output paths
low_candle_path = './csv/swing/swing_low/low_candle/'
low_confirm_path = './csv/swing/swing_low/low_confirm/'
high_candle_path = './csv/swing/swing_high/high_candle/'
high_confirm_path = './csv/swing/swing_high/high_confirm/'

# Create directories if they don't exist
for path in [low_candle_path, low_confirm_path, high_candle_path, high_confirm_path]:
    os.makedirs(path, exist_ok=True)

# Read master CSV
mongodb_data = pd.read_csv('./csv/mongodb.csv')

# Group by symbol
symbol_group = mongodb_data.groupby('symbol')

# Main processing loop
for symbol, df in symbol_group:
    # Identify swing points
    swing_lows, swing_highs = identify_swing_points(df)

    # ---- Swing Highs ----
    if swing_highs:
        high_candle_rows = [df.loc[candle_idx] for candle_idx, _ in swing_highs]
        high_confirm_rows = [df.loc[confirm_idx] for _, confirm_idx in swing_highs]

        new_high_candle_df = pd.DataFrame(high_candle_rows)
        new_high_confirm_df = pd.DataFrame(high_confirm_rows)

        candle_file = f'{high_candle_path}{symbol}.csv'
        confirm_file = f'{high_confirm_path}{symbol}.csv'

        # Handle candle file
        if os.path.exists(candle_file):
            existing = pd.read_csv(candle_file)
            combined = pd.concat([existing, new_high_candle_df])
            combined = combined.drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
        else:
            combined = new_high_candle_df
        combined.to_csv(candle_file, index=False)

        # Handle confirm file
        if os.path.exists(confirm_file):
            existing = pd.read_csv(confirm_file)
            combined = pd.concat([existing, new_high_confirm_df])
            combined = combined.drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
        else:
            combined = new_high_confirm_df
        combined.to_csv(confirm_file, index=False)

    # ---- Swing Lows ----
    if swing_lows:
        low_candle_rows = [df.loc[candle_idx] for candle_idx, _ in swing_lows]
        low_confirm_rows = [df.loc[confirm_idx] for _, confirm_idx in swing_lows]

        new_low_candle_df = pd.DataFrame(low_candle_rows)
        new_low_confirm_df = pd.DataFrame(low_confirm_rows)

        candle_file = f'{low_candle_path}{symbol}.csv'
        confirm_file = f'{low_confirm_path}{symbol}.csv'

        # Handle candle file
        if os.path.exists(candle_file):
            existing = pd.read_csv(candle_file)
            combined = pd.concat([existing, new_low_candle_df])
            combined = combined.drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
        else:
            combined = new_low_candle_df
        combined.to_csv(candle_file, index=False)

        # Handle confirm file
        if os.path.exists(confirm_file):
            existing = pd.read_csv(confirm_file)
            combined = pd.concat([existing, new_low_confirm_df])
            combined = combined.drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
        else:
            combined = new_low_confirm_df
        combined.to_csv(confirm_file, index=False)