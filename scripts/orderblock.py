import os
import pandas as pd
import glob

# Define source paths
low_candle_path = './csv/swing/swing_low/low_candle/'
low_confirm_path = './csv/swing/swing_low/low_confirm/'
high_candle_path = './csv/swing/swing_high/high_candle/'
high_confirm_path = './csv/swing/swing_high/high_confirm/'

# Define output path
orderblock_path = './csv/orderblock/'
os.makedirs(orderblock_path, exist_ok=True)

# All source folders
source_folders = {
    'orderblock': [low_candle_path, high_candle_path],
    'fvg': [low_confirm_path, high_confirm_path]
}

# Helper function to load and tag data
def load_and_tag(folder, tag_type):
    data = []
    for file in glob.glob(os.path.join(folder, '*.csv')):
        symbol = os.path.splitext(os.path.basename(file))[0]
        try:
            df = pd.read_csv(file)
            df['symbol'] = symbol
            df['date'] = pd.to_datetime(df['date'])
            df = df[['symbol', 'date', 'high', 'low']].copy()
            df.rename(columns={
                'high': f'{tag_type} High',
                'low': f'{tag_type} low'
            }, inplace=True)
            data.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

# Load and merge all data
orderblock_df = pd.concat([load_and_tag(p, 'orderblock') for p in source_folders['orderblock']], ignore_index=True)
fvg_df = pd.concat([load_and_tag(p, 'FVG') for p in source_folders['fvg']], ignore_index=True)

# Merge both on symbol and date
merged_df = pd.merge(orderblock_df, fvg_df, on=['symbol', 'date'], how='outer')

# Process each symbol
for symbol in merged_df['symbol'].unique():
    symbol_df = merged_df[merged_df['symbol'] == symbol].copy()
    symbol_df.sort_values('date', inplace=True)

    # Output file path
    out_file = os.path.join(orderblock_path, f'{symbol}.csv')

    # If file exists, append only newer dates
    if os.path.exists(out_file):
        existing = pd.read_csv(out_file)
        existing['date'] = pd.to_datetime(existing['date'])
        last_date = existing['date'].max()
        new_data = symbol_df[symbol_df['date'] > last_date]
        final_df = pd.concat([existing, new_data], ignore_index=True)
    else:
        final_df = symbol_df

    # Final sort and save
    final_df.sort_values('date', inplace=True)
    final_df.to_csv(out_file, index=False)

print("✅ All orderblock files generated/updated successfully.")
