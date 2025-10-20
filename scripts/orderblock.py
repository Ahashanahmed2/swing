import os
import pandas as pd
import glob

# Paths
candle_dir = './csv/swing/candle/'
orderblock_dir = './csv/orderblock/'

# Ensure orderblock directory exists
os.makedirs(orderblock_dir, exist_ok=True)

# Process each candle file
for candle_file in glob.glob(os.path.join(candle_dir, '*.csv')):
    symbol = os.path.splitext(os.path.basename(candle_file))[0]
    candle_df = pd.read_csv(candle_file)
    candle_df['date'] = pd.to_datetime(candle_df['date'])

    # Prepare orderblock rows
    ob_rows = candle_df[['symbol', 'date', 'high', 'low']].copy()
    ob_rows.rename(columns={
        'high': 'orderblock High',
        'low': 'orderblock low'
    }, inplace=True)
    ob_rows['FVG high'] = ob_rows['orderblock High']
    ob_rows['FVG low'] = ob_rows['orderblock low']

    # Load existing orderblock file if exists
    ob_file = os.path.join(orderblock_dir, f'{symbol}.csv')
    if os.path.exists(ob_file):
        existing_df = pd.read_csv(ob_file)
        existing_df['date'] = pd.to_datetime(existing_df['date'])
        last_date = existing_df['date'].max()
        new_rows = ob_rows[ob_rows['date'] > last_date]
        combined_df = pd.concat([existing_df, new_rows], ignore_index=True)
    else:
        combined_df = ob_rows

    # Sort by date
    combined_df.sort_values('date', inplace=True)
    combined_df.to_csv(ob_file, index=False)

print("✅ Orderblock CSV files created/updated successfully.")
