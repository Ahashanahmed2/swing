import pandas as pd
import os
from glob import glob

# Load mongodb.csv
mongo_df = pd.read_csv('./csv/mongodb.csv')

# Prepare output list
output_rows = []

# Group by symbol
for symbol, group in mongo_df.groupby('symbol'):
    latest_row = group.sort_values('date').iloc[-1]
    latest_date = latest_row['date']
    latest_close = latest_row['close']

    ob_file = f'./csv/orderblock/{symbol}.csv'
    if not os.path.exists(ob_file):
        continue

    ob_df = pd.read_csv(ob_file)
    if len(ob_df) < 2 or 'orderblock low' not in ob_df.columns or 'orderblock high' not in ob_df.columns:
        continue

    last = ob_df.iloc[-1]
    second_last = ob_df.iloc[-2]

    ob_low_last = last['orderblock low']
    ob_high_last = last['orderblock high']
    ob_low_second = second_last['orderblock low']
    ob_high_second = second_last['orderblock high']

    # Matching condition
    if (
        ob_high_last < latest_close and
        ob_low_second > ob_high_last and
        ob_high_second < latest_close
    ):
        output_rows.append({
            'SYMBOL': symbol,
            'CLOSE': latest_close,
            'Last OB Date': last['date'] if 'date' in last else '',
            'BOS': ob_low_last,
            'Second Last OB Date': second_last['date'] if 'date' in second_last else '',
            'IDM': ob_high_second
        })

# Save to both locations
if output_rows:
    uptrend_df = pd.DataFrame(output_rows)
    # Save to ./output/uptrand.csv
    # Ensure target folder exists
    os.makedirs('./output/ai_signal', exist_ok=True)
    uptrend_df.to_csv('./output/ai_signal/uptrand.csv', index=False, columns=[
        'SYMBOL', 'CLOSE', 'Last OB Date', 'BOS', 'Second Last OB Date', 'IDM'
    ])
    print("✅ uptrand.csv saved to ./output/ai_signal/")
else:
    print("⚠️ No matching conditions found.")
