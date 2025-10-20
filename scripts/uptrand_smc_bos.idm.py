import pandas as pd
import os
from glob import glob

# Load mongodb.csv
mongo_df = pd.read_csv('./csv/mongodb.csv')

# Prepare output list
output_rows = []

# Group by symbol
for symbol, group in mongo_df.groupby('symbol'):
    # Get latest date row for the symbol
    latest_row = group.sort_values('date').iloc[-1]
    latest_date = latest_row['date']
    latest_close = latest_row['close']

    # Construct orderblock file path
    ob_file = f'./csv/orderblock/{symbol}.csv'
    if not os.path.exists(ob_file):
        continue

    # Load orderblock file
    ob_df = pd.read_csv(ob_file)
    if len(ob_df) < 2:
        continue

    # Get last and second last rows
    last = ob_df.iloc[-1]
    second_last = ob_df.iloc[-2]

    # Extract values
    ob_low_last = last['orderblock low']
    ob_high_last = last['orderblock high']
    ob_low_second = second_last['orderblock low']
    ob_high_second = second_last['orderblock high']

    # Condition check
    if (
        ob_high_last < latest_close and
        ob_low_second > ob_high_last and
        ob_low_second < latest_close
    ):
        # Prepare output row with updated bos = orderblock low
        output_rows.append({
            'symbol': symbol,
            'date': latest_date,
            'close': latest_close,
            'bos': ob_low_last,        # ✅ updated here
            'idm': ob_high_second
        })

# Save to uptrand.csv
if output_rows:
    uptrend_df = pd.DataFrame(output_rows)
    uptrend_df.to_csv('./csv/uptrand.csv', index=False)
    print("✅ uptrand.csv created with", len(output_rows), "rows.")
else:
    print("⚠️ No matching conditions found.")
