import pandas as pd
import os

# Step 1: Load the CSV file
input_file = 'csv/swing/down_to_up.csv'  # Replace with your actual file name
df = pd.read_csv(input_file)

# Step 2: Convert date columns to datetime for comparison
df['date'] = pd.to_datetime(df['date'])
df['orderblock_date'] = pd.to_datetime(df['orderblock_date'])

# Step 3: Sort data by symbol and date
df = df.sort_values(by=['symbol', 'date']).reset_index(drop=True)

# Step 4: Group by symbol and apply the logic (only check last two rows)
filtered_rows = []

for symbol, group in df.groupby('symbol'):
    if len(group) < 2:
        continue  # Need at least two rows per symbol
    
    # Only check the last two rows
    prev_row = group.iloc[-2]  # Second last row
    curr_row = group.iloc[-1]  # Last row

    if (prev_row['trend'] == 'DownTrend' and
        curr_row['trend'] == 'UpTrend' and
        curr_row['date'] > prev_row['date']):
        
        filtered_rows.append({
            'symbol': curr_row['symbol'],
            'date': curr_row['date'],
            'orderblock_date': curr_row['orderblock_date'],
            'orderblock_low': curr_row['orderblock_low'],
            'trend': curr_row['trend']
        })

# Step 5: Create a new DataFrame from filtered rows
result_df = pd.DataFrame(filtered_rows)

# Step 6: Sort by date ascending
result_df = result_df.sort_values(by='date').reset_index(drop=True)

# Step 7: Create directory if not exists and save to CSV
output_dir = 'output/ai_signal'
os.makedirs(output_dir, exist_ok=True)  # Create directory if not exists
output_file = os.path.join(output_dir, 'short_trade.csv')
result_df.to_csv(output_file, index=False)

print(f"Filtered data saved to '{output_file}'")
