import pandas as pd
import os

# Create output directory if it doesn't exist
os.makedirs('./output/ai_signal', exist_ok=True)

# Read the CSV file
df = pd.read_csv('./csv/mongodb.csv')

# Convert date column to datetime for proper sorting
df['date'] = pd.to_datetime(df['date'])

# Sort by symbol and date
df = df.sort_values(['symbol', 'date'])

# Get the last two rows for each symbol
last_two_rows = df.groupby('symbol').tail(2)

# Check if each symbol has at least 2 rows
valid_symbols = last_two_rows.groupby('symbol').filter(lambda x: len(x) == 2)

# Prepare result list
results = []

# Process each symbol
for symbol, group in valid_symbols.groupby('symbol'):
    # Sort by date to ensure correct ordering
    group = group.sort_values('date')
    
    # Get previous and last row
    previous_row = group.iloc[0]
    last_row = group.iloc[1]
    
    # Check conditions
    condition1 = last_row['macd'] > last_row['macd_signal']
    condition2 = previous_row['macd'] < 0
    condition3 = last_row['macd'] > 0
    
    if condition1 and condition2 and condition3:
        results.append({
            'symbol': symbol,
            'close': round(last_row['close'], 2),
            'previous_row_macd': round(previous_row['macd'], 2),
            'last_row_macd': round(last_row['macd'], 2)
        })

# Create result DataFrame
result_df = pd.DataFrame(results)

# Add serial number
result_df.insert(0, 'No', range(1, len(result_df) + 1))

# Rename columns as specified
result_df = result_df.rename(columns={
    'previous_row_macd': 'prm',
    'last_row_macd': 'lrm'
})

# Reorder columns
result_df = result_df[['No', 'symbol', 'close', 'prm', 'lrm']]

# Save to CSV
result_df.to_csv('./output/ai_signal/macd.csv', index=False)

print(f"Process completed. Found {len(result_df)} symbols meeting the criteria.")
print(f"Results saved to: ./output/ai_signal/macd.csv")

# Display first few results if available
if len(result_df) > 0:
    print("\nFirst few results:")
    print(result_df.head())
else:
    print("\nNo symbols found meeting all conditions.")