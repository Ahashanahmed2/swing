import pandas as pd
import os

# Create output directory if it doesn't exist
os.makedirs('./output/ai_signal', exist_ok=True)

# Read the CSV file with proper encoding to handle BOM
df = pd.read_csv('./csv/mongodb.csv', encoding='utf-8-sig')

# Clean column names - remove any BOM or special characters
df.columns = df.columns.str.replace('ï»¿', '').str.strip()

# Clean symbol column - remove any non-alphanumeric characters except common ones
df['symbol'] = df['symbol'].astype(str).str.replace(r'[^a-zA-Z0-9\.\-]', '', regex=True)

# Remove rows with empty symbols
df = df[df['symbol'].str.strip() != '']

# Convert date column to datetime for proper sorting
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Remove rows with invalid dates
df = df.dropna(subset=['date'])

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
    try:
        # Sort by date to ensure correct ordering
        group = group.sort_values('date')

        # Check if we have all required columns
        required_cols = ['macd', 'macd_signal', 'rsi']
        if not all(col in group.columns for col in required_cols):
            print(f"Warning: Missing required columns for {symbol}")
            continue

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
                'close': round(float(last_row['close']), 2) if pd.notna(last_row['close']) else 0,
                'prm': round(float(previous_row['macd']), 2) if pd.notna(previous_row['macd']) else 0,
                'lrm': round(float(last_row['macd']), 2) if pd.notna(last_row['macd']) else 0,
                'rsi': round(float(last_row['rsi']), 2) if pd.notna(last_row['rsi']) else 0
            })
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        continue

# Create result DataFrame
if results:
    result_df = pd.DataFrame(results)
    
    # Add serial number
    result_df.insert(0, 'No', range(1, len(result_df) + 1))
    
    # Reorder columns
    result_df = result_df[['No', 'symbol', 'close', 'prm', 'lrm', 'rsi']]
    
    print(f"Process completed. Found {len(result_df)} symbols meeting the criteria.")
    print("\nFirst few results:")
    print(result_df.head())
else:
    print("No symbols found meeting all conditions.")
    # Create empty DataFrame with headers
    result_df = pd.DataFrame(columns=['No', 'symbol', 'close', 'prm', 'lrm', 'rsi'])
    print("Empty DataFrame created with headers.")

# Save to CSV (এই লাইনটি if-else ব্লকের বাইরে)
result_df.to_csv('./output/ai_signal/macd.csv', index=False)
print(f"Results saved to: ./output/ai_signal/macd.csv")