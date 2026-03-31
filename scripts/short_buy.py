import pandas as pd
import os

# Paths
rsi_path = './csv/rsi_diver.csv'
mongo_path = './csv/mongodb.csv'
output_path2 = './csv/short_buy.csv'

os.makedirs(os.path.dirname(output_path2), exist_ok=True)

# Delete old file
if os.path.exists(output_path2):
    os.remove(output_path2)

# Read data
rsi_df = pd.read_csv(rsi_path)
mongo_df = pd.read_csv(mongo_path)

# Clean column names (spaces to underscores)
rsi_df.columns = rsi_df.columns.str.replace(" ", "_")
mongo_df.columns = mongo_df.columns.str.replace(" ", "_")

# Debug: Print columns to verify
print("RSI columns:", rsi_df.columns.tolist())
print("MongoDB columns:", mongo_df.columns.tolist())

# Ensure dates are datetime
mongo_df['date'] = pd.to_datetime(mongo_df['date'], errors='coerce')

# Create date column in rsi_df from last_row_date
if 'last_row_date' in rsi_df.columns:
    rsi_df['date'] = pd.to_datetime(rsi_df['last_row_date'], errors='coerce')
else:
    print("Error: 'last_row_date' column not found in rsi_df")
    exit(1)

# Drop rows with null dates
mongo_df = mongo_df.dropna(subset=['date'])
rsi_df = rsi_df.dropna(subset=['date'])

# Also drop rows with null low values
rsi_df = rsi_df.dropna(subset=['last_row_low'])

# Check if 'close' column exists in mongo_df
if 'close' not in mongo_df.columns:
    print("Warning: 'close' column not found in mongo_df. Creating from 'high' and 'low'?")
    # If close doesn't exist, you might need to handle it
    # For now, let's add a placeholder
    mongo_df['close'] = (mongo_df['high'] + mongo_df['low']) / 2

# Get latest row for each symbol from MongoDB
latest_rows = mongo_df.sort_values('date').groupby('symbol').last().reset_index()

output_rows = []

print("\n" + "="*80)
print("PROCESSING SIGNALS (DIVERGENCE LOW -> NEXT LOW IS LATEST LOW):")
print("="*80)

for _, rsi_row in rsi_df.iterrows():
    symbol = str(rsi_row['symbol']).strip().upper()
    divergence_low = rsi_row['last_row_low']
    divergence_date = rsi_row['date']

    # Check if symbol exists in latest_rows
    symbol_data = latest_rows[latest_rows['symbol'] == symbol]

    if symbol_data.empty:
        continue

    # Get the latest row for this symbol
    latest_row = symbol_data.iloc[0]

    # Get all rows for this symbol from mongo_df after divergence date
    symbol_all_rows = mongo_df[mongo_df['symbol'] == symbol]
    symbol_all_rows = symbol_all_rows[symbol_all_rows['date'] > divergence_date]

    # Check if there are any rows after divergence
    if len(symbol_all_rows) == 0:
        continue

    # Sort by date to get the first row after divergence
    symbol_all_rows = symbol_all_rows.sort_values('date')

    # Get the first row after divergence (immediate next row)
    first_after_divergence = symbol_all_rows.iloc[0]

    # Check if the latest low is the first low after divergence
    # and it's greater than divergence low
    if (latest_row['low'] == first_after_divergence['low'] and 
        latest_row['low'] > divergence_low):

        # Store signal
        output_rows.append({
            'symbol': symbol,
            'divergence_low': divergence_low,
            'divergence_date': divergence_date.date(),
            'date': latest_row['date'].date(),
            'low': latest_row['low'],
            'close': latest_row['close'],
            'high': latest_row['high']
        })

        print(f"✅ Signal: {symbol} | "
              f"Divergence Low={divergence_low:.2f} on {divergence_date.date()} | "
              f"Next Low={latest_row['low']:.2f} on {latest_row['date'].date()} | "
              f"Close={latest_row['close']:.2f}")

# Create DataFrame
if output_rows:
    df = pd.DataFrame(output_rows)

    # Remove duplicates if any
    df = df.drop_duplicates(subset=['symbol'])

    # Sort by symbol
    df = df.sort_values('symbol').reset_index(drop=True)

    # Add serial number as first column
    df.insert(0, 'no', range(1, len(df) + 1))

    # Set column order
    df = df[['no', 'symbol', 'divergence_low', 'divergence_date', 'date', 'low', 'close', 'high']]

    print(f"\n{'='*80}")
    print(f"✅ TOTAL SIGNALS: {len(df)}")
    print(f"{'='*80}")
    print(df.to_string())

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS:")
    print(f"{'='*80}")
    print(f"Average low: {df['low'].mean():.2f}")
    print(f"Max low: {df['low'].max():.2f}")
    print(f"Min low: {df['low'].min():.2f}")
    print(f"\nAverage close: {df['close'].mean():.2f}")
    print(f"Max close: {df['close'].max():.2f}")
    print(f"Min close: {df['close'].min():.2f}")

else:
    print(f"\n❌ No signals generated")
    df = pd.DataFrame(columns=['no', 'symbol', 'divergence_low', 'divergence_date', 'date', 'low', 'close', 'high'])

# Save to CSV
df.to_csv(output_path2, index=False)
print(f"\n✅ Saved to {output_path2}")
print(f"📁 File contains columns: {', '.join(df.columns)}")