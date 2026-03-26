import pandas as pd
import os

# Paths
rsi_path = './csv/rsi_diver_retest.csv'
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

# Ensure MongoDB date is datetime
mongo_df['date'] = pd.to_datetime(mongo_df['date'], errors='coerce')
mongo_df = mongo_df.dropna(subset=['date'])

# Get latest row for each symbol from MongoDB
latest_rows = mongo_df.sort_values('date').groupby('symbol').last().reset_index()

output_rows = []

print("\n" + "="*80)
print("PROCESSING SIGNALS (LATEST LOW > DIVERGENCE LOW):")
print("="*80)

for _, rsi_row in rsi_df.iterrows():
    symbol = str(rsi_row['symbol']).strip().upper()
    divergence_low = rsi_row['last_row_low']
    
    # Check if symbol exists in latest_rows
    symbol_data = latest_rows[latest_rows['symbol'] == symbol]
    
    if symbol_data.empty:
        continue
    
    # Get the latest row for this symbol
    latest_row = symbol_data.iloc[0]
    
    # Check condition: latest low > divergence low
    if latest_row['low'] > divergence_low:
        
        # Store signal
        output_rows.append({
            'symbol': symbol,
            'divergence_low': divergence_low,
            'date': latest_row['date'].date(),
            'low': latest_row['low'],
            'close': latest_row['close'],
            'high': latest_row['high']
        })
        
        print(f"✅ Signal: {symbol} | "
              f"Divergence Low={divergence_low:.2f} | "
              f"Date={latest_row['date'].date()} | "
              f"Low={latest_row['low']:.2f} | "
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
    df = df[['no', 'symbol', 'divergence_low', 'date', 'low', 'close', 'high']]
    
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
    df = pd.DataFrame(columns=['no', 'symbol', 'divergence_low', 'date', 'low', 'close', 'high'])

# Save to CSV
df.to_csv(output_path2, index=False)
print(f"\n✅ Saved to {output_path2}")
print(f"📁 File contains columns: {', '.join(df.columns)}")