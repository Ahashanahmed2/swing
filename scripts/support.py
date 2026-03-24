import pandas as pd
import numpy as np
import os
from datetime import datetime

def process_stock_data(input_file, output_file):
    """
    Process stock data from CSV file and find support levels
    Stop loop after finding first valid support
    """

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Ensure date column is datetime type
    df['date'] = pd.to_datetime(df['date'])

    # Sort by symbol and date
    df = df.sort_values(['symbol', 'date'], ascending=[True, False])

    results = []

    # Process each symbol separately
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date', ascending=True)

        if len(symbol_data) < 2:
            continue

        # Get the latest row
        latest_row = symbol_data.iloc[-1]
        a_price = latest_row['low']
        a_date = latest_row['date']
        a_close = latest_row['close']

        print(f"\nProcessing {symbol}: Latest low = {a_price} at {a_date.strftime('%Y-%m-%d')}")

        # Flag to track if we found support
        support_found = False

        # Process previous rows (excluding the latest)
        for i in range(len(symbol_data) - 2, -1, -1):
            current_row = symbol_data.iloc[i]
            b_low = current_row['low']
            b_date = current_row['date']
            b_close = current_row['close']

            # Check if current low is below latest low
            if b_low < a_price:
                print(f"  → Found low ({b_low}) below latest low ({a_price}) at {b_date.strftime('%Y-%m-%d')}")
                print(f"  ⛔ Stopping loop for {symbol} - no support found")
                break  # Stop loop for this symbol

            # Check for support level (b_low == a_price)
            if b_low == a_price:
                print(f"  🔍 Found potential support at {b_date.strftime('%Y-%m-%d')} with low = {b_low}")

                # Check rows between b and a
                rows_between = symbol_data.iloc[i+1:-1]
                gap_count = len(rows_between)

                valid_support = True

                # Check each row between b and a
                for _, row in rows_between.iterrows():
                    if row['low'] < b_low:
                        print(f"    ❌ Invalid: Intermediate row at {row['date'].strftime('%Y-%m-%d')} has low {row['low']} < {b_low}")
                        valid_support = False
                        break

                if valid_support and gap_count > 0:
                    print(f"    ✅ FIRST VALID SUPPORT FOUND! Gap = {gap_count} days")
                    results.append({
                        'a_date': a_date.strftime('%Y-%m-%d'),
                        'b_date': b_date.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'close': a_close,
                        'low': a_price,
                        'gap': gap_count
                    })
                    support_found = True
                    break  # Exit loop after finding first valid support

                elif gap_count == 0:
                    print(f"    ❌ Invalid: No gap (consecutive days)")
                else:
                    print(f"    ❌ Invalid: Support condition failed")

        if not support_found:
            if not any(symbol_data.iloc[i]['low'] < a_price for i in range(len(symbol_data) - 1)):
                print(f"  ℹ️ No support found and no low below {a_price} for {symbol}")

    # Create output dataframe
    if results:
        output_df = pd.DataFrame(results)
        output_df = output_df[['a_date', 'b_date', 'symbol', 'close', 'low', 'gap']]
        output_df.columns = ['date', 'support_date', 'symbol', 'close', 'low', 'gap']

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save to CSV
        output_df.to_csv(output_file, index=False)
        print(f"\n✅ Successfully saved {len(output_df)} records to {output_file}")
        print("\n📊 Sample output:")
        print(output_df.head())

        # Print summary statistics
        print("\n📈 Summary Statistics:")
        print(f"Total records: {len(output_df)}")
        print(f"Unique symbols: {output_df['symbol'].nunique()}")
        print(f"Gap range: {output_df['gap'].min()} - {output_df['gap'].max()}")
        print(f"Low range: {output_df['low'].min()} - {output_df['low'].max()}")

    else:
        print("\n❌ No matching patterns found")
        empty_df = pd.DataFrame(columns=['date', 'support_date', 'symbol', 'close', 'low', 'gap'])
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        empty_df.to_csv(output_file, index=False)
        print(f"📁 Created empty file with headers at {output_file}")

def main():
    # Define file paths
    input_file = './csv/mongodb.csv'
    output_file_1 = './output/ai_signal/support.csv'
    output_file_2 = './csv/support.csv'

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file {input_file} not found!")
        return

    # Process the data for first output
    print("=" * 60)
    print("Processing for first output file:")
    print("=" * 60)
    process_stock_data(input_file, output_file_1)

    # Process the data for second output
    print("\n" + "=" * 60)
    print("Processing for second output file:")
    print("=" * 60)
    process_stock_data(input_file, output_file_2)

    print("\n✨ Processing complete!")
    print(f"📁 First output: {output_file_1}")
    print(f"📁 Second output: {output_file_2}")

if __name__ == "__main__":
    main()