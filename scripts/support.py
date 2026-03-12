import pandas as pd
import numpy as np
import os
from datetime import datetime

def process_stock_data(input_file, output_file):
    """
    Process stock data from CSV file and find support levels
    
    Args:
        input_file: path to input CSV file (mongodb.csv)
        output_file: path to output CSV file (./output/ai_signal/support.csv)
    """

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Ensure date column is datetime type
    df['date'] = pd.to_datetime(df['date'])

    # Sort by symbol and date (descending to get latest first)
    df = df.sort_values(['symbol', 'date'], ascending=[True, False])

    results = []

    # Process each symbol separately
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date', ascending=True)  # Sort ascending for processing

        if len(symbol_data) < 2:
            continue

        # Get the latest row (last row after sorting ascending)
        latest_row = symbol_data.iloc[-1]
        a_price = latest_row['low']
        a_date = latest_row['date']

        # Process previous rows (excluding the latest)
        for i in range(len(symbol_data) - 2, -1, -1):
            current_row = symbol_data.iloc[i]
            b_low = current_row['low']
            b_high = current_row['high']
            b_date = current_row['date']

            # Check if a_price is within b's low-high range
            if b_low <= a_price <= b_high:
                # Found potential support level at row b
                # Now check rows between b and a
                rows_between = symbol_data.iloc[i+1:-1]  # Exclude b and a

                valid_support = True
                gap_count = len(rows_between)
                
                # Check if any row's low is below both b_low and a_price
                low_below_both = False
                
                # Check each row between b and a
                for _, row in rows_between.iterrows():
                    # Original condition: If any row's low is within b's low-high range, it's not valid
                    if b_low <= row['low'] <= b_high:
                        valid_support = False
                        break
                    
                    # NEW CONDITION: Check if any row's low is below BOTH b_low and a_price
                    if row['low'] < b_low and row['low'] < a_price:
                        low_below_both = True
                        # No need to break, we can continue checking but mark as invalid

                # Only add to results if:
                # 1. Valid support (no row's low within b's range)
                # 2. No rows with low below both b_low and a_price
                # 3. At least one gap
                if valid_support and not low_below_both and gap_count > 0:
                    results.append({
                        'a_date': a_date.strftime('%Y-%m-%d'),
                        'b_date': b_date.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'close': latest_row['close'],
                        'gap': gap_count,
                        'a_low': a_price,
                        'b_low': b_low,
                        'b_high': b_high
                    })

    # Create output dataframe
    if results:
        output_df = pd.DataFrame(results)
        # Select required columns (including b_date)
        output_df = output_df[['a_date', 'b_date', 'symbol', 'close', 'gap']]

        # Rename columns for better understanding
        output_df.columns = ['date', 'support_date', 'symbol', 'close', 'gap']

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save to CSV
        output_df.to_csv(output_file, index=False)
        print(f"Successfully saved {len(output_df)} records to {output_file}")
        print("\nSample output:")
        print(output_df.head())

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total records: {len(output_df)}")
        print(f"Unique symbols: {output_df['symbol'].nunique()}")
        print(f"Gap range: {output_df['gap'].min()} - {output_df['gap'].max()}")

    else:
        print("No matching patterns found")
        # Create empty dataframe with headers
        empty_df = pd.DataFrame(columns=['date', 'support_date', 'symbol', 'close', 'gap'])
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        empty_df.to_csv(output_file, index=False)
        print(f"Created empty file with headers at {output_file}")

def main():
    # Define file paths
    input_file = './csv/mongodb.csv'
    output_file = './output/ai_signal/support.csv'

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return

    # Process the data
    process_stock_data(input_file, output_file)

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()