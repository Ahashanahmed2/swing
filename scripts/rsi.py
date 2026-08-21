import pandas as pd
import os
from datetime import datetime

def extract_rsi_below_30():
    """
    Extract rows with RSI <= 30 for each symbol's latest date from mongodb.csv
    and save to rsi.csv with new serial numbers starting from 1
    """
    
    # File paths
    input_file = './csv/mongodb.csv'
    output_dir = './output/ai_signal'
    output_file = os.path.join(output_dir, 'rsi.csv')
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating directory: {e}")
        return
    
    try:
        # Step 1: Read the CSV file
        print(f"Reading file: {input_file}")
        df = pd.read_csv(input_file)
        print(f"Total rows in file: {len(df)}")
        print(f"Columns available: {', '.join(df.columns)}")
        
        # Step 2: Check if required columns exist
        required_cols = ['date', 'symbol', 'rsi', 'high']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Error: Missing columns: {missing_cols}")
            print(f"Required: {required_cols}")
            return
        
        # Step 3: Convert date column to datetime for proper sorting
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            print(f"Error converting date column: {e}")
            print("Please make sure the date column is in a valid format.")
            return
        
        # Step 4: Get the latest date for each symbol
        print("\nFinding latest date for each symbol...")
        latest_dates = df.groupby('symbol')['date'].max().reset_index()
        print(f"Total unique symbols: {len(latest_dates)}")
        
        # Step 5: Merge to get only the latest data for each symbol
        latest_data = pd.merge(df, latest_dates, on=['symbol', 'date'], how='inner')
        print(f"Rows with latest data for each symbol: {len(latest_data)}")
        
        # Step 6: From latest data, filter rows where rsi <= 30
        filtered_df = latest_data[latest_data['rsi'] <= 30].copy()
        print(f"Symbols with RSI <= 30 on their latest date: {len(filtered_df)}")
        
        if len(filtered_df) == 0:
            print("\nNo symbols found with RSI <= 30 on their latest date.")
            print("Showing all symbols with their latest RSI values:")
            
            # Show all symbols with their latest RSI
            display_df = latest_data[['symbol', 'date', 'rsi']].copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df = display_df.sort_values('rsi', ascending=True)
            print(display_df.to_string(index=False))
            return
        
        # Step 7: Select only required columns (excluding any existing sl)
        result_df = filtered_df[['date', 'symbol', 'high', 'rsi']].copy()
        
        # Step 8: Sort by rsi ascending (lowest RSI first)
        result_df = result_df.sort_values('rsi', ascending=True)
        
        # Step 9: Format date back to string for output
        result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')
        
        # Step 10: Reset index and create new 'sl' column
        result_df.reset_index(drop=True, inplace=True)
        result_df.insert(0, 'sl', range(1, len(result_df) + 1))
        
        # Step 11: Save to CSV
        result_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully saved to: {output_file}")
        
        # Step 12: Display summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Total symbols processed: {len(latest_dates)}")
        print(f"Symbols with RSI <= 30: {len(result_df)}")
        print(f"Columns: {', '.join(result_df.columns)}")
        print(f"RSI Range: {result_df['rsi'].min():.2f} to {result_df['rsi'].max():.2f}")
        print(f"High Range: {result_df['high'].min():.2f} to {result_df['high'].max():.2f}")
        
        print("\nAll symbols with RSI <= 30 (sorted by RSI):")
        print(result_df.to_string(index=False))
        
        # Step 13: Show all symbols with their latest RSI for comparison
        print("\n" + "="*50)
        print("ALL SYMBOLS WITH LATEST RSI VALUES")
        print("="*50)
        all_latest = latest_data[['symbol', 'date', 'rsi']].copy()
        all_latest['date'] = all_latest['date'].dt.strftime('%Y-%m-%d')
        all_latest = all_latest.sort_values('rsi', ascending=True)
        print(all_latest.to_string(index=False))
        
        print(f"\nFile size: {os.path.getsize(output_file)} bytes")
        print("="*50)
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        print("Please make sure the file exists at the correct path.")
        
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty.")
        
    except pd.errors.ParserError:
        print(f"Error: Could not parse '{input_file}'. Please check if it's a valid CSV file.")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    print("Starting RSI extraction script...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*50)
    extract_rsi_below_30()
    print("\nScript completed.")
