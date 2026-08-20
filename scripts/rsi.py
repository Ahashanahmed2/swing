import pandas as pd
import os
from datetime import datetime

def extract_rsi_below_30():
    """
    Extract rows with RSI <= 30 from mongodb.csv and save to rsi.csv
    with new serial numbers starting from 1
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
        
        # Step 2: Check if 'rsi' column exists
        if 'rsi' not in df.columns:
            print("Error: 'rsi' column not found in the CSV file.")
            print("Available columns:", list(df.columns))
            return
        
        # Step 3: Filter rows where rsi <= 30
        filtered_df = df[df['rsi'] <= 30].copy()
        print(f"Rows with RSI <= 30: {len(filtered_df)}")
        
        if len(filtered_df) == 0:
            print("No rows found with RSI <= 30. Exiting.")
            return
        
        # Step 4: Check required columns
        required_cols = ['date', 'symbol', 'rsi']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Error: Missing columns: {missing_cols}")
            print(f"Required: {required_cols}")
            return
        
        # Step 5: Select only required columns (excluding any existing sl)
        result_df = filtered_df[['date', 'symbol', 'rsi']].copy()
        
        # Step 6: Sort by rsi ascending (lowest RSI first)
        result_df = result_df.sort_values('rsi', ascending=True)
        
        # Step 7: Reset index and create new 'sl' column
        result_df.reset_index(drop=True, inplace=True)
        result_df.insert(0, 'sl', range(1, len(result_df) + 1))
        
        # Step 8: Save to CSV
        result_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully saved to: {output_file}")
        
        # Step 9: Display summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Total records saved: {len(result_df)}")
        print(f"Columns: {', '.join(result_df.columns)}")
        print(f"RSI Range: {result_df['rsi'].min():.2f} to {result_df['rsi'].max():.2f}")
        print(f"Date Range: {result_df['date'].min()} to {result_df['date'].max()}")
        
        print("\nFirst 5 rows:")
        print(result_df.head())
        
        print("\nLast 5 rows:")
        print(result_df.tail())
        
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
