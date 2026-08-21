import pandas as pd
import os
from datetime import datetime

def extract_rsi_below_30():
    """
    Extract rows with RSI <= 30 from mongodb.csv based on latest date
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
        
        # Step 4: Get the latest date from the data
        latest_date = df['date'].max()
        print(f"Latest date in data: {latest_date.strftime('%Y-%m-%d')}")
        
        # Step 5: Filter only the latest date's data
        latest_data = df[df['date'] == latest_date].copy()
        print(f"Rows on latest date: {len(latest_data)}")
        
        # Step 6: From latest data, filter rows where rsi <= 30
        filtered_df = latest_data[latest_data['rsi'] <= 30].copy()
        print(f"Rows with RSI <= 30 on latest date: {len(filtered_df)}")
        
        if len(filtered_df) == 0:
            print(f"No rows found with RSI <= 30 on {latest_date.strftime('%Y-%m-%d')}")
            print("Checking if there are any rows with RSI <= 30 in entire dataset...")
            
            # Check if any rows with RSI <= 30 exist in entire dataset
            any_rsi_below_30 = df[df['rsi'] <= 30]
            if len(any_rsi_below_30) > 0:
                print(f"Found {len(any_rsi_below_30)} rows with RSI <= 30 in entire dataset.")
                print("But none on the latest date. Try checking the previous dates.")
            else:
                print("No rows found with RSI <= 30 anywhere in the dataset.")
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
        print(f"Date: {latest_date.strftime('%Y-%m-%d')}")
        print(f"Total records saved: {len(result_df)}")
        print(f"Columns: {', '.join(result_df.columns)}")
        print(f"RSI Range: {result_df['rsi'].min():.2f} to {result_df['rsi'].max():.2f}")
        print(f"High Range: {result_df['high'].min():.2f} to {result_df['high'].max():.2f}")
        
        print("\nAll rows (sorted by RSI):")
        print(result_df.to_string(index=False))
        
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
