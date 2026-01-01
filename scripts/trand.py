import pandas as pd
import os
from datetime import datetime

from hf_uploader import download_from_hf_or_run_script

# Step 1: Download CSV from HF if needed
if download_from_hf_or_run_script():
    print(f"HF data download success")
# সংক্ষিপ্ত নাম সংরক্ষণ (internal use only)

variable_definitions = {
    'high_one_candle': 'hoc',
    'high_one_candle_high': 'hoch',
    'high_one_candle_low': 'hocl',
    'high_before_one_candle': 'hboc',
    'high_before_one_candle_high': 'hboch',
    'high_before_one_candle_low': 'hbocl',
    'high_before_two_candle': 'hbtc',
    'high_before_two_candle_high': 'hbtch',
    'high_before_two_candle_low': 'hbtcl',
    'high_after_one_candle': 'haoc',
    'high_after_one_candle_high': 'haoch',
    'high_after_one_candle_low': 'haocl',
    'high_after_two_candle': 'hatc',
    'high_after_two_candle_high': 'hatch',
    'high_after_two_candle_low': 'hatcl',
    
    'low_one_candle': 'loc',
    'low_one_candle_high': 'loch',
    'low_one_candle_low': 'locl',
    'low_before_one_candle': 'lboc',
    'low_before_one_candle_high': 'lboch',
    'low_before_one_candle_low': 'lbocl',
    'low_before_two_candle': 'lbtc',
    'low_before_two_candle_high': 'lbtch',
    'low_before_two_candle_low': 'lbtcl',
    'low_after_one_candle': 'laoc',
    'low_after_one_candle_high': 'laoch',
    'low_after_one_candle_low': 'laocl',
    'low_after_two_candle': 'latc',
    'low_after_two_candle_high': 'latch',
    'low_after_two_candle_low': 'latcl'
}

def check_high_swing(symbol_df, idx):
    """
    Check if index idx is a high swing point
    """
    try:
        n = len(symbol_df)
        
        if idx < 2 or idx >= n - 2:
            return False, False
        
        # Get current candle (hoc) - 5th from last
        hoch = symbol_df.iloc[idx]['high']
        hocl = symbol_df.iloc[idx]['low']
        
        # Get BEFORE candles
        hboch = symbol_df.iloc[idx + 1]['high']
        hbocl = symbol_df.iloc[idx + 1]['low']
        
        hbtch = symbol_df.iloc[idx + 2]['high']
        hbtcl = symbol_df.iloc[idx + 2]['low']
        
        # Get AFTER candles
        haoch = symbol_df.iloc[idx - 1]['high']
        haocl = symbol_df.iloc[idx - 1]['low']
        
        hatch = symbol_df.iloc[idx - 2]['high']
        hatcl = symbol_df.iloc[idx - 2]['low']
        
        # Skip condition: hoch == hboch and hocl <= hbocl
        if hoch == hboch and hocl <= hbocl:
            return False, True
        
        # Pass condition 1
        if hoch > hboch and hbocl < hocl and hbtch < hoch and hoch == haoch and hocl <= haocl:
            return False, True
        
        # Pass condition 2
        if hoch > haoch and haocl < hocl and hatch < hoch:
            return False, True
        
        # If none of the above, check if it's a valid high swing
        if hoch > hboch and hoch > haoch:
            return True, False
        
        return False, False
        
    except Exception as e:
        return False, False

def check_low_swing(symbol_df, idx):
    """
    Check if index idx is a low swing point
    """
    try:
        n = len(symbol_df)
        
        if idx < 2 or idx >= n - 2:
            return False, False
        
        # Get current candle (loc) - 5th from last
        loch = symbol_df.iloc[idx]['high']
        locl = symbol_df.iloc[idx]['low']
        
        # Get BEFORE candles
        lboch = symbol_df.iloc[idx + 1]['high']
        lbocl = symbol_df.iloc[idx + 1]['low']
        
        lbtch = symbol_df.iloc[idx + 2]['high']
        lbtcl = symbol_df.iloc[idx + 2]['low']
        
        # Get AFTER candles
        laoch = symbol_df.iloc[idx - 1]['high']
        laocl = symbol_df.iloc[idx - 1]['low']
        
        latch = symbol_df.iloc[idx - 2]['high']
        latcl = symbol_df.iloc[idx - 2]['low']
        
        # Skip condition: locl == lbocl and loch >= lboch
        if locl == lbocl and loch >= lboch:
            return False, True
        
        # Pass condition 1
        if locl < lbocl and lboch > loch and lbtch > locl and locl == laocl and loch >= laoch:
            return False, True
        
        # Pass condition 2
        if locl < laocl and laoch > loch and latcl > locl:
            return False, True
        
        # If none of the above, check if it's a valid low swing
        if locl < lbocl and locl < laocl:
            return True, False
        
        return False, False
        
    except Exception as e:
        return False, False

def process_symbol(symbol, symbol_df):
    """
    Process a single symbol to find swing points
    Returns lists of dates and prices for high and low swings
    """
    # Sort by date descending
    symbol_df = symbol_df.sort_values('date', ascending=False).reset_index(drop=True)
    
    high_dates = []
    high_prices = []
    low_dates = []
    low_prices = []
    
    n = len(symbol_df)
    
    if n < 5:
        return high_dates, high_prices, low_dates, low_prices
    
    idx = 2  # Start from 5th row from last
    
    # Continue until we find 2 highs and 2 lows OR reach end of data
    while idx < n - 2 and (len(high_dates) < 2 or len(low_dates) < 2):
        
        # Check high swing (only if we need more high swings)
        if len(high_dates) < 2:
            is_high, should_skip = check_high_swing(symbol_df, idx)
            if is_high:
                date_val = symbol_df.iloc[idx]['date']
                high_val = symbol_df.iloc[idx]['high']
                high_dates.append(date_val)
                high_prices.append(high_val)
            
            if should_skip:
                idx += 1
                continue
        
        # Check low swing (only if we need more low swings)
        if len(low_dates) < 2:
            is_low, should_skip = check_low_swing(symbol_df, idx)
            if is_low:
                date_val = symbol_df.iloc[idx]['date']
                low_val = symbol_df.iloc[idx]['low']
                low_dates.append(date_val)
                low_prices.append(low_val)
            
            if should_skip:
                idx += 1
                continue
        
        # Move to next candle
        idx += 1
        
        # Early exit if we have enough data
        if len(high_dates) >= 2 and len(low_dates) >= 2:
            break
    
    return high_dates[:2], high_prices[:2], low_dates[:2], low_prices[:2]

def save_to_csv(symbol, high_dates, high_prices, low_dates, low_prices, output_base_dir):
    """
    Save swing points to separate CSV files for high and low swings
    """
    # Create symbol directory
    symbol_dir = os.path.join(output_base_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    
    saved_files = []
    
    # Save high swings to high.csv
    if high_dates and high_prices:
        high_data = []
        for date, price in zip(high_dates, high_prices):
            high_data.append({
                'date': date,
                'price': price
            })
        
        if high_data:
            high_df = pd.DataFrame(high_data)
            high_df = high_df.sort_values('date', ascending=False).reset_index(drop=True)
            
            high_file = os.path.join(symbol_dir, 'high.csv')
            high_df.to_csv(high_file, index=False)
            saved_files.append(('high', high_file))
    
    # Save low swings to low.csv
    if low_dates and low_prices:
        low_data = []
        for date, price in zip(low_dates, low_prices):
            low_data.append({
                'date': date,
                'price': price
            })
        
        if low_data:
            low_df = pd.DataFrame(low_data)
            low_df = low_df.sort_values('date', ascending=False).reset_index(drop=True)
            
            low_file = os.path.join(symbol_dir, 'low.csv')
            low_df.to_csv(low_file, index=False)
            saved_files.append(('low', low_file))
    
    return saved_files

def main():
    # CSV file path
    csv_file_path = './csv/mongodb.csv'
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: File not found: {csv_file_path}")
        return
    
    # Output base directory
    output_base_dir = './csv/trand/'
    os.makedirs(output_base_dir, exist_ok=True)
    
    try:
        # Read CSV file
        print(f"Reading CSV file: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        # Check required columns
        required_columns = ['symbol', 'date', 'high', 'low']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Get unique symbols
        symbols = df['symbol'].unique()
        print(f"Found {len(symbols)} symbols to process")
        
        # Process each symbol
        processed_count = 0
        
        for symbol in symbols:
            # Filter data for current symbol
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Process the symbol
            high_dates, high_prices, low_dates, low_prices = process_symbol(symbol, symbol_data)
            
            # Save to CSV files
            saved_files = save_to_csv(symbol, high_dates, high_prices, low_dates, low_prices, output_base_dir)
            
            if saved_files:
                processed_count += 1
                print(f"\n✓ {symbol}:")
                for swing_type, file_path in saved_files:
                    if swing_type == 'high':
                        print(f"  High swings: {len(high_dates)} saved to {file_path}")
                    else:
                        print(f"  Low swings: {len(low_dates)} saved to {file_path}")
            else:
                print(f"\n✗ {symbol}: No swings found")
        
        print(f"\n{'='*60}")
        print(f"Processing completed! {processed_count} symbols processed")
        print(f"Output directory structure: {output_base_dir}")
        
        # Show directory structure
        if processed_count > 0:
            print(f"\nDirectory structure created:")
            for root, dirs, files in os.walk(output_base_dir):
                level = root.replace(output_base_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    if file.endswith('.csv'):
                        print(f"{subindent}{file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
