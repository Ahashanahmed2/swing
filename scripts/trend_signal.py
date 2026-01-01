import pandas as pd
import os
from datetime import datetime

def create_uptrend_downtrend_signals():
    """
    Create uptrend.csv and downtrend.csv based on price comparison
    """
    
    # Input file paths
    mongodb_csv = './csv/mongodb.csv'
    trand_base_dir = './csv/trand/'
    output_dir = './csv/'
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file paths
    uptrend_file = os.path.join(output_dir, 'uptrend.csv')
    downtrend_file = os.path.join(output_dir, 'downtrend.csv')
    
    # Read mongodb.csv
    print(f"Reading {mongodb_csv}...")
    try:
        mongodb_df = pd.read_csv(mongodb_csv)
    except FileNotFoundError:
        print(f"Error: {mongodb_csv} not found!")
        return
    
    # Check required columns
    required_columns = ['symbol', 'date', 'close']
    for col in required_columns:
        if col not in mongodb_df.columns:
            print(f"Error: Required column '{col}' not found in {mongodb_csv}")
            return
    
    # Convert date to datetime
    mongodb_df['date'] = pd.to_datetime(mongodb_df['date'])
    
    # Get latest close for each symbol
    print("Getting latest close prices for each symbol...")
    latest_data = {}
    
    for symbol in mongodb_df['symbol'].unique():
        symbol_data = mongodb_df[mongodb_df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date', ascending=False).reset_index(drop=True)
        
        if not symbol_data.empty:
            latest_close = symbol_data.iloc[0]['close']
            latest_date = symbol_data.iloc[0]['date']
            latest_data[symbol] = {
                'close': latest_close,
                'date': latest_date
            }
    
    print(f"Found {len(latest_data)} symbols with latest data")
    
    # Initialize lists for uptrend and downtrend signals
    uptrend_signals = []
    downtrend_signals = []
    
    # Process each symbol
    print("\nProcessing symbols for trend signals...")
    
    for symbol, latest_info in latest_data.items():
        symbol_dir = os.path.join(trand_base_dir, symbol)
        
        # Check if high.csv exists
        high_file = os.path.join(symbol_dir, 'high.csv')
        low_file = os.path.join(symbol_dir, 'low.csv')
        
        has_high = os.path.exists(high_file)
        has_low = os.path.exists(low_file)
        
        if not (has_high or has_low):
            continue  # Skip symbols without swing data
        
        latest_close = latest_info['close']
        latest_date = latest_info['date']
        
        # Check UPTREND condition from high.csv
        if has_high:
            try:
                high_df = pd.read_csv(high_file)
                
                # Check if high.csv has at least 2 rows
                if len(high_df) >= 2:
                    # Get the two most recent high swing prices (assuming sorted newest first)
                    row_1_price = high_df.iloc[0]['price'] if 'price' in high_df.columns else high_df.iloc[0]['price']
                    row_2_price = high_df.iloc[1]['price'] if 'price' in high_df.columns else high_df.iloc[1]['price']
                    
                    # UPTREND condition: row_1_price < close > row_2_price and row_1_price < row_2_price
                    # Note: Fixed condition based on your input
                    condition1 = row_1_price < latest_close > row_2_price
                    condition2 = row_1_price < row_2_price
                    
                    if condition1 and condition2:
                        uptrend_signals.append({
                            'no': len(uptrend_signals) + 1,
                            'date': latest_date,
                            'symbol': symbol,
                            'close': latest_close,
                            'row_1_price': row_1_price,
                            'row_2_price': row_2_price,
                            'condition': f"{row_1_price} < {latest_close} > {row_2_price} and {row_1_price} < {row_2_price}"
                        })
                        print(f"✓ UPTREND signal for {symbol}")
                        
            except Exception as e:
                print(f"  Error reading high.csv for {symbol}: {e}")
        
        # Check DOWNTREND condition from low.csv
        if has_low:
            try:
                low_df = pd.read_csv(low_file)
                
                # Check if low.csv has at least 2 rows
                if len(low_df) >= 2:
                    # Get the two most recent low swing prices (assuming sorted newest first)
                    row_1_price = low_df.iloc[0]['price'] if 'price' in low_df.columns else low_df.iloc[0]['price']
                    row_2_price = low_df.iloc[1]['price'] if 'price' in low_df.columns else low_df.iloc[1]['price']
                    
                    # DOWNTREND condition: row_1_price > close < row_2_price and row_1_price > row_2_price
                    condition1 = row_1_price > latest_close < row_2_price
                    condition2 = row_1_price > row_2_price
                    
                    if condition1 and condition2:
                        downtrend_signals.append({
                            'no': len(downtrend_signals) + 1,
                            'date': latest_date,
                            'symbol': symbol,
                            'close': latest_close,
                            'row_1_price': row_1_price,
                            'row_2_price': row_2_price,
                            'condition': f"{row_1_price} > {latest_close} < {row_2_price} and {row_1_price} > {row_2_price}"
                        })
                        print(f"✓ DOWNTREND signal for {symbol}")
                        
            except Exception as e:
                print(f"  Error reading low.csv for {symbol}: {e}")
    
    # Save uptrend signals to CSV
    if uptrend_signals:
        uptrend_df = pd.DataFrame(uptrend_signals)
        uptrend_df.to_csv(uptrend_file, index=False)
        print(f"\n✅ Saved {len(uptrend_signals)} uptrend signals to {uptrend_file}")
        
        # Display sample
        print("\nSample uptrend signals:")
        print(uptrend_df[['no', 'date', 'symbol', 'close']].head().to_string(index=False))
    else:
        print(f"\n❌ No uptrend signals found")
    
    # Save downtrend signals to CSV
    if downtrend_signals:
        downtrend_df = pd.DataFrame(downtrend_signals)
        downtrend_df.to_csv(downtrend_file, index=False)
        print(f"\n✅ Saved {len(downtrend_signals)} downtrend signals to {downtrend_file}")
        
        # Display sample
        print("\nSample downtrend signals:")
        print(downtrend_df[['no', 'date', 'symbol', 'close']].head().to_string(index=False))
    else:
        print(f"\n❌ No downtrend signals found")
    
    # Create summary report
    summary_file = os.path.join(output_dir, 'trend_signals_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("TREND SIGNALS SUMMARY REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total symbols processed: {len(latest_data)}\n")
        f.write(f"Uptrend signals found: {len(uptrend_signals)}\n")
        f.write(f"Downtrend signals found: {len(downtrend_signals)}\n\n")
        
        if uptrend_signals:
            f.write("UPTREND SIGNALS:\n")
            f.write("-" * 30 + "\n")
            for signal in uptrend_signals:
                f.write(f"{signal['no']}. {signal['symbol']} - Close: {signal['close']} (Date: {signal['date']})\n")
        
        if downtrend_signals:
            f.write("\nDOWNTREND SIGNALS:\n")
            f.write("-" * 30 + "\n")
            for signal in downtrend_signals:
                f.write(f"{signal['no']}. {signal['symbol']} - Close: {signal['close']} (Date: {signal['date']})\n")
    
    print(f"\n📊 Summary saved to: {summary_file}")
    print("\n🎯 Trend signal detection completed!")

def main():
    """
    Main function to execute trend signal detection
    """
    print("=" * 60)
    print("TREND SIGNAL DETECTION SCRIPT")
    print("=" * 60)
    
    create_uptrend_downtrend_signals()

if __name__ == "__main__":
    main()