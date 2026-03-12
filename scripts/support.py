import pandas as pd
import numpy as np
import os

def find_support_levels(input_file, output_file, touch_tolerance=0.02):
    """
    Improved support level detection
    
    Args:
        input_file: input CSV path
        output_file: output CSV path
        touch_tolerance: price tolerance for support touches (2% default)
    """
    
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])
    
    results = []
    
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].copy()
        
        if len(symbol_data) < 3:
            continue
        
        # Latest price
        latest = symbol_data.iloc[-1]
        current_low = latest['low']
        current_date = latest['date']
        
        # Find potential support levels
        for i in range(len(symbol_data) - 2, -1, -1):
            potential_support = symbol_data.iloc[i]
            support_low = potential_support['low']
            support_high = potential_support['high']
            support_date = potential_support['date']
            
            # Check if current low is near the support low (within tolerance)
            price_diff = abs(current_low - support_low) / support_low
            if price_diff > touch_tolerance:
                continue
            
            # Check if price is within support range
            if not (support_low <= current_low <= support_high):
                continue
            
            # Check intermediate bars
            intermediate = symbol_data.iloc[i+1:-1]
            valid = True
            touch_count = 1  # Current touch
            
            # Count how many times price touched this support zone
            for _, row in intermediate.iterrows():
                # If price came near support zone
                if abs(row['low'] - support_low) / support_low <= touch_tolerance:
                    touch_count += 1
                
                # If price broke below support significantly
                if row['low'] < support_low * (1 - touch_tolerance):
                    valid = False
                    break
            
            if valid and touch_count >= 2:  # At least 2 touches including current
                gap = len(intermediate)
                
                # Calculate support strength
                strength = "Strong" if touch_count >= 3 else "Moderate"
                
                results.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'support_date': support_date.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'close': latest['close'],
                    'gap': gap,
                    'touch_count': touch_count,
                    'strength': strength,
                    'support_level': round(support_low, 2)
                })
    
    # Save results
    if results:
        output_df = pd.DataFrame(results)
        output_df = output_df[['date', 'support_date', 'symbol', 'close', 'gap', 'touch_count', 'strength', 'support_level']]
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output_df.to_csv(output_file, index=False)
        
        print(f"Found {len(output_df)} support levels")
        print("\nSample with strength:")
        print(output_df.head())
    else:
        # Create empty file with headers
        empty_df = pd.DataFrame(columns=['date', 'support_date', 'symbol', 'close', 'gap', 'touch_count', 'strength', 'support_level'])
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        empty_df.to_csv(output_file, index=False)
        print("No support levels found")

if __name__ == "__main__":
    find_support_levels('./csv/mongodb.csv', './output/ai_signal/support.csv')