# support_resistance.py
# Latest Support and Resistance Level Detection (One per Symbol)

import pandas as pd
import numpy as np
import os
from datetime import datetime

def process_support_resistance(input_file, output_file):
    """
    Process stock data and find LATEST SUPPORT or RESISTANCE level only
    Each symbol gets ONE entry - either support or resistance (whichever is latest)
    """

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Ensure required columns exist
    if 'close' not in df.columns:
        if 'open' in df.columns:
            df['close'] = df['open']
        else:
            df['close'] = (df['high'] + df['low']) / 2

    # Ensure date column is datetime type
    df['date'] = pd.to_datetime(df['date'])

    # Sort by symbol and date
    df = df.sort_values(['symbol', 'date'], ascending=[True, True])

    results = []

    #print("\n" + "=" * 80)
    #print("📊 LATEST SUPPORT & RESISTANCE DETECTION")
    #print("=" * 80)
    #print("(Each symbol will have ONE entry - either Support or Resistance)")
    #print("=" * 80)

    # Process each symbol separately
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date', ascending=True).reset_index(drop=True)

        if len(symbol_data) < 3:
            continue

        # Get the latest row
        latest_row = symbol_data.iloc[-1]
        current_low = latest_row['low']
        current_high = latest_row['high']
        current_date = latest_row['date']
        current_close = latest_row['close']

        #print(f"\n{'='*60}")
        #print(f"🔍 {symbol}")
        #print(f"{'='*60}")
        #print(f"Date: {current_date.strftime('%Y-%m-%d')}")
        #print(f"Low: ${current_low:.2f} | High: ${current_high:.2f} | Close: ${current_close:.2f}")

        # ==================== FIND LATEST SUPPORT ====================
        latest_support = None
        latest_support_date = None
        latest_support_gap = None
        latest_support_strength = None

        #print(f"\n🟢 Searching for SUPPORT...")

        for i in range(len(symbol_data) - 2, -1, -1):
            prev_row = symbol_data.iloc[i]
            prev_low = prev_row['low']
            prev_date = prev_row['date']

            # If previous low is below current low, stop (no more support possible)
            if prev_low < current_low:
                break

            # Check for support level (prev_low == current_low)
            if prev_low == current_low:
                rows_between = symbol_data.iloc[i+1:-1]
                gap_count = len(rows_between)

                valid = True
                for _, row in rows_between.iterrows():
                    if row['low'] < prev_low:
                        valid = False
                        break

                if valid and gap_count > 0:
                    strength = 'Strong' if gap_count > 10 else 'Moderate' if gap_count > 5 else 'Weak'
                    latest_support = prev_low
                    latest_support_date = prev_date
                    latest_support_gap = gap_count
                    latest_support_strength = strength
                    #print(f"  ✅ Found SUPPORT at ${prev_low:.2f} on {prev_date.strftime('%Y-%m-%d')} (Gap: {gap_count} days, {strength})")
                    break

        # ==================== FIND LATEST RESISTANCE ====================
        latest_resistance = None
        latest_resistance_date = None
        latest_resistance_gap = None
        latest_resistance_strength = None

        #print(f"\n🔴 Searching for RESISTANCE...")

        for i in range(len(symbol_data) - 2, -1, -1):
            prev_row = symbol_data.iloc[i]
            prev_high = prev_row['high']
            prev_date = prev_row['date']

            # If previous high is above current high, stop (no more resistance possible)
            if prev_high > current_high:
                break

            # Check for resistance level (prev_high == current_high)
            if prev_high == current_high:
                rows_between = symbol_data.iloc[i+1:-1]
                gap_count = len(rows_between)

                valid = True
                for _, row in rows_between.iterrows():
                    if row['high'] > prev_high:
                        valid = False
                        break

                if valid and gap_count > 0:
                    strength = 'Strong' if gap_count > 10 else 'Moderate' if gap_count > 5 else 'Weak'
                    latest_resistance = prev_high
                    latest_resistance_date = prev_date
                    latest_resistance_gap = gap_count
                    latest_resistance_strength = strength
                    #print(f"  ✅ Found RESISTANCE at ${prev_high:.2f} on {prev_date.strftime('%Y-%m-%d')} (Gap: {gap_count} days, {strength})")
                    break

        # ==================== DETERMINE WHICH IS LATEST ====================
        support_exists = latest_support is not None
        resistance_exists = latest_resistance is not None

        if support_exists and resistance_exists:
            # Compare dates to find which is more recent
            if latest_support_date > latest_resistance_date:
                # Support is more recent
                #print(f"\n📌 LATEST LEVEL: SUPPORT (More recent than Resistance)")
                results.append({
                    'type': 'support',
                    'symbol': symbol,
                    'current_date': current_date.strftime('%Y-%m-%d'),
                    'current_low': round(current_low, 2),
                    'current_high': round(current_high, 2),
                    'current_close': round(current_close, 2),
                    'level_date': latest_support_date.strftime('%Y-%m-%d'),
                    'level_price': round(latest_support, 2),
                    'gap_days': latest_support_gap,
                    'strength': latest_support_strength
                })
            else:
                # Resistance is more recent
                #(f"\n📌 LATEST LEVEL: RESISTANCE (More recent than Support)")
                results.append({
                    'type': 'resistance',
                    'symbol': symbol,
                    'current_date': current_date.strftime('%Y-%m-%d'),
                    'current_low': round(current_low, 2),
                    'current_high': round(current_high, 2),
                    'current_close': round(current_close, 2),
                    'level_date': latest_resistance_date.strftime('%Y-%m-%d'),
                    'level_price': round(latest_resistance, 2),
                    'gap_days': latest_resistance_gap,
                    'strength': latest_resistance_strength
                })

        elif support_exists:
            # Only support found
            #print(f"\n📌 LATEST LEVEL: SUPPORT (Only Support found)")
            results.append({
                'type': 'support',
                'symbol': symbol,
                'current_date': current_date.strftime('%Y-%m-%d'),
                'current_low': round(current_low, 2),
                'current_high': round(current_high, 2),
                'current_close': round(current_close, 2),
                'level_date': latest_support_date.strftime('%Y-%m-%d'),
                'level_price': round(latest_support, 2),
                'gap_days': latest_support_gap,
                'strength': latest_support_strength
            })

        elif resistance_exists:
            # Only resistance found
            #print(f"\n📌 LATEST LEVEL: RESISTANCE (Only Resistance found)")
            results.append({
                'type': 'resistance',
                'symbol': symbol,
                'current_date': current_date.strftime('%Y-%m-%d'),
                'current_low': round(current_low, 2),
                'current_high': round(current_high, 2),
                'current_close': round(current_close, 2),
                'level_date': latest_resistance_date.strftime('%Y-%m-%d'),
                'level_price': round(latest_resistance, 2),
                'gap_days': latest_resistance_gap,
                'strength': latest_resistance_strength
            })

        else:
            pass
            #print(f"\n⚠️ No Support or Resistance found for {symbol}")

    # ==================== CREATE OUTPUT DATAFRAME ====================
    if results:
        output_df = pd.DataFrame(results)

        # Reorder columns
        column_order = [
            'type', 'symbol', 'current_date', 'current_low', 'current_high', 'current_close',
            'level_date', 'level_price', 'gap_days', 'strength'
        ]
        output_df = output_df[column_order]

        # Sort by symbol
        output_df = output_df.sort_values('symbol').reset_index(drop=True)

        # Add serial number
        output_df.insert(0, 'no', range(1, len(output_df) + 1))

        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save to CSV
        output_df.to_csv(output_file, index=False)

        #print("\n" + "=" * 80)
        #print("✅ LATEST SUPPORT & RESISTANCE DETECTION COMPLETE")
        #print("=" * 80)
        #print(f"\n📊 TOTAL SYMBOLS WITH LEVELS: {len(output_df)}")
        #print(f"   Support: {len(output_df[output_df['type'] == 'support'])}")
        #print(f"   Resistance: {len(output_df[output_df['type'] == 'resistance'])}")

        #print("\n📈 DETAILED SUMMARY:")
        #print("=" * 80)
        #print(output_df[['no', 'type', 'symbol', 'current_close', 'level_price', 'gap_days', 'strength']].to_string())

        # Summary statistics
        #print("\n📊 STATISTICS:")
        #print("-" * 50)
        support_df = output_df[output_df['type'] == 'support']
        resistance_df = output_df[output_df['type'] == 'resistance']

        if len(support_df) > 0:
            pass
            #print(f"Support: {len(support_df)} symbols | Avg Gap: {support_df['gap_days'].mean():.1f} days")
        if len(resistance_df) > 0:
            pass
            #print(f"Resistance: {len(resistance_df)} symbols | Avg Gap: {resistance_df['gap_days'].mean():.1f} days")

    else:
        
        #print("\n❌ No support or resistance levels found")
        empty_df = pd.DataFrame(columns=[
            'no', 'type', 'symbol', 'current_date', 'current_low', 'current_high', 
            'current_close', 'level_date', 'level_price', 'gap_days', 'strength'
        ])
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        empty_df.to_csv(output_file, index=False)
        #print(f"📁 Created empty file: {output_file}")

    return results

def main():
    # Define file paths
    input_file = './csv/mongodb.csv'
    output_file_2 = './csv/support_resistance.csv'

    # Check if input file exists
    if not os.path.exists(input_file):
        #print(f"❌ Error: Input file {input_file} not found!")
        #print("Creating sample data...")

        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2024-01-15', freq='D')
        np.random.seed(42)

        symbols = ['MONGODB', 'BTC-USD', 'ETH-USD', 'SOL-USD']
        all_data = []

        for sym in symbols:
            close = 100 + np.random.randn() * 50
            closes = []
            for i in range(len(dates)):
                close = close + np.random.randn() * 2
                closes.append(close)

            df_sym = pd.DataFrame({
                'date': dates,
                'open': [c - np.random.rand() * 2 for c in closes],
                'high': [c + abs(np.random.randn()) * 2 for c in closes],
                'low': [c - abs(np.random.randn()) * 2 for c in closes],
                'close': closes,
                'volume': np.random.randint(100000, 1000000, len(dates)),
                'symbol': sym
            })
            all_data.append(df_sym)

        df = pd.concat(all_data, ignore_index=True)
        df.to_csv(input_file, index=False)
        #print(f"✅ Sample data created at: {input_file}")

    # Process for both output locations
    process_support_resistance(input_file, output_file_2)

    #print("\n" + "=" * 80)
    #print("✨ PROCESSING COMPLETE!")

    #print(f"📁 Output 2: {output_file_2}")
    #print("=" * 80)

if __name__ == "__main__":
    main()
