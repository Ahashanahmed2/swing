# fail_short_buy.py
# Check if breakout failed (price didn't close above breakout high)

import pandas as pd
import os
from datetime import datetime

# =========================
# Configuration
# =========================
SHORT_BUY_CHECK_FILE = './csv/short_buy_check.csv'
MONGO_FILE = './csv/mongodb.csv'
FAIL_SHORT_BUY_FILE = './csv/fail_short_buy.csv'

# =========================
# Load Data
# =========================
print("=" * 80)
print("📊 FAIL SHORT BUY DETECTOR")
print("=" * 80)

# Check if files exist
if not os.path.exists(SHORT_BUY_CHECK_FILE):
    print(f"❌ Error: {SHORT_BUY_CHECK_FILE} not found!")
    print("Please run short_buy_check.py first")
    exit(1)

if not os.path.exists(MONGO_FILE):
    print(f"❌ Error: {MONGO_FILE} not found!")
    exit(1)

# Read data
check_df = pd.read_csv(SHORT_BUY_CHECK_FILE)
mongo_df = pd.read_csv(MONGO_FILE)

print(f"\n📂 Short Buy Check file: {len(check_df)} records")
print(f"📂 MongoDB file: {len(mongo_df)} records")

# =========================
# Prepare MongoDB Data
# =========================
# Convert dates
mongo_df['date'] = pd.to_datetime(mongo_df['date'])
check_df['date'] = pd.to_datetime(check_df['date'])

# Ensure all required columns exist in MongoDB
required_mongo_cols = ['symbol', 'date', 'high', 'low', 'close']
for col in required_mongo_cols:
    if col not in mongo_df.columns:
        if col == 'close':
            if 'open' in mongo_df.columns:
                mongo_df['close'] = mongo_df['open']
            else:
                mongo_df['close'] = (mongo_df['high'] + mongo_df['low']) / 2
        else:
            print(f"❌ Error: Missing required column '{col}' in MongoDB!")
            exit(1)

# Sort MongoDB by date
mongo_df = mongo_df.sort_values(['symbol', 'date']).reset_index(drop=True)

# =========================
# Check Failed Breakouts
# =========================
print("\n" + "=" * 80)
print("🔍 CHECKING FAILED BREAKOUTS")
print("=" * 80)

failed_signals = []

for _, check_row in check_df.iterrows():
    symbol = check_row['symbol']
    check_date = check_row['date']
    
    try:
        # Get the specific row from MongoDB for this symbol and date
        mongo_match = mongo_df[(mongo_df['symbol'] == symbol) & (mongo_df['date'] == check_date)]
        
        if len(mongo_match) == 0:
            print(f"⚠️ {symbol:<12} | Date: {check_date.strftime('%Y-%m-%d')} - Not found in MongoDB")
            continue
        
        # Get data from MongoDB
        mongo_row = mongo_match.iloc[0]
        check_high = mongo_row['high']  # Now from MongoDB
        check_low = mongo_row['low']    # Now from MongoDB
        
        # Get data for this symbol after the check date
        symbol_data = mongo_df[(mongo_df['symbol'] == symbol) & (mongo_df['date'] > check_date)]
        symbol_data = symbol_data.sort_values('date')
        
        # Skip if no data after check date
        if len(symbol_data) == 0:
            continue
        
        # Check if ANY subsequent row closed above the breakout high
        failed = True
        last_date = None
        for _, subsequent_row in symbol_data.iterrows():
            last_date = subsequent_row['date']
            if subsequent_row['close'] > check_high:
                failed = False
                break
        
        # If failed (no close above breakout high)
        if failed:
            # Get the last available date from MongoDB for this symbol
            last_mongo_date = last_date if last_date else check_date
            
            failed_signals.append({
                'date': last_mongo_date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'high': check_high,    # From MongoDB
                'low': check_low       # From MongoDB
            })
            
            print(f"❌ {symbol:<12} | Date: {check_date.strftime('%Y-%m-%d')} | High: ${check_high:.2f} | Low: ${check_low:.2f} - No close above high")
    
    except Exception as e:
        print(f"⚠️ Error processing {symbol}: {e}")
        continue

# =========================
# Save Failed Signals
# =========================
if failed_signals:
    fail_df = pd.DataFrame(failed_signals)
    
    # Add serial number (no) at first column
    fail_df.insert(0, 'no', range(1, len(fail_df) + 1))
    
    # Reorder columns: no, date, symbol, high, low
    fail_df = fail_df[['no', 'date', 'symbol', 'high', 'low']]
    
    # Save to file
    fail_df.to_csv(FAIL_SHORT_BUY_FILE, index=False)
    
    print(f"\n✅ Failed breakouts saved to: {FAIL_SHORT_BUY_FILE}")
    print(f"📋 Columns: {list(fail_df.columns)}")
    print(f"📊 Total failed signals: {len(fail_df)}")
    
    # Display failed signals
    print("\n" + "=" * 80)
    print("📋 FAILED BREAKOUT SIGNALS")
    print("=" * 80)
    print(fail_df.to_string(index=False))
    
else:
    print("\n✅ No failed breakouts found!")
    
    # Create empty file with headers
    empty_fail = pd.DataFrame(columns=['no', 'date', 'symbol', 'high', 'low'])
    empty_fail.to_csv(FAIL_SHORT_BUY_FILE, index=False)
    print(f"📄 Empty file created: {FAIL_SHORT_BUY_FILE}")

# =========================
# Delete all rows from short_buy_check.csv
# =========================
print("\n" + "=" * 80)
print("🗑️ CLEANING UP")
print("=" * 80)

# Create empty DataFrame with same columns
empty_check = pd.DataFrame(columns=['no', 'symbol', 'date', 'high', 'low'])
empty_check.to_csv(SHORT_BUY_CHECK_FILE, index=False)

print(f"✅ All rows deleted from: {SHORT_BUY_CHECK_FILE}")
print(f"📄 Empty file with headers only")

# =========================
# Summary
# =========================
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)
print(f"Total records checked: {len(check_df)}")
print(f"Failed breakouts found: {len(failed_signals)}")
print(f"Success breakouts (not failed): {len(check_df) - len(failed_signals)}")

print("\n" + "=" * 80)
print("✅ DONE!")
print("=" * 80)
