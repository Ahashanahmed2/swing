# fail_short_buy_pass.py
# Check if failed breakout recovers (price breaks above previous high with higher low)

import pandas as pd
import os
from datetime import datetime

# =========================
# Configuration
# =========================
FAIL_SHORT_BUY_FILE = './csv/fail_short_buy.csv'
MONGO_FILE = './csv/mongodb.csv'
PASS_FILE = './output/ai_signal/fail_short_buy_pass.csv'

# =========================
# Load Data
# =========================
print("=" * 80)
print("📊 FAIL SHORT BUY PASS DETECTOR")
print("=" * 80)

# Check if files exist
if not os.path.exists(FAIL_SHORT_BUY_FILE):
    print(f"❌ Error: {FAIL_SHORT_BUY_FILE} not found!")
    print("Please run fail_short_buy.py first")
    exit(1)

if not os.path.exists(MONGO_FILE):
    print(f"❌ Error: {MONGO_FILE} not found!")
    exit(1)

# Read data
fail_df = pd.read_csv(FAIL_SHORT_BUY_FILE)
mongo_df = pd.read_csv(MONGO_FILE)

print(f"\n📂 Fail Short Buy file: {len(fail_df)} records")
print(f"📂 MongoDB file: {len(mongo_df)} records")

# =========================
# Prepare MongoDB Data
# =========================
# Convert dates
mongo_df['date'] = pd.to_datetime(mongo_df['date'])
fail_df['date'] = pd.to_datetime(fail_df['date'])

# Ensure all required columns exist in MongoDB
required_mongo_cols = ['symbol', 'date', 'high', 'low']
for col in required_mongo_cols:
    if col not in mongo_df.columns:
        print(f"❌ Error: Missing required column '{col}' in MongoDB!")
        exit(1)

# Sort MongoDB by date
mongo_df = mongo_df.sort_values(['symbol', 'date']).reset_index(drop=True)

# =========================
# Check Passed (Recovered) Signals
# =========================
print("\n" + "=" * 80)
print("🔍 CHECKING RECOVERED BREAKOUTS")
print("=" * 80)

passed_signals = []
updated_fails = []

for _, fail_row in fail_df.iterrows():
    symbol = fail_row['symbol']
    fail_date = fail_row['date']
    fail_high = fail_row['high']
    fail_low = fail_row['low']
    
    try:
        # Get data for this symbol after the fail date
        symbol_data = mongo_df[(mongo_df['symbol'] == symbol) & (mongo_df['date'] > fail_date)]
        symbol_data = symbol_data.sort_values('date')
        
        # Skip if no data after fail date
        if len(symbol_data) == 0:
            # Keep the existing fail data as is
            updated_fails.append({
                'no': fail_row['no'],
                'date': fail_date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'high': fail_high,
                'low': fail_low
            })
            continue
        
        # Check if ANY subsequent row has:
        # 1. high > fail_high (price breaks above previous high)
        # 2. low > fail_low (new low is higher than previous low)
        passed = False
        for _, subsequent_row in symbol_data.iterrows():
            if (subsequent_row['close'] > fail_high) and (subsequent_row['low'] > fail_low):
                passed = True
                
                # Add to passed signals
                passed_signals.append({
                    'symbol': symbol,
                    #'fail_date': fail_date.strftime('%Y-%m-%d'),
                    #'fail_high': fail_high,
                    #'fail_low': fail_low,
                    'date': subsequent_row['date'].strftime('%Y-%m-%d'),
                    'high': subsequent_row['high'],
                    'low': subsequent_row['low']
                })
                
                print(f"✅ {symbol:<12} | Passed! High: ${subsequent_row['high']:.2f} > ${fail_high:.2f} | Low: ${subsequent_row['low']:.2f} > ${fail_low:.2f}")
                break
        
        # If not passed, update with latest MongoDB data
        if not passed:
            # Get the latest row from MongoDB
            latest_row = symbol_data.iloc[-1]
            
            updated_fails.append({
                'no': fail_row['no'],
                'date': latest_row['date'].strftime('%Y-%m-%d'),
                'symbol': symbol,
                'high': latest_row['high'],
                'low': latest_row['low']
            })
            
            print(f"❌ {symbol:<12} | Updated - Date: {latest_row['date'].strftime('%Y-%m-%d')} | High: ${latest_row['high']:.2f} | Low: ${latest_row['low']:.2f}")
    
    except Exception as e:
        print(f"⚠️ Error processing {symbol}: {e}")
        # Keep original fail data on error
        updated_fails.append({
            'no': fail_row['no'],
            'date': fail_date.strftime('%Y-%m-%d'),
            'symbol': symbol,
            'high': fail_high,
            'low': fail_low
        })
        continue

# =========================
# Save Passed Signals
# =========================
print("\n" + "=" * 80)
print("💾 SAVING RESULTS")
print("=" * 80)

# Create output directory
os.makedirs(os.path.dirname(PASS_FILE), exist_ok=True)

if passed_signals:
    pass_df = pd.DataFrame(passed_signals)
    
    # Add serial number
    pass_df.insert(0, 'no', range(1, len(pass_df) + 1))
    
    # Save passed signals
    pass_df.to_csv(PASS_FILE, index=False)
    
    print(f"\n✅ Passed signals saved to: {PASS_FILE}")
    print(f"📋 Columns: {list(pass_df.columns)}")
    print(f"📊 Total passed signals: {len(pass_df)}")
    
    # Display passed signals
    print("\n" + "=" * 80)
    print("📋 PASSED (RECOVERED) SIGNALS")
    print("=" * 80)
    print(pass_df.to_string(index=False))
    
else:
    print("\n✅ No passed signals found!")
    
    # Create empty file with headers
    empty_pass = pd.DataFrame(columns=['no', 'symbol', 'fail_date', 'fail_high', 'fail_low', 'pass_date', 'pass_high', 'pass_low'])
    empty_pass.to_csv(PASS_FILE, index=False)
    print(f"📄 Empty file created: {PASS_FILE}")

# =========================
# Update fail_short_buy.csv (remove passed symbols, update others)
# =========================
print("\n" + "=" * 80)
print("🔄 UPDATING FAIL SHORT BUY FILE")
print("=" * 80)

if updated_fails:
    updated_fail_df = pd.DataFrame(updated_fails)
    
    # Re-number the serial numbers
    updated_fail_df = updated_fail_df.sort_values('symbol').reset_index(drop=True)
    updated_fail_df['no'] = range(1, len(updated_fail_df) + 1)
    
    # Reorder columns
    updated_fail_df = updated_fail_df[['no', 'date', 'symbol', 'high', 'low']]
    
    # Save updated fail file
    updated_fail_df.to_csv(FAIL_SHORT_BUY_FILE, index=False)
    
    print(f"✅ Updated fail file: {FAIL_SHORT_BUY_FILE}")
    print(f"📊 Remaining failed signals: {len(updated_fail_df)}")
    print(f"📊 Passed (removed): {len(passed_signals)}")
    
    # Display updated fails
    print("\n📋 UPDATED FAILED SIGNALS:")
    print(updated_fail_df.to_string(index=False))
    
else:
    # No fails remaining, create empty file
    empty_fail = pd.DataFrame(columns=['no', 'date', 'symbol', 'high', 'low'])
    empty_fail.to_csv(FAIL_SHORT_BUY_FILE, index=False)
    
    print(f"✅ All failed signals resolved!")
    print(f"📄 Empty file created: {FAIL_SHORT_BUY_FILE}")

# =========================
# Summary
# =========================
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)
original_count = len(fail_df)
passed_count = len(passed_signals)
remaining_count = len(updated_fails)

print(f"Original failed signals: {original_count}")
print(f"Passed (recovered): {passed_count}")
print(f"Still failed: {remaining_count}")
print(f"Success rate: {(passed_count/original_count*100):.1f}%" if original_count > 0 else "N/A")

print("\n" + "=" * 80)
print("✅ DONE!")
print("=" * 80)
