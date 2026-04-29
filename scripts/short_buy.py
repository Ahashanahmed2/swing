# short_buy.py
# Breakout Signal Generator based on RSI Divergence

import pandas as pd
import numpy as np
import os
from datetime import datetime

# =========================
# Configuration
# =========================
RSI_DIVER_FILE = './csv/rsi_diver.csv'
MONGO_FILE = './csv/mongodb.csv'
SHORT_BUY_FILE = './csv/short_buy.csv'

# =========================
# Load Data
# =========================
print("=" * 80)
print("📊 BREAKOUT SIGNAL GENERATOR (Based on RSI Divergence)")
print("=" * 80)

# Check if files exist
if not os.path.exists(RSI_DIVER_FILE):
    print(f"❌ Error: {RSI_DIVER_FILE} not found!")
    print("Please run rsi_diver.py first")
    exit(1)

if not os.path.exists(MONGO_FILE):
    print(f"❌ Error: {MONGO_FILE} not found!")
    exit(1)

# Read data
rsi_df = pd.read_csv(RSI_DIVER_FILE)
mongo_df = pd.read_csv(MONGO_FILE)

print(f"\n📂 RSI Divergence file: {len(rsi_df)} records")
print(f"📂 MongoDB file: {len(mongo_df)} records")

# =========================
# Prepare Data
# =========================
# Convert dates
mongo_df['date'] = pd.to_datetime(mongo_df['date'])

# Ensure required columns exist
if 'close' not in mongo_df.columns:
    if 'open' in mongo_df.columns:
        mongo_df['close'] = mongo_df['open']
    else:
        mongo_df['close'] = (mongo_df['high'] + mongo_df['low']) / 2

# Sort by date
mongo_df = mongo_df.sort_values(['symbol', 'date']).reset_index(drop=True)

# =========================
# Process Bullish Divergence (Long/Buy Signals)
# =========================
print("\n" + "=" * 80)
print("🔍 BULLISH DIVERGENCE - BUY SIGNALS")
print("=" * 80)

bullish_signals = []
bullish_df = rsi_df[rsi_df['divergence_type'] == 'Bullish']

for _, row in bullish_df.iterrows():
    symbol = row['symbol']
    divergence_date = pd.to_datetime(row['last_date'])
    divergence_low = row['last_price']

    # ✅ BUG FIX: Safe data fetching with proper checks
    try:
        # Get data for this symbol
        symbol_data = mongo_df[mongo_df['symbol'] == symbol]
        
        # Skip if no data in MongoDB
        if len(symbol_data) == 0:
            continue
        
        # Get data after divergence date
        symbol_data = symbol_data[symbol_data['date'] > divergence_date]
        symbol_data = symbol_data.sort_values('date')
        
        # Skip if no data after divergence date (trading halted/suspended)
        if len(symbol_data) == 0:
            continue
        
        # Get the first candle after divergence
        first_candle = symbol_data.iloc[0]
        first_low = first_candle['low']

        # Check breakout: First candle's low > divergence low (Breakout)
        if first_low > divergence_low:
            bullish_signals.append({
                'symbol': symbol,
                'signal_type': 'BUY (Bullish Breakout)',
                'divergence_date': divergence_date.strftime('%Y-%m-%d'),
                'divergence_low': divergence_low,
                'breakout_low': first_low,
                'strength': row['strength']
            })

            print(f"✅ {symbol:<12} | Breakout Low: ${first_low:.2f} > Div Low: ${divergence_low:.2f}")
    
    except (IndexError, KeyError) as e:
        # Skip symbols with data issues (trading halted, missing columns, etc.)
        continue
    except Exception as e:
        continue

# =========================
# Process Bearish Divergence (Short/Sell Signals)
# =========================
print("\n" + "=" * 80)
print("🔍 BEARISH DIVERGENCE - SELL/SHORT SIGNALS")
print("=" * 80)

bearish_signals = []
bearish_df = rsi_df[rsi_df['divergence_type'] == 'Bearish']

for _, row in bearish_df.iterrows():
    symbol = row['symbol']
    divergence_date = pd.to_datetime(row['last_date'])
    divergence_high = row['last_price']  # For bearish, last_price is the high

    # ✅ BUG FIX: Safe data fetching with proper checks
    try:
        # Get data for this symbol
        symbol_data = mongo_df[mongo_df['symbol'] == symbol]
        
        # Skip if no data in MongoDB
        if len(symbol_data) == 0:
            continue
        
        # Get data after divergence date
        symbol_data = symbol_data[symbol_data['date'] > divergence_date]
        symbol_data = symbol_data.sort_values('date')
        
        # Skip if no data after divergence date (trading halted/suspended)
        if len(symbol_data) == 0:
            continue
        
        # Get the first candle after divergence
        first_candle = symbol_data.iloc[0]
        first_high = first_candle['high']

        # Check breakout: First candle's high < divergence high (Breakdown)
        if first_high < divergence_high:
            bearish_signals.append({
                'symbol': symbol,
                'signal_type': 'SELL/SHORT (Bearish Breakdown)',
                'divergence_date': divergence_date.strftime('%Y-%m-%d'),
                'divergence_high': divergence_high,
                'breakdown_high': first_high,
                'strength': row['strength']
            })

            print(f"🔻 {symbol:<12} | Breakdown High: ${first_high:.2f} < Div High: ${divergence_high:.2f}")
    
    except (IndexError, KeyError) as e:
        # Skip symbols with data issues (trading halted, missing columns, etc.)
        continue
    except Exception as e:
        continue

# =========================
# Combine All Signals
# =========================
all_signals = bullish_signals + bearish_signals

if all_signals:
    df_signals = pd.DataFrame(all_signals)

    # Sort by symbol and signal_type
    df_signals = df_signals.sort_values(['symbol', 'signal_type']).reset_index(drop=True)

    # Add serial number
    df_signals.insert(0, 'no', range(1, len(df_signals) + 1))

    # Print summary
    print("\n" + "=" * 80)
    print("📊 BREAKOUT SIGNALS SUMMARY")
    print("=" * 80)
    print(f"\nTotal Bullish Breakouts: {len(bullish_signals)}")
    print(f"Total Bearish Breakdowns: {len(bearish_signals)}")
    print(f"Total Signals: {len(df_signals)}")

    print("\n" + "-" * 80)
    print("DETAILED SIGNALS:")
    print("-" * 80)

    for _, row in df_signals.iterrows():
        if 'BUY' in row['signal_type']:
            print(f"\n📈 {row['no']}. {row['symbol']} - {row['signal_type']}")
            print(f"   Divergence Date: {row['divergence_date']} | Low: ${row['divergence_low']:.2f}")
            print(f"   Breakout Low: ${row['breakout_low']:.2f}")
        else:
            print(f"\n📉 {row['no']}. {row['symbol']} - {row['signal_type']}")
            print(f"   Divergence Date: {row['divergence_date']} | High: ${row['divergence_high']:.2f}")
            print(f"   Breakdown High: ${row['breakdown_high']:.2f}")

    # Save short_buy.csv (only bullish signals for compatibility)
    if bullish_signals:
        short_buy_df = pd.DataFrame(bullish_signals)
        short_buy_df = short_buy_df[['symbol', 'divergence_low', 'divergence_date', 'breakout_low']]
        short_buy_df.to_csv(SHORT_BUY_FILE, index=False)
        print(f"\n✅ Short Buy signals saved to: {SHORT_BUY_FILE}")
    else:
        # Create empty file
        empty_df = pd.DataFrame(columns=['symbol', 'divergence_low', 'divergence_date', 'breakout_low'])
        empty_df.to_csv(SHORT_BUY_FILE, index=False)
        print(f"📄 No short buy signals, empty file created: {SHORT_BUY_FILE}")

else:
    print("\n❌ No breakout signals generated")

    # Create empty file
    empty_short = pd.DataFrame(columns=['symbol', 'divergence_low', 'divergence_date', 'breakout_low'])
    empty_short.to_csv(SHORT_BUY_FILE, index=False)

    print(f"📄 Empty files created")

# =========================
# Risk Management Summary
# =========================
print("\n" + "=" * 80)
print("⚠️ RISK MANAGEMENT GUIDELINES")
print("=" * 80)
print("""
1. Position Size: 1-2% of portfolio per trade
2. Risk/Reward: Minimum 1:2
3. Stop Loss: Always use stop loss
4. Take Profit: Book partial profits at each target
5. Trailing Stop: Move stop to entry after 50% gain
6. Confirmation: Wait for candle close above/below breakout level
""")

print("=" * 80)
print("✅ DONE!")
print("=" * 80)