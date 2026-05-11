import pandas as pd
import numpy as np
import os
import sys

# -------------------------------------------------------------------
# Step 1: Load the existing CSV file
# -------------------------------------------------------------------
csv_path = './csv/mongodb.csv'
output_dir = './output/ai_signal'
output_file = os.path.join(output_dir, 'ema_21.csv')

# Check if CSV exists
if not os.path.exists(csv_path):
    print(f"❌ CSV file not found: {csv_path}")
    sys.exit(1)

print(f"📂 Loading data from: {csv_path}")
df = pd.read_csv(csv_path)

# Convert date
df['date'] = pd.to_datetime(df['date'])

# Check if ema_21 column exists
if 'ema_21' not in df.columns:
    print("❌ 'ema_21' column not found in the CSV file.")
    print("Please run the main script first to generate EMA-21 values.")
    sys.exit(1)

print(f"✅ Total records: {len(df)}")
print(f"🏷️ Total symbols: {df['symbol'].nunique()}")
print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")

# -------------------------------------------------------------------
# Step 2: Sort by symbol and date
# -------------------------------------------------------------------
df = df.sort_values(['symbol', 'date'])

# -------------------------------------------------------------------
# Step 3: Find symbols meeting the condition
# -------------------------------------------------------------------
print("\n🔍 Finding symbols where:")
print("   - Latest candle: High > EMA-21")
print("   - AND Latest candle: Low <= EMA-21")
print("   (EMA-21 is INSIDE the latest candle)")

def find_signals(group):
    # Sort by date
    group = group.sort_values('date')
    
    # Need at least 1 row
    if len(group) < 1:
        return pd.DataFrame()
    
    # Get latest row only
    latest = group.iloc[-1]
    
    # Check condition using ema_21 column
    if (pd.notna(latest['ema_21']) and 
        pd.notna(latest['close']) and 
        pd.notna(latest['low'])):
        
        # 🔥 মূল কন্ডিশন:
        # High > EMA-21 AND Low <= EMA-21
        if latest['close'] > latest['ema_21'] and latest['low'] <= latest['ema_21']:
            return pd.DataFrame({
                'symbol': [latest['symbol']],
                'date': [latest['date']],
                'close': [latest['close']],
                'high': [latest['high']],
                'low': [latest['low']],
                'ema_21': [latest['ema_21']]
            })
    
    return pd.DataFrame()

# Process each symbol
signal_dfs = []
for symbol, group in df.groupby('symbol'):
    signal = find_signals(group)
    if not signal.empty:
        signal_dfs.append(signal)
        print(f"✅ {symbol}")

# -------------------------------------------------------------------
# Step 4: Save results
# -------------------------------------------------------------------
# Create output directory
os.makedirs(output_dir, exist_ok=True)

if signal_dfs:
    result_df = pd.concat(signal_dfs, ignore_index=True)
    print(f"\n🎯 Total {len(result_df)} symbols found")
    
    # Show results
    print("\n📊 Signals found:")
    for _, row in result_df.iterrows():
        print(f"   - {row['symbol']}: {row['date'].strftime('%Y-%m-%d')} | "
              f"High: {row['high']:.2f} | Low: {row['low']:.2f} | EMA-21: {row['ema_21']:.2f}")
else:
    result_df = pd.DataFrame(columns=['symbol', 'date', 'close', 'high', 'low', 'ema_21'])
    print("\n❌ No symbols found matching the condition")

# Save to CSV
result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n💾 Results saved to: {output_file}")
print("✅ Done!")
