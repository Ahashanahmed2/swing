import pandas as pd
import os

# Paths
rsi_path = './csv/rsi_diver_retest.csv'
mongo_path = './csv/mongodb.csv'
output_path2 = './csv/short_buy.csv'

os.makedirs(os.path.dirname(output_path2), exist_ok=True)

# Delete old file
if os.path.exists(output_path2):
    os.remove(output_path2)

# Read data
rsi_df = pd.read_csv(rsi_path)
mongo_df = pd.read_csv(mongo_path)

# Clean column names (spaces to underscores)
rsi_df.columns = rsi_df.columns.str.replace(" ", "_")

# Ensure MongoDB date is datetime
mongo_df['date'] = pd.to_datetime(mongo_df['date'], errors='coerce')
mongo_df = mongo_df.dropna(subset=['date'])
mongo_groups = mongo_df.groupby('symbol')

output_rows = []

print("\n" + "="*80)
print("PROCESSING SIGNALS (CLOSE > LAST_LOW AND LOW > LAST_LOW LOGIC):")
print("="*80)

for _, rsi_row in rsi_df.iterrows():
    symbol = str(rsi_row['symbol']).strip().upper()
    last_low = rsi_row['last_row_low']
    last_row_date = pd.to_datetime(rsi_row['last_row_date'], errors='coerce')

    if pd.isna(last_row_date) or symbol not in mongo_groups.groups:
        continue

    symbol_group = mongo_groups.get_group(symbol).sort_values('date').reset_index(drop=True)

    last_row_candidates = symbol_group[symbol_group['date'] == last_row_date]
    if last_row_candidates.empty:
        continue
    last_row = last_row_candidates.iloc[-1]

    # দুটি কন্ডিশন একসাথে চেক করা হচ্ছে
    if not (last_row['close'] > last_low and last_row['low'] > last_low):
        continue

    # সিগন্যাল স্টোর করছি
    output_rows.append({
        'symbol': symbol,
        'date': last_row['date'].date(),
        'buy': last_row['close']
    })

    print(f"✅ Signal: {symbol} | Date={last_row['date'].date()} | Buy={last_row['close']:.2f} | Low={last_row['low']:.2f} | Last_Low={last_low:.2f}")

# DataFrame তৈরি করা
if output_rows:
    df = pd.DataFrame(output_rows)

    # ডুপ্লিকেট রিমুভ (যদি একই symbol এবং date একাধিকবার আসে)
    df = df.drop_duplicates(subset=['symbol', 'date'])

    # প্রথম কলাম হিসেবে সিরিয়াল নাম্বার যোগ করা
    df.insert(0, 'no', range(1, len(df) + 1))

    # কলামের অর্ডার ঠিক করা: no, symbol, date, buy
    df = df[['no', 'symbol', 'date', 'buy']]

    print(f"\n{'='*80}")
    print(f"✅ TOTAL SIGNALS: {len(df)}")
    print(f"{'='*80}")
    print(df.to_string())

else:
    print(f"\n❌ No signals generated")
    df = pd.DataFrame(columns=['no', 'symbol', 'date', 'buy'])

# Save to CSV
df.to_csv(output_path2, index=False)
print(f"\n✅ Saved to {output_path2}")
print(f"📁 File contains columns: {', '.join(df.columns)}")