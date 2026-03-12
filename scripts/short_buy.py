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
print("PROCESSING SIGNALS (LOW > LAST_LOW LOGIC):")
print("="*80)

for _, rsi_row in rsi_df.iterrows():
    symbol = str(rsi_row['symbol']).strip().upper()
    last_low = rsi_row['last_row_low']
    last_row_date = pd.to_datetime(rsi_row['last_row_date'], errors='coerce')
    second_row_date = pd.to_datetime(rsi_row['second_row_date'], errors='coerce')

    if pd.isna(last_row_date) or pd.isna(second_row_date) or symbol not in mongo_groups.groups:
        continue

    symbol_group = mongo_groups.get_group(symbol).sort_values('date').reset_index(drop=True)

    # Find last_row and second_row in MongoDB data
    last_row_candidates = symbol_group[symbol_group['date'] == last_row_date]
    second_row_candidates = symbol_group[symbol_group['date'] == second_row_date]
    
    if last_row_candidates.empty or second_row_candidates.empty:
        continue
    
    last_row = last_row_candidates.iloc[-1]

    # শুধু low > last_low চেক করা হচ্ছে (close চেক করা হচ্ছে না)
    if not (last_row['low'] > last_low):
        continue

    # Count rows between second_row_date and last_row_date (exclusive of both dates)
    mask = (symbol_group['date'] > second_row_date) & (symbol_group['date'] < last_row_date)
    rows_between_count = mask.sum()

    # Store signal with row counts
    output_rows.append({
        'symbol': symbol,
        'date': last_row['date'].date(),
        'buy': last_row['close'],
        'gap': rows_between_count,  # rows between second_row_date and last_row_date
        'second_row_date': second_row_date.date()
    })

    print(f"✅ Signal: {symbol} | Date={last_row['date'].date()} | Buy={last_row['close']:.2f} | "
          f"Low={last_row['low']:.2f} > Last_Low={last_low:.2f} | Gap: {rows_between_count}")

# DataFrame তৈরি করা
if output_rows:
    df = pd.DataFrame(output_rows)

    # ডুপ্লিকেট রিমুভ (যদি একই symbol এবং date একাধিকবার আসে)
    df = df.drop_duplicates(subset=['symbol', 'date'])

    # Sort by gap in descending order (more rows first)
    df = df.sort_values('gap', ascending=False).reset_index(drop=True)

    # প্রথম কলাম হিসেবে সিরিয়াল নাম্বার যোগ করা
    df.insert(0, 'no', range(1, len(df) + 1))

    # কলামের অর্ডার ঠিক করা
    df = df[['no', 'symbol', 'date', 'buy', 'gap', 'second_row_date']]

    print(f"\n{'='*80}")
    print(f"✅ TOTAL SIGNALS: {len(df)}")
    print(f"{'='*80}")
    print(df.to_string())

    # Show summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS:")
    print(f"{'='*80}")
    print(f"Average gap: {df['gap'].mean():.2f}")
    print(f"Max gap: {df['gap'].max()}")
    print(f"Min gap: {df['gap'].min()}")
    print(f"Total gap sum: {df['gap'].sum()}")

else:
    print(f"\n❌ No signals generated")
    df = pd.DataFrame(columns=['no', 'symbol', 'date', 'buy', 'gap', 'second_row_date'])

# Save to CSV
df.to_csv(output_path2, index=False)
print(f"\n✅ Saved to {output_path2}")
print(f"📁 File contains columns: {', '.join(df.columns)}")