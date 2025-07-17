import pandas as pd
import os

# CSV ফাইল থেকে ডেটা পড়া
df = pd.read_csv('./csv/mongodb.csv')

# date কে datetime এ কনভার্ট করা
df['date'] = pd.to_datetime(df['date'])

# output directory তৈরি করা (যদি না থাকে)
os.makedirs('csv/swing/', exist_ok=True)

# ফলাফল রাখার জন্য লিস্ট
results = []

# unique symbols গুলো ধরা
symbols = df['symbol'].unique()

# প্রতিটি symbol এর জন্য লজিক প্রয়োগ
for symbol in symbols:
    symbol_df = df[df['symbol'] == symbol].reset_index(drop=True)

    # প্রতিটি row এর জন্য
    for i in range(len(symbol_df)):
        a_row = symbol_df.iloc[i]

        for j in range(len(symbol_df)):
            if i == j:
                continue  # একই row চেক করবে না

            b_row = symbol_df.iloc[j]

            # শর্তগুলো চেক
            if a_row['low'] > b_row['low'] and a_row['rsi'] < b_row['rsi']:
                # নতুন row তৈরি
                combined_row = {
                    'symbol': symbol,
                    'a_row_date': a_row['date'],
                    'a_row_low': a_row['low'],
                    'a_row_volume': a_row['volume'],
                    'a_row_trades': a_row['trades'],
                    'a_row_rsi': a_row['rsi'],
                    'b_row_date': b_row['date'],
                    'b_row_low': b_row['low'],
                    'b_row_volume': b_row['volume'],
                    'b_row_trades': b_row['trades'],
                    'b_row_rsi': b_row['rsi']
                }
                results.append(combined_row)
                break  # একবার মিললেই রেজাল্টে যোগ করবে

# ফিল্টারড ডেটা থেকে ডেটাফ্রেম তৈরি
filtered_df = pd.DataFrame(results)

# CSV তে সেভ করা
output_path = 'csv/swing/filtered_low_rsi_candles.csv'
filtered_df.to_csv(output_path, index=False)

print(f"ফিল্টারড ডেটা '{output_path}' ফাইলে সেভ করা হয়েছে।")