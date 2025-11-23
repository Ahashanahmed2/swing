import pandas as pd
import os

# ইনপুট ফাইল পাথ
gape_file = "./csv/gape.csv"
mongodb_file = "./csv/mongodb.csv"

# আউটপুট ফাইল পাথ
output_file = "./output/ai_signal/gape_buy.csv"

# ডেটা পড়া
gape_df = pd.read_csv(gape_file)
mongodb_df = pd.read_csv(mongodb_file)

results = []
row_id = 1  # প্রতিটি আউটপুট row এর জন্য সিরিয়াল নাম্বার

for symbol in gape_df['symbol'].unique():
    # gape.csv থেকে ওই symbol এর শেষ row
    symbol_data = gape_df[gape_df['symbol'] == symbol].sort_values(by='last_row_date')
    last_row = symbol_data.iloc[-1]

    A_row_date = last_row['last_row_date']
    last_row_low = last_row['last_row_low']
    last_row_close = last_row['last_row_close']

    # mongodb.csv থেকে ওই symbol এর ডেটা
    mongo_data = mongodb_df[mongodb_df['symbol'] == symbol].sort_values(by='date')

    # ওই তারিখের আগের row খুঁজে বের করা
    prev_rows = mongo_data[mongo_data['date'] < A_row_date]
    if prev_rows.empty:
        continue

    prev_row = prev_rows.iloc[-1]  # শেষ আগের row
    B_row_date = prev_row['date']

    # শর্ত মিলানো
    if (last_row_low > prev_row['low']) and (last_row_close > prev_row['high']):
        results.append({
            'row_id': row_id,
            'symbol': symbol,
            'A_row_date': A_row_date,
            'B_row_date': B_row_date,
            'last_row_low': last_row_low,
            'last_row_close': last_row_close
        })
        row_id += 1

# আউটপুট ডিরেক্টরি তৈরি করা যদি না থাকে
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# ফলাফল CSV তে লেখা
if results:
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"✅ Output saved to {output_file}")
else:
    print("⚠️ কোনো মিল পাওয়া যায়নি, আউটপুট ফাইল তৈরি হয়নি।")