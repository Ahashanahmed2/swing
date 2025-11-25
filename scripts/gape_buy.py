import pandas as pd
import os

# ইনপুট ফাইল পাথ
gape_file = "./csv/gape.csv"
mongodb_file = "./csv/mongodb.csv"

# আউটপুট ফাইল পাথ
output_file1 = "./output/ai_signal/gape_buy.csv"
output_file2 = "./csv/gape_buy.csv"

# ডেটা পড়া
gape_df = pd.read_csv(gape_file)
mongodb_df = pd.read_csv(mongodb_file)

results = []

for _, last_row in gape_df.iterrows():
    symbol = last_row['symbol']
    Arow_date = last_row['last_row_date']
    lastrowhigh = last_row['last_row_high']
    lastrowlow = last_row['last_row_low']
    lastrowclose = last_row['last_row_close']

    # mongodb.csv থেকে ওই symbol এর ডেটা
    mongodata = mongodb_df[mongodb_df['symbol'] == symbol].sort_values(by='date')

    # ওই তারিখের আগের row গুলো
    prevrows = mongodata[mongodata['date'] < Arow_date]
    if prevrows.empty:
        continue

    # শেষ আগের row
    prevrow = prevrows.iloc[-1]
    Browdate = prevrow['date']

    # শর্ত মিলানো
    if (lastrowlow > prevrow['low']) and (lastrowclose > prevrow['high']):
        # pre_candle খুঁজে বের করা
        pre_candle = None
        for i in range(len(prevrows)-1, -1, -1):  # শেষ থেকে লুপ
            row = prevrows.iloc[i]
            if row['low'] <= lastrowhigh <= row['high'] or row['low'] <= lastrowlow <= row['high']:
                pre_candle = row
                break

        if pre_candle is None:
            continue

        pre_candle_date = pre_candle['date']

        # pre_candle ও last_row এর মাঝে যত row আছে
        between_rows = mongodata[(mongodata['date'] > pre_candle_date) & (mongodata['date'] < Arow_date)]
        if between_rows.empty:
            continue

        # low_candle বের করা (যার low সবচেয়ে কম)
        low_candle = between_rows.loc[between_rows['low'].idxmin()]
        low_candle_date = low_candle['date']
        SL = low_candle['low']

        # low_candle ও last_row এর মাঝে কতগুলো candle আছে
        candles_between = mongodata[(mongodata['date'] > low_candle_date) & (mongodata['date'] < Arow_date)]
        candle_count = len(candles_between)

        # gape.csv এর row + নতুন ফিল্ড যুক্ত করা
        result_row = last_row.to_dict()
        result_row.update({
           'low_candle_date': low_candle_date,
           'candle_count': candle_count + 1,
           'SL': SL
        })
        results.append(result_row)

# আউটপুট ডিরেক্টরি তৈরি করা যদি না থাকে
os.makedirs(os.path.dirname(output_file1), exist_ok=True)
os.makedirs(os.path.dirname(output_file2), exist_ok=True)

# ফলাফল CSV তে লেখা (SL ascending, তারপর candle_count ascending)
if results:
    df = pd.DataFrame(results)
    df = df.sort_values(by=['SL', 'candle_count'], ascending=[True, True]).reset_index(drop=True)

    # ✅ ascending sort হওয়ার পর নতুন row_id যুক্ত করা
    df.insert(0, 'row_id', range(1, len(df) + 1))

    df.to_csv(output_file1, index=False)
    df.to_csv(output_file2, index=False)
    print(f"✅ Output saved to {output_file1} and {output_file2}")
else:
    print("⚠️ কোনো মিল পাওয়া যায়নি, আউটপুট ফাইল তৈরি হয়নি।")