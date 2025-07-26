import pandas as pd

# CSV ফাইল লোড
df = pd.read_csv('./csv/mongodb.csv')

# তারিখ অনুযায়ী সাজানো
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by=['symbol', 'date'], inplace=True)

# শর্ত পূরণ করা রো গুলো রাখার জন্য
final_rows = []

# প্রতিটি symbol আলাদাভাবে প্রসেস
for symbol, group in df.groupby('symbol'):
    group = group.reset_index(drop=True)
    
    # শেষ দিনের রো
    last_row = group.iloc[-1]
    
    try:
        rsi_last = float(last_row['rsi'])
    except:
        continue  # যদি RSI NaN বা ভুল ফরম্যাট হয়

    # RSI 30 < rsi < 60 হলে চেক করা শুরু
    if 30 < rsi_last < 60:
        # আগের রো-গুলো উল্টো করে চেক
        for i in range(len(group) - 2, -1, -1):
            try:
                rsi_prev = float(group.loc[i, 'rsi'])
                close = float(group.loc[i, 'close'])
                bb_middle = float(group.loc[i, 'bb_middle'])
                bb_lower = float(group.loc[i, 'bb_lower'])
                macd = float(group.loc[i, 'macd'])
                macd_hist = float(group.loc[i, 'macd_hist'])
            except:
                continue  # যদি কোন ভ্যালু মিসিং থাকে

            # শর্ত মিলিয়ে দেখা
            if rsi_prev <= 30 and bb_lower < close < bb_middle and macd > macd_hist:
                # শুধুমাত্র প্রয়োজনীয় কলাম রেখে যোগ করব
                final_rows.append({
                    'symbol': last_row['symbol'],
                    'date': last_row['date'].strftime('%Y-%m-%d'),
                    'close': last_row['close'],
                    'rsi': rsi_last
                })
                break  # একবার শর্ত মিললে আর চেক করার দরকার নেই

# ফলাফল DataFrame এ রূপান্তর ও CSV ফাইলে সংরক্ষণ
if final_rows:
    output_df = pd.DataFrame(final_rows)
    output_df.to_csv('./csv/filtered_output.csv', index=False)
    print("ফাইল './csv/filtered_output.csv' এ সংরক্ষণ করা হয়েছে।")
else:
    print("কোনো ডেটা শর্ত পূরণ করেনি।")
