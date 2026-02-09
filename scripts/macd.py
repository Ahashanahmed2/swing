import pandas as pd
import os

# CSV ফাইল লোড করুন
df = pd.read_csv('./csv/mongodb.csv')

# তারিখ অনুযায়ী সাজান
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol', 'date'])

# প্রতিটি সিম্বলের জন্য আগের দিনের MACD মান বের করুন
df['prev_macd'] = df.groupby('symbol')['macd'].shift(1)

# ফিল্টার শর্ত:
# 1. macd > macd_signal
# 2. আগের দিনের macd < 0 (নেগেটিভ)
# 3. বর্তমান দিনের macd > 0 (পজিটিভ)
filtered_symbols = df[
    (df['macd'] > df['macd_signal']) &  # শর্ত ১
    (df['prev_macd'] < 0) &              # শর্ত ২: আগের দিন নেগেটিভ
    (df['macd'] > 0)                     # শর্ত ৩: বর্তমান দিন পজিটিভ
]

# প্রতিটি সিম্বলের সর্বশেষ রেকর্ড নিন
latest_filtered = filtered_symbols.groupby('symbol').tail(1)

# প্রয়োজনীয় কলামগুলো নির্বাচন করুন
result = latest_filtered[['symbol', 'date', 'macd', 'macd_signal', 'macd_hist', 'prev_macd']]

# অতিরিক্ত কলাম যোগ করুন ক্রসিং শো করার জন্য
result['cross_type'] = 'Centerline Cross (Negative to Positive)'
result['signal_strength'] = result['macd_hist']

# আউটপুট ডিরেক্টরি তৈরি করুন
output_dir = './output/ai_signal/'
os.makedirs(output_dir, exist_ok=True)

# CSV হিসেবে সেভ করুন
output_path = os.path.join(output_dir, 'centerline_cross_symbols.csv')
result.to_csv(output_path, index=False)

print(f"ফাইল সফলভাবে সেভ হয়েছে: {output_path}")
print(f"টোটাল ক্রসিং সিম্বল: {len(result)}")
print(f"সেন্টারলাইন ক্রসিং (নেগেটিভ থেকে পজিটিভ) শর্ত:")
print("1. macd > macd_signal")
print("2. পূর্বের দিন macd < 0 (নেগেটিভ)")
print("3. বর্তমান দিন macd > 0 (পজিটিভ)")
print("\nক্রসিং সিম্বলগুলোর তালিকা:")
print(result[['symbol', 'date', 'macd', 'prev_macd', 'macd_hist']].to_string(index=False))

# বিস্তারিত বিশ্লেষণ
print("\n" + "="*50)
print("বিস্তারিত বিশ্লেষণ:")
print("="*50)

for _, row in result.iterrows():
    print(f"\nসিম্বল: {row['symbol']}")
    print(f"তারিখ: {row['date'].strftime('%Y-%m-%d')}")
    print(f"বর্তমান MACD: {row['macd']:.4f}")
    print(f"আগের দিন MACD: {row['prev_macd']:.4f}")
    print(f"MACD সিগন্যাল: {row['macd_signal']:.4f}")
    print(f"MACD হিস্টোগ্রাম: {row['macd_hist']:.4f}")
    print(f"ক্রসিং ধরন: {row['cross_type']}")