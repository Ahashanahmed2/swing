import pandas as pd
import os

# CSV ফাইল লোড
df = pd.read_csv('./csv/mongodb.csv')
df['date'] = pd.to_datetime(df['date'])

# প্রতিটি সিম্বলের সর্বশেষ তারিখ বের করুন
latest_dates = df.groupby('symbol')['date'].max().reset_index()

# শুধুমাত্র সর্বশেষ তারিখের রেকর্ডগুলো ফিল্টার করুন
latest_data = pd.merge(df, latest_dates, on=['symbol', 'date'])

# প্রয়োজনীয় কলামগুলো সিলেক্ট করুন
result = latest_data[['symbol', 'date', 'macd', 'macd_signal', 'macd_hist']]

# তারিখ অনুসারে সাজান
result = result.sort_values(['symbol', 'date'])

# আউটপুট ডিরেক্টরি তৈরি করুন
output_dir = './output/ai_signal/'
os.makedirs(output_dir, exist_ok=True)

# CSV হিসেবে সেভ করুন
output_path = os.path.join(output_dir, 'all_macd.csv')
result.to_csv(output_path, index=False)

print(f"ফাইল সেভ হয়েছে: {output_path}")
print(f"টোটাল রেকর্ড: {len(result)}")
print(f"টোটাল সিম্বল: {result['symbol'].nunique()}")
print("\nপ্রথম ১০ টি রেকর্ড:")
print(result.head(10).to_string(index=False))