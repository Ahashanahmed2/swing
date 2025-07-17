import os
import pandas as pd
# সোর্স ডিরেক্টরি
source_dir = "./output/ai_signal/"

# সব .csv ফাইল লোড করুন
csv_files = [f for f in os.listdir(source_dir) if f.endswith('all_signals.csv')]

# ট্রেন্ড ও অ্যাকশন অনুযায়ী গ্রুপ
uptrend_rows = []
downtrend_rows = []
buy_rows = []
sell_rows = []
hold_rows = []

for file in csv_files:
    file_path = os.path.join(source_dir, file)
    try:
        df = pd.read_csv(file_path)
        if 'trend' in df.columns and 'signal_type' in df.columns:
            # ট্রেন্ড অনুযায়ী
            uptrend_rows.append(df[df['trend'].str.lower() == 'uptrand'])
            downtrend_rows.append(df[df['trend'].str.lower() == 'downtrand'])

            # সিগন্যাল টাইপ অনুযায়ী
            buy_rows.append(df[df['signal_type'].str.lower() == 'buy'])
            sell_rows.append(df[df['signal_type'].str.lower() == 'sell'])
            hold_rows.append(df[df['signal_type'].str.lower() == 'hold'])

    except Exception as e:
        print(f"❌ Error processing file {file}: {e}")

# সংযুক্ত ও সংরক্ষণ
def save_combined(rows, filename):
    if rows:
        combined_df = pd.concat(rows, ignore_index=True)
        combined_df.to_csv(os.path.join(source_dir, filename), index=False)
        print(f"✅ Saved: {filename} ({len(combined_df)} rows)")

# 🟢 ট্রেন্ড অনুযায়ী
save_combined(uptrend_rows, "uptrend_signals.csv")
save_combined(downtrend_rows, "downtrend_signals.csv")

# 🟠 সিগন্যাল টাইপ অনুযায়ী
save_combined(buy_rows, "buy_signals.csv")
save_combined(sell_rows, "sell_signals.csv")
save_combined(hold_rows, "hold_signals.csv")

