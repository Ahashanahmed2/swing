import pandas as pd
import os

# 🔍 মূল ফাইলের পাথ
input_path = "./output/ai_signal/all_signals.csv"
output_path = "./output/ai_signal/sorted_signals.csv"

# 📥 CSV লোড করুন
if not os.path.exists(input_path):
    print(f"❌ ফাইল পাওয়া যায়নি: {input_path}")
    exit()

df = pd.read_csv(input_path)

# ✅ ai_score অনুসারে descending (বড় → ছোট) sort করুন
df_sorted = df.sort_values(by="ai_score", ascending=False)

# 💾 নতুন CSV ফাইলে সংরক্ষণ করুন
df_sorted.to_csv(output_path, index=False)

print(f"✅ Sorted signals saved to: {output_path}")
