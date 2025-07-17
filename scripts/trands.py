import os
import pandas as pd

# ইনপুট ফোল্ডারগুলো
input_dirs = [
    "./csv/swing/imbalanceZone/down_to_up",
    "./csv/swing/imbalanceZone/up_to_down"
]

# আউটপুট ফোল্ডারগুলো
output_dirs = [
    "./output/ai_signal",
    "./csv/swing"
]

# সব আউটপুট ফোল্ডার বানিয়ে ফেলা, যদি না থাকে
for out_dir in output_dirs:
    os.makedirs(out_dir, exist_ok=True)

# প্রতিটি ইনপুট ফোল্ডার প্রসেস করা
for dir_path in input_dirs:
    all_data = []
    folder_name = os.path.basename(dir_path.rstrip("/"))  # ফোল্ডারের নাম

    # প্রতিটি CSV ফাইল একত্রিত করা
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(dir_path, file_name)
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
            except Exception as e:
                print(f"Could not read {file_path}: {e}")

    # একত্রিত ডেটা থাকলে দুই জায়গায় ফাইল সেভ করা
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        for out_dir in output_dirs:
            output_file = os.path.join(out_dir, f"{folder_name}.csv")
            combined_df.to_csv(output_file, index=False)
            print(f"✅ Saved: {output_file}")
    else:
        print(f"⚠️ No valid CSV files found in {dir_path}")
