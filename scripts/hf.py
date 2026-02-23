import pandas as pd
import os
from datetime import datetime
from hf_uploader import download_from_hf, REPO_ID, HF_TOKEN

# -------------------------------------------------------------------
# Step 1: Download CSV from HF if needed
# -------------------------------------------------------------------
print("📥 Checking for CSV files from Hugging Face...")

# CSV ফোল্ডার তৈরি করুন
csv_folder = './csv'
os.makedirs(csv_folder, exist_ok=True)

# HF থেকে ডাউনলোড করার চেষ্টা করুন
download_success = download_from_hf(csv_folder, REPO_ID, HF_TOKEN)

if download_success:
    print(f"✅ HF data download success")

    # ডাউনলোড করা ফাইলগুলো দেখান
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    print(f"📊 Found {len(csv_files)} CSV files: {csv_files}")
else:
    print(f"⚠️ No data found in HF. Will work with existing local data or create new.")

# -------------------------------------------------------------------
# Step 2: Main processing function
# -------------------------------------------------------------------
def main():
    # বর্তমান ডিরেক্টরি চেক করুন
    current_dir = os.getcwd()
    print(f"\n📂 বর্তমান ডিরেক্টরি: {current_dir}")

    # CSV ফাইল পাথ নির্ধারণ করুন
    csv_file_path = "./csv/mongodb.csv"

    # ফাইল আছে কিনা চেক করুন
    if not os.path.exists(csv_file_path):
        print(f"⚠️ {csv_file_path} পাওয়া যায়নি, অন্যান্য পাথ চেক করা হচ্ছে...")

        # অন্য পাথ চেষ্টা করুন
        alt_paths = [
            "csv/mongodb.csv",
            os.path.join(current_dir, "csv", "mongodb.csv"),
            "mongodb.csv",
            "./mongodb.csv",
            os.path.join("..", "csv", "mongodb.csv")  # এক লেভেল উপরে চেক করুন
        ]

        found = False
        for path in alt_paths:
            if os.path.exists(path):
                csv_file_path = path
                print(f"✅ ফাইল পাওয়া গেছে: {path}")
                found = True
                break
        
        if not found:
            print("❌ কোনো পাথেই mongodb.csv পাওয়া যায়নি!")
            
            # ডিরেক্টরি লিস্টিং দেখান
            print("\n📋 উপলব্ব ফাইল ও ফোল্ডারসমূহ:")
            for item in os.listdir("."):
                if os.path.isdir(item):
                    print(f"  📁 {item}/")
                    # CSV ফোল্ডারের কন্টেন্ট দেখান
                    if item == "csv" and os.path.exists(item):
                        csv_contents = [f for f in os.listdir(item) if f.endswith('.csv')]
                        if csv_contents:
                            print(f"    CSV files: {csv_contents}")
                else:
                    if item.endswith('.csv'):
                        print(f"  📄 {item}")
            
            return

    # এখন CSV ফাইল পড়ুন
    try:
        df = pd.read_csv(csv_file_path)
        print(f"\n✅ CSV ফাইল সফলভাবে পড়া হয়েছে।")
        print(f"📊 মোট {len(df)} টি রেকর্ড পাওয়া গেছে।")
        print(f"📋 কলামসমূহ: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ CSV ফাইল পড়তে সমস্যা: {e}")
        return

    # ডেট টাইপ কনভার্ট করুন (যদি 'date' কলাম থাকে)
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            print(f"✅ Date column converted successfully")
        except Exception as e:
            print(f"⚠️ Date column convert error: {e}")
    else:
        print(f"⚠️ 'date' column not found in the data")
        print(f"Available columns: {list(df.columns)}")

    # আউটপুট ডিরেক্টরি তৈরি করুন
    output_base_dir = "./csv/trand/"
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"✅ Output directory created: {output_base_dir}")

    # -------------------------------------------------------------------
    # আপনার বাকি প্রসেসিং কোড এখানে যোগ করুন
    # -------------------------------------------------------------------
    
    # উদাহরণ: প্রথম কয়েকটি রো দেখান
    print("\n📊 প্রথম ৫ টি রেকর্ড:")
    print(df.head())
    
    # ডেটা সম্পর্কে মৌলিক তথ্য
    print("\n📊 ডেটা সম্পর্কে তথ্য:")
    print(f"ডেটার আকার: {df.shape}")
    print(f"ডেটার টাইপ:\n{df.dtypes}")
    
    # এখানে আপনার অন্যান্য ফাংশন কল করুন
    # process_data(df, output_base_dir)
    
    print("\n✅ সব কাজ সফলভাবে সম্পন্ন হয়েছে!")

# -------------------------------------------------------------------
# Script execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()