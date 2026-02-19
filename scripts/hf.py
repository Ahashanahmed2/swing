import pandas as pd
import os
from datetime import datetime
from hf_uploader import download_from_hf_or_run_script

# ... আপনার অন্যান্য ফাংশনগুলি আগের মতই থাকবে ...

def main():
    # প্রথমে HF থেকে ডেটা ডাউনলোড করুন
    download_from_hf_or_run_script()
    
    # বর্তমান ডিরেক্টরি চেক করুন
    current_dir = os.getcwd()
    print(f"বর্তমান ডিরেক্টরি: {current_dir}")
    
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
            "./mongodb.csv"
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                csv_file_path = path
                print(f"✅ ফাইল পাওয়া গেছে: {path}")
                break
        else:
            print("❌ কোনো পাথেই mongodb.csv পাওয়া যায়নি!")
            print("উপলব্ধ ফাইল:", os.listdir("."))
            if os.path.exists("csv"):
                print("csv ফোল্ডারের কন্টেন্ট:", os.listdir("csv"))
            return
    
    # এখন CSV ফাইল পড়ুন
    try:
        df = pd.read_csv(csv_file_path)
        print(f"✅ CSV ফাইল সফলভাবে পড়া হয়েছে। {len(df)} রেকর্ড পাওয়া গেছে।")
    except Exception as e:
        print(f"❌ CSV ফাইল পড়তে সমস্যা: {e}")
        return
    
    df['date'] = pd.to_datetime(df['date'])
    
    output_base_dir = "./csv/trand/"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # ... বাকি কোড

if __name__ == "__main__":
    main()