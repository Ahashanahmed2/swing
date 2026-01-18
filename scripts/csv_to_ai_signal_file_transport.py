import os
import shutil

def copy_csv_file():
    # সোর্স ও টার্গেট ফাইলের পাথ
    source_file = "./csv/uptrand.csv"
    target_dir = "./output/ai_signal"
    target_file = os.path.join(target_dir, "uptrand.csv")
    
    try:
        # টার্গেট ডিরেক্টরি তৈরি করুন (যদি না থাকে)
        os.makedirs(target_dir, exist_ok=True)
        
        # ফাইল কপি করুন
        shutil.copy2(source_file, target_file)
        
        print(f"ফাইল সফলভাবে কপি করা হয়েছে:")
        print(f"সোর্স: {source_file}")
        print(f"টার্গেট: {target_file}")
        
    except FileNotFoundError:
        print(f"ত্রুটি: সোর্স ফাইল পাওয়া যায়নি: {source_file}")
    except PermissionError:
        print(f"ত্রুটি: ফাইল অ্যাক্সেস করার অনুমতি নেই")
    except Exception as e:
        print(f"ত্রুটি: {str(e)}")

if __name__ == "__main__":
    copy_csv_file()