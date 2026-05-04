# scripts/hf_download.py
# ✅ Step 1: বাদ (কোনো চেকপয়েন্ট ডিলিট হবে না)
# ✅ Step 2: LLM + PPO চেকপয়েন্ট বাদে বাকি সব ফাইল ডাউনলোড

import os
import time
import shutil
from huggingface_hub import (
    snapshot_download, 
    HfApi, 
    login, 
    create_commit,
    CommitOperationDelete
)

# =========================================================
# Step 1: বাতিল - কোনো চেকপয়েন্ট ডিলিট হবে না
# =========================================================

# (এই ফাংশন রাখা আছে, কিন্তু কল করা হবে না)
def cleanup_hf_checkpoints_before_download(keep_last=1, max_files_per_commit=100, sleep_between_commits=120):
    """এই ফাংশনটি ব্যবহার করা হবে না - কোনো চেকপয়েন্ট ডিলিট হবে না"""
    pass

# =========================================================
# Step 2: HF থেকে চেকপয়েন্ট বাদে বাকি সব ফাইল ডাউনলোড
# =========================================================

def download_from_hf():
    """HF Dataset থেকে ডাউনলোড (PPO ও LLM চেকপয়েন্ট বাদে, বাকি সব)"""
    print("\n📥 Downloading from HF Dataset...")
    print("   ❌ LLM checkpoints excluded")
    print("   ❌ PPO checkpoints excluded")
    print("   ✅ All CSV, models, and other files included")
    print("   🔄 Unlimited auto-retry on rate limit (5 min wait)")
    
    attempt = 0
    while True:  # ⬅️ আনলিমিটেড রিট্রাই লুপ
        attempt += 1
        try:
            snapshot_download(
                repo_id="ahashanahmed/csv",
                repo_type="dataset",
                local_dir="./csv",
                max_workers=2,
                local_dir_use_symlinks=False,
                token=os.getenv("hf_token"),
                resume_download=True,
                tqdm_class=None,
                ignore_patterns=[
                    "checkpoints/*",  # চেকপয়েন্ট ফোল্ডারের সব কিছু বাদ
                ],
            )
            print(f"✅ Download complete! (Total attempts: {attempt})")
            return  # সফল হলে বেরিয়ে যাবে
        except Exception as e:
            if "429" in str(e):
                wait_time = 300  # ৫ মিনিট
                print(f"\n⚠️ Rate limited! (Attempt {attempt})")
                print(f"⏳ Waiting {wait_time//60} minutes for rate limit reset...")
                print(f"📊 Already downloaded files will NOT be re-downloaded (resume mode)")
                time.sleep(wait_time)
                print(f"🔄 Resuming download...")
            else:
                print(f"❌ Download failed: {str(e)[:200]}")
                raise  # ৪২৯ ছাড়া অন্য এরর হলে থামবে

# =========================================================
# Step 3: লোকাল ক্লিনআপ (এই ফাংশনটিও চলবে না, চেকপয়েন্ট নেই)
# =========================================================

def cleanup_old_checkpoints(keep_last=1):
    """এই ফাংশনটিও ব্যবহার করা হবে না"""
    pass

# =========================================================
# এক্সিকিউট
# =========================================================

if __name__ == "__main__":
    print("="*60)
    print("🚀 HF DOWNLOAD (NO Checkpoints - CSV & Models Only)")
    print("="*60)
    
    # Step 1: স্কিপ - কোনো চেকপয়েন্ট ডিলিট হবে না
    print("⏭️  Step 1: Skipped (no checkpoint deletion)")
    
    # Step 2: শুধু CSV, মডেল ও অন্যান্য ফাইল ডাউনলোড (চেকপয়েন্ট বাদ)
    download_from_hf()
    
    # Step 3: স্কিপ - কোনো লোকাল ক্লিনআপ নেই
    print("⏭️  Step 3: Skipped (no local checkpoint cleanup)")
    
    print("\n" + "="*60)
    print("✅ hf_download.py সম্পূর্ণ!")
    print("="*60)
