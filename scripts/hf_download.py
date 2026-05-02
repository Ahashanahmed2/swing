# scripts/hf_download.py
# HF Dataset থেকে সম্পূর্ণ ডাউনলোড (কোনো ফাইল স্কিপ নয়)
# ✅ HF_TOKEN + hf_transfer + max_workers ব্যবহার করে দ্রুত ও নিরাপদ ডাউনলোড

import os
import shutil
from huggingface_hub import snapshot_download, HfApi, login

# =========================================================
# HF Dataset-এ পুরনো চেকপয়েন্ট ডিলিট (ডাউনলোডের আগে)
# =========================================================

def cleanup_hf_checkpoints_before_download(keep_last=1):
    """HF Dataset-এ শুধু শেষ ১টি চেকপয়েন্ট রাখুন, বাকি ডিলিট (ডাউনলোডের আগে)"""
    
    token = os.getenv("hf_token")
    if not token:
        print("ℹ️ No HF_TOKEN, skipping HF cleanup")
        return
    
    try:
        login(token=token)
        api = HfApi(token=token)
        
        print("\n🔍 Checking HF Dataset for old checkpoints (before download)...")
        
        files = api.list_repo_files(repo_id="ahashanahmed/csv", repo_type="dataset")
        
        checkpoint_folders = set()
        for f in files:
            if f.startswith("checkpoints/checkpoint-"):
                folder = f.split("/")[1]  # "checkpoint-80"
                checkpoint_folders.add(folder)
        
        if not checkpoint_folders:
            print("ℹ️ No checkpoints found in HF Dataset")
            return
        
        # স্টেপ নাম্বার অনুযায়ী সর্ট
        def get_step_num(folder):
            try:
                return int(folder.replace("checkpoint-", ""))
            except:
                return 0
        
        checkpoint_list = sorted(checkpoint_folders, key=get_step_num)
        
        if len(checkpoint_list) <= keep_last:
            print(f"✅ Only {len(checkpoint_list)} checkpoints in HF, no cleanup needed")
            return
        
        # শেষ ১টি বাদে বাকি ডিলিট
        to_delete = checkpoint_list[:-keep_last]
        to_keep = checkpoint_list[-keep_last:]
        
        print(f"\n🗑️ Cleaning up HF Dataset checkpoints...")
        print(f"   Keeping last {keep_last}: {to_keep}")
        print(f"   Deleting {len(to_delete)} old checkpoints...")
        
        for folder in to_delete:
            try:
                # ফোল্ডারের সব ফাইল ডিলিট
                folder_files = [f for f in files if f.startswith(f"checkpoints/{folder}/")]
                for file_path in folder_files:
                    api.delete_file(
                        path_in_repo=file_path,
                        repo_id="ahashanahmed/csv",
                        repo_type="dataset",
                        commit_message=f"🗑️ Cleanup: Delete {folder}"
                    )
                print(f"   ✅ Deleted HF: {folder}")
            except Exception as e:
                print(f"   ⚠️ Failed to delete HF {folder}: {e}")
        
        print(f"✅ HF cleanup complete! Kept {len(to_keep)} checkpoint(s).")
        
    except Exception as e:
        print(f"⚠️ HF cleanup failed: {e}")

# =========================================================
# HF ডাউনলোড (কোনো ফাইল স্কিপ নয়)
# =========================================================

def download_from_hf():
    """HF Dataset থেকে সব ফাইল ডাউনলোড - rate-limit safe"""
    print("\n📥 Downloading from HF Dataset: ahashanahmed/csv...")
    print("   ✅ HF_TOKEN: " + ("YES" if os.getenv("hf_token") else "NO (rate limit may apply)"))
    print("   ✅ hf_transfer: " + ("YES" if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") else "NO (slower download)"))
    print("   ✅ max_workers: 2 (safe parallel)")
    print("   ✅ No files skipped - downloading everything")
    
    snapshot_download(
        repo_id="ahashanahmed/csv",
        repo_type="dataset",
        local_dir="./csv",
        max_workers=2,              # ✅ ২টি প্যারালাল ডাউনলোড (দ্রুত + নিরাপদ)
        local_dir_use_symlinks=False,
        token=os.getenv("hf_token"), # ✅ টোকেন = ৫,০০০ req/5min
        resume_download=True,        # ✅ আংশিক ডাউনলোড রিজিউম
        # ✅ কোনো ignore_patterns নেই = সব ফাইল ডাউনলোড হবে
    )
    print("✅ সম্পূর্ণ ডাউনলোড সম্পূর্ণ!")

# =========================================================
# লোকালে পুরনো চেকপয়েন্ট ক্লিনআপ (ডাউনলোডের পরে)
# =========================================================

def cleanup_old_checkpoints(keep_last=1):
    """লোকালে শুধু সর্বশেষ চেকপয়েন্ট রাখুন, বাকি ডিলিট"""
    
    checkpoint_dir = "./csv/llm_model"
    if not os.path.exists(checkpoint_dir):
        print("ℹ️ No llm_model directory found, skipping cleanup")
        return
    
    # সব চেকপয়েন্ট ফোল্ডার খুঁজুন
    import glob
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    
    if not checkpoints:
        print("ℹ️ No checkpoints found, skipping cleanup")
        return
    
    # স্টেপ নাম্বার অনুযায়ী সর্ট করুন
    def get_step_num(path):
        try:
            return int(path.split("-")[-1])
        except:
            return 0
    
    checkpoints = sorted(checkpoints, key=get_step_num)
    
    if len(checkpoints) <= keep_last:
        print(f"✅ Only {len(checkpoints)} checkpoints locally, no cleanup needed")
        return
    
    # পুরনো চেকপয়েন্ট ডিলিট করুন
    to_delete = checkpoints[:-keep_last]
    to_keep = checkpoints[-keep_last:]
    
    print(f"\n🗑️ Cleaning up old local checkpoints...")
    print(f"   Keeping: {[cp.split('/')[-1] for cp in to_keep]}")
    print(f"   Deleting {len(to_delete)} old checkpoints...")
    
    for checkpoint in to_delete:
        try:
            shutil.rmtree(checkpoint)
            print(f"   ✅ Deleted locally: {checkpoint.split('/')[-1]}")
        except Exception as e:
            print(f"   ⚠️ Failed to delete {checkpoint}: {e}")
    
    print(f"✅ Local cleanup complete! Kept {len(to_keep)} checkpoint(s).")

# =========================================================
# এক্সিকিউট
# =========================================================

if __name__ == "__main__":
    print("="*60)
    print("🚀 HF DOWNLOAD & CLEANUP SCRIPT")
    print("="*60)
    print(f"📅 {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Step 1: HF-তে পুরনো চেকপয়েন্ট ডিলিট (ডাউনলোডের আগে)
    cleanup_hf_checkpoints_before_download(keep_last=1)
    
    # Step 2: HF থেকে সম্পূর্ণ ডাউনলোড (কোনো ফাইল স্কিপ নয়)
    download_from_hf()
    
    # Step 3: লোকালে পুরনো চেকপয়েন্ট ডিলিট (শুধু সর্বশেষ রাখুন)
    cleanup_old_checkpoints(keep_last=1)
    
    print("\n" + "="*60)
    print("✅ hf_download.py সম্পূর্ণ!")
    print("="*60)