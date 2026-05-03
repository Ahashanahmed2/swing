# scripts/ppo_download.py
# ✅ Step 1: LLM + PPO পুরনো চেকপয়েন্ট HF থেকে ডিলিট (শুধু latest থাকবে)
# ✅ Step 2: LLM চেকপয়েন্ট বাদে সব ফাইল ডাউনলোড (Auto-retry সহ)

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
# Step 1: HF-তে LLM + PPO পুরনো চেকপয়েন্ট ডিলিট (latest রাখবে)
# =========================================================

def cleanup_hf_checkpoints_before_download(keep_last=1, max_files_per_commit=100, sleep_between_commits=120):
    """HF Dataset-এ LLM + PPO পুরনো চেকপয়েন্ট ডিলিট, শুধু latest রাখবে"""
    
    token = os.getenv("hf_token")
    if not token:
        print("ℹ️ No HF_TOKEN, skipping HF cleanup")
        return
    
    try:
        login(token=token)
        api = HfApi(token=token)
        
        print("\n🔍 Checking HF Dataset for old checkpoints (LLM + PPO)...")
        
        files = api.list_repo_files(repo_id="ahashanahmed/csv", repo_type="dataset")
        all_delete_files = []
        
        # ১. LLM পুরনো চেকপয়েন্ট (checkpoints/checkpoint-*)
        llm_folders = set()
        for f in files:
            if f.startswith("checkpoints/checkpoint-"):
                folder = f.split("/")[1]
                llm_folders.add(folder)
        
        if llm_folders:
            def get_llm_step(folder):
                try: return int(folder.replace("checkpoint-", ""))
                except: return 0
            
            llm_list = sorted(llm_folders, key=get_llm_step)
            
            if len(llm_list) > keep_last:
                to_delete = llm_list[:-keep_last]
                print(f"\n🗑️ LLM: Deleting {len(to_delete)} old checkpoints")
                for folder in to_delete:
                    folder_files = [f for f in files if f.startswith(f"checkpoints/{folder}/")]
                    all_delete_files.extend(folder_files)
        
        # ২. PPO পুরনো চেকপয়েন্ট (checkpoints/{symbol}/*_step*.zip)
        ppo_folders = set()
        for f in files:
            if f.startswith("checkpoints/") and f.endswith(".zip"):
                parts = f.split("/")
                if len(parts) >= 3:
                    symbol = parts[1]
                    if not symbol.startswith("checkpoint-"):
                        ppo_folders.add(symbol)
        
        if ppo_folders:
            print(f"\n🗑️ PPO: Cleaning {len(ppo_folders)} symbols")
            for symbol in ppo_folders:
                symbol_files = [f for f in files if f.startswith(f"checkpoints/{symbol}/") and f.endswith(".zip")]
                
                def get_ppo_step(filepath):
                    try:
                        name = filepath.split("/")[-1].replace(".zip", "")
                        if "_step" in name: return int(name.split("_step")[-1])
                        elif "latest" in name or "best" in name: return 999999
                        return 0
                    except: return 0
                
                symbol_files.sort(key=get_ppo_step)
                
                if len(symbol_files) > keep_last:
                    to_delete = symbol_files[:-keep_last]
                    all_delete_files.extend(to_delete)
                    print(f"   {symbol}: {len(to_delete)} old → delete")
        
        # ব্যাচ ডিলিট
        if all_delete_files:
            total_files = len(all_delete_files)
            total_batches = (total_files + max_files_per_commit - 1) // max_files_per_commit
            
            print(f"\n🚀 Deleting {total_files} files in {total_batches} batches...")
            
            for i in range(0, total_files, max_files_per_commit):
                batch = all_delete_files[i:i+max_files_per_commit]
                batch_num = i // max_files_per_commit + 1
                operations = [CommitOperationDelete(path_in_repo=f) for f in batch]
                
                try:
                    create_commit(
                        repo_id="ahashanahmed/csv",
                        repo_type="dataset",
                        operations=operations,
                        commit_message=f"🗑️ Cleanup batch {batch_num}/{total_batches}",
                        token=token
                    )
                    print(f"   ✅ Batch {batch_num}/{total_batches}: {len(batch)} deleted")
                    if batch_num < total_batches:
                        time.sleep(sleep_between_commits)
                except Exception as e:
                    if "429" in str(e):
                        print(f"   ⚠️ Rate limited! Waiting 5min...")
                        time.sleep(300)
                        try:
                            create_commit(
                                repo_id="ahashanahmed/csv",
                                repo_type="dataset",
                                operations=operations,
                                commit_message=f"🗑️ Cleanup batch {batch_num}/{total_batches} (retry)",
                                token=token
                            )
                            print(f"   ✅ Batch {batch_num}: retry success")
                        except:
                            print(f"   ❌ Batch {batch_num} failed")
        else:
            print("✅ No old checkpoints to delete")
        
    except Exception as e:
        print(f"⚠️ HF cleanup failed: {e}")

# =========================================================
# Step 2: HF থেকে LLM চেকপয়েন্ট বাদে সব ডাউনলোড (Auto-retry)
# =========================================================

def download_from_hf():
    """HF Dataset থেকে ডাউনলোড (resume + auto-retry on rate limit)"""
    print("\n📥 Downloading from HF Dataset...")
    print("   ❌ LLM checkpoints excluded")
    print("   ✅ PPO latest checkpoints included")
    print("   ✅ All CSV, models, and other files included")
    
    max_retries = 5  # ৫ বার পর্যন্ত চেষ্টা করবে
    for attempt in range(1, max_retries + 1):
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
                ignore_patterns=["checkpoints/checkpoint-*"],
            )
            print("✅ Download complete!")
            return  # সফল হলে ফাংশন থেকে বেরিয়ে যাবে
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                wait_time = 900  # ১৫ মিনিট
                print(f"\n⚠️ Rate limited! (Attempt {attempt}/{max_retries})")
                print(f"⏳ Waiting {wait_time//60} minutes for rate limit reset...")
                print(f"📊 Already downloaded files will NOT be re-downloaded (resume mode)")
                time.sleep(wait_time)
                print(f"🔄 Retrying download...")
            else:
                print(f"❌ Download failed: {str(e)[:300]}")
                raise

# =========================================================
# Step 3: লোকালে পুরনো চেকপয়েন্ট ক্লিনআপ
# =========================================================

def cleanup_old_checkpoints(keep_last=1):
    """লোকালে শুধু সর্বশেষ চেকপয়েন্ট রাখুন"""
    checkpoint_dir = "./csv/llm_model"
    if not os.path.exists(checkpoint_dir):
        return
    
    import glob
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not checkpoints:
        return
    
    def get_step_num(path):
        try: return int(path.split("-")[-1])
        except: return 0
    
    checkpoints = sorted(checkpoints, key=get_step_num)
    
    if len(checkpoints) <= keep_last:
        return
    
    to_delete = checkpoints[:-keep_last]
    for checkpoint in to_delete:
        try:
            shutil.rmtree(checkpoint)
        except:
            pass
    print(f"✅ Local cleanup: kept {keep_last} checkpoint(s)")

# =========================================================
# এক্সিকিউট
# =========================================================

if __name__ == "__main__":
    print("="*60)
    print("🚀 PPO DOWNLOAD & CLEANUP (Resume + Auto-retry)")
    print("="*60)
    
    # Step 1: HF-তে LLM + PPO পুরনো চেকপয়েন্ট ডিলিট
    # ⚠️ আগের রানে ক্লিনআপ শেষ হয়ে গেলে স্কিপ করতে পারেন
    # cleanup_hf_checkpoints_before_download(
    #     keep_last=1,
    #     max_files_per_commit=100,
    #     sleep_between_commits=120
    # )

    # Step 2: HF থেকে সব ফাইল ডাউনলোড (LLM চেকপয়েন্ট বাদে, auto-retry সহ)
    download_from_hf()
    
    # Step 3: লোকালে পুরনো চেকপয়েন্ট ডিলিট
    cleanup_old_checkpoints(keep_last=1)
    
    print("\n" + "="*60)
    print("✅ ppo_download.py সম্পূর্ণ!")
    print("="*60)
