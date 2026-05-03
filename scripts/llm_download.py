# scripts/llm_download.py
# ✅ Step 1: শুধু LLM পুরনো চেকপয়েন্ট HF থেকে ডিলিট (PPO untouched)
# ✅ Step 2: PPO চেকপয়েন্ট বাদে সব ফাইল ডাউনলোড

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
# Step 1: HF-তে শুধু LLM পুরনো চেকপয়েন্ট ডিলিট (PPO untouched)
# =========================================================

def cleanup_llm_checkpoints_only(keep_last=1, max_files_per_commit=100, sleep_between_commits=120):
    """HF Dataset-এ শুধু LLM পুরনো চেকপয়েন্ট ডিলিট, PPO untouched"""
    
    token = os.getenv("hf_token")
    if not token:
        print("ℹ️ No HF_TOKEN, skipping HF cleanup")
        return
    
    try:
        login(token=token)
        api = HfApi(token=token)
        
        print("\n🔍 Checking HF Dataset for old LLM checkpoints...")
        
        files = api.list_repo_files(repo_id="ahashanahmed/csv", repo_type="dataset")
        all_delete_files = []
        
        # শুধু LLM পুরনো চেকপয়েন্ট (checkpoints/checkpoint-*)
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
                print(f"\n🗑️ LLM: Deleting {len(to_delete)} old checkpoints (PPO untouched)")
                for folder in to_delete:
                    folder_files = [f for f in files if f.startswith(f"checkpoints/{folder}/")]
                    all_delete_files.extend(folder_files)
        
        # ব্যাচ ডিলিট
        if all_delete_files:
            total_files = len(all_delete_files)
            total_batches = (total_files + max_files_per_commit - 1) // max_files_per_commit
            
            print(f"\n🚀 Deleting {total_files} LLM files in {total_batches} batches...")
            
            for i in range(0, total_files, max_files_per_commit):
                batch = all_delete_files[i:i+max_files_per_commit]
                batch_num = i // max_files_per_commit + 1
                operations = [CommitOperationDelete(path_in_repo=f) for f in batch]
                
                try:
                    create_commit(
                        repo_id="ahashanahmed/csv",
                        repo_type="dataset",
                        operations=operations,
                        commit_message=f"🗑️ LLM Cleanup batch {batch_num}/{total_batches}",
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
                                commit_message=f"🗑️ LLM Cleanup batch {batch_num}/{total_batches} (retry)",
                                token=token
                            )
                            print(f"   ✅ Batch {batch_num}: retry success")
                        except:
                            print(f"   ❌ Batch {batch_num} failed")
        else:
            print("✅ No old LLM checkpoints to delete")
        
    except Exception as e:
        print(f"⚠️ LLM cleanup failed: {e}")

# =========================================================
# Step 2: HF থেকে PPO চেকপয়েন্ট বাদে সব ডাউনলোড (Auto-retry)
# =========================================================

def download_from_hf_with_retry():
    """HF Dataset থেকে ডাউনলোড (প্রতি ৪২৯ এ ৫ মিনিট অপেক্ষা, আনলিমিটেড রিজিউম)"""
    print("\n📥 Downloading from HF Dataset...")
    print("   ✅ LLM checkpoints included")
    print("   ❌ PPO checkpoints excluded")
    print("   ✅ All CSV, models, and other files included")
    
    attempt = 0
    while True:
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
                    "checkpoints/*/ppo_*",
                    "checkpoints/*/*_step*",
                    "checkpoints/*/*_ens*",
                ]
            )
            print(f"✅ Download complete! (Total attempts: {attempt})")
            return
        except Exception as e:
            if "429" in str(e):
                wait_time = 300
                print(f"\n⚠️ Rate limited! (Attempt {attempt})")
                print(f"⏳ Waiting {wait_time//60} minutes for rate limit reset...")
                time.sleep(wait_time)
                print(f"🔄 Resuming download...")
            else:
                print(f"❌ Download failed: {str(e)[:200]}")
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
    print("🚀 LLM CLEANUP & DOWNLOAD (PPO untouched)")
    print("="*60)
    
    # Step 1: শুধু LLM পুরনো চেকপয়েন্ট ডিলিট (PPO untouched)
    cleanup_llm_checkpoints_only(
        keep_last=1,
        max_files_per_commit=100,
        sleep_between_commits=120
    )

    # Step 2: LLM ফাইল ডাউনলোড (PPO চেকপয়েন্ট বাদে)
    download_from_hf_with_retry()
    
    # Step 3: লোকাল ক্লিনআপ
    cleanup_old_checkpoints(keep_last=1)
    
    print("\n" + "="*60)
    print("✅ llm_download.py সম্পূর্ণ!")
    print("="*60)