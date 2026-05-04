# ================== scripts/ppo_download.py ==================
# ✅ Step 1: শুধু PPO পুরনো চেকপয়েন্ট HF থেকে ডিলিট (LLM untouched)
# ✅ Step 2: LLM চেকপয়েন্ট বাদে সব ফাইল ডাউনলোড
# ✅ আনলিমিটেড রিট্রাই সহ

import os
import time
import shutil
import glob
from datetime import datetime
from huggingface_hub import (
    snapshot_download, 
    HfApi, 
    login, 
    create_commit,
    CommitOperationDelete
)

# =========================================================
# CONFIG
# =========================================================

HF_REPO = "ahashanahmed/csv"
LOCAL_DIR = "./csv"

# =========================================================
# Step 1: HF-তে শুধু PPO পুরনো চেকপয়েন্ট ডিলিট (LLM untouched)
# =========================================================

def cleanup_ppo_checkpoints_only(keep_last=1, max_files_per_commit=100, sleep_between_commits=120):
    """HF Dataset-এ শুধু PPO পুরনো চেকপয়েন্ট ডিলিট, LLM untouched"""

    token = os.getenv("hf_token")
    if not token:
        print("ℹ️ No HF_TOKEN, skipping HF cleanup")
        return

    try:
        login(token=token)
        api = HfApi(token=token)

        print("\n🔍 Checking HF Dataset for old PPO checkpoints...")

        files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
        all_delete_files = []

        # শুধু PPO পুরনো চেকপয়েন্ট (checkpoints/{symbol}/*_step*.zip)
        ppo_folders = set()
        for f in files:
            if f.startswith("checkpoints/") and f.endswith(".zip"):
                parts = f.split("/")
                if len(parts) >= 3:
                    symbol = parts[1]
                    if not symbol.startswith("checkpoint-"):
                        ppo_folders.add(symbol)

        if ppo_folders:
            print(f"\n🗑️ PPO: Cleaning {len(ppo_folders)} symbols (LLM untouched)")
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

        # ব্যাচ ডিলিট (আনলিমিটেড রিট্রাই সহ)
        if all_delete_files:
            total_files = len(all_delete_files)
            total_batches = (total_files + max_files_per_commit - 1) // max_files_per_commit

            print(f"\n🚀 Deleting {total_files} PPO files in {total_batches} batches...")

            for i in range(0, total_files, max_files_per_commit):
                batch = all_delete_files[i:i+max_files_per_commit]
                batch_num = i // max_files_per_commit + 1
                operations = [CommitOperationDelete(path_in_repo=f) for f in batch]

                # আনলিমিটেড রিট্রাই
                del_attempt = 0
                while True:
                    del_attempt += 1
                    try:
                        create_commit(
                            repo_id=HF_REPO,
                            repo_type="dataset",
                            operations=operations,
                            commit_message=f"🗑️ PPO Cleanup batch {batch_num}/{total_batches}",
                            token=token
                        )
                        print(f"   ✅ Batch {batch_num}/{total_batches}: {len(batch)} deleted")
                        if batch_num < total_batches:
                            time.sleep(sleep_between_commits)
                        break
                    except Exception as e:
                        if "429" in str(e):
                            print(f"   ⚠️ Rate limited on delete! Waiting 5min...")
                            time.sleep(300)
                        else:
                            print(f"   ❌ Batch {batch_num} failed: {str(e)[:100]}")
                            if del_attempt >= 3:
                                print(f"   ⚠️ Skipping batch {batch_num} after {del_attempt} attempts")
                                break
                            time.sleep(30)
        else:
            print("✅ No old PPO checkpoints to delete")

    except Exception as e:
        print(f"⚠️ PPO cleanup failed: {e}")


# =========================================================
# Step 2: HF থেকে LLM চেকপয়েন্ট বাদে সব ডাউনলোড (UNLIMITED RETRY)
# =========================================================

def download_from_hf():
    """HF Dataset থেকে ডাউনলোড - যতক্ষণ ১০০% না হয়, ততক্ষণ চলবে"""
    print("\n" + "=" * 60)
    print("📥 PPO DOWNLOAD - UNLIMITED RETRY MODE")
    print("=" * 60)
    print(f"   ❌ LLM checkpoints excluded")
    print(f"   ✅ PPO latest checkpoints included")
    print(f"   ✅ All CSV, models, and other files included")
    print(f"   🔄 Will retry FOREVER until 100%")
    print("=" * 60)

    start_time = datetime.now()
    attempt = 0
    base_wait = 300  # 5 minutes

    while True:  # ← আনলিমিটেড লুপ
        attempt += 1
        
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        
        print(f"\n🔄 Attempt {attempt} | ⏰ {datetime.now().strftime('%H:%M:%S')} | ⏱️ {elapsed:.0f} min elapsed")
        print("-" * 50)

        try:
            snapshot_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                local_dir=LOCAL_DIR,
                max_workers=2,
                local_dir_use_symlinks=False,
                token=os.getenv("hf_token"),
                resume_download=True,
                tqdm_class=None,
                ignore_patterns=[
                    "checkpoints/checkpoint-*",
                    "patchtst_models/***",            #tst চেকপয়েন্ট বাদ
                ]
            )
            
            total_time = (datetime.now() - start_time).total_seconds() / 60
            
            print(f"\n{'='*60}")
            print(f"🎉 PPO DOWNLOAD 100% COMPLETE!")
            print(f"{'='*60}")
            print(f"   Total attempts: {attempt}")
            print(f"   Total time: {total_time:.0f} minutes")
            print(f"{'='*60}")
            return True
            
        except Exception as e:
            error_str = str(e)
            
            if "429" in error_str or "rate limit" in error_str.lower():
                # অ্যাডাপ্টিভ ওয়েট
                if attempt > 15:
                    wait_time = 600  # 10 min
                elif attempt > 10:
                    wait_time = 480  # 8 min
                elif attempt > 5:
                    wait_time = 420  # 7 min
                else:
                    wait_time = base_wait  # 5 min
                
                print(f"\n{'='*60}")
                print(f"⚠️ RATE LIMITED (Attempt {attempt})")
                print(f"{'='*60}")
                print(f"   ⏱️ Elapsed: {elapsed:.0f} min")
                print(f"   ⏳ Waiting: {wait_time//60} min {wait_time%60} sec")
                print(f"   🔄 Resume at: {datetime.fromtimestamp(datetime.now().timestamp() + wait_time).strftime('%H:%M:%S')}")
                print(f"   💾 Already downloaded files are SAFE")
                print(f"{'='*60}")
                
                for remaining in range(wait_time, 0, -60):
                    print(f"      {remaining//60} min remaining...")
                    time.sleep(60)
                
                print(f"\n🔄 Resuming download NOW...")
                
            elif "404" in error_str:
                print(f"   ⚠️ Some files not found (404), continuing...")
                continue
                
            elif "connection" in error_str.lower() or "timeout" in error_str.lower():
                print(f"\n⚠️ Network error")
                print(f"🔄 Retrying in 60 seconds...")
                time.sleep(60)
                
            else:
                print(f"\n❌ Error: {error_str[:200]}")
                print(f"🔄 Retrying in 30 seconds...")
                time.sleep(30)
    
    return False


# =========================================================
# Step 3: লোকালে পুরনো চেকপয়েন্ট ক্লিনআপ
# =========================================================

def cleanup_old_checkpoints(keep_last=1):
    """লোকালে শুধু সর্বশেষ চেকপয়েন্ট রাখুন"""
    checkpoint_dir = os.path.join(LOCAL_DIR, "llm_model")
    if not os.path.exists(checkpoint_dir):
        print("   ℹ️ No local llm_model directory")
        return

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not checkpoints:
        print("   ℹ️ No local checkpoints to clean")
        return

    def get_step_num(path):
        try: return int(path.split("-")[-1])
        except: return 0

    checkpoints = sorted(checkpoints, key=get_step_num)

    if len(checkpoints) <= keep_last:
        print(f"   ✅ {len(checkpoints)} checkpoint(s), no cleanup needed")
        return

    to_delete = checkpoints[:-keep_last]
    for checkpoint in to_delete:
        try:
            shutil.rmtree(checkpoint)
        except:
            try:
                os.remove(checkpoint)
            except:
                pass
    print(f"   🗑️ Local cleanup: deleted {len(to_delete)} old checkpoint(s), kept {keep_last}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 PPO CLEANUP & DOWNLOAD (LLM untouched)")
    print("=" * 60)
    print(f"📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Repo: {HF_REPO}")
    print(f"💾 Local: {LOCAL_DIR}")
    print(f"🔄 Mode: UNLIMITED RETRY until 100%")
    print("=" * 60)

    # ============================
    # Step 1: HF-তে PPO পুরনো চেকপয়েন্ট ডিলিট
    # ============================
    print(f"\n{'='*60}")
    print(f"STEP 1: HF PPO CHECKPOINT CLEANUP")
    print(f"{'='*60}")
    
    cleanup_ppo_checkpoints_only(
        keep_last=1,
        max_files_per_commit=100,
        sleep_between_commits=120
    )

    # ============================
    # Step 2: ডাউনলোড (আনলিমিটেড রিট্রাই)
    # ============================
    print(f"\n{'='*60}")
    print(f"STEP 2: UNLIMITED DOWNLOAD")
    print(f"{'='*60}")
    
    download_from_hf()

    # ============================
    # Step 3: লোকাল ক্লিনআপ
    # ============================
    print(f"\n{'='*60}")
    print(f"STEP 3: LOCAL CLEANUP")
    print(f"{'='*60}")
    
    cleanup_old_checkpoints(keep_last=1)

    # ============================
    # ফাইনাল
    # ============================
    print("\n" + "=" * 60)
    print("✅ ppo_download.py সম্পূর্ণ!")
    print(f"📅 End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
