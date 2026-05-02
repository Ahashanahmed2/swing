# scripts/hf_download.py
# HF Dataset থেকে ডাউনলোড + LLM + PPO পুরনো চেকপয়েন্ট একক কমিটে ডিলিট
# ✅ Batch delete to avoid commit rate limit (128/hour)

import os
import shutil
from huggingface_hub import (
    snapshot_download, 
    HfApi, 
    login, 
    create_commit,
    CommitOperationDelete
)

# =========================================================
# HF Dataset-এ পুরনো চেকপয়েন্ট ডিলিট (LLM + PPO) - BATCH
# =========================================================

def cleanup_hf_checkpoints_before_download(keep_last=1):
    """HF Dataset-এ LLM + PPO উভয়ের পুরনো চেকপয়েন্ট একক কমিটে ডিলিট, শুধু শেষ ১টি রাখুন"""
    
    token = os.getenv("hf_token")
    if not token:
        print("ℹ️ No HF_TOKEN, skipping HF cleanup")
        return
    
    try:
        login(token=token)
        api = HfApi(token=token)
        
        print("\n🔍 Checking HF Dataset for old checkpoints (LLM + PPO)...")
        
        files = api.list_repo_files(repo_id="ahashanahmed/csv", repo_type="dataset")
        
        # ============================================================
        # সব ডিলিট অপারেশন একসাথে জমা করুন
        # ============================================================
        delete_operations = []
        
        # ============================================================
        # ১. LLM চেকপয়েন্ট ক্লিনআপ (checkpoints/checkpoint-*)
        # ============================================================
        llm_folders = set()
        for f in files:
            if f.startswith("checkpoints/checkpoint-"):
                folder = f.split("/")[1]
                llm_folders.add(folder)
        
        if llm_folders:
            def get_llm_step(folder):
                try:
                    return int(folder.replace("checkpoint-", ""))
                except:
                    return 0
            
            llm_list = sorted(llm_folders, key=get_llm_step)
            
            if len(llm_list) > keep_last:
                to_delete_llm = llm_list[:-keep_last]
                to_keep_llm = llm_list[-keep_last:]
                
                print(f"\n🗑️ LLM Checkpoints: Deleting {len(to_delete_llm)}, Keeping: {to_keep_llm}")
                
                for folder in to_delete_llm:
                    folder_files = [f for f in files if f.startswith(f"checkpoints/{folder}/")]
                    for file_path in folder_files:
                        delete_operations.append(CommitOperationDelete(path_in_repo=file_path))
        
        # ============================================================
        # ২. PPO চেকপয়েন্ট ক্লিনআপ (checkpoints/{symbol}/*.zip)
        # ============================================================
        ppo_folders = set()
        for f in files:
            if f.startswith("checkpoints/") and f.endswith(".zip"):
                parts = f.split("/")
                if len(parts) >= 3:
                    symbol = parts[1]
                    if not symbol.startswith("checkpoint-"):
                        ppo_folders.add(symbol)
        
        if ppo_folders:
            print(f"\n🗑️ PPO Checkpoints: Cleaning up {len(ppo_folders)} symbols...")
            
            for symbol in ppo_folders:
                try:
                    symbol_files = [
                        f for f in files 
                        if f.startswith(f"checkpoints/{symbol}/") and f.endswith(".zip")
                    ]
                    
                    def get_ppo_step(filepath):
                        try:
                            name = filepath.split("/")[-1].replace(".zip", "")
                            if "_step" in name:
                                return int(name.split("_step")[-1])
                            elif "latest" in name or "best" in name:
                                return 999999
                            return 0
                        except:
                            return 0
                    
                    symbol_files.sort(key=get_ppo_step)
                    
                    if len(symbol_files) > keep_last:
                        to_delete = symbol_files[:-keep_last]
                        to_keep = symbol_files[-keep_last:]
                        
                        for file_path in to_delete:
                            delete_operations.append(CommitOperationDelete(path_in_repo=file_path))
                        
                        print(f"   ✅ {symbol}: Deleted {len(to_delete)} old, Kept {[f.split('/')[-1] for f in to_keep]}")
                
                except Exception as e:
                    print(f"   ⚠️ PPO cleanup failed for {symbol}: {e}")
        
        # ============================================================
        # ৩. PPO প্রগ্রেস/সামারি JSON বাদে অন্যান্য JSON ডিলিট
        # ============================================================
        ppo_json_to_keep = ["ppo_training_progress.json", "ppo_training_summary.json"]
        for f in files:
            if f.startswith("checkpoints/") and f.endswith(".json"):
                name = f.split("/")[-1]
                if name not in ppo_json_to_keep and not f.startswith("checkpoints/checkpoint-"):
                    delete_operations.append(CommitOperationDelete(path_in_repo=f))
        
        # ============================================================
        # ✅ একক কমিটে (বা ব্যাচে) সব ডিলিট
        # ============================================================
        if delete_operations:
            print(f"\n🚀 Creating commits to delete {len(delete_operations)} files...")
            
            # ব্যাচে ভাগ করুন (প্রতি কমিটে সর্বোচ্চ ৫০০ ফাইল)
            batch_size = 500
            total_batches = (len(delete_operations) + batch_size - 1) // batch_size
            
            for i in range(0, len(delete_operations), batch_size):
                batch = delete_operations[i:i+batch_size]
                batch_num = i // batch_size + 1
                try:
                    create_commit(
                        repo_id="ahashanahmed/csv",
                        repo_type="dataset",
                        operations=batch,
                        commit_message=f"🗑️ Batch cleanup {batch_num}/{total_batches}: Delete {len(batch)} old checkpoints",
                        token=token
                    )
                    print(f"   ✅ Batch {batch_num}/{total_batches}: Deleted {len(batch)} files")
                except Exception as e:
                    print(f"   ⚠️ Batch {batch_num} failed: {str(e)[:100]}")
        else:
            print("✅ No old checkpoints to delete")
        
        print(f"\n✅ HF Cleanup Complete!")
        
    except Exception as e:
        print(f"⚠️ HF cleanup failed: {e}")

# =========================================================
# HF ডাউনলোড (চেকপয়েন্ট বাদে)
# =========================================================

def download_from_hf():
    """HF Dataset থেকে সব ফাইল ডাউনলোড - checkpoints/ বাদে (রেট লিমিট এড়াতে)"""
    print("\n📥 Downloading from HF Dataset: ahashanahmed/csv...")
    print("   ⚠️ Skipping checkpoints/ to avoid rate limit")
    print("   ✅ All other files will be downloaded")
    
    snapshot_download(
        repo_id="ahashanahmed/csv",
        repo_type="dataset",
        local_dir="./csv",
        max_workers=2,
        local_dir_use_symlinks=False,
        token=os.getenv("hf_token"),
        resume_download=True,
        ignore_patterns=[
            "checkpoints/**",      # ✅ LLM + PPO সব চেকপয়েন্ট স্কিপ
        ],
    )
    print("✅ Download complete! (checkpoints excluded)")

# =========================================================
# লোকালে পুরনো চেকপয়েন্ট ক্লিনআপ (ডাউনলোডের পরে)
# =========================================================

def cleanup_old_checkpoints(keep_last=1):
    """লোকালে শুধু সর্বশেষ চেকপয়েন্ট রাখুন, বাকি ডিলিট"""
    
    checkpoint_dir = "./csv/llm_model"
    if not os.path.exists(checkpoint_dir):
        print("ℹ️ No llm_model directory found, skipping cleanup")
        return
    
    import glob
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    
    if not checkpoints:
        print("ℹ️ No checkpoints found, skipping cleanup")
        return
    
    def get_step_num(path):
        try:
            return int(path.split("-")[-1])
        except:
            return 0
    
    checkpoints = sorted(checkpoints, key=get_step_num)
    
    if len(checkpoints) <= keep_last:
        print(f"✅ Only {len(checkpoints)} checkpoints locally, no cleanup needed")
        return
    
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
    print("🚀 HF DOWNLOAD & CLEANUP SCRIPT (LLM + PPO - Batch Delete)")
    print("="*60)
    
    # Step 1: HF-তে LLM + PPO পুরনো চেকপয়েন্ট একক কমিটে ডিলিট (শেষ ১টি রাখুন)
    cleanup_hf_checkpoints_before_download(keep_last=1)
    
    # Step 2: HF থেকে বাকি সব ফাইল ডাউনলোড (চেকপয়েন্ট বাদে)
    download_from_hf()
    
    # Step 3: লোকালে পুরনো চেকপয়েন্ট ডিলিট (শুধু সর্বশেষ রাখুন)
    cleanup_old_checkpoints(keep_last=1)
    
    print("\n" + "="*60)
    print("✅ hf_download.py সম্পূর্ণ!")
    print("="*60)