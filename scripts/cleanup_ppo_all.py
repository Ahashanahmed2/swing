# ================== cleanup_ppo_all.py ==================
# Delete ALL PPO models & checkpoints from HF & Local
# Run this before PPO retrain with new observation shape (89)

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, login, create_commit, CommitOperationDelete
import time
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================
HF_REPO = "ahashanahmed/csv"
LOCAL_DIR = "./csv"

# =========================================================
# STEP 1: DELETE FROM HUGGINGFACE
# =========================================================

def delete_ppo_from_hf():
    """Delete ALL PPO-related files from HF Dataset"""
    
    token = os.getenv("HF_TOKEN") or os.getenv("hf_token")
    if not token:
        print("❌ No HF_TOKEN found!")
        return False
    
    try:
        login(token=token)
        api = HfApi(token=token)
        
        print("🔍 Listing all files in HF repo...")
        files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
        print(f"   Total files: {len(files)}")
        
        # Collect ALL PPO-related files
        delete_files = []
        
        for f in files:
            # PPO per_symbol models
            if f.startswith("ppo_models/per_symbol/"):
                delete_files.append(f)
            # PPO ensemble
            elif f.startswith("ppo_models/ensemble/"):
                delete_files.append(f)
            # PPO checkpoints (all symbols in checkpoints/)
            elif f.startswith("checkpoints/") and (f.endswith(".zip") or f.endswith(".json")):
                # Skip LLM checkpoints (checkpoint-*)
                if not f.startswith("checkpoints/checkpoint-"):
                    delete_files.append(f)
            # PPO training progress/summary
            elif f in ["ppo_training_progress.json", "ppo_training_summary.json", "last_ppo_train.txt"]:
                delete_files.append(f)
            # PPO shared model
            elif f.startswith("ppo_models/ppo_shared"):
                delete_files.append(f)
        
        if not delete_files:
            print("✅ No PPO files found on HF!")
            return True
        
        print(f"\n🗑️ Found {len(delete_files)} PPO files to delete:")
        
        # Show summary by type
        per_symbol = [f for f in delete_files if f.startswith("ppo_models/per_symbol/")]
        checkpoint = [f for f in delete_files if f.startswith("checkpoints/")]
        ensemble = [f for f in delete_files if f.startswith("ppo_models/ensemble/")]
        other = [f for f in delete_files if f not in per_symbol + checkpoint + ensemble]
        
        print(f"   PPO Models: {len(per_symbol)}")
        print(f"   Checkpoints: {len(checkpoint)}")
        print(f"   Ensemble: {len(ensemble)}")
        print(f"   Other: {len(other)}")
        
        # Batch delete (100 files per commit)
        batch_size = 100
        total_batches = (len(delete_files) + batch_size - 1) // batch_size
        
        for i in range(0, len(delete_files), batch_size):
            batch = delete_files[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            operations = [CommitOperationDelete(path_in_repo=f) for f in batch]
            
            del_attempt = 0
            while del_attempt < 3:
                del_attempt += 1
                try:
                    create_commit(
                        repo_id=HF_REPO,
                        repo_type="dataset",
                        operations=operations,
                        commit_message=f"🗑️ PPO Cleanup batch {batch_num}/{total_batches}",
                        token=token
                    )
                    print(f"   ✅ Batch {batch_num}/{total_batches}: {len(batch)} files deleted")
                    break
                except Exception as e:
                    if "429" in str(e):
                        print(f"   ⚠️ Rate limited! Waiting 5min...")
                        time.sleep(300)
                    else:
                        print(f"   ❌ Batch {batch_num} failed: {str(e)[:100]}")
                        if del_attempt >= 3:
                            break
                        time.sleep(10)
            
            if batch_num < total_batches:
                time.sleep(2)  # Small delay between batches
        
        print(f"\n✅ HF Cleanup complete!")
        return True
        
    except Exception as e:
        print(f"❌ HF Cleanup failed: {e}")
        return False


# =========================================================
# STEP 2: DELETE FROM LOCAL
# =========================================================

def delete_ppo_local():
    """Delete ALL PPO-related files from local ./csv/"""
    
    print("\n🗑️ Cleaning local PPO files...")
    
    deleted_count = 0
    paths_to_delete = [
        Path("./csv/ppo_models/per_symbol"),
        Path("./csv/ppo_models/ensemble"),
        Path("./csv/ppo_checkpoints"),
        Path("./csv/ppo_models/ppo_shared.zip"),
        Path("./csv/ppo_training_progress.json"),
        Path("./csv/ppo_training_summary.json"),
        Path("./csv/last_ppo_train.txt"),
        Path("./csv/ppo_models/ppo_shared"),
    ]
    
    for path in paths_to_delete:
        if not path.exists():
            print(f"   ⏭️ {path} (not found)")
            continue
        
        try:
            if path.is_dir():
                # Count files before deleting
                file_count = len(list(path.rglob("*")))
                shutil.rmtree(path)
                print(f"   🗑️ {path} ({file_count} files)")
                deleted_count += file_count
            else:
                path.unlink()
                print(f"   🗑️ {path}")
                deleted_count += 1
        except Exception as e:
            print(f"   ❌ {path}: {e}")
    
    print(f"\n✅ Local Cleanup: {deleted_count} files deleted")
    return deleted_count


# =========================================================
# STEP 3: VERIFY
# =========================================================

def verify_cleanup():
    """Verify all PPO files are deleted"""
    
    print("\n🔍 Verifying cleanup...")
    
    # Check local
    local_checks = [
        Path("./csv/ppo_models/per_symbol"),
        Path("./csv/ppo_models/ensemble"),
        Path("./csv/ppo_checkpoints"),
        Path("./csv/ppo_models/ppo_shared.zip"),
        Path("./csv/last_ppo_train.txt"),
    ]
    
    local_remaining = []
    for p in local_checks:
        if p.exists():
            local_remaining.append(str(p))
    
    # Check HF
    token = os.getenv("HF_TOKEN") or os.getenv("hf_token")
    hf_remaining = []
    if token:
        try:
            api = HfApi(token=token)
            files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
            hf_remaining = [f for f in files if f.startswith("ppo_models/per_symbol/") or 
                          (f.startswith("checkpoints/") and not f.startswith("checkpoints/checkpoint-"))]
        except:
            pass
    
    print(f"\n📊 Verification Results:")
    print(f"   Local PPO files remaining: {len(local_remaining)}")
    if local_remaining:
        for f in local_remaining[:5]:
            print(f"      ⚠️ {f}")
    
    print(f"   HF PPO files remaining: {len(hf_remaining)}")
    if hf_remaining:
        for f in hf_remaining[:5]:
            print(f"      ⚠️ {f}")
    
    if not local_remaining and not hf_remaining:
        print(f"\n✅ ALL PPO FILES CLEANED!")
        print(f"   Ready for fresh PPO training with new observation shape (89)")
        return True
    else:
        print(f"\n⚠️ Some PPO files still remain. Run again.")
        return False


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🗑️ PPO COMPLETE CLEANUP")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"📂 HF Repo: {HF_REPO}")
    print(f"💾 Local: {LOCAL_DIR}")
    print("=" * 60)
    
    # Step 1: HF Delete
    print(f"\n{'='*50}")
    print(f"STEP 1: DELETE FROM HUGGINGFACE")
    print(f"{'='*50}")
    delete_ppo_from_hf()
    
    # Step 2: Local Delete
    print(f"\n{'='*50}")
    print(f"STEP 2: DELETE FROM LOCAL")
    print(f"{'='*50}")
    delete_ppo_local()
    
    # Step 3: Verify
    print(f"\n{'='*50}")
    print(f"STEP 3: VERIFY")
    print(f"{'='*50}")
    verify_cleanup()
    
    print(f"\n✅ cleanup_ppo_all.py complete!")