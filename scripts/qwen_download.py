# ================== scripts/qwen_download.py ==================
# ✅ Step 1: শুধু Qwen3 পুরনো চেকপয়েন্ট HF থেকে ডিলিট (PPO untouched)
# ✅ Step 2: PPO চেকপয়েন্ট বাদে সব ফাইল ডাউনলোড (Qwen3 model + data)
# ✅ Step 3: আনলিমিটেড রিট্রাই - ১০০% না হওয়া পর্যন্ত থামবে না

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

# ✅ Qwen3-specific paths
QWEN3_HF_CHECKPOINT_PREFIX = "qwen3_checkpoints/qwen3_checkpoint-"   # HF path
QWEN3_LOCAL_CHECKPOINT_DIR = "./csv/qwen3_checkpoints"                # Local checkpoint dir
QWEN3_MODEL_DIR = "./csv/llm_model_qwen3"                             # Local model dir

MAX_WORKERS = 2

# =========================================================
# Step 1: HF-তে শুধু Qwen3 পুরনো চেকপয়েন্ট ডিলিট (PPO untouched)
# =========================================================

def cleanup_qwen3_checkpoints_only(keep_last=1, max_files_per_commit=100, sleep_between_commits=120):
    """HF Dataset-এ শুধু Qwen3 পুরনো চেকপয়েন্ট ডিলিট, PPO untouched"""

    token = os.getenv("hf_token")
    if not token:
        print("ℹ️ No HF_TOKEN, skipping HF cleanup")
        return

    try:
        login(token=token)
        api = HfApi(token=token)

        print("\n🔍 Checking HF Dataset for old Qwen3 checkpoints...")

        files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
        all_delete_files = []

        # ✅ শুধু Qwen3 পুরনো চেকপয়েন্ট
        qwen3_folders = set()
        for f in files:
            if f.startswith(QWEN3_HF_CHECKPOINT_PREFIX):
                folder = f.split("/")[1]  # qwen3_checkpoint-N
                qwen3_folders.add(folder)

        if qwen3_folders:
            def get_qwen3_step(folder):
                try: return int(folder.replace("qwen3_checkpoint-", ""))
                except: return 0

            qwen3_list = sorted(qwen3_folders, key=get_qwen3_step)

            if len(qwen3_list) > keep_last:
                to_delete = qwen3_list[:-keep_last]
                print(f"\n🗑️ Qwen3: Deleting {len(to_delete)} old checkpoints (PPO untouched)")
                for folder in to_delete:
                    folder_files = [f for f in files if f.startswith(f"qwen3_checkpoints/{folder}/")]
                    all_delete_files.extend(folder_files)

        # ব্যাচ ডিলিট
        if all_delete_files:
            total_files = len(all_delete_files)
            total_batches = (total_files + max_files_per_commit - 1) // max_files_per_commit

            print(f"\n🚀 Deleting {total_files} Qwen3 files in {total_batches} batches...")

            for i in range(0, total_files, max_files_per_commit):
                batch = all_delete_files[i:i+max_files_per_commit]
                batch_num = i // max_files_per_commit + 1
                operations = [CommitOperationDelete(path_in_repo=f) for f in batch]

                del_attempt = 0
                while True:
                    del_attempt += 1
                    try:
                        create_commit(
                            repo_id=HF_REPO,
                            repo_type="dataset",
                            operations=operations,
                            commit_message=f"🗑️ Qwen3 Cleanup batch {batch_num}/{total_batches}",
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
            print("✅ No old Qwen3 checkpoints to delete")

    except Exception as e:
        print(f"⚠️ Qwen3 cleanup failed: {e}")


# =========================================================
# Step 2: HF থেকে PPO চেকপয়েন্ট বাদে সব ডাউনলোড (UNLIMITED RETRY)
# =========================================================

def download_from_hf_with_retry():
    """
    HF Dataset থেকে ডাউনলোড - আনলিমিটেড রিট্রাই
    - ৪২৯ এলে ৫ মিনিট অপেক্ষা
    - যতক্ষণ ১০০% না হবে, ততক্ষণ চলবে
    - ইতিমধ্যে ডাউনলোডেড ফাইল সুরক্ষিত
    - ✅ Qwen3 model + checkpoints INCLUDED
    - ❌ PPO checkpoints EXCLUDED
    """
    print("\n" + "=" * 60)
    print("📥 QWEN3 UNLIMITED DOWNLOAD MODE")
    print("=" * 60)
    print(f"   ✅ Qwen3 model (llm_model_qwen3/) included")
    print(f"   ✅ Qwen3 checkpoints (qwen3_checkpoints/) included")
    print(f"   ✅ All CSV, XGBoost, sector, data included")
    print(f"   ❌ PPO checkpoints excluded")
    print(f"   ❌ PatchTST checkpoints excluded")
    print(f"   🔄 Will retry FOREVER until 100%")
    print("=" * 60)

    start_time = datetime.now()
    attempt = 0
    base_wait = 300    while True:
        attempt += 1
        elapsed = (datetime.now() - start_time).total_seconds() / 60

        print(f"\n🔄 Attempt {attempt} | ⏰ {datetime.now().strftime('%H:%M:%S')} | ⏱️ {elapsed:.0f} min elapsed")
        print("-" * 50)

        try:
            snapshot_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                local_dir=LOCAL_DIR,
                max_workers=MAX_WORKERS,
                local_dir_use_symlinks=False,
                token=os.getenv("hf_token"),
                resume_download=True,
                tqdm_class=None,
                ignore_patterns=[
                    # ❌ PPO checkpoints বাদ
                    "checkpoints/*/ppo_*",
                    "checkpoints/*/*_step*",
                    "checkpoints/*/*_ens*",
                    "ppo_models/**",
                    # ❌ PatchTST checkpoints বাদ
                    "patchtst_models/**",
                    # ❌ Old Qwen2.5 checkpoints বাদ
                    "qwen_checkpoints/**",
                    # ❌ Temp / log ফাইল
                    "*.tmp",
                    "*.log",
                ]
            )

            total_time = (datetime.now() - start_time).total_seconds() / 60

            print(f"\n{'='*60}")
            print(f"🎉 QWEN3 DOWNLOAD 100% COMPLETE!")
            print(f"{'='*60}")
            print(f"   Total attempts: {attempt}")
            print(f"   Total time: {total_time:.0f} minutes")
            print(f"   Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            return True

        except Exception as e:
            error_str = str(e)

            if "429" in error_str or "rate limit" in error_str.lower():
                if attempt > 15:
                    wait_time = 600
                elif attempt > 10:
                    wait_time = 480
                elif attempt > 5:
                    wait_time = 420
                else:
                    wait_time = base_wait

                print(f"\n{'='*60}")
                print(f"⚠️ RATE LIMITED (Attempt {attempt})")
                print(f"{'='*60}")
                print(f"   ⏱️ Elapsed: {elapsed:.0f} min")
                print(f"   ⏳ Waiting: {wait_time//60} min {wait_time%60} sec")
                print(f"   🔄 Resume at: {datetime.fromtimestamp(datetime.now().timestamp() + wait_time).strftime('%H:%M:%S')}")
                print(f"   💾 Already downloaded files are SAFE")
                print(f"{'='*60}")

                for remaining in range(wait_time, 0, -60):
                    mins = remaining // 60
                    print(f"      {mins} min remaining...")
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
# Step 3: লোকালে পুরনো Qwen3 চেকপয়েন্ট ক্লিনআপ
# =========================================================

def cleanup_old_qwen3_checkpoints(keep_last=1):
    """লোকালে শুধু সর্বশেষ Qwen3 চেকপয়েন্ট রাখুন"""
    checkpoint_dir = QWEN3_LOCAL_CHECKPOINT_DIR

    if not os.path.exists(checkpoint_dir):
        print(f"   ℹ️ No local Qwen3 directory: {checkpoint_dir}")
        return

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    checkpoints += glob.glob(os.path.join(checkpoint_dir, "qwen3_checkpoint-*"))

    if not checkpoints:
        print("   ℹ️ No local Qwen3 checkpoints to clean")
        return

    def get_step_num(path):
        try:
            name = os.path.basename(path)
            return int(name.split("-")[-1])
        except: return 0

    checkpoints = sorted(checkpoints, key=get_step_num)

    if len(checkpoints) <= keep_last:
        print(f"   ✅ {len(checkpoints)} Qwen3 checkpoint(s), no cleanup needed")
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

    print(f"   🗑️ Local cleanup: deleted {len(to_delete)} old Qwen3 checkpoint(s), kept {keep_last}")


# =========================================================
# BONUS: Verify Qwen3 Model
# =========================================================

def verify_qwen3_model():
    """Qwen3 model সঠিকভাবে ডাউনলোড হয়েছে কিনা verify করুন"""
    print("\n" + "=" * 60)
    print("🔍 VERIFYING QWEN3 MODEL")
    print("=" * 60)

    if not os.path.exists(QWEN3_MODEL_DIR):
        print(f"   ❌ Qwen3 directory not found: {QWEN3_MODEL_DIR}")
        return False

    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    weight_file = None
    for wf in ["model.safetensors", "pytorch_model.bin"]:
        if os.path.exists(os.path.join(QWEN3_MODEL_DIR, wf)):
            weight_file = wf
            break

    missing = []
    for f in required_files:
        if not os.path.exists(os.path.join(QWEN3_MODEL_DIR, f)):
            missing.append(f)

    if weight_file is None:
        missing.append("model.safetensors/pytorch_model.bin")

    if missing:
        print(f"   ❌ Missing files: {missing}")
        return False

    print(f"   ✅ Qwen3 model ready at {QWEN3_MODEL_DIR}")
    print(f"   📄 Files:")
    for f in sorted(os.listdir(QWEN3_MODEL_DIR)):
        fp = os.path.join(QWEN3_MODEL_DIR, f)
        if os.path.isfile(fp):
            size_mb = os.path.getsize(fp) / 1024 / 1024
            print(f"      • {f} ({size_mb:.1f} MB)")

    ckpts = glob.glob(os.path.join(QWEN3_LOCAL_CHECKPOINT_DIR, "checkpoint-*"))
    ckpts += glob.glob(os.path.join(QWEN3_LOCAL_CHECKPOINT_DIR, "qwen3_checkpoint-*"))
    if ckpts:
        print(f"   📂 Qwen3 checkpoints: {len(ckpts)}")
        for c in sorted(ckpts):
            print(f"      • {os.path.basename(c)}")

    return True


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 QWEN3 CLEANUP & DOWNLOAD (PPO untouched)")
    print("=" * 60)
    print(f"📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Repo: {HF_REPO}")
    print(f"💾 Local: {LOCAL_DIR}")
    print(f"🧠 Qwen3 dir: {QWEN3_MODEL_DIR}")
    print(f"📁 Qwen3 checkpoints: {QWEN3_LOCAL_CHECKPOINT_DIR}")
    print(f"🔄 Mode: UNLIMITED RETRY until 100%")
    print("=" * 60)

    # Step 1: HF cleanup
    print(f"\n{'='*60}")
    print(f"STEP 1: HF QWEN3 CHECKPOINT CLEANUP")
    print(f"{'='*60}")
    cleanup_qwen3_checkpoints_only(
        keep_last=1,
        max_files_per_commit=100,
        sleep_between_commits=120
    )

    # Step 2: Download
    print(f"\n{'='*60}")
    print(f"STEP 2: QWEN3 UNLIMITED DOWNLOAD")
    print(f"{'='*60}")
    download_from_hf_with_retry()

    # Step 3: Local cleanup
    print(f"\n{'='*60}")
    print(f"STEP 3: LOCAL QWEN3 CLEANUP")
    print(f"{'='*60}")
    cleanup_old_qwen3_checkpoints(keep_last=1)

    # Step 4: Verify
    print(f"\n{'='*60}")
    print(f"STEP 4: VERIFY QWEN3 MODEL")
    print(f"{'='*60}")
    verify_qwen3_model()

    print("\n" + "=" * 60)
    print("✅ qwen_download.py সম্পূর্ণ!")
    print(f"📅 End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)