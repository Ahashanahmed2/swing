# ================== scripts/deepseek_download.py ==================
# ✅ DeepSeek-R1-Distill-Qwen-1.5B Downloader
# ✅ Step 1: HF থেকে DeepSeek পুরনো চেকপয়েন্ট ডিলিট (PPO untouched)
# ✅ Step 2: PPO + old checkpoints বাদে সব ডাউনলোড
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

# ✅ DeepSeek-specific paths
DEEPSEEK_HF_CHECKPOINT_PREFIX = "deepseek_checkpoints/deepseek_checkpoint-"
DEEPSEEK_LOCAL_CHECKPOINT_DIR = "./csv/deepseek_checkpoints"
DEEPSEEK_MODEL_DIR = "./csv/llm_model_deepseek"

MAX_WORKERS = 2


# =========================================================
# Step 1: HF-তে DeepSeek পুরনো চেকপয়েন্ট ডিলিট
# =========================================================

def cleanup_deepseek_checkpoints_only(keep_last=1, max_files_per_commit=100, sleep_between_commits=120):
    """HF Dataset-এ শুধু DeepSeek পুরনো চেকপয়েন্ট ডিলিট, PPO untouched"""

    token = os.getenv("hf_token")
    if not token:
        print("ℹ️ No HF_TOKEN, skipping HF cleanup")
        return

    try:
        login(token=token)
        api = HfApi(token=token)

        print("\n🔍 Checking HF Dataset for old DeepSeek checkpoints...")

        files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
        all_delete_files = []

        # ✅ শুধু DeepSeek চেকপয়েন্ট
        deepseek_folders = set()
        for f in files:
            if f.startswith(DEEPSEEK_HF_CHECKPOINT_PREFIX):
                folder = f.split("/")[1]
                deepseek_folders.add(folder)

        if deepseek_folders:
            def get_deepseek_step(folder):
                try:
                    return int(folder.replace("deepseek_checkpoint-", ""))
                except:
                    return 0

            deepseek_list = sorted(deepseek_folders, key=get_deepseek_step)

            if len(deepseek_list) > keep_last:
                to_delete = deepseek_list[:-keep_last]
                print(f"\n🗑️ DeepSeek: Deleting {len(to_delete)} old checkpoints")
                for folder in to_delete:
                    folder_files = [
                        f for f in files
                        if f.startswith(f"deepseek_checkpoints/{folder}/")
                    ]
                    all_delete_files.extend(folder_files)

        if all_delete_files:
            total_files = len(all_delete_files)
            total_batches = (total_files + max_files_per_commit - 1) // max_files_per_commit

            print(f"\n🚀 Deleting {total_files} DeepSeek files in {total_batches} batches...")

            for i in range(0, total_files, max_files_per_commit):
                batch = all_delete_files[i:i + max_files_per_commit]
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
                            commit_message=f"🗑️ DeepSeek Cleanup batch {batch_num}/{total_batches}",
                            token=token
                        )
                        print(f"   ✅ Batch {batch_num}/{total_batches}: {len(batch)} deleted")
                        if batch_num < total_batches:
                            time.sleep(sleep_between_commits)
                        break
                    except Exception as e:
                        if "429" in str(e):
                            print(f"   ⚠️ Rate limited! Waiting 5min...")
                            time.sleep(300)
                        else:
                            print(f"   ❌ Batch {batch_num} failed: {str(e)[:100]}")
                            if del_attempt >= 3:
                                print(f"   ⚠️ Skipping batch {batch_num}")
                                break
                            time.sleep(30)
        else:
            print("✅ No old DeepSeek checkpoints to delete")

    except Exception as e:
        print(f"⚠️ DeepSeek cleanup failed: {e}")


# =========================================================
# Step 2: HF থেকে সব ডাউনলোড (PPO + old checkpoints বাদে)
# =========================================================

def download_from_hf_with_retry():
    """HF থেকে আনলিমিটেড রিট্রাই ডাউনলোড"""

    print("\n" + "=" * 60)
    print("📥 DEEPSEEK UNLIMITED DOWNLOAD MODE")
    print("=" * 60)
    print(f"   ✅ DeepSeek model + checkpoints included")
    print(f"   ✅ All CSV, XGBoost, sector data included")
    print(f"   ❌ PPO / PatchTST / old Qwen checkpoints excluded")
    print(f"   🔄 Will retry FOREVER until 100%")
    print("=" * 60)

    start_time = datetime.now()
    attempt = 0
    base_wait = 300

    while True:
        attempt += 1
        elapsed = (datetime.now() - start_time).total_seconds() / 60

        print(f"\n🔄 Attempt {attempt} | ⏰ {datetime.now().strftime('%H:%M:%S')} | ⏱️ {elapsed:.0f} min")
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
                    # ❌ PPO
                    "checkpoints/*/ppo_*",
                    "checkpoints/*/*_step*",
                    "checkpoints/*/*_ens*",
                    "ppo_models/**",
                    # ❌ PatchTST
                    "patchtst_models/**",
                    # ❌ Old GPT-2 checkpoints
                    "checkpoints/**",
                    # ❌ Old Qwen checkpoints
                    "qwen_checkpoints/**",
                    "qwen3_checkpoints/**",
                    # ❌ Temp/log
                    "*.tmp",
                    "*.log",
                ]
            )

            total_time = (datetime.now() - start_time).total_seconds() / 60

            print(f"\n{'=' * 60}")
            print(f"🎉 DEEPSEEK DOWNLOAD 100% COMPLETE!")
            print(f"{'=' * 60}")
            print(f"   Total attempts: {attempt}")
            print(f"   Total time: {total_time:.0f} minutes")
            print(f"{'=' * 60}")
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

                print(f"\n⚠️ RATE LIMITED (Attempt {attempt})")
                print(f"   ⏳ Waiting: {wait_time // 60} min")

                for remaining in range(wait_time, 0, -60):
                    print(f"      {remaining // 60} min remaining...")
                    time.sleep(60)

                print(f"\n🔄 Resuming download...")

            elif "404" in error_str:
                print(f"   ⚠️ Some files not found (404), continuing...")
                continue

            elif "connection" in error_str.lower() or "timeout" in error_str.lower():
                print(f"\n⚠️ Network error, retrying in 60s...")
                time.sleep(60)

            else:
                print(f"\n❌ Error: {error_str[:200]}")
                print(f"🔄 Retrying in 30s...")
                time.sleep(30)

    return False


# =========================================================
# Step 3: লোকাল DeepSeek চেকপয়েন্ট ক্লিনআপ
# =========================================================

def cleanup_old_deepseek_checkpoints(keep_last=1):
    """লোকালে শুধু সর্বশেষ DeepSeek চেকপয়েন্ট রাখুন"""
    checkpoint_dir = DEEPSEEK_LOCAL_CHECKPOINT_DIR

    if not os.path.exists(checkpoint_dir):
        print(f"   ℹ️ No local DeepSeek directory: {checkpoint_dir}")
        return

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    checkpoints += glob.glob(os.path.join(checkpoint_dir, "deepseek_checkpoint-*"))

    if not checkpoints:
        print("   ℹ️ No local DeepSeek checkpoints to clean")
        return

    def get_step_num(path):
        try:
            name = os.path.basename(path)
            return int(name.split("-")[-1])
        except:
            return 0

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

    print(f"   🗑️ Deleted {len(to_delete)} old DeepSeek checkpoint(s)")


# =========================================================
# BONUS: Verify DeepSeek Model
# =========================================================

def verify_deepseek_model():
    """DeepSeek model verify"""
    print("\n" + "=" * 60)
    print("🔍 VERIFYING DEEPSEEK MODEL")
    print("=" * 60)

    if not os.path.exists(DEEPSEEK_MODEL_DIR):
        print(f"   ❌ Directory not found: {DEEPSEEK_MODEL_DIR}")
        return False

    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    weight_file = None
    for wf in ["model.safetensors", "pytorch_model.bin"]:
        if os.path.exists(os.path.join(DEEPSEEK_MODEL_DIR, wf)):
            weight_file = wf
            break

    missing = []
    for f in required_files:
        if not os.path.exists(os.path.join(DEEPSEEK_MODEL_DIR, f)):
            missing.append(f)

    if weight_file is None:
        missing.append("model.safetensors/pytorch_model.bin")

    if missing:
        print(f"   ❌ Missing files: {missing}")
        return False

    print(f"   ✅ DeepSeek model ready at {DEEPSEEK_MODEL_DIR}")
    for f in sorted(os.listdir(DEEPSEEK_MODEL_DIR)):
        fp = os.path.join(DEEPSEEK_MODEL_DIR, f)
        if os.path.isfile(fp):
            size_mb = os.path.getsize(fp) / 1024 / 1024
            print(f"      • {f} ({size_mb:.1f} MB)")

    return True


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 DEEPSEEK CLEANUP & DOWNLOAD")
    print("=" * 60)
    print(f"📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Repo: {HF_REPO}")
    print(f"🧠 DeepSeek dir: {DEEPSEEK_MODEL_DIR}")
    print("=" * 60)

    cleanup_deepseek_checkpoints_only(keep_last=1)
    download_from_hf_with_retry()
    cleanup_old_deepseek_checkpoints(keep_last=1)
    verify_deepseek_model()

    print("\n" + "=" * 60)
    print("✅ deepseek_download.py সম্পূর্ণ!")
    print("=" * 60)
