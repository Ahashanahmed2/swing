# ================== scripts/qwen_download.py ==================
# FIXED VERSION - matches the fixed qwen_train.py
#
# HF layout expected:
#   final_model_qwen3/latest/      -> downloaded, then installed to ./csv/llm_model_qwen3/
#   qwen3_checkpoints/current/     -> downloaded ONLY if RESUME_FROM_HF_CHECKPOINT=1
#                                     (crash recovery), into ./csv/qwen3_checkpoints/current/
#
# Step 1: delete LEGACY Qwen3 folders on HF (old per-step checkpoints, old per-mode
#         final models). PPO / PatchTST / everything else untouched.
# Step 2: download everything except PPO/PatchTST checkpoints, Qwen3 checkpoints,
#         and the Qwen3 final-model folder (that one is fetched separately).
# Step 3: download final_model_qwen3/latest (+ optional crash checkpoint).
# Step 4: install latest model into ./csv/llm_model_qwen3, tidy local checkpoints, verify.
#
# Do NOT run this while qwen_train.py is running.

import os
import re
import glob
import time
import shutil
from datetime import datetime

from huggingface_hub import (
    snapshot_download,
    HfApi,
    login,
    CommitOperationDelete,
)

# =========================================================
# CONFIG
# =========================================================
HF_REPO = "ahashanahmed/csv"
LOCAL_DIR = "./csv"

QWEN3_HF_FINAL_DIR = "final_model_qwen3"
QWEN3_HF_FINAL_PATH = "final_model_qwen3/latest"
QWEN3_HF_CHECKPOINT_DIR = "qwen3_checkpoints"
QWEN3_HF_CHECKPOINT_PATH = "qwen3_checkpoints/current"

QWEN3_LOCAL_CHECKPOINT_DIR = "./csv/qwen3_checkpoints"
QWEN3_MODEL_DIR = "./csv/llm_model_qwen3"
FINAL_DOWNLOAD_DIR = os.path.join(LOCAL_DIR, QWEN3_HF_FINAL_PATH)

MAX_WORKERS = 2
CLEAN_LEGACY_ON_HF = True                                       # step 1 on/off
RESUME_FROM_HF_CHECKPOINT = os.getenv("RESUME_FROM_HF_CHECKPOINT", "0") == "1"

MAX_OTHER_ERROR_RETRIES = 5     # non-network, non-rate-limit errors
MAX_404_RETRIES = 3


def get_hf_token():
    return os.getenv("HF_TOKEN") or os.getenv("hf_token")


# =========================================================
# Step 1: delete LEGACY Qwen3 folders on HF
# =========================================================

def cleanup_legacy_qwen3_on_hf(max_ops_per_commit=50, sleep_between_commits=120):
    """
    Deletes (as whole folders, one op each):
      - qwen3_checkpoints/qwen3_checkpoint-N/    (old per-step layout)
      - final_model_qwen3/<mode>/ where <mode> != 'latest'   (old per-mode layout)
    Keeps qwen3_checkpoints/current/ and final_model_qwen3/latest/. PPO untouched.
    """
    token = get_hf_token()
    if not token:
        print("ℹ️ No HF token, skipping HF cleanup")
        return

    try:
        login(token=token)
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")

        folders = set()
        for f in files:
            parts = f.split("/")
            if len(parts) < 3:
                continue
            if parts[0] == QWEN3_HF_CHECKPOINT_DIR and parts[1].startswith("qwen3_checkpoint-"):
                folders.add(f"{parts[0]}/{parts[1]}/")
            elif parts[0] == QWEN3_HF_FINAL_DIR and parts[1] != "latest":
                folders.add(f"{parts[0]}/{parts[1]}/")

        if not folders:
            print("✅ No legacy Qwen3 folders to delete")
            return

        folders = sorted(folders)
        print(f"🗑️ Deleting {len(folders)} legacy Qwen3 folder(s) (PPO untouched):")
        for fo in folders:
            print(f"   • {fo}")

        total_batches = (len(folders) + max_ops_per_commit - 1) // max_ops_per_commit
        for i in range(0, len(folders), max_ops_per_commit):
            batch = folders[i:i + max_ops_per_commit]
            batch_num = i // max_ops_per_commit + 1
            ops = [CommitOperationDelete(path_in_repo=p) for p in batch]

            attempt = 0
            while True:
                attempt += 1
                try:
                    api.create_commit(
                        repo_id=HF_REPO, repo_type="dataset", operations=ops,
                        commit_message=f"🗑️ Qwen3 legacy cleanup {batch_num}/{total_batches}")
                    print(f"   ✅ Batch {batch_num}/{total_batches}: {len(batch)} folder(s) deleted")
                    if batch_num < total_batches:
                        time.sleep(sleep_between_commits)
                    break
                except Exception as e:
                    if "429" in str(e):
                        print("   ⚠️ Rate limited on delete! Waiting 5 min...")
                        time.sleep(300)
                    elif attempt >= 3:
                        print(f"   ⚠️ Skipping batch {batch_num}: {str(e)[:120]}")
                        break
                    else:
                        print(f"   ❌ Batch {batch_num} failed: {str(e)[:120]}")
                        time.sleep(30)
    except Exception as e:
        print(f"⚠️ Legacy cleanup failed: {e}")


# =========================================================
# Download helper (retry)
# =========================================================

def _status_code(e):
    resp = getattr(e, "response", None)
    return getattr(resp, "status_code", None)


def download_with_retry(label, **snapshot_kwargs):
    """
    Retries FOREVER on rate limit (429) and network errors.
    Stops on auth errors (401/403) and repo-not-found.
    Limited retries on 404 and unknown errors.
    """
    start = datetime.now()
    attempt = 0
    n404 = 0
    n_other = 0
    base_wait = 300

    while True:
        attempt += 1
        elapsed = (datetime.now() - start).total_seconds() / 60
        print(f"\n🔄 [{label}] Attempt {attempt} | {datetime.now():%H:%M:%S} | {elapsed:.0f} min elapsed")

        try:
            snapshot_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                local_dir=LOCAL_DIR,
                max_workers=MAX_WORKERS,
                token=get_hf_token(),
                **snapshot_kwargs,
            )
            print(f"✅ [{label}] download complete (attempts: {attempt}, "
                  f"{(datetime.now() - start).total_seconds() / 60:.0f} min)")
            return True

        except Exception as e:
            err = str(e)
            code = _status_code(e)
            low = err.lower()

            if code in (401, 403) or "401" in err[:200] or "403" in err[:200] \
                    or "repositorynotfound" in type(e).__name__.lower():
                print(f"❌ [{label}] Auth/permission/repo error - not retrying: {err[:200]}")
                return False

            if code == 429 or "429" in err or "rate limit" in low:
                wait = 600 if attempt > 15 else 480 if attempt > 10 else 420 if attempt > 5 else base_wait
                print(f"⚠️ [{label}] Rate limited. Waiting {wait // 60} min (downloaded files are safe)")
                for remaining in range(wait, 0, -60):
                    print(f"      {remaining // 60} min remaining...")
                    time.sleep(60)

            elif code == 404 or "404" in err[:200]:
                n404 += 1
                print(f"⚠️ [{label}] 404 ({n404}/{MAX_404_RETRIES}): {err[:150]}")
                if n404 >= MAX_404_RETRIES:
                    print(f"❌ [{label}] Giving up after repeated 404")
                    return False
                time.sleep(15)

            elif any(k in low for k in ("connection", "timeout", "timed out", "reset by peer")):
                print(f"⚠️ [{label}] Network error, retrying in 60s: {err[:120]}")
                time.sleep(60)

            else:
                n_other += 1
                print(f"❌ [{label}] Error ({n_other}/{MAX_OTHER_ERROR_RETRIES}): {err[:200]}")
                if n_other >= MAX_OTHER_ERROR_RETRIES:
                    print(f"❌ [{label}] Giving up")
                    return False
                time.sleep(30)


# =========================================================
# Step 2 & 3: downloads
# =========================================================

MAIN_IGNORE_PATTERNS = [
    # PPO checkpoints
    "checkpoints/*/ppo_*",
    "checkpoints/*/*_step*",
    "checkpoints/*/*_ens*",
    "ppo_models/**",
    # PatchTST
    "patchtst_models/**",
    # Old Qwen2.5 + ALL Qwen3 checkpoint/final folders (fetched explicitly below)
    "qwen_checkpoints/**",
    f"{QWEN3_HF_CHECKPOINT_DIR}/**",
    f"{QWEN3_HF_FINAL_DIR}/**",
    # temp / logs
    "*.tmp",
    "*.log",
]


def download_main():
    print("\n" + "=" * 60 + "\n📥 MAIN DOWNLOAD (data, xgboost, sector, ...)\n" + "=" * 60)
    return download_with_retry("main", ignore_patterns=MAIN_IGNORE_PATTERNS)


def download_final_model():
    print("\n" + "=" * 60 + f"\n📥 FINAL MODEL ({QWEN3_HF_FINAL_PATH})\n" + "=" * 60)
    return download_with_retry("final-model", allow_patterns=[f"{QWEN3_HF_FINAL_PATH}/**"])


def download_crash_checkpoint():
    print("\n" + "=" * 60 + f"\n📥 CRASH CHECKPOINT ({QWEN3_HF_CHECKPOINT_PATH})\n" + "=" * 60)
    return download_with_retry("checkpoint", allow_patterns=[f"{QWEN3_HF_CHECKPOINT_PATH}/**"])


# =========================================================
# Step 4: install model + local checkpoint tidy
# =========================================================

def install_latest_model():
    """Copy ./csv/final_model_qwen3/latest -> ./csv/llm_model_qwen3 (replace)."""
    if not os.path.isdir(FINAL_DOWNLOAD_DIR) or \
            not os.path.exists(os.path.join(FINAL_DOWNLOAD_DIR, "config.json")):
        print(f"ℹ️ No {QWEN3_HF_FINAL_PATH} on HF - trainer will start from the base model")
        return False

    tmp = QWEN3_MODEL_DIR + "_install_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(FINAL_DOWNLOAD_DIR, tmp)
    shutil.rmtree(QWEN3_MODEL_DIR, ignore_errors=True)
    os.replace(tmp, QWEN3_MODEL_DIR)
    shutil.rmtree(os.path.join(LOCAL_DIR, QWEN3_HF_FINAL_DIR), ignore_errors=True)  # save disk
    print(f"✅ Installed latest model → {QWEN3_MODEL_DIR}")
    return True


def tidy_local_checkpoints():
    """
    Resume validity (mode/symbols match) is decided by qwen_train.py via run_marker.json.
    Here we only: (a) drop checkpoints from the OLD layout without a marker,
    (b) keep just the newest checkpoint-N, (c) leave 'current' (HF crash copy) alone.
    If RESUME_FROM_HF_CHECKPOINT is off, remove 'current' too.
    """
    d = QWEN3_LOCAL_CHECKPOINT_DIR
    if not os.path.isdir(d):
        print(f"   ℹ️ No local checkpoint dir: {d}")
        return

    if not RESUME_FROM_HF_CHECKPOINT:
        cur = os.path.join(d, "current")
        if os.path.isdir(cur):
            shutil.rmtree(cur, ignore_errors=True)
            print("   🗑️ Removed stale 'current' checkpoint (resume from HF not requested)")

    legacy = glob.glob(os.path.join(d, "qwen3_checkpoint-*"))
    for p in legacy:
        shutil.rmtree(p, ignore_errors=True)
    if legacy:
        print(f"   🗑️ Removed {len(legacy)} legacy 'qwen3_checkpoint-N' folder(s)")

    def step(p):
        m = re.search(r"checkpoint-(\d+)$", p)
        return int(m.group(1)) if m else 0

    ckpts = sorted(glob.glob(os.path.join(d, "checkpoint-*")), key=step)
    if len(ckpts) > 1:
        for p in ckpts[:-1]:
            shutil.rmtree(p, ignore_errors=True)
        print(f"   🗑️ Kept newest checkpoint only, removed {len(ckpts) - 1}")
    else:
        print(f"   ✅ {len(ckpts)} local checkpoint(s), no cleanup needed")


# =========================================================
# Verify
# =========================================================

def verify_qwen3_model():
    print("\n" + "=" * 60 + "\n🔍 VERIFYING QWEN3 MODEL\n" + "=" * 60)

    if not os.path.isdir(QWEN3_MODEL_DIR):
        print(f"   ℹ️ {QWEN3_MODEL_DIR} not found (first run? trainer will use base model)")
        return False

    def has(name):
        return os.path.exists(os.path.join(QWEN3_MODEL_DIR, name))

    missing = []
    if not has("config.json"):
        missing.append("config.json")
    if not has("tokenizer_config.json"):
        missing.append("tokenizer_config.json")
    if not (has("tokenizer.json") or has("vocab.json")):
        missing.append("tokenizer.json / vocab.json")
    if not (has("model.safetensors") or has("model.safetensors.index.json")
            or has("pytorch_model.bin") or has("pytorch_model.bin.index.json")):
        missing.append("model weights")

    if missing:
        print(f"   ❌ Missing: {missing}")
        return False

    print(f"   ✅ Qwen3 model ready at {QWEN3_MODEL_DIR}")
    for f in sorted(os.listdir(QWEN3_MODEL_DIR)):
        fp = os.path.join(QWEN3_MODEL_DIR, f)
        if os.path.isfile(fp):
            print(f"      • {f} ({os.path.getsize(fp) / 1024 / 1024:.1f} MB)")

    ckpts = [p for p in glob.glob(os.path.join(QWEN3_LOCAL_CHECKPOINT_DIR, "*"))
             if os.path.isdir(p)]
    if ckpts:
        print(f"   📂 Local checkpoints: {[os.path.basename(c) for c in sorted(ckpts)]}")
    return True


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("=" * 60 + "\n🚀 QWEN3 CLEANUP & DOWNLOAD (PPO untouched)\n" + "=" * 60)
    print(f"📅 Start: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"📂 Repo: {HF_REPO} | 💾 Local: {LOCAL_DIR}")
    print(f"🧠 Model dir: {QWEN3_MODEL_DIR}")
    print(f"🔁 Resume from HF checkpoint: {RESUME_FROM_HF_CHECKPOINT}")
    print("=" * 60)

    if CLEAN_LEGACY_ON_HF:
        print("\nSTEP 1: LEGACY HF CLEANUP")
        cleanup_legacy_qwen3_on_hf()

    print("\nSTEP 2: MAIN DOWNLOAD")
    main_ok = download_main()

    print("\nSTEP 3: MODEL (+ CHECKPOINT) DOWNLOAD")
    model_ok = download_final_model()
    if RESUME_FROM_HF_CHECKPOINT:
        download_crash_checkpoint()

    print("\nSTEP 4: INSTALL + TIDY + VERIFY")
    if model_ok:
        install_latest_model()
    tidy_local_checkpoints()
    verify_qwen3_model()

    print("\n" + "=" * 60)
    print("✅ qwen_download.py সম্পূর্ণ!" if main_ok else "⚠️ qwen_download.py শেষ, তবে main ডাউনলোড সম্পূর্ণ হয়নি")
    print(f"📅 End: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)
