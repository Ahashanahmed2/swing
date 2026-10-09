# ================== scripts/deepseek_download.py (v2) ==================
# v2 পরিবর্তন:
#  1. adapter_latest → ./csv/llm_model_deepseek_lora (trainer এখান থেকেই চালিয়ে যায়)
#  2. পুরো final_model / checkpoints ডাউনলোড বন্ধ (কয়েক GB বাঁচে)
#  3. শুধু প্রতি মোডের সর্বশেষ checkpoint restore (অসমাপ্ত রান হলে)
#  4. _final ফোল্ডার বাদ, HF_TOKEN/hf_token দুটোই সাপোর্ট
#  5. 404/401 টাইট লুপ বন্ধ, সর্বোচ্চ retry সীমা, ব্যর্থ হলে exit code 1
#  6. verify এখন adapter যাচাই করে (merged মডেল ট্রেনিংয়ে লাগে না)

import os
import re
import sys
import time
import glob
import shutil
from datetime import datetime
from huggingface_hub import (
    snapshot_download,
    HfApi,
    login,
    CommitOperationDelete,
)

HF_REPO = "ahashanahmed/csv"
LOCAL_DIR = "./csv"

DEEPSEEK_HF_CHECKPOINT_PREFIX = "deepseek_checkpoints/deepseek_checkpoint-"
DEEPSEEK_LOCAL_CHECKPOINT_DIR = "./csv/deepseek_checkpoints"
LORA_ADAPTER_DIR = "./csv/llm_model_deepseek_lora"
HF_ADAPTER_LATEST = "final_model_deepseek/adapter_latest"

MAX_WORKERS = 2
MAX_ATTEMPTS = 30

CKPT_RE = re.compile(r"deepseek_checkpoint-(.+)-(\d+)$")


def get_token():
    return os.getenv("HF_TOKEN") or os.getenv("hf_token")


def list_hf_checkpoints(files):
    """{mode: {step: folder}} — _final বাদ।"""
    out = {}
    for f in files:
        if not f.startswith(DEEPSEEK_HF_CHECKPOINT_PREFIX):
            continue
        parts = f.split("/")
        if len(parts) < 3:
            continue
        m = CKPT_RE.match(parts[1])
        if not m:
            continue
        mode, step = m.group(1), int(m.group(2))
        if mode.endswith("_final"):
            continue
        out.setdefault(mode, {})[step] = parts[1]
    return out


# =========================================================
# Step 1: HF-তে পুরনো checkpoint ডিলিট (মোড-ভিত্তিক)
# =========================================================
def cleanup_deepseek_checkpoints_only(keep_last_per_mode=1):
    token = get_token()
    if not token:
        print("ℹ️ No HF token, skipping HF cleanup")
        return
    try:
        login(token=token)
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
        by_mode = list_hf_checkpoints(files)

        folders_to_delete = []
        for mode, steps in by_mode.items():
            old = sorted(steps)[:-keep_last_per_mode] if len(steps) > keep_last_per_mode else []
            for st in old:
                folders_to_delete.append(steps[st])
            if old:
                print(f"🗑️ '{mode}': {len(old)} পুরনো checkpoint মুছছি")

        if not folders_to_delete:
            print("✅ No old DeepSeek checkpoints to delete")
            return

        # ফোল্ডার-লেভেল ডিলিট = এক commit (ফাইল-বাই-ফাইল commit লাগে না)
        api.create_commit(
            repo_id=HF_REPO,
            repo_type="dataset",
            operations=[CommitOperationDelete(path_in_repo=f"deepseek_checkpoints/{fo}/")
                        for fo in folders_to_delete],
            commit_message=f"🗑️ DeepSeek cleanup: {len(folders_to_delete)} folder(s)",
        )
        print(f"   ✅ Deleted {len(folders_to_delete)} folder(s)")
    except Exception as e:
        print(f"⚠️ DeepSeek cleanup failed: {e}")


# =========================================================
# Step 2: মূল ডাউনলোড (ডেটা + tracking; মডেল/checkpoint বাদ)
# =========================================================
def download_from_hf_with_retry():
    print("\n" + "=" * 60 + "\n📥 DEEPSEEK DOWNLOAD\n" + "=" * 60)
    token = get_token()
    start = datetime.now()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        elapsed = (datetime.now() - start).total_seconds() / 60
        print(f"\n🔄 Attempt {attempt}/{MAX_ATTEMPTS} | ⏱️ {elapsed:.0f} min")
        try:
            snapshot_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                local_dir=LOCAL_DIR,
                max_workers=MAX_WORKERS,
                token=token,
                ignore_patterns=[
                    # বড় মডেল/checkpoint আলাদা ধাপে (শুধু দরকারটুকু) নামে
                    "deepseek_checkpoints/**",
                    "final_model_deepseek/**",
                    # অন্য মডেল
                    "checkpoints/**",
                    "ppo_models/**",
                    "patchtst_models/**",
                    "qwen_checkpoints/**",
                    "qwen3_checkpoints/**",
                    "*.tmp",
                    "*.log",
                ],
            )
            print(f"🎉 Download complete ({(datetime.now() - start).total_seconds() / 60:.0f} min)")
            return True
        except Exception as e:
            err = str(e)
            low = err.lower()
            if "401" in err or "403" in err:
                print(f"❌ Auth error (token/permission): {err[:200]}")
                return False
            if "404" in err:
                print(f"❌ Repo/file not found: {err[:200]}")
                return False
            if "429" in err or "rate limit" in low:
                wait = min(600, 300 + 60 * attempt)
                print(f"⚠️ Rate limited → wait {wait // 60} min")
                time.sleep(wait)
            elif "connection" in low or "timeout" in low:
                print("⚠️ Network error → retry in 60s")
                time.sleep(60)
            else:
                print(f"❌ Error: {err[:200]} → retry in 30s")
                time.sleep(30)

    print("❌ Max attempts reached")
    return False


# =========================================================
# Step 3: adapter_latest → trainer-এর adapter ডিরেক্টরি
# =========================================================
def restore_adapter_latest():
    print("\n" + "=" * 60 + "\n🔧 RESTORING adapter_latest\n" + "=" * 60)
    token = get_token()
    tmp = "./csv/_hf_adapter_tmp"
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            allow_patterns=[f"{HF_ADAPTER_LATEST}/*"],
            local_dir=tmp,
            token=token,
        )
        src = os.path.join(tmp, HF_ADAPTER_LATEST)
        if not os.path.exists(os.path.join(src, "adapter_config.json")):
            print("   ℹ️ adapter_latest নেই (প্রথম রান হলে স্বাভাবিক)")
            return False
        shutil.rmtree(LORA_ADAPTER_DIR, ignore_errors=True)
        os.makedirs(os.path.dirname(LORA_ADAPTER_DIR), exist_ok=True)
        shutil.move(src, LORA_ADAPTER_DIR)
        print(f"   ✅ Adapter restored → {LORA_ADAPTER_DIR}")
        return True
    except Exception as e:
        print(f"   ⚠️ Adapter restore failed: {str(e)[:200]}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =========================================================
# Step 4: অসমাপ্ত রানের checkpoint restore (মোড প্রতি শুধু সর্বশেষটা)
# HF: deepseek_checkpoints/deepseek_checkpoint-{mode}-{step}/
# Local: ./csv/deepseek_checkpoints/{mode}/checkpoint-{step}/
#
# ⚠️ trainer সফল হলে HF checkpoint মুছে ফেলে (delete_mode_checkpoints),
#    তাই HF-এ যা বাকি আছে সেটা সত্যিই অসমাপ্ত রান।
# =========================================================
def restore_checkpoints_to_local_structure():
    print("\n" + "=" * 60 + "\n🔧 RESTORING INTERRUPTED CHECKPOINTS\n" + "=" * 60)
    token = get_token()
    if not token:
        print("   ℹ️ No HF token, skipping")
        return
    try:
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
        by_mode = list_hf_checkpoints(files)
        if not by_mode:
            print("   ℹ️ No interrupted checkpoints on HF")
            return

        for mode, steps in by_mode.items():
            step = max(steps)
            hf_folder = steps[step]
            mode_dir = os.path.join(DEEPSEEK_LOCAL_CHECKPOINT_DIR, mode)
            local_target = os.path.join(mode_dir, f"checkpoint-{step}")

            if os.path.exists(os.path.join(local_target, "adapter_config.json")):
                print(f"   ✅ {mode}/checkpoint-{step} already local")
                continue

            print(f"   📥 {hf_folder}")
            tmp = f"./csv/_hf_ckpt_tmp_{mode}_{step}"
            try:
                snapshot_download(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    allow_patterns=f"deepseek_checkpoints/{hf_folder}/*",
                    local_dir=tmp,
                    token=token,
                )
                src = os.path.join(tmp, "deepseek_checkpoints", hf_folder)
                if not os.path.exists(src):
                    print(f"   ⚠️ Source not found: {src}")
                    continue
                shutil.rmtree(mode_dir, ignore_errors=True)  # পুরনো আধা-ভাঙা ফোল্ডার সাফ
                os.makedirs(mode_dir, exist_ok=True)
                shutil.move(src, local_target)
                with open(os.path.join(mode_dir, "IN_PROGRESS"), "w") as fp:
                    fp.write(datetime.now().isoformat())
                print(f"   ✅ Restored {mode}/checkpoint-{step}")
            except Exception as e:
                print(f"   ⚠️ Failed {mode}/{step}: {str(e)[:150]}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        print(f"⚠️ Restore failed: {e}")


# =========================================================
# Step 5: লোকাল checkpoint ক্লিনআপ
# =========================================================
def cleanup_old_deepseek_checkpoints(keep_last_per_mode=1):
    if not os.path.exists(DEEPSEEK_LOCAL_CHECKPOINT_DIR):
        return
    for mode_dir in glob.glob(os.path.join(DEEPSEEK_LOCAL_CHECKPOINT_DIR, "*")):
        if not os.path.isdir(mode_dir):
            continue
        ckpts = glob.glob(os.path.join(mode_dir, "checkpoint-*"))
        if len(ckpts) <= keep_last_per_mode:
            continue

        def step_of(path):
            try:
                return int(os.path.basename(path).split("-")[-1])
            except ValueError:
                return 0

        ckpts.sort(key=step_of)
        for c in ckpts[:-keep_last_per_mode]:
            shutil.rmtree(c, ignore_errors=True)
        print(f"   🗑️ {os.path.basename(mode_dir)}: {len(ckpts) - keep_last_per_mode} পুরনো মুছলাম")


# =========================================================
# Step 6: যাচাই
# =========================================================
def verify_state():
    print("\n" + "=" * 60 + "\n🔍 VERIFY\n" + "=" * 60)
    ok = True

    cfg = os.path.exists(os.path.join(LORA_ADAPTER_DIR, "adapter_config.json"))
    wts = any(os.path.exists(os.path.join(LORA_ADAPTER_DIR, w))
              for w in ("adapter_model.safetensors", "adapter_model.bin"))
    if cfg and wts:
        print(f"   ✅ Adapter ready: {LORA_ADAPTER_DIR} (incremental চলবে)")
    else:
        print("   ℹ️ Adapter নেই → trainer first_train হিসেবে শুরু করবে")
        if os.path.exists("./csv/trained_symbols_deepseek.json"):
            print("   ⚠️ কিন্তু trained_symbols আছে! adapter ছাড়া 'incremental' ফ্রেশ LoRA দিয়ে চলবে")
            ok = False

    for f in ("trained_symbols_deepseek.json", "batch_tracking_deepseek.json",
              "replay_buffer_deepseek.json"):
        path = os.path.join(LOCAL_DIR, f)
        print(f"   {'✅' if os.path.exists(path) else 'ℹ️ (নেই)'} {f}")
    return ok


if __name__ == "__main__":
    print("=" * 60 + "\n🚀 DEEPSEEK CLEANUP & DOWNLOAD (v2)\n" + "=" * 60)
    print(f"📅 {datetime.now():%Y-%m-%d %H:%M:%S} | 📂 {HF_REPO}")

    cleanup_deepseek_checkpoints_only(keep_last_per_mode=1)
    if not download_from_hf_with_retry():
        sys.exit(1)
    restore_adapter_latest()
    restore_checkpoints_to_local_structure()
    cleanup_old_deepseek_checkpoints(keep_last_per_mode=1)
    verify_state()

    print("\n✅ deepseek_download.py সম্পূর্ণ!")
