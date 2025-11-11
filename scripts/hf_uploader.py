from huggingface_hub import login, upload_folder, snapshot_download, HfApi
import os
import shutil
import time
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("hf_token")
USERNAME = "ahashanahmed"
REPO_NAME = "csv"
REPO_ID = f"{USERNAME}/{REPO_NAME}"

# ✅ Login wrapper
def hf_login(token=None):
    if token:
        try:
            login(token=token)
            print("🔐 HF login সফল হয়েছে।")
        except Exception as e:
            print(f"❌ HF login ব্যর্থ: {e}")

# ✅ Local directory validator
def is_valid_directory(local_dir: str) -> bool:
    return os.path.isdir(local_dir) and len(os.listdir(local_dir)) > 0

# ✅ Repo creator with existence check
def create_repo_if_not_exists(repo_id: str = REPO_ID, token: str = HF_TOKEN):
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
        print(f"ℹ️ Repo '{repo_id}' আগে থেকেই আছে।")
    except Exception:
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, token=token)
            print(f"✅ নতুন Repo তৈরি হয়েছে: {repo_id}")
        except Exception as e:
            print(f"❌ Repo তৈরি ব্যর্থ: {e}")

# ✅ Upload with retry logic
def upload_to_hf(folder_path: str = "./csv", repo_id: str = REPO_ID, token: str = HF_TOKEN, retries: int = 3, delay: int = 5):
    hf_login(token)
    create_repo_if_not_exists(repo_id, token)

    if not is_valid_directory(folder_path):
        print(f"⚠️ আপলোডের জন্য ফোল্ডার খালি বা নেই: {folder_path}")
        return False

    print(f"📤 আপলোড শুরু: {folder_path} → {repo_id}")
    for attempt in range(1, retries + 1):
        try:
            upload_folder(folder_path=folder_path, repo_id=repo_id, repo_type="dataset", token=token)
            print("✅ HF আপলোড সফল হয়েছে।")
            return True
        except Exception as e:
            print(f"⏳ আপলোড চেষ্টা {attempt} ব্যর্থ: {e}")
            time.sleep(delay)

    print("❌ HF আপলোড সম্পূর্ণভাবে ব্যর্থ হয়েছে।")
    return False

# ✅ Download fallback with local check
def download_from_hf_or_run_script(repo_id: str = REPO_ID, local_dir: str = "./csv", token: str = HF_TOKEN):
    create_repo_if_not_exists(repo_id, token)

    if is_valid_directory(local_dir):
        print("✅ লোকাল './csv' ফোল্ডার আগে থেকেই আছে। ডাউনলোড প্রয়োজন নেই।")
        return True

    print("🔍 লোকাল './csv' ফোল্ডার নেই, HF থেকে ডাউনলোড চেষ্টা করছি...")

    try:
        hf_login(token)
        temp_path = snapshot_download(repo_id=repo_id, repo_type="dataset", token=token)

        def copy_contents(src_dir, dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
            for item in os.listdir(src_dir):
                s = os.path.join(src_dir, item)
                d = os.path.join(dst_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                elif os.path.isfile(s):
                    shutil.copy2(s, d)

        copy_contents(temp_path, local_dir)
        print("✅ HF থেকে ডাউনলোড সফল, './csv' ফোল্ডারে কপি সম্পন্ন।")
        return True

    except Exception as e:
        print(f"⚠️ HF থেকে ডাউনলোড ব্যর্থ: {e}")
        print("📉 MongoDB থেকে রিড করে লোকালি সেভ করার fallback চালু করা যেতে পারে...")
        return False