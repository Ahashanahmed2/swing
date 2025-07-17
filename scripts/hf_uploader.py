from huggingface_hub import login, upload_folder, snapshot_download, HfApi
import os
import subprocess
from dotenv import load_dotenv
import sys
load_dotenv()

HF_TOKEN = os.getenv("hf_token")
USERNAME = "ahashanahmed"
REPO_NAME = "csv"
REPO_ID = f"{USERNAME}/{REPO_NAME}"

def hf_login(token=None):
    if token:
        login(token=token)
def is_valide_diractory(local_dir:str)->bool:
    return os.path.isdir(local_dir) and len(os.listdir(local_dir))>0
    

def create_repo_if_not_exists(repo_id: str = REPO_ID, token: str = HF_TOKEN):
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type="model", token=token)
        print(f"ℹ️ Repo '{repo_id}' already exists.")
    except:
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, token=token)
        print(f"✅ Created repo: {repo_id}")

def upload_to_hf(folder_path: str = "./csv", repo_id: str = REPO_ID, token: str = HF_TOKEN):
    hf_login(token)
    create_repo_if_not_exists(repo_id, token)
    print(f"📤 Uploading folder: {folder_path} → {repo_id}")
    upload_folder(folder_path=folder_path, repo_id=repo_id, repo_type="model", token=token)
    print("✅ Upload complete.")

def download_from_hf_or_run_script(repo_id: str = REPO_ID, local_dir: str = "./csv", token: str = HF_TOKEN):
    create_repo_if_not_exists()
    # Step 1: লোকাল csv ফোল্ডার আছে কিনা চেক
    if is_valide_diractory(local_dir):
     return print("✅ লোকাল './csv' ফোল্ডার আগে থেকেই আছে। কিছু ডাউনলোড লাগবে না।")
   

    print("🔍 লোকাল './csv' ফোল্ডার নেই, HF থেকে ডাউনলোড চেষ্টা করছি...")

    # Step 2: HF থেকে ডাউনলোড
    try:
        hf_login(token)
        temp_path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=["csv/**"],
            token=token
        )
        hf_csv_path = os.path.join(temp_path, "csv")
        if os.path.exists(hf_csv_path):
            print("📥 HF-এ csv পাওয়া গেছে, লোকাল ./csv-এ কপি করছি...")
            import shutil
            shutil.copytree(hf_csv_path, local_dir)
            print("✅ ডাউনলোড ও কপি সম্পন্ন হয়েছে।")
        else:
            raise FileNotFoundError("❌ HF-এ 'csv' ফোল্ডার পাওয়া যায়নি।")

    except Exception as e:
        print(f"⚠️ HF থেকে ডাউনলোড ব্যর্থ: {e}")
        
     

# ⏯️ চালাতে চাইলে
if __name__ == "__main__":
    download_from_hf_or_run_script()
