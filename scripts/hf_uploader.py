from huggingface_hub import login, upload_folder, snapshot_download, HfApi
import os
from dotenv import load_dotenv
import shutil
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
        api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
        print(f"ℹ️ Repo '{repo_id}' already exists.")
    except:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, token=token)
        print(f"✅ Created repo: {repo_id}")

def upload_to_hf(folder_path: str = "./csv", repo_id: str = REPO_ID, token: str = HF_TOKEN):
    hf_login(token)
    create_repo_if_not_exists(repo_id, token)
    print(f"📤 Uploading folder: {folder_path} → {repo_id}")
    upload_folder(folder_path=folder_path, repo_id=repo_id, repo_type="dataset", token=token)
    print("✅ Upload complete.")

def download_from_hf_or_run_script(repo_id: str = REPO_ID, local_dir: str = "./csv", token: str = HF_TOKEN):
    create_repo_if_not_exists()
    
    if is_valide_diractory(local_dir):
        print("✅ লোকাল './csv' ফোল্ডার আগে থেকেই আছে। কিছু ডাউনলোড লাগবে না।")
        return

    print("🔍 লোকাল './csv' ফোল্ডার নেই, HF থেকে ডাউনলোড চেষ্টা করছি...")

    try:
        hf_login(token)
        temp_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

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
        print("✅ HF থেকে .csv ও ফোল্ডার ./csv ফোল্ডারে কপি করা হয়েছে।")
        return True

    except Exception as e:
        print(f"⚠️ HF থেকে ডাউনলোড ব্যর্থ: {e}")
        print("📉 MongoDB থেকে রিড করে লোকালি সেভ করার চেষ্টা করা হবে...")