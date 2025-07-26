from huggingface_hub import login, upload_folder, snapshot_download, HfApi
import os
<<<<<<< HEAD
=======

>>>>>>> 02b7957 (last update 26-07-2025)
from dotenv import load_dotenv

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
    
    # লোকাল csv ফোল্ডার চেক
    if is_valide_diractory(local_dir):
        print("✅ লোকাল './csv' ফোল্ডার আগে থেকেই আছে। কিছু ডাউনলোড লাগবে না।")
        return

    print("🔍 লোকাল './csv' ফোল্ডার নেই, HF থেকে ডাউনলোড চেষ্টা করছি...")

    try:
        hf_login(token)
        temp_path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
<<<<<<< HEAD
            token=token
=======
            token=token,
>>>>>>> 02b7957 (last update 26-07-2025)
        )

        import shutil
        os.makedirs(local_dir, exist_ok=True)

        # temp_path এর মধ্যে থাকা সব .csv ফাইল ./csv ফোল্ডারে কপি করো
        for file_name in os.listdir(temp_path):
            if file_name.endswith(".csv"):
                full_src = os.path.join(temp_path, file_name)
                full_dst = os.path.join(local_dir, file_name)
                shutil.copy(full_src, full_dst)

        print("✅ HF থেকে .csv ফাইলগুলো ./csv ফোল্ডারে কপি করা হয়েছে।")

    except Exception as e:
        print(f"⚠️ HF থেকে ডাউনলোড ব্যর্থ: {e}")
        print("📉 MongoDB থেকে রিড করে লোকালি সেভ করার চেষ্টা করা হবে...")
<<<<<<< HEAD
       

     
=======
        

>>>>>>> 02b7957 (last update 26-07-2025)

# ⏯️ চালাতে চাইলে
if __name__ == "__main__":
    download_from_hf_or_run_script()
