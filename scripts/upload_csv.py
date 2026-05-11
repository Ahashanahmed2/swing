# upload_csv.py - GitHub Actions-এর জন্য সঠিক সংস্করণ (Unlimited Retry)
import os
import time
from huggingface_hub import HfApi, login
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
HF_TOKEN = os.getenv("hf_token")
REPO_ID = "ahashanahmed/csv"

print("="*60)
print("🚀 UPLOADING TO HUGGING FACE (UNLIMITED RETRY)")
print("="*60)

# Check token
if not HF_TOKEN:
    print("❌ HF_TOKEN not found in environment variables!")
    print("   Please set HF_TOKEN in GitHub Secrets or .env file")
    exit(1)

# =========================================================
# Login with UNLIMITED retry (rate limit handle)
# =========================================================
print("\n🔑 Logging in...")
attempt = 0
while True:
    attempt += 1
    try:
        login(token=HF_TOKEN)
        print(f"✅ Logged in to Hugging Face (Attempt: {attempt})")
        break
    except Exception as e:
        if "429" in str(e):
            wait_time = 300  # 5 minutes
            print(f"⚠️ Rate limited! (Attempt {attempt})")
            print(f"⏳ Waiting {wait_time//60} minutes for rate limit reset...")
            time.sleep(wait_time)
            print(f"🔄 Retrying login...")
        else:
            print(f"❌ Login failed: {e}")
            exit(1)

api = HfApi()

local_folder = "./csv"

# Check folder exists
if not os.path.exists(local_folder):
    print(f"❌ Folder not found: {local_folder}")
    exit(1)

# =========================================================
# Check/Create repo with UNLIMITED retry
# =========================================================
print("\n📁 Checking repository...")
attempt = 0
while True:
    attempt += 1
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="dataset")
        print(f"✅ Repository exists: {REPO_ID} (Attempt: {attempt})")
        break
    except Exception as e:
        if "429" in str(e):
            wait_time = 300
            print(f"⚠️ Rate limited! (Attempt {attempt})")
            print(f"⏳ Waiting {wait_time//60} minutes...")
            time.sleep(wait_time)
            print(f"🔄 Retrying repo check...")
        elif "404" in str(e) or "not found" in str(e).lower():
            print(f"📁 Repository not found. Creating...")
            create_attempt = 0
            while True:
                create_attempt += 1
                try:
                    api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=False)
                    print(f"✅ Created repository: {REPO_ID} (Attempt: {create_attempt})")
                    break
                except Exception as create_e:
                    if "429" in str(create_e):
                        wait_time = 300
                        print(f"⚠️ Rate limited on create! (Attempt {create_attempt})")
                        print(f"⏳ Waiting {wait_time//60} minutes...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Failed to create repo: {create_e}")
                        exit(1)
            break
        else:
            print(f"❌ Repo check failed: {e}")
            exit(1)

# =========================================================
# Files to upload
# =========================================================
files_to_upload = []
total_size = 0

for root, dirs, files in os.walk(local_folder):
    for file in files:
        file_path = os.path.join(root, file)
        relative_path = os.path.relpath(file_path, local_folder)

        # Skip metadata and temp files
        skip_patterns = ['.dataset_metadata.json', '*.tmp', '*.log', '__pycache__', '.DS_Store']
        if any(pattern in relative_path for pattern in skip_patterns):
            continue

        size = os.path.getsize(file_path)
        files_to_upload.append((relative_path, size))
        total_size += size

print(f"\n📁 Files to upload: {len(files_to_upload)} files")
print(f"📊 Total size: {total_size / (1024*1024):.2f} MB")
print(f"🔄 Unlimited auto-retry on rate limit (5 min wait)")

# =========================================================
# Upload with UNLIMITED retry
# =========================================================
print("\n📤 Uploading...")
attempt = 0

while True:
    attempt += 1
    try:
        api.upload_folder(
            folder_path=local_folder,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Auto-update: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ignore_patterns=[
                ".dataset_metadata.json",
                "*.tmp", 
                "*.log", 
                "__pycache__",
                ".DS_Store"
            ],
        )

        print("\n" + "="*60)
        print(f"✅ SUCCESS!")
        print(f"   Files uploaded: {len(files_to_upload)}")
        print(f"   Total attempts: {attempt}")
        print(f"   Commits: 1")
        print("="*60)
        break  # সফল হলে লুপ থেকে বেরিয়ে যাবে

    except Exception as e:
        if "429" in str(e):
            wait_time = 300  # 5 minutes
            print(f"\n⚠️ Rate limited! (Attempt {attempt})")
            print(f"⏳ Waiting {wait_time//60} minutes for rate limit reset...")
            print(f"📊 Already uploaded files will NOT be re-uploaded")
            time.sleep(wait_time)
            print(f"🔄 Resuming upload...")
        else:
            print(f"\n❌ Upload failed: {e}")
            exit(1)