# upload_csv.py - GitHub Actions-এর জন্য সঠিক সংস্করণ
import os
from huggingface_hub import HfApi, login
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
HF_TOKEN = os.getenv("hf_token")
REPO_ID = "ahashanahmed/csv"

print("="*60)
print("🚀 UPLOADING TO HUGGING FACE (SINGLE COMMIT)")
print("="*60)

# Check token
if not HF_TOKEN:
    print("❌ HF_TOKEN not found in environment variables!")
    print("   Please set HF_TOKEN in GitHub Secrets or .env file")
    exit(1)

# Login
try:
    login(token=HF_TOKEN)
    print("✅ Logged in to Hugging Face")
except Exception as e:
    print(f"❌ Login failed: {e}")
    exit(1)

api = HfApi()

local_folder = "./csv"

# Check folder exists
if not os.path.exists(local_folder):
    print(f"❌ Folder not found: {local_folder}")
    exit(1)

# Check if repo exists, create if not
try:
    api.repo_info(repo_id=REPO_ID, repo_type="dataset")
    print(f"✅ Repository exists: {REPO_ID}")
except Exception:
    print(f"📁 Repository not found. Creating...")
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=False)
        print(f"✅ Created repository: {REPO_ID}")
    except Exception as e:
        print(f"❌ Failed to create repo: {e}")
        exit(1)

# Files to upload
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
print(f"💾 Single commit upload")

# Auto-upload (no confirmation needed in GitHub Actions)
print("\n📤 Uploading...")

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
    print(f"   Commits: 1")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    exit(1)