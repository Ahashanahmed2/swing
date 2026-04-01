# upload_models.py - উন্নত সংস্করণ
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

# Login
login(token=HF_TOKEN)
api = HfApi()

local_folder = "./csv"

# চেক: ফোল্ডার আছে কিনা
if not os.path.exists(local_folder):
    print(f"❌ Folder not found: {local_folder}")
    exit(1)

# ফাইলের তালিকা ও সাইজ
files_to_upload = []
total_size = 0

for root, dirs, files in os.walk(local_folder):
    for file in files:
        file_path = os.path.join(root, file)
        relative_path = os.path.relpath(file_path, local_folder)
        
        # মেটাডাটা ফাইল স্কিপ
        if relative_path == '.dataset_metadata.json':
            continue
            
        size = os.path.getsize(file_path)
        files_to_upload.append((relative_path, size))
        total_size += size

print(f"\n📁 Files to upload: {len(files_to_upload)} files")
print(f"📊 Total size: {total_size / (1024*1024):.2f} MB")
print(f"💾 Commit: 1 (single commit)")

# কনফার্ম
print("\n" + "-"*60)
confirm = input("Proceed with upload? (y/n): ")
if confirm.lower() != 'y':
    print("❌ Cancelled")
    exit(0)

# আপলোড
try:
    print("\n📤 Uploading...")
    
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
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Upload failed: {e}")