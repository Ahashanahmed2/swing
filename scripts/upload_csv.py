# upload_models.py - Direct upload script
import os
from huggingface_hub import HfApi, upload_file, login
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("hf_token")
REPO_ID = "ahashanahmed/csv"

print("="*60)
print("🚀 UPLOADING MODELS TO HUGGING FACE")
print("="*60)

# Login
login(token=HF_TOKEN)
api = HfApi()

# Upload all files from ./csv folder
local_folder = "./csv"
uploaded = 0
failed = 0

for root, dirs, files in os.walk(local_folder):
    for file in files:
        file_path = os.path.join(root, file)
        relative_path = os.path.relpath(file_path, local_folder)
        
        # Skip metadata file
        if relative_path == '.dataset_metadata.json':
            continue
            
        try:
            print(f"📤 Uploading: {relative_path} ({os.path.getsize(file_path)/1024:.1f} KB)")
            
            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=relative_path,
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
            uploaded += 1
            print(f"   ✅ Uploaded")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed += 1

print("\n" + "="*60)
print(f"✅ Upload complete!")
print(f"   Uploaded: {uploaded} files")
print(f"   Failed: {failed} files")
print("="*60)