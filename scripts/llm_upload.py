from huggingface_hub import HfApi, upload_file, login
import pandas as pd
import os
from datetime import datetime

def main():
    print("="*60)
    print("🚀 UPLOADING LLM TRAINING DATA TO HUGGING FACE")
    print("="*60)
    
    # ✅ Login করুন
    token = os.getenv("hf_token")
    if token:
        login(token=token)
        print("✅ Logged in to Hugging Face")
    else:
        print("❌ No token found!")
        return
    
    repo_id = "ahashanahmed/LLM_model_stock"
    
    # ... ফাইল তৈরি করুন ...
    
    # ✅ আপলোড করুন (token লাগবে না, login ইতিমধ্যে done)
    try:
        for file in ["training_texts.txt", "market_patterns.csv", "README.md"]:
            upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"✅ Uploaded: {file}")
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    main()