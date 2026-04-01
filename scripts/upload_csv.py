# upload_models.py - Direct model upload
import os
from hf_uploader import SmartDatasetUploader, REPO_ID, HF_TOKEN

print("🚀 Uploading XGBoost models to Hugging Face...")

uploader = SmartDatasetUploader(REPO_ID, HF_TOKEN)

# আপলোড করুন পুরো ./csv ফোল্ডার
uploader.smart_upload(
    local_folder="./csv",
    unique_columns=['symbol', 'date']
)

print("✅ Models uploaded!")