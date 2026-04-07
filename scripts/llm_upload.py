# scripts/llm_upload.py
# LLM ট্রেনিং ডাটা Hugging Face-এ আপলোড করার স্ক্রিপ্ট

import os
import pandas as pd
from huggingface_hub import HfApi, login, upload_file

def main():
    print("="*60)
    print("🚀 UPLOADING LLM TRAINING DATA TO HUGGING FACE")
    print("="*60)
    
    # =========================================================
    # 1. টোকেন চেক করুন
    # =========================================================
    token = os.getenv("hf_token")
    if not token:
        print("❌ HF_TOKEN not found in environment variables!")
        print("   Please set hf_token in GitHub Secrets or .env file")
        return
    
    print("✅ Token found")
    
    # =========================================================
    # 2. Hugging Face-এ লগইন করুন
    # =========================================================
    try:
        login(token=token)
        print("✅ Logged in to Hugging Face")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return
    
    repo_id = "ahashanahmed/LLM_model_stock"
    
    # =========================================================
    # 3. বর্তমান ডিরেক্টরি দেখান
    # =========================================================
    print(f"\n📁 Current working directory: {os.getcwd()}")
    
    # =========================================================
    # 4. training_texts.txt ফাইল তৈরি করুন
    # =========================================================
    training_text = """Symbol: RELIANCE1
Pattern: Cup and Handle detected. Cup bottom at 52, handle between 55-57. 
Breakout above 58 confirmed. Target: 65. Stop loss: 54.

Symbol: KPCL
Pattern: Bull Flag. Flagpole from 45 to 55. Consolidation between 52-54.
Breakout above 55 with volume. Target: 62.

Symbol: TECHNODRUG
Pattern: Double Bottom at 30 and 31. Neckline at 35. Breakout confirmed.
Target: 40.

Symbol: APEXFOOT
Pattern: Bull Flag. Sharp move from 50 to 60, consolidation at 57-59.
Breakout above 60. Target: 70.

Symbol: SONALIANSH
Pattern: Cup and Handle. Cup bottom at 100, handle at 110-115.
Breakout above 118. Target: 135.

Symbol: VFSTDL
Pattern: Double Bottom. Bottom at 25 and 26. Neckline at 30.
Breakout above 30 confirmed. Target: 38.

Symbol: PF1STMF
Pattern: Bull Flag. Sharp rally from 80 to 95. Consolidation at 90-93.
Breakout above 95. Target: 110.
"""
    
    with open("training_texts.txt", "w", encoding="utf-8") as f:
        f.write(training_text)
    print("✅ Created training_texts.txt")
    
    # =========================================================
    # 5. market_patterns.csv ফাইল তৈরি করুন
    # =========================================================
    patterns_data = pd.DataFrame({
        'symbol': ['RELIANCE1', 'KPCL', 'TECHNODRUG', 'APEXFOOT', 'SONALIANSH', 'VFSTDL', 'PF1STMF'],
        'pattern': ['Cup and Handle', 'Bull Flag', 'Double Bottom', 'Bull Flag', 'Cup and Handle', 'Double Bottom', 'Bull Flag'],
        'confidence': [0.85, 0.78, 0.82, 0.71, 0.88, 0.75, 0.80],
        'target': [65, 62, 40, 70, 135, 38, 110],
        'stop_loss': [54, 52, 33, 58, 112, 27, 92],
        'support': [52, 45, 30, 50, 100, 25, 80],
        'resistance': [58, 55, 35, 60, 118, 30, 95]
    })
    patterns_data.to_csv("market_patterns.csv", index=False)
    print("✅ Created market_patterns.csv")
    
    # =========================================================
    # 6. README.md ফাইল তৈরি করুন
    # =========================================================
    readme_text = """# LLM Model Stock Dataset

## 📊 Description
This dataset contains stock market pattern data for training a small LLM (Language Model) for pattern recognition.

## 📁 Files
- `training_texts.txt`: Text descriptions of stock patterns for training
- `market_patterns.csv`: Structured pattern data with confidence scores

## 📈 Pattern Types
- Cup and Handle
- Bull Flag
- Double Bottom

## 🚀 Usage
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("ahashanahmed/LLM_model_stock")

# Access training texts
print(dataset['train']['training_texts.txt'])

# Access pattern data
patterns = pd.read_csv("market_patterns.csv")
"""
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_text)
print("✅ Created README.md")

# =========================================================
# 7. ফাইল চেক করুন
# =========================================================
print("\n📁 Checking created files:")
files_to_upload = ["training_texts.txt", "market_patterns.csv", "README.md"]

for file in files_to_upload:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"   ✅ {file} ({size} bytes)")
    else:
        print(f"   ❌ {file} (NOT FOUND)")
        return

# =========================================================
# 8. Hugging Face-এ আপলোড করুন
# =========================================================
print("\n📤 Uploading to Hugging Face...")
print(f"📦 Repository: {repo_id}")
print("-" * 40)

success_count = 0
fail_count = 0

for file in files_to_upload:
    try:
        upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"   ✅ Uploaded: {file}")
        success_count += 1
    except Exception as e:
        print(f"   ❌ Failed: {file} - {e}")
        fail_count += 1

# =========================================================
# 9. ফলাফল সারাংশ
# =========================================================
print("\n" + "="*60)
print("📊 UPLOAD SUMMARY")
print("="*60)
print(f"   ✅ Successful: {success_count}")
print(f"   ❌ Failed: {fail_count}")
print(f"   📁 Repository: https://huggingface.co/datasets/{repo_id}")
print("="*60)

if success_count == len(files_to_upload):
    print("🎉 All files uploaded successfully!")
else:
    print("⚠️ Some files failed to upload. Please check the errors above.")

if name == "main":
main()