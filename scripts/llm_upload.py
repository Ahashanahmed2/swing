# scripts/llm_upload.py
from huggingface_hub import HfApi, upload_file
import pandas as pd
import os
from datetime import datetime

def main():
    print("="*60)
    print("🚀 UPLOADING LLM TRAINING DATA TO HUGGING FACE")
    print("="*60)
    
    repo_id = "ahashanahmed/LLM_model_stock"
    
    # 1. ট্রেনিং টেক্সট তৈরি করুন
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
"""
    
    with open("training_texts.txt", "w", encoding="utf-8") as f:
        f.write(training_text)
    print("✅ Created training_texts.txt")
    
    # 2. CSV ফাইল তৈরি করুন
    patterns_data = pd.DataFrame({
        'symbol': ['RELIANCE1', 'KPCL', 'TECHNODRUG', 'APEXFOOT', 'SONALIANSH'],
        'pattern': ['Cup and Handle', 'Bull Flag', 'Double Bottom', 'Bull Flag', 'Cup and Handle'],
        'confidence': [0.85, 0.78, 0.82, 0.71, 0.88],
        'target': [65, 62, 40, 70, 135],
        'stop_loss': [54, 52, 33, 58, 112]
    })
    patterns_data.to_csv("market_patterns.csv", index=False)
    print("✅ Created market_patterns.csv")
    
    # 3. README.md তৈরি করুন (সঠিকভাবে)
    readme_text = "# LLM Model Stock Dataset\n\n"
    readme_text += "## Description\n"
    readme_text += "This dataset is used for LLM training for stock market pattern recognition.\n\n"
    readme_text += "## Files\n"
    readme_text += "- `training_texts.txt`: Training text data\n"
    readme_text += "- `market_patterns.csv`: Pattern label data\n\n"
    readme_text += "## Usage\n"
    readme_text += "```python\n"
    readme_text += "from datasets import load_dataset\n"
    readme_text += "dataset = load_dataset(\"ahashanahmed/LLM_model_stock\")\n"
    readme_text += "```\n\n"
    readme_text += "## License\n"
    readme_text += "MIT\n"
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_text)
    print("✅ Created README.md")
    
    # 4. Hugging Face-এ আপলোড করুন
    print("\n📤 Uploading to Hugging Face...")
    
    try:
        for file in ["training_texts.txt", "market_patterns.csv", "README.md"]:
            upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"✅ Uploaded: {file}")
        
        print("\n" + "="*60)
        print("🎉 All files uploaded successfully!")
        print(f"🔗 View your dataset: https://huggingface.co/datasets/{repo_id}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    main()