from huggingface_hub import HfApi, upload_file
import pandas as pd
import os

# API ইনিশিয়ালাইজ
api = HfApi()
repo_id = "ahashanahmed/LLM_model_stock"

# ১. ট্রেনিং টেক্সট ফাইল তৈরি করুন
training_text = """
Symbol: RELIANCE1
Pattern: Cup and Handle detected. Cup bottom at 52, handle between 55-57. 
Breakout above 58 confirmed. Target: 65. Stop loss: 54.

Symbol: KPCL
Pattern: Bull Flag. Flagpole from 45 to 55. Consolidation between 52-54.
Breakout above 55 with volume. Target: 62.

Symbol: TECHNODRUG
Pattern: Double Bottom at 30 and 31. Neckline at 35. Breakout confirmed.
Target: 40.
"""

with open("training_texts.txt", "w", encoding="utf-8") as f:
    f.write(training_text)

# ২. CSV ফাইল তৈরি করুন
patterns_data = pd.DataFrame({
    'symbol': ['RELIANCE1', 'KPCL', 'TECHNODRUG', 'APEXFOOT'],
    'pattern': ['Cup and Handle', 'Bull Flag', 'Double Bottom', 'Bull Flag'],
    'confidence': [0.85, 0.78, 0.82, 0.71],
    'target': [65, 62, 40, 55],
    'stop_loss': [54, 52, 33, 48]
})
patterns_data.to_csv("market_patterns.csv", index=False)

# ৩. README.md তৈরি করুন
readme = """# LLM Model Stock Dataset

## Description
এই ডেটাসেটটি স্টক মার্কেটের প্যাটার্ন রিকগনিশনের জন্য এলএলএম ট্রেনিং করতে ব্যবহৃত হবে।

## Files
- `training_texts.txt`: ট্রেনিং টেক্সট ডেটা
- `market_patterns.csv`: প্যাটার্ন লেবেল ডেটা
- `support_resistance.csv`: সাপোর্ট/রেজিস্ট্যান্স ডেটা (coming soon)

## Usage
```python
from datasets import load_dataset
dataset = load_dataset("ahashanahmed/LLM_model_stock")