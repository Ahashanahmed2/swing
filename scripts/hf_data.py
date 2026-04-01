# fetch_and_analyze.py
import pandas as pd
import requests
from io import StringIO

# Download from Hugging Face
url = "https://huggingface.co/datasets/ahashanahmed/csv/resolve/main/mongodb.csv"
response = requests.get(url)

if response.status_code == 200:
    df = pd.read_csv(StringIO(response.text))
    print("✅ ডাটা সফলভাবে ডাউনলোড হয়েছে")
    print("="*60)
    print(f"📊 মোট রো: {len(df)}")
    print(f"📈 মোট সিম্বল: {df['symbol'].nunique()}")
    print(f"📅 ডাটা রেঞ্জ: {df['date'].min()} থেকে {df['date'].max()}")
    print("\n" + "="*60)
    print("সিম্বল ভিত্তিক রো কাউন্ট:")
    print("="*60)
    
    symbol_counts = df['symbol'].value_counts().sort_values(ascending=False)
    
    # Show all symbols with counts
    for symbol, count in symbol_counts.items():
        status = "✅" if count >= 100 else "⚠️" if count >= 50 else "❌"
        print(f"{status} {symbol}: {count} rows")
    
    print("\n" + "="*60)
    print("সারাংশ:")
    print(f"✅ 100+ rows: {(symbol_counts >= 100).sum()} symbols")
    print(f"⚠️ 50-99 rows: {((symbol_counts >= 50) & (symbol_counts < 100)).sum()} symbols")
    print(f"❌ <50 rows: {(symbol_counts < 50).sum()} symbols")
    
else:
    print(f"❌ ডাউনলোড失敗: {response.status_code}")