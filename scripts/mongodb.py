# scripts/mongodb.py
from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
os.makedirs('./csv/', exist_ok=True)

# MongoDB-এ সংযোগ
mongourl = os.getenv("MONGO_URL")
client = MongoClient(mongourl)
db = client["candleData"]
collection = db["candledatas"]

# CSV ফাইল থেকে বিদ্যমান ডেটা লোড করুন (যদি থাকে)
csv_file_path = './csv/mongodb.csv'
existing_data = pd.DataFrame()

if os.path.exists(csv_file_path):
    existing_data = pd.read_csv(csv_file_path)
    last_date = existing_data['date'].max() if 'date' in existing_data.columns else None
else:
    last_date = None

# MongoDB থেকে শুধুমাত্র নতুন ডেটা কুয়েরি করুন
query = {}
if last_date:
    query['date'] = {'$gt': last_date}

data = list(collection.find(query, {'_id': 0, '__v': 0}))

if data:
    new_data = pd.DataFrame(data)
    
    # নতুন এবং বিদ্যমান ডেটা মার্জ করুন
    if not existing_data.empty:
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        updated_data = new_data
    
    # ডুপ্লিকেট ডেটা রিমুভ করুন (যদি থাকে)
    if 'date' in updated_data.columns:
        updated_data.drop_duplicates(subset=['date'], keep='last', inplace=True)
    
    # CSV ফাইলে সেভ করুন
    updated_data.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
    print(f"{len(new_data)} নতুন রেকর্ড যোগ করা হয়েছে, মোট রেকর্ড: {len(updated_data)}")
else:
    print("কোন নতুন ডেটা পাওয়া যায়নি")

client.close()
