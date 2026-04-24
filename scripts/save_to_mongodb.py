"""
scripts/save_to_mongodb.py
FINAL_AI_SIGNALS.csv থেকে ডেটা MongoDB-তে সেইভ করে
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# =========================================================
# কনফিগারেশন
# =========================================================
MONGODB_URI = os.environ.get("MONGODBEMAIL_URI", "")
DATABASE_NAME = "swing_trading_db"
COLLECTION_NAME = "daily_ai_signals"
CSV_PATH = "./csv/FINAL_AI_SIGNALS.csv"

# =========================================================
# MongoDB কানেকশন
# =========================================================
def connect_mongodb():
    """MongoDB-তে কানেক্ট করে ক্লায়েন্ট রিটার্ন করে"""
    if not MONGODB_URI:
        print("❌ MONGODBEMAIL_URI environment variable not set!")
        return None
    
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
        # চেক করুন কানেকশন ঠিক আছে কিনা
        client.admin.command('ping')
        print("✅ MongoDB Connected Successfully!")
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"❌ MongoDB Connection Failed: {e}")
        return None

# =========================================================
# CSV থেকে ডেটা লোড
# =========================================================
def load_csv_data():
    """CSV ফাইল থেকে ডেটা লোড করে"""
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV file not found: {CSV_PATH}")
        return None
    
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✅ Loaded {len(df)} rows from {CSV_PATH}")
        print(f"   Columns: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ CSV Load Error: {e}")
        return None

# =========================================================
# MongoDB-তে সেইভ
# =========================================================
def save_to_mongodb(df, client):
    """DataFrame MongoDB-তে সেইভ করে"""
    if df is None or client is None:
        return False
    
    try:
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        today = datetime.now().strftime('%Y-%m-%d')
        today_datetime = datetime.now()
        
        # DataFrame থেকে ডিকশনারিতে কনভার্ট
        records = df.to_dict('records')
        
        # প্রতিটি রেকর্ডে তারিখ ও টাইমস্ট্যাম্প যোগ করুন
        for record in records:
            record['analysis_date'] = today
            record['analysis_datetime'] = today_datetime
            record['saved_at'] = datetime.now().isoformat()
            
            # NaN ভ্যালুগুলো None-এ কনভার্ট (MongoDB compatible)
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
        
        # পুরনো ডেটা ডিলিট (একই তারিখের)
        delete_result = collection.delete_many({'analysis_date': today})
        print(f"   🗑️ Deleted {delete_result.deleted_count} old records for {today}")
        
        # নতুন ডেটা ইনসার্ট
        if records:
            insert_result = collection.insert_many(records)
            print(f"   ✅ Inserted {len(insert_result.inserted_ids)} new records")
        
        # ইনডেক্স তৈরি (দ্রুত সার্চের জন্য)
        collection.create_index([('analysis_date', -1)])
        collection.create_index([('symbol', 1)])
        collection.create_index([('final_combined_score', -1)])
        collection.create_index([('final_signal', 1)])
        print(f"   ✅ Indexes created")
        
        return True
    except Exception as e:
        print(f"❌ MongoDB Save Error: {e}")
        return False

# =========================================================
# স্ট্যাটাস চেক
# =========================================================
def check_collection_stats(client):
    """MongoDB কালেকশনের স্ট্যাটাস দেখায়"""
    try:
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        total_docs = collection.count_documents({})
        dates = collection.distinct('analysis_date')
        latest_date = max(dates) if dates else 'No data'
        
        print(f"\n📊 MONGODB COLLECTION STATS:")
        print(f"   📁 Database: {DATABASE_NAME}")
        print(f"   📂 Collection: {COLLECTION_NAME}")
        print(f"   📄 Total Documents: {total_docs}")
        print(f"   📅 Available Dates: {len(dates)}")
        print(f"   📆 Latest Date: {latest_date}")
        
        # আজকের স্ট্যাটাস
        today = datetime.now().strftime('%Y-%m-%d')
        today_count = collection.count_documents({'analysis_date': today})
        print(f"   🎯 Today's Signals: {today_count}")
        
        # সিগন্যাল ডিস্ট্রিবিউশন
        pipeline = [
            {'$match': {'analysis_date': today}},
            {'$group': {'_id': '$final_signal', 'count': {'$sum': 1}}}
        ]
        signal_dist = list(collection.aggregate(pipeline))
        if signal_dist:
            print(f"\n   📈 Today's Signal Distribution:")
            for item in signal_dist:
                print(f"      {item['_id']}: {item['count']}")
        
        return True
    except Exception as e:
        print(f"❌ Stats Error: {e}")
        return False

# =========================================================
# মেইন ফাংশন
# =========================================================
def main():
    print("=" * 70)
    print("💾 MONGODB SAVE SCRIPT - AI TRADING SIGNALS")
    print("=" * 70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 CSV Path: {CSV_PATH}")
    print(f"🗄️  Database: {DATABASE_NAME}.{COLLECTION_NAME}")
    print("=" * 70)
    
    # ১. CSV লোড
    print("\n📂 STEP 1: Loading CSV...")
    df = load_csv_data()
    if df is None:
        print("❌ Failed to load CSV. Exiting.")
        sys.exit(1)
    
    # ২. MongoDB কানেক্ট
    print("\n🔗 STEP 2: Connecting to MongoDB...")
    client = connect_mongodb()
    if client is None:
        print("❌ Failed to connect to MongoDB. Exiting.")
        sys.exit(1)
    
    # ৩. MongoDB-তে সেইভ
    print("\n💾 STEP 3: Saving to MongoDB...")
    success = save_to_mongodb(df, client)
    
    if success:
        print("\n✅ DATA SAVED SUCCESSFULLY!")
    else:
        print("\n❌ Failed to save data!")
    
    # ৪. স্ট্যাটাস চেক
    print("\n📊 STEP 4: Checking Stats...")
    check_collection_stats(client)
    
    # ৫. ক্লোজ
    client.close()
    print("\n" + "=" * 70)
    print("✅ SCRIPT COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()
