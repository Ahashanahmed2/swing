"""
scripts/save_to_mongodb.py
FINAL_AI_SIGNALS.csv + Support/Resistance + MACD + EMA + Daily Buy সব MongoDB-তে সেইভ করে
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

# সব ফাইল কনফিগ
FILES_TO_SAVE = [
    {
        "path": "./output/ai_signal/FINAL_AI_SIGNALS.csv",
        "collection": "daily_ai_signals",
        "has_date": False,
        "description": "AI Trading Signals (LLM + XGBoost + PPO + Elliott Wave)"
    },
    {
        "path": "./output/ai_signal/support_resistant.csv",
        "collection": "support_resistance",
        "has_date": True,
        "date_column": "current_date",
        "description": "Support/Resistance Levels"
    },
    {
        "path": "./output/ai_signal/daily_buy.csv",
        "collection": "daily_buy_signals",
        "has_date": True,
        "date_column": "date",
        "description": "Daily Buy Signals"
    },
    {
        "path": "./output/ai_signal/macd.csv",
        "collection": "macd_signals",
        "has_date": False,
        "description": "MACD Signals"
    },
    {
        "path": "./output/ai_signal/macd_daily.csv",
        "collection": "macd_daily_signals",
        "has_date": False,
        "description": "MACD Daily Signals"
    },
    {
        "path": "./output/ai_signal/ema_200.csv",
        "collection": "ema_200_signals",
        "has_date": False,
        "description": "EMA 200 Signals"
    },
]

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
        client.admin.command('ping')
        print("✅ MongoDB Connected Successfully!")
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"❌ MongoDB Connection Failed: {e}")
        return None

# =========================================================
# CSV থেকে ডেটা লোড
# =========================================================
def load_csv_data(filepath):
    """CSV ফাইল থেকে ডেটা লোড করে"""
    if not os.path.exists(filepath):
        print(f"   ⚠️ File not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)
        print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"   ❌ CSV Load Error: {e}")
        return None

# =========================================================
# MongoDB-তে সেইভ
# =========================================================
def save_to_mongodb(df, client, collection_name, has_date=False, date_column=None, is_ai_signals=False):
    """DataFrame MongoDB-তে সেইভ করে"""
    if df is None or client is None:
        return False

    try:
        db = client[DATABASE_NAME]
        collection = db[collection_name]

        today = datetime.now().strftime('%Y-%m-%d')
        today_datetime = datetime.now()

        # DataFrame থেকে ডিকশনারিতে কনভার্ট
        records = df.to_dict('records')

        # প্রতিটি রেকর্ডে তারিখ ও টাইমস্ট্যাম্প যোগ করুন
        for record in records:
            record['saved_at'] = datetime.now().isoformat()

            # Date কলাম থেকে analysis_date বের করুন
            if has_date and date_column and date_column in df.columns:
                try:
                    date_val = pd.to_datetime(record.get(date_column), format='mixed', errors='coerce')
                    if not pd.isna(date_val):
                        record['analysis_date'] = date_val.strftime('%Y-%m-%d')
                except:
                    pass

            # AI Signals এর জন্য extra ফিল্ড
            if is_ai_signals:
                record['analysis_date'] = today
                record['analysis_datetime'] = today_datetime

            # NaN ভ্যালুগুলো None-এ কনভার্ট (MongoDB compatible)
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None

        # পুরনো ডেটা ডিলিট
        if is_ai_signals:
            delete_result = collection.delete_many({'analysis_date': today})
        else:
            delete_result = collection.delete_many({})
        
        print(f"   🗑️ Deleted {delete_result.deleted_count} old records")

        # নতুন ডেটা ইনসার্ট
        if records:
            insert_result = collection.insert_many(records)
            print(f"   ✅ Inserted {len(insert_result.inserted_ids)} new records")

        # ইনডেক্স তৈরি
        if 'symbol' in df.columns:
            collection.create_index([('symbol', 1)])
        if 'analysis_date' in df.columns or is_ai_signals:
            collection.create_index([('analysis_date', -1)])
        if is_ai_signals:
            collection.create_index([('final_combined_score', -1)])
            collection.create_index([('final_signal', 1)])
        print(f"   ✅ Indexes created")

        return True
    except Exception as e:
        print(f"   ❌ MongoDB Save Error: {e}")
        return False

# =========================================================
# স্ট্যাটাস চেক
# =========================================================
def check_all_collections(client):
    """সব Collection-এর স্ট্যাটাস দেখায়"""
    db = client[DATABASE_NAME]
    
    print(f"\n📊 MONGODB COLLECTIONS STATUS:")
    print("=" * 50)
    
    for file_config in FILES_TO_SAVE:
        collection_name = file_config["collection"]
        collection = db[collection_name]
        count = collection.count_documents({})
        print(f"   📂 {collection_name}: {count} documents")
    
    print("=" * 50)

# =========================================================
# মেইন ফাংশন
# =========================================================
def main():
    print("=" * 70)
    print("💾 MONGODB SAVE SCRIPT - ALL TRADING DATA")
    print("=" * 70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Files to save: {len(FILES_TO_SAVE)}")
    print(f"🗄️  Database: {DATABASE_NAME}")
    print("=" * 70)

    # ১. MongoDB কানেক্ট
    print("\n🔗 Connecting to MongoDB...")
    client = connect_mongodb()
    if client is None:
        print("❌ Failed to connect to MongoDB. Exiting.")
        sys.exit(1)

    total_saved = 0
    total_skipped = 0

    # ২. প্রতিটি ফাইল প্রসেস
    for file_config in FILES_TO_SAVE:
        filepath = file_config["path"]
        collection_name = file_config["collection"]
        has_date = file_config.get("has_date", False)
        date_column = file_config.get("date_column", None)
        is_ai_signals = (collection_name == "daily_ai_signals")
        description = file_config.get("description", "")

        print(f"\n{'='*50}")
        print(f"📂 {description}")
        print(f"   Path: {filepath}")
        print(f"   Collection: {collection_name}")
        print(f"{'='*50}")

        # CSV লোড
        df = load_csv_data(filepath)
        
        if df is None:
            total_skipped += 1
            continue

        # MongoDB-তে সেইভ
        success = save_to_mongodb(
            df, client, collection_name, 
            has_date=has_date, 
            date_column=date_column,
            is_ai_signals=is_ai_signals
        )

        if success:
            total_saved += 1
        else:
            total_skipped += 1

    # ৩. স্ট্যাটাস চেক
    check_all_collections(client)

    # ৪. ক্লোজ
    client.close()
    
    print("\n" + "=" * 70)
    print(f"✅ SCRIPT COMPLETED")
    print(f"   📂 Saved: {total_saved} files")
    print(f"   ⚠️ Skipped: {total_skipped} files")
    print("=" * 70)

if __name__ == "__main__":
    main()