"""
scripts/save_to_mongodb.py
FINAL_AI_SIGNALS.csv + Support/Resistance + EMA + Daily Buy + SWRSI + Strong Ratio সব MongoDB-তে সেইভ করে
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, DuplicateKeyError

# =========================================================
# কনফিগারেশন
# =========================================================
MONGODB_URI = os.environ.get("MONGODBEMAIL_URI", "")
DATABASE_NAME = "swing_trading_db"

# সব ফাইল কনফিগ (macd.csv ও macd_daily.csv বাদ)
FILES_TO_SAVE = [
    {
        "path": "./output/ai_signal/FINAL_AI_SIGNALS.csv",
        "collection": "daily_ai_signals",
        "has_date": True,
        "date_column": "date",
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
        "path": "./output/ai_signal/ema_21.csv",
        "collection": "ema_21_signals",
        "has_date": False,
        "description": "EMA 21 Signals"
    },
    {
        "path": "./output/ai_signal/swrsi.csv",
        "collection": "swrsi_signals",
        "has_date": True,
        "date_column": "signal_date",
        "description": "SWRSI - Sector Weekly + Daily RSI Divergence Signals"
    },
    {
        "path": "./output/ai_signal/strong_ratio.csv",
        "collection": "strong_ratio_signals",
        "has_date": True,
        "date_column": "date",
        "description": "Strong Ratio - RT, BBR, Strong Divergence and Date Data"
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
        if df.empty:
            print(f"   ⚠️ File is empty: {filepath}")
            return None
        print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"   ❌ CSV Load Error: {e}")
        return None

# =========================================================
# Sector Mapping লোড করা
# =========================================================
def load_sector_mapping(csv_path="./csv/mongodb.csv"):
    """
    mongodb.csv ফাইল থেকে sector ডেটা লোড করে
    symbol -> sector ম্যাপিং তৈরি করে
    """
    if not os.path.exists(csv_path):
        print(f"⚠️ Sector file not found: {csv_path}")
        return {}

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"⚠️ Sector file is empty: {csv_path}")
            return {}

        # symbol এবং sector কলাম চেক করুন
        if 'symbol' not in df.columns or 'sector' not in df.columns:
            print(f"⚠️ Sector file missing required columns (symbol, sector)")
            print(f"   Available columns: {df.columns.tolist()}")
            return {}

        # ডুপ্লিকেট সিম্বল থাকলে লেটেস্ট টা নিন
        sector_df = df.dropna(subset=['symbol', 'sector'])
        
        # date কলাম থাকলে সাজান
        if 'date' in sector_df.columns:
            sector_df = sector_df.sort_values('date', ascending=False)
        
        # symbol -> sector ম্যাপিং
        sector_map = {}
        for _, row in sector_df.iterrows():
            symbol = str(row['symbol']).strip().upper()
            sector = str(row['sector']).strip()
            if symbol and sector and sector != 'nan' and sector != 'None':
                sector_map[symbol] = sector

        print(f"✅ Loaded {len(sector_map)} sector mappings from mongodb.csv")
        
        # Show sample sectors
        if sector_map:
            sample = list(sector_map.items())[:5]
            print(f"   Sample: {sample}")
        
        return sector_map

    except Exception as e:
        print(f"❌ Error loading sector mapping: {e}")
        import traceback
        traceback.print_exc()
        return {}

# =========================================================
# হেল্পার ফাংশন: তারিখ পার্স করা
# =========================================================
def parse_date_to_string(date_val):
    """বিভিন্ন ফরম্যাটের তারিখকে YYYY-MM-DD স্ট্রিং-এ কনভার্ট করে"""
    if date_val is None or pd.isna(date_val):
        return None

    try:
        if isinstance(date_val, datetime):
            return date_val.strftime('%Y-%m-%d')
        elif isinstance(date_val, str):
            # বিভিন্ন ফরম্যাট চেক করা
            for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d-%m-%Y', '%m/%d/%Y', '%Y%m%d']:
                try:
                    parsed_date = datetime.strptime(date_val.strip(), fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except:
                    continue
            # যদি কোনো ফরম্যাট ম্যাচ না করে, তাহলে pandas দিয়ে চেষ্টা
            try:
                return pd.to_datetime(date_val).strftime('%Y-%m-%d')
            except:
                return None
        else:
            return pd.to_datetime(date_val).strftime('%Y-%m-%d')
    except Exception:
        return None

# =========================================================
# MongoDB-তে সেইভ (Upsert - কোন ডাটা ডিলিট হবে না)
# =========================================================
def save_to_mongodb(df, client, collection_name, has_date=False, date_column=None, 
                    is_ai_signals=False, is_swrsi=False, sector_map=None):
    """DataFrame MongoDB-তে সেইভ করে - ডুপ্লিকেট ছাড়া, sector সহ"""
    if df is None or client is None:
        return False

    if df.empty:
        print(f"   ⚠️ DataFrame is empty for {collection_name}")
        return True

    try:
        db = client[DATABASE_NAME]
        collection = db[collection_name]

        today = datetime.now().strftime('%Y-%m-%d')
        today_datetime = datetime.now()

        # DataFrame থেকে ডিকশনারিতে কনভার্ট
        records = df.to_dict('records')

        if not records:
            print(f"   ⚠️ No records to process for {collection_name}")
            return True

        # প্রতিটি রেকর্ডে তারিখ ও টাইমস্ট্যাম্প যোগ করুন
        processed_records = []
        sector_added = 0
        sector_missing = 0

        for idx, record in enumerate(records):
            try:
                # Skip if no symbol for collections that need it
                if collection_name not in ["strong_ratio_signals"]:
                    if 'symbol' not in record or pd.isna(record.get('symbol')):
                        print(f"   ⚠️ Skipping record {idx}: missing symbol")
                        continue

                record['saved_at'] = datetime.now().isoformat()
                record['saved_timestamp'] = datetime.now()

                # =============================================
                # 🔑 SECTOR যোগ করুন (শুধু AI signals এর জন্য)
                # =============================================
                if is_ai_signals and sector_map:
                    symbol = record.get('symbol')
                    if symbol:
                        # symbol কে uppercase করুন এবং clean করুন
                        clean_symbol = str(symbol).strip().upper()
                        # sector_map থেকে sector খুঁজুন
                        if clean_symbol in sector_map:
                            record['sector'] = sector_map[clean_symbol]
                            sector_added += 1
                        else:
                            # sector না পেলে 'Other' দিন
                            record['sector'] = 'Other'
                            sector_missing += 1
                    else:
                        record['sector'] = 'Other'
                        sector_missing += 1
                elif is_ai_signals:
                    # sector_map না থাকলে 'Other' দিন
                    record['sector'] = 'Other'

                # =============================================
                # analysis_date নির্ধারণ (সবার জন্য ইউনিফাইড লজিক)
                # =============================================
                analysis_date = None

                # ১ম প্রাধান্য: কনফিগার করা date_column থেকে
                if has_date and date_column:
                    date_val = record.get(date_column)
                    if date_val and not pd.isna(date_val):
                        analysis_date = parse_date_to_string(date_val)

                # ২য় প্রাধান্য: CSV-তে থাকা অন্যান্য সম্ভাব্য date কলাম
                if not analysis_date:
                    possible_date_cols = ['date', 'Date', 'signal_date', 'current_date', 'analysis_date', 'trade_date']
                    for col in possible_date_cols:
                        if col in record and record.get(col) and not pd.isna(record.get(col)):
                            analysis_date = parse_date_to_string(record.get(col))
                            if analysis_date:
                                break

                # ৩য় প্রাধান্য: AI Signals এর জন্য বিশেষ হ্যান্ডলিং
                if not analysis_date and is_ai_signals:
                    # CSV-তে 'date' কলাম থাকলে সেটা ব্যবহার করো
                    if 'date' in record and record.get('date') and not pd.isna(record.get('date')):
                        analysis_date = parse_date_to_string(record.get('date'))
                    elif 'Date' in record and record.get('Date') and not pd.isna(record.get('Date')):
                        analysis_date = parse_date_to_string(record.get('Date'))

                # ৪র্থ প্রাধান্য: SWRSI এর জন্য signal_date ব্যবহার
                if not analysis_date and is_swrsi:
                    if 'signal_date' in record and record.get('signal_date') and not pd.isna(record.get('signal_date')):
                        analysis_date = parse_date_to_string(record.get('signal_date'))

                # ৫ম প্রাধান্য: সব ব্যর্থ হলে আজকের তারিখ
                if not analysis_date:
                    analysis_date = today
                    print(f"   ⚠️ Record {idx}: No date found, using today ({today})")

                record['analysis_date'] = analysis_date

                # AI Signals এর জন্য extra ফিল্ড
                if is_ai_signals:
                    record['analysis_datetime'] = today_datetime
                    # Ensure final signal is properly set
                    if 'final_signal' not in record or pd.isna(record.get('final_signal')):
                        record['final_signal'] = 'NEUTRAL'

                # SWRSI signals এর জন্য extra ফিল্ড
                if is_swrsi:
                    record['analysis_datetime'] = today_datetime
                    # Ensure required fields for SWRSI
                    if 'composite_score' not in record or pd.isna(record.get('composite_score')):
                        record['composite_score'] = 0
                    if 'weekly_strength_label' not in record or pd.isna(record.get('weekly_strength_label')):
                        record['weekly_strength_label'] = 'Weak'

                # Strong Ratio signals এর জন্য extra ফিল্ড
                if collection_name == "strong_ratio_signals":
                    if 'date' not in record or pd.isna(record.get('date')):
                        print(f"   ⚠️ Skipping record {idx}: missing date")
                        continue
                    if 'rt' not in record or pd.isna(record.get('rt')):
                        print(f"   ⚠️ Skipping record {idx}: missing rt")
                        continue

                # NaN ভ্যালুগুলো None-এ কনভার্ট (MongoDB compatible)
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None

                processed_records.append(record)

            except Exception as e:
                print(f"   ⚠️ Error processing record {idx}: {e}")
                continue

        if not processed_records:
            print(f"   ⚠️ No valid records to save for {collection_name}")
            return True

        # Sector statistics
        if is_ai_signals and sector_map:
            print(f"   📊 Sector Stats: {sector_added} found, {sector_missing} set to 'Other'")

        # কোন ডাটা ডিলিট করা হচ্ছে না - শুধু upsert
        inserted_count = 0
        updated_count = 0
        error_count = 0

        for record in processed_records:
            try:
                # strong_ratio_signals এর জন্য symbol কলাম নেই, rt ব্যবহার করা হবে
                if collection_name == "strong_ratio_signals":
                    rt_value = record.get('rt')
                    date_value = record.get('date')

                    if rt_value and date_value:
                        filter_query = {
                            'rt': rt_value,
                            'date': date_value
                        }
                        # Remove _id if exists to avoid conflicts
                        if '_id' in record:
                            del record['_id']

                        result = collection.update_one(
                            filter_query,
                            {'$set': record},
                            upsert=True
                        )
                        if result.upserted_id:
                            inserted_count += 1
                        else:
                            updated_count += 1
                    else:
                        collection.insert_one(record)
                        inserted_count += 1

                elif record.get('symbol') and record.get('analysis_date'):
                    # সিম্বল এবং analysis_date ভিত্তিতে upsert
                    filter_query = {
                        'symbol': record['symbol'],
                        'analysis_date': record['analysis_date']
                    }
                    # Remove _id if exists to avoid conflicts
                    if '_id' in record:
                        del record['_id']

                    result = collection.update_one(
                        filter_query,
                        {'$set': record},
                        upsert=True
                    )
                    if result.upserted_id:
                        inserted_count += 1
                    else:
                        updated_count += 1
                elif record.get('symbol'):
                    # শুধু সিম্বল ভিত্তিতে upsert (যেসব ফাইলে analysis_date নেই)
                    filter_query = {'symbol': record['symbol']}
                    if '_id' in record:
                        del record['_id']

                    result = collection.update_one(
                        filter_query,
                        {'$set': record},
                        upsert=True
                    )
                    if result.upserted_id:
                        inserted_count += 1
                    else:
                        updated_count += 1
                else:
                    # কোন ইউনিক কী না থাকলে সরাসরি ইনসার্ট
                    collection.insert_one(record)
                    inserted_count += 1

            except DuplicateKeyError:
                # Skip duplicates gracefully
                updated_count += 1
            except Exception as e:
                print(f"   ⚠️ Error upserting record: {e}")
                error_count += 1
                continue

        print(f"   ✅ Inserted: {inserted_count} new records")
        print(f"   🔄 Updated: {updated_count} existing records")
        if error_count > 0:
            print(f"   ⚠️ Errors: {error_count} records failed")
        print(f"   📚 All historical data preserved (no deletion)")

        # ইনডেক্স তৈরি (with error handling)
        try:
            if 'symbol' in df.columns or collection_name == "strong_ratio_signals":
                if collection_name == "strong_ratio_signals":
                    collection.create_index([('rt', 1)], background=True)
                    collection.create_index([('date', -1)], background=True)
                else:
                    collection.create_index([('symbol', 1)], background=True)

            # analysis_date ইনডেক্স সব collection-এর জন্য
            collection.create_index([('analysis_date', -1)], background=True)

            if is_ai_signals:
                collection.create_index([('final_combined_score', -1)], background=True)
                collection.create_index([('final_signal', 1)], background=True)
                # Sector index for AI signals
                collection.create_index([('sector', 1)], background=True)

            if is_swrsi:
                collection.create_index([('composite_score', -1)], background=True)
                collection.create_index([('sector', 1)], background=True)
                collection.create_index([('weekly_strength_label', 1)], background=True)

            # ইউনিক কম্পাউন্ড ইনডেক্স (ডুপ্লিকেট প্রতিরোধের জন্য)
            if 'symbol' in df.columns and not is_ai_signals:
                try:
                    collection.create_index([('symbol', 1), ('analysis_date', 1)], unique=True, sparse=True, background=True)
                except Exception as e:
                    # Index might already exist
                    pass

            if collection_name == "strong_ratio_signals":
                try:
                    collection.create_index([('rt', 1), ('date', 1)], unique=True, sparse=True, background=True)
                except Exception as e:
                    # Index might already exist
                    pass

            print(f"   ✅ Indexes verified")
        except Exception as e:
            print(f"   ⚠️ Index creation note: {e}")

        return True

    except Exception as e:
        print(f"   ❌ MongoDB Save Error: {e}")
        import traceback
        traceback.print_exc()
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
        try:
            collection = db[collection_name]
            count = collection.count_documents({})
            print(f"   📂 {collection_name}: {count} documents")
        except Exception as e:
            print(f"   ❌ {collection_name}: Error - {e}")

    print("=" * 50)

# =========================================================
# AI Signals Sector Statistics
# =========================================================
def check_ai_sector_stats(client):
    """AI Signals collection-এর sector statistics দেখায়"""
    try:
        db = client[DATABASE_NAME]
        ai_col = db["daily_ai_signals"]
        
        total = ai_col.count_documents({})
        if total == 0:
            print("\n📊 AI SIGNALS SECTOR STATISTICS:")
            print("-" * 40)
            print("   No AI signals found")
            return
        
        # sector আছে কতগুলো ডকুমেন্টে
        with_sector = ai_col.count_documents({'sector': {'$ne': None, '$ne': ''}})
        
        print(f"\n📊 AI SIGNALS SECTOR STATISTICS:")
        print("-" * 40)
        print(f"   Total AI Signals: {total}")
        print(f"   With Sector: {with_sector} ({with_sector/total*100:.1f}%)")
        print(f"   Without Sector: {total - with_sector} ({(total-with_sector)/total*100:.1f}%)")
        
        # Top sectors
        pipeline = [
            {'$match': {'sector': {'$ne': None, '$ne': ''}}},
            {'$group': {'_id': '$sector', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}},
            {'$limit': 10}
        ]
        top_sectors = list(ai_col.aggregate(pipeline))
        if top_sectors:
            print(f"\n   🏆 Top Sectors:")
            for i, sec in enumerate(top_sectors[:10], 1):
                print(f"     {i}. {sec['_id']}: {sec['count']} signals")
        
        # Latest date sectors
        latest_date = ai_col.distinct('analysis_date', sort=[('analysis_date', -1)])
        if latest_date:
            latest = latest_date[0]
            latest_sectors = ai_col.distinct('sector', {'analysis_date': latest, 'sector': {'$ne': None, '$ne': ''}})
            print(f"\n   📅 Latest Date ({latest}) Sectors: {len(latest_sectors)}")
            if latest_sectors:
                print(f"     {', '.join(latest_sectors[:10])}{'...' if len(latest_sectors) > 10 else ''}")
        
    except Exception as e:
        print(f"⚠️ Could not check AI sector stats: {e}")

# =========================================================
# মেইন ফাংশন
# =========================================================
def main():
    print("=" * 70)
    print("💾 MONGODB SAVE SCRIPT - ALL TRADING DATA + SWRSI + STRONG RATIO + SECTOR")
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

    # ২. 🆕 Sector Mapping লোড করুন
    print("\n📂 Loading sector mapping from mongodb.csv...")
    sector_map = load_sector_mapping("./csv/mongodb.csv")
    
    if sector_map:
        print(f"✅ Sector mapping loaded: {len(sector_map)} symbols")
    else:
        print("⚠️ No sector mapping found, continuing without sector data")
        sector_map = {}

    total_saved = 0
    total_skipped = 0
    total_failed = 0

    # ৩. প্রতিটি ফাইল প্রসেস
    for file_config in FILES_TO_SAVE:
        filepath = file_config["path"]
        collection_name = file_config["collection"]
        has_date = file_config.get("has_date", False)
        date_column = file_config.get("date_column", None)
        is_ai_signals = (collection_name == "daily_ai_signals")
        is_swrsi = (collection_name == "swrsi_signals")
        description = file_config.get("description", "")

        print(f"\n{'='*50}")
        print(f"📂 {description}")
        print(f"   Path: {filepath}")
        print(f"   Collection: {collection_name}")
        if is_ai_signals:
            print(f"   🆕 Will add sector data from mongodb.csv")
        print(f"{'='*50}")

        # CSV লোড
        df = load_csv_data(filepath)

        if df is None:
            print(f"   ⚠️ Skipped: File not found or empty")
            total_skipped += 1
            continue

        # MongoDB-তে সেইভ (sector_map পাস করুন)
        try:
            success = save_to_mongodb(
                df, client, collection_name, 
                has_date=has_date, 
                date_column=date_column,
                is_ai_signals=is_ai_signals,
                is_swrsi=is_swrsi,
                sector_map=sector_map  # 🆕 sector_map পাস করুন
            )

            if success:
                total_saved += 1
                print(f"   ✅ Successfully saved to MongoDB")
            else:
                total_failed += 1
                print(f"   ❌ Failed to save to MongoDB")
        except Exception as e:
            total_failed += 1
            print(f"   ❌ Exception: {e}")
            import traceback
            traceback.print_exc()

    # ৪. স্ট্যাটাস চেক
    try:
        check_all_collections(client)
    except Exception as e:
        print(f"⚠️ Could not check collections: {e}")

    # ৫. Sector Statistics (AI Signals)
    try:
        check_ai_sector_stats(client)
    except Exception as e:
        print(f"⚠️ Could not check AI sector stats: {e}")

    # ৬. SWRSI Summary (extra)
    try:
        db = client[DATABASE_NAME]
        swrsi_col = db["swrsi_signals"]
        if swrsi_col.count_documents({}) > 0:
            print(f"\n📊 SWRSI SIGNALS SUMMARY:")
            print("-" * 40)

            # Total signals
            total = swrsi_col.count_documents({})
            print(f"   Total Signals: {total}")

            # By strength
            for strength in ['Strong', 'Moderate', 'Weak']:
                count = swrsi_col.count_documents({'weekly_strength_label': strength})
                print(f"   {strength}: {count}")

            # High score signals
            high_score = swrsi_col.count_documents({'composite_score': {'$gte': 70}})
            print(f"   High Score (≥70): {high_score}")

            # By sector
            sectors = swrsi_col.distinct('sector')
            print(f"   Sectors: {len(sectors)} ({', '.join(sectors[:5])}{'...' if len(sectors) > 5 else ''})")
    except Exception as e:
        print(f"⚠️ Could not generate SWRSI summary: {e}")

    # ৭. Strong Ratio Summary (extra)
    try:
        strong_ratio_col = db["strong_ratio_signals"]
        if strong_ratio_col.count_documents({}) > 0:
            print(f"\n📊 STRONG RATIO SIGNALS SUMMARY:")
            print("-" * 40)

            total = strong_ratio_col.count_documents({})
            print(f"   Total Records: {total}")

            # Date range
            dates = strong_ratio_col.distinct('date')
            if dates:
                sorted_dates = sorted(dates)[:5]
                print(f"   Dates: {', '.join(sorted_dates)}{'...' if len(dates) > 5 else ''}")

            # Sample records
            sample = strong_ratio_col.find().limit(3)
            print(f"   Sample Data (first 3 records):")
            for doc in sample:
                print(f"     - Date: {doc.get('date', 'N/A')}, RT: {doc.get('rt', 'N/A')}, BBR: {doc.get('bbr', 'N/A')}")
    except Exception as e:
        print(f"⚠️ Could not generate Strong Ratio summary: {e}")

    # ৮. ক্লোজ
    client.close()

    print("\n" + "=" * 70)
    print(f"✅ SCRIPT COMPLETED")
    print(f"   📂 Saved: {total_saved} files")
    print(f"   ⚠️ Skipped: {total_skipped} files")
    print(f"   ❌ Failed: {total_failed} files")
    print("=" * 70)

    # Exit with appropriate code
    if total_failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()