import os
from pymongo import MongoClient
import logging
import sys

# লগিং সেটআপ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment Variables
MONGODBEMAIL_URI = os.environ.get('MONGODBMAIL_URI')
CSV_FILE_PATH = './csv/emails.txt'  # লোকাল ফাইল পাথ

def read_local_emails():
    """লোকাল CSV ফাইল থেকে ইমেল লিস্ট পড়ে"""
    try:
        # ফাইল আছে কিনা চেক করুন
        if not os.path.exists(CSV_FILE_PATH):
            logger.warning(f"ফাইল নেই: {CSV_FILE_PATH}। নতুন ফাইল তৈরি করা হবে।")
            return set()
        
        with open(CSV_FILE_PATH, 'r') as f:
            content = f.read().strip()
        
        emails = {line.strip().lower() for line in content.splitlines() if line.strip()}
        logger.info(f"লোকাল ফাইল থেকে {len(emails)}টি ইমেল পাওয়া গেছে")
        return emails
        
    except Exception as e:
        logger.error(f"লোকাল ফাইল পড়তে সমস্যা: {e}")
        return None

def get_mongodb_emails():
    """MongoDB থেকে ইমেল লিস্ট নেয়"""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client["email_bot_db"]
        collection = db["emails"]
        
        emails = {doc["email"].lower() for doc in collection.find({}, {"email": 1})}
        logger.info(f"MongoDB থেকে {len(emails)}টি ইমেল পাওয়া গেছে")
        return emails, collection
        
    except Exception as e:
        logger.error(f"MongoDB থেকে পড়তে সমস্যা: {e}")
        return None, None

def update_local_file(emails):
    """লোকাল CSV ফাইল আপডেট করে"""
    try:
        # ডিরেক্টরি আছে কিনা চেক করুন
        os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
        
        # ইমেলগুলো সাজিয়ে টেক্সট বানান
        content = "\n".join(sorted(list(emails)))
        
        # ফাইল সেভ করুন
        with open(CSV_FILE_PATH, "w") as f:
            f.write(content)
        
        logger.info(f"✅ লোকাল ফাইল আপডেট হয়েছে: {len(emails)}টি ইমেল")
        logger.info(f"   ফাইল লোকেশন: {CSV_FILE_PATH}")
        return True
        
    except Exception as e:
        logger.error(f"লোকাল ফাইল সেভ করতে সমস্যা: {e}")
        return False

def sync():
    """মূল সিঙ্ক ফাংশন"""
    logger.info("🔄 লোকাল ফাইল ও MongoDB সিঙ্ক্রোনাইজেশন শুরু...")
    
    # ডাটা সংগ্রহ
    local_emails = read_local_emails()
    if local_emails is None:
        return False
    
    mongo_emails, collection = get_mongodb_emails()
    if mongo_emails is None:
        return False
    
    # তুলনা
    only_in_local = local_emails - mongo_emails
    only_in_mongo = mongo_emails - local_emails
    in_both = local_emails & mongo_emails
    
    logger.info(f"📊 বিশ্লেষণ:")
    logger.info(f"   - লোকাল ফাইল: {len(local_emails)}টি")
    logger.info(f"   - MongoDB: {len(mongo_emails)}টি")
    logger.info(f"   - শুধু লোকাল ফাইলে: {len(only_in_local)}টি")
    logger.info(f"   - শুধু MongoDB-তে: {len(only_in_mongo)}টি")
    logger.info(f"   - উভয় জায়গায়: {len(in_both)}টি")
    
    # বিস্তারিত লগ (ঐচ্ছিক)
    if only_in_local:
        logger.info(f"📝 শুধু লোকাল ফাইলে: {', '.join(only_in_local)}")
    if only_in_mongo:
        logger.info(f"📝 শুধু MongoDB-তে: {', '.join(only_in_mongo)}")
    
    # যদি কোন পরিবর্তন না থাকে
    if not only_in_local and not only_in_mongo:
        logger.info("✅ কোন পরিবর্তন নেই। সব সমান।")
        return True
    
    # লোকাল ফাইল আপডেট করার জন্য MongoDB-র ইমেলগুলো নিন
    final_emails = mongo_emails
    
    # লোকাল ফাইল আপডেট করুন
    success = update_local_file(final_emails)
    
    if success:
        logger.info("🎉 সিঙ্ক সম্পন্ন!")
        logger.info(f"📁 লোকাল ফাইল এখন MongoDB-র সাথে সিঙ্ক হয়েছে")
        return True
    else:
        logger.error("❌ সিঙ্ক ব্যর্থ")
        return False

if __name__ == "__main__":
    # MONGODB_URI সেট করা আছে কিনা চেক করুন
    if not MONGODB_URI:
        logger.error("MONGODB_URI environment variable সেট করা নেই!")
        logger.error("দয়া করে .env ফাইল বা GitHub Secrets-এ MONGODB_URI সেট করুন।")
        sys.exit(1)
    
    success = sync()
    sys.exit(0 if success else 1)
