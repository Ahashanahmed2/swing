import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
import sys
import os

def check_date_and_continue():
    url = "https://dsebd.org/dseX_share.php"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ ওয়েবসাইট থেকে ডেটা আনতে সমস্যা হয়েছে: {e}")
        sys.exit(1)  # স্ক্রিপ্ট বন্ধ
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # পৃষ্ঠার শিরোনাম থেকে তারিখ বের করা
    page_text = soup.get_text()
    
    # "Share On" লাইন খোঁজা
    share_on_pattern = r"Share On\s+([A-Za-z]+)\s+(\d+),\s+(\d{4})\s+at\s+(\d+:\d+\s+[AP]M)"
    match = re.search(share_on_pattern, page_text, re.IGNORECASE)
    
    if not match:
        print("❌ পৃষ্ঠায় 'Share On' তারিখ খুঁজে পাওয়া যায়নি।")
        sys.exit(1)  # স্ক্রিপ্ট বন্ধ
    
    month_name, day, year, time_str = match.groups()
    
    # মাসের নামকে সংখ্যায় রূপান্তর
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    month = month_map.get(month_name[:3])
    if not month:
        print(f"❌ অবৈধ মাস: {month_name}")
        sys.exit(1)  # স্ক্রিপ্ট বন্ধ
    
    day = int(day)
    year = int(year)
    
    # আজকের তারিখ
    today = datetime.now()
    
    # পৃষ্ঠার তারিখ ও আজকের তারিখ তুলনা
    if not (today.year == year and today.month == month and today.day == day):
        print(f"❌ আজকের তারিখ ({today.strftime('%Y-%m-%d')}) এবং পৃষ্ঠার তারিখ ({year}-{month:02d}-{day:02d}) মিলছে না।")
        print("📅 ডেটা আপডেট না হওয়া পর্যন্ত অপেক্ষা করুন।")
        sys.exit(0)  # স্ক্রিপ্ট সফলভাবে বন্ধ (কিন্তু তারিখ না মেলায় পরবর্তী স্ক্রিপ্টে যাবে না)
    
    print(f"✅ তারিখ মিলেছে! ({today.strftime('%Y-%m-%d')})")
    print("🔄 পরবর্তী স্ক্রিপ্টে যাওয়া হচ্ছে...\n")
    
    # এখানে কোনো সিগনাল বা ইন্ডিকেটর তৈরি না করে শুধু চেক করে পাস করবে
    # কারণ main.py প্রতিটি স্ক্রিপ্ট আলাদাভাবে চালাচ্ছে

if __name__ == "__main__":
    check_date_and_continue()