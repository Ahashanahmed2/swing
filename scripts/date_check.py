import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
import sys
import subprocess

def check_date_and_run_main_script():
    url = "https://dsebd.org/dseX_share.php"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"ওয়েবসাইট থেকে ডেটা আনতে সমস্যা হয়েছে: {e}")
        sys.exit(1)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # পৃষ্ঠার শিরোনাম থেকে তারিখ বের করা
    page_text = soup.get_text()
    
    # "Share On" লাইন খোঁজা
    share_on_pattern = r"Share On\s+([A-Za-z]+)\s+(\d+),\s+(\d{4})\s+at\s+(\d+:\d+\s+[AP]M)"
    match = re.search(share_on_pattern, page_text, re.IGNORECASE)
    
    if not match:
        print("পৃষ্ঠায় 'Share On' তারিখ খুঁজে পাওয়া যায়নি।")
        sys.exit(1)
    
    month_name, day, year, time_str = match.groups()
    
    # মাসের নামকে সংখ্যায় রূপান্তর
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    month = month_map.get(month_name[:3])
    if not month:
        print(f"অবৈধ মাস: {month_name}")
        sys.exit(1)
    
    day = int(day)
    year = int(year)
    
    # আজকের তারিখ
    today = datetime.now()
    
    # পৃষ্ঠার তারিখ ও আজকের তারিখ তুলনা
    if not (today.year == year and today.month == month and today.day == day):
        print(f"আজকের তারিখ ({today.strftime('%Y-%m-%d')}) এবং পৃষ্ঠার তারিখ ({year}-{month:02d}-{day:02d}) মিলছে না। স্ক্রিপ্ট বন্ধ হচ্ছে।")
        sys.exit(0)
    
    print(f"✅ তারিখ মিলেছে! ({today.strftime('%Y-%m-%d')})")
    print("🎯 মূল ডেটা প্রসেসিং স্ক্রিপ্ট চালু হচ্ছে...\n")
    
    # মূল ডেটা প্রসেসিং স্ক্রিপ্ট চালানো
    try:
        # main_processing_script.py ফাইল থেকে মূল স্ক্রিপ্ট চালানো
        subprocess.run([sys.executable, "main_processing_script.py"], check=True)
    except FileNotFoundError:
        print("❌ মূল প্রসেসিং স্ক্রিপ্ট (main_processing_script.py) খুঁজে পাওয়া যায়নি!")
        print("⚠️ ডিফল্ট ডেটা পার্সিং চলছে...\n")
        
        # ডিফল্ট ডেটা পার্সিং (যদি মূল স্ক্রিপ্ট না থাকে)
        parse_and_display_data(soup)
    except subprocess.CalledProcessError as e:
        print(f"❌ মূল স্ক্রিপ্ট চালাতে সমস্যা হয়েছে: {e}")
        sys.exit(1)

def parse_and_display_data(soup):
    """ডিফল্ট ডেটা পার্সিং ফাংশন"""
    
    # ডেটা টেবিল পার্স করা
    table = soup.find('table')
    if not table:
        print("পৃষ্ঠায় টেবিল খুঁজে পাওয়া যায়নি।")
        return
    
    rows = table.find_all('tr')
    
    # হেডার বের করা (প্রথম সারি)
    header_row = rows[0] if rows else None
    headers = []
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
    
    # যদি হেডার না থাকে, ডিফল্ট হেডার ব্যবহার
    if not headers or len(headers) < 4:
        headers = ['ট্রেডিং কোড', 'সর্বশেষ মূল্য', 'পরিবর্তন', '% পরিবর্তন']
    
    # ডেটা সংগ্রহ
    data_rows = []
    for row in rows[1:]:  # হেডার বাদ দিয়ে
        cols = row.find_all('td')
        if len(cols) >= 4:
            trading_code = cols[0].get_text(strip=True)
            ltp = cols[1].get_text(strip=True)
            change = cols[2].get_text(strip=True)
            percent_change = cols[3].get_text(strip=True)
            
            data_rows.append([trading_code, ltp, change, percent_change])
    
    # ফলাফল দেখানো
    print(f"\n{'ট্রেডিং কোড':<15} {'সর্বশেষ মূল্য':<15} {'পরিবর্তন':<15} {'% পরিবর্তন':<15}")
    print("-" * 60)
    
    # প্রথম ২০টি এন্ট্রি দেখানো
    for row in data_rows[:20]:
        change_symbol = "▲" if "+" in row[2] or (row[2].replace('.', '').replace('-', '').isdigit() and float(row[2]) > 0) else "▼" if "-" in row[2] else "◆"
        print(f"{row[0]:<15} {row[1]:<15} {row[2]:<15} {row[3]:<15} {change_symbol}")
    
    print(f"\n📊 মোট {len(data_rows)}টি এন্ট্রি পাওয়া গেছে।")
    
    # সংক্ষিপ্ত পরিসংখ্যান
    gainers = sum(1 for row in data_rows if '+' in row[3] or (row[3].replace('%', '').replace('.', '').replace('-', '').isdigit() and float(row[3].replace('%', '')) > 0))
    losers = sum(1 for row in data_rows if '-' in row[3] and row[3] != '0.00%')
    unchanged = len(data_rows) - gainers - losers
    
    print(f"\n📈 দর বেড়েছে: {gainers}টি")
    print(f"📉 দর কমেছে: {losers}টি")
    print(f"🔸 অপরিবর্তিত: {unchanged}টি")

# মূল প্রসেসিং স্ক্রিপ্ট (main_processing_script.py)
# এই ফাইলটি আলাদাভাবে তৈরি করতে হবে
MAIN_SCRIPT_CONTENT = '''
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import pandas as pd

def process_dse_data():
    """মূল ডেটা প্রসেসিং ফাংশন"""
    url = "https://dsebd.org/dseX_share.php"
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # ডেটা টেবিল পার্স করা
    table = soup.find('table')
    rows = table.find_all('tr')
    
    # ডেটা সংগ্রহ
    data = []
    for row in rows[1:]:
        cols = row.find_all('td')
        if len(cols) >= 7:  # সম্পূর্ণ ডেটা
            trading_code = cols[0].get_text(strip=True)
            ltp = cols[1].get_text(strip=True)
            high = cols[2].get_text(strip=True)
            low = cols[3].get_text(strip=True)
            close = cols[4].get_text(strip=True)
            ycp = cols[5].get_text(strip=True)
            change = cols[6].get_text(strip=True)
            percent_change = cols[7].get_text(strip=True)
            value = cols[8].get_text(strip=True) if len(cols) > 8 else ''
            volume = cols[9].get_text(strip=True) if len(cols) > 9 else ''
            
            data.append({
                'ট্রেডিং কোড': trading_code,
                'সর্বশেষ মূল্য': ltp,
                'দিনের সর্বোচ্চ': high,
                'দিনের সর্বনিম্ন': low,
                'ক্লোজ মূল্য': close,
                'গতকাল ক্লোজ': ycp,
                'পরিবর্তন': change,
                '% পরিবর্তন': percent_change,
                'ট্রেড ভ্যালু': value,
                'ভলিউম': volume
            })
    
    # CSV ফাইলে সংরক্ষণ
    filename = f"dse_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['ট্রেডিং কোড', 'সর্বশেষ মূল্য', 'দিনের সর্বোচ্চ', 'দিনের সর্বনিম্ন', 
                     'ক্লোজ মূল্য', 'গতকাল ক্লোজ', 'পরিবর্তন', '% পরিবর্তন', 'ট্রেড ভ্যালু', 'ভলিউম']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"✅ ডেটা {filename} ফাইলে সংরক্ষণ করা হয়েছে")
    
    # টপ গেইনার/লুজার বিশ্লেষণ
    df = pd.DataFrame(data)
    
    # % পরিবর্তনকে সংখ্যায় রূপান্তর
    df['% পরিবর্তন (সংখ্যা)'] = df['% পরিবর্তন'].str.replace('%', '').astype(float)
    
    # টপ ১০ গেইনার
    top_gainers = df.nlargest(10, '% পরিবর্তন (সংখ্যা)')[['ট্রেডিং কোড', 'সর্বশেষ মূল্য', '% পরিবর্তন']]
    print("\\n🏆 টপ ১০ গেইনার:")
    print(top_gainers.to_string(index=False))
    
    # টপ ১০ লুজার
    top_losers = df.nsmallest(10, '% পরিবর্তন (সংখ্যা)')[['ট্রেডিং কোড', 'সর্বশেষ মূল্য', '% পরিবর্তন']]
    print("\\n📉 টপ ১০ লুজার:")
    print(top_losers.to_string(index=False))
    
    # সেক্টরওয়াইজ বিশ্লেষণ (যদি ইচ্ছে থাকে)
    # এখানে আপনি সেক্টরভিত্তিক বিশ্লেষণ যোগ করতে পারেন

if __name__ == "__main__":
    process_dse_data()
'''

if __name__ == "__main__":
    # প্রথমে মূল স্ক্রিপ্ট ফাইল তৈরি করা (যদি না থাকে)
    import os
    if not os.path.exists('main_processing_script.py'):
        with open('main_processing_script.py', 'w', encoding='utf-8') as f:
            f.write(MAIN_SCRIPT_CONTENT)
        print("📝 মূল প্রসেসিং স্ক্রিপ্ট (main_processing_script.py) তৈরি করা হয়েছে")
    
    check_date_and_run_main_script()