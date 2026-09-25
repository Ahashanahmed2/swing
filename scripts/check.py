# debug_ltp.py
import requests, re
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = "https://new.dsebd.org/markets/latest-share-price"

s = requests.Session()
s.verify = False
s.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
})

r = s.get(URL, timeout=20)
print("HTTP status:", r.status_code)
print("HTML length:", len(r.text))
print("Has '<table':", '<table' in r.text)
print("Has 'TRADING CODE':", 'TRADING CODE' in r.text)
print("Has '1JANATAMF':", '1JANATAMF' in r.text)
print("Has 'Market closed':", 'Market closed' in r.text)
print("Has 'Market open':", 'Market open' in r.text)

# Count tables and rows
soup = BeautifulSoup(r.text, 'html.parser')
tables = soup.find_all('table')
print("Tables:", len(tables))
for i, t in enumerate(tables[:3]):
    rows = t.find_all('tr')
    print(f"  Table {i}: {len(rows)} rows")

# Show first 800 chars
print("\n--- First 800 chars ---")
print(r.text[:800])
