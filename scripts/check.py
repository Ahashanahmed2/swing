# debug_new_dse.py
import requests, re, json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = "https://new.dsebd.org/markets/latest-share-price"

# === TEST 1: Simple request ===
print("=" * 70)
print("TEST 1: simple requests.get")
print("=" * 70)
try:
    r = requests.get(URL, timeout=20, verify=False)
    print(f"Status: {r.status_code}")
    print(f"Length: {len(r.text)}")
    print(f"Server header: {r.headers.get('Server')}")
    print(f"CF-Ray: {r.headers.get('CF-Ray')}")
    print(f"Content-Type: {r.headers.get('Content-Type')}")
    print(f"First 500 chars:\n{r.text[:500]}")
    if 'tickerInitial' in r.text:
        print("✅ tickerInitial FOUND")
    else:
        print("❌ tickerInitial NOT found")
    if '<table' in r.text:
        print("✅ <table> FOUND")
    if 'TRADING CODE' in r.text:
        print("✅ 'TRADING CODE' FOUND")
    if 'Market closed' in r.text or 'Market open' in r.text:
        print("✅ Market status text FOUND")
    # Header time regex
    m = re.search(r'On\s+(\w+ \d{1,2}, \d{4})\s+at\s+(\d{1,2}:\d{2}\s*[AP]M)', r.text)
    print(f"Header time regex match: {m.group(0) if m else '❌ none'}")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")

# === TEST 2: With browser-like headers ===
print("\n" + "=" * 70)
print("TEST 2: requests with full browser headers")
print("=" * 70)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,'
              'image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9,bn;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Cache-Control': 'max-age=0',
}
try:
    s = requests.Session()
    s.verify = False
    r = s.get(URL, headers=headers, timeout=20)
    print(f"Status: {r.status_code}")
    print(f"Length: {len(r.text)}")
    print(f"tickerInitial: {'✅' if 'tickerInitial' in r.text else '❌'}")
    print(f"<table>:       {'✅' if '<table' in r.text else '❌'}")
    print(f"TRADING CODE:  {'✅' if 'TRADING CODE' in r.text else '❌'}")
    # Data dump (for inspection)
    print("\n--- First 2000 chars ---")
    print(r.text[:2000])
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")


# === TEST 3: curl_cffi (if installed) ===
print("\n" + "=" * 70)
print("TEST 3: curl_cffi impersonate chrome")
print("=" * 70)
try:
    from curl_cffi import requests as cf_requests
    r = cf_requests.get(URL, impersonate="chrome120", timeout=20)
    print(f"Status: {r.status_code}")
    print(f"Length: {len(r.text)}")
    print(f"tickerInitial: {'✅' if 'tickerInitial' in r.text else '❌'}")
    print(f"<table>:       {'✅' if '<table' in r.text else '❌'}")
    print(f"TRADING CODE:  {'✅' if 'TRADING CODE' in r.text else '❌'}")
    m = re.search(r'On\s+(\w+ \d{1,2}, \d{4})\s+at\s+(\d{1,2}:\d{2}\s*[AP]M)', r.text)
    print(f"Header time: {m.group(0) if m else '❌'}")
    # tickerInitial context
    idx = r.text.find('tickerInitial')
    if idx > -1:
        print(f"\ntickerInitial context:\n{r.text[idx:idx+400]}")
    # Table context
    idx2 = r.text.find('TRADING CODE')
    if idx2 > -1:
        print(f"\nTRADING CODE context:\n{r.text[max(0,idx2-200):idx2+800]}")
except ImportError:
    print("⚠️ curl_cffi not installed. Install: pip install curl_cffi")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")