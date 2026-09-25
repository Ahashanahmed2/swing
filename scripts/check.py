# debug_ltp.py — v2
import requests, re, json
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
html = r.text

print("=" * 70)
print("BASIC CHECKS")
print("=" * 70)
print("HTTP status:", r.status_code)
print("HTML length:", len(html))
print("Has '<table':", '<table' in html)
print("Has 'TRADING CODE':", 'TRADING CODE' in html)
print("Has '1JANATAMF':", '1JANATAMF' in html)
print("Has 'Market closed':", 'Market closed' in html)
print("Has 'Market open':", 'Market open' in html)
print("Has 'tickerInitial':", 'tickerInitial' in html)
print("Has '__next_f':", '__next_f' in html)

# ─────────────────────────────────────────────
# METHOD 1: HTML টেবিল
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("METHOD 1: HTML <table> parse")
print("=" * 70)
soup = BeautifulSoup(html, 'html.parser')
tables = soup.find_all('table')
print(f"Tables found: {len(tables)}")
for i, t in enumerate(tables[:5]):
    rows = t.find_all('tr')
    print(f"  Table {i}: {len(rows)} rows")
    for j, row in enumerate(rows[:3]):
        cells = row.find_all(['td', 'th'])
        texts = [c.get_text(strip=True)[:20] for c in cells]
        print(f"    Row {j}: {len(cells)} cells | {texts}")

method1_data = {}
for table in tables:
    for row in table.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) < 3:
            continue
        a = cells[1].find('a')
        symbol = (a.get_text(strip=True) if a else cells[1].get_text(strip=True))
        if not symbol or len(symbol) < 2:
            continue
        try:
            ltp = float(cells[2].get_text(strip=True).replace(',', ''))
            if 0 < ltp < 50000:
                method1_data[symbol.upper().strip()] = ltp
        except (ValueError, IndexError):
            continue
print(f"→ Method 1 symbols: {len(method1_data)}")
if method1_data:
    print(f"   Sample: {list(method1_data.items())[:5]}")


# ─────────────────────────────────────────────
# METHOD 2: tickerInitial JSON — full entries
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("METHOD 2: tickerInitial JSON — FULL DATA")
print("=" * 70)
method2_data = {}
method2_entries = []

idx = html.find('tickerInitial')
print(f"tickerInitial index: {idx}")

if idx > -1:
    # Context দেখি
    print(f"\nContext (±50 to +250):")
    print(html[max(0, idx - 50): idx + 250])
    print()

try:
    m = re.search(r'"tickerInitial"\s*:\s*(\[[^\]]*\])', html)
    if not m:
        m = re.search(r'tickerInitial.{0,5}?(\[\{.*?\}\])', html)

    if m:
        raw = m.group(1)
        try:
            raw_decoded = raw.encode('utf-8').decode('unicode_escape')
        except Exception:
            raw_decoded = raw

        try:
            tickers = json.loads(raw_decoded)
            print(f"✅ Parsed {len(tickers)} ticker entries")
        except Exception as e:
            print(f"❌ JSON parse error: {e}")
            tickers = []

        # প্রতিটি entry-এর কী কী ফিল্ড আছে তা দেখি
        if tickers:
            print(f"\n📋 ALL FIELDS present in first entry:")
            print(json.dumps(tickers[0], indent=2, ensure_ascii=False))

            print(f"\n📋 First 10 entries (raw):")
            for i, t in enumerate(tickers[:10]):
                print(f"  {i+1}. {t}")

            print(f"\n📋 Last 5 entries (raw):")
            for i, t in enumerate(tickers[-5:]):
                print(f"  {len(tickers)-5+i+1}. {t}")

        # সব entry-এর key গুলো collect করি — কোন ফিল্ড আছে বুঝতে
        all_keys = set()
        for t in tickers:
            all_keys.update(t.keys())
        print(f"\n🔑 All distinct keys across entries: {sorted(all_keys)}")

        # LTP extract
        for t in tickers:
            sym = t.get('code')
            price = t.get('price')
            if not sym:
                continue
            try:
                ltp = float(str(price).replace(',', ''))
                if 0 < ltp < 50000:
                    method2_data[sym.upper().strip()] = ltp
                    method2_entries.append({
                        'symbol': sym.upper().strip(),
                        'ltp': ltp,
                        'change': t.get('change'),
                        'delta': t.get('delta'),
                    })
            except (ValueError, TypeError):
                continue

        print(f"\n✅ Extracted {len(method2_data)} symbols with LTP")
        print(f"\n📊 Sample with all fields (first 5):")
        for e in method2_entries[:5]:
            print(f"  {e}")

        print(f"\n📊 Sample with all fields (last 5):")
        for e in method2_entries[-5:]:
            print(f"  {e}")

        # Any entries with high/low/volume?
        fields_with_hlv = [t for t in tickers if any(k in t for k in ['high', 'low', 'volume', 'value', 'trades'])]
        print(f"\n🔍 Entries containing high/low/volume/etc: {len(fields_with_hlv)}")
        if fields_with_hlv:
            print(f"   Example: {fields_with_hlv[0]}")

    else:
        print("❌ Regex didn't match tickerInitial")

except Exception as e:
    print(f"❌ Method 2 error: {e}")


# ─────────────────────────────────────────────
# METHOD 3: __next_f থেকে সমস্ত JSON-like data খুঁজি
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("METHOD 3: Searching for other JSON arrays in __next_f")
print("=" * 70)

# সব সম্ভাব্য data array-র নাম যা Next.js SSR-এ আসে
candidates = [
    'tickerInitial', 'latestSharePrice', 'boardData', 'sharePrice',
    'initialData', 'tableData', 'marketData', 'rows', 'data',
    'latest_price', 'share_price', 'instruments'
]
for name in candidates:
    pattern = rf'"{name}"\s*:\s*(\[[^\]]*\])'
    m = re.search(pattern, html)
    if m:
        try:
            raw = m.group(1).encode('utf-8').decode('unicode_escape')
            arr = json.loads(raw)
            print(f"✅ '{name}' found: {len(arr)} entries")
            if arr and isinstance(arr[0], dict):
                print(f"   Keys: {list(arr[0].keys())}")
                print(f"   First: {arr[0]}")
        except Exception as e:
            print(f"⚠️ '{name}' matched but parse failed: {e}")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Method 1 (HTML table):       {len(method1_data)} symbols")
print(f"Method 2 (tickerInitial):    {len(method2_data)} symbols")
print(f"Entries with full fields:    {len(method2_entries)}")

if method2_entries:
    print(f"\n✅ BEST DATA SOURCE: tickerInitial")
    print(f"   Fields available per symbol: {list(method2_entries[0].keys())}")
    print(f"\n   Full first 5 entries:")
    for e in method2_entries[:5]:
        print(f"     {e}")
