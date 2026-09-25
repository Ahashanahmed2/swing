# debug_ltp.py
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

print("=" * 60)
print("HTTP status:", r.status_code)
print("HTML length:", len(html))
print("Has '<table':", '<table' in html)
print("Has 'TRADING CODE':", 'TRADING CODE' in html)
print("Has '1JANATAMF':", '1JANATAMF' in html)
print("Has 'Market closed':", 'Market closed' in html)
print("Has 'Market open':", 'Market open' in html)
print("Has 'tickerInitial':", 'tickerInitial' in html)
print("Has '__next_f':", '__next_f' in html)
print("=" * 60)

# ─────────────────────────────────────────────
# Method 1: HTML টেবিল
# ─────────────────────────────────────────────
print("\n### METHOD 1: HTML <table> parse")
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
# Method 2: tickerInitial JSON
# ─────────────────────────────────────────────
print("\n### METHOD 2: tickerInitial JSON")
method2_data = {}
idx = html.find('tickerInitial')
print(f"tickerInitial index: {idx}")
if idx > -1:
    print(f"Context (±100): ...{html[max(0,idx-50):idx+200]}...")

try:
    # ["tickerInitial":[{...},{...}],"alerts":...]
    m = re.search(r'"tickerInitial"\s*:\s*(\[[^\]]*\])', html)
    if not m:
        # backup: \u0022 এ escaped হলে
        m = re.search(r'tickerInitial.{0,5}?(\[\{.*?\}\])', html)
    if m:
        raw = m.group(1)
        # Next.js-এ \u003c, \u0026 থাকতে পারে — unicode_escape দিয়ে decode
        try:
            raw_decoded = raw.encode('utf-8').decode('unicode_escape')
        except Exception:
            raw_decoded = raw
        try:
            tickers = json.loads(raw_decoded)
        except Exception as e:
            print(f"   JSON parse error: {e}")
            tickers = []
        print(f"   Parsed tickers: {len(tickers)}")
        for t in tickers:
            sym = t.get('code')
            price = t.get('price')
            if not sym:
                continue
            try:
                ltp = float(str(price).replace(',', ''))
                if 0 < ltp < 50000:
                    method2_data[sym.upper().strip()] = ltp
            except (ValueError, TypeError):
                continue
    else:
        print("   Regex didn't match tickerInitial")
except Exception as e:
    print(f"   Method 2 error: {e}")

print(f"→ Method 2 symbols: {len(method2_data)}")
if method2_data:
    print(f"   Sample: {list(method2_data.items())[:5]}")


# ─────────────────────────────────────────────
# Method 3: Raw regex — escaped টেবিল row থেকে
# ─────────────────────────────────────────────
print("\n### METHOD 3: Regex on raw HTML")
method3_data = {}
# দুটো pattern try — escaped এবং non-escaped
patterns = [
    # non-escaped: >SYMBOL</a></td><td ...>LTP<
    re.compile(r'>([A-Z0-9&\-\.\(\)]+)</a>\s*</td>\s*<td[^>]*>\s*([\d,]+\.\d+)\s*<'),
    # escaped: \u003eSYMBOL\u003c/a\u003e...\u003c
    re.compile(r'\\u003e([A-Z0-9&\-\.\(\)]+)\\u003c/a\\u003e[^<]{0,200}?([\d,]+\.\d+)'),
    # /company/SYMBOL">SYMBOL  pattern
    re.compile(r'/company/([A-Z0-9&\-\.\(\)]+)"[^>]*>\s*([A-Z0-9&\-\.\(\)]+)\s*<[^>]*>[^<]*<[^>]*>\s*([\d,]+\.\d+)'),
]
for i, pat in enumerate(patterns):
    hits = pat.findall(html)
    print(f"   Pattern {i}: {len(hits)} matches")
    if hits:
        print(f"     First 3: {hits[:3]}")

# pattern 0 → (symbol, ltp)
for sym, ltp_s in patterns[0].findall(html):
    try:
        ltp = float(ltp_s.replace(',', ''))
        if 0 < ltp < 50000:
            method3_data[sym.upper().strip()] = ltp
    except ValueError:
        continue
print(f"→ Method 3 symbols: {len(method3_data)}")


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Method 1 (HTML table):      {len(method1_data)} symbols")
print(f"Method 2 (tickerInitial):   {len(method2_data)} symbols")
print(f"Method 3 (regex escaped):   {len(method3_data)} symbols")

best = max([method1_data, method2_data, method3_data], key=len)
print(f"\n✅ BEST method: {len(best)} symbols")
if best:
    print(f"   Sample entries: {list(best.items())[:5]}")


# ─────────────────────────────────────────────
# HTML snippet around "TRADING CODE"
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("HTML CONTEXT around 'TRADING CODE'")
print("=" * 60)
tc_idx = html.find('TRADING CODE')
if tc_idx > -1:
    print(f"Index: {tc_idx}")
    print(html[max(0, tc_idx - 300): tc_idx + 800])
else:
    print("Not found")

print("\n" + "=" * 60)
print("HTML CONTEXT around '1JANATAMF'")
print("=" * 60)
jt_idx = html.find('1JANATAMF')
if jt_idx > -1:
    print(f"Index: {jt_idx}")
    print(html[max(0, jt_idx - 300): jt_idx + 800])
else:
    print("Not found")
