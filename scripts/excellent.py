import pandas as pd
import os

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
source_path = './csv/liquidity.csv'
csv_path = './csv/excellent.csv'

# Ensure output dirs exist
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

# ---------------------------------------------------------
# 🔒 SAFE LOAD & VALIDATE
# ---------------------------------------------------------
if not os.path.exists(source_path):
    print(f"❌ Source file not found: {source_path}")
    exit(1)

try:
    df = pd.read_csv(source_path)
    print(f"✅ Loaded {len(df)} rows from {source_path}")
except Exception as e:
    print(f"❌ Failed to read {source_path}: {e}")
    exit(1)

# Validate required column
if 'liquidity_rating' not in df.columns:
    print(f"⚠️ 'liquidity_rating' column missing in {source_path} → Columns: {list(df.columns)}")
    exit(1)

# ---------------------------------------------------------
# ✅ Filter: 'Excellent' (case-insensitive, robust)
# ---------------------------------------------------------
# Handle NaN and non-string values safely
mask = df['liquidity_rating'].astype(str).str.strip().str.lower() == 'excellent'
filtered = df[mask].copy()

print(f"✅ Filtered {len(filtered)} / {len(df)} stocks with liquidity_rating = 'Excellent'")

# ---------------------------------------------------------
# 🔍 Debug: Show top 5 if none found
# ---------------------------------------------------------
if len(filtered) == 0:
    print("❓ No 'Excellent' stocks found. Top liquidity ratings in data:")
    top_ratings = df['liquidity_rating'].value_counts().head(5)
    for rating, count in top_ratings.items():
        print(f"   • {rating}: {count} stocks")
    # Optional: Still save empty file (for pipeline continuity)
    empty_df = pd.DataFrame(columns=df.columns)
    empty_df.to_csv(csv_path, index=False)
else:
    # Save
    filtered.to_csv(csv_path, index=False)
    print(f"📁 Saved to:\n   → {csv_path}")

# ---------------------------------------------------------
# ✅ Optional: Update system-wide liquidity flag
# ---------------------------------------------------------
# For integration with your main system (e.g., trade_stock.csv)
excellent_symbols = filtered['symbol'].str.upper().tolist()
with open('./csv/excellent_symbols.txt', 'w') as f:
    f.write('\n'.join(excellent_symbols))
print(f"🔖 {len(excellent_symbols)} Excellent symbols saved to ./csv/excellent_symbols.txt")