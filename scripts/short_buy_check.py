# short_buy_check.py
# Extract specific columns from short_buy.csv

import pandas as pd
import os

# =========================
# Configuration
# =========================
SHORT_BUY_FILE = './csv/short_buy.csv'
SHORT_BUY_CHECK_FILE = './csv/short_buy_check.csv'

# =========================
# Load Data
# =========================
print("=" * 80)
print("📊 SHORT BUY CHECK - COLUMN EXTRACTOR")
print("=" * 80)

# Check if file exists
if not os.path.exists(SHORT_BUY_FILE):
    print(f"❌ Error: {SHORT_BUY_FILE} not found!")
    print("Please run short_buy.py first")
    exit(1)

# Read data
short_buy_df = pd.read_csv(SHORT_BUY_FILE)

print(f"\n📂 Input file: {SHORT_BUY_FILE}")
print(f"📂 Records found: {len(short_buy_df)}")

# =========================
# Extract Required Columns
# =========================
# Select only symbol, date, high, low columns
required_columns = ['symbol', 'date', 'high', 'low']
check_df = short_buy_df[required_columns].copy()

# Add serial number (no) column at the front
check_df.insert(0, 'no', range(1, len(check_df) + 1))

# =========================
# Save Output
# =========================
check_df.to_csv(SHORT_BUY_CHECK_FILE, index=False)

print(f"\n✅ Output saved to: {SHORT_BUY_CHECK_FILE}")
print(f"📋 Columns: {list(check_df.columns)}")
print(f"📊 Total records: {len(check_df)}")

# =========================
# Display Summary
# =========================
print("\n" + "=" * 80)
print("📋 SHORT BUY CHECK DATA")
print("=" * 80)

if len(check_df) > 0:
    print(check_df.to_string(index=False))
else:
    print("⚠️ No data found")

print("\n" + "=" * 80)
print("✅ DONE!")
print("=" * 80)
