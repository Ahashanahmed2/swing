import pandas as pd
import os

# --------------------------------------------------
# Paths
# --------------------------------------------------
BUY_FILES = {
    "rsi": "./csv/rsi_30_buy.csv",
    "short": "./csv/short_buy.csv",
    "swing": "./csv/swing_buy.csv",
    "gape": "./csv/gape_buy.csv",
}

UPTREND_FILE = "./csv/uptrand.csv"
DOWNTREND_FILE = "./csv/downtrand.csv"

OUTPUT_FILE = "./output/ai_signal/buy.csv"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# --------------------------------------------------
# Required columns (buy files)
# --------------------------------------------------
BASE_COLUMNS = [
    "No", "date", "symbol", "buy", "SL", "tp",
    "position_size", "exposure_bdt",
    "actual_risk_bdt", "diff", "RRR"
]

# --------------------------------------------------
# Load trend files
# --------------------------------------------------
trend_map = {}

if os.path.exists(UPTREND_FILE):
    up_df = pd.read_csv(UPTREND_FILE)
    for s in up_df["symbol"].dropna().unique():
        trend_map[s] = "uptrend"

if os.path.exists(DOWNTREND_FILE):
    down_df = pd.read_csv(DOWNTREND_FILE)
    for s in down_df["symbol"].dropna().unique():
        trend_map[s] = "downtrend"

print(f"✅ Trend symbols loaded: {len(trend_map)}")

# --------------------------------------------------
# Process buy files
# --------------------------------------------------
final_rows = []

for file_tag, file_path in BUY_FILES.items():

    if not os.path.exists(file_path):
        print(f"⚠️ Buy file not found, skipped: {file_path}")
        continue

    print(f"📂 Processing: {file_path}")
    df = pd.read_csv(file_path)

    # Column validation
    missing_cols = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns in {file_path}: {missing_cols}")
        continue

    # Ensure RRR numeric
    df["RRR"] = pd.to_numeric(df["RRR"], errors="coerce")

    for _, row in df.iterrows():
        symbol = row["symbol"]

        # Skip if not in trend
        if symbol not in trend_map:
            continue

        new_row = {col: row[col] for col in BASE_COLUMNS}
        new_row["file"] = file_tag
        new_row["trand"] = trend_map[symbol]

        final_rows.append(new_row)

# --------------------------------------------------
# Create final DataFrame
# --------------------------------------------------
if not final_rows:
    print("\n❌ No matching buy signals found. buy.csv not created.")
    exit()

final_df = pd.DataFrame(final_rows)

# --------------------------------------------------
# Sort by best RRR (DESC)
# --------------------------------------------------
final_df = final_df.sort_values(
    by="RRR",
    ascending=False
).reset_index(drop=True)

# Reassign No column
final_df["No"] = range(1, len(final_df) + 1)

# --------------------------------------------------
# Reorder columns
# --------------------------------------------------
FINAL_COLUMNS = BASE_COLUMNS + ["file", "trand"]
final_df = final_df[FINAL_COLUMNS]

# --------------------------------------------------
# Save output
# --------------------------------------------------
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ buy.csv generated successfully!")
print(f"📁 Output path: {OUTPUT_FILE}")
print(f"📊 Total signals: {len(final_df)}")

# Show top 5
print("\n🔥 Top 5 signals by RRR:")
print(
    final_df[["No", "symbol", "RRR", "file", "trand"]]
    .head(5)
    .to_string(index=False)
)
