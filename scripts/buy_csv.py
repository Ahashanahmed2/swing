import pandas as pd
import os

# -----------------------------
# Input buy files
# -----------------------------
BUY_FILES = {
    "rsi": "./csv/rsi_30_buy.csv",
    "short": "./csv/short_buy.csv",
    "swing": "./csv/swing_buy.csv",
    "gape": "./csv/gape_buy.csv",
}

UPTREND_FILE = "./csv/uptrend.csv"
DOWNTREND_FILE = "./csv/downtrend.csv"

OUTPUT_FILE = "./csc/buy.csv"

# -----------------------------
# Base columns
# -----------------------------
BASE_COLUMNS = [
    "No", "date", "symbol", "buy", "SL", "tp",
    "position_size", "exposure_bdt",
    "actual_risk_bdt", "diff", "RRR"
]

# -----------------------------
# Load trend symbols (CLEAN)
# -----------------------------
def load_trend_symbols(path):
    if not os.path.exists(path):
        print(f"⚠️ Trend file not found: {path}")
        return set()

    df = pd.read_csv(path)

    if "symbol" not in df.columns:
        print(f"⚠️ 'symbol' column missing in {path}")
        return set()

    return set(
        df["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

uptrend_symbols = load_trend_symbols(UPTREND_FILE)
downtrend_symbols = load_trend_symbols(DOWNTREND_FILE)

# -----------------------------
# Process buy files
# -----------------------------
all_rows = []

for file_key, file_path in BUY_FILES.items():
    if not os.path.exists(file_path):
        print(f"⚠️ Buy file not found: {file_path}")
        continue

    df = pd.read_csv(file_path)

    if "symbol" not in df.columns:
        print(f"⚠️ 'symbol' column missing in {file_path}")
        continue

    # Ensure all base columns exist
    for col in BASE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[BASE_COLUMNS]

    # Clean symbol
    df["symbol"] = (
        df["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # File source
    df["file"] = file_key

    # Detect trend
    def detect_trend(symbol):
        if symbol in uptrend_symbols:
            return "uptrend"
        elif symbol in downtrend_symbols:
            return "downtrend"
        else:
            return "sideways"

    df["trend"] = df["symbol"].apply(detect_trend)

    all_rows.append(df)

# -----------------------------
# Merge all
# -----------------------------
if all_rows:
    final_df = pd.concat(all_rows, ignore_index=True)
else:
    final_df = pd.DataFrame(columns=BASE_COLUMNS + ["file", "trend"])

# -----------------------------
# Safe type conversion
# -----------------------------
if not final_df.empty:
    # Date handling (NaT → very old date)
    final_df["date"] = pd.to_datetime(
        final_df["date"], errors="coerce"
    ).fillna(pd.Timestamp("1970-01-01"))

    # RRR handling (NaN → very low)
    final_df["RRR"] = pd.to_numeric(
        final_df["RRR"], errors="coerce"
    ).fillna(-999)

# -----------------------------
# Trend-wise sorting
# -----------------------------
up_df = final_df[final_df["trend"] == "uptrend"].copy()
side_df = final_df[final_df["trend"] == "sideways"].copy()
down_df = final_df[final_df["trend"] == "downtrend"].copy()

# uptrend → latest date → highest RRR
if not up_df.empty:
    up_df = up_df.sort_values(
        by=["date", "RRR"],
        ascending=[False, False]
    )

# sideways → highest RRR
if not side_df.empty:
    side_df = side_df.sort_values(
        by=["RRR"],
        ascending=[False]
    )

# downtrend → highest RRR
if not down_df.empty:
    down_df = down_df.sort_values(
        by=["RRR"],
        ascending=[False]
    )

# -----------------------------
# Final merge (priority order)
# -----------------------------
final_df = pd.concat(
    [up_df, side_df, down_df],
    ignore_index=True
)

# -----------------------------
# Save output
# -----------------------------
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
final_df.to_csv(OUTPUT_FILE, index=False)

print("✅ Final sorted buy.csv created")
print(f"📂 Path: {OUTPUT_FILE}")
print(f"📊 Total records: {len(final_df)}")
print(
    f"🔼 Uptrend: {len(up_df)} | ➖ Sideways: {len(side_df)} | 🔽 Downtrend: {len(down_df)}"
)