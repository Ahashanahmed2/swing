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

UPTREND_FILE = "./csv/uptrand.csv"
DOWNTREND_FILE = "./csv/downtrand.csv"

OUTPUT_FILE = "./output/ai-signal/buy.csv"

# -----------------------------
# Base columns
# -----------------------------
BASE_COLUMNS = [
    "No", "date", "symbol", "buy", "SL", "tp",
    "position_size", "exposure_bdt",
    "actual_risk_bdt", "diff", "RRR"
]

# -----------------------------
# Load trend symbols
# -----------------------------
def load_trend_symbols(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "symbol" in df.columns:
            return set(df["symbol"].astype(str))
    return set()

uptrend_symbols = load_trend_symbols(UPTREND_FILE)
downtrend_symbols = load_trend_symbols(DOWNTREND_FILE)

# -----------------------------
# Process buy files
# -----------------------------
all_rows = []

for file_key, file_path in BUY_FILES.items():
    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path)

    if "symbol" not in df.columns:
        continue

    # ensure base columns exist
    for col in BASE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[BASE_COLUMNS]

    # add file column
    df["file"] = file_key

    # detect trand
    def detect_trand(symbol):
        symbol = str(symbol)
        if symbol in uptrend_symbols:
            return "uptrand"
        elif symbol in downtrend_symbols:
            return "downtrand"
        else:
            return "sideways"

    df["trand"] = df["symbol"].apply(detect_trand)

    all_rows.append(df)

# -----------------------------
# Merge all
# -----------------------------
final_df = pd.concat(all_rows, ignore_index=True) if all_rows else \
           pd.DataFrame(columns=BASE_COLUMNS + ["file", "trand"])

# -----------------------------
# Format date & RRR
# -----------------------------
final_df["date"] = pd.to_datetime(final_df["date"], errors="coerce")
final_df["RRR"] = pd.to_numeric(final_df["RRR"], errors="coerce")

# -----------------------------
# Trend wise sorting
# -----------------------------
up_df = final_df[final_df["trand"] == "uptrand"].copy()
side_df = final_df[final_df["trand"] == "sideways"].copy()
down_df = final_df[final_df["trand"] == "downtrand"].copy()

# uptrand → latest date, then highest RRR
up_df = up_df.sort_values(
    by=["date", "RRR"],
    ascending=[False, False]
)

# sideways → highest RRR
side_df = side_df.sort_values(
    by=["RRR"],
    ascending=[False]
)

# downtrand → highest RRR
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

print("✅ Final sorted buy.csv created:", OUTPUT_FILE)