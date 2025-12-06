import pandas as pd
import os

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
input_path = "./csv/mongodb.csv"
output_path1 = "./csv/swing_buy.csv"
output_path2 = "./output/ai_signal/swing_buy.csv"

os.makedirs(os.path.dirname(output_path1), exist_ok=True)
os.makedirs(os.path.dirname(output_path2), exist_ok=True)

# ---------------------------------------------------------
# Clear old results (with consistent columns)
# ---------------------------------------------------------
empty_cols = ["no", "date", "symbol", "buy", "SL", "diff"]
empty_df = pd.DataFrame(columns=empty_cols)
empty_df.to_csv(output_path1, index=False)
empty_df.to_csv(output_path2, index=False)

# ---------------------------------------------------------
# Load & validate
# ---------------------------------------------------------
if not os.path.exists(input_path):
    print("❌ mongodb.csv not found!")
    exit()

df = pd.read_csv(input_path)

required_cols = ["date", "symbol", "close", "high", "low"]
for col in required_cols:
    if col not in df.columns:
        raise Exception(f"Column '{col}' missing in mongodb.csv")

df = df.dropna(subset=["date"])
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

results = []

# ---------------------------------------------------------
# Per-symbol latest pattern check
# ---------------------------------------------------------
for symbol, group in df.groupby("symbol", sort=False):
    group = group.sort_values("date").reset_index(drop=True)
    if len(group) < 5:
        continue

    # Get last 5 bars: A (latest), B, C, D, E (oldest)
    A, B, C, D, E = [group.iloc[-i] for i in range(1, 6)]

    buy = SL = None

    # Logic 1
    if (A["close"] > B["high"] and
        B["low"] < C["low"] and
        B["high"] < C["high"] and
        C["high"] < D["high"] and
        C["low"] < D["low"]):
        buy, SL = A["close"], B["low"]

    # Logic 2
    elif (A["close"] > B["high"] and
          B["high"] < C["high"] and
          B["low"] > C["low"] and
          C["high"] < D["high"] and
          C["low"] < D["low"] and
          D["high"] < E["high"] and
          D["low"] < E["low"]):
        buy, SL = A["close"], C["low"]

    if buy is not None:
        results.append({
            "date": A["date"],
            "symbol": symbol,
            "buy": round(buy, 4),
            "SL": round(SL, 4)
        })

# ---------------------------------------------------------
# Create DataFrame, add diff, sort, then number
# ---------------------------------------------------------
if results:
    result_df = pd.DataFrame(results)
    result_df["diff"] = (result_df["buy"] - result_df["SL"]).round(4)
    # Sort by diff ascending → smallest gap first
    result_df = result_df.sort_values("diff").reset_index(drop=True)
    # Add 'no' after sorting
    result_df.insert(0, "no", range(1, len(result_df) + 1))
    # Format date
    result_df["date"] = result_df["date"].dt.strftime("%Y-%m-%d")
else:
    result_df = pd.DataFrame(columns=["no", "date", "symbol", "buy", "SL", "diff"])

# ---------------------------------------------------------
# Save (both paths)
# ---------------------------------------------------------
result_df.to_csv(output_path1, index=False)
result_df.to_csv(output_path2, index=False)

print(f"✅ swing_buy.csv updated with {len(result_df)} signals (sorted by diff ↑).")