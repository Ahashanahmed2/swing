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
# Clear old results
# ---------------------------------------------------------
empty_df = pd.DataFrame(columns=["no", "date", "symbol", "buy", "SL"])
empty_df.to_csv(output_path1, index=False, newline='')  # ← newline=''
empty_df.to_csv(output_path2, index=False, newline='')

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
for symbol, group in df.groupby("symbol", sort=False):  # ← sort=False
    group = group.sort_values("date").reset_index(drop=True)
    if len(group) < 5:
        continue

    A, B, C, D, E = [group.iloc[-i] for i in range(1, 6)]  # clean & pythonic

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
            "no": len(results) + 1,
            "date": A["date"].strftime("%Y-%m-%d"),
            "symbol": symbol,
            "buy": round(buy, 4),
            "SL": round(SL, 4)
        })

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
result_df = pd.DataFrame(results)
result_df.to_csv(output_path1, index=False, newline='')  # ← newline=''
result_df.to_csv(output_path2, index=False, newline='')

print(f"✅ swing_buy.csv updated with {len(results)} signals.")