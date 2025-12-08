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
# Clear old results (with full columns)
# ---------------------------------------------------------
full_cols = ["no", "date", "symbol", "buy", "SL", "tp", "diff", "RRR"]
empty_df = pd.DataFrame(columns=full_cols)
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

# Group by symbol for efficiency
mongo_groups = df.groupby("symbol", sort=False)

results = []

# ---------------------------------------------------------
# Per-symbol latest pattern check (5-bar logic)
# ---------------------------------------------------------
for symbol, group in mongo_groups:
    group = group.sort_values("date").reset_index(drop=True)
    if len(group) < 5:
        continue

    # Get last 5 bars: A (latest), B, C, D, E (oldest)
    A, B, C, D, E = [group.iloc[-i] for i in range(1, 6)]

    buy = SL = SL_source_row = None

    # Logic 1: SL = B["low"]
    if (A["close"] > B["high"] and
        B["low"] < C["low"] and
        B["high"] < C["high"] and
        C["high"] < D["high"] and
        C["low"] < D["low"]):
        buy, SL = A["close"], B["low"]
        SL_source_row = B  # 🔑 B is SL source

    # Logic 2: SL = C["low"]
    elif (A["close"] > B["high"] and
          B["high"] < C["high"] and
          B["low"] > C["low"] and
          C["high"] < D["high"] and
          C["low"] < D["low"] and
          D["high"] < E["high"] and
          D["low"] < E["low"]):
        buy, SL = A["close"], C["low"]
        SL_source_row = C  # 🔑 C is SL source

    if buy is None:
        continue

    # 🔑 NEW: Find tp — scan BACKWARD from SL_source_row in FULL history (unlimited)
    tp = None
    try:
        # Find index of SL_source_row in FULL group (not just last 5)
        sl_idx = group[group["date"] == SL_source_row["date"]].index[0]
    except IndexError:
        # Fallback: closest match
        sl_idx = (abs(group["date"] - SL_source_row["date"])).idxmin()

    # Scan backward: i = index of candidate 's' (need sb = i-2, sa = i-1, s = i)
    for i in range(sl_idx - 1, 1, -1):  # unlimited backward
        try:
            sb = group.iloc[i - 2]
            sa = group.iloc[i - 1]
            s  = group.iloc[i]
        except IndexError:
            break

        # Ensure chronological order
        if not (sb["date"] < sa["date"] < s["date"] < SL_source_row["date"]):
            continue

        # ✅ tp condition: s.high > sa.high and sa.high >= sb.high
        if (s["high"] > sa["high"]) and (sa["high"] >= sb["high"]):
            tp = s["high"]
            break

    if tp is None:
        continue  # skip if no valid tp

    # Append
    results.append({
        "date": A["date"],
        "symbol": symbol,
        "buy": buy,
        "SL": SL,
        "tp": tp,
    })

# ---------------------------------------------------------
# Create DataFrame, compute diff & RRR, filter, sort
# ---------------------------------------------------------
if results:
    result_df = pd.DataFrame(results)
    result_df["buy"] = pd.to_numeric(result_df["buy"], errors="coerce")
    result_df["SL"] = pd.to_numeric(result_df["SL"], errors="coerce")
    result_df["tp"] = pd.to_numeric(result_df["tp"], errors="coerce")

    # Compute diff & RRR
    result_df["diff"] = (result_df["buy"] - result_df["SL"]).round(4)
    result_df["RRR"] = ((result_df["tp"] - result_df["buy"]) / (result_df["buy"] - result_df["SL"])).round(2)

    # ✅ Filter: only valid, positive RRR signals
    result_df = result_df[
        (result_df["buy"] > result_df["SL"]) &
        (result_df["tp"] > result_df["buy"]) &
        (result_df["RRR"] > 0)
    ].reset_index(drop=True)

    if len(result_df) > 0:
        # Sort: highest RRR first → then smallest diff (risk) first
        result_df = result_df.sort_values(["RRR", "diff"], ascending=[False, True]).reset_index(drop=True)
        result_df.insert(0, "no", range(1, len(result_df) + 1))
        result_df["date"] = result_df["date"].dt.strftime("%Y-%m-%d")

        # Final column order
        result_df = result_df[["no", "date", "symbol", "buy", "SL", "tp", "diff", "RRR"]]
    else:
        result_df = pd.DataFrame(columns=["no", "date", "symbol", "buy", "SL", "tp", "diff", "RRR"])
else:
    result_df = pd.DataFrame(columns=["no", "date", "symbol", "buy", "SL", "tp", "diff", "RRR"])

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
result_df.to_csv(output_path1, index=False)
result_df.to_csv(output_path2, index=False)

print(f"✅ swing_buy.csv updated with {len(result_df)} signals:")
if len(result_df) > 0:
    print(f"   📈 Top RRR: {result_df['RRR'].max():.2f} | Avg RRR: {result_df['RRR'].mean():.2f}")
    print(f"   📉 Min diff: {result_df['diff'].min():.4f} | Max diff: {result_df['diff'].max():.4f}")