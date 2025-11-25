import pandas as pd
import os

# Input and output paths
input_path = "./csv/mongodb.csv"
output_path1 = "./csv/gape.csv"
output_path2 = "./output/ai_signal/gape.csv"

# Read CSV
df = pd.read_csv(input_path)

# Ensure sorted by symbol and date
df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)

results = {}

# 👉 প্রথমে gape.csv থেকে cutoff date বের করা
try:
    gape_df = pd.read_csv(output_path1)
    if not gape_df.empty:
        cutoff_date = pd.to_datetime(gape_df.iloc[0]["A_row_date"])
    else:
        cutoff_date = None
except FileNotFoundError:
    cutoff_date = None

# Group by symbol
for symbol, group in df.groupby("symbol"):
    group = group.reset_index(drop=True)

    last_row = group.iloc[-1]

    A_row = None

    # উপরের দিকে লুপ চালানো (শেষ থেকে শুরু করে)
    for i in range(len(group)-2, 0, -1):
        current_row = group.iloc[i]
        prev_row = group.iloc[i-1]

        current_date = pd.to_datetime(current_row["date"])

        # ✅ নতুন লিমিট: gape.csv এর ১ নাম্বার A_row_date পর্যন্ত
        if cutoff_date is not None and current_date < cutoff_date:
            break

        # শর্ত: current_row এর high < prev_row এর low - 0.10
        if current_row["high"] < prev_row["low"] - 0.10:
            below_rows = group.iloc[i+1:]
            # ✅ নতুন শর্ত: নিচের সব row-এর high < current_row এর high
            if (below_rows["high"] < current_row["high"]).all():
                A_row = current_row
                results[symbol] = {
                    "symbol": symbol,
                    "last_row_date": last_row["date"],
                    "last_row_high": last_row["high"],
                    "last_row_low": last_row["low"],
                    "last_row_close": last_row["close"],
                    "A_row_date": A_row["date"],
                    "A_row_high": A_row["high"],
                    "A_row_low": A_row["low"],
                    "A_row_close": A_row["close"],
                    "B_row_date": prev_row["date"],
                    "B_row_low": prev_row["low"]
                }

# Convert results to DataFrame
result_df = pd.DataFrame(results.values())

# Convert date columns to datetime
if not result_df.empty:
    result_df["last_row_date"] = pd.to_datetime(result_df["last_row_date"])
    result_df["A_row_date"] = pd.to_datetime(result_df["A_row_date"])
    result_df["B_row_date"] = pd.to_datetime(result_df["B_row_date"])

    # Calculate difference
    result_df["date_diff"] = (result_df["last_row_date"] - result_df["A_row_date"]).dt.days

    # ✅ Sort by date_diff descending, then last_row_date descending
    result_df = result_df.sort_values(by=["date_diff", "last_row_date"], ascending=[False, False]).reset_index(drop=True)

    

# Ensure output directories exist
os.makedirs(os.path.dirname(output_path1), exist_ok=True)
os.makedirs(os.path.dirname(output_path2), exist_ok=True)

# Save to both paths
result_df.to_csv(output_path1, index=False)
result_df.to_csv(output_path2, index=False)

print("✅ Filtered data saved to gape.csv successfully!")