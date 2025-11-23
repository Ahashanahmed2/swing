import pandas as pd
import os

# Input and output paths
input_path = "./csv/mongodb.csv"
output_path1 = "./csv/gape.csv"
output_path2 = "./output/signal_ai/gape.csv"

# Read CSV
df = pd.read_csv(input_path)

# Ensure sorted by symbol and date
df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)

results = {}

# Group by symbol
for symbol, group in df.groupby("symbol"):
    group = group.reset_index(drop=True)

    last_row = group.iloc[-1]
    last_row_date = pd.to_datetime(last_row["date"])
    limit_date = last_row_date - pd.Timedelta(days=180)  # ৬ মাসের লিমিট

    A_row = None

    # উপরের দিকে লুপ চালানো
    for i in range(len(group)-2, 0, -1):
        current_row = group.iloc[i]
        prev_row = group.iloc[i-1]
        current_date = pd.to_datetime(current_row["date"])

        if current_date < limit_date:
            break

        # শর্ত: current_row এর high < prev_row এর low - 0.10
        if current_row["high"] < prev_row["low"] - 0.10:
            below_rows = group.iloc[i+1:]
            if (below_rows["high"] < current_row["high"]).all():
                # ✅ Gap validation
                if i+1 < len(group):
                    next_row = group.iloc[i+1]
                    gap_high = max(current_row["high"], next_row["high"])
                    gap_low = min(current_row["low"], next_row["low"])

                    # মাঝখানের row গুলো
                    between_rows = group.iloc[i+2:]
                    # যদি কোনো row-এর high gap range-এর মধ্যে পড়ে → বাতিল
                    if ((between_rows["high"] >= gap_low) & (between_rows["high"] <= gap_high)).any():
                        break  # symbol বাতিল হবে

                # সব শর্ত পূরণ হলে result-এ যোগ করুন
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
                    "A_row_close": A_row["close"]
                }

# Convert results to DataFrame
result_df = pd.DataFrame(results.values())

# Convert date columns to datetime
if not result_df.empty:
    result_df["last_row_date"] = pd.to_datetime(result_df["last_row_date"])
    result_df["A_row_date"] = pd.to_datetime(result_df["A_row_date"])

    # Calculate difference
    result_df["date_diff"] = (result_df["last_row_date"] - result_df["A_row_date"]).dt.days

    # Sort by date_diff descending, then last_row_date descending
    result_df = result_df.sort_values(by=["date_diff", "last_row_date"], ascending=[False, False]).reset_index(drop=True)
    result_df["No"] = range(1, len(result_df) + 1)

# Ensure output directories exist
os.makedirs(os.path.dirname(output_path1), exist_ok=True)
os.makedirs(os.path.dirname(output_path2), exist_ok=True)

# Save to both paths
result_df.to_csv(output_path1, index=False)
result_df.to_csv(output_path2, index=False)

print("✅ Filtered data saved to gape.csv successfully!")
