import pandas as pd
import os

# Input and output paths
input_path = "./csv/mongodb.csv"
output_path1 = "./csv/gape.csv"
output_path2 = "./output/ai_signal/gape.csv"
# Read CSV
df = pd.read_csv('./csv/mongodb.csv')

# Ensure sorted by symbol and date
df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)

results = []

# Group by symbol
for symbol, group in df.groupby("symbol"):
    group = group.reset_index(drop=True)

    # Check last row condition: last high < previous low - 0.10
    if len(group) >= 2:
        last_row = group.iloc[-1]
        prev_row = group.iloc[-2]

        if last_row["high"] < prev_row["low"] - 0.10:
            # Define A row
            A_row = last_row
            A_high = A_row["high"]
            A_index = group.index[-1]

            # Check if at least 3 rows exist below A_row
            if len(group) - (A_index + 1) >= 3:
                below_rows = group.iloc[A_index+1:]

                # Condition: all below rows high < A_high
                if (below_rows["high"] < A_high).all():
                    results.append({
                        "No": len(results) + 1,
                        "symbol": symbol,
                        "last_row_date": last_row["date"],
                        "last_row_high": last_row["high"],
                        "last_row_low": last_row["low"],
                        "last_row_close": last_row["close"],
                        "A_row_date": A_row["date"],
                        "A_row_high": A_row["high"],
                        "A_row_low": A_row["low"],
                        "A_row_close": A_row["close"]
                    })

# Convert results to DataFrame
result_df = pd.DataFrame(results)

# 👉 Convert date columns to datetime
result_df["last_row_date"] = pd.to_datetime(result_df["last_row_date"])
result_df["A_row_date"] = pd.to_datetime(result_df["A_row_date"])

# 👉 Calculate difference (last_row_date - A_row_date)
result_df["date_diff"] = (result_df["last_row_date"] - result_df["A_row_date"]).dt.days

# 👉 Sort by date_diff descending, then last_row_date descending
result_df = result_df.sort_values(by=["date_diff", "last_row_date"], ascending=[False, False]).reset_index(drop=True)

# 👉 Reassign serial No after sorting
result_df["No"] = range(1, len(result_df) + 1)

# Ensure output directories exist
os.makedirs(os.path.dirname(output_path1), exist_ok=True)
os.makedirs(os.path.dirname(output_path2), exist_ok=True)

# Save to both paths
result_df.to_csv(output_path1, index=False)
result_df.to_csv(output_path2, index=False)

print("Filtered data saved to gape.csv successfully!")