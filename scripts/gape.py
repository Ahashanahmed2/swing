import pandas as pd
import os

# Input and output paths
input_path = "./csv/mongodb.csv"
output_path1 = "./csv/gape.csv"
output_path2 = "./output/aisignal/gape.csv"

# Read CSV
df = pd.read_csv(input_path)

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
                        "lastrowdate": last_row["date"],
                        "lastrowhigh": last_row["high"],
                        "lastrowlow": last_row["low"],
                        "lastrowclose": last_row["close"],
                        "Arowdate": A_row["date"],
                        "Arowhigh": A_row["high"],
                        "Arowlow": A_row["low"],
                        "Arowclose": A_row["close"]
                    })

# Convert results to DataFrame
result_df = pd.DataFrame(results)

# Convert date columns to datetime
result_df["lastrowdate"] = pd.to_datetime(result_df["lastrowdate"])
result_df["Arowdate"] = pd.to_datetime(result_df["Arowdate"])

# Calculate difference (lastrowdate - Arowdate)
result_df["datediff"] = (result_df["lastrowdate"] - result_df["Arowdate"]).dt.days

# Sort by datediff descending, then lastrowdate descending
result_df = result_df.sort_values(by=["datediff", "lastrowdate"], ascending=[False, False]).reset_index(drop=True)

# Reassign serial No after sorting
result_df["No"] = range(1, len(result_df) + 1)

# Ensure output directories exist
os.makedirs(os.path.dirname(output_path1), exist_ok=True)
os.makedirs(os.path.dirname(output_path2), exist_ok=True)

# Save to both paths
result_df.to_csv(output_path1, index=False)
result_df.to_csv(output_path2, index=False)

print("Filtered data saved to gape.csv successfully!")