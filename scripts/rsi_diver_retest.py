import pandas as pd
import os

# Paths
source_path = './csv/rsi_diver.csv'           # read-only
target_path = './csv/rsi_diver_retest.csv'    # update + append

# Load source (read-only)
df_source = pd.read_csv(source_path)

# Load or create target
if os.path.exists(target_path):
    df_target = pd.read_csv(target_path)
else:
    df_target = pd.DataFrame(columns=df_source.columns)

# Ensure both contain 'symbol'
if 'symbol' not in df_source.columns or 'symbol' not in df_target.columns:
    raise ValueError("Both CSV files must contain 'symbol' column.")

# Set index for easy update
df_source.set_index('symbol', inplace=True)
df_target.set_index('symbol', inplace=True)

# -------- UPDATE PART --------
# Update only symbols that exist in both files
common_symbols = df_target.index.intersection(df_source.index)
df_target.loc[common_symbols] = df_source.loc[common_symbols]

# -------- APPEND NEW SYMBOLS --------
# Find symbols that exist in source but not in target
new_symbols = df_source.index.difference(df_target.index)

# Append those new rows
df_target = pd.concat([df_target, df_source.loc[new_symbols]])

# Remove duplicates if any
df_target = df_target[~df_target.index.duplicated(keep='first')]

# Save output
df_target.reset_index(inplace=True)
df_target.to_csv(target_path, index=False)

print("Update + Append Complete.")