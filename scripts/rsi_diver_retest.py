import pandas as pd
import os

# Paths
source_path = './csv/rsi_diver.csv'
target_path = './csv/rsi_diver_retest.csv'

# Load source data
df_source = pd.read_csv(source_path)

# Load target data if exists
if os.path.exists(target_path):
    df_target = pd.read_csv(target_path)
else:
    df_target = pd.DataFrame(columns=df_source.columns)

# Merge logic
# Update existing symbols
df_target.set_index('symbol', inplace=True)
df_source.set_index('symbol', inplace=True)

# Update old symbols with new data
df_target.update(df_source)

# Find new symbols to append
new_symbols = df_source.index.difference(df_target.index)
df_new = df_source.loc[new_symbols]

# Append new symbols
df_combined = pd.concat([df_target, df_new])

# Reset index and save
df_combined.reset_index(inplace=True)
df_combined.to_csv(target_path, index=False)