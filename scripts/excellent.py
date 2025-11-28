import pandas as pd
import os

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
source_path = './csv/liquidity.csv'
output_path = './output/ai_signal/excellen.csv'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------
df = pd.read_csv(source_path)

# ---------------------------------------------------------
# Filter rows where LIQUIDITY_RATING == 'Excellen'
# ---------------------------------------------------------
filtered = df[df['LIQUIDITY_RATING'] == 'Excellen']

# ---------------------------------------------------------
# Save to output
# ---------------------------------------------------------
filtered.to_csv(output_path, index=False)

print(f"Saved {len(filtered)} rows to {output_path}")