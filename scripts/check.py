import pandas as pd
from pathlib import Path

# mongodb থেকে unique sectors
df = pd.read_csv('./csv/mongodb.csv')
mongo_sectors = set(
    df.dropna(subset=['sector'])
      .groupby('symbol')['sector'].last()
      .str.strip()
      .unique()
)

# sector CSV থেকে unique sectors
sector_dir = Path('./csv/sector')
csv_sectors = set()
for f in (sector_dir / 'daily').glob('*.csv'):
    s = pd.read_csv(f)
    if 'sector' in s.columns:
        csv_sectors.add(str(s['sector'].iloc[0]).strip())

print("=" * 70)
print("MONGO SECTORS:")
for s in sorted(mongo_sectors):
    print(f"  [{repr(s)}]")

print("\nSECTOR CSV SECTORS:")
for s in sorted(csv_sectors):
    print(f"  [{repr(s)}]")

print("\n" + "=" * 70)
print("❌ IN MONGO BUT NOT IN CSV:")
for s in sorted(mongo_sectors - csv_sectors):
    print(f"  [{repr(s)}]")

print("\n❌ IN CSV BUT NOT IN MONGO:")
for s in sorted(csv_sectors - mongo_sectors):
    print(f"  [{repr(s)}]")

print("\n✅ MATCHED:")
for s in sorted(mongo_sectors & csv_sectors):
    print(f"  [{repr(s)}]")
