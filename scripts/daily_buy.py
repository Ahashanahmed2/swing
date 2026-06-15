import pandas as pd
import os
from pathlib import Path

files_info = {
    "short": "./csv/short_buy.csv",
    "gape": "./csv/gape_buy.csv", 
    "rsi": "./csv/rsi_30_buy.csv",
    "swing": "./csv/swing_buy.csv",
    "uptrend": "./output/ai_signal/uptrand_buy.csv",
    "fail_short": "./output/ai_signal/fail_short_buy_pass.csv",
    "bullish_strong": "./output/ai_signal/bullish_strong.csv",
    "macd": "./output/ai_signal/macd.csv"
}

output_dir = "./output/ai_signal/"
output_file = os.path.join(output_dir, "daily_buy.csv")
Path(output_dir).mkdir(parents=True, exist_ok=True)

date_columns = ['date', 'Date', 'DATE', 'timestamp', 'Timestamp', 'TIMESTAMP', 'signal_date']

# Columns to exclude (will not be saved)
exclude_columns = ['buy', 'dl', 'dd', 'no', 'sl', 'Unnamed: 0']

# Store all data before final grouping
all_records = []

# Helper function to get date column
def get_date_column(df):
    for col in date_columns:
        if col in df.columns:
            return col
    return None

# Step 1: Process each file
for file_key, file_path in files_info.items():
    try:
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        if df.empty:
            print(f"⚠️ Empty file: {file_path}")
            continue

        print(f"\n📂 Processing {file_key}: {len(df)} rows")
        
        # Find symbol column
        symbol_col = None
        for col in ['SYMBOL', 'symbol', 'Symbol', 'Trading Code', 'trading_code']:
            if col in df.columns:
                symbol_col = col
                break
        
        if symbol_col is None:
            print(f"   ⚠️ No symbol column found in {file_key}")
            continue
        
        # Find date column
        date_col = get_date_column(df)
        
        # Filter by latest date if date column exists
        if date_col and file_key not in ["fail_short", "bullish_strong", "macd"]:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            
            if not df.empty:
                latest_date = df[date_col].max()
                df = df[df[date_col] == latest_date]
                print(f"   📅 Latest date: {latest_date.strftime('%Y-%m-%d')}, Rows: {len(df)}")
        
        # Process each row
        for _, row in df.iterrows():
            symbol = str(row[symbol_col]).strip()
            if not symbol or symbol == 'nan':
                continue
            
            # Create record
            record = {
                'symbol': symbol,
                'source_file': file_key,
                'original_date': row[date_col] if date_col and date_col in row else None
            }
            
            # Add all other columns (except excluded ones)
            for col in df.columns:
                if col != symbol_col and col.lower() not in [x.lower() for x in exclude_columns]:
                    # Skip date column if already added
                    if date_col and col == date_col:
                        continue
                    # Convert to string for non-numeric to avoid issues
                    val = row[col]
                    if pd.isna(val):
                        continue
                    # Use original column name
                    record[col] = val
            
            all_records.append(record)
            
    except Exception as e:
        print(f"❌ Error processing {file_key}: {e}")
        continue

# Step 2: Create DataFrame from all records
if not all_records:
    print("❌ No valid data found")
    exit()

result_df = pd.DataFrame(all_records)
print(f"\n📊 Total records before dedup: {len(result_df)}")

# Step 3: Remove duplicates (keep first occurrence, but merge source files)
# First, create a dictionary to collect unique symbols
symbol_data = {}

for _, row in result_df.iterrows():
    symbol = row['symbol']
    
    if symbol not in symbol_data:
        # First time seeing this symbol - store all data
        symbol_data[symbol] = row.to_dict()
    else:
        # Already exists - merge source files
        existing = symbol_data[symbol]
        
        # Merge source_file (comma separated)
        existing_sources = existing.get('source_file', '').split(',')
        new_source = row.get('source_file', '')
        if new_source and new_source not in existing_sources:
            existing_sources.append(new_source)
            existing['source_file'] = ','.join(existing_sources)
        
        # For numeric columns, take max or keep first? 
        # Priority: keep the value from main files first
        for col in row.index:
            if col in ['symbol', 'source_file', 'original_date']:
                continue
            
            current_val = row[col]
            existing_val = existing.get(col)
            
            # If existing is None/NaN and current has value, use current
            if pd.isna(existing_val) and not pd.isna(current_val):
                existing[col] = current_val
            # If both have values, keep the non-NaN (or could do max/min based on column type)
            elif not pd.isna(current_val) and not pd.isna(existing_val):
                # For numeric columns, take max
                if isinstance(current_val, (int, float)) and isinstance(existing_val, (int, float)):
                    existing[col] = max(existing_val, current_val)

# Step 4: Convert back to DataFrame
final_df = pd.DataFrame(symbol_data.values())
print(f"📊 Unique symbols after dedup: {len(final_df)}")

# Step 5: Add high from mongodb.csv (if exists)
mongo_path = './csv/mongodb.csv'
if os.path.exists(mongo_path):
    try:
        mongo_df = pd.read_csv(mongo_path)
        if 'date' in mongo_df.columns:
            mongo_df['date'] = pd.to_datetime(mongo_df['date'])
            
            # Get latest high for each symbol
            latest_high = mongo_df.sort_values('date').groupby('symbol').tail(1)[['symbol', 'high']]
            high_map = dict(zip(latest_high['symbol'], latest_high['high']))
            
            # Add high to final dataframe
            final_df['high'] = final_df['symbol'].map(high_map)
            print(f"   ✅ Added high from MongoDB for {len(high_map)} symbols")
    except Exception as e:
        print(f"   ⚠️ Error loading mongodb.csv: {e}")

# Step 6: Remove any remaining excluded columns
cols_to_remove = [col for col in final_df.columns if col.lower() in [x.lower() for x in exclude_columns]]
if cols_to_remove:
    final_df = final_df.drop(columns=cols_to_remove)
    print(f"   🧹 Removed excluded columns: {cols_to_remove}")

# Step 7: Column ordering
priority_cols = ['symbol', 'source_file', 'high', 'original_date']
existing_priority = [col for col in priority_cols if col in final_df.columns]
other_cols = [col for col in final_df.columns if col not in existing_priority]
final_df = final_df[existing_priority + other_cols]

# Step 8: Save to CSV
final_df.to_csv(output_file, index=False)
print(f"\n✅ Saved {len(final_df)} unique symbols to {output_file}")

# Step 9: Summary
print(f"\n📊 SUMMARY:")
print(f"   Total unique symbols: {len(final_df)}")
print(f"   Columns: {list(final_df.columns)}")
print(f"   Source files distribution:")
source_counts = final_df['source_file'].value_counts()
for source, count in source_counts.items():
    print(f"      - {source}: {count} symbols")

# Step 10: Show sample of duplicate sources (where multiple files have same symbol)
print(f"\n📋 Symbols from multiple sources:")
multi_source = final_df[final_df['source_file'].str.contains(',')]
if len(multi_source) > 0:
    for _, row in multi_source.head(10).iterrows():
        print(f"   {row['symbol']}: {row['source_file']}")
else:
    print("   No symbols from multiple sources")

# Step 11: Debug - show if any duplicate still exists
duplicate_check = final_df['symbol'].duplicated().sum()
if duplicate_check > 0:
    print(f"\n⚠️ WARNING: Still {duplicate_check} duplicate symbols found!")
    duplicates = final_df[final_df['symbol'].duplicated(keep=False)].sort_values('symbol')
    print(duplicates[['symbol', 'source_file']].head(10))
else:
    print(f"\n✅ No duplicate symbols in final output")
