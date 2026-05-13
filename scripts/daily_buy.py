import pandas as pd
import os
from pathlib import Path

files_info = {
    "short": "./csv/short_buy.csv",
    "gape": "./csv/gape_buy.csv", 
    "rsi": "./csv/rsi_30_buy.csv",
    "swing": "./csv/swing_buy.csv",
    "uptrend": "./output/ai_signal/uptrand_buy.csv"
}

output_dir = "./output/ai_signal/"
output_file = os.path.join(output_dir, "daily_buy.csv")
Path(output_dir).mkdir(parents=True, exist_ok=True)

latest_date_data = []
file_latest_dates = {}
file_dataframes = {}  # ✅ DataFrame cache

date_columns = ['date', 'Date', 'DATE', 'timestamp', 'Timestamp', 'TIMESTAMP']

# Step 1: Find latest dates AND cache DataFrames
for file_key, file_path in files_info.items():
    try:
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        if df.empty:
            continue

        date_column = None
        for col in date_columns:
            if col in df.columns:
                date_column = col
                break

        if date_column is None:
            continue
        
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])
        
        if not df.empty:
            latest_date = df[date_column].max()
            file_latest_dates[file_key] = latest_date
            file_dataframes[file_key] = df  # ✅ Cache
    except:
        continue

# Step 2: Collect data using cached DataFrames
if file_latest_dates:
    overall_latest_date = max(file_latest_dates.values())
    
    for file_key, df in file_dataframes.items():  # ✅ ক্যাশ থেকে নেওয়া
        try:
            date_column = None
            for col in date_columns:
                if col in df.columns:
                    date_column = col
                    break

            if date_column is None:
                continue
            
            latest_data = df[df[date_column] == overall_latest_date]
            
            if len(latest_data) == 0:
                continue
            
            for _, row in latest_data.iterrows():
                if 'symbol' in df.columns:
                    symbol = str(row['symbol'])
                    
                    if symbol:
                        row_data = row.to_dict()
                        row_data['symbol'] = symbol
                        row_data['file'] = file_key
                        latest_date_data.append(row_data)
        except:
            continue
    
    if latest_date_data:
        result_df = pd.DataFrame(latest_date_data)
        
        # close -> buy conversion
        close_columns = [col for col in result_df.columns if col.lower() == 'close']
        for close_col in close_columns:
            if 'buy' not in result_df.columns:
                result_df['buy'] = result_df[close_col]
            result_df = result_df.drop(columns=[close_col])
        
        # Column ordering
        ordered_columns = []
        for col in ['symbol', 'file', 'buy']:
            if col in result_df.columns:
                ordered_columns.append(col)
        
        for col in result_df.columns:
            if col not in ordered_columns:
                ordered_columns.append(col)
        
        result_df = result_df[ordered_columns]
        result_df = result_df.drop_duplicates(subset=['symbol'], keep='first')
        
        # MongoDB high merge
        mongo_path = './csv/mongodb.csv'
        if os.path.exists(mongo_path):
            mongo_df = pd.read_csv(mongo_path)
            mongo_df['date'] = pd.to_datetime(mongo_df['date'])
            
            if 'symbol' in result_df.columns and len(result_df) > 0:
                target_symbols = result_df['symbol'].unique()
                mongo_filtered = mongo_df[mongo_df['symbol'].isin(target_symbols)]
                latest = mongo_filtered.sort_values('date').groupby('symbol').tail(1)[['symbol', 'high']]
                
                high_map = dict(zip(latest['symbol'], latest['high']))
                result_df['high'] = result_df['symbol'].map(high_map)  # ✅ buy fillna বাদ
        
        result_df.to_csv(output_file, index=False)
        print(f"✅ Saved {len(result_df)} symbols to {output_file}")
else:
    print("❌ No valid data found")
