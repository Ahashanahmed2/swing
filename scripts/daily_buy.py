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
    "macd": "./output/ai_signal/macd.csv"  # ✅ নতুন যোগ করা
}

output_dir = "./output/ai_signal/"
output_file = os.path.join(output_dir, "daily_buy.csv")
Path(output_dir).mkdir(parents=True, exist_ok=True)

latest_date_data = []
file_latest_dates = {}
file_dataframes = {}

date_columns = ['date', 'Date', 'DATE', 'timestamp', 'Timestamp', 'TIMESTAMP']

# Columns to exclude (will not be saved)
exclude_columns = ['buy', 'dl', 'dd', 'date', 'no', 'sl']

# Step 1: Find latest dates AND cache DataFrames
for file_key, file_path in files_info.items():
    try:
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        if df.empty:
            continue

        if file_key in ["fail_short", "bullish_strong", "macd"]:  # ✅ macd যোগ করা
            file_dataframes[file_key] = df
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
            file_dataframes[file_key] = df
    except:
        continue

# Step 2: Collect data using cached DataFrames
if file_latest_dates:
    overall_latest_date = max(file_latest_dates.values())
    
    for file_key, df in file_dataframes.items():
        try:
            if file_key in ["fail_short", "bullish_strong", "macd"]:  # ✅ macd যোগ করা
                symbol_col = None
                for col in ['SYMBOL', 'symbol', 'Symbol']:
                    if col in df.columns:
                        symbol_col = col
                        break
                
                if symbol_col:
                    for _, row in df.iterrows():
                        symbol = str(row[symbol_col]).strip()
                        if symbol:
                            row_data = {'symbol': symbol, 'file': file_key}
                            for col in df.columns:
                                if col != symbol_col and col.lower() not in [x.lower() for x in exclude_columns]:
                                    row_data[col] = row[col]
                            latest_date_data.append(row_data)
                continue

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
                        row_data = {}
                        for col in df.columns:
                            if col.lower() not in [x.lower() for x in exclude_columns]:
                                row_data[col] = row[col]
                        row_data['symbol'] = symbol
                        row_data['file'] = file_key
                        latest_date_data.append(row_data)
        except:
            continue
    
    if latest_date_data:
        result_df = pd.DataFrame(latest_date_data)
        
        # Remove any remaining excluded columns (case-insensitive)
        cols_to_remove = []
        for col in result_df.columns:
            if col.lower() in [x.lower() for x in exclude_columns]:
                cols_to_remove.append(col)
        
        if cols_to_remove:
            result_df = result_df.drop(columns=cols_to_remove)
        
        # close -> buy conversion (but buy will be removed later)
        close_columns = [col for col in result_df.columns if col.lower() == 'close']
        for close_col in close_columns:
            if 'buy' not in result_df.columns:
                result_df['buy'] = result_df[close_col]
            result_df = result_df.drop(columns=[close_col])
        
        # Remove buy column if exists
        if 'buy' in result_df.columns:
            result_df = result_df.drop(columns=['buy'])
        
        # Group by symbol
        agg_dict = {'file': lambda x: ','.join(sorted(set(x)))}
        
        other_cols = [col for col in result_df.columns if col not in ['symbol', 'file']]
        for col in other_cols:
            agg_dict[col] = 'first'
        
        result_df = result_df.groupby('symbol', as_index=False).agg(agg_dict)
        
        # Column ordering
        ordered_columns = []
        for col in ['symbol', 'file']:
            if col in result_df.columns:
                ordered_columns.append(col)
        
        for col in result_df.columns:
            if col not in ordered_columns:
                ordered_columns.append(col)
        
        result_df = result_df[ordered_columns]
        
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
                result_df['high'] = result_df['symbol'].map(high_map)
        
        # Final check: remove excluded columns one more time before saving
        final_cols_to_remove = []
        for col in result_df.columns:
            if col.lower() in [x.lower() for x in exclude_columns]:
                final_cols_to_remove.append(col)
        
        if final_cols_to_remove:
            result_df = result_df.drop(columns=final_cols_to_remove)
        
        result_df.to_csv(output_file, index=False)
        print(f"✅ Saved {len(result_df)} symbols to {output_file}")
        
        print(f"\n📊 Summary:")
        print(f"   Total unique symbols: {len(result_df)}")
        print(f"   Columns saved: {list(result_df.columns)}")
        print(f"   Excluded columns: {exclude_columns}")
        
else:
    print("❌ No valid data found")
