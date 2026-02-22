import pandas as pd
import os
import numpy as np

gape_file = "./csv/gape.csv"
mongodb_file = "./csv/mongodb.csv"
output_file2 = "./csv/gape_buy.csv"


os.makedirs(os.path.dirname(output_file2), exist_ok=True)

# Load existing data if file exists
existing_df = pd.DataFrame()
if os.path.exists(output_file2):
    existing_df = pd.read_csv(output_file2)
    print(f"📂 Existing file loaded with {len(existing_df)} signals")

# Load data
gape_df = pd.read_csv(gape_file)
mongodb_df = pd.read_csv(mongodb_file)

# Parse dates
gape_df['last_row_date'] = pd.to_datetime(gape_df['last_row_date'], errors='coerce')
mongodb_df['date'] = pd.to_datetime(mongodb_df['date'], errors='coerce')
mongodb_df = mongodb_df.dropna(subset=['date'])

# Group MongoDB by symbol for efficiency
mongo_groups = mongodb_df.groupby('symbol')

results = []

for _, r in gape_df.iterrows():
    symbol = str(r['symbol']).strip().upper()
    date = r['last_row_date']
    last_row_close = r['last_row_close']
    last_row_high = r['last_row_high']
    last_row_low = r['last_row_low']

    if pd.isna(date) or pd.isna(last_row_close) or symbol not in mongo_groups.groups:
        continue

    sym_data = mongo_groups.get_group(symbol).sort_values('date').reset_index(drop=True)

    # Get rows before trigger date
    prev_rows = sym_data[sym_data['date'] < date]
    if prev_rows.empty:
        continue
    prev_row = prev_rows.iloc[-1]

    # Entry condition
    if not (last_row_low > prev_row['low'] and last_row_close > prev_row['high']):
        continue

    # Find pre_candle (overlap with last_high or last_low)
    pre_candle = None
    for i in range(len(prev_rows)-1, -1, -1):
        row = prev_rows.iloc[i]
        if (row['low'] <= last_row_high <= row['high']) or (row['low'] <= last_row_low <= row['high']):
            pre_candle = row
            break
    if pre_candle is None:
        continue

    # Rows between pre_candle and trigger
    between = sym_data[
        (sym_data['date'] > pre_candle['date']) & 
        (sym_data['date'] < date)
    ]
    if between.empty:
        continue

    # SL = lowest low in between segment
    low_candle = between.loc[between['low'].idxmin()]
    SL_price = low_candle['low']
    buy_price = last_row_close

    # ✅ শুধু বেসিক তথ্য সংরক্ষণ (tp_price বাদ)
    results.append({
        'symbol': symbol,
        'date': date.date(),
        'buy_price': buy_price,
        'SL_price': SL_price
    })

# Build new signals DataFrame
if results:
    new_df = pd.DataFrame(results)
    new_df['buy_price'] = pd.to_numeric(new_df['buy_price'], errors='coerce')
    new_df['SL_price'] = pd.to_numeric(new_df['SL_price'], errors='coerce')

    # Filter valid signals (buy > SL)
    new_df = new_df[new_df['buy_price'] > new_df['SL_price']].reset_index(drop=True)

    if len(new_df) > 0:
        # Sort by date (newest first)
        new_df = new_df.sort_values(['date'], ascending=[False]).reset_index(drop=True)
        new_df.insert(0, 'No', range(1, len(new_df) + 1))

        # ✅ ফাইনাল কলাম (শুধু buy_price ও SL_price)
        new_df = new_df[[
            'No', 'symbol', 'date', 'buy_price', 'SL_price'
        ]]
    else:
        new_df = pd.DataFrame()
else:
    new_df = pd.DataFrame()

# Merge with existing data (remove duplicates based on symbol)
if not existing_df.empty and not new_df.empty:
    # Check if existing file has the correct format
    required_cols = ['symbol', 'date', 'buy_price', 'SL_price']
    if not all(col in existing_df.columns for col in required_cols):
        # If old format, we'll create new file with new format
        print("🔄 Existing file has old format - creating new file with updated format")
        final_df = new_df.copy()
    else:
        # Get existing symbols
        existing_symbols = set(existing_df['symbol'].unique())
        
        # Filter out symbols that already exist
        new_symbols_df = new_df[~new_df['symbol'].isin(existing_symbols)]
        
        if not new_symbols_df.empty:
            # Adjust 'No' column for new symbols
            new_symbols_df = new_symbols_df.reset_index(drop=True)
            new_symbols_df.insert(0, 'No', range(1, len(new_symbols_df) + 1))
            
            # Combine existing and new data
            final_df = pd.concat([existing_df, new_symbols_df], ignore_index=True)
            print(f"➕ Added {len(new_symbols_df)} new symbols to existing {len(existing_df)} signals")
        else:
            final_df = existing_df
            print("⏭️ No new symbols to add")
        
elif not new_df.empty:
    final_df = new_df
    print(f"🆕 Created new file with {len(new_df)} signals")
else:
    final_df = existing_df if not existing_df.empty else pd.DataFrame()
    print("⚠️ No signals found")

# Save final DataFrame
if not final_df.empty:
    # Ensure correct column order
    column_order = [
        'No', 'symbol', 'date', 'buy_price', 'SL_price'
    ]
    
    # Make sure all columns exist
    for col in column_order:
        if col not in final_df.columns:
            if col == 'No':
                final_df.insert(0, 'No', range(1, len(final_df) + 1))
            else:
                final_df[col] = None
    
    final_df = final_df[column_order]
    final_df.to_csv(output_file2, index=False)
    
    print(f"✅ gape_buy.csv saved with {len(final_df)} signals")
    if len(final_df) > 0:
        print(f"   📈 Latest signal: {final_df.iloc[0]['symbol']} - Buy: {final_df.iloc[0]['buy_price']:.2f}, SL: {final_df.iloc[0]['SL_price']:.2f}")
    
    # Show newly added symbols if any
    if not existing_df.empty and not new_df.empty and 'new_symbols_df' in locals() and not new_symbols_df.empty:
        new_added = set(new_symbols_df['symbol'].unique())
        if new_added:
            print(f"   🆕 New symbols added: {', '.join(sorted(new_added))}")
else:
    # Save empty DataFrame with correct columns
    empty_df = pd.DataFrame(columns=[
        'No', 'symbol', 'date', 'buy_price', 'SL_price'
    ])
    empty_df.to_csv(output_file2, index=False)
    print("⚠️ No signals found - empty file saved")