import pandas as pd
import os

# File paths
rsi_path = './csv/rsi_diver.csv'
mongo_path = './csv/mongodb.csv'

# Check if RSI divergence file exists and is not empty
if not os.path.exists(rsi_path) or os.path.getsize(rsi_path) == 0:
    print("⚠️ RSI divergence file missing or empty. Skipping confirmation step.")
else:
    # Load RSI divergence file
    rsi_df = pd.read_csv(rsi_path)

    # Check if mongodb.csv exists and is not empty
    if not os.path.exists(mongo_path) or os.path.getsize(mongo_path) == 0:
        raise ValueError("❌ mongodb.csv file is missing or empty.")

    mongo_df = pd.read_csv(mongo_path)

    # Ensure 'close' column exists
    if 'close' not in mongo_df.columns:
        raise ValueError("❌ mongodb.csv must contain a 'close' column.")

    # Convert date columns to datetime
    rsi_df['last row date'] = pd.to_datetime(rsi_df['last row date'])
    mongo_df['date'] = pd.to_datetime(mongo_df['date'])

    # Prepare list for confirmed signals
    confirmed_list = []

    # Iterate over RSI divergence symbols
    for _, row in rsi_df.iterrows():
        symbol = row['symbol']
        last_high = row['last row high']
        last_date = row['last row date']

        # Filter mongodb.csv for matching symbol and date > last row date
        future_rows = mongo_df[(mongo_df['symbol'] == symbol) & (mongo_df['date'] > last_date)]

        # Check if any future row has close > last_high
        for _, future_row in future_rows.iterrows():
            if future_row['close'] > last_high:
                confirmed_list.append(row)
                break  # Only one confirmation needed

    # Save confirmed signals to both paths
    confirmed_df = pd.DataFrame(confirmed_list)

    # Ensure output directory exists
    os.makedirs('./output/ai_signal', exist_ok=True)

    confirmed_df.to_csv('./output/ai_signal/confirm_rsi_diver.csv', index=False)
    confirmed_df.to_csv('./csv/confirm_rsi_diver.csv', index=False)

    print("✅ Confirmed RSI divergence saved to both output files.")
