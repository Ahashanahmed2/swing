import pandas as pd
import os

import numpy as np
import ta  # ensure ta-lib or pandas-ta is installed: pip install ta

DATA_FOLDER = "./csv"  # নিশ্চিত করুন আপনার CSV ফাইল এই ফোল্ডারে আছে

def load_data():
    all_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    if not all_files:
        print("❌ কোনো CSV ফাইল পাওয়া যায়নি।")
        return pd.DataFrame()

    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(os.path.join(DATA_FOLDER, file))
            df['symbol'] = os.path.splitext(file)[0]  # ফাইল নাম থেকে symbol তৈরি
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ {file} ফাইল লোডে সমস্যা: {e}")
            continue

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.sort_values(['symbol', 'date']).reset_index(drop=True)
        return combined_df
    else:
        return pd.DataFrame()


def calculate_indicators(df):
    df = df.copy()
    try:
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()

        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()

        # অপশনাল: Zigzag mock, যতক্ষণ না আপনি সত্যিকারের zigzag যুক্ত করেন
        df['zigzag'] = df['close'].rolling(window=5).mean()

    except Exception as e:
        print(f"⚠️ ইন্ডিকেটর হিসাব করতে সমস্যা হয়েছে: {e}")

    return df
