import pandas as pd
import os
from datetime import datetime

from hf_uploader import download_from_hf_or_run_script


# --------------------------------------------------
# Step 1: Download CSV from HF if needed
# --------------------------------------------------
if download_from_hf_or_run_script():
    print("HF data download success")


# --------------------------------------------------
# Swing check functions (UNCHANGED LOGIC)
# --------------------------------------------------
def check_high_swing(symbol_df, idx):
    try:
        n = len(symbol_df)
        if idx < 2 or idx >= n - 2:
            return False, False

        hoch = symbol_df.iloc[idx]['high']
        hocl = symbol_df.iloc[idx]['low']

        hboch = symbol_df.iloc[idx + 1]['high']
        hbocl = symbol_df.iloc[idx + 1]['low']

        hbtch = symbol_df.iloc[idx + 2]['high']
        hbtcl = symbol_df.iloc[idx + 2]['low']

        haoch = symbol_df.iloc[idx - 1]['high']
        haocl = symbol_df.iloc[idx - 1]['low']

        hatch = symbol_df.iloc[idx - 2]['high']
        hatcl = symbol_df.iloc[idx - 2]['low']

        # ----------------------------------
        # 🚫 SKIP CONDITION (YOUR RULE)
        # ----------------------------------
        if hoch == haoch and hocl <= haocl:
            return False, False   # 👉 skip this candle

        # ----------------------------------
        # ❌ Invalid / fake high
        # ----------------------------------
        if hoch == hboch and hocl <= hbocl:
            return False, True

        if (
            hoch > hboch
            and hbocl < hocl
            and hbtcl < hoch      # future LOW must be below

        ):
            return False, True

        if hoch > haoch and haocl < hocl and hatch < hoch:
            return False, True

        # ----------------------------------
        # ✅ VALID HIGH SWING
        # ----------------------------------
        if hoch > hboch and hoch > haoch:
            return True, False

        return False, False

    except Exception:
        return False, False


def check_low_swing(symbol_df, idx):
    try:
        n = len(symbol_df)
        if idx < 2 or idx >= n - 2:
            return False, False

        loch = symbol_df.iloc[idx]['high']
        locl = symbol_df.iloc[idx]['low']

        lboch = symbol_df.iloc[idx + 1]['high']
        lbocl = symbol_df.iloc[idx + 1]['low']

        lbtch = symbol_df.iloc[idx + 2]['high']
        lbtcl = symbol_df.iloc[idx + 2]['low']

        laoch = symbol_df.iloc[idx - 1]['high']
        laocl = symbol_df.iloc[idx - 1]['low']

        latch = symbol_df.iloc[idx - 2]['high']
        latcl = symbol_df.iloc[idx - 2]['low']

        # ----------------------------------
        # 🚫 SKIP CONDITION (YOUR RULE)
        # ----------------------------------
        if locl == laocl and loch >= laoch:
            return False, False   # 👉 skip, go next candle

        # ----------------------------------
        # ❌ Invalid / fake low
        # ----------------------------------
        if locl == lbocl and loch >= lboch:
            return False, True

        if (
            locl < lbocl
            and lboch > loch
            and lbtcl > locl      # fixed

        ):
            return False, True

        if locl < laocl and laoch > loch and latcl > locl:
            return False, True

        # ----------------------------------
        # ✅ VALID LOW SWING
        # ----------------------------------
        if locl < lbocl and locl < laocl:
            return True, False

        return False, False

    except Exception:
        return False, False


# --------------------------------------------------
# Core processing logic (শেষ থেকে পিছনে মাত্র ২টি করে)
# --------------------------------------------------
def process_symbol(symbol, symbol_df):
    """
    Data sorted ASC (oldest first)
    শেষ row থেকে পিছনে স্ক্যান করে সর্বশেষ ২ high ও ২ low swing খুঁজে আনে
    একই লুপে নিকটতমগুলো নির্বাচন করে
    """
    # ডেটা পুরানো থেকে নতুন ক্রমে সাজানো (ascending)
    symbol_df = symbol_df.sort_values('date', ascending=True).reset_index(drop=True)

    high_dates, high_prices = [], []
    low_dates, low_prices = [], []

    n = len(symbol_df)
    if n < 5:
        return high_dates, high_prices, low_dates, low_prices

    # 🔥 শেষ row থেকে পিছনের দিকে স্ক্যান
    for i in range(n - 1, 1, -1):
        idx = i
        
        # শুধুমাত্র HIGH swing চেক (যদি ২টি এখনও না পাওয়া যায়)
        if len(high_dates) < 2:
            is_high, is_fake_high = check_high_swing(symbol_df, idx)
            if is_high and not is_fake_high:
                # নিকটতম high প্রথমে রাখতে, আমরা reverse order এ রাখব
                # এবং পরে reverse করব
                high_dates.append(symbol_df.iloc[idx]['date'])
                high_prices.append(symbol_df.iloc[idx]['high'])

        # শুধুমাত্র LOW swing চেক (যদি ২টি এখনও না পাওয়া যায়)
        if len(low_dates) < 2:
            is_low, is_fake_low = check_low_swing(symbol_df, idx)
            if is_low and not is_fake_low:
                # নিকটতম low প্রথমে রাখতে, আমরা reverse order এ রাখব
                # এবং পরে reverse করব
                low_dates.append(symbol_df.iloc[idx]['date'])
                low_prices.append(symbol_df.iloc[idx]['low'])

        # যদি ২টি high এবং ২টি low swing পেয়ে যায়, লুপ বন্ধ করুন
        if len(high_dates) >= 2 and len(low_dates) >= 2:
            break

    # 🔥 যেহেতু আমরা শেষ থেকে পিছনে স্ক্যান করেছি,
    # তাই প্রথমে পাওয়া swing আসলে নিকটবর্তী
    # আমাদের শুধু ক্রম ঠিক করতে হবে:
    # high_dates এবং low_dates ইতিমধ্যেই নিকটতম থেকে দূরের ক্রমে আছে
    # (কারণ আমরা append করেছি যেভাবে পেয়েছি)
    
    # কিন্তু আমরা চাই: প্রথমে নিকটতম high, তারপর দ্বিতীয় high
    # প্রথমে নিকটতম low, তারপর দ্বিতীয় low
    # বর্তমানে এটি ঠিকই আছে কারণ:
    # যখন i = n-1 (শেষ row) থেকে শুরু করে পিছনে যাচ্ছি,
    # তখন প্রথম যে valid swing পাবো সেটাই সবচেয়ে নিকটবর্তী
    
    return high_dates[:2], high_prices[:2], low_dates[:2], low_prices[:2]


# --------------------------------------------------
# Save results
# --------------------------------------------------
def save_to_csv(symbol, high_dates, high_prices, low_dates, low_prices, output_base_dir):

    symbol_dir = os.path.join(output_base_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    if high_dates:
        high_df = pd.DataFrame({
            "date": pd.to_datetime(high_dates),
            "price": high_prices
        }).sort_values("date", ascending=False).reset_index(drop=True)

        high_df.to_csv(
            os.path.join(symbol_dir, "high.csv"),
            index=False,
            date_format="%Y-%m-%d"
        )

    if low_dates:
        low_df = pd.DataFrame({
            "date": pd.to_datetime(low_dates),
            "price": low_prices
        }).sort_values("date", ascending=False).reset_index(drop=True)

        low_df.to_csv(
            os.path.join(symbol_dir, "low.csv"),
            index=False,
            date_format="%Y-%m-%d"
        )


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    csv_file_path = "./csv/mongodb.csv"
    output_base_dir = "./csv/trand/"
    os.makedirs(output_base_dir, exist_ok=True)

    df = pd.read_csv(csv_file_path)
    df['date'] = pd.to_datetime(df['date'])

    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()

        high_dates, high_prices, low_dates, low_prices = process_symbol(symbol, symbol_df)

        if high_dates or low_dates:
            save_to_csv(symbol, high_dates, high_prices, low_dates, low_prices, output_base_dir)
            print(f"✓ {symbol} | High:{len(high_dates)} Low:{len(low_dates)}")
        else:
            print(f"✗ {symbol} | No swings")


if __name__ == "__main__":
    main()