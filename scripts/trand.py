import pandas as pd
import os
from datetime import datetime

#from hf_uploader import download_from_hf_or_run_script


# --------------------------------------------------
# Step 1: Download CSV from HF if needed
# --------------------------------------------------
#if download_from_hf_or_run_script():
    #print("HF data download success")


# --------------------------------------------------
# Swing check functions (UNCHANGED LOGIC)
# --------------------------------------------------
def check_high_swing(symbol_df, idx):
    try:
        n = len(symbol_df)
        if idx < 2 or idx >= n - 2:
            return False, False

        # -------- Current candle --------
        cur_high = symbol_df.iloc[idx]['high']
        cur_low  = symbol_df.iloc[idx]['low']

        # -------- Previous candles --------
        prev1_high = symbol_df.iloc[idx - 1]['high']
        prev1_low  = symbol_df.iloc[idx - 1]['low']

        prev2_high = symbol_df.iloc[idx - 2]['high']
        prev2_low  = symbol_df.iloc[idx - 2]['low']

        # -------- Next candles --------
        next1_high = symbol_df.iloc[idx + 1]['high']
        next1_low  = symbol_df.iloc[idx + 1]['low']

        next2_high = symbol_df.iloc[idx + 2]['high']
        next2_low  = symbol_df.iloc[idx + 2]['low']

        # ----------------------------------
        # 🚫 SKIP CONDITION (YOUR RULE)
        # same high as previous & low not higher
        # ----------------------------------
        if cur_high == prev1_high and cur_low <= prev1_low:
            return False, True   # skip candle

        # ----------------------------------
        # ❌ Fake high (same as next)
        # ----------------------------------
        if cur_high == next1_high and cur_low <= next1_low:
            return False, True

        # ----------------------------------
        # ✅ VALID HIGH SWING
        # hocl > hocl+2 AND hocl > hocl-2
        # ----------------------------------
        if (
            cur_high >= next1_high
            and cur_high >= prev1_high
            and cur_low > next1_low
            and cur_low > prev1_low
            and prev1_high > prev2_high
            and next1_high > next2_high
        ):
            return True, False

        return False, False

    except Exception:
        return False, False


def check_low_swing(symbol_df, idx):
    try:
        n = len(symbol_df)
        if idx < 2 or idx >= n - 2:
            return False, False

        # -------- Current candle --------
        cur_high = symbol_df.iloc[idx]['high']
        cur_low  = symbol_df.iloc[idx]['low']

        # -------- Previous candles --------
        prev1_high = symbol_df.iloc[idx - 1]['high']
        prev1_low  = symbol_df.iloc[idx - 1]['low']

        prev2_high = symbol_df.iloc[idx - 2]['high']
        prev2_low  = symbol_df.iloc[idx - 2]['low']

        # -------- Next candles --------
        next1_high = symbol_df.iloc[idx + 1]['high']
        next1_low  = symbol_df.iloc[idx + 1]['low']

        next2_high = symbol_df.iloc[idx + 2]['high']
        next2_low  = symbol_df.iloc[idx + 2]['low']

        # ----------------------------------
        # 🚫 SKIP CONDITION
        # same low as previous & high not lower
        # ----------------------------------
        if cur_low == prev1_low and cur_high >= prev1_high:
            return False, True

        # ----------------------------------
        # ❌ Fake low (same as next)
        # ----------------------------------
        if cur_low == next1_low and cur_high >= next1_high:
            return False, True

        # ----------------------------------
        # ✅ VALID LOW SWING
        # locl < locl+2 AND locl < locl-2
        # ----------------------------------
        if (
            cur_low <= next1_low
            and cur_low <= prev1_low
            and cur_high < next1_high
            and cur_high < prev1_high
            and next1_low < next2_low
            and prev1_low < prev2_low
        ):
            return True, False

        return False, False

    except Exception:
        return False, False


# --------------------------------------------------
# Core processing logic (শেষ থেকে পিছনে শুধু ২টি করে)
# --------------------------------------------------
def process_symbol(symbol, symbol_df):
    """
    Data sorted ASC (oldest first)
    শেষ row থেকে পিছনে স্ক্যান করে মাত্র ২ high ও ২ low swing খুঁজে আনে
    """
    # ডেটা পুরানো থেকে নতুন ক্রমে সাজানো (ascending)
    symbol_df = symbol_df.sort_values('date', ascending=True).reset_index(drop=True)
    
    high_dates, high_prices = [], []
    low_dates, low_prices = [], []
    
    n = len(symbol_df)
    if n < 5:
        return high_dates, high_prices, low_dates, low_prices
    
    # 🔥 শেষ থেকে পিছনের দিকে স্ক্যান
    # সর্বশেষ ক্যান্ডেল থেকে শুরু করে পিছনে যাবে
    i = n - 3  # শেষ থেকে ৩য় index থেকে শুরু (কারণ check ফাংশনে idx+2 দরকার)
    
    while i >= 2 and (len(high_dates) < 2 or len(low_dates) < 2):
        
        # HIGH swing চেক (যদি ২টি এখনও না পাওয়া যায়)
        if len(high_dates) < 2:
            is_high, is_fake_high = check_high_swing(symbol_df, i)
            if is_high and not is_fake_high:
                high_dates.append(symbol_df.iloc[i]['date'])
                high_prices.append(symbol_df.iloc[i]['high'])
        
        # LOW swing চেক (যদি ২টি এখনও না পাওয়া যায়)
        if len(low_dates) < 2:
            is_low, is_fake_low = check_low_swing(symbol_df, i)
            if is_low and not is_fake_low:
                low_dates.append(symbol_df.iloc[i]['date'])
                low_prices.append(symbol_df.iloc[i]['low'])
        
        i -= 1  # পরবর্তী পুরানো ক্যান্ডেলে যান
    
    # তারিখের ক্রম ঠিক রাখুন (নতুন থেকে পুরানো)
    high_dates.reverse()
    high_prices.reverse()
    low_dates.reverse()
    low_prices.reverse()
    
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