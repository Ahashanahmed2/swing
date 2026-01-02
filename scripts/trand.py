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
            and hoch == haoch
            and hocl <= haocl
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
            and locl == laocl
            and loch >= laoch
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
# Core processing logic (LATEST FIRST)
# --------------------------------------------------
def process_symbol(symbol, symbol_df):
    """
    Data sorted DESC
    Scan latest → older
    Output always latest first
    """

    symbol_df = symbol_df.sort_values('date', ascending=False).reset_index(drop=True)

    high_dates, high_prices = [], []
    low_dates, low_prices = [], []

    n = len(symbol_df)
    if n < 5:
        return high_dates, high_prices, low_dates, low_prices

    # Scan from newest candle (index 2) to older
    for idx in range(2, n - 2):

        # ---------- HIGH ----------
        is_high, _ = check_high_swing(symbol_df, idx)
        if is_high:
            high_dates.append(symbol_df.iloc[idx]['date'])
            high_prices.append(symbol_df.iloc[idx]['high'])

        # ---------- LOW ----------
        is_low, _ = check_low_swing(symbol_df, idx)
        if is_low:
            low_dates.append(symbol_df.iloc[idx]['date'])
            low_prices.append(symbol_df.iloc[idx]['low'])

    # keep only latest 2 swings
    high_df = pd.DataFrame({"date": high_dates, "price": high_prices})
    low_df  = pd.DataFrame({"date": low_dates,  "price": low_prices})

    high_df = high_df.sort_values("date", ascending=False).head(2)
    low_df  = low_df.sort_values("date", ascending=False).head(2)

    return (
        high_df["date"].tolist(),
        high_df["price"].tolist(),
        low_df["date"].tolist(),
        low_df["price"].tolist(),
    )


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