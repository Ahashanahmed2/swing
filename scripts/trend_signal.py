#trend_signal.py
import pandas as pd
import os
from datetime import datetime


# --------------------------------------------------
# Helper: merge old + new & detect new symbols
# --------------------------------------------------
def merge_and_track_new_symbols(old_df, new_df, symbol_col='symbol'):
    if old_df is None or old_df.empty:
        return new_df, new_df.copy()

    old_symbols = set(old_df[symbol_col])
    new_symbols = set(new_df[symbol_col])

    # keep only still-valid symbols
    common_symbols = old_symbols & new_symbols
    preserved_df = old_df[old_df[symbol_col].isin(common_symbols)]

    # brand new symbols
    new_only_df = new_df[~new_df[symbol_col].isin(old_symbols)]

    final_df = pd.concat([preserved_df, new_only_df], ignore_index=True)
    return final_df, new_only_df


def create_uptrend_signals():

    mongodb_csv = './csv/mongodb.csv'
    trand_base_dir = './csv/trand/'
    output_dir = './csv/'

    os.makedirs(output_dir, exist_ok=True)

    uptrend_file = os.path.join(output_dir, 'uptrand.csv')

    print(f"Reading {mongodb_csv}...")

    try:
        mongodb_df = pd.read_csv(mongodb_csv)
    except FileNotFoundError:
        print(f"❌ Error: {mongodb_csv} not found!")
        return

    for col in ['symbol', 'date', 'close']:
        if col not in mongodb_df.columns:
            print(f"❌ Missing column: {col}")
            return

    mongodb_df['date'] = pd.to_datetime(mongodb_df['date'])

    # latest candle per symbol
    latest_data = {}
    for symbol in mongodb_df['symbol'].unique():
        df = mongodb_df[mongodb_df['symbol'] == symbol] \
            .sort_values('date', ascending=False)

        if not df.empty:
            latest_data[symbol] = {
                'close': df.iloc[0]['close'],
                'date': df.iloc[0]['date']
            }

    print(f"✅ Found {len(latest_data)} symbols")

    uptrend_signals = []

    # --------------------------------------------------
    # Signal detection
    # --------------------------------------------------
    for symbol, info in latest_data.items():
        symbol_dir = os.path.join(trand_base_dir, symbol)
        high_file = os.path.join(symbol_dir, 'high.csv')

        latest_close = info['close']
        latest_date = info['date']

        # ---------------- UPTREND ----------------
        if os.path.exists(high_file):
            try:
                high_df = pd.read_csv(high_file)
                high_df['date'] = pd.to_datetime(high_df['date'])

                if len(high_df) >= 2:
                    p1_price = float(high_df.iloc[0]['price'])
                    p2_price = float(high_df.iloc[1]['price'])

                    p1_date = high_df.iloc[0]['date']
                    p2_date = high_df.iloc[1]['date']

                    if p1_price < latest_close > p2_price and p1_price < p2_price:
                        uptrend_signals.append({
                            'date': latest_date,
                            'symbol': symbol,
                            'close': latest_close,
                            'p1_date': p1_date,
                            'p2_date': p2_date
                        })
            except Exception as e:
                print(f"⚠️ High error ({symbol}): {e}")

    # --------------------------------------------------
    # SAVE UPTREND
    # --------------------------------------------------
    if uptrend_signals:
        new_up_df = pd.DataFrame(uptrend_signals)
        
        # পুরোনো ডেটা লোড করা
        old_up_df = None
        if os.path.exists(uptrend_file):
            try:
                old_up_df = pd.read_csv(uptrend_file)
                if old_up_df.empty:
                    old_up_df = None
            except pd.errors.EmptyDataError:
                print(f"⚠️ {uptrend_file} ফাইলটি খালি ছিল, নতুন ফাইল তৈরি হচ্ছে")
                old_up_df = None
            except Exception as e:
                print(f"⚠️ {uptrend_file} পড়তে সমস্যা: {e}, নতুন ফাইল তৈরি হচ্ছে")
                old_up_df = None
        
        # শুধুমাত্র নতুন সিম্বল অ্যাড হবে, পুরোনোটা অপরিবর্তিত থাকবে
        final_up_df, _ = merge_and_track_new_symbols(
            old_up_df, new_up_df
        )

        final_up_df.to_csv(uptrend_file, index=False)

        print("✅ Uptrend updated")

    else:
        # আগের ডেটা মুছে না, শুধু নতুন signal না থাকলে পুরোনো ফাইল রেখে দিবে
        if not os.path.exists(uptrend_file):
            pd.DataFrame().to_csv(uptrend_file, index=False)
        print("❌ No uptrend signals found")

    print("\n🎯 Trend breakout detection completed!")


def main():
    print("=" * 60)
    print("TREND BREAKOUT DETECTION")
    print("=" * 60)
    create_uptrend_signals()


if __name__ == "__main__":
    main()
