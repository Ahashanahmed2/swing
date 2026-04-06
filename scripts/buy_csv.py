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


def create_uptrend_downtrend_signals():
    mongodb_csv = './csv/mongodb.csv'
    trand_base_dir = './csv/trand/'
    output_dir = './csv/'
    
    os.makedirs(output_dir, exist_ok=True)

    uptrend_file = os.path.join(output_dir, 'uptrand.csv')
    downtrend_file = os.path.join(output_dir, 'downtrand.csv')

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
    downtrend_signals = []

    # --------------------------------------------------
    # Signal detection
    # --------------------------------------------------
    for symbol, info in latest_data.items():
        symbol_dir = os.path.join(trand_base_dir, symbol)
        high_file = os.path.join(symbol_dir, 'high.csv')
        low_file = os.path.join(symbol_dir, 'low.csv')

        latest_close = info['close']
        latest_date = info['date']

        # ---------------- UPTREND ----------------
        if os.path.exists(high_file):
            try:
                high_df = pd.read_csv(high_file)
                if 'high' in high_df.columns:
                    recent_high = high_df['high'].iloc[-1] if not high_df.empty else 0
                    if latest_close > recent_high:
                        uptrend_signals.append({
                            'symbol': symbol,
                            'date': latest_date.strftime('%Y-%m-%d'),
                            'close': latest_close,
                            'break_high': recent_high
                        })
            except Exception as e:
                print(f"⚠️ Error reading {high_file}: {e}")

        # ---------------- DOWNTREND ----------------
        if os.path.exists(low_file):
            try:
                low_df = pd.read_csv(low_file)
                if 'low' in low_df.columns:
                    recent_low = low_df['low'].iloc[-1] if not low_df.empty else 0
                    if latest_close < recent_low:
                        downtrend_signals.append({
                            'symbol': symbol,
                            'date': latest_date.strftime('%Y-%m-%d'),
                            'close': latest_close,
                            'break_low': recent_low
                        })
            except Exception as e:
                print(f"⚠️ Error reading {low_file}: {e}")

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    if uptrend_signals:
        uptrend_df = pd.DataFrame(uptrend_signals)
        uptrend_df.to_csv(uptrend_file, index=False)
        print(f"✅ Uptrend signals saved: {len(uptrend_signals)} symbols → {uptrend_file}")
    else:
        print(f"⚠️ No uptrend signals found")

    if downtrend_signals:
        downtrend_df = pd.DataFrame(downtrend_signals)
        downtrend_df.to_csv(downtrend_file, index=False)
        print(f"✅ Downtrend signals saved: {len(downtrend_signals)} symbols → {downtrend_file}")
    else:
        print(f"⚠️ No downtrend signals found")


# --------------------------------------------------
# Main execution
# --------------------------------------------------
if __name__ == "__main__":
    create_uptrend_downtrend_signals()
    print("\n✅ buy_csv.py completed successfully!")
