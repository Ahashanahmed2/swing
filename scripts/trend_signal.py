import pandas as pd
import os
from datetime import datetime


def create_uptrend_downtrend_signals():
    """
    Create uptrend.csv and downtrend.csv based on price comparison
    """

    # Input file paths
    mongodb_csv = './csv/mongodb.csv'
    trand_base_dir = './csv/trand/'
    output_dir = './csv/'

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Output file paths
    uptrend_file = os.path.join(output_dir, 'uptrand.csv')
    downtrend_file = os.path.join(output_dir, 'downtrand.csv')

    # Read mongodb.csv
    print(f"Reading {mongodb_csv}...")
    try:
        mongodb_df = pd.read_csv(mongodb_csv)
    except FileNotFoundError:
        print(f"❌ Error: {mongodb_csv} not found!")
        return

    # Check required columns
    required_columns = ['symbol', 'date', 'close']
    for col in required_columns:
        if col not in mongodb_df.columns:
            print(f"❌ Required column '{col}' not found in {mongodb_csv}")
            return

    # Convert date to datetime
    mongodb_df['date'] = pd.to_datetime(mongodb_df['date'])

    # Get latest close for each symbol
    latest_data = {}
    for symbol in mongodb_df['symbol'].unique():
        symbol_data = mongodb_df[mongodb_df['symbol'] == symbol] \
            .sort_values('date', ascending=False)

        if not symbol_data.empty:
            latest_data[symbol] = {
                'close': symbol_data.iloc[0]['close'],
                'date': symbol_data.iloc[0]['date']
            }

    print(f"✅ Found {len(latest_data)} symbols")

    uptrend_signals = []
    downtrend_signals = []

    # Process symbols
    for symbol, latest_info in latest_data.items():
        symbol_dir = os.path.join(trand_base_dir, symbol)

        high_file = os.path.join(symbol_dir, 'high.csv')
        low_file = os.path.join(symbol_dir, 'low.csv')

        latest_close = latest_info['close']
        latest_date = latest_info['date']

        # ---------------- UPTREND ----------------
        if os.path.exists(high_file):
            try:
                high_df = pd.read_csv(high_file)

                if len(high_df) >= 2:
                    row_1_price = float(high_df.iloc[0]['price'])
                    row_2_price = float(high_df.iloc[1]['price'])

                    if (
                        row_1_price < latest_close > row_2_price
                        and row_1_price < row_2_price
                    ):
                        uptrend_signals.append({
                            'no': len(uptrend_signals) + 1,
                            'date': latest_date,
                            'symbol': symbol,
                            'close': latest_close,
                            'row_1_price': row_1_price,
                            'row_2_price': row_2_price,
                            'pattern': "Descending Highs Breakout"
                        })
            except Exception as e:
                print(f"⚠️ High file error ({symbol}): {e}")

        # ---------------- DOWNTREND ----------------
        if os.path.exists(low_file):
            try:
                low_df = pd.read_csv(low_file)

                if len(low_df) >= 2:
                    row_1_price = float(low_df.iloc[0]['price'])
                    row_2_price = float(low_df.iloc[1]['price'])

                    if (
                        row_1_price > latest_close < row_2_price
                        and row_1_price > row_2_price
                    ):
                        downtrend_signals.append({
                            'no': len(downtrend_signals) + 1,
                            'date': latest_date,
                            'symbol': symbol,
                            'close': latest_close,
                            'row_1_price': row_1_price,
                            'row_2_price': row_2_price,
                            'pattern': "Ascending Lows Breakdown"
                        })
            except Exception as e:
                print(f"⚠️ Low file error ({symbol}): {e}")

    # Save CSV files
    if uptrend_signals:
        pd.DataFrame(uptrend_signals).to_csv(uptrend_file, index=False)
        print(f"✅ Uptrend saved: {uptrend_file}")
    else:
        print("❌ No uptrend signals found")

    if downtrend_signals:
        pd.DataFrame(downtrend_signals).to_csv(downtrend_file, index=False)
        print(f"✅ Downtrend saved: {downtrend_file}")
    else:
        print("❌ No downtrend signals found")

    print("\n🎯 Trend breakout/breakdown detection completed!")


def main():
    print("=" * 60)
    print("TREND BREAKOUT / BREAKDOWN DETECTION")
    print("=" * 60)
    create_uptrend_downtrend_signals()


if __name__ == "__main__":
    main()