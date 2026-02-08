import pandas as pd
import numpy as np
import ta
import os
import traceback

def calculate_macd_historical_full(group):
    """
    Calculate MACD using full historical data for each symbol.
    Returns the group with 'macd', 'macd_signal', and 'macd_hist' columns.
    """
    group = group.copy()
    group = group.sort_values('date').reset_index(drop=True)

    if len(group) < 26:
        group['macd'] = np.nan
        group['macd_signal'] = np.nan
        group['macd_hist'] = np.nan
        return group

    try:
        # ✅ Use FULL historical data to compute MACD (correct approach)
        macd_indicator = ta.trend.MACD(close=group['close'])
        group['macd'] = macd_indicator.macd()
        group['macd_signal'] = macd_indicator.macd_signal()
        group['macd_hist'] = macd_indicator.macd_diff()
    except Exception as e:
        print(f"⚠️ MACD calculation error for {group['symbol'].iloc[0]}: {e}")
        group['macd'] = np.nan
        group['macd_signal'] = np.nan
        group['macd_hist'] = np.nan

    return group

def process_macd_signals():
    # File paths
    input_file = "./csv/mongodb.csv"
    output_dir = "./output/ai_signal"
    output_file = os.path.join(output_dir, "macd.csv")

    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"📂 Reading file: {input_file}")
        df = pd.read_csv(input_file)

        # Column validation
        print(f"📋 Input columns ({len(df.columns)}):")
        print(df.columns.tolist()[:10], "...")
        if 'date' not in df.columns:
            print("❌ 'date' column missing!")
            return None

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        required_cols = ['symbol', 'date', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return None

        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])

        # Data stats
        print(f"\n📊 Data Summary:")
        symbol_stats = df.groupby('symbol').size().reset_index(name='total_days')
        last_dates = df.groupby('symbol')['date'].max().reset_index(name='last_date')
        symbol_stats = pd.merge(symbol_stats, last_dates, on='symbol')

        print(f"  - Total symbols: {len(symbol_stats)}")
        print(f"  - Avg days/symbol: {symbol_stats['total_days'].mean():.1f}")
        print(f"  - Date range: {df['date'].min().date()} to {df['date'].max().date()}")

        low_data = symbol_stats[symbol_stats['total_days'] < 26]
        if len(low_data) > 0:
            print(f"  ⚠️ {len(low_data)} symbols have <26 days (insufficient for MACD)")

        # -------------------------------------------------------------------
        # Step 1: Compute MACD using FULL historical data
        # -------------------------------------------------------------------
        print("\n📈 Computing MACD using full historical data...")
        df = df.groupby('symbol', group_keys=False).apply(calculate_macd_historical_full)

        # -------------------------------------------------------------------
        # Step 2: Get last day for each symbol
        # -------------------------------------------------------------------
        print("\n🎯 Extracting last trading day for each symbol...")
        last_dates_df = df.groupby('symbol')['date'].max().reset_index()
        last_day_data = []

        for _, row in last_dates_df.iterrows():
            symbol = row['symbol']
            last_date = row['date']
            match = df[(df['symbol'] == symbol) & (df['date'] == last_date)]
            if not match.empty:
                last_day_data.append(match.iloc[0])
        last_day_df = pd.DataFrame(last_day_data)
        valid_macd_df = last_day_df.dropna(subset=['macd', 'macd_signal', 'macd_hist'])
        print(f"✅ Found last-day data for {len(last_day_df)} symbols")
        print(f"📊 Valid MACD values for {len(valid_macd_df)} symbols")

        # Show first 5
        if not valid_macd_df.empty:
            print(f"\n🔍 First 5 symbols with MACD:")
            print("="*70)
            for _, row in valid_macd_df.head(5).iterrows():
                print(f"{row['symbol']}: date={row['date'].date()}, "
                      f"Close={row['close']:.2f}, "
                      f"MACD={row['macd']:.6f}, "
                      f"Signal={row['macd_signal']:.6f}, "
                      f"Hist={row['macd_hist']:.6f}")

        # -------------------------------------------------------------------
        # Step 3: Find bullish crossover signals
        # Conditions:
        #   1. Today: MACD > Signal
        #   2. Yesterday: MACD Histogram < 0
        #   3. Today: MACD Histogram > 0
        # -------------------------------------------------------------------
        print(f"\n🔍 Searching for bullish MACD crossover signals...")

        results = []
        for _, last_row in valid_macd_df.iterrows():
            symbol = last_row['symbol']
            last_date = last_row['date']

            # Get all data for this symbol, sorted by date
            symbol_data = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)

            # Find index of last day
            last_idx = symbol_data[symbol_data['date'] == last_date].index
            if len(last_idx) == 0:
                continue
            last_idx = last_idx[0]

            # Need at least one previous day
            if last_idx == 0:
                continue

            prev_row = symbol_data.iloc[last_idx - 1]

            # Skip if previous histogram is NaN
            if pd.isna(prev_row['macd_hist']):
                continue

            # Conditions            cond1 = last_row['macd'] > last_row['macd_signal']          # MACD above signal
            cond2 = prev_row['macd_hist'] < 0                           # Prev hist negative
            cond3 = last_row['macd_hist'] > 0                           # Today hist positive

            if cond1 and cond2 and cond3:
                print(f"✅ {symbol}: {last_date.date()}")
                print(f"   Prev ({prev_row['date'].date()}) hist: {prev_row['macd_hist']:.6f}")
                print(f"   Today hist: {last_row['macd_hist']:.6f}")
                print(f"   MACD: {last_row['macd']:.6f} > Signal: {last_row['macd_signal']:.6f}")
                print(f"   Close: {last_row['close']:.2f}")
                print("-" * 60)

                results.append({
                    'symbol': symbol,
                    'date': last_date,
                    'close': last_row['close'],
                    'macd': last_row['macd'],
                    'macd_signal': last_row['macd_signal'],
                    'macd_hist': last_row['macd_hist'],
                    'prev_macd_hist': prev_row['macd_hist'],
                    'prev_date': prev_row['date']
                })

        # -------------------------------------------------------------------
        # Step 4: Save results
        # -------------------------------------------------------------------
        print("\n" + "="*80)
        if results:
            result_df = pd.DataFrame(results)
            result_df = result_df.sort_values('date', ascending=False)
            result_df.insert(0, 'No', range(1, len(result_df) + 1))

            # Round numeric columns
            num_cols = ['macd', 'macd_signal', 'macd_hist', 'prev_macd_hist']
            for col in num_cols:
                result_df[col] = result_df[col].round(6)

            # Final column order
            output_df = result_df[['No', 'symbol', 'date', 'close',
                                   'macd', 'macd_signal', 'macd_hist', 'prev_macd_hist']]
            output_df.to_csv(output_file, index=False)

            print(f"✅ Found {len(result_df)} bullish MACD crossover signals!")
            print(f"💾 Saved to: {output_file}")

            print(f"\n📈 Signal Details:")
            print("="*100)
            for _, row in result_df.iterrows():
                print(f"{row['No']:3d}. {row['symbol']:10} {row['date'].date()} "
                      f"Close: {row['close']:8.2f} | "                      f"MACD: {row['macd']:7.4f} > {row['macd_signal']:7.4f} | "
                      f"Hist: {row['prev_macd_hist']:7.4f} → {row['macd_hist']:7.4f}")
        else:
            print("❌ No bullish MACD crossover signals found.")
            empty_df = pd.DataFrame(columns=[
                'No', 'symbol', 'date', 'close',
                'macd', 'macd_signal', 'macd_hist', 'prev_macd_hist'
            ])
            empty_df.to_csv(output_file, index=False)
            print(f"💾 Empty file saved: {output_file}")

        return results if results else None

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return None

if __name__ == "__main__":
    process_macd_signals()