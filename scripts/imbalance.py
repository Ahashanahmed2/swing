import pandas as pd
import os

# Paths
low_candle_path = './csv/swing/swing_low/low_candle/'
low_confirm_path = './csv/swing/swing_low/low_confirm/'
high_candle_path = './csv/swing/swing_high/high_candle/'
high_confirm_path = './csv/swing/swing_high/high_confirm/'
down_zone_path = './csv/swing/imbalanceZone/down_to_up/'
up_zone_path = './csv/swing/imbalanceZone/up_to_down/'
mongodb_csv = './csv/mongodb.csv'

# Create output dirs if not exists
for path in [down_zone_path, up_zone_path]:
    os.makedirs(path, exist_ok=True)

# ✅ Load mongodb.csv once and filter columns
mongo_df = pd.read_csv(mongodb_csv, parse_dates=['date'], usecols=['symbol', 'date', 'rsi'])

# Combine all unique symbols from both low and high candle folders
symbols_low = {f.replace('.csv', '') for f in os.listdir(low_candle_path) if f.endswith('.csv')}
symbols_high = {f.replace('.csv', '') for f in os.listdir(high_candle_path) if f.endswith('.csv')}
symbols = sorted(symbols_low.union(symbols_high))  # unique list

for symbol in symbols:
    try:
        # ---- Swing Low Zone (Down) ----
        low_candle_file = f'{low_candle_path}{symbol}.csv'
        low_confirm_file = f'{low_confirm_path}{symbol}.csv'
        output_down_file = f'{down_zone_path}{symbol}.csv'

        if os.path.exists(low_candle_file) and os.path.exists(low_confirm_file):
            low_candle_df = pd.read_csv(low_candle_file)
            low_confirm_df = pd.read_csv(low_confirm_file)

            low_candle_df['date'] = pd.to_datetime(low_candle_df['date'])
            low_confirm_df['date'] = pd.to_datetime(low_confirm_df['date'])

            uptrend_rows = []

            for _, candle_row in low_candle_df.iterrows():
                confirm_matches = low_confirm_df[
                    (low_confirm_df['symbol'] == candle_row['symbol']) &
                    (low_confirm_df['date'] > candle_row['date'])
                ]
                if confirm_matches.empty:
                    continue
                confirm_row = confirm_matches.sort_values('date').iloc[0]

                high = candle_row['low']
                low = confirm_row['low']
                fib_382 = low + (high - low) * 0.382
                fib_50 = (high + low) / 2
                fib_618 = low + (high - low) * 0.618

                # ✅ Get RSI from mongodb
                rsi_value = mongo_df[
                    (mongo_df['symbol'] == symbol) &
                    (mongo_df['date'] == confirm_row['date'])
                ]['rsi']
                rsi_value = rsi_value.iloc[0] if not rsi_value.empty else None

                uptrend_rows.append({
                    'symbol': symbol,
                    'date': confirm_row['date'],
                    'low': high,
                    'high': low,
                    'orderblock_date': candle_row['date'],
                    'orderblock_low': candle_row['low'],
                    'orderblock_high': candle_row['high'],
                    'fvg_low': candle_row['high'],
                    'fvg_high': confirm_row['low'],
                    'fib_38.2%': round(fib_382,2),
                    'fib_50%':round(fib_50,2),
                    'fib_61.8%': round(fib_618,2),
                    'rsi': round(rsi_value,2),
                    'trend': 'UpTrend'
                })

            if uptrend_rows:
                new_df = pd.DataFrame(uptrend_rows)
                if os.path.exists(output_down_file):
                    old_df = pd.read_csv(output_down_file)
                    old_df['date'] = pd.to_datetime(old_df['date'])
                    combined_df = pd.concat([old_df, new_df], ignore_index=True)
                    combined_df.drop_duplicates(subset=['symbol', 'date'], inplace=True)
                else:
                    combined_df = new_df
                combined_df.to_csv(output_down_file, index=False)
        else:
            print(f"[INFO] Skipping DOWN zone for {symbol} — missing low candle or confirm file")

        # ---- Swing High Zone (Up) ----
        high_candle_file = f'{high_candle_path}{symbol}.csv'
        high_confirm_file = f'{high_confirm_path}{symbol}.csv'
        output_up_file = f'{up_zone_path}{symbol}.csv'

        if os.path.exists(high_candle_file) and os.path.exists(high_confirm_file):
            high_candle_df = pd.read_csv(high_candle_file)
            high_confirm_df = pd.read_csv(high_confirm_file)

            high_candle_df['date'] = pd.to_datetime(high_candle_df['date'])
            high_confirm_df['date'] = pd.to_datetime(high_confirm_df['date'])

            downtrend_rows = []

            for _, candle_row in high_candle_df.iterrows():
                confirm_matches = high_confirm_df[
                    (high_confirm_df['symbol'] == candle_row['symbol']) &
                    (high_confirm_df['date'] > candle_row['date'])
                ]
                if confirm_matches.empty:
                    continue
                confirm_row = confirm_matches.sort_values('date').iloc[0]

                low = candle_row['high']
                high = confirm_row['high']
                fib_382 = low + (high - low) * 0.382
                fib_50 = (high + low) / 2
                fib_618 = low + (high - low) * 0.618

                # ✅ Get RSI from mongodb
                rsi_value = mongo_df[
                    (mongo_df['symbol'] == symbol) &
                    (mongo_df['date'] == confirm_row['date'])
                ]['rsi']
                rsi_value = rsi_value.iloc[0] if not rsi_value.empty else None

                downtrend_rows.append({
                    'symbol': symbol,
                    'date': confirm_row['date'],
                    'low': high,
                    'high': low,
                    'orderblock_date': candle_row['date'],
                    'orderblock_low': candle_row['low'],
                    'orderblock_high': candle_row['high'],
                    'fvg_low': candle_row['low'],
                    'fvg_high': confirm_row['high'],
                    'fib_38.2%': round(fib_382,2),
                    'fib_50%': round(fib_50,2),
                    'fib_61.8%': round(fib_618,2),
                    'rsi': round(rsi_value,2),
                    'trend': 'DownTrend'
                })

            if downtrend_rows:
                new_df = pd.DataFrame(downtrend_rows)
                if os.path.exists(output_up_file):
                    old_df = pd.read_csv(output_up_file)
                    old_df['date'] = pd.to_datetime(old_df['date'])
                    combined_df = pd.concat([old_df, new_df], ignore_index=True)
                    combined_df.drop_duplicates(subset=['symbol', 'date'], inplace=True)
                else:
                    combined_df = new_df
                combined_df.to_csv(output_up_file, index=False)
        else:
            print(f"[INFO] Skipping UP zone for {symbol} — missing high candle or confirm file")

    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")
