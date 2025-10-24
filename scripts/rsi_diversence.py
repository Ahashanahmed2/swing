import pandas as pd
import os

# Load mongodb.csv
mongo_df = pd.read_csv('./csv/mongodb.csv')
mongo_df['date'] = pd.to_datetime(mongo_df['date'])

# Prepare output list
all_divergence_rows = []

# Group by symbol
for symbol, group in mongo_df.groupby('symbol'):
    latest_row = group.sort_values('date').iloc[-1]
    latest_close = latest_row['close']

    ob_file = f'./csv/orderblock/{symbol}.csv'
    if not os.path.exists(ob_file):
        continue

    ob_df = pd.read_csv(ob_file)
    if len(ob_df) < 3 or 'orderblock low' not in ob_df.columns or 'orderblock high' not in ob_df.columns or 'rsi' not in ob_df.columns:
        continue

    ob_df['date'] = pd.to_datetime(ob_df['date'])

    # Latest two rows
    last = ob_df.iloc[-1]
    second_last = ob_df.iloc[-2]

    ob_low_last = last['orderblock low']
    ob_high_last = last['orderblock high']
    ob_low_second = second_last['orderblock low']
    ob_high_second = second_last['orderblock high']

    # Matching condition for uptrend
    if (
        ob_high_last < latest_close and
        ob_low_second > ob_high_last and
        ob_high_second < latest_close
    ):
        # RSI divergence check
        for i in range(len(ob_df) - 2, -1, -1):
            candidate = ob_df.iloc[i]
            if (
                candidate['orderblock low'] < ob_low_last and
                candidate['rsi'] > last['rsi']
            ):
                start_date = pd.to_datetime(candidate['date'])
                end_date = pd.to_datetime(last['date'])
                start_price = candidate['orderblock low']
                end_price = ob_low_last
                days_diff = (end_date - start_date).days
                slope = round((end_price - start_price) / days_diff, 2) if days_diff != 0 else 0.00

                # Intermediate price validation
                intermediate = mongo_df[
                    (mongo_df['symbol'] == symbol) &
                    (mongo_df['date'] > start_date) &
                    (mongo_df['date'] < end_date)
                ]
                if (intermediate['low'] < start_price).any():
                    continue

                # Trendline break validation
                symbol_mongo = mongo_df[
                    (mongo_df['symbol'] == symbol) &
                    (mongo_df['date'] >= start_date) &
                    (mongo_df['date'] <= end_date)
                ].copy()

                if symbol_mongo.empty:
                    break

                symbol_mongo['trendline'] = symbol_mongo['date'].apply(
                    lambda d: round(start_price + slope * (d - start_date).days, 2)
                )

                if (symbol_mongo['low'] < symbol_mongo['trendline']).any():
                    continue

                # Wave-0
                wave_0 = end_price
                wave_0_date = end_date

                # Wave-1
                wave_1_index = ob_df.index.get_loc(last.name) + 1
                if wave_1_index >= len(ob_df):
                    break
                wave_1_row = ob_df.iloc[wave_1_index]
                wave_1 = wave_1_row['orderblock high']
                wave_1_date = wave_1_row['date']

                # Fibonacci retracement
                fib_0_5 = round(wave_0 + 0.5 * (wave_1 - wave_0), 2)

                # Wave-2 detection
                wave_2 = ''
                wave_2_date = None
                ob_after_wave1 = ob_df.iloc[wave_1_index + 1:]
                for _, row in ob_after_wave1.iterrows():
                    ob_low = row['orderblock low']
                    if wave_0 < ob_low < fib_0_5:
                        wave_2 = ob_low
                        wave_2_date = row['date']
                        break

                # Wave-3
                wave_3 = ''
                fib_0_23 = ''
                if wave_2 != '':
                    wave_3 = round(wave_2 + 1.618 * (wave_1 - wave_0), 2)
                    fib_0_23 = round(wave_2 + 0.236 * (wave_3 - wave_2), 2)

                # Wave-4 detection
                wave_4 = ''
                wave_4_date = None
                if wave_3 != '':
                    wave_3_index = ob_after_wave1.index[0] + 1
                    ob_after_wave3 = ob_df.iloc[wave_3_index:]
                    for _, row in ob_after_wave3.iterrows():
                        ob_low = row['orderblock low']
                        if ob_low > wave_1 and ob_low > fib_0_23:
                            wave_4 = ob_low
                            wave_4_date = row['date']
                            break

                # Wave-5 projection
                wave_5 = ''
                if wave_4 != '' and wave_3 != '':
                    wave_5 = round(wave_4 + 1.618 * (wave_4 - wave_3), 2)

                # Final validation
                uptrend_valid = True
                if wave_4 != '':
                    if wave_4 < wave_1 or wave_4 < fib_0_23:
                        uptrend_valid = False
                    else:
                        post_wave4 = mongo_df[
                            (mongo_df['symbol'] == symbol) &
                            (mongo_df['date'] > wave_4_date)
                        ]
                        if (post_wave4['close'] < wave_1).any():
                            uptrend_valid = False
                        if wave_5 != '' and (post_wave4['close'] >= wave_5).any():
                            continue  # wave-5 target hit → skip saving

                # Final row
                all_divergence_rows.append({
                    'SYMBOL': symbol,
                    'Start OB Date': start_date,
                    'Start OB Low': start_price,
                    'End OB Date': end_date,
                    'End OB Low': wave_0,
                    'wave-0': wave_0,
                    'wave-1': wave_1,
                    'wave-2': wave_2,
                    'wave-3': wave_3,
                    'wave-4': wave_4,
                    'wave-5': wave_5,
                    'Uptrend Valid': uptrend_valid
                })
                break

# Save CSVs
os.makedirs('./output/ai_signal', exist_ok=True)
os.makedirs('./csv', exist_ok=True)

if all_divergence_rows:
    all_df = pd.DataFrame(all_divergence_rows)
    all_df.to_csv('./output/ai_signal/all_divergence.csv', index=False)
    all_df.to_csv('./csv/all_divergence.csv', index=False)
    print("✅ all_divergence.csv saved.")
else:
    print("⚠️ No valid RSI divergence found.")
