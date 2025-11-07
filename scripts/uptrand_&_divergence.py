import pandas as pd
import os

# ---------- 1. Load mongodb.csv ----------
mongo_df = pd.read_csv('./csv/mongodb.csv')
mongo_df['date'] = pd.to_datetime(mongo_df['date'])

uptrend_rows   = []
rsi_div_rows   = []

# ---------- 2. Process each symbol ----------
for symbol, group in mongo_df.groupby('symbol'):
    latest_row = group.sort_values('date').iloc[-1]
    latest_close = latest_row['close']

    ob_file = f'./csv/orderblock/{symbol}.csv'
    if not os.path.exists(ob_file):
        continue

    ob_df = pd.read_csv(ob_file)
    if len(ob_df) < 2 or \
       'orderblock low' not in ob_df.columns or \
       'orderblock high' not in ob_df.columns or \
       'rsi' not in ob_df.columns:
        continue

    ob_df['date'] = pd.to_datetime(ob_df['date'])

    last        = ob_df.iloc[-1]
    second_last = ob_df.iloc[-2]

    ob_low_last   = last['orderblock low']
    ob_high_last  = last['orderblock high']
    ob_low_second = second_last['orderblock low']
    ob_high_second= second_last['orderblock high']

    # ---------- Uptrend condition ----------
    if (
        ob_high_last < latest_close and
        ob_low_second > ob_high_last and
        ob_high_second < latest_close
    ):
        uptrend_rows.append({
            'SYMBOL': symbol,
            'CLOSE': latest_close,
            'Last OB Date': last.get('date', ''),
            'BOS': ob_low_last,
            'Second Last OB Date': second_last.get('date', ''),
            'IDM': ob_high_second
        })

        # ---------- up_candle.csv update (1st script logic) ----------
        up_candle_new = {
            'symbol': symbol,
            'date': latest_row['date'].strftime('%Y-%m-%d'),
            'high': latest_row['high'],
            'low': latest_row['low'],
            'ob_low_last_date': last['date'].strftime('%Y-%m-%d'),
            'ob_low_last': ob_low_last
        }

        up_candle_file = './csv/up_candle.csv'
        if os.path.exists(up_candle_file):
            existing = pd.read_csv(up_candle_file)
            existing = existing[
                ~((existing['symbol'] == symbol) & (existing['date'] == up_candle_new['date']))
            ]
        else:
            existing = pd.DataFrame()

        updated = pd.concat([existing, pd.DataFrame([up_candle_new])], ignore_index=True)
        updated.to_csv(up_candle_file, index=False)
        # -------------------------------------------------------------

        # ---------- RSI divergence loop (2nd script logic) ----------
        for i in range(len(ob_df) - 2, -1, -1):
            candidate = ob_df.iloc[i]
            if (
                candidate['orderblock low'] > ob_low_last and   # price lower low
                candidate['rsi'] < last['rsi']                  # RSI higher low
            ):
                start_date = pd.to_datetime(candidate['date'])
                end_date   = pd.to_datetime(last['date'])
                start_price= candidate['orderblock low']
                end_price  = ob_low_last
                days_diff  = (end_date - start_date).days
                slope      = round((end_price - start_price) / days_diff, 2) if days_diff != 0 else 0.00

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

                # Append only one row per symbol
                rsi_div_rows.append({
                    'SYMBOL': symbol,
                    'Start OB Date': start_date,
                    'Start OB Low': start_price,
                    'End OB Date': end_date,
                    'End OB Low': end_price
                })
                break  # Stop after first match

# ---------- 3. Save outputs ----------
os.makedirs('./output/ai_signal', exist_ok=True)

if uptrend_rows:
    pd.DataFrame(uptrend_rows).to_csv('./output/ai_signal/uptrand.csv', index=False)
    print("✅ uptrand.csv saved to ./output/ai_signal/")
else:
    print("⚠️ No matching uptrend conditions found.")

if rsi_div_rows:
    pd.DataFrame(rsi_div_rows).to_csv('./output/ai_signal/rsi_divergence.csv', index=False)
    print("✅ rsi_divergence.csv saved to ./output/ai_signal/")
else:
    print("⚠️ No RSI divergence found.")
