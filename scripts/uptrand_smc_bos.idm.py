import pandas as pd
import os

# ---------- 1. Existing code (unchanged) ----------
mongo_df = pd.read_csv('./csv/mongodb.csv')
mongo_df['date'] = pd.to_datetime(mongo_df['date'])

uptrend_rows = []
rsi_div_rows = []

for symbol, group in mongo_df.groupby('symbol'):
    latest_row = group.sort_values('date').iloc[-1]
    latest_close = latest_row['close']

    ob_file = f'./csv/orderblock/{symbol}.csv'
    if not os.path.exists(ob_file):
        continue

    ob_df = pd.read_csv(ob_file)
    if len(ob_df) < 2 or 'orderblock low' not in ob_df.columns or 'orderblock high' not in ob_df.columns or 'rsi' not in ob_df.columns:
        continue

    ob_df['date'] = pd.to_datetime(ob_df['date'])

    last = ob_df.iloc[-1]
    second_last = ob_df.iloc[-2]

    ob_low_last = last['orderblock low']
    ob_high_last = last['orderblock high']
    ob_low_second = second_last['orderblock low']
    ob_high_second = second_last['orderblock high']

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

        # --------- 2. NEW: collect latest candle info for up_candle.csv ---------
        up_candle_new = {
            'symbol': symbol,
            'date': latest_row['date'].strftime('%Y-%m-%d'),
            'high': latest_row['high'],
            'low': latest_row['low']
        }

        up_candle_file = './csv/up_candle.csv'
        # Load existing data if file exists
        if os.path.exists(up_candle_file):
            existing = pd.read_csv(up_candle_file)
            # drop duplicates (same symbol+date) just in case
            existing = existing[
                ~((existing['symbol'] == symbol) & (existing['date'] == up_candle_new['date']))
            ]
        else:
            existing = pd.DataFrame()

        updated = pd.concat([existing, pd.DataFrame([up_candle_new])], ignore_index=True)
        updated.to_csv(up_candle_file, index=False)
        # -----------------------------------------------------------------------

        # RSI divergence loop (unchanged)
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
                slope = (end_price - start_price) / days_diff if days_diff != 0 else 0

                symbol_mongo = mongo_df[
                    (mongo_df['symbol'] == symbol) &
                    (mongo_df['date'] >= start_date) &
                    (mongo_df['date'] <= end_date)
                ].copy()

                if symbol_mongo.empty:
                    break

                symbol_mongo['trendline'] = symbol_mongo['date'].apply(
                    lambda d: start_price + slope * (d - start_date).days
                )

                for _, row in symbol_mongo.iterrows():
                    rsi_div_rows.append({
                        'SYMBOL': symbol,
                        'CLOSE': row['close'],
                        'Date': row['date'],
                        'Trendline': row['trendline'],
                        'RSI': row['rsi'],
                        'Start OB Date': start_date,
                        'Start OB Low': start_price,
                        'End OB Date': end_date,
                        'End OB Low': end_price,
                        'Slope': slope
                    })
                break

# ---------- 3. Save final outputs ----------
os.makedirs('./output/ai_signal', exist_ok=True)

if uptrend_rows:
    uptrend_df = pd.DataFrame(uptrend_rows)
    uptrend_df.to_csv('./output/ai_signal/uptrand.csv', index=False)
    print("✅ uptrand.csv saved to ./output/ai_signal/")
else:
    print("⚠️ No matching uptrend conditions found.")

if rsi_div_rows:
    rsi_df = pd.DataFrame(rsi_div_rows)
    rsi_df.to_csv('./output/ai_signal/rsi_divergence.csv', index=False)
    print("✅ rsi_divergence.csv saved to ./output/ai_signal/")
else:
    print("⚠️ No RSI divergence found.")
