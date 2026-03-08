import pandas as pd
import os



# Ensure the output directory exists
os.makedirs('./output/ai_signal', exist_ok=True)

# Load and prepare data
df = pd.read_csv('./csv/mongodb.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[['symbol', 'date', 'low', 'high', 'rsi']]

divergence_dict = {}

for symbol, group in df.groupby('symbol'):
    group_sorted = group.sort_values(by='date', ascending=False).reset_index(drop=True)
    if len(group_sorted) < 2:
        continue

    last_row = group_sorted.iloc[0]
    last_date = last_row['date']
    last_low = last_row['low']
    last_rsi = last_row['rsi']

    for i in range(1, len(group_sorted)):
        upper_row = group_sorted.iloc[i]
        upper_date = upper_row['date']
        upper_low = upper_row['low']
        upper_rsi = upper_row['rsi']

        if (
            last_rsi >= upper_rsi and
            last_low <= upper_low and
            last_date > upper_date
        ):
            # Calculate line equation: y = m*x + c
            x1 = upper_date.timestamp()
            y1 = upper_low
            x2 = last_date.timestamp()
            y2 = last_low

            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1

            # Filter rows between upper_date and last_date
            mask = (group_sorted['date'] >= upper_date) & (group_sorted['date'] <= last_date)
            segment = group_sorted[mask]

            # Check if any candle low breaks the line
            broken = False
            for _, row in segment.iterrows():
                x = row['date'].timestamp()
                expected_low = m * x + c
                if row['low'] < expected_low:
                    broken = True
                    break

            if not broken:
                divergence_dict[symbol] = {
                    'symbol': symbol,
                    'last row date': last_date.strftime('%Y-%m-%d'),
                    'last row low': last_low,
                    'last row high': last_row['high'],
                    'last row rsi': last_rsi,
                    'second row date': upper_date.strftime('%Y-%m-%d'),
                    'second row low': upper_low,
                    'second row rsi': upper_rsi
                }
            break  # Only first valid match needed

# Save output
output_df = pd.DataFrame(divergence_dict.values())
output_df.to_csv('./csv/rsi_diver.csv', index=False)
output_df.to_csv('./output/ai_signal/rsi_diver.csv', index=False)
