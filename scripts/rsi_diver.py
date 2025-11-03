import pandas as pd

# Load the CSV file
df = pd.read_csv('./csv/mongodb.csv')

# Ensure proper column names
required_columns = ['symbol', 'date', 'low', 'high', 'rsi']
df = df[required_columns]

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Prepare dictionary to store latest valid divergence per symbol
divergence_dict = {}

# Group by symbol
for symbol, group in df.groupby('symbol'):
    group_sorted = group.sort_values(by='date', ascending=False).reset_index(drop=True)
    if len(group_sorted) < 2:
        continue

    last_row = group_sorted.iloc[0]
    last_low = last_row['low']
    last_rsi = last_row['rsi']
    last_date = last_row['date']

    # Check upper rows one by one
    for i in range(1, len(group_sorted)):
        upper_row = group_sorted.iloc[i]
        upper_low = upper_row['low']
        upper_rsi = upper_row['rsi']
        upper_date = upper_row['date']

        # Condition: last rsi > upper rsi AND last low ≤ upper low AND same date
        if (
            last_rsi > upper_rsi and
            last_low <= upper_low and
            last_date == upper_date
        ):
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

# Convert to DataFrame and save
output_df = pd.DataFrame(divergence_dict.values())
output_df.to_csv('./csv/rsi_diver.csv', index=False)

print("RSI divergence saved to ./rsi_divergence.csv")
