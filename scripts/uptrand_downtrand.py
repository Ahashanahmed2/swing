import pandas as pd
import os
from datetime import datetime

# Paths
mongodb_path = "./csv/mongodb.csv"
high_zone_path = "./csv/swing/imbalanceZone/down_to_up/"
low_zone_path = "./csv/swing/imbalanceZone/up_to_down/"
output_path = "./output/ai_signal/trend_changes.csv"

# Today's date
today = pd.Timestamp(datetime.now().date())

# Columns to load
usecols = ['symbol', 'date', 'close']

# ✅ Load MongoDB CSV
try:
    df_main = pd.read_csv(mongodb_path, usecols=usecols, parse_dates=['date'])
except Exception as e:
    print(f"❌ Failed to read mongodb.csv: {e}")
    exit()

# Ensure numeric + clean data
df_main['close'] = pd.to_numeric(df_main['close'], errors='coerce')
df_main.dropna(subset=['close'], inplace=True)

# Group by symbol
grouped = df_main.groupby('symbol')

# ✅ Store today's trend changes
today_trend_changes = []

# ✅ Check trend change function
def check_trend_change(symbol, df_group):
    file_high = os.path.join(high_zone_path, f"{symbol}.csv")
    file_low = os.path.join(low_zone_path, f"{symbol}.csv")

    df_group_sorted = df_group.sort_values('date')
    last_date = df_group_sorted.iloc[-1]['date']
    last_close = df_group_sorted.iloc[-1]['close']

    try:
        # 🔽 যদি high zone থাকে (UpTrend)
        if os.path.exists(file_high):
            df_high = pd.read_csv(file_high)
            df_high['high'] = pd.to_numeric(df_high['high'], errors='coerce')
            last_high = df_high['high'].dropna().iloc[-1]

            if last_close < last_high and last_date.date() == today.date():
                print(f"[{symbol}] 🔻 UpTrend → DownTrend")
                os.remove(file_high)
                df_high['trand'] = 'DownTrend'
                df_high.to_csv(file_low, index=False)
                today_trend_changes.append({
                    'symbol': symbol,
                    'trend_change': '🔻 UpTrend → DownTrend',
                    'date': today.date()
                })

        # 🔼 যদি low zone থাকে (DownTrend)
        elif os.path.exists(file_low):
            df_low = pd.read_csv(file_low)
            df_low['low'] = pd.to_numeric(df_low['low'], errors='coerce')
            last_low = df_low['low'].dropna().iloc[-1]

            if last_close > last_low and last_date.date() == today.date():
                print(f"[{symbol}] 🔺 DownTrend → UpTrend")
                os.remove(file_low)
                df_low['trand'] = 'UpTrend'
                df_low.to_csv(file_high, index=False)
                today_trend_changes.append({
                    'symbol': symbol,
                    'trend_change': '🔺 DownTrend → UpTrend',
                    'date': today.date()
                })

    except Exception as e:
        print(f"[{symbol}] ❌ Trend check error: {e}")

# ✅ Run for all symbols
print("🔍 Checking trend changes...")
for symbol, df_group in grouped:
    check_trend_change(symbol, df_group)

# ✅ Save only if today's trend changes occurred
if today_trend_changes:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_changes = pd.DataFrame(today_trend_changes)
    df_changes.to_csv(output_path, index=False)
    print(f"✅ Today's trend changes saved to {output_path}")
else:
    print("ℹ️ No trend changes detected for today. No CSV file created.")

print("✅ Trend analysis completed.")
