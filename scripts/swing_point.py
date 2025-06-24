import pandas as pd
from ta.volatility import BollingerBands
# from ta.momentum import RSIIndicator  # Uncomment if needed later
import numpy as np

def identify_swing_points(df):
    # Bollinger Bands calculate
    bb = BollingerBands(close=df['close'], window=18, window_dev=2, fillna=True)
    df['upper'] = bb.bollinger_hband()
    df['middle'] = bb.bollinger_mavg()
    df['lower'] = bb.bollinger_lband()

    swing_lows = []
    swing_highs = []

    # Loop through data to find swing points
    for i in range(2, len(df) - 4):
        candle = df.iloc[i]
        prev_candle = df.iloc[i - 1]

        # --- Swing Low Condition ---
        if (
            candle['low'] < prev_candle['low'] and
            candle['close'] < candle['middle']
        ):
            for j in range(1, 4):
                current = df.iloc[i + j]
                prev = df.iloc[i + j - 1]

                if (current['close'] > prev['high'] and
                    df.iloc[i + 2]['close'] > current['high']):
                    swing_lows.append((df.index[i], df.index[i + j + 1]))
                    print(f"swing_lows: {swing_lows}")
                    break

        # --- Swing High Condition ---
        if (
            candle['high'] > prev_candle['high'] and
            candle['close'] > candle['middle']
        ):
            for j in range(1, 4):
                current = df.iloc[i + j]
                prev = df.iloc[i + j - 1]

                if (current['close'] < prev['low'] and
                    df.iloc[i + 2]['close'] < current['low']):
                    swing_highs.append((df.index[i], df.index[i + j + 1]))
                    print(f"swing_highs: {swing_highs}")
                    break

    return swing_lows, swing_highs
