import pandas as pd
from ta.momentum import RSIIndicator
import numpy as np

def identify_swing_points(df):
    # RSI Indicator calculate
    rsi_indicator = RSIIndicator(close=df['close'], window=14, fillna=True)
    df['rsi'] = rsi_indicator.rsi()

    swing_lows = []
    swing_highs = []

    # Loop through data to find swing points
    for i in range(2, len(df) - 6):
        candle = df.iloc[i]
        prev_candle = df.iloc[i - 1]

        # --- Swing Low Condition ---
        if (
            candle['low'] < prev_candle['low']
        ):
            for j in range(1, 6):
                current = df.iloc[i + j]
                prev = df.iloc[i + j - 1]

                if (current['close'] > prev['high'] and
                    df.iloc[i + 2]['close'] > current['high']):
                    swing_lows.append((df.index[i], df.index[i + j + 1]))
                    print(f"swing_lows: {swing_lows}")
                    break

        # --- Swing High Condition ---
        if (
            candle['high'] > prev_candle['high']
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
