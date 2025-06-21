#scripts/swing_point.py
import pandas as pd
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator


import numpy as np



def identify_swing_points(df):
    # Bollinger Bands calculate
    bb = BollingerBands(close=df['close'], window=18, window_dev=2, fillna=True)
    df['upper'] = bb.bollinger_hband()
    df['middle'] = bb.bollinger_mavg()
    df['lower'] = bb.bollinger_lband()


    rsi = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi.rsi()


    swing_lows = []
    swing_highs = []

    # Loop through data to find swing lows
    for i in range(2, len(df) - 2):
        candle = df.iloc[i]
        prev_candle = df.iloc[i - 1]
        next_candle = df.iloc[i + 1]
        next2_candle = df.iloc[i + 2]

        # Swing Low condition
        if (
            candle['low'] < prev_candle['low'] and
            candle['low'] <= candle['lower'] and
            candle['close'] < candle['middle'] and
            next_candle['close'] > candle['high'] and
            next_candle['high'] < next2_candle['close']
        ):
            swing_lows.append((df.index[i],df.index[i+2]))
        #__SH condition__
        if(  candle['high'] > prev_candle['high'] and
             candle['close'] > candle['middle'] and
             next_candle['close'] < candle['low'] and 
             next2_candle['close'] < next_candle['close']):

             swing_highs.append((df.index[i],df.index[i+2]))

    

    return swing_lows, swing_highs