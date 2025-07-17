import pandas as pd
import pandas_ta as ta

def compute_indicators(df):
    """
    Compute basic technical indicators using pandas-ta
    Required columns: ['open', 'high', 'low', 'close', 'volume']
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    # Sort by timestamp to ensure indicators are computed correctly
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')

    # RSI (Relative Strength Index)
    df['rsi_14'] = ta.rsi(df['close'], length=14)

    # EMA (Exponential Moving Average)
    df['ema_20'] = ta.ema(df['close'], length=20)

    # SMA (Simple Moving Average)
    df['sma_50'] = ta.sma(df['close'], length=50)

    # MACD
    macd = ta.macd(df['close'])
    df = pd.concat([df, macd], axis=1)

    # Bollinger Bands
    bbands = ta.bbands(df['close'])
    df = pd.concat([df, bbands], axis=1)

    # Stochastic RSI
    stoch_rsi = ta.stochrsi(df['close'])
    df = pd.concat([df, stoch_rsi], axis=1)

    return df
