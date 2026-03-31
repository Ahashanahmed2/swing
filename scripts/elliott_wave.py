import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =========================
# Config
# =========================
INPUT_FILE = "./csv/mongodb.csv"
OUTPUT_DIR = "./output/ai_signal"
OUTPUT_FILE = f"{OUTPUT_DIR}/Elliott_wave.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Indicators
# =========================
def calculate_indicators(df):
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

# =========================
# ZigZag
# =========================
def detect_zigzag(df, pct=3):
    swings = []
    last_price = df['close'].iloc[0]
    trend = None

    for i in range(1, len(df)):
        change = (df['close'].iloc[i] - last_price) / last_price * 100

        if trend is None:
            if abs(change) > pct:
                trend = 'up' if change > 0 else 'down'
                last_price = df['close'].iloc[i]
                swings.append({'price': last_price, 'type': trend})

        elif trend == 'up':
            if df['close'].iloc[i] > last_price:
                last_price = df['close'].iloc[i]
                swings[-1]['price'] = last_price
            elif change < -pct:
                trend = 'down'
                last_price = df['close'].iloc[i]
                swings.append({'price': last_price, 'type': 'down'})

        elif trend == 'down':
            if df['close'].iloc[i] < last_price:
                last_price = df['close'].iloc[i]
                swings[-1]['price'] = last_price
            elif change > pct:
                trend = 'up'
                last_price = df['close'].iloc[i]
                swings.append({'price': last_price, 'type': 'up'})

    return swings

# =========================
# Helper
# =========================
def bullish_divergence(df):
    if len(df) < 10:
        return False
    return (df['close'].iloc[-1] < df['close'].iloc[-5] and
            df['rsi'].iloc[-1] > df['rsi'].iloc[-5])

# =========================
# Wave Detection
# =========================
def detect_wave_for_symbol(df, symbol):

    if len(df) < 30:
        return {
            'symbol': symbol,
            'current_wave': 'No Data',
            'confidence_score': 0
        }

    df = calculate_indicators(df)
    swings = detect_zigzag(df)

    if len(swings) < 3:
        return {
            'symbol': symbol,
            'current_wave': 'Sideways',
            'confidence_score': 30
        }

    types = [s['type'] for s in swings]
    prices = [s['price'] for s in swings]

    price = df['close'].iloc[-1]
    rsi = round(df['rsi'].iloc[-1], 1)

    confidence = 50

    current_wave = "Unknown"
    direction = "Sideways"

    # =========================
    # Bullish Impulse
    # =========================
    if len(types) >= 5 and types[-5:] == ['up','down','up','down','up']:

        p = prices[-5:]

        wave1 = abs(p[1] - p[0])
        wave2 = abs(p[2] - p[1])
        wave3 = abs(p[3] - p[2])

        retrace = wave2 / wave1 if wave1 else 0

        if 0.5 <= retrace <= 0.7:
            confidence += 10

        if wave3 > wave1:
            confidence += 15

        if bullish_divergence(df):
            confidence += 10

        current_wave = "Impulse"
        direction = "Bullish"

        entry = f"{price*0.97:.2f}-{price:.2f}"
        sl = f"{price*0.94:.2f}"
        tp = f"{price*1.12:.2f}"

    # =========================
    # Bearish Impulse
    # =========================
    elif len(types) >= 5 and types[-5:] == ['down','up','down','up','down']:

        p = prices[-5:]

        wave1 = abs(p[1] - p[0])
        wave2 = abs(p[2] - p[1])
        wave3 = abs(p[3] - p[2])

        if wave1:
            retrace = wave2 / wave1
            if 0.5 <= retrace <= 0.7:
                confidence += 10

        if wave3 > wave1:
            confidence += 15

        current_wave = "Impulse"
        direction = "Bearish"

        entry = f"{price*1.00:.2f}-{price*1.03:.2f}"
        sl = f"{price*1.05:.2f}"
        tp = f"{price*0.90:.2f}"

    # =========================
    # Correction
    # =========================
    elif len(types) >= 3 and types[-3:] == ['down','up','down']:

        p = prices[-3:]
        a = abs(p[1] - p[0])
        b = abs(p[2] - p[1])

        ratio = b / a if a else 0

        if ratio < 0.8:
            current_wave = "Zigzag"
            confidence += 10
        elif 0.9 <= ratio <= 1.1:
            current_wave = "Flat"
            confidence += 15

        if bullish_divergence(df):
            confidence += 10

        direction = "Bullish"

        entry = f"{price*0.96:.2f}"
        sl = f"{price*0.92:.2f}"
        tp = f"{price*1.10:.2f}"

    else:
        current_wave = "Sideways"
        direction = "Range"
        entry, sl, tp = "N/A", "N/A", "N/A"
        confidence = 30

    confidence = min(confidence, 100)

    return {
        'symbol': symbol,
        'price': round(price,2),
        'wave': current_wave,
        'direction': direction,
        'entry': entry,
        'sl': sl,
        'tp': tp,
        'confidence': confidence,
        'rsi': rsi
    }

# =========================
# Main
# =========================
def main():
    df = pd.read_csv(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])

    results = []

    for symbol in df['symbol'].unique():
        s_df = df[df['symbol']==symbol].sort_values('date')
        res = detect_wave_for_symbol(s_df, symbol)
        results.append(res)

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_FILE, index=False)

    print("✅ Done:", OUTPUT_FILE)

if __name__ == "__main__":
    main()