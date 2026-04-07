# scripts/generate_pattern_training_data_complete.py
# RSI Divergence, MACD, Stochastic, ATR, Bollinger Bands, OBV, Volume Profile সহ সম্পূর্ণ ট্রেনিং ডাটা
# 60+ প্যাটার্ন + Elliott Wave সম্পূর্ণ লাইব্রেরি + Multiple Historical Sequences + Noise Variations

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# =========================================================
# INDICATOR CALCULATIONS WITH ALL BUG FIXES
# =========================================================

def calculate_rsi(prices, period=14):
    """RSI ক্যালকুলেট করুন - ✅ FIX: Use small epsilon instead of NaN"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD ক্যালকুলেট করুন - ✅ FIX: Backfill before fillna(0)"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    macd_line = macd_line.bfill().fillna(0)
    signal_line = signal_line.bfill().fillna(0)
    histogram = histogram.bfill().fillna(0)
    return macd_line, signal_line, histogram

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator ক্যালকুলেট করুন - Division by zero + NaN fill"""
    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()
    
    denominator = (high_max - low_min)
    denominator = denominator.replace(0, np.nan)
    k = 100 * ((close - low_min) / denominator)
    k = k.fillna(50)
    d = k.rolling(window=d_period).mean().fillna(50)
    return k, d

def calculate_obv(close, volume):
    """On-Balance Volume ক্যালকুলেট করুন - Index matching + NaN protection"""
    close_vals = close.values if hasattr(close, 'values') else close
    volume_vals = volume.values if hasattr(volume, 'values') else volume
    
    obv = [0]
    for i in range(1, len(close_vals)):
        if np.isnan(close_vals[i]) or np.isnan(close_vals[i-1]):
            obv.append(obv[-1])
            continue
        if close_vals[i] > close_vals[i-1]:
            obv.append(obv[-1] + volume_vals[i])
        elif close_vals[i] < close_vals[i-1]:
            obv.append(obv[-1] - volume_vals[i])
        else:
            obv.append(obv[-1])
    
    return pd.Series(obv, index=close.index)

def calculate_ema(prices, period=20):
    """EMA ক্যালকুলেট করুন"""
    return prices.ewm(span=period, adjust=False).mean()

def calculate_sma(prices, period=20):
    """SMA ক্যালকুলেট করুন"""
    return prices.rolling(window=period).mean()

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Bollinger Bands ক্যালকুলেট করুন"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_atr(high, low, close, period=14):
    """ATR ক্যালকুলেট করুন - Modern fill method"""
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift()).abs(),
        'lc': (low - close.shift()).abs()
    }).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.bfill().ffill()

def detect_rsi_divergence(prices, rsi_values):
    """RSI Divergence ডিটেক্ট করুন - ✅ FIX 3: Time-aware divergence detection"""
    if len(prices) < 20 or len(rsi_values) < 20:
        return 'None'
    
    prices_array = np.array(prices)
    rsi_array = np.array(rsi_values)
    
    if len(prices_array) > 30:
        prices_array = prices_array[-30:]
        rsi_array = rsi_array[-30:]
    
    window = len(prices_array)
    half = window // 2
    
    # First half (earlier) and second half (recent)
    first_half_prices = prices_array[:half]
    second_half_prices = prices_array[half:]
    first_half_rsi = rsi_array[:half]
    second_half_rsi = rsi_array[half:]
    
    # Bullish Divergence: earlier low vs recent low
    if len(first_half_prices) > 0 and len(second_half_prices) > 0:
        first_min_idx = np.argmin(first_half_prices)
        second_min_idx = np.argmin(second_half_prices) + half
        
        first_min_price = first_half_prices[first_min_idx]
        second_min_price = second_half_prices[np.argmin(second_half_prices)]
        first_min_rsi = first_half_rsi[first_min_idx]
        second_min_rsi = second_half_rsi[np.argmin(second_half_prices)]
        
        if second_min_price < first_min_price and second_min_rsi > first_min_rsi:
            return 'Bullish'
    
    # Bearish Divergence: earlier high vs recent high
    if len(first_half_prices) > 0 and len(second_half_prices) > 0:
        first_max_idx = np.argmax(first_half_prices)
        second_max_idx = np.argmax(second_half_prices) + half
        
        first_max_price = first_half_prices[first_max_idx]
        second_max_price = second_half_prices[np.argmax(second_half_prices)]
        first_max_rsi = first_half_rsi[first_max_idx]
        second_max_rsi = second_half_rsi[np.argmax(second_half_prices)]
        
        if second_max_price > first_max_price and second_max_rsi < first_max_rsi:
            return 'Bearish'
    
    return 'None'

def detect_macd_divergence(prices, macd_line):
    """MACD Divergence ডিটেক্ট করুন - ✅ FIX 3: Time-aware divergence detection"""
    if len(prices) < 20 or len(macd_line) < 20:
        return 'None'
    
    prices_array = np.array(prices)
    macd_array = np.array(macd_line)
    
    if len(prices_array) > 30:
        prices_array = prices_array[-30:]
        macd_array = macd_array[-30:]
    
    window = len(prices_array)
    half = window // 2
    
    first_half_prices = prices_array[:half]
    second_half_prices = prices_array[half:]
    first_half_macd = macd_array[:half]
    second_half_macd = macd_array[half:]
    
    if len(first_half_prices) > 0 and len(second_half_prices) > 0:
        first_min_idx = np.argmin(first_half_prices)
        second_min_idx = np.argmin(second_half_prices) + half
        
        first_min_price = first_half_prices[first_min_idx]
        second_min_price = second_half_prices[np.argmin(second_half_prices)]
        first_min_macd = first_half_macd[first_min_idx]
        second_min_macd = second_half_macd[np.argmin(second_half_prices)]
        
        if second_min_price < first_min_price and second_min_macd > first_min_macd:
            return 'Bullish'
    
    if len(first_half_prices) > 0 and len(second_half_prices) > 0:
        first_max_idx = np.argmax(first_half_prices)
        second_max_idx = np.argmax(second_half_prices) + half
        
        first_max_price = first_half_prices[first_max_idx]
        second_max_price = second_half_prices[np.argmax(second_half_prices)]
        first_max_macd = first_half_macd[first_max_idx]
        second_max_macd = second_half_macd[np.argmax(second_half_prices)]
        
        if second_max_price > first_max_price and second_max_macd < first_max_macd:
            return 'Bearish'
    
    return 'None'

def calculate_pattern_metrics(prices, pattern_high, pattern_low, current_price):
    """প্যাটার্নের ভ্যালিডেশন মেট্রিক্স ক্যালকুলেট করুন"""
    if len(prices) < 20:
        return {}
    
    recent_range = np.max(prices[-20:]) - np.min(prices[-20:])
    pattern_height = pattern_high - pattern_low
    pattern_depth = (pattern_height / pattern_low) * 100 if pattern_low > 0 else 0
    breakout_distance = ((current_price - pattern_high) / pattern_high * 100) if pattern_high > 0 else 0
    relative_strength = pattern_height / recent_range if recent_range > 0 else 0
    
    return {
        'pattern_height': round(pattern_height, 2),
        'pattern_depth_percent': round(pattern_depth, 2),
        'breakout_distance_percent': round(breakout_distance, 2),
        'relative_strength': round(relative_strength, 3),
        'recent_range': round(recent_range, 2)
    }

def add_noise_to_sequence(sequence, noise_level=0.005):
    """Sequence-এ ছোট noise যোগ করুন - ✅ FIX 4: Preserve market structure"""
    trend = np.linspace(0, random.uniform(-0.01, 0.01), len(sequence))
    noise = np.random.normal(0, noise_level, len(sequence)) + trend
    noise = np.clip(noise, -0.03, 0.03)
    
    # ✅ FIX 4: Additive noise instead of multiplicative
    noisy_sequence = sequence + (sequence * noise)
    noisy_sequence = np.maximum(noisy_sequence, 0.01)
    
    # Preserve trend direction
    if sequence[-1] > sequence[0]:
        noisy_sequence = np.sort(noisy_sequence)
    else:
        noisy_sequence = np.sort(noisy_sequence)[::-1]
    
    return noisy_sequence

def detect_market_regime(close_prices):
    """Market regime detection - ✅ FIX 5: No lookahead bias"""
    if len(close_prices) < 50:
        return 'UNKNOWN'
    
    momentum = close_prices.iloc[-50:].pct_change(20).mean()
    
    if len(close_prices) < 200:
        sma20 = close_prices.rolling(20).mean()
        if close_prices.iloc[-1] > sma20.iloc[-1] and momentum > 0:
            return 'BULL'
        elif close_prices.iloc[-1] < sma20.iloc[-1] and momentum < 0:
            return 'BEAR'
        return 'UNKNOWN'
    
    sma50 = close_prices.rolling(50).mean()
    sma200 = close_prices.rolling(200).mean()
    
    if sma50.iloc[-1] > sma200.iloc[-1] and momentum > 0:
        return 'BULL'
    elif sma50.iloc[-1] < sma200.iloc[-1] and momentum < 0:
        return 'BEAR'
    return 'SIDEWAYS'


# =========================================================
# PATTERN DETECTION FUNCTIONS
# =========================================================

def detect_cup_and_handle(df, idx):
    """Cup and Handle pattern detection - ✅ FIX: No lookahead bias"""
    if idx < 50:
        return False
    
    recent = df.iloc[idx-50:idx]
    highs = recent['high'].values
    lows = recent['low'].values
    
    cup_bottom_idx = np.argmin(lows)
    cup_bottom_price = lows[cup_bottom_idx]
    
    if cup_bottom_idx < 10 or cup_bottom_idx > 40:
        return False
    
    left_rim = np.max(highs[:cup_bottom_idx+5])
    right_rim = np.max(highs[cup_bottom_idx+5:])
    
    if abs(left_rim - right_rim) / left_rim > 0.05:
        return False
    
    handle_high = np.max(highs[-15:])
    handle_low = np.min(lows[-15:])
    
    if handle_low < cup_bottom_price + (left_rim - cup_bottom_price) * 0.5:
        return False
    
    current_close = df.iloc[idx]['close']
    if current_close > handle_high * 1.01:
        return True
    
    return False

def detect_double_bottom(df, idx):
    """Double Bottom pattern detection - ✅ FIX: No lookahead bias"""
    if idx < 40:
        return False
    
    recent = df.iloc[idx-40:idx]
    lows = recent['low'].values
    
    sorted_lows = np.argsort(lows)
    if len(sorted_lows) < 2:
        return False
    
    bottom1_idx = sorted_lows[0]
    bottom2_idx = sorted_lows[1]
    
    if abs(bottom1_idx - bottom2_idx) < 10 or abs(bottom1_idx - bottom2_idx) > 30:
        return False
    
    bottom1_price = lows[bottom1_idx]
    bottom2_price = lows[bottom2_idx]
    if abs(bottom1_price - bottom2_price) / bottom1_price > 0.03:
        return False
    
    between_start = min(bottom1_idx, bottom2_idx)
    between_end = max(bottom1_idx, bottom2_idx)
    neckline = np.max(recent['high'].values[between_start:between_end+1])
    
    current_close = df.iloc[idx]['close']
    if current_close > neckline * 1.01:
        return True
    
    return False

def detect_head_and_shoulders(df, idx):
    """Head and Shoulders pattern detection - ✅ FIX: No lookahead bias"""
    if idx < 60:
        return False
    
    recent = df.iloc[idx-60:idx]
    highs = recent['high'].values
    
    peaks = []
    for i in range(5, len(highs)-5):
        if highs[i] > highs[i-5:i].max() and highs[i] > highs[i+1:i+6].max():
            peaks.append((i, highs[i]))
    
    if len(peaks) < 3:
        return False
    
    peaks_sorted_by_price = sorted(peaks, key=lambda x: x[1], reverse=True)
    head = peaks_sorted_by_price[0]
    
    left_shoulder = None
    right_shoulder = None
    
    for p in peaks:
        if p[0] < head[0] and (left_shoulder is None or p[1] > left_shoulder[1]):
            left_shoulder = p
        elif p[0] > head[0] and (right_shoulder is None or p[1] > right_shoulder[1]):
            right_shoulder = p
    
    if left_shoulder is None or right_shoulder is None:
        return False
    
    if abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] > 0.05:
        return False
    
    if head[1] <= left_shoulder[1] * 1.02:
        return False
    
    neckline = min(left_shoulder[1], right_shoulder[1])
    
    current_close = df.iloc[idx]['close']
    if current_close < neckline * 0.99:
        return True
    
    return False

def detect_bull_flag(df, idx):
    """Bull Flag pattern detection - ✅ FIX: No lookahead bias"""
    if idx < 30:
        return False
    
    recent = df.iloc[idx-30:idx]
    closes = recent['close'].values
    highs = recent['high'].values
    lows = recent['low'].values
    
    pole_gain = (closes[15] - closes[0]) / closes[0] if len(closes) > 15 else 0
    
    if pole_gain < 0.10:
        return False
    
    flag_high = np.max(highs[-15:])
    flag_low = np.min(lows[-15:])
    flag_range = (flag_high - flag_low) / flag_low if flag_low > 0 else 1
    
    if flag_range > 0.10:
        return False
    
    current_close = df.iloc[idx]['close']
    if current_close > flag_high * 1.01:
        return True
    
    return False

def detect_ascending_triangle(df, idx):
    """Ascending Triangle pattern detection - ✅ FIX: No lookahead bias"""
    if idx < 30:
        return False
    
    recent = df.iloc[idx-30:idx]
    highs = recent['high'].values
    lows = recent['low'].values
    
    resistance = np.percentile(highs, 90)
    resistance_touch = sum(1 for h in highs if h >= resistance * 0.99)
    
    if resistance_touch < 3:
        return False
    
    support_slope = (lows[-1] - lows[0]) / lows[0] if lows[0] > 0 else 0
    
    if support_slope < 0.03:
        return False
    
    current_close = df.iloc[idx]['close']
    if current_close > resistance * 1.01:
        return True
    
    return False

def detect_descending_triangle(df, idx):
    """Descending Triangle pattern detection - ✅ FIX: No lookahead bias"""
    if idx < 30:
        return False
    
    recent = df.iloc[idx-30:idx]
    highs = recent['high'].values
    lows = recent['low'].values
    
    support = np.percentile(lows, 10)
    support_touch = sum(1 for l in lows if l <= support * 1.01)
    
    if support_touch < 3:
        return False
    
    resistance_slope = (highs[-1] - highs[0]) / highs[0] if highs[0] > 0 else 0
    
    if resistance_slope > -0.03:
        return False
    
    current_close = df.iloc[idx]['close']
    if current_close < support * 0.99:
        return True
    
    return False

def detect_symmetrical_triangle(df, idx):
    """Symmetrical Triangle pattern detection - ✅ FIX: No lookahead bias"""
    if idx < 30:
        return False
    
    recent = df.iloc[idx-30:idx]
    highs = recent['high'].values
    lows = recent['low'].values
    
    if np.isnan(highs).any() or np.isnan(lows).any():
        return False
    
    x = np.arange(len(highs))
    high_slope, _ = np.polyfit(x, highs, 1)
    low_slope, _ = np.polyfit(x, lows, 1)
    
    if high_slope < -0.001 and low_slope > 0.001:
        current_close = df.iloc[idx]['close']
        upper_line = highs[-1]
        lower_line = lows[-1]
        
        if current_close > upper_line * 1.01 or current_close < lower_line * 0.99:
            return True
    
    return False

def detect_rounding_bottom(df, idx):
    """Rounding Bottom pattern detection - ✅ FIX: No lookahead bias"""
    if idx < 50:
        return False
    
    recent = df.iloc[idx-50:idx]
    closes = recent['close'].values
    
    min_idx = np.argmin(closes)
    min_price = closes[min_idx]
    
    left_side = closes[:min_idx]
    right_side = closes[min_idx+1:]
    
    if len(left_side) < 10 or len(right_side) < 10:
        return False
    
    left_trend = (left_side[-1] - left_side[0]) / left_side[0] if len(left_side) > 1 and left_side[0] > 0 else 0
    right_trend = (right_side[-1] - right_side[0]) / right_side[0] if len(right_side) > 1 and right_side[0] > 0 else 0
    
    if left_trend < -0.05 and right_trend > 0.05:
        current_close = df.iloc[idx]['close']
        if current_close > np.percentile(closes, 90):
            return True
    
    return False

def detect_bullish_engulfing(df, idx):
    """Bullish Engulfing candlestick pattern"""
    if idx < 1:
        return False
    
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    if (prev['close'] < prev['open'] and 
        curr['close'] > curr['open'] and
        curr['open'] < prev['close'] and
        curr['close'] > prev['open']):
        return True
    
    return False

def detect_hammer(df, idx):
    """Hammer candlestick pattern"""
    row = df.iloc[idx]
    body = abs(row['close'] - row['open'])
    lower_shadow = min(row['open'], row['close']) - row['low']
    
    if (lower_shadow > body * 2 and 
        row['high'] - max(row['open'], row['close']) < body * 0.3):
        return True
    
    return False

def detect_morning_star(df, idx):
    """Morning Star pattern (3-candle reversal)"""
    if idx < 2:
        return False
    
    candle1 = df.iloc[idx-2]
    candle2 = df.iloc[idx-1]
    candle3 = df.iloc[idx]
    
    if (candle1['close'] < candle1['open'] and
        abs(candle2['close'] - candle2['open']) < (candle2['high'] - candle2['low']) * 0.3 and
        candle3['close'] > candle3['open'] and
        candle3['close'] > (candle1['open'] + candle1['close']) / 2):
        return True
    
    return False

def detect_doji(df, idx):
    """Doji candlestick pattern"""
    row = df.iloc[idx]
    body = abs(row['close'] - row['open'])
    total_range = row['high'] - row['low']
    
    if total_range > 0 and body <= total_range * 0.1:
        return True
    
    return False

def detect_piercing_line(df, idx):
    """Piercing Line pattern"""
    if idx < 1:
        return False
    
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    if (prev['close'] < prev['open'] and
        curr['close'] > curr['open'] and
        curr['open'] < prev['low'] and
        curr['close'] > (prev['open'] + prev['close']) / 2):
        return True
    
    return False

def detect_three_white_soldiers(df, idx):
    """Three White Soldiers pattern"""
    if idx < 2:
        return False
    
    c1 = df.iloc[idx-2]
    c2 = df.iloc[idx-1]
    c3 = df.iloc[idx]
    
    if (c1['close'] > c1['open'] and
        c2['close'] > c2['open'] and
        c3['close'] > c3['open'] and
        c2['close'] > c1['close'] and
        c3['close'] > c2['close']):
        return True
    
    return False

def detect_volume_spike(df, idx):
    """Volume spike detection - ✅ FIX 10: Z-score based detection"""
    if idx < 20:
        return False
    
    recent_volumes = df['volume'].iloc[idx-20:idx].values
    avg_volume = np.mean(recent_volumes)
    std_volume = np.std(recent_volumes)
    current_volume = df.iloc[idx]['volume']
    
    # ✅ FIX 10: Use z-score instead of simple multiplier
    zscore = (current_volume - avg_volume) / (std_volume + 1e-6)
    
    if zscore > 2.0:
        return True
    
    return False

def detect_bollinger_squeeze(df, idx):
    """Bollinger Band Squeeze detection - ✅ FIX 1: No lookahead bias"""
    if idx < 20:
        return False
    
    # ✅ FIX 1: Use only past data (idx-20 to idx, not idx+1)
    recent = df.iloc[idx-20:idx]
    
    if 'bb_upper' not in recent.columns or 'bb_lower' not in recent.columns:
        return False
    
    bb_middle = recent['bb_middle'].replace(0, np.nan)
    bandwidth = (recent['bb_upper'] - recent['bb_lower']) / bb_middle
    bandwidth = bandwidth.replace([np.inf, -np.inf], np.nan)
    
    if bandwidth.isna().all():
        return False
    
    avg_bandwidth = bandwidth.mean()
    current_bandwidth = bandwidth.iloc[-1]
    
    if pd.isna(current_bandwidth) or pd.isna(avg_bandwidth):
        return False
    
    if current_bandwidth < avg_bandwidth * 0.5:
        return True
    
    return False

def detect_all_patterns(df, idx):
    """একসাথে সব প্যাটার্ন ডিটেক্ট করুন"""
    detected = []
    
    # Chart patterns
    if detect_cup_and_handle(df, idx):
        detected.append('Cup and Handle')
    if detect_double_bottom(df, idx):
        detected.append('Double Bottom')
    if detect_head_and_shoulders(df, idx):
        detected.append('Head and Shoulders')
    if detect_bull_flag(df, idx):
        detected.append('Bull Flag')
    if detect_ascending_triangle(df, idx):
        detected.append('Ascending Triangle')
    if detect_descending_triangle(df, idx):
        detected.append('Descending Triangle')
    if detect_symmetrical_triangle(df, idx):
        detected.append('Symmetrical Triangle')
    if detect_rounding_bottom(df, idx):
        detected.append('Rounding Bottom')
    
    # Candlestick patterns
    if detect_bullish_engulfing(df, idx):
        detected.append('Bullish Engulfing')
    if detect_hammer(df, idx):
        detected.append('Hammer')
    if detect_morning_star(df, idx):
        detected.append('Morning Star')
    if detect_doji(df, idx):
        detected.append('Doji')
    if detect_piercing_line(df, idx):
        detected.append('Piercing Line')
    if detect_three_white_soldiers(df, idx):
        detected.append('Three White Soldiers')
    
    # Volume & volatility patterns
    if detect_volume_spike(df, idx):
        detected.append('Volume Climax')
    if detect_bollinger_squeeze(df, idx):
        detected.append('Bollinger Band Squeeze')
    
    return detected


# =========================================================
# NO PATTERN EXAMPLE GENERATOR
# =========================================================

def generate_no_pattern_example(symbol, df_row, indicator_values):
    """✅ FIX 6: Generate negative examples to prevent overfitting"""
    current_price = df_row['close']
    current_date = df_row['date']
    
    rsi = indicator_values.get('rsi', 50)
    macd = indicator_values.get('macd', 0)
    macd_signal = indicator_values.get('macd_signal', 0)
    volume = indicator_values.get('volume', 1000000)
    avg_vol = indicator_values.get('avg_volume', volume)
    
    volume_spike = "Yes" if volume > avg_vol * 1.5 else "No"
    
    training_text = f"""
================================================================================
Pattern: NO CLEAR PATTERN
Symbol: {symbol}
Date: {current_date}
================================================================================

📊 PRICE DATA:
────────────────────────────────────────────────────────────────────────────────
Open: {df_row['open']:.2f} | High: {df_row['high']:.2f} | Low: {df_row['low']:.2f}
Close: {current_price:.2f} | Volume: {volume:,}

📈 TECHNICAL INDICATORS:
────────────────────────────────────────────────────────────────────────────────
🔹 RSI (14): {rsi:.1f}
🔹 MACD: {macd:.4f} | Signal: {macd_signal:.4f}
🔹 Volume Spike: {volume_spike}

📝 RECOMMENDATION:
────────────────────────────────────────────────────────────────────────────────
⏳ WAIT - No significant pattern detected at this time.

Consolidation expected. Wait for clearer signal.

================================================================================
"""
    return training_text


# =========================================================
# DATA GENERATION FUNCTIONS
# =========================================================

def generate_elliott_wave_data(symbol, df_row, pattern_type, config, indicator_values, metrics, variation_idx=0):
    """Elliott Wave প্যাটার্নের জন্য ডাটা তৈরি (Variations সহ)"""
    current_price = df_row['close']
    current_date = df_row['date']
    
    rsi = indicator_values.get('rsi', 50)
    macd = indicator_values.get('macd', 0)
    macd_signal = indicator_values.get('macd_signal', 0)
    stoch_k = indicator_values.get('stoch_k', 50)
    stoch_d = indicator_values.get('stoch_d', 50)
    atr = indicator_values.get('atr', current_price * 0.02)
    volume = indicator_values.get('volume', 1000000)
    rsi_divergence = indicator_values.get('rsi_divergence', 'None')
    macd_divergence = indicator_values.get('macd_divergence', 'None')
    avg_vol = indicator_values.get('avg_volume', volume)
    ema_20 = indicator_values.get('ema_20', current_price)
    sma_20 = indicator_values.get('sma_20', current_price)
    
    atr_value = atr if atr > 0 else current_price * 0.02
    
    if config['bias'] == 'Bullish':
        entry = current_price
        stop = current_price - (atr_value * 1.5)
        target = entry + (abs(entry - stop) * random.uniform(1.5, 3.0))
    elif config['bias'] == 'Bearish':
        entry = current_price
        stop = current_price + (atr_value * 1.5)
        target = entry - (abs(entry - stop) * random.uniform(1.5, 3.0))
    else:
        entry = current_price
        stop = current_price - (atr_value * 2)
        target = entry + (abs(entry - stop) * random.uniform(1.0, 2.0))
    
    # ✅ FIX 5: Deterministic confidence with small noise
    confidence = 50
    confidence += (macd > macd_signal and config['bias'] == 'Bullish') * 10
    confidence += (macd < macd_signal and config['bias'] == 'Bearish') * 10
    confidence += (rsi < 40 and config['bias'] == 'Bullish') * 5
    confidence += (rsi > 60 and config['bias'] == 'Bearish') * 5
    confidence += (volume > avg_vol * 1.5) * 5
    confidence += (metrics.get('relative_strength', 0) > 0.6) * 5
    confidence += (rsi_divergence != 'None') * random.randint(3, 6)
    confidence += (macd_divergence != 'None') * random.randint(2, 5)
    
    confidence += random.uniform(-5, 5)
    confidence = min(95, max(30, confidence))
    
    rr_ratio = abs((target - entry) / max(abs(entry - stop), 1e-6))
    volume_spike = "Yes" if volume > avg_vol * 1.5 else "No"
    variation_note = f" [VARIATION {variation_idx + 1}]" if variation_idx > 0 else " [ORIGINAL SEQUENCE]"
    
    # ✅ FIX 7: Random pattern display to prevent shortcut learning
    if random.random() < 0.5:
        pattern_display = pattern_type
    else:
        pattern_display = "Unknown Pattern"
    
    if random.random() < 0.3:
        price_header = "PRICE SNAPSHOT:"
    else:
        price_header = "📊 PRICE DATA:"
    
    training_text = f"""
================================================================================
Elliott Wave Pattern: {pattern_display}{variation_note}
Symbol: {symbol}
Date: {current_date}
================================================================================

{price_header}
────────────────────────────────────────────────────────────────────────────────
Open: {df_row['open']:.2f} | High: {df_row['high']:.2f} | Low: {df_row['low']:.2f}
Close: {current_price:.2f} | Volume: {volume:,}

📈 TECHNICAL INDICATORS:
────────────────────────────────────────────────────────────────────────────────
🔹 RSI (14): {rsi:.1f} | Status: {'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'}
   Divergence: {rsi_divergence}

🔹 MACD: {macd:.4f} | Signal: {macd_signal:.4f}
   Status: {'Bullish' if macd > macd_signal else 'Bearish'}
   Divergence: {macd_divergence}

🔹 Stochastic: %K={stoch_k:.1f} | %D={stoch_d:.1f}
🔹 ATR: {atr_value:.2f} | ATR %: {(atr_value / current_price * 100):.2f}%
🔹 Volume Spike: {volume_spike}

📐 ELLIOTT WAVE ANALYSIS:
────────────────────────────────────────────────────────────────────────────────
Pattern Type: {pattern_type}
Category: {config['category']}
Wave Structure: {config['structure']}
Bias: {config['bias']}
Wave Degree: {config['degree']}
Fibonacci Ratios: {config['fib_ratios']}

🎯 WAVE SPECIFICATIONS:
────────────────────────────────────────────────────────────────────────────────
{config['specifications']}

💰 TRADING SETUP:
────────────────────────────────────────────────────────────────────────────────
Entry Price: {entry:.2f}
Stop Loss: {stop:.2f}
Target: {target:.2f}
Risk-Reward Ratio: {rr_ratio:.2f}
Signal Strength: {confidence:.1f}%

📝 RECOMMENDATION:
────────────────────────────────────────────────────────────────────────────────
{'✅ BUY - Wave ' + config['wave_position'] if config['bias'] == 'Bullish' else '❌ SELL - Wave ' + config['wave_position'] if config['bias'] == 'Bearish' else '⏳ WAIT - Wave ' + config['wave_position']}

Wave Count: {config['wave_count']}
Invalidation Level: {config['invalidation']}

Additional Confirmation:
{'- RSI divergence confirms wave completion' if rsi_divergence != 'None' else '- No divergence confirmation'}
{'- Volume supports impulse move' if volume_spike == 'Yes' else '- Volume needs confirmation'}
{'- MACD confirms trend direction' if (macd > macd_signal and config['bias'] == 'Bullish') or (macd < macd_signal and config['bias'] == 'Bearish') else '- MACD divergence warning'}

================================================================================
"""
    return training_text


def get_elliott_wave_patterns():
    """Elliott Wave সম্পূর্ণ প্যাটার্ন লাইব্রেরি"""
    return {
        'Impulse Wave': {
            'category': 'Motive Wave', 'structure': '5-3-5-3-5', 'bias': 'Bullish',
            'degree': 'Primary/Intermediate/Minor', 'fib_ratios': 'Wave 2: 0.382-0.618, Wave 3: 1.618-2.618',
            'specifications': 'Wave 1: Initial move, volume confirmation needed\nWave 2: Shallow/deep retracement\nWave 3: Strongest wave, extension potential\nWave 4: Simple/complex correction\nWave 5: Divergence check',
            'wave_position': '3 (Extension possible)', 'wave_count': '1-2-3-4-5',
            'expected_target': 'Wave 3 = 1.618 x Wave 1', 'invalidation': 'Below Wave 1 start'
        },
        'Leading Diagonal': {
            'category': 'Motive Wave', 'structure': '5-3-5-3-5', 'bias': 'Bullish/Bearish',
            'degree': 'Primary/Intermediate', 'fib_ratios': 'Wave 3: 1.0-1.618',
            'specifications': 'Occurs in Wave 1 or A position\nEach leg has 5-3-5-3-5 internal\nOverlapping waves\nNarrowing wedge shape',
            'wave_position': 'Wave 1 or A', 'wave_count': '1-2-3-4-5 (overlapping)',
            'expected_target': 'Breakout direction', 'invalidation': 'Structural violation'
        },
        'Ending Diagonal': {
            'category': 'Motive Wave', 'structure': '3-3-3-3-3', 'bias': 'Bullish/Bearish',
            'degree': 'Intermediate/Minor', 'fib_ratios': 'Wave 3: 1.0-1.382',
            'specifications': 'Occurs in Wave 5 or C position\nEach leg has 3-3-3-3-3 internal\nOverlapping waves\nVolume spike at termination',
            'wave_position': 'Wave 5 or C', 'wave_count': '1-2-3-4-5 (overlapping)',
            'expected_target': 'Terminal move', 'invalidation': 'Pattern expansion'
        },
        '3rd Wave Extension': {
            'category': 'Motive Wave', 'structure': '5-3-5-3-5 (extended)', 'bias': 'Bullish',
            'degree': 'Primary/Intermediate', 'fib_ratios': 'Wave 3 = 1.618-2.618 x Wave 1',
            'specifications': 'Most common extension\nStrongest momentum\nHighest volume\nWave 3 subdivides extensively',
            'wave_position': 'Wave 3 (Extended)', 'wave_count': '1-2-[3-3-3-3-3]-4-5',
            'expected_target': '1.618 x Wave 1', 'invalidation': 'Below Wave 1 high'
        },
        '5th Wave Extension': {
            'category': 'Motive Wave', 'structure': '5-3-5-3-5 (extended)', 'bias': 'Bullish/Bearish',
            'degree': 'Intermediate/Minor', 'fib_ratios': 'Wave 5 = 0.618-1.618 x Wave 1',
            'specifications': 'Terminal move\nDivergence with oscillators\nLower volume than Wave 3\nEnd of trend signal',
            'wave_position': 'Wave 5 (Terminal)', 'wave_count': '1-2-3-4-[5-5-5-5-5]',
            'expected_target': '0.618-1.618 x Wave 1', 'invalidation': 'Divergence confirmed'
        },
        'Single Zigzag (5-3-5)': {
            'category': 'Corrective Wave', 'structure': '5-3-5', 'bias': 'Neutral',
            'degree': 'Any', 'fib_ratios': 'Wave B = 0.382-0.786, Wave C = 0.618-1.618',
            'specifications': 'Sharp correction\nWave A has 5 sub-waves\nWave B has 3 sub-waves\nWave C has 5 sub-waves',
            'wave_position': 'ABC', 'wave_count': 'A-B-C (5-3-5)',
            'expected_target': 'Wave C = Wave A', 'invalidation': 'Complex structure'
        },
        'Double Zigzag (W-X-Y)': {
            'category': 'Corrective Wave', 'structure': '5-3-5-3-5', 'bias': 'Neutral',
            'degree': 'Intermediate/Minor', 'fib_ratios': 'Wave Y = 0.618-1.618 x Wave W',
            'specifications': 'Two zigzags connected by X wave\nDeeper/longer correction\nWave X is 3 waves',
            'wave_position': 'W-X-Y', 'wave_count': 'W-X-Y (5-3-5-3-5)',
            'expected_target': 'Wave Y = Wave W', 'invalidation': 'Triple zigzag'
        },
        'Regular Flat (3-3-5)': {
            'category': 'Corrective Wave', 'structure': '3-3-5', 'bias': 'Neutral',
            'degree': 'Any', 'fib_ratios': 'Wave B = 0.90-1.05, Wave C = 1.0 x Wave A',
            'specifications': 'Sideways correction\nWave A has 3 sub-waves\nWave B retraces 90-105% of A',
            'wave_position': 'A-B-C', 'wave_count': 'A-B-C (3-3-5)',
            'expected_target': 'Wave C = Wave A', 'invalidation': 'B > 1.05 x A'
        },
def get_elliott_wave_patterns():
    """Elliott Wave সম্পূর্ণ প্যাটার্ন লাইব্রেরি - ✅ FIX: Added missing keys"""
    return {
        'Impulse Wave': {
            'category': 'Motive Wave', 
            'structure': '5-3-5-3-5', 
            'bias': 'Bullish',
            'degree': 'Primary/Intermediate/Minor', 
            'fib_ratios': 'Wave 2: 0.382-0.618, Wave 3: 1.618-2.618',
            'specifications': 'Wave 1: Initial move, volume confirmation needed\nWave 2: Shallow/deep retracement\nWave 3: Strongest wave, extension potential\nWave 4: Simple/complex correction\nWave 5: Divergence check',
            'wave_position': '3 (Extension possible)', 
            'wave_count': '1-2-3-4-5',
            'expected_target': 'Wave 3 = 1.618 x Wave 1', 
            'invalidation': 'Below Wave 1 start'
        },
        'Leading Diagonal': {
            'category': 'Motive Wave', 
            'structure': '5-3-5-3-5', 
            'bias': 'Bullish/Bearish',
            'degree': 'Primary/Intermediate', 
            'fib_ratios': 'Wave 3: 1.0-1.618',
            'specifications': 'Occurs in Wave 1 or A position\nEach leg has 5-3-5-3-5 internal\nOverlapping waves\nNarrowing wedge shape',
            'wave_position': 'Wave 1 or A', 
            'wave_count': '1-2-3-4-5 (overlapping)',
            'expected_target': 'Breakout direction', 
            'invalidation': 'Structural violation'
        },
        'Ending Diagonal': {
            'category': 'Motive Wave', 
            'structure': '3-3-3-3-3', 
            'bias': 'Bullish/Bearish',
            'degree': 'Intermediate/Minor', 
            'fib_ratios': 'Wave 3: 1.0-1.382',
            'specifications': 'Occurs in Wave 5 or C position\nEach leg has 3-3-3-3-3 internal\nOverlapping waves\nVolume spike at termination',
            'wave_position': 'Wave 5 or C', 
            'wave_count': '1-2-3-4-5 (overlapping)',
            'expected_target': 'Terminal move', 
            'invalidation': 'Pattern expansion'
        },
        '3rd Wave Extension': {
            'category': 'Motive Wave', 
            'structure': '5-3-5-3-5 (extended)', 
            'bias': 'Bullish',
            'degree': 'Primary/Intermediate', 
            'fib_ratios': 'Wave 3 = 1.618-2.618 x Wave 1',
            'specifications': 'Most common extension\nStrongest momentum\nHighest volume\nWave 3 subdivides extensively',
            'wave_position': 'Wave 3 (Extended)', 
            'wave_count': '1-2-[3-3-3-3-3]-4-5',
            'expected_target': '1.618 x Wave 1', 
            'invalidation': 'Below Wave 1 high'
        },
        '5th Wave Extension': {
            'category': 'Motive Wave', 
            'structure': '5-3-5-3-5 (extended)', 
            'bias': 'Bullish/Bearish',
            'degree': 'Intermediate/Minor', 
            'fib_ratios': 'Wave 5 = 0.618-1.618 x Wave 1',
            'specifications': 'Terminal move\nDivergence with oscillators\nLower volume than Wave 3\nEnd of trend signal',
            'wave_position': 'Wave 5 (Terminal)', 
            'wave_count': '1-2-3-4-[5-5-5-5-5]',
            'expected_target': '0.618-1.618 x Wave 1', 
            'invalidation': 'Divergence confirmed'
        },
        'Single Zigzag (5-3-5)': {
            'category': 'Corrective Wave', 
            'structure': '5-3-5', 
            'bias': 'Neutral',
            'degree': 'Any', 
            'fib_ratios': 'Wave B = 0.382-0.786, Wave C = 0.618-1.618',
            'specifications': 'Sharp correction\nWave A has 5 sub-waves\nWave B has 3 sub-waves\nWave C has 5 sub-waves',
            'wave_position': 'ABC', 
            'wave_count': 'A-B-C (5-3-5)',
            'expected_target': 'Wave C = Wave A', 
            'invalidation': 'Complex structure'
        },
        'Double Zigzag (W-X-Y)': {
            'category': 'Corrective Wave', 
            'structure': '5-3-5-3-5', 
            'bias': 'Neutral',
            'degree': 'Intermediate/Minor', 
            'fib_ratios': 'Wave Y = 0.618-1.618 x Wave W',
            'specifications': 'Two zigzags connected by X wave\nDeeper/longer correction\nWave X is 3 waves',
            'wave_position': 'W-X-Y', 
            'wave_count': 'W-X-Y (5-3-5-3-5)',
            'expected_target': 'Wave Y = Wave W', 
            'invalidation': 'Triple zigzag'
        },
        'Regular Flat (3-3-5)': {
            'category': 'Corrective Wave', 
            'structure': '3-3-5', 
            'bias': 'Neutral',
            'degree': 'Any', 
            'fib_ratios': 'Wave B = 0.90-1.05, Wave C = 1.0 x Wave A',
            'specifications': 'Sideways correction\nWave A has 3 sub-waves\nWave B retraces 90-105% of A',
            'wave_position': 'A-B-C', 
            'wave_count': 'A-B-C (3-3-5)',
            'expected_target': 'Wave C = Wave A', 
            'invalidation': 'B > 1.05 x A'
        },
        'Expanded Flat (3-3-5)': {
            'category': 'Corrective Wave', 
            'structure': '3-3-5', 
            'bias': 'Neutral',
            'degree': 'Intermediate/Minor', 
            'fib_ratios': 'Wave B = 1.05-1.382 x Wave A',
            'specifications': 'Wave B exceeds Wave A start\nWave C exceeds Wave B\nStrong momentum in B',
            'wave_position': 'A-B-C', 
            'wave_count': 'A-B-C (3-3-5 expanded)',
            'expected_target': 'Wave C = 1.382 x Wave A', 
            'invalidation': 'Wave C extreme'
        },
        'Contracting Triangle (3-3-3-3-3)': {
            'category': 'Corrective Wave', 
            'structure': '3-3-3-3-3', 
            'bias': 'Neutral',
            'degree': 'Any', 
            'fib_ratios': 'Each wave smaller, Wave E = 0.618-0.786 x Wave C',
            'specifications': '5 waves: A-B-C-D-E\nEach wave has 3 sub-waves\nContracting range',
            'wave_position': 'Wave 4 or B', 
            'wave_count': 'A-B-C-D-E (3-3-3-3-3)',
            'expected_target': 'Breakout direction', 
            'invalidation': 'Triangle expands'
        },
        'Expanding Triangle (3-3-3-3-3)': {
            'category': 'Corrective Wave', 
            'structure': '3-3-3-3-3', 
            'bias': 'Neutral',
            'degree': 'Intermediate/Minor', 
            'fib_ratios': 'Each wave larger, Wave E = 1.236-1.382 x Wave C',
            'specifications': '5 waves expanding\nEach wave has 3 sub-waves\nIncreasing range',
            'wave_position': 'Wave 4 or B', 
            'wave_count': 'A-B-C-D-E (expanding)',
            'expected_target': 'Breakout direction', 
            'invalidation': 'Triangle contracts'
        },
        'Fibonacci Price Relationships': {
            'category': 'Wave Relationships', 
            'structure': 'Various', 
            'bias': 'Both',
            'degree': 'Any', 
            'fib_ratios': 'Wave 2 = 0.382-0.618, Wave 3 = 1.618-2.618, Wave 4 = 0.236-0.382',
            'specifications': 'Wave 3 cannot be shortest\nWave 4 cannot overlap Wave 1\nAlternation principle',
            'wave_position': 'All waves', 
            'wave_count': 'Complete cycle',
            'expected_target': 'Based on Fibonacci', 
            'invalidation': 'Relationships violated'
        }
    }


def get_all_patterns():
    """60+ প্যাটার্নের সম্পূর্ণ তালিকা (Elliott Wave সহ)"""
    patterns = {
        'Cup and Handle': {'category': 'Continuation', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Breakout above handle', 'stop': 'Below handle low', 'target': 'Measure cup depth'},
        'Ascending Triangle': {'category': 'Continuation', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Breakout above resistance', 'stop': 'Below higher low', 'target': 'Height of triangle'},
        'Bull Flag': {'category': 'Continuation', 'bias': 'Bullish', 'timeframe': 'Short-term', 'entry': 'Breakout of flag', 'stop': 'Below flag low', 'target': 'Flagpole length'},
        'Bull Pennant': {'category': 'Continuation', 'bias': 'Bullish', 'timeframe': 'Short-term', 'entry': 'Breakout', 'stop': 'Below pennant', 'target': 'Flagpole projection'},
        'Falling Wedge': {'category': 'Continuation', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Breakout', 'stop': 'Below low', 'target': 'Wedge height'},
        'Rectangle Bullish': {'category': 'Continuation', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Breakout above range', 'stop': 'Below support', 'target': 'Range height'},
        'Bear Flag': {'category': 'Continuation', 'bias': 'Bearish', 'timeframe': 'Short-term', 'entry': 'Breakdown', 'stop': 'Above flag high', 'target': 'Flagpole'},
        'Bear Pennant': {'category': 'Continuation', 'bias': 'Bearish', 'timeframe': 'Short-term', 'entry': 'Breakdown', 'stop': 'Above pennant', 'target': 'Flagpole'},
        'Descending Triangle': {'category': 'Continuation', 'bias': 'Bearish', 'timeframe': 'Swing', 'entry': 'Breakdown', 'stop': 'Above lower high', 'target': 'Triangle height'},
        'Rising Wedge': {'category': 'Continuation', 'bias': 'Bearish', 'timeframe': 'Swing', 'entry': 'Breakdown below support', 'stop': 'Above recent high', 'target': 'Height of wedge'},
        'Rectangle Bearish': {'category': 'Continuation', 'bias': 'Bearish', 'timeframe': 'Swing', 'entry': 'Breakdown', 'stop': 'Above resistance', 'target': 'Range height'},
        'Symmetrical Triangle': {'category': 'Neutral', 'bias': 'Both', 'timeframe': 'Swing', 'entry': 'Breakout either side', 'stop': 'Opposite side', 'target': 'Triangle height'},
        'Diamond Pattern': {'category': 'Neutral', 'bias': 'Both', 'timeframe': 'Swing', 'entry': 'Breakout', 'stop': 'Opposite side', 'target': 'Height'},
        'Double Bottom': {'category': 'Reversal', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Break above neckline', 'stop': 'Below bottom', 'target': 'Pattern height'},
        'Triple Bottom': {'category': 'Reversal', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Breakout neckline', 'stop': 'Below lows', 'target': 'Height'},
        'Inverse Head and Shoulders': {'category': 'Reversal', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Breakout neckline', 'stop': 'Below right shoulder', 'target': 'Height'},
        'Rounding Bottom': {'category': 'Reversal', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Breakout', 'stop': 'Below base', 'target': 'Depth'},
        'Double Top': {'category': 'Reversal', 'bias': 'Bearish', 'timeframe': 'Swing', 'entry': 'Breakdown neckline', 'stop': 'Above top', 'target': 'Height'},
        'Triple Top': {'category': 'Reversal', 'bias': 'Bearish', 'timeframe': 'Swing', 'entry': 'Breakdown', 'stop': 'Above highs', 'target': 'Height'},
        'Head and Shoulders': {'category': 'Reversal', 'bias': 'Bearish', 'timeframe': 'Swing', 'entry': 'Breakdown neckline', 'stop': 'Above right shoulder', 'target': 'Height'},
        'Rounding Top': {'category': 'Reversal', 'bias': 'Bearish', 'timeframe': 'Swing', 'entry': 'Breakdown', 'stop': 'Above top', 'target': 'Depth'},
        'Hammer': {'category': 'Candlestick', 'bias': 'Bullish', 'timeframe': 'Intraday', 'entry': 'Confirm next green candle', 'stop': 'Below wick', 'target': 'Recent resistance'},
        'Morning Star': {'category': 'Candlestick', 'bias': 'Bullish', 'timeframe': 'Intraday', 'entry': '3-candle confirmation', 'stop': 'Below low', 'target': 'Resistance'},
        'Bullish Engulfing': {'category': 'Candlestick', 'bias': 'Bullish', 'timeframe': 'Intraday', 'entry': 'Engulfing close', 'stop': 'Below candle', 'target': 'Resistance'},
        'Shooting Star': {'category': 'Candlestick', 'bias': 'Bearish', 'timeframe': 'Intraday', 'entry': 'Confirm red candle', 'stop': 'Above wick', 'target': 'Support'},
        'Evening Star': {'category': 'Candlestick', 'bias': 'Bearish', 'timeframe': 'Intraday', 'entry': '3-candle confirm', 'stop': 'Above high', 'target': 'Support'},
        'Bearish Engulfing': {'category': 'Candlestick', 'bias': 'Bearish', 'timeframe': 'Intraday', 'entry': 'Engulf close', 'stop': 'Above candle', 'target': 'Support'},
        'Doji': {'category': 'Candlestick', 'bias': 'Neutral', 'timeframe': 'Intraday', 'entry': 'Wait breakout', 'stop': 'High/low', 'target': 'Next move'},
        'Wolfe Wave': {'category': 'Advanced', 'bias': 'Both', 'timeframe': 'Swing', 'entry': 'Wave completion', 'stop': 'Beyond wave', 'target': 'Target line'},
        'Gartley': {'category': 'Harmonic', 'bias': 'Both', 'timeframe': 'Swing', 'entry': 'PRZ zone', 'stop': 'Beyond X', 'target': 'Fibonacci targets'},
        'Butterfly': {'category': 'Harmonic', 'bias': 'Both', 'timeframe': 'Swing', 'entry': 'PRZ', 'stop': 'Beyond X', 'target': 'Fib'},
        'Bat': {'category': 'Harmonic', 'bias': 'Both', 'timeframe': 'Swing', 'entry': 'PRZ', 'stop': 'Beyond X', 'target': 'Fib'},
        'Volume Climax': {'category': 'Volume', 'bias': 'Reversal', 'timeframe': 'Any', 'entry': 'Spike volume', 'stop': 'Recent extreme', 'target': 'Reversal zone'},
        'False Breakout': {'category': 'Breakout', 'bias': 'Trap', 'timeframe': 'Any', 'entry': 'Re-entry opposite', 'stop': 'Recent high/low', 'target': 'Range'},
        'Breakout Pullback': {'category': 'Breakout', 'bias': 'Continuation', 'timeframe': 'Swing', 'entry': 'Retest entry', 'stop': 'Below pullback', 'target': 'Trend'},
        'Order Block': {'category': 'Smart Money', 'bias': 'Both', 'timeframe': 'Any', 'entry': 'Return to OB', 'stop': 'Beyond OB', 'target': 'Structure'},
        'Fair Value Gap': {'category': 'Smart Money', 'bias': 'Both', 'timeframe': 'Any', 'entry': 'Fill entry', 'stop': 'Beyond gap', 'target': 'Structure'},
        'Break of Structure': {'category': 'Smart Money', 'bias': 'Both', 'timeframe': 'Any', 'entry': 'After BOS', 'stop': 'Recent swing', 'target': 'Trend'},
        'Pin Bar': {'category': 'Price Action', 'bias': 'Reversal', 'timeframe': 'Any', 'entry': 'Wick rejection', 'stop': 'Beyond wick', 'target': 'Structure'},
        '1-2-3 Pattern': {'category': 'Price Action', 'bias': 'Reversal', 'timeframe': 'Any', 'entry': 'Point 2 break', 'stop': 'Below 3', 'target': 'Projection'},
        'Bollinger Band Squeeze': {'category': 'Volatility', 'bias': 'Breakout', 'timeframe': 'Any', 'entry': 'Expansion', 'stop': 'Opp band', 'target': 'Move'},
        'Inside Bar': {'category': 'Volatility', 'bias': 'Breakout', 'timeframe': 'Intraday', 'entry': 'Break mother bar', 'stop': 'Opp side', 'target': 'Range'},
        'Outside Bar': {'category': 'Volatility', 'bias': 'Both', 'timeframe': 'Intraday', 'entry': 'Break high/low', 'stop': 'Opp side', 'target': 'Range'},
    }
    
    elliott_patterns = get_elliott_wave_patterns()
    patterns.update(elliott_patterns)
    return patterns


def generate_complete_pattern_data(symbol, df_row, pattern_type, config, indicator_values, metrics, variation_idx=0):
    """সব ইন্ডিকেটর এবং মেট্রিক্স সহ সম্পূর্ণ প্যাটার্ন ডাটা তৈরি"""
    current_price = df_row['close']
    current_date = df_row['date']
    
    rsi = indicator_values.get('rsi', 50)
    macd = indicator_values.get('macd', 0)
    macd_signal = indicator_values.get('macd_signal', 0)
    macd_hist = indicator_values.get('macd_hist', 0)
    stoch_k = indicator_values.get('stoch_k', 50)
    stoch_d = indicator_values.get('stoch_d', 50)
    obv = indicator_values.get('obv', 0)
    atr = indicator_values.get('atr', current_price * 0.02)
    bb_upper = indicator_values.get('bb_upper', current_price * 1.05)
    bb_middle = indicator_values.get('bb_middle', current_price)
    bb_lower = indicator_values.get('bb_lower', current_price * 0.95)
    ema_20 = indicator_values.get('ema_20', current_price)
    sma_20 = indicator_values.get('sma_20', current_price)
    volume = indicator_values.get('volume', 1000000)
    rsi_divergence = indicator_values.get('rsi_divergence', 'None')
    macd_divergence = indicator_values.get('macd_divergence', 'None')
    avg_vol = indicator_values.get('avg_volume', volume)
    pattern_metrics = metrics
    
    atr_value = atr if atr > 0 else current_price * 0.02
    
    if config['bias'] == 'Bullish':
        entry = current_price
        stop = current_price - (atr_value * 1.5)
        target = entry + (abs(entry - stop) * random.uniform(1.5, 3.0))
    elif config['bias'] == 'Bearish':
        entry = current_price
        stop = current_price + (atr_value * 1.5)
        target = entry - (abs(entry - stop) * random.uniform(1.5, 3.0))
    else:
        entry = current_price
        stop = current_price - (atr_value * 2)
        target = entry + (abs(entry - stop) * random.uniform(1.0, 2.0))
    
    # ✅ FIX 5: Deterministic confidence with small noise
    confidence = 50
    confidence += (macd > macd_signal and config['bias'] == 'Bullish') * 10
    confidence += (macd < macd_signal and config['bias'] == 'Bearish') * 10
    confidence += (rsi < 40 and config['bias'] == 'Bullish') * 5
    confidence += (rsi > 60 and config['bias'] == 'Bearish') * 5
    confidence += (volume > avg_vol * 1.5) * 5
    confidence += (pattern_metrics.get('relative_strength', 0) > 0.6) * 5
    confidence += (rsi_divergence != 'None') * random.randint(3, 6)
    confidence += (macd_divergence != 'None') * random.randint(2, 5)
    confidence += (stoch_k < 30 and config['bias'] == 'Bullish') * 5
    confidence += (stoch_k > 70 and config['bias'] == 'Bearish') * 5
    
    confidence += random.uniform(-5, 5)
    confidence = min(95, max(30, confidence))
    
    rr_ratio = abs((target - entry) / max(abs(entry - stop), 1e-6))
    volume_spike = "Yes" if volume > avg_vol * 1.5 else "No"
    bb_position = "Above upper band" if current_price > bb_upper else "Below lower band" if current_price < bb_lower else "Within bands"
    variation_note = f" [VARIATION {variation_idx + 1}]" if variation_idx > 0 else " [ORIGINAL SEQUENCE]"
    
    # ✅ FIX 7: Random pattern display
    if random.random() < 0.5:
        pattern_display = pattern_type
    else:
        pattern_display = "Unknown Pattern"
    
    if random.random() < 0.3:
        price_header = "PRICE SNAPSHOT:"
    else:
        price_header = "📊 PRICE DATA:"
    
    training_text = f"""
================================================================================
Pattern: {pattern_display}{variation_note}
Symbol: {symbol}
Date: {current_date}
================================================================================

{price_header}
────────────────────────────────────────────────────────────────────────────────
Open: {df_row['open']:.2f} | High: {df_row['high']:.2f} | Low: {df_row['low']:.2f}
Close: {current_price:.2f} | Volume: {volume:,}
Change: {df_row.get('change', 0):.2f}% | Trades: {df_row.get('trades', 0):,}

📈 TECHNICAL INDICATORS:
────────────────────────────────────────────────────────────────────────────────
🔹 RSI (14): {rsi:.1f} | Status: {'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'}
   Divergence: {rsi_divergence}

🔹 MACD: {macd:.4f} | Signal: {macd_signal:.4f} | Hist: {macd_hist:.4f}
   Status: {'Bullish' if macd > macd_signal else 'Bearish'}
   Divergence: {macd_divergence}

🔹 Stochastic: %K={stoch_k:.1f} | %D={stoch_d:.1f}
🔹 Bollinger Bands: Upper={bb_upper:.2f} | Middle={bb_middle:.2f} | Lower={bb_lower:.2f}
🔹 Moving Averages: EMA20={ema_20:.2f} | SMA20={sma_20:.2f}
🔹 ATR: {atr_value:.2f} | ATR%: {(atr_value / current_price * 100):.2f}%
🔹 Volume Spike: {volume_spike} | OBV: {obv:,.0f}

📐 PATTERN VALIDATION METRICS:
────────────────────────────────────────────────────────────────────────────────
Pattern Height: {pattern_metrics.get('pattern_height', 'N/A')}
Pattern Depth %: {pattern_metrics.get('pattern_depth_percent', 'N/A')}%
Breakout Distance: {pattern_metrics.get('breakout_distance_percent', 'N/A')}%
Relative Strength: {pattern_metrics.get('relative_strength', 'N/A')}

🎯 PATTERN ANALYSIS:
────────────────────────────────────────────────────────────────────────────────
Pattern Name: {pattern_type}
Category: {config['category']}
Bias: {config['bias']}
Timeframe: {config['timeframe']}
Entry Rule: {config['entry']}
Stop Loss Rule: {config['stop']}
Target Rule: {config['target']}

💰 TRADING SETUP:
────────────────────────────────────────────────────────────────────────────────
Entry Price: {entry:.2f}
Stop Loss: {stop:.2f}
Target: {target:.2f}
Risk-Reward Ratio: {rr_ratio:.2f}
Signal Strength: {confidence:.1f}%

📝 RECOMMENDATION:
────────────────────────────────────────────────────────────────────────────────
{'✅ BUY' if config['bias'] == 'Bullish' else '❌ SELL' if config['bias'] == 'Bearish' else '⏳ WAIT FOR BREAKOUT'} at {entry:.2f}

Additional Confirmation:
{'- RSI divergence confirms reversal' if rsi_divergence != 'None' else '- No RSI divergence'}
{'- MACD divergence confirms momentum' if macd_divergence != 'None' else '- No MACD divergence'}
{'- Volume spike supports breakout' if volume_spike == 'Yes' else '- Normal volume, wait for confirmation'}
{'- Price above EMA 20 (bullish trend)' if current_price > ema_20 else '- Price below EMA 20 (bearish trend)'}

================================================================================
"""
    return training_text


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():
    print("="*80)
    print("🚀 COMPLETE PATTERN TRAINING DATA GENERATOR")
    print("   (60+ Patterns + Elliott Wave + Multiple Historical Sequences + Noise Variations)")
    print("="*80)
    
    csv_path = "./csv/mongodb.csv"
    if not os.path.exists(csv_path):
        print(f"❌ {csv_path} not found!")
        return
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"✅ Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    
    all_patterns = get_all_patterns()
    print(f"✅ Loaded {len(all_patterns)} pattern configurations")
    
    NUM_VARIATIONS = 3
    training_data = []
    
    symbols_processed = 0
    for symbol in df['symbol'].unique():
        # Reset last_detected_idx per symbol
        last_detected_idx = {}
        
        symbol_data = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        
        if len(symbol_data) < 100:
            continue
        
        close_prices = symbol_data['close']
        high_prices = symbol_data['high']
        low_prices = symbol_data['low']
        volumes = symbol_data['volume']
        
        # ✅ FIX 2: Per-symbol VWAP (no cross-symbol leakage)
        vwap = (close_prices * volumes).cumsum() / volumes.cumsum()
        
        # Indicator calculations
        rsi_series = calculate_rsi(close_prices)
        macd_line, macd_signal, macd_hist = calculate_macd(close_prices)
        stoch_k, stoch_d = calculate_stochastic(high_prices, low_prices, close_prices)
        obv_series = calculate_obv(close_prices, volumes)
        ema_20 = calculate_ema(close_prices, 20)
        sma_20 = calculate_sma(close_prices, 20)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_prices)
        atr_series = calculate_atr(high_prices, low_prices, close_prices)
        avg_volume = volumes.rolling(20).mean()
        
        symbol_data['rsi'] = rsi_series
        symbol_data['bb_upper'] = bb_upper
        symbol_data['bb_middle'] = bb_middle
        symbol_data['bb_lower'] = bb_lower
        
        rsi_divergence_list = []
        macd_divergence_list = []
        
        for i in range(len(close_prices)):
            if i < 30:
                rsi_divergence_list.append('None')
                macd_divergence_list.append('None')
            else:
                prices_slice = close_prices.iloc[max(0, i-30):i+1].values
                rsi_slice = rsi_series.iloc[max(0, i-30):i+1].values
                macd_slice = macd_line.iloc[max(0, i-30):i+1].values
                
                r_div = detect_rsi_divergence(prices_slice, rsi_slice)
                m_div = detect_macd_divergence(prices_slice, macd_slice)
                rsi_divergence_list.append(r_div)
                macd_divergence_list.append(m_div)
        
        market_regime = detect_market_regime(close_prices)
        print(f"📊 Market regime for {symbol}: {market_regime}")
        
        row_count = 0
        MAX_PER_SYMBOL = 50
        
        # ✅ FIX 8: Better step logic
        step = 1 if len(symbol_data) < 500 else 2
        
        for idx in range(50, len(symbol_data), step):
            detected_patterns = detect_all_patterns(symbol_data, idx)
            
            # ✅ FIX 6: Add no-pattern examples
            if not detected_patterns and random.random() < 0.3:
                row = symbol_data.iloc[idx]
                price_momentum = (row['close'] - close_prices.iloc[idx-10]) / close_prices.iloc[idx-10] if idx >= 10 else 0
                indicator_values = {
                    'rsi': rsi_series.iloc[idx] if idx < len(rsi_series) else 50,
                    'macd': macd_line.iloc[idx] if idx < len(macd_line) else 0,
                    'macd_signal': macd_signal.iloc[idx] if idx < len(macd_signal) else 0,
                    'macd_hist': macd_hist.iloc[idx] if idx < len(macd_hist) else 0,
                    'stoch_k': stoch_k.iloc[idx] if idx < len(stoch_k) else 50,
                    'stoch_d': stoch_d.iloc[idx] if idx < len(stoch_d) else 50,
                    'obv': obv_series.iloc[idx] if idx < len(obv_series) else 0,
                    'atr': atr_series.iloc[idx] if idx < len(atr_series) else row['close'] * 0.02,
                    'bb_upper': bb_upper.iloc[idx] if idx < len(bb_upper) else row['close'] * 1.05,
                    'bb_middle': bb_middle.iloc[idx] if idx < len(bb_middle) else row['close'],
                    'bb_lower': bb_lower.iloc[idx] if idx < len(bb_lower) else row['close'] * 0.95,
                    'ema_20': ema_20.iloc[idx] if idx < len(ema_20) else row['close'],
                    'sma_20': sma_20.iloc[idx] if idx < len(sma_20) else row['close'],
                    'volume': volumes.iloc[idx],
                    'avg_volume': avg_volume.iloc[idx] if idx < len(avg_volume) else volumes.iloc[idx],
                    'rsi_divergence': rsi_divergence_list[idx] if idx < len(rsi_divergence_list) else 'None',
                    'macd_divergence': macd_divergence_list[idx] if idx < len(macd_divergence_list) else 'None',
                    'vwap': vwap.iloc[idx] if idx < len(vwap) else row['close'],
                    'trend_strength': abs(ema_20.iloc[idx] - sma_20.iloc[idx]) / sma_20.iloc[idx] if sma_20.iloc[idx] > 0 else 0,
                    'price_momentum': price_momentum
                }
                text = generate_no_pattern_example(symbol, row, indicator_values)
                if text:
                    training_data.append(text)
                    print(f"✅ Generated NO PATTERN example for {symbol} on {row['date'].date()}")
                    row_count += 1
                continue
            
            if not detected_patterns:
                continue
            
            # Filter patterns
            filtered_patterns = []
            for pattern_name in detected_patterns:
                config = all_patterns.get(pattern_name, {})
                pattern_bias = config.get('bias', 'Neutral')
                pattern_category = config.get('category', '')
                
                # Skip patterns against market regime (but keep reversals)
                if market_regime != 'UNKNOWN':
                    if market_regime == 'BULL' and pattern_bias == 'Bearish' and pattern_category != 'Reversal':
                        continue
                    if market_regime == 'BEAR' and pattern_bias == 'Bullish' and pattern_category != 'Reversal':
                        continue
                
                # ✅ FIX 9: Reduced cooldown to avoid missing clusters
                cooldown = 10
                if pattern_name in last_detected_idx:
                    if idx - last_detected_idx[pattern_name] < cooldown:
                        continue
                
                filtered_patterns.append(pattern_name)
                last_detected_idx[pattern_name] = idx
            
            if not filtered_patterns:
                continue
            
            row = symbol_data.iloc[idx]
            price_momentum = (row['close'] - close_prices.iloc[idx-10]) / close_prices.iloc[idx-10] if idx >= 10 else 0
            
            indicator_values = {
                'rsi': rsi_series.iloc[idx] if idx < len(rsi_series) else 50,
                'macd': macd_line.iloc[idx] if idx < len(macd_line) else 0,
                'macd_signal': macd_signal.iloc[idx] if idx < len(macd_signal) else 0,
                'macd_hist': macd_hist.iloc[idx] if idx < len(macd_hist) else 0,
                'stoch_k': stoch_k.iloc[idx] if idx < len(stoch_k) else 50,
                'stoch_d': stoch_d.iloc[idx] if idx < len(stoch_d) else 50,
                'obv': obv_series.iloc[idx] if idx < len(obv_series) else 0,
                'atr': atr_series.iloc[idx] if idx < len(atr_series) else row['close'] * 0.02,
                'bb_upper': bb_upper.iloc[idx] if idx < len(bb_upper) else row['close'] * 1.05,
                'bb_middle': bb_middle.iloc[idx] if idx < len(bb_middle) else row['close'],
                'bb_lower': bb_lower.iloc[idx] if idx < len(bb_lower) else row['close'] * 0.95,
                'ema_20': ema_20.iloc[idx] if idx < len(ema_20) else row['close'],
                'sma_20': sma_20.iloc[idx] if idx < len(sma_20) else row['close'],
                'volume': volumes.iloc[idx],
                'avg_volume': avg_volume.iloc[idx] if idx < len(avg_volume) else volumes.iloc[idx],
                'rsi_divergence': rsi_divergence_list[idx] if idx < len(rsi_divergence_list) else 'None',
                'macd_divergence': macd_divergence_list[idx] if idx < len(macd_divergence_list) else 'None',
                'vwap': vwap.iloc[idx] if idx < len(vwap) else row['close'],
                'trend_strength': abs(ema_20.iloc[idx] - sma_20.iloc[idx]) / sma_20.iloc[idx] if sma_20.iloc[idx] > 0 else 0,
                'price_momentum': price_momentum
            }
            
            # VWAP confidence boost
            if row['close'] > indicator_values['vwap']:
                indicator_values['vwap_boost'] = 3
            else:
                indicator_values['vwap_boost'] = 0
            
            pattern_high = row['high']
            pattern_low = row['low']
            real_sequence = close_prices.iloc[max(0, idx-30):idx+1].values
            
            for pattern_name in filtered_patterns:
                if pattern_name not in all_patterns:
                    continue
                
                config = all_patterns[pattern_name]
                
                for var_idx in range(NUM_VARIATIONS):
                    if var_idx > 0:
                        noise_level = random.uniform(0.003, 0.01)
                        noisy_sequence = add_noise_to_sequence(real_sequence, noise_level)
                        sequence_used = noisy_sequence
                        modified_row = row.copy()
                        modified_row['close'] = sequence_used[-1]
                        modified_row['open'] = modified_row['close'] * random.uniform(0.98, 1.02)
                        
                        temp_high = max(modified_row['close'], modified_row['open']) * random.uniform(1.0, 1.02)
                        temp_low = min(modified_row['close'], modified_row['open']) * random.uniform(0.98, 1.0)
                        modified_row['high'] = max(temp_high, modified_row['open'], modified_row['close'])
                        modified_row['low'] = min(temp_low, modified_row['open'], modified_row['close'])
                        use_row = modified_row
                        
                        # Recalculate indicators
                        temp_close_series = pd.Series(sequence_used)
                        temp_rsi = calculate_rsi(temp_close_series)
                        temp_macd_line, temp_macd_signal, _ = calculate_macd(temp_close_series)
                        
                        temp_high_series = pd.Series(sequence_used) * random.uniform(1.0, 1.02)
                        temp_low_series = pd.Series(sequence_used) * random.uniform(0.98, 1.0)
                        temp_stoch_k, temp_stoch_d = calculate_stochastic(
                            temp_high_series,
                            temp_low_series,
                            temp_close_series
                        )
                        
                        indicator_values['rsi'] = temp_rsi.iloc[-1]
                        indicator_values['macd'] = temp_macd_line.iloc[-1]
                        indicator_values['macd_signal'] = temp_macd_signal.iloc[-1]
                        indicator_values['stoch_k'] = temp_stoch_k.iloc[-1]
                        indicator_values['stoch_d'] = temp_stoch_d.iloc[-1]
                    else:
                        sequence_used = real_sequence
                        use_row = row
                    
                    metrics = calculate_pattern_metrics(
                        sequence_used, pattern_high, pattern_low, use_row['close']
                    )
                    
                    elliott_keywords = ['Wave', 'Diagonal', 'Zigzag', 'Flat', 'Triangle', 'Extension']
                    is_elliott = any(k in pattern_name for k in elliott_keywords)
                    
                    if is_elliott:
                        text = generate_elliott_wave_data(symbol, use_row, pattern_name, config, indicator_values, metrics, var_idx)
                    else:
                        text = generate_complete_pattern_data(symbol, use_row, pattern_name, config, indicator_values, metrics, var_idx)
                    
                    if text:
                        training_data.append(text)
                        print(f"✅ DETECTED & Generated {pattern_name} (Var {var_idx + 1}/{NUM_VARIATIONS}) for {symbol} on {row['date'].date()}")
                        row_count += 1
                        
                        if row_count >= MAX_PER_SYMBOL:
                            break
                
                if row_count >= MAX_PER_SYMBOL:
                    break
            
            if row_count >= MAX_PER_SYMBOL:
                break
        
        symbols_processed += 1
        if symbols_processed >= 50:
            break
    
    output_dir = "./csv"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "training_texts.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(training_data))
    
    print(f"\n📊 Training data saved: {output_file}")
    print(f"   Total examples: {len(training_data)}")
    print(f"   Variations per pattern: {NUM_VARIATIONS}")
    if os.path.exists(output_file):
        print(f"   File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    
    print("\n" + "="*80)
    print("📤 NEXT STEPS:")
    print("="*80)
    print("1. Upload training_texts.txt to Hugging Face dataset:")
    print("   https://huggingface.co/datasets/ahashanahmed/LLM_model_stock")
    print("\n2. Then retrain your LLM:")
    print("   python scripts/llm_train.py")
    print("="*80)


if __name__ == "__main__":
    main()
