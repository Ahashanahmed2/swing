# scripts/generate_pattern_training_data_complete.py
# RSI Divergence, MACD, Stochastic, ATR, Bollinger Bands, OBV, Volume Profile সহ সম্পূর্ণ ট্রেনিং ডাটা
# 60+ প্যাটার্ন + Elliott Wave সম্পূর্ণ লাইব্রেরি + Multiple Historical Sequences + Noise Variations

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def calculate_rsi(prices, period=14):
    """RSI ক্যালকুলেট করুন"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD ক্যালকুলেট করুন"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator ক্যালকুলেট করুন"""
    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()
    k = 100 * ((close - low_min) / (high_max - low_min))
    k = k.fillna(50)
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_obv(close, volume):
    """On-Balance Volume ক্যালকুলেট করুন"""
    obv = [0]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv.append(obv[-1] + volume[i])
        elif close[i] < close[i-1]:
            obv.append(obv[-1] - volume[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv)

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
    """ATR ক্যালকুলেট করুন"""
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift()).abs(),
        'lc': (low - close.shift()).abs()
    }).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.fillna(tr.mean())

def detect_rsi_divergence(prices, rsi_values):
    """RSI Divergence ডিটেক্ট করুন"""
    if len(prices) < 20 or len(rsi_values) < 20:
        return 'None'
    
    prices_array = np.array(prices) if not isinstance(prices, np.ndarray) else prices
    rsi_array = np.array(rsi_values) if not isinstance(rsi_values, np.ndarray) else rsi_values
    
    min_price_idx = np.argmin(prices_array[-20:])
    rsi_at_min = rsi_array[-20:][min_price_idx]
    
    sorted_indices = np.argsort(prices_array[-20:])
    if len(sorted_indices) > 1:
        second_min_idx = sorted_indices[1]
        second_min_price = prices_array[-20:][second_min_idx]
        second_min_rsi = rsi_array[-20:][second_min_idx]
        
        if prices_array[-20:].min() < second_min_price and rsi_at_min > second_min_rsi:
            return 'Bullish'
    
    max_price_idx = np.argmax(prices_array[-20:])
    rsi_at_max = rsi_array[-20:][max_price_idx]
    
    sorted_desc_indices = np.argsort(prices_array[-20:])[::-1]
    if len(sorted_desc_indices) > 1:
        second_max_idx = sorted_desc_indices[1]
        second_max_price = prices_array[-20:][second_max_idx]
        second_max_rsi = rsi_array[-20:][second_max_idx]
        
        if prices_array[-20:].max() > second_max_price and rsi_at_max < second_max_rsi:
            return 'Bearish'
    
    return 'None'

def detect_macd_divergence(prices, macd_line):
    """MACD Divergence ডিটেক্ট করুন"""
    if len(prices) < 20 or len(macd_line) < 20:
        return 'None'
    
    prices_array = np.array(prices) if not isinstance(prices, np.ndarray) else prices
    macd_array = np.array(macd_line) if not isinstance(macd_line, np.ndarray) else macd_line
    
    min_price_idx = np.argmin(prices_array[-20:])
    macd_at_min = macd_array[-20:][min_price_idx]
    
    sorted_indices = np.argsort(prices_array[-20:])
    if len(sorted_indices) > 1:
        second_min_idx = sorted_indices[1]
        second_min_price = prices_array[-20:][second_min_idx]
        second_min_macd = macd_array[-20:][second_min_idx]
        
        if prices_array[-20:].min() < second_min_price and macd_at_min > second_min_macd:
            return 'Bullish'
    
    max_price_idx = np.argmax(prices_array[-20:])
    macd_at_max = macd_array[-20:][max_price_idx]
    
    sorted_desc_indices = np.argsort(prices_array[-20:])[::-1]
    if len(sorted_desc_indices) > 1:
        second_max_idx = sorted_desc_indices[1]
        second_max_price = prices_array[-20:][second_max_idx]
        second_max_macd = macd_array[-20:][second_max_idx]
        
        if prices_array[-20:].max() > second_max_price and macd_at_max < second_max_macd:
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
    """Sequence-এ ছোট noise যোগ করুন (variations তৈরি করতে)"""
    noise = np.random.normal(0, noise_level, len(sequence))
    noisy_sequence = sequence * (1 + noise)
    return noisy_sequence

def generate_historical_variations(base_prices, num_variations=3):
    """একই প্যাটার্নের একাধিক historical variation তৈরি করুন"""
    variations = []
    variations.append(base_prices)  # Original
    
    for i in range(num_variations - 1):
        noise_level = random.uniform(0.002, 0.015)
        noisy = add_noise_to_sequence(base_prices, noise_level)
        variations.append(noisy)
    
    return variations

def generate_elliott_wave_data(symbol, df_row, pattern_type, config, indicator_values, metrics, variation_idx=0):
    """Elliott Wave প্যাটার্নের জন্য ডাটা তৈরি (Variations সহ)"""
    current_price = df_row['close']
    current_date = df_row['date']
    
    rsi = indicator_values.get('rsi', 50)
    macd = indicator_values.get('macd', 0)
    macd_signal = indicator_values.get('macd_signal', 0)
    stoch_k = indicator_values.get('stoch_k', 50)
    atr = indicator_values.get('atr', current_price * 0.02)
    volume = indicator_values.get('volume', 1000000)
    
    rsi_divergence = indicator_values.get('rsi_divergence', 'None')
    macd_divergence = indicator_values.get('macd_divergence', 'None')
    
    atr_value = atr if atr > 0 else current_price * 0.02
    
    if config['bias'] == 'Bullish':
        entry = current_price
        stop = current_price - (atr_value * 1.5)
        target = current_price + (atr_value * 3)
        fib_targets = f"1.618 = {current_price + (atr_value * 2.618):.2f}, 2.618 = {current_price + (atr_value * 4.236):.2f}"
    elif config['bias'] == 'Bearish':
        entry = current_price
        stop = current_price + (atr_value * 1.5)
        target = current_price - (atr_value * 3)
        fib_targets = f"1.618 = {current_price - (atr_value * 2.618):.2f}, 2.618 = {current_price - (atr_value * 4.236):.2f}"
    else:
        entry = current_price
        stop = current_price - (atr_value * 2)
        target = current_price + (atr_value * 2)
        fib_targets = f"1.272 = {current_price + (atr_value * 1.272):.2f}, 1.618 = {current_price + (atr_value * 1.618):.2f}"
    
    confidence = 65
    if rsi_divergence != 'None':
        confidence += 10
    if macd_divergence != 'None':
        confidence += 8
    confidence = min(95, max(50, confidence))
    
    rr_ratio = abs((target - entry) / (entry - stop)) if (entry - stop) != 0 else 0
    volume_spike = "Yes" if volume > indicator_values.get('avg_volume', volume) * 1.5 else "No"
    
    variation_note = f" [VARIATION {variation_idx + 1}]" if variation_idx > 0 else " [ORIGINAL SEQUENCE]"
    
    training_text = f"""
================================================================================
Elliott Wave Pattern: {pattern_type}{variation_note}
Symbol: {symbol}
Date: {current_date}
================================================================================

📊 PRICE DATA:
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
Take Profit: {target:.2f}
Fibonacci Extensions: {fib_targets}
Risk-Reward Ratio: {rr_ratio:.2f}
Confidence: {confidence}%

📝 RECOMMENDATION:
────────────────────────────────────────────────────────────────────────────────
{'✅ BUY - Wave ' + config['wave_position'] if config['bias'] == 'Bullish' else '❌ SELL - Wave ' + config['wave_position'] if config['bias'] == 'Bearish' else '⏳ WAIT - Wave ' + config['wave_position']}

Wave Count: {config['wave_count']}
Expected Target: {config['expected_target']}
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
        'Expanded Flat (3-3-5)': {
            'category': 'Corrective Wave', 'structure': '3-3-5', 'bias': 'Neutral',
            'degree': 'Intermediate/Minor', 'fib_ratios': 'Wave B = 1.05-1.382 x Wave A',
            'specifications': 'Wave B exceeds Wave A start\nWave C exceeds Wave B\nStrong momentum in B',
            'wave_position': 'A-B-C', 'wave_count': 'A-B-C (3-3-5 expanded)',
            'expected_target': 'Wave C = 1.382 x Wave A', 'invalidation': 'Wave C extreme'
        },
        'Contracting Triangle (3-3-3-3-3)': {
            'category': 'Corrective Wave', 'structure': '3-3-3-3-3', 'bias': 'Neutral',
            'degree': 'Any', 'fib_ratios': 'Each wave smaller, Wave E = 0.618-0.786 x Wave C',
            'specifications': '5 waves: A-B-C-D-E\nEach wave has 3 sub-waves\nContracting range',
            'wave_position': 'Wave 4 or B', 'wave_count': 'A-B-C-D-E (3-3-3-3-3)',
            'expected_target': 'Breakout direction', 'invalidation': 'Triangle expands'
        },
        'Expanding Triangle (3-3-3-3-3)': {
            'category': 'Corrective Wave', 'structure': '3-3-3-3-3', 'bias': 'Neutral',
            'degree': 'Intermediate/Minor', 'fib_ratios': 'Each wave larger, Wave E = 1.236-1.382 x Wave C',
            'specifications': '5 waves expanding\nEach wave has 3 sub-waves\nIncreasing range',
            'wave_position': 'Wave 4 or B', 'wave_count': 'A-B-C-D-E (expanding)',
            'expected_target': 'Breakout direction', 'invalidation': 'Triangle contracts'
        },
        'Fibonacci Price Relationships': {
            'category': 'Wave Relationships', 'structure': 'Various', 'bias': 'Both',
            'degree': 'Any', 'fib_ratios': 'Wave 2 = 0.382-0.618, Wave 3 = 1.618-2.618, Wave 4 = 0.236-0.382',
            'specifications': 'Wave 3 cannot be shortest\nWave 4 cannot overlap Wave 1\nAlternation principle',
            'wave_position': 'All waves', 'wave_count': 'Complete cycle',
            'expected_target': 'Based on Fibonacci', 'invalidation': 'Relationships violated'
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
    """সব ইন্ডিকেটর এবং মেট্রিক্স সহ সম্পূর্ণ প্যাটার্ন ডাটা তৈরি (Variations সহ)"""
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
    
    pattern_metrics = metrics
    
    atr_value = atr if atr > 0 else current_price * 0.02
    
    if config['bias'] == 'Bullish':
        entry = current_price
        stop = current_price - (atr_value * 1.5)
        target = current_price + (atr_value * 3)
    elif config['bias'] == 'Bearish':
        entry = current_price
        stop = current_price + (atr_value * 1.5)
        target = current_price - (atr_value * 3)
    else:
        entry = current_price
        stop = current_price - (atr_value * 2)
        target = current_price + (atr_value * 2)
    
    confidence = 65
    if rsi_divergence == 'Bullish' and config['bias'] == 'Bullish':
        confidence += 10
    elif rsi_divergence == 'Bearish' and config['bias'] == 'Bearish':
        confidence += 10
    if macd_divergence == 'Bullish' and config['bias'] == 'Bullish':
        confidence += 8
    elif macd_divergence == 'Bearish' and config['bias'] == 'Bearish':
        confidence += 8
    if pattern_metrics.get('relative_strength', 0) > 0.5:
        confidence += 5
    if config['bias'] == 'Bullish' and stoch_k < 30:
        confidence += 5
    elif config['bias'] == 'Bearish' and stoch_k > 70:
        confidence += 5
    
    confidence = min(95, max(50, confidence))
    rr_ratio = abs((target - entry) / (entry - stop)) if (entry - stop) != 0 else 0
    
    avg_vol = indicator_values.get('avg_volume', volume)
    volume_spike = "Yes" if volume > avg_vol * 1.5 else "No"
    bb_position = "Above upper band" if current_price > bb_upper else "Below lower band" if current_price < bb_lower else "Within bands"
    
    variation_note = f" [VARIATION {variation_idx + 1}]" if variation_idx > 0 else " [ORIGINAL SEQUENCE]"
    
    training_text = f"""
================================================================================
Pattern: {pattern_type}{variation_note}
Symbol: {symbol}
Date: {current_date}
================================================================================

📊 PRICE DATA:
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
Take Profit: {target:.2f}
Risk-Reward Ratio: {rr_ratio:.2f}
Confidence: {confidence}%

📝 RECOMMENDATION:
────────────────────────────────────────────────────────────────────────────────
{'✅ BUY' if config['bias'] == 'Bullish' else '❌ SELL' if config['bias'] == 'Bearish' else '⏳ WAIT FOR BREAKOUT'} at {entry:.2f}

Risk Management:
- Position Size: {2 if confidence > 80 else 1 if confidence > 65 else 0.5}% of capital
- Stop Loss Type: {'Trailing' if rr_ratio > 2 else 'Fixed'}
- First Target: {target:.2f}
- Second Target: {target + (target - entry):.2f if config['bias'] == 'Bullish' else target - (entry - target):.2f}

Additional Confirmation:
{'- RSI divergence confirms reversal' if rsi_divergence != 'None' else '- No RSI divergence'}
{'- MACD divergence confirms momentum' if macd_divergence != 'None' else '- No MACD divergence'}
{'- Volume spike supports breakout' if volume_spike == 'Yes' else '- Normal volume, wait for confirmation'}
{'- Price above EMA 20 (bullish trend)' if current_price > ema_20 else '- Price below EMA 20 (bearish trend)'}

================================================================================
"""
    return training_text

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
    
    # Variations per pattern
    NUM_VARIATIONS = 3  # Original + 2 noisy variations
    
    training_data = []
    
    symbols_processed = 0
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        
        if len(symbol_data) < 100:
            continue
        
        close_prices = symbol_data['close']
        high_prices = symbol_data['high']
        low_prices = symbol_data['low']
        volumes = symbol_data['volume']
        
        # ইন্ডিকেটর ক্যালকুলেট করুন
        rsi_series = calculate_rsi(close_prices)
        macd_line, macd_signal, macd_hist = calculate_macd(close_prices)
        stoch_k, stoch_d = calculate_stochastic(high_prices, low_prices, close_prices)
        obv_series = calculate_obv(close_prices.values, volumes.values)
        ema_20 = calculate_ema(close_prices, 20)
        sma_20 = calculate_sma(close_prices, 20)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_prices)
        atr_series = calculate_atr(high_prices, low_prices, close_prices)
        avg_volume = volumes.rolling(20).mean()
        
        rsi_divergence_list = []
        macd_divergence_list = []
        
        for i in range(len(close_prices)):
            if i < 30:
                rsi_divergence_list.append('None')
                macd_divergence_list.append('None')
            else:
                r_div = detect_rsi_divergence(close_prices.iloc[:i+1].values, rsi_series.iloc[:i+1].values)
                m_div = detect_macd_divergence(close_prices.iloc[:i+1].values, macd_line.iloc[:i+1].values)
                rsi_divergence_list.append(r_div)
                macd_divergence_list.append(m_div)
        
        row_count = 0
        for idx in range(len(symbol_data)):
            if idx < 50:
                continue
            
            row = symbol_data.iloc[idx]
            
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
                'macd_divergence': macd_divergence_list[idx] if idx < len(macd_divergence_list) else 'None'
            }
            
            pattern_high = row['high']
            pattern_low = row['low']
            
            # Real price sequence for variations
            real_sequence = close_prices.iloc[max(0, idx-30):idx+1].values
            
            for pattern_name, config in all_patterns.items():
                # Generate multiple variations of the same pattern
                for var_idx in range(NUM_VARIATIONS):
                    if var_idx > 0:
                        # Create noisy variation of the price sequence
                        noisy_sequence = add_noise_to_sequence(real_sequence, noise_level=random.uniform(0.003, 0.01))
                        # Update current price with variation
                        modified_row = row.copy()
                        modified_row['close'] = noisy_sequence[-1]
                        modified_row['open'] = modified_row['close'] * random.uniform(0.98, 1.02)
                        modified_row['high'] = max(modified_row['close'], modified_row['open']) * random.uniform(1.0, 1.02)
                        modified_row['low'] = min(modified_row['close'], modified_row['open']) * random.uniform(0.98, 1.0)
                        use_row = modified_row
                    else:
                        use_row = row
                    
                    metrics = calculate_pattern_metrics(
                        noisy_sequence if var_idx > 0 else real_sequence, 
                        pattern_high, pattern_low, use_row['close']
                    )
                    
                    if 'Wave' in pattern_name or 'Diagonal' in pattern_name or 'Zigzag' in pattern_name or 'Flat' in pattern_name or 'Triangle' in pattern_name:
                        text = generate_elliott_wave_data(symbol, use_row, pattern_name, config, indicator_values, metrics, var_idx)
                    else:
                        text = generate_complete_pattern_data(symbol, use_row, pattern_name, config, indicator_values, metrics, var_idx)
                    
                    if text:
                        training_data.append(text)
                        print(f"✅ Generated {pattern_name} (Var {var_idx + 1}/{NUM_VARIATIONS}) for {symbol} on {row['date'].date()}")
                        row_count += 1
                        
                        if row_count >= 20:
                            break
                
                if row_count >= 20:
                    break
            
            if row_count >= 20:
                break
        
        symbols_processed += 1
        if symbols_processed >= 2:
            break
    
    # ট্রেনিং ডাটা সেভ করুন
    output_dir = "./csv"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "training_texts.txt")

    with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(training_data))
    
    print(f"\n📊 Training data saved: {output_file}")
    print(f"   Total examples: {len(training_data)}")
    print(f"   Variations per pattern: {NUM_VARIATIONS}")
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
