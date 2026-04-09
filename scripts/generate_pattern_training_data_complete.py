# scripts/generate_pattern_training_data_complete.py
# RSI Divergence, MACD, Stochastic, ATR, Bollinger Bands, OBV, Volume Profile সহ সম্পূর্ণ ট্রেনিং ডাটা
# 130+ প্যাটার্ন + Elliott Wave + SMC সম্পূর্ণ লাইব্রেরি + Multiple Historical Sequences + Noise Variations
# ✅ NEW: Sector Rotation + Symbol Ranking + Wyckoff + Forward-Looking Analysis + 150+ Candles

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta


# =========================================================
# TRAINING CONFIGURATION
# =========================================================

# Symbol limits
MAX_SYMBOLS = 380       # Process all 380 symbols
MAX_PER_SYMBOL = 10     # 60 examples per symbol (balanced)

# Time control
MAX_EXAMPLES_PER_RUN = 5000  # Max examples to generate (prevents timeout)

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

    first_half_prices = prices_array[:half]
    second_half_prices = prices_array[half:]
    first_half_rsi = rsi_array[:half]
    second_half_rsi = rsi_array[half:]

    if len(first_half_prices) > 0 and len(second_half_prices) > 0:
        first_min_idx = np.argmin(first_half_prices)
        second_min_idx = np.argmin(second_half_prices) + half

        first_min_price = first_half_prices[first_min_idx]
        second_min_price = second_half_prices[np.argmin(second_half_prices)]
        first_min_rsi = first_half_rsi[first_min_idx]
        second_min_rsi = second_half_rsi[np.argmin(second_half_prices)]

        if second_min_price < first_min_price and second_min_rsi > first_min_rsi:
            return 'Bullish'

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

    noisy_sequence = sequence + (sequence * noise)
    noisy_sequence = np.maximum(noisy_sequence, 0.01)

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
# NEW: ADVANCED PRICE SEQUENCE GENERATOR (150+ Candles)
# =========================================================

def find_swing_points(highs, lows, window=5):
    """Swing Highs and Lows খুঁজে বের করুন"""
    swing_highs = []
    swing_lows = []
    
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i-window:i+window+1]):
            swing_lows.append(lows[i])
    
    return swing_highs, swing_lows


def detect_bos_from_swings(swing_highs, swing_lows, current_price):
    """Swing points থেকে BOS ডিটেক্ট করুন"""
    if len(swing_highs) >= 2 and current_price > swing_highs[-2]:
        return True, "BULLISH BOS (Break of Structure)"
    elif len(swing_lows) >= 2 and current_price < swing_lows[-2]:
        return True, "BEARISH BOS (Break of Structure)"
    return False, None


def find_support_resistance(highs, lows, closes, tolerance=0.02):
    """Support এবং Resistance levels খুঁজে বের করুন"""
    all_levels = list(highs) + list(lows)
    levels = {}
    
    for price in all_levels:
        found = False
        for key in levels:
            if abs(price - key) / key < tolerance:
                levels[key] += 1
                found = True
                break
        if not found:
            levels[price] = 1
    
    sorted_levels = sorted(levels.items(), key=lambda x: x[1], reverse=True)
    
    current_price = closes[-1]
    resistance = [price for price, count in sorted_levels if price > current_price and count >= 2]
    support = [price for price, count in sorted_levels if price < current_price and count >= 2]
    
    return {'resistance': sorted(resistance, reverse=True), 'support': sorted(support)}


def count_touches(prices, level, tolerance=0.015):
    """একটি লেভেল কতবার টাচ হয়েছে তা গণনা করুন"""
    return sum(1 for p in prices if abs(p - level) / level < tolerance)


def generate_advanced_price_sequence(symbol_data, idx, lookback=150):
    """১৫০+ ক্যান্ডেল সহ সম্পূর্ণ প্রাইস অ্যাকশন বিশ্লেষণ"""
    start_idx = max(0, idx - lookback)
    sequence_data = symbol_data.iloc[start_idx:idx+1].copy()
    
    if len(sequence_data) < 50:
        return "Insufficient data for 150+ candle analysis.", False, False, False
    
    closes = sequence_data['close'].values
    highs = sequence_data['high'].values
    lows = sequence_data['low'].values
    volumes = sequence_data['volume'].values
    
    text = "📊 COMPREHENSIVE PRICE ANALYSIS (150+ Candles):\n"
    text += "="*80 + "\n\n"
    
    # ========== 1. PRICE DATA TABLE ==========
    text += "📋 PRICE DATA (Last 30 candles):\n"
    text += "─"*80 + "\n"
    text += "Date       | Open   | High   | Low    | Close  | Volume   | Range\n"
    
    recent_data = sequence_data.iloc[-30:]
    for _, row in recent_data.iterrows():
        date_str = str(row['date'])[:10]
        range_val = row['high'] - row['low']
        text += f"{date_str} | {row['open']:7.2f} | {row['high']:7.2f} | {row['low']:7.2f} | {row['close']:7.2f} | {int(row['volume']):8,} | {range_val:.2f}\n"
    
    # Summary of previous candles
    older_data = sequence_data.iloc[:-30] if len(sequence_data) > 30 else sequence_data.iloc[:0]
    if len(older_data) > 0:
        text += f"\n📊 Previous {len(older_data)} candles summary:\n"
        text += f"   High: {older_data['high'].max():.2f} | Low: {older_data['low'].min():.2f}\n"
        text += f"   Avg Close: {older_data['close'].mean():.2f} | Avg Volume: {older_data['volume'].mean():,.0f}\n"
    
    # ========== 2. TREND ANALYSIS ==========
    text += "\n📈 TREND ANALYSIS:\n"
    text += "─"*80 + "\n"
    
    sma20 = np.mean(closes[-20:])
    sma50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma20
    sma150 = np.mean(closes) if len(closes) >= 150 else sma50
    
    current_price = closes[-1]
    
    text += f"Short-term Trend (20):  {'BULLISH 📈' if current_price > sma20 else 'BEARISH 📉'}\n"
    text += f"Medium-term Trend (50): {'BULLISH 📈' if current_price > sma50 else 'BEARISH 📉'}\n"
    text += f"Long-term Trend (150):  {'BULLISH 📈' if current_price > sma150 else 'BEARISH 📉'}\n"
    
    price_change_20 = (closes[-1] - closes[-20]) / closes[-20] * 100 if len(closes) >= 20 else 0
    price_change_50 = (closes[-1] - closes[-50]) / closes[-50] * 100 if len(closes) >= 50 else price_change_20
    price_change_150 = (closes[-1] - closes[0]) / closes[0] * 100
    
    text += f"\nPrice Change (20d):  {price_change_20:+.2f}%\n"
    text += f"Price Change (50d):  {price_change_50:+.2f}%\n"
    text += f"Price Change (150d): {price_change_150:+.2f}%\n"
    
    # ========== 3. MARKET STRUCTURE (SMC) ==========
    text += "\n🏗️ MARKET STRUCTURE (SMC):\n"
    text += "─"*80 + "\n"
    
    swing_highs, swing_lows = find_swing_points(highs, lows)
    
    recent_sh = swing_highs[-5:] if len(swing_highs) >= 5 else swing_highs
    recent_sl = swing_lows[-5:] if len(swing_lows) >= 5 else swing_lows
    
    text += f"Swing Highs: {', '.join([f'{h:.2f}' for h in recent_sh])}\n"
    text += f"Swing Lows:  {', '.join([f'{l:.2f}' for l in recent_sl])}\n"
    
    if len(swing_highs) >= 3:
        if swing_highs[-1] > swing_highs[-2] and swing_highs[-2] > swing_highs[-3]:
            text += "Structure: HIGHER HIGHS (Bullish Continuation) 📈\n"
        elif swing_highs[-1] < swing_highs[-2] and swing_highs[-2] < swing_highs[-3]:
            text += "Structure: LOWER HIGHS (Bearish Continuation) 📉\n"
        else:
            text += "Structure: CHOPPY (Consolidation) ↔️\n"
    
    if len(swing_lows) >= 3:
        if swing_lows[-1] > swing_lows[-2] and swing_lows[-2] > swing_lows[-3]:
            text += "Structure: HIGHER LOWS (Bullish Support) ✅\n"
        elif swing_lows[-1] < swing_lows[-2] and swing_lows[-2] < swing_lows[-3]:
            text += "Structure: LOWER LOWS (Bearish Continuation) ❌\n"
    
    bos_detected, bos_type = detect_bos_from_swings(swing_highs, swing_lows, current_price)
    if bos_detected:
        text += f"\n⚡ {bos_type} DETECTED!\n"
    
    # ========== 4. SUPPORT & RESISTANCE ==========
    text += "\n🎯 SUPPORT & RESISTANCE LEVELS:\n"
    text += "─"*80 + "\n"
    
    levels = find_support_resistance(highs, lows, closes)
    
    text += "Resistance Levels:\n"
    for level in levels['resistance'][:3]:
        touches = count_touches(highs, level)
        text += f"  R{level:.2f} ({touches} touches) - {'STRONG' if touches >= 3 else 'WEAK'}\n"
    
    text += "\nSupport Levels:\n"
    for level in levels['support'][:3]:
        touches = count_touches(lows, level)
        text += f"  S{level:.2f} ({touches} touches) - {'STRONG' if touches >= 3 else 'WEAK'}\n"
    
    nearest_resistance = min([r for r in levels['resistance'] if r > current_price], default=current_price * 1.1)
    nearest_support = max([s for s in levels['support'] if s < current_price], default=current_price * 0.9)
    
    text += f"\nCurrent Price: {current_price:.2f}\n"
    text += f"Nearest Resistance: {nearest_resistance:.2f} ({(nearest_resistance/current_price-1)*100:+.1f}% away)\n"
    text += f"Nearest Support: {nearest_support:.2f} ({(nearest_support/current_price-1)*100:+.1f}% away)\n"
    
    # ========== 5. FIBONACCI LEVELS ==========
    text += "\n📐 FIBONACCI RETRACEMENT:\n"
    text += "─"*80 + "\n"
    
    high_150 = np.max(highs)
    low_150 = np.min(lows)
    high_idx = np.argmax(highs)
    low_idx = np.argmin(lows)
    
    if high_idx > low_idx:
        fib_range = high_150 - low_150
        text += f"Swing Low: {low_150:.2f} → Swing High: {high_150:.2f} (Range: {fib_range:.2f})\n\n"
        text += "Retracement Levels:\n"
        for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
            fib_price = high_150 - (fib_range * level)
            text += f"  {level:.3f} ({level*100:.1f}%): {fib_price:.2f}\n"
    else:
        fib_range = high_150 - low_150
        text += f"Swing High: {high_150:.2f} → Swing Low: {low_150:.2f} (Range: {fib_range:.2f})\n\n"
        text += "Retracement Levels:\n"
        for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
            fib_price = low_150 + (fib_range * level)
            text += f"  {level:.3f} ({level*100:.1f}%): {fib_price:.2f}\n"
    
    # ========== 6. VOLUME ANALYSIS ==========
    text += "\n📊 VOLUME ANALYSIS:\n"
    text += "─"*80 + "\n"
    
    avg_volume_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
    avg_volume_50 = np.mean(volumes[-50:]) if len(volumes) >= 50 else avg_volume_20
    current_volume = volumes[-1]
    
    text += f"Current Volume: {current_volume:,.0f}\n"
    text += f"Avg Volume (20): {avg_volume_20:,.0f} ({'ABOVE' if current_volume > avg_volume_20 else 'BELOW'} average)\n"
    text += f"Avg Volume (50): {avg_volume_50:,.0f}\n"
    text += f"Volume Trend: {'INCREASING 📈' if avg_volume_20 > avg_volume_50 else 'DECREASING 📉'}\n"
    
    volume_spikes = []
    for i in range(20, len(volumes)):
        if volumes[i] > np.mean(volumes[i-20:i]) * 1.5:
            volume_spikes.append((i, volumes[i]))
    
    has_volume_spike = len(volume_spikes) > 0
    
    # ========== 7. VOLATILITY ANALYSIS ==========
    text += "\n📉 VOLATILITY ANALYSIS:\n"
    text += "─"*80 + "\n"
    
    # Simple ATR calculation
    atr_values = []
    for i in range(len(highs)):
        if i < 1:
            atr_values.append(highs[i] - lows[i])
        else:
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            atr_values.append(tr)
    atr_values = np.array(atr_values)
    
    current_atr = atr_values[-1]
    avg_atr = np.mean(atr_values)
    
    text += f"Current ATR: {current_atr:.2f} ({current_atr/current_price*100:.2f}% of price)\n"
    text += f"Average ATR: {avg_atr:.2f}\n"
    text += f"Volatility Regime: {'HIGH' if current_atr > avg_atr * 1.2 else 'LOW' if current_atr < avg_atr * 0.8 else 'NORMAL'}\n"
    
    high_volatility = current_atr > avg_atr * 1.2
    
    # ========== 8. PATTERN EVOLUTION ==========
    text += "\n🔄 PATTERN EVOLUTION (How the pattern formed):\n"
    text += "─"*80 + "\n"
    
    if price_change_20 > 5 and price_change_50 > 10:
        text += "• Strong uptrend over last 50 candles\n"
        text += "• Multiple higher highs and higher lows formed\n"
    elif price_change_20 < -5 and price_change_50 < -10:
        text += "• Strong downtrend over last 50 candles\n"
        text += "• Multiple lower lows and lower highs formed\n"
    else:
        text += "• Price has been consolidating\n"
        text += "• Range-bound movement with no clear trend\n"
    
    last_5 = closes[-5:]
    if len(last_5) >= 5:
        if all(last_5[i] > last_5[i-1] for i in range(1, 5)):
            text += "• Last 5 candles: All bullish (Strong momentum)\n"
        elif all(last_5[i] < last_5[i-1] for i in range(1, 5)):
            text += "• Last 5 candles: All bearish (Strong selling pressure)\n"
        else:
            text += "• Last 5 candles: Mixed (Indecision)\n"
    
    return text, bos_detected, has_volume_spike, high_volatility


# =========================================================
# NEW: WYCKOFF CYCLE & VOLUME-PRICE ANALYSIS
# =========================================================

def detect_volume_price_cycle(symbol_data, idx, lookback=150):
    """Wyckoff Cycle এবং Volume-Price Action Pattern ডিটেক্ট করুন"""
    
    start_idx = max(0, idx - lookback)
    sequence_data = symbol_data.iloc[start_idx:idx+1].copy()
    
    if len(sequence_data) < 60:
        return "Insufficient data for Wyckoff analysis.", {}
    
    closes = sequence_data['close'].values
    volumes = sequence_data['volume'].values
    highs = sequence_data['high'].values
    lows = sequence_data['low'].values
    
    # Volume averages
    vol_10 = np.mean(volumes[-10:]) if len(volumes) >= 10 else np.mean(volumes)
    vol_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else vol_10
    vol_50 = np.mean(volumes[-50:]) if len(volumes) >= 50 else vol_20
    vol_100 = np.mean(volumes[-100:]) if len(volumes) >= 100 else vol_50
    
    current_volume = volumes[-1]
    current_price = closes[-1]
    
    volume_trend = "INCREASING" if vol_10 > vol_20 > vol_50 else "DECREASING" if vol_10 < vol_20 < vol_50 else "NEUTRAL"
    
    avg_volume_50 = vol_50
    volume_ratio = current_volume / avg_volume_50 if avg_volume_50 > 0 else 1
    
    is_volume_spike = volume_ratio >= 1.5
    is_volume_dry = volume_ratio <= 0.5
    
    # Price position
    high_50 = np.max(highs[-50:]) if len(highs) >= 50 else np.max(highs)
    low_50 = np.min(lows[-50:]) if len(lows) >= 50 else np.min(lows)
    range_50 = high_50 - low_50
    price_position = (current_price - low_50) / range_50 if range_50 > 0 else 0.5
    
    price_change_5 = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
    price_change_10 = (closes[-1] - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 else 0
    price_change_20 = (closes[-1] - closes[-20]) / closes[-20] * 100 if len(closes) >= 20 else 0
    
    # Phase detection
    phase = "UNKNOWN"
    confidence_boost = 0
    prediction_text = ""
    
    if volume_trend == "DECREASING" and price_position < 0.3 and abs(price_change_20) < 10:
        phase = "ACCUMULATION"
        confidence_boost = 5
        prediction_text = """
📊 WYCKOFF ANALYSIS - ACCUMULATION PHASE:
────────────────────────────────────────────────────────────────────────────────
Current Phase: ACCUMULATION (Smart Money Buying)
• Volume is drying up - Smart money accumulating quietly
• Price is in lower range - Institutional buying zone
• Expect: Potential markup phase soon

🎯 PREDICTION: Watch for volume surge + price breakout above resistance
Once volume increases significantly with price moving up → STRONG BUY SIGNAL
"""
    elif is_volume_spike and price_change_5 > 0 and price_change_10 > 0 and price_position > 0.5:
        phase = "MARKUP"
        confidence_boost = 20
        prediction_text = f"""
📊 WYCKOFF ANALYSIS - MARKUP PHASE (BREAKOUT DETECTED!):
────────────────────────────────────────────────────────────────────────────────
Current Phase: MARKUP (Strong Uptrend Confirmed)
• Volume Surge: {volume_ratio:.1f}x average volume ⚡
• Price Movement: +{price_change_5:.1f}% (5d) | +{price_change_10:.1f}% (10d)
• Price Position: {price_position*100:.0f}% of recent range

✅ CONFIRMED SIGNALS:
• Volume expansion with price increase
• Breaking above accumulation range
• Smart money pushing price higher

🎯 PREDICTION: CONTINUED UPTREND
"""
    elif (is_volume_spike or volume_trend == "INCREASING") and price_position > 0.7 and price_change_10 < 5:
        phase = "DISTRIBUTION"
        confidence_boost = -5
        prediction_text = """
📊 WYCKOFF ANALYSIS - DISTRIBUTION PHASE:
────────────────────────────────────────────────────────────────────────────────
Current Phase: DISTRIBUTION (Smart Money Selling)
• High volume but price struggling to go higher
• Price is in upper range - Institutional selling zone
• Expect: Potential markdown phase soon

⚠️ CAUTION: Volume is high but price not advancing significantly
Smart money may be distributing to retail buyers
"""
    elif price_change_20 < -5 and price_position < 0.5:
        phase = "MARKDOWN"
        confidence_boost = -10
        prediction_text = """
📊 WYCKOFF ANALYSIS - MARKDOWN PHASE:
────────────────────────────────────────────────────────────────────────────────
Current Phase: MARKDOWN (Downtrend)
• Price declining with increasing volume
• Breaking below support levels
• Smart money has distributed and price is falling

📉 BEARISH: Wait for accumulation phase before entering
"""
    elif volume_trend == "DECREASING" and 0.3 <= price_position <= 0.7 and abs(price_change_20) < 8:
        phase = "RE-ACCUMULATION"
        confidence_boost = 8
        prediction_text = """
📊 WYCKOFF ANALYSIS - RE-ACCUMULATION PHASE:
────────────────────────────────────────────────────────────────────────────────
Current Phase: RE-ACCUMULATION (Consolidation before next move up)
• Volume decreasing - smart money accumulating again
• Price holding in middle/upper range
• Higher low formed - Bullish structure

🎯 PREDICTION: Expect another markup phase
Watch for volume expansion with price breakout
"""
    
    # Divergence detection
    divergence_type = None
    divergence_text = ""
    
    if price_change_10 < -3 and current_volume < avg_volume_50 * 0.7:
        divergence_type = "BULLISH_VOLUME"
        divergence_text = """
⚠️ BULLISH VOLUME DIVERGENCE DETECTED:
• Price is falling but volume is drying up
• Selling pressure is weakening
• Potential reversal signal - Watch for volume expansion
"""
        confidence_boost += 8
    elif price_change_10 > 3 and current_volume < avg_volume_50 * 0.7:
        divergence_type = "BEARISH_VOLUME"
        divergence_text = """
⚠️ BEARISH VOLUME DIVERGENCE DETECTED:
• Price is rising but volume is declining
• Buying pressure is weakening
• Potential pullback or reversal signal
"""
        confidence_boost -= 5
    elif is_volume_spike and price_change_5 > 2:
        divergence_type = "VOLUME_PRICE_CONFIRMATION"
        divergence_text = f"""
✅ VOLUME-PRICE CONFIRMATION (STRONG BULLISH):
• Volume is {volume_ratio:.1f}x average ⚡
• Price is up {price_change_5:.1f}% 
• This confirms genuine buying interest
• High probability of continued upward movement
"""
        confidence_boost += 12
    
    # Breakout prediction
    will_breakout_soon = False
    breakout_confidence = 0
    
    if phase == "ACCUMULATION":
        days_in_accumulation = 0
        for i in range(min(30, len(volumes))):
            if volumes[-1-i] < avg_volume_50 * 0.8:
                days_in_accumulation += 1
        
        if days_in_accumulation >= 10 and price_position < 0.4:
            will_breakout_soon = True
            breakout_confidence = min(70, 40 + days_in_accumulation)
    elif phase == "RE-ACCUMULATION":
        if volume_trend == "DECREASING" and price_change_20 > -5:
            will_breakout_soon = True
            breakout_confidence = 65
    elif phase == "MARKUP" and is_volume_spike:
        will_breakout_soon = True
        breakout_confidence = 85
    
    analysis_text = f"""
📊 VOLUME-PRICE CYCLE ANALYSIS (Wyckoff Method):
================================================================================

🔍 CURRENT MARKET PHASE: {phase}
────────────────────────────────────────────────────────────────────────────────

📈 VOLUME ANALYSIS:
  Current Volume: {current_volume:,.0f}
  vs 10d Avg: {(current_volume/vol_10 - 1)*100:+.1f}%
  vs 20d Avg: {(current_volume/vol_20 - 1)*100:+.1f}%
  vs 50d Avg: {(current_volume/vol_50 - 1)*100:+.1f}%
  Volume Trend: {volume_trend}
  Volume Spike: {'YES ⚡' if is_volume_spike else 'NO'}
  Volume Dry-up: {'YES 💧' if is_volume_dry else 'NO'}

💰 PRICE ANALYSIS:
  Current Price: {current_price:.2f}
  5d Change: {price_change_5:+.2f}%
  10d Change: {price_change_10:+.2f}%
  20d Change: {price_change_20:+.2f}%
  Price Position: {price_position*100:.0f}% of 50d range

{divergence_text}
{prediction_text}

🎯 BREAKOUT PREDICTION:
────────────────────────────────────────────────────────────────────────────────
Will Volume + Price Increase Soon? {'✅ YES' if will_breakout_soon else '❌ NOT YET'}
Confidence: {breakout_confidence}%

Key Levels to Watch:
• Resistance: {high_50:.2f} (Breakout confirmation level)
• Support: {low_50:.2f} (Invalidation level)
• Target if breaks out: {high_50 + range_50*0.5:.2f}

================================================================================
"""
    
    return analysis_text, {
        'phase': phase,
        'will_breakout_soon': will_breakout_soon,
        'breakout_confidence': breakout_confidence,
        'volume_spike': is_volume_spike,
        'volume_ratio': volume_ratio,
        'price_position': price_position,
        'confidence_boost': confidence_boost,
        'volume_trend': volume_trend,
        'divergence': divergence_type
    }


# =========================================================
# NEW: SECTOR ROTATION & SYMBOL RANKING
# =========================================================

def get_sector_analysis(sector, symbol, current_price):
    """Sector-based analysis এবং confidence boost প্রদান করুন"""
    
    sector_strength = {
        'Pharmaceuticals & Chemicals': {'strength': 'Strong', 'confidence_boost': 5},
        'Bank': {'strength': 'Moderate', 'confidence_boost': 2},
        'Telecommunication': {'strength': 'Strong', 'confidence_boost': 4},
        'Fuel & Power': {'strength': 'Moderate', 'confidence_boost': 2},
        'Engineering': {'strength': 'Moderate', 'confidence_boost': 1},
        'Food & Allied': {'strength': 'Strong', 'confidence_boost': 3},
        'Textile': {'strength': 'Weak', 'confidence_boost': -2},
        'Financial Institutions': {'strength': 'Moderate', 'confidence_boost': 1},
        'Services & Real Estate': {'strength': 'Moderate', 'confidence_boost': 0},
        'Cement': {'strength': 'Moderate', 'confidence_boost': 1},
        'IT': {'strength': 'Strong', 'confidence_boost': 4},
        'Insurance': {'strength': 'Moderate', 'confidence_boost': 0},
    }
    
    sector_info = sector_strength.get(str(sector), {'strength': 'Unknown', 'confidence_boost': 0})
    
    rotation_signals = ['None', 'Rotation In', 'Rotation Out', 'Sector Leadership']
    rotation = random.choice(rotation_signals)
    
    if rotation == 'Rotation In':
        sector_info['confidence_boost'] += 3
        additional_note = f"\n- Sector Rotation: {sector} is attracting capital inflow"
    elif rotation == 'Rotation Out':
        sector_info['confidence_boost'] -= 3
        additional_note = f"\n- Sector Rotation: {sector} is seeing capital outflow"
    elif rotation == 'Sector Leadership':
        sector_info['confidence_boost'] += 5
        additional_note = f"\n- Sector Leadership: {sector} is leading the market"
    else:
        additional_note = ""
    
    peer_rank = random.choice(['Top 25%', 'Top 50%', 'Average', 'Below Average'])
    
    return {
        'strength': sector_info['strength'],
        'rotation': rotation,
        'peer_rank': peer_rank,
        'confidence_boost': sector_info['confidence_boost'],
        'additional_note': additional_note
    }


def detect_elliott_wave_position(data):
    """Elliott Wave পজিশন ডিটেক্ট করুন (সিম্পলিফাইড)"""
    if len(data) < 100:
        return 'Unknown', 'Neutral'
    
    closes = data['close'].values
    highs = data['high'].values
    lows = data['low'].values
    
    swing_highs = []
    swing_lows = []
    
    for i in range(5, len(highs) - 5):
        if highs[i] == max(highs[i-5:i+6]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i-5:i+6]):
            swing_lows.append((i, lows[i]))
    
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return 'Unknown', 'Neutral'
    
    recent_highs = [h for _, h in swing_highs[-3:]]
    recent_lows = [l for _, l in swing_lows[-3:]]
    
    if len(recent_highs) >= 3 and len(recent_lows) >= 3:
        if recent_highs[-1] > recent_highs[-2] and recent_lows[-1] > recent_lows[-2]:
            if recent_highs[-1] > recent_highs[-2] * 1.1:
                return 'Wave 3', 'Bullish'
            elif recent_highs[-1] > recent_highs[-2]:
                return 'Wave 5', 'Bullish'
            else:
                return 'Wave 1', 'Bullish'
        elif recent_highs[-1] < recent_highs[-2] and recent_lows[-1] < recent_lows[-2]:
            return 'Wave C', 'Bearish'
        else:
            return 'Wave 4 or B', 'Neutral'
    
    return 'Unknown', 'Neutral'


def detect_smc_structure_for_symbol(data):
    """SMC মার্কেট স্ট্রাকচার ডিটেক্ট করুন"""
    if len(data) < 50:
        return 'Unknown'
    
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    volumes = data['volume'].values
    current_price = closes[-1]
    
    swing_highs = []
    swing_lows = []
    
    for i in range(5, len(highs) - 5):
        if highs[i] == max(highs[i-5:i+6]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i-5:i+6]):
            swing_lows.append(lows[i])
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 'Unknown'
    
    structures = []
    
    if current_price > swing_highs[-2]:
        structures.append('BOS Bullish')
    elif current_price < swing_lows[-2]:
        structures.append('BOS Bearish')
    
    if len(swing_lows) >= 2 and swing_lows[-1] > swing_lows[-2]:
        structures.append('HL')
    if len(swing_highs) >= 2 and swing_highs[-1] > swing_highs[-2]:
        structures.append('HH')
    
    if structures:
        return ' + '.join(structures[:2])
    
    return 'Neutral'


def generate_forward_looking_analysis(symbol_data, idx, lookback=150):
    """ভবিষ্যতের সম্ভাব্য মূল্য আচরণ বিশ্লেষণ"""
    start_idx = max(0, idx - lookback)
    sequence_data = symbol_data.iloc[start_idx:idx+1].copy()
    
    if len(sequence_data) < 50:
        return "Insufficient data for forward-looking analysis."
    
    closes = sequence_data['close'].values
    highs = sequence_data['high'].values
    lows = sequence_data['low'].values
    volumes = sequence_data['volume'].values
    current_price = closes[-1]
    
    swing_highs, swing_lows = find_swing_points(highs, lows)
    
    structure_desc = "Neutral"
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        if swing_highs[-1] > swing_highs[-2] and swing_lows[-1] > swing_lows[-2]:
            structure_desc = "Higher High (HH) and Higher Low (HL)"
        elif swing_highs[-1] < swing_highs[-2] and swing_lows[-1] < swing_lows[-2]:
            structure_desc = "Lower High (LH) and Lower Low (LL)"
    
    bos_detected, bos_type = detect_bos_from_swings(swing_highs, swing_lows, current_price)
    wave_position, wave_bias = detect_elliott_wave_position(sequence_data)
    
    analysis_text = """
🔮 FORWARD-LOOKING ANALYSIS (Expected Price Behavior):
────────────────────────────────────────────────────────────────────────────────
Based on current structure, here is the expected behavior:

"""
    analysis_text += f"Current Structure: {structure_desc}."
    if bos_detected:
        analysis_text += f" A {bos_type} has just occurred"
        analysis_text += " on high volume." if volumes[-1] > np.mean(volumes[-20:]) * 1.5 else "."
    analysis_text += "\n\n"
    
    if 'HH' in structure_desc and 'HL' in structure_desc:
        analysis_text += """Typical Next Move: After establishing a bullish structure, price often pulls back to a newly formed Order Block (OB) or Fair Value Gap (FVG) for a 'retest' before continuing higher. This retest offers a high-probability entry for a swing trade.

What to Watch For:
1. A pullback to the nearest OB/FVG zone for potential long entry confirmation.
2. A daily close below the most recent HL would be an early warning sign of weakness.
3. A sharp, high-volume reversal breaking below the recent HL could signal a Change of Character (CHoCH) to the downside.
"""
    elif 'LH' in structure_desc and 'LL' in structure_desc:
        analysis_text += """Typical Next Move: In a bearish structure, price typically retraces to a breaker block or an FVG before resuming its downtrend. This retracement offers a potential short entry.

What to Watch For:
1. A pullback to the nearest supply zone (Bearish OB) for potential short entry confirmation.
2. A daily close above the most recent LH would be an early sign of bullish strength.
3. A strong, high-volume breakout above the recent LH could signal a bullish CHoCH.
"""
    elif bos_detected and 'Bullish' in bos_type:
        analysis_text += """Typical Next Move: A bullish Break of Structure is a strong momentum signal. The immediate move often continues, but a pullback to the 'breaker' (the level that was broken) is common. This retest is a key area to watch for adding to longs.

What to Watch For:
1. Continuation of the breakout candle towards the next resistance level.
2. A successful retest of the breakout level (now turned support).
3. A failed retest (price falling back below the breakout level) would be a bearish trap.
"""
    elif wave_position == 'Wave 3' and wave_bias == 'Bullish':
        analysis_text += """Elliott Wave Context: Price is currently in a powerful Wave 3. This wave is typically the strongest and longest, often reaching the 1.618 Fibonacci extension of Wave 1. The expectation is for a sustained bullish move with brief, shallow pullbacks.

What to Watch For:
1. Price to target the 1.618 Fibonacci extension level.
2. Any pullback should be supported by the top of Wave 1 (no overlap rule).
3. A daily close below the Wave 1 high would be a major warning sign that the wave count is invalid.
"""
    else:
        analysis_text += """Typical Next Move: The market is currently in a phase of consolidation or transition. The price is likely to remain range-bound or choppy until a clear Break of Structure (BOS) or Change of Character (CHoCH) occurs.

What to Watch For:
1. A clear breakout above or below the recent trading range on increased volume.
2. Formation of a new Order Block or FVG that could dictate the next directional move.
3. Avoid taking large positions until a new structural trend is confirmed.
"""
    
    if wave_position != 'Unknown':
        analysis_text += f"\nElliott Wave Context: Currently in {wave_position}. "
        if wave_position == 'Wave 3':
            analysis_text += "This wave is typically the strongest, often reaching 1.618-2.618 of Wave 1."
        elif wave_position == 'Wave 5':
            analysis_text += "This is the final motive wave, watch for RSI divergence as a sign of an impending reversal."
        elif wave_position == 'Wave C':
            analysis_text += "This is the final corrective wave, often terminating near the 1.0 or 1.618 extension of Wave A."
        analysis_text += "\n"
    
    return analysis_text


# =========================================================
# SMC PATTERN DETECTION FUNCTIONS (NEW)
# =========================================================

def detect_order_block(df, idx):
    """Order Block ডিটেক্ট করুন - Bullish/Bearish"""
    if idx < 5:
        return None
    
    recent = df.iloc[max(0, idx-5):idx+1]
    result = []
    
    for i in range(len(recent)-2, 0, -1):
        if recent.iloc[i]['close'] < recent.iloc[i]['open']:
            next_candles = recent.iloc[i+1:]
            if len(next_candles) > 0:
                avg_gain = (next_candles['close'].values - next_candles['open'].values).mean()
                if avg_gain > 0:
                    result.append('Bullish Order Block')
            break
    
    for i in range(len(recent)-2, 0, -1):
        if recent.iloc[i]['close'] > recent.iloc[i]['open']:
            next_candles = recent.iloc[i+1:]
            if len(next_candles) > 0:
                avg_loss = (next_candles['open'].values - next_candles['close'].values).mean()
                if avg_loss > 0:
                    result.append('Bearish Order Block')
            break
    
    return result if result else None


def detect_fair_value_gap(df, idx):
    """Fair Value Gap (FVG) ডিটেক্ট করুন"""
    if idx < 3:
        return None
    
    candle1 = df.iloc[idx-2]
    candle2 = df.iloc[idx-1]
    candle3 = df.iloc[idx]
    
    result = []
    
    if candle1['low'] > candle3['high']:
        result.append('Bullish FVG')
    
    if candle1['high'] < candle3['low']:
        result.append('Bearish FVG')
    
    return result if result else None


def detect_liquidity_pools(df, idx):
    """Liquidity Pools (Equal Highs/Lows) ডিটেক্ট করুন"""
    if idx < 20:
        return None
    
    recent = df.iloc[idx-20:idx]
    highs = recent['high'].values
    lows = recent['low'].values
    
    result = []
    high_tolerance = np.mean(highs) * 0.005
    
    for i in range(len(highs)-1):
        for j in range(i+1, len(highs)):
            if abs(highs[i] - highs[j]) <= high_tolerance:
                result.append('Buy Side Liquidity')
                break
        if 'Buy Side Liquidity' in result:
            break
    
    low_tolerance = np.mean(lows) * 0.005
    for i in range(len(lows)-1):
        for j in range(i+1, len(lows)):
            if abs(lows[i] - lows[j]) <= low_tolerance:
                result.append('Sell Side Liquidity')
                break
        if 'Sell Side Liquidity' in result:
            break
    
    if 'Buy Side Liquidity' in result:
        result.append('Equal Highs')
    if 'Sell Side Liquidity' in result:
        result.append('Equal Lows')
    
    current_price = df.iloc[idx]['close']
    if current_price > np.max(highs) * 1.001:
        result.append('Liquidity Sweep')
    elif current_price < np.min(lows) * 0.999:
        result.append('Liquidity Sweep')
    
    return result if result else None


def detect_market_structure_smc(df, idx):
    """Market Structure (BOS/CHoCH/MSS) ডিটেক্ট করুন"""
    if idx < 30:
        return None
    
    recent = df.iloc[idx-30:idx]
    highs = recent['high'].values
    lows = recent['low'].values
    closes = recent['close'].values
    
    result = []
    
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(highs)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])
    
    if len(swing_highs) >= 2:
        if swing_highs[-1] > swing_highs[-2]:
            result.append('Higher High (HH)')
        else:
            result.append('Lower High (LH)')
    
    if len(swing_lows) >= 2:
        if swing_lows[-1] > swing_lows[-2]:
            result.append('Higher Low (HL)')
        else:
            result.append('Lower Low (LL)')
    
    if len(swing_highs) >= 2 and swing_highs[-1] > swing_highs[-2]:
        result.append('Break of Structure (BOS)')
    
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        if closes[-1] < swing_lows[-2]:
            result.append('Change of Character (CHoCH)')
        elif closes[-1] > swing_highs[-2]:
            result.append('Change of Character (CHoCH)')
    
    return result if result else None


def detect_ote_entry(df, idx):
    """Optimal Trade Entry (OTE) Zone ডিটেক্ট করুন"""
    if idx < 20:
        return None
    
    recent = df.iloc[idx-20:idx]
    swing_high = recent['high'].max()
    swing_low = recent['low'].min()
    current_price = df.iloc[idx]['close']
    
    range_size = swing_high - swing_low
    if range_size <= 0:
        return None
    
    result = []
    
    fib_618 = swing_low + range_size * 0.618
    fib_786 = swing_low + range_size * 0.786
    if fib_618 <= current_price <= fib_786:
        result.append('Optimal Trade Entry (OTE)')
        result.append('Discount Zone Ugc')
    
    fib_618_short = swing_high - range_size * 0.618
    fib_786_short = swing_high - range_size * 0.786
    if fib_786_short <= current_price <= fib_618_short:
        result.append('Optimal Trade Entry (OTE)')
        result.append('Premium Zone Ugc')
    
    return result if result else None


def detect_smc_manipulation(df, idx):
    """SMC Manipulation patterns (Judas Swing, Power of 3, Fake Breakout)"""
    if idx < 10:
        return None
    
    recent = df.iloc[idx-10:idx]
    result = []
    
    highs = recent['high'].values
    lows = recent['low'].values
    closes = recent['close'].values
    
    if len(highs) >= 5:
        recent_high = np.max(highs[:-1])
        if highs[-1] > recent_high * 1.005 and closes[-1] < recent_high:
            result.append('Fake Breakout')
            result.append('Bull Trap Ugc')
    
    if len(lows) >= 5:
        recent_low = np.min(lows[:-1])
        if lows[-1] < recent_low * 0.995 and closes[-1] > recent_low:
            result.append('Fake Breakout')
            result.append('Bear Trap Ugc')
    
    return result if result else None


def detect_smc_hybrid(df, idx):
    """Hybrid SMC patterns (OB+FVG, Sweep+CHoCH)"""
    ob = detect_order_block(df, idx)
    fvg = detect_fair_value_gap(df, idx)
    liq = detect_liquidity_pools(df, idx)
    ms = detect_market_structure_smc(df, idx)
    
    result = []
    
    if ob and fvg:
        result.append('OB + FVG Combo Ugc')
    
    if liq and 'Liquidity Sweep' in liq:
        if ms and 'Change of Character (CHoCH)' in ms:
            result.append('Liquidity Sweep + CHoCH Ugc')
    
    if fvg and ob:
        result.append('Breaker + FVG Ugc')
    
    return result if result else None


# =========================================================
# ORIGINAL PATTERN DETECTION FUNCTIONS (UNCHANGED)
# =========================================================

def detect_cup_and_handle(df, idx):
    """Cup and Handle pattern detection"""
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
    """Double Bottom pattern detection"""
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
    """Head and Shoulders pattern detection"""
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
    """Bull Flag pattern detection"""
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
    """Ascending Triangle pattern detection"""
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
    """Descending Triangle pattern detection"""
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
    """Symmetrical Triangle pattern detection"""
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
    """Rounding Bottom pattern detection"""
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
    """Volume spike detection"""
    if idx < 20:
        return False

    recent_volumes = df['volume'].iloc[idx-20:idx].values
    avg_volume = np.mean(recent_volumes)
    std_volume = np.std(recent_volumes)
    current_volume = df.iloc[idx]['volume']

    zscore = (current_volume - avg_volume) / (std_volume + 1e-6)

    if zscore > 2.0:
        return True

    return False


def detect_bollinger_squeeze(df, idx):
    """Bollinger Band Squeeze detection"""
    if idx < 20:
        return False

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


# =========================================================
# COMPLETE PATTERN DETECTION (ORIGINAL + SMC)
# =========================================================

def detect_all_patterns(df, idx):
    """একসাথে সব প্যাটার্ন ডিটেক্ট করুন - Original + SMC"""
    detected = []

    # ========== ORIGINAL CHART PATTERNS ==========
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

    # ========== CANDLESTICK PATTERNS ==========
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

    # ========== VOLUME & VOLATILITY ==========
    if detect_volume_spike(df, idx):
        detected.append('Volume Climax')
    if detect_bollinger_squeeze(df, idx):
        detected.append('Bollinger Band Squeeze')

    # ========== SMC PATTERNS (NEW) ==========
    ob = detect_order_block(df, idx)
    if ob:
        detected.extend(ob)
    
    fvg = detect_fair_value_gap(df, idx)
    if fvg:
        detected.extend(fvg)
    
    liq = detect_liquidity_pools(df, idx)
    if liq:
        detected.extend(liq)
    
    ms = detect_market_structure_smc(df, idx)
    if ms:
        detected.extend(ms)
    
    ote = detect_ote_entry(df, idx)
    if ote:
        detected.extend(ote)
    
    manip = detect_smc_manipulation(df, idx)
    if manip:
        detected.extend(manip)
    
    hybrid = detect_smc_hybrid(df, idx)
    if hybrid:
        detected.extend(hybrid)

    # Remove duplicates
    return list(set(detected))


# =========================================================
# NO PATTERN EXAMPLE GENERATOR
# =========================================================

def generate_no_pattern_example(symbol, df_row, indicator_values):
    """Generate negative examples to prevent overfitting"""
    current_price = df_row['close']
    current_date = df_row['date']
    sector = df_row.get('sector', 'Unknown')

    rsi = indicator_values.get('rsi', 50)
    macd = indicator_values.get('macd', 0)
    macd_signal = indicator_values.get('macd_signal', 0)
    volume = indicator_values.get('volume', 1000000)
    avg_vol = indicator_values.get('avg_volume', volume)

    volume_spike = "Yes" if volume > avg_vol * 1.5 else "No"
    
    sector_info = f"""
🏭 SECTOR: {sector}
""" if sector != 'Unknown' else ""

    training_text = f"""
================================================================================
Pattern: NO CLEAR PATTERN
Symbol: {symbol}
Date: {current_date}
================================================================================
{sector_info}
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

def generate_elliott_wave_data(symbol, df_row, pattern_type, config, indicator_values, metrics, variation_idx=0, symbol_data=None, idx=None):
    """Elliott Wave প্যাটার্নের জন্য ডাটা তৈরি (Variations সহ)"""
    if config.get('category') not in ['Motive Wave', 'Corrective Wave', 'Wave Relationships', 'Combination']:
        return generate_complete_pattern_data(symbol, df_row, pattern_type, config, indicator_values, metrics, variation_idx, symbol_data, idx)

    current_price = df_row['close']
    current_date = df_row['date']
    sector = df_row.get('sector', 'Unknown')

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

    confidence = 50
    confidence += (macd > macd_signal and config['bias'] == 'Bullish') * 10
    confidence += (macd < macd_signal and config['bias'] == 'Bearish') * 10
    confidence += (rsi < 40 and config['bias'] == 'Bullish') * 5
    confidence += (rsi > 60 and config['bias'] == 'Bearish') * 5
    confidence += (volume > avg_vol * 1.5) * 5
    confidence += (metrics.get('relative_strength', 0) > 0.6) * 5
    confidence += (rsi_divergence != 'None') * random.randint(3, 6)
    confidence += (macd_divergence != 'None') * random.randint(2, 5)

    # ✅ Sector-based confidence
    sector_analysis = get_sector_analysis(sector, symbol, current_price)
    confidence += sector_analysis.get('confidence_boost', 0)

    confidence += random.uniform(-5, 5)
    confidence = min(95, max(30, confidence))

    rr_ratio = abs((target - entry) / max(abs(entry - stop), 1e-6))
    volume_spike = "Yes" if volume > avg_vol * 1.5 else "No"
    variation_note = f" [VARIATION {variation_idx + 1}]" if variation_idx > 0 else " [ORIGINAL SEQUENCE]"

    if random.random() < 0.5:
        pattern_display = pattern_type
    else:
        pattern_display = "Unknown Pattern"

    if random.random() < 0.3:
        price_header = "PRICE SNAPSHOT:"
    else:
        price_header = "📊 PRICE DATA:"

    # ✅ Sector details
    sector_details = f"""
🏭 SECTOR INFORMATION:
────────────────────────────────────────────────────────────────────────────────
Sector: {sector}
Sector Strength: {sector_analysis.get('strength', 'Neutral')}
Sector Rotation Signal: {sector_analysis.get('rotation', 'None')}
Peer Comparison: {sector_analysis.get('peer_rank', 'N/A')}
"""

    # ✅ Advanced Price Sequence
    price_sequence_text = ""
    if symbol_data is not None and idx is not None:
        price_sequence_text, bos_detected, vol_spike, high_vol = generate_advanced_price_sequence(symbol_data, idx)
        if bos_detected:
            confidence += 10
        if vol_spike:
            confidence += 5
        if high_vol:
            confidence += 3

    # ✅ Wyckoff Analysis
    wyckoff_text = ""
    wyckoff_data = {}
    if symbol_data is not None and idx is not None:
        wyckoff_text, wyckoff_data = detect_volume_price_cycle(symbol_data, idx)
        confidence += wyckoff_data.get('confidence_boost', 0)

    # ✅ Forward-Looking Analysis
    forward_text = ""
    if symbol_data is not None and idx is not None:
        forward_text = generate_forward_looking_analysis(symbol_data, idx)

    training_text = f"""
================================================================================
Elliott Wave Pattern: {pattern_display}{variation_note}
Symbol: {symbol}
Date: {current_date}
================================================================================

{sector_details}
{price_sequence_text}
{wyckoff_text}
{forward_text}

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
{sector_analysis.get('additional_note', '')}

================================================================================
"""
    return training_text


def get_elliott_wave_patterns():
    """Elliott Wave সম্পূর্ণ প্যাটার্ন লাইব্রেরি"""
    return {
        'Impulse Wave': {
            'category': 'Motive Wave', 
            'structure': '5-3-5-3-5', 
            'bias': 'Bullish',
            'degree': 'Primary/Intermediate/Minor', 
            'fib_ratios': 'Wave 2: 0.382-0.618, Wave 3: 1.618-2.618',
            'specifications': 'Wave 1: Initial move\nWave 2: Shallow/deep retracement\nWave 3: Strongest wave\nWave 4: Simple/complex correction\nWave 5: Divergence check',
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
            'specifications': 'Occurs in Wave 1 or A position\nOverlapping waves\nNarrowing wedge shape',
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
            'specifications': 'Occurs in Wave 5 or C position\nOverlapping waves\nVolume spike at termination',
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
            'specifications': 'Most common extension\nStrongest momentum\nHighest volume',
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
            'specifications': 'Terminal move\nDivergence with oscillators\nLower volume than Wave 3',
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
            'specifications': 'Two zigzags connected by X wave\nDeeper/longer correction',
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
            'specifications': 'Wave B exceeds Wave A start\nWave C exceeds Wave B',
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
            'fib_ratios': 'Wave E = 0.618-0.786 x Wave C',
            'specifications': '5 waves: A-B-C-D-E\nContracting range',
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
            'fib_ratios': 'Wave E = 1.236-1.382 x Wave C',
            'specifications': '5 waves expanding\nIncreasing range',
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
            'fib_ratios': 'Wave 2 = 0.382-0.618, Wave 3 = 1.618-2.618',
            'specifications': 'Wave 3 cannot be shortest\nWave 4 cannot overlap Wave 1',
            'wave_position': 'All waves', 
            'wave_count': 'Complete cycle',
            'expected_target': 'Based on Fibonacci', 
            'invalidation': 'Relationships violated'
        }
    }


def get_all_patterns():
    """130+ প্যাটার্নের সম্পূর্ণ তালিকা (Elliott Wave + SMC সহ)"""
    patterns = {
        # ========== CHART PATTERNS ==========
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
        
        # ========== CANDLESTICK PATTERNS ==========
        'Hammer': {'category': 'Candlestick', 'bias': 'Bullish', 'timeframe': 'Intraday', 'entry': 'Confirm next green candle', 'stop': 'Below wick', 'target': 'Recent resistance'},
        'Morning Star': {'category': 'Candlestick', 'bias': 'Bullish', 'timeframe': 'Intraday', 'entry': '3-candle confirmation', 'stop': 'Below low', 'target': 'Resistance'},
        'Bullish Engulfing': {'category': 'Candlestick', 'bias': 'Bullish', 'timeframe': 'Intraday', 'entry': 'Engulfing close', 'stop': 'Below candle', 'target': 'Resistance'},
        'Piercing Line': {'category': 'Candlestick', 'bias': 'Bullish', 'timeframe': 'Intraday', 'entry': 'Confirm next candle', 'stop': 'Below low', 'target': 'Resistance'},
        'Three White Soldiers': {'category': 'Candlestick', 'bias': 'Bullish', 'timeframe': 'Intraday', 'entry': 'After 3rd candle', 'stop': 'Below first candle', 'target': 'Recent high'},
        'Shooting Star': {'category': 'Candlestick', 'bias': 'Bearish', 'timeframe': 'Intraday', 'entry': 'Confirm red candle', 'stop': 'Above wick', 'target': 'Support'},
        'Evening Star': {'category': 'Candlestick', 'bias': 'Bearish', 'timeframe': 'Intraday', 'entry': '3-candle confirm', 'stop': 'Above high', 'target': 'Support'},
        'Bearish Engulfing': {'category': 'Candlestick', 'bias': 'Bearish', 'timeframe': 'Intraday', 'entry': 'Engulf close', 'stop': 'Above candle', 'target': 'Support'},
        'Doji': {'category': 'Candlestick', 'bias': 'Neutral', 'timeframe': 'Intraday', 'entry': 'Wait breakout', 'stop': 'High/low', 'target': 'Next move'},
        
        # ========== HARMONIC PATTERNS ==========
        'Gartley': {'category': 'Harmonic', 'bias': 'Both', 'timeframe': 'Swing', 'entry': 'PRZ zone', 'stop': 'Beyond X', 'target': 'Fibonacci targets'},
        'Butterfly': {'category': 'Harmonic', 'bias': 'Both', 'timeframe': 'Swing', 'entry': 'PRZ', 'stop': 'Beyond X', 'target': 'Fib'},
        'Bat': {'category': 'Harmonic', 'bias': 'Both', 'timeframe': 'Swing', 'entry': 'PRZ', 'stop': 'Beyond X', 'target': 'Fib Ugc'},
        
        # ========== VOLUME & VOLATILITY ==========
        'Volume Climax': {'category': 'Volume', 'bias': 'Reversal', 'timeframe': 'Any', 'entry': 'Spike volume', 'stop': 'Recent extreme', 'target': 'Reversal zone'},
        'Bollinger Band Squeeze': {'category': 'Volatility', 'bias': 'Breakout', 'timeframe': 'Any', 'entry': 'Expansion', 'stop': 'Opp band', 'target': 'Move'},
        'Inside Bar': {'category': 'Volatility', 'bias': 'Breakout', 'timeframe': 'Intraday', 'entry': 'Break mother bar', 'stop': 'Opp side', 'target': 'Range'},
        'Outside Bar': {'category': 'Volatility', 'bias': 'Both', 'timeframe': 'Intraday', 'entry': 'Break high/low', 'stop': 'Opp side', 'target': 'Range Ugc'},
        
        # ========== BREAKOUT PATTERNS ==========
        'False Breakout': {'category': 'Breakout', 'bias': 'Trap', 'timeframe': 'Any', 'entry': 'Re-entry opposite', 'stop': 'Recent high/low', 'target': 'Range Ugc'},
        'Breakout Pullback': {'category': 'Breakout', 'bias': 'Continuation', 'timeframe': 'Swing', 'entry': 'Retest entry', 'stop': 'Below pullback', 'target': 'Trend Ugc'},
        
        # ========== PRICE ACTION ==========
        'Pin Bar': {'category': 'Price Action', 'bias': 'Reversal', 'timeframe': 'Any', 'entry': 'Wick rejection', 'stop': 'Beyond wick', 'target': 'Structure Ugc'},
        '1-2-3 Pattern': {'category': 'Price Action', 'bias': 'Reversal', 'timeframe': 'Any', 'entry': 'Point 2 break', 'stop': 'Below 3', 'target': 'Projection Ugc'},
        'Wolfe Wave': {'category': 'Advanced', 'bias': 'Both', 'timeframe': 'Swing', 'entry': 'Wave completion', 'stop': 'Beyond wave', 'target': 'Target line Ugc'},
        
        # ========== SMC PATTERNS (NEW - 70+) ==========
        'Break of Structure (BOS)': {'category': 'SMC', 'bias': 'Both', 'timeframe': 'Swing', 'entry': 'Retest of broken structure', 'stop': 'Beyond recent swing', 'target': 'Next structure Ugc'},
        'Change of Character (CHoCH)': {'category': 'SMC', 'bias': 'Both', 'timeframe': 'Swing', 'entry': 'After CHoCH confirmation', 'stop': 'Beyond CHoCH level', 'target': 'First OB/FVG Ugc'},
        'Higher High (HH)': {'category': 'SMC', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Pullback to HL', 'stop': 'Below HL', 'target': 'Next resistance Ugc'},
        'Higher Low (HL)': {'category': 'SMC', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Bounce from HL', 'stop': 'Below HL', 'target': 'Previous HH Ugc'},
        'Lower High (LH)': {'category': 'SMC', 'bias': 'Bearish', 'timeframe': 'Swing', 'entry': 'Rejection at LH', 'stop': 'Above LH', 'target': 'Previous LL Ugc'},
        'Lower Low (LL)': {'category': 'SMC', 'bias': 'Bearish', 'timeframe': 'Swing', 'entry': 'Break of LL', 'stop': 'Above recent LH', 'target': 'Next support Ugc'},
        'Liquidity Sweep': {'category': 'SMC', 'bias': 'Both', 'timeframe': 'Any', 'entry': 'After sweep confirmation', 'stop': 'Beyond swept level', 'target': 'Opposite liquidity Ugc'},
        'Buy Side Liquidity': {'category': 'SMC', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Wait for BSL sweep', 'stop': 'Below recent low', 'target': 'BSL level Ugc'},
        'Sell Side Liquidity': {'category': 'SMC', 'bias': 'Bearish', 'timeframe': 'Swing', 'entry': 'Wait for SSL sweep', 'stop': 'Above recent high', 'target': 'SSL level Ugc'},
        'Equal Highs': {'category': 'SMC', 'bias': 'Bearish', 'timeframe': 'Swing', 'entry': 'Sell at equal highs rejection', 'stop': 'Above equal highs', 'target': 'SSL Ugc'},
        'Equal Lows': {'category': 'SMC', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Buy at equal lows bounce', 'stop': 'Below equal lows', 'target': 'BSL Ugc'},
        'Bullish Order Block': {'category': 'SMC', 'bias': 'Bullish', 'timeframe': 'Any', 'entry': 'OB retest', 'stop': 'Below OB low', 'target': 'Next liquidity Ugc'},
        'Bearish Order Block': {'category': 'SMC', 'bias': 'Bearish', 'timeframe': 'Any', 'entry': 'OB retest', 'stop': 'Above OB high', 'target': 'Next liquidity Ugc'},
        'Fair Value Gap (FVG)': {'category': 'SMC', 'bias': 'Both', 'timeframe': 'Any', 'entry': 'FVG fill', 'stop': 'Beyond FVG', 'target': 'Next OB Ugc'},
        'Bullish FVG': {'category': 'SMC', 'bias': 'Bullish', 'timeframe': 'Any', 'entry': 'FVG fill', 'stop': 'Below FVG', 'target': 'Next OB Ugc'},
        'Bearish FVG': {'category': 'SMC', 'bias': 'Bearish', 'timeframe': 'Any', 'entry': 'FVG fill', 'stop': 'Above FVG', 'target': 'Next OB Ugc'},
        'Optimal Trade Entry (OTE)': {'category': 'SMC', 'bias': 'Both', 'timeframe': 'Short-term', 'entry': '0.618-0.786 Fib zone', 'stop': 'Beyond 0.786', 'target': 'Swing high/low Ugc'},
        'Discount Zone': {'category': 'SMC', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Below 50% retracement', 'stop': 'Below discount zone', 'target': 'Premium zone Ugc'},
        'Premium Zone': {'category': 'SMC', 'bias': 'Bearish', 'timeframe': 'Swing', 'entry': 'Above 50% retracement', 'stop': 'Above premium zone', 'target': 'Discount zone Ugc'},
        'Judas Swing': {'category': 'SMC', 'bias': 'Both', 'timeframe': 'Daily', 'entry': 'After Judas Swing trap', 'stop': 'Beyond swing extreme', 'target': 'Opposite extreme Ugc'},
        'Fake Breakout': {'category': 'SMC', 'bias': 'Both', 'timeframe': 'Any', 'entry': 'Trade the reversal', 'stop': 'Beyond fakeout candle', 'target': 'Opposite side Ugc'},
        'Bull Trap': {'category': 'SMC', 'bias': 'Bearish', 'timeframe': 'Swing', 'entry': 'Sell after bull trap', 'stop': 'Above trap high', 'target': 'Recent low Ugc'},
        'Bear Trap': {'category': 'SMC', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Buy after bear trap', 'stop': 'Below trap low', 'target': 'Recent high Ugc'},
        'OB + FVG Combo': {'category': 'SMC', 'bias': 'Both', 'timeframe': 'Any', 'entry': 'Confluence zone', 'stop': 'Beyond confluence', 'target': 'Next structure Ugc'},
        'Liquidity Sweep + CHoCH': {'category': 'SMC', 'bias': 'Both', 'timeframe': 'Swing', 'entry': 'After sweep + CHoCH', 'stop': 'Beyond CHoCH', 'target': 'Opposite liquidity Ugc'},
        'Breaker + FVG': {'category': 'SMC', 'bias': 'Both', 'timeframe': 'Swing', 'entry': 'Breaker with FVG confluence', 'stop': 'Beyond confluence', 'target': 'Next OB Ugc'},
    }

    elliott_patterns = get_elliott_wave_patterns()
    patterns.update(elliott_patterns)
    
    return patterns


def generate_complete_pattern_data(symbol, df_row, pattern_type, config, indicator_values, metrics, variation_idx=0, symbol_data=None, idx=None):
    """সব ইন্ডিকেটর এবং মেট্রিক্স সহ সম্পূর্ণ প্যাটার্ন ডাটা তৈরি"""
    current_price = df_row['close']
    current_date = df_row['date']
    sector = df_row.get('sector', 'Unknown')

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

    # ✅ Sector-based confidence
    sector_analysis = get_sector_analysis(sector, symbol, current_price)
    confidence += sector_analysis.get('confidence_boost', 0)

    confidence += random.uniform(-5, 5)
    confidence = min(95, max(30, confidence))

    rr_ratio = abs((target - entry) / max(abs(entry - stop), 1e-6))
    volume_spike = "Yes" if volume > avg_vol * 1.5 else "No"
    bb_position = "Above upper band" if current_price > bb_upper else "Below lower band" if current_price < bb_lower else "Within bands"
    variation_note = f" [VARIATION {variation_idx + 1}]" if variation_idx > 0 else " [ORIGINAL SEQUENCE]"

    if random.random() < 0.5:
        pattern_display = pattern_type
    else:
        pattern_display = "Unknown Pattern"

    if random.random() < 0.3:
        price_header = "PRICE SNAPSHOT:"
    else:
        price_header = "📊 PRICE DATA:"

    # ✅ Sector details
    sector_details = f"""
🏭 SECTOR INFORMATION:
────────────────────────────────────────────────────────────────────────────────
Sector: {sector}
Sector Strength: {sector_analysis.get('strength', 'Neutral')}
Sector Rotation Signal: {sector_analysis.get('rotation', 'None')}
Peer Comparison: {sector_analysis.get('peer_rank', 'N/A')}
"""

    # ✅ Advanced Price Sequence
    price_sequence_text = ""
    if symbol_data is not None and idx is not None:
        price_sequence_text, bos_detected, vol_spike, high_vol = generate_advanced_price_sequence(symbol_data, idx)
        if bos_detected:
            confidence += 10
        if vol_spike:
            confidence += 5
        if high_vol:
            confidence += 3

    # ✅ Wyckoff Analysis
    wyckoff_text = ""
    wyckoff_data = {}
    if symbol_data is not None and idx is not None:
        wyckoff_text, wyckoff_data = detect_volume_price_cycle(symbol_data, idx)
        confidence += wyckoff_data.get('confidence_boost', 0)

    # ✅ Forward-Looking Analysis
    forward_text = ""
    if symbol_data is not None and idx is not None:
        forward_text = generate_forward_looking_analysis(symbol_data, idx)

    # SMC specific details
    smc_details = ""
    if config.get('category') == 'SMC':
        smc_details = f"""
🎯 SMC ANALYSIS:
────────────────────────────────────────────────────────────────────────────────
Pattern Type: {pattern_type}
Category: Smart Money Concepts
Bias: {config['bias']}
Timeframe: {config['timeframe']}
Entry Rule: {config['entry']}
Stop Loss Rule: {config['stop']}
Target Rule: {config['target']}
"""

    training_text = f"""
================================================================================
Pattern: {pattern_display}{variation_note}
Symbol: {symbol}
Date: {current_date}
================================================================================

{sector_details}
{price_sequence_text}
{wyckoff_text}
{forward_text}

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
{smc_details}
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
{sector_analysis.get('additional_note', '')}

================================================================================
"""
    return training_text


# =========================================================
# MAIN FUNCTION (UNCHANGED STRUCTURE)
# =========================================================

def main():
    print("="*80)
    print("🚀 COMPLETE PATTERN TRAINING DATA GENERATOR")
    print("   (130+ Patterns + Elliott Wave + SMC + Multiple Historical Sequences)")
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
        last_detected_idx = {}

        symbol_data = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)

        if len(symbol_data) < 100:
            continue

        close_prices = symbol_data['close']
        high_prices = symbol_data['high']
        low_prices = symbol_data['low']
        volumes = symbol_data['volume']

        vwap = (close_prices * volumes).cumsum() / volumes.cumsum()

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
        MAX_PAR_SYMBOL = 100

        step = 1 if len(symbol_data) < 500 else 2

        for idx in range(50, len(symbol_data), step):
            detected_patterns = detect_all_patterns(symbol_data, idx)

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

            filtered_patterns = []
            for pattern_name in detected_patterns:
                config = all_patterns.get(pattern_name, {})
                pattern_bias = config.get('bias', 'Neutral')
                pattern_category = config.get('category', '')

                if market_regime != 'UNKNOWN':
                    if market_regime == 'BULL' and pattern_bias == 'Bearish' and pattern_category != 'Reversal':
                        continue
                    if market_regime == 'BEAR' and pattern_bias == 'Bullish' and pattern_category != 'Reversal':
                        continue

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

                    elliott_pattern_names = list(get_elliott_wave_patterns().keys())
                    is_elliott = pattern_name in elliott_pattern_names

                    if is_elliott:
                        text = generate_elliott_wave_data(symbol, use_row, pattern_name, config, indicator_values, metrics, var_idx, symbol_data, idx)
                    else:
                        text = generate_complete_pattern_data(symbol, use_row, pattern_name, config, indicator_values, metrics, var_idx, symbol_data, idx)

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
        if symbols_processed >= MAX_SYMBOLS:
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