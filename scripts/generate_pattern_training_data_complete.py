# scripts/generate_pattern_training_data_complete.py
# RSI Divergence, MACD, Stochastic, ATR, Bollinger Bands, OBV, Volume Profile সহ সম্পূর্ণ ট্রেনিং ডাটা
# 130+ প্যাটার্ন + Elliott Wave + SMC সম্পূর্ণ লাইব্রেরি + Multiple Historical Sequences + Noise Variations
# ✅ PRIORITY-BASED TRAINING - Elliott Wave (3.0x) | SMC (2.5x) | Divergence (2.0x) | Candlestick (1.5x)

import pandas as pd
import numpy as np
import os
import random
import json
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
np.seterr(divide='ignore', invalid='ignore')

# Optional ML imports
try:
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import cosine
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not installed. ML features disabled.")

# Optional DTW import
try:
    from fastdtw import fastdtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("⚠️ fastdtw not installed. Pattern similarity disabled.")

# =========================================================
# TRAINING CONFIGURATION - AUTO
# =========================================================

try:
    df_temp = pd.read_csv('./csv/mongodb.csv')
    MAX_SYMBOLS = df_temp['symbol'].nunique()
    print(f"✅ Auto-detected {MAX_SYMBOLS} symbols from mongodb.csv")
except:
    MAX_SYMBOLS = 396
    print(f"⚠️ Using fallback: {MAX_SYMBOLS} symbols")

TOTAL_PATTERNS = None
MIN_EXAMPLES_PER_PATTERN = 5
MAX_EXAMPLES_PER_PATTERN = 10
NUM_VARIATIONS = 4
MAX_PER_SYMBOL = None
MAX_EXAMPLES_PER_RUN = 100000

ELLIOTT_LOOKBACK = 300
SWING_WINDOW = 5
HT_SWING_WINDOW = 20

FIB_RETRACEMENT = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXTENSION = [1.272, 1.618, 2.0, 2.618, 4.236]

generated_patterns_tracker = defaultdict(lambda: defaultdict(int))
elliott_wave_tracker = defaultdict(list)
mistake_log = []
elliott_backtester = None

# =========================================================
# PRIORITY-BASED TRAINING CONFIGURATION
# =========================================================

BASE_MIN_EXAMPLES = 50
BASE_MAX_EXAMPLES = 100

PATTERN_PRIORITY = {
    'Impulse Wave': 3.0, 'Leading Diagonal': 3.0, 'Ending Diagonal': 3.0,
    '3rd Wave Extension': 3.0, '5th Wave Extension': 3.0, 'Single Zigzag': 3.0,
    'Double Zigzag': 3.0, 'Regular Flat': 3.0, 'Expanded Flat': 3.0,
    'Contracting Triangle': 3.0, 'Expanding Triangle': 3.0,
    'Break of Structure (BOS)': 2.5, 'Bullish Order Block': 2.5,
    'Bearish Order Block': 2.5, 'Fair Value Gap (FVG)': 2.5,
    'Optimal Trade Entry (OTE)': 2.5, 'Liquidity Sweep': 2.5,
    'Change of Character (CHoCH)': 2.5, 'Breaker Block': 2.5,
    'Mitigation Block': 2.5, 'Rejection Block': 2.5,
    'RSI Divergence': 2.0, 'MACD Divergence': 2.0, 'Hidden Divergence': 2.0,
    'Hammer': 1.5, 'Shooting Star': 1.5, 'Doji': 1.5, 'Engulfing': 1.5,
    'Bullish Engulfing': 1.5, 'Morning Star': 1.5, 'Evening Star': 1.5,
    'Three White Soldiers': 1.5, 'Three Black Crows': 1.5, 'Piercing Line': 1.5,
    'Cup and Handle': 1.0, 'Double Bottom': 1.0, 'Head and Shoulders': 1.0,
    'Bull Flag': 1.0, 'Ascending Triangle': 1.0, 'Descending Triangle': 1.0,
    'Symmetrical Triangle': 1.0, 'Rounding Bottom': 1.0,
    'Volume Climax': 1.2, 'Bollinger Band Squeeze': 1.2,
}
DEFAULT_PRIORITY = 1.0

def get_pattern_limits(pattern_name):
    multiplier = PATTERN_PRIORITY.get(pattern_name, DEFAULT_PRIORITY)
    return int(BASE_MIN_EXAMPLES * multiplier), int(BASE_MAX_EXAMPLES * multiplier), multiplier

def get_pattern_category(pattern_name):
    if pattern_name in ['Impulse Wave', 'Leading Diagonal', 'Ending Diagonal', 
                        '3rd Wave Extension', '5th Wave Extension', 'Single Zigzag',
                        'Double Zigzag', 'Regular Flat', 'Expanded Flat',
                        'Contracting Triangle', 'Expanding Triangle']:
        return 'Elliott Wave'
    elif pattern_name in ['Break of Structure (BOS)', 'Bullish Order Block', 
                          'Bearish Order Block', 'Fair Value Gap (FVG)', 
                          'Optimal Trade Entry (OTE)', 'Liquidity Sweep',
                          'Change of Character (CHoCH)', 'Breaker Block',
                          'Mitigation Block', 'Rejection Block']:
        return 'SMC'
    elif 'Divergence' in pattern_name:
        return 'Divergence'
    elif pattern_name in ['Hammer', 'Shooting Star', 'Doji', 'Engulfing',
                          'Bullish Engulfing', 'Morning Star', 'Evening Star',
                          'Three White Soldiers', 'Three Black Crows', 'Piercing Line']:
        return 'Candlestick'
    else:
        return 'Basic'

# =========================================================
# SAFETY HELPER FUNCTIONS
# =========================================================

def safe_divide(a, b, default=0.0):
    try:
        if b == 0 or np.isnan(a) or np.isnan(b): return default
        result = a / b
        if np.isnan(result) or np.isinf(result): return default
        return result
    except: return default

def safe_mean(arr, default=0.0):
    try:
        if arr is None or len(arr) == 0: return default
        arr = np.array(arr); arr = arr[~np.isnan(arr)]
        if len(arr) == 0: return default
        return float(np.mean(arr))
    except: return default

def safe_max(arr, default=0.0):
    try:
        if arr is None or len(arr) == 0: return default
        arr = np.array(arr); arr = arr[~np.isnan(arr)]
        if len(arr) == 0: return default
        return float(np.max(arr))
    except: return default

def safe_min(arr, default=0.0):
    try:
        if arr is None or len(arr) == 0: return default
        arr = np.array(arr); arr = arr[~np.isnan(arr)]
        if len(arr) == 0: return default
        return float(np.min(arr))
    except: return default

def clean_array(arr):
    try:
        arr = np.array(arr)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    except: return np.array([])

# =========================================================
# ELLIOTT WAVE BACKTESTER CLASS
# =========================================================

class ElliottWaveBacktester:
    def __init__(self):
        self.predictions = []
        self.results = []
        self.accuracy_log = []
        self.backtest_file = "./csv/elliott_backtest.json"
        self.load_history()

    def load_history(self):
        if os.path.exists(self.backtest_file):
            try:
                with open(self.backtest_file, 'r') as f:
                    data = json.load(f)
                    self.predictions = data.get('predictions', [])
                    self.results = data.get('results', [])
                    self.accuracy_log = data.get('accuracy_log', [])
            except: pass

    def save_history(self):
        os.makedirs("./csv", exist_ok=True)
        with open(self.backtest_file, 'w') as f:
            json.dump({'predictions': self.predictions[-1000:], 'results': self.results[-1000:], 'accuracy_log': self.accuracy_log[-100:]}, f, indent=2)

    def add_prediction(self, symbol, date, predicted_wave, predicted_target, current_price):
        pred_id = len(self.predictions)
        self.predictions.append({'id': pred_id, 'symbol': symbol, 'date': str(date)[:10], 'predicted_wave': predicted_wave, 'predicted_target': float(predicted_target) if not np.isnan(predicted_target) else float(current_price), 'current_price': float(current_price) if not np.isnan(current_price) else 100.0, 'timestamp': datetime.now().isoformat()})
        self.save_history()
        return pred_id

    def verify_prediction(self, pred_id, actual_future_price):
        if pred_id >= len(self.predictions): return None
        pred = self.predictions[pred_id]
        if actual_future_price <= 0 or np.isnan(actual_future_price): return None
        error = abs(pred['predicted_target'] - actual_future_price) / actual_future_price
        accurate = error < 0.05
        result = {'prediction_id': pred_id, 'symbol': pred['symbol'], 'date': pred['date'], 'accurate': accurate, 'error_percent': error * 100}
        self.results.append(result)
        total = len(self.results)
        accurate_count = sum(1 for r in self.results if r['accurate'])
        accuracy = safe_divide(accurate_count, total) * 100 if total > 0 else 0
        self.accuracy_log.append({'total_predictions': total, 'accuracy': accuracy, 'date': datetime.now().isoformat()})
        self.save_history()
        return result

    def get_performance_report(self):
        if not self.results: return "No predictions verified yet"
        total = len(self.results)
        accurate = sum(1 for r in self.results if r['accurate'])
        avg_error = safe_mean([r['error_percent'] for r in self.results], 0)
        recent_10 = self.results[-10:] if len(self.results) >= 10 else self.results
        recent_accuracy = sum(1 for r in recent_10 if r['accurate']) / len(recent_10) * 100 if recent_10 else 0
        return f"""
📊 ELLIOTT WAVE PERFORMANCE REPORT:
────────────────────────────────────────────────────────────────────────────────
Total Predictions: {total}
Accurate Predictions: {accurate}
Accuracy Rate: {accurate/total*100:.1f}%
Average Error: {avg_error:.2f}%
Recent Accuracy (Last {len(recent_10)}): {recent_accuracy:.1f}%
"""

# =========================================================
# SIMPLE BACKTESTER CLASS
# =========================================================

class SimpleBacktester:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.trades = []
        self.equity_curve = []

    def run_backtest(self, symbol_data, signals):
        capital = self.initial_capital
        position = 0
        for i, (idx, row) in enumerate(symbol_data.iterrows()):
            signal = signals[i] if i < len(signals) else 'HOLD'
            if signal == 'BUY' and position == 0:
                if row['close'] > 0 and not np.isnan(row['close']):
                    position = capital / row['close']
                    capital = 0
                    self.trades.append({'type': 'BUY', 'price': row['close'], 'date': row['date']})
            elif signal == 'SELL' and position > 0:
                if row['close'] > 0 and not np.isnan(row['close']):
                    capital = position * row['close']
                    position = 0
                    self.trades.append({'type': 'SELL', 'price': row['close'], 'date': row['date']})
            equity = capital + (position * row['close'] if position > 0 else 0)
            if not np.isnan(equity): self.equity_curve.append(equity)
        return self.get_performance()

    def get_performance(self):
        if len(self.equity_curve) == 0: return {'total_return': 0, 'total_trades': 0, 'win_rate': 0, 'equity_curve': []}
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital * 100
        winning_trades = sum(1 for i in range(1, len(self.trades), 2) if i < len(self.trades) and self.trades[i]['price'] > self.trades[i-1]['price'])
        total_trades = len(self.trades) // 2
        return {'total_return': total_return, 'total_trades': total_trades, 'win_rate': winning_trades / total_trades * 100 if total_trades > 0 else 0, 'equity_curve': self.equity_curve[-50:] if len(self.equity_curve) >= 50 else self.equity_curve}

# =========================================================
# PRICE ACTION TOOLS
# =========================================================

def calculate_trend_line(highs, lows, closes, lookback=50):
    highs, lows, closes = clean_array(highs), clean_array(lows), clean_array(closes)
    if len(highs) < lookback: return None
    recent_highs, recent_lows, recent_closes = highs[-lookback:], lows[-lookback:], closes[-lookback:]
    swing_highs, swing_lows, swing_high_indices, swing_low_indices = [], [], [], []
    for i in range(2, len(recent_highs) - 2):
        if i-2 >= 0 and i+3 <= len(recent_highs):
            if recent_highs[i] == max(recent_highs[i-2:i+3]):
                swing_highs.append(recent_highs[i]); swing_high_indices.append(i)
        if i-2 >= 0 and i+3 <= len(recent_lows):
            if recent_lows[i] == min(recent_lows[i-2:i+3]):
                swing_lows.append(recent_lows[i]); swing_low_indices.append(i)
    trend_lines = {'support_lines': [], 'resistance_lines': [], 'current_support': None, 'current_resistance': None}
    if len(swing_lows) >= 2:
        for i in range(len(swing_lows)-1):
            for j in range(i+1, len(swing_lows)):
                x1, y1 = swing_low_indices[i], swing_lows[i]
                x2, y2 = swing_low_indices[j], swing_lows[j]
                if x2 > x1 and x2 - x1 > 0:
                    slope = (y2 - y1) / (x2 - x1)
                    valid, touch_count = True, 0
                    for k in range(x1, min(x2+20, len(recent_lows))):
                        expected = slope * k + y1 - slope * x1
                        if recent_lows[k] < expected * 0.99: valid = False; break
                        if expected > 0 and abs(recent_lows[k] - expected) / expected < 0.01: touch_count += 1
                    if valid and touch_count >= 2:
                        current_level = slope * (len(recent_lows)-1) + y1 - slope * x1
                        trend_lines['support_lines'].append({'start_price': float(y1), 'end_price': float(y2), 'current_level': float(current_level), 'slope': float(slope), 'type': 'SUPPORT', 'strength': 'STRONG' if touch_count >= 3 else 'MODERATE', 'touches': touch_count})
    if len(swing_highs) >= 2:
        for i in range(len(swing_highs)-1):
            for j in range(i+1, len(swing_highs)):
                x1, y1 = swing_high_indices[i], swing_highs[i]
                x2, y2 = swing_high_indices[j], swing_highs[j]
                if x2 > x1 and x2 - x1 > 0:
                    slope = (y2 - y1) / (x2 - x1)
                    valid, touch_count = True, 0
                    for k in range(x1, min(x2+20, len(recent_highs))):
                        expected = slope * k + y1 - slope * x1
                        if recent_highs[k] > expected * 1.01: valid = False; break
                        if expected > 0 and abs(recent_highs[k] - expected) / expected < 0.01: touch_count += 1
                    if valid and touch_count >= 2:
                        current_level = slope * (len(recent_highs)-1) + y1 - slope * x1
                        trend_lines['resistance_lines'].append({'start_price': float(y1), 'end_price': float(y2), 'current_level': float(current_level), 'slope': float(slope), 'type': 'RESISTANCE', 'strength': 'STRONG' if touch_count >= 3 else 'MODERATE', 'touches': touch_count})
    current_price = float(recent_closes[-1]) if len(recent_closes) > 0 else 100.0
    if trend_lines['support_lines']:
        supports_below = [s for s in trend_lines['support_lines'] if s['current_level'] < current_price]
        if supports_below: trend_lines['current_support'] = max(supports_below, key=lambda x: x['current_level'])
    if trend_lines['resistance_lines']:
        resistances_above = [r for r in trend_lines['resistance_lines'] if r['current_level'] > current_price]
        if resistances_above: trend_lines['current_resistance'] = min(resistances_above, key=lambda x: x['current_level'])
    return trend_lines

def calculate_parallel_channel(trend_lines, current_price):
    if not trend_lines or current_price <= 0 or np.isnan(current_price): return None
    channels = []
    if trend_lines.get('current_support'):
        s = trend_lines['current_support']
        support_level = s.get('current_level', current_price * 0.95)
        if support_level <= 0: support_level = current_price * 0.95
        channel_top = current_price * 1.05
        channels.append({'type': 'ASCENDING' if s.get('slope', 0) > 0 else 'DESCENDING' if s.get('slope', 0) < 0 else 'HORIZONTAL', 'support_line': support_level, 'support_slope': s.get('slope', 0), 'mid_line': support_level + (channel_top - support_level) / 2, 'channel_top': channel_top, 'width': channel_top - support_level, 'width_percent': safe_divide(channel_top - support_level, support_level) * 100, 'position': 'UPPER' if current_price > support_level + (channel_top - support_level) * 0.7 else 'LOWER' if current_price < support_level + (channel_top - support_level) * 0.3 else 'MIDDLE'})
    if trend_lines.get('current_resistance'):
        r = trend_lines['current_resistance']
        resistance_level = r.get('current_level', current_price * 1.05)
        if resistance_level <= 0: resistance_level = current_price * 1.05
        channel_bottom = current_price * 0.95
        channels.append({'type': 'ASCENDING' if r.get('slope', 0) > 0 else 'DESCENDING' if r.get('slope', 0) < 0 else 'HORIZONTAL', 'resistance_line': resistance_level, 'resistance_slope': r.get('slope', 0), 'mid_line': channel_bottom + (resistance_level - channel_bottom) / 2, 'channel_bottom': channel_bottom, 'width': resistance_level - channel_bottom, 'width_percent': safe_divide(resistance_level - channel_bottom, channel_bottom) * 100, 'position': 'UPPER' if current_price > channel_bottom + (resistance_level - channel_bottom) * 0.7 else 'LOWER' if current_price < channel_bottom + (resistance_level - channel_bottom) * 0.3 else 'MIDDLE'})
    return channels if channels else None

def calculate_ray_lines(symbol_data, lookback=100):
    if symbol_data is None: return None
    closes, highs, lows = clean_array(symbol_data['close'].values), clean_array(symbol_data['high'].values), clean_array(symbol_data['low'].values)
    if len(closes) < lookback: return None
    recent_highs, recent_lows = highs[-lookback:], lows[-lookback:]
    current_price = closes[-1] if len(closes) > 0 else 100.0
    rays, pivot_highs, pivot_lows = [], [], []
    for i in range(5, len(recent_highs)-5):
        if i-5 >= 0 and i+6 <= len(recent_highs):
            if recent_highs[i] == max(recent_highs[i-5:i+6]): pivot_highs.append((i, recent_highs[i]))
    for i in range(5, len(recent_lows)-5):
        if i-5 >= 0 and i+6 <= len(recent_lows):
            if recent_lows[i] == min(recent_lows[i-5:i+6]): pivot_lows.append((i, recent_lows[i]))
    if len(pivot_lows) >= 2:
        for i in range(len(pivot_lows)-1):
            for j in range(i+1, len(pivot_lows)):
                if pivot_lows[j][1] > pivot_lows[i][1]:
                    x1, y1 = pivot_lows[i]; x2, y2 = pivot_lows[j]
                    if x2 > x1:
                        slope = (y2 - y1) / (x2 - x1)
                        projections = {}
                        for step in [5, 10, 20, 50]:
                            future_x = len(recent_lows) + step
                            projections[f'{step}_days'] = float(slope * (future_x - x1) + y1)
                        rays.append({'type': 'BULLISH_RAY', 'start_price': float(y1), 'current_support': float(slope * (len(recent_lows)-1 - x1) + y1), 'slope': float(slope), 'projections': projections, 'angle': float(np.degrees(np.arctan(slope)))})
    if len(pivot_highs) >= 2:
        for i in range(len(pivot_highs)-1):
            for j in range(i+1, len(pivot_highs)):
                if pivot_highs[j][1] < pivot_highs[i][1]:
                    x1, y1 = pivot_highs[i]; x2, y2 = pivot_highs[j]
                    if x2 > x1:
                        slope = (y2 - y1) / (x2 - x1)
                        projections = {}
                        for step in [5, 10, 20, 50]:
                            future_x = len(recent_highs) + step
                            projections[f'{step}_days'] = float(slope * (future_x - x1) + y1)
                        rays.append({'type': 'BEARISH_RAY', 'start_price': float(y1), 'current_resistance': float(slope * (len(recent_highs)-1 - x1) + y1), 'slope': float(slope), 'projections': projections, 'angle': float(np.degrees(np.arctan(slope)))})
    return rays if rays else None

def calculate_trend_based_fib_extension(symbol_data, lookback=100):
    if symbol_data is None: return None
    closes, highs, lows = clean_array(symbol_data['close'].values), clean_array(symbol_data['high'].values), clean_array(symbol_data['low'].values)
    if len(closes) < lookback: return None
    recent_lows, recent_highs = lows[-lookback:], highs[-lookback:]
    swing_low, swing_high = safe_min(recent_lows, 0), safe_max(recent_highs, 0)
    if swing_low <= 0 or swing_high <= 0: return None
    swing_low_idx = np.argmin(recent_lows) if len(recent_lows) > 0 else 0
    swing_high_idx = np.argmax(recent_highs) if len(recent_highs) > 0 else 0
    current_price = closes[-1] if len(closes) > 0 else 0.0
    is_uptrend = swing_high_idx > swing_low_idx
    fib_levels = {'trend': 'UPTREND' if is_uptrend else 'DOWNTREND', 'swing_low': float(swing_low), 'swing_high': float(swing_high), 'retracement': {}, 'extension': {}}
    range_size = swing_high - swing_low
    if range_size <= 0: return None
    if is_uptrend:
        for fib in FIB_RETRACEMENT: fib_levels['retracement'][f'{fib:.3f}'] = float(swing_high - range_size * fib)
        for fib in FIB_EXTENSION: fib_levels['extension'][f'{fib:.3f}'] = float(swing_low + range_size * fib)
    else:
        for fib in FIB_RETRACEMENT: fib_levels['retracement'][f'{fib:.3f}'] = float(swing_low + range_size * fib)
        for fib in FIB_EXTENSION: fib_levels['extension'][f'{fib:.3f}'] = float(swing_high - range_size * fib)
    current_zone = None
    for level_name, price in fib_levels['retracement'].items():
        if is_uptrend:
            if current_price <= price: current_zone = f'Below {level_name} retracement'; break
        else:
            if current_price >= price: current_zone = f'Above {level_name} retracement'; break
    fib_levels['current_zone'] = current_zone if current_zone else 'Between retracement levels'
    return fib_levels

def calculate_fixed_range_volume_profile(symbol_data, lookback=100, num_bins=20):
    if symbol_data is None or len(symbol_data) < lookback: return None
    closes, volumes, highs, lows = clean_array(symbol_data['close'].values), clean_array(symbol_data['volume'].values), clean_array(symbol_data['high'].values), clean_array(symbol_data['low'].values)
    if len(closes) < lookback: return None
    recent_closes, recent_volumes = closes[-lookback:], volumes[-lookback:]
    recent_highs, recent_lows = highs[-lookback:], lows[-lookback:]
    valid_mask = ~(np.isnan(recent_closes) | np.isnan(recent_volumes) | np.isnan(recent_highs) | np.isnan(recent_lows))
    if np.sum(valid_mask) < 10: return None
    recent_closes, recent_volumes = recent_closes[valid_mask], recent_volumes[valid_mask]
    recent_highs, recent_lows = recent_highs[valid_mask], recent_lows[valid_mask]
    range_high, range_low = safe_max(recent_highs, 0), safe_min(recent_lows, 0)
    range_size = range_high - range_low
    if range_size <= 0: return None
    bin_size = range_size / num_bins
    if bin_size <= 0: return None
    volume_by_price = defaultdict(float)
    for i in range(len(recent_closes)):
        price, vol = recent_closes[i], recent_volumes[i]
        if price <= 0 or vol <= 0: continue
        bin_idx = int((price - range_low) / bin_size)
        bin_idx = max(0, min(num_bins-1, bin_idx))
        bin_center = range_low + (bin_idx + 0.5) * bin_size
        volume_by_price[float(bin_center)] += float(vol)
    if not volume_by_price: return None
    poc = max(volume_by_price, key=volume_by_price.get) if volume_by_price else None
    total_volume = sum(volume_by_price.values())
    if total_volume <= 0: return None
    sorted_bins = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
    cumulative_vol, value_area_bins = 0, []
    for price, vol in sorted_bins:
        cumulative_vol += vol; value_area_bins.append(price)
        if cumulative_vol >= total_volume * 0.7: break
    vah = max(value_area_bins) if value_area_bins else range_high
    val = min(value_area_bins) if value_area_bins else range_low
    current_price = float(recent_closes[-1]) if len(recent_closes) > 0 else 0.0
    return {'range_high': float(range_high), 'range_low': float(range_low), 'poc': float(poc) if poc else None, 'value_area_high': float(vah), 'value_area_low': float(val), 'current_position': 'ABOVE_VA' if current_price > vah else 'BELOW_VA' if current_price < val else 'IN_VA', 'volume_profile': dict(sorted(volume_by_price.items())), 'total_volume': float(total_volume)}

def predict_price_from_trend_lines(trend_lines, current_price, days_forward=10):
    if not trend_lines or current_price <= 0: return None
    predictions = []
    if trend_lines.get('current_support'):
        s = trend_lines['current_support']; level = s.get('current_level', current_price)
        if level > 0 and current_price < level * 1.02:
            predictions.append({'type': 'BOUNCE_FROM_SUPPORT', 'expected_direction': 'UP', 'target': float(level * 1.05), 'invalidation': float(level * 0.98), 'confidence': 'HIGH' if s.get('strength') == 'STRONG' else 'MEDIUM'})
    if trend_lines.get('current_resistance'):
        r = trend_lines['current_resistance']; level = r.get('current_level', current_price)
        if level > 0 and current_price > level * 0.98:
            predictions.append({'type': 'REJECTION_FROM_RESISTANCE', 'expected_direction': 'DOWN', 'target': float(level * 0.95), 'invalidation': float(level * 1.02), 'confidence': 'HIGH' if r.get('strength') == 'STRONG' else 'MEDIUM'})
        if level > 0 and current_price > level * 1.01:
            predictions.append({'type': 'BULLISH_BREAKOUT', 'expected_direction': 'UP', 'target': float(current_price * 1.05), 'invalidation': float(level), 'confidence': 'HIGH'})
    if trend_lines.get('current_support'):
        s = trend_lines['current_support']; level = s.get('current_level', current_price)
        if level > 0 and current_price < level * 0.99:
            predictions.append({'type': 'BEARISH_BREAKDOWN', 'expected_direction': 'DOWN', 'target': float(current_price * 0.95), 'invalidation': float(level), 'confidence': 'HIGH'})
    return predictions if predictions else None

def analyze_price_action_complete(symbol_data, idx, lookback=100):
    if symbol_data is None or idx is None: return "Insufficient data.", None, None, None, None, None, None
    if idx < 0 or idx >= len(symbol_data): return "Invalid index.", None, None, None, None, None, None
    closes, highs, lows = clean_array(symbol_data['close'].values), clean_array(symbol_data['high'].values), clean_array(symbol_data['low'].values)
    if len(closes) == 0: return "Insufficient data.", None, None, None, None, None, None
    current_price = float(closes[-1]) if len(closes) > 0 else 100.0
    if len(closes) < 50: return "Insufficient data.", None, None, None, None, None, None
    trend_lines = calculate_trend_line(highs, lows, closes, min(lookback, len(closes)))
    channels = calculate_parallel_channel(trend_lines, current_price) if trend_lines else None
    rays = calculate_ray_lines(symbol_data, min(lookback, len(closes)))
    fib_ext = calculate_trend_based_fib_extension(symbol_data, min(lookback, len(closes)))
    try: vol_profile = calculate_fixed_range_volume_profile(symbol_data, min(lookback, len(closes)))
    except: vol_profile = None
    predictions = predict_price_from_trend_lines(trend_lines, current_price) if trend_lines else None
    report = "\n📐 PRICE ACTION COMPLETE ANALYSIS:\n" + "="*80 + "\n"
    if trend_lines:
        report += "\n📏 TREND LINES:\n" + "─"*80 + "\n"
        if trend_lines.get('current_support'):
            s = trend_lines['current_support']
            report += f"\n🔹 SUPPORT: Level={s['current_level']:.2f} Slope={s['slope']:.4f} Strength={s['strength']}\n"
        if trend_lines.get('current_resistance'):
            r = trend_lines['current_resistance']
            report += f"\n🔸 RESISTANCE: Level={r['current_level']:.2f} Slope={r['slope']:.4f} Strength={r['strength']}\n"
    if channels:
        report += "\n📊 PARALLEL CHANNELS:\n" + "─"*80 + "\n"
        for i, ch in enumerate(channels[:2]):
            report += f"\n🔹 CHANNEL {i+1}: {ch['type']} | Support: {ch.get('support_line', ch.get('channel_bottom', 0)):.2f} | Resistance: {ch.get('channel_top', ch.get('resistance_line', 0)):.2f}\n"
    if rays:
        report += "\n🔆 RAY LINES:\n" + "─"*80 + "\n"
        for ray in rays[:3]:
            report += f"\n🔹 {ray['type']}: Level={ray.get('current_support', ray.get('current_resistance', 0)):.2f} Angle={ray['angle']:.1f}°\n"
    if fib_ext:
        report += f"\n📐 FIBONACCI: {fib_ext['trend']} | Swing: {fib_ext['swing_low']:.2f}-{fib_ext['swing_high']:.2f} | Zone: {fib_ext['current_zone']}\n"
    if vol_profile:
        report += f"\n📊 VOLUME PROFILE: Range={vol_profile['range_low']:.2f}-{vol_profile['range_high']:.2f} | POC={vol_profile['poc']:.2f if vol_profile['poc'] else 'N/A'} | Position={vol_profile['current_position']}\n"
    if predictions:
        report += "\n🎯 PREDICTIONS:\n" + "─"*80 + "\n"
        for pred in predictions:
            report += f"\n🔹 {pred['type']}: {pred['expected_direction']} | Target={pred['target']:.2f} | Confidence={pred['confidence']}\n"
    report += "\n" + "="*80 + "\n"
    return report, trend_lines, channels, rays, fib_ext, vol_profile, predictions

# =========================================================
# INDICATOR CALCULATIONS
# =========================================================

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1: return pd.Series([50] * len(prices), index=prices.index)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_rsi_series(prices, period=14):
    if len(prices) < period + 1: return np.full(len(prices), 50.0)
    prices = clean_array(prices)
    deltas = np.diff(prices)
    if len(deltas) < period: return np.full(len(prices), 50.0)
    seed = deltas[:period+1]
    up = np.sum(seed[seed >= 0]) / period if len(seed[seed >= 0]) > 0 else 0
    down = -np.sum(seed[seed < 0]) / period if len(seed[seed < 0]) > 0 else 0
    if down == 0: rs = 100.0
    else: rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs) if rs != float('inf') else 100.0
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0: upval, downval = delta, 0
        else: upval, downval = 0, -delta
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        if down == 0: rsi[i] = 100.0
        else:
            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
    return np.clip(rsi, 0, 100)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal: return pd.Series([0]*len(prices)), pd.Series([0]*len(prices)), pd.Series([0]*len(prices))
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.fillna(0), signal_line.fillna(0), histogram.fillna(0)

def calculate_macd_series(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal: return np.zeros(len(prices)), np.zeros(len(prices)), np.zeros(len(prices))
    prices = clean_array(prices)
    exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean().values
    exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean().values
    macd_line = exp1 - exp2
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    macd_line = np.nan_to_num(macd_line, nan=0.0)
    signal_line = np.nan_to_num(signal_line, nan=0.0)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()
    denominator = (high_max - low_min).replace(0, np.nan)
    k = 100 * ((close - low_min) / denominator)
    k = k.fillna(50)
    d = k.rolling(window=d_period).mean().fillna(50)
    return k, d

def calculate_obv(close, volume):
    close_vals = clean_array(close.values)
    volume_vals = clean_array(volume.values)
    obv = [0.0]
    for i in range(1, len(close_vals)):
        if close_vals[i] > close_vals[i-1]: obv.append(obv[-1] + volume_vals[i])
        elif close_vals[i] < close_vals[i-1]: obv.append(obv[-1] - volume_vals[i])
        else: obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

def calculate_ema(prices, period=20): return prices.ewm(span=period, adjust=False).mean()
def calculate_sma(prices, period=20): return prices.rolling(window=period).mean()

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    sma = sma.fillna(method='bfill').fillna(prices.iloc[0] if len(prices) > 0 else 0)
    std = std.fillna(0)
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_atr(high, low, close, period=14):
    tr = pd.DataFrame({'hl': high - low, 'hc': (high - close.shift()).abs(), 'lc': (low - close.shift()).abs()}).max(axis=1)
    return tr.rolling(window=period).mean().fillna(method='bfill').fillna(0)

def calculate_atr_series(highs, lows, closes, period=14):
    highs, lows, closes = clean_array(highs), clean_array(lows), clean_array(closes)
    if len(highs) < 2: return np.full(len(highs), 0.01)
    atr = []
    for i in range(len(highs)):
        if i < 1: tr = highs[i] - lows[i]; atr.append(max(tr, 0.01))
        else: tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])); atr.append(max(tr, 0.01))
    atr_series = []
    for i in range(len(atr)):
        if i < period: atr_series.append(safe_mean(atr[:i+1], 0.01))
        else: atr_series.append(safe_mean(atr[i-period+1:i+1], 0.01))
    return np.array(atr_series)

# =========================================================
# DIVERGENCE DETECTION
# =========================================================

def detect_rsi_divergence(prices, rsi_values):
    if len(prices) < 20 or len(rsi_values) < 20: return 'None'
    prices_array = clean_array(prices)
    rsi_array = clean_array(rsi_values)
    if len(prices_array) > 30: prices_array, rsi_array = prices_array[-30:], rsi_array[-30:]
    window = len(prices_array)
    half = window // 2
    if half == 0: return 'None'
    first_half_prices = prices_array[:half]
    second_half_prices = prices_array[half:]
    first_half_rsi = rsi_array[:half]
    second_half_rsi = rsi_array[half:]
    if len(first_half_prices) > 0 and len(second_half_prices) > 0:
        first_min_idx = np.argmin(first_half_prices)
        second_min_idx = np.argmin(second_half_prices)
        if second_half_prices[second_min_idx] < first_half_prices[first_min_idx] and second_half_rsi[second_min_idx] > first_half_rsi[first_min_idx]:
            return 'Bullish'
    if len(first_half_prices) > 0 and len(second_half_prices) > 0:
        first_max_idx = np.argmax(first_half_prices)
        second_max_idx = np.argmax(second_half_prices)
        if second_half_prices[second_max_idx] > first_half_prices[first_max_idx] and second_half_rsi[second_max_idx] < first_half_rsi[first_max_idx]:
            return 'Bearish'
    return 'None'

def detect_macd_divergence(prices, macd_line):
    if len(prices) < 20 or len(macd_line) < 20: return 'None'
    prices_array = clean_array(prices)
    macd_array = clean_array(macd_line)
    if len(prices_array) > 30: prices_array, macd_array = prices_array[-30:], macd_array[-30:]
    window = len(prices_array)
    half = window // 2
    if half == 0: return 'None'
    first_half_prices = prices_array[:half]
    second_half_prices = prices_array[half:]
    first_half_macd = macd_array[:half]
    second_half_macd = macd_array[half:]
    if len(first_half_prices) > 0 and len(second_half_prices) > 0:
        first_min_idx = np.argmin(first_half_prices)
        second_min_idx = np.argmin(second_half_prices)
        if second_half_prices[second_min_idx] < first_half_prices[first_min_idx] and second_half_macd[second_min_idx] > first_half_macd[first_min_idx]:
            return 'Bullish'
    if len(first_half_prices) > 0 and len(second_half_prices) > 0:
        first_max_idx = np.argmax(first_half_prices)
        second_max_idx = np.argmax(second_half_prices)
        if second_half_prices[second_max_idx] > first_half_prices[first_max_idx] and second_half_macd[second_max_idx] < first_half_macd[first_max_idx]:
            return 'Bearish'
    return 'None'

# =========================================================
# PATTERN METRICS & NOISE
# =========================================================

def calculate_pattern_metrics(prices, pattern_high, pattern_low, current_price):
    if len(prices) < 20: return {}
    prices = clean_array(prices)
    recent_range = safe_max(prices[-20:], 0) - safe_min(prices[-20:], 0)
    if pattern_low <= 0: pattern_low = current_price * 0.95
    if pattern_high <= 0: pattern_high = current_price * 1.05
    pattern_height = pattern_high - pattern_low
    pattern_depth = safe_divide(pattern_height, pattern_low) * 100
    breakout_distance = safe_divide(current_price - pattern_high, pattern_high) * 100
    relative_strength = safe_divide(pattern_height, recent_range, 0)
    return {'pattern_height': round(float(pattern_height), 2), 'pattern_depth_percent': round(float(pattern_depth), 2), 'breakout_distance_percent': round(float(breakout_distance), 2), 'relative_strength': round(float(relative_strength), 3), 'recent_range': round(float(recent_range), 2)}

def add_noise_to_sequence(sequence, noise_level=0.005):
    sequence = clean_array(sequence)
    if len(sequence) == 0: return sequence
    trend = np.linspace(0, random.uniform(-0.01, 0.01), len(sequence))
    noise = np.random.normal(0, noise_level, len(sequence)) + trend
    noise = np.clip(noise, -0.03, 0.03)
    noisy_sequence = sequence + (sequence * noise)
    noisy_sequence = np.maximum(noisy_sequence, 0.01)
    if sequence[-1] > sequence[0]: noisy_sequence = np.sort(noisy_sequence)
    else: noisy_sequence = np.sort(noisy_sequence)[::-1]
    return noisy_sequence

def detect_market_regime(close_prices):
    if len(close_prices) < 50: return 'UNKNOWN'
    momentum = safe_mean(close_prices.iloc[-50:].pct_change(20).dropna(), 0)
    if len(close_prices) < 200:
        sma20 = close_prices.rolling(20).mean()
        if len(sma20) > 0 and len(close_prices) > 0:
            if close_prices.iloc[-1] > sma20.iloc[-1] and momentum > 0: return 'BULL'
            elif close_prices.iloc[-1] < sma20.iloc[-1] and momentum < 0: return 'BEAR'
        return 'UNKNOWN'
    sma50 = close_prices.rolling(50).mean()
    sma200 = close_prices.rolling(200).mean()
    if len(sma50) > 0 and len(sma200) > 0:
        if sma50.iloc[-1] > sma200.iloc[-1] and momentum > 0: return 'BULL'
        elif sma50.iloc[-1] < sma200.iloc[-1] and momentum < 0: return 'BEAR'
    return 'SIDEWAYS'

# =========================================================
# SWING POINTS & MARKET STRUCTURE
# =========================================================

def find_swing_points(highs, lows, window=5):
    highs, lows = clean_array(highs), clean_array(lows)
    swing_highs, swing_lows = [], []
    for i in range(window, len(highs) - window):
        if i-window >= 0 and i+window+1 <= len(highs):
            if highs[i] == max(highs[i-window:i+window+1]): swing_highs.append(highs[i])
        if i-window >= 0 and i+window+1 <= len(lows):
            if lows[i] == min(lows[i-window:i+window+1]): swing_lows.append(lows[i])
    return swing_highs, swing_lows

def find_swing_points_with_indices(highs, lows, window=5):
    highs, lows = clean_array(highs), clean_array(lows)
    if len(highs) < window * 2 + 1: return [], [], [], []
    swing_highs, swing_lows, swing_high_indices, swing_low_indices = [], [], [], []
    for i in range(window, len(highs) - window):
        if i-window >= 0 and i+window+1 <= len(highs):
            if highs[i] == max(highs[i-window:i+window+1]):
                swing_highs.append(highs[i]); swing_high_indices.append(i)
        if i-window >= 0 and i+window+1 <= len(lows):
            if lows[i] == min(lows[i-window:i+window+1]):
                swing_lows.append(lows[i]); swing_low_indices.append(i)
    return swing_highs, swing_lows, swing_high_indices, swing_low_indices

def detect_bos_from_swings(swing_highs, swing_lows, current_price):
    if len(swing_highs) >= 2 and current_price > swing_highs[-2]: return True, "BULLISH BOS"
    elif len(swing_lows) >= 2 and current_price < swing_lows[-2]: return True, "BEARISH BOS"
    return False, None

def find_support_resistance(highs, lows, closes, tolerance=0.02):
    highs, lows, closes = clean_array(highs), clean_array(lows), clean_array(closes)
    all_levels = list(highs) + list(lows)
    levels = {}
    for price in all_levels:
        if price <= 0: continue
        found = False
        for key in list(levels.keys()):
            if abs(price - key) / key < tolerance: levels[key] += 1; found = True; break
        if not found: levels[price] = 1
    sorted_levels = sorted(levels.items(), key=lambda x: x[1], reverse=True)
    current_price = closes[-1] if len(closes) > 0 else 0.0
    resistance = [price for price, count in sorted_levels if price > current_price and count >= 2]
    support = [price for price, count in sorted_levels if price < current_price and count >= 2]
    return {'resistance': sorted(resistance, reverse=True), 'support': sorted(support)}

def count_touches(prices, level, tolerance=0.015):
    prices = clean_array(prices)
    if level <= 0: return 0
    return sum(1 for p in prices if p > 0 and abs(p - level) / level < tolerance)

# =========================================================
# ADVANCED PRICE SEQUENCE GENERATOR
# =========================================================

def generate_advanced_price_sequence(symbol_data, idx, lookback=150):
    if symbol_data is None or idx is None: return "Insufficient data.", False, False, False
    start_idx = max(0, idx - lookback)
    sequence_data = symbol_data.iloc[start_idx:idx+1].copy()
    if len(sequence_data) < 50: return "Insufficient data.", False, False, False
    closes = clean_array(sequence_data['close'].values)
    highs = clean_array(sequence_data['high'].values)
    lows = clean_array(sequence_data['low'].values)
    volumes = clean_array(sequence_data['volume'].values)
    current_price = closes[-1] if len(closes) > 0 else 100.0
    text = "📊 COMPREHENSIVE PRICE ANALYSIS (150+ Candles):\n" + "="*80 + "\n\n"
    text += "📋 PRICE DATA (Last 30 candles):\n" + "─"*80 + "\n"
    text += "Date       | Open   | High   | Low    | Close  | Volume   | Range\n"
    recent_data = sequence_data.iloc[-30:]
    for _, row in recent_data.iterrows():
        date_str = str(row['date'])[:10]
        range_val = row['high'] - row['low']
        text += f"{date_str} | {row['open']:7.2f} | {row['high']:7.2f} | {row['low']:7.2f} | {row['close']:7.2f} | {int(row['volume']):8,} | {range_val:.2f}\n"
    older_data = sequence_data.iloc[:-30] if len(sequence_data) > 30 else sequence_data.iloc[:0]
    if len(older_data) > 0:
        text += f"\n📊 Previous {len(older_data)} candles summary:\n"
        text += f"   High: {older_data['high'].max():.2f} | Low: {older_data['low'].min():.2f}\n"
        text += f"   Avg Close: {older_data['close'].mean():.2f} | Avg Volume: {older_data['volume'].mean():,.0f}\n"
    text += "\n📈 TREND ANALYSIS:\n" + "─"*80 + "\n"
    sma20 = safe_mean(closes[-20:], current_price)
    sma50 = safe_mean(closes[-50:], sma20) if len(closes) >= 50 else sma20
    sma150 = safe_mean(closes, sma50)
    text += f"Short-term Trend (20):  {'BULLISH 📈' if current_price > sma20 else 'BEARISH 📉'}\n"
    text += f"Medium-term Trend (50): {'BULLISH 📈' if current_price > sma50 else 'BEARISH 📉'}\n"
    text += f"Long-term Trend (150):  {'BULLISH 📈' if current_price > sma150 else 'BEARISH 📉'}\n"
    price_change_20 = safe_divide(closes[-1] - closes[-20], closes[-20]) * 100 if len(closes) >= 20 and closes[-20] > 0 else 0
    price_change_50 = safe_divide(closes[-1] - closes[-50], closes[-50]) * 100 if len(closes) >= 50 and closes[-50] > 0 else price_change_20
    price_change_150 = safe_divide(closes[-1] - closes[0], closes[0]) * 100 if len(closes) > 0 and closes[0] > 0 else 0
    text += f"\nPrice Change (20d):  {price_change_20:+.2f}%\nPrice Change (50d):  {price_change_50:+.2f}%\nPrice Change (150d): {price_change_150:+.2f}%\n"
    text += "\n🏗️ MARKET STRUCTURE (SMC):\n" + "─"*80 + "\n"
    swing_highs, swing_lows = find_swing_points(highs, lows)
    recent_sh = swing_highs[-5:] if len(swing_highs) >= 5 else swing_highs
    recent_sl = swing_lows[-5:] if len(swing_lows) >= 5 else swing_lows
    text += f"Swing Highs: {', '.join([f'{h:.2f}' for h in recent_sh])}\n"
    text += f"Swing Lows:  {', '.join([f'{l:.2f}' for l in recent_sl])}\n"
    if len(swing_highs) >= 3:
        if swing_highs[-1] > swing_highs[-2] and swing_highs[-2] > swing_highs[-3]: text += "Structure: HIGHER HIGHS 📈\n"
        elif swing_highs[-1] < swing_highs[-2] and swing_highs[-2] < swing_highs[-3]: text += "Structure: LOWER HIGHS 📉\n"
        else: text += "Structure: CHOPPY ↔️\n"
    if len(swing_lows) >= 3:
        if swing_lows[-1] > swing_lows[-2] and swing_lows[-2] > swing_lows[-3]: text += "Structure: HIGHER LOWS ✅\n"
        elif swing_lows[-1] < swing_lows[-2] and swing_lows[-2] < swing_lows[-3]: text += "Structure: LOWER LOWS ❌\n"
    bos_detected, bos_type = detect_bos_from_swings(swing_highs, swing_lows, current_price)
    if bos_detected: text += f"\n⚡ {bos_type} DETECTED!\n"
    text += "\n🎯 SUPPORT & RESISTANCE:\n" + "─"*80 + "\n"
    levels = find_support_resistance(highs, lows, closes)
    text += "Resistance:\n"
    for level in levels['resistance'][:3]:
        touches = count_touches(highs, level)
        text += f"  R{level:.2f} ({touches} touches) - {'STRONG' if touches >= 3 else 'WEAK'}\n"
    text += "\nSupport:\n"
    for level in levels['support'][:3]:
        touches = count_touches(lows, level)
        text += f"  S{level:.2f} ({touches} touches) - {'STRONG' if touches >= 3 else 'WEAK'}\n"
    nearest_resistance = min([r for r in levels['resistance'] if r > current_price], default=current_price * 1.1)
    nearest_support = max([s for s in levels['support'] if s < current_price], default=current_price * 0.9)
    text += f"\nCurrent Price: {current_price:.2f}\n"
    text += f"Nearest Resistance: {nearest_resistance:.2f}\nNearest Support: {nearest_support:.2f}\n"
    text += "\n📊 VOLUME ANALYSIS:\n" + "─"*80 + "\n"
    avg_volume_20 = safe_mean(volumes[-20:], 0) if len(volumes) >= 20 else safe_mean(volumes, 0)
    avg_volume_50 = safe_mean(volumes[-50:], avg_volume_20) if len(volumes) >= 50 else avg_volume_20
    current_volume = volumes[-1] if len(volumes) > 0 else 0
    text += f"Current Volume: {current_volume:,.0f}\nAvg Volume (20): {avg_volume_20:,.0f}\nAvg Volume (50): {avg_volume_50:,.0f}\n"
    volume_spikes = []
    for i in range(20, len(volumes)):
        avg_prev = safe_mean(volumes[i-20:i], 1)
        if avg_prev > 0 and volumes[i] > avg_prev * 1.5: volume_spikes.append((i, volumes[i]))
    has_volume_spike = len(volume_spikes) > 0
    text += "\n📉 VOLATILITY ANALYSIS:\n" + "─"*80 + "\n"
    atr_values = calculate_atr_series(highs, lows, closes)
    current_atr = atr_values[-1] if len(atr_values) > 0 else current_price * 0.02
    avg_atr = safe_mean(atr_values, current_atr)
    text += f"Current ATR: {current_atr:.2f}\nAverage ATR: {avg_atr:.2f}\n"
    high_volatility = current_atr > avg_atr * 1.2
    return text, bos_detected, has_volume_spike, high_volatility

# =========================================================
# WYCKOFF CYCLE & SECTOR ANALYSIS
# =========================================================

def detect_volume_price_cycle(symbol_data, idx, lookback=150):
    if symbol_data is None or idx is None: return "Insufficient data.", {}
    start_idx = max(0, idx - lookback)
    sequence_data = symbol_data.iloc[start_idx:idx+1].copy()
    if len(sequence_data) < 60: return "Insufficient data.", {}
    closes = clean_array(sequence_data['close'].values)
    volumes = clean_array(sequence_data['volume'].values)
    vol_10 = safe_mean(volumes[-10:], 0) if len(volumes) >= 10 else safe_mean(volumes, 0)
    vol_20 = safe_mean(volumes[-20:], vol_10) if len(volumes) >= 20 else vol_10
    vol_50 = safe_mean(volumes[-50:], vol_20) if len(volumes) >= 50 else vol_20
    vol_50 = max(vol_50, 1)
    current_volume = volumes[-1] if len(volumes) > 0 else 0
    current_price = closes[-1] if len(closes) > 0 else 100.0
    volume_trend = "INCREASING" if vol_10 > vol_20 > vol_50 else "DECREASING" if vol_10 < vol_20 < vol_50 else "NEUTRAL"
    volume_ratio = safe_divide(current_volume, vol_50, 1)
    is_volume_spike = volume_ratio >= 1.5
    price_change_5 = safe_divide(closes[-1] - closes[-5], closes[-5]) * 100 if len(closes) >= 5 and closes[-5] > 0 else 0
    price_change_10 = safe_divide(closes[-1] - closes[-10], closes[-10]) * 100 if len(closes) >= 10 and closes[-10] > 0 else 0
    price_change_20 = safe_divide(closes[-1] - closes[-20], closes[-20]) * 100 if len(closes) >= 20 and closes[-20] > 0 else 0
    phase = "UNKNOWN"
    confidence_boost = 0
    if volume_trend == "DECREASING" and abs(price_change_20) < 10: phase = "ACCUMULATION"; confidence_boost = 5
    elif is_volume_spike and price_change_5 > 0: phase = "MARKUP"; confidence_boost = 20
    elif is_volume_spike and price_change_10 < 5: phase = "DISTRIBUTION"; confidence_boost = -5
    elif price_change_20 < -5: phase = "MARKDOWN"; confidence_boost = -10
    will_breakout_soon = phase in ["ACCUMULATION", "MARKUP"]
    breakout_confidence = 65 if will_breakout_soon else 30
    analysis_text = f"""
📊 WYCKOFF ANALYSIS:
================================================================================
Current Phase: {phase}
Volume Trend: {volume_trend} | Volume Spike: {'YES' if is_volume_spike else 'NO'}
Price Change (5d): {price_change_5:+.2f}% | (10d): {price_change_10:+.2f}%
Breakout Soon: {'✅ YES' if will_breakout_soon else '❌ NO'} (Confidence: {breakout_confidence}%)
================================================================================
"""
    return analysis_text, {'phase': phase, 'will_breakout_soon': will_breakout_soon, 'breakout_confidence': breakout_confidence, 'volume_spike': is_volume_spike, 'confidence_boost': confidence_boost}

def get_sector_analysis(sector, symbol, current_price):
    sector_strength = {'Pharmaceuticals & Chemicals': {'strength': 'Strong', 'confidence_boost': 5}, 'Bank': {'strength': 'Moderate', 'confidence_boost': 2}, 'IT': {'strength': 'Strong', 'confidence_boost': 4}}
    sector_info = sector_strength.get(str(sector), {'strength': 'Unknown', 'confidence_boost': 0})
    rotation = random.choice(['None', 'Rotation In', 'Rotation Out'])
    if rotation == 'Rotation In': sector_info['confidence_boost'] += 3
    elif rotation == 'Rotation Out': sector_info['confidence_boost'] -= 3
    return {'strength': sector_info['strength'], 'rotation': rotation, 'confidence_boost': sector_info['confidence_boost'], 'additional_note': ''}