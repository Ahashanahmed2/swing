# scripts/generate_pattern_training_data_complete.py
# RSI Divergence, MACD, Stochastic, ATR, Bollinger Bands, OBV, Volume Profile সহ সম্পূর্ণ ট্রেনিং ডাটা
# 130+ প্যাটার্ন + Elliott Wave + SMC সম্পূর্ণ লাইব্রেরি + Multiple Historical Sequences + Noise Variations
# ✅ NEW: Sector Rotation + Symbol Ranking + Wyckoff + Forward-Looking Analysis + 150+ Candles
# ✅ NEW: Complete Elliott Wave Detection with Sub-waves + Fibonacci + Invalidation
# ✅ NEW: Multi-Timeframe + Fibonacci Time Zones + Volume Confluence + ML Pattern Matching + Backtesting
# ✅ NEW: Price Action Tools - Trend Lines, Ray, Parallel Channel, Trend-Based Fib Extension, Fixed Range Volume Profile
# ✅ NEW: Fixed Range Liquidity, Gap Analysis, Volatility Skew, Z-Score, LSTM Prediction, Risk Metrics, Supply/Demand, Anchored VWAP, Order Book Simulation, Backtester
# ✅ NEW: AUTO CONFIG - Auto MAX_SYMBOLS, Auto MAX_PER_SYMBOL, Pattern Tracking, Coverage Report
# ✅ NEW: PRIORITY-BASED TRAINING - Elliott Wave (3.0x) | SMC (2.5x) | Divergence (2.0x) | Candlestick (1.5x)

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
# PRIORITY-BASED TRAINING CONFIGURATION (NEW)
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
    min_examples = int(BASE_MIN_EXAMPLES * multiplier)
    max_examples = int(BASE_MAX_EXAMPLES * multiplier)
    return min_examples, max_examples, multiplier

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
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0: return default
        return float(np.mean(arr))
    except: return default

def safe_max(arr, default=0.0):
    try:
        if arr is None or len(arr) == 0: return default
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0: return default
        return float(np.max(arr))
    except: return default

def safe_min(arr, default=0.0):
    try:
        if arr is None or len(arr) == 0: return default
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0: return default
        return float(np.min(arr))
    except: return default

def clean_array(arr):
    try:
        arr = np.array(arr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    except: return np.array([])

def fill_na_series(series, method='ffill'):
    try:
        if isinstance(series, pd.Series):
            return series.fillna(method='ffill').fillna(method='bfill').fillna(0)
        else:
            arr = np.array(series)
            arr = np.nan_to_num(arr, nan=0.0)
            return arr
    except: return series

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
            json.dump({
                'predictions': self.predictions[-1000:],
                'results': self.results[-1000:],
                'accuracy_log': self.accuracy_log[-100:]
            }, f, indent=2)

    def add_prediction(self, symbol, date, predicted_wave, predicted_target, current_price):
        pred_id = len(self.predictions)
        self.predictions.append({
            'id': pred_id, 'symbol': symbol, 'date': str(date)[:10],
            'predicted_wave': predicted_wave,
            'predicted_target': float(predicted_target) if not np.isnan(predicted_target) else float(current_price),
            'current_price': float(current_price) if not np.isnan(current_price) else 100.0,
            'timestamp': datetime.now().isoformat()
        })
        self.save_history()
        return pred_id

    def verify_prediction(self, pred_id, actual_future_price):
        if pred_id >= len(self.predictions): return None
        pred = self.predictions[pred_id]
        if actual_future_price <= 0 or np.isnan(actual_future_price): return None
        error = abs(pred['predicted_target'] - actual_future_price) / actual_future_price
        accurate = error < 0.05
        result = {'prediction_id': pred_id, 'symbol': pred['symbol'], 'date': pred['date'],
                  'accurate': accurate, 'error_percent': error * 100}
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
        if len(self.equity_curve) == 0:
            return {'total_return': 0, 'total_trades': 0, 'win_rate': 0, 'equity_curve': []}
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital * 100
        winning_trades = sum(1 for i in range(1, len(self.trades), 2) if i < len(self.trades) and self.trades[i]['price'] > self.trades[i-1]['price'])
        total_trades = len(self.trades) // 2
        return {'total_return': total_return, 'total_trades': total_trades,
                'win_rate': winning_trades / total_trades * 100 if total_trades > 0 else 0,
                'equity_curve': self.equity_curve[-50:] if len(self.equity_curve) >= 50 else self.equity_curve}

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
                        trend_lines['support_lines'].append({
                            'start_price': float(y1), 'end_price': float(y2), 'current_level': float(current_level),
                            'slope': float(slope), 'type': 'SUPPORT',
                            'strength': 'STRONG' if touch_count >= 3 else 'MODERATE', 'touches': touch_count})
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
                        trend_lines['resistance_lines'].append({
                            'start_price': float(y1), 'end_price': float(y2), 'current_level': float(current_level),
                            'slope': float(slope), 'type': 'RESISTANCE',
                            'strength': 'STRONG' if touch_count >= 3 else 'MODERATE', 'touches': touch_count})
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
        channels.append({
            'type': 'ASCENDING' if s.get('slope', 0) > 0 else 'DESCENDING' if s.get('slope', 0) < 0 else 'HORIZONTAL',
            'support_line': support_level, 'support_slope': s.get('slope', 0),
            'mid_line': support_level + (channel_top - support_level) / 2, 'channel_top': channel_top,
            'width': channel_top - support_level,
            'width_percent': safe_divide(channel_top - support_level, support_level) * 100,
            'position': 'UPPER' if current_price > support_level + (channel_top - support_level) * 0.7 else
                       'LOWER' if current_price < support_level + (channel_top - support_level) * 0.3 else 'MIDDLE'})
    if trend_lines.get('current_resistance'):
        r = trend_lines['current_resistance']
        resistance_level = r.get('current_level', current_price * 1.05)
        if resistance_level <= 0: resistance_level = current_price * 1.05
        channel_bottom = current_price * 0.95
        channels.append({
            'type': 'ASCENDING' if r.get('slope', 0) > 0 else 'DESCENDING' if r.get('slope', 0) < 0 else 'HORIZONTAL',
            'resistance_line': resistance_level, 'resistance_slope': r.get('slope', 0),
            'mid_line': channel_bottom + (resistance_level - channel_bottom) / 2, 'channel_bottom': channel_bottom,
            'width': resistance_level - channel_bottom,
            'width_percent': safe_divide(resistance_level - channel_bottom, channel_bottom) * 100,
            'position': 'UPPER' if current_price > channel_bottom + (resistance_level - channel_bottom) * 0.7 else
                       'LOWER' if current_price < channel_bottom + (resistance_level - channel_bottom) * 0.3 else 'MIDDLE'})
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
                        rays.append({'type': 'BULLISH_RAY', 'start_price': float(y1),
                                     'current_support': float(slope * (len(recent_lows)-1 - x1) + y1),
                                     'slope': float(slope), 'projections': projections,
                                     'angle': float(np.degrees(np.arctan(slope)))})
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
                        rays.append({'type': 'BEARISH_RAY', 'start_price': float(y1),
                                     'current_resistance': float(slope * (len(recent_highs)-1 - x1) + y1),
                                     'slope': float(slope), 'projections': projections,
                                     'angle': float(np.degrees(np.arctan(slope)))})
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
    fib_levels = {'trend': 'UPTREND' if is_uptrend else 'DOWNTREND', 'swing_low': float(swing_low),
                  'swing_high': float(swing_high), 'retracement': {}, 'extension': {}}
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
    return {'range_high': float(range_high), 'range_low': float(range_low), 'poc': float(poc) if poc else None,
            'value_area_high': float(vah), 'value_area_low': float(val),
            'current_position': 'ABOVE_VA' if current_price > vah else 'BELOW_VA' if current_price < val else 'IN_VA',
            'volume_profile': dict(sorted(volume_by_price.items())), 'total_volume': float(total_volume)}

def predict_price_from_trend_lines(trend_lines, current_price, days_forward=10):
    if not trend_lines or current_price <= 0: return None
    predictions = []
    if trend_lines.get('current_support'):
        s = trend_lines['current_support']; level = s.get('current_level', current_price)
        if level > 0 and current_price < level * 1.02:
            predictions.append({'type': 'BOUNCE_FROM_SUPPORT', 'expected_direction': 'UP',
                                'target': float(level * 1.05), 'invalidation': float(level * 0.98),
                                'confidence': 'HIGH' if s.get('strength') == 'STRONG' else 'MEDIUM'})
    if trend_lines.get('current_resistance'):
        r = trend_lines['current_resistance']; level = r.get('current_level', current_price)
        if level > 0 and current_price > level * 0.98:
            predictions.append({'type': 'REJECTION_FROM_RESISTANCE', 'expected_direction': 'DOWN',
                                'target': float(level * 0.95), 'invalidation': float(level * 1.02),
                                'confidence': 'HIGH' if r.get('strength') == 'STRONG' else 'MEDIUM'})
        if level > 0 and current_price > level * 1.01:
            predictions.append({'type': 'BULLISH_BREAKOUT', 'expected_direction': 'UP',
                                'target': float(current_price * 1.05), 'invalidation': float(level), 'confidence': 'HIGH'})
    if trend_lines.get('current_support'):
        s = trend_lines['current_support']; level = s.get('current_level', current_price)
        if level > 0 and current_price < level * 0.99:
            predictions.append({'type': 'BEARISH_BREAKDOWN', 'expected_direction': 'DOWN',
                                'target': float(current_price * 0.95), 'invalidation': float(level), 'confidence': 'HIGH'})
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
# NEW FUNCTIONS - LIQUIDITY, GAP, VOLATILITY, Z-SCORE, LSTM, RISK
# =========================================================

def detect_fixed_range_liquidity(symbol_data, lookback=200):
    if symbol_data is None: return None
    closes, volumes = clean_array(symbol_data['close'].values), clean_array(symbol_data['volume'].values)
    if len(closes) < lookback: return None
    volume_by_price = {}
    for price, vol in zip(closes[-lookback:], volumes[-lookback:]):
        if price <= 0: continue
        price_level = round(price, -1)
        volume_by_price[price_level] = volume_by_price.get(price_level, 0) + vol
    if not volume_by_price: return None
    sorted_levels = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
    current_price = float(closes[-1]) if len(closes) > 0 else 0.0
    return {'highest_liquidity': float(sorted_levels[0][0]) if sorted_levels else None,
            'liquidity_levels': [(float(k), float(v)) for k, v in sorted_levels[:5]],
            'current_position': 'ABOVE' if current_price > sorted_levels[0][0] else 'BELOW' if sorted_levels else 'UNKNOWN'}

def analyze_gaps(symbol_data, idx):
    if symbol_data is None or idx < 1: return None
    prev_close = symbol_data['close'].iloc[idx-1]
    curr_open = symbol_data['open'].iloc[idx]
    if prev_close <= 0: return None
    gap_pct = (curr_open - prev_close) / prev_close * 100
    if abs(gap_pct) < 0.5: return {'type': 'NO_GAP', 'gap_percent': 0.0, 'fill_probability': 0, 'expected_fill_days': 0}
    historical_gaps = []
    for i in range(max(20, idx-100), idx):
        prev_c = symbol_data['close'].iloc[i-1]
        curr_o = symbol_data['open'].iloc[i]
        if prev_c > 0:
            gap = (curr_o - prev_c) / prev_c * 100
            if abs(gap) > 0.5:
                filled = False
                for j in range(i, min(i+10, idx)):
                    if (gap > 0 and symbol_data['low'].iloc[j] <= prev_c) or (gap < 0 and symbol_data['high'].iloc[j] >= prev_c):
                        filled = True; break
                historical_gaps.append(filled)
    fill_prob = sum(historical_gaps) / len(historical_gaps) * 100 if historical_gaps else 70
    return {'type': 'GAP_UP' if gap_pct > 0 else 'GAP_DOWN', 'gap_percent': float(gap_pct),
            'fill_probability': float(fill_prob), 'expected_fill_days': 3 if abs(gap_pct) < 1 else 7}

def analyze_volatility_skew(symbol_data):
    if symbol_data is None: return None
    returns = symbol_data['close'].pct_change().dropna()
    returns = clean_array(returns.values)
    if len(returns) < 60: return None
    vol_5d = safe_divide(np.std(returns[-5:]) * np.sqrt(252), 1, 0) * 100 if len(returns) >= 5 else 0
    vol_20d = safe_divide(np.std(returns[-20:]) * np.sqrt(252), 1, 0) * 100 if len(returns) >= 20 else 0
    term_structure = 'CONTANGO' if vol_20d > vol_5d else 'BACKWARDATION'
    return {'vol_5d': float(vol_5d), 'vol_20d': float(vol_20d), 'term_structure': term_structure,
            'signal': 'BULLISH' if term_structure == 'CONTANGO' else 'CAUTION'}

def calculate_zscore_signals(symbol_data, lookback=50):
    if symbol_data is None: return None
    closes = clean_array(symbol_data['close'].values)
    if len(closes) < lookback: return None
    sma = safe_mean(closes[-lookback:], 0)
    std = safe_divide(np.std(closes[-lookback:]), 1, 1)
    current_price = float(closes[-1]) if len(closes) > 0 else 0.0
    zscore = (current_price - sma) / std if std > 0 else 0
    return {'zscore': float(zscore), 'signal': 'OVERSOLD' if zscore < -2 else 'OVERBOUGHT' if zscore > 2 else 'NEUTRAL',
            'mean_reversion_target': float(sma), 'confidence': float(min(95, abs(zscore) * 20))}

def predict_price_lstm(symbol_data, lookback=50, forecast=5):
    if symbol_data is None: return None
    closes = clean_array(symbol_data['close'].values)
    if len(closes) < lookback: return None
    recent_closes = closes[-lookback:]
    x = np.arange(len(recent_closes))
    try: slope, intercept = np.polyfit(x, recent_closes, 1)
    except: slope, intercept = 0, safe_mean(recent_closes, 100)
    predictions = [float(slope * (len(recent_closes) + i) + intercept) for i in range(1, forecast+1)]
    return {'method': 'LSTM (Simulated)', 'forecast_days': forecast, 'predictions': predictions,
            'trend': 'UP' if slope > 0 else 'DOWN', 'confidence': float(min(80, abs(slope) * 100))}

def detect_all_candlestick_patterns(df, idx):
    patterns = []
    if detect_hammer(df, idx): patterns.append('Hammer')
    if detect_shooting_star(df, idx): patterns.append('Shooting Star')
    if detect_doji(df, idx): patterns.append('Doji')
    if detect_spinning_top(df, idx): patterns.append('Spinning Top')
    if detect_marubozu(df, idx): patterns.append('Marubozu')
    if detect_engulfing(df, idx): patterns.append('Engulfing')
    if detect_harami(df, idx): patterns.append('Harami')
    if detect_piercing_line(df, idx): patterns.append('Piercing Line')
    if detect_dark_cloud_cover(df, idx): patterns.append('Dark Cloud Cover')
    if detect_tweezer_top_bottom(df, idx): patterns.append('Tweezer')
    if detect_morning_star(df, idx): patterns.append('Morning Star')
    if detect_evening_star(df, idx): patterns.append('Evening Star')
    if detect_three_white_soldiers(df, idx): patterns.append('Three White Soldiers')
    if detect_three_black_crows(df, idx): patterns.append('Three Black Crows')
    if detect_three_inside_up_down(df, idx): patterns.append('Three Inside Up/Down')
    if detect_abandoned_baby(df, idx): patterns.append('Abandoned Baby')
    return patterns

def calculate_risk_metrics(symbol_data, risk_free_rate=0.04):
    if symbol_data is None: return None
    returns = symbol_data['close'].pct_change().dropna()
    returns = clean_array(returns.values)
    if len(returns) < 50: return None
    var_95 = np.percentile(returns, 5) * 100
    cvar_95 = safe_mean(returns[returns <= np.percentile(returns, 5)], 0) * 100
    excess_returns = returns - risk_free_rate/252
    std_returns = np.std(returns)
    sharpe = np.sqrt(252) * safe_mean(excess_returns, 0) / std_returns if std_returns > 0 else 0
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1
    sortino = np.sqrt(252) * safe_mean(returns, 0) / downside_std if downside_std > 0 else 0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = safe_min(drawdown, 0) * 100
    return {'var_95': float(var_95), 'cvar_95': float(cvar_95), 'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino), 'max_drawdown': float(max_dd),
            'risk_level': 'HIGH' if var_95 < -3 else 'MEDIUM' if var_95 < -1.5 else 'LOW'}

def detect_supply_demand_zones(df, idx, lookback=100):
    if df is None or idx < lookback: return None
    highs, lows = clean_array(df['high'].values[-lookback:]), clean_array(df['low'].values[-lookback:])
    zones = []
    current_price = float(df['close'].iloc[idx]) if not np.isnan(df['close'].iloc[idx]) else 100.0
    for i in range(20, len(highs)-10):
        if i+5 <= len(highs) and i+5 <= len(lows):
            range_high, range_low = safe_max(highs[i:i+5], current_price), safe_min(lows[i:i+5], current_price)
            if range_low > 0 and (range_high - range_low) / range_low < 0.02:
                prev_move = highs[i-1] - lows[i-5] if i >= 5 else 0
                if prev_move > 0:
                    zones.append({'type': 'DEMAND_ZONE' if i+5 < len(highs) and highs[i+5] > range_high else 'SUPPLY_ZONE',
                                  'level_high': float(range_high), 'level_low': float(range_low),
                                  'freshness': 'FRESH' if current_price > range_high else 'TESTED',
                                  'strength': 'STRONG' if i > 50 else 'MODERATE'})
    return zones[:5] if zones else None

def calculate_anchored_vwap(symbol_data, anchor_date_idx):
    if symbol_data is None or anchor_date_idx >= len(symbol_data): return None
    anchored_data = symbol_data.iloc[anchor_date_idx:]
    if len(anchored_data) < 2: return None
    prices = (anchored_data['high'] + anchored_data['low'] + anchored_data['close']) / 3
    volumes = clean_array(anchored_data['volume'].values)
    vwap_series = (prices * volumes).cumsum() / volumes.cumsum()
    vwap_series = vwap_series.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    current_price = float(symbol_data['close'].iloc[-1]) if len(symbol_data) > 0 else 0.0
    vwap_value = float(vwap_series.iloc[-1]) if len(vwap_series) > 0 and not np.isnan(vwap_series.iloc[-1]) else current_price
    if vwap_value <= 0: vwap_value = current_price
    return {'anchored_vwap': vwap_value, 'deviation': safe_divide(current_price - vwap_value, vwap_value) * 100,
            'position': 'ABOVE' if current_price > vwap_value else 'BELOW',
            'signal': 'BULLISH' if current_price > vwap_value and len(vwap_series) >= 2 and current_price > vwap_series.iloc[-2] else 'BEARISH'}

def simulate_order_book(symbol_data, idx):
    if symbol_data is None or idx < 20: return None
    current_price = float(symbol_data['close'].iloc[idx]) if not np.isnan(symbol_data['close'].iloc[idx]) else 100.0
    liquidity_levels = {
        'bids': [{'price': current_price * 0.998, 'size': random.randint(10000, 50000)},
                 {'price': current_price * 0.995, 'size': random.randint(20000, 100000)},
                 {'price': current_price * 0.990, 'size': random.randint(50000, 200000)}],
        'asks': [{'price': current_price * 1.002, 'size': random.randint(10000, 50000)},
                 {'price': current_price * 1.005, 'size': random.randint(20000, 100000)},
                 {'price': current_price * 1.010, 'size': random.randint(50000, 200000)}]}
    total_bids = sum(b['size'] for b in liquidity_levels['bids'])
    total_asks = sum(a['size'] for a in liquidity_levels['asks'])
    return {'liquidity_levels': liquidity_levels, 'bid_ask_ratio': safe_divide(total_bids, total_asks, 1),
            'imbalance': 'BUY_PRESSURE' if total_bids > total_asks * 1.2 else 'SELL_PRESSURE' if total_asks > total_bids * 1.2 else 'BALANCED',
            'nearest_support': float(liquidity_levels['bids'][0]['price']),
            'nearest_resistance': float(liquidity_levels['asks'][0]['price'])}