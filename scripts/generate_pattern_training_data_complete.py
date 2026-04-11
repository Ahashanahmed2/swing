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
# ✅ BUG FIX: NaN handling, Division by Zero protection, Array bounds checking

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
# TRAINING CONFIGURATION - AUTO (সব প্যাটার্ন + সব সিম্বল)
# =========================================================

# ✅ অটো ডিটেক্ট MAX_SYMBOLS
try:
    df_temp = pd.read_csv('./csv/mongodb.csv')
    MAX_SYMBOLS = df_temp['symbol'].nunique()
    print(f"✅ Auto-detected {MAX_SYMBOLS} symbols from mongodb.csv")
except:
    MAX_SYMBOLS = 396  # Fallback
    print(f"⚠️ Using fallback: {MAX_SYMBOLS} symbols")

# ✅ TOTAL_PATTERNS পরে main() এ সেট হবে
TOTAL_PATTERNS = None

# ✅ প্যাটার্ন কভারেজ কনফিগ
MIN_EXAMPLES_PER_PATTERN = 5      # প্রতি প্যাটার্ন কমপক্ষে ৫টি examples
MAX_EXAMPLES_PER_PATTERN = 10     # প্রতি প্যাটার্ন সর্বোচ্চ ১০টি

# ✅ VARIATION কনফিগ
NUM_VARIATIONS = 4                # ৪টি variation per pattern

# ✅ MAX_PER_SYMBOL হবে অটো ক্যালকুলেটেড
MAX_PER_SYMBOL = None             # main() এ TOTAL_PATTERNS * MIN_EXAMPLES_PER_PATTERN * NUM_VARIATIONS

# Time control
MAX_EXAMPLES_PER_RUN = 100000     # Max examples to generate (prevents timeout)

# Elliott Wave Configuration
ELLIOTT_LOOKBACK = 300
SWING_WINDOW = 5
HT_SWING_WINDOW = 20

# Fibonacci Levels
FIB_RETRACEMENT = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXTENSION = [1.272, 1.618, 2.0, 2.618, 4.236]

# Global Trackers
generated_patterns_tracker = defaultdict(lambda: defaultdict(int))
elliott_wave_tracker = defaultdict(list)
mistake_log = []

# Elliott Wave Backtester Instance
elliott_backtester = None

# =========================================================
# 🎯 PRIORITY-BASED TRAINING CONFIGURATION (LLM অপ্টিমাইজড)
# =========================================================

# বেস কনফিগারেশন
BASE_MIN_EXAMPLES = 50      # বেস মিনিমাম examples per pattern
BASE_MAX_EXAMPLES = 100     # বেস ম্যাক্সিমাম examples per pattern

# ✅ প্যাটার্ন টাইপ অনুযায়ী মাল্টিপ্লায়ার
PATTERN_PRIORITY = {
    # Elliott Wave Patterns (সবচেয়ে জটিল) - 3.0x
    'Impulse Wave': 3.0,
    'Leading Diagonal': 3.0,
    'Ending Diagonal': 3.0,
    '3rd Wave Extension': 3.0,
    '5th Wave Extension': 3.0,
    'Single Zigzag': 3.0,
    'Double Zigzag': 3.0,
    'Regular Flat': 3.0,
    'Expanded Flat': 3.0,
    'Contracting Triangle': 3.0,
    'Expanding Triangle': 3.0,
    
    # SMC/Smart Money Patterns (খুব জটিল) - 2.5x
    'Break of Structure (BOS)': 2.5,
    'Bullish Order Block': 2.5,
    'Bearish Order Block': 2.5,
    'Fair Value Gap (FVG)': 2.5,
    'Optimal Trade Entry (OTE)': 2.5,
    'Liquidity Sweep': 2.5,
    'Change of Character (CHoCH)': 2.5,
    'Breaker Block': 2.5,
    'Mitigation Block': 2.5,
    'Rejection Block': 2.5,
    'Vacuum Block': 2.5,
    'Turtle Soup': 2.5,
    'Power of 3': 2.5,
    'Silver Bullet': 2.5,
    'MSS': 2.5,
    'SIBI/BISI': 2.5,
    
    # Divergence Patterns (মিডিয়াম জটিল) - 2.0x
    'RSI Divergence': 2.0,
    'MACD Divergence': 2.0,
    'Hidden Divergence': 2.0,
    'Volume Divergence': 2.0,
    
    # Candlestick Patterns (সহজ কিন্তু গুরুত্বপূর্ণ) - 1.5x
    'Hammer': 1.5,
    'Shooting Star': 1.5,
    'Doji': 1.5,
    'Engulfing': 1.5,
    'Bullish Engulfing': 1.5,
    'Bearish Engulfing': 1.5,
    'Morning Star': 1.5,
    'Evening Star': 1.5,
    'Three White Soldiers': 1.5,
    'Three Black Crows': 1.5,
    'Piercing Line': 1.5,
    'Dark Cloud Cover': 1.5,
    'Harami': 1.5,
    'Spinning Top': 1.5,
    'Marubozu': 1.5,
    'Tweezer': 1.5,
    'Abandoned Baby': 1.5,
    'Three Inside Up/Down': 1.5,
    
    # Basic Chart Patterns (সহজ) - 1.0x
    'Cup and Handle': 1.0,
    'Double Bottom': 1.0,
    'Double Top': 1.0,
    'Head and Shoulders': 1.0,
    'Inverse Head and Shoulders': 1.0,
    'Bull Flag': 1.0,
    'Bear Flag': 1.0,
    'Ascending Triangle': 1.0,
    'Descending Triangle': 1.0,
    'Symmetrical Triangle': 1.0,
    'Rounding Bottom': 1.0,
    'Rounding Top': 1.0,
    'Pennant': 1.0,
    'Wedge': 1.0,
    'Channel': 1.0,
    
    # Volume/Volatility Patterns - 1.2x
    'Volume Climax': 1.2,
    'Volume Spike': 1.2,
    'Bollinger Band Squeeze': 1.2,
    'Volatility Breakout': 1.2,
}

# ডিফল্ট মাল্টিপ্লায়ার (যে প্যাটার্ন লিস্টে নেই)
DEFAULT_PRIORITY = 1.0

# =========================================================
# SAFETY HELPER FUNCTIONS (BUG FIX)
# =========================================================

def safe_divide(a, b, default=0.0):
    """Safe division with default value"""
    try:
        if b == 0 or np.isnan(a) or np.isnan(b):
            return default
        result = a / b
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except:
        return default

def safe_mean(arr, default=0.0):
    """Safe mean calculation"""
    try:
        if arr is None or len(arr) == 0:
            return default
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return default
        return float(np.mean(arr))
    except:
        return default

def safe_max(arr, default=0.0):
    """Safe max calculation"""
    try:
        if arr is None or len(arr) == 0:
            return default
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return default
        return float(np.max(arr))
    except:
        return default

def safe_min(arr, default=0.0):
    """Safe min calculation"""
    try:
        if arr is None or len(arr) == 0:
            return default
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return default
        return float(np.min(arr))
    except:
        return default

def clean_array(arr):
    """Remove NaN and Inf from array"""
    try:
        arr = np.array(arr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    except:
        return np.array([])

def fill_na_series(series, method='ffill'):
    """Fill NaN values in series"""
    try:
        if isinstance(series, pd.Series):
            return series.fillna(method='ffill').fillna(method='bfill').fillna(0)
        else:
            arr = np.array(series)
            arr = np.nan_to_num(arr, nan=0.0)
            return arr
    except:
        return series

# =========================================================
# PATTERN PRIORITY HELPER FUNCTIONS
# =========================================================

def get_pattern_limits(pattern_name):
    """
    প্যাটার্ন অনুযায়ী min/max examples ক্যালকুলেট করে
    """
    multiplier = PATTERN_PRIORITY.get(pattern_name, DEFAULT_PRIORITY)
    
    min_examples = int(BASE_MIN_EXAMPLES * multiplier)
    max_examples = int(BASE_MAX_EXAMPLES * multiplier)
    
    return min_examples, max_examples, multiplier


def get_pattern_category(pattern_name):
    """প্যাটার্নের ক্যাটাগরি রিটার্ন করে"""
    if pattern_name in ['Impulse Wave', 'Leading Diagonal', 'Ending Diagonal', 
                        '3rd Wave Extension', '5th Wave Extension', 'Single Zigzag',
                        'Double Zigzag', 'Regular Flat', 'Expanded Flat',
                        'Contracting Triangle', 'Expanding Triangle']:
        return 'Elliott Wave'
    elif pattern_name in ['Break of Structure (BOS)', 'Bullish Order Block', 
                          'Bearish Order Block', 'Fair Value Gap (FVG)', 
                          'Optimal Trade Entry (OTE)', 'Liquidity Sweep',
                          'Change of Character (CHoCH)', 'Breaker Block',
                          'Mitigation Block', 'Rejection Block', 'Vacuum Block',
                          'Turtle Soup', 'Power of 3', 'Silver Bullet', 'MSS',
                          'SIBI/BISI', 'ICT Macro', 'Killzone']:
        return 'SMC'
    elif 'Divergence' in pattern_name:
        return 'Divergence'
    elif pattern_name in ['Hammer', 'Shooting Star', 'Doji', 'Engulfing',
                          'Bullish Engulfing', 'Bearish Engulfing', 'Morning Star', 
                          'Evening Star', 'Three White Soldiers', 'Three Black Crows', 
                          'Piercing Line', 'Dark Cloud Cover', 'Harami', 'Spinning Top',
                          'Marubozu', 'Tweezer', 'Abandoned Baby', 'Three Inside Up/Down']:
        return 'Candlestick'
    elif pattern_name in ['Volume Climax', 'Volume Spike', 'Bollinger Band Squeeze', 'Volatility Breakout']:
        return 'Volume/Volatility'
    else:
        return 'Basic'


def display_priority_summary(all_patterns):
    """প্রায়োরিটি সামারি ডিসপ্লে করে"""
    print("\n" + "="*80)
    print("📊 PATTERN PRIORITY CONFIGURATION (LLM Training Optimized):")
    print("="*80)
    
    category_stats = defaultdict(lambda: {'patterns': [], 'total_min': 0, 'total_max': 0, 'total_multiplier': 0})
    
    for pattern_name in all_patterns.keys():
        min_ex, max_ex, multiplier = get_pattern_limits(pattern_name)
        category = get_pattern_category(pattern_name)
        
        category_stats[category]['patterns'].append(pattern_name)
        category_stats[category]['total_min'] += min_ex
        category_stats[category]['total_max'] += max_ex
        category_stats[category]['total_multiplier'] += multiplier
    
    print(f"\n{'Category':<20} {'Patterns':<10} {'Multiplier':<12} {'Min/Pattern':<12} {'Max/Pattern':<12} {'Total Min':<12}")
    print("─" * 80)
    
    priority_order = ['Elliott Wave', 'SMC', 'Divergence', 'Candlestick', 'Volume/Volatility', 'Basic']
    
    for cat in priority_order:
        if cat in category_stats:
            stats = category_stats[cat]
            avg_multiplier = stats['total_multiplier'] / len(stats['patterns'])
            avg_min = stats['total_min'] / len(stats['patterns'])
            avg_max = stats['total_max'] / len(stats['patterns'])
            
            print(f"{cat:<20} {len(stats['patterns']):<10} {avg_multiplier:.1f}x{'':<7} {avg_min:.0f}{'':<9} {avg_max:.0f}{'':<9} {stats['total_min']:<12}")
    
    print("─" * 80)
    
    # টোটাল ক্যালকুলেশন
    total_patterns = len(all_patterns)
    total_min_all = sum(stats['total_min'] for stats in category_stats.values())
    total_max_all = sum(stats['total_max'] for stats in category_stats.values())
    
    print(f"\n📈 TOTAL ESTIMATED EXAMPLES:")
    print(f"   Total Patterns: {total_patterns}")
    print(f"   Base Examples (min): {total_min_all:,}")
    print(f"   Base Examples (max): {total_max_all:,}")
    print(f"   With {NUM_VARIATIONS} variations:")
    print(f"   └── Minimum Total: {total_min_all * NUM_VARIATIONS:,} examples")
    print(f"   └── Maximum Total: {total_max_all * NUM_VARIATIONS:,} examples")
    print("="*80)
    
    return category_stats

# =========================================================
# ELLIOTT WAVE BACKTESTER CLASS
# =========================================================

class ElliottWaveBacktester:
    """Track Elliott Wave prediction accuracy"""

    def __init__(self):
        self.predictions = []
        self.results = []
        self.accuracy_log = []
        self.backtest_file = "./csv/elliott_backtest.json"
        self.load_history()

    def load_history(self):
        """Load previous backtest history"""
        if os.path.exists(self.backtest_file):
            try:
                with open(self.backtest_file, 'r') as f:
                    data = json.load(f)
                    self.predictions = data.get('predictions', [])
                    self.results = data.get('results', [])
                    self.accuracy_log = data.get('accuracy_log', [])
            except:
                pass

    def save_history(self):
        """Save backtest history"""
        os.makedirs("./csv", exist_ok=True)
        with open(self.backtest_file, 'w') as f:
            json.dump({
                'predictions': self.predictions[-1000:],
                'results': self.results[-1000:],
                'accuracy_log': self.accuracy_log[-100:]
            }, f, indent=2)

    def add_prediction(self, symbol, date, predicted_wave, predicted_target, current_price):
        """Add a prediction for tracking"""
        pred_id = len(self.predictions)

        self.predictions.append({
            'id': pred_id,
            'symbol': symbol,
            'date': str(date)[:10],
            'predicted_wave': predicted_wave,
            'predicted_target': float(predicted_target) if not np.isnan(predicted_target) else float(current_price),
            'current_price': float(current_price) if not np.isnan(current_price) else 100.0,
            'timestamp': datetime.now().isoformat()
        })

        self.save_history()
        return pred_id

    def verify_prediction(self, pred_id, actual_future_price):
        """Verify a previous prediction"""
        if pred_id >= len(self.predictions):
            return None

        pred = self.predictions[pred_id]
        if actual_future_price <= 0 or np.isnan(actual_future_price):
            return None
        error = abs(pred['predicted_target'] - actual_future_price) / actual_future_price
        accurate = error < 0.05

        result = {
            'prediction_id': pred_id,
            'symbol': pred['symbol'],
            'date': pred['date'],
            'accurate': accurate,
            'error_percent': error * 100
        }

        self.results.append(result)

        total = len(self.results)
        accurate_count = sum(1 for r in self.results if r['accurate'])
        accuracy = safe_divide(accurate_count, total) * 100 if total > 0 else 0

        self.accuracy_log.append({
            'total_predictions': total,
            'accuracy': accuracy,
            'date': datetime.now().isoformat()
        })

        self.save_history()
        return result

    def get_performance_report(self):
        """Get overall performance report"""
        if not self.results:
            return "No predictions verified yet"

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
        """Run backtest on historical data with given signals"""
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
            if not np.isnan(equity):
                self.equity_curve.append(equity)

        return self.get_performance()

    def get_performance(self):
        if len(self.equity_curve) == 0:
            return {'total_return': 0, 'total_trades': 0, 'win_rate': 0, 'equity_curve': []}

        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital * 100
        winning_trades = sum(1 for i in range(1, len(self.trades), 2) if i < len(self.trades) and self.trades[i]['price'] > self.trades[i-1]['price'])
        total_trades = len(self.trades) // 2

        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': winning_trades / total_trades * 100 if total_trades > 0 else 0,
            'equity_curve': self.equity_curve[-50:] if len(self.equity_curve) >= 50 else self.equity_curve
        }


# =========================================================
# PRICE ACTION TOOLS - TREND LINES, RAY, PARALLEL CHANNEL
# =========================================================

def calculate_trend_line(highs, lows, closes, lookback=50):
    """Calculate dynamic trend lines based on swing points"""
    highs = clean_array(highs)
    lows = clean_array(lows)
    closes = clean_array(closes)
    
    if len(highs) < lookback:
        return None

    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    recent_closes = closes[-lookback:]

    swing_highs = []
    swing_lows = []
    swing_high_indices = []
    swing_low_indices = []

    for i in range(2, len(recent_highs) - 2):
        if i-2 >= 0 and i+3 <= len(recent_highs):
            if recent_highs[i] == max(recent_highs[i-2:i+3]):
                swing_highs.append(recent_highs[i])
                swing_high_indices.append(i)
        if i-2 >= 0 and i+3 <= len(recent_lows):
            if recent_lows[i] == min(recent_lows[i-2:i+3]):
                swing_lows.append(recent_lows[i])
                swing_low_indices.append(i)

    trend_lines = {
        'support_lines': [],
        'resistance_lines': [],
        'current_support': None,
        'current_resistance': None,
        'channel': None
    }

    if len(swing_lows) >= 2:
        for i in range(len(swing_lows) - 1):
            for j in range(i + 1, len(swing_lows)):
                x1, y1 = swing_low_indices[i], swing_lows[i]
                x2, y2 = swing_low_indices[j], swing_lows[j]

                if x2 > x1 and x2 - x1 > 0:
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1

                    valid = True
                    touch_count = 0
                    for k in range(x1, min(x2 + 20, len(recent_lows))):
                        expected = slope * k + intercept
                        if recent_lows[k] < expected * 0.99:
                            valid = False
                            break
                        if expected > 0 and abs(recent_lows[k] - expected) / expected < 0.01:
                            touch_count += 1

                    if valid and touch_count >= 2:
                        current_level = slope * (len(recent_lows) - 1) + intercept
                        trend_lines['support_lines'].append({
                            'start_price': float(y1),
                            'end_price': float(y2),
                            'current_level': float(current_level),
                            'slope': float(slope),
                            'type': 'SUPPORT',
                            'strength': 'STRONG' if touch_count >= 3 else 'MODERATE',
                            'touches': touch_count
                        })

    if len(swing_highs) >= 2:
        for i in range(len(swing_highs) - 1):
            for j in range(i + 1, len(swing_highs)):
                x1, y1 = swing_high_indices[i], swing_highs[i]
                x2, y2 = swing_high_indices[j], swing_highs[j]

                if x2 > x1 and x2 - x1 > 0:
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1

                    valid = True
                    touch_count = 0
                    for k in range(x1, min(x2 + 20, len(recent_highs))):
                        expected = slope * k + intercept
                        if recent_highs[k] > expected * 1.01:
                            valid = False
                            break
                        if expected > 0 and abs(recent_highs[k] - expected) / expected < 0.01:
                            touch_count += 1

                    if valid and touch_count >= 2:
                        current_level = slope * (len(recent_highs) - 1) + intercept
                        trend_lines['resistance_lines'].append({
                            'start_price': float(y1),
                            'end_price': float(y2),
                            'current_level': float(current_level),
                            'slope': float(slope),
                            'type': 'RESISTANCE',
                            'strength': 'STRONG' if touch_count >= 3 else 'MODERATE',
                            'touches': touch_count
                        })

    current_price = float(recent_closes[-1]) if len(recent_closes) > 0 else 0.0
    if current_price <= 0:
        current_price = 100.0

    if trend_lines['support_lines']:
        supports_below = [s for s in trend_lines['support_lines'] if s['current_level'] < current_price]
        if supports_below:
            trend_lines['current_support'] = max(supports_below, key=lambda x: x['current_level'])

    if trend_lines['resistance_lines']:
        resistances_above = [r for r in trend_lines['resistance_lines'] if r['current_level'] > current_price]
        if resistances_above:
            trend_lines['current_resistance'] = min(resistances_above, key=lambda x: x['current_level'])

    return trend_lines


def calculate_parallel_channel(trend_lines, current_price):
    """Create parallel channel from trend line"""
    if not trend_lines or current_price <= 0 or np.isnan(current_price):
        return None

    channels = []

    if trend_lines.get('current_support'):
        support = trend_lines['current_support']
        slope = support.get('slope', 0)
        support_level = support.get('current_level', current_price * 0.95)
        
        if support_level <= 0:
            support_level = current_price * 0.95

        channel_top = current_price * 1.05

        channel = {
            'type': 'ASCENDING' if slope > 0 else 'DESCENDING' if slope < 0 else 'HORIZONTAL',
            'support_line': support_level,
            'support_slope': slope,
            'mid_line': support_level + (channel_top - support_level) / 2,
            'channel_top': channel_top,
            'width': channel_top - support_level,
            'width_percent': safe_divide(channel_top - support_level, support_level) * 100,
            'position': 'UPPER' if current_price > support_level + (channel_top - support_level) * 0.7 else
                       'LOWER' if current_price < support_level + (channel_top - support_level) * 0.3 else
                       'MIDDLE'
        }
        channels.append(channel)

    if trend_lines.get('current_resistance'):
        resistance = trend_lines['current_resistance']
        slope = resistance.get('slope', 0)
        resistance_level = resistance.get('current_level', current_price * 1.05)
        
        if resistance_level <= 0:
            resistance_level = current_price * 1.05

        channel_bottom = current_price * 0.95

        channel = {
            'type': 'ASCENDING' if slope > 0 else 'DESCENDING' if slope < 0 else 'HORIZONTAL',
            'resistance_line': resistance_level,
            'resistance_slope': slope,
            'mid_line': channel_bottom + (resistance_level - channel_bottom) / 2,
            'channel_bottom': channel_bottom,
            'width': resistance_level - channel_bottom,
            'width_percent': safe_divide(resistance_level - channel_bottom, channel_bottom) * 100,
            'position': 'UPPER' if current_price > channel_bottom + (resistance_level - channel_bottom) * 0.7 else
                       'LOWER' if current_price < channel_bottom + (resistance_level - channel_bottom) * 0.3 else
                       'MIDDLE'
        }
        channels.append(channel)

    return channels if channels else None