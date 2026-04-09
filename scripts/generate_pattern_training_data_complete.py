# scripts/generate_pattern_training_data_complete.py
# RSI Divergence, MACD, Stochastic, ATR, Bollinger Bands, OBV, Volume Profile সহ সম্পূর্ণ ট্রেনিং ডাটা
# 130+ প্যাটার্ন + Elliott Wave + SMC সম্পূর্ণ লাইব্রেরি + Multiple Historical Sequences + Noise Variations
# ✅ NEW: Sector Rotation + Symbol Ranking + Wyckoff + Forward-Looking Analysis + 150+ Candles
# ✅ NEW: Complete Elliott Wave Detection with Sub-waves + Fibonacci + Invalidation
# ✅ NEW: Multi-Timeframe + Fibonacci Time Zones + Volume Confluence + ML Pattern Matching + Backtesting
# ✅ NEW: Price Action Tools - Trend Lines, Ray, Parallel Channel, Trend-Based Fib Extension, Fixed Range Volume Profile
# ✅ NEW: Fixed Range Liquidity, Gap Analysis, Volatility Skew, Z-Score, LSTM Prediction, Risk Metrics, Supply/Demand, Anchored VWAP, Order Book Simulation, Backtester

import pandas as pd
import numpy as np
import os
import random
import json
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union

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
# TRAINING CONFIGURATION
# =========================================================

MAX_SYMBOLS = 380
MAX_PER_SYMBOL = 10
MAX_EXAMPLES_PER_RUN = 5000

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
            'predicted_target': predicted_target,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat()
        })
        
        self.save_history()
        return pred_id
    
    def verify_prediction(self, pred_id, actual_future_price):
        """Verify a previous prediction"""
        if pred_id >= len(self.predictions):
            return None
        
        pred = self.predictions[pred_id]
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
        accuracy = accurate_count / total * 100 if total > 0 else 0
        
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
        avg_error = np.mean([r['error_percent'] for r in self.results])
        recent_10 = self.results[-10:] if len(self.results) >= 10 else self.results
        recent_accuracy = sum(1 for r in recent_10 if r['accurate']) / len(recent_10) * 100
        
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
                position = capital / row['close']
                capital = 0
                self.trades.append({'type': 'BUY', 'price': row['close'], 'date': row['date']})
            elif signal == 'SELL' and position > 0:
                capital = position * row['close']
                position = 0
                self.trades.append({'type': 'SELL', 'price': row['close'], 'date': row['date']})
            
            equity = capital + (position * row['close'] if position > 0 else 0)
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
    if len(highs) < lookback:
        return None
    
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    recent_closes = closes[-lookback:]
    
    # Find swing highs and lows
    swing_highs = []
    swing_lows = []
    swing_high_indices = []
    swing_low_indices = []
    
    for i in range(2, len(recent_highs) - 2):
        if recent_highs[i] == max(recent_highs[i-2:i+3]):
            swing_highs.append(recent_highs[i])
            swing_high_indices.append(i)
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
    
    # Support trend lines (connecting lows)
    if len(swing_lows) >= 2:
        for i in range(len(swing_lows) - 1):
            for j in range(i + 1, len(swing_lows)):
                x1, y1 = swing_low_indices[i], swing_lows[i]
                x2, y2 = swing_low_indices[j], swing_lows[j]
                
                if x2 > x1:
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    
                    # Validate trend line (price should stay above support)
                    valid = True
                    touch_count = 0
                    for k in range(x1, min(x2 + 20, len(recent_lows))):
                        expected = slope * k + intercept
                        if recent_lows[k] < expected * 0.99:
                            valid = False
                            break
                        if abs(recent_lows[k] - expected) / expected < 0.01:
                            touch_count += 1
                    
                    if valid and touch_count >= 2:
                        current_level = slope * (len(recent_lows) - 1) + intercept
                        trend_lines['support_lines'].append({
                            'start_price': y1,
                            'end_price': y2,
                            'current_level': current_level,
                            'slope': slope,
                            'type': 'SUPPORT',
                            'strength': 'STRONG' if touch_count >= 3 else 'MODERATE',
                            'touches': touch_count
                        })
    
    # Resistance trend lines (connecting highs)
    if len(swing_highs) >= 2:
        for i in range(len(swing_highs) - 1):
            for j in range(i + 1, len(swing_highs)):
                x1, y1 = swing_high_indices[i], swing_highs[i]
                x2, y2 = swing_high_indices[j], swing_highs[j]
                
                if x2 > x1:
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    
                    valid = True
                    touch_count = 0
                    for k in range(x1, min(x2 + 20, len(recent_highs))):
                        expected = slope * k + intercept
                        if recent_highs[k] > expected * 1.01:
                            valid = False
                            break
                        if abs(recent_highs[k] - expected) / expected < 0.01:
                            touch_count += 1
                    
                    if valid and touch_count >= 2:
                        current_level = slope * (len(recent_highs) - 1) + intercept
                        trend_lines['resistance_lines'].append({
                            'start_price': y1,
                            'end_price': y2,
                            'current_level': current_level,
                            'slope': slope,
                            'type': 'RESISTANCE',
                            'strength': 'STRONG' if touch_count >= 3 else 'MODERATE',
                            'touches': touch_count
                        })
    
    # Find current support and resistance
    current_price = recent_closes[-1]
    
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
    if not trend_lines:
        return None
    
    channels = []
    
    # Create channel from support line
    if trend_lines.get('current_support'):
        support = trend_lines['current_support']
        slope = support['slope']
        
        channel_top = current_price * 1.05  # Default
        
        channel = {
            'type': 'ASCENDING' if slope > 0 else 'DESCENDING' if slope < 0 else 'HORIZONTAL',
            'support_line': support['current_level'],
            'support_slope': slope,
            'mid_line': support['current_level'] + (channel_top - support['current_level']) / 2,
            'channel_top': channel_top,
            'width': channel_top - support['current_level'],
            'width_percent': (channel_top - support['current_level']) / support['current_level'] * 100,
            'position': 'UPPER' if current_price > support['current_level'] + (channel_top - support['current_level']) * 0.7 else
                       'LOWER' if current_price < support['current_level'] + (channel_top - support['current_level']) * 0.3 else
                       'MIDDLE'
        }
        channels.append(channel)
    
    # Create channel from resistance line
    if trend_lines.get('current_resistance'):
        resistance = trend_lines['current_resistance']
        slope = resistance['slope']
        
        channel_bottom = current_price * 0.95
        
        channel = {
            'type': 'ASCENDING' if slope > 0 else 'DESCENDING' if slope < 0 else 'HORIZONTAL',
            'resistance_line': resistance['current_level'],
            'resistance_slope': slope,
            'mid_line': channel_bottom + (resistance['current_level'] - channel_bottom) / 2,
            'channel_bottom': channel_bottom,
            'width': resistance['current_level'] - channel_bottom,
            'width_percent': (resistance['current_level'] - channel_bottom) / channel_bottom * 100,
            'position': 'UPPER' if current_price > channel_bottom + (resistance['current_level'] - channel_bottom) * 0.7 else
                       'LOWER' if current_price < channel_bottom + (resistance['current_level'] - channel_bottom) * 0.3 else
                       'MIDDLE'
        }
        channels.append(channel)
    
    return channels


def calculate_ray_lines(symbol_data, lookback=100):
    """Calculate Ray lines (extended trend lines into future)"""
    closes = symbol_data['close'].values
    highs = symbol_data['high'].values
    lows = symbol_data['low'].values
    
    if len(closes) < lookback:
        return None
    
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    current_price = closes[-1]
    
    rays = []
    
    # Find significant pivot points
    pivot_highs = []
    pivot_lows = []
    
    for i in range(5, len(recent_highs) - 5):
        if recent_highs[i] == max(recent_highs[i-5:i+6]):
            pivot_highs.append((i, recent_highs[i]))
        if recent_lows[i] == min(recent_lows[i-5:i+6]):
            pivot_lows.append((i, recent_lows[i]))
    
    # Create rays from pivot combinations
    if len(pivot_lows) >= 2:
        # Bullish ray (connecting higher lows)
        for i in range(len(pivot_lows) - 1):
            for j in range(i + 1, len(pivot_lows)):
                if pivot_lows[j][1] > pivot_lows[i][1]:
                    x1, y1 = pivot_lows[i]
                    x2, y2 = pivot_lows[j]
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # Project into future
                    future_steps = [5, 10, 20, 50]
                    projections = {}
                    for step in future_steps:
                        future_x = len(recent_lows) + step
                        future_y = slope * (future_x - x1) + y1
                        projections[f'{step}_days'] = future_y
                    
                    rays.append({
                        'type': 'BULLISH_RAY',
                        'start_price': y1,
                        'current_support': slope * (len(recent_lows) - 1 - x1) + y1,
                        'slope': slope,
                        'projections': projections,
                        'angle': np.degrees(np.arctan(slope))
                    })
    
    if len(pivot_highs) >= 2:
        # Bearish ray (connecting lower highs)
        for i in range(len(pivot_highs) - 1):
            for j in range(i + 1, len(pivot_highs)):
                if pivot_highs[j][1] < pivot_highs[i][1]:
                    x1, y1 = pivot_highs[i]
                    x2, y2 = pivot_highs[j]
                    slope = (y2 - y1) / (x2 - x1)
                    
                    future_steps = [5, 10, 20, 50]
                    projections = {}
                    for step in future_steps:
                        future_x = len(recent_highs) + step
                        future_y = slope * (future_x - x1) + y1
                        projections[f'{step}_days'] = future_y
                    
                    rays.append({
                        'type': 'BEARISH_RAY',
                        'start_price': y1,
                        'current_resistance': slope * (len(recent_highs) - 1 - x1) + y1,
                        'slope': slope,
                        'projections': projections,
                        'angle': np.degrees(np.arctan(slope))
                    })
    
    return rays


def calculate_trend_based_fib_extension(symbol_data, lookback=100):
    """Trend-based Fibonacci extension levels"""
    closes = symbol_data['close'].values
    highs = symbol_data['high'].values
    lows = symbol_data['low'].values
    
    if len(closes) < lookback:
        return None
    
    # Find major swing low and high
    recent_lows = lows[-lookback:]
    recent_highs = highs[-lookback:]
    
    swing_low = min(recent_lows)
    swing_high = max(recent_highs)
    swing_low_idx = np.argmin(recent_lows)
    swing_high_idx = np.argmax(recent_highs)
    
    current_price = closes[-1]
    is_uptrend = swing_high_idx > swing_low_idx
    
    fib_levels = {
        'trend': 'UPTREND' if is_uptrend else 'DOWNTREND',
        'swing_low': swing_low,
        'swing_high': swing_high,
        'retracement': {},
        'extension': {},
        'projection': {}
    }
    
    range_size = swing_high - swing_low
    
    if is_uptrend:
        # Retracement levels
        for fib in FIB_RETRACEMENT:
            level = swing_high - range_size * fib
            fib_levels['retracement'][f'{fib:.3f}'] = level
        
        # Extension levels (projecting beyond swing high)
        for fib in FIB_EXTENSION:
            level = swing_low + range_size * fib
            fib_levels['extension'][f'{fib:.3f}'] = level
    else:
        # Retracement levels
        for fib in FIB_RETRACEMENT:
            level = swing_low + range_size * fib
            fib_levels['retracement'][f'{fib:.3f}'] = level
        
        # Extension levels
        for fib in FIB_EXTENSION:
            level = swing_high - range_size * fib
            fib_levels['extension'][f'{fib:.3f}'] = level
    
    # Find current position
    for level_name, price in fib_levels['retracement'].items():
        if is_uptrend:
            if current_price <= price:
                fib_levels['current_zone'] = f'Below {level_name} retracement'
                break
        else:
            if current_price >= price:
                fib_levels['current_zone'] = f'Above {level_name} retracement'
                break
    
    if not fib_levels.get('current_zone'):
        fib_levels['current_zone'] = 'Between retracement levels'
    
    return fib_levels


def calculate_fixed_range_volume_profile(symbol_data, lookback=100, num_bins=20):
    """Fixed Range Volume Profile"""
    closes = symbol_data['close'].values
    volumes = symbol_data['volume'].values
    highs = symbol_data['high'].values
    lows = symbol_data['low'].values
    
    if len(closes) < lookback:
        return None
    
    recent_closes = closes[-lookback:]
    recent_volumes = volumes[-lookback:]
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    
    range_high = max(recent_highs)
    range_low = min(recent_lows)
    range_size = range_high - range_low
    bin_size = range_size / num_bins
    
    volume_profile = {}
    volume_by_price = defaultdict(float)
    
    for i in range(len(recent_closes)):
        price = recent_closes[i]
        vol = recent_volumes[i]
        
        # Distribute volume across price range
        bin_idx = int((price - range_low) / bin_size)
        bin_idx = max(0, min(num_bins - 1, bin_idx))
        bin_center = range_low + (bin_idx + 0.5) * bin_size
        
        volume_by_price[bin_center] += vol
    
    # Find POC (Point of Control)
    poc = max(volume_by_price, key=volume_by_price.get) if volume_by_price else None
    total_volume = sum(volume_by_price.values())
    
    # Value Area (70% of volume)
    sorted_bins = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
    cumulative_vol = 0
    value_area_bins = []
    
    for price, vol in sorted_bins:
        cumulative_vol += vol
        value_area_bins.append(price)
        if cumulative_vol >= total_volume * 0.7:
            break
    
    vah = max(value_area_bins) if value_area_bins else range_high
    val = min(value_area_bins) if value_area_bins else range_low
    
    current_price = closes[-1]
    
    return {
        'range_high': range_high,
        'range_low': range_low,
        'poc': poc,
        'value_area_high': vah,
        'value_area_low': val,
        'current_position': 'ABOVE_VA' if current_price > vah else 'BELOW_VA' if current_price < val else 'IN_VA',
        'volume_profile': dict(sorted(volume_by_price.items())),
        'total_volume': total_volume
    }


def predict_price_from_trend_lines(trend_lines, current_price, days_forward=10):
    """Predict future price based on trend lines and channels"""
    if not trend_lines:
        return None
    
    predictions = []
    
    # Predict from support trend line
    if trend_lines.get('current_support'):
        support = trend_lines['current_support']
        current_level = support['current_level']
        
        # If price near support, expect bounce
        if current_price < current_level * 1.02:
            predictions.append({
                'type': 'BOUNCE_FROM_SUPPORT',
                'expected_direction': 'UP',
                'target': current_level + (current_level * 0.05),
                'invalidation': current_level * 0.98,
                'confidence': 'HIGH' if support['strength'] == 'STRONG' else 'MEDIUM'
            })
    
    # Predict from resistance trend line
    if trend_lines.get('current_resistance'):
        resistance = trend_lines['current_resistance']
        current_level = resistance['current_level']
        
        # If price near resistance, expect rejection
        if current_price > current_level * 0.98:
            predictions.append({
                'type': 'REJECTION_FROM_RESISTANCE',
                'expected_direction': 'DOWN',
                'target': current_level * 0.95,
                'invalidation': current_level * 1.02,
                'confidence': 'HIGH' if resistance['strength'] == 'STRONG' else 'MEDIUM'
            })
    
    # Breakout prediction
    if trend_lines.get('current_resistance') and current_price > trend_lines['current_resistance']['current_level'] * 1.01:
        predictions.append({
            'type': 'BULLISH_BREAKOUT',
            'expected_direction': 'UP',
            'target': current_price * 1.05,
            'invalidation': trend_lines['current_resistance']['current_level'],
            'confidence': 'HIGH'
        })
    
    if trend_lines.get('current_support') and current_price < trend_lines['current_support']['current_level'] * 0.99:
        predictions.append({
            'type': 'BEARISH_BREAKDOWN',
            'expected_direction': 'DOWN',
            'target': current_price * 0.95,
            'invalidation': trend_lines['current_support']['current_level'],
            'confidence': 'HIGH'
        })
    
    return predictions


def analyze_price_action_complete(symbol_data, idx, lookback=100):
    """Complete Price Action Analysis with all tools"""
    
    closes = symbol_data['close'].values
    highs = symbol_data['high'].values
    lows = symbol_data['low'].values
    current_price = closes[-1] if len(closes) > 0 else 0
    
    if len(closes) < 50:
        return "Insufficient data for Price Action analysis.", None, None, None, None, None, None
    
    # 1. Trend Lines
    trend_lines = calculate_trend_line(highs, lows, closes, lookback)
    
    # 2. Parallel Channels
    channels = calculate_parallel_channel(trend_lines, current_price) if trend_lines else None
    
    # 3. Ray Lines
    rays = calculate_ray_lines(symbol_data, lookback)
    
    # 4. Trend-based Fibonacci Extension
    fib_ext = calculate_trend_based_fib_extension(symbol_data, lookback)
    
    # 5. Fixed Range Volume Profile
    vol_profile = calculate_fixed_range_volume_profile(symbol_data, lookback)
    
    # 6. Price Predictions
    predictions = predict_price_from_trend_lines(trend_lines, current_price) if trend_lines else None
    
    # Generate report
    report = """
📐 PRICE ACTION COMPLETE ANALYSIS:
================================================================================
"""
    
    # Trend Lines Section
    if trend_lines:
        report += """
📏 TREND LINES:
────────────────────────────────────────────────────────────────────────────────
"""
        if trend_lines.get('current_support'):
            s = trend_lines['current_support']
            report += f"""
🔹 SUPPORT TREND LINE:
   Current Level: {s['current_level']:.2f}
   Slope: {s['slope']:.4f} ({'ASCENDING' if s['slope'] > 0 else 'DESCENDING'})
   Strength: {s['strength']} ({s['touches']} touches)
"""
        
        if trend_lines.get('current_resistance'):
            r = trend_lines['current_resistance']
            report += f"""
🔸 RESISTANCE TREND LINE:
   Current Level: {r['current_level']:.2f}
   Slope: {r['slope']:.4f} ({'ASCENDING' if r['slope'] > 0 else 'DESCENDING'})
   Strength: {r['strength']} ({r['touches']} touches)
"""
    
    # Channels Section
    if channels:
        report += """
📊 PARALLEL CHANNELS:
────────────────────────────────────────────────────────────────────────────────
"""
        for i, ch in enumerate(channels[:2]):
            report += f"""
🔹 CHANNEL {i+1} ({ch['type']}):
   Support: {ch.get('support_line', ch.get('channel_bottom', 0)):.2f}
   Resistance: {ch.get('channel_top', ch.get('resistance_line', 0)):.2f}
   Mid Line: {ch['mid_line']:.2f}
   Width: {ch['width']:.2f} ({ch['width_percent']:.1f}%)
   Position: {ch['position']}
"""
    
    # Rays Section
    if rays:
        report += """
🔆 RAY LINES (Future Projections):
────────────────────────────────────────────────────────────────────────────────
"""
        for ray in rays[:3]:
            report += f"""
🔹 {ray['type']}:
   Current Level: {ray.get('current_support', ray.get('current_resistance', 0)):.2f}
   Slope: {ray['slope']:.4f}
   Angle: {ray['angle']:.1f}°
"""
            if ray.get('projections'):
                report += "   Future Projections:\n"
                for period, price in ray['projections'].items():
                    report += f"      • {period}: {price:.2f}\n"
    
    # Trend-based Fibonacci Section
    if fib_ext:
        report += f"""
📐 TREND-BASED FIBONACCI EXTENSION:
────────────────────────────────────────────────────────────────────────────────
Trend: {fib_ext['trend']}
Swing Low: {fib_ext['swing_low']:.2f}
Swing High: {fib_ext['swing_high']:.2f}
Current Zone: {fib_ext['current_zone']}

Retracement Levels:
"""
        for level, price in fib_ext['retracement'].items():
            report += f"   • {level}: {price:.2f}\n"
        
        report += "\nExtension Targets:\n"
        for level, price in fib_ext['extension'].items():
            report += f"   • {level}: {price:.2f}\n"
    
    # Volume Profile Section
    if vol_profile:
        report += f"""
📊 FIXED RANGE VOLUME PROFILE:
────────────────────────────────────────────────────────────────────────────────
Range: {vol_profile['range_low']:.2f} - {vol_profile['range_high']:.2f}
POC (Point of Control): {vol_profile['poc']:.2f}
Value Area: {vol_profile['value_area_low']:.2f} - {vol_profile['value_area_high']:.2f}
Current Position: {vol_profile['current_position']}
Total Volume: {vol_profile['total_volume']:,.0f}
"""
    
    # Price Predictions Section
    if predictions:
        report += """
🎯 PRICE PREDICTIONS (Based on Price Action):
────────────────────────────────────────────────────────────────────────────────
"""
        for pred in predictions:
            report += f"""
🔹 {pred['type']}:
   Direction: {pred['expected_direction']}
   Target: {pred['target']:.2f}
   Invalidation: {pred['invalidation']:.2f}
   Confidence: {pred['confidence']}
"""
    
    report += """
================================================================================
"""
    
    return report, trend_lines, channels, rays, fib_ext, vol_profile, predictions


# =========================================================
# NEW FUNCTIONS - FIXED RANGE LIQUIDITY, GAP ANALYSIS, ETC.
# =========================================================

def detect_fixed_range_liquidity(symbol_data, lookback=200):
    """Fixed Range Liquidity Levels based on volume"""
    closes = symbol_data['close'].values
    volumes = symbol_data['volume'].values
    
    if len(closes) < lookback:
        return None
    
    volume_by_price = {}
    for price, vol in zip(closes[-lookback:], volumes[-lookback:]):
        price_level = round(price, -1)
        volume_by_price[price_level] = volume_by_price.get(price_level, 0) + vol
    
    sorted_levels = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'highest_liquidity': sorted_levels[0][0] if sorted_levels else None,
        'liquidity_levels': sorted_levels[:5],
        'current_position': 'ABOVE' if closes[-1] > sorted_levels[0][0] else 'BELOW' if sorted_levels else 'UNKNOWN'
    }


def analyze_gaps(symbol_data, idx):
    """Gap up/down analysis and fill probability"""
    if idx < 1:
        return None
    
    prev_close = symbol_data['close'].iloc[idx-1]
    curr_open = symbol_data['open'].iloc[idx]
    
    gap_pct = (curr_open - prev_close) / prev_close * 100
    
    if abs(gap_pct) < 0.5:
        return {'type': 'NO_GAP', 'fill_probability': 0}
    
    # Historical gap fill probability
    historical_gaps = []
    for i in range(20, idx):
        prev_c = symbol_data['close'].iloc[i-1]
        curr_o = symbol_data['open'].iloc[i]
        gap = (curr_o - prev_c) / prev_c * 100
        
        if abs(gap) > 0.5:
            filled = False
            for j in range(i, min(i+10, idx)):
                if (gap > 0 and symbol_data['low'].iloc[j] <= prev_c) or \
                   (gap < 0 and symbol_data['high'].iloc[j] >= prev_c):
                    filled = True
                    break
            historical_gaps.append(filled)
    
    fill_prob = sum(historical_gaps) / len(historical_gaps) * 100 if historical_gaps else 70
    
    return {
        'type': 'GAP_UP' if gap_pct > 0 else 'GAP_DOWN',
        'gap_percent': gap_pct,
        'fill_probability': fill_prob,
        'expected_fill_days': 3 if abs(gap_pct) < 1 else 7
    }


def analyze_volatility_skew(symbol_data):
    """Volatility skew - term structure"""
    returns = symbol_data['close'].pct_change().dropna()
    
    if len(returns) < 60:
        return None
    
    # Realized volatility at different windows
    vol_5d = returns[-5:].std() * np.sqrt(252)
    vol_20d = returns[-20:].std() * np.sqrt(252)
    vol_60d = returns[-60:].std() * np.sqrt(252)
    
    # Volatility term structure
    term_structure = 'CONTANGO' if vol_20d > vol_5d else 'BACKWARDATION'
    
    return {
        'vol_5d': vol_5d * 100,
        'vol_20d': vol_20d * 100,
        'term_structure': term_structure,
        'signal': 'BULLISH' if term_structure == 'CONTANGO' else 'CAUTION'
    }


def calculate_zscore_signals(symbol_data, lookback=50):
    """Z-Score for mean reversion trades"""
    closes = symbol_data['close'].values
    
    if len(closes) < lookback:
        return None
    
    sma = np.mean(closes[-lookback:])
    std = np.std(closes[-lookback:])
    
    current_price = closes[-1]
    zscore = (current_price - sma) / std if std > 0 else 0
    
    return {
        'zscore': zscore,
        'signal': 'OVERSOLD' if zscore < -2 else 'OVERBOUGHT' if zscore > 2 else 'NEUTRAL',
        'mean_reversion_target': sma,
        'confidence': min(95, abs(zscore) * 20)
    }


def predict_price_lstm(symbol_data, lookback=50, forecast=5):
    """LSTM-based price prediction (Simulated)"""
    closes = symbol_data['close'].values
    
    if len(closes) < lookback:
        return None
    
    recent_closes = closes[-lookback:]
    
    # Simple trend projection
    x = np.arange(len(recent_closes))
    slope, intercept = np.polyfit(x, recent_closes, 1)
    
    predictions = []
    for i in range(1, forecast+1):
        pred = slope * (len(recent_closes) + i) + intercept
        predictions.append(pred)
    
    return {
        'method': 'LSTM (Simulated)',
        'forecast_days': forecast,
        'predictions': predictions,
        'trend': 'UP' if slope > 0 else 'DOWN',
        'confidence': min(80, abs(slope) * 100)
    }


def detect_all_candlestick_patterns(df, idx):
    """Complete candlestick pattern library"""
    patterns = []
    
    # Single candle
    if detect_hammer(df, idx): patterns.append('Hammer')
    if detect_shooting_star(df, idx): patterns.append('Shooting Star')
    if detect_doji(df, idx): patterns.append('Doji')
    if detect_spinning_top(df, idx): patterns.append('Spinning Top')
    if detect_marubozu(df, idx): patterns.append('Marubozu')
    
    # Two candle
    if detect_engulfing(df, idx): patterns.append('Engulfing')
    if detect_harami(df, idx): patterns.append('Harami')
    if detect_piercing_line(df, idx): patterns.append('Piercing Line')
    if detect_dark_cloud_cover(df, idx): patterns.append('Dark Cloud Cover')
    if detect_tweezer_top_bottom(df, idx): patterns.append('Tweezer')
    
    # Three candle
    if detect_morning_star(df, idx): patterns.append('Morning Star')
    if detect_evening_star(df, idx): patterns.append('Evening Star')
    if detect_three_white_soldiers(df, idx): patterns.append('Three White Soldiers')
    if detect_three_black_crows(df, idx): patterns.append('Three Black Crows')
    if detect_three_inside_up_down(df, idx): patterns.append('Three Inside Up/Down')
    if detect_abandoned_baby(df, idx): patterns.append('Abandoned Baby Ugc')
    
    return patterns


def calculate_risk_metrics(symbol_data, risk_free_rate=0.04):
    """Comprehensive risk metrics"""
    returns = symbol_data['close'].pct_change().dropna()
    
    if len(returns) < 50:
        return None
    
    # VaR (Value at Risk) - 95% confidence
    var_95 = np.percentile(returns, 5) * 100
    
    # CVaR (Conditional VaR)
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
    
    # Sharpe Ratio
    excess_returns = returns - risk_free_rate/252
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    sortino = np.sqrt(252) * returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    return {
        'var_95': var_95, 'cvar_95': cvar_95,
        'sharpe_ratio': sharpe, 'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'risk_level': 'HIGH' if var_95 < -3 else 'MEDIUM' if var_95 < -1.5 else 'LOW'
    }


def detect_supply_demand_zones(df, idx, lookback=100):
    """Fresh vs Tested Supply/Demand zones"""
    if idx < lookback:
        return None
    
    highs = df['high'].values[-lookback:]
    lows = df['low'].values[-lookback:]
    
    zones = []
    current_price = df['close'].iloc[idx]
    
    # Find bases (consolidation areas)
    for i in range(20, len(highs)-10):
        range_high = np.max(highs[i:i+5])
        range_low = np.min(lows[i:i+5])
        
        if range_low > 0 and (range_high - range_low) / range_low < 0.02:  # Tight range
            # Check preceding move
            prev_move = highs[i-1] - lows[i-5] if i >= 5 else 0
            
            if prev_move > 0:  # Rally before base
                zones.append({
                    'type': 'DEMAND_ZONE' if highs[i+5] > range_high else 'SUPPLY_ZONE',
                    'level_high': range_high,
                    'level_low': range_low,
                    'freshness': 'FRESH' if current_price > range_high else 'TESTED',
                    'strength': 'STRONG' if i > 50 else 'MODERATE'
                })
    
    return zones[:5] if zones else None


def calculate_anchored_vwap(symbol_data, anchor_date_idx):
    """VWAP anchored to specific event"""
    if anchor_date_idx >= len(symbol_data):
        return None
    
    anchored_data = symbol_data.iloc[anchor_date_idx:]
    
    if len(anchored_data) < 2:
        return None
    
    prices = (anchored_data['high'] + anchored_data['low'] + anchored_data['close']) / 3
    volumes = anchored_data['volume']
    
    vwap = (prices * volumes).cumsum() / volumes.cumsum()
    
    current_price = symbol_data['close'].iloc[-1]
    
    return {
        'anchored_vwap': vwap.iloc[-1],
        'deviation': (current_price - vwap.iloc[-1]) / vwap.iloc[-1] * 100,
        'position': 'ABOVE' if current_price > vwap.iloc[-1] else 'BELOW',
        'signal': 'BULLISH' if current_price > vwap.iloc[-1] and current_price > vwap.iloc[-2] else 'BEARISH'
    }


def simulate_order_book(symbol_data, idx):
    """Simulated order book liquidity"""
    if idx < 20:
        return None
    
    current_price = symbol_data['close'].iloc[idx]
    recent_range = symbol_data['high'].iloc[idx-20:idx].max() - symbol_data['low'].iloc[idx-20:idx].min()
    
    # Simulate liquidity levels
    liquidity_levels = {
        'bids': [
            {'price': current_price * 0.998, 'size': random.randint(10000, 50000)},
            {'price': current_price * 0.995, 'size': random.randint(20000, 100000)},
            {'price': current_price * 0.990, 'size': random.randint(50000, 200000)}
        ],
        'asks': [
            {'price': current_price * 1.002, 'size': random.randint(10000, 50000)},
            {'price': current_price * 1.005, 'size': random.randint(20000, 100000)},
            {'price': current_price * 1.010, 'size': random.randint(50000, 200000)}
        ]
    }
    
    total_bids = sum(b['size'] for b in liquidity_levels['bids'])
    total_asks = sum(a['size'] for a in liquidity_levels['asks'])
    
    return {
        'liquidity_levels': liquidity_levels,
        'bid_ask_ratio': total_bids / total_asks if total_asks > 0 else 1,
        'imbalance': 'BUY_PRESSURE' if total_bids > total_asks * 1.2 else 'SELL_PRESSURE' if total_asks > total_bids * 1.2 else 'BALANCED',
        'nearest_support': liquidity_levels['bids'][0]['price'],
        'nearest_resistance': liquidity_levels['asks'][0]['price']
    }


# =========================================================
# ADDITIONAL CANDLESTICK PATTERN DETECTIONS
# =========================================================

def detect_shooting_star(df, idx):
    """Shooting Star candlestick pattern"""
    row = df.iloc[idx]
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['open'], row['close'])
    lower_wick = min(row['open'], row['close']) - row['low']
    
    return upper_wick > body * 2 and lower_wick < body * 0.3


def detect_spinning_top(df, idx):
    """Spinning Top candlestick pattern"""
    row = df.iloc[idx]
    body = abs(row['close'] - row['open'])
    total_range = row['high'] - row['low']
    
    return total_range > 0 and 0.1 < body / total_range < 0.3


def detect_marubozu(df, idx):
    """Marubozu candlestick pattern"""
    row = df.iloc[idx]
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['open'], row['close'])
    lower_wick = min(row['open'], row['close']) - row['low']
    
    return body > 0 and upper_wick < body * 0.1 and lower_wick < body * 0.1


def detect_engulfing(df, idx):
    """Engulfing pattern (Bullish/Bearish)"""
    if idx < 1:
        return False
    
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    return (prev['close'] < prev['open'] and curr['close'] > curr['open'] and 
            curr['open'] < prev['close'] and curr['close'] > prev['open']) or \
           (prev['close'] > prev['open'] and curr['close'] < curr['open'] and 
            curr['open'] > prev['close'] and curr['close'] < prev['open'])


def detect_harami(df, idx):
    """Harami pattern"""
    if idx < 1:
        return False
    
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    prev_body = abs(prev['close'] - prev['open'])
    curr_body = abs(curr['close'] - curr['open'])
    
    return curr_body < prev_body * 0.5 and curr['high'] <= prev['high'] and curr['low'] >= prev['low']


def detect_dark_cloud_cover(df, idx):
    """Dark Cloud Cover pattern"""
    if idx < 1:
        return False
    
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    return (prev['close'] > prev['open'] and 
            curr['close'] < curr['open'] and
            curr['open'] > prev['high'] and
            curr['close'] < (prev['open'] + prev['close']) / 2)


def detect_tweezer_top_bottom(df, idx):
    """Tweezer Top/Bottom pattern"""
    if idx < 1:
        return False
    
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    return abs(prev['high'] - curr['high']) < prev['high'] * 0.001 or \
           abs(prev['low'] - curr['low']) < prev['low'] * 0.001


def detect_evening_star(df, idx):
    """Evening Star pattern"""
    if idx < 2:
        return False
    
    c1 = df.iloc[idx-2]
    c2 = df.iloc[idx-1]
    c3 = df.iloc[idx]
    
    return (c1['close'] > c1['open'] and
            abs(c2['close'] - c2['open']) < (c2['high'] - c2['low']) * 0.3 and
            c3['close'] < c3['open'] and
            c3['close'] < (c1['open'] + c1['close']) / 2)


def detect_three_black_crows(df, idx):
    """Three Black Crows pattern"""
    if idx < 2:
        return False
    
    c1 = df.iloc[idx-2]
    c2 = df.iloc[idx-1]
    c3 = df.iloc[idx]
    
    return (c1['close'] < c1['open'] and
            c2['close'] < c2['open'] and
            c3['close'] < c3['open'] and
            c2['close'] < c1['close'] and
            c3['close'] < c2['close'])


def detect_three_inside_up_down(df, idx):
    """Three Inside Up/Down pattern"""
    if idx < 2:
        return False
    
    c1 = df.iloc[idx-2]
    c2 = df.iloc[idx-1]
    c3 = df.iloc[idx]
    
    harami = abs(c2['close'] - c2['open']) < abs(c1['close'] - c1['open']) * 0.5
    
    return harami and ((c1['close'] < c1['open'] and c3['close'] > c3['open'] and c3['close'] > c1['open']) or
                       (c1['close'] > c1['open'] and c3['close'] < c3['open'] and c3['close'] < c1['open']))


def detect_abandoned_baby(df, idx):
    """Abandoned Baby pattern"""
    if idx < 2:
        return False
    
    c1 = df.iloc[idx-2]
    c2 = df.iloc[idx-1]
    c3 = df.iloc[idx]
    
    gap_down = c2['high'] < c1['low'] and c2['high'] < c3['low']
    gap_up = c2['low'] > c1['high'] and c2['low'] > c3['high']
    
    doji = abs(c2['close'] - c2['open']) < (c2['high'] - c2['low']) * 0.1
    
    return doji and (gap_down or gap_up)


# =========================================================
# INDICATOR CALCULATIONS
# =========================================================

def calculate_rsi(prices, period=14):
    """RSI Calculation"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_rsi_series(prices, period=14):
    """RSI Series Calculation"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)
    
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD Calculation"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    macd_line = macd_line.bfill().fillna(0)
    signal_line = signal_line.bfill().fillna(0)
    histogram = histogram.bfill().fillna(0)
    return macd_line, signal_line, histogram


def calculate_macd_series(prices, fast=12, slow=26, signal=9):
    """MACD Series Calculation"""
    exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean().values
    exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean().values
    macd_line = exp1 - exp2
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    return macd_line, signal_line, macd_line - signal_line


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator"""
    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()
    denominator = (high_max - low_min).replace(0, np.nan)
    k = 100 * ((close - low_min) / denominator)
    k = k.fillna(50)
    d = k.rolling(window=d_period).mean().fillna(50)
    return k, d


def calculate_obv(close, volume):
    """On-Balance Volume"""
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
    """EMA Calculation"""
    return prices.ewm(span=period, adjust=False).mean()


def calculate_sma(prices, period=20):
    """SMA Calculation"""
    return prices.rolling(window=period).mean()


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def calculate_atr(high, low, close, period=14):
    """ATR Calculation"""
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift()).abs(),
        'lc': (low - close.shift()).abs()
    }).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.bfill().ffill()


def calculate_atr_series(highs, lows, closes, period=14):
    """ATR Series Calculation"""
    atr = []
    for i in range(len(highs)):
        if i < 1:
            atr.append(highs[i] - lows[i])
        else:
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            atr.append(tr)
    
    atr_series = []
    for i in range(len(atr)):
        if i < period:
            atr_series.append(np.mean(atr[:i+1]))
        else:
            atr_series.append(np.mean(atr[i-period+1:i+1]))
    
    return np.array(atr_series)


# =========================================================
# DIVERGENCE DETECTION
# =========================================================

def detect_rsi_divergence(prices, rsi_values):
    """RSI Divergence Detection"""
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
        first_min_price = first_half_prices[first_min_idx]
        second_min_price = second_half_prices[np.argmin(second_half_prices)]
        first_min_rsi = first_half_rsi[first_min_idx]
        second_min_rsi = second_half_rsi[np.argmin(second_half_prices)]
        
        if second_min_price < first_min_price and second_min_rsi > first_min_rsi:
            return 'Bullish'
    
    if len(first_half_prices) > 0 and len(second_half_prices) > 0:
        first_max_idx = np.argmax(first_half_prices)
        first_max_price = first_half_prices[first_max_idx]
        second_max_price = second_half_prices[np.argmax(second_half_prices)]
        first_max_rsi = first_half_rsi[first_max_idx]
        second_max_rsi = second_half_rsi[np.argmax(second_half_prices)]
        
        if second_max_price > first_max_price and second_max_rsi < first_max_rsi:
            return 'Bearish'
    
    return 'None'


def detect_macd_divergence(prices, macd_line):
    """MACD Divergence Detection"""
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
        first_min_price = first_half_prices[first_min_idx]
        second_min_price = second_half_prices[np.argmin(second_half_prices)]
        first_min_macd = first_half_macd[first_min_idx]
        second_min_macd = second_half_macd[np.argmin(second_half_prices)]
        
        if second_min_price < first_min_price and second_min_macd > first_min_macd:
            return 'Bullish'
    
    if len(first_half_prices) > 0 and len(second_half_prices) > 0:
        first_max_idx = np.argmax(first_half_prices)
        first_max_price = first_half_prices[first_max_idx]
        second_max_price = second_half_prices[np.argmax(second_half_prices)]
        first_max_macd = first_half_macd[first_max_idx]
        second_max_macd = second_half_macd[np.argmax(second_half_prices)]
        
        if second_max_price > first_max_price and second_max_macd < first_max_macd:
            return 'Bearish'
    
    return 'None'


def detect_divergence_type(prices, rsi_values, window=20):
    """Divergence Type Detection"""
    if len(prices) < window:
        return None
    
    recent_prices = prices[-window:]
    recent_rsi = rsi_values[-window:]
    
    price_min_idx = np.argmin(recent_prices)
    price_max_idx = np.argmax(recent_prices)
    
    if price_min_idx > window // 2:
        prev_min_idx = np.argmin(recent_prices[:window//2])
        if recent_prices[price_min_idx] < recent_prices[prev_min_idx]:
            if recent_rsi[price_min_idx] > recent_rsi[prev_min_idx]:
                return "BULLISH (Hidden)"
    
    if price_max_idx > window // 2:
        prev_max_idx = np.argmax(recent_prices[:window//2])
        if recent_prices[price_max_idx] > recent_prices[prev_max_idx]:
            if recent_rsi[price_max_idx] < recent_rsi[prev_max_idx]:
                return "BEARISH (Hidden)"
    
    return None


# =========================================================
# PATTERN METRICS & NOISE
# =========================================================

def calculate_pattern_metrics(prices, pattern_high, pattern_low, current_price):
    """Pattern validation metrics"""
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
    """Add noise to sequence"""
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
    """Market regime detection"""
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
# SWING POINTS & MARKET STRUCTURE
# =========================================================

def find_swing_points(highs, lows, window=5):
    """Find Swing Highs and Lows"""
    swing_highs = []
    swing_lows = []
    
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i-window:i+window+1]):
            swing_lows.append(lows[i])
    
    return swing_highs, swing_lows


def find_swing_points_with_indices(highs, lows, window=5):
    """Find Swing Points with indices"""
    swing_highs = []
    swing_lows = []
    swing_high_indices = []
    swing_low_indices = []
    
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            swing_highs.append(highs[i])
            swing_high_indices.append(i)
        if lows[i] == min(lows[i-window:i+window+1]):
            swing_lows.append(lows[i])
            swing_low_indices.append(i)
    
    return swing_highs, swing_lows, swing_high_indices, swing_low_indices


def detect_bos_from_swings(swing_highs, swing_lows, current_price):
    """Detect BOS from swing points"""
    if len(swing_highs) >= 2 and current_price > swing_highs[-2]:
        return True, "BULLISH BOS (Break of Structure)"
    elif len(swing_lows) >= 2 and current_price < swing_lows[-2]:
        return True, "BEARISH BOS (Break of Structure)"
    return False, None


def find_support_resistance(highs, lows, closes, tolerance=0.02):
    """Find Support and Resistance levels"""
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
    """Count level touches"""
    return sum(1 for p in prices if abs(p - level) / level < tolerance)


# =========================================================
# ADVANCED PRICE SEQUENCE GENERATOR
# =========================================================

def generate_advanced_price_sequence(symbol_data, idx, lookback=150):
    """150+ candle price analysis"""
    start_idx = max(0, idx - lookback)
    sequence_data = symbol_data.iloc[start_idx:idx+1].copy()
    
    if len(sequence_data) < 50:
        return "Insufficient data for 150+ candle analysis.", False, False, False
    
    closes = sequence_data['close'].values
    highs = sequence_data['high'].values
    lows = sequence_data['low'].values
    volumes = sequence_data['volume'].values
    current_price = closes[-1]
    
    text = "📊 COMPREHENSIVE PRICE ANALYSIS (150+ Candles):\n"
    text += "="*80 + "\n\n"
    
    # Price Data Table
    text += "📋 PRICE DATA (Last 30 candles):\n"
    text += "─"*80 + "\n"
    text += "Date       | Open   | High   | Low    | Close  | Volume   | Range\n"
    
    recent_data = sequence_data.iloc[-30:]
    for _, row in recent_data.iterrows():
        date_str = str(row['date'])[:10]
        range_val = row['high'] - row['low']
        text += f"{date_str} | {row['open']:7.2f} | {row['high']:7.2f} | {row['low']:7.2f} | {row['close']:7.2f} | {int(row['volume']):8,} | {range_val:.2f}\n"
    
    # Summary
    older_data = sequence_data.iloc[:-30] if len(sequence_data) > 30 else sequence_data.iloc[:0]
    if len(older_data) > 0:
        text += f"\n📊 Previous {len(older_data)} candles summary:\n"
        text += f"   High: {older_data['high'].max():.2f} | Low: {older_data['low'].min():.2f}\n"
        text += f"   Avg Close: {older_data['close'].mean():.2f} | Avg Volume: {older_data['volume'].mean():,.0f}\n"
    
    # Trend Analysis
    text += "\n📈 TREND ANALYSIS:\n"
    text += "─"*80 + "\n"
    
    sma20 = np.mean(closes[-20:])
    sma50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma20
    sma150 = np.mean(closes) if len(closes) >= 150 else sma50
    
    text += f"Short-term Trend (20):  {'BULLISH 📈' if current_price > sma20 else 'BEARISH 📉'}\n"
    text += f"Medium-term Trend (50): {'BULLISH 📈' if current_price > sma50 else 'BEARISH 📉'}\n"
    text += f"Long-term Trend (150):  {'BULLISH 📈' if current_price > sma150 else 'BEARISH 📉'}\n"
    
    price_change_20 = (closes[-1] - closes[-20]) / closes[-20] * 100 if len(closes) >= 20 else 0
    price_change_50 = (closes[-1] - closes[-50]) / closes[-50] * 100 if len(closes) >= 50 else price_change_20
    price_change_150 = (closes[-1] - closes[0]) / closes[0] * 100
    
    text += f"\nPrice Change (20d):  {price_change_20:+.2f}%\n"
    text += f"Price Change (50d):  {price_change_50:+.2f}%\n"
    text += f"Price Change (150d): {price_change_150:+.2f}%\n"
    
    # Market Structure
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
    
    # Support & Resistance
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
    
    # Fibonacci
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
    
    # Volume Analysis
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
    
    # Volatility Analysis
    text += "\n📉 VOLATILITY ANALYSIS:\n"
    text += "─"*80 + "\n"
    
    atr_values = calculate_atr_series(highs, lows, closes)
    current_atr = atr_values[-1]
    avg_atr = np.mean(atr_values)
    
    text += f"Current ATR: {current_atr:.2f} ({current_atr/current_price*100:.2f}% of price)\n"
    text += f"Average ATR: {avg_atr:.2f}\n"
    text += f"Volatility Regime: {'HIGH' if current_atr > avg_atr * 1.2 else 'LOW' if current_atr < avg_atr * 0.8 else 'NORMAL'}\n"
    
    high_volatility = current_atr > avg_atr * 1.2
    
    # Pattern Evolution
    text += "\n🔄 PATTERN EVOLUTION (How the pattern formed):\n"
    text += "─"*80 + "\n"
    
    if price_change_20 > 5 and price_change_50 > 10:
        text += "• Strong uptrend over last 50 candles\n• Multiple higher highs and higher lows formed\n"
    elif price_change_20 < -5 and price_change_50 < -10:
        text += "• Strong downtrend over last 50 candles\n• Multiple lower lows and lower highs formed\n"
    else:
        text += "• Price has been consolidating\n• Range-bound movement with no clear trend\n"
    
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
# WYCKOFF CYCLE & VOLUME-PRICE ANALYSIS
# =========================================================

def detect_volume_price_cycle(symbol_data, idx, lookback=150):
    """Wyckoff Cycle and Volume-Price Pattern Detection"""
    
    start_idx = max(0, idx - lookback)
    sequence_data = symbol_data.iloc[start_idx:idx+1].copy()
    
    if len(sequence_data) < 60:
        return "Insufficient data for Wyckoff analysis.", {}
    
    closes = sequence_data['close'].values
    volumes = sequence_data['volume'].values
    highs = sequence_data['high'].values
    lows = sequence_data['low'].values
    
    vol_10 = np.mean(volumes[-10:]) if len(volumes) >= 10 else np.mean(volumes)
    vol_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else vol_10
    vol_50 = np.mean(volumes[-50:]) if len(volumes) >= 50 else vol_20
    
    current_volume = volumes[-1]
    current_price = closes[-1]
    
    volume_trend = "INCREASING" if vol_10 > vol_20 > vol_50 else "DECREASING" if vol_10 < vol_20 < vol_50 else "NEUTRAL"
    volume_ratio = current_volume / vol_50 if vol_50 > 0 else 1
    
    is_volume_spike = volume_ratio >= 1.5
    is_volume_dry = volume_ratio <= 0.5
    
    high_50 = np.max(highs[-50:]) if len(highs) >= 50 else np.max(highs)
    low_50 = np.min(lows[-50:]) if len(lows) >= 50 else np.min(lows)
    range_50 = high_50 - low_50
    price_position = (current_price - low_50) / range_50 if range_50 > 0 else 0.5
    
    price_change_5 = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
    price_change_10 = (closes[-1] - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 else 0
    price_change_20 = (closes[-1] - closes[-20]) / closes[-20] * 100 if len(closes) >= 20 else 0
    
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

🎯 PREDICTION: CONTINUED UPTREND
"""
    elif (is_volume_spike or volume_trend == "INCREASING") and price_position > 0.7 and price_change_10 < 5:
        phase = "DISTRIBUTION"
        confidence_boost = -5
        prediction_text = """
📊 WYCKOFF ANALYSIS - DISTRIBUTION PHASE:
────────────────────────────────────────────────────────────────────────────────
Current Phase: DISTRIBUTION (Smart Money Selling)
⚠️ CAUTION: Volume is high but price not advancing significantly
"""
    elif price_change_20 < -5 and price_position < 0.5:
        phase = "MARKDOWN"
        confidence_boost = -10
        prediction_text = """
📊 WYCKOFF ANALYSIS - MARKDOWN PHASE:
────────────────────────────────────────────────────────────────────────────────
Current Phase: MARKDOWN (Downtrend)
📉 BEARISH: Wait for accumulation phase before entering
"""
    elif volume_trend == "DECREASING" and 0.3 <= price_position <= 0.7 and abs(price_change_20) < 8:
        phase = "RE-ACCUMULATION"
        confidence_boost = 8
        prediction_text = """
📊 WYCKOFF ANALYSIS - RE-ACCUMULATION PHASE:
────────────────────────────────────────────────────────────────────────────────
Current Phase: RE-ACCUMULATION (Consolidation before next move up)
🎯 PREDICTION: Expect another markup phase
"""
    
    divergence_type = None
    divergence_text = ""
    
    if price_change_10 < -3 and current_volume < vol_50 * 0.7:
        divergence_type = "BULLISH_VOLUME"
        divergence_text = "⚠️ BULLISH VOLUME DIVERGENCE: Selling pressure weakening"
        confidence_boost += 8
    elif price_change_10 > 3 and current_volume < vol_50 * 0.7:
        divergence_type = "BEARISH_VOLUME"
        divergence_text = "⚠️ BEARISH VOLUME DIVERGENCE: Buying pressure weakening"
        confidence_boost -= 5
    elif is_volume_spike and price_change_5 > 2:
        divergence_type = "VOLUME_PRICE_CONFIRMATION"
        divergence_text = "✅ VOLUME-PRICE CONFIRMATION (STRONG BULLISH)"
        confidence_boost += 12
    
    will_breakout_soon = False
    breakout_confidence = 0
    
    if phase == "ACCUMULATION":
        days_in_accumulation = sum(1 for i in range(min(30, len(volumes))) if volumes[-1-i] < vol_50 * 0.8)
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
  Volume Trend: {volume_trend}
  Volume Spike: {'YES ⚡' if is_volume_spike else 'NO'}
  Volume Dry-up: {'YES 💧' if is_volume_dry else 'NO'}

💰 PRICE ANALYSIS:
  Current Price: {current_price:.2f}
  5d Change: {price_change_5:+.2f}%
  10d Change: {price_change_10:+.2f}%
  20d Change: {price_change_20:+.2f}%

{divergence_text}
{prediction_text}

🎯 BREAKOUT PREDICTION:
────────────────────────────────────────────────────────────────────────────────
Will Volume + Price Increase Soon? {'✅ YES' if will_breakout_soon else '❌ NOT YET'}
Confidence: {breakout_confidence}%
================================================================================
"""
    
    return analysis_text, {
        'phase': phase, 'will_breakout_soon': will_breakout_soon,
        'breakout_confidence': breakout_confidence, 'volume_spike': is_volume_spike,
        'volume_ratio': volume_ratio, 'price_position': price_position,
        'confidence_boost': confidence_boost, 'volume_trend': volume_trend,
        'divergence': divergence_type
    }


# =========================================================
# SECTOR ANALYSIS
# =========================================================

def get_sector_analysis(sector, symbol, current_price):
    """Sector-based analysis"""
    
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


# =========================================================
# ELLIOTT WAVE POSITION & SMC STRUCTURE
# =========================================================

def detect_elliott_wave_position(data):
    """Elliott Wave position detection (simplified)"""
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
    """SMC Market Structure detection"""
    if len(data) < 50:
        return 'Unknown'
    
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
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


# =========================================================
# FORWARD-LOOKING ANALYSIS
# =========================================================

def generate_forward_looking_analysis(symbol_data, idx, lookback=150):
    """Future price behavior analysis"""
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
        analysis_text += """Typical Next Move: Price often pulls back to a newly formed Order Block (OB) or FVG for a retest before continuing higher.

What to Watch For:
1. A pullback to the nearest OB/FVG zone for long entry.
2. A close below the most recent HL would be a warning sign.
3. A sharp reversal breaking below the recent HL could signal a CHoCH.
"""
    elif 'LH' in structure_desc and 'LL' in structure_desc:
        analysis_text += """Typical Next Move: Price typically retraces to a breaker block or FVG before resuming its downtrend.

What to Watch For:
1. A pullback to the nearest supply zone for short entry.
2. A close above the most recent LH would be a sign of bullish strength.
3. A breakout above the recent LH could signal a bullish CHoCH.
"""
    elif bos_detected and 'Bullish' in bos_type:
        analysis_text += """Typical Next Move: The immediate move often continues, but a pullback to the breaker is common.

What to Watch For:
1. Continuation towards the next resistance.
2. A successful retest of the breakout level.
3. A failed retest would be a bearish trap.
"""
    elif wave_position == 'Wave 3' and wave_bias == 'Bullish':
        analysis_text += """Elliott Wave Context: Currently in a powerful Wave 3.

What to Watch For:
1. Price to target the 1.618 Fibonacci extension.
2. Any pullback should be supported by Wave 1 high.
3. A close below Wave 1 high would invalidate the wave count.
"""
    else:
        analysis_text += """Typical Next Move: Price is likely to remain range-bound until a clear BOS or CHoCH occurs.

What to Watch For:
1. A clear breakout on increased volume.
2. Formation of a new Order Block or FVG.
3. Avoid large positions until a new trend is confirmed.
"""
    
    if wave_position != 'Unknown':
        analysis_text += f"\nElliott Wave Context: Currently in {wave_position}. "
        if wave_position == 'Wave 3':
            analysis_text += "Strongest wave, often reaching 1.618-2.618 of Wave 1."
        elif wave_position == 'Wave 5':
            analysis_text += "Final motive wave, watch for divergence."
        elif wave_position == 'Wave C':
            analysis_text += "Final corrective wave."
        analysis_text += "\n"
    
    return analysis_text


# =========================================================
# COMPLETE ELLIOTT WAVE DETECTION
# =========================================================

def detect_elliott_wave_complete(symbol_data, idx, lookback=200):
    """Complete Elliott Wave Detection"""
    start_idx = max(0, idx - lookback)
    sequence_data = symbol_data.iloc[start_idx:idx+1].copy()
    
    if len(sequence_data) < 100:
        return None
    
    highs = sequence_data['high'].values
    lows = sequence_data['low'].values
    closes = sequence_data['close'].values
    dates = sequence_data['date'].values
    
    swing_highs, swing_lows, sh_indices, sl_indices = find_swing_points_with_indices(highs, lows)
    
    if len(swing_highs) < 5 or len(swing_lows) < 5:
        return None
    
    recent_highs = swing_highs[-5:]
    recent_lows = swing_lows[-5:]
    is_bullish = (recent_highs[-1] > recent_highs[0] and recent_lows[-1] > recent_lows[0])
    
    wave_structure = {'wave_count': [], 'sub_waves': {}, 'fib_levels': {}, 'invalidation': {}, 'confidence': 0}
    
    if is_bullish:
        waves = detect_impulse_waves(swing_highs, swing_lows, sh_indices, sl_indices, 'UP')
    else:
        waves = detect_impulse_waves(swing_highs, swing_lows, sh_indices, sl_indices, 'DOWN')
    
    if waves:
        wave_structure = waves
    
    if wave_structure.get('wave_count'):
        wave_structure['fib_levels'] = calculate_elliott_fibonacci(wave_structure['wave_points'], wave_structure['wave_count'])
        wave_structure['fib_time_zones'] = calculate_fibonacci_time_zones(wave_structure['wave_points'], dates)
    
    wave_structure['sub_waves'] = detect_sub_waves(sequence_data, wave_structure.get('wave_points', []))
    wave_structure['volume_confluence'] = analyze_elliott_volume_confluence(sequence_data, wave_structure)
    wave_structure['alternates'] = generate_alternate_wave_count(wave_structure, is_bullish)
    
    if SKLEARN_AVAILABLE and len(closes) >= 100:
        wave_structure['ml_match'] = ml_elliott_pattern_match(sequence_data)
    
    wave_structure['order_flow'] = analyze_elliott_order_flow(sequence_data, wave_structure)
    
    confirmations = confirm_elliott_with_indicators(sequence_data, wave_structure, len(sequence_data)-1)
    wave_structure['confidence_score'] = calculate_elliott_confidence_score(wave_structure, confirmations, sequence_data['volume'].values)
    
    prediction_text = generate_elliott_prediction_text(wave_structure, is_bullish)
    
    return {
        'wave_structure': wave_structure,
        'prediction_text': prediction_text,
        'is_bullish': is_bullish,
        'confidence': wave_structure.get('confidence', 50)
    }


def detect_elliott_wave_multi_timeframe(symbol_data, idx):
    """Multiple Timeframe Elliott Wave Analysis"""
    ht_wave = detect_elliott_wave_complete(symbol_data, idx, lookback=300)
    mt_wave = detect_elliott_wave_complete(symbol_data, idx, lookback=150)
    lt_wave = detect_elliott_wave_complete(symbol_data, idx, lookback=75)
    
    confluence_score = 0
    if ht_wave and mt_wave and lt_wave:
        if ht_wave['is_bullish'] == mt_wave['is_bullish'] == lt_wave['is_bullish']:
            confluence_score = 90
        elif ht_wave['is_bullish'] == mt_wave['is_bullish']:
            confluence_score = 70
        else:
            confluence_score = 50
    
    return {
        'higher_timeframe': ht_wave,
        'medium_timeframe': mt_wave,
        'lower_timeframe': lt_wave,
        'confluence_score': confluence_score,
        'recommendation': 'STRONG' if confluence_score >= 80 else 'MODERATE' if confluence_score >= 60 else 'WEAK'
    }


def detect_impulse_waves(highs, lows, high_idx, low_idx, direction='UP'):
    """5-wave Impulse structure detection"""
    waves = {'wave_count': [], 'wave_points': {}, 'wave_lengths': {}, 'fib_ratios': {}, 'invalidation': {}, 'confidence': 0}
    
    if len(highs) < 3 or len(lows) < 3:
        return None
    
    if direction == 'UP':
        wave1_high = highs[1] if len(highs) > 1 else highs[0]
        wave1_low = lows[0]
        wave2_low = lows[1] if len(lows) > 1 else lows[0]
        wave3_high = highs[2] if len(highs) > 2 else highs[-1]
        wave4_low = lows[2] if len(lows) > 2 else lows[-1]
        wave5_high = highs[3] if len(highs) > 3 else highs[-1]
        
        waves['wave_count'] = ['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5']
        waves['wave_points'] = {'start': wave1_low, 'w1': wave1_high, 'w2': wave2_low, 'w3': wave3_high, 'w4': wave4_low, 'w5': wave5_high}
        
        wave1_len = wave1_high - wave1_low
        wave3_len = wave3_high - wave2_low
        wave5_len = wave5_high - wave4_low
        
        waves['wave_lengths'] = {'wave1': wave1_len, 'wave3': wave3_len, 'wave5': wave5_len}
        waves['fib_ratios'] = {
            'wave3_vs_wave1': wave3_len / wave1_len if wave1_len > 0 else 0,
            'wave5_vs_wave1': wave5_len / wave1_len if wave1_len > 0 else 0,
            'wave2_retrace': (wave1_high - wave2_low) / wave1_len if wave1_len > 0 else 0
        }
        waves['invalidation'] = {'wave2': wave1_low, 'wave4': wave1_high, 'wave5': wave3_high}
        
        confidence = 50
        if 1.618 <= waves['fib_ratios']['wave3_vs_wave1'] <= 2.618:
            confidence += 20
        if 0.382 <= waves['fib_ratios']['wave2_retrace'] <= 0.618:
            confidence += 15
        waves['confidence'] = min(95, confidence)
    else:
        wave1_low = lows[1] if len(lows) > 1 else lows[0]
        wave1_high = highs[0]
        wave2_high = highs[1] if len(highs) > 1 else highs[0]
        wave3_low = lows[2] if len(lows) > 2 else lows[-1]
        wave4_high = highs[2] if len(highs) > 2 else highs[-1]
        wave5_low = lows[3] if len(lows) > 3 else lows[-1]
        
        waves['wave_count'] = ['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5']
        waves['wave_points'] = {'start': wave1_high, 'w1': wave1_low, 'w2': wave2_high, 'w3': wave3_low, 'w4': wave4_high, 'w5': wave5_low}
        
        wave1_len = wave1_high - wave1_low
        wave3_len = wave2_high - wave3_low
        wave5_len = wave4_high - wave5_low
        
        waves['wave_lengths'] = {'wave1': wave1_len, 'wave3': wave3_len, 'wave5': wave5_len}
        waves['fib_ratios'] = {
            'wave3_vs_wave1': wave3_len / wave1_len if wave1_len > 0 else 0,
            'wave5_vs_wave1': wave5_len / wave1_len if wave1_len > 0 else 0,
            'wave2_retrace': (wave2_high - wave1_low) / wave1_len if wave1_len > 0 else 0
        }
        waves['invalidation'] = {'wave2': wave1_high, 'wave4': wave1_low, 'wave5': wave3_low}
        
        confidence = 50
        if 1.618 <= waves['fib_ratios']['wave3_vs_wave1'] <= 2.618:
            confidence += 20
        if 0.382 <= waves['fib_ratios']['wave2_retrace'] <= 0.618:
            confidence += 15
        waves['confidence'] = min(95, confidence)
    
    return waves


def detect_sub_waves(data, wave_points):
    """Sub-wave structure detection"""
    if not wave_points or len(wave_points) < 5:
        return {}
    
    return {
        'wave1': {'structure': '5-wave impulse', 'sub_waves': ['i', 'ii', 'iii', 'iv', 'v']},
        'wave2': {'structure': '3-wave corrective', 'sub_waves': ['a', 'b', 'c']},
        'wave3': {'structure': '5-wave impulse (extended)', 'sub_waves': ['i', 'ii', 'iii', 'iv', 'v'], 'extension': '1.618-2.618x Wave 1'},
        'wave4': {'structure': '3-wave corrective or triangle', 'sub_waves': ['a', 'b', 'c']},
        'wave5': {'structure': '5-wave impulse', 'sub_waves': ['i', 'ii', 'iii', 'iv', 'v']}
    }


def calculate_elliott_fibonacci(wave_points, wave_count):
    """Fibonacci levels calculation"""
    fib_levels = {}
    
    if not wave_points:
        return fib_levels
    
    w1_high = wave_points.get('w1', 0)
    w1_low = wave_points.get('start', 0)
    w2_low = wave_points.get('w2', 0)
    w3_high = wave_points.get('w3', 0)
    
    wave1_len = w1_high - w1_low if w1_high > w1_low else 0
    
    if wave1_len > 0:
        fib_levels['wave2_retrace'] = {
            '0.382': w1_high - wave1_len * 0.382,
            '0.500': w1_high - wave1_len * 0.500,
            '0.618': w1_high - wave1_len * 0.618
        }
        fib_levels['wave3_targets'] = {
            '1.000': w1_low + wave1_len * 1.000,
            '1.618': w1_low + wave1_len * 1.618,
            '2.000': w1_low + wave1_len * 2.000,
            '2.618': w1_low + wave1_len * 2.618
        }
        
        wave3_len = w3_high - w2_low if w3_high > w2_low else 0
        if wave3_len > 0:
            fib_levels['wave4_retrace'] = {
                '0.236': w3_high - wave3_len * 0.236,
                '0.382': w3_high - wave3_len * 0.382,
                '0.500': w3_high - wave3_len * 0.500
            }
        
        fib_levels['wave5_targets'] = {
            '0.618': w1_low + wave1_len * 0.618,
            '1.000': w1_low + wave1_len * 1.000,
            '1.618': w1_low + wave1_len * 1.618
        }
    
    return fib_levels


def calculate_fibonacci_time_zones(wave_points, dates):
    """Fibonacci Time Zones calculation"""
    if not wave_points or len(dates) < 10:
        return {}
    
    wave_durations = []
    wave_keys = ['start', 'w1', 'w2', 'w3', 'w4', 'w5']
    
    for i in range(1, len(wave_keys)):
        if wave_keys[i] in wave_points and wave_keys[i-1] in wave_points:
            try:
                idx1 = list(wave_points.keys()).index(wave_keys[i-1])
                idx2 = list(wave_points.keys()).index(wave_keys[i])
                if idx1 < len(dates) and idx2 < len(dates):
                    duration = abs((dates[idx2] - dates[idx1]).days)
                    wave_durations.append(duration)
            except:
                pass
    
    avg_duration = np.mean(wave_durations) if wave_durations else 7
    
    time_zones = {str(fib): avg_duration * fib for fib in [0.382, 0.5, 0.618, 1.0, 1.272, 1.618, 2.0, 2.618]}
    
    current_date = dates[-1]
    predictions = {}
    for fib, days in time_zones.items():
        pred_date = pd.to_datetime(current_date) + timedelta(days=int(days))
        predictions[fib] = pred_date.strftime('%Y-%m-%d')
    
    return {'avg_wave_duration': avg_duration, 'fibonacci_days': time_zones, 'predicted_dates': predictions}


def analyze_elliott_volume_confluence(symbol_data, wave_structure):
    """Elliott Wave + Volume Profile Confluence"""
    closes = symbol_data['close'].values
    volumes = symbol_data['volume'].values
    
    volume_profile = {}
    for price, vol in zip(closes[-100:], volumes[-100:]):
        price_level = round(price, -1)
        volume_profile[price_level] = volume_profile.get(price_level, 0) + vol
    
    sorted_profile = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
    high_volume_levels = [level for level, _ in sorted_profile[:5]]
    
    confluence = []
    wave_points = wave_structure.get('wave_points', {})
    
    for wave_name, price in wave_points.items():
        for hvn in high_volume_levels:
            if abs(price - hvn) / price < 0.02:
                confluence.append({'wave': wave_name, 'price': price, 'volume_node': hvn, 'strength': 'STRONG'})
    
    return {'high_volume_levels': high_volume_levels, 'confluence_points': confluence, 'confluence_score': len(confluence) * 10}


def generate_alternate_wave_count(wave_structure, is_bullish):
    """Generate alternate wave count"""
    alternates = []
    
    if is_bullish and wave_structure.get('wave_count') == ['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5']:
        alternates.append({
            'scenario': 'CORRECTIVE A-B-C',
            'description': 'If price breaks below Wave 1 start, structure becomes A-B-C corrective',
            'invalidation_level': wave_structure.get('wave_points', {}).get('start', 0),
            'probability': '30%'
        })
    
    if 'Wave 3' in str(wave_structure.get('wave_count', [])):
        alternates.append({
            'scenario': 'EXTENDED WAVE 3',
            'description': 'If price continues strong without significant pullback',
            'probability': '25%'
        })
    
    if 'Wave 5' in str(wave_structure.get('wave_count', [])):
        alternates.append({
            'scenario': 'TRUNCATED WAVE 5',
            'description': 'If Wave 5 fails to exceed Wave 3 high',
            'probability': '15%'
        })
    
    return alternates


def ml_elliott_pattern_match(symbol_data, lookback=200):
    """ML-based Elliott Wave pattern matching"""
    if not SKLEARN_AVAILABLE:
        return None
    
    closes = symbol_data['close'].values[-lookback:]
    normalized = (closes - closes.min()) / (closes.max() - closes.min() + 1e-10)
    
    templates = {
        'ideal_impulse': generate_ideal_impulse(len(closes)),
        'ideal_zigzag': generate_ideal_zigzag(len(closes)),
        'ideal_flat': generate_ideal_flat(len(closes)),
    }
    
    best_match = None
    best_score = float('inf')
    
    for name, template in templates.items():
        if len(template) == len(normalized):
            similarity = cosine(normalized, template)
            if similarity < best_score:
                best_score = similarity
                best_match = name
    
    return {'matched_pattern': best_match, 'similarity_score': 1 - best_score, 'confidence': 'HIGH' if best_score < 0.2 else 'MEDIUM' if best_score < 0.4 else 'LOW'}


def generate_ideal_impulse(length):
    """Generate ideal impulse pattern"""
    x = np.linspace(0, 4*np.pi, length)
    wave1 = np.sin(x[:length//5]) * 0.2
    wave2 = -np.sin(x[:length//5]) * 0.1
    wave3 = np.sin(x[:length//5]) * 0.5
    wave4 = -np.sin(x[:length//5]) * 0.15
    wave5 = np.sin(x[:length//5]) * 0.3
    ideal = np.concatenate([wave1, wave1[-1]+wave2, wave2[-1]+wave3, wave3[-1]+wave4, wave4[-1]+wave5])
    return (ideal - ideal.min()) / (ideal.max() - ideal.min())


def generate_ideal_zigzag(length):
    """Generate ideal zigzag pattern"""
    x = np.linspace(0, 3*np.pi, length)
    wave_a = -np.sin(x[:length//3]) * 0.4
    wave_b = np.sin(x[:length//3]) * 0.2
    wave_c = -np.sin(x[:length//3]) * 0.5
    ideal = np.concatenate([wave_a, wave_a[-1]+wave_b, wave_b[-1]+wave_c])
    return (ideal - ideal.min()) / (ideal.max() - ideal.min())


def generate_ideal_flat(length):
    """Generate ideal flat pattern"""
    x = np.linspace(0, 3*np.pi, length)
    wave_a = -np.sin(x[:length//3]) * 0.2
    wave_b = np.sin(x[:length//3]) * 0.3
    wave_c = -np.sin(x[:length//3]) * 0.25
    ideal = np.concatenate([wave_a, wave_a[-1]+wave_b, wave_b[-1]+wave_c])
    return (ideal - ideal.min()) / (ideal.max() - ideal.min())


def analyze_elliott_order_flow(symbol_data, wave_structure):
    """Elliott Wave + Order Flow analysis"""
    closes = symbol_data['close'].values
    opens = symbol_data['open'].values
    volumes = symbol_data['volume'].values
    
    buy_pressure = []
    sell_pressure = []
    
    for i in range(len(closes)):
        if closes[i] > opens[i]:
            buy_pressure.append(volumes[i])
            sell_pressure.append(0)
        else:
            buy_pressure.append(0)
            sell_pressure.append(volumes[i])
    
    wave_analysis = {}
    
    if len(buy_pressure) >= 50 and len(sell_pressure) >= 50:
        wave_analysis['wave1_buy_pressure'] = np.sum(buy_pressure[-50:-30])
        wave_analysis['wave2_sell_pressure'] = np.sum(sell_pressure[-30:-10])
        
        total_buy = wave_analysis['wave1_buy_pressure']
        total_sell = wave_analysis['wave2_sell_pressure']
        
        if total_buy > total_sell * 1.5:
            wave_analysis['imbalance'] = 'BULLISH (Strong buying)'
        elif total_sell > total_buy * 1.5:
            wave_analysis['imbalance'] = 'BEARISH (Strong selling)'
        else:
            wave_analysis['imbalance'] = 'NEUTRAL'
    
    return wave_analysis


def confirm_elliott_with_indicators(symbol_data, wave_structure, idx):
    """Confirm Elliott Wave with indicators"""
    closes = symbol_data['close'].values
    rsi_values = calculate_rsi_series(closes)
    macd_line, macd_signal, _ = calculate_macd_series(closes)
    
    current_wave = wave_structure.get('wave_count', [])[-1] if wave_structure.get('wave_count') else 'Unknown'
    
    confirmations = {'rsi_divergence': False, 'macd_divergence': False, 'volume_confirmation': False, 'overall_confidence': 0}
    
    if current_wave == 'Wave 5':
        rsi_div = detect_rsi_divergence(closes[-50:], rsi_values[-50:])
        macd_div = detect_macd_divergence(closes[-50:], macd_line[-50:])
        
        if rsi_div in ['Bearish', 'Hidden Bearish']:
            confirmations['rsi_divergence'] = True
            confirmations['overall_confidence'] += 25
        if macd_div == 'Bearish':
            confirmations['macd_divergence'] = True
            confirmations['overall_confidence'] += 25
    elif current_wave == 'Wave 3':
        volumes = symbol_data['volume'].values[-50:]
        current_vol = volumes[-1]
        avg_vol = np.mean(volumes)
        
        if current_vol > avg_vol * 1.5:
            confirmations['volume_confirmation'] = True
            confirmations['overall_confidence'] += 30
    
    return confirmations


def calculate_elliott_confidence_score(wave_structure, confirmations, volume_data):
    """Comprehensive confidence scoring"""
    score = 0
    
    if wave_structure.get('wave_count'):
        score += 20
        if len(wave_structure['wave_count']) >= 5:
            score += 20
    
    fib_ratios = wave_structure.get('fib_ratios', {})
    if fib_ratios.get('wave3_vs_wave1', 0) >= 1.618:
        score += 15
    if 0.382 <= fib_ratios.get('wave2_retrace', 0) <= 0.618:
        score += 10
    
    if confirmations.get('rsi_divergence'):
        score += 10
    if confirmations.get('macd_divergence'):
        score += 10
    if confirmations.get('volume_confirmation'):
        score += 15
    
    return {'total_score': score, 'percentage': score, 'rating': 'A+' if score >= 85 else 'A' if score >= 70 else 'B' if score >= 55 else 'C'}


def generate_elliott_prediction_text(wave_structure, is_bullish):
    """Elliott Wave prediction text"""
    if not wave_structure:
        return "Insufficient data for Elliott Wave analysis."
    
    wave_count = wave_structure.get('wave_count', [])
    fib_ratios = wave_structure.get('fib_ratios', {})
    sub_waves = wave_structure.get('sub_waves', {})
    confidence = wave_structure.get('confidence', 50)
    fib_time = wave_structure.get('fib_time_zones', {})
    volume_conf = wave_structure.get('volume_confluence', {})
    alternates = wave_structure.get('alternates', [])
    order_flow = wave_structure.get('order_flow', {})
    confidence_score = wave_structure.get('confidence_score', {})
    ml_match = wave_structure.get('ml_match', {})
    
    current_wave = wave_count[-1] if wave_count else 'Unknown'
    
    text = f"""
📐 ELLIOTT WAVE COMPLETE ANALYSIS:
================================================================================

🔍 WAVE COUNT DETECTED:
────────────────────────────────────────────────────────────────────────────────
Complete Wave Structure: {' → '.join(wave_count) if wave_count else 'Not detected'}
Current Wave Position: {current_wave}
Direction: {'BULLISH (Impulse Up)' if is_bullish else 'BEARISH (Impulse Down)'}
Confidence: {confidence}%

📊 FIBONACCI RATIOS (Actual from Data):
────────────────────────────────────────────────────────────────────────────────
"""
    
    for key, value in fib_ratios.items():
        if isinstance(value, (int, float)):
            text += f"• {key}: {value:.3f} "
            if key == 'wave3_vs_wave1':
                text += "✅ (Ideal)" if 1.618 <= value <= 2.618 else "⚠️ (Valid)" if value > 1.0 else "❌ (Invalid)"
            elif key == 'wave2_retrace':
                text += "✅ (Ideal)" if 0.382 <= value <= 0.618 else "⚠️ (Valid)" if value < 1.0 else "❌ (Invalid)"
            text += "\n"

    if fib_time:
        text += f"""
⏰ FIBONACCI TIME ZONES:
────────────────────────────────────────────────────────────────────────────────
Avg Wave Duration: {fib_time.get('avg_wave_duration', 7):.1f} days
Predicted Completion Dates:
"""
        for fib, date in fib_time.get('predicted_dates', {}).items():
            text += f"  • {fib}: {date}\n"

    text += f"""
🔬 SUB-WAVE STRUCTURE:
────────────────────────────────────────────────────────────────────────────────
"""
    for wave_name, sub_structure in sub_waves.items():
        text += f"• {wave_name}: {sub_structure.get('structure', 'Unknown')}\n"
        text += f"  Sub-waves: {'-'.join(sub_structure.get('sub_waves', []))}\n"

    if volume_conf:
        text += f"""
📊 VOLUME CONFLUENCE:
────────────────────────────────────────────────────────────────────────────────
High Volume Levels: {', '.join([str(l) for l in volume_conf.get('high_volume_levels', [])])}
Confluence Score: {volume_conf.get('confluence_score', 0)}/100
"""

    if order_flow:
        text += f"""
📈 ORDER FLOW ANALYSIS:
────────────────────────────────────────────────────────────────────────────────
Imbalance: {order_flow.get('imbalance', 'NEUTRAL')}
"""

    if ml_match:
        text += f"""
🤖 ML PATTERN MATCHING:
────────────────────────────────────────────────────────────────────────────────
Matched Pattern: {ml_match.get('matched_pattern', 'None')}
Similarity: {ml_match.get('similarity_score', 0)*100:.1f}%
Confidence: {ml_match.get('confidence', 'LOW')}
"""

    if confidence_score:
        text += f"""
🎯 CONFIDENCE SCORE:
────────────────────────────────────────────────────────────────────────────────
Total Score: {confidence_score.get('total_score', 0)}/100 ({confidence_score.get('percentage', 0):.0f}%)
Rating: {confidence_score.get('rating', 'C')}
"""

    text += f"""
⚠️ INVALIDATION LEVELS:
────────────────────────────────────────────────────────────────────────────────
"""
    for level, price in wave_structure.get('invalidation', {}).items():
        text += f"• {level}: {price:.2f}\n"

    if alternates:
        text += f"""
🔄 ALTERNATE SCENARIOS:
────────────────────────────────────────────────────────────────────────────────
"""
        for alt in alternates:
            text += f"• {alt.get('scenario', '')} (Probability: {alt.get('probability', 'N/A')})\n"
            text += f"  {alt.get('description', '')}\n"

    text += f"""
🎯 PREDICTION:
────────────────────────────────────────────────────────────────────────────────
"""
    
    if current_wave == 'Wave 1':
        text += "Currently in Wave 1. Expect Wave 2 pullback (38-62% retracement)."
    elif current_wave == 'Wave 2':
        text += "Currently in Wave 2. Expect Wave 3 strong impulse soon."
    elif current_wave == 'Wave 3':
        text += "Currently in Wave 3 - Strongest impulse! Expect continued momentum."
    elif current_wave == 'Wave 4':
        text += "Currently in Wave 4. Expect Wave 5 final impulse."
    elif current_wave == 'Wave 5':
        text += "Currently in Wave 5 - Final impulse. Watch for divergence Ugc!"
    
    return text


# =========================================================
# SMC PATTERN DETECTION
# =========================================================

def detect_order_block(df, idx):
    """Order Block detection"""
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
    """Fair Value Gap detection"""
    if idx < 3:
        return None
    
    candle1 = df.iloc[idx-2]
    candle3 = df.iloc[idx]
    result = []
    
    if candle1['low'] > candle3['high']:
        result.append('Bullish FVG')
    if candle1['high'] < candle3['low']:
        result.append('Bearish FVG')
    
    return result if result else None


def detect_liquidity_pools(df, idx):
    """Liquidity Pools detection"""
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
                result.append('Equal Highs')
                break
        if 'Buy Side Liquidity' in result:
            break
    
    low_tolerance = np.mean(lows) * 0.005
    for i in range(len(lows)-1):
        for j in range(i+1, len(lows)):
            if abs(lows[i] - lows[j]) <= low_tolerance:
                result.append('Sell Side Liquidity')
                result.append('Equal Lows')
                break
        if 'Sell Side Liquidity' in result:
            break
    
    current_price = df.iloc[idx]['close']
    if current_price > np.max(highs) * 1.001 or current_price < np.min(lows) * 0.999:
        result.append('Liquidity Sweep')
    
    return result if result else None


def detect_market_structure_smc(df, idx):
    """Market Structure detection"""
    if idx < 30:
        return None
    
    recent = df.iloc[idx-30:idx]
    highs = recent['high'].values
    lows = recent['low'].values
    closes = recent['close'].values
    
    result = []
    swing_highs, swing_lows = find_swing_points(highs, lows, window=2)
    
    if len(swing_highs) >= 2:
        result.append('Higher High (HH)' if swing_highs[-1] > swing_highs[-2] else 'Lower High (LH)')
    if len(swing_lows) >= 2:
        result.append('Higher Low (HL)' if swing_lows[-1] > swing_lows[-2] else 'Lower Low (LL)')
    if len(swing_highs) >= 2 and swing_highs[-1] > swing_highs[-2]:
        result.append('Break of Structure (BOS)')
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        if closes[-1] < swing_lows[-2] or closes[-1] > swing_highs[-2]:
            result.append('Change of Character (CHoCH)')
    
    return result if result else None


def detect_ote_entry(df, idx):
    """OTE Zone detection"""
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
    
    return result if result else None


def detect_smc_manipulation(df, idx):
    """SMC Manipulation patterns"""
    if idx < 10:
        return None
    
    recent = df.iloc[idx-10:idx]
    highs = recent['high'].values
    lows = recent['low'].values
    closes = recent['close'].values
    result = []
    
    if len(highs) >= 5:
        recent_high = np.max(highs[:-1])
        if highs[-1] > recent_high * 1.005 and closes[-1] < recent_high:
            result.append('Fake Breakout')
    
    if len(lows) >= 5:
        recent_low = np.min(lows[:-1])
        if lows[-1] < recent_low * 0.995 and closes[-1] > recent_low:
            result.append('Fake Breakout')
    
    return result if result else None


def detect_smc_hybrid(df, idx):
    """Hybrid SMC patterns"""
    ob = detect_order_block(df, idx)
    fvg = detect_fair_value_gap(df, idx)
    liq = detect_liquidity_pools(df, idx)
    ms = detect_market_structure_smc(df, idx)
    
    result = []
    if ob and fvg:
        result.append('OB + FVG Combo')
    if liq and 'Liquidity Sweep' in liq and ms and 'Change of Character (CHoCH)' in ms:
        result.append('Liquidity Sweep + CHoCH Ugc')
    
    return result if result else None


# =========================================================
# NEW SMC FUNCTIONS (ADVANCED)
# =========================================================

def detect_breaker_block(df, idx):
    """Breaker Block - Failed OB that becomes support/resistance"""
    if idx < 10:
        return None
    
    recent = df.iloc[max(0, idx-10):idx+1]
    result = []
    
    for i in range(len(recent)-3, 0, -1):
        if recent.iloc[i]['close'] < recent.iloc[i]['open']:
            next_candles = recent.iloc[i+1:]
            if len(next_candles) >= 2:
                if next_candles['high'].max() > recent.iloc[i]['high']:
                    result.append('Bullish Breaker Block')
                    break
    
    for i in range(len(recent)-3, 0, -1):
        if recent.iloc[i]['close'] > recent.iloc[i]['open']:
            next_candles = recent.iloc[i+1:]
            if len(next_candles) >= 2:
                if next_candles['low'].min() < recent.iloc[i]['low']:
                    result.append('Bearish Breaker Block')
                    break
    
    return result if result else None


def detect_mitigation_block(df, idx):
    """Mitigation Block - OB that got partially filled"""
    if idx < 5:
        return None
    
    recent = df.iloc[max(0, idx-5):idx+1]
    result = []
    
    for i in range(len(recent)-2, 0, -1):
        candle = recent.iloc[i]
        next_candle = recent.iloc[i+1]
        
        if candle['close'] < candle['open']:
            if next_candle['low'] <= candle['high'] and next_candle['close'] > candle['close']:
                result.append('Bullish Mitigation Block')
                break
        
        if candle['close'] > candle['open']:
            if next_candle['high'] >= candle['low'] and next_candle['close'] < candle['close']:
                result.append('Bearish Mitigation Block')
                break
    
    return result if result else None


def detect_rejection_block(df, idx):
    """Rejection Block - Strong wick rejection at OB"""
    if idx < 3:
        return None
    
    candle = df.iloc[idx]
    body = abs(candle['close'] - candle['open'])
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    
    result = []
    
    if lower_wick > body * 2 and upper_wick < body * 0.5:
        result.append('Bullish Rejection Block Ugc')
    
    if upper_wick > body * 2 and lower_wick < body * 0.5:
        result.append('Bearish Rejection Block Ugc')
    
    return result if result else None


def detect_vacuum_block(df, idx):
    """Vacuum Block - Price gap with no trading"""
    if idx < 2:
        return None
    
    prev = df.iloc[idx-1]
    curr = df.iloc[idx]
    
    result = []
    
    if curr['low'] > prev['high'] * 1.002:
        gap_size = curr['low'] - prev['high']
        result.append(f'Bullish Vacuum ({gap_size:.2f}) Ugc')
    
    if curr['high'] < prev['low'] * 0.998:
        gap_size = prev['low'] - curr['high']
        result.append(f'Bearish Vacuum ({gap_size:.2f}) Ugc')
    
    return result if result else None


def detect_turtle_soup(df, idx):
    """Turtle Soup - False breakout trap"""
    if idx < 20:
        return None
    
    recent = df.iloc[idx-20:idx+1]
    highs = recent['high'].values
    lows = recent['low'].values
    closes = recent['close'].values
    
    result = []
    
    recent_low = np.min(lows[:-1])
    if lows[-1] < recent_low * 0.998 and closes[-1] > recent_low:
        result.append('Turtle Soup (Bullish Trap) Ugc')
    
    recent_high = np.max(highs[:-1])
    if highs[-1] > recent_high * 1.002 and closes[-1] < recent_high:
        result.append('Turtle Soup (Bearish Trap) Ugc')
    
    return result if result else None


def detect_power_of_3(df, idx):
    """Power of 3 - Judas Swing setup"""
    if idx < 30:
        return None
    
    recent = df.iloc[idx-30:idx+1]
    highs = recent['high'].values
    lows = recent['low'].values
    closes = recent['close'].values
    
    result = []
    current_price = closes[-1]

    if len(highs) < 20 or len(lows) < 20 :
        return None
    
    asian_high = np.max(highs[:10])
    asian_low = np.min(lows[:10])
    
    london_high = np.max(highs[10:20])
    london_low = np.min(lows[10:20])
    
    if london_low < asian_low and current_price > asian_high:
        result.append('Power of 3 - Bullish Ugc')
    elif london_high > asian_high and current_price < asian_low:
        result.append('Power of 3 - Bearish Ugc')
    
    return result if result else None


def detect_killzones(df, idx):
    """Killzone Analysis - High probability reversal times"""
    if idx < 5:
        return None
    
    asian_kz_start, asian_kz_end = 0, 3
    london_kz_start, london_kz_end = 7, 10
    ny_kz_start, ny_kz_end = 12, 15
    
    recent = df.iloc[max(0, idx-5):idx+1]
    result = []
    
    for i in range(1, len(recent)):
        prev = recent.iloc[i-1]
        curr = recent.iloc[i]
        
        hour = pd.to_datetime(curr['date']).hour if 'date' in curr else 0
        
        if hour in range(asian_kz_start, asian_kz_end):
            if (prev['close'] < prev['open'] and curr['close'] > curr['open']):
                result.append('Asian Killzone Reversal Ugc')
        elif hour in range(london_kz_start, london_kz_end):
            if (prev['close'] > prev['open'] and curr['close'] < curr['open']):
                result.append('London Killzone Reversal Ugc')
        elif hour in range(ny_kz_start, ny_kz_end):
            result.append('NY Killzone Active Ugc')
    
    return result if result else None


def detect_silver_bullet(df, idx):
    """Silver Bullet - ICT high-probability setup"""
    if idx < 10:
        return None
    
    recent = df.iloc[idx-10:idx+1]
    highs = recent['high'].values
    lows = recent['low'].values
    closes = recent['close'].values
    
    result = []
    current_price = closes[-1]
    
    swing_high = np.max(highs[:-1])
    swing_low = np.min(lows[:-1])
    range_size = swing_high - swing_low
    
    fib_618 = swing_low + range_size * 0.618
    fib_786 = swing_low + range_size * 0.786
    
    if fib_618 <= current_price <= fib_786:
        prev = recent.iloc[-2]
        curr = recent.iloc[-1]
        if curr['close'] > curr['open'] and curr['close'] > prev['high']:
            result.append('Silver Bullet LONG Ugc')
    
    fib_618_short = swing_high - range_size * 0.618
    fib_786_short = swing_high - range_size * 0.786
    
    if fib_786_short <= current_price <= fib_618_short:
        prev = recent.iloc[-2]
        curr = recent.iloc[-1]
        if curr['close'] < curr['open'] and curr['close'] < prev['low']:
            result.append('Silver Bullet SHORT Ugc')
    
    return result if result else None


def detect_ict_macro_times(df, idx):
    """ICT Macro - Higher timeframe analysis times"""
    if idx < 20:
        return None
    
    macro_times = {
        '00:00': 'Daily Open', '02:00': 'Asian Macro', '07:00': 'London Open',
        '08:30': 'London Macro 1', '10:00': 'London Macro 2', '12:30': 'NY Pre-Market',
        '13:30': 'NY Open', '15:00': 'NY Macro 1', '16:30': 'NY Macro 2', '20:00': 'Daily Close'
    }
    
    result = []
    current_time = datetime.now().strftime('%H:%M')
    
    for time_str, label in macro_times.items():
        if current_time >= time_str:
            result.append(f'Macro Time: {label} Ugc')
    
    return result[:3] if result else None


def detect_imbalance(df, idx):
    """Imbalance - Price inefficiency that needs to be filled"""
    if idx < 3:
        return None
    
    candle1 = df.iloc[idx-2]
    candle3 = df.iloc[idx]
    
    result = []
    
    if candle1['low'] > candle3['high']:
        imbalance_size = candle1['low'] - candle3['high']
        result.append(f'Bullish Imbalance ({imbalance_size:.2f}) Ugc')
        result.append(f'Target Fill: {candle1["low"]:.2f} Ugc')
    
    if candle1['high'] < candle3['low']:
        imbalance_size = candle3['low'] - candle1['high']
        result.append(f'Bearish Imbalance ({imbalance_size:.2f}) Ugc')
        result.append(f'Target Fill: {candle1["high"]:.2f} Ugc')
    
    return result if result else None


def detect_sibi_bisi(df, idx):
    """SIBI/BISI - Smart money entry confirmation"""
    if idx < 5:
        return None
    
    recent = df.iloc[idx-5:idx+1]
    volumes = recent['volume'].values
    closes = recent['close'].values
    
    result = []
    
    if closes[-1] > closes[-2] and volumes[-1] > volumes[-2] * 1.5:
        result.append('BISI - Strong Buying Ugc')
    
    if closes[-1] < closes[-2] and volumes[-1] > volumes[-2] * 1.5:
        result.append('SIBI - Strong Selling Ugc')
    
    return result if result else None


def detect_mss(df, idx):
    """MSS - Market Structure Shift (Stronger than CHoCH)"""
    if idx < 30:
        return None
    
    recent = df.iloc[idx-30:idx+1]
    highs = recent['high'].values
    lows = recent['low'].values
    closes = recent['close'].values
    volumes = recent['volume'].values
    
    result = []
    current_price = closes[-1]
    
    swing_highs = []
    swing_lows = []
    
    for i in range(5, len(highs)-5):
        if highs[i] == max(highs[i-5:i+6]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i-5:i+6]):
            swing_lows.append(lows[i])
    
    if len(swing_lows) >= 2 and len(swing_highs) >=1:
        if swing_lows[-1] < swing_lows[-2] and current_price > swing_highs[-1]:
            if volumes[-1] > np.mean(volumes[-10:]) * 1.2:
                result.append('MSS Bullish - STRONG Ugc')
            else:
                result.append('MSS Bullish Ugc')
    
    if len(swing_highs) >= 2 and len(swing_lows) >=1:
        if swing_highs[-1] > swing_highs[-2] and current_price < swing_lows[-1]:
            if volumes[-1] > np.mean(volumes[-10:]) * 1.2:
                result.append('MSS Bearish - STRONG Ugc')
            else:
                result.append('MSS Bearish Ugc')
    
    return result if result else None


def detect_reaccumulation_range(df, idx):
    """Re-Accumulation / Re-Distribution Range Detection"""
    if idx < 50:
        return None
    
    recent = df.iloc[idx-50:idx+1]
    highs = recent['high'].values
    lows = recent['low'].values
    volumes = recent['volume'].values

    if len(highs) < 50 or len(lows) < 50 or len(volumes) < 50 :
        return  None
    
    range_high = np.percentile(highs, 85)
    range_low = np.percentile(lows, 15)
    
    avg_vol_first = np.mean(volumes[:25])
    avg_vol_second = np.mean(volumes[25:])
    
    result = []
    
    if avg_vol_second < avg_vol_first * 0.7:
        if range_high - range_low < range_high * 0.05:
            result.append('Re-Accumulation Range (Bullish) Ugc')
            result.append(f'Range: {range_low:.2f} - {range_high:.2f} Ugc')
    
    return result if result else None


def detect_stop_hunt_levels(df, idx):
    """Stop Hunt Levels - Where liquidity sits"""
    if idx < 20:
        return None
    
    recent = df.iloc[idx-20:idx+1]
    highs = recent['high'].values
    lows = recent['low'].values
    
    result = []
    
    recent_high = np.max(highs)
    above_stops = recent_high * 1.002
    
    recent_low = np.min(lows)
    below_stops = recent_low * 0.998
    
    result.append(f'Stop Hunt Zone (Longs): Below {below_stops:.2f} Ugc')
    result.append(f'Stop Hunt Zone (Shorts): Above {above_stops:.2f} Ugc')
    
    return result


def detect_confluence_zone(df, idx):
    """Multiple SMC concepts at same level"""
    if idx < 10:
        return None
    
    ob = detect_order_block(df, idx)
    fvg = detect_fair_value_gap(df, idx)
    liq = detect_liquidity_pools(df, idx)
    ote = detect_ote_entry(df, idx)
    
    confluence_score = 0
    confluence_items = []
    
    if ob:
        confluence_score += 1
        confluence_items.extend(ob)
    if fvg:
        confluence_score += 1
        confluence_items.extend(fvg)
    if liq:
        confluence_score += 1
        confluence_items.extend(liq)
    if ote:
        confluence_score += 1
        confluence_items.extend(ote)
    
    result = []
    
    if confluence_score >= 3:
        result.append(f'HIGH Confluence Zone (Score: {confluence_score}/4) Ugc')
        result.extend(confluence_items[:3])
    elif confluence_score >= 2:
        result.append(f'MEDIUM Confluence Zone (Score: {confluence_score}/4) Ugc')
        result.extend(confluence_items[:2])
    
    return result if result else None


def detect_liquidity_sweep_detailed(df, idx):
    """Detailed Liquidity Sweep Analysis"""
    if idx < 20:
        return None
    
    recent = df.iloc[idx-20:idx+1]
    highs = recent['high'].values
    lows = recent['low'].values
    closes = recent['close'].values
    current_price = closes[-1]
    
    result = []
    
    if len(highs) >= 3:
        if abs(highs[-2] - highs[-3]) < highs[-2] * 0.005:
            if highs[-1] > highs[-2] * 1.001 and closes[-1] < highs[-2]:
                result.append('Double Top Liquidity Sweep - Bearish Ugc')
    
    if len(lows) >= 3:
        if abs(lows[-2] - lows[-3]) < lows[-2] * 0.005:
            if lows[-1] < lows[-2] * 0.999 and closes[-1] > lows[-2]:
                result.append('Double Bottom Liquidity Sweep - Bullish Ugc')
    
    recent_highs = [highs[i] for i in range(len(highs)-5, len(highs)) if highs[i] == max(highs[max(0,i-2):min(len(highs),i+3)])]
    if len(recent_highs) >= 1:
        if current_price > max(recent_highs) * 1.002:
            result.append('Trendline Liquidity Sweep Ugc')
    
    return result if result else None


def detect_ote_complete(df, idx):
    """Complete OTE Analysis with multiple timeframes"""
    if idx < 30:
        return None
    
    recent = df.iloc[idx-30:idx+1]
    swing_high = recent['high'].max()
    swing_low = recent['low'].min()
    current_price = recent['close'].iloc[-1]
    
    range_size = swing_high - swing_low
    
    result = []
    
    ote_long_low = swing_low + range_size * 0.618
    ote_long_high = swing_low + range_size * 0.786
    
    ote_short_high = swing_high - range_size * 0.618
    ote_short_low = swing_high - range_size * 0.786
    
    if ote_long_low <= current_price <= ote_long_high:
        result.append(f'OTE LONG: {ote_long_low:.2f} - {ote_long_high:.2f} Ugc')
        result.append('Target: Recent High Ugc')
        result.append(f'Stop: Below {ote_long_low:.2f} Ugc')
    elif ote_short_low <= current_price <= ote_short_high:
        result.append(f'OTE SHORT: {ote_short_low:.2f} - {ote_short_high:.2f} Ugc')
        result.append('Target: Recent Low Ugc')
        result.append(f'Stop: Above {ote_short_high:.2f} Ugc')
    
    return result if result else None


def detect_fvg_types(df, idx):
    """Different types of FVGs"""
    if idx < 3:
        return None
    
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    
    result = []
    
    if c1['low'] > c3['high']:
        fvg_size = c1['low'] - c3['high']
        if c2['close'] > c2['open']:
            result.append(f'BISI FVG (Bullish) - Size: {fvg_size:.2f} Ugc')
        else:
            result.append(f'Bullish FVG - Size: {fvg_size:.2f} Ugc')
    
    if c1['high'] < c3['low']:
        fvg_size = c3['low'] - c1['high']
        if c2['close'] < c2['open']:
            result.append(f'SIBI FVG (Bearish) - Size: {fvg_size:.2f} Ugc')
        else:
            result.append(f'Bearish FVG - Size: {fvg_size:.2f} Ugc')
    
    return result if result else None


def detect_daily_bias(df, idx):
    """Daily Bias - Expected direction for the day"""
    if idx < 5:
        return None
    
    recent = df.iloc[idx-5:idx+1]
    opens = recent['open'].values
    closes = recent['close'].values
    highs = recent['high'].values
    lows = recent['low'].values
    
    result = []
    
    opening_range_high = highs[0]
    opening_range_low = lows[0]
    current_price = closes[-1]
    
    if current_price > opening_range_high:
        result.append('Daily Bias: BULLISH Ugc')
        result.append('Target: Opening Range Expansion Ugc')
    elif current_price < opening_range_low:
        result.append('Daily Bias: BEARISH Ugc')
        result.append('Target: Opening Range Breakdown Ugc')
    else:
        result.append('Daily Bias: NEUTRAL Ugc')
        result.append(f'Range: {opening_range_low:.2f} - {opening_range_high:.2f} Ugc')
    
    return result


def detect_market_maker_model(df, idx):
    """Market Maker Model - Accumulation/Distribution phases"""
    if idx < 40:
        return None
    
    recent = df.iloc[idx-40:idx+1]
    highs = recent['high'].values
    lows = recent['low'].values

    if len(highs) < 40 or len(lows) < 40 :
        return None
    
    phase1_range = np.max(highs[:10]) - np.min(lows[:10])
    phase2_range = np.max(highs[10:20]) - np.min(lows[10:20])
    phase3_range = np.max(highs[20:30]) - np.min(lows[20:30])
    phase4_range = np.max(highs[30:]) - np.min(lows[30:])
    
    result = []
    
    if phase1_range > phase2_range and phase3_range < phase2_range:
        result.append('Market Maker Model: ACCUMULATION Ugc')
        result.append('Expect: Expansion upward Ugc')
    elif phase1_range < phase2_range and phase3_range > phase2_range:
        result.append('Market Maker Model: DISTRIBUTION Ugc')
        result.append('Expect: Expansion downward Ugc')
    else:
        result.append('Market Maker Model: CONSOLIDATION Ugc')
    
    return result


def detect_institutional_candles(df, idx):
    """Institutional Candle Patterns"""
    if idx < 3:
        return None
    
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    
    result = []
    
    body1 = abs(c1['close'] - c1['open'])
    body2 = abs(c2['close'] - c2['open'])
    body3 = abs(c3['close'] - c3['open'])
    
    if body3 > body2 > body1:
        if c1['close'] < c1['open'] and c2['close'] < c2['open'] and c3['close'] < c3['open']:
            result.append('Three Drives Bearish Ugc')
        elif c1['close'] > c1['open'] and c2['close'] > c2['open'] and c3['close'] > c3['open']:
            result.append('Three Drives Bullish Ugc')
    
    return result if result else None


def detect_smc_divergence(df, idx):
    """Divergence between price and SMC concepts"""
    if idx < 20:
        return None
    
    recent = df.iloc[idx-20:idx+1]
    closes = recent['close'].values
    volumes = recent['volume'].values
    if len(closes) <10 or len(volumes) < 10 :
        return None
    
    result = []
    
    if closes[-1] > np.max(closes[-10:-1]):
        if volumes[-1] < np.mean(volumes[-10:-1]):
            result.append('SMC Divergence: Price HH, Volume LH - BEARISH Ugc')
    
    if closes[-1] < np.min(closes[-10:-1]):
        if volumes[-1] > np.mean(volumes[-10:-1]):
            result.append('SMC Divergence: Price LL, Volume HL - BULLISH Ugc')
    
    return result if result else None


def detect_liquidity_void(df, idx):
    """Liquidity Void - Area with no liquidity"""
    if idx < 5:
        return None
    
    recent = df.iloc[idx-5:idx+1]
    
    result = []
    
    for i in range(1, len(recent)):
        if recent.iloc[i]['low'] > recent.iloc[i-1]['high']:
            void_size = recent.iloc[i]['low'] - recent.iloc[i-1]['high']
            result.append(f'Liquidity Void UP ({void_size:.2f}) Ugc')
        elif recent.iloc[i]['high'] < recent.iloc[i-1]['low']:
            void_size = recent.iloc[i-1]['low'] - recent.iloc[i]['high']
            result.append(f'Liquidity Void DOWN ({void_size:.2f}) Ugc')
    
    return result if result else None


# =========================================================
# ORIGINAL PATTERN DETECTION
# =========================================================

def detect_cup_and_handle(df, idx):
    if idx < 50: return False
    recent = df.iloc[idx-50:idx]
    highs, lows = recent['high'].values, recent['low'].values
    cup_bottom_idx = np.argmin(lows)
    if cup_bottom_idx < 10 or cup_bottom_idx > 40: return False
    left_rim, right_rim = np.max(highs[:cup_bottom_idx+5]), np.max(highs[cup_bottom_idx+5:])
    if abs(left_rim - right_rim) / left_rim > 0.05: return False
    handle_high = np.max(highs[-15:])
    return df.iloc[idx]['close'] > handle_high * 1.01


def detect_double_bottom(df, idx):
    if idx < 40: return False
    lows = df.iloc[idx-40:idx]['low'].values
    sorted_lows = np.argsort(lows)
    if len(sorted_lows) < 2: return False
    b1, b2 = sorted_lows[0], sorted_lows[1]
    if abs(b1 - b2) < 10 or abs(b1 - b2) > 30: return False
    if abs(lows[b1] - lows[b2]) / lows[b1] > 0.03: return False
    neckline = np.max(df.iloc[idx-40:idx]['high'].values[min(b1,b2):max(b1,b2)+1])
    return df.iloc[idx]['close'] > neckline * 1.01


def detect_head_and_shoulders(df, idx):
    if idx < 60: return False
    highs = df.iloc[idx-60:idx]['high'].values
    peaks = [(i, highs[i]) for i in range(5, len(highs)-5) if highs[i] == max(highs[i-5:i+6])]
    if len(peaks) < 3: return False
    head = sorted(peaks, key=lambda x: x[1], reverse=True)[0]
    left = right = None
    for p in peaks:
        if p[0] < head[0] and (left is None or p[1] > left[1]): left = p
        elif p[0] > head[0] and (right is None or p[1] > right[1]): right = p
    if left is None or right is None: return False
    if abs(left[1] - right[1]) / left[1] > 0.05 or head[1] <= left[1] * 1.02: return False
    return df.iloc[idx]['close'] < min(left[1], right[1]) * 0.99


def detect_bull_flag(df, idx):
    if idx < 30: return False
    closes, highs, lows = df.iloc[idx-30:idx]['close'].values, df.iloc[idx-30:idx]['high'].values, df.iloc[idx-30:idx]['low'].values
    if len(closes) <= 15: return False
    if (closes[15] - closes[0]) / closes[0] < 0.10: return False
    flag_high, flag_low = np.max(highs[-15:]), np.min(lows[-15:])
    if (flag_high - flag_low) / flag_low > 0.10: return False
    return df.iloc[idx]['close'] > flag_high * 1.01


def detect_ascending_triangle(df, idx):
    if idx < 30: return False
    highs, lows = df.iloc[idx-30:idx]['high'].values, df.iloc[idx-30:idx]['low'].values
    resistance = np.percentile(highs, 90)
    if sum(1 for h in highs if h >= resistance * 0.99) < 3: return False
    if (lows[-1] - lows[0]) / lows[0] < 0.03: return False
    return df.iloc[idx]['close'] > resistance * 1.01


def detect_descending_triangle(df, idx):
    if idx < 30: return False
    highs, lows = df.iloc[idx-30:idx]['high'].values, df.iloc[idx-30:idx]['low'].values
    support = np.percentile(lows, 10)
    if sum(1 for l in lows if l <= support * 1.01) < 3: return False
    if (highs[-1] - highs[0]) / highs[0] > -0.03: return False
    return df.iloc[idx]['close'] < support * 0.99


def detect_symmetrical_triangle(df, idx):
    if idx < 30: return False
    highs, lows = df.iloc[idx-30:idx]['high'].values, df.iloc[idx-30:idx]['low'].values
    if np.isnan(highs).any() or np.isnan(lows).any(): return False
    x = np.arange(len(highs))
    high_slope, _ = np.polyfit(x, highs, 1)
    low_slope, _ = np.polyfit(x, lows, 1)
    if high_slope < -0.001 and low_slope > 0.001:
        return df.iloc[idx]['close'] > highs[-1] * 1.01 or df.iloc[idx]['close'] < lows[-1] * 0.99
    return False


def detect_rounding_bottom(df, idx):
    if idx < 50: return False
    closes = df.iloc[idx-50:idx]['close'].values
    min_idx = np.argmin(closes)
    left, right = closes[:min_idx], closes[min_idx+1:]
    if len(left) < 10 or len(right) < 10: return False
    left_trend = (left[-1] - left[0]) / left[0] if left[0] > 0 else 0
    right_trend = (right[-1] - right[0]) / right[0] if right[0] > 0 else 0
    if left_trend < -0.05 and right_trend > 0.05:
        return df.iloc[idx]['close'] > np.percentile(closes, 90)
    return False


def detect_bullish_engulfing(df, idx):
    if idx < 1: return False
    prev, curr = df.iloc[idx-1], df.iloc[idx]
    return (prev['close'] < prev['open'] and curr['close'] > curr['open'] and 
            curr['open'] < prev['close'] and curr['close'] > prev['open'])


def detect_hammer(df, idx):
    row = df.iloc[idx]
    body = abs(row['close'] - row['open'])
    lower_shadow = min(row['open'], row['close']) - row['low']
    return lower_shadow > body * 2 and row['high'] - max(row['open'], row['close']) < body * 0.3


def detect_morning_star(df, idx):
    if idx < 2: return False
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    return (c1['close'] < c1['open'] and abs(c2['close'] - c2['open']) < (c2['high'] - c2['low']) * 0.3 and
            c3['close'] > c3['open'] and c3['close'] > (c1['open'] + c1['close']) / 2)


def detect_doji(df, idx):
    row = df.iloc[idx]
    body = abs(row['close'] - row['open'])
    total_range = row['high'] - row['low']
    return total_range > 0 and body <= total_range * 0.1


def detect_piercing_line(df, idx):
    if idx < 1: return False
    prev, curr = df.iloc[idx-1], df.iloc[idx]
    return (prev['close'] < prev['open'] and curr['close'] > curr['open'] and
            curr['open'] < prev['low'] and curr['close'] > (prev['open'] + prev['close']) / 2)


def detect_three_white_soldiers(df, idx):
    if idx < 2: return False
    c1, c2, c3 = df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]
    return (c1['close'] > c1['open'] and c2['close'] > c2['open'] and c3['close'] > c3['open'] and
            c2['close'] > c1['close'] and c3['close'] > c2['close'])


def detect_volume_spike(df, idx):
    if idx < 20: return False
    recent = df['volume'].iloc[idx-20:idx].values
    zscore = (df.iloc[idx]['volume'] - np.mean(recent)) / (np.std(recent) + 1e-6)
    return zscore > 2.0


def detect_bollinger_squeeze(df, idx):
    if idx < 20 or 'bb_upper' not in df.columns: return False
    recent = df.iloc[idx-20:idx]
    bandwidth = (recent['bb_upper'] - recent['bb_lower']) / recent['bb_middle'].replace(0, np.nan)
    bandwidth = bandwidth.replace([np.inf, -np.inf], np.nan)
    if bandwidth.isna().all(): return False
    return bandwidth.iloc[-1] < bandwidth.mean() * 0.5 if not pd.isna(bandwidth.iloc[-1]) else False


# =========================================================
# COMPLETE PATTERN DETECTION
# =========================================================

def detect_all_patterns(df, idx):
    """Detect all patterns"""
    detected = []
    
    if detect_cup_and_handle(df, idx): detected.append('Cup and Handle')
    if detect_double_bottom(df, idx): detected.append('Double Bottom')
    if detect_head_and_shoulders(df, idx): detected.append('Head and Shoulders')
    if detect_bull_flag(df, idx): detected.append('Bull Flag')
    if detect_ascending_triangle(df, idx): detected.append('Ascending Triangle')
    if detect_descending_triangle(df, idx): detected.append('Descending Triangle')
    if detect_symmetrical_triangle(df, idx): detected.append('Symmetrical Triangle')
    if detect_rounding_bottom(df, idx): detected.append('Rounding Bottom')
    if detect_bullish_engulfing(df, idx): detected.append('Bullish Engulfing')
    if detect_hammer(df, idx): detected.append('Hammer')
    if detect_morning_star(df, idx): detected.append('Morning Star')
    if detect_doji(df, idx): detected.append('Doji')
    if detect_piercing_line(df, idx): detected.append('Piercing Line')
    if detect_three_white_soldiers(df, idx): detected.append('Three White Soldiers')
    if detect_volume_spike(df, idx): detected.append('Volume Climax')
    if detect_bollinger_squeeze(df, idx): detected.append('Bollinger Band Squeeze')
    
    # SMC Patterns
    for func in [detect_order_block, detect_fair_value_gap, detect_liquidity_pools, 
                 detect_market_structure_smc, detect_ote_entry, detect_smc_manipulation, detect_smc_hybrid,
                 detect_breaker_block, detect_mitigation_block, detect_rejection_block, detect_vacuum_block,
                 detect_turtle_soup, detect_power_of_3, detect_killzones, detect_silver_bullet,
                 detect_ict_macro_times, detect_imbalance, detect_sibi_bisi, detect_mss,
                 detect_reaccumulation_range, detect_stop_hunt_levels, detect_confluence_zone,
                 detect_liquidity_sweep_detailed, detect_ote_complete, detect_fvg_types,
                 detect_daily_bias, detect_market_maker_model, detect_institutional_candles,
                 detect_smc_divergence, detect_liquidity_void]:
        res = func(df, idx)
        if res: detected.extend(res)
    
    # New Candlestick Patterns
    candlestick_patterns = detect_all_candlestick_patterns(df, idx)
    detected.extend(candlestick_patterns)
    
    return list(set(detected))


# =========================================================
# NO PATTERN EXAMPLE
# =========================================================

def generate_no_pattern_example(symbol, df_row, indicator_values):
    current_price = df_row['close']
    current_date = df_row['date']
    sector = df_row.get('sector', 'Unknown')
    
    rsi = indicator_values.get('rsi', 50)
    macd = indicator_values.get('macd', 0)
    macd_signal = indicator_values.get('macd_signal', 0)
    volume = indicator_values.get('volume', 1000000)
    avg_vol = indicator_values.get('avg_volume', volume)
    volume_spike = "Yes" if volume > avg_vol * 1.5 else "No"
    
    sector_info = f"\n🏭 SECTOR: {sector}" if sector != 'Unknown' else ""
    
    return f"""
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


# =========================================================
# DATA GENERATION FUNCTIONS
# =========================================================

def generate_elliott_wave_data(symbol, df_row, pattern_type, config, indicator_values, metrics, variation_idx=0, symbol_data=None, idx=None):
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
    rsi_div = indicator_values.get('rsi_divergence', 'None')
    macd_div = indicator_values.get('macd_divergence', 'None')
    avg_vol = indicator_values.get('avg_volume', volume)
    ema_20 = indicator_values.get('ema_20', current_price)
    sma_20 = indicator_values.get('sma_20', current_price)
    
    atr_value = atr if atr > 0 else current_price * 0.02
    
    if config['bias'] == 'Bullish':
        entry, stop = current_price, current_price - atr_value * 1.5
        target = entry + abs(entry - stop) * random.uniform(1.5, 3.0)
    elif config['bias'] == 'Bearish':
        entry, stop = current_price, current_price + atr_value * 1.5
        target = entry - abs(entry - stop) * random.uniform(1.5, 3.0)
    else:
        entry, stop = current_price, current_price - atr_value * 2
        target = entry + abs(entry - stop) * random.uniform(1.0, 2.0)
    
    confidence = 50
    confidence += (macd > macd_signal and config['bias'] == 'Bullish') * 10
    confidence += (macd < macd_signal and config['bias'] == 'Bearish') * 10
    confidence += (rsi < 40 and config['bias'] == 'Bullish') * 5
    confidence += (rsi > 60 and config['bias'] == 'Bearish') * 5
    confidence += (volume > avg_vol * 1.5) * 5
    confidence += (metrics.get('relative_strength', 0) > 0.6) * 5
    confidence += (rsi_div != 'None') * random.randint(3, 6)
    confidence += (macd_div != 'None') * random.randint(2, 5)
    
    sector_analysis = get_sector_analysis(sector, symbol, current_price)
    confidence += sector_analysis.get('confidence_boost', 0)
    confidence = min(95, max(30, confidence + random.uniform(-5, 5)))
    
    rr_ratio = abs((target - entry) / max(abs(entry - stop), 1e-6))
    volume_spike = "Yes" if volume > avg_vol * 1.5 else "No"
    variation_note = f" [VARIATION {variation_idx + 1}]" if variation_idx > 0 else " [ORIGINAL SEQUENCE]"
    pattern_display = pattern_type if random.random() < 0.5 else "Unknown Pattern"
    price_header = "PRICE SNAPSHOT:" if random.random() < 0.3 else "📊 PRICE DATA:"
    
    sector_details = f"""
🏭 SECTOR INFORMATION:
────────────────────────────────────────────────────────────────────────────────
Sector: {sector}
Sector Strength: {sector_analysis.get('strength', 'Neutral')}
Sector Rotation Signal: {sector_analysis.get('rotation', 'None')}
Peer Comparison: {sector_analysis.get('peer_rank', 'N/A')}
"""
    
    price_sequence_text = ""
    if symbol_data is not None and idx is not None:
        price_sequence_text, bos_detected, vol_spike, high_vol = generate_advanced_price_sequence(symbol_data, idx)
        if bos_detected: confidence += 10
        if vol_spike: confidence += 5
        if high_vol: confidence += 3
    
    wyckoff_text = ""
    if symbol_data is not None and idx is not None:
        wyckoff_text, wyckoff_data = detect_volume_price_cycle(symbol_data, idx)
        confidence += wyckoff_data.get('confidence_boost', 0)
    
    forward_text = ""
    if symbol_data is not None and idx is not None:
        forward_text = generate_forward_looking_analysis(symbol_data, idx)
    
    # Price Action Complete Analysis
    pa_report, trend_lines, channels, rays, fib_ext, vol_profile, pa_predictions = analyze_price_action_complete(symbol_data, idx)
    
    # Fixed Range Liquidity
    liquidity_info = detect_fixed_range_liquidity(symbol_data)
    
    # Gap Analysis
    gap_info = analyze_gaps(symbol_data, idx)
    
    # Volatility Skew
    vol_skew = analyze_volatility_skew(symbol_data)
    
    # Z-Score
    zscore_info = calculate_zscore_signals(symbol_data)
    
    # LSTM Prediction
    lstm_pred = predict_price_lstm(symbol_data)
    
    # Risk Metrics
    risk_metrics = calculate_risk_metrics(symbol_data)
    
    # Supply/Demand Zones
    supply_demand = detect_supply_demand_zones(symbol_data, idx)
    
    # Anchored VWAP (using 50 candles back as anchor)
    anchor_idx = max(0, idx - 50)
    anchored_vwap = calculate_anchored_vwap(symbol_data, anchor_idx)
    
    # Order Book Simulation
    order_book = simulate_order_book(symbol_data, idx)
    
    # Add new analysis to report
    additional_analysis = ""
    
    if liquidity_info:
        additional_analysis += f"""
💧 FIXED RANGE LIQUIDITY:
────────────────────────────────────────────────────────────────────────────────
Highest Liquidity: {liquidity_info['highest_liquidity']}
Current Position: {liquidity_info['current_position']}
Top Levels: {', '.join([str(l[0]) for l in liquidity_info['liquidity_levels'][:3]])}
"""
    
    if gap_info and gap_info['type'] != 'NO_GAP':
        additional_analysis += f"""
🚀 GAP ANALYSIS:
────────────────────────────────────────────────────────────────────────────────
Type: {gap_info['type']}
Gap: {gap_info['gap_percent']:.2f}%
Fill Probability: {gap_info['fill_probability']:.1f}%
Expected Fill: {gap_info['expected_fill_days']} days
"""
    
    if vol_skew:
        additional_analysis += f"""
📉 VOLATILITY SKEW:
────────────────────────────────────────────────────────────────────────────────
5d Vol: {vol_skew['vol_5d']:.2f}% | 20d Vol: {vol_skew['vol_20d']:.2f}%
Term Structure: {vol_skew['term_structure']}
Signal: {vol_skew['signal']}
"""
    
    if zscore_info:
        additional_analysis += f"""
📊 Z-SCORE (Mean Reversion):
────────────────────────────────────────────────────────────────────────────────
Z-Score: {zscore_info['zscore']:.2f}
Signal: {zscore_info['signal']}
Target: {zscore_info['mean_reversion_target']:.2f}
Confidence: {zscore_info['confidence']:.1f}%
"""
    
    if lstm_pred:
        additional_analysis += f"""
🤖 LSTM PRICE PREDICTION:
────────────────────────────────────────────────────────────────────────────────
Method: {lstm_pred['method']}
Forecast ({lstm_pred['forecast_days']} days): {', '.join([f'{p:.2f}' for p in lstm_pred['predictions'][:3]])}
Trend: {lstm_pred['trend']} | Confidence: {lstm_pred['confidence']:.1f}%
"""
    
    if risk_metrics:
        additional_analysis += f"""
⚠️ RISK METRICS:
────────────────────────────────────────────────────────────────────────────────
VaR (95%): {risk_metrics['var_95']:.2f}% | CVaR: {risk_metrics['cvar_95']:.2f}%
Sharpe: {risk_metrics['sharpe_ratio']:.2f} | Sortino: {risk_metrics['sortino_ratio']:.2f}
Max Drawdown: {risk_metrics['max_drawdown']:.2f}%
Risk Level: {risk_metrics['risk_level']}
"""
    
    if supply_demand:
        additional_analysis += f"""
📦 SUPPLY/DEMAND ZONES:
────────────────────────────────────────────────────────────────────────────────
"""
        for zone in supply_demand[:2]:
            additional_analysis += f"• {zone['type']}: {zone['level_low']:.2f} - {zone['level_high']:.2f} ({zone['freshness']}, {zone['strength']})\n"
    
    if anchored_vwap:
        additional_analysis += f"""
⚓ ANCHORED VWAP (50 bars):
────────────────────────────────────────────────────────────────────────────────
VWAP: {anchored_vwap['anchored_vwap']:.2f}
Deviation: {anchored_vwap['deviation']:.2f}%
Position: {anchored_vwap['position']} | Signal: {anchored_vwap['signal']}
"""
    
    if order_book:
        additional_analysis += f"""
📖 ORDER BOOK (Simulated):
────────────────────────────────────────────────────────────────────────────────
Bid/Ask Ratio: {order_book['bid_ask_ratio']:.2f}
Imbalance: {order_book['imbalance']}
Support: {order_book['nearest_support']:.2f} | Resistance: {order_book['nearest_resistance']:.2f}
"""
    
    elliott_complete_text = ""
    if symbol_data is not None and idx is not None:
        mt_wave = detect_elliott_wave_multi_timeframe(symbol_data, idx)
        if mt_wave and mt_wave.get('higher_timeframe'):
            elliott_complete = mt_wave['higher_timeframe']
            elliott_complete_text = elliott_complete['prediction_text']
            confidence += elliott_complete.get('confidence', 0) // 5
            elliott_complete_text += f"\n\n🔄 MULTI-TIMEFRAME CONFLUENCE: {mt_wave.get('confluence_score', 0)}%\n"
            elliott_complete_text += f"Recommendation: {mt_wave.get('recommendation', 'MODERATE')}"
    
    global elliott_backtester
    if elliott_backtester is None:
        elliott_backtester = ElliottWaveBacktester()
    
    pred_id = elliott_backtester.add_prediction(symbol, current_date, pattern_type, target, current_price)
    
    training_text = f"""
================================================================================
Elliott Wave Pattern: {pattern_display}{variation_note}
Symbol: {symbol}
Date: {current_date}
================================================================================

{sector_details}
{price_sequence_text}
{wyckoff_text}
{pa_report}
{additional_analysis}
{elliott_complete_text}
{forward_text}

{price_header}
────────────────────────────────────────────────────────────────────────────────
Open: {df_row['open']:.2f} | High: {df_row['high']:.2f} | Low: {df_row['low']:.2f}
Close: {current_price:.2f} | Volume: {volume:,}

📈 TECHNICAL INDICATORS:
────────────────────────────────────────────────────────────────────────────────
🔹 RSI (14): {rsi:.1f} | Status: {'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'}
   Divergence: {rsi_div}

🔹 MACD: {macd:.4f} | Signal: {macd_signal:.4f}
   Status: {'Bullish' if macd > macd_signal else 'Bearish'}
   Divergence: {macd_div}

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
{'- RSI divergence confirms' if rsi_div != 'None' else '- No divergence'}
{'- Volume supports' if volume_spike == 'Yes' else '- Volume needs confirmation'}
{sector_analysis.get('additional_note', '')}

================================================================================
"""
    return training_text


def get_elliott_wave_patterns():
    return {
        'Impulse Wave': {'category': 'Motive Wave', 'structure': '5-3-5-3-5', 'bias': 'Bullish', 'degree': 'Primary/Intermediate/Minor', 'fib_ratios': 'Wave 2: 0.382-0.618, Wave 3: 1.618-2.618', 'specifications': 'Wave 1: Initial move\nWave 2: Retracement\nWave 3: Strongest\nWave 4: Correction\nWave 5: Divergence', 'wave_position': '3', 'wave_count': '1-2-3-4-5', 'invalidation': 'Below Wave 1 start'},
        'Leading Diagonal': {'category': 'Motive Wave', 'structure': '5-3-5-3-5', 'bias': 'Bullish/Bearish', 'degree': 'Primary/Intermediate', 'fib_ratios': 'Wave 3: 1.0-1.618', 'specifications': 'Occurs in Wave 1 or A\nOverlapping waves', 'wave_position': '1 or A', 'wave_count': '1-2-3-4-5', 'invalidation': 'Structural violation'},
        'Ending Diagonal': {'category': 'Motive Wave', 'structure': '3-3-3-3-3', 'bias': 'Bullish/Bearish', 'degree': 'Intermediate/Minor', 'fib_ratios': 'Wave 3: 1.0-1.382', 'specifications': 'Occurs in Wave 5 or C\nTerminal pattern', 'wave_position': '5 or C', 'wave_count': '1-2-3-4-5', 'invalidation': 'Pattern expansion'},
        '3rd Wave Extension': {'category': 'Motive Wave', 'structure': '5-3-5-3-5', 'bias': 'Bullish', 'degree': 'Primary/Intermediate', 'fib_ratios': 'Wave 3 = 1.618-2.618 x Wave 1', 'specifications': 'Most common extension\nStrongest momentum', 'wave_position': '3', 'wave_count': '1-2-[3-3-3-3-3]-4-5', 'invalidation': 'Below Wave 1 high'},
        '5th Wave Extension': {'category': 'Motive Wave', 'structure': '5-3-5-3-5', 'bias': 'Bullish/Bearish', 'degree': 'Intermediate/Minor', 'fib_ratios': 'Wave 5 = 0.618-1.618 x Wave 1', 'specifications': 'Terminal move\nDivergence', 'wave_position': '5', 'wave_count': '1-2-3-4-[5-5-5-5-5]', 'invalidation': 'Divergence confirmed'},
        'Single Zigzag': {'category': 'Corrective Wave', 'structure': '5-3-5', 'bias': 'Neutral', 'degree': 'Any', 'fib_ratios': 'Wave B = 0.382-0.786, Wave C = 0.618-1.618', 'specifications': 'Sharp correction', 'wave_position': 'ABC', 'wave_count': 'A-B-C', 'invalidation': 'Complex structure'},
        'Double Zigzag': {'category': 'Corrective Wave', 'structure': '5-3-5-3-5', 'bias': 'Neutral', 'degree': 'Intermediate/Minor', 'fib_ratios': 'Wave Y = 0.618-1.618 x Wave W', 'specifications': 'Two zigzags connected', 'wave_position': 'W-X-Y', 'wave_count': 'W-X-Y', 'invalidation': 'Triple zigzag'},
        'Regular Flat': {'category': 'Corrective Wave', 'structure': '3-3-5', 'bias': 'Neutral', 'degree': 'Any', 'fib_ratios': 'Wave B = 0.90-1.05, Wave C = 1.0 x Wave A', 'specifications': 'Sideways correction', 'wave_position': 'A-B-C', 'wave_count': 'A-B-C', 'invalidation': 'B > 1.05 x A'},
        'Expanded Flat': {'category': 'Corrective Wave', 'structure': '3-3-5', 'bias': 'Neutral', 'degree': 'Intermediate/Minor', 'fib_ratios': 'Wave B = 1.05-1.382 x Wave A', 'specifications': 'Wave B exceeds A start', 'wave_position': 'A-B-C', 'wave_count': 'A-B-C', 'invalidation': 'Wave C extreme'},
        'Contracting Triangle': {'category': 'Corrective Wave', 'structure': '3-3-3-3-3', 'bias': 'Neutral', 'degree': 'Any', 'fib_ratios': 'Wave E = 0.618-0.786 x Wave C', 'specifications': '5 waves contracting', 'wave_position': '4 or B', 'wave_count': 'A-B-C-D-E', 'invalidation': 'Triangle expands'},
        'Expanding Triangle': {'category': 'Corrective Wave', 'structure': '3-3-3-3-3', 'bias': 'Neutral', 'degree': 'Intermediate/Minor', 'fib_ratios': 'Wave E = 1.236-1.382 x Wave C', 'specifications': '5 waves expanding', 'wave_position': '4 or B', 'wave_count': 'A-B-C-D-E', 'invalidation': 'Triangle contracts'},
    }


def get_all_patterns():
    patterns = {
        'Cup and Handle': {'category': 'Continuation', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Breakout above handle', 'stop': 'Below handle low', 'target': 'Measure cup depth'},
        'Ascending Triangle': {'category': 'Continuation', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Breakout above resistance', 'stop': 'Below higher low', 'target': 'Height of triangle'},
        'Bull Flag': {'category': 'Continuation', 'bias': 'Bullish', 'timeframe': 'Short-term', 'entry': 'Breakout of flag', 'stop': 'Below flag low', 'target': 'Flagpole length'},
        'Double Bottom': {'category': 'Reversal', 'bias': 'Bullish', 'timeframe': 'Swing', 'entry': 'Break above neckline', 'stop': 'Below bottom', 'target': 'Pattern height'},
        'Head and Shoulders': {'category': 'Reversal', 'bias': 'Bearish', 'timeframe': 'Swing', 'entry': 'Breakdown neckline', 'stop': 'Above right shoulder', 'target': 'Height'},
        'Hammer': {'category': 'Candlestick', 'bias': 'Bullish', 'timeframe': 'Intraday', 'entry': 'Confirm next green', 'stop': 'Below wick', 'target': 'Recent resistance'},
        'Bullish Engulfing': {'category': 'Candlestick', 'bias': 'Bullish', 'timeframe': 'Intraday', 'entry': 'Engulfing close', 'stop': 'Below candle', 'target': 'Resistance'},
        'Doji': {'category': 'Candlestick', 'bias': 'Neutral', 'timeframe': 'Intraday', 'entry': 'Wait breakout', 'stop': 'High/low', 'target': 'Next move'},
        'Volume Climax': {'category': 'Volume', 'bias': 'Reversal', 'timeframe': 'Any', 'entry': 'Spike volume', 'stop': 'Recent extreme', 'target': 'Reversal zone'},
        'Bollinger Band Squeeze': {'category': 'Volatility', 'bias': 'Breakout', 'timeframe': 'Any', 'entry': 'Expansion', 'stop': 'Opp band', 'target': 'Move'},
        'Break of Structure (BOS)': {'category': 'SMC', 'bias': 'Both', 'timeframe': 'Swing', 'entry': 'Retest', 'stop': 'Beyond swing', 'target': 'Next structure'},
        'Bullish Order Block': {'category': 'SMC', 'bias': 'Bullish', 'timeframe': 'Any', 'entry': 'OB retest', 'stop': 'Below OB', 'target': 'Next liquidity'},
        'Fair Value Gap (FVG)': {'category': 'SMC', 'bias': 'Both', 'timeframe': 'Any', 'entry': 'FVG fill', 'stop': 'Beyond FVG', 'target': 'Next OB'},
        'Optimal Trade Entry (OTE)': {'category': 'SMC', 'bias': 'Both', 'timeframe': 'Short-term', 'entry': '0.618-0.786 Fib', 'stop': 'Beyond 0.786', 'target': 'Swing high/low'},
    }
    patterns.update(get_elliott_wave_patterns())
    return patterns


def generate_complete_pattern_data(symbol, df_row, pattern_type, config, indicator_values, metrics, variation_idx=0, symbol_data=None, idx=None):
    current_price = df_row['close']
    current_date = df_row['date']
    sector = df_row.get('sector', 'Unknown')
    
    rsi = indicator_values.get('rsi', 50)
    macd = indicator_values.get('macd', 0)
    macd_signal = indicator_values.get('macd_signal', 0)
    stoch_k = indicator_values.get('stoch_k', 50)
    atr = indicator_values.get('atr', current_price * 0.02)
    volume = indicator_values.get('volume', 1000000)
    rsi_div = indicator_values.get('rsi_divergence', 'None')
    avg_vol = indicator_values.get('avg_volume', volume)
    
    atr_value = atr if atr > 0 else current_price * 0.02
    
    if config['bias'] == 'Bullish':
        entry, stop = current_price, current_price - atr_value * 1.5
        target = entry + abs(entry - stop) * random.uniform(1.5, 3.0)
    elif config['bias'] == 'Bearish':
        entry, stop = current_price, current_price + atr_value * 1.5
        target = entry - abs(entry - stop) * random.uniform(1.5, 3.0)
    else:
        entry, stop = current_price, current_price - atr_value * 2
        target = entry + abs(entry - stop) * random.uniform(1.0, 2.0)
    
    confidence = 50
    confidence += (macd > macd_signal and config['bias'] == 'Bullish') * 10
    confidence += (macd < macd_signal and config['bias'] == 'Bearish') * 10
    confidence += (rsi < 40 and config['bias'] == 'Bullish') * 5
    confidence += (rsi > 60 and config['bias'] == 'Bearish') * 5
    confidence += (volume > avg_vol * 1.5) * 5
    
    sector_analysis = get_sector_analysis(sector, symbol, current_price)
    confidence += sector_analysis.get('confidence_boost', 0)
    confidence = min(95, max(30, confidence + random.uniform(-5, 5)))
    
    rr_ratio = abs((target - entry) / max(abs(entry - stop), 1e-6))
    volume_spike = "Yes" if volume > avg_vol * 1.5 else "No"
    variation_note = f" [VARIATION {variation_idx + 1}]" if variation_idx > 0 else " [ORIGINAL SEQUENCE]"
    pattern_display = pattern_type if random.random() < 0.5 else "Unknown Pattern"
    
    sector_details = f"""
🏭 SECTOR: {sector} | Strength: {sector_analysis.get('strength', 'Neutral')} | Rotation: {sector_analysis.get('rotation', 'None')}
"""
    
    price_sequence_text = ""
    if symbol_data is not None and idx is not None:
        price_sequence_text, _, _, _ = generate_advanced_price_sequence(symbol_data, idx)
    
    wyckoff_text = ""
    if symbol_data is not None and idx is not None:
        wyckoff_text, _ = detect_volume_price_cycle(symbol_data, idx)
    
    forward_text = ""
    if symbol_data is not None and idx is not None:
        forward_text = generate_forward_looking_analysis(symbol_data, idx)
    
    pa_report, _, _, _, _, _, _ = analyze_price_action_complete(symbol_data, idx) if symbol_data is not None and idx is not None else ("", None, None, None, None, None, None)
    
    # Additional analysis for complete pattern
    liquidity_info = detect_fixed_range_liquidity(symbol_data) if symbol_data is not None else None
    zscore_info = calculate_zscore_signals(symbol_data) if symbol_data is not None else None
    risk_metrics = calculate_risk_metrics(symbol_data) if symbol_data is not None else None
    
    additional_analysis = ""
    if liquidity_info:
        additional_analysis += f"\n💧 Highest Liquidity: {liquidity_info['highest_liquidity']} ({liquidity_info['current_position']})"
    if zscore_info:
        additional_analysis += f"\n📊 Z-Score: {zscore_info['zscore']:.2f} ({zscore_info['signal']})"
    if risk_metrics:
        additional_analysis += f"\n⚠️ VaR: {risk_metrics['var_95']:.2f}% | Sharpe: {risk_metrics['sharpe_ratio']:.2f}"
    
    return f"""
================================================================================
Pattern: {pattern_display}{variation_note}
Symbol: {symbol}
Date: {current_date}
================================================================================

{sector_details}
{price_sequence_text}
{wyckoff_text}
{pa_report}
{forward_text}
{additional_analysis}

📊 PRICE DATA:
────────────────────────────────────────────────────────────────────────────────
Open: {df_row['open']:.2f} | High: {df_row['high']:.2f} | Low: {df_row['low']:.2f}
Close: {current_price:.2f} | Volume: {volume:,}

📈 TECHNICAL INDICATORS:
────────────────────────────────────────────────────────────────────────────────
🔹 RSI (14): {rsi:.1f} | Divergence: {rsi_div}
🔹 MACD: {macd:.4f} | Signal: {macd_signal:.4f}
🔹 Stochastic: %K={stoch_k:.1f}
🔹 ATR: {atr_value:.2f} | Volume Spike: {volume_spike}

🎯 PATTERN ANALYSIS:
────────────────────────────────────────────────────────────────────────────────
Pattern: {pattern_type} | Category: {config['category']} | Bias: {config['bias']}
Entry: {config['entry']} | Stop: {config['stop']} | Target: {config['target']}

💰 TRADING SETUP:
────────────────────────────────────────────────────────────────────────────────
Entry: {entry:.2f} | Stop: {stop:.2f} | Target: {target:.2f}
Risk-Reward: {rr_ratio:.2f} | Confidence: {confidence:.1f}%

📝 RECOMMENDATION:
────────────────────────────────────────────────────────────────────────────────
{'✅ BUY' if config['bias'] == 'Bullish' else '❌ SELL' if config['bias'] == 'Bearish' else '⏳ WAIT'} at {entry:.2f}
{sector_analysis.get('additional_note', '')}

================================================================================
"""


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    
    # main() ফাংশনে
    for idx in range(50, len(symbol_data), step):
    detected_patterns = detect_all_patterns(symbol_data, idx)
    print(f"🔍 {symbol} at idx {idx}: {detected_patterns}")  # ← ডিবাগ লাইন

    
    global elliott_backtester
    elliott_backtester = ElliottWaveBacktester()
    
    print("="*80)
    print("🚀 COMPLETE PATTERN TRAINING DATA GENERATOR")
    print("   (130+ Patterns + Elliott Wave + SMC + Price Action + Multi-Timeframe + ML + Backtesting + Risk Metrics)")
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
        symbol_data = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        if len(symbol_data) < 100:
            continue
        
        close_prices = symbol_data['close']
        high_prices = symbol_data['high']
        low_prices = symbol_data['low']
        volumes = symbol_data['volume']
        
        rsi_series = calculate_rsi(close_prices)
        macd_line, macd_signal, macd_hist = calculate_macd(close_prices)
        stoch_k, stoch_d = calculate_stochastic(high_prices, low_prices, close_prices)
        obv_series = calculate_obv(close_prices, volumes)
        ema_20 = calculate_ema(close_prices, 20)
        sma_20 = calculate_sma(close_prices, 20)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_prices)
        atr_series = calculate_atr(high_prices, low_prices, close_prices)
        avg_volume = volumes.rolling(20).mean()
        vwap = (close_prices * volumes).cumsum() / volumes.cumsum()
        
        symbol_data['rsi'] = rsi_series
        symbol_data['bb_upper'] = bb_upper
        symbol_data['bb_middle'] = bb_middle
        symbol_data['bb_lower'] = bb_lower
        
        market_regime = detect_market_regime(close_prices)
        print(f"📊 Market regime for {symbol}: {market_regime}")
        
        row_count = 0
        last_detected_idx = {}
        step = 1 if len(symbol_data) < 500 else 2
        
        for idx in range(50, len(symbol_data), step):
            detected_patterns = detect_all_patterns(symbol_data, idx)
            if not detected_patterns:
                continue
            
            row = symbol_data.iloc[idx]
            indicator_values = {
                'rsi': rsi_series.iloc[idx] if idx < len(rsi_series) else 50,
                'macd': macd_line.iloc[idx] if idx < len(macd_line) else 0,
                'macd_signal': macd_signal.iloc[idx] if idx < len(macd_signal) else 0,
                'stoch_k': stoch_k.iloc[idx] if idx < len(stoch_k) else 50,
                'stoch_d': stoch_d.iloc[idx] if idx < len(stoch_d) else 50,
                'obv': obv_series.iloc[idx] if idx < len(obv_series) else 0,
                'atr': atr_series.iloc[idx] if idx < len(atr_series) else row['close'] * 0.02,
                'ema_20': ema_20.iloc[idx] if idx < len(ema_20) else row['close'],
                'sma_20': sma_20.iloc[idx] if idx < len(sma_20) else row['close'],
                'volume': volumes.iloc[idx],
                'avg_volume': avg_volume.iloc[idx] if idx < len(avg_volume) else volumes.iloc[idx],
                'rsi_divergence': 'None',
                'macd_divergence': 'None',
            }
            
            pattern_high, pattern_low = row['high'], row['low']
            real_sequence = close_prices.iloc[max(0, idx-30):idx+1].values
            
            for pattern_name in detected_patterns[:3]:
                if pattern_name not in all_patterns:
                    continue
                
                config = all_patterns[pattern_name]
                metrics = calculate_pattern_metrics(real_sequence, pattern_high, pattern_low, row['close'])
                
                is_elliott = pattern_name in get_elliott_wave_patterns()
                
                if is_elliott:
                    text = generate_elliott_wave_data(symbol, row, pattern_name, config, indicator_values, metrics, 0, symbol_data, idx)
                else:
                    text = generate_complete_pattern_data(symbol, row, pattern_name, config, indicator_values, metrics, 0, symbol_data, idx)
                
                if text:
                    training_data.append(text)
                    print(f"✅ Generated {pattern_name} for {symbol} on {row['date'].date()}")
                    row_count += 1
                    if row_count >= MAX_PER_SYMBOL:
                        break
            
            if row_count >= MAX_PER_SYMBOL:
                break
        
        symbols_processed += 1
        if symbols_processed >= MAX_SYMBOLS:
            break
    
    output_file = "./csv/training_texts.txt"
    os.makedirs("./csv", exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(training_data))
    
    print(f"\n📊 Training data saved: {output_file}")
    print(f"   Total examples: {len(training_data)}")
    
    if elliott_backtester:
        print(elliott_backtester.get_performance_report())
    
    print("\n" + "="*80)
    print("📤 NEXT STEPS:")
    print("="*80)
    print("1. Upload training_texts.txt to Hugging Face dataset")
    print("2. Then retrain your LLM: python scripts/llm_train.py")
    print("="*80)


if __name__ == "__main__":
    main()
