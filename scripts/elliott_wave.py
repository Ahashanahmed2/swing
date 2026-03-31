# elliott-wave-complete.py
# Complete Elliott Wave Detection System with All Patterns

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# Configuration
# =========================
INPUT_FILE = "./csv/mongodb.csv"
OUTPUT_DIR = "./output/ai_signal"
OUTPUT_FILE = f"{OUTPUT_DIR}/Elliott_wave_complete.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Calculate Indicators
# =========================
def calculate_indicators(df):
    """Calculate technical indicators"""
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume MA
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    return df

# =========================
# ZigZag Detection
# =========================
def detect_zigzag(df, pct=3, min_bars=3):
    """Detect swing points"""
    swings = []
    last_pivot_idx = 0
    last_pivot_price = df['close'].iloc[0]
    trend = None
    
    for i in range(1, len(df)):
        change = (df['close'].iloc[i] - last_pivot_price) / last_pivot_price * 100
        bars_since_last = i - last_pivot_idx
        
        if trend is None:
            if abs(change) > pct:
                trend = 'up' if change > 0 else 'down'
                last_pivot_price = df['close'].iloc[i]
                last_pivot_idx = i
                swings.append({
                    'index': i,
                    'price': last_pivot_price,
                    'type': trend,
                    'date': df['date'].iloc[i],
                    'volume': df['volume'].iloc[i],
                    'rsi': df['rsi'].iloc[i]
                })
        
        elif trend == 'up':
            if df['close'].iloc[i] > last_pivot_price:
                last_pivot_price = df['close'].iloc[i]
                last_pivot_idx = i
                swings[-1].update({
                    'index': i,
                    'price': last_pivot_price,
                    'date': df['date'].iloc[i]
                })
            elif change < -pct and bars_since_last >= min_bars:
                trend = 'down'
                last_pivot_price = df['close'].iloc[i]
                last_pivot_idx = i
                swings.append({
                    'index': i,
                    'price': last_pivot_price,
                    'type': 'down',
                    'date': df['date'].iloc[i],
                    'volume': df['volume'].iloc[i],
                    'rsi': df['rsi'].iloc[i]
                })
        
        elif trend == 'down':
            if df['close'].iloc[i] < last_pivot_price:
                last_pivot_price = df['close'].iloc[i]
                last_pivot_idx = i
                swings[-1].update({
                    'index': i,
                    'price': last_pivot_price,
                    'date': df['date'].iloc[i]
                })
            elif change > pct and bars_since_last >= min_bars:
                trend = 'up'
                last_pivot_price = df['close'].iloc[i]
                last_pivot_idx = i
                swings.append({
                    'index': i,
                    'price': last_pivot_price,
                    'type': 'up',
                    'date': df['date'].iloc[i],
                    'volume': df['volume'].iloc[i],
                    'rsi': df['rsi'].iloc[i]
                })
    
    return swings

# =========================
# Complete Elliott Wave Analysis
# =========================
class ElliottWaveAnalyzer:
    def __init__(self, swings, df):
        self.swings = swings
        self.df = df
        self.current_price = df['close'].iloc[-1]
        self.current_date = df['date'].iloc[-1]
        
    def analyze(self):
        """Complete wave analysis"""
        result = {
            'current_main_wave': None,
            'current_sub_wave': None,
            'wave_type': None,
            'direction': None,
            'next_wave': None,
            'next_subwave': None,
            'entry_zone': None,
            'entry_time': None,
            'stop_loss': None,
            'take_profit': None,
            'confidence': 0,
            'higher_timeframe_wave': None,
            'higher_timeframe_subwave': None,
            'validation_status': []
        }
        
        # Detect current wave pattern
        pattern = self.detect_current_pattern()
        
        if pattern:
            result.update(pattern)
            result['validation_status'] = self.validate_wave_rules()
        
        # Higher timeframe analysis
        result['higher_timeframe_wave'], result['higher_timeframe_subwave'] = self.analyze_higher_timeframe()
        
        # Generate future projections
        future = self.generate_future_projection(result)
        result.update(future)
        
        return result
    
    def detect_current_pattern(self):
        """Detect current Elliott Wave pattern"""
        if len(self.swings) < 3:
            return {
                'current_main_wave': 'No Clear Wave',
                'current_sub_wave': 'Consolidation',
                'wave_type': 'Neutral',
                'direction': 'Sideways',
                'confidence': 20
            }
        
        recent = self.swings[-8:] if len(self.swings) >= 8 else self.swings
        types = [s['type'] for s in recent]
        prices = [s['price'] for s in recent]
        volumes = [s['volume'] for s in recent]
        
        # ==================== IMPULSE WAVE DETECTION ====================
        # Bullish Impulse: up-down-up-down-up
        if len(types) >= 5 and types[-5:] == ['up', 'down', 'up', 'down', 'up']:
            # Check validation rules
            wave1_range = abs(prices[-5] - prices[-4])
            wave2_range = abs(prices[-4] - prices[-3])
            wave3_range = abs(prices[-3] - prices[-2])
            wave4_range = abs(prices[-2] - prices[-1])
            
            # Wave 2 cannot retrace more than 100% of Wave 1
            if wave2_range <= wave1_range:
                # Wave 3 cannot be shortest
                if wave3_range > wave1_range and wave3_range > wave4_range:
                    # Wave 4 cannot overlap Wave 1
                    if (prices[-2] > prices[-5] if types[-3] == 'up' else prices[-2] < prices[-5]):
                        
                        # Determine which wave we're in
                        if self.is_wave_complete(recent[-1], recent[-2]):
                            current_wave = "Wave 5"
                            sub_wave = "5-v (Final)"
                            next_wave = "ABC Correction"
                            next_subwave = "Wave A"
                            entry = "SELL - Take profits"
                            entry_time = "Now"
                            tp = self.current_price * 0.92
                            sl = self.current_price * 1.02
                        elif len(recent) >= 4 and self.is_wave_complete(recent[-2], recent[-3]):
                            current_wave = "Wave 4"
                            sub_wave = "4-C (Zigzag)"
                            next_wave = "Wave 5"
                            next_subwave = "5-i"
                            entry = "BUY - Wave 4 bottom"
                            entry_time = "3-5 days"
                            tp = self.current_price * 1.08
                            sl = self.current_price * 0.95
                        elif len(recent) >= 3 and self.is_wave_complete(recent[-3], recent[-4]):
                            current_wave = "Wave 3"
                            sub_wave = "3-iii (Mega Extension)"
                            next_wave = "Wave 4"
                            next_subwave = "4-A"
                            entry = "HOLD - Wave 3 in progress"
                            entry_time = "N/A"
                            tp = self.current_price * 1.15
                            sl = self.current_price * 0.92
                        else:
                            current_wave = "Wave 2"
                            sub_wave = "2-C (Double Zigzag)"
                            next_wave = "Wave 3"
                            next_subwave = "3-i"
                            entry = "BUY ZONE - Strong entry"
                            entry_time = "Now - 2 days"
                            tp = self.current_price * 1.12
                            sl = self.current_price * 0.94
                        
                        return {
                            'current_main_wave': current_wave,
                            'current_sub_wave': sub_wave,
                            'wave_type': 'Impulse Wave',
                            'direction': 'Bullish',
                            'next_wave': next_wave,
                            'next_subwave': next_subwave,
                            'entry_zone': entry,
                            'entry_time': entry_time,
                            'stop_loss': sl,
                            'take_profit': tp,
                            'confidence': self.calculate_confidence(recent, volumes, 'impulse')
                        }
        
        # Bearish Impulse: down-up-down-up-down
        if len(types) >= 5 and types[-5:] == ['down', 'up', 'down', 'up', 'down']:
            wave1_range = abs(prices[-5] - prices[-4])
            wave3_range = abs(prices[-3] - prices[-2])
            
            if wave3_range > wave1_range:
                if self.is_wave_complete(recent[-1], recent[-2]):
                    return {
                        'current_main_wave': 'Wave 5',
                        'current_sub_wave': '5-v (Final)',
                        'wave_type': 'Impulse Wave',
                        'direction': 'Bearish',
                        'next_wave': 'ABC Correction',
                        'next_subwave': 'Wave A',
                        'entry_zone': 'SELL - Take profits',
                        'entry_time': 'Now',
                        'stop_loss': self.current_price * 0.98,
                        'take_profit': self.current_price * 0.92,
                        'confidence': 80
                    }
                else:
                    return {
                        'current_main_wave': 'Wave 3',
                        'current_sub_wave': '3-iii (Mega Extension)',
                        'wave_type': 'Impulse Wave',
                        'direction': 'Bearish',
                        'next_wave': 'Wave 4',
                        'next_subwave': '4-A',
                        'entry_zone': 'SELL - Short entry',
                        'entry_time': 'Now',
                        'stop_loss': self.current_price * 1.02,
                        'take_profit': self.current_price * 0.88,
                        'confidence': 75
                    }
        
        # ==================== ZIGZAG CORRECTION (5-3-5) ====================
        if len(types) >= 3 and types[-3:] == ['down', 'up', 'down']:
            a_move = abs(prices[-3] - prices[-2])
            b_move = abs(prices[-2] - prices[-1])
            retrace = b_move / a_move if a_move > 0 else 0
            
            if 0.382 <= retrace <= 0.618:
                if self.is_wave_complete(recent[-1], recent[-2]):
                    return {
                        'current_main_wave': 'Wave C',
                        'current_sub_wave': 'C-1 (Impulse)',
                        'wave_type': 'Zigzag Correction',
                        'direction': 'Bullish (Ending)',
                        'next_wave': 'New Impulse Wave 1',
                        'next_subwave': '1-i',
                        'entry_zone': 'BUY - Correction ending',
                        'entry_time': '3-7 days',
                        'stop_loss': self.current_price * 0.96,
                        'take_profit': self.current_price * 1.10,
                        'confidence': 70
                    }
                else:
                    return {
                        'current_main_wave': 'Wave B',
                        'current_sub_wave': 'B-2 (Flat)',
                        'wave_type': 'Zigzag Correction',
                        'direction': 'Counter-trend',
                        'next_wave': 'Wave C',
                        'next_subwave': 'C-1',
                        'entry_zone': 'WAIT - Let B complete',
                        'entry_time': '5-10 days',
                        'stop_loss': None,
                        'take_profit': None,
                        'confidence': 60
                    }
        
        # ==================== FLAT CORRECTION (3-3-5) ====================
        if len(types) >= 3 and types[-3:] == ['down', 'up', 'down']:
            a_move = abs(prices[-3] - prices[-2])
            b_move = abs(prices[-2] - prices[-1])
            ratio = b_move / a_move if a_move > 0 else 0
            
            # Regular Flat (B = A)
            if 0.9 <= ratio <= 1.1:
                return {
                    'current_main_wave': 'Wave C',
                    'current_sub_wave': 'C-1 (Impulse)',
                    'wave_type': 'Regular Flat Correction',
                    'direction': 'Sideways',
                    'next_wave': 'Trend Resumption',
                    'next_subwave': 'Wave 1',
                    'entry_zone': 'BUY - After C completes',
                    'entry_time': '5-12 days',
                    'stop_loss': self.current_price * 0.95,
                    'take_profit': self.current_price * 1.08,
                    'confidence': 75
                }
            
            # Expanded Flat (B > A)
            elif 1.2 <= ratio <= 1.618:
                return {
                    'current_main_wave': 'Wave C',
                    'current_sub_wave': 'C-2 (Ending Diagonal)',
                    'wave_type': 'Expanded Flat Correction',
                    'direction': 'Complex',
                    'next_wave': 'Trend Resumption',
                    'next_subwave': 'Wave 3',
                    'entry_zone': 'CAUTION - Wait for confirmation',
                    'entry_time': '8-15 days',
                    'stop_loss': None,
                    'take_profit': None,
                    'confidence': 65
                }
        
        # ==================== TRIANGLE CORRECTION (3-3-3-3-3) ====================
        if len(types) >= 5:
            # Check for alternating pattern
            if all(types[i] != types[i+1] for i in range(len(types)-1)):
                highs = [prices[i] for i in range(len(prices)) if types[i] == 'up']
                lows = [prices[i] for i in range(len(prices)) if types[i] == 'down']
                
                if len(highs) >= 3 and len(lows) >= 2:
                    # Contracting Triangle
                    if highs[0] > highs[1] and lows[0] < lows[1]:
                        return {
                            'current_main_wave': 'Wave E',
                            'current_sub_wave': 'E (Final)',
                            'wave_type': 'Contracting Triangle',
                            'direction': 'Neutral - Breakout Soon',
                            'next_wave': 'Breakout',
                            'next_subwave': 'Wave 3 of Impulse',
                            'entry_zone': 'Breakout Entry',
                            'entry_time': '3-10 days',
                            'stop_loss': self.current_price * 0.97,
                            'take_profit': self.current_price * 1.12,
                            'confidence': 80
                        }
        
        # ==================== DIAGONAL PATTERNS ====================
        # Leading Diagonal (Wave 1 or A)
        if len(types) >= 5 and types[-5:] == ['up', 'down', 'up', 'down', 'up']:
            if self.has_overlap(prices):
                return {
                    'current_main_wave': 'Wave 1 or A',
                    'current_sub_wave': '5-v (Leading Diagonal)',
                    'wave_type': 'Leading Diagonal',
                    'direction': 'Bullish',
                    'next_wave': 'Wave 2 or B',
                    'next_subwave': 'Correction',
                    'entry_zone': 'WAIT - Let correction complete',
                    'entry_time': '5-10 days',
                    'stop_loss': None,
                    'take_profit': None,
                    'confidence': 70
                }
        
        # Ending Diagonal (Wave 5 or C)
        if len(types) >= 5 and types[-5:] == ['up', 'down', 'up', 'down', 'up']:
            if self.has_convergence(prices):
                return {
                    'current_main_wave': 'Wave 5 or C',
                    'current_sub_wave': '5-iii (Ending Diagonal)',
                    'wave_type': 'Ending Diagonal',
                    'direction': 'Bullish (Exhaustion)',
                    'next_wave': 'Reversal',
                    'next_subwave': 'Wave A',
                    'entry_zone': 'SELL - Reversal expected',
                    'entry_time': 'Now - 3 days',
                    'stop_loss': self.current_price * 1.02,
                    'take_profit': self.current_price * 0.90,
                    'confidence': 75
                }
        
        # ==================== DOUBLE ZIGZAG (W-X-Y) ====================
        if len(swings) >= 5:
            w = self.swings[-5:-3]
            x = self.swings[-3]
            y = self.swings[-2:]
            
            if len(w) >= 2 and len(y) >= 2:
                if w[0]['type'] != w[1]['type'] and y[0]['type'] != y[1]['type']:
                    return {
                        'current_main_wave': 'Wave Y',
                        'current_sub_wave': 'Y-C (Final)',
                        'wave_type': 'Double Zigzag',
                        'direction': 'Bullish',
                        'next_wave': 'New Impulse Wave 1',
                        'next_subwave': '1-i',
                        'entry_zone': 'BUY - Complex correction ending',
                        'entry_time': '5-15 days',
                        'stop_loss': self.current_price * 0.95,
                        'take_profit': self.current_price * 1.12,
                        'confidence': 68
                    }
        
        # Default
        return {
            'current_main_wave': 'Consolidation',
            'current_sub_wave': 'No Clear Structure',
            'wave_type': 'Neutral',
            'direction': 'Sideways',
            'next_wave': 'Uncertain',
            'next_subwave': 'Wait for breakout',
            'entry_zone': 'NO ENTRY',
            'entry_time': 'Monitor only',
            'stop_loss': None,
            'take_profit': None,
            'confidence': 30
        }
    
    def validate_wave_rules(self):
        """Validate Elliott Wave rules"""
        validations = []
        
        if len(self.swings) >= 3:
            recent = self.swings[-3:]
            prices = [s['price'] for s in recent]
            
            # Rule: Wave 2 cannot retrace >100% of Wave 1
            if len(self.swings) >= 4:
                w1_range = abs(self.swings[-4]['price'] - self.swings[-3]['price'])
                w2_range = abs(self.swings[-3]['price'] - self.swings[-2]['price'])
                if w2_range <= w1_range:
                    validations.append("✓ Wave 2 rule: PASS")
                else:
                    validations.append("✗ Wave 2 rule: FAIL - Wave 2 retraced >100%")
            
            # Rule: Wave 3 cannot be shortest
            if len(self.swings) >= 5:
                w1_range = abs(self.swings[-5]['price'] - self.swings[-4]['price'])
                w3_range = abs(self.swings[-3]['price'] - self.swings[-2]['price'])
                w5_range = abs(self.swings[-2]['price'] - self.swings[-1]['price'])
                if w3_range > w1_range and w3_range > w5_range:
                    validations.append("✓ Wave 3 rule: PASS - Wave 3 is longest")
                else:
                    validations.append("✗ Wave 3 rule: FAIL - Wave 3 is shortest")
            
            # Rule: Wave 4 cannot overlap Wave 1
            if len(self.swings) >= 5:
                w1_high = max(self.swings[-5]['price'], self.swings[-4]['price'])
                w4_low = min(self.swings[-2]['price'], self.swings[-1]['price'])
                if (self.swings[-5]['type'] == 'up' and w4_low > w1_high) or \
                   (self.swings[-5]['type'] == 'down' and w4_low < w1_high):
                    validations.append("✓ Wave 4 rule: PASS - No overlap")
                else:
                    validations.append("✗ Wave 4 rule: FAIL - Wave 4 overlaps Wave 1")
        
        return validations
    
    def analyze_higher_timeframe(self):
        """Analyze larger structure (weekly/daily)"""
        if len(self.swings) < 10:
            return "Insufficient Data", "N/A"
        
        # Look for completed cycles
        cycles = []
        for i in range(0, len(self.swings) - 5, 5):
            cycle = self.swings[i:i+5]
            types = [c['type'] for c in cycle]
            
            if types == ['up', 'down', 'up', 'down', 'up']:
                cycles.append(('Bullish Impulse', cycle[-1]['date']))
            elif types == ['down', 'up', 'down', 'up', 'down']:
                cycles.append(('Bearish Impulse', cycle[-1]['date']))
        
        if cycles:
            last_cycle = cycles[-1]
            return f"{last_cycle[0]} (Completed)", f"Wave 5 of {last_cycle[0]}"
        
        return "Wave 3 of Larger Impulse", "Sub-wave iii of Larger Cycle"
    
    def generate_future_projection(self, result):
        """Generate future wave projections with timeline"""
        current_wave = result.get('current_main_wave', '')
        direction = result.get('direction', '')
        
        projections = []
        
        if 'Wave 2' in current_wave:
            projections.append({
                'wave': 'Wave 3',
                'subwave': '3-i to 3-v',
                'timeline': '2-4 months',
                'direction': 'Up' if 'Bullish' in direction else 'Down'
            })
            projections.append({
                'wave': 'Wave 4',
                'subwave': 'Correction',
                'timeline': '1-2 months after Wave 3',
                'direction': 'Opposite'
            })
            projections.append({
                'wave': 'Wave 5',
                'subwave': 'Final Push',
                'timeline': '3-6 months',
                'direction': 'Same as Wave 1'
            })
        
        elif 'Wave 3' in current_wave:
            projections.append({
                'wave': 'Wave 4',
                'subwave': 'Flat/Triangle',
                'timeline': '1-2 months',
                'direction': 'Opposite'
            })
            projections.append({
                'wave': 'Wave 5',
                'subwave': 'Ending Diagonal possible',
                'timeline': '2-3 months after Wave 4',
                'direction': 'Same as Wave 3'
            })
        
        elif 'Wave 4' in current_wave:
            projections.append({
                'wave': 'Wave 5',
                'subwave': 'Final Wave',
                'timeline': '1-3 months',
                'direction': 'Same as Wave 3'
            })
            projections.append({
                'wave': 'ABC Correction',
                'subwave': 'A-B-C',
                'timeline': '2-4 months after Wave 5',
                'direction': 'Opposite'
            })
        
        elif 'Wave C' in current_wave or 'Wave 5' in current_wave:
            projections.append({
                'wave': 'New Impulse Wave 1',
                'subwave': 'Starting',
                'timeline': '1-2 months',
                'direction': 'Opposite to current'
            })
            projections.append({
                'wave': 'Wave 3',
                'subwave': 'Strongest',
                'timeline': '3-5 months',
                'direction': 'Same as Wave 1'
            })
        
        return {
            'future_projections': projections,
            'projection_table': self.create_projection_table(projections)
        }
    
    def create_projection_table(self, projections):
        """Create HTML/Text table for projections"""
        if not projections:
            return "No clear projections available"
        
        table = "\n" + "=" * 70 + "\n"
        table += "🔮 FUTURE WAVE PROJECTIONS\n"
        table += "=" * 70 + "\n"
        table += f"{'Wave':<15} {'Sub-wave':<20} {'Timeline':<20} {'Direction':<15}\n"
        table += "-" * 70 + "\n"
        
        for proj in projections:
            table += f"{proj['wave']:<15} {proj['subwave']:<20} {proj['timeline']:<20} {proj['direction']:<15}\n"
        
        table += "=" * 70
        return table
    
    def calculate_confidence(self, swings, volumes, pattern_type):
        """Calculate confidence score"""
        confidence = 50  # Base
        
        # Volume confirmation
        if volumes and volumes[-1] > np.mean(volumes):
            confidence += 15
        
        # RSI confirmation
        rsi = self.df['rsi'].iloc[-1]
        if pattern_type == 'impulse' and 30 <= rsi <= 70:
            confidence += 15
        elif pattern_type == 'corrective' and (rsi < 30 or rsi > 70):
            confidence += 15
        
        # Wave count
        if len(swings) >= 5:
            confidence += 10
        
        # ATR volatility
        if self.df['atr'].iloc[-1] > self.df['atr'].rolling(50).mean().iloc[-1]:
            confidence += 10
        
        return min(confidence, 100)
    
    def is_wave_complete(self, current, previous):
        """Check if current wave is complete"""
        # Check for reversal patterns
        if current['type'] != previous['type']:
            return True
        
        # Check for momentum loss
        if current['rsi'] < previous['rsi'] and current['type'] == previous['type']:
            return True
        
        return False
    
    def has_overlap(self, prices):
        """Check if waves overlap (diagonal characteristic)"""
        if len(prices) >= 5:
            return prices[-3] < prices[-5] if prices[-5] < prices[-4] else prices[-3] > prices[-5]
        return False
    
    def has_convergence(self, prices):
        """Check for convergence (ending diagonal)"""
        if len(prices) >= 5:
            highs = [prices[i] for i in range(len(prices)) if i % 2 == 0]
            if len(highs) >= 3:
                return highs[0] > highs[1] > highs[2] or highs[0] < highs[1] < highs[2]
        return False

# =========================
# Create Output DataFrame
# =========================
def create_output(analysis, df, symbol):
    """Create final output with all required fields"""
    
    # Create projection table string
    projection_str = ""
    if analysis.get('projection_table'):
        projection_str = analysis['projection_table']
    
    # Get validation status
    validation_str = " | ".join(analysis.get('validation_status', []))
    
    output = {
        'symbol': symbol,
        'তারিখ': df['date'].iloc[-1].strftime('%Y-%m-%d'),
        'বর্তমান_প্রাইস': round(df['close'].iloc[-1], 2),
        
        # Main Wave Analysis
        'এলিয়ট_ওয়েব_বর্তমান': analysis['current_main_wave'],
        'সাব_ওয়েব_বর্তমান': analysis['current_sub_wave'],
        'ওয়েব_টাইপ': analysis['wave_type'],
        'ট্রেন্ড_ডিরেকশন': analysis['direction'],
        
        # Next Wave Projection
        'পরবর্তী_ওয়েব': analysis['next_wave'],
        'পরবর্তী_সাব_ওয়েব': analysis['next_subwave'],
        
        # Entry Strategy
        'এন্ট্রি_জোন': analysis['entry_zone'],
        'এন্ট্রির_সময়': analysis['entry_time'],
        'স্টপ_লস': round(analysis['stop_loss'], 2) if analysis['stop_loss'] else 'N/A',
        'টেক_প্রফিট': round(analysis['take_profit'], 2) if analysis['take_profit'] else 'N/A',
        
        # Confidence & Validation
        'কনফিডেন্স_স্কোর': analysis['confidence'],
        'ভ্যালিডেশন_স্ট্যাটাস': validation_str if validation_str else 'Validating...',
        
        # Higher Timeframe
        'হায়ার_টাইমফ্রেম_ওয়েব': analysis['higher_timeframe_wave'],
        'হায়ার_টাইমফ্রেম_সাবওয়েব': analysis['higher_timeframe_subwave'],
        
        # Future Projections
        'ভবিষ্যত_প্রজেকশন': projection_str,
        
        # Current Market Data
        'আরএসআই': round(df['rsi'].iloc[-1], 1),
        'ভলিউম_রেশিও': round(df['volume'].iloc[-1] / df['volume_ma'].iloc[-1], 2),
        'এটিআর': round(df['atr'].iloc[-1], 2),
        
        # Timestamp
        'বিশ্লেষণের_সময়': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return output

# =========================
# Main Execution
# =========================
def main():
    print("=" * 80)
    print("🌊 COMPLETE ELLIOTT WAVE ANALYSIS SYSTEM")
    print("=" * 80)
    
    # Load data
    print(f"\n📂 Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Get symbol
    symbol = df['symbol'].iloc[-1] if 'symbol' in df.columns else 'UNKNOWN'
    
    # Detect swings
    swings = detect_zigzag(df, pct=3, min_bars=3)
    print(f"   Found {len(swings)} swing points")
    
    # Analyze
    analyzer = ElliottWaveAnalyzer(swings, df)
    analysis = analyzer.analyze()
    
    # Create output
    output_data = create_output(analysis, df, symbol)
    
    # Save to CSV
    output_df = pd.DataFrame([output_data])
    output_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 ANALYSIS RESULTS")
    print("=" * 80)
    print(f"\n📍 Symbol: {symbol}")
    print(f"📅 Date: {output_data['তারিখ']}")
    print(f"💰 Current Price: ${output_data['বর্তমান_প্রাইস']}")
    print(f"🎯 RSI: {output_data['আরএসআই']}")
    
    print("\n" + "-" * 50)
    print("🔍 CURRENT WAVE POSITION")
    print("-" * 50)
    print(f"Main Wave: {output_data['এলিয়ট_ওয়েব_বর্তমান']}")
    print(f"Sub-wave: {output_data['সাব_ওয়েব_বর্তমান']}")
    print(f"Wave Type: {output_data['ওয়েব_টাইপ']}")
    print(f"Direction: {output_data['ট্রেন্ড_ডিরেকশন']}")
    
    print("\n" + "-" * 50)
    print("➡️ NEXT WAVE PROJECTION")
    print("-" * 50)
    print(f"Next Wave: {output_data['পরবর্তী_ওয়েব']}")
    print(f"Next Sub-wave: {output_data['পরবর্তী_সাব_ওয়েব']}")
    
    print("\n" + "-" * 50)
    print("💰 ENTRY STRATEGY")
    print("-" * 50)
    print(f"Entry Zone: {output_data['এন্ট্রি_জোন']}")
    print(f"Entry Time: {output_data['এন্ট্রির_সময়']}")
    print(f"Stop Loss: {output_data['স্টপ_লস']}")
    print(f"Take Profit: {output_data['টেক_প্রফিট']}")
    
    print("\n" + "-" * 50)
    print("📈 CONFIDENCE & VALIDATION")
    print("-" * 50)
    print(f"Confidence Score: {output_data['কনফিডেন্স_স্কোর']}%")
    print(f"Validation: {output_data['ভ্যালিডেশন_স্ট্যাটাস'][:100]}...")
    
    print("\n" + "-" * 50)
    print("🌍 HIGHER TIMEFRAME")
    print("-" * 50)
    print(f"Higher TF Wave: {output_data['হায়ার_টাইমফ্রেম_ওয়েব']}")
    print(f"Higher TF Sub-wave: {output_data['হায়ার_টাইমফ্রেম_সাবওয়েব']}")
    
    print("\n" + "=" * 80)
    print(f"✅ Complete! Saved to: {OUTPUT_FILE}")
    print("=" * 80)
    
    return output_df

if __name__ == "__main__":
    result = main()
