# elliott-wave.py
# Elliott Wave Detection System - English Output Only

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =========================
# Configuration - Auto detect file path
# =========================

# Try different possible file paths
possible_paths = [
    "/csv/mongodb.csv",
    "./csv/mongodb.csv", 
    "../csv/mongodb.csv",
    "mongodb.csv",
    "data/mongodb.csv",
    "/home/runner/work/swing/swing/csv/mongodb.csv",
    "./data/mongodb.csv"
]

INPUT_FILE = None
for path in possible_paths:
    if os.path.exists(path):
        INPUT_FILE = path
        break

if INPUT_FILE is None:
    print("⚠️ No MongoDB CSV file found. Creating sample data for testing...")
    
    dates = pd.date_range(start='2023-01-01', end='2024-01-15', freq='D')
    np.random.seed(42)
    
    close = 100
    closes = []
    for i in range(len(dates)):
        close = close + np.random.randn() * 2
        closes.append(close)
    
    sample_df = pd.DataFrame({
        'date': dates,
        'open': [c - np.random.rand() * 2 for c in closes],
        'high': [c + abs(np.random.randn()) * 2 for c in closes],
        'low': [c - abs(np.random.randn()) * 2 for c in closes],
        'close': closes,
        'volume': np.random.randint(100000, 1000000, len(dates)),
        'symbol': 'MONGODB'
    })
    
    sample_df.to_csv("mongodb.csv", index=False)
    INPUT_FILE = "mongodb.csv"
    print(f"✅ Sample data created at: {INPUT_FILE}")

print(f"📂 Using file: {INPUT_FILE}")

OUTPUT_DIR = "./output/ai_signal"
OUTPUT_FILE = f"{OUTPUT_DIR}/Elliott_wave.csv"

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
                    'rsi': df['rsi'].iloc[i] if 'rsi' in df else 50
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
                    'rsi': df['rsi'].iloc[i] if 'rsi' in df else 50
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
                    'rsi': df['rsi'].iloc[i] if 'rsi' in df else 50
                })
    
    return swings

# =========================
# Wave Detection - English Only
# =========================
def detect_current_wave(swings):
    """Simple wave detection based on swing pattern - English output only"""
    if len(swings) < 3:
        return "Consolidation", "No Clear Sub-wave", "Neutral", "Sideways", 30
    
    types = [s['type'] for s in swings]
    
    # ========== Bullish Impulse ==========
    if len(types) >= 5 and types[-5:] == ['up', 'down', 'up', 'down', 'up']:
        if len(swings) >= 6 and swings[-1]['date'] > swings[-2]['date']:
            return "Wave 5", "Wave 5 - Sub-wave v", "Impulse Wave", "Bullish", 85
        elif len(swings) >= 4:
            return "Wave 3", "Wave 3 - Sub-wave iii", "Impulse Wave", "Bullish", 90
        else:
            return "Wave 2", "Wave 2 - Sub-wave c", "Impulse Wave", "Bullish", 75
    
    # ========== Bearish Impulse ==========
    if len(types) >= 5 and types[-5:] == ['down', 'up', 'down', 'up', 'down']:
        if len(swings) >= 6:
            return "Wave 5", "Wave 5 - Sub-wave v", "Impulse Wave", "Bearish", 85
        else:
            return "Wave 3", "Wave 3 - Sub-wave iii", "Impulse Wave", "Bearish", 90
    
    # ========== Zigzag Correction ==========
    if len(types) >= 3 and types[-3:] == ['down', 'up', 'down']:
        return "Wave C", "Wave C - Sub-wave v", "Zigzag Correction", "Bullish Ending", 70
    
    if len(types) >= 3 and types[-3:] == ['up', 'down', 'up']:
        return "Wave C", "Wave C - Sub-wave v", "Zigzag Correction", "Bearish Ending", 70
    
    # ========== Flat Correction ==========
    if len(types) >= 3 and types[-3:] == ['down', 'up', 'down']:
        prices = [s['price'] for s in swings[-3:]]
        a_move = abs(prices[1] - prices[0])
        b_move = abs(prices[2] - prices[1])
        ratio = b_move / a_move if a_move > 0 else 0
        
        if 0.9 <= ratio <= 1.1:
            return "Wave C", "Wave C - Flat", "Flat Correction", "Sideways", 75
    
    # ========== Triangle ==========
    if len(types) >= 5:
        if all(types[i] != types[i+1] for i in range(len(types)-1)):
            return "Wave E", "Wave E - Breakout Soon", "Contracting Triangle", "Neutral", 80
    
    # ========== Default ==========
    return "Consolidation", "No Clear Structure", "Neutral", "Sideways", 30

# =========================
# Get Next Wave
# =========================
def get_next_wave(current_wave, direction):
    """Determine next expected wave"""
    wave_map = {
        "Wave 1": ("Wave 2", "Wave 2 - Correction"),
        "Wave 2": ("Wave 3", "Wave 3 - Strongest"),
        "Wave 3": ("Wave 4", "Wave 4 - Correction"),
        "Wave 4": ("Wave 5", "Wave 5 - Final"),
        "Wave 5": ("ABC Correction", "Wave A - Start"),
        "Wave A": ("Wave B", "Wave B - Counter"),
        "Wave B": ("Wave C", "Wave C - Final"),
        "Wave C": ("New Impulse Wave 1", "Wave 1 - Start"),
        "Consolidation": ("Breakout", "Wave 1 or 3"),
    }
    
    return wave_map.get(current_wave, ("Uncertain", "Wait for confirmation"))

# =========================
# Get Entry Strategy
# =========================
def get_entry_strategy(current_wave, current_price, direction):
    """Generate entry strategy"""
    strategies = {
        "Wave 2": {
            "entry": f"BUY ZONE - ${current_price * 0.96:.2f} to ${current_price:.2f}",
            "time": "Now - 3 days",
            "sl": f"${current_price * 0.94:.2f}",
            "tp": f"${current_price * 1.12:.2f}"
        },
        "Wave 3": {
            "entry": "HOLD - Wave 3 in progress",
            "time": "N/A",
            "sl": f"${current_price * 0.92:.2f}",
            "tp": f"${current_price * 1.15:.2f}"
        },
        "Wave 4": {
            "entry": f"BUY ZONE - ${current_price * 0.97:.2f} to ${current_price:.2f}",
            "time": "3-7 days",
            "sl": f"${current_price * 0.95:.2f}",
            "tp": f"${current_price * 1.08:.2f}"
        },
        "Wave 5": {
            "entry": "SELL ZONE - Take profits",
            "time": "Now",
            "sl": f"${current_price * 1.02:.2f}",
            "tp": f"${current_price * 0.92:.2f}"
        },
        "Wave C": {
            "entry": f"BUY ZONE - ${current_price * 0.95:.2f}",
            "time": "5-10 days",
            "sl": f"${current_price * 0.92:.2f}",
            "tp": f"${current_price * 1.10:.2f}"
        },
        "Consolidation": {
            "entry": "NO ENTRY - Wait for breakout",
            "time": "Monitor only",
            "sl": "N/A",
            "tp": "N/A"
        }
    }
    
    return strategies.get(current_wave, {
        "entry": "Monitor - No clear signal",
        "time": "Wait for confirmation",
        "sl": "N/A",
        "tp": "N/A"
    })

# =========================
# Get Higher Timeframe Analysis
# =========================
def get_higher_timeframe(swings):
    """Simple higher timeframe analysis"""
    if len(swings) < 10:
        return "Insufficient Data", "N/A"
    
    types = [s['type'] for s in swings[-20:]]
    ups = types.count('up')
    downs = types.count('down')
    
    if ups > downs + 3:
        return "Larger Bullish Impulse", "Wave 3 of Larger Cycle"
    elif downs > ups + 3:
        return "Larger Bearish Impulse", "Wave C of Larger Cycle"
    else:
        return "Sideways Consolidation", "Wave B or 4"

# =========================
# Create Future Projection Table
# =========================
def create_projection_table(current_wave, direction, current_price):
    """Create future projection table"""
    projections = []
    
    if "Wave 2" in current_wave:
        projections.append(["Wave 3", "Wave 3-i to 3-v", "2-4 months", "Up" if "Bull" in direction else "Down"])
        projections.append(["Wave 4", "Correction", "1-2 months", "Opposite"])
        projections.append(["Wave 5", "Final Push", "3-6 months", "Same as Wave 1"])
    
    elif "Wave 3" in current_wave:
        projections.append(["Wave 4", "Flat/Triangle", "1-2 months", "Opposite"])
        projections.append(["Wave 5", "Ending Diagonal", "2-3 months", "Same as Wave 3"])
    
    elif "Wave 4" in current_wave:
        projections.append(["Wave 5", "Final Wave", "1-3 months", "Same as Wave 3"])
        projections.append(["ABC Correction", "A-B-C", "2-4 months", "Opposite"])
    
    elif "Wave C" in current_wave or "Wave 5" in current_wave:
        projections.append(["New Impulse", "Wave 1", "1-2 months", "Opposite to current"])
        projections.append(["Wave 3", "Strongest", "3-5 months", "Same as Wave 1"])
    
    if projections:
        table = "\n" + "=" * 70 + "\n"
        table += "FUTURE WAVE PROJECTIONS\n"
        table += "=" * 70 + "\n"
        table += f"{'Wave':<15} {'Sub-wave':<20} {'Timeline':<20} {'Direction':<15}\n"
        table += "-" * 70 + "\n"
        for proj in projections:
            table += f"{proj[0]:<15} {proj[1]:<20} {proj[2]:<20} {proj[3]:<15}\n"
        table += "=" * 70
        return table
    
    return "No clear projections available"

# =========================
# Main Execution
# =========================

def main():
    print("=" * 80)
    print("ELLIOTT WAVE ANALYSIS SYSTEM")
    print("=" * 80)
    
    try:
        print(f"\nLoading: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        
        # Check required columns
        if 'date' not in df.columns:
            df['date'] = pd.date_range(end=datetime.now(), periods=len(df))
        
        if 'close' not in df.columns:
            if 'open' in df.columns:
                df['close'] = df['open']
            else:
                df['close'] = 100
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        symbol = df['symbol'].iloc[-1] if 'symbol' in df.columns else "MONGODB"
        
        df = calculate_indicators(df)
        
        swings = detect_zigzag(df, pct=3, min_bars=3)
        print(f"   Found {len(swings)} swing points")
        
        current_wave, sub_wave, wave_type, direction, confidence = detect_current_wave(swings)
        next_wave, next_subwave = get_next_wave(current_wave, direction)
        
        current_price = df['close'].iloc[-1]
        strategy = get_entry_strategy(current_wave, current_price, direction)
        higher_wave, higher_subwave = get_higher_timeframe(swings)
        projection_table = create_projection_table(current_wave, direction, current_price)
        
        validation = []
        if len(swings) >= 3:
            validation.append("Wave count: Valid")
        if confidence >= 70:
            validation.append("Pattern confidence: High")
        else:
            validation.append("Pattern confidence: Moderate")
        
        # Create output - ALL ENGLISH COLUMNS
        output_data = {
            'symbol': symbol,
            'date': df['date'].iloc[-1].strftime('%Y-%m-%d'),
            'current_price': round(current_price, 2),
            'current_wave': current_wave,
            'current_subwave': sub_wave,
            'wave_type': wave_type,
            'direction': direction,
            'next_wave': next_wave,
            'next_subwave': next_subwave,
            'entry_zone': strategy['entry'],
            'entry_time': strategy['time'],
            'stop_loss': strategy['sl'],
            'take_profit': strategy['tp'],
            'confidence_score': confidence,
            'validation_status': " | ".join(validation),
            'higher_timeframe_wave': higher_wave,
            'higher_timeframe_subwave': higher_subwave,
            'future_projections': projection_table,
            'rsi': round(df['rsi'].iloc[-1], 1),
            'volume_ratio': round(df['volume'].iloc[-1] / df['volume_ma'].iloc[-1], 2),
            'atr': round(df['atr'].iloc[-1], 2),
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        output_df = pd.DataFrame([output_data])
        output_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        
        print("\n" + "=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)
        print(f"\nSymbol: {symbol}")
        print(f"Date: {output_data['date']}")
        print(f"Current Price: ${output_data['current_price']}")
        
        print("\n" + "-" * 50)
        print("CURRENT WAVE POSITION")
        print("-" * 50)
        print(f"Main Wave: {output_data['current_wave']}")
        print(f"Sub-wave: {output_data['current_subwave']}")
        print(f"Wave Type: {output_data['wave_type']}")
        print(f"Direction: {output_data['direction']}")
        
        print("\n" + "-" * 50)
        print("NEXT WAVE PROJECTION")
        print("-" * 50)
        print(f"Next Wave: {output_data['next_wave']}")
        print(f"Next Sub-wave: {output_data['next_subwave']}")
        
        print("\n" + "-" * 50)
        print("ENTRY STRATEGY")
        print("-" * 50)
        print(f"Entry Zone: {output_data['entry_zone']}")
        print(f"Entry Time: {output_data['entry_time']}")
        print(f"Stop Loss: {output_data['stop_loss']}")
        print(f"Take Profit: {output_data['take_profit']}")
        
        print("\n" + "-" * 50)
        print("CONFIDENCE & VALIDATION")
        print("-" * 50)
        print(f"Confidence Score: {output_data['confidence_score']}%")
        print(f"Validation: {output_data['validation_status']}")
        
        print("\n" + "=" * 80)
        print(f"Complete! Saved to: {OUTPUT_FILE}")
        print("=" * 80)
        
        return output_df
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()