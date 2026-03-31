# elliott-wave.py
# Elliott Wave Detection - Group by Symbol

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =========================
# Configuration
# =========================
INPUT_FILE = "/csv/mongodb.csv"
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
# Wave Detection for a Symbol
# =========================
def detect_wave_for_symbol(df, symbol_name):
    """Detect Elliott Wave for a single symbol"""
    
    if len(df) < 30:
        return {
            'symbol': symbol_name,
            'current_wave': 'Insufficient Data',
            'current_subwave': 'Need more data',
            'wave_type': 'N/A',
            'direction': 'N/A',
            'next_wave': 'N/A',
            'next_subwave': 'N/A',
            'entry_zone': 'N/A',
            'entry_time': 'N/A',
            'stop_loss': 'N/A',
            'take_profit': 'N/A',
            'confidence': 0,
            'rsi': 50,
            'price': df['close'].iloc[-1] if len(df) > 0 else 0,
            'date': df['date'].iloc[-1].strftime('%Y-%m-%d') if len(df) > 0 else datetime.now().strftime('%Y-%m-%d')
        }
    
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Detect swings
    swings = detect_zigzag(df, pct=3, min_bars=3)
    
    if len(swings) < 3:
        return {
            'symbol': symbol_name,
            'current_wave': 'Consolidation',
            'current_subwave': 'No clear structure',
            'wave_type': 'Neutral',
            'direction': 'Sideways',
            'next_wave': 'Breakout',
            'next_subwave': 'Wait for confirmation',
            'entry_zone': 'Monitor only',
            'entry_time': 'N/A',
            'stop_loss': 'N/A',
            'take_profit': 'N/A',
            'confidence': 30,
            'rsi': round(df['rsi'].iloc[-1], 1),
            'price': round(df['close'].iloc[-1], 2),
            'date': df['date'].iloc[-1].strftime('%Y-%m-%d')
        }
    
    # Detect wave pattern
    types = [s['type'] for s in swings]
    current_price = df['close'].iloc[-1]
    
    # ========== Bullish Impulse (up-down-up-down-up) ==========
    if len(types) >= 5 and types[-5:] == ['up', 'down', 'up', 'down', 'up']:
        # Check wave positions
        if len(swings) >= 6 and swings[-1]['date'] > swings[-2]['date']:
            current_wave = "Wave 5"
            sub_wave = "Wave 5 - Sub-wave v"
            wave_type = "Impulse Wave"
            direction = "Bullish"
            next_wave = "ABC Correction"
            next_subwave = "Wave A - Start"
            entry_zone = "SELL ZONE - Take profits"
            entry_time = "Now"
            stop_loss = f"${current_price * 1.02:.2f}"
            take_profit = f"${current_price * 0.92:.2f}"
            confidence = 85
        elif len(swings) >= 4:
            current_wave = "Wave 3"
            sub_wave = "Wave 3 - Sub-wave iii (Strongest)"
            wave_type = "Impulse Wave"
            direction = "Bullish"
            next_wave = "Wave 4"
            next_subwave = "Wave 4 - Correction"
            entry_zone = "HOLD - Wave 3 in progress"
            entry_time = "N/A"
            stop_loss = f"${current_price * 0.92:.2f}"
            take_profit = f"${current_price * 1.15:.2f}"
            confidence = 90
        else:
            current_wave = "Wave 2"
            sub_wave = "Wave 2 - Sub-wave c (Entry Zone)"
            wave_type = "Impulse Wave"
            direction = "Bullish"
            next_wave = "Wave 3"
            next_subwave = "Wave 3 - Sub-wave i"
            entry_zone = f"BUY ZONE - ${current_price * 0.96:.2f} to ${current_price:.2f}"
            entry_time = "Now - 3 days"
            stop_loss = f"${current_price * 0.94:.2f}"
            take_profit = f"${current_price * 1.12:.2f}"
            confidence = 75
    
    # ========== Bearish Impulse (down-up-down-up-down) ==========
    elif len(types) >= 5 and types[-5:] == ['down', 'up', 'down', 'up', 'down']:
        if len(swings) >= 6:
            current_wave = "Wave 5"
            sub_wave = "Wave 5 - Sub-wave v"
            wave_type = "Impulse Wave"
            direction = "Bearish"
            next_wave = "ABC Correction"
            next_subwave = "Wave A - Start"
            entry_zone = "BUY ZONE - Cover shorts"
            entry_time = "Now"
            stop_loss = f"${current_price * 0.98:.2f}"
            take_profit = f"${current_price * 0.92:.2f}"
            confidence = 85
        else:
            current_wave = "Wave 3"
            sub_wave = "Wave 3 - Sub-wave iii (Strongest)"
            wave_type = "Impulse Wave"
            direction = "Bearish"
            next_wave = "Wave 4"
            next_subwave = "Wave 4 - Correction"
            entry_zone = "SELL ZONE - Short entry"
            entry_time = "Now"
            stop_loss = f"${current_price * 1.02:.2f}"
            take_profit = f"${current_price * 0.88:.2f}"
            confidence = 90
    
    # ========== Zigzag Correction (down-up-down) ==========
    elif len(types) >= 3 and types[-3:] == ['down', 'up', 'down']:
        current_wave = "Wave C"
        sub_wave = "Wave C - Sub-wave v"
        wave_type = "Zigzag Correction"
        direction = "Bullish Ending"
        next_wave = "New Impulse Wave 1"
        next_subwave = "Wave 1 - Sub-wave i"
        entry_zone = f"BUY ZONE - ${current_price * 0.95:.2f}"
        entry_time = "5-10 days"
        stop_loss = f"${current_price * 0.92:.2f}"
        take_profit = f"${current_price * 1.10:.2f}"
        confidence = 70
    
    elif len(types) >= 3 and types[-3:] == ['up', 'down', 'up']:
        current_wave = "Wave C"
        sub_wave = "Wave C - Sub-wave v"
        wave_type = "Zigzag Correction"
        direction = "Bearish Ending"
        next_wave = "New Impulse Wave 1"
        next_subwave = "Wave 1 - Sub-wave i"
        entry_zone = f"SELL ZONE - ${current_price * 1.05:.2f}"
        entry_time = "5-10 days"
        stop_loss = f"${current_price * 1.08:.2f}"
        take_profit = f"${current_price * 0.92:.2f}"
        confidence = 70
    
    # ========== Flat Correction ==========
    elif len(types) >= 3 and types[-3:] == ['down', 'up', 'down']:
        prices = [s['price'] for s in swings[-3:]]
        a_move = abs(prices[1] - prices[0])
        b_move = abs(prices[2] - prices[1])
        ratio = b_move / a_move if a_move > 0 else 0
        
        if 0.9 <= ratio <= 1.1:
            current_wave = "Wave C"
            sub_wave = "Wave C - Flat Correction"
            wave_type = "Regular Flat Correction"
            direction = "Sideways"
            next_wave = "Trend Resumption"
            next_subwave = "Wave 1 - Sub-wave i"
            entry_zone = f"BUY ZONE - ${current_price * 0.97:.2f}"
            entry_time = "5-12 days"
            stop_loss = f"${current_price * 0.95:.2f}"
            take_profit = f"${current_price * 1.08:.2f}"
            confidence = 75
    
    # ========== Triangle Pattern ==========
    elif len(types) >= 5:
        # Check for alternating pattern (characteristic of triangles)
        alternating = all(types[i] != types[i+1] for i in range(len(types)-1))
        if alternating:
            current_wave = "Wave E"
            sub_wave = "Wave E - Breakout Soon"
            wave_type = "Contracting Triangle"
            direction = "Neutral"
            next_wave = "Breakout"
            next_subwave = "Wave 3 - Sub-wave iii"
            entry_zone = "WAIT - Breakout entry"
            entry_time = "3-10 days"
            stop_loss = f"${current_price * 0.97:.2f}"
            take_profit = f"${current_price * 1.12:.2f}"
            confidence = 80
    
    # ========== Default - Consolidation ==========
    else:
        current_wave = "Consolidation"
        sub_wave = "No clear structure"
        wave_type = "Neutral"
        direction = "Sideways"
        next_wave = "Breakout"
        next_subwave = "Wait for confirmation"
        entry_zone = "Monitor only - No entry"
        entry_time = "N/A"
        stop_loss = "N/A"
        take_profit = "N/A"
        confidence = 30
    
    # Get RSI and other indicators
    rsi_val = round(df['rsi'].iloc[-1], 1)
    
    return {
        'symbol': symbol_name,
        'date': df['date'].iloc[-1].strftime('%Y-%m-%d'),
        'current_price': round(current_price, 2),
        'current_wave': current_wave,
        'current_subwave': sub_wave,
        'wave_type': wave_type,
        'direction': direction,
        'next_wave': next_wave,
        'next_subwave': next_subwave,
        'entry_zone': entry_zone,
        'entry_time': entry_time,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'confidence_score': confidence,
        'rsi': rsi_val,
        'swing_points': len(swings),
        'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# =========================
# Main Execution
# =========================

def main():
    print("=" * 80)
    print("ELLIOTT WAVE ANALYSIS - GROUP BY SYMBOL")
    print("=" * 80)
    
    try:
        # Check if input file exists
        if not os.path.exists(INPUT_FILE):
            print(f"\n⚠️ File not found: {INPUT_FILE}")
            print("Creating sample data for testing...")
            
            # Create sample data with multiple symbols
            dates = pd.date_range(start='2023-01-01', end='2024-01-15', freq='D')
            np.random.seed(42)
            
            symbols = ['MONGODB', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD']
            all_data = []
            
            for sym in symbols:
                close = 100 + np.random.randn() * 50
                closes = []
                for i in range(len(dates)):
                    close = close + np.random.randn() * 2
                    closes.append(close)
                
                df_sym = pd.DataFrame({
                    'date': dates,
                    'open': [c - np.random.rand() * 2 for c in closes],
                    'high': [c + abs(np.random.randn()) * 2 for c in closes],
                    'low': [c - abs(np.random.randn()) * 2 for c in closes],
                    'close': closes,
                    'volume': np.random.randint(100000, 1000000, len(dates)),
                    'symbol': sym
                })
                all_data.append(df_sym)
            
            df = pd.concat(all_data, ignore_index=True)
            df.to_csv(INPUT_FILE, index=False)
            print(f"✅ Sample data created with {len(symbols)} symbols")
        
        # Load data
        print(f"\n📂 Loading: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        df['date'] = pd.to_datetime(df['date'])
        
        # Get unique symbols
        symbols = df['symbol'].unique()
        print(f"📊 Found {len(symbols)} symbols: {', '.join(symbols)}")
        
        # Analyze each symbol
        results = []
        for i, symbol in enumerate(symbols, 1):
            print(f"\n🔍 Analyzing [{i}/{len(symbols)}]: {symbol}")
            
            # Filter data for this symbol
            symbol_df = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)
            
            # Skip if not enough data
            if len(symbol_df) < 30:
                print(f"   ⚠️ Insufficient data ({len(symbol_df)} rows). Minimum 30 rows required.")
                results.append({
                    'symbol': symbol,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'current_price': 0,
                    'current_wave': 'Insufficient Data',
                    'current_subwave': 'Need 30+ days data',
                    'wave_type': 'N/A',
                    'direction': 'N/A',
                    'next_wave': 'N/A',
                    'next_subwave': 'N/A',
                    
                    'confidence_score': 0,
                   
                })
                continue
            
            # Detect wave for this symbol
            result = detect_wave_for_symbol(symbol_df, symbol)
            results.append(result)
            
            # Print brief result
            print(f"   ✅ Wave: {result['current_wave']} | Sub-wave: {result['current_subwave']} | Conf: {result['confidence_score']}%")
        
        # Create output dataframe (one row per symbol)
        output_df = pd.DataFrame(results)
        
        # Reorder columns
        column_order = [
            'symbol', 'date', 'current_price', 'current_wave', 'current_subwave',
            'wave_type', 'direction', 'next_wave', 'next_subwave',
            'entry_zone', 'entry_time', 'stop_loss', 'take_profit',
            'confidence_score', 'rsi', 'swing_points', 'analysis_time'
        ]
        
        # Keep only columns that exist
        existing_cols = [col for col in column_order if col in output_df.columns]
        output_df = output_df[existing_cols]
        
        # Save to CSV
        output_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        
        # Print summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("=" * 80)
        print(f"\nTotal Symbols Analyzed: {len(results)}")
        print(f"Output File: {OUTPUT_FILE}")
        
        print("\n" + "-" * 80)
        print("SYMBOL WAVE SUMMARY")
        print("-" * 80)
        print(f"{'Symbol':<12} {'Current Wave':<20} {'Direction':<12} {'Confidence':<10}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['symbol']:<12} {r['current_wave']:<20} {r['direction']:<12} {r['confidence_score']:<10}%")
        
        print("\n" + "=" * 80)
        print(f"✅ Complete! Saved to: {OUTPUT_FILE}")
        print("=" * 80)
        
        return output_df
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()