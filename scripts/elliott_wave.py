# ================== elliott_wave_detector.py ==================
# ✅ প্রতিটি symbol-এর জন্য Elliott Wave ডিটেক্ট
# ✅ Main Wave, Sub-Wave, Sub-Sub-Wave সহ সম্পূর্ণ বিশ্লেষণ
# ✅ generate_pattern_training_data_complete.py ব্যবহার করে
# ✅ Output: ./csv/elliott_wave_detailed.csv

import pandas as pd
import numpy as np
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# =========================================================
# PATH SETUP
# =========================================================
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = "./csv/mongodb.csv"
OUTPUT_CSV = "./output/ai_signal/elliott_wave_detailed.csv"

import os

# ✅ ডিরেক্টরি না থাকলে তৈরি করুন
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# =========================================================
# IMPORT ELLIOTT WAVE DETECTOR
# =========================================================
try:
    from scripts.generate_pattern_training_data_complete import (
        detect_elliott_wave_complete,
        detect_elliott_wave_patterns_from_complete,
        find_swing_points,
        find_swing_points_with_indices
    )
    ELLIOTT_AVAILABLE = True
    print("✅ Elliott Wave module loaded")
except ImportError as e:
    print(f"⚠️ Elliott Wave module not available: {e}")
    ELLIOTT_AVAILABLE = False

# =========================================================
# FALLBACK ZIGZAG DETECTOR
# =========================================================
def detect_zigzag_fallback(df, pct=2.5):
    """Fallback swing detector"""
    swings = []
    last_price = df['close'].iloc[0]
    last_idx = 0
    trend = None

    for i in range(1, len(df)):
        change = (df['close'].iloc[i] - last_price) / last_price * 100

        if trend is None:
            if abs(change) > pct:
                trend = 'up' if change > 0 else 'down'
                last_price = df['close'].iloc[i]
                last_idx = i
                swings.append({'price': last_price, 'type': trend, 'idx': i})

        elif trend == 'up':
            if df['close'].iloc[i] > last_price:
                last_price = df['close'].iloc[i]
                last_idx = i
                swings[-1]['price'] = last_price
                swings[-1]['idx'] = i
            elif change < -pct:
                trend = 'down'
                last_price = df['close'].iloc[i]
                last_idx = i
                swings.append({'price': last_price, 'type': 'down', 'idx': i})

        elif trend == 'down':
            if df['close'].iloc[i] < last_price:
                last_price = df['close'].iloc[i]
                last_idx = i
                swings[-1]['price'] = last_price
                swings[-1]['idx'] = i
            elif change > pct:
                trend = 'up'
                last_price = df['close'].iloc[i]
                last_idx = i
                swings.append({'price': last_price, 'type': 'up', 'idx': i})

    return swings


# =========================================================
# MAIN DETECTION FUNCTION
# =========================================================
def detect_elliott_wave_detailed(symbol, symbol_df):
    """
    Complete Elliott Wave Detection for a single symbol
    Returns detailed wave structure
    """
    
    result = {
        'symbol': symbol,
        'main_wave': 'Unknown',
        'sub_wave': 'N/A',
        'sub_sub_wave': 'N/A',
        'wave_count': 'N/A',
        'wave_structure': 'N/A',
        'current_wave': 'Unknown',
        'wave_position': 'N/A',
        'is_bullish': False,
        'confidence': 0,
        'total_waves_detected': 0,
        'pattern_type': 'None',
        'swing_count': 0
    }
    
    if len(symbol_df) < 30:
        result['main_wave'] = 'Insufficient Data'
        result['sub_wave'] = f'Need 30+ candles (has {len(symbol_df)})'
        return result
    
    try:
        # =============================================
        # METHOD 1: Complete Elliott Wave Detection
        # =============================================
        if ELLIOTT_AVAILABLE:
            try:
                elliott_result = detect_elliott_wave_complete(
                    symbol_df, 
                    len(symbol_df) - 1, 
                    lookback=200
                )
                
                if elliott_result and elliott_result.get('wave_structure'):
                    wave_structure = elliott_result['wave_structure']
                    
                    # Main wave count
                    wave_count = wave_structure.get('wave_count', [])
                    if wave_count:
                        result['wave_count'] = ' _ '.join(wave_count)
                        result['current_wave'] = wave_count[-1]
                        result['total_waves_detected'] = len(wave_count)
                        
                        # Determine main wave type
                        if len(wave_count) >= 5:
                            result['main_wave'] = 'Impulse Wave (5-Wave)'
                        elif len(wave_count) >= 3:
                            result['main_wave'] = 'Corrective Wave (ABC)'
                        else:
                            result['main_wave'] = 'Developing Wave'
                    
                    # Sub-waves
                    sub_waves = wave_structure.get('sub_waves', {})
                    if sub_waves:
                        sub_wave_parts = []
                        sub_sub_parts = []
                        
                        for wave_name, sub_info in sub_waves.items():
                            if isinstance(sub_info, dict):
                                structure = sub_info.get('structure', 'N/A')
                                sub_list = sub_info.get('sub_waves', [])
                                
                                # Sub-wave description
                                sub_wave_parts.append(f"{wave_name}:{structure}({'-'.join(sub_list)})")
                                
                                # Sub-sub-wave (nested)
                                nested = sub_info.get('nested_waves', {})
                                if nested:
                                    for nested_name, nested_info in nested.items():
                                        if isinstance(nested_info, dict):
                                            nested_struct = nested_info.get('structure', 'N/A')
                                            sub_sub_parts.append(f"{wave_name}.{nested_name}:{nested_struct}")
                        
                        if sub_wave_parts:
                            result['sub_wave'] = ' | '.join(sub_wave_parts)
                        if sub_sub_parts:
                            result['sub_sub_wave'] = ' | '.join(sub_sub_parts)
                    
                    # Confidence
                    result['confidence'] = wave_structure.get('confidence', 0)
                    
                    # Bullish/Bearish
                    result['is_bullish'] = elliott_result.get('is_bullish', False)
                    
                    # Pattern type
                    pattern_type = elliott_result.get('pattern_type', '')
                    if pattern_type:
                        result['pattern_type'] = pattern_type
                    elif result['is_bullish']:
                        result['pattern_type'] = 'Bullish Elliott Wave'
                    else:
                        result['pattern_type'] = 'Bearish Elliott Wave'
                    
                    # Wave position details
                    if result['current_wave'] != 'Unknown':
                        result['wave_position'] = f"Currently in {result['current_wave']}"
                        if result['is_bullish']:
                            result['wave_position'] += " (Uptrend)"
                        else:
                            result['wave_position'] += " (Downtrend)"
                    
                    return result
                    
            except Exception as e:
                pass  # Fall through to fallback
        
        # =============================================
        # METHOD 2: Pattern-based Detection
        # =============================================
        if ELLIOTT_AVAILABLE:
            try:
                patterns = detect_elliott_wave_patterns_from_complete(symbol_df)
                
                if patterns and len(patterns) > 0:
                    latest_pattern = patterns[-1] if isinstance(patterns, list) else patterns
                    
                    if isinstance(latest_pattern, dict):
                        result['main_wave'] = latest_pattern.get('pattern_name', 'Pattern Detected')
                        result['sub_wave'] = latest_pattern.get('description', 'N/A')
                        result['confidence'] = latest_pattern.get('confidence', 40)
                        result['pattern_type'] = latest_pattern.get('pattern_type', 'Elliott Pattern')
                        result['is_bullish'] = latest_pattern.get('is_bullish', False)
                        
                        return result
            except:
                pass
        
        # =============================================
        # METHOD 3: Swing Analysis (Fallback)
        # =============================================
        swings = detect_zigzag_fallback(symbol_df)
        result['swing_count'] = len(swings)
        
        if len(swings) >= 5:
            types = [s['type'] for s in swings[-5:]]
            prices = [s['price'] for s in swings[-5:]]
            
            if types == ['up', 'down', 'up', 'down', 'up']:
                result['main_wave'] = 'Impulse Wave (Bullish)'
                result['current_wave'] = 'Wave 5'
                result['wave_count'] = '1  2  3  4  5'
                result['sub_wave'] = '5-Wave Bullish Impulse'
                result['is_bullish'] = True
                result['confidence'] = 50
                result['total_waves_detected'] = 5
                result['pattern_type'] = 'Bullish 5-Wave Impulse'
                
            elif types == ['down', 'up', 'down', 'up', 'down']:
                result['main_wave'] = 'Impulse Wave (Bearish)'
                result['current_wave'] = 'Wave 5'
                result['wave_count'] = '1 2  3  4  5'
                result['sub_wave'] = '5-Wave Bearish Impulse'
                result['is_bullish'] = False
                result['confidence'] = 50
                result['total_waves_detected'] = 5
                result['pattern_type'] = 'Bearish 5-Wave Impulse'
                
        elif len(swings) >= 3:
            types = [s['type'] for s in swings[-3:]]
            prices = [s['price'] for s in swings[-3:]]
            
            if types == ['down', 'up', 'down']:
                result['main_wave'] = 'Corrective Wave (ABC)'
                result['current_wave'] = 'Wave C'
                result['wave_count'] = 'A  B  C'
                result['sub_wave'] = 'ABC Bearish Correction'
                result['is_bullish'] = False
                result['confidence'] = 40
                result['total_waves_detected'] = 3
                result['pattern_type'] = 'Bearish ABC Correction'
                
            elif types == ['up', 'down', 'up']:
                result['main_wave'] = 'Corrective Wave (ABC)'
                result['current_wave'] = 'Wave C'
                result['wave_count'] = 'A  B  C'
                result['sub_wave'] = 'ABC Bullish Correction'
                result['is_bullish'] = True
                result['confidence'] = 40
                result['total_waves_detected'] = 3
                result['pattern_type'] = 'Bullish ABC Correction'
        
        elif len(swings) >= 1:
            last_swing = swings[-1]
            result['main_wave'] = f"Developing Wave ({last_swing['type'].title()} Swing)"
            result['current_wave'] = 'Wave 1'
            result['wave_count'] = '1'
            result['sub_wave'] = 'Early Stage - Pattern Forming'
            result['confidence'] = 20
            result['total_waves_detected'] = 1
            
    except Exception as e:
        result['main_wave'] = f'Error: {str(e)[:50]}'
        result['sub_wave'] = 'Detection failed'
    
    return result


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 70)
    print("🌊 ELLIOTT WAVE DETAILED DETECTOR")
    print("   Source: generate_pattern_training_data_complete.py + mongodb.csv")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading market data...")
    if not os.path.exists(INPUT_CSV):
        print(f"❌ {INPUT_CSV} not found!")
        return
    
    df = pd.read_csv(INPUT_CSV)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    symbols = sorted(df['symbol'].unique())
    print(f"   ✅ {len(symbols)} symbols loaded")
    
    # Process each symbol
    results = []
    total = len(symbols)
    
    print(f"\n🔍 Detecting Elliott Waves for {total} symbols...")
    
    for i, symbol in enumerate(symbols, 1):
        if i % 50 == 0:
            print(f"   Progress: {i}/{total}...")
        
        symbol_df = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        result = detect_elliott_wave_detailed(symbol, symbol_df)
        results.append(result)
    
    # Create DataFrame
    output_df = pd.DataFrame(results)
    
    # Sort by confidence
    output_df = output_df.sort_values('confidence', ascending=False)
    # =========================================================

    # =========================================================
    # সেভ করার আগে SUB_WAVE truncate
    max_len = 20  # সর্বোচ্চ ৮০ ক্যারেক্টার

    for col in ['sub_wave', 'sub_sub_wave', 'wave_structure','confidence']:
        if col in output_df.columns:
            output_df[col] = output_df[col].astype(str).str.slice(0, max_len)

    # Wave count-এ arrow replace
    #output_df['wave_count'] = output_df['wave_count'].astype(str).str.replace('→', '->', regex=False)

    

    # সেভ
    output_df[cols_to_save].to_csv(OUTPUT_CSV, index=False)
    print(f"\n{'='*70}")
    print(f"{'='*70}")
    print(f"   Total symbols: {len(output_df)}")
    print(f"   Output: {OUTPUT_CSV}")
    
    # Distribution
    print(f"\n📈 WAVE DISTRIBUTION:")
    wave_counts = output_df['main_wave'].value_counts().head(10)
    for wave, count in wave_counts.items():
        pct = count / len(output_df) * 100
        print(f"   {wave:<40}: {count:>4} ({pct:>5.1f}%)")
    
    # Bullish/Bearish
    bullish = output_df['is_bullish'].sum()
    bearish = len(output_df) - bullish
    print(f"\n🐂 Bullish: {bullish} ({bullish/len(output_df)*100:.1f}%)")
    print(f"🐻 Bearish: {bearish} ({bearish/len(output_df)*100:.1f}%)")
    
    # Top confidence
    print(f"\n⭐ TOP CONFIDENCE DETECTIONS:")
    top5 = output_df[output_df['confidence'] > 0].head(5)
    if len(top5) > 0:
        for _, row in top5.iterrows():
            print(f"   {row['symbol']:<15} | {row['main_wave']:<30} | Conf: {row['confidence']}% | Bullish: {row['is_bullish']}")
    else:
        print("   (No high confidence detections yet)")
    
    # Sample
    print(f"\n📋 SAMPLE OUTPUT (5 rows):")
    cols = ['symbol', 'main_wave', 'current_wave', 'confidence', 'is_bullish', 'pattern_type']
    print(output_df[cols].head(5).to_string())
    
    print(f"\n{'='*70}")
    print(f"✅ Done! Output: {OUTPUT_CSV}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
