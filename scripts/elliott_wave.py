import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# =========================
# কনফিগারেশন
# =========================
INPUT_FILE = "./csv/mongodb.csv"
OUTPUT_FILE = "./csv/Elliott_wave.csv"

# =========================
# ফিবোনাচি টুলস
# =========================
def fibonacci_retrace(high, low, price):
    """ফিবোনাচি রিট্রেসমেন্ট লেভেল চেক"""
    diff = high - low
    if diff == 0:
        return 0
    retrace = (price - low) / diff
    return retrace

# =========================
# ZigZag ডিটেকশন (ইম্প্রুভড)
# =========================
def detect_zigzag(df, pct=2.5):
    """সুইং হাই-লো ডিটেকশন"""
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

# =========================
# মোমেন্টাম চেক
# =========================
def check_momentum(df, start_idx, end_idx):
    """মোমেন্টাম স্ট্রেংথ চেক"""
    if start_idx >= len(df) or end_idx >= len(df):
        return 0
    price_change = abs(df['close'].iloc[end_idx] - df['close'].iloc[start_idx])
    volume_avg = df['volume'].iloc[start_idx:end_idx+1].mean() if 'volume' in df.columns else 0
    return price_change

# =========================
# ডাইভারজেন্স চেক
# =========================
def check_divergence(df, price_swings, rsi_swings):
    """বুলিশ/বিয়ারিশ ডাইভারজেন্স"""
    if len(price_swings) < 2 or len(rsi_swings) < 2:
        return False
    
    price_diff = price_swings[-1] - price_swings[-2]
    rsi_diff = rsi_swings[-1] - rsi_swings[-2]
    
    # বুলিশ ডাইভারজেন্স: প্রাইস নিচে, RSI উপরে
    if price_diff < 0 and rsi_diff > 0:
        return "Bullish"
    # বিয়ারিশ ডাইভারজেন্স: প্রাইস উপরে, RSI নিচে
    elif price_diff > 0 and rsi_diff < 0:
        return "Bearish"
    return False

# =========================
# ইম্পালস ওয়েভ ডিটেকশন
# =========================
def detect_impulse_wave(swings, df, rsi_values):
    """৫-ওয়েভ ইম্পালস প্যাটার্ন ডিটেক্ট"""
    if len(swings) < 5:
        return None, None
    
    types = [s['type'] for s in swings[-5:]]
    prices = [s['price'] for s in swings[-5:]]
    idxs = [s['idx'] for s in swings[-5:]]
    
    # বুলিশ ইম্পালস: up-down-up-down-up
    if types == ['up', 'down', 'up', 'down', 'up']:
        wave1 = abs(prices[1] - prices[0])
        wave2 = abs(prices[2] - prices[1])
        wave3 = abs(prices[3] - prices[2])
        wave4 = abs(prices[4] - prices[3])
        
        # ওয়েভ ২ রিট্রেসমেন্ট চেক (শ্যালো/ডিপ)
        retrace_2 = wave2 / wave1 if wave1 else 0
        
        # ওয়েভ ৩ এক্সটেনশন চেক
        wave3_ext = wave3 > wave1 * 1.618
        
        # ওয়েভ ৪ রিট্রেসমেন্ট চেক
        retrace_4 = wave4 / wave3 if wave3 else 0
        
        # ডাইভারজেন্স চেক
        divergence = check_divergence(df, [prices[-2], prices[-1]], 
                                       [rsi_values[idxs[-2]], rsi_values[idxs[-1]]]) if len(rsi_values) > max(idxs) else False
        
        # সাব-ওয়েভ আইডেন্টিফিকেশন
        sub_wave = "Wave 5 (Bullish Impulse)"
        
        # ওয়েভ ৩ এক্সটেনশন
        if wave3_ext:
            sub_wave = "Wave 3 Extension (Bullish) - Strong Momentum"
        # ট্রাঙ্কেশন
        elif divergence == "Bearish" and wave3 < wave1:
            sub_wave = "Wave 5 Truncation (Bullish) - Divergence Warning"
        
        return "Impulse Wave (Bullish)", sub_wave
    
    # বিয়ারিশ ইম্পালস: down-up-down-up-down
    elif types == ['down', 'up', 'down', 'up', 'down']:
        wave1 = abs(prices[1] - prices[0])
        wave2 = abs(prices[2] - prices[1])
        wave3 = abs(prices[3] - prices[2])
        
        wave3_ext = wave3 > wave1 * 1.618
        
        sub_wave = "Wave 5 (Bearish Impulse)"
        
        if wave3_ext:
            sub_wave = "Wave 3 Extension (Bearish) - Strong Downside"
        
        return "Impulse Wave (Bearish)", sub_wave
    
    return None, None

# =========================
# ডায়াগোনাল প্যাটার্ন ডিটেকশন
# =========================
def detect_diagonal_pattern(swings, df):
    """লিডিং এবং এন্ডিং ডায়াগোনাল ডিটেক্ট"""
    if len(swings) < 5:
        return None, None
    
    types = [s['type'] for s in swings[-5:]]
    prices = [s['price'] for s in swings[-5:]]
    
    # লিডিং ডায়াগোনাল (Wave 1 বা A)
    if types in [['up', 'down', 'up', 'down', 'up'], ['down', 'up', 'down', 'up', 'down']]:
        # চেক করছি ওয়েভগুলি ওভারল্যাপ করে কিনা
        overlaps = False
        for i in range(1, 4):
            if (prices[i] > prices[i-1] and prices[i] > prices[i+1]) or \
               (prices[i] < prices[i-1] and prices[i] < prices[i+1]):
                overlaps = True
        
        if overlaps:
            direction = "Bullish" if types[0] == 'up' else "Bearish"
            return "Leading Diagonal", f"Wave 1 or A ({direction}) - 5-3-5-3-5 Structure"
    
    # এন্ডিং ডায়াগোনাল (Wave 5 বা C)
    if len(swings) >= 5:
        # চেক করছি সংকুচিত হচ্ছে কিনা
        if abs(prices[-1] - prices[0]) < abs(prices[-2] - prices[1]):
            direction = "Bullish" if types[0] == 'up' else "Bearish"
            return "Ending Diagonal", f"Wave 5 or C ({direction}) - Terminal Pattern"
    
    return None, None

# =========================
# জিগজ্যাগ ফ্যামিলি ডিটেকশন
# =========================
def detect_zigzag_family(swings):
    """সিঙ্গেল, ডাবল জিগজ্যাগ ডিটেক্ট"""
    if len(swings) < 3:
        return None, None
    
    types = [s['type'] for s in swings[-3:]]
    prices = [s['price'] for s in swings[-3:]]
    
    # সিঙ্গেল জিগজ্যাগ (5-3-5 structure)
    if types == ['down', 'up', 'down'] or types == ['up', 'down', 'up']:
        a = abs(prices[1] - prices[0])
        b = abs(prices[2] - prices[1])
        ratio = b / a if a else 0
        
        if ratio < 0.8:  # শার্প কারেকশন
            direction = "Bearish" if types[0] == 'down' else "Bullish"
            return "Zigzag (Single)", f"5-3-5 Structure - Sharp {direction} Correction"
        
        # ডাবল জিগজ্যাগ চেক (W-X-Y)
        if len(swings) >= 7:
            types7 = [s['type'] for s in swings[-7:]]
            if types7 == ['down', 'up', 'down', 'up', 'down', 'up', 'down'] or \
               types7 == ['up', 'down', 'up', 'down', 'up', 'down', 'up']:
                return "Double Zigzag", "W-X-Y Pattern - Complex Correction"
    
    return None, None

# =========================
# ফ্ল্যাট ফ্যামিলি ডিটেকশন
# =========================
def detect_flat_family(swings):
    """রেগুলার, এক্সপ্যান্ডেড, রানিং ফ্ল্যাট ডিটেক্ট"""
    if len(swings) < 3:
        return None, None
    
    types = [s['type'] for s in swings[-3:]]
    prices = [s['price'] for s in swings[-3:]]
    
    # ফ্ল্যাট প্যাটার্ন (3-3-5 structure)
    if types == ['down', 'up', 'down'] or types == ['up', 'down', 'up']:
        a = abs(prices[1] - prices[0])
        b = abs(prices[2] - prices[1])
        ratio = b / a if a else 0
        
        # রেগুলার ফ্ল্যাট: B ≈ A, C ≈ A
        if 0.9 <= ratio <= 1.1:
            direction = "Bearish" if types[0] == 'down' else "Bullish"
            return "Regular Flat", f"3-3-5 Structure - B={ratio:.2f}A, Sideways Correction"
        
        # এক্সপ্যান্ডেড ফ্ল্যাট: B > A, C > B
        elif ratio > 1.1 and len(swings) >= 4:
            c = abs(prices[2] - prices[1]) if len(prices) > 2 else 0
            if c > b:
                return "Expanded Flat", "B > A, C > B - Extended Correction"
        
        # রানিং ফ্ল্যাট: B > A, C < A
        elif ratio > 1.1 and len(swings) >= 4:
            c = abs(prices[2] - prices[1]) if len(prices) > 2 else 0
            if c < a:
                return "Running Flat", "B > A, C < A - Strong Trend Continuation"
    
    return None, None

# =========================
# ট্রায়াঙ্গেল ফ্যামিলি ডিটেকশন
# =========================
def detect_triangle_pattern(swings):
    """কন্ট্রাক্টিং এবং এক্সপ্যান্ডিং ট্রায়াঙ্গেল ডিটেক্ট"""
    if len(swings) < 5:
        return None, None
    
    types = [s['type'] for s in swings[-5:]]
    prices = [s['price'] for s in swings[-5:]]
    
    # ট্রায়াঙ্গেলের জন্য ৩-৩-৩-৩-৩ স্ট্রাকচার
    # অল্টারনেটিং টাইপ চেক
    alternating = True
    for i in range(1, len(types)):
        if types[i] == types[i-1]:
            alternating = False
            break
    
    if alternating and len(types) >= 5:
        # কন্ট্রাক্টিং ট্রায়াঙ্গেল (সংকুচিত)
        ranges = []
        for i in range(0, len(prices)-1, 2):
            if i+1 < len(prices):
                ranges.append(abs(prices[i+1] - prices[i]))
        
        if len(ranges) >= 2 and ranges[-1] < ranges[0]:
            direction = "Bullish" if types[0] == 'up' else "Bearish"
            return "Contracting Triangle", f"3-3-3-3-3 Structure - Converging Range, {direction} Breakout Expected"
        
        # এক্সপ্যান্ডিং ট্রায়াঙ্গেল (প্রসারিত)
        elif len(ranges) >= 2 and ranges[-1] > ranges[0]:
            return "Expanding Triangle", "3-3-3-3-3 Structure - Expanding Range, Volatility Increasing"
    
    return None, None

# =========================
# ওয়েভ এক্সটেনশন ডিটেকশন
# =========================
def detect_wave_extension(swings):
    """ওয়েভ এক্সটেনশন ভেরিয়েশন ডিটেক্ট"""
    if len(swings) < 3:
        return None
    
    types = [s['type'] for s in swings[-3:]]
    prices = [s['price'] for s in swings[-3:]]
    
    if len(prices) >= 3:
        wave1 = abs(prices[1] - prices[0])
        wave3 = abs(prices[2] - prices[1])
        
        # 3rd Wave Extension
        if wave3 > wave1 * 1.618:
            direction = "Bullish" if types[1] == 'up' else "Bearish"
            return f"3rd Wave Extension", f"1.618-2.618 Fibonacci Target - Strong {direction} Momentum"
        
        # 5th Wave Extension (শেষ ৫ ওয়েভের জন্য)
        if len(swings) >= 5:
            prices5 = [s['price'] for s in swings[-5:]]
            wave5 = abs(prices5[-1] - prices5[-2])
            wave3_ext = abs(prices5[-3] - prices5[-4]) if len(prices5) >= 4 else 0
            
            if wave5 > wave3_ext * 1.618:
                return "5th Wave Extension", "Terminal Movement - Exhaustion Likely"
    
    return None

# =========================
# মেইন ওয়েভ ডিটেকশন ফাংশন
# =========================
def detect_elliott_wave(df, symbol):
    """সম্পূর্ণ এলিয়ট ওয়েভ প্যাটার্ন লাইব্রেরি বিশ্লেষণ"""
    
    if len(df) < 30:
        return {
            'SYMBOL': symbol,
            'WAVE': 'No Data',
            'SUB_WAVE': 'Insufficient Data (Need 30+ candles)'
        }
    
    # RSI ক্যালকুলেশন
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    rsi_values = (100 - (100 / (1 + rs))).fillna(50).values
    
    # ZigZag ডিটেকশন
    swings = detect_zigzag(df)
    
    if len(swings) < 3:
        return {
            'SYMBOL': symbol,
            'WAVE': 'Sideways / Range Bound',
            'SUB_WAVE': 'No Clear Elliott Pattern - Consolidation Phase'
        }
    
    wave = "Unknown"
    sub_wave = "Analyzing Pattern..."
    
    # 1. ইম্পালস ওয়েভ ডিটেকশন
    impulse_wave, impulse_sub = detect_impulse_wave(swings, df, rsi_values)
    if impulse_wave:
        wave = impulse_wave
        sub_wave = impulse_sub
    
    # 2. ডায়াগোনাল প্যাটার্ন ডিটেকশন
    diagonal_wave, diagonal_sub = detect_diagonal_pattern(swings, df)
    if diagonal_wave and wave == "Unknown":
        wave = diagonal_wave
        sub_wave = diagonal_sub
    
    # 3. জিগজ্যাগ ফ্যামিলি ডিটেকশন
    zigzag_wave, zigzag_sub = detect_zigzag_family(swings)
    if zigzag_wave and wave == "Unknown":
        wave = zigzag_wave
        sub_wave = zigzag_sub
    
    # 4. ফ্ল্যাট ফ্যামিলি ডিটেকশন
    flat_wave, flat_sub = detect_flat_family(swings)
    if flat_wave and wave == "Unknown":
        wave = flat_wave
        sub_wave = flat_sub
    
    # 5. ট্রায়াঙ্গেল ফ্যামিলি ডিটেকশন
    triangle_wave, triangle_sub = detect_triangle_pattern(swings)
    if triangle_wave and wave == "Unknown":
        wave = triangle_wave
        sub_wave = triangle_sub
    
    # 6. ওয়েভ এক্সটেনশন ডিটেকশন (ইমপালসের ভেতরে)
    extension, extension_sub = detect_wave_extension(swings), None
    if extension and "Extension" not in sub_wave:
        if wave != "Unknown":
            sub_wave = f"{sub_wave} | {extension[0]}: {extension[1]}" if isinstance(extension, tuple) else f"{sub_wave} | {extension}"
    
    # যদি কোনো প্যাটার্ন না পাওয়া যায়
    if wave == "Unknown":
        if len(swings) >= 3:
            types = [s['type'] for s in swings[-3:]]
            if types in [['down', 'up', 'down'], ['up', 'down', 'up']]:
                wave = "Corrective Wave"
                sub_wave = "Minor Correction - Incomplete Pattern"
            else:
                wave = "Complex Correction"
                sub_wave = "Mixed Pattern - Awaiting Confirmation"
        else:
            wave = "Undefined"
            sub_wave = "Insufficient Swing Points"
    
    return {
        'SYMBOL': symbol,
        'WAVE': wave,
        'SUB_WAVE': sub_wave
    }

# =========================
# মেইন ফাংশন
# =========================
def main():
    #print("=" * 80)
    #print("এলিয়ট ওয়েভ সম্পূর্ণ প্যাটার্ন লাইব্রেরি অ্যানালাইসিস")
    #print("=" * 80)
    
    df = pd.read_csv(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])
    
    # ভলিউম কলাম চেক
    if 'volume' not in df.columns:
        df['volume'] = 1000000  # ডামি ভলিউম
    
    results = []
    symbols = df['symbol'].unique()
    
    for i, symbol in enumerate(symbols, 1):
        s_df = df[df['symbol'] == symbol].sort_values('date')
        res = detect_elliott_wave(s_df, symbol)
        results.append(res)
        #print(f"[{i}/{len(symbols)}] প্রসেসিং: {symbol} -> {res['WAVE']}")
    
    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_FILE, index=False)
    
    #print("\n" + "=" * 80)
    #print(f"✅ কাজ সম্পন্ন হয়েছে: {OUTPUT_FILE}")
    #print(f"📊 মোট {len(out)}টি সিম্বল প্রসেস করা হয়েছে")
    #print("=" * 80)
    
    # প্যাটার্ন ডিস্ট্রিবিউশন দেখানো
    #print("\n📈 প্যাটার্ন ডিস্ট্রিবিউশন:")
    pattern_counts = out['WAVE'].value_counts()
    for pattern, count in pattern_counts.items():
        print(f"   {pattern}: {count} ({count/len(out)*100:.1f}%)")

if __name__ == "__main__":
    main()
