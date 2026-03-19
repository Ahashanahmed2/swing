import os
import pandas as pd
import numpy as np

def zigzag(high, low, change_threshold=0.05):
    """
    Simple zigzag indicator that marks swing highs and lows based on percentage change.
    Returns list of tuples (index, price, type).
    """
    zigzag_points = []
    trend = None
    last_pivot_idx = 0
    last_pivot_price = low.iloc[0]

    for i in range(1, len(high)-1):
        if trend is None or trend == 'down':
            if low.iloc[i] < last_pivot_price * (1 - change_threshold):
                if low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i+1]:
                    zigzag_points.append((i, low.iloc[i], 'low'))
                    last_pivot_idx = i
                    last_pivot_price = low.iloc[i]
                    trend = 'up'
        if trend is None or trend == 'up':
            if high.iloc[i] > last_pivot_price * (1 + change_threshold):
                if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i+1]:
                    zigzag_points.append((i, high.iloc[i], 'high'))
                    last_pivot_idx = i
                    last_pivot_price = high.iloc[i]
                    trend = 'down'

    # Include first point
    if len(zigzag_points) == 0 or zigzag_points[0][0] != 0:
        first_type = 'low' if low.iloc[0] < high.iloc[0] else 'high'
        zigzag_points.insert(0, (0, low.iloc[0] if first_type=='low' else high.iloc[0], first_type))
    # Include last point
    last_idx = len(high)-1
    if zigzag_points[-1][0] != last_idx:
        last_type = 'low' if low.iloc[last_idx] < high.iloc[last_idx] else 'high'
        zigzag_points.append((last_idx, low.iloc[last_idx] if last_type=='low' else high.iloc[last_idx], last_type))

    return zigzag_points

def detect_impulse_waves(swings, prices, fib_tolerance=0.1):
    """
    Detect 5-wave impulse patterns from a list of swing points.
    Returns list of pattern dictionaries.
    """
    patterns = []
    if len(swings) < 6:
        return patterns

    for j in range(len(swings) - 5):
        pts = swings[j:j+6]
        types = [p[2] for p in pts]
        if not all(types[k] != types[k+1] for k in range(5)):
            continue

        if types[0] == 'low' and types[1] == 'high':  # Bullish
            wave1 = pts[1][1] - pts[0][1]
            wave2 = pts[2][1] - pts[1][1]
            wave3 = pts[3][1] - pts[2][1]
            wave4 = pts[4][1] - pts[3][1]
            wave5 = pts[5][1] - pts[4][1]
            if wave1 <= 0 or wave2 >= 0 or wave3 <= 0 or wave4 >= 0 or wave5 <= 0:
                continue
            pattern_type = 'bullish'
        elif types[0] == 'high' and types[1] == 'low':  # Bearish
            wave1 = pts[1][1] - pts[0][1]
            wave2 = pts[2][1] - pts[1][1]
            wave3 = pts[3][1] - pts[2][1]
            wave4 = pts[4][1] - pts[3][1]
            wave5 = pts[5][1] - pts[4][1]
            if wave1 >= 0 or wave2 <= 0 or wave3 >= 0 or wave4 <= 0 or wave5 >= 0:
                continue
            pattern_type = 'bearish'
        else:
            continue

        # Wave 2 cannot retrace more than 100% of wave 1
        if abs(wave2) >= abs(wave1):
            continue

        # Wave 3 is not the shortest among 1,3,5
        if abs(wave3) < abs(wave1) or abs(wave3) < abs(wave5):
            continue

        # Wave 4 does not overlap wave 1 price territory
        if pattern_type == 'bullish':
            if pts[4][1] <= pts[1][1]:
                continue
        else:  # bearish
            if pts[4][1] >= pts[1][1]:
                continue

        # Fibonacci relationships (simplified)
        fib_382 = 0.382
        fib_618 = 0.618
        fib_500 = 0.5

        retrace2 = abs(wave2) / abs(wave1)
        if not (fib_382 - fib_tolerance <= retrace2 <= fib_618 + fib_tolerance) and not (fib_500 - fib_tolerance <= retrace2 <= fib_500 + fib_tolerance):
            continue

        ratio3 = abs(wave3) / abs(wave1)
        if ratio3 < 1.0 - fib_tolerance:
            continue

        retrace4 = abs(wave4) / abs(wave3)
        if not (fib_382 - fib_tolerance <= retrace4 <= fib_500 + fib_tolerance):
            continue

        pattern = {
            'start_idx': pts[0][0],
            'end_idx': pts[5][0],
            'type': pattern_type,
            'wave_points': [p[1] for p in pts],
            'swing_indices': [p[0] for p in pts]
        }
        patterns.append(pattern)

    return patterns

def add_smc_features(group, pattern):
    """
    Add basic Smart Money Concepts (SMC) features for a given pattern.
    """
    high = group['high']
    low = group['low']
    close = group['close']
    dates = group['date']
    start_idx = pattern['start_idx']
    end_idx = pattern['end_idx']
    
    # Get last swing high and low within the pattern range (excluding the last point? we can take overall)
    # For simplicity, we take the highest high and lowest low in the pattern range
    last_swing_high = high.iloc[start_idx:end_idx+1].max()
    last_swing_low = low.iloc[start_idx:end_idx+1].min()
    
    # Check for Break of Structure (BOS)
    # For bullish: if price breaks above previous swing high
    # For bearish: if price breaks below previous swing low
    # We'll consider the swing points from zigzag
    swings = zigzag(high, low, change_threshold=0.03)  # reuse threshold
    # Filter swings within pattern range
    pattern_swings = [s for s in swings if start_idx <= s[0] <= end_idx]
    bos_occurred = False
    if pattern['type'] == 'bullish':
        # Check if any high after a low is greater than previous high
        for i in range(1, len(pattern_swings)):
            if pattern_swings[i][2] == 'high' and pattern_swings[i][1] > pattern_swings[i-1][1]:
                bos_occurred = True
                break
    else:  # bearish
        for i in range(1, len(pattern_swings)):
            if pattern_swings[i][2] == 'low' and pattern_swings[i][1] < pattern_swings[i-1][1]:
                bos_occurred = True
                break

    # Simple Order Block detection: last two candles before the final swing
    # For a bullish order block, we look at the last down candle before a swing low
    # For simplicity, we take average of high/low of the two candles before the final point
    if end_idx >= 2:
        ob_high = high.iloc[end_idx-2:end_idx].max()
        ob_low = low.iloc[end_idx-2:end_idx].min()
        order_block_price = (ob_high + ob_low) / 2
    else:
        order_block_price = None

    return {
        'last_swing_high': last_swing_high,
        'last_swing_low': last_swing_low,
        'bos_occurred': bos_occurred,
        'order_block_price': order_block_price
    }

def main(csv_path, output_csv='elliott_wave_output.csv'):
    # Load data
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values(['symbol', 'date'])

    all_patterns = []

    for symbol, group in df.groupby('symbol'):
        print(f"\nProcessing {symbol}...")
        group = group.reset_index(drop=True)
        high = group['high']
        low = group['low']
        close = group['close']
        dates = group['date']

        # Get zigzag swing points
        swings = zigzag(high, low, change_threshold=0.03)  # 3% threshold

        # Detect impulse waves
        patterns = detect_impulse_waves(swings, close)

        # Convert patterns to readable format with dates and SMC (excluding price columns)
        for p in patterns:
            # Add SMC features
            smc = add_smc_features(group, p)
            
            record = {
                'symbol': symbol,
                'pattern_type': p['type'],
                'start_date': dates.iloc[p['start_idx']],
                'end_date': dates.iloc[p['end_idx']],
                # SMC columns with simplified names
                'swing_high': smc['last_swing_high'],
                'swing_low': smc['last_swing_low'],
                'bos': smc['bos_occurred'],
                'order_block': smc['order_block_price']
            }
            all_patterns.append(record)

        print(f"Found {len(patterns)} impulse patterns for {symbol}")

    # Save all patterns to CSV
    if all_patterns:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        result_df = pd.DataFrame(all_patterns)
        result_df.to_csv(output_csv, index=False)
        print(f"\n✅ Patterns saved to {output_csv}")
    else:
        print("\nNo Elliott Wave patterns detected.")

    return all_patterns

if __name__ == "__main__":
    csv_file = "./csv/mongodb.csv"  # আপনার ইনপুট ফাইলের পাথ দিন
    output_path = "./output/ai_signal/elliott_wave_output.csv"  # কাঙ্ক্ষিত আউটপুট পাথ
    main(csv_file, output_csv=output_path)