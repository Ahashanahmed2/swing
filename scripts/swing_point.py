def identify_swing_points(df, use_rsi=False, rsi_threshold_low=30, rsi_threshold_high=70):
    from ta.momentum import RSIIndicator

    if use_rsi:
        rsi_indicator = RSIIndicator(close=df['close'], window=14, fillna=True)
        df['rsi'] = rsi_indicator.rsi()

    swing_lows = []
    swing_highs = []

    for i in range(1, len(df) - 7):
        current = df.iloc[i]
        prev = df.iloc[i - 1]
        next1 = df.iloc[i + 1]

        # === ✅ Swing Low Logic ===
        if current['low'] < prev['low'] and current['low'] < next1['low']:
            condition_pairs = [
                (i,     i + 2, i + 1),
                (i + 1, i + 3, i + 2),
                (i + 2, i + 4, i + 3),
                (i + 3, i + 5, i + 4),
                (i + 4, i + 6, i + 5),
            ]
            for high_idx, low_idx, close_compare_idx in condition_pairs:
                if low_idx >= len(df):
                    break
                if (
                    df.iloc[high_idx]['high'] < df.iloc[low_idx]['low'] and
                    df.iloc[low_idx]['close'] > df.iloc[close_compare_idx]['high']
                ):
                    if not use_rsi or df['rsi'].iloc[i] < rsi_threshold_low:
                        swing_lows.append((i, low_idx))
                    break

        # === 🔼 Swing High Logic ===
        if current['high'] > prev['high'] and current['high'] > next1['high']:
            condition_pairs = [
                (i,     i + 2, i + 1),
                (i + 1, i + 3, i + 2),
                (i + 2, i + 4, i + 3),
                (i + 3, i + 5, i + 4),
                (i + 4, i + 6, i + 5),
            ]
            for low_idx, high_idx, close_compare_idx in condition_pairs:
                if high_idx >= len(df):
                    break
                if (
                    df.iloc[low_idx]['low'] > df.iloc[high_idx]['high'] and
                    df.iloc[high_idx]['close'] < df.iloc[close_compare_idx]['low']
                ):
                    if not use_rsi or df['rsi'].iloc[i] > rsi_threshold_high:
                        swing_highs.append((i, high_idx))
                    break

    return swing_lows, swing_highs
