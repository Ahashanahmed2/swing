def identifyswingpoints(df, usersi=False, rsithresholdlow=30, rsithreshold_high=70):
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
            for highidx, lowidx, closecompareidx in condition_pairs:
                if low_idx >= len(df):
                    break
                if (
                    df.iloc[highidx]['high'] < df.iloc[lowidx]['low'] and
                    df.iloc[lowidx]['close'] > df.iloc[closecompare_idx]['high']
                ):
                    if not usersi or df['rsi'].iloc[i] < rsithreshold_low:
                        swinglows.append((i, lowidx))
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
            for lowidx, highidx, closecompareidx in condition_pairs:
                if high_idx >= len(df):
                    break
                if (
                    df.iloc[lowidx]['low'] > df.iloc[highidx]['high'] and
                    df.iloc[highidx]['close'] < df.iloc[closecompare_idx]['low']
                ):
                    if not usersi or df['rsi'].iloc[i] > rsithreshold_high:
                        swinghighs.append((i, highidx))
                    break

    return swinglows, swinghighs.
