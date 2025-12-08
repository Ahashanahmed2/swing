# ---------------------------------------------------------
# SAVE BUY SIGNALS TO TWO LOCATIONS (with tp, RRR, diff & sorted)
# ---------------------------------------------------------

if buy_rows:
    # Initial DataFrame: date, symbol, close, SL
    buy_df = pd.DataFrame(buy_rows, columns=['date', 'symbol', 'close', 'SL'])
    buy_df['date'] = pd.to_datetime(buy_df['date'])

    # 🔑 NEW: Add 'tp' and 'RRR'
    tp_list = []
    rrr_list = []
    diff_list = []

    for _, row in buy_df.iterrows():
        symbol = row['symbol']
        buy_date = row['date']
        buy_price = row['close']
        SL_price = row['SL']

        # Get symbol data up to buy_date (inclusive)
        sym_data = mongodb[mongodb['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        buy_idx = sym_data[sym_data['date'] == buy_date].index
        if len(buy_idx) == 0:
            tp_list.append(None)
            rrr_list.append(None)
            diff_list.append(buy_price - SL_price)
            continue
        buy_idx = buy_idx[0]

        # 🔑 Find SL source: base_low came from rsi_30 row → date = base_date
        # We stored base_date in rsi30 (original), so find matching row
        base_row = rsi30[rsi30['symbol'] == symbol]
        if base_row.empty:
            tp_list.append(None)
            rrr_list.append(None)
            diff_list.append(buy_price - SL_price)
            continue
        base_date = base_row.iloc[0]['date']

        # Find index of base_date (SL source candle)
        try:
            sl_idx = sym_data[sym_data['date'] == base_date].index[0]
        except IndexError:
            # Fallback: nearest
            sl_idx = (abs(sym_data['date'] - base_date)).idxmin()

        # 🔍 Search BACKWARD from sl_idx for tp condition
        tp = None
        for i in range(sl_idx - 1, 1, -1):
            try:
                sb = sym_data.iloc[i - 2]
                sa = sym_data.iloc[i - 1]
                s  = sym_data.iloc[i]
            except IndexError:
                break

            if not (sb['date'] < sa['date'] < s['date'] < base_date):
                continue

            if (s['high'] > sa['high']) and (sa['high'] >= sb['high']):
                tp = s['high']
                break

        # Compute RRR only if valid tp
        diff = buy_price - SL_price
        diff_list.append(diff)

        if tp is not None and buy_price > SL_price and tp > buy_price:
            rrr = (tp - buy_price) / diff
            tp_list.append(tp)
            rrr_list.append(round(rrr, 2))
        else:
            tp_list.append(None)
            rrr_list.append(None)

    # Add columns
    buy_df['tp'] = tp_list
    buy_df['diff'] = [round(d, 4) for d in diff_list]
    buy_df['RRR'] = rrr_list

    # ✅ Filter: keep only valid RRR > 0
    buy_df = buy_df.dropna(subset=['tp', 'RRR']).reset_index(drop=True)
    buy_df = buy_df[buy_df['RRR'] > 0].reset_index(drop=True)

    if len(buy_df) > 0:
        # Sort: highest RRR first → then smallest diff
        buy_df = buy_df.sort_values(['RRR', 'diff'], ascending=[False, True]).reset_index(drop=True)
        buy_df.insert(0, 'No', range(1, len(buy_df) + 1))

        # Final column order & formatting
        buy_df['date'] = buy_df['date'].dt.strftime("%Y-%m-%d")
        buy_df['close'] = buy_df['close'].round(4)
        buy_df['SL'] = buy_df['SL'].round(4)
        buy_df['tp'] = buy_df['tp'].round(4)
        buy_df = buy_df[['No', 'date', 'symbol', 'close', 'SL', 'tp', 'diff', 'RRR']]
    else:
        buy_df = pd.DataFrame(columns=['No', 'date', 'symbol', 'close', 'SL', 'tp', 'diff', 'RRR'])

else:
    buy_df = pd.DataFrame(columns=['No', 'date', 'symbol', 'close', 'SL', 'tp', 'diff', 'RRR'])

# -------------------------------
# DELETE OLD ./csv/rsi_30_buy.csv
# -------------------------------
for f in [BUY_PATH_CSV, BUY_PATH_OUTPUT]:
    if os.path.exists(f):
        os.remove(f)

# Save
buy_df.to_csv(BUY_PATH_OUTPUT, index=False)
buy_df.to_csv(BUY_PATH_CSV, index=False)

if len(buy_df) > 0:
    log(f"✅ {len(buy_df)} BUY signals generated (with tp & RRR).")
    log(f"📈 Max RRR: {buy_df['RRR'].max():.2f} | Avg RRR: {buy_df['RRR'].mean():.2f}")
else:
    log("ℹ No valid BUY signals (tp/RRR missing or invalid).")