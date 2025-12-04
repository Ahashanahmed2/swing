import os
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------
# SAFE Backtest Function
# ---------------------------------------------------------
def backtest_signals(signal_csv_path,
                     main_df_path='./csv/mongodb.csv',
                     result_dir='./csv/backtest_result',
                     ai_signal_dir='./output/ai_signal'):

    # ----------------------------
    # Load CSV safely
    # ----------------------------
    if not os.path.isfile(signal_csv_path):
        print(f"⚠️ Signal file not found: {signal_csv_path} → Skipping.")
        return

    try:
        signals = pd.read_csv(signal_csv_path)
    except Exception as e:
        print(f"⚠️ Failed to read {signal_csv_path}: {e}")
        return

    required_cols = ['symbol', 'entrydate', 'buyprice', 'exittargetprice',
                     'stoploss', 'signaltype', 'profit', 'confidence',
                     'trend', 'ai_score', 'riskrewardratio']

    # Missing column protection
    for col in required_cols:
        if col not in signals.columns:
            signals[col] = None

    signals['entrydate'] = pd.to_datetime(signals['entrydate'], errors='coerce')

    # ----------------------------
    # Load main mongodb.csv
    # ----------------------------
    if not os.path.isfile(main_df_path):
        print(f"❌ mongodb.csv not found → {main_df_path}")
        return

    try:
        main_df = pd.read_csv(main_df_path)
        main_df['date'] = pd.to_datetime(main_df['date'], errors='coerce')
    except Exception as e:
        print(f"⚠️ Failed to read mongodb.csv: {e}")
        return

    results = []
    processed_symbols = set()

    # ---------------------------------------------------------
    # Main backtest loop
    # ---------------------------------------------------------
    for _, row in signals.iterrows():
        symbol = row['symbol']
        entry_date = row['entrydate']

        if pd.isna(symbol) or pd.isna(entry_date):
            continue

        buy_price = row['buyprice']
        exit_price = row['exittargetprice']
        stop_loss = row['stoploss']
        signal_type = row['signaltype']

        df = main_df[(main_df['symbol'] == symbol) &
                     (main_df['date'] > entry_date)].sort_values(by='date')

        outcome = 'HOLD'
        exit_day = None
        duration_days = None

        if not df.empty:
            for _, future_row in df.iterrows():
                close = future_row['close']
                if close >= exit_price:
                    outcome = 'TP'
                    exit_day = future_row['date']
                    break
                elif close <= stop_loss:
                    outcome = 'SL'
                    exit_day = future_row['date']
                    break

        if outcome in ['TP', 'SL']:
            duration_days = (exit_day.date() - entry_date.date()).days
            processed_symbols.add(symbol)

        results.append({
            'symbol': symbol,
            'entry_date': entry_date.date(),
            'exit_date': exit_day.date() if exit_day else None,
            'signal_type': signal_type,
            'outcome': outcome,
            'buy_price': buy_price,
            'exit_price': exit_price,
            'stop_loss': stop_loss,
            'profit': row['profit'],
            'confidence': row['confidence'],
            'trend': row['trend'],
            'ai_score': row['ai_score'],
            'risk_reward_ratio': row['riskrewardratio'],
            'duration_days': duration_days
        })

    # ---------------------------------------------------------
    # Save Results
    # ---------------------------------------------------------
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ai_signal_dir, exist_ok=True)

    date_str = os.path.basename(signal_csv_path).split('.')[0]

    result_path = os.path.join(result_dir, f"{date_str}_backtest.csv")
    ai_signal_path = os.path.join(ai_signal_dir, f"{date_str}_ai_signal.csv")

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values(by='duration_days', na_position='last')

    df_result.to_csv(result_path, index=False)
    df_result.to_csv(ai_signal_path, index=False)

    print(f"✅ Backtest complete for {date_str}")
    print(f"📁 Saved to: {result_path}")
    print(f"📁 Also saved to: {ai_signal_path}")

    # ---------------------------------------------------------
    # Remove processed signals
    # ---------------------------------------------------------
    updated_signals = signals[~signals['symbol'].isin(processed_symbols)]

    if updated_signals.empty:
        os.remove(signal_csv_path)
        print(f"🗑️ Deleted empty signal file: {signal_csv_path}")
    else:
        updated_signals.to_csv(signal_csv_path, index=False)
        print(f"✂️ Updated signal file: {signal_csv_path}")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    summary = df_result['outcome'].value_counts()
    print("\n📈 Backtest Summary:")
    print(summary)

    tp = summary.get('TP', 0)
    sl = summary.get('SL', 0)
    accuracy = round((tp / (tp + sl)) * 100, 2) if (tp + sl) > 0 else 0
    print(f"🎯 Take Profit Accuracy: {accuracy}%")



# ---------------------------------------------------------
# SAFE Batch Backtest Runner
# ---------------------------------------------------------
def run_backtest_on_all_signals(signal_dir='./csv/all_signal',
                                main_df_path='./csv/mongodb.csv',
                                result_dir='./csv/backtest_result',
                                ai_signal_dir='./output/ai_signal'):

    # 🔹 Folder না থাকলে Error নয় → Skip করবে
    if not os.path.isdir(signal_dir):
        print(f"⚠️ Signal folder not found: {signal_dir} → Skipping all backtests.")
        return

    signal_files = sorted([f for f in os.listdir(signal_dir) if f.endswith('.csv')])

    if not signal_files:
        print("⚠️ No signal files found.")
        return

    for file in signal_files:
        print(f"\n🔍 Backtesting: {file}")
        signal_path = os.path.join(signal_dir, file)

        try:
            backtest_signals(signal_path, main_df_path, result_dir, ai_signal_dir)
        except Exception as e:
            print(f"❌ ব্যাকটেস্ট ব্যর্থ ({file}): {e}")


# ---------------------------------------------------------
# AUTO EXECUTE
# ---------------------------------------------------------
if __name__ == "__main__":
    run_backtest_on_all_signals()