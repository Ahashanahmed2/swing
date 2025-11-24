import os
import pandas as pd
from datetime import datetime

def backtest_signals(signal_csv_path, main_df_path='./csv/mongodb.csv', output_dir='./output/backtest_results'):
    signals = pd.read_csv(signal_csv_path)
    signals['entrydate'] = pd.to_datetime(signals['entrydate'])

    main_df = pd.read_csv(main_df_path)
    main_df['date'] = pd.to_datetime(main_df['date'])

    results = []

    for _, row in signals.iterrows():
        symbol = row['symbol']
        entry_date = row['entrydate']
        buy_price = row['buyprice']
        exit_price = row['exittargetprice']
        stop_loss = row['stoploss']
        signal_type = row['signaltype']

        df = main_df[(main_df['symbol'] == symbol) & (main_df['date'] > entry_date)].sort_values(by='date')

        outcome = 'HOLD'
        exit_day = None
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
            'risk_reward_ratio': row['riskrewardratio']
        })

    os.makedirs(output_dir, exist_ok=True)
    date_str = os.path.basename(signal_csv_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{date_str}_backtest.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"✅ Backtest complete for {date_str}. Results saved to: {output_path}")

    # Summary
    df_result = pd.DataFrame(results)
    summary = df_result['outcome'].value_counts()
    print("\n📈 Backtest Summary:")
    print(summary)

    tp = summary.get('TP', 0)
    sl = summary.get('SL', 0)
    accuracy = round((tp / (tp + sl)) * 100, 2) if (tp + sl) > 0 else 0
    print(f"🎯 Take Profit Accuracy: {accuracy}%")

def run_backtest_on_all_signals(signal_dir='./csv/all_signal', main_df_path='./csv/mongodb.csv', output_dir='./output/backtest_results'):
    signal_files = sorted([f for f in os.listdir(signal_dir) if f.endswith('.csv')])

    if not signal_files:
        print("⚠️ কোনো সিগন্যাল ফাইল পাওয়া যায়নি")
        return

    for file in signal_files:
        signal_path = os.path.join(signal_dir, file)
        print(f"\n🔍 Backtesting: {file}")
        try:
            backtest_signals(signal_path, main_df_path, output_dir)
        except Exception as e:
            print(f"❌ ব্যাকটেস্ট ব্যর্থ: {file} → {e}")

# ✅ Run all backtests
if __name__ == "__main__":
    run_backtest_on_all_signals()