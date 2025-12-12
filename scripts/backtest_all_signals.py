import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# ---------------------------------------------------------
# 🔧 Load config (for risk context)
# ---------------------------------------------------------
CONFIG_PATH = "./config.json"
TOTAL_CAPITAL = 500000
RISK_PERCENT = 0.01
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
        TOTAL_CAPITAL = cfg.get("total_capital", 500000)
        RISK_PERCENT = cfg.get("risk_percent", 0.01)

# ---------------------------------------------------------
# 📊 Load system intelligence
# ---------------------------------------------------------
strategy_metrics_path = "./output/ai_signal/strategy_metrics.csv"
symbol_ref_path = "./output/ai_signal/symbol_reference_metrics.csv"

strategy_metrics = pd.read_csv(strategy_metrics_path) if os.path.exists(strategy_metrics_path) else pd.DataFrame()
symbol_ref_metrics = pd.read_csv(symbol_ref_path) if os.path.exists(symbol_ref_path) else pd.DataFrame()

# ---------------------------------------------------------
# SAFE Backtest Function — ENHANCED
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
        print(f"📊 Loaded {len(signals)} signals from {os.path.basename(signal_csv_path)}")
    except Exception as e:
        print(f"⚠️ Failed to read {signal_csv_path}: {e}")
        return

    # ✅ Normalize column names (handle both 'entrydate' & 'entry_date')
    col_map = {
        'entrydate': 'entry_date',
        'buyprice': 'buy_price',
        'exittargetprice': 'exit_target_price',
        'stoploss': 'stop_loss',
        'riskrewardratio': 'risk_reward_ratio'
    }
    signals.rename(columns=col_map, inplace=True)

    required_cols = ['symbol', 'entry_date', 'buy_price', 'exit_target_price',
                     'stop_loss', 'signal_type', 'profit', 'confidence',
                     'trend', 'ai_score', 'risk_reward_ratio']

    # Missing column protection
    for col in required_cols:
        if col not in signals.columns:
            signals[col] = None

    signals['entry_date'] = pd.to_datetime(signals['entry_date'], errors='coerce')
    signals['symbol'] = signals['symbol'].str.upper()

    # ----------------------------
    # Load main mongodb.csv
    # ----------------------------
    if not os.path.isfile(main_df_path):
        print(f"❌ mongodb.csv not found → {main_df_path}")
        return

    try:
        main_df = pd.read_csv(main_df_path)
        main_df['date'] = pd.to_datetime(main_df['date'], errors='coerce')
        main_df['symbol'] = main_df['symbol'].str.upper()
    except Exception as e:
        print(f"⚠️ Failed to read mongodb.csv: {e}")
        return

    results = []
    processed_symbols = set()

    # ---------------------------------------------------------
    # ✅ ENHANCED Backtest Loop — with SYSTEM METRICS
    # ---------------------------------------------------------
    for _, row in signals.iterrows():
        symbol = row['symbol']
        entry_date = row['entry_date']

        if pd.isna(symbol) or pd.isna(entry_date):
            continue

        buy_price = row['buy_price']
        exit_price = row['exit_target_price']
        stop_loss = row['stop_loss']
        signal_type = row['signal_type']

        # Get SYSTEM CONTEXT for this symbol
        sys_context = {
            "position_size": 0,
            "exposure_bdt": 0,
            "actual_risk_bdt": 0,
            "rrr_system": 0.0,
            "win_percent": 50.0,
            "expectancy_bdt": 0.0,
            "strategy": "SWING"
        }

        # ✅ Pull from symbol_reference_metrics
        if not symbol_ref_metrics.empty:
            ref_row = symbol_ref_metrics[
                (symbol_ref_metrics['Symbol'].str.upper() == symbol) &
                (symbol_ref_metrics['Reference'] == 'SWING')
            ]
            if not ref_row.empty:
                sys_context["win_percent"] = float(ref_row.iloc[0].get('Win%', 50.0))
                sys_context["expectancy_bdt"] = float(ref_row.iloc[0].get('Expectancy (BDT)', 0.0))
                sys_context["rrr_system"] = float(ref_row.iloc[0].get('RRR', 0.0))
                sys_context["strategy"] = 'SWING'

        # Filter future data
        df = main_df[
            (main_df['symbol'] == symbol) &
            (main_df['date'] > entry_date)
        ].sort_values(by='date')

        # Initialize result
        outcome = 'HOLD'
        exit_day = None
        duration_days = None
        exit_price_actual = None
        profit_actual = 0.0
        risk_actual = 0.0
        rrr_actual = 0.0

        # ✅ Simulate position based on YOUR system
        position_size = int(row.get('position_size', sys_context["position_size"]) or 0)
        if position_size <= 0 and buy_price > 0 and stop_loss < buy_price:
            risk_per_share = buy_price - stop_loss
            if risk_per_share > 0:
                risk_bdt = TOTAL_CAPITAL * RISK_PERCENT
                position_size = int(risk_bdt / risk_per_share)
                position_size = max(1, position_size)

        exposure_bdt = position_size * buy_price
        risk_bdt = position_size * (buy_price - stop_loss)

        # Scan for TP/SL
        if not df.empty:
            for _, future_row in df.iterrows():
                close = future_row['close']
                if close >= exit_price:
                    outcome = 'TP'
                    exit_day = future_row['date']
                    exit_price_actual = exit_price
                    profit_actual = (exit_price - buy_price) * position_size
                    risk_actual = risk_bdt
                    rrr_actual = profit_actual / risk_actual if risk_actual > 0 else 0.0
                    break
                elif close <= stop_loss:
                    outcome = 'SL'
                    exit_day = future_row['date']
                    exit_price_actual = stop_loss
                    profit_actual = (stop_loss - buy_price) * position_size
                    risk_actual = risk_bdt
                    rrr_actual = 0.0
                    break

        if outcome in ['TP', 'SL']:
            duration_days = (exit_day.date() - entry_date.date()).days
            processed_symbols.add(symbol)

        # ✅ Enrich result with SYSTEM METRICS
        results.append({
            'symbol': symbol,
            'entry_date': entry_date.date(),
            'exit_date': exit_day.date() if exit_day else None,
            'signal_type': signal_type,
            'outcome': outcome,
            'buy_price': float(buy_price),
            'exit_price_target': float(exit_price),
            'exit_price_actual': float(exit_price_actual) if exit_price_actual else float(buy_price),
            'stop_loss': float(stop_loss),
            'profit_target': float(row.get('profit', 0)),
            'profit_actual': round(profit_actual, 2),
            'risk_actual': round(risk_actual, 2),
            'rrr_actual': round(rrr_actual, 2),
            'confidence': row.get('confidence', '0%'),
            'trend': row.get('trend', 'neutral'),
            'ai_score': float(row.get('ai_score', 0)),
            'risk_reward_ratio': float(row.get('risk_reward_ratio', 0)),
            'duration_days': duration_days,
            # ✅ SYSTEM INTELLIGENCE
            'position_size': int(position_size),
            'exposure_bdt': round(exposure_bdt, 0),
            'actual_risk_bdt': round(risk_actual, 0),
            'win_percent': float(sys_context["win_percent"]),
            'expectancy_bdt': float(sys_context["expectancy_bdt"]),
            'rrr_system': float(sys_context["rrr_system"]),
            'strategy': sys_context["strategy"]
        })

    # ---------------------------------------------------------
    # Save Results
    # ---------------------------------------------------------
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ai_signal_dir, exist_ok=True)

    date_str = os.path.splitext(os.path.basename(signal_csv_path))[0]

    result_path = os.path.join(result_dir, f"{date_str}_backtest.csv")
    ai_signal_path = os.path.join(ai_signal_dir, "profit-loss.csv")  # ← reuse your system's format

    df_result = pd.DataFrame(results)
    if not df_result.empty:
        df_result = df_result.sort_values(by=['expectancy_bdt', 'win_percent', 'rrr_system'], 
                                          ascending=[False, False, False])

        df_result.to_csv(result_path, index=False)
        print(f"✅ Backtest complete for {date_str} → {len(df_result)} trades")
        print(f"📁 Saved to: {result_path}")

        # 🔁 Append to profit-loss.csv (for your main system)
        if os.path.exists(ai_signal_path):
            existing = pd.read_csv(ai_signal_path)
            combined = pd.concat([existing, df_result], ignore_index=True)
        else:
            combined = df_result.copy()
        combined.to_csv(ai_signal_path, index=False)
        print(f"🔁 Updated: {ai_signal_path}")
    else:
        print("⚠️ No valid trades to save.")

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
    # ✅ ENHANCED Summary — with SYSTEM METRICS
    # ---------------------------------------------------------
    if not df_result.empty:
        tp = len(df_result[df_result['outcome'] == 'TP'])
        sl = len(df_result[df_result['outcome'] == 'SL'])
        hold = len(df_result[df_result['outcome'] == 'HOLD'])
        total = tp + sl

        accuracy = round((tp / total) * 100, 2) if total > 0 else 0
        avg_expectancy = df_result['expectancy_bdt'].mean()
        avg_win_pct = df_result['win_percent'].mean()
        avg_rrr = df_result['rrr_actual'].replace([np.inf, -np.inf], 0).mean()

        print("\n" + "="*60)
        print("📊 BACKTEST SUMMARY (SYSTEM-OPTIMIZED)")
        print("="*60)
        print(f"✅ TP: {tp:2d} | ❌ SL: {sl:2d} | ⏳ HOLD: {hold:2d} | 🎯 Accuracy: {accuracy:5.1f}%")
        print(f"💰 Avg Expectancy  : {avg_expectancy:7.1f} BDT/trade")
        print(f"📈 Avg Win%        : {avg_win_pct:6.1f}%")
        print(f"⚖️  Avg RRR         : {avg_rrr:6.2f}")
        print(f"🧮 Avg Position    : {df_result['position_size'].mean():7.0f} shares")

        # Top 3 performers
        top3 = df_result.nlargest(3, 'expectancy_bdt')
        print("\n🏆 Top 3 High-Expectancy Trades:")
        for _, r in top3.iterrows():
            print(f"   • {r['symbol']:10} | Exp: {r['expectancy_bdt']:5.0f} BDT | Win%: {r['win_percent']:4.0f}% | RRR: {r['rrr_actual']:3.1f}")

        print("="*60)

        # 🔔 Alert if expectancy < 0
        if avg_expectancy < 0:
            print("\n❗ Warning: Negative expectancy — review strategy parameters!")
        elif avg_expectancy > 100:
            print("\n🚀 Excellent! Expectancy > 100 BDT/trade.")


# ---------------------------------------------------------
# SAFE Batch Backtest Runner
# ---------------------------------------------------------
def run_backtest_on_all_signals(signal_dir='./csv/all_signal',
                                main_df_path='./csv/mongodb.csv',
                                result_dir='./csv/backtest_result',
                                ai_signal_dir='./output/ai_signal'):

    if not os.path.isdir(signal_dir):
        print(f"⚠️ Signal folder not found: {signal_dir} → Skipping all backtests.")
        return

    signal_files = sorted([f for f in os.listdir(signal_dir) if f.endswith('.csv')])

    if not signal_files:
        print("⚠️ No signal files found.")
        return

    print(f"🔍 Found {len(signal_files)} signal files to backtest")

    for file in signal_files:
        print(f"\n{'─'*50}")
        signal_path = os.path.join(signal_dir, file)
        backtest_signals(signal_path, main_df_path, result_dir, ai_signal_dir)


# ---------------------------------------------------------
# AUTO EXECUTE
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting SYSTEM-OPTIMIZED Backtest...")
    run_backtest_on_all_signals()
    print("\n🎉 All backtests completed!")