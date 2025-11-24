import pandas as pd
import os
from datetime import datetime
from env import TradeEnv
from stable_baselines3 import DQN
import numpy as np

def generate_signals():
    # 📥 Load Accuracy Report
    accuracy_df = pd.read_csv("./csv/accuracy_by_symbol.csv")

    # 🧠 Load Trained Model
    try:
        model = DQN.load("./csv/dqn_retrained")
        print("✅ মডেল সফলভাবে লোড হয়েছে")
    except Exception as e:
        print(f"❌ মডেল লোড করতে ব্যর্থ: {e}")
        return

    # 📊 Load Main Data
    main_df = pd.read_csv("./csv/mongodb.csv")
    if 'symbol' not in main_df.columns:
        print("❌ 'symbol' column not found in main_df")
        return

    unique_symbols = main_df["symbol"].dropna().unique()
    print(f"🔎 মোট {len(unique_symbols)}টি symbol পাওয়া গেছে")

    # 📂 Load Feature Sets
    gape_path = "./csv/gape.csv"
    gapebuy_path = "./csv/gape_buy.csv"
    shortbuy_path = "./csv/short_buy.csv"
    rsi_diver_path = "./csv/rsi_diver.csv"
    rsi_diver_retest_path = "./csv/rsi_diver_retest.csv"

    os.makedirs("./output/ai_signal", exist_ok=True)
    output_path = "./output/ai_signal/all_signals.csv"
    all_signals = []

    for symbol in unique_symbols:
        try:
            symbol_df = main_df[main_df["symbol"] == symbol].copy()
            if symbol_df.empty:
                continue

            symbol_df = symbol_df.sort_values(by="date")
            symbol_df = symbol_df.tail(60)

            env = TradeEnv(
                maindf=symbol_df,
                gape_path=gape_path,
                gapebuy_path=gapebuy_path,
                shortbuy_path=shortbuy_path,
                rsi_diver_path=rsi_diver_path,
                rsi_diver_retest_path=rsi_diver_retest_path
            )

            obs, _ = env.reset()
            terminated = truncated = False
            last_reward = 0.0
            last_action = 0

            while not (terminated or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                last_reward = reward
                last_action = int(action) if isinstance(action, (int, np.integer)) else int(action.item())

            price = info['price']
            profit = round((price * 1.05) - price, 2)
            confidence_score = min(100, max(0, int((last_reward + 10) / 20 * 100)))

            row_match = accuracy_df[accuracy_df['symbol'] == symbol]
            ai_score = float(row_match['accuracy (%)'].iloc[0]) if not row_match.empty else 0.0
            ai_action = row_match['ai_action'].iloc[0] if not row_match.empty and 'ai_action' in row_match.columns else ['Hold', 'Buy', 'Sell'][last_action]

            # 🔍 Filter weak signals
            if ai_score < 60 or confidence_score < 50:
                continue

            signal = {
                "symbol": symbol,
                "entry_date": str(symbol_df['date'].max()),
                "buy_price": round(price, 2),
                "exit_target_price": round(price * 1.05, 2),
                "profit": profit,
                "confidence": f"{confidence_score}%",
                "trend": "uptrend" if last_reward > 0 else "downtrend",
                "signal_type": ai_action,
                "stop_loss": round(price * 0.97, 2),
                "risk_reward_ratio": round(abs(profit / (price * 0.03)), 2),
                "ai_score": ai_score
            }

            print(f"✅ Signal: {symbol} → {signal['signal_type']} ({signal['ai_score']}%)")
            all_signals.append(signal)

        except Exception as e:
            print(f"❌ {symbol} প্রসেস করতে ব্যর্থ: {e}")

    if all_signals:
        pd.DataFrame(all_signals).to_csv(output_path, index=False)
        print(f"✅ মোট {len(all_signals)}টি শক্তিশালী সিগন্যাল সেভ হয়েছে: {output_path}")
    else:
        print("⚠️ কোনো শক্তিশালী সিগন্যাল পাওয়া যায়নি।")

generate_signals()