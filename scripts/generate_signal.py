import os
import pandas as pd
import numpy as np
from datetime import datetime
from env import TradeEnv
from stable_baselines3 import PPO
import json

def generate_signals():
    print("🚀 সিগন্যাল জেনারেশন শুরু হচ্ছে...")

    # 🔧 Load config
    CONFIG_PATH = "./config.json"
    TOTAL_CAPITAL = 500000
    RISK_PERCENT = 0.01
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
            TOTAL_CAPITAL = cfg.get("total_capital", 500000)
            RISK_PERCENT = cfg.get("risk_percent", 0.01)
    print(f"✅ Config: Capital = {TOTAL_CAPITAL:,.0f} BDT, Risk = {RISK_PERCENT*100:.1f}%")

    # 📥 Load critical system files
    accuracy_path = "./csv/accuracy_by_symbol.csv"
    trade_stock_path = "./csv/trade_stock.csv"
    strategy_metrics_path = "./output/ai_signal/strategy_metrics.csv"
    symbol_ref_path = "./output/ai_signal/symbol_reference_metrics.csv"

    accuracy_df = pd.read_csv(accuracy_path) if os.path.exists(accuracy_path) else pd.DataFrame()
    trade_stock_df = pd.read_csv(trade_stock_path) if os.path.exists(trade_stock_path) else pd.DataFrame()
    strategy_metrics = pd.read_csv(strategy_metrics_path) if os.path.exists(strategy_metrics_path) else pd.DataFrame()
    symbol_ref_metrics = pd.read_csv(symbol_ref_path) if os.path.exists(symbol_ref_path) else pd.DataFrame()

    # 🧠 Load Trained PPO Model
    try:
        model = PPO.load("./csv/ppo_retrained.zip")
        print("✅ PPO মডেল সফলভাবে লোড হয়েছে")
    except Exception as e:
        print(f"❌ PPO মডেল লোড করতে ব্যর্থ: {e}")
        return

    # 📊 Load Main Data
    main_path = "./csv/mongodb.csv"
    if not os.path.exists(main_path):
        print("❌ মূল ডেটা ফাইল পাওয়া যায়নি:", main_path)
        return

    main_df = pd.read_csv(main_path)
    required_cols = ['RSI', 'confidence', 'ai_score', 'duration_days', 'close']
    for col in required_cols:
        if col not in main_df.columns:
            main_df[col] = 0

    if 'symbol' not in main_df.columns:
        print("❌ 'symbol' column not found in main_df")
        return

    main_df["date"] = pd.to_datetime(main_df["date"], errors="coerce")
    main_df = main_df.dropna(subset=["date"])

    unique_symbols = main_df["symbol"].dropna().str.upper().unique()
    print(f"🔎 মোট {len(unique_symbols)}টি symbol পাওয়া গেছে")

    # 📂 Feature sets (signals)
    gape_path = "./csv/gape.csv"
    gapebuy_path = "./csv/gape_buy.csv"
    shortbuy_path = "./csv/short_buy.csv"
    rsi_diver_path = "./csv/rsi_diver.csv"
    rsi_diver_retest_path = "./csv/rsi_diver_retest.csv"

    os.makedirs("./csv/all_signal", exist_ok=True)
    output_path = "./csv/all_signals.csv"
    all_signals = []

    for symbol in unique_symbols:
        try:
            symbol_df = main_df[main_df["symbol"].str.upper() == symbol].copy()
            if symbol_df.empty:
                continue

            symbol_df = symbol_df.sort_values("date").tail(60)

            # ✅ Get SYSTEM CONTEXT for this symbol
            system_context = {
                "has_gape_signal": False,
                "has_gapebuy": False,
                "has_shortbuy": False,
                "has_rsi_diver": False,
                "has_rsi_retest": False,
                "position_size": 0,
                "exposure_bdt": 0,
                "actual_risk_bdt": 0,
                "RRR": 0.0,
                "strategy_win_pct": 50.0,
                "symbol_ref_win_pct": 50.0,
                "expectancy_bdt": 0.0
            }

            # --- Signal flags ---
            if os.path.exists(gape_path):
                gape_tmp = pd.read_csv(gape_path)
                system_context["has_gape_signal"] = symbol in gape_tmp["symbol"].str.upper().values
            if os.path.exists(gapebuy_path):
                gapebuy_tmp = pd.read_csv(gapebuy_path)
                system_context["has_gapebuy"] = symbol in gapebuy_tmp["symbol"].str.upper().values
            if os.path.exists(shortbuy_path):
                shortbuy_tmp = pd.read_csv(shortbuy_path)
                system_context["has_shortbuy"] = symbol in shortbuy_tmp["symbol"].str.upper().values
            if os.path.exists(rsi_diver_path):
                rsi_diver_tmp = pd.read_csv(rsi_diver_path)
                system_context["has_rsi_diver"] = symbol in rsi_diver_tmp["symbol"].str.upper().values
            if os.path.exists(rsi_diver_retest_path):
                rsi_retest_tmp = pd.read_csv(rsi_diver_retest_path)
                system_context["has_rsi_retest"] = symbol in rsi_retest_tmp["symbol"].str.upper().values

            # --- YOUR SYSTEM'S POSITION & RISK ---
            if not trade_stock_df.empty:
                open_signals = trade_stock_df[
                    (trade_stock_df["symbol"].str.upper() == symbol) &
                    (pd.to_datetime(trade_stock_df["date"]) <= symbol_df["date"].max())
                ]
                if not open_signals.empty:
                    sig = open_signals.iloc[-1]
                    system_context["position_size"] = int(sig.get("position_size", 0))
                    system_context["exposure_bdt"] = float(sig.get("exposure_bdt", 0))
                    system_context["actual_risk_bdt"] = float(sig.get("actual_risk_bdt", 0))
                    system_context["RRR"] = float(sig.get("RRR", 0))

            # --- Strategy metrics ---
            if not strategy_metrics.empty:
                swing_row = strategy_metrics[strategy_metrics["Reference"] == "SWING"]
                if not swing_row.empty:
                    system_context["strategy_win_pct"] = float(swing_row.iloc[0]["Win%"])

            # --- Symbol × Strategy metrics (POWERGRID + RSI) ---
            if not symbol_ref_metrics.empty:
                sym_ref = symbol_ref_metrics[
                    (symbol_ref_metrics["Symbol"].str.upper() == symbol) &
                    (symbol_ref_metrics["Reference"] == "SWING")
                ]
                if not sym_ref.empty:
                    system_context["symbol_ref_win_pct"] = float(sym_ref.iloc[0]["Win%"])
                    system_context["expectancy_bdt"] = float(sym_ref.iloc[0]["Expectancy (BDT)"])

            # ✅ Create environment with SYSTEM CONTEXT
            env = TradeEnv(
                maindf=symbol_df,
                gape_path=gape_path,
                gapebuy_path=gapebuy_path,
                shortbuy_path=shortbuy_path,
                rsi_diver_path=rsi_diver_path,
                rsi_diver_retest_path=rsi_diver_retest_path,
                trade_stock_path=trade_stock_path,
                metrics_path=strategy_metrics_path,
                symbol_ref_path=symbol_ref_path,
                config_path=CONFIG_PATH
            )

            obs, _ = env.reset()
            terminated = truncated = False
            last_reward = 0.0
            last_action = 0
            step_rewards = []

            # Run episode
            while not (terminated or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                step_rewards.append(reward)
                last_reward = reward
                last_action = int(action) if isinstance(action, (int, np.integer)) else int(action.item())

            # Get final price
            if symbol_df.empty:
                continue
            price = float(symbol_df["close"].iloc[-1])

            # ✅ ENHANCED SIGNAL LOGIC — USE YOUR SYSTEM'S METRICS
            rrr = system_context["RRR"]
            expectancy = system_context["expectancy_bdt"]
            win_pct = system_context["symbol_ref_win_pct"]
            position_size = system_context["position_size"]
            risk_bdt = system_context["actual_risk_bdt"]

            # Adjust TP/SL based on RRR & expectancy
            if rrr > 0 and risk_bdt > 0:
                tp = price + (price - (price - risk_bdt / position_size)) * rrr if position_size > 0 else price * 1.05
                sl = price - risk_bdt / position_size if position_size > 0 else price * 0.97
            else:
                tp = price * 1.05
                sl = price * 0.97

            profit = round(tp - price, 2)
            risk_amt = round(price - sl, 2)
            calc_rrr = round(profit / risk_amt, 2) if risk_amt > 0 else 0.0

            # ✅ Confidence = hybrid of AI reward + system expectancy
            ai_confidence = min(100, max(0, int((np.mean(step_rewards) + 10) / 20 * 100)))
            sys_confidence = min(95, win_pct + 0.1 * expectancy)  # e.g., 65% Win + 200 BDT → 85%
            final_confidence = int(0.4 * ai_confidence + 0.6 * sys_confidence)

            # Get accuracy score
            row_match = accuracy_df[accuracy_df['symbol'].str.upper() == symbol]
            ai_score = float(row_match['accuracy (%)'].iloc[0]) if not row_match.empty else 0.0
            ai_action = row_match['ai_action'].iloc[0] if not row_match.empty and 'ai_action' in row_match.columns else ['Hold', 'Buy', 'Sell'][last_action]

            # 🔍 Filter: Only high-potential signals
            if (
                ai_score < 60 or 
                final_confidence < 65 or 
                expectancy < 50 or 
                win_pct < 55 or
                calc_rrr < 1.5
            ):
                continue

            signal = {
                "symbol": symbol,
                "entry_date": str(symbol_df['date'].max().date()),
                "buy_price": round(price, 2),
                "exit_target_price": round(tp, 2),
                "stop_loss": round(sl, 2),
                "profit": profit,
                "risk": risk_amt,
                "risk_reward_ratio": calc_rrr,
                "position_size": position_size,
                "exposure_bdt": round(system_context["exposure_bdt"], 0),
                "actual_risk_bdt": round(risk_bdt, 0),
                "confidence": f"{final_confidence}%",
                "trend": "uptrend" if last_reward > 0 else "downtrend",
                "signal_type": ai_action,
                "ai_score": round(ai_score, 1),
                "win_percent": round(win_pct, 1),
                "expectancy_bdt": round(expectancy, 1),
                "rrr_system": round(rrr, 2),
                "ppo_reward_avg": round(np.mean(step_rewards), 2)
            }

            print(f"✅ {symbol:10} | {signal['signal_type']:4} | Pos: {position_size:4} | Exp: {signal['exposure_bdt']:7,.0f} | Conf: {final_confidence:3}% | Exp(BDT): {expectancy:5.0f} | RRR: {calc_rrr:3.1f}")
            all_signals.append(signal)

        except Exception as e:
            print(f"❌ {symbol} প্রসেস করতে ব্যর্থ: {e}")

    # 🔚 Save results
    if all_signals:
        df = pd.DataFrame(all_signals)
        
        # ✅ Sort by expectancy (BDT) → highest edge first
        df = df.sort_values(["expectancy_bdt", "win_percent", "rrr_system"], ascending=[False, False, False]).reset_index(drop=True)
        df.insert(0, 'no', range(1, len(df)+1))

        # Final column order
        cols = [
            'no', 'symbol', 'entry_date', 'signal_type', 'buy_price', 'stop_loss', 'exit_target_price',
            'profit', 'risk', 'risk_reward_ratio',
            'position_size', 'exposure_bdt', 'actual_risk_bdt',
            'confidence', 'ai_score', 'win_percent', 'expectancy_bdt', 'rrr_system', 'ppo_reward_avg'
        ]
        df = df.reindex(columns=cols)

        df.to_csv(output_path, index=False)
        print(f"\n✅ মোট {len(all_signals)}টি শক্তিশালী সিগন্যাল সেভ হয়েছে: {output_path}")

        entry_date = datetime.now().strftime("%Y-%m-%d")
        dated_output = f"./csv/all_signal/{entry_date}.csv"
        df.to_csv(dated_output, index=False)
        print(f"📅 অতিরিক্ত সেভ হয়েছে: {dated_output}")

        # 📊 Summary
        print("\n" + "="*80)
        print("📊 SIGNAL SUMMARY")
        print("="*80)
        print(f"📈 Avg Win%       : {df['win_percent'].mean():.1f}%")
        print(f"💰 Avg Expectancy : {df['expectancy_bdt'].mean():.1f} BDT/trade")
        print(f"⚖️  Avg RRR        : {df['risk_reward_ratio'].mean():.2f}")
        print(f"🧮 Avg Position    : {df['position_size'].mean():.0f} shares")
        print(f"🎯 Top Symbol      : {df.iloc[0]['symbol']} ({df.iloc[0]['expectancy_bdt']} BDT expectancy)")
        print("="*80)

    else:
        print("⚠️ কোনো শক্তিশালী সিগন্যাল পাওয়া যায়নি।")

# 🔁 Run
if __name__ == "__main__":
    generate_signals()