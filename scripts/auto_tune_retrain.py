import os
import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env import TradeEnv
import json

# 📂 Paths
main_df_path = './csv/mongodb.csv'
backtest_dir = './csv/backtest_result'
model_path = './csv/dqn_retrained'
liquidity_path = './csv/liquidity_system.csv'  # ✅ NEW

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
print(f"✅ Config: Capital = {TOTAL_CAPITAL:,.0f} BDT, Risk = {RISK_PERCENT*100:.1f}%")

# ---------------------------------------------------------
# 📊 Load critical system metrics
# ---------------------------------------------------------
strategy_metrics_path = "./output/ai_signal/strategy_metrics.csv"
symbol_ref_path = "./output/ai_signal/symbol_reference_metrics.csv"

strategy_metrics = pd.read_csv(strategy_metrics_path) if os.path.exists(strategy_metrics_path) else pd.DataFrame()
symbol_ref_metrics = pd.read_csv(symbol_ref_path) if os.path.exists(symbol_ref_path) else pd.DataFrame()

# ---------------------------------------------------------
# SAFE: Ensure folders exist
# ---------------------------------------------------------
os.makedirs(backtest_dir, exist_ok=True)

# ---------------------------------------------------------
# Load MongoDB main_df safely
# ---------------------------------------------------------
if not os.path.isfile(main_df_path):
    print(f"❌ MongoDB CSV missing → {main_df_path}")
    exit()

main_df = pd.read_csv(main_df_path)
main_df['date'] = pd.to_datetime(main_df['date'], errors='coerce').dt.date
main_df['symbol'] = main_df['symbol'].str.upper()

print(f"📊 Loaded {len(main_df)} rows from mongodb.csv")

# ---------------------------------------------------------
# Load backtest files safely
# ---------------------------------------------------------
if not os.path.isdir(backtest_dir):
    print(f"⚠️ Backtest folder missing → {backtest_dir}")
    exit()

backtest_files = [f for f in os.listdir(backtest_dir) if f.endswith('.csv')]

if not backtest_files:
    print("⚠️ No backtest files found → skipping retrain.")
    exit()

# Concatenate all backtests
bt_df_list = []
for f in backtest_files:
    try:
        df = pd.read_csv(os.path.join(backtest_dir, f))
        df['symbol'] = df['symbol'].str.upper()
        bt_df_list.append(df)
    except Exception as e:
        print(f"⚠️ Failed to load backtest file {f}: {e}")

if not bt_df_list:
    print("⚠️ No valid backtest files → retrain aborted.")
    exit()

bt_df = pd.concat(bt_df_list, ignore_index=True)
print(f"📈 Loaded {len(bt_df)} backtest trades from {len(backtest_files)} files")

# ---------------------------------------------------------
# ✅ ENHANCED SYMBOL SELECTION: Use YOUR SYSTEM'S METRICS
# ---------------------------------------------------------
# Base: TP/SL counts
tp_symbols = bt_df[bt_df['outcome'] == 'TP']['symbol'].value_counts()
sl_symbols = bt_df[bt_df['outcome'] == 'SL']['symbol'].value_counts()

# ✅ Get symbols with HIGH EXPECTANCY (BDT) from your system
high_expectancy_symbols = []
if not symbol_ref_metrics.empty:
    good_syms = symbol_ref_metrics[symbol_ref_metrics['Expectancy (BDT)'] > 100]
    high_expectancy_symbols = good_syms['Symbol'].str.upper().unique().tolist()
    print(f"🎯 {len(high_expectancy_symbols)} symbols with >100 BDT expectancy found")

# ✅ Get symbols with HIGH Win% (>65%)
high_win_symbols = []
if not symbol_ref_metrics.empty:
    good_win = symbol_ref_metrics[symbol_ref_metrics['Win%'] > 65]
    high_win_symbols = good_win['Symbol'].str.upper().unique().tolist()
    print(f"✅ {len(high_win_symbols)} symbols with >65% Win% found")

# ✅ Get symbols with MODERATE+ Liquidity (liquidity_score >= 0.4)
high_liquidity_symbols = []
if os.path.exists(liquidity_path):
    try:
        liquidity_df = pd.read_csv(liquidity_path)
        liquidity_df['symbol'] = liquidity_df['symbol'].str.upper()
        good_liq = liquidity_df[liquidity_df['liquidity_score'] >= 0.4]
        high_liquidity_symbols = good_liq['symbol'].unique().tolist()
        print(f"💧 {len(high_liquidity_symbols)} symbols with Moderate+ liquidity (score ≥ 0.4)")
    except Exception as e:
        print(f"⚠️ Failed to load liquidity_system.csv: {e}")

# 🔍 Combine: quality_symbols = Win%>65 ∩ Exp>100 ∩ Liq≥Moderate
quality_symbols = set(high_expectancy_symbols) & set(high_win_symbols)
if high_liquidity_symbols:
    quality_symbols = quality_symbols & set(high_liquidity_symbols)
print(f"🌟 {len(quality_symbols)} high-quality symbols: {list(quality_symbols)[:5]}...")

# ---------------------------------------------------------
# Filter main_df using SYSTEM-INTELLIGENT criteria
# ---------------------------------------------------------
filtered_df = main_df.copy()

# 1. Must be in quality_symbols
if quality_symbols:
    filtered_df = filtered_df[filtered_df['symbol'].isin(quality_symbols)]

# 2. Remove blacklisted (SL ≥ 3)
blacklist = sl_symbols[sl_symbols >= 3].index.tolist()
filtered_df = filtered_df[~filtered_df['symbol'].isin(blacklist)]

# 3. Keep only fast TP symbols (TP ≤ 3 days)
fast_tp = bt_df[(bt_df['outcome'] == 'TP') & (bt_df['duration_days'] <= 3)]
fast_tp_symbols = fast_tp['symbol'].unique().tolist()
if fast_tp_symbols:
    filtered_df = filtered_df[filtered_df['symbol'].isin(fast_tp_symbols)]

# ✅ Add SYSTEM FEATURES to observation
if not filtered_df.empty:
    def add_system_features(row):
        sym = row['symbol']
        win_pct = 50.0
        expectancy = 0.0
        if not symbol_ref_metrics.empty:
            ref_row = symbol_ref_metrics[
                (symbol_ref_metrics['Symbol'].str.upper() == sym) &
                (symbol_ref_metrics['Reference'] == 'SWING')
            ]
            if not ref_row.empty:
                win_pct = float(ref_row.iloc[0]['Win%'])
                expectancy = float(ref_row.iloc[0]['Expectancy (BDT)'])
        return pd.Series([win_pct, expectancy], index=['system_win_pct', 'system_expectancy_bdt'])

    filtered_df[['system_win_pct', 'system_expectancy_bdt']] = filtered_df.apply(add_system_features, axis=1)

print(f"🔍 Final training set: {len(filtered_df)} rows across {filtered_df['symbol'].nunique()} symbols")

# ---------------------------------------------------------
# Empty dataset check
# ---------------------------------------------------------
if filtered_df.empty:
    print("⚠️ Filtered training data is empty → Retrain skipped.")
    exit()

# ---------------------------------------------------------
# Prepare RL Environment — with FULL SYSTEM CONTEXT
# ---------------------------------------------------------
env = TradeEnv(
    maindf=filtered_df,
    gape_path="./csv/gape.csv",
    gapebuy_path="./csv/gape_buy.csv",
    shortbuy_path="./csv/short_buy.csv",
    rsi_diver_path="./csv/rsi_diver.csv",
    rsi_diver_retest_path="./csv/rsi_diver_retest.csv",
    trade_stock_path="./csv/trade_stock.csv",
    metrics_path=strategy_metrics_path,
    symbol_ref_path=symbol_ref_path,
    config_path=CONFIG_PATH,
    liquidity_path=liquidity_path  # ✅ CRITICAL: Enable liquidity awareness
)
env = DummyVecEnv([lambda: env])

# ---------------------------------------------------------
# Load or create model — DQN-OPTIMIZED
# ---------------------------------------------------------
if os.path.exists(model_path + ".zip"):
    model = DQN.load(model_path, env=env, device='cpu')
    print("✅ Loaded existing model for fine-tuning")
else:
    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=2000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=8,
        target_update_interval=500,   # ✅ Optimized for DSE (faster target sync)
        exploration_fraction=0.3,      # ✅ More exploration
        exploration_final_eps=0.02,
        device='cpu'
    )
    print("🆕 Created new model with DSE-Optimized DQN params")

# ---------------------------------------------------------
# 🚀 TRAINING with LIQUIDITY-AWARE LOGGING
# ---------------------------------------------------------
print(f"\n🚀 Retraining DQN with {len(filtered_df)} rows of LIQUIDITY-OPTIMIZED data...")
print("   🔁 200,000 timesteps (≈ 15-20 mins)")

# Custom callback
from stable_baselines3.common.callbacks import BaseCallback

class LogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.steps = 0

    def _on_step(self) -> bool:
        self.steps += 1
        if self.steps % 20000 == 0:
            avg_reward = np.mean(self.model.ep_info_buffer or [0])
            print(f"   📊 Step {self.steps:6d} | Avg Reward: {avg_reward:+.3f}")
        return True

callback = LogCallback()

model.learn(
    total_timesteps=200_000,
    callback=callback,
    log_interval=1000,
    progress_bar=False
)

# ---------------------------------------------------------
# ✅ Save & LIQUIDITY-AWARE Report
# ---------------------------------------------------------
model.save(model_path)
print(f"\n✅ Model saved at {model_path}.zip")

# 📊 Training summary with liquidity stats
if hasattr(model, 'ep_info_buffer') and model.ep_info_buffer:
    rewards = [ep['r'] for ep in model.ep_info_buffer]
    
    # Get liquidity distribution of trained symbols
    trained_syms = filtered_df['symbol'].unique()
    liq_distribution = {}
    if os.path.exists(liquidity_path):
        try:
            liq_df = pd.read_csv(liquidity_path)
            liq_df['symbol'] = liq_df['symbol'].str.upper()
            trained_liq = liq_df[liq_df['symbol'].isin(trained_syms)]
            liq_distribution = trained_liq['liquidity_rating'].value_counts().to_dict()
        except:
            pass

    print("\n" + "="*60)
    print("📊 DQN TRAINING SUMMARY (LIQUIDITY-OPTIMIZED)")
    print("="*60)
    print(f"📈 Avg Episode Reward   : {np.mean(rewards):+.3f}")
    print(f"📉 Min/Max Reward        : {np.min(rewards):+.3f} / {np.max(rewards):+.3f}")
    print(f"🧮 Trained on           : {len(trained_syms)} symbols")
    print(f"🎯 Top symbols          : {list(quality_symbols)[:5]}")
    if liq_distribution:
        print("💧 Liquidity distribution:")
        for liq, cnt in sorted(liq_distribution.items()):
            print(f"   • {liq:<10} : {cnt:2d} symbols")
    print("="*60)

# 🔁 Optional: Auto-generate signals
auto_signal = input("\n🔄 Generate signals with new DQN model? (y/n): ").strip().lower()
if auto_signal == 'y':
    try:
        from generate_signals import generate_signals
        print("\n🔍 Generating signals with new DQN model...\n")
        generate_signals()
    except Exception as e:
        print(f"⚠️ Signal generation failed: {e}")