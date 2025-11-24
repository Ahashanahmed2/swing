import os
import pandas as pd
from datetime import datetime
from stable_baselines3 import DQN
from env import TradeEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# 📂 Paths
main_df_path = './csv/mongodb.csv'
backtest_dir = './csv/backtest_result'  # ✅ Updated path
model_path = './csv/dqn_retrained'

# 📊 Load main_df
main_df = pd.read_csv(main_df_path)
main_df['date'] = pd.to_datetime(main_df['date']).dt.date

# 🔍 Step 1: Aggregate backtest results
backtest_files = [f for f in os.listdir(backtest_dir) if f.endswith('.csv')]
bt_df = pd.concat([pd.read_csv(os.path.join(backtest_dir, f)) for f in backtest_files], ignore_index=True)

# ✅ Sort by duration_days (fastest outcomes first)
bt_df = bt_df.sort_values(by='duration_days', na_position='last')

# ✅ Step 2: Filter strong signals
tp_symbols = bt_df[bt_df['outcome'] == 'TP']['symbol'].value_counts()
sl_symbols = bt_df[bt_df['outcome'] == 'SL']['symbol'].value_counts()

# ❌ Blacklist symbols with 3+ SL hits
blacklist = sl_symbols[sl_symbols >= 3].index.tolist()

# ✅ Whitelist symbols with 2+ TP hits
whitelist = tp_symbols[tp_symbols >= 2].index.tolist()

# 🧹 Filter training data
filtered_df = main_df[main_df['symbol'].isin(whitelist) & ~main_df['symbol'].isin(blacklist)]

# ✅ Optional: Keep only fast TP symbols (e.g., TP within 3 days)
fast_tp = bt_df[(bt_df['outcome'] == 'TP') & (bt_df['duration_days'] <= 3)]
fast_tp_symbols = fast_tp['symbol'].unique().tolist()
filtered_df = filtered_df[filtered_df['symbol'].isin(fast_tp_symbols)]

if filtered_df.empty:
    print("⚠️ Filtered training data is empty. Retrain skipped.")
    exit()

# ✅ Prepare environment
env = TradeEnv(
    maindf=filtered_df,
    gape_path="./csv/gape.csv",
    gapebuy_path="./csv/gape_buy.csv",
    shortbuy_path="./csv/short_buy.csv",
    rsi_diver_path="./csv/rsi_diver.csv",
    rsi_diver_retest_path="./csv/rsi_diver_retest.csv"
)
env = DummyVecEnv([lambda: env])

# ✅ Load or create model
if os.path.exists(model_path + ".zip"):
    model = DQN.load(model_path, env=env)
    print("✅ Loaded existing model for fine-tuning")
else:
    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=128,
        tau=0.01,
        gamma=0.95,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        device='cpu'
    )
    print("🆕 Created new model")

# 🧠 Train
print("🚀 Retraining model with auto-tuned data...")
model.learn(total_timesteps=200_000)
model.save(model_path)
print(f"✅ Model saved at {model_path}")
