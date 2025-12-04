import os
import pandas as pd
from datetime import datetime
from stable_baselines3 import DQN
from env import TradeEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# 📂 Paths
main_df_path = './csv/mongodb.csv'
backtest_dir = './csv/backtest_result'
model_path = './csv/dqn_retrained'

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
        bt_df_list.append(df)
    except Exception as e:
        print(f"⚠️ Failed to load backtest file {f}: {e}")

if not bt_df_list:
    print("⚠️ No valid backtest files → retrain aborted.")
    exit()

bt_df = pd.concat(bt_df_list, ignore_index=True)

# ---------------------------------------------------------
# Sort by fastest duration
# ---------------------------------------------------------
bt_df = bt_df.sort_values(by='duration_days', na_position='last')

# ---------------------------------------------------------
# TP / SL Symbol Stats
# ---------------------------------------------------------
tp_symbols = bt_df[bt_df['outcome'] == 'TP']['symbol'].value_counts()
sl_symbols = bt_df[bt_df['outcome'] == 'SL']['symbol'].value_counts()

# ❌ Blacklist (bad symbols)
blacklist = sl_symbols[sl_symbols >= 3].index.tolist()

# ✅ Whitelist (high performing)
whitelist = tp_symbols[tp_symbols >= 2].index.tolist()

# ---------------------------------------------------------
# Filter main_df using whitelist/blacklist
# ---------------------------------------------------------
filtered_df = main_df[
    main_df['symbol'].isin(whitelist) &
    ~main_df['symbol'].isin(blacklist)
]

# Keep only fast TP symbols (TP ≤ 3 days)
fast_tp = bt_df[(bt_df['outcome'] == 'TP') & (bt_df['duration_days'] <= 3)]
fast_tp_symbols = fast_tp['symbol'].unique().tolist()
filtered_df = filtered_df[filtered_df['symbol'].isin(fast_tp_symbols)]

# ---------------------------------------------------------
# Empty dataset check
# ---------------------------------------------------------
if filtered_df.empty:
    print("⚠️ Filtered training data is empty → Retrain skipped.")
    exit()

# ---------------------------------------------------------
# Prepare RL Environment
# ---------------------------------------------------------
env = TradeEnv(
    maindf=filtered_df,
    gape_path="./csv/gape.csv",
    gapebuy_path="./csv/gape_buy.csv",
    shortbuy_path="./csv/short_buy.csv",
    rsi_diver_path="./csv/rsi_diver.csv",
    rsi_diver_retest_path="./csv/rsi_diver_retest.csv"
)
env = DummyVecEnv([lambda: env])

# ---------------------------------------------------------
# Load or create model
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 🚀 Train the Model
# ---------------------------------------------------------
print("🚀 Retraining model with auto-tuned data...")
model.learn(total_timesteps=200_000)

model.save(model_path)
print(f"✅ Model saved at {model_path}")