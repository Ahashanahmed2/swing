# eval_model.py
import os
import pandas as pd
from env import TradeEnv
from stable_baselines3 import DQN

# Load data helpers
def load_all_csv_from_folder(folder_path):
    dfs = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            try:
                dfs.append(pd.read_csv(os.path.join(folder_path, file)))
            except Exception as e:
                print(f"Error reading {file}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# Load all CSVs
main_df = pd.read_csv('./csv/mongodb.csv')
imbalance_high = load_all_csv_from_folder('./csv/swing/imbalanceZone/down_to_up')
imbalance_low = load_all_csv_from_folder('./csv/swing/imbalanceZone/up_to_down')
swing_high_candle = load_all_csv_from_folder('./csv/swing/swing_high/high_candle')
swing_high_confirm = load_all_csv_from_folder('./csv/swing/swing_high/high_confirm')
swing_low_candle = load_all_csv_from_folder('./csv/swing/swing_low/low_candle')
swing_low_confirm = load_all_csv_from_folder('./csv/swing/swing_low/low_confirm')
rsi_divergence = load_all_csv_from_folder('./csv/swing/rsi_divergences')

# Load trained model
print("\U0001F4E5 Loading trained model...")
model = DQN.load("./csv/dqn_retrained")

# Create environment
env = TradeEnv(
    main_df,
    imbalance_high,
    imbalance_low,
    swing_high_candle,
    swing_high_confirm,
    swing_low_candle,
    swing_low_confirm,
    rsi_divergence
)

# Run evaluation
obs, info = env.reset()
terminated = False
truncated = False

while not (terminated or truncated):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Optional: disable if running headless
