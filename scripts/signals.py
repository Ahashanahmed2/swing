import os
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from env import TradeEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# ---------- 📦 Load Data ----------
def load_csv_safe(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.fillna(0, inplace=True)
        print(f"✅ Loaded: {path} ({len(df)} rows)")
        return df
    else:
        print(f"❌ File not found: {path}")
        return pd.DataFrame()

def load_all_csv_from_folder(folder):
    dfs = []
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(folder, f))
                df.fillna(0, inplace=True)
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

main_df = pd.read_csv('./csv/mongodb.csv')
imbalance_high = load_all_csv_from_folder('./csv/swing/imbalanceZone/down_to_up')
imbalance_low = load_all_csv_from_folder('./csv/swing/imbalanceZone/up_to_down')
swing_high_candle = load_all_csv_from_folder('./csv/swing/swing_high/high_candle')
swing_high_confirm = load_all_csv_from_folder('./csv/swing/swing_high/high_confirm')
swing_low_candle = load_all_csv_from_folder('./csv/swing/swing_low/low_candle')
swing_low_confirm = load_all_csv_from_folder('./csv/swing/swing_low/low_confirm')
down_to_up =pd.read_csv("./csv/swing/down_to_up.csv")
up_to_down = pd.read_csv("./csv/swing/up_to_down.csv")
rsi_divergence = pd.DataFrame()

# ---------- 🧠 Load Model ----------
model_path = "./csv/dqn_retrained.zip"
if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    exit()
model = DQN.load(model_path)

# ---------- 🌍 Create Environment ----------
env = TradeEnv(
    main_df=main_df,
    imbalance_high_df=imbalance_high,
    imbalance_low_df=imbalance_low,
    swing_high_candle_df=swing_high_candle,
    swing_high_confirm_df=swing_high_confirm,
    swing_low_candle_df=swing_low_candle,
    swing_low_confirm_df=swing_low_confirm,
    rsi_divergence_df=rsi_divergence,
    down_to_up_df=down_to_up,
    up_to_down_df=up_to_down
)
env = DummyVecEnv([lambda: env])
obs = env.reset()

# ---------- 🚀 Run Model Prediction ----------
signals = []
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    info = info[0] if isinstance(info, list) else info

    signals.append({
        'step': len(signals),
        'symbol': info.get('symbol', ''),
        'price': info.get('price'),
        'action': ['Hold', 'Buy', 'Sell'][action[0] if isinstance(action, np.ndarray) else action],
        'cash': info.get('cash'),
        'stock': info.get('stock'),
        'portfolio_value': info.get('portfolio_value')
    })

# ---------- 💾 Save CSV ----------
output_path = "./csv/signals.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
pd.DataFrame(signals).to_csv(output_path, index=False)
print(f"✅ signals.csv saved to: {output_path}")
