import os
import pandas as pd
import warnings
from stable_baselines3 import DQN
from env import TradeEnv  # ✅ নিশ্চিত করুন env.py তে TradeEnv ক্লাস আছে
#from loss_logging_callback import LossLoggingCallback  # যদি আলাদা ফাইলে থাকে
#callback = LossLoggingCallback()
# ⚠️ FutureWarning বন্ধ করি
warnings.simplefilter(action='ignore', category=FutureWarning)

# 📂 Helper: একক CSV ফাইল লোডার
def load_csv_safe(path):
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✅ Loaded: {path} ({len(df)} rows)")
            return df
        else:
            print(f"⚠️ File not found: {path}")
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
    return pd.DataFrame()

# 📂 Helper: একটি ফোল্ডারের সব CSV একত্রে লোড করা
def load_all_csv_from_folder(folder_path):
    dfs = []
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                full_path = os.path.join(folder_path, file)
                try:
                    df = pd.read_csv(full_path)
                    if not {'symbol', 'date'}.issubset(df.columns):
                        print(f"⚠️ Warning: {file} missing 'symbol' or 'date'")
                    dfs.append(df)
                    print(f"✅ Loaded: {file} ({len(df)} rows)")
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
    else:
        print(f"⚠️ Folder not found: {folder_path}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ✅ Load all required datasets
main_df = pd.read_csv('./csv/mongodb.csv')
filtered_df = pd.read_csv('./csv/swing/filtered_low_rsi_candles.csv')
imbalance_high = load_all_csv_from_folder('./csv/swing/imbalanceZone/down_to_up')
imbalance_low = load_all_csv_from_folder('./csv/swing/imbalanceZone/up_to_down')
swing_high_candle = load_all_csv_from_folder('./csv/swing/swing_high/high_candle')
swing_high_confirm = load_all_csv_from_folder('./csv/swing/swing_high/high_confirm')
swing_low_candle = load_all_csv_from_folder('./csv/swing/swing_low/low_candle')
swing_low_confirm = load_all_csv_from_folder('./csv/swing/swing_low/low_confirm')
down_to_up = pd.read_csv("./csv/swing/down_to_up.csv")
up_to_down = pd.read_csv("./csv/swing/up_to_down.csv")
# rsi_divergence = load_all_csv_from_folder('/home/ahsan/Music/swing/csv/swing/rsi_divergences')  # ঐচ্ছিক

# ✅ Environment তৈরি
try:
    env = TradeEnv(
         main_df,
         filtered_df,
       imbalance_high,
    imbalance_low,
    swing_high_candle,
    swing_high_confirm,
    swing_low_candle,
    swing_low_confirm,
    down_to_up,
    up_to_down

        # rsi_divergence_df=rsi_divergence
    )
    print("✅ Environment initialized")
except Exception as e:
    print(f"❌ Failed to initialize environment: {e}")
    exit()

from stable_baselines3.common.vec_env import DummyVecEnv

# ✅ Environment wrap
env = DummyVecEnv([lambda: env])

# ✅ Model creation
try:
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
        #tensorboard_log="./dqn_tensorboard/"
    )
    print("✅ DQN model created")

except Exception as e:
    print(f"❌ Failed to create DQN model: {e}")
    exit()

# ✅ Training
try:
    print("🚀 Training the DQN model...")
    model.learn(total_timesteps=200_000)
    print("✅ Training complete")

except Exception as e:
    print(f"❌ Training failed: {e}")
    exit()
# 💾 প্রশিক্ষিত মডেল সংরক্ষণ
try:
    save_path = './csv/dqn_retrained'
    model.save(save_path)
    print(f"✅ Model saved at: {save_path}")
except Exception as e:
    print(f"❌ Failed to save model: {e}")
