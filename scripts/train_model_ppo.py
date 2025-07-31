import os
import pandas as pd
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import TradeEnv  # নিশ্চিত হও যে env.py তে TradeEnv আছে

# ⚠️ Warning অফ
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Helper Functions ---
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

# --- Load Datasets ---
main_df = load_csv_safe('./csv/mongodb.csv')
filtered_df = load_csv_safe('./csv/swing/filtered_low_rsi_candles.csv')
imbalance_high = load_all_csv_from_folder('./csv/swing/imbalanceZone/down_to_up')
imbalance_low = load_all_csv_from_folder('./csv/swing/imbalanceZone/up_to_down')
swing_high_candle = load_all_csv_from_folder('./csv/swing/swing_high/high_candle')
swing_high_confirm = load_all_csv_from_folder('./csv/swing/swing_high/high_confirm')
swing_low_candle = load_all_csv_from_folder('./csv/swing/swing_low/low_candle')
swing_low_confirm = load_all_csv_from_folder('./csv/swing/swing_low/low_confirm')
rsi_divergences = load_csv_safe("./csv/swing/rsi_divergences/rsi_divergences.csv")
filtered_output_path = './csv/filtered_output.csv'
filtered_output = pd.read_csv(filtered_output_path) if os.path.exists(filtered_output_path) and not pd.read_csv(filtered_output_path).empty else pd.DataFrame()
down_to_up = load_csv_safe("./csv/swing/down_to_up.csv")
up_to_down = load_csv_safe("./csv/swing/up_to_down.csv")

# --- Create Environment ---
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
        rsi_divergences,
        filtered_output,
        down_to_up,
        up_to_down
    )
    print("✅ Environment initialized")
except Exception as e:
    print(f"❌ Failed to initialize environment: {e}")
    exit()

# --- Wrap Environment ---
env = DummyVecEnv([lambda: env])

# --- Create PPO Model ---
try:
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=10,
        device='cpu'
        #tensorboard_log="./ppo_tensorboard/"
    )
    print("✅ PPO model created")

except Exception as e:
    print(f"❌ Failed to create PPO model: {e}")
    exit()

# --- Train PPO Model ---
try:
    print("🚀 Training the PPO model...")
    model.learn(total_timesteps=200_000)
    print("✅ PPO Training complete")

except Exception as e:
    print(f"❌ Training failed: {e}")
    exit()

# --- Save PPO Model ---
try:
    save_path = './csv/ppo_retrained'
    model.save(save_path)
    print(f"✅ PPO model saved at: {save_path}")
except Exception as e:
    print(f"❌ Failed to save PPO model: {e}")
