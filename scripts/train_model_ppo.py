import os
import pandas as pd
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import TradeEnv  # নিশ্চিত হও যে env.py তে আপডেটেড TradeEnv আছে

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

# --- Load Datasets ---
main_df = load_csv_safe('./csv/mongodb.csv')

# ✅ নতুন CSV ফাইলগুলো লোড করুন
gape_df = load_csv_safe("./csv/gape.csv")
gapebuy_df = load_csv_safe("./csv/gape_buy.csv")
shortbuy_df = load_csv_safe("./csv/short_buy.csv")
rsi_diver_df = load_csv_safe("./csv/rsi_diver.csv")
rsi_diver_retest_df = load_csv_safe("./csv/rsi_diver_retest.csv")

# --- Create Environment ---
try:
    env = TradeEnv(
        maindf=main_df,
        gape_path="./csv/gape.csv",
        gapebuy_path="./csv/gape_buy.csv",
        shortbuy_path="./csv/short_buy.csv",
        rsi_diver_path="./csv/rsi_diver.csv",
        rsi_diver_retest_path="./csv/rsi_diver_retest.csv"
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