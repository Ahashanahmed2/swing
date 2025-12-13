# train_model_ppo.py
import os
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.logger import configure
import gymnasium as gym

# Import your custom environment
from env import TradeEnv  # assuming your env is in trade_env.py

# 🔧 Configuration
MODEL_DIR = "./models"
LOG_DIR = "./logs"
CSV_DIR = "./csv"
OUTPUT_DIR = "./output/ai_signal"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 📁 Load main data (assumed to be preprocessed OHLCV + indicators)
MAIN_DATA_PATH = "./csv/main_data.csv"
if not os.path.exists(MAIN_DATA_PATH):
    raise FileNotFoundError(f"❌ Required data file not found: {MAIN_DATA_PATH}")

print("📥 Loading main dataset...")
maindf = pd.read_csv(MAIN_DATA_PATH)
maindf['date'] = pd.to_datetime(maindf['date'], errors='coerce')
print(f"✅ Loaded {len(maindf)} rows from {MAIN_DATA_PATH}")

# 🌍 Wrap environment for vectorization & monitoring
def make_env():
    def _init():
        env = TradeEnv(
            maindf=maindf,
            gape_path=os.path.join(CSV_DIR, "gape.csv"),
            gapebuy_path=os.path.join(CSV_DIR, "gape_buy.csv"),
            shortbuy_path=os.path.join(CSV_DIR, "short_buy.csv"),
            rsi_diver_path=os.path.join(CSV_DIR, "rsi_diver.csv"),
            rsi_diver_retest_path=os.path.join(CSV_DIR, "rsi_diver_retest.csv"),
            trade_stock_path=os.path.join(CSV_DIR, "trade_stock.csv"),
            metrics_path=os.path.join(OUTPUT_DIR, "strategy_metrics.csv"),
            symbol_ref_path=os.path.join(OUTPUT_DIR, "symbol_reference_metrics.csv"),
            liquidity_path=os.path.join(CSV_DIR, "liquidity_system.csv"),
            config_path="./config.json"
        )
        env = Monitor(env)
        return env
    return _init

# 🧪 Create training & evaluation environments
print("⚙️ Creating environments...")
env = DummyVecEnv([make_env()])
eval_env = DummyVecEnv([make_env()])

# ⚖️ Normalize observations & rewards for faster convergence
print("⚖️ Applying VecNormalize...")
vec_norm = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
vec_norm_eval = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

# 📝 Setup logging
logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])

# 🤖 Initialize PPO model
print("🤖 Initializing PPO model...")
model = PPO(
    "MlpPolicy",
    vec_norm,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=False,
    verbose=1,
    tensorboard_log=LOG_DIR,
    seed=42
)
model.set_logger(logger)

# 📌 Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,  # every 50k steps
    save_path=MODEL_DIR,
    name_prefix="ppo_dse"
)

eval_callback = EvalCallback(
    vec_norm_eval,
    best_model_save_path=os.path.join(MODEL_DIR, "best"),
    log_path=LOG_DIR,
    eval_freq=25_000,      # every 25k training steps
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

# 🚀 Training
TOTAL_TIMESTEPS = 500_000
print(f"🚀 Starting PPO training for {TOTAL_TIMESTEPS:,} timesteps...")
start_time = time.time()

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        tb_log_name="ppo_run"
    )
    train_time = time.time() - start_time
    print(f"✅ Training completed in {train_time/60:.1f} minutes.")
except KeyboardInterrupt:
    print("🛑 Training interrupted by user.")
except Exception as e:
    print(f"❌ Training failed: {e}")
    raise

# 💾 Save final model
final_model_path = os.path.join(MODEL_DIR, "ppo_retrained.zip")
model.save(final_model_path)
vec_norm.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
print(f"💾 Final model saved to: {final_model_path}")
print(f"📊 VecNormalize stats saved to: {MODEL_DIR}/vec_normalize.pkl")

# 🧪 Optional: Run quick evaluation
print("\n🔍 Running final evaluation...")
vec_norm_eval.training = False
vec_norm_eval.norm_reward = False
obs = vec_norm_eval.reset()
total_reward = 0
done = False
steps = 0
max_steps = 1000

while not done and steps < max_steps:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done_array, info = vec_norm_eval.step(action)
    done = done_array[0]
    total_reward += reward[0]
    steps += 1

print(f"📊 Final eval episode reward: {total_reward:.2f} over {steps} steps")

# 🎉 Done!
print("\n🎉 PPO training pipeline completed successfully!")
print(f"→ Best model: {os.path.join(MODEL_DIR, 'best', 'best_model.zip')}")
print(f"→ Final model: {final_model_path}")
print(f"→ Logs & TensorBoard: {LOG_DIR}")
print("\nTo launch TensorBoard: `tensorboard --logdir ./logs`")