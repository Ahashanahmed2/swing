import os
import pandas as pd
import numpy as np
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env import TradeEnv
import json

# ⚠️ FutureWarning অফ
warnings.simplefilter(action='ignore', category=FutureWarning)

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
# 📊 Load system intelligence files
# ---------------------------------------------------------
strategy_metrics_path = "./output/ai_signal/strategy_metrics.csv"
symbol_ref_path = "./output/ai_signal/symbol_reference_metrics.csv"

strategy_metrics = pd.read_csv(strategy_metrics_path) if os.path.exists(strategy_metrics_path) else pd.DataFrame()
symbol_ref_metrics = pd.read_csv(symbol_ref_path) if os.path.exists(symbol_ref_path) else pd.DataFrame()

# --- Helper Function ---
def load_csv_safe(path, required_columns=None):
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✅ Loaded: {path} ({len(df)} rows)")
            if required_columns:
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = 0
            return df
        else:
            print(f"⚠️ File not found: {path}")
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
    return pd.DataFrame()

# --- Load Datasets ---
required_cols = ['RSI', 'confidence', 'ai_score', 'duration_days', 'close', 'symbol', 'date']
main_df = load_csv_safe('./csv/mongodb.csv', required_columns=required_cols)
gape_df = load_csv_safe('./csv/gape.csv')
gapebuy_df = load_csv_safe('./csv/gape_buy.csv')
shortbuy_df = load_csv_safe('./csv/short_buy.csv')
rsi_diver_df = load_csv_safe('./csv/rsi_diver.csv')
rsi_diver_retest_df = load_csv_safe('./csv/rsi_diver_retest.csv')

# --- ✅ ENHANCED DATA FILTERING: Use YOUR SYSTEM'S METRICS ---
if not main_df.empty and not symbol_ref_metrics.empty:
    print("\n🔍 Optimizing training data using system metrics...")

    # Get symbols with: Win% > 65% AND Expectancy > 100 BDT
    high_quality = symbol_ref_metrics[
        (symbol_ref_metrics['Win%'] > 65) &
        (symbol_ref_metrics['Expectancy (BDT)'] > 100)
    ]
    quality_symbols = high_quality['Symbol'].str.upper().unique().tolist()
    
    print(f"🎯 {len(quality_symbols)} high-quality symbols: {quality_symbols[:5]}...")

    # Filter main_df
    main_df['symbol'] = main_df['symbol'].str.upper()
    main_df = main_df[main_df['symbol'].isin(quality_symbols)]
    print(f"✅ Filtered to {len(main_df)} rows ({main_df['symbol'].nunique()} symbols)")

# --- Create Environment — with SYSTEM CONTEXT ---
try:
    env = TradeEnv(
        maindf=main_df,
        gape_path="./csv/gape.csv",
        gapebuy_path="./csv/gape_buy.csv",
        shortbuy_path="./csv/short_buy.csv",
        rsi_diver_path="./csv/rsi_diver.csv",
        rsi_diver_retest_path="./csv/rsi_diver_retest.csv",
        trade_stock_path="./csv/trade_stock.csv",          # ← your open signals
        metrics_path=strategy_metrics_path,                # ← strategy metrics
        symbol_ref_path=symbol_ref_path,                   # ← symbol×strategy metrics
        config_path=CONFIG_PATH
    )
    print("✅ Environment initialized with SYSTEM INTELLIGENCE")
except Exception as e:
    print(f"❌ Failed to initialize environment: {e}")
    exit()

env = DummyVecEnv([lambda: env])

# --- ✅ ENHANCED PPO: Swing-Optimized Hyperparameters ---
try:
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=2.5e-4,       # ↓ slightly lower for stability
        n_steps=2048,
        batch_size=128,             # ↑ larger batch (better gradient est.)
        n_epochs=15,                # ↑ more epochs per update
        gamma=0.985,                # ↑ longer horizon (swing trades hold days)
        gae_lambda=0.95,
        clip_range=0.15,            # ↓ tighter clipping (more stable)
        ent_coef=0.005,             # ↓ less entropy (more deterministic)
        vf_coef=0.6,                # ↑ value function weight (better risk estimation)
        max_grad_norm=0.5,          # ↓ gradient clipping
        device='cpu'
    )
    print("✅ PPO model created with SWING-OPTIMIZED parameters")
except Exception as e:
    print(f"❌ Failed to create PPO model: {e}")
    exit()

# --- 📊 Custom Callback for Monitoring ---
class TrainingCallback(BaseCallback):
    def __init__(self, check_freq=2000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                avg_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                self.rewards.append(avg_reward)
                print(f"   📊 Step {self.num_timesteps:6d} | Avg Reward: {avg_reward:+.3f}")
        return True

callback = TrainingCallback(check_freq=2000)

# --- 🚀 Train PPO Model ---
try:
    print(f"\n🚀 Training PPO (200,000 timesteps) with SYSTEM-OPTIMIZED data...")
    model.learn(
        total_timesteps=200_000,
        callback=callback,
        log_interval=1000,
        progress_bar=False
    )
    print("✅ PPO Training complete")
except Exception as e:
    print(f"❌ Training failed: {e}")
    exit()

# --- ✅ Save & Report ---
try:
    save_path = './csv/ppo_retrained'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\n✅ PPO model saved at: {save_path}.zip")

    # 📊 Final report
    if callback.rewards:
        print("\n" + "="*50)
        print("📊 PPO TRAINING SUMMARY")
        print("="*50)
        print(f"📈 Avg Final Reward  : {np.mean(callback.rewards[-5:]):+.3f}")
        print(f"📉 Min Reward         : {np.min(callback.rewards):+.3f}")
        print(f"📈 Max Reward         : {np.max(callback.rewards):+.3f}")
        print(f"🎯 Trained on         : {main_df['symbol'].nunique()} high-quality symbols")
        if not symbol_ref_metrics.empty:
            top_symbol = symbol_ref_metrics.loc[symbol_ref_metrics['Expectancy (BDT)'].idxmax()]
            print(f"🏆 Top Symbol         : {top_symbol['Symbol']} ({top_symbol['Expectancy (BDT)']:.0f} BDT)")
        print("="*50)

except Exception as e:
    print(f"❌ Failed to save PPO model: {e}")

# --- 🔁 Optional: Auto-test with generate_signals ---
auto_test = input("\n🧪 Test new model with signal generation? (y/n): ").strip().lower()
if auto_test == 'y':
    try:
        from generate_signals import generate_signals
        print("\n🔍 Generating signals with new PPO model...\n")
        generate_signals()
    except Exception as e:
        print(f"⚠️ Signal test failed: {e}")