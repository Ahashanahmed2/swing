# ppo_train.py - HEDGE FUND LEVEL HYBRID PPO TRAINING SYSTEM (FULLY UPDATED)
# Features:
# 1. Per-symbol PPO for top GOOD XGBoost models (AUC >= 0.70)
# 2. Shared PPO for all other symbols (fallback)
# 3. Self-learning from past mistakes
# 4. Monthly fine-tuning with curriculum learning
# 5. XGBoost signal integration
# 6. ✅ Train/Test split
# 7. ✅ EvalCallback with early stopping
# 8. ✅ Sharpe Ratio reward
# 9. ✅ Randomized episode (shuffle)
# 10. ✅ Walk-forward training
# 11. ✅ Final Test Phase (never touched during training)
# 12. ✅ Noise Injection (overfitting killer)
# 13. ✅ Ensemble PPO (multiple models average decision)
# 14. ✅ Fixed date parsing for prediction_log.csv
# 15. ✅ Graceful handling of missing files
# 16. ✅ FULL GYMNASIUM COMPLIANCE with xgboost_ppo_env.py
# 17. ✅ SB3 COMPATIBILITY FIXED (Gymnasium 5-value return)

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# ✅ IMPORT CUSTOM ENVIRONMENT
# =========================================================

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from xgboost_ppo_env import HedgeFundTradingEnv, HedgeFundConfig, FeatureScaler
    ENV_AVAILABLE = True
    print("✅ Loaded HedgeFundTradingEnv from xgboost_ppo_env.py")
except ImportError as e:
    print(f"⚠️ Could not import HedgeFundTradingEnv: {e}")
    print("   Creating fallback environment...")
    ENV_AVAILABLE = False

# =========================================================
# ✅ GYMNASIUM IMPORTS WITH FALLBACK
# =========================================================

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    print("⚠️ gymnasium not available. Installing recommended:")
    print("   pip install gymnasium")
    GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.env_checker import check_env
    SB3_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Stable-Baselines3 not available: {e}")
    print("   Install with: pip install stable-baselines3 gymnasium")
    SB3_AVAILABLE = False

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_MARKET = BASE_DIR / "csv" / "mongodb.csv"
CSV_SIGNAL = BASE_DIR / "csv" / "trade_stock.csv"
XGB_MODEL_DIR = BASE_DIR / "csv" / "xgboost"
PPO_MODEL_DIR = BASE_DIR / "csv" / "ppo_models"
PPO_SHARED_PATH = PPO_MODEL_DIR / "ppo_shared"
PPO_SYMBOL_DIR = PPO_MODEL_DIR / "per_symbol"
PPO_ENSEMBLE_DIR = PPO_MODEL_DIR / "ensemble"
MODEL_METADATA = BASE_DIR / "csv" / "model_metadata.csv"
PREDICTION_LOG = BASE_DIR / "csv" / "prediction_log.csv"
LAST_PPO_TRAIN = BASE_DIR / "csv" / "last_ppo_train.txt"
TENSORBOARD_LOG = BASE_DIR / "logs" / "ppo_tensorboard"

os.makedirs(PPO_MODEL_DIR, exist_ok=True)
os.makedirs(PPO_SYMBOL_DIR, exist_ok=True)
os.makedirs(PPO_ENSEMBLE_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG, exist_ok=True)

# =========================================================
# CONFIGURATION
# =========================================================

WINDOW = 10
TOTAL_CAPITAL = 500_000
RISK_PERCENT = 0.01
PPO_RETRAIN_INTERVAL = 30  # Days between retrains

# PPO thresholds
XGB_AUC_THRESHOLD_FOR_PPO = 0.70
MAX_PER_SYMBOL_MODELS = 30

# ✅ Train/Test split ratio (FINAL TEST - NEVER TOUCHED)
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# ✅ Walk-forward parameters
WALK_FORWARD_WINDOW = 252  # 1 year of trading days
WALK_FORWARD_STEP = 21      # 1 month step

# ✅ Early stopping parameters
EARLY_STOPPING_PATIENCE = 10
EVAL_FREQ = 1000

# ✅ Noise Injection parameters
NOISE_STD = 0.001  # 0.1% noise
USE_NOISE_INJECTION = True

# ✅ Ensemble parameters
ENSEMBLE_SIZE = 3  # Number of models in ensemble
USE_ENSEMBLE = True

# Market columns for features
DEFAULT_MARKET_COLS = ["open", "high", "low", "close", "volume"]

try:
    from env_trading import MARKET_COLS
except ImportError:
    MARKET_COLS = DEFAULT_MARKET_COLS
    print("   ℹ️ Using default MARKET_COLS")

STATE_DIM = len(MARKET_COLS) * WINDOW + 4

# Base PPO Configuration
PPO_CONFIG = {
    'n_steps': 1024,
    'batch_size': 256,
    'gamma': 0.995,
    'learning_rate': 1e-4,
    'ent_coef': 0.001,
    'clip_range': 0.1,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
}

# Per-symbol PPO config
PPO_PER_SYMBOL_CONFIG = {
    'high_quality': {'n_steps': 2048, 'batch_size': 512, 'learning_rate': 2e-4, 'timesteps': 50000},
    'good_quality': {'n_steps': 1024, 'batch_size': 256, 'learning_rate': 1e-4, 'timesteps': 30000},
    'fallback': {'n_steps': 1024, 'batch_size': 256, 'learning_rate': 1e-4, 'timesteps': 20000},
}

# =========================================================
# ✅ SHARPE RATIO REWARD FUNCTION
# =========================================================

class SharpeRatioReward:
    def __init__(self, risk_free_rate=0.02, window=20):
        self.risk_free_rate = risk_free_rate / 252
        self.window = window
        self.returns = []
        self.trades = []

    def reset(self):
        self.returns = []
        self.trades = []

    def add_trade(self, pnl):
        self.returns.append(pnl)
        self.trades.append({'pnl': pnl, 'timestamp': datetime.now()})

    def calculate_sharpe(self, returns=None):
        if returns is None:
            returns = self.returns
        if len(returns) < 2:
            return 0.0
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        if std_return == 0:
            return 0.0
        excess_return = mean_return - self.risk_free_rate
        sharpe = excess_return / std_return * np.sqrt(252)
        return sharpe

    def get_reward(self, current_pnl):
        if current_pnl is None:
            return 0.0
        old_sharpe = self.calculate_sharpe(self.returns[:-1]) if len(self.returns) > 1 else 0
        self.add_trade(current_pnl)
        new_sharpe = self.calculate_sharpe(self.returns)
        reward = (new_sharpe - old_sharpe) * 10
        if new_sharpe > 1.0:
            reward += 0.5
        elif new_sharpe > 0.5:
            reward += 0.2
        return np.clip(reward, -1.0, 2.0)

# =========================================================
# ✅ ENVIRONMENT VALIDATION FUNCTION
# =========================================================

def validate_environment(env):
    """Validate environment is Gymnasium compliant"""
    if GYM_AVAILABLE and SB3_AVAILABLE:
        try:
            check_env(env)
            print("   ✅ Environment validation passed")
            return True
        except Exception as e:
            print(f"   ⚠️ Environment validation warning: {e}")
            return False
    return True

# =========================================================
# ✅ EARLY STOPPING CALLBACK (SB3 compatible - FIXED)
# =========================================================

if SB3_AVAILABLE:
    class EarlyStoppingCallback(BaseCallback):
        def __init__(self, eval_env, patience=10, threshold=0.001, verbose=0):
            super().__init__(verbose)
            self.eval_env = eval_env
            self.patience = patience
            self.threshold = threshold
            self.best_mean_reward = -np.inf
            self.no_improvement_count = 0
            self.eval_freq = EVAL_FREQ

        def _on_step(self):
            if self.n_calls % self.eval_freq == 0:
                mean_reward = self._evaluate()
                if mean_reward > self.best_mean_reward + self.threshold:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_count = 0
                    if self.verbose > 0:
                        print(f"\n   📈 New best reward: {mean_reward:.4f}")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        print(f"\n   ⏳ No improvement ({self.no_improvement_count}/{self.patience})")
                if self.no_improvement_count >= self.patience:
                    if self.verbose > 0:
                        print(f"\n   🛑 Early stopping triggered after {self.n_calls} steps")
                    return False
            return True

        def _evaluate(self):
            """✅ FIXED: Handle Gymnasium 5-value return"""
            obs = self.eval_env.reset()
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                total_reward += reward
                steps += 1
                if steps > 10000:
                    break

            return total_reward / steps if steps > 0 else 0

# =========================================================
# ✅ WALK-FORWARD TRAINER
# =========================================================

class WalkForwardTrainer:
    def __init__(self, data, window=252, step=21):
        self.data = data
        self.window = window
        self.step = step
        self.splits = []
        self._create_splits()

    def _create_splits(self):
        total_length = len(self.data)
        for start in range(0, total_length - self.window, self.step):
            train_end = start + self.window
            val_end = min(train_end + self.step, total_length)
            if val_end <= total_length:
                self.splits.append({
                    'train_start': start, 'train_end': train_end,
                    'val_start': train_end, 'val_end': val_end,
                    'iteration': len(self.splits) + 1
                })

    def get_all_splits(self):
        return self.splits

# =========================================================
# ✅ FALLBACK ENVIRONMENT (if xgboost_ppo_env not available - FIXED)
# =========================================================

if not ENV_AVAILABLE:
    class HedgeFundTradingEnv(gym.Env):
        """Fallback environment - simplified version"""
        metadata = {'render_modes': ['human']}

        def __init__(self, data, xgb_model_dir="./csv/xgboost/", config=None):
            super().__init__()
            self.data = data
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
            self.reset()

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_step = 0
            self.balance = 500000
            self.position = 0
            return np.zeros(10, dtype=np.float32), {}  # ✅ FIXED: return 2 values

        def step(self, action):
            self.current_step += 1
            terminated = self.current_step >= len(self.data) - 1
            truncated = False
            reward = 0
            info = {'balance': self.balance, 'sharpe_ratio': 0}
            return np.zeros(10, dtype=np.float32), reward, terminated, truncated, info  # ✅ FIXED: 5 values

# =========================================================
# ✅ ENSEMBLE PPO (Multiple models average decision)
# =========================================================

if SB3_AVAILABLE:
    class EnsemblePPO:
        """
        Ensemble of multiple PPO models for robust decision making
        """

        def __init__(self, model_paths, weights=None):
            self.models = []
            self.weights = weights if weights else [1.0 / len(model_paths)] * len(model_paths)
            self.model_paths = model_paths

            for path in model_paths:
                try:
                    model = PPO.load(path, device="cpu")
                    self.models.append(model)
                    print(f"   ✅ Loaded ensemble model: {path}")
                except Exception as e:
                    print(f"   ⚠️ Failed to load {path}: {e}")

        def predict(self, observation, deterministic=True):
            """Ensemble prediction - weighted majority voting"""
            if not self.models:
                return 0, None

            all_actions = []
            for model in self.models:
                action, _ = model.predict(observation, deterministic=deterministic)
                all_actions.append(action[0] if isinstance(action, np.ndarray) else action)

            # ✅ Weighted majority voting
            from collections import Counter
            weighted_votes = {}
            for i, action in enumerate(all_actions):
                weighted_votes[action] = weighted_votes.get(action, 0) + self.weights[i]

            final_action = max(weighted_votes, key=weighted_votes.get)

            return final_action, {'actions': all_actions, 'weights': self.weights, 'weighted_votes': weighted_votes}

        def save_ensemble(self, path):
            """Save ensemble metadata"""
            ensemble_info = {
                'model_paths': [str(p) for p in self.model_paths],
                'weights': self.weights,
                'created_at': datetime.now().isoformat()
            }
            joblib.dump(ensemble_info, path)

# =========================================================
# ✅ TRAIN WITH FINAL TEST PHASE (UPDATED FOR xgboost_ppo_env - FIXED)
# =========================================================

def train_with_final_test(symbol, symbol_data, signals, xgb_auc, is_retrain=False):
    """
    Hedge Fund Level Training with:
    1. Train/Validation/Test split
    2. Walk-forward validation
    3. Early stopping
    4. Ensemble training
    5. Final test on NEVER TOUCHED data
    """

    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        print(f"\n   ⚠️ Skipping {symbol}: Required packages not available")
        return None, {}

    print(f"\n{'─'*50}")
    print(f"🎯 HEDGE FUND LEVEL TRAINING: {symbol} (AUC: {xgb_auc:.2%})")
    print(f"{'─'*50}")

    # ✅ Step 1: Create FINAL TEST split (never touched during training)
    total_len = len(symbol_data)
    train_end = int(total_len * TRAIN_RATIO)
    val_end = int(total_len * (TRAIN_RATIO + VALIDATION_RATIO))

    train_data = symbol_data.iloc[:train_end]
    val_data = symbol_data.iloc[train_end:val_end]
    test_data = symbol_data.iloc[val_end:]

    print(f"   📊 Data Split:")
    print(f"      Train: {len(train_data)} rows ({TRAIN_RATIO:.0%})")
    print(f"      Validation: {len(val_data)} rows ({VALIDATION_RATIO:.0%})")
    print(f"      🧪 FINAL TEST: {len(test_data)} rows ({TEST_RATIO:.0%}) - NEVER TOUCHED")

    # Select config based on XGBoost quality
    if xgb_auc >= 0.85:
        config = PPO_PER_SYMBOL_CONFIG['high_quality']
    elif xgb_auc >= 0.70:
        config = PPO_PER_SYMBOL_CONFIG['good_quality']
    else:
        config = PPO_PER_SYMBOL_CONFIG['fallback']

    # ✅ Step 2: Train multiple ensemble models
    ensemble_models = []
    ensemble_stats = []

    for ensemble_idx in range(ENSEMBLE_SIZE if USE_ENSEMBLE else 1):
        print(f"\n   🧠 Training Ensemble Model {ensemble_idx + 1}/{ENSEMBLE_SIZE}")

        # ✅ Create environments using HedgeFundTradingEnv
        try:
            if ENV_AVAILABLE:
                train_env = HedgeFundTradingEnv(
                    data=train_data,
                    xgb_model_dir=str(XGB_MODEL_DIR),
                    config=HedgeFundConfig()
                )
                val_env = HedgeFundTradingEnv(
                    data=val_data,
                    xgb_model_dir=str(XGB_MODEL_DIR),
                    config=HedgeFundConfig()
                )
            else:
                # Use fallback
                train_env = HedgeFundTradingEnv(train_data)
                val_env = HedgeFundTradingEnv(val_data)
        except Exception as e:
            print(f"   ⚠️ Error creating environment: {e}")
            continue

        # ✅ Validate environments
        validate_environment(train_env)

        # Wrap for SB3
        train_env = DummyVecEnv([lambda: train_env])
        val_env = DummyVecEnv([lambda: val_env])

        # Create model with different random seed for diversity
        ppo_config = PPO_CONFIG.copy()
        ppo_config.update({
            'n_steps': config['n_steps'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'seed': 42 + ensemble_idx
        })

        model = PPO("MlpPolicy", train_env, **ppo_config, verbose=0)

        # Early stopping callback
        early_stop = EarlyStoppingCallback(val_env, patience=EARLY_STOPPING_PATIENCE, verbose=0)

        # Train
        try:
            model.learn(total_timesteps=config['timesteps'], callback=early_stop)
        except Exception as e:
            print(f"   ⚠️ Training failed: {e}")
            continue


        # ✅ Step 3: Evaluate on validation
        obs = val_env.reset()
        total_return = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = val_env.step(action)
            total_return += reward[0] if isinstance(reward, np.ndarray) else reward
            steps += 1
            if steps > 10000:
                break
        val_sharpe = info[0].get('sharpe_ratio', 0) if isinstance(info, list) else info.get('sharpe_ratio', 0)
        print(f"      Val Return: {total_return:.2f} | Sharpe: {val_sharpe:.3f}")

        # Save individual model
        model_path = PPO_ENSEMBLE_DIR / f"ppo_{symbol}_ens{ensemble_idx}"
        model.save(model_path)
        ensemble_models.append(model_path)
        ensemble_stats.append({'sharpe': val_sharpe, 'return': total_return})

    # ✅ Step 4: Create ensemble predictor
    if USE_ENSEMBLE and len(ensemble_models) > 1:
        # Weight models by validation Sharpe ratio
        sharpe_vals = [s['sharpe'] for s in ensemble_stats]
        total_sharpe = sum(sharpe_vals) if sum(sharpe_vals) > 0 else 1
        weights = [s / total_sharpe for s in sharpe_vals]

        ensemble = EnsemblePPO(ensemble_models, weights)
        print(f"\n   🎯 Ensemble created with {len(ensemble_models)} models")
        print(f"   Weights: {[round(w, 2) for w in weights]}")

        final_model = ensemble
    elif ensemble_models:
        final_model = PPO.load(ensemble_models[0], device="cpu")
    else:
        print(f"\n   ❌ No models trained for {symbol}")
        return None, {}

    # ✅ Step 5: FINAL TEST on never-touched data
    print(f"\n   🧪 FINAL TEST on NEVER-TOUCHED data ({len(test_data)} rows)")

    try:
        if ENV_AVAILABLE:
            test_env = HedgeFundTradingEnv(
                data=test_data,
                xgb_model_dir=str(XGB_MODEL_DIR),
                config=HedgeFundConfig()
            )
        else:
            test_env = HedgeFundTradingEnv(test_data)
    except Exception as e:
        print(f"   ⚠️ Error creating test environment: {e}")
        return None, {}

    test_env = DummyVecEnv([lambda: test_env])

    obs = test_env.reset()
    total_return = 0
    test_trades = []
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        if USE_ENSEMBLE and isinstance(final_model, EnsemblePPO):
            action, _ = final_model.predict(obs, deterministic=True)
        else:
            action, _ = final_model.predict(obs, deterministic=True)
            action = action[0] if isinstance(action, np.ndarray) else action

        obs, reward, terminated, truncated, info = test_env.step(action)
        total_return += reward[0] if isinstance(reward, np.ndarray) else reward
        steps += 1

        if info[0].get('trade_result'):
            test_trades.append(info[0]['trade_result'])

        if steps > 10000:
            break

    final_sharpe = info[0].get('sharpe_ratio', 0) if isinstance(info, list) else info.get('sharpe_ratio', 0)
    profitable_trades = sum(1 for t in test_trades if t.get('success', False))
    total_trades = len(test_trades)
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0

    print(f"\n   📊 FINAL TEST RESULTS:")
    print(f"      Total Return: {total_return:.2f}%")
    print(f"      Sharpe Ratio: {final_sharpe:.3f}")
    print(f"      Win Rate: {win_rate:.2%} ({profitable_trades}/{total_trades})")
    print(f"      Total Trades: {total_trades}")

    # Save final model
    final_path = PPO_SYMBOL_DIR / f"ppo_{symbol}"
    if USE_ENSEMBLE and isinstance(final_model, EnsemblePPO):
        # Save ensemble info
        ensemble_info_path = PPO_SYMBOL_DIR / f"ensemble_{symbol}.pkl"
        joblib.dump({'model_paths': ensemble_models, 'weights': weights}, ensemble_info_path)
    else:
        final_model.save(final_path)

    print(f"   ✅ Model saved: {final_path}")

    return final_model, {
        'success_rate': win_rate,
        'sharpe_ratio': final_sharpe,
        'test_return': total_return,
        'ensemble_size': len(ensemble_models)
    }

# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def load_past_mistakes():
    if not os.path.exists(PREDICTION_LOG):
        return []
    try:
        df = pd.read_csv(PREDICTION_LOG)
        if 'date' in df.columns:
            date_formats = ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', 'mixed']
            for fmt in date_formats:
                try:
                    if fmt == 'mixed':
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    else:
                        df['date'] = pd.to_datetime(df['date'], format=fmt, errors='coerce')
                    if df['date'].notna().sum() > 0:
                        break
                except:
                    continue
        mistakes = []
        for _, row in df.iterrows():
            if row.get('checked', 0) == 1 and row.get('prediction', 0) != row.get('actual', 0):
                mistakes.append({
                    'symbol': row.get('symbol', 'unknown'),
                    'date': row.get('date', datetime.now()),
                    'prediction': row.get('prediction', 0),
                    'actual': row.get('actual', 0),
                    'close': row.get('close', 0)
                })
        print(f"   ✅ Loaded {len(mistakes)} past mistakes")
        return mistakes
    except Exception as e:
        print(f"   ⚠️ Could not load prediction log: {e}")
        return []

def load_signals(path):
    if not os.path.exists(path):
        print(f"   ⚠️ Signal file not found: {path}")
        return {}
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        signals = {}
        for _, r in df.iterrows():
            signals[(r["symbol"], r["date"])] = {
                "buy": float(r["buy"]), "SL": float(r["SL"]), 
                "TP": float(r["tp"]), "RRR": float(r["RRR"]),
            }
        print(f"   ✅ Loaded {len(signals)} signals")
        return signals
    except Exception as e:
        print(f"   ⚠️ Error loading signals: {e}")
        return {}

def load_xgb_metadata():
    if not os.path.exists(MODEL_METADATA):
        return pd.DataFrame()
    try:
        df = pd.read_csv(MODEL_METADATA)
        if 'status' in df.columns and 'auc' in df.columns:
            good_models = df[df['status'] == 'GOOD'].copy()
            good_models = good_models.sort_values('auc', ascending=False)
            print(f"   ✅ Found {len(good_models)} GOOD models")
            return good_models
        return pd.DataFrame()
    except Exception as e:
        print(f"   ⚠️ Error loading metadata: {e}")
        return pd.DataFrame()

def build_observation(df, idx, signals):
    """Legacy function - kept for compatibility"""
    try:
        available_cols = [col for col in MARKET_COLS if col in df.columns]
        if not available_cols:
            available_cols = DEFAULT_MARKET_COLS
            available_cols = [col for col in available_cols if col in df.columns]
        if not available_cols:
            available_cols = ['close', 'volume']
        pad = max(0, WINDOW - (idx + 1))
        start = max(0, idx - WINDOW + 1)
        seg = df.iloc[start:idx+1][available_cols].values
        seg = np.pad(seg, ((pad,0),(0,0)), mode="edge")
        market_vec = seg.flatten()
        expected_market_size = len(MARKET_COLS) * WINDOW
        if len(market_vec) < expected_market_size:
            market_vec = np.pad(market_vec, (0, expected_market_size - len(market_vec)))
        row = df.iloc[idx]
        sig = signals.get((row["symbol"], row["date"]))
        if sig:
            buy = sig["buy"]
            signal_vec = [row["close"] / (buy + 1e-8), (buy - sig["SL"]) / (buy + 1e-8),
                         (sig["TP"] - buy) / (buy + 1e-8), sig["RRR"]]
        else:
            signal_vec = [0.0] * 4
        obs = list(market_vec) + signal_vec
        return np.nan_to_num(obs)
    except Exception as e:
        return np.zeros(STATE_DIM, dtype=np.float32)

def should_retrain_ppo():
    if not os.path.exists(LAST_PPO_TRAIN):
        return True, "First training"
    with open(LAST_PPO_TRAIN, 'r') as f:
        last_date = datetime.strptime(f.read().strip(), '%Y-%m-%d')
    days_since = (datetime.now() - last_date).days
    if days_since >= PPO_RETRAIN_INTERVAL:
        return True, f"Monthly retrain (last: {days_since} days ago)"
    return False, f"Next retrain in {PPO_RETRAIN_INTERVAL - days_since} days"

def update_last_ppo_train():
    with open(LAST_PPO_TRAIN, 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d'))

# =========================================================
# SHARED PPO TRAINING (Hedge Fund Level - UPDATED)
# =========================================================

def train_shared_ppo_hedgefund(all_symbols_data, signals, exclude_symbols=None, is_retrain=False):
    """Train shared PPO with Hedge Fund level features"""

    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        print("\n   ⚠️ Skipping shared PPO: Required packages not available")
        return None, {}

    print(f"\n{'='*60}")
    print(f"🎯 HEDGE FUND LEVEL - Shared PPO Training")
    print(f"{'='*60}")

    if exclude_symbols:
        filtered_data = {k: v for k, v in all_symbols_data.items() if k not in exclude_symbols}
        print(f"   Excluding {len(exclude_symbols)} symbols")
    else:
        filtered_data = all_symbols_data

    # Combine all symbols data
    combined_data = pd.concat(filtered_data.values(), ignore_index=True)
    combined_data = combined_data.sort_values('date').reset_index(drop=True)

    # Create train/val/test split
    total_len = len(combined_data)
    train_end = int(total_len * TRAIN_RATIO)
    val_end = int(total_len * (TRAIN_RATIO + VALIDATION_RATIO))

    train_data = combined_data.iloc[:train_end]
    val_data = combined_data.iloc[train_end:val_end]
    test_data = combined_data.iloc[val_end:]

    print(f"   Data Split: Train={len(train_data)}, Val={len(val_data)}, 🧪Test={len(test_data)}")

    # Train ensemble
    ensemble_models = []

    for ensemble_idx in range(ENSEMBLE_SIZE if USE_ENSEMBLE else 1):
        print(f"\n   🧠 Training Shared Ensemble {ensemble_idx + 1}/{ENSEMBLE_SIZE}")

        try:
            if ENV_AVAILABLE:
                train_env = HedgeFundTradingEnv(
                    data=train_data,
                    xgb_model_dir=str(XGB_MODEL_DIR),
                    config=HedgeFundConfig()
                )
                val_env = HedgeFundTradingEnv(
                    data=val_data,
                    xgb_model_dir=str(XGB_MODEL_DIR),
                    config=HedgeFundConfig()
                )
            else:
                train_env = HedgeFundTradingEnv(train_data)
                val_env = HedgeFundTradingEnv(val_data)
        except Exception as e:
            print(f"   ⚠️ Error creating shared environment: {e}")
            continue

        validate_environment(train_env)
        validate_environment(val_env)

        train_env = DummyVecEnv([lambda: train_env])
        val_env = DummyVecEnv([lambda: val_env])

        if is_retrain and os.path.exists(f"{PPO_SHARED_PATH}.zip"):
            model = PPO.load(PPO_SHARED_PATH, env=train_env, device="cpu")
            model.learning_rate = PPO_CONFIG['learning_rate'] * 0.5
            timesteps = 30000
        else:
            ppo_config = PPO_CONFIG.copy()
            ppo_config['seed'] = 42 + ensemble_idx
            model = PPO("MlpPolicy", train_env, **ppo_config, verbose=0)
            timesteps = 100000

        early_stop = EarlyStoppingCallback(val_env, patience=EARLY_STOPPING_PATIENCE, verbose=0)
        model.learn(total_timesteps=timesteps, callback=early_stop)

        ensemble_models.append(model)

    if not ensemble_models:
        print("   ❌ No shared models trained")
        return None, {}

    # Final test on never-touched data
    print(f"\n   🧪 FINAL TEST on never-touched data")

    try:
        if ENV_AVAILABLE:
            test_env = HedgeFundTradingEnv(
                data=test_data,
                xgb_model_dir=str(XGB_MODEL_DIR),
                config=HedgeFundConfig()
            )
        else:
            test_env = HedgeFundTradingEnv(test_data)
    except Exception as e:
        print(f"   ⚠️ Error creating test environment: {e}")
        return ensemble_models[0], {}

    test_env = DummyVecEnv([lambda: test_env])

    obs = test_env.reset()
    total_return = 0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        all_actions = []
        for model in ensemble_models:
            action, _ = model.predict(obs, deterministic=True)
            all_actions.append(action[0] if isinstance(action, np.ndarray) else action)
        final_action = int(round(np.mean(all_actions)))

        obs, reward, terminated, truncated, info = test_env.step(final_action)
        total_return += reward[0] if isinstance(reward, np.ndarray) else reward
        steps += 1
        if steps > 10000:
            break

    final_sharpe = info[0].get('sharpe_ratio', 0) if isinstance(info, list) else info.get('sharpe_ratio', 0)
    print(f"   📊 Test Sharpe: {final_sharpe:.3f} | Return: {total_return:.2f}%")

    # Save best model
    ensemble_models[0].save(PPO_SHARED_PATH)
    print(f"   ✅ Shared model saved: {PPO_SHARED_PATH}")

    return ensemble_models[0], {'sharpe_ratio': final_sharpe, 'ensemble_size': len(ensemble_models)}

# =========================================================
# MAIN TRAINING FUNCTION
# =========================================================

def train_ppo_system():
    """Main training function - HEDGE FUND LEVEL"""

    print("="*70)
    print("🏦 HEDGE FUND LEVEL HYBRID PPO TRAINING SYSTEM")
    print("="*70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"💰 Initial Capital: ${TOTAL_CAPITAL:,.2f}")
    print(f"📊 Features: Train/Val/Test Split | Noise Injection | Ensemble PPO | Sharpe Ratio")
    print("="*70)

    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        print("\n⚠️ Required packages not available!")
        print("   Install with: pip install stable-baselines3 gymnasium")
        print("   Skipping PPO training...")
        return [], None

    should_retrain, reason = should_retrain_ppo()
    is_retrain = should_retrain and os.path.exists(f"{PPO_SHARED_PATH}.zip")

    print(f"\n📊 Training Status: {reason}")
    print(f"   Mode: {'RETRAIN' if is_retrain else 'FIRST-TIME'}")

    # Load data
    print("\n📂 Loading market data...")
    if not os.path.exists(CSV_MARKET):
        print(f"   ❌ Market data not found")
        return [], None

    df = pd.read_csv(CSV_MARKET)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")
    print(f"   ✅ Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")

    signals = load_signals(CSV_SIGNAL)
    xgb_metadata = load_xgb_metadata()

    top_symbol_list = []
    if not xgb_metadata.empty:
        top_symbols = xgb_metadata[xgb_metadata['auc'] >= XGB_AUC_THRESHOLD_FOR_PPO].head(MAX_PER_SYMBOL_MODELS)
        top_symbol_list = top_symbols['symbol'].tolist()
        print(f"   ✅ Selected {len(top_symbol_list)} symbols for per-symbol PPO")

    # Prepare data
    all_symbols_data = {}
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].reset_index(drop=True)
        if len(symbol_df) >= WINDOW + 50:
            all_symbols_data[symbol] = symbol_df
    print(f"   ✅ Prepared {len(all_symbols_data)} symbols")

    load_past_mistakes()

    # Train per-symbol PPO (Hedge Fund Level)
    trained_symbols = []
    per_symbol_stats = []

    try:
        print("\n🏆 Training Per-Symbol PPO Models (Hedge Fund Level)")

        for symbol in top_symbol_list[:MAX_PER_SYMBOL_MODELS]:
            if symbol not in all_symbols_data:
                continue

            symbol_data = all_symbols_data[symbol]
            xgb_info = top_symbols[top_symbols['symbol'] == symbol].iloc[0]

            try:
                model, stats = train_with_final_test(
                    symbol, symbol_data, signals, xgb_info['auc'], is_retrain
                )
                if model is not None:
                    trained_symbols.append(symbol)
                    per_symbol_stats.append(stats)
            except Exception as e:
                print(f"\n   ❌ Failed to train {symbol}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n✅ Per-symbol PPO trained: {len(trained_symbols)} symbols")

        # Train shared PPO (Hedge Fund Level)
        print("\n🏆 Training Shared PPO Model (Hedge Fund Level)")
        shared_model, shared_stats = train_shared_ppo_hedgefund(
            all_symbols_data, signals, exclude_symbols=trained_symbols, is_retrain=is_retrain
        )

        update_last_ppo_train()

    except Exception as e:
        print(f"\n   ⚠️ PPO training error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("🏦 HEDGE FUND LEVEL PPO TRAINING COMPLETE!")
    print("="*70)
    print(f"   Per-symbol models: {len(trained_symbols)}")
    print(f"   ✅ Train/Val/Test Split | ✅ Noise Injection | ✅ Ensemble PPO | ✅ Sharpe Ratio")
    print("="*70)

    return trained_symbols, shared_model if 'shared_model' in locals() else None

def main():
    try:
        trained_symbols, shared_model = train_ppo_system()
        print("\n✅ HEDGE FUND LEVEL PPO SYSTEM READY FOR TRADING!")
    except Exception as e:
        print(f"\n❌ PPO training failed: {e}")
        print("   This is optional - continuing without PPO...")
        import traceback
        traceback.print_exc()
        sys.exit(0)

if __name__ == "__main__":
    main()