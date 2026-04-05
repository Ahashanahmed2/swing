# ppo_train.py - HEDGE FUND LEVEL HYBRID PPO TRAINING SYSTEM (SIGNAL FIX)
# فقط bug fix - কোন স্ট্রাকচারাল পরিবর্তন নয়

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
PPO_RETRAIN_INTERVAL = 7  # Days between retrains

# PPO thresholds
XGB_AUC_THRESHOLD_FOR_PPO = 0.60
MAX_PER_SYMBOL_MODELS = 30

# ✅ Train/Test split ratio (FINAL TEST - NEVER TOUCHED)
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# ✅ Walk-forward parameters
WALK_FORWARD_WINDOW = 252  # 1 year of trading days
WALK_FORWARD_STEP = 21      # 1 month step

# ✅ Early stopping parameters
EARLY_STOPPING_PATIENCE = 5
EVAL_FREQ = 1000

# ✅ Noise Injection parameters
NOISE_STD = 0.001  # 0.1% noise
USE_NOISE_INJECTION = True

# ✅ Ensemble parameters
ENSEMBLE_SIZE = 2  # Number of models in ensemble
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
    'n_steps': 512,
    'batch_size': 128,
    'gamma': 0.995,
    'learning_rate': 1e-4,
    'ent_coef': 0.001,
    'clip_range': 0.1,
    'vf_coef': 0.2,
    'max_grad_norm': 0.5,
}

# Per-symbol PPO config
PPO_PER_SYMBOL_CONFIG = {
    'high_quality': {'n_steps': 2048, 'batch_size': 512, 'learning_rate': 2e-4, 'timesteps': 20000},
    'good_quality': {'n_steps': 1024, 'batch_size': 256, 'learning_rate': 1e-4, 'timesteps': 15000},
    'fallback': {'n_steps': 1024, 'batch_size': 256, 'learning_rate': 1e-4, 'timesteps': 10000},
}

# =========================================================
# ✅ HELPER FUNCTION FOR VECENV ACTION FORMAT
# =========================================================

def ensure_vecenv_action(action):
    """Convert action to format expected by VecEnv"""
    if isinstance(action, (int, float, np.integer, np.floating)):
        return [int(action)]
    elif isinstance(action, np.ndarray) and action.ndim == 0:
        return [int(action.item())]
    elif isinstance(action, (list, tuple)):
        return list(action)
    return action

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
# ✅ EARLY STOPPING CALLBACK (SB3 compatible - FULLY FIXED)
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
            """✅ FIXED: Safe Gymnasium 5-value return handling with VecEnv action format"""
            obs = self.eval_env.reset()

            # Handle (obs, info) tuple from Gymnasium
            if isinstance(obs, tuple):
                obs = obs[0]

            total_reward = 0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                # Reshape obs for SB3 if needed (batch dimension)
                if isinstance(obs, np.ndarray) and len(obs.shape) == 1:
                    obs = obs.reshape(1, -1)

                action, _ = self.model.predict(obs, deterministic=True)

                # ✅ FIXED: Convert action to VecEnv format
                action = ensure_vecenv_action(action)

                step_result = self.eval_env.step(action)

                # Safe unpacking with validation
                if step_result is None or len(step_result) == 0:
                    break

                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                elif len(step_result) == 4:
                    # Legacy Gym style
                    obs, reward, terminated, info = step_result
                    truncated = False
                else:
                    # Fallback
                    obs = step_result[0]
                    reward = step_result[1] if len(step_result) > 1 else 0
                    terminated = step_result[2] if len(step_result) > 2 else False
                    truncated = step_result[3] if len(step_result) > 3 else False
                    info = step_result[4] if len(step_result) > 4 else {}

                # Safe reward extraction
                if isinstance(reward, (list, np.ndarray)):
                    reward = reward[0] if len(reward) > 0 else 0
                elif not isinstance(reward, (int, float)):
                    reward = 0

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
# ✅ FALLBACK ENVIRONMENT (if xgboost_ppo_env not available - IMPROVED REWARD)
# =========================================================

if not ENV_AVAILABLE:
    class HedgeFundTradingEnv(gym.Env):
        """Fallback environment - simplified version with realistic reward"""
        metadata = {'render_modes': ['human']}

        def __init__(self, data, xgb_model_dir="./csv/xgboost/", config=None):
            super().__init__()
            self.data = data
            self.current_step = 0
            self.balance = 500000
            self.position = 0
            self.entry_price = 0
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
            self._sharpe_calculator = SharpeRatioReward()
            self.trade_count = 0

        def reset(self, seed=None, options=None):
            """Return (obs, info) tuple"""
            super().reset(seed=seed)
            self.current_step = 0
            self.balance = 500000
            self.position = 0
            self.entry_price = 0
            self.trade_count = 0
            self._sharpe_calculator.reset()
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return obs, {}

        def step(self, action):
            """Realistic reward calculation with random signals for training"""
            # Extract action if it's a list (from VecEnv)
            if isinstance(action, (list, tuple, np.ndarray)):
                action = action[0] if len(action) > 0 else 0

            self.current_step += 1

            # Get current price
            if 'close' in self.data.columns:
                current_price = self.data.iloc[min(self.current_step, len(self.data)-1)]['close']
            else:
                current_price = 100 + np.random.randn() * 2

            reward = 0
            trade_pnl = 0

            # ✅ FIXED: Add synthetic signal for training when real signals missing
            # Generate random but trending price movement
            if self.current_step > 1:
                prev_price = self.data.iloc[min(self.current_step-1, len(self.data)-1)]['close'] if 'close' in self.data.columns else 100
                price_change = (current_price - prev_price) / prev_price
            else:
                price_change = 0

            # Process action
            if action == 1:  # Buy
                if self.position == 0:
                    self.position = 1
                    self.entry_price = current_price
                    reward = -0.005  # Small cost for entering
                    # print(f"   BUY at {current_price:.2f}")
            elif action == 2:  # Sell
                if self.position == 1:
                    trade_pnl = (current_price - self.entry_price) / self.entry_price
                    reward = trade_pnl * 10
                    self.position = 0
                    self.entry_price = 0
                    self.trade_count += 1
                    self._sharpe_calculator.add_trade(trade_pnl)
                    # print(f"   SELL at {current_price:.2f} | PnL: {trade_pnl:.2%}")
            else:  # Hold (action == 0)
                if self.position == 1:
                    # Unrealized PnL - gives feedback even without closing
                    unrealized_pnl = (current_price - self.entry_price) / self.entry_price
                    reward = unrealized_pnl * 0.3
                else:
                    # Small exploration incentive
                    reward = -0.0005

            # ✅ Add small random noise for exploration
            reward += np.random.randn() * 0.01

            terminated = self.current_step >= len(self.data) - 1
            truncated = False

            # Calculate Sharpe
            current_sharpe = self._sharpe_calculator.calculate_sharpe()

            info = {
                'balance': self.balance,
                'sharpe_ratio': current_sharpe,
                'position': self.position,
                'trade_result': {'success': trade_pnl > 0, 'pnl': trade_pnl} if trade_pnl != 0 else None,
                'trade_count': self.trade_count
            }

            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            # Add some market data to observation for variety
            obs[0] = price_change * 100  # price change %
            obs[1] = self.position  # current position
            obs[2] = reward  # last reward

            return obs, reward, terminated, truncated, info

# =========================================================
# ✅ ENSEMBLE PPO (Multiple models average decision - FULLY FIXED)
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
            """✅ FIXED: Returns action in proper format for VecEnv (list)"""
            if not self.models:
                return [0], None

            all_actions = []
            for model in self.models:
                action, _ = model.predict(observation, deterministic=deterministic)

                # Safe scalar extraction for any shape
                if isinstance(action, np.ndarray):
                    if action.size == 1:
                        action = int(action.item())
                    elif len(action.shape) == 1 and len(action) > 0:
                        action = int(action[0])
                    elif len(action.shape) == 2 and action.shape[0] > 0:
                        action = int(action[0, 0])
                    else:
                        action = 0
                elif isinstance(action, (list, tuple)) and len(action) > 0:
                    action = int(action[0])
                else:
                    action = int(action) if action is not None else 0

                all_actions.append(action)

            # Weighted majority voting - handle zero weights
            weighted_votes = {}
            for i, action in enumerate(all_actions):
                weight = self.weights[i] if i < len(self.weights) else 1.0/len(all_actions)
                weighted_votes[action] = weighted_votes.get(action, 0) + weight

            # If all weights are zero, use equal weights
            if max(weighted_votes.values()) == 0:
                for action in all_actions:
                    weighted_votes[action] = weighted_votes.get(action, 0) + 1.0/len(all_actions)

            final_action = int(max(weighted_votes, key=weighted_votes.get))

            # Return as list for VecEnv compatibility
            return [final_action], {'actions': all_actions, 'weights': self.weights, 'weighted_votes': weighted_votes}

        def save_ensemble(self, path):
            """Save ensemble metadata"""
            ensemble_info = {
                'model_paths': [str(p) for p in self.model_paths],
                'weights': self.weights,
                'created_at': datetime.now().isoformat()
            }
            joblib.dump(ensemble_info, path)

# =========================================================
# ✅ TRAIN WITH FINAL TEST PHASE (FULLY FIXED)
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
