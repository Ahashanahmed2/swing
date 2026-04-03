# ppo_train_hybrid.py - COMPLETE FIXED HYBRID SOLUTION
"""
HYBRID PPO TRAINING SYSTEM - FULLY FIXED
✅ Fixed Gymnasium compatibility (5 return values)
✅ Fixed reset/step methods
✅ Fixed DummyVecEnv handling
✅ Fixed validation loops
✅ 5-action space (Hold/Buy/Sell/Add/Reduce)
✅ Dynamic XGBoost per-step prediction
✅ Professional risk management
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# ✅ IMPORTS WITH FALLBACK
# =========================================================

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    print("⚠️ gymnasium not available")
    GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_checker import check_env
    SB3_AVAILABLE = True
except ImportError:
    print("⚠️ Stable-Baselines3 not available")
    SB3_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn not available")
    SKLEARN_AVAILABLE = False

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
PPO_RETRAIN_INTERVAL = 30

XGB_AUC_THRESHOLD_FOR_PPO = 0.70
MAX_PER_SYMBOL_MODELS = 30

# Train/Test split
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Walk-forward parameters
WALK_FORWARD_WINDOW = 252
WALK_FORWARD_STEP = 21

# Early stopping
EARLY_STOPPING_PATIENCE = 10
EVAL_FREQ = 1000

# Ensemble
ENSEMBLE_SIZE = 3
USE_ENSEMBLE = True

# Advanced features
USE_ADVANCED_RISK = True
USE_REWARD_CLIPPING = True
MAX_DRAWDOWN_LIMIT = 0.25

# PPO Configuration
PPO_CONFIG = {
    'n_steps': 1024,
    'batch_size': 256,
    'gamma': 0.995,
    'learning_rate': 1e-4,
    'ent_coef': 0.001,
    'clip_range': 0.1,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'tensorboard_log': str(TENSORBOARD_LOG),
}

PPO_PER_SYMBOL_CONFIG = {
    'high_quality': {'n_steps': 2048, 'batch_size': 512, 'learning_rate': 2e-4, 'timesteps': 50000},
    'good_quality': {'n_steps': 1024, 'batch_size': 256, 'learning_rate': 1e-4, 'timesteps': 30000},
    'fallback': {'n_steps': 1024, 'batch_size': 256, 'learning_rate': 1e-4, 'timesteps': 20000},
}

# Market columns
MARKET_COLS = ["open", "high", "low", "close", "volume"]
STATE_DIM = len(MARKET_COLS) * WINDOW + 4

# =========================================================
# ✅ FEATURE SCALER
# =========================================================

class FeatureScaler:
    _instance = None
    _scaler = None
    
    def __new__(cls):
        if cls._instance is None and SKLEARN_AVAILABLE:
            cls._instance = super().__new__(cls)
            cls._scaler = StandardScaler()
        return cls._instance
    
    def fit(self, data):
        if self._scaler:
            self._scaler.fit(data)
    
    def transform(self, data):
        if self._scaler:
            return self._scaler.transform(data)
        return data
    
    def fit_transform(self, data):
        if self._scaler:
            return self._scaler.fit_transform(data)
        return data

# =========================================================
# ✅ SHARPE RATIO REWARD
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
        return np.clip(sharpe, -2, 2)

    def get_reward(self, current_pnl):
        if current_pnl is None:
            return 0.0
        old_sharpe = self.calculate_sharpe(self.returns[:-1]) if len(self.returns) > 1 else 0
        self.add_trade(current_pnl)
        new_sharpe = self.calculate_sharpe(self.returns)
        reward = (new_sharpe - old_sharpe) * 10
        if USE_REWARD_CLIPPING:
            reward = np.clip(reward, -1.0, 2.0)
        return reward

# =========================================================
# ✅ HYBRID HEDGE FUND ENVIRONMENT (FIXED)
# =========================================================

class HybridHedgeFundEnv(gym.Env):
    """
    Hybrid Environment - FULLY GYMNASIUM COMPLIANT
    Returns exactly 5 values from step() and 2 from reset()
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 symbol_dfs: Dict[str, pd.DataFrame],
                 signals: Dict,
                 build_observation_func,
                 window: int,
                 state_dim: int,
                 total_capital: float = 500_000,
                 risk_percent: float = 0.01,
                 symbol_name: str = "hybrid",
                 shuffle_episodes: bool = True,
                 use_noise: bool = True,
                 xgb_model_dir: str = None):
        
        super().__init__()
        
        self.symbol_dfs = symbol_dfs
        self.signals = signals
        self.build_observation = build_observation_func
        self.window = window
        self.state_dim = state_dim
        self.total_capital = total_capital
        self.risk_percent = risk_percent
        self.symbol_name = symbol_name
        self.shuffle_episodes = shuffle_episodes
        self.use_noise = use_noise
        self.xgb_model_dir = xgb_model_dir or str(XGB_MODEL_DIR)
        
        # Load XGBoost models
        self.xgb_models = self._load_xgb_models()
        
        self.symbols = list(symbol_dfs.keys())
        self.symbol_order = self.symbols.copy()
        self.sharpe_calculator = SharpeRatioReward()
        
        # 5-action space
        self.action_space = spaces.Discrete(5)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Risk management tracking
        self.entry_price = 0
        self.highest_price = 0
        self.stop_loss_pct = 0.03
        self.trailing_stop_pct = 0.02
        
        self.reset()
    
    def _load_xgb_models(self):
        models = {}
        if os.path.exists(self.xgb_model_dir):
            for file in os.listdir(self.xgb_model_dir):
                if file.endswith('.joblib'):
                    symbol = file.replace('.joblib', '')
                    try:
                        models[symbol] = joblib.load(os.path.join(self.xgb_model_dir, file))
                    except Exception:
                        pass
        return models
    
    def _get_xgb_signal(self, symbol, features_dict):
        if symbol in self.xgb_models:
            try:
                model = self.xgb_models[symbol]
                features = [
                    features_dict.get('close', 0),
                    features_dict.get('volume', 0),
                    features_dict.get('rsi', 50),
                    features_dict.get('macd', 0),
                ]
                features_array = np.array(features).reshape(1, -1)
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(features_array)[0, 1]
                else:
                    prob = 0.5
                return prob * 100, 1 if prob > 0.5 else 0
            except Exception:
                pass
        return 50, 0
    
    def _check_stop_loss(self, current_price, entry_price, high_price):
        if self.position == 0 or entry_price == 0:
            return False
        fixed_sl = entry_price * (1 - self.stop_loss_pct)
        trailing_sl = high_price * (1 - self.trailing_stop_pct)
        stop_price = max(fixed_sl, trailing_sl)
        return current_price <= stop_price
    
    def _add_noise(self, price):
        if self.use_noise:
            noise = np.random.normal(1, 0.001)
            return price * noise
        return price
    
    def _calculate_drawdown(self, equity_curve):
        if len(equity_curve) < 2:
            return 0.0
        peak = max(equity_curve)
        current = equity_curve[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0
        return np.clip(drawdown, 0, 1)
    
    def reset(self, seed=None, options=None):
        """Gymnasium reset - returns (obs, info)"""
        super().reset(seed=seed)
        
        if self.shuffle_episodes:
            self.symbol_order = np.random.permutation(self.symbols).tolist()
        else:
            self.symbol_order = self.symbols.copy()
        
        self.current_symbol_idx = 0
        self._load_current_symbol()
        self.current_step = self.window
        self.capital = self.total_capital
        self.position = 0
        self.entry_price = 0
        self.highest_price = 0
        self.total_reward = 0
        self.trades = []
        self.equity_curve = [self.capital]
        self.sharpe_calculator.reset()
        
        observation = self._get_obs()
        info = {
            'balance': self.capital,
            'symbol': self.current_symbol if self.current_symbol else 'NONE',
            'total_return': 0,
            'sharpe_ratio': 0,
            'drawdown': 0
        }
        
        return observation, info
    
    def _load_current_symbol(self):
        if self.current_symbol_idx < len(self.symbol_order):
            self.current_symbol = self.symbol_order[self.current_symbol_idx]
            self.current_df = self.symbol_dfs[self.current_symbol]
        else:
            self.current_symbol = None
            self.current_df = None
    
    def _get_obs(self):
        if self.current_symbol is None or self.current_step >= len(self.current_df):
            return np.zeros(self.state_dim, dtype=np.float32)
        
        obs = self.build_observation(self.current_df, self.current_step, self.signals)
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """
        Gymnasium step - returns (obs, reward, terminated, truncated, info)
        """
        # Handle case when no symbol is loaded
        if self.current_symbol is None:
            obs = self._get_obs()
            info = {'balance': self.capital, 'symbol': 'NONE', 'error': 'no_symbol'}
            return obs, 0.0, True, False, info
        
        # Handle end of data
        if self.current_step >= len(self.current_df) - 1:
            return self._move_to_next_symbol()
        
        row = self.current_df.iloc[self.current_step]
        next_row = self.current_df.iloc[self.current_step + 1]
        
        # Apply noise
        price = self._add_noise(row['close'])
        next_price = self._add_noise(next_row['close'])
        high_price = self._add_noise(row['high'])
        
        # Get dynamic XGBoost signal
        xgb_conf, xgb_pred = self._get_xgb_signal(self.current_symbol, row.to_dict())
        
        reward = 0.0
        terminated = False
        truncated = False
        trade_result = None
        
        # Check stop-loss
        if USE_ADVANCED_RISK and self.position > 0:
            if self._check_stop_loss(price, self.entry_price, self.highest_price):
                action = 2  # Force sell
        
        # Execute action (5-action space)
        if action == 1:  # BUY
            if self.position == 0:
                sig = self.signals.get((self.current_symbol, row['date']))
                buy_price = sig['buy'] if sig else None
                
                if buy_price and price <= buy_price * 1.02:
                    risk_amount = self.capital * self.risk_percent
                    shares = risk_amount / (price * 0.03)
                    shares = min(shares, self.capital / price)
                    self.position = shares
                    self.entry_price = price
                    self.highest_price = price
                    self.capital -= shares * price
                    reward -= 0.001
                    
                    if xgb_conf > 70 and xgb_pred == 1:
                        reward += 0.02
        
        elif action == 2:  # SELL
            if self.position > 0:
                sell_amount = self.position * price * 0.999
                pnl = (price - self.entry_price) / self.entry_price
                sharpe_reward = self.sharpe_calculator.get_reward(pnl)
                reward = pnl * 10 + sharpe_reward
                
                if USE_REWARD_CLIPPING:
                    reward = np.clip(reward, -1.0, 2.0)
                
                trade_result = {
                    'symbol': self.current_symbol,
                    'pnl': pnl,
                    'success': pnl > 0
                }
                self.trades.append(trade_result)
                self.capital += sell_amount
                self.position = 0
                self.entry_price = 0
                self.highest_price = 0
                self.equity_curve.append(self.capital)
        
        elif action == 3:  # ADD to position
            if self.position > 0:
                add_amount = self.capital * 0.15
                shares = add_amount / price
                self.position += shares
                self.capital -= add_amount
                reward -= 0.0005
                
                if xgb_conf > 80 and xgb_pred == 1:
                    reward += 0.01
        
        elif action == 4:  # REDUCE position
            if self.position > 0:
                reduce_shares = self.position * 0.5
                sell_amount = reduce_shares * price * 0.999
                self.capital += sell_amount
                self.position -= reduce_shares
                reward -= 0.0005
        
        # Hold action reward
        elif action == 0:
            if self.position > 0 and next_price > price:
                reward += 0.001 * (1 + (price - self.entry_price) / self.entry_price)
            elif self.position == 0 and next_price > price:
                reward += 0.0005
        
        self.current_step += 1
        self.total_reward += reward
        
        # Update highest price for trailing stop
        if price > self.highest_price:
            self.highest_price = price
        
        # Drawdown protection
        current_equity = self.capital + (self.position * price if self.position > 0 else 0)
        temp_curve = self.equity_curve + [current_equity]
        drawdown = self._calculate_drawdown(temp_curve)
        
        if USE_ADVANCED_RISK and drawdown > MAX_DRAWDOWN_LIMIT:
            reward -= drawdown * 2.0
            terminated = True
        
        info = {
            'balance': self.capital,
            'symbol': self.current_symbol,
            'trade_result': trade_result,
            'total_return': (self.capital / self.total_capital - 1) * 100,
            'sharpe_ratio': self.sharpe_calculator.calculate_sharpe(),
            'drawdown': drawdown,
            'xgb_confidence': xgb_conf
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _move_to_next_symbol(self):
        """Move to next symbol - returns 5 values"""
        self.current_symbol_idx += 1
        
        if self.current_symbol_idx >= len(self.symbol_order):
            # Episode finished
            observation = self._get_obs()
            reward = 0.0
            terminated = True
            truncated = False
            info = {
                'balance': self.capital,
                'symbol': 'END',
                'total_return': (self.capital / self.total_capital - 1) * 100,
                'sharpe_ratio': self.sharpe_calculator.calculate_sharpe(),
                'total_trades': len(self.trades)
            }
            return observation, reward, terminated, truncated, info
        else:
            # Move to next symbol
            self._load_current_symbol()
            self.current_step = self.window
            self.position = 0
            self.entry_price = 0
            self.highest_price = 0
            
            observation = self._get_obs()
            reward = 0.0
            terminated = False
            truncated = False
            info = {
                'balance': self.capital,
                'symbol': self.current_symbol,
                'total_return': (self.capital / self.total_capital - 1) * 100,
                'sharpe_ratio': self.sharpe_calculator.calculate_sharpe()
            }
            return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        if mode == 'human' and self.current_symbol:
            print(f"Symbol: {self.current_symbol}, Step: {self.current_step}, "
                  f"Capital: ${self.capital:,.2f}, Position: {self.position:.2f}")

# =========================================================
# ✅ EARLY STOPPING CALLBACK
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
                else:
                    self.no_improvement_count += 1
                if self.no_improvement_count >= self.patience:
                    return False
            return True
        
        def _evaluate(self):
            obs, _ = self.eval_env.reset()
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
# ✅ ENSEMBLE PPO
# =========================================================

if SB3_AVAILABLE:
    class EnsemblePPO:
        def __init__(self, model_paths, weights=None):
            self.models = []
            self.weights = weights if weights else [1.0 / len(model_paths)] * len(model_paths)
            for path in model_paths:
                try:
                    self.models.append(PPO.load(path, device="cpu"))
                except Exception as e:
                    print(f"   ⚠️ Failed to load {path}: {e}")
        
        def predict(self, observation, deterministic=True):
            if not self.models:
                return 0, None
            all_actions = []
            for model in self.models:
                action, _ = model.predict(observation, deterministic=deterministic)
                all_actions.append(action[0] if isinstance(action, np.ndarray) else action)
            final_action = int(round(np.average(all_actions, weights=self.weights)))
            return final_action, {'actions': all_actions, 'weights': self.weights}

# =========================================================
# ✅ UTILITY FUNCTIONS
# =========================================================

def build_observation(df, idx, signals):
    """Build observation vector for environment"""
    try:
        available_cols = [col for col in MARKET_COLS if col in df.columns]
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
            signal_vec = [
                row["close"] / (buy + 1e-8), 
                (buy - sig["SL"]) / (buy + 1e-8),
                (sig["TP"] - buy) / (buy + 1e-8), 
                sig["RRR"]
            ]
        else:
            signal_vec = [0.0] * 4
        
        obs = list(market_vec) + signal_vec
        return np.nan_to_num(obs)
    except Exception as e:
        return np.zeros(STATE_DIM, dtype=np.float32)

def load_signals(path):
    if not os.path.exists(path):
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
        return signals
    except Exception:
        return {}

def load_xgb_metadata():
    if not os.path.exists(MODEL_METADATA):
        return pd.DataFrame()
    try:
        df = pd.read_csv(MODEL_METADATA)
        if 'status' in df.columns and 'auc' in df.columns:
            good_models = df[df['status'] == 'GOOD'].copy()
            good_models = good_models.sort_values('auc', ascending=False)
            return good_models
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

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
# ✅ TRAINING FUNCTION (FIXED)
# =========================================================

def train_hybrid_ppo(symbol, symbol_data, signals, xgb_auc, is_retrain=False):
    """Train hybrid PPO model with fixed Gymnasium compatibility"""
    
    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        print(f"\n   ⚠️ Skipping {symbol}: Required packages not available")
        return None, {}
    
    print(f"\n{'─'*50}")
    print(f"🎯 HYBRID TRAINING: {symbol} (AUC: {xgb_auc:.2%})")
    print(f"   Features: 5-Actions | Stop-Loss | Dynamic XGBoost")
    print(f"{'─'*50}")
    
    # Data split
    total_len = len(symbol_data)
    train_end = int(total_len * TRAIN_RATIO)
    val_end = int(total_len * (TRAIN_RATIO + VALIDATION_RATIO))
    
    train_data = symbol_data.iloc[:train_end]
    val_data = symbol_data.iloc[train_end:val_end]
    test_data = symbol_data.iloc[val_end:]
    
    print(f"   📊 Data: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Select config based on XGBoost quality
    if xgb_auc >= 0.85:
        config = PPO_PER_SYMBOL_CONFIG['high_quality']
    elif xgb_auc >= 0.70:
        config = PPO_PER_SYMBOL_CONFIG['good_quality']
    else:
        config = PPO_PER_SYMBOL_CONFIG['fallback']
    
    # Train ensemble models
    ensemble_models = []
    ensemble_stats = []
    
    for ensemble_idx in range(ENSEMBLE_SIZE if USE_ENSEMBLE else 1):
        print(f"\n   🧠 Training Model {ensemble_idx + 1}/{ENSEMBLE_SIZE}")
        
        # Create environments
        train_dfs = {symbol: train_data}
        val_dfs = {symbol: val_data}
        
        train_env_raw = HybridHedgeFundEnv(
            symbol_dfs=train_dfs,
            signals=signals,
            build_observation_func=build_observation,
            window=WINDOW,
            state_dim=STATE_DIM,
            total_capital=TOTAL_CAPITAL,
            risk_percent=RISK_PERCENT,
            symbol_name=f"{symbol}_train_ens{ensemble_idx}",
            shuffle_episodes=True,
            use_noise=True,
            xgb_model_dir=str(XGB_MODEL_DIR)
        )
        
        val_env_raw = HybridHedgeFundEnv(
            symbol_dfs=val_dfs,
            signals=signals,
            build_observation_func=build_observation,
            window=WINDOW,
            state_dim=STATE_DIM,
            total_capital=TOTAL_CAPITAL,
            risk_percent=RISK_PERCENT,
            symbol_name=f"{symbol}_val_ens{ensemble_idx}",
            shuffle_episodes=False,
            use_noise=False,
            xgb_model_dir=str(XGB_MODEL_DIR)
        )
        
        # Wrap for SB3
        train_env = DummyVecEnv([lambda: train_env_raw])
        val_env = DummyVecEnv([lambda: val_env_raw])
        
        ppo_config = PPO_CONFIG.copy()
        ppo_config.update({
            'n_steps': config['n_steps'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'seed': 42 + ensemble_idx
        })
        
        try:
            model = PPO("MlpPolicy", train_env, **ppo_config, verbose=0)
            early_stop = EarlyStoppingCallback(val_env, patience=EARLY_STOPPING_PATIENCE, verbose=0)
            model.learn(total_timesteps=config['timesteps'], callback=early_stop)
        except Exception as e:
            print(f"   ⚠️ Training failed: {e}")
            continue
        
        # Evaluate on validation
        obs = val_env.reset()
        total_return = 0
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action_val = action[0] if isinstance(action, np.ndarray) else action
            obs, reward, terminated, truncated, info = val_env.step([action_val])
            reward_val = reward[0] if isinstance(reward, np.ndarray) else reward
            total_return += reward_val
            steps += 1
            if steps > 10000:
                break
        
        # Extract sharpe from info
        if isinstance(info, list) and len(info) > 0:
            val_sharpe = info[0].get('sharpe_ratio', 0)
        else:
            val_sharpe = 0
        
        print(f"      Val Return: {total_return:.2f} | Sharpe: {val_sharpe:.3f}")
        
        model_path = PPO_ENSEMBLE_DIR / f"ppo_{symbol}_ens{ensemble_idx}"
        model.save(model_path)
        ensemble_models.append(model_path)
        ensemble_stats.append({'sharpe': val_sharpe, 'return': total_return})
    
    if not ensemble_models:
        print(f"\n   ❌ No models trained for {symbol}")
        return None, {}
    
    # Create ensemble
    if USE_ENSEMBLE and len(ensemble_models) > 1:
        sharpe_vals = [max(0.1, s['sharpe']) for s in ensemble_stats]
        total_sharpe = sum(sharpe_vals)
        weights = [s / total_sharpe for s in sharpe_vals]
        final_model = EnsemblePPO(ensemble_models, weights)
        print(f"\n   🎯 Ensemble created with {len(ensemble_models)} models")
    else:
        final_model = PPO.load(ensemble_models[0], device="cpu")
    
    # FINAL TEST on never-touched data
    print(f"\n   🧪 FINAL TEST on test data ({len(test_data)} rows)")
    
    test_dfs = {symbol: test_data}
    test_env_raw = HybridHedgeFundEnv(
        symbol_dfs=test_dfs,
        signals=signals,
        build_observation_func=build_observation,
        window=WINDOW,
        state_dim=STATE_DIM,
        total_capital=TOTAL_CAPITAL,
        risk_percent=RISK_PERCENT,
        symbol_name=f"{symbol}_test",
        shuffle_episodes=False,
        use_noise=False,
        xgb_model_dir=str(XGB_MODEL_DIR)
    )
    test_env = DummyVecEnv([lambda: test_env_raw])
    
    obs = test_env.reset()
    total_return = 0
    test_trades = []
    steps = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        if isinstance(final_model, EnsemblePPO):
            action, _ = final_model.predict(obs, deterministic=True)
        else:
            action, _ = final_model.predict(obs, deterministic=True)
            action = action[0] if isinstance(action, np.ndarray) else action
        
        obs, reward, terminated, truncated, info = test_env.step([action])
        reward_val = reward[0] if isinstance(reward, np.ndarray) else reward
        total_return += reward_val
        steps += 1
        
        if isinstance(info, list) and len(info) > 0 and info[0].get('trade_result'):
            test_trades.append(info[0]['trade_result'])
        
        if steps > 10000:
            break
    
    # Extract final metrics
    if isinstance(info, list) and len(info) > 0:
        final_sharpe = info[0].get('sharpe_ratio', 0)
    else:
        final_sharpe = 0
    
    profitable_trades = sum(1 for t in test_trades if t.get('success', False))
    total_trades = len(test_trades)
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    
    print(f"\n   📊 FINAL TEST RESULTS:")
    print(f"      Return: {total_return:.2f}% | Sharpe: {final_sharpe:.3f}")
    print(f"      Win Rate: {win_rate:.2%} ({profitable_trades}/{total_trades})")
    
    # Save model
    final_path = PPO_SYMBOL_DIR / f"ppo_{symbol}"
    if not isinstance(final_model, EnsemblePPO):
        final_model.save(final_path)
    
    return final_model, {
        'sharpe_ratio': final_sharpe,
        'test_return': total_return,
        'win_rate': win_rate,
        'ensemble_size': len(ensemble_models)
    }

# =========================================================
# ✅ MAIN FUNCTION
# =========================================================

def main():
    print("="*70)
    print("🏦 HYBRID PPO TRAINING SYSTEM (FIXED)")
    print("="*70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"💰 Capital: ${TOTAL_CAPITAL:,.2f}")
    print(f"🎯 Actions: 5 (Hold/Buy/Sell/Add/Reduce)")
    print(f"🛡️ Risk: Stop-Loss | Trailing Stop | Drawdown Protection")
    print("="*70)
    
    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        print("\n⚠️ Required packages not available!")
        return
    
    # Load data
    if not os.path.exists(CSV_MARKET):
        print(f"\n❌ Market data not found")
        return
    
    df = pd.read_csv(CSV_MARKET)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")
    
    signals = load_signals(CSV_SIGNAL)
    xgb_metadata = load_xgb_metadata()
    
    # Select top symbols
    if xgb_metadata.empty:
        print("\n⚠️ No XGBoost models found")
        return
    
    top_symbols = xgb_metadata[xgb_metadata['auc'] >= XGB_AUC_THRESHOLD_FOR_PPO].head(MAX_PER_SYMBOL_MODELS)
    top_symbol_list = top_symbols['symbol'].tolist()
    print(f"\n✅ Selected {len(top_symbol_list)} symbols for training")
    
    # Prepare data
    all_symbols_data = {}
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].reset_index(drop=True)
        if len(symbol_df) >= WINDOW + 50:
            all_symbols_data[symbol] = symbol_df
    
    # Train per-symbol
    trained_symbols = []
    
    for symbol in top_symbol_list[:MAX_PER_SYMBOL_MODELS]:
        if symbol not in all_symbols_data:
            continue
        
        symbol_data = all_symbols_data[symbol]
        xgb_info = top_symbols[top_symbols['symbol'] == symbol].iloc[0]
        
        try:
            model, stats = train_hybrid_ppo(
                symbol, symbol_data, signals, xgb_info['auc'], False
            )
            if model is not None:
                trained_symbols.append(symbol)
        except Exception as e:
            print(f"\n   ❌ Failed: {e}")
    
    print("\n" + "="*70)
    print(f"🏦 TRAINING COMPLETE! ✅ Trained: {len(trained_symbols)} symbols")
    print("="*70)

if __name__ == "__main__":
    main()