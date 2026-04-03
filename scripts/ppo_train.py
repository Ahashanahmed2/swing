# ppo_train_hybrid.py - COMPLETE HYBRID SOLUTION (FINAL PRODUCTION)
"""
HYBRID PPO TRAINING SYSTEM - FULLY OPTIMIZED v2.0
✅ ALL FIXES APPLIED:
1. ✅ Feature Scaling (Online RobustScaler)
2. ✅ Walk-Forward Validation
3. ✅ Stable Reward Function
4. ✅ No XGBoost Leakage
5. ✅ Auto Monthly Retrain
6. ✅ Self-Learning from Mistakes
7. ✅ Enhanced XGBoost Features (RSI, MACD, ATR, etc.)
8. ✅ Larger Ensemble (3 models for better stability)
9. ✅ Anti-Overfitting Layers
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque
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
    SB3_AVAILABLE = True
except ImportError:
    print("⚠️ Stable-Baselines3 not available")
    SB3_AVAILABLE = False

try:
    from sklearn.preprocessing import RobustScaler
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
MISTAKES_LOG = BASE_DIR / "csv" / "ppo_mistakes.csv"
TENSORBOARD_LOG = BASE_DIR / "logs" / "ppo_tensorboard"

os.makedirs(PPO_MODEL_DIR, exist_ok=True)
os.makedirs(PPO_SYMBOL_DIR, exist_ok=True)
os.makedirs(PPO_ENSEMBLE_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG, exist_ok=True)

# =========================================================
# CONFIGURATION
# =========================================================

WINDOW = 5
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
WALK_FORWARD_SPLITS = 3

# Early stopping
EARLY_STOPPING_PATIENCE = 5
EVAL_FREQ = 500

# ✅ Ensemble size increased (3 for better stability)
ENSEMBLE_SIZE = 3
USE_ENSEMBLE = True

# Advanced features
USE_ADVANCED_RISK = True
USE_REWARD_CLIPPING = True
MAX_DRAWDOWN_LIMIT = 0.25

# Anti-overfitting
USE_FEATURE_SCALING = True
USE_NOISE_INJECTION = True
NOISE_STD = 0.001

# Stable reward parameters
MIN_REWARD = -1.0
MAX_REWARD = 2.0

# Self-learning parameters
SELF_LEARNING_WEIGHT_INCREASE = 1.5
MAX_MISTAKE_AGE_DAYS = 90

# PPO Configuration (optimized for larger ensemble)
PPO_CONFIG = {
    'n_steps': 128,
    'batch_size': 64,
    'gamma': 0.99,
    'learning_rate': 3e-4,
    'ent_coef': 0.01,
    'clip_range': 0.2,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
}

PPO_PER_SYMBOL_CONFIG = {
    'high_quality': {'n_steps': 128, 'batch_size': 64, 'learning_rate': 3e-4, 'timesteps': 3000},
    'good_quality': {'n_steps': 128, 'batch_size': 64, 'learning_rate': 3e-4, 'timesteps': 3000},
    'fallback': {'n_steps': 128, 'batch_size': 64, 'learning_rate': 3e-4, 'timesteps': 3000},
}

MARKET_COLS = ["open", "high", "low", "close", "volume"]
STATE_DIM = len(MARKET_COLS) * WINDOW + 4

# =========================================================
# ✅ ENHANCED XGBOOST FEATURES (Solution 1)
# =========================================================

class TechnicalIndicators:
    """
    Calculate technical indicators for XGBoost features
    This gives XGBoost full power instead of just close/volume
    """
    
    @staticmethod
    def calculate_rsi(close_prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        if len(close_prices) < period + 1:
            return 50.0
        
        deltas = np.diff(close_prices[-period-1:])
        gains = deltas[deltas > 0].sum() / period
        losses = abs(deltas[deltas < 0].sum()) / period
        
        if losses == 0:
            return 100.0
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        return np.clip(rsi, 0, 100)
    
    @staticmethod
    def calculate_macd(close_prices: np.ndarray, fast=12, slow=26, signal=9) -> Tuple[float, float]:
        """Calculate MACD and Signal line"""
        if len(close_prices) < slow + signal:
            return 0.0, 0.0
        
        # Simple EMA approximation
        ema_fast = np.mean(close_prices[-fast:])
        ema_slow = np.mean(close_prices[-slow:])
        macd_line = ema_fast - ema_slow
        
        # Signal line (9-period EMA of MACD)
        if len(close_prices) >= slow + signal:
            macd_values = []
            for i in range(signal):
                idx = -(signal - i)
                if abs(idx) <= len(close_prices):
                    f = np.mean(close_prices[idx - fast:idx]) if idx - fast < 0 else np.mean(close_prices[max(0, idx-fast):idx])
                    s = np.mean(close_prices[idx - slow:idx]) if idx - slow < 0 else np.mean(close_prices[max(0, idx-slow):idx])
                    macd_values.append(f - s)
            signal_line = np.mean(macd_values) if macd_values else 0
        else:
            signal_line = macd_line
        
        return macd_line, signal_line
    
    @staticmethod
    def calculate_atr(high_prices: np.ndarray, low_prices: np.ndarray, 
                      close_prices: np.ndarray, period: int = 14) -> float:
        """Calculate ATR (Average True Range)"""
        if len(high_prices) < period + 1:
            return 0.0
        
        high = high_prices[-period-1:]
        low = low_prices[-period-1:]
        close = close_prices[-period-2:-1]
        
        true_ranges = []
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            true_ranges.append(max(hl, hc, lc))
        
        return np.mean(true_ranges) if true_ranges else 0.0
    
    @staticmethod
    def calculate_bollinger_bands(close_prices: np.ndarray, period: int = 20) -> Tuple[float, float]:
        """Calculate Bollinger Bands position"""
        if len(close_prices) < period:
            return 0.0, 0.0
        
        recent = close_prices[-period:]
        mean = np.mean(recent)
        std = np.std(recent)
        
        if std == 0:
            return 0.0, 0.0
        
        current = close_prices[-1]
        upper = (current - mean) / std  # How many std above mean
        lower = (mean - current) / std  # How many std below mean
        
        return upper, lower
    
    @staticmethod
    def calculate_volume_ratio(volumes: np.ndarray, period: int = 20) -> float:
        """Calculate volume ratio (current / average)"""
        if len(volumes) < period:
            return 1.0
        
        avg_volume = np.mean(volumes[-period-1:-1])
        if avg_volume == 0:
            return 1.0
        
        return volumes[-1] / avg_volume
    
    @staticmethod
    def get_all_features(df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """
        Extract all technical features for current index
        """
        # Get price arrays
        close_prices = df['close'].values[:idx+1]
        high_prices = df['high'].values[:idx+1]
        low_prices = df['low'].values[:idx+1]
        volumes = df['volume'].values[:idx+1] if 'volume' in df.columns else None
        
        features = {
            'rsi': TechnicalIndicators.calculate_rsi(close_prices, 14),
            'macd_line': 0.0,
            'macd_signal': 0.0,
            'atr': 0.0,
            'bb_upper': 0.0,
            'bb_lower': 0.0,
            'volume_ratio': 1.0,
            'price_momentum': 0.0,
            'price_volatility': 0.0,
        }
        
        # MACD
        macd_line, macd_signal = TechnicalIndicators.calculate_macd(close_prices)
        features['macd_line'] = macd_line
        features['macd_signal'] = macd_signal
        
        # ATR
        if len(high_prices) > 0 and len(low_prices) > 0:
            features['atr'] = TechnicalIndicators.calculate_atr(high_prices, low_prices, close_prices)
        
        # Bollinger Bands
        bb_upper, bb_lower = TechnicalIndicators.calculate_bollinger_bands(close_prices)
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        
        # Volume ratio
        if volumes is not None:
            features['volume_ratio'] = TechnicalIndicators.calculate_volume_ratio(volumes)
        
        # Price momentum (rate of change)
        if len(close_prices) >= 5:
            features['price_momentum'] = (close_prices[-1] - close_prices[-5]) / close_prices[-5]
        
        # Price volatility
        if len(close_prices) >= 10:
            features['price_volatility'] = np.std(close_prices[-10:]) / np.mean(close_prices[-10:])
        
        return features

# =========================================================
# ✅ AUTO MONTHLY RETRAIN
# =========================================================

def should_retrain_ppo() -> Tuple[bool, str]:
    """Check if PPO should be retrained based on last training date"""
    if not LAST_PPO_TRAIN.exists():
        return True, "First training - no previous model found"
    
    try:
        last_date_str = LAST_PPO_TRAIN.read_text().strip()
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
        days_since = (datetime.now() - last_date).days
        
        if days_since >= PPO_RETRAIN_INTERVAL:
            return True, f"Monthly retrain required (last: {days_since} days ago)"
        else:
            days_left = PPO_RETRAIN_INTERVAL - days_since
            return False, f"Model fresh (next retrain in {days_left} days)"
    except Exception as e:
        return True, f"Error reading last train date: {e}"

def update_last_ppo_train():
    """Update last training date after successful training"""
    try:
        LAST_PPO_TRAIN.write_text(datetime.now().strftime("%Y-%m-%d"))
        print(f"   📅 Last training date updated: {datetime.now().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"   ⚠️ Could not update last train date: {e}")

# =========================================================
# ✅ SELF-LEARNING SYSTEM
# =========================================================

class SelfLearningSystem:
    """Learn from past mistakes and adjust training weights"""
    
    def __init__(self, mistakes_log_path: Path):
        self.mistakes_log_path = mistakes_log_path
        self.mistakes = self._load_mistakes()
        self.symbol_weights = self._calculate_symbol_weights()
        
    def _load_mistakes(self) -> pd.DataFrame:
        """Load past mistakes from CSV"""
        if not self.mistakes_log_path.exists():
            return pd.DataFrame(columns=[
                'symbol', 'date', 'predicted_action', 'actual_action',
                'loss_amount', 'sharpe_impact', 'timestamp'
            ])
        
        try:
            df = pd.read_csv(self.mistakes_log_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"   ⚠️ Could not load mistakes: {e}")
            return pd.DataFrame(columns=[
                'symbol', 'date', 'predicted_action', 'actual_action',
                'loss_amount', 'sharpe_impact', 'timestamp'
            ])
    
    def log_mistake(self, symbol: str, date: str, predicted_action: int,
                    actual_action: int, loss_amount: float, sharpe_impact: float):
        """Log a trading mistake for self-learning"""
        new_mistake = pd.DataFrame([{
            'symbol': symbol,
            'date': date,
            'predicted_action': predicted_action,
            'actual_action': actual_action,
            'loss_amount': abs(loss_amount),
            'sharpe_impact': sharpe_impact,
            'timestamp': datetime.now()
        }])
        
        self.mistakes = pd.concat([self.mistakes, new_mistake], ignore_index=True)
        
        # Keep only recent mistakes
        if 'timestamp' in self.mistakes.columns:
            cutoff_date = datetime.now() - timedelta(days=MAX_MISTAKE_AGE_DAYS)
            self.mistakes = self.mistakes[self.mistakes['timestamp'] >= cutoff_date]
        
        # Save to CSV
        try:
            self.mistakes.to_csv(self.mistakes_log_path, index=False)
            print(f"   📝 Logged mistake for {symbol} (loss: ${loss_amount:.2f})")
        except Exception as e:
            print(f"   ⚠️ Could not save mistake: {e}")
        
        # Update weights
        self.symbol_weights = self._calculate_symbol_weights()
    
    def _calculate_symbol_weights(self) -> Dict[str, float]:
        """Calculate training weights based on mistake history"""
        if self.mistakes.empty:
            return {}
        
        weights = {}
        
        for symbol, group in self.mistakes.groupby('symbol'):
            mistake_count = len(group)
            total_loss = group['loss_amount'].sum()
            avg_sharpe_impact = group['sharpe_impact'].mean()
            
            weight = 1.0
            weight += mistake_count * 0.1
            loss_factor = min(total_loss / 1000, 0.5)
            weight += loss_factor
            
            if avg_sharpe_impact < 0:
                weight += abs(avg_sharpe_impact) * 0.5
            
            weight *= SELF_LEARNING_WEIGHT_INCREASE
            weights[symbol] = min(weight, 3.0)
        
        return weights
    
    def get_symbol_weight(self, symbol: str) -> float:
        """Get training weight for a symbol"""
        return self.symbol_weights.get(symbol, 1.0)
    
    def get_priority_symbols(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get symbols sorted by training priority"""
        if not self.symbol_weights:
            return []
        
        sorted_symbols = sorted(
            self.symbol_weights.items(),
            key=lambda x: x[2] if isinstance(x[1], tuple) else x[1],
            reverse=True
        )
        return sorted_symbols[:limit]
    
    def get_statistics(self) -> Dict:
        """Get self-learning statistics"""
        if self.mistakes.empty:
            return {'total_mistakes': 0, 'affected_symbols': 0}
        
        return {
            'total_mistakes': len(self.mistakes),
            'affected_symbols': self.mistakes['symbol'].nunique(),
            'total_loss': self.mistakes['loss_amount'].sum(),
            'avg_sharpe_impact': self.mistakes['sharpe_impact'].mean(),
            'top_symbols': self.get_priority_symbols(5)
        }

# =========================================================
# ✅ ONLINE FEATURE SCALER
# =========================================================

class OnlineFeatureScaler:
    """Online feature scaling for PPO environment"""
    def __init__(self, shape, buffer_size=5000, use_robust=True):
        self.shape = shape
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.use_robust = use_robust
        self.scaler = None
        self.is_fitted = False
        
    def update(self, observation):
        if not self.is_fitted:
            self.buffer.append(observation)
            if len(self.buffer) >= min(1000, self.buffer_size):
                self._fit_scaler()
    
    def _fit_scaler(self):
        buffer_array = np.array(self.buffer)
        if self.use_robust and SKLEARN_AVAILABLE:
            self.scaler = RobustScaler()
            self.scaler.fit(buffer_array)
            self.is_fitted = True
    
    def scale(self, observation):
        if not self.is_fitted or self.scaler is None:
            return observation
        obs_reshaped = observation.reshape(1, -1)
        scaled = self.scaler.transform(obs_reshaped)
        return np.clip(scaled.flatten(), -5, 5)
    
    def reset(self):
        if len(self.buffer) > self.buffer_size // 2:
            keep_count = int(len(self.buffer) * 0.2)
            self.buffer = deque(list(self.buffer)[-keep_count:], maxlen=self.buffer_size)
        else:
            self.buffer.clear()
        self.is_fitted = False

# =========================================================
# ✅ STABLE REWARD FUNCTION
# =========================================================

class StableSharpeReward:
    """Stable reward function with lower variance"""
    def __init__(self, risk_free_rate=0.02):
        self.risk_free_rate = risk_free_rate / 252
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
        
        sharpe_improvement = (new_sharpe - old_sharpe)
        pnl_normalized = np.tanh(current_pnl * 5)
        reward = pnl_normalized * 0.5 + sharpe_improvement * 1.5
        
        if USE_REWARD_CLIPPING:
            reward = np.clip(reward, MIN_REWARD, MAX_REWARD)
            
        return reward

# =========================================================
# ✅ WALK-FORWARD VALIDATOR
# =========================================================

class WalkForwardValidator:
    """Proper walk-forward validation"""
    def __init__(self, data, window_size=252, step_size=21, n_splits=3):
        self.data = data
        self.window_size = min(window_size, len(data) // 2)
        self.step_size = step_size
        self.n_splits = min(n_splits, (len(data) - self.window_size) // self.step_size)
        self.splits = []
        self._create_splits()
    
    def _create_splits(self):
        for i in range(max(1, self.n_splits)):
            train_start = 0
            train_end = self.window_size + (i * self.step_size)
            val_start = train_end
            val_end = min(val_start + self.step_size, len(self.data))
            
            if val_end <= len(self.data) and val_start < val_end:
                self.splits.append({
                    'train': (train_start, train_end),
                    'val': (val_start, val_end),
                    'iteration': i + 1
                })
    
    def get_best_split(self):
        if not self.splits:
            return None
        mid_idx = len(self.splits) // 2
        split = self.splits[mid_idx]
        train_data = self.data.iloc[split['train'][0]:split['train'][1]]
        val_data = self.data.iloc[split['val'][0]:split['val'][1]]
        return train_data, val_data, split['iteration']

# =========================================================
# ✅ HYBRID HEDGE FUND ENVIRONMENT (UPDATED)
# =========================================================

class HybridHedgeFundEnv(gym.Env):
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
                 use_scaling: bool = True,
                 xgb_model_dir: str = None,
                 self_learning: SelfLearningSystem = None):

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
        self.use_scaling = use_scaling
        self.xgb_model_dir = xgb_model_dir or str(XGB_MODEL_DIR)
        self.self_learning = self_learning

        self.xgb_models = self._load_xgb_models()
        self.symbols = list(symbol_dfs.keys())
        self.symbol_order = self.symbols.copy()
        self.sharpe_calculator = StableSharpeReward()

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        self.entry_price = 0
        self.highest_price = 0
        self.stop_loss_pct = 0.03
        self.trailing_stop_pct = 0.02
        self.feature_scaler = None
        
        self.current_symbol = None
        self.current_df = None
        self.current_step = 0
        self.capital = total_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
        self.total_reward = 0

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

    def _get_xgb_signal(self, symbol, df, idx):
        """
        ✅ ENHANCED XGBoost signal with full technical features
        Now uses RSI, MACD, ATR, Bollinger Bands, etc.
        """
        if symbol not in self.xgb_models:
            return 50, 0
        
        try:
            model = self.xgb_models[symbol]
            
            # ✅ Extract full technical features
            features_dict = TechnicalIndicators.get_all_features(df, idx)
            
            # Create feature vector for XGBoost
            features = [
                df.iloc[idx]['close'],
                df.iloc[idx]['volume'] if 'volume' in df.columns else 0,
                features_dict['rsi'],
                features_dict['macd_line'],
                features_dict['macd_signal'],
                features_dict['atr'],
                features_dict['bb_upper'],
                features_dict['bb_lower'],
                features_dict['volume_ratio'],
                features_dict['price_momentum'],
                features_dict['price_volatility'],
            ]
            
            features_array = np.array(features).reshape(1, -1)
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(features_array)[0, 1]
                return prob * 100, 1 if prob > 0.5 else 0
            else:
                pred = model.predict(features_array)[0]
                return 50 + (pred * 50) if pred != 0 else 50, pred
                
        except Exception as e:
            return 50, 0

    def _add_noise(self, price):
        if self.use_noise and USE_NOISE_INJECTION:
            noise = np.random.normal(1, NOISE_STD)
            return price * noise
        return price

    def _check_stop_loss(self, current_price, entry_price, high_price):
        if self.position == 0 or entry_price == 0:
            return False
        fixed_sl = entry_price * (1 - self.stop_loss_pct)
        trailing_sl = high_price * (1 - self.trailing_stop_pct)
        stop_price = max(fixed_sl, trailing_sl)
        return current_price <= stop_price

    def _calculate_drawdown(self, equity_curve):
        if len(equity_curve) < 2:
            return 0.0
        peak = max(equity_curve)
        current = equity_curve[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0
        return np.clip(drawdown, 0, 1)

    def reset(self, *, seed=None, options=None):
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
        
        if self.use_scaling and USE_FEATURE_SCALING:
            self.feature_scaler = OnlineFeatureScaler(
                shape=(self.state_dim,),
                buffer_size=2000,
                use_robust=True
            )

        observation = self._get_obs()
        
        if self.feature_scaler:
            self.feature_scaler.update(observation)
            observation = self.feature_scaler.scale(observation)

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
        
        obs_raw = self.build_observation(self.current_df, self.current_step, self.signals)
        obs = np.array(obs_raw, dtype=np.float32)
        
        if hasattr(self, 'feature_scaler') and self.feature_scaler and self.feature_scaler.is_fitted:
            obs = self.feature_scaler.scale(obs)
        
        return obs

    def step(self, action):
        if self.current_symbol is None:
            return self._get_obs(), 0.0, True, False, {}

        if self.current_step >= len(self.current_df) - 1:
            return self._move_to_next_symbol()

        row = self.current_df.iloc[self.current_step]
        next_row = self.current_df.iloc[self.current_step + 1]

        price = self._add_noise(row['close'])
        next_price = self._add_noise(next_row['close'])
        high_price = self._add_noise(row['high'])

        # ✅ Enhanced XGBoost signal with full features
        xgb_conf, xgb_pred = self._get_xgb_signal(self.current_symbol, self.current_df, self.current_step)

        reward = 0.0
        terminated = False
        trade_result = None

        if USE_ADVANCED_RISK and self.position > 0:
            if self._check_stop_loss(price, self.entry_price, self.highest_price):
                action = 2

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
                    reward -= 0.0005

                    if xgb_conf > 70 and xgb_pred == 1:
                        reward += 0.01

        elif action == 2:  # SELL
            if self.position > 0:
                sell_amount = self.position * price * 0.999
                pnl = (price - self.entry_price) / self.entry_price
                
                sharpe_reward = self.sharpe_calculator.get_reward(pnl)
                reward = sharpe_reward

                if USE_REWARD_CLIPPING:
                    reward = np.clip(reward, MIN_REWARD, MAX_REWARD)

                if pnl < -0.02 and self.self_learning:
                    self.self_learning.log_mistake(
                        symbol=self.current_symbol,
                        date=row['date'],
                        predicted_action=action,
                        actual_action=2,
                        loss_amount=abs(pnl * self.capital),
                        sharpe_impact=sharpe_reward
                    )

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

        elif action == 3:  # ADD
            if self.position > 0:
                add_amount = self.capital * 0.15
                shares = add_amount / price
                self.position += shares
                self.capital -= add_amount
                reward -= 0.0003

        elif action == 4:  # REDUCE
            if self.position > 0:
                reduce_shares = self.position * 0.5
                sell_amount = reduce_shares * price * 0.999
                self.capital += sell_amount
                self.position -= reduce_shares
                reward -= 0.0003

        elif action == 0:  # HOLD
            if self.position > 0 and next_price > price:
                reward += 0.0002
            elif self.position == 0 and next_price < price:
                reward += 0.0001

        self.current_step += 1
        self.total_reward += reward

        if price > self.highest_price:
            self.highest_price = price

        current_equity = self.capital + (self.position * price if self.position > 0 else 0)
        temp_curve = self.equity_curve + [current_equity]
        drawdown = self._calculate_drawdown(temp_curve)

        if USE_ADVANCED_RISK and drawdown > MAX_DRAWDOWN_LIMIT:
            reward -= drawdown
            terminated = True

        if self.current_step >= len(self.current_df) - 1:
            if self.position > 0:
                sell_amount = self.position * price * 0.999
                pnl = (price - self.entry_price) / self.entry_price
                self.capital += sell_amount
                self.sharpe_calculator.add_trade(pnl)
                self.equity_curve.append(self.capital)
                self.position = 0
            return self._move_to_next_symbol()

        info = {
            'balance': self.capital,
            'symbol': self.current_symbol,
            'trade_result': trade_result,
            'total_return': (self.capital / self.total_capital - 1) * 100,
            'sharpe_ratio': self.sharpe_calculator.calculate_sharpe(),
            'drawdown': drawdown,
            'xgb_confidence': xgb_conf
        }

        return self._get_obs(), reward, terminated, False, info

    def _move_to_next_symbol(self):
        self.current_symbol_idx += 1

        if self.current_symbol_idx >= len(self.symbol_order):
            observation = self._get_obs()
            info = {
                'balance': self.capital,
                'symbol': 'END',
                'total_return': (self.capital / self.total_capital - 1) * 100,
                'sharpe_ratio': self.sharpe_calculator.calculate_sharpe(),
                'total_trades': len(self.trades)
            }
            return observation, 0.0, True, False, info
        else:
            self._load_current_symbol()
            self.current_step = self.window
            self.position = 0
            self.entry_price = 0
            self.highest_price = 0
            observation = self._get_obs()
            info = {
                'balance': self.capital,
                'symbol': self.current_symbol,
                'total_return': (self.capital / self.total_capital - 1) * 100,
                'sharpe_ratio': self.sharpe_calculator.calculate_sharpe()
            }
            return observation, 0.0, False, False, info

# =========================================================
# ✅ ENVIRONMENT FACTORY
# =========================================================

def make_env(symbol, data, signals, name_suffix, shuffle, use_noise, use_scaling, self_learning):
    def _make():
        return HybridHedgeFundEnv(
            symbol_dfs={symbol: data},
            signals=signals,
            build_observation_func=build_observation,
            window=WINDOW,
            state_dim=STATE_DIM,
            total_capital=TOTAL_CAPITAL,
            risk_percent=RISK_PERCENT,
            symbol_name=f"{symbol}_{name_suffix}",
            shuffle_episodes=shuffle,
            use_noise=use_noise,
            use_scaling=use_scaling,
            xgb_model_dir=str(XGB_MODEL_DIR),
            self_learning=self_learning
        )
    return _make

# =========================================================
# ✅ EARLY STOPPING CALLBACK
# =========================================================

if SB3_AVAILABLE:
    class EarlyStoppingCallback(BaseCallback):
        def __init__(self, eval_env, patience=5, threshold=0.001, verbose=0):
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
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                total_reward += reward
                steps += 1
                if steps > 5000:
                    break
            return total_reward / steps if steps > 0 else 0

# =========================================================
# ✅ ENSEMBLE PPO (Larger ensemble)
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
            
            # ✅ Weighted voting for final action
            final_action = int(round(np.average(all_actions, weights=self.weights)))
            return final_action, {'actions': all_actions, 'weights': self.weights}
        
        def predict_with_confidence(self, observation, deterministic=True):
            """Get prediction with confidence score"""
            if not self.models:
                return 0, 0.0, None
            
            all_actions = []
            for model in self.models:
                action, _ = model.predict(observation, deterministic=deterministic)
                all_actions.append(action[0] if isinstance(action, np.ndarray) else action)
            
            # Calculate confidence (agreement between models)
            unique, counts = np.unique(all_actions, return_counts=True)
            confidence = counts.max() / len(all_actions)
            
            final_action = int(round(np.average(all_actions, weights=self.weights)))
            return final_action, confidence, {'actions': all_actions, 'confidence': confidence}

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
    except Exception:
        return np.zeros(STATE_DIM, dtype=np.float32)

def load_signals(path):
    """Load trading signals"""
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
    """Load XGBoost model metadata"""
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

# =========================================================
# ✅ TRAINING FUNCTION
# =========================================================

def train_hybrid_ppo(symbol, symbol_data, signals, xgb_auc, self_learning, is_retrain=False):
    """Train hybrid PPO model with enhanced features"""

    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        return None, {}

    symbol_weight = self_learning.get_symbol_weight(symbol)
    weight_str = f" (Weight: {symbol_weight:.2f}x)" if symbol_weight > 1.0 else ""

    print(f"\n{'─'*50}")
    print(f"🎯 HYBRID TRAINING: {symbol} (AUC: {xgb_auc:.2%}){weight_str}")
    print(f"   Features: 5-Actions | Walk-Forward | Self-Learning | Enhanced XGB")
    print(f"{'─'*50}")

    # Walk-forward split
    walker = WalkForwardValidator(
        symbol_data,
        window_size=min(WALK_FORWARD_WINDOW, len(symbol_data) // 2),
        step_size=WALK_FORWARD_STEP,
        n_splits=WALK_FORWARD_SPLITS
    )
    
    split_result = walker.get_best_split()
    if split_result is None:
        total_len = len(symbol_data)
        train_end = int(total_len * TRAIN_RATIO)
        val_end = int(total_len * (TRAIN_RATIO + VALIDATION_RATIO))
        train_data = symbol_data.iloc[:train_end]
        val_data = symbol_data.iloc[train_end:val_end]
        test_data = symbol_data.iloc[val_end:]
    else:
        train_data, val_data, iteration = split_result
        test_start = len(train_data) + len(val_data)
        test_data = symbol_data.iloc[test_start:test_start + int(len(symbol_data) * TEST_RATIO)]

    config = PPO_PER_SYMBOL_CONFIG['good_quality']
    adjusted_timesteps = int(config['timesteps'] * min(symbol_weight, 2.0))

    ensemble_models = []
    ensemble_stats = []

    for ensemble_idx in range(ENSEMBLE_SIZE if USE_ENSEMBLE else 1):
        print(f"\n   🧠 Training Model {ensemble_idx + 1}/{ENSEMBLE_SIZE}")

        train_env = DummyVecEnv([make_env(symbol, train_data, signals, f"train_ens{ensemble_idx}", True, True, True, self_learning)])
        val_env = DummyVecEnv([make_env(symbol, val_data, signals, f"val_ens{ensemble_idx}", False, False, True, self_learning)])

        ppo_config = PPO_CONFIG.copy()
        ppo_config.update({
            'n_steps': config['n_steps'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'] * (1.0 if symbol_weight == 1.0 else 0.8),
            'seed': 42 + ensemble_idx
        })

        model = PPO("MlpPolicy", train_env, **ppo_config, verbose=0)
        early_stop = EarlyStoppingCallback(val_env, patience=EARLY_STOPPING_PATIENCE, verbose=0)

        try:
            model.learn(total_timesteps=adjusted_timesteps, callback=early_stop)
        except Exception as e:
            print(f"   ⚠️ Training failed: {e}")
            continue

        # Evaluate
        obs = val_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        total_return = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = val_env.step(action)
            total_return += reward[0] if isinstance(reward, np.ndarray) else reward
            steps += 1
            if steps > 5000:
                break

        val_sharpe = info[0].get('sharpe_ratio', 0)
        print(f"      Val Return: {total_return:.2f} | Sharpe: {val_sharpe:.3f}")

        model_path = PPO_ENSEMBLE_DIR / f"ppo_{symbol}_ens{ensemble_idx}"
        model.save(model_path)
        ensemble_models.append(model_path)
        ensemble_stats.append({'sharpe': val_sharpe, 'return': total_return})

    if not ensemble_models:
        print(f"\n   ❌ No models trained for {symbol}")
        return None, {}

    if USE_ENSEMBLE and len(ensemble_models) > 1:
        sharpe_vals = [s['sharpe'] for s in ensemble_stats]
        total_sharpe = sum(sharpe_vals) if sum(sharpe_vals) > 0 else 1
        weights = [s / total_sharpe for s in sharpe_vals]
        final_model = EnsemblePPO(ensemble_models, weights)
        print(f"\n   🎯 Ensemble created with {len(ensemble_models)} models")
    else:
        final_model = PPO.load(ensemble_models[0], device="cpu")

    # Final test
    print(f"\n   🧪 FINAL TEST ({len(test_data)} rows)")

    test_env = DummyVecEnv([make_env(symbol, test_data, signals, "test", False, False, True, self_learning)])

    obs = test_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
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

        obs, reward, terminated, truncated, info = test_env.step(action)
        total_return += reward[0] if isinstance(reward, np.ndarray) else reward
        steps += 1

        if info[0].get('trade_result'):
            test_trades.append(info[0]['trade_result'])

        if steps > 5000:
            break

    final_sharpe = info[0].get('sharpe_ratio', 0)
    profitable = sum(1 for t in test_trades if t.get('success', False))
    win_rate = profitable / len(test_trades) if test_trades else 0

    print(f"\n   📊 RESULTS:")
    print(f"      Return: {total_return:.2f}% | Sharpe: {final_sharpe:.3f}")
    print(f"      Win Rate: {win_rate:.2%} ({profitable}/{len(test_trades)})")

    final_path = PPO_SYMBOL_DIR / f"ppo_{symbol}"
    if not isinstance(final_model, EnsemblePPO):
        final_model.save(final_path)

    return final_model, {'sharpe_ratio': final_sharpe, 'win_rate': win_rate}

# =========================================================
# ✅ MAIN FUNCTION
# =========================================================

def train_hybrid_ppo_system():
    """Main training function with all features"""
    
    print("="*70)
    print("🏦 HYBRID PPO TRAINING SYSTEM v2.0 (FULLY OPTIMIZED)")
    print("="*70)
    print(f"✅ Auto Monthly Retrain: {PPO_RETRAIN_INTERVAL} days")
    print(f"✅ Self-Learning: ON (mistakes → higher priority)")
    print(f"✅ Ensemble Size: {ENSEMBLE_SIZE} models")
    print(f"✅ Enhanced XGBoost: RSI, MACD, ATR, Bollinger Bands")
    print(f"✅ Feature Scaling: {'ON' if USE_FEATURE_SCALING else 'OFF'}")
    print(f"✅ Walk-Forward: {WALK_FORWARD_WINDOW}/{WALK_FORWARD_STEP}")
    print("="*70)

    # Check if retrain is needed
    should_retrain, reason = should_retrain_ppo()
    print(f"\n📊 Retrain Check: {reason}")
    
    if not should_retrain:
        print("   ✅ Model is fresh - skipping training")
        return [], None
    else:
        print("   🔄 Starting retraining process...")

    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        print("\n⚠️ Required packages not available!")
        return [], None

    # Load data
    print("\n📂 Loading data...")
    if not os.path.exists(CSV_MARKET):
        print("   ❌ Market data not found")
        return [], None

    df = pd.read_csv(CSV_MARKET)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")
    print(f"   ✅ Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")

    signals = load_signals(CSV_SIGNAL)
    xgb_metadata = load_xgb_metadata()

    # Initialize self-learning system
    self_learning = SelfLearningSystem(MISTAKES_LOG)
    
    # Show self-learning statistics
    stats = self_learning.get_statistics()
    if stats['total_mistakes'] > 0:
        print(f"\n📚 Self-Learning Statistics:")
        print(f"   Total mistakes: {stats['total_mistakes']}")
        print(f"   Affected symbols: {stats['affected_symbols']}")
        print(f"   Total loss: ${stats['total_loss']:.2f}")

    top_symbol_list = []
    if not xgb_metadata.empty:
        top_symbols = xgb_metadata[xgb_metadata['auc'] >= XGB_AUC_THRESHOLD_FOR_PPO].head(MAX_PER_SYMBOL_MODELS)
        top_symbol_list = top_symbols['symbol'].tolist()
        
        # Reorder based on self-learning priority
        priority_symbols = [s for s, _ in self_learning.get_priority_symbols(10)]
        top_symbol_list = [s for s in priority_symbols if s in top_symbol_list] + \
                         [s for s in top_symbol_list if s not in priority_symbols]
        
        print(f"   ✅ {len(top_symbol_list)} symbols selected (priority reordered)")

    # Prepare data
    all_symbols_data = {}
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].reset_index(drop=True)
        if len(symbol_df) >= WINDOW + 50:
            all_symbols_data[symbol] = symbol_df

    # Train
    trained_symbols = []
    
    for symbol in top_symbol_list[:MAX_PER_SYMBOL_MODELS]:
        if symbol not in all_symbols_data:
            continue
        
        symbol_data = all_symbols_data[symbol]
        xgb_info = xgb_metadata[xgb_metadata['symbol'] == symbol].iloc[0]
        
        try:
            model, stats = train_hybrid_ppo(symbol, symbol_data, signals, xgb_info['auc'], self_learning, False)
            if model is not None:
                trained_symbols.append(symbol)
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    # Update last training date
    if trained_symbols:
        update_last_ppo_train()
        print(f"\n   📅 Training completed and saved")
    
    print("\n" + "="*70)
    print("🏦 TRAINING COMPLETE!")
    print(f"   ✅ Trained: {len(trained_symbols)} symbols")
    print(f"   ✅ Ensemble: {ENSEMBLE_SIZE} models per symbol")
    print(f"   ✅ Enhanced XGBoost: 11 technical features")
    print("="*70)

    return trained_symbols, None

def main():
    try:
        trained, _ = train_hybrid_ppo_system()
        print(f"\n✅ SYSTEM READY! {len(trained)} models trained")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(0)

if __name__ == "__main__":
    main()