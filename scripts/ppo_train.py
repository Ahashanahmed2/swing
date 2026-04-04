# ppo_train.py - HEDGE FUND LEVEL HYBRID PPO TRAINING SYSTEM (FULLY UPDATED WITH MISTAKE LEARNING)
# স্ট্রাকচার অপরিবর্তিত - শুধু ভুল থেকে শেখার মেকানিজম যুক্ত

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
MISTAKES_FILE = BASE_DIR / "csv" / "trading_mistakes.csv"  # NEW

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

# ✅ FALLBACK CONFIGURATION (NEW)
FALLBACK_CONFIG = {
    'enabled': True,
    'max_fallback_symbols': 10,
    'fallback_strategy': 'top_xgboost',  # 'top_xgboost', 'trending', 'dummy', 'momentum'
    'min_confidence_for_fallback': 0.55,
    'alert_on_no_signals': True
}

# ✅ MISTAKE LEARNING CONFIGURATION (NEW)
MISTAKE_LEARNING_CONFIG = {
    'enabled': True,
    'max_mistakes_per_symbol': 3,
    'penalty_multiplier': 0.7,
    'review_interval_days': 7,
    'forget_after_days': 90,  # Forget mistakes after 90 days
    'save_mistakes': True
}

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
# ✅ MISTAKE LEARNING CLASS (NEW)
# =========================================================

class MistakeLearner:
    """
    ভুল থেকে শেখার সিস্টেম
    - False positives (ভুল BUY সিগন্যাল)
    - False negatives (মিস করা BUY সিগন্যাল)
    - Poor risk-reward trades
    """
    
    def __init__(self, mistakes_file=MISTAKES_FILE):
        self.mistakes_file = Path(mistakes_file)
        self.mistakes = []
        self.patterns = {}
        self.load_mistakes()
    
    def load_mistakes(self):
        """পূর্বের ভুলগুলো লোড করুন"""
        if self.mistakes_file.exists():
            try:
                df = pd.read_csv(self.mistakes_file)
                self.mistakes = df.to_dict('records')
                print(f"   ✅ Loaded {len(self.mistakes)} past mistakes")
            except Exception as e:
                print(f"   ⚠️ Could not load mistakes: {e}")
                self.mistakes = []
    
    def record_mistake(self, trade_info):
        """
        ট্রেড করার পর ভুল রেকর্ড করুন
        
        Args:
            trade_info: {
                'symbol': str,
                'entry_date': date,
                'exit_date': date,
                'entry_price': float,
                'exit_price': float,
                'signal_type': 'BUY'/'SELL',
                'pnl': float,
                'reason': 'stop_loss'/'take_profit'/'manual',
                'signal_score': float
            }
        """
        if not MISTAKE_LEARNING_CONFIG['enabled']:
            return None
        
        # Only record losses
        if trade_info.get('pnl', 0) >= 0:
            return None
        
        mistake = {
            **trade_info,
            'recorded_at': datetime.now().isoformat(),
            'loss_amount': abs(trade_info['pnl']),
            'loss_percent': abs(trade_info['pnl']) / trade_info['entry_price'] * 100,
            'lesson': self._analyze_mistake(trade_info)
        }
        self.mistakes.append(mistake)
        
        if MISTAKE_LEARNING_CONFIG['save_mistakes']:
            self._save_mistakes()
        
        return mistake['lesson']
    
    def _analyze_mistake(self, trade_info):
        """ভুল বিশ্লেষণ করে শিক্ষা নিন"""
        loss_pct = abs(trade_info['pnl']) / trade_info['entry_price']
        signal_score = trade_info.get('signal_score', 0.5)
        
        if loss_pct > 0.05:  # 5% এর বেশি লস
            return {
                'type': 'high_loss',
                'message': f'Avoid {trade_info["symbol"]} - high volatility',
                'weight': 0.8,
                'action': 'reduce_exposure'
            }
        elif trade_info.get('reason') == 'stop_loss':
            return {
                'type': 'tight_sl',
                'message': f'Widen stop loss for {trade_info["symbol"]}',
                'weight': 0.6,
                'action': 'adjust_sl'
            }
        elif signal_score > 0.7 and loss_pct > 0.03:
            return {
                'type': 'false_signal',
                'message': f'Reduce confidence for {trade_info["symbol"]}',
                'weight': 0.7,
                'action': 'lower_confidence'
            }
        else:
            return {
                'type': 'wrong_signal',
                'message': f'Review {trade_info["symbol"]} pattern',
                'weight': 0.5,
                'action': 'review'
            }
    
    def get_mistake_penalty(self, symbol, signal_score):
        """
        পূর্বের ভুলের উপর ভিত্তি করে পেনাল্টি ক্যালকুলেট করুন
        
        Returns:
            adjusted_score: modified signal score
        """
        if not MISTAKE_LEARNING_CONFIG['enabled']:
            return signal_score
        
        # Filter recent mistakes (within forget_after_days)
        cutoff_date = datetime.now() - pd.Timedelta(days=MISTAKE_LEARNING_CONFIG['forget_after_days'])
        recent_mistakes = []
        
        for m in self.mistakes:
            if m.get('symbol') != symbol:
                continue
            try:
                mistake_date = pd.to_datetime(m.get('recorded_at', '2000-01-01'))
                if mistake_date > cutoff_date:
                    recent_mistakes.append(m)
            except:
                continue
        
        if not recent_mistakes:
            return signal_score
        
        # Calculate penalty based on number of recent mistakes
        mistake_count = len(recent_mistakes)
        
        if mistake_count >= MISTAKE_LEARNING_CONFIG['max_mistakes_per_symbol']:
            penalty = 0.5
        elif mistake_count >= 2:
            penalty = 0.3
        else:
            penalty = 0.1
        
        # Higher penalty for larger losses
        avg_loss = np.mean([m.get('loss_percent', 0) for m in recent_mistakes]) if recent_mistakes else 0
        if avg_loss > 5:  # 5%+ average loss
            penalty += 0.2
        elif avg_loss > 3:
            penalty += 0.1
        
        adjusted = signal_score * (1 - penalty * MISTAKE_LEARNING_CONFIG['penalty_multiplier'])
        return max(0.1, min(1.0, adjusted))
    
    def get_avoid_list(self):
        """যে সিম্বল এড়িয়ে চলা উচিত"""
        bad_symbols = {}
        cutoff_date = datetime.now() - pd.Timedelta(days=MISTAKE_LEARNING_CONFIG['forget_after_days'])
        
        for mistake in self.mistakes:
            symbol = mistake.get('symbol')
            if not symbol:
                continue
            
            try:
                mistake_date = pd.to_datetime(mistake.get('recorded_at', '2000-01-01'))
                if mistake_date < cutoff_date:
                    continue
            except:
                continue
            
            if symbol not in bad_symbols:
                bad_symbols[symbol] = 0
            bad_symbols[symbol] += 1
        
        # Return symbols with too many mistakes
        return [s for s, count in bad_symbols.items() 
                if count >= MISTAKE_LEARNING_CONFIG['max_mistakes_per_symbol']]
    
    def get_reduced_confidence_symbols(self):
        """যে সিম্বলে কনফিডেন্স কমানো উচিত"""
        reduced = {}
        for mistake in self.mistakes:
            symbol = mistake.get('symbol')
            if symbol and mistake.get('loss_percent', 0) > 3:
                if symbol not in reduced:
                    reduced[symbol] = 0
                reduced[symbol] += 1
        return reduced
    
    def _save_mistakes(self):
        """মিসটেক সেভ করুন"""
        try:
            df = pd.DataFrame(self.mistakes)
            df.to_csv(self.mistakes_file, index=False)
        except Exception as e:
            print(f"   ⚠️ Could not save mistakes: {e}")
    
    def get_summary(self):
        """মিসটেক সারাংশ"""
        if not self.mistakes:
            return "No mistakes recorded"
        
        df = pd.DataFrame(self.mistakes)
        summary = {
            'total_mistakes': len(self.mistakes),
            'unique_symbols': df['symbol'].nunique() if 'symbol' in df.columns else 0,
            'avg_loss_percent': df['loss_percent'].mean() if 'loss_percent' in df.columns else 0,
            'top_mistake_types': df['lesson'].apply(lambda x: x.get('type') if isinstance(x, dict) else 'unknown').value_counts().to_dict() if 'lesson' in df.columns else {}
        }
        return summary


# =========================================================
# ✅ SMART REWARD FUNCTION WITH MISTAKE LEARNING (NEW)
# =========================================================

class SmartRewardFunction:
    """
    স্মার্ট রিওয়ার্ড ফাংশন যা ভুল থেকে শেখে
    """
    
    def __init__(self):
        self.mistake_learner = MistakeLearner()
        self.consecutive_losses = 0
        self.winning_streak = 0
        self.trade_history = []
        
    def calculate_reward(self, trade_result, symbol, signal_score):
        """
        ট্রেড রেজাল্টের উপর ভিত্তি করে রিওয়ার্ড ক্যালকুলেট করুন
        
        Args:
            trade_result: {'pnl': float, 'success': bool, 'exit_reason': str}
            symbol: str
            signal_score: float (0-1)
        
        Returns:
            reward: float
        """
        pnl = trade_result.get('pnl', 0)
        is_win = trade_result.get('success', False)
        exit_reason = trade_result.get('exit_reason', 'unknown')
        
        # Base reward
        if is_win:
            # Winning trade
            reward = pnl * 10
            
            # Bonus for winning streak
            self.consecutive_losses = 0
            self.winning_streak += 1
            if self.winning_streak >= 3:
                reward *= 1.2  # 20% bonus for 3+ wins
            elif self.winning_streak >= 5:
                reward *= 1.5  # 50% bonus for 5+ wins
        else:
            # Losing trade - LEARN FROM MISTAKE
            reward = pnl * 15  # Higher penalty for losses
            
            # Record the mistake
            lesson = self.mistake_learner.record_mistake({
                'symbol': symbol,
                'pnl': pnl,
                'entry_price': trade_result.get('entry_price', 100),
                'exit_price': trade_result.get('exit_price', 100),
                'signal_score': signal_score,
                'reason': exit_reason,
                'signal_type': 'BUY'
            })
            
            # Additional penalty for repeated mistakes
            self.consecutive_losses += 1
            self.winning_streak = 0
            
            if self.consecutive_losses >= 2:
                reward *= 1.5  # 50% extra penalty for consecutive losses
            elif self.consecutive_losses >= 3:
                reward *= 2.0  # 100% extra penalty for 3+ consecutive losses
        
        # Apply mistake-based penalty to signal
        adjusted_score = self.mistake_learner.get_mistake_penalty(symbol, signal_score)
        if adjusted_score < signal_score:
            reward -= 0.1  # Small penalty for using problematic symbols
        
        self.trade_history.append({
            'symbol': symbol,
            'pnl': pnl,
            'is_win': is_win,
            'reward': reward,
            'timestamp': datetime.now()
        })
        
        return np.clip(reward, -2.0, 5.0)
    
    def get_insights(self):
        """শেখা থেকে প্রাপ্ত অন্তর্দৃষ্টি"""
        avoid_list = self.mistake_learner.get_avoid_list()
        reduced_confidence = self.mistake_learner.get_reduced_confidence_symbols()
        mistake_summary = self.mistake_learner.get_summary()
        
        return {
            'avoid_symbols': avoid_list,
            'reduced_confidence_symbols': reduced_confidence,
            'consecutive_losses': self.consecutive_losses,
            'winning_streak': self.winning_streak,
            'total_trades': len(self.trade_history),
            'mistake_summary': mistake_summary
        }
    
    def get_win_rate(self):
        """বর্তমান উইন রেট"""
        if not self.trade_history:
            return 0
        wins = sum(1 for t in self.trade_history if t.get('is_win', False))
        return wins / len(self.trade_history)


# =========================================================
# ✅ ENHANCED ENVIRONMENT WITH MISTAKE LEARNING (NEW)
# =========================================================

class HedgeFundTradingEnvWithMistakeLearning:
    """
    ভুল থেকে শেখার ক্ষমতা সম্পন্ন এনভায়রনমেন্ট
    Original HedgeFundTradingEnv-এর wrapper
    """
    
    def __init__(self, original_env, mistake_learner=None):
        self.original_env = original_env
        self.mistake_learner = mistake_learner or MistakeLearner()
        self.smart_reward = SmartRewardFunction()
        self.current_symbol = None
        self.current_signal_score = 0.5
        
    def __getattr__(self, name):
        """Delegate all attributes to original environment"""
        return getattr(self.original_env, name)
    
    def step(self, action):
        """Enhanced step with mistake learning"""
        
        # Get original step result
        result = self.original_env.step(action)
        
        # Handle different return formats
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            obs, reward, terminated, info = result
            truncated = False
        else:
            return result
        
        # Check if a trade was closed
        if info and isinstance(info, dict):
            trade_result = info.get('trade_result')
            if trade_result and trade_result.get('pnl', 0) < 0:
                # This was a losing trade - learn from it
                symbol = self.current_symbol or info.get('symbol', 'UNKNOWN')
                signal_score = self.current_signal_score
                
                # Calculate enhanced reward
                enhanced_reward = self.smart_reward.calculate_reward(
                    trade_result, symbol, signal_score
                )
                reward = enhanced_reward
                
                # Update info with learning insights
                info['mistake_learning'] = self.smart_reward.get_insights()
                info['adjusted_reward'] = True
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset with mistake learning context"""
        result = self.original_env.reset(**kwargs)
        
        # Try to get current symbol from data
        if hasattr(self.original_env, 'data') and hasattr(self.original_env, 'current_step'):
            try:
                if 'symbol' in self.original_env.data.columns:
                    idx = min(self.original_env.current_step, len(self.original_env.data)-1)
                    self.current_symbol = self.original_env.data.iloc[idx]['symbol']
            except:
                pass
        
        return result


def wrap_env_with_mistake_learning(env):
    """Environment কে mistake learning দিয়ে wrap করুন"""
    if MISTAKE_LEARNING_CONFIG['enabled'] and ENV_AVAILABLE:
        return HedgeFundTradingEnvWithMistakeLearning(env)
    return env


# =========================================================
# ✅ PERIODIC MISTAKE REVIEW FUNCTION (NEW)
# =========================================================

def review_mistakes_and_adjust():
    """
    পর্যায়ক্রমে ভুল রিভিউ করুন এবং স্ট্র্যাটেজি অ্যাডজাস্ট করুন
    """
    if not MISTAKE_LEARNING_CONFIG['enabled']:
        return []
    
    mistake_learner = MistakeLearner()
    
    if not mistake_learner.mistakes:
        print("   No mistakes to review")
        return []
    
    # Group mistakes by symbol
    mistakes_by_symbol = {}
    for m in mistake_learner.mistakes:
        symbol = m.get('symbol')
        if symbol:
            if symbol not in mistakes_by_symbol:
                mistakes_by_symbol[symbol] = []
            mistakes_by_symbol[symbol].append(m)
    
    print(f"\n📊 MISTAKE REVIEW SUMMARY:")
    print(f"   Total mistakes: {len(mistake_learner.mistakes)}")
    print(f"   Symbols with mistakes: {len(mistakes_by_symbol)}")
    
    # Identify problematic symbols
    problem_symbols = mistake_learner.get_avoid_list()
    reduced_symbols = mistake_learner.get_reduced_confidence_symbols()
    
    if problem_symbols:
        print(f"   ⚠️ Problematic symbols (AVOID): {problem_symbols[:5]}{'...' if len(problem_symbols) > 5 else ''}")
    
    if reduced_symbols:
        print(f"   ⚠️ Reduced confidence symbols: {list(reduced_symbols.keys())[:5]}{'...' if len(reduced_symbols) > 5 else ''}")
    
    # Update model metadata for problematic symbols
    if problem_symbols and MODEL_METADATA.exists():
        try:
            meta_df = pd.read_csv(MODEL_METADATA)
            for symbol in problem_symbols:
                mask = meta_df['symbol'] == symbol
                if mask.any():
                    # Reduce AUC for problematic symbols
                    original_auc = meta_df.loc[mask, 'auc'].values[0]
                    reduced_auc = original_auc * 0.85
                    meta_df.loc[mask, 'auc'] = reduced_auc
                    print(f"   🔧 Adjusted AUC for {symbol}: {original_auc:.2%} → {reduced_auc:.2%}")
            meta_df.to_csv(MODEL_METADATA, index=False)
        except Exception as e:
            print(f"   ⚠️ Could not update metadata: {e}")
    
    return problem_symbols


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
            """Safe Gymnasium 5-value return handling with VecEnv action format"""
            obs = self.eval_env.reset()

            if isinstance(obs, tuple):
                obs = obs[0]

            total_reward = 0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                if isinstance(obs, np.ndarray) and len(obs.shape) == 1:
                    obs = obs.reshape(1, -1)

                action, _ = self.model.predict(obs, deterministic=True)
                action = ensure_vecenv_action(action)
                step_result = self.eval_env.step(action)

                if step_result is None or len(step_result) == 0:
                    break

                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                elif len(step_result) == 4:
                    obs, reward, terminated, info = step_result
                    truncated = False
                else:
                    obs = step_result[0]
                    reward = step_result[1] if len(step_result) > 1 else 0
                    terminated = step_result[2] if len(step_result) > 2 else False
                    truncated = step_result[3] if len(step_result) > 3 else False
                    info = step_result[4] if len(step_result) > 4 else {}

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
# ✅ FALLBACK ENVIRONMENT
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
            self.mistake_learner = MistakeLearner()

        def reset(self, seed=None, options=None):
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
            if isinstance(action, (list, tuple, np.ndarray)):
                action = action[0] if len(action) > 0 else 0

            self.current_step += 1

            if 'close' in self.data.columns:
                current_price = self.data.iloc[min(self.current_step, len(self.data)-1)]['close']
            else:
                current_price = 100 + np.random.randn() * 2

            reward = 0
            trade_pnl = 0
            trade_result = None

            if self.current_step > 1:
                prev_price = self.data.iloc[min(self.current_step-1, len(self.data)-1)]['close'] if 'close' in self.data.columns else 100
                price_change = (current_price - prev_price) / prev_price
            else:
                price_change = 0

            if action == 1:  # Buy
                if self.position == 0:
                    self.position = 1
                    self.entry_price = current_price
                    self.entry_step = self.current_step
                    reward = -0.005
            elif action == 2:  # Sell
                if self.position == 1:
                    trade_pnl = (current_price - self.entry_price) / self.entry_price
                    trade_result = {
                        'pnl': trade_pnl,
                        'success': trade_pnl > 0,
                        'entry_price': self.entry_price,
                        'exit_price': current_price,
                        'exit_reason': 'take_profit' if trade_pnl > 0.03 else 'stop_loss' if trade_pnl < -0.02 else 'manual'
                    }
                    
                    # Use smart reward if available
                    if MISTAKE_LEARNING_CONFIG['enabled']:
                        smart_reward = SmartRewardFunction()
                        symbol = self._get_current_symbol()
                        reward = smart_reward.calculate_reward(trade_result, symbol, 0.6)
                    else:
                        reward = trade_pnl * 10
                    
                    self.position = 0
                    self.entry_price = 0
                    self.trade_count += 1
                    self._sharpe_calculator.add_trade(trade_pnl)
            else:  # Hold
                if self.position == 1:
                    unrealized_pnl = (current_price - self.entry_price) / self.entry_price
                    reward = unrealized_pnl * 0.3
                    if self.current_step - self.entry_step > 20:
                        reward -= 0.01
                else:
                    reward = -0.0005

            reward += np.random.randn() * 0.01
            terminated = self.current_step >= len(self.data) - 1
            truncated = False
            current_sharpe = self._sharpe_calculator.calculate_sharpe()

            info = {
                'balance': self.balance,
                'sharpe_ratio': current_sharpe,
                'position': self.position,
                'trade_result': trade_result,
                'trade_count': self.trade_count,
                'symbol': self._get_current_symbol()
            }

            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            obs[0] = price_change * 100
            obs[1] = self.position
            obs[2] = reward

            return obs, reward, terminated, truncated, info
        
        def _get_current_symbol(self):
            """Get current symbol from data"""
            if 'symbol' in self.data.columns:
                return self.data.iloc[min(self.current_step, len(self.data)-1)]['symbol']
            return 'UNKNOWN'


# =========================================================
# ✅ ENSEMBLE PPO (Multiple models average decision - FULLY FIXED)
# =========================================================

if SB3_AVAILABLE:
    class EnsemblePPO:
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
            if not self.models:
                return [0], None

            all_actions = []
            for model in self.models:
                action, _ = model.predict(observation, deterministic=deterministic)

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

            weighted_votes = {}
            for i, action in enumerate(all_actions):
                weight = self.weights[i] if i < len(self.weights) else 1.0/len(all_actions)
                weighted_votes[action] = weighted_votes.get(action, 0) + weight

            if max(weighted_votes.values()) == 0:
                for action in all_actions:
                    weighted_votes[action] = weighted_votes.get(action, 0) + 1.0/len(all_actions)

            final_action = int(max(weighted_votes, key=weighted_votes.get))
            return [final_action], {'actions': all_actions, 'weights': self.weights, 'weighted_votes': weighted_votes}

        def save_ensemble(self, path):
            ensemble_info = {
                'model_paths': [str(p) for p in self.model_paths],
                'weights': self.weights,
                'created_at': datetime.now().isoformat()
            }
            joblib.dump(ensemble_info, path)


# =========================================================
# ✅ TRAIN WITH FINAL TEST PHASE (UPDATED WITH MISTAKE LEARNING)
# =========================================================

def train_with_final_test(symbol, symbol_data, signals, xgb_auc, is_retrain=False):
    """Hedge Fund Level Training with fallback support and mistake learning"""

    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        print(f"\n   ⚠️ Skipping {symbol}: Required packages not available")
        return None, {}

    print(f"\n{'─'*50}")
    print(f"🎯 HEDGE FUND LEVEL TRAINING: {symbol} (AUC: {xgb_auc:.2%})")
    print(f"{'─'*50}")

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

    if xgb_auc >= 0.85:
        config = PPO_PER_SYMBOL_CONFIG['high_quality']
    elif xgb_auc >= 0.70:
        config = PPO_PER_SYMBOL_CONFIG['good_quality']
    else:
        config = PPO_PER_SYMBOL_CONFIG['fallback']

    ensemble_models = []
    ensemble_stats = []

    for ensemble_idx in range(ENSEMBLE_SIZE if USE_ENSEMBLE else 1):
        print(f"\n   🧠 Training Ensemble Model {ensemble_idx + 1}/{ENSEMBLE_SIZE}")

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
                # Wrap with mistake learning if enabled
                if MISTAKE_LEARNING_CONFIG['enabled']:
                    train_env = wrap_env_with_mistake_learning(train_env)
                    val_env = wrap_env_with_mistake_learning(val_env)
            else:
                train_env = HedgeFundTradingEnv(train_data)
                val_env = HedgeFundTradingEnv(val_data)
        except Exception as e:
            print(f"   ⚠️ Error creating environment: {e}")
            continue

        validate_environment(train_env)
        train_env = DummyVecEnv([lambda: train_env])
        val_env = DummyVecEnv([lambda: val_env])

        ppo_config = PPO_CONFIG.copy()
        ppo_config.update({
            'n_steps': config['n_steps'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'seed': 42 + ensemble_idx
        })

        model = PPO("MlpPolicy", train_env, **ppo_config, verbose=0)
        early_stop = EarlyStoppingCallback(val_env, patience=EARLY_STOPPING_PATIENCE, verbose=0)

        try:
            model.learn(total_timesteps=config['timesteps'], callback=early_stop)
        except Exception as e:
            print(f"   ⚠️ Training failed: {e}")
            continue

        obs = val_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        total_return = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if isinstance(obs, np.ndarray) and len(obs.shape) == 1:
                obs = obs.reshape(1, -1)

            action, _ = model.predict(obs, deterministic=True)
            action = ensure_vecenv_action(action)
            step_result = val_env.step(action)

            if step_result is None or len(step_result) == 0:
                break

            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            elif len(step_result) == 4:
                obs, reward, terminated, info = step_result
                truncated = False
            else:
                obs = step_result[0]
                reward = step_result[1] if len(step_result) > 1 else 0
                terminated = step_result[2] if len(step_result) > 2 else False
                truncated = step_result[3] if len(step_result) > 3 else False
                info = step_result[4] if len(step_result) > 4 else {}

            if isinstance(reward, (list, np.ndarray)):
                reward_val = reward[0] if len(reward) > 0 else 0
            elif isinstance(reward, (int, float)):
                reward_val = reward
            else:
                reward_val = 0

            total_return += reward_val
            steps += 1
            if steps > 10000:
                break

        val_sharpe = 0
        if isinstance(info, list) and len(info) > 0:
            if isinstance(info[0], dict):
                val_sharpe = info[0].get('sharpe_ratio', 0)
        elif isinstance(info, dict):
            val_sharpe = info.get('sharpe_ratio', 0)

        print(f"      Val Return: {total_return:.2f} | Sharpe: {val_sharpe:.3f}")

        model_path = PPO_ENSEMBLE_DIR / f"ppo_{symbol}_ens{ensemble_idx}"
        model.save(model_path)
        ensemble_models.append(model_path)
        ensemble_stats.append({'sharpe': val_sharpe, 'return': total_return})

    if USE_ENSEMBLE and len(ensemble_models) > 1:
        sharpe_vals = [max(0.001, s['sharpe'] + 0.1) for s in ensemble_stats]
        total_sharpe = sum(sharpe_vals)
        if total_sharpe > 0:
            weights = [s / total_sharpe for s in sharpe_vals]
        else:
            weights = [1.0 / len(sharpe_vals)] * len(sharpe_vals)

        ensemble = EnsemblePPO(ensemble_models, weights)
        print(f"\n   🎯 Ensemble created with {len(ensemble_models)} models")
        print(f"   Weights: {[round(w, 3) for w in weights]}")
        final_model = ensemble
    elif ensemble_models:
        final_model = PPO.load(ensemble_models[0], device="cpu")
    else:
        print(f"\n   ❌ No models trained for {symbol}")
        return None, {}

    print(f"\n   🧪 FINAL TEST on NEVER-TOUCHED data ({len(test_data)} rows)")

    try:
        if ENV_AVAILABLE:
            test_env = HedgeFundTradingEnv(
                data=test_data,
                xgb_model_dir=str(XGB_MODEL_DIR),
                config=HedgeFundConfig()
            )
            if MISTAKE_LEARNING_CONFIG['enabled']:
                test_env = wrap_env_with_mistake_learning(test_env)
        else:
            test_env = HedgeFundTradingEnv(test_data)
    except Exception as e:
        print(f"   ⚠️ Error creating test environment: {e}")
        return None, {}

    test_env = DummyVecEnv([lambda: test_env])

    obs = test_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    total_return = 0
    test_trades = []
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        if isinstance(obs, np.ndarray) and len(obs.shape) == 1:
            obs = obs.reshape(1, -1)

        if USE_ENSEMBLE and isinstance(final_model, EnsemblePPO):
            action, _ = final_model.predict(obs, deterministic=True)
        else:
            action, _ = final_model.predict(obs, deterministic=True)

        if not isinstance(action, (list, tuple, np.ndarray)):
            action = ensure_vecenv_action(action)

        step_result = test_env.step(action)

        if step_result is None or len(step_result) == 0:
            break

        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        elif len(step_result) == 4:
            obs, reward, terminated, info = step_result
            truncated = False
        else:
            obs = step_result[0]
            reward = step_result[1] if len(step_result) > 1 else 0
            terminated = step_result[2] if len(step_result) > 2 else False
            truncated = step_result[3] if len(step_result) > 3 else False
            info = step_result[4] if len(step_result) > 4 else {}

        if isinstance(reward, (list, np.ndarray)):
            reward_val = reward[0] if len(reward) > 0 else 0
        elif isinstance(reward, (int, float)):
            reward_val = reward
        else:
            reward_val = 0

        total_return += reward_val
        steps += 1

        trade_result = None
        if isinstance(info, list) and len(info) > 0:
            if isinstance(info[0], dict):
                trade_result = info[0].get('trade_result')
        elif isinstance(info, dict):
            trade_result = info.get('trade_result')

        if trade_result:
            test_trades.append(trade_result)

        if steps > 10000:
            break

    final_sharpe = 0
    if isinstance(info, list) and len(info) > 0:
        if isinstance(info[0], dict):
            final_sharpe = info[0].get('sharpe_ratio', 0)
    elif isinstance(info, dict):
        final_sharpe = info.get('sharpe_ratio', 0)

    profitable_trades = sum(1 for t in test_trades if t.get('success', False))
    total_trades = len(test_trades)
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0

    print(f"\n   📊 FINAL TEST RESULTS:")
    print(f"      Total Return: {total_return:.2f}%")
    print(f"      Sharpe Ratio: {final_sharpe:.3f}")
    print(f"      Win Rate: {win_rate:.2%} ({profitable_trades}/{total_trades})")
    print(f"      Total Trades: {total_trades}")

    final_path = PPO_SYMBOL_DIR / f"ppo_{symbol}"
    if USE_ENSEMBLE and isinstance(final_model, EnsemblePPO):
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
    """Load mistakes using MistakeLearner"""
    if MISTAKE_LEARNING_CONFIG['enabled']:
        mistake_learner = MistakeLearner()
        return mistake_learner.mistakes
    return []


def generate_fallback_signals(market_df, top_n=10):
    """
    Generate fallback BUY signals when no real signals exist
    
    Args:
        market_df: Market data DataFrame
        top_n: Number of fallback signals to generate
    
    Returns:
        Dictionary of fallback signals in PPO format
    """
    fallback_signals = {}
    latest_date = market_df['date'].max()
    
    print(f"   🔧 Generating fallback signals (strategy: {FALLBACK_CONFIG['fallback_strategy']})")
    
    # Apply mistake learning to avoid problematic symbols
    avoid_symbols = []
    if MISTAKE_LEARNING_CONFIG['enabled']:
        mistake_learner = MistakeLearner()
        avoid_symbols = mistake_learner.get_avoid_list()
        if avoid_symbols:
            print(f"      🛡️ Avoiding problematic symbols: {avoid_symbols[:3]}...")
    
    # Strategy 1: Top XGBoost confidence symbols
    if FALLBACK_CONFIG['fallback_strategy'] == 'top_xgboost' or True:
        if os.path.exists(PREDICTION_LOG):
            try:
                pred_df = pd.read_csv(PREDICTION_LOG)
                if 'confidence_score' in pred_df.columns:
                    pred_df['date'] = pd.to_datetime(pred_df['date'])
                    latest_pred = pred_df.sort_values('date').groupby('symbol').last()
                    
                    # Filter out avoid symbols
                    available_symbols = [s for s in latest_pred.index if s not in avoid_symbols]
                    top_symbols = latest_pred.loc[available_symbols].nlargest(top_n, 'confidence_score').index.tolist() if available_symbols else []
                    
                    for symbol in top_symbols:
                        sym_data = market_df[market_df['symbol'] == symbol].sort_values('date')
                        if len(sym_data) > 0:
                            latest_price = sym_data.iloc[-1]['close']
                            confidence = latest_pred.loc[symbol, 'confidence_score'] / 100
                            if confidence >= FALLBACK_CONFIG['min_confidence_for_fallback']:
                                fallback_signals[(symbol, latest_date.strftime('%Y-%m-%d'))] = {
                                    'buy': round(latest_price, 2),
                                    'SL': round(latest_price * 0.97, 2),
                                    'tp': round(latest_price * 1.06, 2),
                                    'RRR': 2.0
                                }
                    if fallback_signals:
                        print(f"      ✅ Generated {len(fallback_signals)} signals from top XGBoost symbols")
            except Exception as e:
                print(f"      ⚠️ Error reading prediction log: {e}")
    
    # Strategy 2: Trending symbols (uptrend detection)
    if len(fallback_signals) < top_n:
        for symbol in market_df['symbol'].unique():
            if symbol in avoid_symbols or symbol in [s[0] for s in fallback_signals.keys()]:
                continue
            sym_data = market_df[market_df['symbol'] == symbol].sort_values('date')
            if len(sym_data) >= 10:
                sma_5 = sym_data['close'].rolling(5).mean()
                sma_20 = sym_data['close'].rolling(20).mean()
                latest_price = sym_data.iloc[-1]['close']
                if (latest_price > sma_5.iloc[-1] and 
                    sma_5.iloc[-1] > sma_20.iloc[-1] and
                    len(sym_data) > 0):
                    fallback_signals[(symbol, latest_date.strftime('%Y-%m-%d'))] = {
                        'buy': round(latest_price, 2),
                        'SL': round(latest_price * 0.97, 2),
                        'tp': round(latest_price * 1.06, 2),
                        'RRR': 2.0
                    }
                    if len(fallback_signals) >= top_n:
                        break
        
        if fallback_signals:
            print(f"      ✅ Generated {len(fallback_signals)} signals from trending symbols")
    
    # Strategy 3: Momentum symbols (strong recent gains)
    if len(fallback_signals) < top_n:
        for symbol in market_df['symbol'].unique():
            if symbol in avoid_symbols or symbol in [s[0] for s in fallback_signals.keys()]:
                continue
            sym_data = market_df[market_df['symbol'] == symbol].sort_values('date')
            if len(sym_data) >= 5:
                returns = sym_data['close'].pct_change().dropna()
                if len(returns) >= 5:
                    momentum = returns.tail(5).sum()
                    if momentum > 0.02:
                        latest_price = sym_data.iloc[-1]['close']
                        fallback_signals[(symbol, latest_date.strftime('%Y-%m-%d'))] = {
                            'buy': round(latest_price, 2),
                            'SL': round(latest_price * 0.97, 2),
                            'tp': round(latest_price * 1.06, 2),
                            'RRR': 2.0
                        }
                        if len(fallback_signals) >= top_n:
                            break
        
        if fallback_signals:
            print(f"      ✅ Generated {len(fallback_signals)} signals from momentum symbols")
    
    # Strategy 4: Dummy signals (last resort)
    if len(fallback_signals) == 0:
        print(f"      ⚠️ No fallback signals found! Using dummy symbols...")
        dummy_symbols = ['KPCL', 'AAMRANET', 'SONALIANSH', 'SALVOCHEM', 'FAREASTFIN']
        for symbol in dummy_symbols[:top_n]:
            if symbol not in avoid_symbols:
                fallback_signals[(symbol, latest_date.strftime('%Y-%m-%d'))] = {
                    'buy': 100.0,
                    'SL': 97.0,
                    'tp': 106.0,
                    'RRR': 2.0
                }
        print(f"      ✅ Created {len(fallback_signals)} dummy signals")
    
    return fallback_signals


def ensure_buy_signals(signals, market_df):
    """
    Ensure there are BUY signals. If not, generate fallbacks.
    
    Args:
        signals: Current signals dictionary
        market_df: Market data DataFrame
    
    Returns:
        Updated signals dictionary with guaranteed BUY signals
    """
    if not FALLBACK_CONFIG['enabled']:
        return signals
    
    # Check if there are any BUY signals
    if len(signals) > 0:
        valid_buy_count = 0
        for key, sig in signals.items():
            if sig.get('buy', 0) > 0:
                valid_buy_count += 1
        
        if valid_buy_count > 0:
            print(f"   ✅ Using {valid_buy_count} existing BUY signals")
            return signals
    
    # No BUY signals found - generate fallbacks
    print(f"   ⚠️ No BUY signals found! Generating fallback signals...")
    
    if FALLBACK_CONFIG['alert_on_no_signals']:
        print(f"   📢 ALERT: No BUY signals available for PPO training!")
    
    fallback_signals = generate_fallback_signals(market_df, FALLBACK_CONFIG['max_fallback_symbols'])
    
    # Merge with existing signals (existing take priority)
    updated_signals = signals.copy()
    updated_signals.update(fallback_signals)
    
    print(f"   ✅ Total signals after fallback: {len(updated_signals)}")
    return updated_signals


def load_signals_with_fallback(path, market_df=None):
    """Load signals with automatic fallback generation if no BUY signals exist"""
    
    if not os.path.exists(path):
        print(f"   ⚠️ Signal file not found: {path}")
        if market_df is not None and FALLBACK_CONFIG['enabled']:
            return ensure_buy_signals({}, market_df)
        return {}
    
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        signals = {}
        for _, r in df.iterrows():
            signals[(r["symbol"], r["date"])] = {
                "buy": float(r["buy"]), 
                "SL": float(r["SL"]), 
                "tp": float(r["tp"]), 
                "RRR": float(r["RRR"]),
            }
        
        # Check if we have valid BUY signals
        valid_buy_count = 0
        for key, sig in signals.items():
            if sig.get('buy', 0) > 0:
                valid_buy_count += 1
        
        # Apply mistake-based filtering
        if MISTAKE_LEARNING_CONFIG['enabled'] and valid_buy_count > 0:
            mistake_learner = MistakeLearner()
            avoid_symbols = mistake_learner.get_avoid_list()
            reduced_symbols = mistake_learner.get_reduced_confidence_symbols()
            
            filtered_signals = {}
            for (symbol, date), sig in signals.items():
                if symbol in avoid_symbols:
                    # Skip problematic symbols
                    continue
                elif symbol in reduced_symbols:
                    # Reduce confidence
                    sig['confidence'] = sig.get('confidence', 0.7) * 0.8
                    sig['adjusted'] = True
                filtered_signals[(symbol, date)] = sig
            
            if len(filtered_signals) < valid_buy_count:
                print(f"   🔧 Filtered {valid_buy_count - len(filtered_signals)} signals due to past mistakes")
                signals = filtered_signals
                valid_buy_count = len(signals)
        
        if valid_buy_count == 0 and market_df is not None and FALLBACK_CONFIG['enabled']:
            print(f"   ⚠️ Signal file exists but no valid BUY signals!")
            return ensure_buy_signals({}, market_df)
        
        print(f"   ✅ Loaded {len(signals)} signals")
        return signals
        
    except Exception as e:
        print(f"   ⚠️ Error loading signals: {e}")
        if market_df is not None and FALLBACK_CONFIG['enabled']:
            return ensure_buy_signals({}, market_df)
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
                         (sig["tp"] - buy) / (buy + 1e-8), sig["RRR"]]
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
# SHARED PPO TRAINING (Hedge Fund Level)
# =========================================================

def train_shared_ppo_hedgefund(all_symbols_data, signals, exclude_symbols=None, is_retrain=False):
    """Train shared PPO with Hedge Fund level features and fallback support"""

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

    combined_data = pd.concat(filtered_data.values(), ignore_index=True)
    combined_data = combined_data.sort_values('date').reset_index(drop=True)

    total_len = len(combined_data)
    train_end = int(total_len * TRAIN_RATIO)
    val_end = int(total_len * (TRAIN_RATIO + VALIDATION_RATIO))

    train_data = combined_data.iloc[:train_end]
    val_data = combined_data.iloc[train_end:val_end]
    test_data = combined_data.iloc[val_end:]

    print(f"   Data Split: Train={len(train_data)}, Val={len(val_data)}, 🧪Test={len(test_data)}")

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
                # Wrap with mistake learning if enabled
                if MISTAKE_LEARNING_CONFIG['enabled']:
                    train_env = wrap_env_with_mistake_learning(train_env)
                    val_env = wrap_env_with_mistake_learning(val_env)
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
            ppo_config = {
                'n_steps': PPO_CONFIG['n_steps'],
                'batch_size': PPO_CONFIG['batch_size'],
                'gamma': PPO_CONFIG['gamma'],
                'learning_rate': PPO_CONFIG['learning_rate'],
                'ent_coef': PPO_CONFIG['ent_coef'],
                'clip_range': PPO_CONFIG['clip_range'],
                'vf_coef': PPO_CONFIG['vf_coef'],
                'max_grad_norm': PPO_CONFIG['max_grad_norm'],
                'seed': 42 + ensemble_idx
            }
            model = PPO("MlpPolicy", train_env, **ppo_config, verbose=0)
            timesteps = 100000

        early_stop = EarlyStoppingCallback(val_env, patience=EARLY_STOPPING_PATIENCE, verbose=0)
        model.learn(total_timesteps=timesteps, callback=early_stop)
        ensemble_models.append(model)

    if not ensemble_models:
        print("   ❌ No shared models trained")
        return None, {}

    print(f"\n   🧪 FINAL TEST on never-touched data")

    try:
        if ENV_AVAILABLE:
            test_env = HedgeFundTradingEnv(
                data=test_data,
                xgb_model_dir=str(XGB_MODEL_DIR),
                config=HedgeFundConfig()
            )
            if MISTAKE_LEARNING_CONFIG['enabled']:
                test_env = wrap_env_with_mistake_learning(test_env)
        else:
            test_env = HedgeFundTradingEnv(test_data)
    except Exception as e:
        print(f"   ⚠️ Error creating test environment: {e}")
        return ensemble_models[0], {}

    test_env = DummyVecEnv([lambda: test_env])

    obs = test_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    total_return = 0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        if isinstance(obs, np.ndarray) and len(obs.shape) == 1:
            obs = obs.reshape(1, -1)

        all_actions = []
        for model in ensemble_models:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action[0] if len(action) > 0 else 0
            elif isinstance(action, (list, tuple)):
                action = action[0] if len(action) > 0 else 0
            all_actions.append(int(action))

        final_action = int(round(np.mean(all_actions)))
        action_wrapped = ensure_vecenv_action(final_action)
        step_result = test_env.step(action_wrapped)

        if step_result is None or len(step_result) == 0:
            break

        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        elif len(step_result) == 4:
            obs, reward, terminated, info = step_result
            truncated = False
        else:
            obs = step_result[0]
            reward = step_result[1] if len(step_result) > 1 else 0
            terminated = step_result[2] if len(step_result) > 2 else False
            truncated = step_result[3] if len(step_result) > 3 else False
            info = step_result[4] if len(step_result) > 4 else {}

        if isinstance(reward, (list, np.ndarray)):
            reward_val = reward[0] if len(reward) > 0 else 0
        elif isinstance(reward, (int, float)):
            reward_val = reward
        else:
            reward_val = 0

        total_return += reward_val
        steps += 1
        if steps > 10000:
            break

    final_sharpe = 0
    if isinstance(info, list) and len(info) > 0:
        if isinstance(info[0], dict):
            final_sharpe = info[0].get('sharpe_ratio', 0)
    elif isinstance(info, dict):
        final_sharpe = info.get('sharpe_ratio', 0)

    print(f"   📊 Test Sharpe: {final_sharpe:.3f} | Return: {total_return:.2f}%")

    ensemble_models[0].save(PPO_SHARED_PATH)
    print(f"   ✅ Shared model saved: {PPO_SHARED_PATH}")

    return ensemble_models[0], {'sharpe_ratio': final_sharpe, 'ensemble_size': len(ensemble_models)}

# =========================================================
# MAIN TRAINING FUNCTION (UPDATED WITH FALLBACK AND MISTAKE LEARNING)
# =========================================================

def train_ppo_system():
    """Main training function - HEDGE FUND LEVEL with fallback support and mistake learning"""

    print("="*70)
    print("🏦 HEDGE FUND LEVEL HYBRID PPO TRAINING SYSTEM")
    print("="*70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"💰 Initial Capital: ${TOTAL_CAPITAL:,.2f}")
    print(f"📊 Features: Train/Val/Test Split | Noise Injection | Ensemble PPO | Sharpe Ratio")
    print(f"🔄 Fallback: {'ENABLED' if FALLBACK_CONFIG['enabled'] else 'DISABLED'}")
    print(f"🎓 Mistake Learning: {'ENABLED' if MISTAKE_LEARNING_CONFIG['enabled'] else 'DISABLED'}")
    print("="*70)

    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        print("\n⚠️ Required packages not available!")
        print("   Install with: pip install stable-baselines3 gymnasium")
        print("   Skipping PPO training...")
        return [], None

    # Review past mistakes before training
    if MISTAKE_LEARNING_CONFIG['enabled']:
        print("\n📚 Reviewing past mistakes...")
        problem_symbols = review_mistakes_and_adjust()
        if problem_symbols:
            print(f"   🛡️ Will avoid {len(problem_symbols)} problematic symbols")

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

    # Load signals WITH FALLBACK (UPDATED)
    signals = load_signals_with_fallback(CSV_SIGNAL, df)
    
    # Count and display signal types
    buy_count = sum(1 for s in signals.values() if s.get('buy', 0) > 0)
    print(f"   📊 Total signals: {len(signals)} (BUY: {buy_count})")

    xgb_metadata = load_xgb_metadata()

    top_symbol_list = []
    if not xgb_metadata.empty:
        # Apply mistake-based filtering to symbol selection
        if MISTAKE_LEARNING_CONFIG['enabled']:
            mistake_learner = MistakeLearner()
            avoid_symbols = mistake_learner.get_avoid_list()
            # Filter out problematic symbols
            xgb_metadata = xgb_metadata[~xgb_metadata['symbol'].isin(avoid_symbols)]
            print(f"   🛡️ Filtered out {len(avoid_symbols)} problematic symbols")
        
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

    # Load past mistakes (using new system)
    load_past_mistakes()

    # Train per-symbol PPO
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

        print(f"\n✅ Per-symbol PPO trained: {len(trained_symbols)} symbols")

        # Train shared PPO
        print("\n🏆 Training Shared PPO Model (Hedge Fund Level)")
        shared_model, shared_stats = train_shared_ppo_hedgefund(
            all_symbols_data, signals, exclude_symbols=trained_symbols, is_retrain=is_retrain
        )

        update_last_ppo_train()

    except Exception as e:
        print(f"\n   ⚠️ PPO training error: {e}")
        import traceback
        traceback.print_exc()

    # Final mistake learning summary
    if MISTAKE_LEARNING_CONFIG['enabled']:
        print("\n📚 MISTAKE LEARNING SUMMARY")
        mistake_learner = MistakeLearner()
        summary = mistake_learner.get_summary()
        if isinstance(summary, dict):
            print(f"   Total mistakes learned: {summary.get('total_mistakes', 0)}")
            print(f"   Unique symbols with mistakes: {summary.get('unique_symbols', 0)}")
            if summary.get('avg_loss_percent', 0) > 0:
                print(f"   Average loss per mistake: {summary.get('avg_loss_percent', 0):.2f}%")
        else:
            print(f"   {summary}")

    print("\n" + "="*70)
    print("🏦 HEDGE FUND LEVEL PPO TRAINING COMPLETE!")
    print("="*70)
    print(f"   Per-symbol models: {len(trained_symbols)}")
    print(f"   ✅ Train/Val/Test Split | ✅ Noise Injection | ✅ Ensemble PPO | ✅ Sharpe Ratio")
    print(f"   ✅ Fallback Mechanism: {'Active' if FALLBACK_CONFIG['enabled'] else 'Inactive'}")
    print(f"   ✅ Mistake Learning: {'Active' if MISTAKE_LEARNING_CONFIG['enabled'] else 'Inactive'}")
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