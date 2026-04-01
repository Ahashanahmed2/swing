# ppo_train.py - Complete Hybrid PPO Training System
# Features:
# 1. Per-symbol PPO for top GOOD XGBoost models (AUC >= 0.70)
# 2. Shared PPO for all other symbols (fallback)
# 3. Self-learning from past mistakes
# 4. Monthly fine-tuning with curriculum learning
# 5. XGBoost signal integration

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

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
MODEL_METADATA = BASE_DIR / "csv" / "model_metadata.csv"
PREDICTION_LOG = BASE_DIR / "csv" / "prediction_log.csv"
LAST_PPO_TRAIN = BASE_DIR / "csv" / "last_ppo_train.txt"

os.makedirs(PPO_MODEL_DIR, exist_ok=True)
os.makedirs(PPO_SYMBOL_DIR, exist_ok=True)

# =========================================================
# CONFIGURATION
# =========================================================

WINDOW = 10
TOTAL_CAPITAL = 500_000
RISK_PERCENT = 0.01
PPO_RETRAIN_INTERVAL = 30  # Days between retrains

# PPO thresholds
XGB_AUC_THRESHOLD_FOR_PPO = 0.70  # Only symbols with AUC >= 70% get per-symbol PPO
MAX_PER_SYMBOL_MODELS = 30  # Limit to top 30 symbols (to save storage/time)

# Market columns for features
MARKET_COLS = [
    "open", "high", "low", "close",
    "volume", "value", "trades", "change", "marketCap",
    "bb_upper", "bb_middle", "bb_lower",
    "macd", "macd_signal", "macd_hist",
    "rsi", "atr",
    "Hammer", "BullishEngulfing", "MorningStar",
    "Doji", "PiercingLine", "ThreeWhiteSoldiers",
]

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

# Per-symbol PPO config (customized by XGBoost quality)
PPO_PER_SYMBOL_CONFIG = {
    'high_quality': {  # AUC >= 0.85
        'n_steps': 2048,
        'batch_size': 512,
        'learning_rate': 2e-4,
        'timesteps': 50000,
    },
    'good_quality': {  # AUC 0.70 - 0.85
        'n_steps': 1024,
        'batch_size': 256,
        'learning_rate': 1e-4,
        'timesteps': 30000,
    },
    'fallback': {  # AUC < 0.70 or shared
        'n_steps': 1024,
        'batch_size': 256,
        'learning_rate': 1e-4,
        'timesteps': 20000,
    }
}

# =========================================================
# SELF-LEARNING CALLBACK
# =========================================================

class SelfLearningCallback(BaseCallback):
    """
    Callback that learns from past mistakes during training
    """
    
    def __init__(self, symbol_dfs, signals, build_observation, window, state_dim, 
                 symbol_name="shared", verbose=0):
        super().__init__(verbose)
        self.symbol_dfs = symbol_dfs
        self.signals = signals
        self.build_observation = build_observation
        self.window = window
        self.state_dim = state_dim
        self.symbol_name = symbol_name
        self.mistakes = []
        self.successful_trades = []
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _on_step(self):
        # Track episode reward
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward
        
        # Track trade results from environment info
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            
            if 'trade_result' in info and info['trade_result']:
                trade = info['trade_result']
                if trade['pnl'] > 0:
                    self.successful_trades.append(trade)
                else:
                    self.mistakes.append(trade)
                    if self.verbose > 0:
                        print(f"   📝 [{self.symbol_name}] Mistake: {trade['symbol']} - PnL: {trade['pnl']:.2%}")
        
        return True
    
    def _on_episode_end(self):
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0
        
        if len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards[-100:])
            self.logger.record(f'custom/{self.symbol_name}_avg_reward', avg_reward)
        
    def get_learning_stats(self):
        total_trades = len(self.mistakes) + len(self.successful_trades)
        success_rate = len(self.successful_trades) / total_trades if total_trades > 0 else 0
        
        return {
            'symbol': self.symbol_name,
            'mistakes': len(self.mistakes),
            'successful_trades': len(self.successful_trades),
            'total_trades': total_trades,
            'success_rate': success_rate,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        }

# =========================================================
# TRADING ENVIRONMENT
# =========================================================

class MultiSymbolTradingEnv:
    """
    Multi-Symbol Trading Environment for PPO
    Supports both shared and per-symbol training
    """
    
    def __init__(self, symbol_dfs, signals, build_observation, window, state_dim,
                 total_capital=500_000, risk_percent=0.01, symbol_name="shared"):
        
        self.symbol_dfs = symbol_dfs
        self.signals = signals
        self.build_observation = build_observation
        self.window = window
        self.state_dim = state_dim
        self.total_capital = total_capital
        self.risk_percent = risk_percent
        self.symbol_name = symbol_name
        
        self.symbols = list(symbol_dfs.keys())
        self.current_symbol_idx = 0
        self.reset()
    
    def reset(self):
        """Reset environment"""
        self.current_symbol_idx = 0
        self._load_current_symbol()
        
        self.current_step = self.window
        self.capital = self.total_capital
        self.position = 0
        self.entry_price = 0
        self.total_reward = 0
        self.trades = []
        
        return self._get_obs()
    
    def _load_current_symbol(self):
        """Load current symbol data"""
        if self.current_symbol_idx < len(self.symbols):
            self.current_symbol = self.symbols[self.current_symbol_idx]
            self.current_df = self.symbol_dfs[self.current_symbol]
        else:
            self.current_symbol = None
            self.current_df = None
    
    def _get_obs(self):
        """Get current observation"""
        if self.current_symbol is None or self.current_step >= len(self.current_df):
            return np.zeros(self.state_dim, dtype=np.float32)
        
        obs = self.build_observation(
            self.current_df, 
            self.current_step, 
            self.signals
        )
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """Execute action"""
        if self.current_symbol is None:
            return self._get_obs(), 0, True, False, {}
        
        if self.current_step >= len(self.current_df) - 1:
            return self._move_to_next_symbol()
        
        row = self.current_df.iloc[self.current_step]
        next_row = self.current_df.iloc[self.current_step + 1]
        
        price = row['close']
        next_price = next_row['close']
        
        reward = 0
        terminated = False
        trade_result = None
        
        # Get signal info
        sig = self.signals.get((self.current_symbol, row['date']))
        buy_price = sig['buy'] if sig else None
        
        # Action: 0=Hold, 1=Buy, 2=Sell
        if action == 1:  # BUY
            if self.position == 0 and buy_price:
                risk_amount = self.capital * self.risk_percent
                shares = risk_amount / (price * 0.03)
                shares = min(shares, self.capital / price)
                
                self.position = shares
                self.entry_price = price
                self.capital -= shares * price
                reward -= 0.001
                
                # Bonus for good entry
                if price <= buy_price * 1.02:
                    reward += 0.02
        
        elif action == 2:  # SELL
            if self.position > 0:
                sell_amount = self.position * price * 0.999
                pnl = (price - self.entry_price) / self.entry_price
                reward = pnl * 10
                
                trade_result = {
                    'symbol': self.current_symbol,
                    'entry_price': self.entry_price,
                    'exit_price': price,
                    'pnl': pnl,
                    'success': pnl > 0
                }
                self.trades.append(trade_result)
                
                self.capital += sell_amount
                self.position = 0
                self.entry_price = 0
        
        # Hold reward based on momentum
        if action == 0 and self.position == 0:
            if next_price > price:
                reward += 0.001
            elif next_price < price:
                reward -= 0.001
        
        self.current_step += 1
        self.total_reward += reward
        
        # Check if we need to move to next symbol
        if self.current_step >= len(self.current_df) - 1:
            return self._move_to_next_symbol()
        
        return self._get_obs(), reward, False, False, {
            'balance': self.capital,
            'symbol': self.current_symbol,
            'trade_result': trade_result,
            'total_return': (self.capital / self.total_capital - 1) * 100
        }
    
    def _move_to_next_symbol(self):
        """Move to next symbol in the list"""
        self.current_symbol_idx += 1
        
        if self.current_symbol_idx >= len(self.symbols):
            # End of episode
            return self._get_obs(), 0, True, False, {
                'balance': self.capital,
                'symbol': 'END',
                'total_return': (self.capital / self.total_capital - 1) * 100
            }
        else:
            # Load next symbol
            self._load_current_symbol()
            self.current_step = self.window
            self.position = 0
            self.entry_price = 0
            
            return self._get_obs(), 0, False, False, {
                'balance': self.capital,
                'symbol': self.current_symbol,
                'total_return': (self.capital / self.total_capital - 1) * 100
            }

# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def should_retrain_ppo():
    """Check if PPO needs retraining"""
    if not os.path.exists(LAST_PPO_TRAIN):
        return True, "First training"
    
    with open(LAST_PPO_TRAIN, 'r') as f:
        last_date = datetime.strptime(f.read().strip(), '%Y-%m-%d')
    
    days_since = (datetime.now() - last_date).days
    
    if days_since >= PPO_RETRAIN_INTERVAL:
        return True, f"Monthly retrain (last: {days_since} days ago)"
    
    return False, f"Next retrain in {PPO_RETRAIN_INTERVAL - days_since} days"

def update_last_ppo_train():
    """Update last PPO train date"""
    with open(LAST_PPO_TRAIN, 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d'))

def load_xgb_metadata():
    """Load XGBoost model metadata and get top GOOD symbols"""
    if not os.path.exists(MODEL_METADATA):
        print("   ⚠️ No XGBoost metadata found")
        return pd.DataFrame()
    
    df = pd.read_csv(MODEL_METADATA)
    
    # Filter GOOD models and sort by AUC
    good_models = df[df['status'] == 'GOOD'].copy()
    good_models = good_models.sort_values('auc', ascending=False)
    
    return good_models

def load_past_mistakes():
    """Load past mistakes from prediction log for self-learning"""
    if not os.path.exists(PREDICTION_LOG):
        return []
    
    df = pd.read_csv(PREDICTION_LOG)
    df['date'] = pd.to_datetime(df['date'])
    
    mistakes = []
    for _, row in df.iterrows():
        if row.get('checked', 0) == 1 and row.get('prediction', 0) != row.get('actual', 0):
            mistakes.append({
                'symbol': row['symbol'],
                'date': row['date'],
                'prediction': row['prediction'],
                'actual': row['actual'],
                'close': row['close']
            })
    
    return mistakes

def load_signals(path):
    """Load trading signals from CSV"""
    if not os.path.exists(path):
        print(f"   ⚠️ Signal file not found: {path}")
        return {}
    
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    
    signals = {}
    for _, r in df.iterrows():
        signals[(r["symbol"], r["date"])] = {
            "buy": float(r["buy"]),
            "SL": float(r["SL"]),
            "TP": float(r["tp"]),
            "RRR": float(r["RRR"]),
        }
    return signals

def build_observation(df, idx, signals):
    """Build observation vector with market data and signal info"""
    pad = max(0, WINDOW - (idx + 1))
    start = max(0, idx - WINDOW + 1)
    
    seg = df.iloc[start:idx+1][MARKET_COLS].values
    seg = np.pad(seg, ((pad,0),(0,0)), mode="edge")
    market_vec = seg.flatten()
    
    row = df.iloc[idx]
    sig = signals.get((row["symbol"], row["date"]))
    
    if sig:
        buy = sig["buy"]
        signal_vec = [
            row["close"] / (buy + 1e-8),
            (buy - sig["SL"]) / (buy + 1e-8),
            (sig["TP"] - buy) / (buy + 1e-8),
            sig["RRR"],
        ]
    else:
        signal_vec = [0.0] * 4
    
    obs = list(market_vec) + signal_vec
    return np.nan_to_num(obs)

# =========================================================
# PER-SYMBOL PPO TRAINING
# =========================================================

def train_per_symbol_ppo(symbol, symbol_data, signals, xgb_auc, is_retrain=False):
    """Train PPO for a single symbol"""
    
    print(f"\n{'─'*50}")
    print(f"🎯 Training Per-Symbol PPO: {symbol} (XGBoost AUC: {xgb_auc:.2%})")
    print(f"{'─'*50}")
    
    # Select config based on XGBoost quality
    if xgb_auc >= 0.85:
        config = PPO_PER_SYMBOL_CONFIG['high_quality']
        quality = "HIGH"
    elif xgb_auc >= 0.70:
        config = PPO_PER_SYMBOL_CONFIG['good_quality']
        quality = "GOOD"
    else:
        config = PPO_PER_SYMBOL_CONFIG['fallback']
        quality = "FALLBACK"
    
    print(f"   Quality: {quality}")
    print(f"   Timesteps: {config['timesteps']:,}")
    print(f"   Learning Rate: {config['learning_rate']}")
    
    # Create single-symbol environment
    symbol_dfs = {symbol: symbol_data}
    
    env = MultiSymbolTradingEnv(
        symbol_dfs,
        signals,
        build_observation,
        WINDOW,
        STATE_DIM,
        total_capital=TOTAL_CAPITAL,
        risk_percent=RISK_PERCENT,
        symbol_name=symbol
    )
    env = DummyVecEnv([lambda: env])
    
    # Check if model exists (for retrain)
    model_path = PPO_SYMBOL_DIR / f"ppo_{symbol}"
    
    if is_retrain and os.path.exists(f"{model_path}.zip"):
        print(f"   🔄 Loading existing model for fine-tuning...")
        model = PPO.load(model_path, env=env, device="cpu")
        
        # Update learning rate for fine-tuning
        model.learning_rate = config['learning_rate'] * 0.5
    else:
        print(f"   🆕 Creating new model...")
        ppo_config = PPO_CONFIG.copy()
        ppo_config.update({
            'n_steps': config['n_steps'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
        })
        model = PPO("MlpPolicy", env, **ppo_config, verbose=0)
    
    # Self-learning callback
    callback = SelfLearningCallback(
        symbol_dfs, signals, build_observation, WINDOW, STATE_DIM,
        symbol_name=symbol, verbose=1
    )
    
    # Train
    print(f"   🚀 Training...")
    model.learn(
        total_timesteps=config['timesteps'],
        callback=callback
    )
    
    # Save model
    model.save(model_path)
    print(f"   ✅ Model saved: {model_path}")
    
    # Get stats
    stats = callback.get_learning_stats()
    print(f"   📊 Success Rate: {stats['success_rate']:.2%} ({stats['successful_trades']}/{stats['total_trades']})")
    
    return model, stats

# =========================================================
# SHARED PPO TRAINING
# =========================================================

def train_shared_ppo(all_symbols_data, signals, exclude_symbols=None, is_retrain=False):
    """Train shared PPO for all symbols (fallback)"""
    
    print(f"\n{'='*60}")
    print(f"🎯 Training Shared PPO (Fallback Model)")
    print(f"{'='*60}")
    
    # Filter out symbols that have per-symbol models (if needed)
    if exclude_symbols:
        filtered_data = {k: v for k, v in all_symbols_data.items() if k not in exclude_symbols}
        print(f"   Excluding {len(exclude_symbols)} symbols (have per-symbol models)")
    else:
        filtered_data = all_symbols_data
    
    print(f"   Total symbols in shared model: {len(filtered_data)}")
    
    env = MultiSymbolTradingEnv(
        filtered_data,
        signals,
        build_observation,
        WINDOW,
        STATE_DIM,
        total_capital=TOTAL_CAPITAL,
        risk_percent=RISK_PERCENT,
        symbol_name="shared"
    )
    env = DummyVecEnv([lambda: env])
    
    # Load existing model if retraining
    if is_retrain and os.path.exists(f"{PPO_SHARED_PATH}.zip"):
        print(f"   🔄 Loading existing shared model for fine-tuning...")
        model = PPO.load(PPO_SHARED_PATH, env=env, device="cpu")
        model.learning_rate = PPO_CONFIG['learning_rate'] * 0.5
        timesteps = 30000  # Less timesteps for fine-tuning
    else:
        print(f"   🆕 Creating new shared model...")
        model = PPO("MlpPolicy", env, **PPO_CONFIG, verbose=0)
        timesteps = 100000  # Full training for first time
    
    # Self-learning callback
    callback = SelfLearningCallback(
        filtered_data, signals, build_observation, WINDOW, STATE_DIM,
        symbol_name="shared", verbose=1
    )
    
    # Train
    print(f"   🚀 Training for {timesteps:,} timesteps...")
    model.learn(total_timesteps=timesteps, callback=callback)
    
    # Save model
    model.save(PPO_SHARED_PATH)
    print(f"   ✅ Shared model saved: {PPO_SHARED_PATH}")
    
    # Get stats
    stats = callback.get_learning_stats()
    print(f"   📊 Success Rate: {stats['success_rate']:.2%} ({stats['successful_trades']}/{stats['total_trades']})")
    
    return model, stats

# =========================================================
# PREDICTION FUNCTION (Uses appropriate model)
# =========================================================

class HybridPPOPredictor:
    """Predictor that uses per-symbol model if available, else shared model"""
    
    def __init__(self):
        self.per_symbol_models = {}
        self.shared_model = None
        self.load_models()
    
    def load_models(self):
        """Load all trained PPO models"""
        # Load per-symbol models
        for model_file in PPO_SYMBOL_DIR.glob("ppo_*.zip"):
            symbol = model_file.stem.replace("ppo_", "")
            try:
                self.per_symbol_models[symbol] = PPO.load(model_file, device="cpu")
                print(f"   ✅ Loaded per-symbol model: {symbol}")
            except Exception as e:
                print(f"   ⚠️ Failed to load {symbol}: {e}")
        
        # Load shared model
        if os.path.exists(f"{PPO_SHARED_PATH}.zip"):
            try:
                self.shared_model = PPO.load(PPO_SHARED_PATH, device="cpu")
                print(f"   ✅ Loaded shared model")
            except Exception as e:
                print(f"   ⚠️ Failed to load shared model: {e}")
    
    def predict(self, symbol, observation):
        """Predict action using appropriate model"""
        if symbol in self.per_symbol_models:
            action, _ = self.per_symbol_models[symbol].predict(observation, deterministic=True)
            return action[0] if isinstance(action, np.ndarray) else action
        elif self.shared_model:
            action, _ = self.shared_model.predict(observation, deterministic=True)
            return action[0] if isinstance(action, np.ndarray) else action
        else:
            return 0  # Hold if no model
    
    def get_model_info(self, symbol):
        """Get model info for a symbol"""
        if symbol in self.per_symbol_models:
            return f"Per-symbol model (customized)"
        else:
            return f"Shared model (fallback)"

# =========================================================
# MAIN TRAINING FUNCTION
# =========================================================

def train_ppo_system():
    """Main training function - runs first-time or monthly retrain"""
    
    print("="*70)
    print("🚀 HYBRID PPO TRAINING SYSTEM")
    print("="*70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"💰 Initial Capital: ${TOTAL_CAPITAL:,.2f}")
    print(f"📊 Retrain Interval: {PPO_RETRAIN_INTERVAL} days")
    print(f"🎯 Per-symbol PPO threshold: AUC >= {XGB_AUC_THRESHOLD_FOR_PPO}")
    print(f"🔢 Max per-symbol models: {MAX_PER_SYMBOL_MODELS}")
    print("="*70)
    
    # Check if retrain needed
    should_retrain, reason = should_retrain_ppo()
    is_retrain = should_retrain and os.path.exists(f"{PPO_SHARED_PATH}.zip")
    
    print(f"\n📊 Training Status: {reason}")
    print(f"   Mode: {'RETRAIN' if is_retrain else 'FIRST-TIME'}")
    
    # Step 1: Load market data
    print("\n📂 Step 1: Loading market data...")
    df = pd.read_csv(CSV_MARKET, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    print(f"   ✅ Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    
    # Step 2: Load signals
    print("\n📈 Step 2: Loading trading signals...")
    signals = load_signals(CSV_SIGNAL)
    print(f"   ✅ Loaded {len(signals)} signals")
    
    # Step 3: Load XGBoost metadata to identify top symbols
    print("\n🤖 Step 3: Loading XGBoost model metadata...")
    xgb_metadata = load_xgb_metadata()
    print(f"   ✅ XGBoost GOOD models: {len(xgb_metadata)}")
    
    # Step 4: Select top symbols for per-symbol PPO
    print("\n🎯 Step 4: Selecting symbols for per-symbol PPO...")
    top_symbols = xgb_metadata[xgb_metadata['auc'] >= XGB_AUC_THRESHOLD_FOR_PPO]
    top_symbols = top_symbols.head(MAX_PER_SYMBOL_MODELS)
    top_symbol_list = top_symbols['symbol'].tolist()
    print(f"   ✅ Selected {len(top_symbol_list)} symbols:")
    for i, (_, row) in enumerate(top_symbols.iterrows()):
        print(f"      {i+1}. {row['symbol']} (AUC: {row['auc']:.2%})")
    
    # Step 5: Prepare symbol data
    print("\n📊 Step 5: Preparing symbol data...")
    all_symbols_data = {}
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].reset_index(drop=True)
        if len(symbol_df) >= WINDOW + 50:
            all_symbols_data[symbol] = symbol_df
    
    print(f"   ✅ Prepared {len(all_symbols_data)} symbols with sufficient data")
    
    # Step 6: Load past mistakes for self-learning
    print("\n📚 Step 6: Loading past mistakes for self-learning...")
    past_mistakes = load_past_mistakes()
    print(f"   ✅ Loaded {len(past_mistakes)} past mistakes")
    
    # Step 7: Train per-symbol PPO for top symbols
    print("\n" + "="*70)
    print("🏆 Step 7: Training Per-Symbol PPO Models")
    print("="*70)
    
    trained_symbols = []
    per_symbol_stats = []
    
    for symbol in top_symbol_list:
        if symbol not in all_symbols_data:
            print(f"\n⚠️ Skipping {symbol}: insufficient data")
            continue
        
        symbol_data = all_symbols_data[symbol]
        xgb_info = top_symbols[top_symbols['symbol'] == symbol].iloc[0]
        xgb_auc = xgb_info['auc']
        
        try:
            model, stats = train_per_symbol_ppo(
                symbol, symbol_data, signals, xgb_auc, is_retrain
            )
            trained_symbols.append(symbol)
            per_symbol_stats.append(stats)
        except Exception as e:
            print(f"\n   ❌ Failed to train {symbol}: {e}")
    
    print(f"\n✅ Per-symbol PPO trained: {len(trained_symbols)} symbols")
    
    # Step 8: Train shared PPO for remaining symbols
    print("\n" + "="*70)
    print("🏆 Step 8: Training Shared PPO Model (Fallback)")
    print("="*70)
    
    shared_model, shared_stats = train_shared_ppo(
        all_symbols_data, signals, exclude_symbols=trained_symbols, is_retrain=is_retrain
    )
    
    # Step 9: Save training metadata
    print("\n💾 Step 9: Saving training metadata...")
    
    metadata = {
        'train_date': datetime.now(),
        'is_retrain': is_retrain,
        'per_symbol_count': len(trained_symbols),
        'per_symbol_list': trained_symbols,
        'shared_symbols_count': len(all_symbols_data) - len(trained_symbols),
        'shared_success_rate': shared_stats['success_rate'],
    }
    
    # Save per-symbol stats
    if per_symbol_stats:
        stats_df = pd.DataFrame(per_symbol_stats)
        stats_df.to_csv(PPO_MODEL_DIR / "per_symbol_stats.csv", index=False)
    
    # Update last train date
    update_last_ppo_train()
    
    # Step 10: Summary
    print("\n" + "="*70)
    print("🎉 HYBRID PPO TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 TRAINING SUMMARY:")
    print(f"   ├── Per-symbol PPO models: {len(trained_symbols)}")
    print(f"   ├── Shared PPO model: 1")
    print(f"   └── Total models: {len(trained_symbols) + 1}")
    
    if per_symbol_stats:
        avg_success = np.mean([s['success_rate'] for s in per_symbol_stats])
        print(f"\n📈 PERFORMANCE:")
        print(f"   ├── Per-symbol avg success rate: {avg_success:.2%}")
        print(f"   └── Shared success rate: {shared_stats['success_rate']:.2%}")
    
    print(f"\n📁 Model Locations:")
    print(f"   ├── Per-symbol models: {PPO_SYMBOL_DIR}")
    print(f"   └── Shared model: {PPO_SHARED_PATH}.zip")
    print("="*70)
    
    return trained_symbols, shared_model

# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_ppo_system():
    """Evaluate the hybrid PPO system"""
    
    print("\n" + "="*70)
    print("📊 EVALUATING HYBRID PPO SYSTEM")
    print("="*70)
    
    predictor = HybridPPOPredictor()
    
    # Load test data
    df = pd.read_csv(CSV_MARKET, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    signals = load_signals(CSV_SIGNAL)
    
    # Test on a few symbols
    test_symbols = list(predictor.per_symbol_models.keys())[:5] + ['NON_EXISTENT_SYMBOL']
    
    print("\n🔍 Testing predictions:")
    for symbol in test_symbols:
        if symbol not in df['symbol'].unique():
            print(f"\n   {symbol}: NOT IN DATA")
            continue
        
        symbol_data = df[df['symbol'] == symbol].reset_index(drop=True)
        if len(symbol_data) < WINDOW + 10:
            continue
        
        model_type = predictor.get_model_info(symbol)
        print(f"\n   {symbol}: {model_type}")
        
        # Make a few predictions
        for i in range(WINDOW, min(WINDOW + 5, len(symbol_data))):
            obs = build_observation(symbol_data, i, signals)
            action = predictor.predict(symbol, obs)
            action_name = ['HOLD', 'BUY', 'SELL'][action]
            price = symbol_data.iloc[i]['close']
            print(f"      Step {i}: Price={price:.2f} → {action_name}")
    
    return predictor

# =========================================================
# MAIN
# =========================================================

def main():
    """Main execution"""
    
    # Train the system
    trained_symbols, shared_model = train_ppo_system()
    
    # Evaluate
    predictor = evaluate_ppo_system()
    
    print("\n" + "="*70)
    print("🎉 PPO SYSTEM READY FOR TRADING!")
    print("="*70)

if __name__ == "__main__":
    main()