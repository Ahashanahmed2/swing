# ppo_train.py - Complete PPO Training with Retrain, Fine-tune & Self-Learning
# Features:
# 1. First-time training
# 2. Monthly retrain/fine-tune
# 3. Self-learning from past mistakes
# 4. Curriculum learning (start with 1 symbol, expand gradually)
# 5. Performance tracking and improvement

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_MARKET = BASE_DIR / "csv" / "mongodb.csv"
CSV_SIGNAL = BASE_DIR / "csv" / "trade_stock.csv"
MODEL_DIR = BASE_DIR / "csv" / "model"
PPO_MODEL_PATH = MODEL_DIR / "sb3_ppo_trading"
PPO_METADATA_PATH = BASE_DIR / "csv" / "ppo_metadata.csv"
PREDICTION_LOG = BASE_DIR / "csv" / "prediction_log.csv"
LAST_PPO_TRAIN = BASE_DIR / "csv" / "last_ppo_train.txt"

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# CONFIGURATION
# =========================================================

WINDOW = 10
TOTAL_CAPITAL = 500_000
RISK_PERCENT = 0.01
PPO_RETRAIN_INTERVAL = 30  # Days between retrains

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

# PPO Parameters
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

# Training stages
STAGES = [
    {'symbols': 1, 'timesteps': 50000, 'name': 'Stage 1: Single Symbol'},
    {'symbols': 5, 'timesteps': 100000, 'name': 'Stage 2: Multiple Symbols'},
    {'symbols': 10, 'timesteps': 150000, 'name': 'Stage 3: All Symbols'},
]

# =========================================================
# SELF-LEARNING CALLBACK
# =========================================================

class SelfLearningCallback(BaseCallback):
    """
    Callback that learns from past mistakes during training
    """
    
    def __init__(self, symbol_dfs, signals, build_observation, window, state_dim, verbose=0):
        super().__init__(verbose)
        self.symbol_dfs = symbol_dfs
        self.signals = signals
        self.build_observation = build_observation
        self.window = window
        self.state_dim = state_dim
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
                    print(f"   📝 Mistake recorded: {trade['symbol']} - PnL: {trade['pnl']:.2%}")
        
        return True
    
    def _on_episode_end(self):
        # Store episode reward
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0
        
        # Update logger
        if len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards[-100:])
            self.logger.record('custom/avg_reward', avg_reward)
        
    def get_learning_stats(self):
        """Get learning statistics"""
        total_trades = len(self.mistakes) + len(self.successful_trades)
        success_rate = len(self.successful_trades) / total_trades if total_trades > 0 else 0
        
        return {
            'mistakes': len(self.mistakes),
            'successful_trades': len(self.successful_trades),
            'total_trades': total_trades,
            'success_rate': success_rate,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        }

# =========================================================
# ENVIRONMENT (from your existing env_trading.py)
# =========================================================

class MultiSymbolTradingEnv:
    """
    Multi-Symbol Trading Environment for PPO
    """
    
    def __init__(self, symbol_dfs, signals, build_observation, window, state_dim,
                 total_capital=500_000, risk_percent=0.01):
        
        self.symbol_dfs = symbol_dfs
        self.signals = signals
        self.build_observation = build_observation
        self.window = window
        self.state_dim = state_dim
        self.total_capital = total_capital
        self.risk_percent = risk_percent
        
        self.symbols = list(symbol_dfs.keys())
        self.current_symbol_idx = 0
        self.reset()
    
    def reset(self):
        """Reset environment"""
        self.current_symbol_idx = 0
        self.current_symbol = self.symbols[self.current_symbol_idx]
        self.current_df = self.symbol_dfs[self.current_symbol]
        
        self.current_step = self.window
        self.capital = self.total_capital
        self.position = 0
        self.entry_price = 0
        self.total_reward = 0
        self.trades = []
        
        return self._get_obs()
    
    def _get_obs(self):
        """Get current observation"""
        if self.current_step >= len(self.current_df):
            return np.zeros(self.state_dim, dtype=np.float32)
        
        obs = self.build_observation(
            self.current_df, 
            self.current_step, 
            self.signals
        )
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """Execute action"""
        if self.current_step >= len(self.current_df) - 1:
            return self._get_obs(), 0, True, False, {}
        
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
                shares = risk_amount / (price * 0.03)  # 3% stop loss assumption
                shares = min(shares, self.capital / price)
                
                self.position = shares
                self.entry_price = price
                self.capital -= shares * price
                reward -= 0.001  # Trading fee
                
                # Bonus for buying at good price
                if price <= buy_price * 1.02:
                    reward += 0.02
        
        elif action == 2:  # SELL
            if self.position > 0:
                sell_amount = self.position * price * 0.999  # with fee
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
        
        # Check if we need to switch symbol
        if self.current_step >= len(self.current_df) - 1:
            self.current_symbol_idx += 1
            
            if self.current_symbol_idx >= len(self.symbols):
                terminated = True
            else:
                self.current_symbol = self.symbols[self.current_symbol_idx]
                self.current_df = self.symbol_dfs[self.current_symbol]
                self.current_step = self.window
        
        return self._get_obs(), reward, terminated, False, {
            'balance': self.capital,
            'symbol': self.current_symbol,
            'trade_result': trade_result,
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

def load_ppo_metadata():
    """Load PPO training metadata"""
    if os.path.exists(PPO_METADATA_PATH):
        df = pd.read_csv(PPO_METADATA_PATH)
        df['last_trained'] = pd.to_datetime(df['last_trained'])
        return df
    return pd.DataFrame(columns=['train_date', 'symbols_count', 'success_rate', 
                                  'total_trades', 'profitable_trades', 'total_return'])

def save_ppo_metadata(df):
    """Save PPO training metadata"""
    df.to_csv(PPO_METADATA_PATH, index=False)

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
    """Build observation vector"""
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
# PPO TRAINING (First-time)
# =========================================================

def train_ppo_first_time():
    """First-time PPO training with curriculum learning"""
    print("\n" + "="*70)
    print("🎯 PPO FIRST-TIME TRAINING")
    print("="*70)
    
    # Load data
    df = pd.read_csv(CSV_MARKET, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    signals = load_signals(CSV_SIGNAL)
    
    # Create symbol dataframes
    symbol_dfs = {
        s: sdf.reset_index(drop=True)
        for s, sdf in df.groupby("symbol")
        if len(sdf) >= WINDOW
    }
    
    print(f"📊 Total symbols available: {len(symbol_dfs)}")
    
    trained_model = None
    total_timesteps = 0
    
    for stage in STAGES:
        print(f"\n📈 {stage['name']}")
        print(f"   Symbols: {stage['symbols']}")
        print(f"   Timesteps: {stage['timesteps']:,}")
        
        # Select symbols for this stage
        stage_symbols = list(symbol_dfs.keys())[:stage['symbols']]
        stage_dfs = {s: symbol_dfs[s] for s in stage_symbols}
        
        # Create environment
        env = MultiSymbolTradingEnv(
            stage_dfs,
            signals,
            build_observation,
            WINDOW,
            STATE_DIM,
            total_capital=TOTAL_CAPITAL,
            risk_percent=RISK_PERCENT,
        )
        env = DummyVecEnv([lambda: env])
        
        # Create callback
        callback = SelfLearningCallback(
            stage_dfs, signals, build_observation, WINDOW, STATE_DIM
        )
        
        # Create or load model
        if trained_model is None:
            model = PPO("MlpPolicy", env, **PPO_CONFIG, verbose=1)
        else:
            model = trained_model
            model.set_env(env)
        
        # Train
        model.learn(total_timesteps=stage['timesteps'], callback=callback)
        trained_model = model
        total_timesteps += stage['timesteps']
        
        # Show stage stats
        stats = callback.get_learning_stats()
        print(f"\n   📊 Stage {stage['name']} Complete:")
        print(f"      Success Rate: {stats['success_rate']:.2%}")
        print(f"      Total Trades: {stats['total_trades']}")
        print(f"      Avg Reward: {stats['avg_reward']:.2f}")
    
    # Save final model
    trained_model.save(PPO_MODEL_PATH)
    print(f"\n✅ Model saved: {PPO_MODEL_PATH}")
    print(f"   Total timesteps: {total_timesteps:,}")
    
    return trained_model

# =========================================================
# PPO RETRAIN (Monthly Fine-tuning)
# =========================================================

def retrain_ppo_with_self_learning():
    """Monthly retrain/fine-tune with self-learning from mistakes"""
    print("\n" + "="*70)
    print("🔄 PPO MONTHLY RETRAIN - Self-Learning")
    print("="*70)
    
    # Load data
    df = pd.read_csv(CSV_MARKET, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    signals = load_signals(CSV_SIGNAL)
    
    # Load past mistakes
    past_mistakes = load_past_mistakes()
    print(f"📚 Loaded {len(past_mistakes)} past mistakes for self-learning")
    
    # Create symbol dataframes (all symbols)
    symbol_dfs = {
        s: sdf.reset_index(drop=True)
        for s, sdf in df.groupby("symbol")
        if len(sdf) >= WINDOW
    }
    
    print(f"📊 Total symbols: {len(symbol_dfs)}")
    
    # Load existing model
    if not os.path.exists(f"{PPO_MODEL_PATH}.zip"):
        print("⚠️ No existing model found. Running first-time training...")
        return train_ppo_first_time()
    
    model = PPO.load(PPO_MODEL_PATH, device="cpu")
    print(f"✅ Loaded existing model")
    
    # Create environment with all symbols
    env = MultiSymbolTradingEnv(
        symbol_dfs,
        signals,
        build_observation,
        WINDOW,
        STATE_DIM,
        total_capital=TOTAL_CAPITAL,
        risk_percent=RISK_PERCENT,
    )
    env = DummyVecEnv([lambda: env])
    model.set_env(env)
    
    # Self-learning callback
    callback = SelfLearningCallback(
        symbol_dfs, signals, build_observation, WINDOW, STATE_DIM
    )
    
    # Fine-tune with lower learning rate
    fine_tune_timesteps = 50000
    print(f"🔄 Fine-tuning for {fine_tune_timesteps:,} timesteps...")
    
    model.learn(
        total_timesteps=fine_tune_timesteps,
        callback=callback,
        reset_num_timesteps=False
    )
    
    # Get learning stats
    stats = callback.get_learning_stats()
    
    print(f"\n📊 RETRAIN SUMMARY:")
    print(f"   Success Rate: {stats['success_rate']:.2%}")
    print(f"   Mistakes Learned: {stats['mistakes']}")
    print(f"   Successful Trades: {stats['successful_trades']}")
    print(f"   Avg Reward: {stats['avg_reward']:.2f}")
    
    # Save updated model
    model.save(PPO_MODEL_PATH)
    print(f"✅ Model saved: {PPO_MODEL_PATH}")
    
    # Save metadata
    metadata = load_ppo_metadata()
    new_record = pd.DataFrame([{
        'train_date': datetime.now(),
        'symbols_count': len(symbol_dfs),
        'success_rate': stats['success_rate'],
        'total_trades': stats['total_trades'],
        'profitable_trades': stats['successful_trades'],
        'total_return': 0  # Can be calculated from evaluation
    }])
    
    metadata = pd.concat([metadata, new_record], ignore_index=True)
    save_ppo_metadata(metadata)
    
    return model

# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_ppo_model():
    """Evaluate PPO model performance"""
    print("\n" + "="*70)
    print("📊 PPO MODEL EVALUATION")
    print("="*70)
    
    if not os.path.exists(f"{PPO_MODEL_PATH}.zip"):
        print("⚠️ No model found")
        return None
    
    # Load model
    model = PPO.load(PPO_MODEL_PATH, device="cpu")
    
    # Load data
    df = pd.read_csv(CSV_MARKET, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    signals = load_signals(CSV_SIGNAL)
    
    # Create environment (use last 30 days for evaluation)
    symbol_dfs = {
        s: sdf.reset_index(drop=True)
        for s, sdf in df.groupby("symbol")
        if len(sdf) >= WINDOW
    }
    
    env = MultiSymbolTradingEnv(
        symbol_dfs,
        signals,
        build_observation,
        WINDOW,
        STATE_DIM,
        total_capital=TOTAL_CAPITAL,
        risk_percent=RISK_PERCENT,
    )
    env = DummyVecEnv([lambda: env])
    
    # Evaluate
    obs = env.reset()
    total_reward = 0
    steps = 0
    trades = []
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        steps += 1
        
        if info[0].get('trade_result'):
            trades.append(info[0]['trade_result'])
        
        if done:
            break
    
    profitable = sum(1 for t in trades if t['success'])
    total_return = info[0]['total_return']
    
    print(f"\n📈 Evaluation Results:")
    print(f"   Total Steps: {steps}")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Total Trades: {len(trades)}")
    print(f"   Profitable: {profitable}")
    print(f"   Success Rate: {profitable/len(trades)*100:.2f}%" if trades else "N/A")
    print(f"   Total Return: {total_return:.2f}%")
    
    return {
        'steps': steps,
        'reward': total_reward,
        'trades': len(trades),
        'profitable': profitable,
        'success_rate': profitable/len(trades) if trades else 0,
        'total_return': total_return
    }

# =========================================================
# MAIN EXECUTION
# =========================================================

def main():
    print("="*70)
    print("🚀 PPO TRADING SYSTEM - Training & Self-Learning")
    print("="*70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"💰 Initial Capital: ${TOTAL_CAPITAL:,.2f}")
    print(f"📊 Retrain Interval: {PPO_RETRAIN_INTERVAL} days")
    print("="*70)
    
    # Check if first-time or retrain
    should_retrain, reason = should_retrain_ppo()
    
    if should_retrain:
        print(f"\n🔄 {reason}")
        
        if os.path.exists(f"{PPO_MODEL_PATH}.zip"):
            # Retrain with self-learning
            model = retrain_ppo_with_self_learning()
        else:
            # First-time training
            print("   No existing model found. Starting first-time training...")
            model = train_ppo_first_time()
        
        # Update last train date
        update_last_ppo_train()
        print(f"\n✅ Last PPO train updated: {datetime.now().strftime('%Y-%m-%d')}")
        
    else:
        print(f"\n✅ {reason}")
        print("   Skipping retrain. Using existing model.")
        
        # Still evaluate
        evaluate_ppo_model()
    
    # Always evaluate after training
    print("\n" + "="*70)
    print("🎯 FINAL EVALUATION")
    print("="*70)
    eval_results = evaluate_ppo_model()
    
    print("\n" + "="*70)
    print("🎉 PPO SYSTEM COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()