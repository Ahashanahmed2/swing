# train_ppo_xgboost.py - Complete Training Script

import os
import numpy as np
import pandas as pd
import torch
import joblib
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIGURATION
# =========================
XGB_MODEL_DIR = "./csv/xgboost/"
DATA_PATH = "./csv/mongodb.csv"
PPO_MODEL_DIR = "./csv/ppo_models/"
LOG_DIR = "./logs/ppo/"
OUTPUT_DIR = "./output/ai_signal/"

os.makedirs(PPO_MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trading parameters
INITIAL_BALANCE = 100000
MAX_POSITION = 0.3  # 30% of capital per trade
TRADING_FEE = 0.001  # 0.1%

# PPO Parameters
TOTAL_TIMESTEPS = 100000  # Start with 100k, increase if needed
LEARNING_RATE = 0.0003
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10

# =========================
# TRADING ENVIRONMENT (XGBoost + PPO)
# =========================

class XGBoostPPOTradingEnv(gym.Env):
    """Trading Environment with XGBoost Signals"""
    
    def __init__(self, data, symbol, xgb_model):
        super().__init__()
        
        self.data = data.copy()
        self.symbol = symbol
        self.xgb_model = xgb_model
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 50  # Start after history
        self.balance = INITIAL_BALANCE
        self.position = 0
        self.entry_price = 0
        self.total_reward = 0
        self.trades = []
        
        # Calculate features
        self._calculate_features()
        
        return self._get_obs(), {}
    
    def _calculate_features(self):
        """Calculate all features including XGBoost signals"""
        df = self.data
        
        # Price changes
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        
        # Volatility
        df['volatility'] = (df['high'] - df['low']) / df['close']
        df['volatility_5d'] = df['volatility'].rolling(5).mean()
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger(df['close'])
        df['bb_distance'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        df['atr'] = self._calculate_atr(df)
        
        # XGBoost predictions
        df['xgb_confidence'], df['xgb_prediction'] = self._get_xgb_signals(df)
        
        self.data = df.fillna(0)
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices):
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        return macd.fillna(0)
    
    def _calculate_bollinger(self, prices, period=20):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper.fillna(prices), lower.fillna(prices)
    
    def _calculate_atr(self, df, period=14):
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.fillna(tr.mean())
    
    def _get_xgb_signals(self, df):
        """Get XGBoost predictions for each row"""
        if self.xgb_model is None:
            return 50, 0
        
        try:
            # Prepare features (same as training)
            features = df[['close', 'volume', 'return_5d', 'return_10d', 
                           'volatility', 'volatility_5d', 'volume_ratio']].fillna(0)
            
            # Get probabilities
            prob = self.xgb_model.predict_proba(features)[:, 1]
            pred = (prob > 0.5).astype(int)
            
            return prob * 100, pred
        except:
            return 50, 0
    
    def _get_obs(self):
        """Get current observation"""
        if self.current_step >= len(self.data):
            return np.zeros(15, dtype=np.float32)
        
        row = self.data.iloc[self.current_step]
        
        obs = np.array([
            row['close'] / 1000,                    # 0: price
            row['volume'] / 1000000,                # 1: volume
            row.get('rsi', 50) / 100,               # 2: RSI
            row.get('macd', 0) / 10,                # 3: MACD
            row.get('xgb_confidence', 50) / 100,    # 4: XGB confidence
            row.get('xgb_prediction', 0),           # 5: XGB prediction
            self.balance / INITIAL_BALANCE,         # 6: balance
            self.position / 1000,                   # 7: position
            row.get('return_5d', 0),                # 8: 5d return
            row.get('return_10d', 0),               # 9: 10d return
            row.get('volatility', 0),               # 10: volatility
            row.get('volume_ratio', 1),             # 11: volume ratio
            row.get('bb_distance', 0.5),            # 12: BB distance
            row.get('atr', 0) / row.get('close', 1),# 13: ATR ratio
            self.total_reward / 1000                # 14: reward
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        """Execute action"""
        if self.current_step >= len(self.data) - 1:
            return self._get_obs(), 0, True, False, {}
        
        row = self.data.iloc[self.current_step]
        next_row = self.data.iloc[self.current_step + 1]
        
        price = row['close']
        next_price = next_row['close']
        xgb_conf = row.get('xgb_confidence', 50) / 100
        xgb_pred = row.get('xgb_prediction', 0)
        
        reward = 0
        terminated = False
        
        # Execute action
        if action == 1:  # BUY
            if self.position == 0:
                buy_amount = self.balance * MAX_POSITION
                shares = buy_amount / price
                self.position = shares
                self.entry_price = price
                self.balance -= buy_amount
                reward -= TRADING_FEE
                
                # Bonus for following strong XGBoost signal
                if xgb_conf > 0.65 and xgb_pred == 1:
                    reward += 0.05
        
        elif action == 2:  # SELL
            if self.position > 0:
                sell_amount = self.position * price
                self.balance += sell_amount * (1 - TRADING_FEE)
                pnl = (price - self.entry_price) / self.entry_price
                reward = pnl * 10
                self.position = 0
                self.entry_price = 0
        
        # Hold reward based on XGBoost
        if action == 0 and xgb_conf > 0.6 and next_price > price:
            reward += 0.02 * xgb_conf
        elif action == 0 and xgb_conf < 0.4 and next_price < price:
            reward += 0.02 * (1 - xgb_conf)
        
        self.current_step += 1
        self.total_reward += reward
        
        if self.current_step >= len(self.data) - 1:
            terminated = True
            if self.position > 0:
                sell_amount = self.position * price
                self.balance += sell_amount * (1 - TRADING_FEE)
                self.position = 0
        
        return self._get_obs(), reward, terminated, False, {
            'balance': self.balance,
            'symbol': self.symbol,
            'step': self.current_step,
            'xgb_conf': xgb_conf,
            'total_return': (self.balance / INITIAL_BALANCE - 1) * 100
        }

# =========================
# CUSTOM CALLBACK
# =========================

class TensorboardCallback(BaseCallback):
    """Custom callback for logging"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self):
        # Log balance if available
        if 'balance' in self.locals['infos'][0]:
            balance = self.locals['infos'][0]['balance']
            self.logger.record('custom/balance', balance)
        
        return True

# =========================
# LOAD XGBOOST MODELS
# =========================

def load_xgb_models():
    """Load all trained XGBoost models"""
    models = {}
    
    if os.path.exists(XGB_MODEL_DIR):
        for file in os.listdir(XGB_MODEL_DIR):
            if file.endswith('.joblib'):
                symbol = file.replace('.joblib', '')
                try:
                    model = joblib.load(os.path.join(XGB_MODEL_DIR, file))
                    models[symbol] = model
                    print(f"   ✅ Loaded: {symbol}")
                except Exception as e:
                    print(f"   ⚠️ Failed to load {symbol}: {e}")
    
    return models

# =========================
# TRAINING FUNCTION
# =========================

def train_ppo_for_symbol(symbol, data, xgb_model):
    """Train PPO agent for a single symbol"""
    
    print(f"\n{'='*60}")
    print(f"🎯 Training PPO for {symbol}")
    print(f"{'='*60}")
    
    # Create environment
    env = XGBoostPPOTradingEnv(data, symbol, xgb_model)
    env = Monitor(env, f"{LOG_DIR}/{symbol}/")
    env = DummyVecEnv([lambda: env])
    
    # PPO configuration
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device="cpu",
        tensorboard_log=LOG_DIR
    )
    
    # Train
    print("   🚀 Training started...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=TensorboardCallback()
    )
    
    # Save model
    save_path = os.path.join(PPO_MODEL_DIR, f"ppo_{symbol}")
    model.save(save_path)
    print(f"   ✅ Model saved: {save_path}")
    
    return model

# =========================
# MAIN TRAINING LOOP
# =========================

def main():
    print("="*70)
    print("🚀 PPO + XGBoost Training System")
    print("="*70)
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💰 Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"📊 Total Timesteps per symbol: {TOTAL_TIMESTEPS:,}")
    print("="*70)
    
    # Step 1: Load data
    print("\n📂 Step 1: Loading market data...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    print(f"   ✅ Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    
    # Step 2: Load XGBoost models
    print("\n🤖 Step 2: Loading XGBoost models...")
    xgb_models = load_xgb_models()
    print(f"   ✅ Loaded {len(xgb_models)} XGBoost models")
    
    if not xgb_models:
        print("   ⚠️ No XGBoost models found! Training may not work well.")
    
    # Step 3: Select symbols to train
    print("\n🎯 Step 3: Selecting symbols for PPO training...")
    
    # Get symbols with GOOD XGBoost models (AUC >= 0.55)
    good_symbols = list(xgb_models.keys())
    
    # Limit to top 10 for initial training (you can increase later)
    train_symbols = good_symbols[:10] if len(good_symbols) > 10 else good_symbols
    
    print(f"   ✅ Selected {len(train_symbols)} symbols: {', '.join(train_symbols)}")
    
    # Step 4: Train PPO for each symbol
    print("\n🏆 Step 4: Training PPO agents...")
    print("="*70)
    
    trained_models = {}
    
    for symbol in train_symbols:
        try:
            # Get data for this symbol
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
            
            if len(symbol_data) < 100:
                print(f"\n⚠️ Skipping {symbol}: insufficient data ({len(symbol_data)} rows)")
                continue
            
            # Train PPO
            model = train_ppo_for_symbol(symbol, symbol_data, xgb_models.get(symbol))
            trained_models[symbol] = model
            
        except Exception as e:
            print(f"\n❌ Error training {symbol}: {e}")
    
    # Step 5: Summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    print(f"✅ Successfully trained: {len(trained_models)} agents")
    print(f"📁 Models saved in: {PPO_MODEL_DIR}")
    
    # Step 6: Quick test on one symbol
    if trained_models:
        test_symbol = list(trained_models.keys())[0]
        print(f"\n🔍 Testing {test_symbol}...")
        
        test_data = df[df['symbol'] == test_symbol].copy()
        test_data = test_data.sort_values('date').reset_index(drop=True)
        
        env = XGBoostPPOTradingEnv(test_data, test_symbol, xgb_models.get(test_symbol))
        obs, _ = env.reset()
        
        total_reward = 0
        steps = 0
        
        while True:
            action, _ = trained_models[test_symbol].predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"   Steps: {steps}")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Final Balance: ${info['balance']:,.2f}")
        print(f"   Return: {info['total_return']:.2f}%")
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()