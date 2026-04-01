# xgboost_ppo_env.py - XGBoost + PPO Trading Environment

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import joblib
import os

class XGBoostPPOTradingEnv(gym.Env):
    """
    Trading Environment with XGBoost signals for PPO agent
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 model_dir: str = "./csv/xgboost/",
                 initial_balance: float = 100000,
                 max_position: float = 0.3,
                 trading_fee: float = 0.001):
        
        super(XGBoostPPOTradingEnv, self).__init__()
        
        self.data = data
        self.model_dir = model_dir
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.trading_fee = trading_fee
        
        # Load XGBoost models for all symbols
        self.xgb_models = self._load_xgb_models()
        
        # Get unique symbols
        self.symbols = data['symbol'].unique()
        self.current_symbol_idx = 0
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 
        # [price_norm, volume_norm, rsi_norm, macd_norm, 
        #  xgb_conf, xgb_pred, balance_norm, position_norm,
        #  return_5d, return_10d, volatility, volume_ratio,
        #  bb_upper_dist, bb_lower_dist, atr_ratio]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        
        self.reset()
    
    def _load_xgb_models(self):
        """Load all trained XGBoost models"""
        models = {}
        if os.path.exists(self.model_dir):
            for file in os.listdir(self.model_dir):
                if file.endswith('.joblib'):
                    symbol = file.replace('.joblib', '')
                    models[symbol] = joblib.load(os.path.join(self.model_dir, file))
                    print(f"✅ Loaded XGBoost model: {symbol}")
        return models
    
    def _get_xgb_signal(self, symbol, features_df):
        """Get XGBoost prediction for current symbol"""
        if symbol in self.xgb_models:
            model = self.xgb_models[symbol]
            # Get latest features for prediction
            latest = features_df.iloc[-1:].fillna(0)
            
            # Check if features match model expectations
            try:
                prob = model.predict_proba(latest)[0, 1]
                pred = 1 if prob > 0.5 else 0
                return prob * 100, pred  # confidence %, prediction
            except:
                pass
        return 50, 0  # Default: neutral
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Select a random symbol
        self.current_symbol_idx = np.random.randint(0, len(self.symbols))
        self.current_symbol = self.symbols[self.current_symbol_idx]
        
        # Get data for this symbol
        self.symbol_data = self.data[self.data['symbol'] == self.current_symbol].copy()
        self.symbol_data = self.symbol_data.sort_values('date').reset_index(drop=True)
        
        # Calculate features for XGBoost
        self.symbol_data = self._calculate_features(self.symbol_data)
        
        # Get XGBoost signals
        xgb_conf, xgb_pred = self._get_xgb_signal(
            self.current_symbol, 
            self.symbol_data
        )
        
        # Add XGBoost signals to data
        self.symbol_data['xgb_confidence'] = xgb_conf
        self.symbol_data['xgb_prediction'] = xgb_pred
        
        self.current_step = 20  # Start after some history
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_reward = 0
        self.trades = []
        
        return self._get_obs(), {}
    
    def _calculate_features(self, df):
        """Calculate features for XGBoost (same as training)"""
        # Price changes
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        
        # Volatility
        df['volatility'] = (df['high'] - df['low']) / df['close']
        df['volatility_5d'] = df['volatility'].rolling(5).mean()
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Technical indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_distance'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        df['atr'] = self._calculate_atr(df)
        
        return df
    
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
    
    def _get_obs(self):
        """Get current observation"""
        if self.current_step >= len(self.symbol_data):
            return np.zeros(15, dtype=np.float32)
        
        row = self.symbol_data.iloc[self.current_step]
        
        obs = np.array([
            row['close'] / 1000,                    # Normalized price
            row['volume'] / 1000000,                # Normalized volume
            row.get('rsi', 50) / 100,               # RSI
            row.get('macd', 0) / 10,                # MACD
            row.get('xgb_confidence', 50) / 100,    # XGBoost confidence
            row.get('xgb_prediction', 0),           # XGBoost prediction
            self.balance / self.initial_balance,    # Normalized balance
            self.position / 1000,                   # Normalized position
            row.get('return_5d', 0),                # 5-day return
            row.get('return_10d', 0),               # 10-day return
            row.get('volatility', 0),               # Volatility
            row.get('volume_ratio', 1),             # Volume ratio
            row.get('bb_distance', 0.5),            # Bollinger distance
            row.get('atr', 0) / row.get('close', 1),# ATR ratio
            self.total_reward / 1000                # Normalized reward
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        """Execute action and return next state"""
        if self.current_step >= len(self.symbol_data) - 1:
            return self._get_obs(), 0, True, False, {}
        
        row = self.symbol_data.iloc[self.current_step]
        next_row = self.symbol_data.iloc[self.current_step + 1]
        
        price = row['close']
        next_price = next_row['close']
        xgb_conf = row.get('xgb_confidence', 50) / 100
        xgb_pred = row.get('xgb_prediction', 0)
        
        reward = 0
        terminated = False
        truncated = False
        
        # Execute action
        if action == 1:  # BUY
            if self.position == 0:
                buy_amount = self.balance * self.max_position
                shares = buy_amount / price
                self.position = shares
                self.entry_price = price
                self.balance -= buy_amount
                reward -= self.trading_fee
                
                # Bonus for following strong XGBoost signal
                if xgb_conf > 0.7 and xgb_pred == 1:
                    reward += 0.05
        
        elif action == 2:  # SELL
            if self.position > 0:
                sell_amount = self.position * price
                self.balance += sell_amount * (1 - self.trading_fee)
                pnl = (price - self.entry_price) / self.entry_price
                reward = pnl * 10
                self.position = 0
                self.entry_price = 0
        
        # Hold action reward based on XGBoost
        if action == 0 and xgb_conf > 0.6 and next_price > price:
            reward += 0.02 * xgb_conf
        elif action == 0 and xgb_conf < 0.4 and next_price < price:
            reward += 0.02 * (1 - xgb_conf)
        
        # Move to next step
        self.current_step += 1
        self.total_reward += reward
        
        # Check if episode ends
        if self.current_step >= len(self.symbol_data) - 1:
            terminated = True
            if self.position > 0:
                sell_amount = self.position * price
                self.balance += sell_amount * (1 - self.trading_fee)
                self.position = 0
        
        return self._get_obs(), reward, terminated, truncated, {
            'balance': self.balance,
            'symbol': self.current_symbol,
            'step': self.current_step,
            'xgb_conf': xgb_conf
        
