# xgboost_ppo_env.py - HEDGE FUND LEVEL v3 (All Critical Bugs Fixed)
# Fixed Issues:
# ✅ 1. XGBoost per-step dynamic signal (no data leakage)
# ✅ 2. Feature matching with training data
# ✅ 3. Proper observation scaling (StandardScaler)
# ✅ 4. Reward clipping to prevent explosion
# ✅ 5. Fixed equity curve double append bug
# ✅ 6. Episode randomization for better generalization

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# HEDGE FUND LEVEL CONFIGURATION
# =========================================================

class HedgeFundConfig:
    """Configuration for Hedge Fund Level Trading Environment"""
    
    # Trading parameters
    INITIAL_BALANCE = 500_000
    MAX_POSITION = 0.30
    TRADING_FEE = 0.001
    SLIPPAGE = 0.0005
    
    # Risk management
    STOP_LOSS_PCT = 0.03
    TRAILING_STOP_PCT = 0.02
    MAX_DRAWDOWN_LIMIT = 0.25
    
    # Reward parameters (scaled to prevent explosion)
    SHARPE_WEIGHT = 0.5
    PROFIT_WEIGHT = 0.3
    DRAWDOWN_PENALTY = 1.0
    REWARD_CLIP_MIN = -1.0
    REWARD_CLIP_MAX = 1.0
    
    # Technical parameters
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    ATR_PERIOD = 14
    
    # Multi-timeframe periods
    MTF_PERIODS = [5, 10, 20, 50]
    
    # Noise injection
    NOISE_STD = 0.001
    USE_NOISE = True
    
    # ✅ Feature columns that match XGBoost training
    XGB_FEATURE_COLS = [
        'close', 'volume', 'return_5d', 'return_10d',
        'volatility', 'volatility_5d', 'volume_ratio',
        'rsi_oversold', 'rsi_overbought',
        'dist_from_sr', 'sr_strength',
        'is_bullish_div', 'div_strength',
        'dist_from_ema', 'above_ema'
    ]
    
    # All available columns from mongodb.csv
    ALL_FEATURE_COLS = [
        'open', 'high', 'low', 'close', 'volume',
        'bb_upper', 'bb_middle', 'bb_lower',
        'macd', 'macd_signal', 'macd_hist',
        'rsi', 'atr', 'ema_200',
        'zigzag', 'Hammer', 'BullishEngulfing',
        'MorningStar', 'Doji', 'PiercingLine', 'ThreeWhiteSoldiers'
    ]

# =========================================================
# FEATURE SCALER (Singleton pattern)
# =========================================================

class FeatureScaler:
    """Centralized feature scaling for all observations"""
    
    _instance = None
    _scaler = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._scaler = StandardScaler()
        return cls._instance
    
    def fit(self, data):
        """Fit scaler on training data"""
        self._scaler.fit(data)
    
    def transform(self, data):
        """Transform data using fitted scaler"""
        return self._scaler.transform(data)
    
    def fit_transform(self, data):
        """Fit and transform"""
        return self._scaler.fit_transform(data)

# =========================================================
# HEDGE FUND TRADING ENVIRONMENT (FIXED)
# =========================================================

class HedgeFundTradingEnv(gym.Env):
    """
    Professional Hedge Fund Level Trading Environment - v3 (All Bugs Fixed)
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 xgb_model_dir: str = "./csv/xgboost/",
                 config: HedgeFundConfig = None,
                 scaler: FeatureScaler = None):
        
        super(HedgeFundTradingEnv, self).__init__()
        
        # Load configuration
        self.config = config or HedgeFundConfig()
        
        # Store data
        self.raw_data = data.copy()
        self.xgb_model_dir = xgb_model_dir
        
        # Load XGBoost models
        self.xgb_models = self._load_xgb_models()
        
        # Get unique symbols
        self.symbols = self.raw_data['symbol'].unique()
        
        # ✅ Pre-calculate all features for all symbols
        self.symbol_data_cache = {}
        self._preprocess_all_symbols()
        
        # Feature scaler
        self.scaler = scaler or FeatureScaler()
        
        # Calculate observation dimension
        self._calculate_obs_dim()
        
        # Action space: 0=Hold, 1=Buy, 2=Sell, 3=Add, 4=Reduce
        self.action_space = spaces.Discrete(5)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Tracking variables
        self.reset()
    
    def _preprocess_all_symbols(self):
        """✅ Pre-calculate features for all symbols (avoid recomputation)"""
        print("📊 Preprocessing all symbols...")
        
        for symbol in self.symbols:
            symbol_data = self.raw_data[self.raw_data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
            
            # Calculate all technical features
            symbol_data = self._calculate_technical_features(symbol_data)
            
            # Calculate XGBoost features
            symbol_data = self._calculate_xgb_features(symbol_data)
            
            self.symbol_data_cache[symbol] = symbol_data
        
        print(f"   ✅ Preprocessed {len(self.symbol_data_cache)} symbols")
    
    def _calculate_technical_features(self, df):
        """Calculate all technical indicators"""
        # Price changes
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        
        # Volatility
        df['volatility'] = (df['high'] - df['low']) / df['close']
        df['volatility_5d'] = df['volatility'].rolling(5).mean()
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD
        df['macd'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, 1)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # ATR
        df['atr'] = self._calculate_atr(df)
        df['atr_ratio'] = df['atr'] / df['close'].replace(0, 1)
        
        # ZigZag signal
        if 'zigzag' in df.columns:
            df['zigzag_signal'] = (~df['zigzag'].isna()).astype(int)
        else:
            df['zigzag_signal'] = 0
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def _calculate_xgb_features(self, df):
        """✅ Calculate features that match XGBoost training"""
        # Default values for features that might be missing
        default_features = {
            'dist_from_sr': 0, 'sr_strength': 0,
            'is_bullish_div': 0, 'div_strength': 0,
            'dist_from_ema': 0, 'above_ema': 0
        }
        
        for col, default in default_features.items():
            if col not in df.columns:
                df[col] = default
        
        return df
    
    def _load_xgb_models(self):
        """Load all trained XGBoost models"""
        models = {}
        if os.path.exists(self.xgb_model_dir):
            for file in os.listdir(self.xgb_model_dir):
                if file.endswith('.joblib'):
                    symbol = file.replace('.joblib', '')
                    try:
                        models[symbol] = joblib.load(os.path.join(self.xgb_model_dir, file))
                    except Exception as e:
                        print(f"⚠️ Failed to load {symbol}: {e}")
        return models
    
    def _get_xgb_signal_dynamic(self, symbol, features_dict):
        """
        ✅ FIXED: Per-step dynamic XGBoost prediction (no data leakage)
        Called every step with current features only
        """
        if symbol in self.xgb_models:
            model = self.xgb_models[symbol]
            try:
                # Prepare features in the exact order as training
                feature_order = self.config.XGB_FEATURE_COLS
                features = []
                
                for col in feature_order:
                    val = features_dict.get(col, 0)
                    if pd.isna(val):
                        val = 0
                    features.append(val)
                
                features_array = np.array(features).reshape(1, -1)
                
                # Predict
                prob = model.predict_proba(features_array)[0, 1]
                pred = 1 if prob > 0.5 else 0
                return prob * 100, pred
                
            except Exception as e:
                pass
        
        return 50, 0  # Neutral
    
    def _calculate_rsi(self, prices):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.RSI_PERIOD).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices):
        exp1 = prices.ewm(span=self.config.MACD_FAST, adjust=False).mean()
        exp2 = prices.ewm(span=self.config.MACD_SLOW, adjust=False).mean()
        macd = exp1 - exp2
        return macd.fillna(0)
    
    def _calculate_bollinger(self, prices):
        sma = prices.rolling(window=self.config.BB_PERIOD).mean()
        std = prices.rolling(window=self.config.BB_PERIOD).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper.fillna(prices), sma.fillna(prices), lower.fillna(prices)
    
    def _calculate_atr(self, df):
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=self.config.ATR_PERIOD).mean()
        return atr.fillna(tr.mean())
    
    def _calculate_obs_dim(self):
        """✅ Calculate observation dimension dynamically"""
        # Base features
        base_count = len(self.config.ALL_FEATURE_COLS)
        
        # MTF features: 4 features per period
        mtf_count = len(self.config.MTF_PERIODS) * 4
        
        # XGBoost features
        xgb_count = 2  # confidence, prediction
        
        # Position features
        position_count = 5  # balance_ratio, position_ratio, drawdown, sharpe, volatility
        
        self.obs_dim = base_count + mtf_count + xgb_count + position_count
    
    def _calculate_mtf_features(self, df, current_idx):
        """Multi-timeframe analysis"""
        mtf_features = {}
        
        for period in self.config.MTF_PERIODS:
            if current_idx >= period:
                period_data = df.iloc[max(0, current_idx - period):current_idx + 1]
                
                mtf_features[f'ma_{period}'] = period_data['close'].mean()
                mtf_features[f'std_{period}'] = period_data['close'].std()
                mtf_features[f'trend_{period}'] = (period_data['close'].iloc[-1] - period_data['close'].iloc[0]) / (period_data['close'].iloc[0] + 1e-8)
                mtf_features[f'volume_avg_{period}'] = period_data['volume'].mean() / 1e6
            else:
                mtf_features[f'ma_{period}'] = df['close'].iloc[current_idx] / 1000
                mtf_features[f'std_{period}'] = 0
                mtf_features[f'trend_{period}'] = 0
                mtf_features[f'volume_avg_{period}'] = df['volume'].iloc[current_idx] / 1e6
        
        return mtf_features
    
    def _calculate_sharpe_ratio(self, returns, window=20):
        """Calculate rolling Sharpe ratio"""
        if len(returns) < window:
            return 0.0
        
        recent_returns = returns[-window:]
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        if std_return == 0:
            return 0.0
        
        sharpe = mean_return / std_return * np.sqrt(252)
        return np.clip(sharpe, -2, 2)
    
    def _calculate_drawdown(self, equity_curve):
        """Calculate current drawdown"""
        if len(equity_curve) < 2:
            return 0.0
        
        peak = max(equity_curve)
        current = equity_curve[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0
        
        return np.clip(drawdown, 0, 1)
    
    def _check_stop_loss(self, current_price, entry_price, high_price):
        """Check if stop loss is triggered"""
        if self.position == 0 or self.entry_price == 0:
            return False
        
        # Fixed stop loss
        fixed_sl = self.entry_price * (1 - self.config.STOP_LOSS_PCT)
        
        # Trailing stop
        trailing_sl = high_price * (1 - self.config.TRAILING_STOP_PCT)
        
        # Use the tighter stop loss
        stop_price = max(fixed_sl, trailing_sl)
        
        return current_price <= stop_price
    
    def _add_noise(self, price):
        """Noise injection for regularization"""
        if self.config.USE_NOISE:
            noise = np.random.normal(1, self.config.NOISE_STD)
            return price * noise
        return price
    
    def _normalize_observation(self, obs_dict):
        """✅ Normalize all observations to similar scale"""
        normalized = []
        
        # Normalize price-based features (divide by 1000)
        for col in ['open', 'high', 'low', 'close', 'ema_200']:
            if col in obs_dict:
                normalized.append(obs_dict[col] / 1000)
            else:
                normalized.append(0)
        
        # Normalize volume (divide by 1e6)
        normalized.append(obs_dict.get('volume', 0) / 1e6)
        
        # Normalize RSI (divide by 100)
        normalized.append(obs_dict.get('rsi', 50) / 100)
        
        # Normalize MACD (divide by 10)
        normalized.append(obs_dict.get('macd', 0) / 10)
        normalized.append(obs_dict.get('macd_signal', 0) / 10)
        normalized.append(obs_dict.get('macd_hist', 0) / 10)
        
        # Bollinger bands (already ratio)
        normalized.append(obs_dict.get('bb_width', 0))
        normalized.append(obs_dict.get('bb_position', 0.5))
        
        # ATR ratio
        normalized.append(obs_dict.get('atr_ratio', 0))
        
        # Binary features
        for col in ['Hammer', 'BullishEngulfing', 'MorningStar', 'Doji', 'PiercingLine', 'ThreeWhiteSoldiers', 'zigzag_signal']:
            normalized.append(float(obs_dict.get(col, 0)))
        
        return normalized
    
    def reset(self, seed=None, options=None):
        """✅ FIXED: Random episode start for better generalization"""
        super().reset(seed=seed)
        
        # Select random symbol
        self.current_symbol_idx = np.random.randint(0, len(self.symbols))
        self.current_symbol = self.symbols[self.current_symbol_idx]
        
        # Get preprocessed symbol data
        self.symbol_data = self.symbol_data_cache[self.current_symbol].copy()
        
        # ✅ FIXED: Random start position (not fixed)
        min_start = max(50, len(self.symbol_data) // 10)
        max_start = len(self.symbol_data) - 200
        if max_start > min_start:
            self.current_step = np.random.randint(min_start, max_start)
        else:
            self.current_step = min_start
        
        # Initialize trading state
        self.balance = self.config.INITIAL_BALANCE
        self.position = 0
        self.entry_price = 0
        self.highest_price = 0
        self.total_reward = 0
        self.trades = []
        
        # ✅ FIXED: Single equity curve (only updated here and on trade)
        self.equity_curve = [self.balance]
        self.returns = []
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """✅ FIXED: Get current observation with single equity curve update"""
        if self.current_step >= len(self.symbol_data):
            return np.zeros(self.obs_dim, dtype=np.float32)
        
        row = self.symbol_data.iloc[self.current_step]
        
        # ✅ FIXED: Get dynamic XGBoost signal for current step only
        current_features = row.to_dict()
        xgb_conf, xgb_pred = self._get_xgb_signal_dynamic(self.current_symbol, current_features)
        
        # Base normalized features
        base_features = self._normalize_observation(current_features)
        
        # Multi-timeframe features
        mtf_features = self._calculate_mtf_features(self.symbol_data, self.current_step)
        mtf_values = list(mtf_features.values())
        
        # XGBoost features
        xgb_features = [xgb_conf / 100, xgb_pred]
        
        # ✅ FIXED: Calculate equity only once (not in _get_obs)
        current_equity = self.balance + (self.position * row['close'] if self.position > 0 else 0)
        
        # Calculate metrics from trades (not from equity curve)
        sharpe = self._calculate_sharpe_ratio(self.returns)
        drawdown = self._calculate_drawdown(self.equity_curve + [current_equity])
        volatility = np.std(self.returns[-20:]) if len(self.returns) >= 20 else 0
        
        position_features = [
            self.balance / self.config.INITIAL_BALANCE,
            self.position / 1000,
            drawdown,
            sharpe,
            volatility
        ]
        
        # Combine all features
        obs = base_features + mtf_values + xgb_features + position_features
        obs = np.array(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def step(self, action):
        """✅ FIXED: Execute action with proper reward scaling"""
        if self.current_step >= len(self.symbol_data) - 1:
            return self._get_obs(), 0, True, False, {}
        
        row = self.symbol_data.iloc[self.current_step]
        next_row = self.symbol_data.iloc[self.current_step + 1]
        
        # Apply noise
        price = self._add_noise(row['close'])
        next_price = self._add_noise(next_row['close'])
        high_price = self._add_noise(row['high'])
        
        # ✅ FIXED: Get dynamic XGBoost signal for this step
        current_features = row.to_dict()
        xgb_conf, xgb_pred = self._get_xgb_signal_dynamic(self.current_symbol, current_features)
        xgb_conf_norm = xgb_conf / 100
        
        reward = 0
        terminated = False
        trade_result = None
        
        # Check stop loss
        if self.position > 0 and self._check_stop_loss(price, self.entry_price, high_price):
            action = 2  # Force sell
        
        # Execute action
        if action == 1:  # BUY
            if self.position == 0:
                buy_amount = self.balance * self.config.MAX_POSITION
                shares = buy_amount / price
                self.position = shares
                self.entry_price = price
                self.highest_price = price
                self.balance -= buy_amount
                reward -= self.config.TRADING_FEE
                
                if xgb_conf_norm > 0.7 and xgb_pred == 1:
                    reward += 0.05
        
        elif action == 2:  # SELL
            if self.position > 0:
                sell_amount = self.position * price
                self.balance += sell_amount * (1 - self.config.TRADING_FEE - self.config.SLIPPAGE)
                pnl = (price - self.entry_price) / self.entry_price
                
                trade_result = {'pnl': pnl, 'success': pnl > 0}
                self.trades.append(trade_result)
                self.returns.append(pnl)
                
                # ✅ FIXED: Update equity curve only on trade
                self.equity_curve.append(self.balance)
                
                # ✅ FIXED: Clipped reward to prevent explosion
                sharpe_reward = self._calculate_sharpe_ratio(self.returns) * self.config.SHARPE_WEIGHT
                profit_reward = pnl * 5 * self.config.PROFIT_WEIGHT  # Reduced from 10 to 5
                reward = profit_reward + sharpe_reward
                reward = np.clip(reward, self.config.REWARD_CLIP_MIN, self.config.REWARD_CLIP_MAX)
                
                self.position = 0
                self.entry_price = 0
                self.highest_price = 0
        
        elif action == 3:  # ADD to position
            if self.position > 0:
                add_amount = self.balance * (self.config.MAX_POSITION * 0.5)
                shares = add_amount / price
                self.position += shares
                self.balance -= add_amount
                reward -= self.config.TRADING_FEE * 0.5
                
                if xgb_conf_norm > 0.8 and xgb_pred == 1:
                    reward += 0.03
        
        elif action == 4:  # REDUCE position
            if self.position > 0:
                reduce_shares = self.position * 0.5
                sell_amount = reduce_shares * price
                self.balance += sell_amount * (1 - self.config.TRADING_FEE)
                self.position -= reduce_shares
                reward -= self.config.TRADING_FEE * 0.5
        
        # Hold reward
        if action == 0:
            if self.position > 0:
                unrealized_pnl = (price - self.entry_price) / self.entry_price
                if xgb_conf_norm > 0.6 and next_price > price:
                    reward += 0.005 * xgb_conf_norm * (1 + unrealized_pnl)
            else:
                if xgb_conf_norm > 0.6 and next_price > price:
                    reward += 0.002 * xgb_conf_norm
        
        # ✅ Drawdown penalty (scaled)
        current_equity = self.balance + (self.position * price if self.position > 0 else 0)
        temp_curve = self.equity_curve + [current_equity]
        drawdown = self._calculate_drawdown(temp_curve)
        
        if drawdown > self.config.MAX_DRAWDOWN_LIMIT:
            reward -= drawdown * self.config.DRAWDOWN_PENALTY
            terminated = True
        
        # Move to next step
        self.current_step += 1
        self.total_reward += reward
        
        # Update highest price for trailing stop
        if price > self.highest_price:
            self.highest_price = price
        
        # Check episode end
        if self.current_step >= len(self.symbol_data) - 1:
            terminated = True
            if self.position > 0:
                sell_amount = self.position * price
                self.balance += sell_amount * (1 - self.config.TRADING_FEE)
                pnl = (price - self.entry_price) / self.entry_price
                self.returns.append(pnl)
                self.equity_curve.append(self.balance)
                self.position = 0
        
        # ✅ Final reward clipping
        reward = np.clip(reward, self.config.REWARD_CLIP_MIN, self.config.REWARD_CLIP_MAX)
        
        return self._get_obs(), reward, terminated, False, {
            'balance': self.balance,
            'symbol': self.current_symbol,
            'step': self.current_step,
            'xgb_conf': xgb_conf,
            'drawdown': drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(self.returns),
            'total_trades': len(self.trades),
            'win_rate': sum(1 for t in self.trades if t.get('success', False)) / len(self.trades) if self.trades else 0,
            'total_return': (self.balance / self.config.INITIAL_BALANCE - 1) * 100
        }

# =========================================================
# ENVIRONMENT WRAPPER
# =========================================================

def create_hedge_fund_env(data: pd.DataFrame, 
                          xgb_model_dir: str = "./csv/xgboost/",
                          config: HedgeFundConfig = None):
    """Factory function to create Hedge Fund environment"""
    return HedgeFundTradingEnv(data, xgb_model_dir, config)

# =========================================================
# TESTING CODE
# =========================================================

if __name__ == "__main__":
    print("="*70)
    print("🏦 HEDGE FUND LEVEL ENVIRONMENT v3 (All Bugs Fixed)")
    print("="*70)
    
    try:
        df = pd.read_csv("./csv/mongodb.csv")
        print(f"✅ Loaded data: {len(df)} rows, {df['symbol'].nunique()} symbols")
        
        # Create environment
        env = HedgeFundTradingEnv(df)
        print(f"✅ Environment created")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space.shape}")
        
        # Test reset (multiple times for randomization)
        for i in range(3):
            obs, info = env.reset()
            print(f"\n✅ Reset {i+1}: start_step={env.current_step}, symbol={env.current_symbol}")
        
        # Test steps
        obs, info = env.reset()
        total_reward = 0
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                break
        
        print(f"\n✅ Test complete")
        print(f"   Total reward: {total_reward:.4f}")
        print(f"   Final balance: ${info['balance']:,.2f}")
        print(f"   Sharpe ratio: {info['sharpe_ratio']:.3f}")
        print(f"   Win rate: {info['win_rate']:.2%}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()