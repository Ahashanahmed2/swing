# train_ppo_xgboost.py - Complete Training Script with Advanced Features + Telegram Bot
# ✅ Multi-Agent Ensemble, Risk Management, Performance Metrics, Portfolio Optimizer
# ✅ Adaptive LR Scheduler, Experience Replay, Market Regime Detector, Hyperparameter Optimizer, Backtesting Engine
# ✅ Telegram Bot Notifications
# ✅ NEW: PPO Checkpoint Callback (every 20,000 steps)

import os
import numpy as np
import pandas as pd
import torch
import joblib
import random
import requests
from datetime import datetime
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import warnings
warnings.filterwarnings('ignore')

# Optional Optuna import
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not installed. Hyperparameter optimization disabled.")

# =========================
# TELEGRAM NOTIFICATION FUNCTION
# =========================

def send_telegram_message(message, token=None, chat_id=None, parse_mode='HTML'):
    """টেলিগ্রামে মেসেজ পাঠান"""
    token = token or os.getenv("TELEGRAM_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("⚠️ Telegram credentials not found")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": parse_mode
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.json()
    except Exception as e:
        print(f"⚠️ Telegram send failed: {e}")
        return False


class TelegramNotifier:
    """টেলিগ্রাম নোটিফিকেশন ম্যানেজার"""
    
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)
        
        if self.enabled:
            print("✅ Telegram notifications enabled")
        else:
            print("⚠️ Telegram notifications disabled (missing credentials)")
    
    def send(self, message, parse_mode='HTML'):
        """মেসেজ পাঠান"""
        return send_telegram_message(message, self.token, self.chat_id, parse_mode)
    
    def send_training_start(self, total_symbols, total_timesteps):
        """ট্রেনিং শুরু নোটিফিকেশন"""
        msg = f"""
🚀 <b>PPO + XGBoost Training Started</b>
📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
💰 Initial Balance: ${INITIAL_BALANCE:,.2f}
📊 Symbols to train: {total_symbols}
📈 Total Timesteps per symbol: {total_timesteps:,}
"""
        return self.send(msg)
    
    def send_training_complete(self, trained_count, best_symbol=None, best_return=None):
        """ট্রেনিং সম্পূর্ণ নোটিফিকেশন"""
        msg = f"""
✅ <b>PPO + XGBoost Training Completed</b>
📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🤖 Agents Trained: {trained_count}
"""
        if best_symbol and best_return:
            msg += f"\n🏆 Best Performer: {best_symbol} ({best_return:.1f}% return)"
        
        return self.send(msg)
    
    def send_symbol_training_start(self, symbol, step, total):
        """সিম্বল ট্রেনিং শুরু"""
        msg = f"""
🎯 <b>Training PPO for {symbol}</b>
📊 Progress: {step}/{total}
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self.send(msg)
    
    def send_symbol_training_complete(self, symbol, metrics):
        """সিম্বল ট্রেনিং সম্পূর্ণ"""
        win_rate = metrics.get('win_rate', 0)
        total_return = metrics.get('total_return', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        
        msg = f"""
✅ <b>{symbol} Training Complete</b>
📊 Win Rate: {win_rate:.1f}%
📈 Total Return: {total_return:.1f}%
📐 Sharpe Ratio: {sharpe:.2f}
💾 Model saved: ppo_{symbol}
"""
        return self.send(msg)
    
    def send_error(self, symbol, error_msg):
        """এরর নোটিফিকেশন"""
        msg = f"""
⚠️ <b>Error Training {symbol}</b>
❌ {error_msg[:200]}
📅 {datetime.now().strftime('%H:%M:%S')}
"""
        return self.send(msg)
    
    def send_portfolio_summary(self, summary_df):
        """পোর্টফোলিও সামারি"""
        if summary_df.empty:
            return
        
        avg_return = summary_df['total_return'].mean()
        avg_win_rate = summary_df['win_rate'].mean()
        top_3 = summary_df.nlargest(3, 'total_return')
        
        msg = f"""
📊 <b>Portfolio Performance Summary</b>
────────────────────────────────
📈 Average Return: {avg_return:.1f}%
🎯 Average Win Rate: {avg_win_rate:.1f}%
🤖 Total Agents: {len(summary_df)}

🏆 <b>Top 3 Performers:</b>
"""
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            msg += f"\n{i}. {row['symbol']}: {row['total_return']:.1f}% return, {row['win_rate']:.1f}% win rate"
        
        return self.send(msg)
    
    def send_backtest_result(self, symbol, metrics):
        """ব্যাকটেস্ট রেজাল্ট"""
        msg = f"""
📊 <b>Backtest Result: {symbol}</b>
────────────────────────────────
🎯 Win Rate: {metrics.get('win_rate', 0):.1f}%
📈 Total Return: {metrics.get('total_return', 0):.1f}%
📐 Sharpe: {metrics.get('sharpe_ratio', 0):.2f}
💼 Total Trades: {metrics.get('total_trades', 0)}
📉 Max Drawdown: {metrics.get('max_drawdown', 0):.1f}%
"""
        return self.send(msg)


# =========================
# CONFIGURATION
# =========================
XGB_MODEL_DIR = "./csv/xgboost/"
DATA_PATH = "./csv/mongodb.csv"
PPO_MODEL_DIR = "./csv/ppo_models/"
LOG_DIR = "./logs/ppo/"
ENSEMBLE_MODEL_DIR = "./csv/ppo_models/ensemble/"
PERFORMANCE_LOG_DIR = "./csv/performance/"

os.makedirs(PPO_MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ENSEMBLE_MODEL_DIR, exist_ok=True)
os.makedirs(PERFORMANCE_LOG_DIR, exist_ok=True)

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

# Risk Management Parameters
MAX_DRAWDOWN_LIMIT = 0.20  # 20% max drawdown
VAR_LIMIT = 0.02  # 2% VaR limit
SHARPE_MIN = 0.5  # Minimum Sharpe ratio

# Experience Replay Parameters
REPLAY_BUFFER_CAPACITY = 10000
REPLAY_BATCH_SIZE = 128

# ✅ NEW: Checkpoint config
CHECKPOINT_SAVE_FREQ = 20000  # Save checkpoint every 20,000 steps
CHECKPOINT_DIR = "./csv/ppo_models/checkpoints/"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Global Telegram Notifier
telegram = TelegramNotifier()

# =========================
# RISK MANAGEMENT MODULE
# =========================

class RiskManager:
    """Dynamic risk management based on volatility"""
    
    def __init__(self):
        self.max_drawdown_limit = MAX_DRAWDOWN_LIMIT
        self.var_limit = VAR_LIMIT
        self.sharpe_min = SHARPE_MIN
        self.peak_balance = INITIAL_BALANCE
        self.current_drawdown = 0
    
    def calculate_position_size(self, balance, volatility, confidence):
        """Kelly Criterion based position sizing"""
        base_size = balance * 0.1  # 10% base
        
        # Adjust for volatility
        vol_adjustment = 1.0 / (1 + volatility * 10)
        
        # Adjust for confidence
        conf_adjustment = confidence / 100
        
        # Kelly formula: f = (bp - q) / b
        win_prob = confidence / 100
        loss_prob = 1 - win_prob
        kelly = (win_prob * 2 - loss_prob) / 2
        kelly = max(0.01, min(0.25, kelly))  # Cap at 25%
        
        position = balance * kelly * vol_adjustment
        return min(position, balance * MAX_POSITION)
    
    def check_stop_loss(self, entry_price, current_price, atr):
        """Dynamic stop loss based on ATR"""
        stop_loss = entry_price - 2 * atr
        return current_price <= stop_loss
    
    def check_take_profit(self, entry_price, current_price, atr, rrr=2):
        """Risk-Reward based take profit"""
        take_profit = entry_price + (rrr * 2 * atr)
        return current_price >= take_profit
    
    def update_drawdown(self, current_balance):
        """Update current drawdown"""
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
        return self.current_drawdown
    
    def should_stop_trading(self, current_balance):
        """Check if should stop due to max drawdown"""
        dd = self.update_drawdown(current_balance)
        return dd >= self.max_drawdown_limit


# =========================
# PERFORMANCE METRICS TRACKER
# =========================

class PerformanceTracker:
    """Track all trading metrics"""
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.trades = []
        self.daily_returns = []
        self.equity_curve = [INITIAL_BALANCE]
        self.winning_streak = 0
        self.losing_streak = 0
        self.current_streak = 0
        
    def add_trade(self, trade):
        self.trades.append(trade)
        self.equity_curve.append(trade['balance_after'])
        
        # Update streak
        if trade['pnl'] > 0:
            if self.current_streak > 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
            self.winning_streak = max(self.winning_streak, self.current_streak)
        else:
            if self.current_streak < 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1
            self.losing_streak = max(self.losing_streak, abs(self.current_streak))
        
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {}
        
        returns = [t['pnl_percent'] for t in self.trades]
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r <= 0]
        
        return {
            'symbol': self.symbol,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100 if self.trades else 0,
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe(returns),
            'profit_factor': self._calculate_profit_factor(),
            'expectancy': np.mean(returns) if returns else 0,
            'winning_streak': self.winning_streak,
            'losing_streak': self.losing_streak,
            'final_balance': self.equity_curve[-1],
            'total_return': (self.equity_curve[-1] / INITIAL_BALANCE - 1) * 100 if self.equity_curve else 0
        }
    
    def _calculate_max_drawdown(self):
        if len(self.equity_curve) < 2:
            return 0
        peak = self.equity_curve[0]
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
        return max_dd * 100
    
    def _calculate_sharpe(self, returns, risk_free=0.02):
        if len(returns) < 2:
            return 0
        excess = np.mean(returns) / 100 - risk_free/252
        std = np.std(returns) / 100
        return excess / std * np.sqrt(252) if std > 0 else 0
    
    def _calculate_profit_factor(self):
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def save_report(self):
        """Save performance report to CSV"""
        metrics = self.calculate_metrics()
        df = pd.DataFrame([metrics])
        filepath = os.path.join(PERFORMANCE_LOG_DIR, f"{self.symbol}_performance.csv")
        df.to_csv(filepath, index=False)
        return metrics


# =========================
# PORTFOLIO OPTIMIZER
# =========================

class PortfolioOptimizer:
    """Optimize portfolio allocation across symbols"""
    
    def __init__(self, symbols_data):
        self.symbols_data = symbols_data
        self.correlation_matrix = None
        
    def calculate_correlation(self):
        """Calculate correlation matrix between symbols"""
        returns_dict = {}
        for symbol, data in self.symbols_data.items():
            if len(data) > 50:
                returns_dict[symbol] = data['close'].pct_change().dropna()
        
        if len(returns_dict) < 2:
            return None
        
        returns_df = pd.DataFrame(returns_dict)
        self.correlation_matrix = returns_df.corr()
        return self.correlation_matrix
    
    def optimize_weights(self, expected_returns, method='sharpe'):
        """Optimize portfolio weights"""
        if method == 'sharpe':
            weights = self._max_sharpe_weights(expected_returns)
        elif method == 'min_variance':
            weights = self._min_variance_weights()
        elif method == 'equal':
            weights = {sym: 1.0/len(expected_returns) for sym in expected_returns.keys()}
        else:
            weights = expected_returns
        
        return weights
    
    def _max_sharpe_weights(self, expected_returns):
        """Calculate max Sharpe ratio weights"""
        total = sum(expected_returns.values())
        if total == 0:
            n = len(expected_returns)
            return {sym: 1.0/n for sym in expected_returns.keys()}
        return {sym: val/total for sym, val in expected_returns.items()}
    
    def _min_variance_weights(self):
        """Calculate minimum variance weights"""
        n = len(self.symbols_data)
        return {sym: 1.0/n for sym in self.symbols_data.keys()}
    
    def get_diversified_symbols(self, max_correlation=0.7):
        """Select diversified symbols"""
        if self.correlation_matrix is None:
            self.calculate_correlation()
        
        if self.correlation_matrix is None:
            return list(self.symbols_data.keys())
        
        selected = []
        for sym in self.correlation_matrix.columns:
            if not selected:
                selected.append(sym)
            else:
                max_corr = max(self.correlation_matrix.loc[sym, selected].abs())
                if max_corr < max_correlation:
                    selected.append(sym)
        
        return selected


# =========================
# ADAPTIVE LEARNING RATE SCHEDULER
# =========================

class AdaptiveLRScheduler:
    """Adaptive learning rate based on performance"""
    
    def __init__(self, initial_lr=LEARNING_RATE, patience=10, factor=0.5):
        self.lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.best_reward = -np.inf
        self.epochs_without_improvement = 0
        self.history = []
    
    def update(self, current_reward):
        """Update learning rate based on performance"""
        self.history.append(current_reward)
        
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        if self.epochs_without_improvement >= self.patience:
            self.lr *= self.factor
            self.epochs_without_improvement = 0
            print(f"   📉 Reducing learning rate to {self.lr:.6f}")
        
        return self.lr
    
    def get_lr(self):
        return self.lr


# =========================
# EXPERIENCE REPLAY BUFFER
# =========================

class ExperienceReplayBuffer:
    """Store and sample past experiences"""
    
    def __init__(self, capacity=REPLAY_BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        return len(self) >= batch_size


# =========================
# MARKET REGIME DETECTOR
# =========================

class MarketRegimeDetector:
    """Detect market regime for adaptive strategy"""
    
    def __init__(self):
        self.regimes = ['TRENDING_UP', 'TRENDING_DOWN', 'VOLATILE', 'RANGING', 'UNKNOWN']
    
    def detect(self, data, idx):
        """Detect current market regime"""
        if idx < 50:
            return 'UNKNOWN'
        
        recent = data.iloc[idx-50:idx]
        
        # Calculate metrics
        returns = recent['close'].pct_change().dropna()
        if len(returns) < 10:
            return 'UNKNOWN'
        
        trend = returns.mean() * 252  # Annualized trend
        volatility = returns.std() * np.sqrt(252)
        adx = self._calculate_adx(recent)
        
        if volatility > 0.3:
            return 'VOLATILE'
        elif adx < 20:
            return 'RANGING'
        elif trend > 0.1:
            return 'TRENDING_UP'
        elif trend < -0.1:
            return 'TRENDING_DOWN'
        else:
            return 'RANGING'
    
    def _calculate_adx(self, df, period=14):
        """Calculate ADX indicator"""
        if len(df) < period:
            return 20
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([high - low, (high - close.shift()).abs(), 
                       (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1] if not adx.empty and not pd.isna(adx.iloc[-1]) else 20
    
    def get_strategy_adjustment(self, regime):
        """Get strategy adjustment based on regime"""
        adjustments = {
            'TRENDING_UP': {'position_multiplier': 1.2, 'stop_loss_multiplier': 1.5},
            'TRENDING_DOWN': {'position_multiplier': 0.8, 'stop_loss_multiplier': 1.2},
            'VOLATILE': {'position_multiplier': 0.5, 'stop_loss_multiplier': 2.0},
            'RANGING': {'position_multiplier': 0.7, 'stop_loss_multiplier': 1.0},
            'UNKNOWN': {'position_multiplier': 0.5, 'stop_loss_multiplier': 1.0}
        }
        return adjustments.get(regime, adjustments['UNKNOWN'])


# =========================
# BACKTESTING ENGINE
# =========================

class BacktestEngine:
    """Comprehensive backtesting with transaction costs"""
    
    def __init__(self, initial_capital=INITIAL_BALANCE, commission=TRADING_FEE, slippage=0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_manager = RiskManager()
        self.performance_tracker = None
        self.regime_detector = MarketRegimeDetector()
        
    def run(self, data, model, xgb_model, symbol="BACKTEST"):
        """Run backtest"""
        self.performance_tracker = PerformanceTracker(symbol)
        env = XGBoostPPOTradingEnv(data, symbol, xgb_model, self.risk_manager)
        obs, _ = env.reset()
        
        trades = []
        current_capital = self.initial_capital
        entry_price = 0
        
        while True:
            # Detect market regime
            if env.current_step < len(data):
                regime = self.regime_detector.detect(data, env.current_step)
                adjustment = self.regime_detector.get_strategy_adjustment(regime)
            else:
                regime = 'UNKNOWN'
                adjustment = {'position_multiplier': 1.0, 'stop_loss_multiplier': 1.0}
            
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Apply slippage to price
            current_price = data.iloc[env.current_step]['close'] if env.current_step < len(data) else data.iloc[-1]['close']
            if action[0] == 1:  # Buy
                execution_price = current_price * (1 + self.slippage)
            elif action[0] == 2:  # Sell
                execution_price = current_price * (1 - self.slippage)
            else:
                execution_price = current_price
            
            # Adjust position size based on regime
            if action[0] == 1:
                env.max_position_override = MAX_POSITION * adjustment['position_multiplier']
                entry_price = execution_price
            
            obs, reward, terminated, truncated, info = env.step(action[0])
            
            # Track trade
            if action[0] == 2 and env.position == 0:  # Sell completed
                pnl = (execution_price - entry_price) / entry_price * 100
                trade = {
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'exit_price': execution_price,
                    'pnl_percent': pnl,
                    'pnl': info['balance'] - current_capital,
                    'balance_after': info['balance'],
                    'regime': regime
                }
                self.performance_tracker.add_trade(trade)
                trades.append(trade)
                current_capital = info['balance']
            
            # Check risk limits
            if self.risk_manager.should_stop_trading(info['balance']):
                print(f"   ⚠️ Max drawdown reached, stopping backtest")
                break
            
            if terminated or truncated:
                break
        
        return self._analyze_results(trades)
    
    def _analyze_results(self, trades):
        """Analyze backtest results"""
        metrics = self.performance_tracker.calculate_metrics()
        
        # Add backtest-specific metrics
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            regimes = {}
            for t in trades:
                reg = t['regime']
                if reg not in regimes:
                    regimes[reg] = {'trades': 0, 'wins': 0, 'pnl': 0}
                regimes[reg]['trades'] += 1
                if t['pnl'] > 0:
                    regimes[reg]['wins'] += 1
                regimes[reg]['pnl'] += t['pnl_percent']
            
            metrics['regime_performance'] = regimes
        
        return metrics


# =========================
# MULTI-AGENT ENSEMBLE TRADING ENV
# =========================

class EnsembleTradingEnv(gym.Env):
    """Multiple agents voting system"""
    
    def __init__(self, data, symbol, xgb_model, ppo_models):
        super().__init__()
        self.data = data
        self.symbol = symbol
        self.xgb_model = xgb_model
        self.ppo_models = ppo_models
        self.risk_manager = RiskManager()
        self.performance_tracker = PerformanceTracker(symbol)
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 50
        self.balance = INITIAL_BALANCE
        self.position = 0
        self.entry_price = 0
        self.total_reward = 0
        self._calculate_features()
        return self._get_obs(), {}
    
    def _calculate_features(self):
        df = self.data
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['volatility'] = (df['high'] - df['low']) / df['close']
        df['volatility_5d'] = df['volatility'].rolling(5).mean()
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'] = self._calculate_macd(df['close'])
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger(df['close'])
        df['bb_distance'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['atr'] = self._calculate_atr(df)
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
        tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.fillna(tr.mean())
    
    def _get_xgb_signals(self, df):
        if self.xgb_model is None:
            return 50, 0
        try:
            features = df[['close', 'volume', 'return_5d', 'return_10d', 
                           'volatility', 'volatility_5d', 'volume_ratio']].fillna(0)
            prob = self.xgb_model.predict_proba(features)[:, 1]
            pred = (prob > 0.5).astype(int)
            return prob * 100, pred
        except:
            return 50, 0
    
    def _get_obs(self):
        if self.current_step >= len(self.data):
            return np.zeros(20, dtype=np.float32)
        
        row = self.data.iloc[self.current_step]
        
        obs = np.array([
            row['close'] / 1000, row['volume'] / 1000000,
            row.get('rsi', 50) / 100, row.get('macd', 0) / 10,
            row.get('xgb_confidence', 50) / 100, row.get('xgb_prediction', 0),
            self.balance / INITIAL_BALANCE, self.position / 1000,
            row.get('return_5d', 0), row.get('return_10d', 0),
            row.get('volatility', 0), row.get('volume_ratio', 1),
            row.get('bb_distance', 0.5), row.get('atr', 0) / row.get('close', 1),
            self.total_reward / 1000, 0, 0, 0, 0, 0
        ], dtype=np.float32)
        
        return obs
    
    def _get_ensemble_action(self, obs):
        """Get voted action from all agents"""
        votes = {0: 0, 1: 0, 2: 0}
        
        # XGBoost vote (weight=0.3)
        xgb_pred = int(obs[5])
        votes[xgb_pred] += 0.3
        
        # PPO agents votes (weight=0.7 total)
        if self.ppo_models:
            for name, model in self.ppo_models.items():
                action, _ = model.predict(obs, deterministic=True)
                votes[action[0]] += 0.7 / len(self.ppo_models)
        
        return max(votes, key=votes.get)
    
    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return self._get_obs(), 0, True, False, {}
        
        row = self.data.iloc[self.current_step]
        next_row = self.data.iloc[self.current_step + 1]
        
        price = row['close']
        next_price = next_row['close']
        xgb_conf = row.get('xgb_confidence', 50) / 100
        
        reward = 0
        terminated = False
        
        if action == 1:  # BUY
            if self.position == 0:
                volatility = row.get('volatility', 0.02)
                position_size = self.risk_manager.calculate_position_size(self.balance, volatility, xgb_conf * 100)
                shares = position_size / price
                self.position = shares
                self.entry_price = price
                self.balance -= position_size
                reward -= TRADING_FEE
        
        elif action == 2:  # SELL
            if self.position > 0:
                sell_amount = self.position * price
                self.balance += sell_amount * (1 - TRADING_FEE)
                pnl = (price - self.entry_price) / self.entry_price
                reward = pnl * 10
                
                # Track trade
                trade = {
                    'pnl_percent': pnl * 100,
                    'pnl': sell_amount - (self.position * self.entry_price),
                    'balance_after': self.balance
                }
                self.performance_tracker.add_trade(trade)
                
                self.position = 0
                self.entry_price = 0
        
        self.current_step += 1
        self.total_reward += reward
        
        if self.current_step >= len(self.data) - 1:
            terminated = True
            if self.position > 0:
                sell_amount = self.position * price
                self.balance += sell_amount * (1 - TRADING_FEE)
                self.position = 0
        
        info = {
            'balance': self.balance,
            'symbol': self.symbol,
            'total_return': (self.balance / INITIAL_BALANCE - 1) * 100
        }
        
        return self._get_obs(), reward, terminated, False, info


# =========================
# TRADING ENVIRONMENT (XGBoost + PPO)
# =========================

class XGBoostPPOTradingEnv(gym.Env):
    """Trading Environment with XGBoost Signals"""

    def __init__(self, data, symbol, xgb_model, risk_manager=None):
        super().__init__()

        self.data = data.copy()
        self.symbol = symbol
        self.xgb_model = xgb_model
        self.risk_manager = risk_manager or RiskManager()
        self.max_position_override = MAX_POSITION

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
        atr = row.get('atr', price * 0.02)

        reward = 0
        terminated = False

        # Execute action
        if action == 1:  # BUY
            if self.position == 0:
                volatility = row.get('volatility', 0.02)
                position_size = self.risk_manager.calculate_position_size(
                    self.balance, volatility, xgb_conf * 100
                )
                position_size = min(position_size, self.balance * self.max_position_override)
                
                shares = position_size / price
                self.position = shares
                self.entry_price = price
                self.balance -= position_size
                reward -= TRADING_FEE

                # Bonus for following strong XGBoost signal
                if xgb_conf > 0.65 and xgb_pred == 1:
                    reward += 0.05

        elif action == 2:  # SELL
            if self.position > 0:
                # Check if stop loss or take profit triggered
                stop_triggered = self.risk_manager.check_stop_loss(self.entry_price, price, atr)
                take_profit_triggered = self.risk_manager.check_take_profit(self.entry_price, price, atr)
                
                sell_amount = self.position * price
                self.balance += sell_amount * (1 - TRADING_FEE)
                pnl = (price - self.entry_price) / self.entry_price
                reward = pnl * 10
                
                # Penalize stop loss
                if stop_triggered:
                    reward -= 0.1
                # Bonus for take profit
                if take_profit_triggered:
                    reward += 0.1
                    
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

        # Check risk limits
        if self.risk_manager.should_stop_trading(self.balance):
            terminated = True

        info = {
            'balance': self.balance,
            'symbol': self.symbol,
            'step': self.current_step,
            'xgb_conf': xgb_conf,
            'total_return': (self.balance / INITIAL_BALANCE - 1) * 100
        }

        return self._get_obs(), reward, terminated, False, info


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
# HYPERPARAMETER OPTIMIZER
# =========================

def optimize_hyperparameters(symbol_data, xgb_model, n_trials=50):
    """Optimize PPO hyperparameters using Optuna"""
    if not OPTUNA_AVAILABLE:
        print("   ⚠️ Optuna not available, using default hyperparameters")
        return {
            'learning_rate': LEARNING_RATE,
            'n_steps': N_STEPS,
            'batch_size': BATCH_SIZE,
            'gamma': 0.99,
            'ent_coef': 0.01
        }
    
    def objective(trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        n_steps = trial.suggest_int('n_steps', 512, 4096, step=512)
        batch_size = trial.suggest_int('batch_size', 32, 256, step=32)
        gamma = trial.suggest_float('gamma', 0.9, 0.999)
        ent_coef = trial.suggest_float('ent_coef', 0.001, 0.1, log=True)
        
        # Create and train model
        env = XGBoostPPOTradingEnv(symbol_data, 'OPTIMIZE', xgb_model)
        env = DummyVecEnv([lambda: env])
        
        model = PPO(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            ent_coef=ent_coef,
            verbose=0,
            device="cpu"
        )
        
        model.learn(total_timesteps=20000)
        
        # Evaluate
        total_reward = 0
        obs = env.reset()
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action[0])
            total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            if done:
                break
        
        return total_reward
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params


# =========================
# TRAINING FUNCTION
# =========================

def train_ppo_for_symbol(symbol, data, xgb_model, optimize=False, step=0, total=0):
    """Train PPO agent for a single symbol"""

    print(f"\n{'='*60}")
    print(f"🎯 Training PPO for {symbol}")
    print(f"{'='*60}")
    
    # Send Telegram notification
    telegram.send_symbol_training_start(symbol, step, total)

    # Optimize hyperparameters if requested
    if optimize and OPTUNA_AVAILABLE:
        print("   🔧 Optimizing hyperparameters...")
        best_params = optimize_hyperparameters(data, xgb_model, n_trials=20)
        lr = best_params.get('learning_rate', LEARNING_RATE)
        n_steps = best_params.get('n_steps', N_STEPS)
        batch_size = best_params.get('batch_size', BATCH_SIZE)
        gamma = best_params.get('gamma', 0.99)
        ent_coef = best_params.get('ent_coef', 0.01)
        print(f"   ✅ Best params: LR={lr:.6f}, n_steps={n_steps}, batch={batch_size}")
    else:
        lr = LEARNING_RATE
        n_steps = N_STEPS
        batch_size = BATCH_SIZE
        gamma = 0.99
        ent_coef = 0.01

    # Create environment
    risk_mgr = RiskManager()
    env = XGBoostPPOTradingEnv(data, symbol, xgb_model, risk_mgr)
    env = Monitor(env, f"{LOG_DIR}/{symbol}/")
    env = DummyVecEnv([lambda: env])

    # Adaptive LR scheduler
    lr_scheduler = AdaptiveLRScheduler(initial_lr=lr)

    # PPO configuration
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        activation_fn=torch.nn.ReLU
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=N_EPOCHS,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device="cpu",
        tensorboard_log=LOG_DIR
    )

    # ✅ NEW: Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_SAVE_FREQ,
        save_path=f"{CHECKPOINT_DIR}/{symbol}/",
        name_prefix=f"ppo_{symbol}",
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    print(f"   📦 Checkpoints will be saved every {CHECKPOINT_SAVE_FREQ} steps to {CHECKPOINT_DIR}/{symbol}/")

    # Train
    print("   🚀 Training started...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[TensorboardCallback(), checkpoint_callback]
    )

    # Save final model
    save_path = os.path.join(PPO_MODEL_DIR, f"ppo_{symbol}")
    model.save(save_path)
    print(f"   ✅ Final model saved: {save_path}")

    return model


# =========================
# MAIN TRAINING LOOP
# =========================

def main():
    print("="*70)
    print("🚀 PPO + XGBoost Training System (Advanced + Telegram + Checkpoints)")
    print("="*70)
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💰 Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"📊 Total Timesteps per symbol: {TOTAL_TIMESTEPS:,}")
    print(f"📦 Checkpoint Save Frequency: {CHECKPOINT_SAVE_FREQ:,} steps")
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

    # Step 3: Portfolio optimization for symbol selection
    print("\n🎯 Step 3: Portfolio optimization for symbol selection...")
    
    symbols_data = {}
    for symbol in list(xgb_models.keys())[:20]:  # Limit to 20 for speed
        symbol_data = df[df['symbol'] == symbol].copy()
        if len(symbol_data) >= 100:
            symbols_data[symbol] = symbol_data.sort_values('date').reset_index(drop=True)
    
    portfolio_optimizer = PortfolioOptimizer(symbols_data)
    correlation_matrix = portfolio_optimizer.calculate_correlation()
    
    if correlation_matrix is not None:
        print(f"   ✅ Calculated correlation for {len(correlation_matrix)} symbols")
        diversified_symbols = portfolio_optimizer.get_diversified_symbols(max_correlation=0.7)
    else:
        diversified_symbols = list(symbols_data.keys())
    
    train_symbols = diversified_symbols[:10] if len(diversified_symbols) > 10 else diversified_symbols
    print(f"   ✅ Selected {len(train_symbols)} diversified symbols: {', '.join(train_symbols[:5])}...")

    # Send training start notification
    telegram.send_training_start(len(train_symbols), TOTAL_TIMESTEPS)

    # Step 4: Train PPO for each symbol
    print("\n🏆 Step 4: Training PPO agents...")
    print("="*70)

    trained_models = {}
    performance_summary = []
    total_symbols = len(train_symbols)

    for i, symbol in enumerate(train_symbols):
        try:
            # Get data for this symbol
            symbol_data = symbols_data.get(symbol)
            if symbol_data is None:
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('date').reset_index(drop=True)

            if len(symbol_data) < 100:
                print(f"\n⚠️ Skipping {symbol}: insufficient data ({len(symbol_data)} rows)")
                continue

            # Train PPO
            model = train_ppo_for_symbol(symbol, symbol_data, xgb_models.get(symbol), 
                                         optimize=False, step=i+1, total=total_symbols)
            trained_models[symbol] = model

            # Quick backtest
            print(f"\n   📊 Backtesting {symbol}...")
            backtest_engine = BacktestEngine()
            metrics = backtest_engine.run(symbol_data, model, xgb_models.get(symbol), symbol)
            
            if metrics:
                performance_summary.append(metrics)
                print(f"      Win Rate: {metrics.get('win_rate', 0):.1f}%")
                print(f"      Total Return: {metrics.get('total_return', 0):.1f}%")
                print(f"      Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
                
                # Send backtest result
                telegram.send_backtest_result(symbol, metrics)
            
            # Send symbol training complete
            if metrics:
                telegram.send_symbol_training_complete(symbol, metrics)

        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Error training {symbol}: {error_msg}")
            telegram.send_error(symbol, error_msg)

    # Step 5: Train Ensemble Model (if multiple models)
    if len(trained_models) >= 2:
        print("\n🤝 Step 5: Training Ensemble Model...")
        print("="*70)
        
        # Pick a symbol for ensemble training
        ensemble_symbol = train_symbols[0]
        ensemble_data = symbols_data.get(ensemble_symbol)
        
        if ensemble_data is not None:
            ensemble_env = EnsembleTradingEnv(
                ensemble_data, 
                ensemble_symbol, 
                xgb_models.get(ensemble_symbol), 
                trained_models
            )
            
            print(f"   ✅ Ensemble model ready for {ensemble_symbol}")
            print(f"   📊 Ensemble Performance:")
            metrics = ensemble_env.performance_tracker.calculate_metrics()
            if metrics:
                print(f"      Win Rate: {metrics.get('win_rate', 0):.1f}%")

    # Step 6: Summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    print(f"✅ Successfully trained: {len(trained_models)} agents")
    print(f"📁 Final models saved in: {PPO_MODEL_DIR}")
    print(f"📦 Checkpoints saved in: {CHECKPOINT_DIR}")
    
    # Save performance summary
    if performance_summary:
        summary_df = pd.DataFrame(performance_summary)
        summary_path = os.path.join(PERFORMANCE_LOG_DIR, "training_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"📊 Performance summary saved: {summary_path}")
        
        # Print top performers
        if 'total_return' in summary_df.columns:
            top_performers = summary_df.nlargest(5, 'total_return')
            print("\n🏆 Top 5 Performers:")
            best_symbol = None
            best_return = 0
            for _, row in top_performers.iterrows():
                print(f"   {row['symbol']}: {row['total_return']:.1f}% return, {row.get('win_rate', 0):.1f}% win rate")
                if row['total_return'] > best_return:
                    best_return = row['total_return']
                    best_symbol = row['symbol']
            
            # Send portfolio summary
            telegram.send_portfolio_summary(summary_df)
            
            # Send training complete
            telegram.send_training_complete(len(trained_models), best_symbol, best_return)

    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()