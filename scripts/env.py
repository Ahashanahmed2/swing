# envs/trading_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class TradingEnv(gym.Env):
    """
    একটি ট্রেডিং এনভায়রনমেন্ট যেখানে PPO এজেন্ট শেখবে 
    পজিশন সাইজ, স্টপ লস এবং টেক প্রফিট অপ্টিমাইজ করতে
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        signal_data: pd.DataFrame,
        market_data: pd.DataFrame,
        symbol: str = "POWERGRID",
        initial_capital: float = 500_000,
        risk_per_trade: float = 0.01,  # 1% per trade
        transaction_cost: float = 0.001,  # 0.1% per transaction
        max_position_ratio: float = 0.5,  # maximum 50% of capital in one position
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.transaction_cost = transaction_cost
        self.max_position_ratio = max_position_ratio
        self.render_mode = render_mode
        
        # ডাটা ফিল্টার এবং প্রিপ্রসেস
        self.signal_data = signal_data[signal_data['symbol'] == symbol].reset_index(drop=True)
        self.market_data = market_data[market_data['symbol'] == symbol].reset_index(drop=True)
        
        if len(self.signal_data) == 0:
            raise ValueError(f"No signal data for symbol: {symbol}")
        if len(self.market_data) == 0:
            raise ValueError(f"No market data for symbol: {symbol}")
        
        # ডেটা মার্জ
        self.data = pd.merge(
            self.signal_data,
            self.market_data,
            on=['symbol', 'date'],
            how='inner',
            suffixes=('_signal', '_market')
        ).sort_values('date').reset_index(drop=True)
        
        if len(self.data) == 0:
            raise ValueError(f"No common data after merging for symbol: {symbol}")
        
        print(f"✅ Environment created for {symbol} with {len(self.data)} data points")
        
        # কলাম ভ্যালিডেশন
        required_signal_cols = ['buy', 'SL', 'tp', 'diff', 'RRR1', 'position_size']
        for col in required_signal_cols:
            if col not in self.data.columns:
                self.data[col] = 0.0
                print(f"⚠️ Warning: {col} column not found, using default value 0")
        
        # টেকনিক্যাল ইন্ডিকেটরস (যদি না থাকে)
        for col in ['rsi', 'macd', 'macd_hist', 'atr', 'volume', 'marketCap']:
            if col not in self.data.columns:
                self.data[col] = 0.0
        
        # ক্যান্ডলেস্টিক প্যাটার্নস (যদি না থাকে)
        for pattern in ['Hammer', 'BullishEngulfing', 'MorningStar', 'Doji']:
            if pattern not in self.data.columns:
                self.data[pattern] = False
        
        # অ্যাকশন স্পেস: [position_ratio, sl_multiplier, tp_multiplier, hold_ratio]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.5, 1.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 3.0, 5.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # অবজারভেশন স্পেস: টেকনিক্যাল ফিচার + পজিশন স্টেট
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self._get_obs_dim(),), 
            dtype=np.float32
        )
        
        # এনভায়রনমেন্ট স্টেট
        self.current_step = 0
        self.balance = initial_capital
        self.position = 0  # শেয়ার সংখ্যা
        self.entry_price = 0.0
        self.current_sl = 0.0
        self.current_tp = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_portfolio_value = initial_capital
        self.max_drawdown = 0.0
        self.portfolio_history = []
        self.trade_history = []
        
        # স্টেট ভেরিফিকেশন
        self._verify_state()
    
    def _get_obs_dim(self) -> int:
        """অবজারভেশনের ডাইমেনশন রিটার্ন করে"""
        return 30  # পূর্বে 32 ছিল, এখানে 30 করলাম
    
    def _verify_state(self):
        """এনভায়রনমেন্ট স্টেট ভেরিফাই করে"""
        assert self.balance >= 0, f"Balance cannot be negative: {self.balance}"
        assert self.position >= 0, f"Position cannot be negative: {self.position}"
        if self.position > 0:
            assert self.entry_price > 0, f"Entry price must be positive when position open: {self.entry_price}"
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        এনভায়রনমেন্ট রিসেট করে
        """
        super().reset(seed=seed)
        
        # র্যান্ডম স্টার্টিং পয়েন্ট (ঐচ্ছিক)
        if options and options.get('random_start', False) and len(self.data) > 100:
            self.current_step = self.np_random.integers(0, len(self.data) - 100)
        else:
            self.current_step = 0
        
        # এনভায়রনমেন্ট স্টেট রিসেট
        self.balance = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.current_sl = 0.0
        self.current_tp = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_portfolio_value = self.initial_capital
        self.max_drawdown = 0.0
        self.portfolio_history = []
        self.trade_history = []
        
        # প্রথম অবজারভেশন
        obs = self._get_observation()
        
        # রিসেট ইনফো
        info = {
            "balance": self.balance,
            "position": self.position,
            "step": self.current_step,
            "date": self.data.iloc[self.current_step]['date'] if len(self.data) > 0 else None
        }
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """
        বর্তমান অবজারভেশন ভেক্টর তৈরি করে
        """
        if self.current_step >= len(self.data):
            return np.zeros(self._get_obs_dim(), dtype=np.float32)
        
        row = self.data.iloc[self.current_step]
        
        # নরমালাইজেশন ফাংশন
        def normalize(value, min_val, max_val):
            if max_val - min_val == 0:
                return 0.0
            return (value - min_val) / (max_val - min_val)
        
        # মূল ফিচারস
        features = []
        
        # 1. সিগন্যাল ফিচারস (0-1 রেঞ্জে নরমালাইজ)
        signal_features = [
            normalize(row['buy'], self.data['buy'].min(), self.data['buy'].max()),
            normalize(row['SL'], self.data['SL'].min(), self.data['SL'].max()),
            normalize(row['tp'], self.data['tp'].min(), self.data['tp'].max()),
            row['diff'] / row['buy'] if row['buy'] > 0 else 0.0,  # % ডিফারেন্স
            min(row['RRR1'], 5.0) / 5.0 if 'RRR1' in row else 0.5,  # RRR নরমালাইজ
            min(row['position_size'], 10000) / 10000  # পজিশন সাইজ নরমালাইজ
        ]
        features.extend(signal_features)
        
        # 2. মার্কেট ফিচারস
        market_features = [
            row['open'] / row['close'] - 1 if row['close'] > 0 else 0.0,
            row['high'] / row['close'] - 1 if row['close'] > 0 else 0.0,
            row['low'] / row['close'] - 1 if row['close'] > 0 else 0.0,
            normalize(row['volume'], self.data['volume'].min(), self.data['volume'].max()) if 'volume' in self.data.columns else 0.0,
            normalize(row.get('marketCap', 0), 0, 1e12) / 1e12,  # মার্কেট ক্যাপ
            row.get('rsi', 50) / 100.0,  # RSI (0-1)
            row.get('macd', 0) / 100.0 if abs(row.get('macd', 0)) > 0 else 0.0,
            row.get('macd_hist', 0) / 50.0,
            row.get('atr', 0) / row['close'] if row['close'] > 0 else 0.01
        ]
        features.extend(market_features)
        
        # 3. ক্যান্ডলেস্টিক প্যাটার্নস (বুলিয়ান থেকে ফ্লোট)
        patterns = [
            float(row.get('Hammer', False)),
            float(row.get('BullishEngulfing', False)),
            float(row.get('MorningStar', False)),
            float(row.get('Doji', False))
        ]
        features.extend(patterns)
        
        # 4. পজিশন এবং একাউন্ট স্টেট
        position_state = [
            self.position / 10000.0,  # নরমালাইজড পজিশন
            (self.balance - self.initial_capital) / self.initial_capital,  # PnL%
            self.total_pnl / self.initial_capital,
            self.max_drawdown,
            self.position * self.entry_price / self.balance if self.balance > 0 else 0.0,  # পজিশন এক্সপোজার %
            1.0 if self.position > 0 else 0.0,  # ইন পজিশন ফ্ল্যাগ
            (row['buy'] - row.get('SL', row['buy']*0.95)) / row['buy'] if row['buy'] > 0 else 0.0  % রিস্ক
        ]
        features.extend(position_state)
        
        # 5. টাইম ফিচারস
        if 'date' in row and pd.notna(row['date']):
            try:
                date_obj = pd.to_datetime(row['date'])
                day_of_year = date_obj.dayofyear
                features.append(np.sin(2 * np.pi * day_of_year / 365))
                features.append(np.cos(2 * np.pi * day_of_year / 365))
            except:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        # অ্যারে কনভার্ট এবং প্যাড করা
        obs_array = np.array(features, dtype=np.float32)
        
        # ডাইমেনশন চেক
        if len(obs_array) < self._get_obs_dim():
            padding = np.zeros(self._get_obs_dim() - len(obs_array), dtype=np.float32)
            obs_array = np.concatenate([obs_array, padding])
        elif len(obs_array) > self._get_obs_dim():
            obs_array = obs_array[:self._get_obs_dim()]
        
        # NaN চেক
        if np.any(np.isnan(obs_array)):
            obs_array = np.nan_to_num(obs_array, nan=0.0)
        
        return obs_array
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        একটি স্টেপ এগিয়ে নেয় এবং রিওয়ার্ড ক্যালকুলেট করে
        """
        # চেক যদি এপিসোড শেষ হয়ে যায়
        if self.current_step >= len(self.data) - 1:
            done = True
            obs = self._get_observation()
            reward = 0.0
            info = self._get_info()
            return obs, reward, done, False, info
        
        # কারেন্ট এবং নেক্সট রো
        current_row = self.data.iloc[self.current_step]
        next_row = self.data.iloc[self.current_step + 1]
        
        # অ্যাকশন আনপ্যাক এবং ক্লিপ
        pos_ratio, sl_mult, tp_mult, hold_ratio = action
        pos_ratio = np.clip(pos_ratio, 0.0, 2.0)
        sl_mult = np.clip(sl_mult, 0.5, 3.0)
        tp_mult = np.clip(tp_mult, 1.0, 5.0)
        hold_ratio = np.clip(hold_ratio, 0.0, 1.0)
        
        # ইনিশিয়াল স্টেট
        initial_portfolio_value = self._get_portfolio_value(current_row['close'])
        reward = 0.0
        trade_executed = False
        
        # 1. এক্সিস্টিং পজিশন ক্লোজ (যদি hold_ratio কম হয়)
        if self.position > 0 and hold_ratio < 0.5:
            # আংশিক বা সম্পূর্ণ ক্লোজ
            close_ratio = 1.0 - hold_ratio
            shares_to_close = int(self.position * close_ratio)
            
            if shares_to_close > 0:
                # ট্রানজেকশন কস্ট
                close_price = next_row['open']
                transaction_cost_amount = shares_to_close * close_price * self.transaction_cost
                
                # PnL ক্যালকুলেট
                pnl = shares_to_close * (close_price - self.entry_price) - transaction_cost_amount
                
                # ব্যালেন্স আপডেট
                self.balance += shares_to_close * close_price - transaction_cost_amount
                self.position -= shares_to_close
                
                # ট্রেড হিস্ট্রি
                self.total_trades += 1
                self.total_pnl += pnl
                if pnl > 0:
                    self.winning_trades += 1
                
                trade_executed = True
                
                # যদি সব শেয়ার ক্লোজ হয়
                if self.position == 0:
                    self.entry_price = 0.0
                    self.current_sl = 0.0
                    self.current_tp = 0.0
        
        # 2. নতুন পজিশন ওপেন (যদি পজিশন না থাকে)
        if self.position == 0 and pos_ratio > 0.1:
            # স্টপ লস এবং টেক প্রফিট ক্যালকুলেট
            atr = current_row.get('atr', current_row['buy'] * 0.02)
            new_sl = current_row['buy'] - (sl_mult * atr)
            new_tp = current_row['buy'] + (tp_mult * atr)
            
            # সেন্সিবল লিমিটস
            new_sl = max(new_sl, current_row['buy'] * 0.85)  # ম্যাক্স 15% লস
            new_tp = min(new_tp, current_row['buy'] * 1.50)  # ম্যাক্স 50% প্রফিট
            
            # পজিশন সাইজ ক্যালকুলেট
            risk_per_share = current_row['buy'] - new_sl
            if risk_per_share <= 0:
                risk_per_share = current_row['buy'] * 0.01  # 1% ডিফল্ট
            
            # রিস্ক-বেসড পজিশন সাইজ
            max_risk_amount = self.balance * self.risk_per_trade
            max_shares_by_risk = int(max_risk_amount / risk_per_share)
            
            # ক্যাপিটাল-বেসড পজিশন সাইজ
            max_shares_by_capital = int(self.balance * self.max_position_ratio / current_row['buy'])
            
            # বেস পজিশন সাইজ
            base_shares = current_row.get('position_size', 100)
            
            # ফাইনাল পজিশন সাইজ
            desired_shares = int(base_shares * pos_ratio)
            actual_shares = min(
                desired_shares,
                max_shares_by_risk,
                max_shares_by_capital
            )
            
            # মিনিমাম শেয়ার চেক
            if actual_shares >= 10:  # কমপক্ষে 10 শেয়ার
                # ট্রানজেকশন কস্ট
                transaction_cost_amount = actual_shares * current_row['buy'] * self.transaction_cost
                
                # বিনিয়োগ পরিমাণ
                investment = actual_shares * current_row['buy'] + transaction_cost_amount
                
                if investment <= self.balance:
                    # পজিশন ওপেন
                    self.position = actual_shares
                    self.entry_price = current_row['buy']
                    self.current_sl = new_sl
                    self.current_tp = new_tp
                    self.balance -= investment
                    
                    trade_executed = True
                    
                    # ট্রেড হিস্ট্রি
                    self.trade_history.append({
                        'step': self.current_step,
                        'date': current_row['date'],
                        'action': 'BUY',
                        'shares': actual_shares,
                        'price': current_row['buy'],
                        'sl': new_sl,
                        'tp': new_tp,
                        'investment': investment
                    })
        
        # 3. এক্সিস্টিং পজিশনের জন্য SL/TP চেক
        if self.position > 0:
            # SL/TP হিট চেক
            sl_hit = next_row['low'] <= self.current_sl
            tp_hit = next_row['high'] >= self.current_tp
            
            if sl_hit or tp_hit:
                # এক্সিট প্রাইস (যেটা প্রথমে ট্রিগার হয়েছে)
                if sl_hit:
                    exit_price = min(self.current_sl, next_row['open'])
                    exit_type = 'SL'
                else:
                    exit_price = max(self.current_tp, next_row['open'])
                    exit_type = 'TP'
                
                # ট্রানজেকশন কস্ট
                transaction_cost_amount = self.position * exit_price * self.transaction_cost
                
                # PnL ক্যালকুলেট
                pnl = self.position * (exit_price - self.entry_price) - transaction_cost_amount
                
                # ব্যালেন্স আপডেট
                self.balance += self.position * exit_price - transaction_cost_amount
                
                # ট্র্যাকিং
                self.total_trades += 1
                self.total_pnl += pnl
                if pnl > 0:
                    self.winning_trades += 1
                
                # ট্রেড হিস্ট্রি
                self.trade_history.append({
                    'step': self.current_step,
                    'date': next_row['date'],
                    'action': exit_type,
                    'shares': self.position,
                    'price': exit_price,
                    'pnl': pnl
                })
                
                # পজিশন রিসেট
                self.position = 0
                self.entry_price = 0.0
                self.current_sl = 0.0
                self.current_tp = 0.0
                
                trade_executed = True
        
        # 4. পোর্টফোলিও ভ্যালু আপডেট এবং ড্রঅডাউন ক্যালকুলেট
        current_portfolio_value = self._get_portfolio_value(next_row['close'])
        self.max_portfolio_value = max(self.max_portfolio_value, current_portfolio_value)
        
        current_drawdown = (self.max_portfolio_value - current_portfolio_value) / self.max_portfolio_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # 5. রিওয়ার্ড ক্যালকুলেশন
        # 5.1. পোর্টফোলিও রিটার্ন
        portfolio_return = (current_portfolio_value - initial_portfolio_value) / initial_portfolio_value
        
        # 5.2. রিওয়ার্ড কম্পোনেন্টস
        reward_components = {
            'return': portfolio_return * 100,  # স্কেলিং
            'drawdown_penalty': -self.max_drawdown * 50,
            'trade_execution_bonus': 0.1 if trade_executed else 0.0,
            'position_penalty': -0.01 if self.position > 0 else 0.0  # হোল্ডিং কস্ট
        }
        
        # 5.3. ট্রেড কোয়ালিটি বোনাস
        if trade_executed and 'pnl' in locals() and pnl > 0:
            reward_components['win_bonus'] = 0.5
        elif trade_executed and 'pnl' in locals() and pnl < 0:
            reward_components['loss_penalty'] = -0.2
        
        # 5.4. ফাইনাল রিওয়ার্ড
        reward = sum(reward_components.values())
        
        # স্টেপ ইনক্রিমেন্ট
        self.current_step += 1
        
        # এপিসোড শেষ চেক
        done = self.current_step >= len(self.data) - 1
        
        # পোর্টফোলিও হিস্ট্রি আপডেট
        self.portfolio_history.append({
            'step': self.current_step,
            'date': next_row['date'],
            'portfolio_value': current_portfolio_value,
            'balance': self.balance,
            'position_value': self.position * next_row['close'] if self.position > 0 else 0.0,
            'drawdown': current_drawdown
        })
        
        # অবজারভেশন এবং ইনফো
        obs = self._get_observation()
        info = self._get_info()
        info.update({
            'portfolio_return': portfolio_return,
            'reward_components': reward_components,
            'trade_executed': trade_executed
        })
        
        return obs, reward, done, False, info
    
    def _get_portfolio_value(self, current_price: float) -> float:
        """বর্তমান পোর্টফোলিও ভ্যালু রিটার্ন করে"""
        position_value = self.position * current_price if self.position > 0 else 0.0
        return self.balance + position_value
    
    def _get_info(self) -> Dict:
        """এপিসোড ইনফো রিটার্ন করে"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        
        current_price = self.data.iloc[min(self.current_step, len(self.data)-1)]['close']
        portfolio_value = self._get_portfolio_value(current_price)
        
        return {
            "balance": float(self.balance),
            "position": int(self.position),
            "entry_price": float(self.entry_price),
            "current_sl": float(self.current_sl),
            "current_tp": float(self.current_tp),
            "portfolio_value": float(portfolio_value),
            "total_trades": int(self.total_trades),
            "winning_trades": int(self.winning_trades),
            "win_rate": float(win_rate),
            "total_pnl": float(self.total_pnl),
            "max_drawdown": float(self.max_drawdown),
            "step": int(self.current_step),
            "date": str(self.data.iloc[min(self.current_step, len(self.data)-1)]['date']) if len(self.data) > 0 else None
        }
    
    def render(self, mode: str = 'human'):
        """এনভায়রনমেন্ট রেন্ডার করে (সিম্পল ভার্সন)"""
        if mode == 'human':
            info = self._get_info()
            current_price = self.data.iloc[min(self.current_step, len(self.data)-1)]['close']
            
            print(f"\n{'='*60}")
            print(f"Symbol: {self.symbol} | Step: {self.current_step}/{len(self.data)}")
            print(f"Date: {info['date']} | Price: {current_price:.2f}")
            print(f"{'-'*60}")
            print(f"Portfolio Value: {info['portfolio_value']:,.2f}")
            print(f"Balance: {info['balance']:,.2f}")
            print(f"Position: {info['position']} shares @ {info['entry_price']:.2f}")
            print(f"Current P&L: {info['total_pnl']:,.2f}")
            print(f"Win Rate: {info['win_rate']:.1f}% ({info['winning_trades']}/{info['total_trades']})")
            print(f"Max Drawdown: {info['max_drawdown']:.2%}")
            print(f"{'='*60}")
    
    def get_trade_history(self) -> pd.DataFrame:
        """ট্রেড হিস্ট্রি DataFrame হিসেবে রিটার্ন করে"""
        if not self.trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_history)
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """পোর্টফোলিও হিস্ট্রি DataFrame হিসেবে রিটার্ন করে"""
        if not self.portfolio_history:
            return pd.DataFrame()
        return pd.DataFrame(self.portfolio_history)


# টেস্ট করার জন্য
if __name__ == "__main__":
    # টেস্ট ডাটা তৈরি
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    test_signals = pd.DataFrame({
        'symbol': ['TEST'] * len(dates),
        'date': dates,
        'buy': np.random.uniform(100, 200, len(dates)),
        'SL': np.random.uniform(90, 180, len(dates)),
        'tp': np.random.uniform(120, 250, len(dates)),
        'diff': np.random.uniform(5, 20, len(dates)),
        'RRR1': np.random.uniform(1.5, 3.0, len(dates)),
        'position_size': np.random.randint(100, 1000, len(dates))
    })
    
    test_market = pd.DataFrame({
        'symbol': ['TEST'] * len(dates),
        'date': dates,
        'open': np.random.uniform(100, 200, len(dates)),
        'high': np.random.uniform(110, 220, len(dates)),
        'low': np.random.uniform(90, 190, len(dates)),
        'close': np.random.uniform(100, 200, len(dates)),
        'volume': np.random.randint(10000, 100000, len(dates)),
        'marketCap': np.random.uniform(1e9, 1e10, len(dates)),
        'rsi': np.random.uniform(30, 70, len(dates)),
        'macd': np.random.uniform(-5, 5, len(dates)),
        'macd_hist': np.random.uniform(-2, 2, len(dates)),
        'atr': np.random.uniform(2, 10, len(dates))
    })
    
    # এনভায়রনমেন্ট টেস্ট
    try:
        env = TradingEnv(test_signals, test_market, symbol="TEST", initial_capital=100000)
        obs, info = env.reset()
        print(f"✅ Environment created successfully")
        print(f"Observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")
        
        # কয়েকটা স্টেপ রান
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"Step {i}: Reward={reward:.4f}, Portfolio={info['portfolio_value']:.2f}, Position={info['position']}")
            if done:
                break
        
        env.render()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()