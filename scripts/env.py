# envs/trading_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        signal_data,
        market_data,
        symbol="POWERGRID",
        initial_capital=500_000,
        risk_per_trade=0.01,
        render_mode=None
    ):
        super().__init__()
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade

        # Filter data
        self.signal_data = signal_data[signal_data['symbol'] == symbol].reset_index(drop=True)
        self.market_data = market_data[market_data['symbol'] == symbol].reset_index(drop=True)
        
        # Merge on date
        self.data = pd.merge(
            self.signal_data,
            self.market_data,
            on=['symbol', 'date'],
            how='inner'
        ).sort_values('date').reset_index(drop=True)
        
        if len(self.data) == 0:
            raise ValueError(f"No data for symbol: {symbol}")
        
        self.current_step = 0
        self.reset()

        # Action space: continuous (for PPO)
        # [position_size_ratio (0~2), sl_offset (0~3*atr), tp_offset (1~4*atr), close_ratio (0~1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 1.0, 0.0]),
            high=np.array([2.0, 3.0, 4.0, 1.0]),
            dtype=np.float32
        )

        # Observation space: 30+ features
        self.obs_dim = 32
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_capital
        self.position = 0          # number of shares
        self.entry_price = 0.0
        self.current_sl = 0.0
        self.current_tp = 0.0
        self.trades = []
        self.cum_pnl = 0.0
        self.max_balance = self.balance
        self.drawdown = 0.0
        
        return self._get_obs(), {}

    def _get_obs(self):
        if self.current_step >= len(self.data):
            # Dummy obs at end
            return np.zeros(self.obs_dim, dtype=np.float32)
        
        row = self.data.iloc[self.current_step]
        
        # Normalized features
        obs = np.array([
            # Signal features
            row['buy'],
            row['SL'],
            row['tp'],
            row['diff'],
            row['RRR1'],
            row['position_size'],
            
            # Market features
            row['open'], row['high'], row['low'], row['close'],
            row['volume'] / 1e6,
            row['marketCap'] / 1e9,
            row['rsi'] / 100,
            row['macd'] / 10,
            row['macd_hist'] / 5,
            row['atr'] / row['close'],
            
            # Patterns (binary → float)
            float(row['Hammer']),
            float(row['BullishEngulfing']),
            float(row['MorningStar']),
            float(row['Doji']),
            
            # Position context
            self.position / 1000,  # normalize shares
            (self.balance - self.initial_capital) / self.initial_capital,  # PnL%
            self.cum_pnl / self.initial_capital,
            self.drawdown,
            
            # Risk context
            (row['buy'] - row['SL']) / row['buy'],  # % risk if taken
            row['atr'] / row['close'],
            
            # Time features (sine/cosine for day-of-year)
            np.sin(2 * np.pi * pd.to_datetime(row['date']).dayofyear / 365),
            np.cos(2 * np.pi * pd.to_datetime(row['date']).dayofyear / 365),
        ], dtype=np.float32)
        
        # Pad to obs_dim
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)), 'constant')
        return obs[:self.obs_dim]

    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            done = True
            reward = 0.0
            return self._get_obs(), reward, done, False, self._get_info()

        row = self.data.iloc[self.current_step]
        next_row = self.data.iloc[self.current_step + 1]
        
        # Unpack action
        pos_ratio, sl_mult, tp_mult, close_ratio = action
        pos_ratio = np.clip(pos_ratio, 0, 2)
        sl_mult = np.clip(sl_mult, 0.5, 3.0)
        tp_mult = np.clip(tp_mult, 1.0, 4.0)
        close_ratio = np.clip(close_ratio, 0, 1)

        done = False
        reward = 0.0

        # ——————— EXECUTION LOGIC ———————
        # 1. Close part of position?
        if self.position > 0 and close_ratio > 0:
            shares_to_close = int(self.position * close_ratio)
            exit_price = next_row['open']  # assume execution at next open
            pnl = shares_to_close * (exit_price - self.entry_price)
            self.balance += pnl
            self.position -= shares_to_close
            self.trades.append({
                'date': next_row['date'],
                'type': 'partial_close',
                'shares': shares_to_close,
                'price': exit_price,
                'pnl': pnl
            })

        # 2. Adjust SL/TP
        new_sl = row['buy'] - sl_mult * row['atr']
        new_tp = row['buy'] + tp_mult * row['atr']
        self.current_sl = new_sl
        self.current_tp = new_tp

        # 3. Open new position (if signal active & no position)
        if self.position == 0 and pos_ratio > 0:
            risk_per_share = row['buy'] - new_sl
            if risk_per_share <= 0:
                risk_per_share = 0.01 * row['buy']  # fallback
            
            max_shares_by_risk = int((self.balance * self.risk_per_trade) / risk_per_share)
            desired_shares = int(row['position_size'] * pos_ratio)
            shares = min(desired_shares, max_shares_by_risk, int(self.balance / row['buy']))
            
            if shares > 0:
                self.position = shares
                self.entry_price = row['buy']
                self.current_sl = new_sl
                self.current_tp = new_tp
                self.trades.append({
                    'date': row['date'],
                    'type': 'entry',
                    'shares': shares,
                    'price': row['buy'],
                    'sl': new_sl,
                    'tp': new_tp
                })

        # 4. Check SL/TP hit during next period (simplified: use OHLC)
        if self.position > 0:
            hit_sl = next_row['low'] <= self.current_sl
            hit_tp = next_row['high'] >= self.current_tp
            
            if hit_sl or hit_tp:
                exit_price = self.current_sl if hit_sl else self.current_tp
                pnl = self.position * (exit_price - self.entry_price)
                self.balance += pnl
                self.trades.append({
                    'date': next_row['date'],
                    'type': 'exit',
                    'shares': self.position,
                    'price': exit_price,
                    'pnl': pnl,
                    'reason': 'SL' if hit_sl else 'TP'
                })
                self.position = 0

        # ——————— REWARD DESIGN ———————
        # Risk-adjusted return + drawdown penalty
        prev_balance = self.balance - (pnl if 'pnl' in locals() else 0)
        step_pnl_pct = (self.balance - prev_balance) / prev_balance if prev_balalance > 0 else 0
        
        self.max_balance = max(self.max_balance, self.balance)
        self.drawdown = (self.max_balance - self.balance) / self.max_balance
        
        # Reward = Sharpe-like + penalty for drawdown & turnover
        reward = step_pnl_pct * 100  # scale to %
        reward -= 0.5 * self.drawdown  # drawdown penalty
        if action[3] > 0.5:  # excessive closing
            reward -= 0.1

        self.cum_pnnl += step_pnl_pct

        # Advance
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        return self._get_obs(), reward, done, False, self._get_info()

    def _get_info(self):
        return {
            "balance": self.balance,
            "position": self.position,
            "drawdown": self.drawdown,
            "trades": len(self.trades)
        }

    def render(self):
        pass