import gymnasium as gym
import os
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradeEnv(gym.Env):
    def __init__(self,
                 maindf,
                 gape_path="./csv/gape.csv",
                 gapebuy_path="./csv/gape_buy.csv",
                 shortbuy_path="./csv/short_buy.csv",
                 rsi_diver_path="./csv/rsi_diver.csv",
                 rsi_diver_retest_path="./csv/rsi_diver_retest.csv",
                 trade_stock_path="./csv/trade_stock.csv",
                 metrics_path="./output/ai_signal/strategy_metrics.csv",
                 symbol_ref_path="./output/ai_signal/symbol_reference_metrics.csv",
                 config_path="./config.json"):

        super(TradeEnv, self).__init__()

        self.maindf = maindf.copy().reset_index(drop=True)
        
        # --- Load signal files ---
        self.gape_df = pd.read_csv(gape_path) if os.path.exists(gape_path) else pd.DataFrame()
        self.gapebuy_df = pd.read_csv(gapebuy_path) if os.path.exists(gapebuy_path) else pd.DataFrame()
        self.shortbuy_df = pd.read_csv(shortbuy_path) if os.path.exists(shortbuy_path) else pd.DataFrame()
        self.rsi_diver_df = pd.read_csv(rsi_diver_path) if os.path.exists(rsi_diver_path) else pd.DataFrame()
        self.rsi_diver_retest_df = pd.read_csv(rsi_diver_retest_path) if os.path.exists(rsi_diver_retest_path) else pd.DataFrame()

        # --- 🔑 CRITICAL: Load your SYSTEM'S INTELLIGENCE ---
        self.trade_stock_df = pd.read_csv(trade_stock_path) if os.path.exists(trade_stock_path) else pd.DataFrame()
        self.strategy_metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
        self.symbol_ref_metrics = pd.read_csv(symbol_ref_path) if os.path.exists(symbol_ref_path) else pd.DataFrame()

        # Load config for risk context
        self.TOTAL_CAPITAL = 500000
        self.RISK_PERCENT = 0.01
        if os.path.exists(config_path):
            import json
            with open(config_path) as f:
                cfg = json.load(f)
                self.TOTAL_CAPITAL = cfg.get("total_capital", 500000)
                self.RISK_PERCENT = cfg.get("risk_percent", 0.01)

        # Environment state
        self.cash = 10000
        self.stock = 0
        self.last_price = None
        self.current_step = 0
        self.total_steps = len(self.maindf) - 1
        self.last_obs = None

        # --- ✅ ENHANCED OBSERVATION SPACE ---
        # Base features (20)
        self.base_features = [
            'open', 'high', 'low', 'close', 'volume', 'value', 'trades', 'change', 'marketCap',
            'RSI', 'bb_upper', 'bb_middle', 'bb_lower',
            'macd', 'macd_signal', 'macd_hist', 'zigzag',
            'Hammer', 'BullishEngulfing', 'MorningStar'
        ]
        
        # System intelligence features (12 new)
        self.sys_features = [
            'has_gape_signal', 'has_gapebuy', 'has_shortbuy',
            'has_rsi_diver', 'has_rsi_retest',
            'position_size', 'exposure_ratio', 'risk_ratio',
            'RRR', 'strategy_win_pct', 'symbol_ref_win_pct', 'expectancy_bdt'
        ]
        
        obs_dim = len(self.base_features) + len(self.sys_features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell

    def _get_system_context(self, symbol, date):
        """✅ Inject your system's signals & metrics into observation"""
        context = {
            'has_gape_signal': 0,
            'has_gapebuy': 0,
            'has_shortbuy': 0,
            'has_rsi_diver': 0,
            'has_rsi_retest': 0,
            'position_size': 0,
            'exposure_ratio': 0.0,
            'risk_ratio': 0.0,
            'RRR': 0.0,
            'strategy_win_pct': 50.0,      # default neutral
            'symbol_ref_win_pct': 50.0,    # default neutral
            'expectancy_bdt': 0.0
        }

        # --- Signal flags ---
        if not self.gape_df.empty and symbol in self.gape_df['symbol'].values:
            context['has_gape_signal'] = 1
        if not self.gapebuy_df.empty and symbol in self.gapebuy_df['symbol'].values:
            context['has_gapebuy'] = 1
        if not self.shortbuy_df.empty and symbol in self.shortbuy_df['symbol'].values:
            context['has_shortbuy'] = 1
        if not self.rsi_diver_df.empty and symbol in self.rsi_diver_df['symbol'].values:
            context['has_rsi_diver'] = 1
        if not self.rsi_diver_retest_df.empty and symbol in self.rsi_diver_retest_df['symbol'].values:
            context['has_rsi_retest'] = 1

        # --- 🔑 CRITICAL: Pull position & risk from YOUR SYSTEM ---
        if not self.trade_stock_df.empty:
            # Find open signal for this symbol (closest date)
            open_signals = self.trade_stock_df[
                (self.trade_stock_df['symbol'] == symbol) &
                (pd.to_datetime(self.trade_stock_df['date']) <= date)
            ]
            if not open_signals.empty:
                sig = open_signals.iloc[-1]
                pos = sig.get('position_size', 0)
                exp = sig.get('exposure_bdt', 0)
                risk = sig.get('actual_risk_bdt', 0)
                rrr = sig.get('RRR', 0)
                
                context['position_size'] = pos
                context['exposure_ratio'] = exp / self.TOTAL_CAPITAL
                context['risk_ratio'] = risk / self.TOTAL_CAPITAL
                context['RRR'] = rrr

        # --- Strategy performance (swing/gape/etc) ---
        if not self.strategy_metrics.empty:
            # Assume all signals are 'swing' for simplicity — update as needed
            swing_row = self.strategy_metrics[self.strategy_metrics['Reference'] == 'SWING']
            if not swing_row.empty:
                context['strategy_win_pct'] = float(swing_row.iloc[0]['Win%'])

        # --- Symbol × Strategy performance (POWERGRID + RSI) ---
        if not self.symbol_ref_metrics.empty:
            sym_ref = self.symbol_ref_metrics[
                (self.symbol_ref_metrics['Symbol'] == symbol) &
                (self.symbol_ref_metrics['Reference'] == 'SWING')  # or infer from signal
            ]
            if not sym_ref.empty:
                context['symbol_ref_win_pct'] = float(sym_ref.iloc[0]['Win%'])
                context['expectancy_bdt'] = float(sym_ref.iloc[0]['Expectancy (BDT)'])

        return context

    def get_obs(self):
        row = self.maindf.iloc[self.current_step]
        symbol = row.get('symbol', '')
        date = row.get('date', pd.Timestamp.now())

        # Base features
        obs_list = []
        for col in self.base_features:
            val = row.get(col, 0)
            if col in ['Hammer', 'BullishEngulfing', 'MorningStar']:
                val = 1 if str(val).upper() == 'TRUE' else 0
            obs_list.append(float(val))

        # ✅ System intelligence features
        sys_ctx = self._get_system_context(symbol, date)
        for col in self.sys_features:
            obs_list.append(float(sys_ctx[col]))

        obs_array = np.array(obs_list, dtype=np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=0.0, neginf=0.0)
        return obs_array

    def step(self, action):
        row = self.maindf.iloc[self.current_step]
        price = row.get('close', 0)
        symbol = row.get('symbol', '')
        date = row.get('date', pd.Timestamp.now())
        volume = row.get('volume', 0)
        rsi = row.get('RSI', 50)
        macd = row.get('macd', 0)
        macd_signal = row.get('macd_signal', 0)
        zigzag = row.get('zigzag', 0)
        change = row.get('change', 0)

        reward = 0.0
        done = False

        # --- ✅ ENHANCED REWARD: Risk-Adjusted & System-Aware ---
        # Base action reward
        if action == 1:  # Buy
            if self.cash >= price:
                self.stock += 1
                self.cash -= price
                self.last_price = price
                reward = -0.05  # transaction cost
            else:
                reward = -1.0  # punish over-leverage

        elif action == 2 and self.stock > 0:  # Sell
            self.stock -= 1
            self.cash += price
            profit = price - self.last_price if self.last_price else 0
            # ✅ Reward scaled by EXPECTANCY (BDT) — your system's edge!
            sys_ctx = self._get_system_context(symbol, date)
            expectancy_bdt = sys_ctx['expectancy_bdt']
            if profit > 0:
                reward = profit * (1 + 0.001 * expectancy_bdt)  # +0.1% per BDT expectancy
            else:
                reward = profit * 0.5  # punish loss more

        # --- Signal-based bonuses (from YOUR system) ---
        sys_ctx = self._get_system_context(symbol, date)
        
        # Bonus for buying when system has high-confidence signal
        if action == 1:
            if sys_ctx['has_gapebuy'] or sys_ctx['has_rsi_retest']:
                # ✅ Bonus = RRR * Win% * position_size (normalized)
                bonus = sys_ctx['RRR'] * (sys_ctx['symbol_ref_win_pct'] / 100) * (sys_ctx['position_size'] / 1000)
                reward += min(bonus, 2.0)  # cap at 2.0

            # Extra for high-expectancy symbols
            if sys_ctx['expectancy_bdt'] > 100:
                reward += 0.5

        # Pattern & indicator bonuses (existing + tuned)
        if action == 1:
            if rsi < 30:
                reward += 0.5
            if macd > macd_signal:
                reward += 0.25
            if volume > self.maindf['volume'].mean() * 1.5:
                reward += 0.3

        # Volatility adjustment
        volatility = (row.get('high', 0) - row.get('low', 0)) / price if price > 0 else 0
        if volatility > 0.05:
            reward *= 1.2

        # --- Risk penalty: punish deviating from YOUR system's position_size ---
        target_pos = sys_ctx['position_size']
        current_pos = self.stock
        if action == 1 and current_pos > target_pos + 100:  # over-positioned
            reward -= 1.0
        if action == 0 and target_pos > 0 and current_pos == 0:  # missed signal
            reward -= 0.5

        # Step forward
        self.current_step += 1
        done = self.current_step >= self.total_steps

        # Get next obs (or last if done)
        obs = self.get_obs() if not done else self.last_obs
        self.last_obs = obs

        # Clip & clean reward
        reward = np.clip(reward, -5, 5)
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        return obs, reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        self.cash = 10000
        self.stock = 0
        self.last_price = None
        self.current_step = 0
        self.last_obs = self.get_obs()
        return self.last_obs, {}

    def render(self):
        r = self.maindf.iloc[self.current_step]
        price = float(r.get('close', 0))
        symbol = r.get('symbol', '')
        value = self.cash + self.stock * price
        print(f"[{symbol}] Step {self.current_step}: Cash {self.cash:7.2f} | Stock {self.stock:3d} | Value {value:8.2f} | PnL {value - 10000:+7.2f}")