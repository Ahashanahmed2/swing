import gymnasium as gym
import os
import numpy as np
import pandas as pd
from gymnasium import spaces
import json

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
                 liquidity_path="./csv/liquidity_system.csv",  # ✅ NEW
                 config_path="./config.json"):

        super(TradeEnv, self).__init__()

        self.maindf = maindf.copy().reset_index(drop=True)
        self.maindf['date'] = pd.to_datetime(self.maindf['date'])
        self.maindf['symbol'] = self.maindf['symbol'].str.upper()

        # --- Load signal files ---
        self.gape_df = self._safe_load(gape_path, ['symbol'])
        self.gapebuy_df = self._safe_load(gapebuy_path, ['symbol', 'date'])
        self.shortbuy_df = self._safe_load(shortbuy_path, ['symbol', 'date'])
        self.rsi_diver_df = self._safe_load(rsi_diver_path, ['symbol'])
        self.rsi_diver_retest_df = self._safe_load(rsi_diver_retest_path, ['symbol'])

        # --- Load SYSTEM INTELLIGENCE ---
        self.trade_stock_df = self._safe_load(trade_stock_path, ['symbol', 'date', 'position_size', 'RRR'])
        self.strategy_metrics = self._safe_load(metrics_path, ['Reference', 'Win%'])
        self.symbol_ref_metrics = self._safe_load(symbol_ref_path, ['Symbol', 'Reference', 'Win%', 'Expectancy (BDT)'])
        self.liquidity_df = self._safe_load(liquidity_path, ['symbol', 'liquidity_score'])  # ✅ NEW

        # Load config
        self._load_config(config_path)

        # Environment state
        self.cash = self.TOTAL_CAPITAL  # ✅ Start with full capital
        self.positions = {}  # {symbol: {'shares': int, 'avg_price': float, 'sl': float, 'tp': float}}
        self.current_step = 0
        self.total_steps = len(self.maindf) - 1
        self.last_obs = None

        # --- ✅ ENHANCED OBSERVATION SPACE ---
        self.base_features = [
            'open', 'high', 'low', 'close', 'volume', 'value', 'trades', 'change', 'marketCap',
            'RSI', 'bb_upper', 'bb_middle', 'bb_lower',
            'macd', 'macd_signal', 'macd_hist', 'zigzag'
        ]
        self.pattern_cols = ['Hammer', 'BullishEngulfing', 'MorningStar']
        self.sys_features = [
            'has_signal_today', 'position_size_suggested', 'rrr_signal',
            'win_pct_symbol', 'expectancy_bdt', 'liquidity_score',
            'portfolio_exposure_ratio', 'unrealized_pnl_pct'
        ]

        obs_dim = len(self.base_features) + len(self.pattern_cols) + len(self.sys_features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell (all)

    def _safe_load(self, path, required_cols):
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = 0
                return df
        except Exception as e:
            print(f"⚠️ Failed to load {path}: {e}")
        return pd.DataFrame(columns=required_cols)

    def _load_config(self, config_path):
        self.TOTAL_CAPITAL = 500000
        self.RISK_PERCENT = 0.01
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
                self.TOTAL_CAPITAL = cfg.get("total_capital", 500000)
                self.RISK_PERCENT = cfg.get("risk_percent", 0.01)

    def _get_open_signal(self, symbol, date):
        """✅ Get TODAY'S open signal from trade_stock.csv"""
        if self.trade_stock_df.empty:
            return None

        # Find signal for this symbol ON THIS DATE
        signals = self.trade_stock_df[
            (self.trade_stock_df['symbol'] == symbol) &
            (pd.to_datetime(self.trade_stock_df['date']).dt.date == date.date())
        ]
        return signals.iloc[0] if not signals.empty else None

    def _get_system_context(self, symbol, date):
        context = {
            'has_signal_today': 0,
            'position_size_suggested': 0,
            'rrr_signal': 0.0,
            'win_pct_symbol': 50.0,
            'expectancy_bdt': 0.0,
            'liquidity_score': 0.5,
            'portfolio_exposure_ratio': 0.0,
            'unrealized_pnl_pct': 0.0
        }

        # --- Open signal today? ---
        signal = self._get_open_signal(symbol, date)
        if signal is not None:
            context['has_signal_today'] = 1
            context['position_size_suggested'] = int(signal.get('position_size', 0))
            context['rrr_signal'] = float(signal.get('RRR', 0))

        # --- Symbol performance ---
        if not self.symbol_ref_metrics.empty:
            ref_row = self.symbol_ref_metrics[
                (self.symbol_ref_metrics['Symbol'].str.upper() == symbol) &
                (self.symbol_ref_metrics['Reference'] == 'SWING')
            ]
            if not ref_row.empty:
                context['win_pct_symbol'] = float(ref_row.iloc[0].get('Win%', 50.0))
                context['expectancy_bdt'] = float(ref_row.iloc[0].get('Expectancy (BDT)', 0.0))

        # --- Liquidity ---
        if not self.liquidity_df.empty:
            liq_row = self.liquidity_df[self.liquidity_df['symbol'].str.upper() == symbol]
            if not liq_row.empty:
                context['liquidity_score'] = float(liq_row.iloc[0].get('liquidity_score', 0.5))

        # --- Portfolio context ---
        total_exposure = sum(p['shares'] * p['avg_price'] for p in self.positions.values())
        context['portfolio_exposure_ratio'] = total_exposure / self.TOTAL_CAPITAL

        # Unrealized PnL
        if symbol in self.positions:
            pos = self.positions[symbol]
            current_price = self.maindf.iloc[self.current_step]['close']
            unrealized = (current_price - pos['avg_price']) / pos['avg_price'] * 100
            context['unrealized_pnl_pct'] = unrealized

        return context

    def get_obs(self):
        row = self.maindf.iloc[self.current_step]
        symbol = row.get('symbol', '')
        date = row.get('date', pd.Timestamp.now())

        # Base features
        obs = []
        for col in self.base_features:
            val = row.get(col, 0)
            obs.append(float(val))

        # Pattern flags
        for col in self.pattern_cols:
            val = row.get(col, 0)
            obs.append(1.0 if str(val).upper() == 'TRUE' else 0.0)

        # System intelligence
        sys_ctx = self._get_system_context(symbol, date)
        for col in self.sys_features:
            obs.append(float(sys_ctx[col]))

        obs_array = np.array(obs, dtype=np.float32)
        return np.nan_to_num(obs_array, nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, action):
        row = self.maindf.iloc[self.current_step]
        symbol = row['symbol']
        price = row['close']
        date = row['date']

        reward = 0.0
        info = {}

        # Get system signal for TODAY
        signal = self._get_open_signal(symbol, date)
        sys_ctx = self._get_system_context(symbol, date)

        # --- ✅ DSE-REALISTIC EXECUTION: Use SUGGESTED position_size ---
        if action == 1 and signal is not None:  # Buy signal + action=Buy
            suggested_shares = int(signal.get('position_size', 0))
            cost = suggested_shares * price

            if self.cash >= cost and suggested_shares > 0:
                # Open new position
                self.positions[symbol] = {
                    'shares': suggested_shares,
                    'avg_price': price,
                    'sl': float(signal.get('SL', price * 0.97)),
                    'tp': float(signal.get('tp', price * 1.05))
                }
                self.cash -= cost

                # ✅ Reward = scaled by expectancy & liquidity
                bonus = (
                    sys_ctx['expectancy_bdt'] * 0.001 +   # 0.1% per BDT
                    sys_ctx['liquidity_score'] * 0.5 +    # up to +0.5
                    (sys_ctx['win_pct_symbol'] - 50) * 0.01  # +0.1% per 1% Win% above 50
                )
                reward = 1.0 + bonus

            else:
                reward = -2.0  # punish missed opportunity

        elif action == 2 and symbol in self.positions:  # Sell all
            pos = self.positions[symbol]
            proceeds = pos['shares'] * price
            profit = proceeds - (pos['shares'] * pos['avg_price'])

            self.cash += proceeds
            del self.positions[symbol]

            # ✅ Reward based on outcome
            if profit > 0:
                reward = profit * (1 + 0.001 * sys_ctx['expectancy_bdt'])
            else:
                reward = profit * 2.0  # punish loss more

        # --- Auto-exit on SL/TP (DSE-realistic) ---
        if symbol in self.positions:
            pos = self.positions[symbol]
            if price <= pos['sl']:
                # SL hit — auto sell
                shares = pos['shares']
                proceeds = shares * price
                self.cash += proceeds
                del self.positions[symbol]
                reward += -1.0  # additional penalty for SL
                info['exit_reason'] = 'SL'
            elif price >= pos['tp']:
                # TP hit — auto sell
                shares = pos['shares']
                proceeds = shares * price
                self.cash += proceeds
                del self.positions[symbol]
                reward += +0.5  # bonus for TP
                info['exit_reason'] = 'TP'

        # Step forward
        self.current_step += 1
        done = self.current_step >= self.total_steps

        # Final portfolio value
        portfolio_value = self.cash + sum(
            p['shares'] * self.maindf.iloc[min(self.current_step, len(self.maindf)-1)]['close']
            for p in self.positions.values()
        )
        info['portfolio_value'] = portfolio_value
        info['positions'] = len(self.positions)

        # Clip reward
        reward = np.clip(reward, -5, 5)
        obs = self.get_obs() if not done else self.last_obs
        self.last_obs = obs

        return obs, reward, done, False, info

    def reset(self, *, seed=None, options=None):
        self.cash = self.TOTAL_CAPITAL
        self.positions = {}
        self.current_step = 0
        self.last_obs = self.get_obs()
        return self.last_obs, {}

    def render(self):
        r = self.maindf.iloc[self.current_step]
        symbol = r['symbol']
        price = r['close']
        portfolio_value = self.cash + sum(p['shares'] * price for p in self.positions.values())
        pnl = portfolio_value - self.TOTAL_CAPITAL
        pos_str = ", ".join([f"{s}: {p['shares']}" for s, p in self.positions.items()])
        print(f"[{symbol}] Cash: {self.cash:8.0f} | PnL: {pnl:+7.0f} | Pos: {pos_str or 'None'}")