import gymnasium as gym
import os
import numpy as np
import pandas as pd
from gymnasium import spaces
import json

class TradeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

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
                 liquidity_path="./csv/liquidity_system.csv",
                 config_path="./config.json"):

        super(TradeEnv, self).__init__()

        # ✅ 1. Clean maindf
        self.maindf = maindf.copy().reset_index(drop=True)
        self.maindf = self.maindf.fillna({
            'open': self.maindf['close'],
            'high': self.maindf['close'],
            'low': self.maindf['close'],
            'volume': 0,
            'value': 0,
            'trades': 0,
            'change': 0,
            'marketCap': 0,
            'RSI': 50,
            'bb_upper': self.maindf['close'],
            'bb_middle': self.maindf['close'],
            'bb_lower': self.maindf['close'],
            'macd': 0,
            'macd_signal': 0,
            'macd_hist': 0,
            'zigzag': 0,
            'Hammer': 'FALSE',
            'BullishEngulfing': 'FALSE',
            'MorningStar': 'FALSE'
        })
        self.maindf['date'] = pd.to_datetime(self.maindf['date'], errors='coerce')
        self.maindf['symbol'] = self.maindf['symbol'].str.strip().str.upper()
        self.maindf = self.maindf.dropna(subset=['date', 'symbol', 'close'])

        # ✅ 2. Load signal files
        self.gape_df = self._safe_load(gape_path, ['symbol'])
        self.gapebuy_df = self._safe_load(gapebuy_path, ['symbol', 'date'])
        self.shortbuy_df = self._safe_load(shortbuy_path, ['symbol', 'date'])
        self.rsi_diver_df = self._safe_load(rsi_diver_path, ['symbol'])
        self.rsi_diver_retest_df = self._safe_load(rsi_diver_retest_path, ['symbol'])

        # ✅ 3. Load SYSTEM INTELLIGENCE files
        self.trade_stock_df = self._load_and_clean_trade_stock(trade_stock_path)
        self.strategy_metrics = self._safe_load(metrics_path, ['Reference', 'Win%'])
        self.symbol_ref_metrics = self._safe_load(symbol_ref_path, ['Symbol', 'Reference', 'Win%', 'Expectancy (BDT)'])
        self.liquidity_df = self._safe_load(liquidity_path, ['symbol', 'liquidity_score'])

        # ✅ 4. Load config
        self._load_config(config_path)

        # ✅ 5. Env state
        self.cash = float(self.TOTAL_CAPITAL)
        self.positions = {}
        self.current_step = 0
        self.total_steps = max(0, len(self.maindf) - 1)
        self.last_obs = None

        # ✅ 6. Observation schema
        self.base_features = [
            'open', 'high', 'low', 'close', 'volume', 'value', 'trades', 'change', 'marketCap',
            'RSI', 'bb_upper', 'bb_middle', 'bb_lower',
            'macd', 'macd_signal', 'macd_hist', 'zigzag'
        ]  # 17
        self.pattern_cols = ['Hammer', 'BullishEngulfing', 'MorningStar']  # 3
        self.sys_features = [
            'has_signal_today', 'position_size_suggested', 'rrr_signal',
            'win_pct_symbol', 'expectancy_bdt', 'liquidity_score',
            'portfolio_exposure_ratio', 'unrealized_pnl_pct'
        ]  # 8

        self.obs_dim = len(self.base_features) + len(self.pattern_cols) + len(self.sys_features)
        print(f"✅ TradeEnv initialized with observation size = {self.obs_dim}")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell

    def _safe_load(self, path, required_cols):
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                df = pd.read_csv(path, low_memory=False)
                df.columns = df.columns.str.strip()
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = 0
                return df
        except Exception as e:
            print(f"⚠️ Failed to load {path}: {e}")
        return pd.DataFrame(columns=required_cols)

    def _load_and_clean_trade_stock(self, path):
        df = self._safe_load(path, ['symbol', 'date', 'buy', 'SL', 'tp', 'RRR', 'position_size'])

        # Standardize cols
        df['symbol'] = df['symbol'].str.strip().str.upper()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Critical: Remove rows with missing buy/SL/tp
        original_len = len(df)
        df = df.dropna(subset=['buy', 'SL', 'tp', 'position_size'])
        if len(df) < original_len:
            print(f"🧹 Cleaned {original_len - len(df)} invalid rows from {path} (missing buy/SL/tp/position_size)")

        # Convert & sanitize numeric cols
        for col in ['buy', 'SL', 'tp', 'RRR', 'position_size']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0).clip(lower=0)

        # Enforce valid logic: SL < buy < tp
        mask = (df['SL'] < df['buy']) & (df['buy'] < df['tp']) & (df['RRR'] > 0) & (df['position_size'] > 0)
        invalid = (~mask).sum()
        if invalid > 0:
            print(f"🧹 Removed {invalid} rows with invalid SL/buy/tp or RRR ≤ 0")
            df = df[mask].copy()

        return df.reset_index(drop=True)

    def _load_config(self, config_path):
        self.TOTAL_CAPITAL = 500000.0
        self.RISK_PERCENT = 0.01
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                    self.TOTAL_CAPITAL = float(cfg.get("total_capital", 500000))
                    self.RISK_PERCENT = float(cfg.get("risk_percent", 0.01))
            except Exception as e:
                print(f"⚠️ Config load error: {e}")

    def _get_open_signal(self, symbol, date):
        if self.trade_stock_df.empty:
            return None

        # Date-aware exact match
        try:
            target_date = pd.Timestamp(date).date()
        except:
            return None

        signal_row = self.trade_stock_df[
            (self.trade_stock_df['symbol'] == symbol) &
            (self.trade_stock_df['date'].dt.date == target_date)
        ]

        if signal_row.empty:
            return None

        s = signal_row.iloc[0].to_dict()

        # Final safety: ensure all critical fields are valid numbers
        for k in ['buy', 'SL', 'tp', 'RRR', 'position_size']:
            val = s.get(k, 0)
            if pd.isna(val) or not np.isfinite(val):
                return None  # ← Skip signal if ANY critical field is bad

        # Enforce types & bounds
        return {
            'buy': float(s['buy']),
            'SL': float(s['SL']),
            'tp': float(s['tp']),
            'RRR': float(max(0.1, s['RRR'])),  # min RRR = 0.1
            'position_size': int(max(1, round(s['position_size']))),
        }

    def _get_system_context(self, symbol, date):
        ctx = {
            'has_signal_today': 0,
            'position_size_suggested': 0,
            'rrr_signal': 0.0,
            'win_pct_symbol': 50.0,
            'expectancy_bdt': 0.0,
            'liquidity_score': 0.5,
            'portfolio_exposure_ratio': 0.0,
            'unrealized_pnl_pct': 0.0
        }

        # Signal?
        signal = self._get_open_signal(symbol, date)
        if signal:
            ctx.update({
                'has_signal_today': 1,
                'position_size_suggested': signal['position_size'],
                'rrr_signal': signal['RRR']
            })

        # Symbol metrics?
        if not self.symbol_ref_metrics.empty:
            ref = self.symbol_ref_metrics[
                (self.symbol_ref_metrics['Symbol'] == symbol) &
                (self.symbol_ref_metrics['Reference'] == 'SWING')
            ]
            if not ref.empty:
                row = ref.iloc[0]
                ctx['win_pct_symbol'] = float(np.clip(row.get('Win%', 50.0), 0, 100))
                ctx['expectancy_bdt'] = float(row.get('Expectancy (BDT)', 0.0))

        # Liquidity?
        if not self.liquidity_df.empty:
            liq = self.liquidity_df[self.liquidity_df['symbol'] == symbol]
            if not liq.empty:
                ctx['liquidity_score'] = float(np.clip(liq.iloc[0].get('liquidity_score', 0.5), 0, 1))

        # Portfolio context
        total_exposed = sum(p['shares'] * p['avg_price'] for p in self.positions.values())
        ctx['portfolio_exposure_ratio'] = float(np.clip(total_exposed / self.TOTAL_CAPITAL, 0, 1))

        if symbol in self.positions:
            pos = self.positions[symbol]
            current_price = float(self.maindf.iloc[min(self.current_step, len(self.maindf)-1)]['close'])
            avg_price = pos['avg_price']
            if avg_price > 0:
                ctx['unrealized_pnl_pct'] = (current_price - avg_price) / avg_price * 100

        return ctx

    def get_obs(self):
        if self.current_step >= len(self.maindf):
            # Return zero obs if out of bounds (fallback)
            return np.zeros(self.obs_dim, dtype=np.float32)

        row = self.maindf.iloc[self.current_step]
        symbol = str(row.get('symbol', 'UNKNOWN')).upper()
        date = row.get('date', pd.Timestamp.now())

        obs = []

        # Base features (17)
        for col in self.base_features:
            val = row.get(col, 0)
            try:
                obs.append(float(val))
            except (ValueError, TypeError):
                obs.append(0.0)

        # Patterns (3)
        for col in self.pattern_cols:
            val = str(row.get(col, 'FALSE')).upper()
            obs.append(1.0 if val == 'TRUE' else 0.0)

        # System context (8)
        sys_ctx = self._get_system_context(symbol, date)
        for col in self.sys_features:
            val = sys_ctx.get(col, 0.0)
            obs.append(float(val))

        # Final safety
        obs = np.array(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        if len(obs) != self.obs_dim:
            print(f"❗ Obs length fix: {len(obs)} → {self.obs_dim}")
            if len(obs) < self.obs_dim:
                obs = np.pad(obs, (0, self.obs_dim - len(obs)), constant_values=0.0)
            else:
                obs = obs[:self.obs_dim]

        return obs

    def step(self, action):
        if self.current_step >= len(self.maindf):
            obs = self.last_obs if self.last_obs is not None else np.zeros(self.obs_dim, dtype=np.float32)
            return obs, 0.0, True, False, {"terminated": True}

        row = self.maindf.iloc[self.current_step]
        symbol = str(row.get('symbol', 'N/A')).upper()
        price = float(row.get('close', 0.0))
        date = row.get('date', pd.Timestamp.now())

        reward = 0.0
        info = {}

        # Get signal
        signal = self._get_open_signal(symbol, date)

        # --- ACTION: BUY (1) ---
        if action == 1 and signal:
            shares = signal['position_size']
            cost = shares * price

            if shares > 0 and cost > 0 and self.cash >= cost and price >= signal['buy'] * 0.99:
                # Valid entry
                self.positions[symbol] = {
                    'shares': shares,
                    'avg_price': price,
                    'sl': signal['SL'],
                    'tp': signal['tp']
                }
                self.cash -= cost

                # Reward: positive + system boost
                bonus = (
                    min(2.0, signal['RRR'] * 0.2) +  # RRR bonus
                    (sys_ctx := self._get_system_context(symbol, date))['liquidity_score'] * 0.3 +
                    (sys_ctx['win_pct_symbol'] - 50) * 0.01
                )
                reward = 1.0 + np.clip(bonus, -0.5, 2.0)
            else:
                reward = -1.5  # Invalid buy attempt

        # --- ACTION: SELL (2) ---
        elif action == 2 and symbol in self.positions:
            pos = self.positions[symbol]
            proceeds = pos['shares'] * price
            profit = proceeds - (pos['shares'] * pos['avg_price'])

            self.cash += proceeds
            del self.positions[symbol]

            # Reward based on outcome
            if profit > 0:
                reward = 1.0 + profit / self.TOTAL_CAPITAL * 100  # scaled profit
            else:
                reward = -1.0 + profit / self.TOTAL_CAPITAL * 200  # penalize loss more

        # --- Auto-exit: SL/TP ---
        if symbol in self.positions:
            pos = self.positions[symbol]
            sl, tp = pos['sl'], pos['tp']
            if price <= sl and sl > 0:
                proceeds = pos['shares'] * price
                self.cash += proceeds
                del self.positions[symbol]
                reward += -1.0
                info['exit'] = 'SL'
            elif price >= tp and tp > 0:
                proceeds = pos['shares'] * price
                self.cash += proceeds
                del self.positions[symbol]
                reward += +0.8
                info['exit'] = 'TP'

        # Step forward
        self.current_step += 1
        done = self.current_step >= self.total_steps

        # Portfolio value
        current_price = price
        if self.current_step < len(self.maindf):
            current_price = float(self.maindf.iloc[self.current_step]['close'])
        portfolio_value = self.cash + sum(
            p['shares'] * current_price for p in self.positions.values()
        )
        info.update({
            'portfolio_value': float(portfolio_value),
            'cash': float(self.cash),
            'positions': len(self.positions),
            'symbol': symbol,
            'price': price,
            'step': self.current_step
        })

        obs = self.get_obs() if not done else (self.last_obs if self.last_obs is not None else np.zeros(self.obs_dim))
        self.last_obs = obs

        return np.array(obs, dtype=np.float32), float(np.clip(reward, -5, 5)), done, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = float(self.TOTAL_CAPITAL)
        self.positions = {}
        self.current_step = 0
        self.last_obs = self.get_obs()
        return self.last_obs, {}

    def render(self):
        if self.current_step >= len(self.maindf):
            print("⏹️ Env terminated.")
            return

        r = self.maindf.iloc[self.current_step]
        symbol = r.get('symbol', 'N/A')
        price = float(r.get('close', 0))
        pv = self.cash + sum(p['shares'] * price for p in self.positions.values())
        pnl = pv - self.TOTAL_CAPITAL
        pos_str = ", ".join([f"{s}×{p['shares']}" for s, p in self.positions.items()])
        print(f"[{symbol:10}] 💰 Cash: {self.cash:8.0f} | 📈 PnL: {pnl:+7.0f} | 📦 Pos: {pos_str or '—'}")