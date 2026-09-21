# ================== env_trading.py ==================
# FINAL VERSION — SIGNAL-FREE PURE RL (50-DIM)
# ✅ Dynamic SL/TP based on volatility (adaptive)
# ✅ Reduced SL penalty (0.2 → 0.05) + TP bonus
# ✅ Sector files loaded ONLY ONCE per env instance
# ✅ RSI/SR files cached at class level
# ✅ Trade tracking for metrics
# ✅ Patience for slow learning

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

# =========================================================
# ALL IMPORTS
# =========================================================

try:
    from sector_features import SectorFeatureEngine
    SECTOR_AVAILABLE = True
except ImportError:
    SECTOR_AVAILABLE = False

try:
    from patch_tst_predictor import PatchTSTIntegration
    PATCHTST_AVAILABLE = True
except ImportError:
    PATCHTST_AVAILABLE = False

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


# =========================================================
# TIER 1: MARKET MICROSTRUCTURE
# =========================================================

class MarketMicrostructure:
    @staticmethod
    def order_flow_imbalance(df):
        df = df.copy()
        df['price_change'] = df['close'].diff()
        df['volume_direction'] = np.where(df['price_change'] > 0, df['volume'],
                                  np.where(df['price_change'] < 0, -df['volume'], 0))
        df['ofi'] = df['volume_direction'].rolling(10).sum() / (df['volume'].rolling(10).sum() + 1e-8)
        return df['ofi'].fillna(0)

    @staticmethod
    def vwap_deviation(df):
        df = df.copy()
        df['cum_pv'] = (df['close'] * df['volume']).cumsum()
        df['cum_vol'] = df['volume'].cumsum()
        df['vwap'] = df['cum_pv'] / df['cum_vol']
        df['vwap_dev'] = (df['close'] - df['vwap']) / df['vwap']
        return df['vwap_dev'].fillna(0)

    @staticmethod
    def spread_proxy(df):
        df = df.copy()
        df['spread'] = (df['high'] - df['low']) / df['close']
        df['spread_ma'] = df['spread'].rolling(20).mean()
        df['spread_z'] = (df['spread'] - df['spread_ma']) / (df['spread'].rolling(20).std() + 1e-8)
        return df['spread_z'].fillna(0)

    @staticmethod
    def amihud_illiquidity(df):
        df = df.copy()
        df['daily_return'] = df['close'].pct_change()
        df['dollar_volume'] = df['close'] * df['volume']
        df['illiq'] = abs(df['daily_return']) / (df['dollar_volume'] + 1e-8)
        return df['illiq'].rolling(20).mean().fillna(0)

    @staticmethod
    def turnover_ratio(df):
        df = df.copy()
        if 'trades' in df.columns and 'marketCap' in df.columns:
            df['turnover_ratio'] = df['volume'] / (df['marketCap'] + 1e-8)
            return df['turnover_ratio'].fillna(0)
        return pd.Series(0, index=df.index)

    @staticmethod
    def bid_ask_bounce(df):
        df = df.copy()
        df['high_low_range'] = df['high'] - df['low']
        df['bounce'] = (df['close'] - df['low']) / (df['high_low_range'] + 1e-8)
        return df['bounce'].fillna(0.5)

    @staticmethod
    def compute_all(df):
        return pd.DataFrame({
            'ofi': MarketMicrostructure.order_flow_imbalance(df),
            'vwap_dev': MarketMicrostructure.vwap_deviation(df),
            'spread_z': MarketMicrostructure.spread_proxy(df),
            'illiq': MarketMicrostructure.amihud_illiquidity(df),
            'turnover_ratio': MarketMicrostructure.turnover_ratio(df),
            'bounce': MarketMicrostructure.bid_ask_bounce(df)
        }).fillna(0)


# =========================================================
# TIER 1: SYNTHETIC GREEKS
# =========================================================

class SyntheticGreeks:
    @staticmethod
    def delta(df, sector_returns, window=20):
        stock_returns = df['close'].pct_change()
        if len(stock_returns) > window:
            cov = stock_returns.rolling(window).cov(sector_returns)
            var = sector_returns.rolling(window).var()
            return (cov / (var + 1e-8)).fillna(1.0)
        return pd.Series(1.0, index=df.index)

    @staticmethod
    def gamma(delta_series):
        return delta_series.diff().fillna(0)

    @staticmethod
    def vega(df, vix_proxy):
        returns = df['close'].pct_change()
        return returns.rolling(20).corr(vix_proxy).fillna(0)


# =========================================================
# TIER 2: REGIME DETECTION (HMM)
# =========================================================

class MarketRegimeHMM:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = None
        self.regime_map = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}
        self.current_regime = 'SIDEWAYS'
        self.fitted = False

    def fit(self, returns, volumes):
        if not HMM_AVAILABLE or len(returns) < 50:
            return np.zeros(len(returns))
        try:
            features = np.column_stack([returns.fillna(0).values, volumes.fillna(0).values])
            if len(features) > 2000:
                features = features[-2000:]
            self.model = hmm.GaussianHMM(n_components=self.n_regimes,
                                          covariance_type="diag", n_iter=100)
            self.model.fit(features)
            self.fitted = True
            return self.model.predict(features)
        except:
            return np.zeros(len(returns))

    def predict(self, returns, volumes):
        if not self.fitted or self.model is None:
            return 1
        try:
            n = min(30, len(returns))
            if n < 5:
                return 1
            features = np.column_stack([
                returns.fillna(0).values[-n:],
                volumes.fillna(0).values[-n:]
            ])
            return int(self.model.predict(features)[-1])
        except:
            return 1

    def get_regime_multipliers(self, state):
        regime = self.regime_map.get(state, 'SIDEWAYS')
        if regime == 'BULL':
            return {'position_mult': 1.5, 'stop_mult': 1.2, 'reward_bonus': 1.15}
        elif regime == 'BEAR':
            return {'position_mult': 0.5, 'stop_mult': 0.8, 'reward_bonus': 0.85}
        else:
            return {'position_mult': 1.0, 'stop_mult': 1.0, 'reward_bonus': 1.0}


# =========================================================
# TIER 2: GARCH VOLATILITY
# =========================================================

def forecast_volatility(returns, horizon=5):
    if not ARCH_AVAILABLE or len(returns) < 30:
        return float(np.std(returns)) if len(returns) > 0 else 0.02
    try:
        returns_clean = returns.dropna()
        if len(returns_clean) < 30:
            return float(np.std(returns_clean)) if len(returns_clean) > 0 else 0.02
        model = arch_model(returns_clean, vol='Garch', p=1, q=1)
        fitted = model.fit(disp='off')
        forecast = fitted.forecast(horizon=horizon)
        return float(np.sqrt(forecast.variance.values[-1, -1]))
    except:
        return float(np.std(returns)) if len(returns) > 0 else 0.02


# =========================================================
# TIER 1: SECTOR LEADER DETECTOR
# =========================================================

class SectorLeaderDetector:
    def __init__(self):
        self.leaders = {}

    def detect_leader(self, sector_data):
        from collections import defaultdict
        symbols = sector_data['symbol'].unique()
        if len(symbols) < 2:
            return symbols[0] if len(symbols) > 0 else None

        leader_scores = defaultdict(int)
        for sym in symbols:
            sym_data = sector_data[sector_data['symbol'] == sym].sort_values('date')
            sym_returns = sym_data['close'].pct_change().dropna()
            for other_sym in symbols:
                if other_sym != sym:
                    other_data = sector_data[sector_data['symbol'] == other_sym].sort_values('date')
                    other_returns = other_data['close'].pct_change().dropna()
                    min_len = min(len(sym_returns), len(other_returns)) - 1
                    if min_len > 10:
                        lead_corr = sym_returns.iloc[:min_len].corr(other_returns.iloc[1:min_len+1])
                        if abs(lead_corr) > 0.6:
                            leader_scores[sym] += 1

        if leader_scores:
            return max(leader_scores, key=leader_scores.get)
        return symbols[0]


# =========================================================
# TIER 3: PORTFOLIO OPTIMIZATION
# =========================================================

class PortfolioOptimizer:
    @staticmethod
    def risk_parity_weights(returns_df):
        cov = returns_df.cov().values
        inv_vol = 1.0 / np.sqrt(np.diag(cov) + 1e-8)
        weights = inv_vol / inv_vol.sum()
        return weights

    @staticmethod
    def min_variance_weights(returns_df):
        cov = returns_df.cov().values
        n = len(cov)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        def portfolio_var(w):
            return w @ cov @ w

        constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})
        bounds = [(0, 0.3) for _ in range(n)]
        try:
            result = minimize(portfolio_var, np.ones(n)/n, bounds=bounds,
                              constraints=constraints, method='SLSQP')
            return result.x if result.success else np.ones(n)/n
        except:
            return np.ones(n)/n


# =========================================================
# RSI DIVERGENCE — CLASS-LEVEL CACHE
# =========================================================

class RSIDivergenceFeatures:
    _CACHE = {}

    def __init__(self, csv_path="./csv/rsi_diver.csv"):
        if csv_path in RSIDivergenceFeatures._CACHE:
            self.data = RSIDivergenceFeatures._CACHE[csv_path]
        else:
            self.data = self._load(csv_path)
            RSIDivergenceFeatures._CACHE[csv_path] = self.data

    def _load(self, path):
        if not Path(path).exists():
            return {}
        try:
            df = pd.read_csv(path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            data = {}
            for sym in df['symbol'].unique():
                data[sym] = df[df['symbol'] == sym].reset_index(drop=True)
            print(f"✅ RSI Divergence: {len(data)} symbols (cached)")
            return data
        except:
            return {}

    def get_features(self, symbol, current_date):
        if not self.data or symbol not in self.data:
            return np.zeros(3, dtype=np.float32)
        try:
            df = self.data[symbol]
            current_dt = pd.to_datetime(current_date)
            recent = df[df['date'] <= current_dt].tail(1)
            if recent.empty:
                return np.zeros(3, dtype=np.float32)

            row = recent.iloc[-1]
            div_type = str(row.get('divergence_type', 'NONE')).upper()
            div_signal = 1.0 if 'BULLISH' in div_type else 0.0 if 'BEARISH' in div_type else 0.5

            strength = str(row.get('divergence_strength', 'NONE')).upper()
            strength_map = {'STRONG': 1.0, 'MODERATE': 0.6, 'WEAK': 0.3}
            div_strength = strength_map.get(strength, 0.0)

            rsi = float(row.get('rsi', 50))
            rsi_norm = np.clip((rsi - 30) / 40, 0, 1)

            return np.array([div_signal, div_strength, rsi_norm], dtype=np.float32)
        except:
            return np.zeros(3, dtype=np.float32)


# =========================================================
# SUPPORT / RESISTANCE — CLASS-LEVEL CACHE
# =========================================================

class SupportResistanceFeatures:
    _CACHE = {}

    def __init__(self, csv_path="./csv/support_resistance.csv"):
        if csv_path in SupportResistanceFeatures._CACHE:
            self.data = SupportResistanceFeatures._CACHE[csv_path]
        else:
            self.data = self._load(csv_path)
            SupportResistanceFeatures._CACHE[csv_path] = self.data

    def _load(self, path):
        if not Path(path).exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
            if 'current_date' in df.columns:
                df['current_date'] = pd.to_datetime(df['current_date'])
            print(f"✅ Support/Resistance: {df['symbol'].nunique()} symbols (cached)")
            return df
        except:
            return pd.DataFrame()

    def get_features(self, symbol, current_date, current_close):
        if self.data.empty or symbol not in self.data['symbol'].values:
            return np.zeros(3, dtype=np.float32)
        try:
            current_dt = pd.to_datetime(current_date)
            sym_data = self.data[self.data['symbol'] == symbol]
            recent = sym_data[sym_data['current_date'] <= current_dt].tail(1)
            if recent.empty:
                return np.zeros(3, dtype=np.float32)

            row = recent.iloc[-1]
            level_price = float(row['level_price'])
            distance_pct = (current_close - level_price) / current_close if current_close > 0 else 0

            strength_str = str(row.get('strength', 'Weak')).capitalize()
            strength_map = {'Weak': 0.3, 'Moderate': 0.6, 'Strong': 1.0}
            strength_val = strength_map.get(strength_str, 0.5)

            level_type = str(row.get('type', '')).lower()
            type_val = 1.0 if level_type == 'support' else -1.0 if level_type == 'resistance' else 0.0

            return np.array([distance_pct, strength_val, type_val], dtype=np.float32)
        except:
            return np.zeros(3, dtype=np.float32)


# =========================================================
# MAIN ENVIRONMENT
# =========================================================

class MultiSymbolTradingEnv(gym.Env):
    """
    Multi-symbol trading environment for PPO.
    Action per symbol: 0=HOLD, 1=BUY, 2=SELL

    ✅ KEY FIXES:
        - Dynamic SL/TP based on entry volatility
        - Small SL penalty, TP bonus
        - Sector loaded once per env
        - Patience for slow learning
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        symbol_dfs,
        build_observation,
        window,
        state_dim,
        total_capital=500_000,
        risk_percent=0.01,
        sector_engine=None,
        xgb_models=None,
        agentic_loop=None,
        patch_tst=None,
        # Default SL/TP (will be overridden by dynamic)
        sl_pct=0.04,
        tp_pct=0.08,
        # Dynamic SL/TP multipliers (relative to entry vol)
        sl_vol_mult=2.0,       # SL = 2×entry_vol
        tp_vol_mult=4.0,       # TP = 4×entry_vol (2:1 R:R)
        min_sl_pct=0.03,       # min 3% SL
        max_sl_pct=0.10,       # max 10% SL
        min_tp_pct=0.06,       # min 6% TP
        max_tp_pct=0.20,       # max 20% TP
        # Reward shaping
        max_position_pct=0.30,
        hold_penalty=0.0001,
        illegal_action_penalty=0.002,
        open_cost=0.001,
        sl_penalty=0.05,
        tp_bonus=0.10,
        signals=None,
    ):
        super().__init__()

        self.symbols = list(symbol_dfs.keys())
        self.dfs = symbol_dfs
        self.build_observation = build_observation
        self.window = window
        self.state_dim = state_dim

        self.total_capital = total_capital
        self.risk_percent = risk_percent

        # Dynamic SL/TP
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.sl_vol_mult = sl_vol_mult
        self.tp_vol_mult = tp_vol_mult
        self.min_sl_pct = min_sl_pct
        self.max_sl_pct = max_sl_pct
        self.min_tp_pct = min_tp_pct
        self.max_tp_pct = max_tp_pct

        # Reward shaping
        self.max_position_pct = max_position_pct
        self.hold_penalty = hold_penalty
        self.illegal_action_penalty = illegal_action_penalty
        self.open_cost = open_cost
        self.sl_penalty = sl_penalty
        self.tp_bonus = tp_bonus

        self.n_symbols = len(self.symbols)
        self.max_steps = max(len(df) for df in self.dfs.values())

        # External models
        self.xgb_models = xgb_models or {}
        self.agentic_loop = agentic_loop

        # Tier components
        self.micro = MarketMicrostructure()
        self.leader_detector = SectorLeaderDetector()
        self.sector_leaders = {}
        self.greeks = SyntheticGreeks()
        self.sector_returns_cache = None
        self.vix_proxy_cache = None

        self.regime_model = MarketRegimeHMM(n_regimes=3)
        self.current_regime = 'SIDEWAYS'
        self.current_state = 1
        self.regime_fitted = False

        self.optimizer = PortfolioOptimizer()
        self.portfolio_weights = None

        self.patch_tst = patch_tst
        self.rsi_div = RSIDivergenceFeatures()
        self.sr_features = SupportResistanceFeatures()

        # Sector
        self.sector_engine = sector_engine
        self.sector_features_enabled = False

        if self.sector_engine is not None and SECTOR_AVAILABLE:
            try:
                combined_df = pd.concat(self.dfs.values(), ignore_index=True)
                self.sector_engine.update(combined_df)
                self.sector_features_enabled = True
            except:
                self.sector_features_enabled = False

        self.effective_state_dim = self.state_dim

        self.action_space = spaces.MultiDiscrete([3] * self.n_symbols)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_symbols, self.effective_state_dim),
            dtype=np.float32,
        )

        self._last_trades = []
        self._sector_initialized = False
        self._regime_fitted_flag = False
        self._entry_vol = {}

    # -------------------------------------------------
    # Tier 1: Microstructure
    # -------------------------------------------------
    def _get_microstructure_features(self, df, idx):
        if idx < 20:
            return np.zeros(6, dtype=np.float32)
        try:
            df_slice = df.iloc[:idx+1].copy()
            micro_df = self.micro.compute_all(df_slice)
            last_row = micro_df.iloc[-1]
            return np.array([
                last_row['ofi'], last_row['vwap_dev'], last_row['spread_z'],
                last_row['illiq'], last_row['turnover_ratio'], last_row['bounce']
            ], dtype=np.float32)
        except:
            return np.zeros(6, dtype=np.float32)

    # -------------------------------------------------
    # Tier 1: Greeks
    # -------------------------------------------------
    def _get_greek_features(self, df, idx):
        if idx < 20:
            return np.zeros(3, dtype=np.float32)
        try:
            df_slice = df.iloc[:idx+1]
            if self.sector_returns_cache is None:
                combined = pd.concat(self.dfs.values(), ignore_index=True)
                if 'date' in combined.columns:
                    self.sector_returns_cache = combined.groupby('date')['close'].mean().pct_change()
                else:
                    self.sector_returns_cache = combined['close'].pct_change()
            if self.vix_proxy_cache is None:
                self.vix_proxy_cache = df_slice['close'].pct_change().rolling(20).std()

            delta = self.greeks.delta(df_slice, self.sector_returns_cache)
            gamma = self.greeks.gamma(delta)
            vega = self.greeks.vega(df_slice, self.vix_proxy_cache)

            return np.array([
                delta.iloc[-1] if not delta.empty else 1.0,
                gamma.iloc[-1] if not gamma.empty else 0.0,
                vega.iloc[-1] if not vega.empty else 0.0
            ], dtype=np.float32)
        except:
            return np.zeros(3, dtype=np.float32)

    # -------------------------------------------------
    # Tier 2: Regime
    # -------------------------------------------------
    def _update_regime_state(self, df, idx):
        if idx < 50 or not self.regime_fitted:
            return
        try:
            df_slice = df.iloc[:idx+1]
            returns = df_slice['close'].pct_change().fillna(0)
            volumes = df_slice['volume'].fillna(0)
            self.current_state = self.regime_model.predict(returns, volumes)
            self.current_regime = self.regime_model.regime_map.get(self.current_state, 'SIDEWAYS')
        except:
            pass

    # -------------------------------------------------
    # Sector leaders
    # -------------------------------------------------
    def _detect_sector_leaders(self):
        if not self.sector_features_enabled or self.sector_engine is None:
            return
        try:
            combined = pd.concat(self.dfs.values(), ignore_index=True)
            sectors = self.sector_engine.get_all_sectors()
            for sector in sectors:
                sector_symbols = self.sector_engine.get_symbols_in_sector(sector)
                sector_data = combined[combined['symbol'].isin(sector_symbols)]
                if len(sector_data) > 50:
                    leader = self.leader_detector.detect_leader(sector_data)
                    if leader:
                        self.sector_leaders[sector] = leader
        except:
            pass

    # -------------------------------------------------
    # Portfolio weights
    # -------------------------------------------------
    def _calculate_portfolio_weights(self):
        try:
            returns_dict = {}
            for s in self.symbols:
                df = self.dfs[s]
                if len(df) > 20:
                    returns_dict[s] = df['close'].pct_change().dropna()
            if len(returns_dict) > 1:
                returns_df = pd.DataFrame(returns_dict).dropna()
                if len(returns_df) > 20:
                    self.portfolio_weights = self.optimizer.risk_parity_weights(returns_df)
        except:
            self.portfolio_weights = None

    # -------------------------------------------------
    # Sector reward multiplier
    # -------------------------------------------------
    def _get_sector_reward_multiplier(self, symbol):
        if not self.sector_features_enabled or self.sector_engine is None:
            return 1.0
        try:
            sector = self.sector_engine.get_sector(symbol)
            top3 = [s for s, _ in self.sector_engine.get_top_sectors(3)]
            bottom2 = [s for s, _ in self.sector_engine.get_bottom_sectors(2)]
            if sector in top3: return 1.15
            elif sector in bottom2: return 0.90
            return 1.0
        except:
            return 1.0

    # -------------------------------------------------
    # RESET
    # -------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.balance = {s: self.total_capital for s in self.symbols}
        self.position = {s: 0 for s in self.symbols}
        self.entry_price = {s: 0.0 for s in self.symbols}
        self._last_trades = []
        self._entry_vol = {}
        self.sector_returns_cache = None
        self.vix_proxy_cache = None
        self.current_state = 1
        self.current_regime = 'SIDEWAYS'

        if not self._regime_fitted_flag and HMM_AVAILABLE:
            try:
                combined = pd.concat(self.dfs.values(), ignore_index=True)
                if 'date' in combined.columns:
                    returns = combined.groupby('date')['close'].mean().pct_change().fillna(0)
                    volumes = combined.groupby('date')['volume'].mean().fillna(0)
                else:
                    returns = combined['close'].pct_change().fillna(0)
                    volumes = combined['volume'].fillna(0)
                self.regime_model.fit(returns, volumes)
                self.regime_fitted = True
                self._regime_fitted_flag = True
            except:
                self.regime_fitted = False

        if not self._sector_initialized:
            if self.sector_features_enabled and self.sector_engine is not None:
                try:
                    combined_df = pd.concat(self.dfs.values(), ignore_index=True)
                    self.sector_engine.update(combined_df)
                    self._detect_sector_leaders()
                    self._calculate_portfolio_weights()
                    self._sector_initialized = True
                except:
                    pass

        return self._get_obs(), {}

    # -------------------------------------------------
    # OBSERVATION
    # -------------------------------------------------
    def _get_obs(self):
        obs = []
        for s in self.symbols:
            df = self.dfs[s]
            if self.t < len(df):
                self._update_regime_state(df, self.t)
                o = self.build_observation(df, self.t)
                o = np.asarray(o, dtype=np.float32).flatten()
                if len(o) < self.effective_state_dim:
                    o = np.pad(o, (0, self.effective_state_dim - len(o)))
                elif len(o) > self.effective_state_dim:
                    o = o[:self.effective_state_dim]
                o = np.nan_to_num(o)
            else:
                o = np.zeros(self.effective_state_dim, dtype=np.float32)
            obs.append(o)

        return np.asarray(obs, dtype=np.float32)

    # -------------------------------------------------
    # STEP — dynamic SL/TP
    # -------------------------------------------------
    def step(self, actions):
        self._last_trades = []

        if np.isscalar(actions):
            actions_list = [int(actions)] * self.n_symbols
        elif isinstance(actions, np.ndarray):
            actions_list = actions.flatten().tolist()
        elif isinstance(actions, list):
            actions_list = actions
        else:
            actions_list = [0] * self.n_symbols

        rewards = []
        done_flags = []

        for i, s in enumerate(self.symbols):
            action = int(actions_list[i]) if i < len(actions_list) else 0
            df = self.dfs[s]

            if self.t >= len(df):
                rewards.append(0.0)
                done_flags.append(True)
                continue

            row = df.iloc[self.t]
            price = float(row["close"])
            reward = 0.0

            # ---- Auto SL/TP + manual close ----
            if self.position[s] > 0:
                entry = self.entry_price[s]

                # Dynamic SL/TP from entry vol
                vol = self._entry_vol.get(s, 0.02)
                dyn_sl = float(np.clip(vol * self.sl_vol_mult, self.min_sl_pct, self.max_sl_pct))
                dyn_tp = float(np.clip(vol * self.tp_vol_mult, self.min_tp_pct, self.max_tp_pct))

                sl_price = entry * (1.0 - dyn_sl)
                tp_price = entry * (1.0 + dyn_tp)

                should_close = False
                close_reason = None

                if price <= sl_price:
                    should_close = True
                    close_reason = 'sl'
                elif price >= tp_price:
                    should_close = True
                    close_reason = 'tp'
                elif action == 2:
                    should_close = True
                    close_reason = 'signal'

                if should_close:
                    pnl = (price - entry) * self.position[s]
                    self.balance[s] += self.position[s] * price

                    risk_amount = self.total_capital * self.risk_percent
                    reward = float(np.tanh(pnl / (risk_amount * 0.5)))

                    reward *= self._get_sector_reward_multiplier(s)

                    if self.regime_fitted:
                        mults = self.regime_model.get_regime_multipliers(self.current_state)
                        reward *= mults['reward_bonus']

                    if close_reason == 'sl':
                        reward -= self.sl_penalty
                    if close_reason == 'tp':
                        reward += self.tp_bonus

                    self._last_trades.append({
                        'success': bool(pnl > 0),
                        'pnl': float(pnl),
                        'entry_price': float(entry),
                        'exit_price': float(price),
                        'exit_reason': str(close_reason),
                        'symbol': s,
                    })

                    self.position[s] = 0
                    self.entry_price[s] = 0.0
                    self._entry_vol[s] = 0.02

            # ---- Open new position ----
            if action == 1 and self.position[s] == 0:
                if self.t > 20:
                    recent_vol = df['close'].iloc[max(0, self.t-20):self.t+1].pct_change().std()
                    recent_vol = max(float(recent_vol) if pd.notna(recent_vol) else 0.02, 0.005)
                else:
                    recent_vol = 0.02

                risk_amount = self.total_capital * self.risk_percent
                position_value = risk_amount / (recent_vol * 3.0)
                position_value = min(position_value, self.balance[s] * self.max_position_pct)

                shares = int(position_value / price) if price > 0 else 0

                if self.regime_fitted:
                    mults = self.regime_model.get_regime_multipliers(self.current_state)
                    shares = int(shares * mults['position_mult'])

                shares = max(shares, 0)

                if shares > 0 and shares * price <= self.balance[s]:
                    self.position[s] = shares
                    self.entry_price[s] = price
                    self.balance[s] -= shares * price
                    self._entry_vol[s] = recent_vol
                    reward -= self.open_cost

            # ---- Inaction penalties ----
            if action == 0 and self.position[s] == 0:
                reward -= self.hold_penalty

            if action == 2 and self.position[s] == 0:
                reward -= self.illegal_action_penalty

            rewards.append(float(reward))

            is_done = (self.t >= len(df) - 1) and (self.t >= 30)
            done_flags.append(is_done)

        self.t += 1
        terminated = all(done_flags)
        truncated = False

        info = {
            'trade_result': self._last_trades[0] if self._last_trades else None,
            'trades': self._last_trades,
        }
        return self._get_obs(), float(np.sum(rewards)), terminated, truncated, info

    # -------------------------------------------------
    # RENDER
    # -------------------------------------------------
    def render(self):
        print(f"\nStep {self.t}")
        for s in self.symbols:
            print(f"{s} | Balance: {self.balance[s]:.2f} | Position: {self.position[s]}")

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------
    def get_sector_summary(self):
        if self.sector_engine:
            return self.sector_engine.get_summary()
        return {}

    def get_top_sectors(self, n=3):
        if self.sector_engine:
            return self.sector_engine.get_top_sectors(n)
        return []

    def export_sector_rankings(self, path="./csv/sector_rankings.csv"):
        if self.sector_engine:
            return self.sector_engine.export_rankings(path)
        return None

    def get_regime(self):
        return self.current_regime

    def get_leaders(self):
        return self.sector_leaders

    def get_weights(self):
        return self.portfolio_weights