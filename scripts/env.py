import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import math

class TradeEnv(gym.Env):
    def __init__(self,
        main_df,
        filtered_df,
        imbalance_high_df,
        imbalance_low_df,
        swing_high_candle_df,
        swing_high_confirm_df,
        swing_low_candle_df,
        swing_low_confirm_df,
        rsi_divergence_df,
        filtered_output,
        down_to_up_df,
        up_to_down_df
    ):
        super().__init__()

        self.main_df = main_df.reset_index(drop=True)
        self.filtered_df = filtered_df if filtered_df is not None else pd.DataFrame()
        self.imbalance_high_df = imbalance_high_df if imbalance_high_df is not None else pd.DataFrame()
        self.imbalance_low_df = imbalance_low_df if imbalance_low_df is not None else pd.DataFrame()
        self.swing_high_candle_df = swing_high_candle_df if swing_high_candle_df is not None else pd.DataFrame()
        self.swing_high_confirm_df = swing_high_confirm_df if swing_high_confirm_df is not None else pd.DataFrame()
        self.swing_low_candle_df = swing_low_candle_df if swing_low_candle_df is not None else pd.DataFrame()
        self.swing_low_confirm_df = swing_low_confirm_df if swing_low_confirm_df is not None else pd.DataFrame()
        self.rsi_divergence_df = rsi_divergence_df if rsi_divergence_df is not None else pd.DataFrame()
        self.filtered_output = filtered_output if filtered_output is not None else pd.DataFrame()
        self.down_to_up_df = down_to_up_df if down_to_up_df is not None else pd.DataFrame()
        self.up_to_down_df = up_to_down_df if up_to_down_df is not None else pd.DataFrame()

        self.total_steps = len(self.main_df) - 1

        self.action_space = spaces.Discrete(3)

        # Temp obs to determine shape
        tmp_obs = self._make_initial_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(tmp_obs),), dtype=np.float32)

        self.reset()

    def _safe_float(self, val, default=0.0):
        try:
            f = float(val)
            if math.isnan(f):
                return default
            return f
        except (ValueError, TypeError):
            return default

    def _clean_confidence(self, val):
        try:
            if isinstance(val, str) and '%' in val:
                val = val.replace('%', '')
            return float(val) / 100.0
        except Exception:
            return 0.0

    def _map_trend(self, trend_str):
        trend_str = str(trend_str).lower()
        if "down" in trend_str:
            return -1
        elif "up" in trend_str:
            return 1
        return 0

    def _pattern_flag(self, pattern, row):
        return 1.0 if row.get(pattern, False) else 0.0

    def reset(self, *, seed=None, options=None):
        
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = 100000.0
        self.stock = 0
        self.last_price = self._safe_float(self.main_df.iloc[0].get('close'))
        obs = self._get_obs()
        self.last_obs = obs  # ✅ সংরক্ষণ
        return obs, {}

    def _make_initial_obs(self):
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        try:
            r = self.main_df.iloc[self.current_step]
            price = self._safe_float(r.get('close'))
            symbol = r.get('symbol', '')
            matched = 0

            rsi_div = 0.0
            if not self.rsi_divergence_df.empty:
                try:
                    match = self.rsi_divergence_df[(self.rsi_divergence_df['symbol'] == symbol) & (self.rsi_divergence_df['date'] == date)]
                    if not match.empty:
                        rsi_div = self._safe_float(match['rsi_divergence'].iloc[0])
                except Exception:
                    pass

            ob_low = ob_high = fvg_low = fvg_high = f38 = f50 = f61 = 0.0
            trnd = 0.0
            if not self.imbalance_high_df.empty:
                try:
                    ib = self.imbalance_high_df[(self.imbalance_high_df['symbol'] == symbol) & (self.imbalance_high_df['date'] == date)]
                    if not ib.empty:
                        ib0 = ib.iloc[0]
                        ob_low = self._safe_float(ib0.get('orderblock_low'))
                        ob_high = self._safe_float(ib0.get('orderblock_high'))
                        fvg_low = self._safe_float(ib0.get('fvg_low'))
                        fvg_high = self._safe_float(ib0.get('fvg_high'))
                        f38 = self._safe_float(ib0.get('fib_38.2%'))
                        f50 = self._safe_float(ib0.get('fib_50%'))
                        f61 = self._safe_float(ib0.get('fib_61.8%'))
                        trnd = self._map_trend(ib0.get('trand'))
                except Exception:
                    pass

            du_ob_low = du_fvg_low = du_fib50 = du_trend = 0.0
            if not self.down_to_up_df.empty:
                try:
                    du_match = self.down_to_up_df[(self.down_to_up_df['SYMBOL'] == symbol) & (self.down_to_up_df['DATE'] == date)]
                    if not du_match.empty:
                        row = du_match.iloc[0]
                        du_ob_low = self._safe_float(row.get('ORDERBLOCK_LOW'))
                        du_fvg_low = self._safe_float(row.get('FVG_LOW'))
                        du_fib50 = self._safe_float(row.get('FIB_50%'))
                        du_trend = self._map_trend(row.get('TRAND'))
                except Exception:
                    pass

            ud_ob_high = ud_fvg_high = ud_fib50 = ud_trend = 0.0
            if not self.up_to_down_df.empty:
                try:
                    ud_match = self.up_to_down_df[(self.up_to_down_df['SYMBOL'] == symbol) & (self.up_to_down_df['DATE'] == date)]
                    if not ud_match.empty:
                        row = ud_match.iloc[0]
                        ud_ob_high = self._safe_float(row.get('ORDERBLOCK_HIGH'))
                        ud_fvg_high = self._safe_float(row.get('FVG_HIGH'))
                        ud_fib50 = self._safe_float(row.get('FIB_50%'))
                        ud_trend = self._map_trend(row.get('TRAND'))
                except Exception:
                    pass

            bb_upper = self._safe_float(r.get('bb_upper'))
            bb_middle = self._safe_float(r.get('bb_middle'))
            bb_lower = self._safe_float(r.get('bb_lower'))
            macd = self._safe_float(r.get('macd'))
            macd_signal = self._safe_float(r.get('macd_signal'))
            macd_hist = self._safe_float(r.get('macd_hist'))
            zigzag = self._safe_float(r.get('zigzag'))
            rsi = self._safe_float(r.get('rsi'))

            hammer = self._pattern_flag('Hammer', r)
            bullish_engulf = self._pattern_flag('BullishEngulfing', r)
            morning_star = self._pattern_flag('MorningStar', r)
            piercing = self._pattern_flag('PiercingLine', r)
            three_white = self._pattern_flag('ThreeWhiteSoldiers', r)
            doji = self._pattern_flag('Doji', r) if trnd == -1 else 0.0

            hf = lf = 0

            
            # ✅ MACD cross
            macd = self._safe_float(r.get('macd'))
            macd_signal = self._safe_float(r.get('macd_signal'))
            macd_cross = 1.0 if macd > macd_signal else -1.0

            # ✅ BB Position: normalized between bb_upper & bb_lower
            bb_upper = self._safe_float(r.get('bb_upper'))
            bb_lower = self._safe_float(r.get('bb_lower'))
            bb_position = 0.5  # default
            if bb_upper != bb_lower:
                bb_position = (price - bb_lower) / (bb_upper - bb_lower)
                bb_position = np.clip(bb_position, 0.0, 1.0)

            # ✅ RSI trend (change in 3 days)
            rsi = self._safe_float(r.get('rsi'))
            rsi_trend = 0.0
            if self.current_step >= 3:
                rsi_prev = self._safe_float(self.main_df.iloc[self.current_step - 3].get('rsi'))
                rsi_trend = rsi - rsi_prev

            # ✅ Price change 1d
            price_change_1d = 0.0
            if self.current_step >= 1:
                prev_close = self._safe_float(self.main_df.iloc[self.current_step - 1].get('close'))
                if prev_close != 0:
                    price_change_1d = (price - prev_close) / prev_close

            # ✅ Price change 3d
            price_change_3d = 0.0
            if self.current_step >= 3:
                close_3d = self._safe_float(self.main_df.iloc[self.current_step - 3].get('close'))
                if close_3d != 0:
                    price_change_3d = (price - close_3d) / close_3d
            rsi_trend = np.clip(rsi_trend / 100.0, -1.0, 1.0)
            price_change_1d = np.clip(price_change_1d, -1.0, 1.0)
            price_change_3d = np.clip(price_change_3d, -1.0, 1.0)
            # ... ✅ আপনার পূর্বের সকল obs উপাদান এখানে থাকবে .
            try:
                if not self.imbalance_high_df.empty:
                    hf = int(((self.imbalance_high_df['low'] <= price) & (self.imbalance_high_df['high'] >= price)).any())
                if not self.imbalance_low_df.empty:
                    lf = int(((self.imbalance_low_df['low'] <= price) & (self.imbalance_low_df['high'] >= price)).any())

            
            except Exception:
                pass

            swing_high_rsi = self._safe_float(self.swing_high_candle_df.get('rsi', pd.Series([0.0])).iloc[0]) if not self.swing_high_candle_df.empty else 0.0
            swing_low_rsi = self._safe_float(self.swing_low_candle_df.get('rsi', pd.Series([0.0])).iloc[0]) if not self.swing_low_candle_df.empty else 0.0
            matched = 0
            if not self.filtered_output.empty:
                matched = int(
                ((self.filtered_output['symbol'] == symbol) & 
                (self.filtered_output['date'] == pd.to_datetime(date).strftime('%Y-%m-%d'))).any()
        )

            confidence = self._clean_confidence(r.get('CONFIDENCE'))
           
            

            obs = np.array([
                self._safe_float(r.get('marketCap')),
                self._safe_float(r.get('change')),
                self._safe_float(r.get('trades')),
                self._safe_float(r.get('value')),
                self._safe_float(r.get('volume')),
                price,
                self._safe_float(r.get('open')),
                self._safe_float(r.get('high')),
                self._safe_float(r.get('low')),
                self.current_step,

                f38, f50, f61, hf, lf,

                rsi_div,
                swing_high_rsi,
                swing_low_rsi,

                ob_low, ob_high, fvg_low, fvg_high,
                f38, f50, f61, trnd,

                bb_upper, bb_middle, bb_lower,
                macd, macd_signal, macd_hist,
                zigzag, rsi,

                hammer, bullish_engulf, morning_star,
                piercing, three_white, doji,

                du_ob_low, du_fvg_low, du_fib50, du_trend,
                ud_ob_high, ud_fvg_high, ud_fib50, ud_trend,

                bb_position,
                macd_cross,
                rsi_trend,
                price_change_1d,
                price_change_3d,
                matched,

                confidence
            ], dtype=np.float32)

            if np.isnan(obs).any():
                print(f"\U0001f6a8 NaN in observation at step {self.current_step}")
                print(obs)
                raise ValueError("NaN found in observation")

            return obs
        except Exception as e:
            print(f"Error in _get_obs: {e}")
            raise

    def step(self, action):
        is_matched = False
        r = self.main_df.iloc[self.current_step]
        price = self._safe_float(r.get('close'))
        confidence = self._clean_confidence(r.get('CONFIDENCE'))
        symbol = r.get('symbol', '')
        date = r.get('date', None)

        reward = 0.0
        if action == 1 and self.cash >= price:
            self.stock += 1
            self.cash -= price
            self.last_price = price  # ✅ move here, only when bought
            reward = -0.1 * (1 - confidence)
        elif action == 2 and self.stock > 0:
            self.stock -= 1
            self.cash += price
            profit = price - self.last_price
            reward = profit * (1 + confidence * 1.5) if profit > 0 else profit
        elif action == 0:
        # ✅ Hold penalty
            reward = -0.01  # Encourage exploration

       
        self.current_step += 1
        done = self.current_step >= self.total_steps

        if not done:
          obs = self._get_obs()
          self.last_obs = obs  # ✅ সর্বশেষ সঠিক obs রাখুন
        else:
          obs = self.last_obs  # ✅ শেষ observation ফিরিয়ে দিন

        if math.isnan(reward):
            print(f"\U0001f6a8 NaN in reward at step {self.current_step}")
            print(f"Price: {price}, Last: {self.last_price}, Confidence: {confidence}")
            raise ValueError("NaN found in reward")
        
        if not self.filtered_output.empty:
            is_matched = (
            (self.filtered_output['symbol'] == symbol) &
            (self.filtered_output['date'] == date)
        ).any()
        if is_matched and action == 1:
            reward += 1.0 
        reward = np.clip(reward, -10, 10)

        info = {
            'cash': self.cash,
            'stock': self.stock,
            'portfolio_value': self.cash + self.stock * price,
            'price': price,
            'action': action,
            'symbol': self.main_df.iloc[min(self.current_step, self.total_steps)]['symbol'], # ✅ যোগ করুন
            'reward': reward,
            'confidence': confidence,
            'last_price': self.last_price
        }

        return obs, reward, done, False, info

    def render(self):
        r = self.main_df.iloc[self.current_step]
        price = self._safe_float(r.get('close'))
        print(f"Step {self.current_step}: Cash {self.cash:.2f}, Stock {self.stock}, Value {self.cash + self.stock * price:.2f}")
