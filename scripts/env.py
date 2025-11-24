import gymnasium as gym
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
                 rsi_diver_retest_path="./csv/rsi_diver_retest.csv"):

        super(TradeEnv, self).__init__()

        self.maindf = maindf.copy().reset_index(drop=True)
        self.gape_df = pd.read_csv(gape_path) if os.path.exists(gape_path) else pd.DataFrame()
        self.gapebuy_df = pd.read_csv(gapebuy_path) if os.path.exists(gapebuy_path) else pd.DataFrame()
        self.shortbuy_df = pd.read_csv(shortbuy_path) if os.path.exists(shortbuy_path) else pd.DataFrame()
        self.rsi_diver_df = pd.read_csv(rsi_diver_path) if os.path.exists(rsi_diver_path) else pd.DataFrame()
        self.rsi_diver_retest_df = pd.read_csv(rsi_diver_retest_path) if os.path.exists(rsi_diver_retest_path) else pd.DataFrame()

        self.cash = 10000
        self.stock = 0
        self.last_price = None
        self.current_step = 0
        self.total_steps = len(self.maindf) - 1
        self.last_obs = None

        obs_shape = self.get_obs().shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        self.cash = 10000
        self.stock = 0
        self.last_price = None
        self.current_step = 0
        self.last_obs = self.get_obs()
        return self.last_obs, {}

    def get_obs(self):
        window = 20
        start = max(0, self.current_step - window + 1)
        obs_df = self.maindf.iloc[start:self.current_step + 1]

        features = [
            'open', 'high', 'low', 'close', 'volume', 'value', 'trades', 'change', 'marketCap',
            'RSI', 'bb_upper', 'bb_middle', 'bb_lower',
            'macd', 'macd_signal', 'macd_hist', 'zigzag',
            'Hammer', 'BullishEngulfing', 'MorningStar',
            'Doji', 'PiercingLine', 'ThreeWhiteSoldiers',
            'confidence', 'ai_score', 'duration_days'
        ]

        obs_df = obs_df.copy()
        for col in features:
            if col not in obs_df.columns:
                obs_df[col] = 0

        pattern_flags = ['Hammer', 'BullishEngulfing', 'MorningStar', 'Doji', 'PiercingLine', 'ThreeWhiteSoldiers']
        for col in pattern_flags:
            obs_df[col] = obs_df[col].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0}).fillna(0)

        obs_df['confidence'] = obs_df['confidence'].fillna(0) / 100
        obs_df['ai_score'] = obs_df['ai_score'].fillna(0) / 100
        obs_df['duration_days'] = obs_df['duration_days'].fillna(0) / 10

        if len(obs_df) < window:
            pad = pd.DataFrame(np.zeros((window - len(obs_df), len(features))), columns=features)
            obs_df = pd.concat([pad, obs_df], ignore_index=True)

        obs_array = obs_df[features].values.flatten().astype(np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=0.0, neginf=0.0)
        return obs_array

    def step(self, action):
        row = self.maindf.iloc[self.current_step]
        price = row.get('close', 0)
        symbol = row.get('symbol', '')
        confidence = row.get('confidence', 0)
        reward = 0.0

        if action == 1:
            self.stock += 1
            self.cash -= price
            self.last_price = price
            reward = -0.1 * (1 - confidence)
        elif action == 2 and self.stock > 0:
            self.stock -= 1
            self.cash += price
            profit = price - self.last_price if self.last_price else 0
            reward = profit * (1 + confidence * 1.5) if profit > 0 else profit
        elif action == 0:
            reward = -0.01

        try:
            if action == 1 and any(str(row.get(pat)).upper() == 'TRUE' for pat in ['BullishEngulfing', 'MorningStar', 'ThreeWhiteSoldiers']):
                reward += 1.0
            if action == 1 and row.get('RSI', 0) < 30:
                reward += 0.5
            if action == 2 and row.get('RSI', 100) > 70:
                reward += 0.5
            if action == 1 and row.get('macd', 0) > row.get('macd_signal', 0):
                reward += 0.25
            if action == 2 and row.get('macd', 0) < row.get('macd_signal', 0):
                reward += 0.25
            if action == 1 and row.get('zigzag', 0) > 0:
                reward += 0.25
            if action == 2 and row.get('zigzag', 0) < 0:
                reward += 0.25
            if action == 1 and row.get('volume', 0) > self.maindf['volume'].mean() * 1.5:
                reward += 0.5
            if action == 1 and row.get('trades', 0) > self.maindf['trades'].mean():
                reward += 0.25
            if action == 1 and row.get('marketCap', 0) > 1e9:
                reward += 0.25
            if action == 1 and row.get('change', 0) > 0:
                reward += 0.25
            if action == 2 and row.get('change', 0) < 0:
                reward += 0.25
        except:
            pass

        volatility = row.get('high', 0) - row.get('low', 0)
        if volatility > row.get('close', 1) * 0.05:
            reward *= 1.2

        if symbol in self.gape_df.get('symbol', []):
            reward += 0.3
        if symbol in self.gapebuy_df.get('symbol', []):
            reward += 0.5
        if symbol in self.shortbuy_df.get('symbol', []):
            reward += 0.3
        if symbol in self.rsi_diver_df.get('symbol', []):
            reward += 0.4
        if symbol in self.rsi_diver_retest_df.get('symbol', []):
            reward += 0.2

        duration_days = row.get('duration_days', None)
        outcome = row.get('outcome', None)

        if outcome == 'TP' and duration_days is not None:
            reward += max(1.0 - (duration_days * 0.2), 0.2)
        elif outcome == 'SL':
            reward -= 1.0
        elif outcome == 'HOLD':
            reward -= 0.1

        self.current_step += 1
        done = self.current_step >= self.total_steps
        obs = self.get_obs() if not done else self.last_obs
        self.last_obs = obs
        reward = np.clip(reward, -10, 10)
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        return obs, reward, done, False, {}

    def render(self):
        r = self.maindf.iloc[self.current_step]
        price = float(r.get('close', 0))
        print(f"Step {self.current_step}: Cash {self.cash:.2f}, Stock {self.stock}, Value {self.cash + self.stock * price:.2f}")
