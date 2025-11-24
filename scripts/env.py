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

        self.maindf = maindf.copy()
        self.gape_df = pd.read_csv(gape_path)
        self.gapebuy_df = pd.read_csv(gapebuy_path)
        self.shortbuy_df = pd.read_csv(shortbuy_path)
        self.rsi_diver_df = pd.read_csv(rsi_diver_path)
        self.rsi_diver_retest_df = pd.read_csv(rsi_diver_retest_path)

        self.cash = 10000
        self.stock = 0
        self.last_price = None
        self.current_step = 0
        self.total_steps = len(maindf) - 1
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

        obs_df = obs_df[features].copy()

        pattern_flags = ['Hammer', 'BullishEngulfing', 'MorningStar', 'Doji', 'PiercingLine', 'ThreeWhiteSoldiers']
        for col in pattern_flags:
            obs_df[col] = obs_df[col].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0}).fillna(0)

        obs_df['confidence'] = obs_df['confidence'] / 100
        obs_df['ai_score'] = obs_df['ai_score'] / 100
        obs_df['duration_days'] = obs_df['duration_days'].fillna(0) / 10

        if len(obs_df) < window:
            pad = pd.DataFrame(np.zeros((window - len(obs_df), len(features))), columns=features)
            obs_df = pd.concat([pad, obs_df], ignore_index=True)

        return obs_df.values.flatten()

    def step(self, action):
        row = self.maindf.iloc[self.current_step]
        price = row['close']
        symbol = row['symbol']
        date = row['date']
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
            profit = price - self.last_price
            reward = profit * (1 + confidence * 1.5) if profit > 0 else profit
        elif action == 0:
            reward = -0.01

        if action == 1 and any(row.get(pat) == 'TRUE' for pat in ['BullishEngulfing', 'MorningStar', 'ThreeWhiteSoldiers']):
            reward += 1.0
        if action == 1 and row['RSI'] < 30:
            reward += 0.5
        if action == 2 and row['RSI'] > 70:
            reward += 0.5
        if action == 1 and row['macd'] > row['macd_signal']:
            reward += 0.25
        if action == 2 and row['macd'] < row['macd_signal']:
            reward += 0.25
        if action == 1 and row['zigzag'] > 0:
            reward += 0.25
        if action == 2 and row['zigzag'] < 0:
            reward += 0.25
        if action == 1 and row['volume'] > self.maindf['volume'].mean() * 1.5:
            reward += 0.5
        if action == 1 and row['trades'] > self.maindf['trades'].mean():
            reward += 0.25
        if action == 1 and row['marketCap'] > 1e9:
            reward += 0.25
        if action == 1 and row['change'] > 0:
            reward += 0.25
        if action == 2 and row['change'] < 0:
            reward += 0.25

        volatility = row['high'] - row['low']
        if volatility > row['close'] * 0.05:
            reward *= 1.2

        if symbol in self.gape_df['symbol'].values:
            reward += 0.3
        if symbol in self.gapebuy_df['symbol'].values:
            reward += 0.5
        if symbol in self.shortbuy_df['symbol'].values:
            reward += 0.3
        if symbol in self.rsi_diver_df['symbol'].values:
            reward += 0.4
        if symbol in self.rsi_diver_retest_df['symbol'].values:
            reward += 0.2

        duration_days = row.get('duration_days', None)
        outcome = row.get('outcome', None)

        if outcome == 'TP':
            if duration_days is not None:
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

        return obs, reward, done, False, {}

    def render(self):
        r = self.maindf.iloc[self.current_step]
        price = float(r.get('close'))
        print(f"Step {self.current_step}: Cash {self.cash:.2f}, Stock {self.stock}, Value {self.cash + self.stock * price:.2f}")