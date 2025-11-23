import gym
import numpy as np
import pandas as pd
import math

class TradeEnv(gym.Env):
    def __init__(self,
                 maindf,
                 filtered_output,
                 gape_path="./csv/gape.csv",
                 gapebuy_path="./csv/gape_buy.csv",
                 shortbuy_path="./csv/short_buy.csv",
                 rsi_diver_path="./csv/rsi_diver.csv",
                 rsi_diver_retest_path="./csv/rsi_diver_retest.csv"):
        
        super(TradeEnv, self).__init__()
        
        # ✅ মূল ডেটা
        self.maindf = maindf
        self.filtered_output = filtered_output

        # ✅ নতুন CSV ফাইলগুলো লোড করুন
        self.gape_df = pd.read_csv(gape_path)
        self.gapebuy_df = pd.read_csv(gapebuy_path)
        self.shortbuy_df = pd.read_csv(shortbuy_path)
        self.rsi_diver_df = pd.read_csv(rsi_diver_path)
        self.rsi_diver_retest_df = pd.read_csv(rsi_diver_retest_path)

        # ✅ প্রাথমিক অবস্থা
        self.cash = 10000
        self.stock = 0
        self.last_price = None
        self.current_step = 0
        self.total_steps = len(maindf) - 1
        self.last_obs = None

    def step(self, action, price, confidence, symbol, date):
        # ✅ Action অনুযায়ী reward
        if action == 1:  # Buy
            self.stock += 1
            self.cash -= price
            self.last_price = price
            reward = -0.1 * (1 - confidence)

        elif action == 2 and self.stock > 0:  # Sell
            self.stock -= 1
            self.cash += price
            profit = price - self.last_price
            reward = profit * (1 + confidence * 1.5) if profit > 0 else profit

        elif action == 0:  # Hold
            reward = -0.01

        self.current_step += 1
        done = self.current_step >= self.total_steps

        if not done:
            obs = self.get_obs()
            self.last_obs = obs
        else:
            obs = self.last_obs

        if math.isnan(reward):
            print(f"🚨 NaN in reward at step {self.current_step}")
            print(f"Price: {price}, Last: {self.last_price}, Confidence: {confidence}")
            raise ValueError("NaN found in reward")

        # ✅ filtered_output match হলে বোনাস রিওয়ার্ড
        if not self.filtered_output.empty:
            is_matched = (
                (self.filtered_output['symbol'] == symbol) &
                (self.filtered_output['date'] == date)
            ).any()
            if is_matched and action == 1:
                reward += 1.0

        # ✅ CSV ভিত্তিক রিওয়ার্ড লজিক
        if symbol in self.gapebuy_df['symbol'].values:
            reward += 1.0

        if symbol in self.shortbuy_df['SYMBOL'].values:
            reward += 1.0

        if symbol in self.gape_df['symbol'].values:
            reward += 0.5

        if symbol in self.rsi_diver_df['MBOL'].values:  # rsi_diver.csv
            reward += 0.25

        if symbol in self.rsi_diver_retest_df['MBOL'].values:  # rsi_diver_retest.csv
            reward += 0.5

        reward = np.clip(reward, -10, 10)

        info = {
            'cash': self.cash,
            'stock': self.stock,
            'portfolio_value': self.cash + self.stock * price,
            'price': price,
            'action': action,
            'symbol': self.maindf.iloc[min(self.current_step, self.total_steps)]['symbol'],
            'reward': reward,
            'confidence': confidence,
            'lastprice': self.last_price
        }

        return obs, reward, done, False, info

    def get_obs(self):
        # ✅ observation logic
        return self.maindf.iloc[self.current_step].to_dict()

    def render(self):
        r = self.maindf.iloc[self.current_step]
        price = float(r.get('close'))
        print(f"Step {self.current_step}: Cash {self.cash:.2f}, Stock {self.stock}, Value {self.cash + self.stock * price:.2f}")