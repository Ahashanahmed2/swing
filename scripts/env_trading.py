# env_trading.py ==================

import gym from gym import spaces import numpy as np

class MultiSymbolTradingEnv(gym.Env): metadata = {"render.modes": ["human"]}

def __init__(self, symbol_dfs, signals, build_observation,
             window, state_dim, total_capital=500000, risk_percent=0.01):
    super().__init__()

    self.symbols = list(symbol_dfs.keys())
    self.dfs = symbol_dfs
    self.signals = signals
    self.build_observation = build_observation
    self.window = window
    self.state_dim = state_dim

    self.total_capital = total_capital
    self.risk_percent = risk_percent

    self.n_symbols = len(self.symbols)
    self.max_steps = max(len(df) for df in self.dfs.values())

    self.action_space = spaces.MultiDiscrete([3] * self.n_symbols)
    self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(self.n_symbols, self.state_dim),
        dtype=np.float32
    )

    self.reset()

def reset(self):
    self.t = 0
    self.balance = {s: self.total_capital for s in self.symbols}
    self.position = {s: 0 for s in self.symbols}
    self.entry_price = {s: 0.0 for s in self.symbols}
    return self._get_obs()

def _get_obs(self):
    obs = []
    for s in self.symbols:
        df = self.dfs[s]
        if self.t < len(df):
            o = self.build_observation(df, self.t, self.signals)
        else:
            o = np.zeros(self.state_dim)
        obs.append(o)
    return np.array(obs, dtype=np.float32)

def step(self, actions):
    rewards = []
    done_flags = []

    for i, s in enumerate(self.symbols):
        action = actions[i]
        df = self.dfs[s]

        if self.t >= len(df):
            rewards.append(0.0)
            done_flags.append(True)
            continue

        row = df.iloc[self.t]
        price = row["close"]
        sig = self.signals.get((s, row["date"]))
        reward = 0.0

        # -------- BUY --------
        if action == 1 and self.position[s] == 0 and sig:
            buy = sig["buy"]
            sl = sig["SL"]
            risk_amount = self.total_capital * self.risk_percent
            risk_per_share = max(buy - sl, 1e-8)

            shares = int(risk_amount / risk_per_share)

            if shares > 0 and price <= buy:
                self.position[s] = shares
                self.entry_price[s] = price
                self.balance[s] -= shares * price

        # -------- SELL (TP / SL) --------
        if self.position[s] > 0 and sig:
            if price >= sig["TP"] or price <= sig["SL"]:
                self.balance[s] += self.position[s] * price
                pnl = (price - self.entry_price[s]) * self.position[s]
                reward = pnl / (abs(self.entry_price[s] * self.position[s]) + 1e-8)
                self.position[s] = 0

        rewards.append(reward)
        done_flags.append(self.t == len(df) - 1)

    self.t += 1
    done = all(done_flags)
    return self._get_obs(), float(np.sum(rewards)), done, {}

def render(self, mode="human"):
    print(f"Step {self.t}")
    for s in self.symbols:
        print(f"{s} | Balance {self.balance[s]:.2f} | Pos {self.position[s]}")

================== PPO_trading.py ==================

import pandas as pd from pathlib import Path from stable_baselines3 import PPO from stable_baselines3.common.vec_env import DummyVecEnv

from env_trading import MultiSymbolTradingEnv

---------------- Paths ----------------

BASE_DIR = Path(file).resolve().parent.parent CSV_MARKET = BASE_DIR / "csv" / "mongodb.csv" CSV_SIGNAL = BASE_DIR / "csv" / "trade_stock.csv" MODEL_PATH = BASE_DIR / "csv" / "model" / "sb3_ppo_trading"

WINDOW = 10 MARKET_COLS = [ "open","high","low","close", "volume","value","trades","change","marketCap", "bb_upper","bb_middle","bb_lower", "macd","macd_signal","macd_hist", "rsi","atr", "Hammer","BullishEngulfing","MorningStar", "Doji","PiercingLine","ThreeWhiteSoldiers", ] STATE_DIM = len(MARKET_COLS) * WINDOW + 4

---------------- Load signals ----------------

def load_signals(path): df = pd.read_csv(path, parse_dates=["date"]) df["date"] = df["date"].dt.strftime("%Y-%m-%d") sig = {} for _, r in df.iterrows(): sig[(r["symbol"], r["date"]) ] = { "buy": float(r["buy"]), "SL": float(r["SL"]), "TP": float(r["tp"]), "RRR": float(r["RRR"]), } return sig

---------------- Observation builder ----------------

def build_observation(df, idx, signals): pad = max(0, WINDOW - (idx + 1)) start = max(0, idx - WINDOW + 1) seg = df.iloc[start:idx+1][MARKET_COLS].values seg = np.pad(seg, ((pad,0),(0,0)), mode="edge") market_vec = seg.flatten()

row = df.iloc[idx]
sig = signals.get((row["symbol"], row["date"]))
if sig:
    buy = sig["buy"]
    signal_vec = [
        row["close"]/(buy+1e-8),
        (buy-sig["SL"])/(buy+1e-8),
        (sig["TP"]-buy)/(buy+1e-8),
        sig["RRR"],
    ]
else:
    signal_vec = [0.0]*4

obs = list(market_vec) + signal_vec
return np.nan_to_num(obs)

---------------- Main ----------------

if name == "main": df = pd.read_csv(CSV_MARKET, parse_dates=["date"]) df["date"] = df["date"].dt.strftime("%Y-%m-%d")

signals = load_signals(CSV_SIGNAL)

symbol_dfs = { s: sdf.reset_index(drop=True) for s, sdf in df.groupby("symbol") if len(sdf)>=WINDOW }

env = DummyVecEnv([
    lambda: MultiSymbolTradingEnv(symbol_dfs, signals, build_observation, WINDOW, STATE_DIM)
])

model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=256,
    gamma=0.99,
    learning_rate=3e-4,
    verbose=1
)

model.learn(total_timesteps=300_000)
model.save(MODEL_PATH)