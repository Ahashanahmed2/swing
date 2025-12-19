# ================== env_trading.py ==================


import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MultiSymbolTradingEnv(gym.Env):
    """
    Multi-symbol trading environment for PPO
    Action per symbol:
        0 = HOLD
        1 = BUY
        2 = SELL (manual exit, TP/SL auto handled)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        symbol_dfs,
        signals,
        build_observation,
        window,
        state_dim,
        total_capital=500_000,
        risk_percent=0.01,
    ):
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

        # -------- Spaces --------
        self.action_space = spaces.MultiDiscrete([3] * self.n_symbols)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_symbols, self.state_dim),
            dtype=np.float32,
        )

        self.reset()

    # -------------------------------------------------
    # RESET
    # -------------------------------------------------
    def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    self.t = 0
    self.balance = {s: self.total_capital for s in self.symbols}
    self.position = {s: 0 for s in self.symbols}
    self.entry_price = {s: 0.0 for s in self.symbols}

    return self._get_obs(), {}

    # -------------------------------------------------
    # OBSERVATION
    # -------------------------------------------------
    def _get_obs(self):
        obs = []

        for s in self.symbols:
            df = self.dfs[s]
            if self.t < len(df):
                o = self.build_observation(df, self.t, self.signals)
            else:
                o = np.zeros(self.state_dim)

            obs.append(o)

        return np.asarray(obs, dtype=np.float32)

    # -------------------------------------------------
    # STEP
    # -------------------------------------------------
    def step(self, actions):
        rewards = []
        done_flags = []

        for i, s in enumerate(self.symbols):
            action = int(actions[i])
            df = self.dfs[s]

            if self.t >= len(df):
                rewards.append(0.0)
                done_flags.append(True)
                continue

            row = df.iloc[self.t]
            price = float(row["close"])
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

            # -------- SELL / EXIT --------
            if self.position[s] > 0 and sig:
                if price >= sig["TP"] or price <= sig["SL"] or action == 2:
                    pnl = (price - self.entry_price[s]) * self.position[s]
                    self.balance[s] += self.position[s] * price

                    reward = pnl / (
                        abs(self.entry_price[s] * self.position[s]) + 1e-8
                    )

                    self.position[s] = 0
                    self.entry_price[s] = 0.0

            rewards.append(reward)
            done_flags.append(self.t >= len(df) - 1)

        self.t += 1
        done = all(done_flags)

        return self._get_obs(), float(np.sum(rewards)), done, {}

    # -------------------------------------------------
    # RENDER
    # -------------------------------------------------
    def render(self, mode="human"):
        print(f"\nStep {self.t}")
        for s in self.symbols:
            print(
                f"{s} | Balance: {self.balance[s]:.2f} | Position: {self.position[s]}"
            )