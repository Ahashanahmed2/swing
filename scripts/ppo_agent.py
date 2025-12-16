# src/ppo_agent.py  (Updated Version)
import os, sys, logging, math, random
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pandas as pd
import numpy as np

from config import MONGODB_PATH, PPO_MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: %s", device)


# ============================================================================
# Helper: Feature Builder
# ============================================================================
def build_state_vector(df: pd.DataFrame, idx: int, window: int, state_dim: int) -> np.ndarray:
    """Build fixed-length state vector from rolling window."""
    if idx < 0 or idx >= len(df):
        return None
    cols = ["open", "close", "high", "low", "volume", "rsi", "macd", "macd_signal",
            "bb_upper", "bb_middle", "bb_lower"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in market data: {missing}")

    pad_needed = max(0, window - (idx + 1))
    start = max(0, idx - window + 1)
    segment = df.iloc[start: idx + 1][cols].values  # (t, 11)
    state = segment[-window:]                       # last `window` rows
    state = np.pad(state, ((pad_needed, 0), (0, 0)), "edge").flatten()  # (110,)
    return state[:state_dim]


# ============================================================================
# PPO Agent
# ============================================================================
class PPOTradingAgent:
    def __init__(self,
                 state_dim: int = 110,
                 action_dim: int = 3,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 eps_clip: float = 0.2,
                 k_epochs: int = 10,
                 batch_size: int = 64):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.window = 10

        # Networks
        self.policy = self._make_net(action_dim, softmax=True).to(device)
        self.value = self._make_net(1, softmax=False).to(device)
        self.optimizer = optim.Adam(list(self.policy.parameters()) +
                                    list(self.value.parameters()), lr=lr)

        self.memory = []        # (s, a, logp, r, v, done)

    # -------------------- Architecture --------------------
    def _make_net(self, out_size: int, softmax: bool):
        layers = [nn.Linear(self.state_dim, 128),
                  nn.ReLU(),
                  nn.Linear(128, 128),
                  nn.ReLU(),
                  nn.Linear(128, out_size)]
        if softmax:
            layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers)

    # -------------------- Interaction --------------------
    def get_state(self, df, idx):
        return build_state_vector(df, idx, self.window, self.state_dim)

    @torch.no_grad()
    def select_action(self, state: np.ndarray):
        s = torch.tensor(state, dtype=torch.float32, device=device)
        probs = self.policy(s)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value(s)
        return action.item(), log_prob.item(), value.item()

    # -------------------- Training episode --------------------
    def train_episode(self, df, initial_balance: float = 100_000):
        balance, position, entry_price = initial_balance, 0.0, 0.0
        ep_rewards, ep_vals = [], []

        for idx in range(len(df)):
            state = self.get_state(df, idx)
            if state is None:
                continue
            action, log_prob, value = self.select_action(state)
            price = df.iloc[idx]["close"]

            reward = 0.0
            if action == 1 and position == 0:          # buy
                position = balance / price
                entry_price = price
                balance = 0.0
            elif action == 2 and position > 0:         # sell
                balance = position * price
                profit = balance - (position * entry_price)
                reward = profit / entry_price
                position = 0.0

            next_state = self.get_state(df, idx + 1)
            done = 1.0 if (idx == len(df) - 1) else 0.0

            self.memory.append(
                (state, action, log_prob, reward, value, done)
            )
            ep_rewards.append(reward)
            ep_vals.append(value)

            # Update on batch
            if len(self.memory) >= self.batch_size:
                self._update()

        # End-of-episode final reward
        final_balance = balance + (position * df.iloc[-1]["close"] if position > 0 else 0)
        return final_balance

    # -------------------- PPO Update --------------------
    def _update(self):
        if len(self.memory) == 0:
            return
        states, actions, old_logp, rewards, values, dones = zip(*self.memory)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        old_logp = torch.tensor(old_logp, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        values = torch.tensor(values, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        # ----- GAE -----
        deltas = rewards + self.gamma * values.roll(-1) * (1 - dones) - values
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ----- PPO epochs -----
        for _ in range(self.k_epochs):
            idx = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                sl = idx[start: start + self.batch_size]
                b_state, b_action, b_old_logp, b_ret, b_adv = \
                    states[sl], actions[sl], old_logp[sl], returns[sl], advantages[sl]

                probs = self.policy(b_state)
                dist = Categorical(probs)
                new_logp = dist.log_prob(b_action)
                ratio = torch.exp(new_logp - b_old_logp)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = self.value(b_state).squeeze()
                value_loss = nn.MSELoss()(value_pred, b_ret)

                loss = policy_loss + 0.5 * value_loss - 0.01 * dist.entropy().mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), 0.5)
                self.optimizer.step()

        self.memory.clear()

    # -------------------- Save / Load --------------------
    def save_model(self):
        Path(PPO_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, PPO_MODEL_PATH)
        log.info("Model saved → %s", PPO_MODEL_PATH)

    def load_model(self):
        ckpt = torch.load(PPO_MODEL_PATH, map_location=device)
        self.policy.load_state_dict(ckpt["policy"])
        self.value.load_state_dict(ckpt["value"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        log.info("Model loaded ← %s", PPO_MODEL_PATH)


# ============================================================================
# Stand-alone train
# ============================================================================
if __name__ == "__main__":
    df = pd.read_csv(MONGODB_PATH)
    agent = PPOTradingAgent(state_dim=110, action_dim=3)

    EPISODES = 50
    for ep in range(1, EPISODES + 1):
        final = agent.train_episode(df)
        log.info("Episode %d | Final Balance %.2f", ep, final)
        if ep % 10 == 0:
            agent.save_model()

    agent.save_model()
