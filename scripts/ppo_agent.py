# src/ppo_agent_with_signal.py
import os
import sys
import random
import logging
import math
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# --------------- path resolver ---------------
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_MARKET = BASE_DIR / "csv" / "mongodb.csv"
CSV_SIGNAL = BASE_DIR / "csv" / "trade_stock.csv"
MODEL_DIR = BASE_DIR / "csv" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "ppo_signal_model.pt"

sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# 1. Load trade signals (new format)
# ------------------------------------------------------------------------------
def load_signals(csv_path: Path) -> dict:
    """
    Returns: {(symbol, date): {"buy":float, "SL":float, "TP":float,
                               "position_size":int, "exposure_bdt":float,
                               "actual_risk_bdt":float, "diff":float, "RRR":float}}
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    required = ["No", "date", "symbol", "buy", "SL", "tp",
                "position_size", "exposure_bdt", "actual_risk_bdt", "diff", "RRR"]
    if not all(c in df.columns for c in required):
        raise KeyError(f"Missing cols in trade_stock.csv: {required}")
    signals = {}
    for _, row in df.iterrows():
        key = (row["symbol"], row["date"])
        signals[key] = {
            "buy": float(row["buy"]),
            "SL": float(row["SL"]),
            "TP": float(row["tp"]),
            "position_size": int(row["position_size"]),
            "exposure_bdt": float(row["exposure_bdt"]),
            "actual_risk_bdt": float(row["actual_risk_bdt"]),
            "diff": float(row["diff"]),
            "RRR": float(row["RRR"]),
        }
    return signals

# ------------------------------------------------------------------------------
# 2. Build observation: 110 price + 4 signal = 114
# ------------------------------------------------------------------------------
def build_observation(df_market, idx, window, state_dim, signals_dict):
    # 110 price features
    cols = ["open", "close", "high", "low", "volume", "rsi", "macd", "macd_signal",
            "bb_upper", "bb_middle", "bb_lower"]
    pad_needed = max(0, window - (idx + 1))
    start = max(0, idx - window + 1)
    seg = df_market.iloc[start: idx + 1][cols].values
    seg = np.pad(seg, ((pad_needed, 0), (0, 0)), "edge").flatten()
    price_vec = seg[:110]

    # 4 signal features
    row = df_market.iloc[idx]
    sym, date = row["symbol"], row["date"]
    sig = signals_dict.get((sym, date))
    if sig:
        close = row["close"]
        buy_p = sig["buy"]
        sl_p = sig["SL"]
        tp_p = sig["TP"]
        rrr = sig["RRR"]
        feat = np.array([
            close / (buy_p or 1e-8),  # normalized price
            (buy_p - sl_p) / (buy_p or 1e-8),  # SL %
            (tp_p - buy_p) / (buy_p or 1e-8),  # TP %
            rrr,
        ])
    else:
        feat = np.zeros(4)

    obs = np.concatenate([price_vec, feat])
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    return obs

# ------------------------------------------------------------------------------
# 3. PPO Agent
# ------------------------------------------------------------------------------
class PPOTradingAgent:
    def __init__(self, state_dim=114, action_dim=3, lr=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.eps_clip = 0.2
        self.k_epochs = 10
        self.batch_size = 64

        self.policy = self._make_net(action_dim, softmax=True).to(device)
        self.value = self._make_net(1, softmax=False).to(device)
        self.optimizer = optim.Adam(list(self.policy.parameters()) +
                                    list(self.value.parameters()), lr=lr)
        self.memory = []
        self.trade_log = []

    def _make_net(self, out_size, softmax):
        layers = [nn.Linear(self.state_dim, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, out_size)]
        if softmax:
            layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def select_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            return random.randint(0, self.action_dim - 1), 0.0, 0.0
        probs = self.policy(obs)
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones(self.action_dim, device=device) / self.action_dim
        dist = Categorical(probs)
        action = dist.sample()
        value = self.value(obs)
        return action.item(), dist.log_prob(action).item(), value.item()

    # ---------- train ----------
    def train_episode(self, df_market, signals_dict, initial_balance=100000):
        balance, position, entry_price = initial_balance, 0.0, 0.0
        ep_reward = 0.0

        for idx in range(len(df_market)):
            obs = build_observation(df_market, idx, 10, self.state_dim, signals_dict)
            action, log_prob, value = self.select_action(obs)
            price = df_market.iloc[idx]["close"]
            sym = df_market.iloc[idx]["symbol"]
            date = df_market.iloc[idx]["date"]
            reward = 0.0

            # BUY (use CSV size)
            row_sig = signals_dict.get((sym, date))
            if row_sig and action == 1 and position == 0 and price <= row_sig["buy"]:
                shares = row_sig["position_size"]
                if shares > 0:
                    entry_price = price
                    position = shares
                    balance -= shares * entry_price
                    # লগিং
                    self.trade_log.append({
                        "No": len(self.trade_log) + 1,
                        "date": date,
                        "symbol": sym,
                        "buy": entry_price,
                        "SL": row_sig["SL"],
                        "tp": row_sig["TP"],
                        "position_size": shares,
                        "exposure_bdt": shares * entry_price,
                        "actual_risk_bdt": row_sig["actual_risk_bdt"],
                        "diff": 0.0,
                        "RRR": row_sig["RRR"],
                    })

            # SELL (TP hit)
            if row_sig and action == 2 and position > 0 and price >= row_sig["TP"]:
                balance = position * price
                profit = balance - (position * entry_price)
                if self.trade_log:
                    self.trade_log[-1]["diff"] = profit
                position = 0.0
                reward = profit / (entry_price or 1e-8) * 0.01  # scale

            next_obs = build_observation(df_market, idx + 1, 10, self.state_dim, signals_dict)
            done = 1.0 if (idx == len(df_market) - 1) else 0.0
            self.memory.append((obs, action, log_prob, reward, value, done))
            ep_reward += reward

            if len(self.memory) >= self.batch_size:
                self._update()

        final_balance = balance + (position * df_market.iloc[-1]["close"] if position > 0 else 0)
        final_balance = max(final_balance, 0.0)
        return final_balance, ep_reward

    # ---------- PPO update ----------
    def _update(self):
        if not self.memory:
            return
        states, actions, old_logp, rewards, values, dones = zip(*self.memory)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        old_logp = torch.tensor(old_logp, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        values = torch.tensor(values, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        deltas = rewards + self.gamma * values.roll(-1) * (1 - dones) - values
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        idx_all = torch.randperm(len(states))
        for _ in range(self.k_epochs):
            for start in range(0, len(states), self.batch_size):
                sl = idx_all[start: start + self.batch_size]
                b_s, b_a, b_old_logp, b_ret, b_adv = (
                    states[sl], actions[sl], old_logp[sl], returns[sl], advantages[sl]
                )

                probs = self.policy(b_s)
                dist = Categorical(probs)
                new_logp = dist.log_prob(b_a)
                ratio = torch.exp(new_logp - b_old_logp)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = self.value(b_s).squeeze()
                value_loss = nn.MSELoss()(value_pred, b_ret)

                loss = policy_loss + 0.5 * value_loss - 0.01 * dist.entropy().mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()), 0.5
                )
                self.optimizer.step())

        self.memory.clear()

    # ---------- save ----------
    def save_model(self):
        torch.save({"policy": self.policy.state_dict(),
                    "value": self.value.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, MODEL_PATH)
        log.info("Model saved → %s", MODEL_PATH)


# ==============================================================================
# Train + Console Table
# ==============================================================================
if __name__ == "__main__":
    if not CSV_MARKET.exists():
        raise FileNotFoundError(CSV_MARKET)
    if not CSV_SIGNAL.exists():
        raise FileNotFoundError(CSV_SIGNAL)

    # --- OPTIONAL: last 90 days filter ---
    df_market = pd.read_csv(CSV_MARKET, parse_dates=["date"])
    # cutoff = df_market["date"].max() - pd.Timedelta(days=90)
    # df_market = df_market[df_market["date"] >= cutoff]
    df_market["date"] = df_market["date"].dt.strftime("%Y-%m-%d")

    signals = load_signals(CSV_SIGNAL)

    agent = PPOTradingAgent(state_dim=114, action_dim=3)
    EPISODES = 50
    for ep in range(1, EPISODES + 1):
        final_balance, ep_reward = agent.train_episode(df_market, signals)
        # >>> Console Table Print <<<
        if agent.trade_log:
            print("\n>>> Episode Trade Summary <<<")
            df_print = pd.DataFrame(agent.trade_log)
            print(df_print.to_string(index=False))
            agent.trade_log.clear()
        log.info("Episode %d | Final Balance %.2f | Reward %.4f", ep, final_balance, ep_reward)
        if ep % 10 == 0:
            agent.save_model()

    agent.save_model()
