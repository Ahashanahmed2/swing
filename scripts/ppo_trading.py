# ================== PPO_trading.py ==================

import pandas as pd
import numpy as np

from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env_trading import MultiSymbolTradingEnv

# ---------------- Paths ----------------

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_MARKET = BASE_DIR / "csv" / "mongodb.csv"
CSV_SIGNAL = BASE_DIR / "csv" / "trade_stock.csv"
MODEL_PATH = BASE_DIR / "csv" / "model" / "sb3_ppo_trading"

# ---------------- Config ----------------

WINDOW = 10

MARKET_COLS = [
    "open","high","low","close",
    "volume","value","trades","change","marketCap",
    "bb_upper","bb_middle","bb_lower",
    "macd","macd_signal","macd_hist",
    "rsi","atr",
    "Hammer","BullishEngulfing","MorningStar",
    "Doji","PiercingLine","ThreeWhiteSoldiers",
]

STATE_DIM = len(MARKET_COLS) * WINDOW + 4

# ---------------- Load signals ----------------

def load_signals(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    sig = {}
    for _, r in df.iterrows():
        sig[(r["symbol"], r["date"])] = {
            "buy": float(r["buy"]),
            "SL": float(r["SL"]),
            "TP": float(r["tp"]),
            "RRR": float(r["RRR"]),
        }
    return sig

# ---------------- Observation builder ----------------

def build_observation(df, idx, signals):
    pad = max(0, WINDOW - (idx + 1))
    start = max(0, idx - WINDOW + 1)

    seg = df.iloc[start:idx+1][MARKET_COLS].values
    seg = np.pad(seg, ((pad,0),(0,0)), mode="edge")
    market_vec = seg.flatten()

    row = df.iloc[idx]
    sig = signals.get((row["symbol"], row["date"]))

    if sig:
        buy = sig["buy"]
        signal_vec = [
            row["close"] / (buy + 1e-8),
            (buy - sig["SL"]) / (buy + 1e-8),
            (sig["TP"] - buy) / (buy + 1e-8),
            sig["RRR"],
        ]
    else:
        signal_vec = [0.0] * 4

    obs = list(market_vec) + signal_vec
    return np.nan_to_num(obs)

# ---------------- Main ----------------

if __name__ == "__main__":

    df = pd.read_csv(CSV_MARKET, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    signals = load_signals(CSV_SIGNAL)

    symbol_dfs = {
        s: sdf.reset_index(drop=True)
        for s, sdf in df.groupby("symbol")
        if len(sdf) >= WINDOW
    }

    env = DummyVecEnv([
        lambda: MultiSymbolTradingEnv(
            symbol_dfs,
            signals,
            build_observation,
            WINDOW,
            STATE_DIM
        )
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

    model.learn(total_timesteps=50_000)
    model.save(MODEL_PATH)
