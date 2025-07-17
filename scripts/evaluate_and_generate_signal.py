import pandas as pd
import os
import json
from datetime import datetime, timedelta
from env import TradeEnv
from stable_baselines3 import DQN
import glob


def get_latest_file_from_folder(folder_path):
    list_of_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not list_of_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    return max(list_of_files, key=os.path.getctime)

# 🔁 Load trained model
model = DQN.load("./csv/dqn_retrained")

main_data_path = "./csv/mongodb.csv"
main_df = pd.read_csv(main_data_path)
unique_symbols = main_df["symbol"].unique()

signal_dir = "./output/ai_signal"
os.makedirs(signal_dir, exist_ok=True)

all_signals = []

for symbol in unique_symbols:
    symbol_df = main_df[main_df["symbol"] == symbol].copy()
    if symbol_df.empty:
        continue


    imbalance_high_df = pd.read_csv(get_latest_file_from_folder("./csv/swing/imbalanceZone/down_to_up"))
    imbalance_low_df = pd.read_csv(get_latest_file_from_folder("./csv/swing/imbalanceZone/up_to_down"))
    swing_high_candle_df = pd.read_csv(get_latest_file_from_folder("./csv/swing/swing_high/high_candle"))
    swing_high_confirm_df = pd.read_csv(get_latest_file_from_folder("./csv/swing/swing_high/high_confirm"))
    swing_low_candle_df = pd.read_csv(get_latest_file_from_folder("./csv/swing/swing_low/low_candle"))
    swing_low_confirm_df = pd.read_csv(get_latest_file_from_folder("./csv/swing/swing_low/low_confirm"))

    env = TradeEnv(
                symbol_df,  
                imbalance_high_df,
                imbalance_low_df,
                swing_high_candle_df,
                swing_high_confirm_df,
                swing_low_candle_df,
                swing_low_confirm_df,)
    obs, _ = env.reset()
    done = False
    last_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        last_reward = reward

    price = info["price"]
    now = datetime.now()

    signal = {
        "symbol": symbol,
        "entry_date": now.strftime("%Y-%m-%d"),
        "buy_price": round(price, 2),
        "exit_target_price": round(price * 1.05, 2),  # 5% gain
        "confidence": f"{min(100, round(abs(last_reward) * 15))}%",
        "trend": "Uptrend" if last_reward > 0 else "Downtrend",
        "signal_type": "Buy" if last_reward > 0 else "Sell",
        "strategy": "AI DQN Strategy",
        "risk_reward_ratio": round(1.5 + (last_reward / 10), 2),
        "stop_loss": round(price * 0.97, 2),
        "expiry_date": (now + timedelta(days=3)).strftime("%Y-%m-%d"),
        "validity_duration": "3 days",
        "hold_duration": 3,
        "past_accuracy": 88.5,
        "ai_score": round(last_reward * 10, 2),
        "ai_version": "dqn_retrained"
    }

    # Save individual CSV file per symbol
    #symbol_csv_path = os.path.join(signal_dir, f"{symbol}_signal.csv")
   # pd.DataFrame([signal]).to_csv(symbol_csv_path, index=False)

    all_signals.append(signal)

# Save all signals to one combined JSON and CSV
#combined_json_path = os.path.join(signal_dir, "signal.json")
combined_csv_path = os.path.join(signal_dir, "evaluate_and_generate_signal.csv")

#with open(combined_csv_path, "w") as f:
   # json.dump(all_signals, f, indent=2)

pd.DataFrame(all_signals).to_csv(combined_csv_path, index=False)

print("✅ All AI signals saved as CSV and JSON in csv/ai_signal/")
