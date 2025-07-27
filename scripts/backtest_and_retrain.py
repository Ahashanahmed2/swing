import os
import pandas as pd
from datetime import datetime
from stable_baselines3 import DQN
from env import TradeEnv
from stable_baselines3.common.vec_env import DummyVecEnv
#from loss_logging_callback import LossLoggingCallback  # যদি আলাদা ফাইলে থাকে
#callback = LossLoggingCallback()
# 📂 Paths
main_df_path = './csv/mongodb.csv'
last_train_path = './csv/last_train_date.txt'
initial_date_str = "2025-07-12"

# 📊 Load main_df
main_df = pd.read_csv(main_df_path)
main_df['date'] = pd.to_datetime(main_df['date']).dt.date

# 🕒 Step 1: Get last_train_date or create with initial
last_train_date = None
if os.path.exists(last_train_path):
    try:
        with open(last_train_path, "r") as f:
            last_train_date_str = f.read().strip()
            if last_train_date_str:
                last_train_date = datetime.strptime(last_train_date_str, "%Y-%m-%d").date()
                print(f"📅 Last trained on: {last_train_date}")
            else:
                raise ValueError("⛔ Empty date in file")
    except Exception as e:
        print(f"⚠️ Could not read date: {e}")
else:
    print("📁 File does not exist. Will create it with initial date.")

# যদি না থাকে, তাহলে 2025-05-13 থেকে শুরু করবে এবং সেইভও করবে
if last_train_date is None:
    last_train_date = datetime.strptime(initial_date_str, "%Y-%m-%d").date()
    os.makedirs(os.path.dirname(last_train_path), exist_ok=True)
    with open(last_train_path, "w") as f:
        f.write(initial_date_str)
    print(f"📄 Created last_train_date.txt with {initial_date_str}")

# 🔍 Step 2: Filter new data
new_data = main_df[main_df["date"] > last_train_date]

if new_data.empty:
    print("🟡 No new data found. Training skipped.")
    exit()

# ✅ Load all required datasets
filtered_df = pd.read_csv('./csv/swing/filtered_low_rsi_candles.csv')
                            
imbalance_high = pd.concat([pd.read_csv('./csv/swing/imbalanceZone/down_to_up/' + f)
                            for f in os.listdir('./csv/swing/imbalanceZone/down_to_up') if f.endswith('.csv')])
imbalance_low = pd.concat([pd.read_csv('./csv/swing/imbalanceZone/up_to_down/' + f)
                           for f in os.listdir('./csv/swing/imbalanceZone/up_to_down') if f.endswith('.csv')])
swing_high_candle = pd.concat([pd.read_csv('./csv/swing/swing_high/high_candle/' + f)
                               for f in os.listdir('./csv/swing/swing_high/high_candle') if f.endswith('.csv')])
swing_high_confirm = pd.concat([pd.read_csv('./csv/swing/swing_high/high_confirm/' + f)
                                for f in os.listdir('./csv/swing/swing_high/high_confirm') if f.endswith('.csv')])
swing_low_candle = pd.concat([pd.read_csv('./csv/swing/swing_low/low_candle/' + f)
                              for f in os.listdir('./csv/swing/swing_low/low_candle') if f.endswith('.csv')])
swing_low_confirm = pd.concat([pd.read_csv('./csv/swing/swing_low/low_confirm/' + f)
                               for f in os.listdir('./csv/swing/swing_low/low_confirm') if f.endswith('.csv')])
rsi_divergences = pd.read_csv("./csv/swing/rsi_divergences/rsi_divergences.csv")
filtered_output_path = './csv/filtered_output.csv'
filtered_output = pd.read_csv(filtered_output_path) if os.path.exists(filtered_output_path) and not pd.read_csv(filtered_output_path).empty else pd.DataFrame()

down_to_up = pd.read_csv("./csv/swing/down_to_up.csv")
up_to_down = pd.read_csv("./csv/swing/up_to_down.csv")

# ✅ Prepare environment
env = TradeEnv(
    main_df,
    filtered_df,
    imbalance_high,
    imbalance_low,
    swing_high_candle,
    swing_high_confirm,
    swing_low_candle,
    swing_low_confirm,
    rsi_divergences,
    filtered_output,
    down_to_up,
    up_to_down
)
env = DummyVecEnv([lambda: env])

# ✅ Load previous model or create new one
model_path = "./csv/dqn_retrained"
if os.path.exists(model_path + ".zip"):
    model = DQN.load(model_path, env=env)
    print("✅ Loaded existing model for fine-tuning")
else:
    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=128,
        tau=0.01,
        gamma=0.95,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        device='cpu'
    )
    print("🆕 Created new model")

# 🧠 Train
print("🚀 Training model...")
model.learn(total_timesteps=200_000)
model.save(model_path)
print(f"✅ Model saved at {model_path}")

# ✅ Update last_train_date with latest available date
latest_date = new_data["date"].max()
try:
    with open(last_train_path, "w") as f:
        f.write(str(latest_date))
    print(f"📝 Updated last_train_date.txt to {latest_date}")
except Exception as e:
    print(f"❌ Failed to update last_train_date.txt: {e}")
