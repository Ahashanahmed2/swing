# config.py
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR = os.path.join(BASE_DIR, "src")

# Data file paths
TRADE_STOCK_PATH = os.path.join(CSV_DIR, "trade_stock.csv")
MONGODB_PATH = os.path.join(CSV_DIR, "mongodb.csv")

# Model save paths
XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.pkl")
PPO_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_agent.pth")

# Trading parameters
INITIAL_BALANCE = 100000
MAX_POSITION_SIZE = 0.1  # 10%
MAX_PORTFOLIO_RISK = 0.02  # 2%
STOP_LOSS_PCT = 0.05  # 5%

# Create directories if they don't exist
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SRC_DIR, exist_ok=True)