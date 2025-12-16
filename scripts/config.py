# scripts/config.py
import os

# ==================== PATH CONFIGURATION ====================
# Get the root directory (where csv/ folder is located)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
CSV_DIR = os.path.join(ROOT_DIR, "csv")
MODELS_DIR = os.path.join(CSV_DIR, "models")  # csv/models/ directory

# Data file paths
TRADE_STOCK_PATH = os.path.join(CSV_DIR, "trade_stock.csv")
MONGODB_PATH = os.path.join(CSV_DIR, "mongodb.csv")

# Model save paths
XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.pkl")
PPO_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_agent.pth")

# ==================== TRADING PARAMETERS ====================
INITIAL_BALANCE = 500000  # 500,000 BDT
MAX_POSITION_SIZE = 0.1   # 10%
MAX_PORTFOLIO_RISK = 0.01 # 1.0% per trade
STOP_LOSS_PCT = 0.05      # 5%
RISK_PERCENT = 0.01       # 1% risk per trade

# ==================== CREATE DIRECTORIES ====================
def create_directories():
    """Create all necessary directories"""
    directories = [CSV_DIR, MODELS_DIR]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory ensured: {directory}")

# Create directories when config is imported
create_directories()

# ==================== DEBUG INFO ====================
if __name__ == "__main__":
    print("=== Configuration Debug Info ===")
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"CSV_DIR: {CSV_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    
    print(f"\nTRADE_STOCK_PATH: {TRADE_STOCK_PATH}")
    print(f"MONGODB_PATH: {MONGODB_PATH}")
    print(f"XGBOOST_MODEL_PATH: {XGBOOST_MODEL_PATH}")
    
    # Check if directories exist
    print("\n=== Directory Existence Check ===")
    for path, name in [(CSV_DIR, "csv/"), 
                       (MODELS_DIR, "csv/models/")]:
        exists = os.path.exists(path)
        status = "✅ EXISTS" if exists else "❌ NOT FOUND"
        print(f"{name}: {status}")
    
    # Check if files exist
    print("\n=== File Existence Check ===")
    for path, name in [(TRADE_STOCK_PATH, "trade_stock.csv"), 
                       (MONGODB_PATH, "mongodb.csv")]:
        exists = os.path.exists(path)
        status = "✅ EXISTS" if exists else "❌ NOT FOUND"
        print(f"{name}: {status}")