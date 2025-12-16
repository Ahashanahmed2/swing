# scripts/xgboost_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
import sys

class XGBoostTradingModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def _get_csv_paths(self):
        """Get correct CSV file paths based on execution context"""
        # Get current working directory
        current_dir = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        print(f"Current directory: {current_dir}")
        print(f"Script directory: {script_dir}")
        
        # Try different possible paths
        possible_paths = [
            os.path.join(current_dir, "csv", "mongodb.csv"),  # main.py from root
            os.path.join(current_dir, "mongodb.csv"),          # direct execution
            os.path.join(script_dir, "..", "csv", "mongodb.csv"),  # from scripts dir
            os.path.join("csv", "mongodb.csv"),                # relative path
            "./csv/mongodb.csv",                               # another relative
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found mongodb.csv at: {path}")
                mongodb_path = path
                trade_stock_path = path.replace("mongodb.csv", "trade_stock.csv")
                return mongodb_path, trade_stock_path
        
        # If not found, show error
        print("❌ Could not find CSV files in any location")
        print("   Checked locations:")
        for path in possible_paths:
            print(f"   - {path} (exists: {os.path.exists(path)})")
        
        raise FileNotFoundError("CSV files not found. Please ensure trade_stock.csv and mongodb.csv are in csv/ directory")

    def load_and_prepare_data(self):
        """লোড এবং প্রিপেয়ার ডেটা"""
        print("\n📊 Loading and preparing data...")
        
        # Get correct file paths
        mongodb_path, trade_stock_path = self._get_csv_paths()
        
        # Load data
        try:
            market_data = pd.read_csv(mongodb_path)
            trade_data = pd.read_csv(trade_stock_path)
        except Exception as e:
            print(f"❌ Error loading CSV files: {e}")
            raise
        
        print(f"   Market data shape: {market_data.shape}")
        print(f"   Trade data shape: {trade_data.shape}")
        
        # Quick check if data looks okay
        print(f"   Market columns: {list(market_data.columns)[:10]}...")
        print(f"   Trade columns: {list(trade_data.columns)}")
        
        # Feature engineering
        market_data['returns'] = market_data['close'].pct_change()
        market_data['volatility'] = market_data['returns'].rolling(window=20).std()
        market_data['volume_change'] = market_data['volume'].pct_change()
        
        # Merge for labels
        merged_data = pd.merge(market_data, trade_data[['symbol', 'date', 'buy']], 
                              on=['symbol', 'date'], how='left')
        
        # Create target (1 for buy signal)
        merged_data['target'] = (merged_data['buy'].notna()).astype(int)
        
        buy_signals = merged_data['target'].sum()
        print(f"   Buy signals found: {buy_signals} out of {len(merged_data)} samples")
        
        # Features
        self.feature_names = ['open', 'close', 'high', 'low', 'volume', 'rsi', 'macd', 
                             'macd_signal', 'bb_upper', 'bb_lower', 'atr', 'returns',
                             'volatility', 'volume_change']
        
        # Encode symbol
        if 'symbol' in merged_data.columns:
            merged_data['symbol_encoded'] = self.label_encoder.fit_transform(merged_data['symbol'])
            self.feature_names.append('symbol_encoded')
        
        # Clean data
        merged_data = merged_data.dropna(subset=self.feature_names + ['target'])
        
        return merged_data[self.feature_names], merged_data['target']
    
    def train(self, X, y, test_size=0.2):
        """ট্রেইন মডেল"""
        print("\n🤖 Training XGBoost model...")
        
        if len(X) < 20:
            print(f"⚠️ Only {len(X)} samples. Skipping training.")
            return 0.0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📈 Model Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        return accuracy
    
    def save_model(self):
        """সেভ মডেল"""
        if self.model is None:
            print("⚠️ No model to save")
            return
        
        # Save to csv/models directory
        models_dir = os.path.join("csv", "models")
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, "xgboost_model.pkl")
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }, model_path)
        
        print(f"✅ Model saved to {model_path}")

def main():
    """মেইন এক্সিকিউশন ফাংশন - main.py থেকে কল হলে"""
    try:
        print("=" * 60)
        print("XGBOOST MODEL - Starting training...")
        print("=" * 60)
        
        model = XGBoostTradingModel()
        X, y = model.load_and_prepare_data()
        
        if len(X) > 0:
            accuracy = model.train(X, y)
            if accuracy > 0:
                model.save_model()
                print(f"\n✅ Training completed! Accuracy: {accuracy:.4f}")
            else:
                print("\n⚠️ Training skipped - not enough data or poor accuracy")
        else:
            print("\n❌ No data available for training")
        
    except FileNotFoundError as e:
        print(f"\n❌ File error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()