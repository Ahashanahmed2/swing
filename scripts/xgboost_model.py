# scripts/xgboost_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import os
import sys

# Add current directory to path for config import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ফাইল পাথগুলো সরাসরি ডিফাইন করুন
MONGODB_PATH = "./csv/mongodb.csv"
TRADE_STOCK_PATH = "./csv/trade_stock.csv"
XGBOOST_MODEL_PATH = "./csv/models/xgboost_model.pkl"

class XGBoostTradingModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
        # Verify paths and create models directory
        self._setup_directories()
        self._verify_paths()

    def _setup_directories(self):
        """Create models directory if it doesn't exist"""
        models_dir = os.path.dirname(XGBOOST_MODEL_PATH)
        os.makedirs(models_dir, exist_ok=True)
        print(f"✅ Models directory ensured: {models_dir}")

    def _verify_paths(self):
        """Verify that required files exist"""
        print("🔍 Verifying file paths...")
        
        # Check CSV files
        required_files = [
            (MONGODB_PATH, "mongodb.csv"),
            (TRADE_STOCK_PATH, "trade_stock.csv")
        ]
        
        for path, name in required_files:
            if not os.path.exists(path):
                print(f"❌ {name} not found at: {path}")
                print(f"   Please ensure '{name}' exists in the csv/ directory")
                print(f"   Current csv directory: {os.path.dirname(path)}")
                if os.path.exists(os.path.dirname(path)):
                    print(f"   Files in directory:")
                    for file in os.listdir(os.path.dirname(path)):
                        print(f"     - {file}")
                raise FileNotFoundError(f"{name} not found at: {path}")
            else:
                print(f"✅ {name} found: {path}")
        
        print("✅ All required files verified!")

    def load_and_prepare_data(self):
        """লোড এবং প্রিপেয়ার ডেটা"""
        print("\n📊 Loading and preparing data...")
        
        # Load data with error handling
        try:
            market_data = pd.read_csv(MONGODB_PATH)
            trade_data = pd.read_csv(TRADE_STOCK_PATH)
        except Exception as e:
            print(f"❌ Error loading CSV files: {e}")
            raise
        
        print(f"   Market data shape: {market_data.shape}")
        print(f"   Trade data shape: {trade_data.shape}")
        
        # Display column information
        print(f"   Market data columns: {list(market_data.columns)}")
        print(f"   Trade data columns: {list(trade_data.columns)}")
        
        # Check required columns
        required_market_cols = ['symbol', 'date', 'open', 'close', 'high', 'low', 
                               'volume', 'rsi', 'macd', 'macd_signal', 
                               'bb_upper', 'bb_lower', 'atr']
        
        required_trade_cols = ['symbol', 'date', 'buy']
        
        # Check for missing columns
        missing_market = [col for col in required_market_cols if col not in market_data.columns]
        missing_trade = [col for col in required_trade_cols if col not in trade_data.columns]
        
        if missing_market:
            print(f"⚠️ Warning - Missing columns in market data: {missing_market}")
            print(f"   Available columns: {[col for col in market_data.columns if col in required_market_cols]}")
        
        if missing_trade:
            print(f"⚠️ Warning - Missing columns in trade data: {missing_trade}")
            print(f"   Available columns: {[col for col in trade_data.columns if col in required_trade_cols]}")

        # Feature engineering
        print("   Creating features...")
        market_data['returns'] = market_data['close'].pct_change()
        market_data['volatility'] = market_data['returns'].rolling(window=20).std()
        market_data['volume_change'] = market_data['volume'].pct_change()
        
        # Ensure date columns are in correct format for merging
        market_data['date'] = pd.to_datetime(market_data['date'])
        trade_data['date'] = pd.to_datetime(trade_data['date'])
        
        # Merge with trade data for labels
        print("   Merging data...")
        merged_data = pd.merge(market_data, trade_data[['symbol', 'date', 'buy']], 
                              on=['symbol', 'date'], how='left')

        # Create target variable (1 for buy signal, 0 otherwise)
        merged_data['target'] = (merged_data['buy'].notna()).astype(int)
        
        buy_signals = merged_data['target'].sum()
        total_samples = len(merged_data)
        print(f"   Buy signals found: {buy_signals} out of {total_samples} samples ({buy_signals/total_samples:.2%})")

        # Select features
        self.feature_names = ['open', 'close', 'high', 'low', 'volume', 'rsi', 'macd', 
                             'macd_signal', 'bb_upper', 'bb_lower', 'atr', 'returns',
                             'volatility', 'volume_change']

        # Check which features are available
        available_features = [col for col in self.feature_names if col in merged_data.columns]
        missing_features = [col for col in self.feature_names if col not in merged_data.columns]
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            self.feature_names = available_features
            print(f"   Using available features: {self.feature_names}")

        # Encode symbol if needed
        if 'symbol' in merged_data.columns:
            print("   Encoding symbols...")
            merged_data['symbol_encoded'] = self.label_encoder.fit_transform(merged_data['symbol'])
            self.feature_names.append('symbol_encoded')

        # Drop NaN values
        before_drop = merged_data.shape[0]
        merged_data = merged_data.dropna(subset=self.feature_names + ['target'])
        after_drop = merged_data.shape[0]
        
        print(f"   Dropped {before_drop - after_drop} rows with NaN values")
        print(f"   Final data shape: {merged_data.shape}")

        return merged_data[self.feature_names], merged_data['target']
    
    def train(self, X, y, test_size=0.2):
        """ট্রেইন মডেল"""
        print("\n🤖 Training XGBoost model...")
        
        # Check if we have enough data
        if len(X) < 100:
            print(f"⚠️ Warning: Only {len(X)} samples available. Model may not generalize well.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Testing samples: {X_test.shape[0]}")
        print(f"   Features: {X_train.shape[1]}")

        # Create and train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        print("   Fitting model...")
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate additional metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"\n📈 Model Performance:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🔝 Top 10 Important Features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")

        return accuracy
    
    def predict(self, market_data):
        """প্রেডিক্ট সিগন্যাল"""
        if self.model is None:
            raise ValueError("Model not trained. Please train or load the model first.")
        
        # Prepare features for prediction
        market_data = market_data.copy()
        
        # Add engineered features if not present
        if 'returns' not in market_data.columns:
            market_data['returns'] = market_data['close'].pct_change()
        if 'volatility' not in market_data.columns:
            market_data['volatility'] = market_data['returns'].rolling(window=20).std()
        if 'volume_change' not in market_data.columns:
            market_data['volume_change'] = market_data['volume'].pct_change()

        # Encode symbols
        if 'symbol' in market_data.columns:
            # Handle unseen symbols
            known_symbols = set(self.label_encoder.classes_)
            current_symbols = set(market_data['symbol'].unique())
            
            # For symbols not seen during training, use a default encoding
            market_data['symbol_encoded'] = market_data['symbol'].apply(
                lambda x: self.label_encoder.transform([x])[0] if x in known_symbols else -1
            )

        # Ensure all features are present
        missing_features = [f for f in self.feature_names if f not in market_data.columns]
        if missing_features:
            print(f"⚠️ Warning: Missing features for prediction: {missing_features}")
            for feature in missing_features:
                market_data[feature] = 0

        # Get predictions
        predictions = self.model.predict(market_data[self.feature_names].fillna(0))
        probabilities = self.model.predict_proba(market_data[self.feature_names].fillna(0))[:, 1]

        return predictions, probabilities
    
    def save_model(self):
        """সেভ মডেল"""
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(XGBOOST_MODEL_PATH), exist_ok=True)
        
        # Save the model
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }, XGBOOST_MODEL_PATH)
        
        print(f"✅ Model saved to {XGBOOST_MODEL_PATH}")
    
    def load_model(self):
        """লোড মডেল"""
        if not os.path.exists(XGBOOST_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {XGBOOST_MODEL_PATH}")
        
        loaded_data = joblib.load(XGBOOST_MODEL_PATH)
        self.model = loaded_data['model']
        self.label_encoder = loaded_data['label_encoder']
        self.feature_names = loaded_data['feature_names']
        
        print(f"✅ Model loaded from {XGBOOST_MODEL_PATH}")
        print(f"   Model type: {type(self.model).__name__}")
        print(f"   Number of features: {len(self.feature_names)}")
        print(f"   Feature names: {self.feature_names}")

def main():
    """মেইন এক্সিকিউশন ফাংশন"""
    try:
        print("=" * 60)
        print("XGBOOST TRADING MODEL TRAINING")
        print("=" * 60)
        
        # Create model instance
        model = XGBoostTradingModel()
        
        # Load and prepare data
        print("\n📥 Loading data...")
        X, y = model.load_and_prepare_data()
        
        # Check if we have enough data
        if len(X) < 10:
            print(f"\n❌ Insufficient data: Only {len(X)} samples available")
            print("   Need at least 10 samples to train model")
            return None
        
        # Train model
        print("\n🎯 Training model...")
        accuracy = model.train(X, y)
        
        # Save model
        print("\n💾 Saving model...")
        model.save_model()
        
        print("\n" + "=" * 60)
        print(f"✅ TRAINING COMPLETED! Accuracy: {accuracy:.4f}")
        print("=" * 60)
        
        return model
        
    except FileNotFoundError as e:
        print(f"\n❌ File error: {e}")
        print("\n💡 Solution: Please ensure:")
        print("   1. Your CSV files are in the 'csv/' directory")
        print("   2. Files are named correctly: trade_stock.csv and mongodb.csv")
        print("   3. You're running the script from the correct directory")
        print(f"   Current directory: {os.getcwd()}")
        return None
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()