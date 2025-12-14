import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib

class XGBoostTradingModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_and_prepare_data(self, mongodb_path, trade_stock_path):
        # Load data
        market_data = pd.read_csv(mongodb_path)
        trade_data = pd.read_csv(trade_stock_path)
        
        # Feature engineering
        market_data['returns'] = market_data['close'].pct_change()
        market_data['volatility'] = market_data['returns'].rolling(window=20).std()
        market_data['volume_change'] = market_data['volume'].pct_change()
        
        # Merge with trade data for labels
        merged_data = pd.merge(market_data, trade_data[['symbol', 'date', 'buy']], 
                              on=['symbol', 'date'], how='left')
        
        # Create target variable (1 for buy signal, 0 otherwise)
        merged_data['target'] = (merged_data['buy'].notna()).astype(int)
        
        # Select features
        features = ['open', 'close', 'high', 'low', 'volume', 'rsi', 'macd', 
                   'macd_signal', 'bb_upper', 'bb_lower', 'atr', 'returns',
                   'volatility', 'volume_change']
        
        # Encode symbol if needed
        if 'symbol' in merged_data.columns:
            merged_data['symbol_encoded'] = self.label_encoder.fit_transform(merged_data['symbol'])
            features.append('symbol_encoded')
        
        # Drop NaN values
        merged_data = merged_data.dropna(subset=features + ['target'])
        
        return merged_data[features], merged_data['target']
    
    def train(self, X, y, test_size=0.2):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
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
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict_signals(self, market_data):
        # Prepare features for prediction
        market_data = market_data.copy()
        market_data['returns'] = market_data['close'].pct_change()
        market_data['volatility'] = market_data['returns'].rolling(window=20).std()
        market_data['volume_change'] = market_data['volume'].pct_change()
        
        if 'symbol' in market_data.columns:
            market_data['symbol_encoded'] = self.label_encoder.transform(market_data['symbol'])
        
        features = ['open', 'close', 'high', 'low', 'volume', 'rsi', 'macd', 
                   'macd_signal', 'bb_upper', 'bb_lower', 'atr', 'returns',
                   'volatility', 'volume_change']
        
        if 'symbol_encoded' in market_data.columns:
            features.append('symbol_encoded')
        
        # Get predictions
        predictions = self.model.predict(market_data[features].fillna(0))
        probabilities = self.model.predict_proba(market_data[features].fillna(0))[:, 1]
        
        return predictions, probabilities
    
    def save_model(self, path='xgboost_model.pkl'):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='xgboost_model.pkl'):
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")

# Usage
if __name__ == "__main__":
    xgb_model = XGBoostTradingModel()
    X, y = xgb_model.load_and_prepare_data("./csv/mongodb.csv", "./csv/trade_stock.csv")
    accuracy = xgb_model.train(X, y)
    xgb_model.save_model()