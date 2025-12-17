# scripts/xgboost_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import joblib
import os
import sys
from imblearn.over_sampling import SMOTE
from collections import Counter

class XGBoostTradingModel:
    def __init__(self, use_smote=True):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.use_smote = use_smote
        
    def load_and_prepare_data(self):
        """লোড এবং প্রিপেয়ার ডেটা"""
        print("\n📊 Loading and preparing data...")
        
        try:
            market_data = pd.read_csv("csv/mongodb.csv")
            trade_data = pd.read_csv("csv/trade_stock.csv")
        except Exception as e:
            print(f"❌ Error loading CSV files: {e}")
            raise
        
        print(f"   Market data shape: {market_data.shape}")
        print(f"   Trade data shape: {trade_data.shape}")
        
        # Feature engineering
        print("   Creating features...")
        market_data['returns'] = market_data['close'].pct_change()
        market_data['volatility'] = market_data['returns'].rolling(window=20).std()
        market_data['volume_change'] = market_data['volume'].pct_change()
        market_data['price_range'] = (market_data['high'] - market_data['low']) / market_data['close']
        
        # Additional features
        market_data['ma_5'] = market_data['close'].rolling(window=5).mean()
        market_data['ma_20'] = market_data['close'].rolling(window=20).mean()
        market_data['ma_ratio'] = market_data['ma_5'] / market_data['ma_20']
        market_data['rsi_signal'] = (market_data['rsi'] > 30) & (market_data['rsi'] < 70)
        market_data['macd_signal'] = market_data['macd'] > market_data['macd_signal']
        
        # Ensure date columns are in correct format
        market_data['date'] = pd.to_datetime(market_data['date'])
        trade_data['date'] = pd.to_datetime(trade_data['date'])
        
        # Merge with trade data for labels
        print("   Merging data...")
        merged = pd.merge(market_data, trade_data[['symbol', 'date', 'buy', 'SL', 'tp', 'RRR']], 
                         on=['symbol', 'date'], how='left')
        
        # Create target variable (1 for buy signal, 0 otherwise)
        merged['target'] = merged['buy'].notna().astype(int)
        
        # Create multi-class target for better learning
        # 0: no signal, 1: buy signal with good RRR (>1.5), 2: buy signal with moderate RRR (1.0-1.5)
        merged['target_multi'] = 0  # default: no signal
        merged.loc[merged['buy'].notna() & (merged['RRR'] > 1.5), 'target_multi'] = 2  # strong buy
        merged.loc[merged['buy'].notna() & (merged['RRR'] <= 1.5) & (merged['RRR'] > 0), 'target_multi'] = 1  # moderate buy
        
        buy_signals = merged['target'].sum()
        total_samples = len(merged)
        print(f"   Buy signals found: {buy_signals} out of {total_samples} samples ({buy_signals/total_samples:.2%})")
        print(f"   Target distribution: {merged['target_multi'].value_counts().to_dict()}")
        
        # Select features
        self.feature_names = [
            'open', 'close', 'high', 'low', 'volume', 
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'atr',
            'returns', 'volatility', 'volume_change', 'price_range',
            'ma_5', 'ma_20', 'ma_ratio'
        ]
        
        # শুধু available কলামগুলো নিন
        self.feature_names = [col for col in self.feature_names if col in merged.columns]
        print(f"   Using {len(self.feature_names)} features")
        
        # Encode symbol
        if 'symbol' in merged.columns:
            merged['symbol_encoded'] = self.label_encoder.fit_transform(merged['symbol'])
            self.feature_names.append('symbol_encoded')
        
        # Add technical indicator signals as features
        if 'rsi_signal' in merged.columns:
            merged['rsi_signal'] = merged['rsi_signal'].astype(int)
            self.feature_names.append('rsi_signal')
        if 'macd_signal' in merged.columns:
            merged['macd_signal'] = merged['macd_signal'].astype(int)
            self.feature_names.append('macd_signal')
        
        # Drop NaN values
        before_drop = merged.shape[0]
        merged = merged.dropna(subset=self.feature_names + ['target', 'target_multi'])
        after_drop = merged.shape[0]
        
        print(f"   Dropped {before_drop - after_drop} rows with NaN values")
        print(f"   Final data shape: {merged.shape}")
        
        return merged[self.feature_names], merged['target'], merged['target_multi']
    
    def train(self, X, y_binary, y_multi, test_size=0.3):
        """ট্রেইন মডেল"""
        print("\n🤖 Training XGBoost model...")
        
        if len(X) < 100:
            print(f"   ⚠️ Only {len(X)} samples available")
            print("   Need at least 100 samples for meaningful training")
            return 0.0, 0.0
        
        # Split data
        X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = train_test_split(
            X, y_binary, y_multi, test_size=test_size, random_state=42, stratify=y_binary
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        print(f"   Class distribution - Binary: {Counter(y_train_bin)}")
        print(f"   Class distribution - Multi: {Counter(y_train_multi)}")
        
        # Apply SMOTE if enabled
        if self.use_smote and len(np.unique(y_train_bin)) > 1:
            print("   Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_bin)
            print(f"   After SMOTE: {Counter(y_train_balanced)}")
            X_train = X_train_balanced
            y_train_bin = y_train_balanced
        
        # Calculate class weights for imbalance
        class_counts = np.bincount(y_train_bin)
        if len(class_counts) > 1 and class_counts[1] > 0:
            scale_pos_weight = class_counts[0] / class_counts[1]
            print(f"   Class weight (scale_pos_weight): {scale_pos_weight:.2f}")
        else:
            scale_pos_weight = 1.0
        
        # Create and train binary classification model
        print("   Training binary classification model...")
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        )
        
        self.model.fit(
            X_train, y_train_bin,
            eval_set=[(X_test, y_test_bin)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Evaluate binary model
        y_pred_bin = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy_bin = accuracy_score(y_test_bin, y_pred_bin)
        f1_bin = f1_score(y_test_bin, y_pred_bin, zero_division=0)
        precision_bin = precision_score(y_test_bin, y_pred_bin, zero_division=0)
        recall_bin = recall_score(y_test_bin, y_pred_bin, zero_division=0)
        
        try:
            auc_bin = roc_auc_score(y_test_bin, y_pred_proba)
        except:
            auc_bin = 0.0
        
        print(f"\n📈 Binary Model Performance:")
        print(f"   Accuracy: {accuracy_bin:.4f}")
        print(f"   F1-Score: {f1_bin:.4f}")
        print(f"   Precision: {precision_bin:.4f}")
        print(f"   Recall: {recall_bin:.4f}")
        print(f"   AUC-ROC: {auc_bin:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🔝 Top 15 Important Features:")
            for idx, row in feature_importance.head(15).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return accuracy_bin, f1_bin
    
    def save_model(self):
        """সেভ মডেল"""
        if self.model is None:
            print("   ⚠️ No model to save")
            return
        
        # Save model
        model_dir = "csv/models"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "xgboost_model.pkl")
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'feature_importances': dict(zip(self.feature_names, self.model.feature_importances_))
        }, model_path)
        
        print(f"   ✅ Model saved to: {model_path}")
        
        # Save feature importance as CSV
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = os.path.join(model_dir, "feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        print(f"   ✅ Feature importance saved to: {importance_path}")
    
    def predict_with_threshold(self, X, threshold=0.5):
        """প্রেডিক্ট with custom threshold"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities

def main():
    """মেইন ফাংশন"""
    print("=" * 70)
    print("XGBOOST TRADING MODEL - ADVANCED TRAINING")
    print("=" * 70)
    
    try:
        # Create and train model with SMOTE
        model = XGBoostTradingModel(use_smote=True)
        
        # Load data
        print("\n📥 Loading data...")
        X, y_binary, y_multi = model.load_and_prepare_data()
        
        if len(X) == 0:
            print("❌ No data available for training")
            return
        
        # Train if enough data
        if len(X) >= 100:
            accuracy, f1 = model.train(X, y_binary, y_multi)
            
            if f1 > 0:  # F1-score check for meaningful model
                model.save_model()
                
                print(f"\n" + "=" * 70)
                print(f"✅ TRAINING COMPLETED SUCCESSFULLY!")
                print(f"   Accuracy: {accuracy:.4f}")
                print(f"   F1-Score: {f1:.4f}")
                print("=" * 70)
                
                # Generate prediction report
                print("\n📊 Generating sample predictions...")
                sample_size = min(100, len(X))
                sample_X = X.iloc[:sample_size]
                
                predictions, probabilities = model.predict_with_threshold(sample_X, threshold=0.3)
                
                report_df = pd.DataFrame({
                    'probability': probabilities[:10],
                    'prediction': predictions[:10],
                    'actual': y_binary.iloc[:10].values if len(y_binary) >= 10 else [0]*10
                })
                print("\nSample predictions (threshold=0.3):")
                print(report_df.to_string(index=False))
                
            else:
                print("\n⚠️ Model trained but F1-score is zero")
                print("   Consider adjusting class weights or threshold")
        else:
            print(f"\n⚠️ Insufficient data: {len(X)} samples (need at least 100)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Script completed")

if __name__ == "__main__":
    main()