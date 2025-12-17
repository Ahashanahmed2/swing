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
        
    def load_and_prepare_data(self):
        """লোড এবং প্রিপেয়ার ডেটা"""
        print("\n📊 Loading and preparing data...")
        
        try:
            # Load data
            market_data = pd.read_csv("csv/mongodb.csv")
            trade_data = pd.read_csv("csv/trade_stock.csv")
        except Exception as e:
            print(f"❌ Error loading CSV files: {e}")
            raise
        
        print(f"   Market data shape: {market_data.shape}")
        print(f"   Trade data shape: {trade_data.shape}")
        
        # ডিবাগিং: কলামগুলো দেখুন
        print(f"\n   Trade columns: {list(trade_data.columns)}")
        print(f"   Market columns: {list(market_data.columns)[:15]}...")
        
        # 'buy' কলাম আছে কিনা চেক করুন
        if 'buy' not in trade_data.columns:
            print("❌ 'buy' column not found in trade_stock.csv")
            print("   Available columns in trade data:", list(trade_data.columns))
            # বিকল্প কলাম খুঁজুন
            if 'Buy' in trade_data.columns:
                trade_data.rename(columns={'Buy': 'buy'}, inplace=True)
                print("   ✅ Renamed 'Buy' to 'buy'")
            elif 'BUY' in trade_data.columns:
                trade_data.rename(columns={'BUY': 'buy'}, inplace=True)
                print("   ✅ Renamed 'BUY' to 'buy'")
            else:
                # numeric কলাম খুঁজুন
                numeric_cols = trade_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    trade_data['buy'] = trade_data[numeric_cols[0]]
                    print(f"   ⚠️ Created 'buy' column from '{numeric_cols[0]}'")
                else:
                    # ডিফল্ট buy কলাম তৈরি করুন
                    trade_data['buy'] = 1
                    print("   ⚠️ Created default 'buy' column with value 1")
        
        # Feature engineering
        print("   Creating features...")
        market_data['returns'] = market_data['close'].pct_change()
        market_data['volatility'] = market_data['returns'].rolling(window=20).std()
        market_data['volume_change'] = market_data['volume'].pct_change()
        
        # Ensure date columns are in correct format
        market_data['date'] = pd.to_datetime(market_data['date'])
        trade_data['date'] = pd.to_datetime(trade_data['date'])
        
        # ডিবাগিং: merge করার আগে
        print(f"   Before merge - Market date type: {market_data['date'].dtype}")
        print(f"   Before merge - Trade date type: {trade_data['date'].dtype}")
        print(f"   Market symbols: {market_data['symbol'].unique()[:5]}...")
        print(f"   Trade symbols: {trade_data['symbol'].unique()[:5]}...")
        
        # Merge with trade data for labels
        print("   Merging data...")
        merged = pd.merge(market_data, 
                         trade_data[['symbol', 'date', 'buy']], 
                         on=['symbol', 'date'], 
                         how='left')
        
        print(f"   After merge shape: {merged.shape}")
        print(f"   Merged columns: {list(merged.columns)}")
        
        # 'buy' কলাম merge হয়েছে কিনা চেক করুন
        if 'buy' not in merged.columns:
            print("❌ 'buy' column not in merged dataframe after merge!")
            print("   Trying alternative merge...")
            # Alternative merge
            merged = market_data.copy()
            # Add buy column manually
            buy_dict = trade_data.set_index(['symbol', 'date'])['buy'].to_dict()
            merged['buy'] = merged.apply(
                lambda row: buy_dict.get((row['symbol'], row['date']), np.nan), 
                axis=1
            )
        
        # Create target variable (1 for buy signal, 0 otherwise)
        merged['target'] = merged['buy'].notna().astype(int)
        
        buy_signals = merged['target'].sum()
        total_samples = len(merged)
        print(f"   Buy signals found: {buy_signals} out of {total_samples} samples ({buy_signals/total_samples:.2%})")
        
        # Select features
        self.feature_names = ['open', 'close', 'high', 'low', 'volume', 'rsi', 'macd', 
                             'macd_signal', 'bb_upper', 'bb_lower', 'atr', 'returns',
                             'volatility', 'volume_change']
        
        # শুধু available কলামগুলো নিন
        self.feature_names = [col for col in self.feature_names if col in merged.columns]
        print(f"   Available features: {self.feature_names}")
        
        # Encode symbol
        if 'symbol' in merged.columns:
            merged['symbol_encoded'] = self.label_encoder.fit_transform(merged['symbol'])
            self.feature_names.append('symbol_encoded')
        
        # Drop NaN values
        before_drop = merged.shape[0]
        merged = merged.dropna(subset=self.feature_names + ['target'])
        after_drop = merged.shape[0]
        
        print(f"   Dropped {before_drop - after_drop} rows with NaN values")
        print(f"   Final data shape: {merged.shape}")
        
        # ডিবাগিং: কিছু স্যাম্পল দেখুন
        if len(merged) > 0:
            print(f"\n   Sample targets: {merged['target'].value_counts().to_dict()}")
            print(f"   Sample buy values (non-null): {merged['buy'].dropna().unique()[:5]}")
        
        return merged[self.feature_names], merged['target']
    
    def train(self, X, y, test_size=0.3):
        """ট্রেইন মডেল"""
        print("\n🤖 Training XGBoost model...")
        
        if len(X) < 50:
            print(f"   ⚠️ Only {len(X)} samples available")
            print("   Need at least 50 samples for meaningful training")
            return 0.0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        # Create and train model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📈 Model Performance:")
        print(f"   Accuracy: {accuracy:.4f}")
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
            'feature_names': self.feature_names
        }, model_path)
        
        print(f"   ✅ Model saved to: {model_path}")

def debug_merge_issue():
    """ডিবাগ ফাংশন merge সমস্যা খুঁজে বের করার জন্য"""
    print("\n🔧 Debugging merge issue...")
    
    try:
        market_data = pd.read_csv("csv/mongodb.csv")
        trade_data = pd.read_csv("csv/trade_stock.csv")
        
        print(f"Market data shape: {market_data.shape}")
        print(f"Trade data shape: {trade_data.shape}")
        
        # ডেটা টাইপ চেক করুন
        print(f"\nMarket 'date' dtype: {market_data['date'].dtype}")
        print(f"Trade 'date' dtype: {trade_data['date'].dtype}")
        
        # কনভার্ট করুন
        market_data['date'] = pd.to_datetime(market_data['date'])
        trade_data['date'] = pd.to_datetime(trade_data['date'])
        
        # ইউনিক ভ্যালু চেক করুন
        print(f"\nUnique market dates: {market_data['date'].nunique()}")
        print(f"Unique trade dates: {trade_data['date'].nunique()}")
        
        print(f"\nMarket dates range: {market_data['date'].min()} to {market_data['date'].max()}")
        print(f"Trade dates range: {trade_data['date'].min()} to {trade_data['date'].max()}")
        
        # Merge চেষ্টা করুন
        merged = pd.merge(market_data, trade_data[['symbol', 'date', 'buy']], 
                         on=['symbol', 'date'], how='left')
        
        print(f"\nAfter merge shape: {merged.shape}")
        print(f"Columns in merged: {list(merged.columns)}")
        
        # 'buy' কলাম আছে কিনা
        if 'buy' in merged.columns:
            non_null_buy = merged['buy'].notna().sum()
            print(f"'buy' column exists with {non_null_buy} non-null values")
        else:
            print("❌ 'buy' column not in merged dataframe!")
            
            # কেন merge হচ্ছে না দেখা যাক
            common_symbols = set(market_data['symbol']).intersection(set(trade_data['symbol']))
            common_dates = set(market_data['date']).intersection(set(trade_data['date']))
            
            print(f"\nCommon symbols: {len(common_symbols)}")
            print(f"Common dates: {len(common_dates)}")
            
            if len(common_symbols) == 0:
                print("❌ No common symbols between market and trade data!")
                print(f"Market symbols: {market_data['symbol'].unique()[:10]}")
                print(f"Trade symbols: {trade_data['symbol'].unique()[:10]}")
            
            if len(common_dates) == 0:
                print("❌ No common dates between market and trade data!")
                print(f"Market dates: {market_data['date'].unique()[:10]}")
                print(f"Trade dates: {trade_data['date'].unique()[:10]}")
        
    except Exception as e:
        print(f"❌ Debug error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """মেইন ফাংশন"""
    print("=" * 60)
    print("XGBoost Trading Model Training")
    print("=" * 60)
    
    try:
        # প্রথমে ডিবাগ করুন
        debug_merge_issue()
        
        # Create and train model
        model = XGBoostTradingModel()
        
        # Load data
        print("\n📥 Loading data...")
        X, y = model.load_and_prepare_data()
        
        if len(X) == 0:
            print("❌ No data available for training")
            return
        
        # Train if enough data
        if len(X) >= 20:
            accuracy = model.train(X, y)
            
            if accuracy > 0:
                model.save_model()
                print(f"\n✅ Training completed! Accuracy: {accuracy:.4f}")
            else:
                print("\n⚠️ Training completed with zero accuracy")
        else:
            print(f"\n⚠️ Insufficient data: {len(X)} samples (need at least 20)")
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("   Please ensure trade_stock.csv and mongodb.csv are in csv/ directory")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Script completed")

if __name__ == "__main__":
    main()