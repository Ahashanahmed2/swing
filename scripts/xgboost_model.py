# scripts/xgboost_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

class XGBoostTradingModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self):
        """লোড এবং প্রিপেয়ার ডেটা - ALL FEATURES FROM mongodb.csv"""
        print("\n📊 Loading and preparing data...")
        
        try:
            market_data = pd.read_csv("csv/mongodb.csv")
            trade_data = pd.read_csv("csv/trade_stock.csv")
        except Exception as e:
            print(f"❌ Error loading CSV files: {e}")
            raise
        
        print(f"   Market data shape: {market_data.shape}")
        print(f"   Trade data shape: {trade_data.shape}")
        
        # Show available columns
        print(f"\n   Available market columns ({len(market_data.columns)}):")
        for col in market_data.columns:
            print(f"     - {col}")
        
        # 1. ADVANCED FEATURE ENGINEERING USING ALL COLUMNS
        print("\n   Creating advanced features from all available columns...")
        
        # Group by symbol for time-series operations
        market_data = market_data.sort_values(['symbol', 'date'])
        
        # A. PRICE ACTION FEATURES
        market_data['returns'] = market_data.groupby('symbol')['close'].pct_change()
        market_data['returns_5'] = market_data.groupby('symbol')['close'].pct_change(5)
        market_data['returns_10'] = market_data.groupby('symbol')['close'].pct_change(10)
        
        market_data['high_low_range'] = (market_data['high'] - market_data['low']) / market_data['close']
        market_data['close_to_open'] = (market_data['close'] - market_data['open']) / market_data['open']
        market_data['body_size'] = abs(market_data['close'] - market_data['open']) / (market_data['high'] - market_data['low'] + 1e-8)
        
        # B. VOLUME ANALYSIS FEATURES
        market_data['volume_change'] = market_data.groupby('symbol')['volume'].pct_change()
        market_data['volume_ma_5'] = market_data.groupby('symbol')['volume'].rolling(5).mean().reset_index(level=0, drop=True)
        market_data['volume_ma_20'] = market_data.groupby('symbol')['volume'].rolling(20).mean().reset_index(level=0, drop=True)
        market_data['volume_ratio_5'] = market_data['volume'] / market_data['volume_ma_5']
        market_data['volume_ratio_20'] = market_data['volume'] / market_data['volume_ma_20']
        
        # Value-based volume features
        market_data['value_per_trade'] = market_data['value'] / (market_data['trades'] + 1)
        market_data['avg_trade_size'] = market_data['volume'] / (market_data['trades'] + 1)
        
        # C. MOVING AVERAGES
        for window in [5, 10, 20, 50]:
            market_data[f'ma_{window}'] = market_data.groupby('symbol')['close'].rolling(window).mean().reset_index(level=0, drop=True)
            market_data[f'ma_{window}_ratio'] = market_data['close'] / market_data[f'ma_{window}']
        
        # D. BOLLINGER BANDS FEATURES (you have bb_upper, bb_middle, bb_lower)
        market_data['bb_width'] = (market_data['bb_upper'] - market_data['bb_lower']) / market_data['bb_middle']
        market_data['bb_position'] = (market_data['close'] - market_data['bb_lower']) / (market_data['bb_upper'] - market_data['bb_lower'])
        market_data['near_bb_upper'] = (market_data['close'] > market_data['bb_upper'] * 0.95).astype(int)
        market_data['near_bb_lower'] = (market_data['close'] < market_data['bb_lower'] * 1.05).astype(int)
        market_data['bb_squeeze'] = (market_data['bb_width'] < market_data['bb_width'].rolling(20).mean()).astype(int)
        
        # E. MACD FEATURES (you have macd, macd_signal, macd_hist)
        market_data['macd_diff'] = market_data['macd'] - market_data['macd_signal']
        market_data['macd_cross'] = ((market_data['macd'] > market_data['macd_signal']) & 
                                    (market_data.groupby('symbol')['macd'].shift(1) <= 
                                     market_data.groupby('symbol')['macd_signal'].shift(1))).astype(int)
        market_data['macd_trend'] = market_data['macd'].diff()
        market_data['macd_hist_change'] = market_data['macd_hist'].diff()
        
        # F. RSI FEATURES
        market_data['rsi_level'] = pd.cut(market_data['rsi'], 
                                         bins=[0, 30, 40, 60, 70, 100],
                                         labels=['oversold', 'weak_oversold', 'neutral', 'weak_overbought', 'overbought'])
        
        # One-hot encode RSI levels
        rsi_dummies = pd.get_dummies(market_data['rsi_level'], prefix='rsi')
        market_data = pd.concat([market_data, rsi_dummies], axis=1)
        
        # G. CANDLESTICK PATTERNS (you have Hammer, BullishEngulfing, etc.)
        candlestick_cols = ['Hammer', 'BullishEngulfing', 'MorningStar', 'Doji', 
                           'PiercingLine', 'ThreeWhiteSoldiers']
        
        # Count bullish patterns
        market_data['bullish_patterns'] = market_data[candlestick_cols].sum(axis=1)
        market_data['any_bullish_pattern'] = (market_data['bullish_patterns'] > 0).astype(int)
        
        # H. ATR FEATURES
        market_data['atr_pct'] = market_data['atr'] / market_data['close'] * 100
        market_data['atr_ratio'] = market_data['atr'] / market_data['atr'].rolling(20).mean()
        market_data['high_low_atr'] = (market_data['high'] - market_data['low']) / market_data['atr']
        
        # I. MARKET CAP FEATURES
        market_data['marketCap_log'] = np.log1p(market_data['marketCap'])
        market_data['cap_category'] = pd.qcut(market_data['marketCap'], 4, labels=['small', 'mid', 'large', 'xlarge'])
        cap_dummies = pd.get_dummies(market_data['cap_category'], prefix='cap')
        market_data = pd.concat([market_data, cap_dummies], axis=1)
        
        # J. ZIGZAG FEATURES
        market_data['zigzag_change'] = market_data['zigzag'].diff()
        market_data['zigzag_trend'] = np.where(market_data['zigzag_change'] > 0, 1, 
                                              np.where(market_data['zigzag_change'] < 0, -1, 0))
        
        # K. MOMENTUM FEATURES
        for window in [5, 10, 20]:
            market_data[f'momentum_{window}'] = market_data['close'] - market_data.groupby('symbol')['close'].shift(window)
            market_data[f'roc_{window}'] = market_data.groupby('symbol')['close'].pct_change(window) * 100
        
        # L. VOLATILITY FEATURES
        market_data['volatility_5'] = market_data.groupby('symbol')['returns'].rolling(5).std().reset_index(level=0, drop=True)
        market_data['volatility_20'] = market_data.groupby('symbol')['returns'].rolling(20).std().reset_index(level=0, drop=True)
        market_data['volatility_ratio'] = market_data['volatility_5'] / market_data['volatility_20']
        
        # M. PRICE POSITION FEATURES
        market_data['price_position'] = (market_data['close'] - market_data['low']) / (market_data['high'] - market_data['low'] + 1e-8)
        market_data['gap_up'] = (market_data['open'] > market_data.groupby('symbol')['close'].shift(1)).astype(int)
        market_data['gap_down'] = (market_data['open'] < market_data.groupby('symbol')['close'].shift(1)).astype(int)
        
        # Ensure date columns are in correct format
        market_data['date'] = pd.to_datetime(market_data['date'])
        trade_data['date'] = pd.to_datetime(trade_data['date'])
        
        # 2. CREATE FUTURE TARGETS (PREDICT 3 DAYS AHEAD)
        print("   Creating future prediction targets...")
        
        # Shift trade signals 3 days forward
        trade_data_future = trade_data.copy()
        trade_data_future['date'] = trade_data_future['date'] + pd.Timedelta(days=3)
        
        # Merge with market data
        merged = pd.merge(market_data, trade_data_future[['symbol', 'date', 'buy', 'RRR']], 
                         on=['symbol', 'date'], how='left')
        
        # Create target variable
        merged['target'] = merged['buy'].notna().astype(int)
        
        # Create multi-class target based on RRR
        merged['target_rrr'] = 0  # default: no signal
        merged.loc[merged['buy'].notna() & (merged['RRR'] > 1.5), 'target_rrr'] = 2  # strong buy
        merged.loc[merged['buy'].notna() & (merged['RRR'] <= 1.5) & (merged['RRR'] > 0), 'target_rrr'] = 1  # moderate buy
        
        buy_signals = merged['target'].sum()
        total_samples = len(merged)
        print(f"   FUTURE Buy signals: {buy_signals} out of {total_samples} samples ({buy_signals/total_samples:.2%})")
        
        # 3. SELECT ALL AVAILABLE FEATURES
        print("   Selecting features...")
        
        # Define feature categories
        price_features = ['open', 'close', 'high', 'low', 'returns', 'returns_5', 'returns_10',
                         'high_low_range', 'close_to_open', 'body_size', 'price_position',
                         'gap_up', 'gap_down']
        
        volume_features = ['volume', 'volume_change', 'volume_ma_5', 'volume_ma_20',
                          'volume_ratio_5', 'volume_ratio_20', 'value', 'trades',
                          'value_per_trade', 'avg_trade_size', 'change']
        
        ma_features = [f'ma_{w}' for w in [5, 10, 20, 50]] + [f'ma_{w}_ratio' for w in [5, 10, 20, 50]]
        
        bb_features = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
                      'near_bb_upper', 'near_bb_lower', 'bb_squeeze']
        
        macd_features = ['macd', 'macd_signal', 'macd_hist', 'macd_diff', 'macd_cross',
                        'macd_trend', 'macd_hist_change']
        
        rsi_features = ['rsi'] + [col for col in merged.columns if 'rsi_' in col]
        
        pattern_features = candlestick_cols + ['bullish_patterns', 'any_bullish_pattern']
        
        atr_features = ['atr', 'atr_pct', 'atr_ratio', 'high_low_atr']
        
        market_cap_features = ['marketCap', 'marketCap_log'] + [col for col in merged.columns if 'cap_' in col]
        
        zigzag_features = ['zigzag', 'zigzag_change', 'zigzag_trend']
        
        momentum_features = [f'momentum_{w}' for w in [5, 10, 20]] + [f'roc_{w}' for w in [5, 10, 20]]
        
        volatility_features = ['volatility_5', 'volatility_20', 'volatility_ratio']
        
        # Combine all features
        all_features = (price_features + volume_features + ma_features + bb_features +
                       macd_features + rsi_features + pattern_features + atr_features +
                       market_cap_features + zigzag_features + momentum_features + volatility_features)
        
        # Filter to only available features
        self.feature_names = [col for col in all_features if col in merged.columns]
        
        # Add symbol encoding
        if 'symbol' in merged.columns:
            merged['symbol_encoded'] = self.label_encoder.fit_transform(merged['symbol'])
            self.feature_names.append('symbol_encoded')
        
        print(f"   Selected {len(self.feature_names)} features for modeling")
        
        # 4. HANDLE MISSING VALUES
        print("   Handling missing values...")
        before_drop = merged.shape[0]
        
        # Fill NaN with forward/backward fill for time-series
        for col in self.feature_names:
            if col != 'symbol_encoded':
                merged[col] = merged.groupby('symbol')[col].ffill().bfill()
        
        # Drop remaining NaN
        merged = merged.dropna(subset=self.feature_names + ['target'])
        after_drop = merged.shape[0]
        
        print(f"   Dropped {before_drop - after_drop} rows with NaN values")
        print(f"   Final data shape: {merged.shape}")
        
        # 5. DATA ANALYSIS
        print(f"\n📊 Data Analysis:")
        print(f"   Total samples: {len(merged)}")
        print(f"   Buy signals: {merged['target'].sum()} ({merged['target'].mean()*100:.2f}%)")
        print(f"   No signals: {len(merged) - merged['target'].sum()} ({(1 - merged['target'].mean())*100:.2f}%)")
        
        if 'target_rrr' in merged.columns:
            rrr_counts = merged['target_rrr'].value_counts().sort_index()
            print(f"   Signal strength distribution:")
            for val, count in rrr_counts.items():
                label = {0: 'No signal', 1: 'Moderate buy', 2: 'Strong buy'}.get(val, val)
                print(f"     {label}: {count} ({count/len(merged)*100:.2f}%)")
        
        return merged[self.feature_names], merged['target'], merged.get('target_rrr', None)
    
    def train_model(self, X, y, y_rrr=None):
        """ট্রেইন XGBoost model with all features"""
        print("\n🤖 Training XGBoost model with all available features...")
        
        if len(X) < 100:
            print(f"   ⚠️ Only {len(X)} samples available")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Class distribution - Train: {np.bincount(y_train)}")
        print(f"   Class distribution - Test: {np.bincount(y_test)}")
        
        # Calculate class weights for imbalance
        if y_train.sum() > 0:
            scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
            print(f"   Class weight (scale_pos_weight): {scale_pos_weight:.2f}")
        else:
            scale_pos_weight = 1.0
            print("   ⚠️ No positive samples!")
            return None
        
        # Train XGBoost model
        print("   Training model...")
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='aucpr',
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False
        )
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=0
        )
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Print results
        print(f"\n📈 Model Performance:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        print(f"   AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        print("\n📊 Confusion Matrix:")
        print(metrics['confusion_matrix'])
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🔝 Top 25 Most Important Features:")
            for idx, row in importance_df.head(25).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
            
            # Save feature importance
            self.save_feature_importance(importance_df)
        
        return metrics
    
    def save_model(self, metrics):
        """সেভ মডেল এবং মেট্রিক্স"""
        if self.model is None:
            print("   ⚠️ No model to save")
            return
        
        model_dir = "csv/models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "xgboost_model.pkl")
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'feature_importances': dict(zip(self.feature_names, self.model.feature_importances_)),
            'metrics': metrics
        }, model_path)
        
        print(f"\n✅ Model saved to: {model_path}")
        
        # Save metrics report
        metrics_path = os.path.join(model_dir, "model_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("XGBoost Trading Model - Performance Metrics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {metrics['f1']:.4f}\n")
            f.write(f"AUC-ROC:   {metrics['auc_roc']:.4f}\n")
            f.write(f"\nTotal Features: {len(self.feature_names)}\n")
            f.write(f"Training Samples: {self.model.n_features_in_}\n")
        
        print(f"✅ Metrics saved to: {metrics_path}")
    
    def save_feature_importance(self, importance_df):
        """Save feature importance analysis"""
        model_dir = "csv/models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save as CSV
        csv_path = os.path.join(model_dir, "feature_importance.csv")
        importance_df.to_csv(csv_path, index=False)
        
        # Group by feature category
        categories = {
            'price': ['open', 'close', 'high', 'low', 'returns', 'price_position'],
            'volume': ['volume', 'value', 'trades', 'volume_ratio'],
            'technical': ['rsi', 'macd', 'bb_', 'atr', 'zigzag'],
            'patterns': ['Hammer', 'BullishEngulfing', 'Doji', 'bullish_patterns'],
            'momentum': ['momentum', 'roc', 'ma_ratio'],
            'volatility': ['volatility', 'bb_width', 'high_low_range']
        }
        
        # Calculate category importance
        category_importance = {}
        for category, keywords in categories.items():
            cat_features = [f for f in importance_df['feature'] if any(k in f for k in keywords)]
            cat_importance = importance_df[importance_df['feature'].isin(cat_features)]['importance'].sum()
            category_importance[category] = cat_importance
        
        # Save category analysis
        cat_path = os.path.join(model_dir, "feature_categories.txt")
        with open(cat_path, 'w') as f:
            f.write("Feature Importance by Category\n")
            f.write("=" * 50 + "\n\n")
            for category, importance in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{category:15} {importance:.4f} ({importance/sum(category_importance.values())*100:.1f}%)\n")
        
        print(f"✅ Feature analysis saved to: {csv_path}")

def main():
    """মেইন ফাংশন"""
    print("=" * 70)
    print("XGBOOST TRADING MODEL - USING ALL mongodb.csv FEATURES")
    print("=" * 70)
    
    try:
        # Create model
        model = XGBoostTradingModel()
        
        # Load and prepare data
        print("\n📥 Loading and preparing data with ALL features...")
        X, y, y_rrr = model.load_and_prepare_data()
        
        if len(X) == 0:
            print("❌ No data available for training")
            return
        
        # Train model
        metrics = model.train_model(X, y, y_rrr)
        
        if metrics and metrics['f1'] > 0.1:  # Minimum F1 threshold
            model.save_model(metrics)
            
            print(f"\n" + "=" * 70)
            print("✅ MODEL TRAINING SUCCESSFUL!")
            print("=" * 70)
            print(f"\n📊 Key Insights:")
            print(f"   • Model uses {len(model.feature_names)} features from mongodb.csv")
            print(f"   • Best metric: F1-Score = {metrics['f1']:.4f}")
            print(f"   • Recall: {metrics['recall']:.4f} (can detect buy signals)")
            print(f"   • Check 'csv/models/' folder for detailed analysis")
            print("=" * 70)
            
            # Generate sample predictions
            print("\n🎯 Sample Predictions (first 10 test samples):")
            sample_indices = np.random.choice(len(X), min(10, len(X)), replace=False)
            sample_X = X.iloc[sample_indices]
            
            if hasattr(model.model, 'predict_proba'):
                probabilities = model.model.predict_proba(sample_X)[:, 1]
                predictions = (probabilities > 0.3).astype(int)
                
                sample_results = pd.DataFrame({
                    'Probability': probabilities,
                    'Prediction': predictions,
                    'Actual': y.iloc[sample_indices].values if len(y) > max(sample_indices) else [0]*len(sample_indices)
                })
                print(sample_results.to_string(index=False))
        else:
            print(f"\n⚠️ Model performance insufficient (F1={metrics['f1']:.4f} if available)")
            print("   Consider:")
            print("   1. Getting more trade data")
            print("   2. Adjusting the prediction horizon")
            print("   3. Feature selection/engineering")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Script completed")

if __name__ == "__main__":
    main()