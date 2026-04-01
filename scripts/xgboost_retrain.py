# xgboost_retrain.py - Complete Updated Version
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIGURATION
# =========================
DATA_PATH = './csv/mongodb.csv'
MODEL_DIR = './csv/xgboost/'
os.makedirs(MODEL_DIR, exist_ok=True)

# Schedule tracking files
LAST_RETRAIN_FILE = './csv/last_retrain.txt'
LAST_FINETUNE_FILE = './csv/last_finetune.txt'
LAST_TUNING_FILE = './csv/last_tuning.txt'

# Schedule intervals (in days)
DAILY_FINETUNE = 1      # প্রতিদিন fine-tune
WEEKLY_RETRAIN = 7      # সাপ্তাহিক full retrain
MONTHLY_TUNING = 30     # মাসিক hyperparameter tuning

# Minimum samples per symbol
MIN_SAMPLES_PER_SYMBOL = 30
MIN_TARGET_RATIO = 0.15
MAX_TARGET_RATIO = 0.85

# Base model parameters
BASE_MODEL_PARAMS = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'random_state': 42,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}

# Hyperparameter tuning grid
TUNING_GRID = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.03, 0.05, 0.07],
    'n_estimators': [150, 200, 250],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1, 2]
}

print("="*80)
print("🤖 XGBOOST AUTOMATED LEARNING SYSTEM")
print("="*80)
print(f"📅 Daily Fine-tune | 📆 Weekly Retrain | 📅 Monthly Hyperparameter Tuning")
print(f"📊 Min Samples: {MIN_SAMPLES_PER_SYMBOL}")
print("="*80)

# =========================
# HELPER FUNCTIONS
# =========================

def check_last_run(file_path, days_interval):
    """Check if enough days have passed since last run"""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                last_date = datetime.strptime(f.read().strip(), '%Y-%m-%d')
        except:
            last_date = datetime(2000, 1, 1)
    else:
        last_date = datetime(2000, 1, 1)
    
    today = datetime.today()
    days_since = (today - last_date).days
    needed = days_since >= days_interval
    
    return last_date, today, days_since, needed

def update_last_run(file_path, date):
    """Update last run date"""
    with open(file_path, 'w') as f:
        f.write(date.strftime('%Y-%m-%d'))

def load_data():
    """Load and prepare data"""
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    return df

def engineer_features(df):
    """Add engineered features"""
    print("   🔧 Adding engineered features...")
    
    # Price changes
    df['return'] = df.groupby('symbol')['close'].pct_change()
    df['return_3d'] = df.groupby('symbol')['close'].pct_change(3)
    df['return_5d'] = df.groupby('symbol')['close'].pct_change(5)
    
    # Volatility
    df['volatility'] = (df['high'] - df['low']) / df['close']
    df['volatility_5d'] = df.groupby('symbol')['volatility'].transform(lambda x: x.rolling(5).mean())
    
    # Volume features
    df['volume_ma'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['volume_spike'] = (df['volume'] > df['volume_ma'] * 1.5).astype(int)
    
    # Support/Resistance
    df['resistance_20d'] = df.groupby('symbol')['high'].transform(lambda x: x.rolling(20).max())
    df['support_20d'] = df.groupby('symbol')['low'].transform(lambda x: x.rolling(20).min())
    df['dist_to_resistance'] = (df['resistance_20d'] - df['close']) / df['close'] * 100
    df['dist_to_support'] = (df['close'] - df['support_20d']) / df['close'] * 100
    
    # MACD signals
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
    
    # RSI signals
    if 'rsi' in df.columns:
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_cross_above_30'] = ((df['rsi'] >= 30) & (df['rsi'].shift(1) < 30)).astype(int)
        df['rsi_cross_below_70'] = ((df['rsi'] <= 70) & (df['rsi'].shift(1) > 70)).astype(int)
        df['rsi_bullish_divergence'] = ((df['low'] <= df['low'].shift(1)) & 
                                         (df['rsi'] > df['rsi'].shift(1))).astype(int)
    
    # Target
    df['future_return'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
    df['target'] = (df['future_return'] > 0.02).astype(int)
    
    # Drop NaN target only
    initial_len = len(df)
    df = df.dropna(subset=['target'])
    print(f"   📊 Rows after target drop: {len(df):,} (dropped {initial_len - len(df):,})")
    
    return df

def get_features(df):
    """Get list of available features"""
    feature_cols = [
        'open', 'high', 'low', 'volume', 'value', 'trades', 'change', 'marketCap',
        'bb_upper', 'bb_middle', 'bb_lower', 'macd', 'macd_signal', 'macd_hist',
        'rsi', 'atr', 'zigzag', 'ema_200',
        'return', 'return_3d', 'return_5d',
        'volatility', 'volatility_5d',
        'volume_ratio', 'volume_spike',
        'dist_to_resistance', 'dist_to_support',
        'macd_cross_up', 'macd_cross_down',
        'rsi_oversold', 'rsi_overbought', 'rsi_cross_above_30', 'rsi_cross_below_70',
        'rsi_bullish_divergence'
    ]
    
    return [f for f in feature_cols if f in df.columns]

def hyperparameter_tuning(X_train, y_train):
    """Monthly hyperparameter tuning"""
    print("   🔧 Running hyperparameter tuning...")
    
    from sklearn.model_selection import GridSearchCV
    
    # Use smaller subset for faster tuning
    if len(X_train) > 5000:
        X_sample = X_train.sample(n=5000, random_state=42)
        y_sample = y_train.loc[X_sample.index]
    else:
        X_sample = X_train
        y_sample = y_train
    
    grid_search = GridSearchCV(
        xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
        TUNING_GRID,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_sample, y_sample)
    
    print(f"   ✅ Best params: {grid_search.best_params_}")
    print(f"   📊 Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_

# =========================
# ADAPTIVE MODEL CLASS
# =========================

class AdaptiveXGBoost:
    """Adaptive XGBoost with daily fine-tuning, weekly retrain, monthly tuning"""
    
    def __init__(self, symbol, base_params):
        self.symbol = symbol
        self.base_params = base_params.copy()
        self.model = None
        self.mistake_log = []
        self.performance_history = []
        self.model_path = os.path.join(MODEL_DIR, f'xgb_model_{symbol}.joblib')
        
    def load_model(self):
        """Load existing model if exists"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False
    
    def save_model(self):
        """Save model to disk"""
        joblib.dump(self.model, self.model_path)
        
    def train_fresh(self, X_train, y_train, X_val=None, y_val=None):
        """Fresh training"""
        print(f"   🆕 Fresh training...")
        self.model = xgb.XGBClassifier(**self.base_params)
        self.model.fit(X_train, y_train)
        
        if X_val is not None:
            self._evaluate(X_val, y_val)
        
        self.save_model()
        
    def fine_tune(self, X_train, y_train, X_val=None, y_val=None):
        """Daily fine-tuning on new data"""
        print(f"   🔧 Fine-tuning on new data...")
        
        if self.model is None:
            self.train_fresh(X_train, y_train, X_val, y_val)
            return
        
        # Continue training on new data
        self.model.fit(X_train, y_train, xgb_model=self.model)
        
        if X_val is not None:
            self._evaluate(X_val, y_val)
        
        self.save_model()
        
    def retrain(self, X_train, y_train, X_val=None, y_val=None):
        """Weekly full retrain"""
        print(f"   🔄 Full retraining...")
        self.model = xgb.XGBClassifier(**self.base_params)
        self.model.fit(X_train, y_train)
        
        if X_val is not None:
            self._evaluate(X_val, y_val)
        
        self.save_model()
    
    def retrain_with_best_params(self, X_train, y_train, best_params, X_val=None, y_val=None):
        """Retrain with optimized hyperparameters"""
        print(f"   🎯 Retraining with optimized parameters...")
        
        # Update base params with best params
        for key, value in best_params.items():
            self.base_params[key] = value
        
        self.model = xgb.XGBClassifier(**self.base_params)
        self.model.fit(X_train, y_train)
        
        if X_val is not None:
            self._evaluate(X_val, y_val)
        
        self.save_model()
        
    def _evaluate(self, X_test, y_test):
        """Evaluate model and track mistakes"""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # Track mistakes for hard negative mining
        mistakes = []
        for i, (true, pred, proba) in enumerate(zip(y_test, y_pred, y_proba)):
            if true != pred:
                mistakes.append({
                    'true': true,
                    'pred': pred,
                    'confidence': proba,
                    'date': datetime.now()
                })
        
        self.mistake_log.extend(mistakes)
        self.performance_history.append({
            'date': datetime.now(), 
            'accuracy': acc, 
            'auc': auc
        })
        
        print(f"   ✅ Acc: {acc:.2%}, AUC: {auc:.2%}, Mistakes: {len(mistakes)}")
        
        return acc, auc, mistakes
    
    def learn_from_mistakes(self, X_mistakes, y_mistakes):
        """Hard negative mining - learn from past mistakes"""
        if len(self.mistake_log) > 50 and X_mistakes is not None and len(X_mistakes) > 0:
            print(f"   📚 Learning from {len(self.mistake_log)} past mistakes...")
            
            # Higher weight for mistakes with high confidence
            sample_weights = []
            for m in self.mistake_log[-100:]:
                weight = 2.0 if m['confidence'] > 0.7 else 1.0
                sample_weights.append(weight)
            
            self.model.fit(X_mistakes, y_mistakes, 
                          sample_weight=sample_weights[:len(X_mistakes)],
                          xgb_model=self.model)
            
            self.save_model()
            print(f"   ✅ Learned from mistakes!")

# =========================
# MAIN EXECUTION
# =========================

def main():
    # Check schedules
    _, today, _, retrain_needed = check_last_run(LAST_RETRAIN_FILE, WEEKLY_RETRAIN)
    _, _, _, finetune_needed = check_last_run(LAST_FINETUNE_FILE, DAILY_FINETUNE)
    _, _, _, tuning_needed = check_last_run(LAST_TUNING_FILE, MONTHLY_TUNING)
    
    print(f"\n📅 Schedule Check:")
    print(f"   Daily Fine-tune: {'✅ NEEDED' if finetune_needed else '❌ NOT NEEDED'}")
    print(f"   Weekly Retrain:  {'✅ NEEDED' if retrain_needed else '❌ NOT NEEDED'}")
    print(f"   Monthly Tuning:  {'✅ NEEDED' if tuning_needed else '❌ NOT NEEDED'}")
    
    # Load data
    print("\n📂 Loading data...")
    df = load_data()
    print(f"   Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols")
    
    # Engineer features
    print("\n🔧 Engineering features...")
    df = engineer_features(df)
    features = get_features(df)
    print(f"   Using {len(features)} features")
    
    # Global best params for monthly tuning
    global_best_params = None
    
    if tuning_needed:
        print("\n" + "="*80)
        print("🎯 MONTHLY HYPERPARAMETER TUNING")
        print("="*80)
        
        # Use all symbols data for tuning
        all_data = df[features].copy()
        all_target = df['target'].copy()
        
        # Handle missing values
        for col in all_data.columns:
            if all_data[col].isnull().any():
                all_data[col] = all_data[col].fillna(all_data[col].median())
        
        if len(all_data) > 1000:
            X_tune, X_tune_val, y_tune, y_tune_val = train_test_split(
                all_data, all_target, test_size=0.2, random_state=42
            )
            
            global_best_params = hyperparameter_tuning(X_tune, y_tune)
            
            # Update base params globally
            for key, value in global_best_params.items():
                BASE_MODEL_PARAMS[key] = value
                print(f"   ✅ Updated {key}: {value}")
    
    # Process each symbol
    print("\n" + "="*80)
    print("🤖 TRAINING MODELS BY SYMBOL")
    print("="*80)
    
    results = []
    training_stats = []
    total_trained = 0
    total_loaded = 0
    total_skipped = 0
    
    for symbol, group in df.groupby('symbol'):
        if len(group) < MIN_SAMPLES_PER_SYMBOL:
            total_skipped += 1
            continue
        
        print(f"\n{'─'*50}")
        print(f"🔹 {symbol} ({len(group)} rows)")
        
        # Prepare data
        group = group.sort_values('date')
        X = group[features].copy()
        y = group['target'].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        
        target_ratio = y.mean()
        if target_ratio < MIN_TARGET_RATIO or target_ratio > MAX_TARGET_RATIO:
            print(f"   ⚠️ Skipped (target ratio: {target_ratio:.2%})")
            total_skipped += 1
            continue
        
        # Split data
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Create adaptive model
        adaptive_model = AdaptiveXGBoost(symbol, BASE_MODEL_PARAMS)
        
        # Check if model exists
        model_exists = adaptive_model.load_model()
        
        # Decide training strategy
        if tuning_needed and global_best_params:
            adaptive_model.retrain_with_best_params(X_train, y_train, global_best_params, X_test, y_test)
            total_trained += 1
            
        elif retrain_needed:
            adaptive_model.retrain(X_train, y_train, X_test, y_test)
            total_trained += 1
            
        elif finetune_needed and model_exists:
            adaptive_model.fine_tune(X_train, y_train, X_test, y_test)
            total_trained += 1
            
        elif not model_exists:
            adaptive_model.train_fresh(X_train, y_train, X_test, y_test)
            total_trained += 1
            
        else:
            print(f"   📂 Using existing model")
            total_loaded += 1
        
        # Learn from mistakes (hard negative mining)
        if len(adaptive_model.mistake_log) > 50:
            y_pred_test = adaptive_model.model.predict(X_test)
            mistake_indices = [i for i in range(len(y_test)) if y_test.iloc[i] != y_pred_test[i]]
            if mistake_indices:
                X_mistakes = X_test.iloc[mistake_indices]
                y_mistakes = y_test.iloc[mistake_indices]
                adaptive_model.learn_from_mistakes(X_mistakes, y_mistakes)
        
        # Predict on all data
        X_full = X.copy()
        for col in X_full.columns:
            if X_full[col].isnull().any():
                X_full[col] = X_full[col].fillna(X_full[col].median())
        
        group['confidence_score'] = adaptive_model.model.predict_proba(X_full)[:, 1] * 100
        group['prediction'] = (group['confidence_score'] > 50).astype(int)
        group['signal_strength'] = pd.cut(
            group['confidence_score'],
            bins=[0, 30, 50, 70, 100],
            labels=['Weak', 'Moderate', 'Strong', 'Very Strong']
        )
        
        # Save results
        results.append(group[['symbol', 'date', 'close', 'confidence_score', 'prediction', 'signal_strength']])
        
        # Track stats
        last_acc = adaptive_model.performance_history[-1]['accuracy'] if adaptive_model.performance_history else 0
        training_stats.append({
            'symbol': symbol,
            'samples': len(group),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'accuracy': last_acc,
            'target_ratio': target_ratio
        })
    
    # Save all predictions
    if results:
        final_df = pd.concat(results, ignore_index=True)
        output_path = './csv/xgb_confidence.csv'
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ Predictions saved: {output_path}")
        print(f"   Total rows: {len(final_df):,}")
    
    # Save training summary
    if training_stats:
        stats_df = pd.DataFrame(training_stats)
        stats_path = os.path.join(MODEL_DIR, 'training_summary.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"✅ Training summary saved: {stats_path}")
        
        print(f"\n📊 Training Summary:")
        print(f"   Average accuracy: {stats_df['accuracy'].mean():.2%}")
        print(f"   Best accuracy: {stats_df['accuracy'].max():.2%}")
        print(f"   Worst accuracy: {stats_df['accuracy'].min():.2%}")
    
    # Update last run dates
    if finetune_needed:
        update_last_run(LAST_FINETUNE_FILE, datetime.today())
        print("✅ Daily fine-tune date updated")
    
    if retrain_needed:
        update_last_run(LAST_RETRAIN_FILE, datetime.today())
        print("✅ Weekly retrain date updated")
    
    if tuning_needed:
        update_last_run(LAST_TUNING_FILE, datetime.today())
        print("✅ Monthly tuning date updated")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ XGBOOST AUTOMATED LEARNING COMPLETE!")
    print("="*80)
    print(f"📊 Models trained: {total_trained}")
    print(f"📂 Models loaded: {total_loaded}")
    print(f"⚠️ Symbols skipped: {total_skipped}")
    print(f"📁 Models saved in: {MODEL_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()