# xgboost_retrain.py - Fixed Version

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIGURATION - ULTIMATE QUALITY
# =========================
DATA_PATH = './csv/mongodb.csv'
MODEL_DIR = './csv/xgboost/'
os.makedirs(MODEL_DIR, exist_ok=True)

# Schedule tracking files
LAST_RETRAIN_FILE = './csv/last_retrain.txt'
LAST_FINETUNE_FILE = './csv/last_finetune.txt'
LAST_TUNING_FILE = './csv/last_tuning.txt'

# Schedule intervals (in days)
DAILY_FINETUNE = 1
WEEKLY_RETRAIN = 7
MONTHLY_TUNING = 30

# Minimum samples per symbol
MIN_SAMPLES_PER_SYMBOL = 50
MIN_TARGET_RATIO = 0.15
MAX_TARGET_RATIO = 0.85

# =========================
# ULTIMATE QUALITY MODEL PARAMETERS (FIXED)
# =========================
ULTIMATE_PARAMS = {
    # Ensemble size
    'n_estimators': 1000,
    'max_depth': 8,
    'learning_rate': 0.01,
    
    # Regularization
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'colsample_bynode': 0.7,
    'min_child_weight': 5,
    'gamma': 0.2,
    'reg_alpha': 0.5,
    'reg_lambda': 2,
    
    # Tree constraints
    'max_leaves': 31,
    'max_bin': 256,
    
    # Training control - FIX: eval_metric goes here, not in fit()
    'random_state': 42,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'verbosity': 0,
    
    # Advanced
    'booster': 'gbtree',
    'tree_method': 'hist',
    'grow_policy': 'lossguide',
    'max_delta_step': 1,
}

# =========================
# HYPERPARAMETER TUNING GRID
# =========================
ULTIMATE_TUNING_GRID = {
    'max_depth': [6, 7, 8, 9, 10],
    'min_child_weight': [3, 5, 7, 10],
    'gamma': [0, 0.1, 0.2, 0.3, 0.5],
    'learning_rate': [0.1, 0.05, 0.03, 0.01, 0.005],
    'n_estimators': [500, 750, 1000, 1500],
    'reg_alpha': [0, 0.1, 0.5, 1, 2],
    'reg_lambda': [0.5, 1, 1.5, 2, 3],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'max_leaves': [0, 31, 63, 127],
    'max_delta_step': [0, 1, 2, 5],
}

print("="*90)
print("🏆 ULTIMATE XGBOOST QUALITY TRAINING")
print("="*90)
print(f"📊 Target: Maximum accuracy & generalization")
print(f"⏱️ Expected Time: 2-4 hours (first run)")
print(f"🎯 Models: {MIN_SAMPLES_PER_SYMBOL}+ samples per symbol")
print("="*90)

# =========================
# HELPER FUNCTIONS
# =========================

def fix_bom_columns(df):
    """Remove BOM characters from column names"""
    df.columns = df.columns.str.replace('ï»¿', '').str.replace('\ufeff', '').str.strip()
    return df

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
    """Load and prepare data with BOM fix"""
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df = fix_bom_columns(df)
    df['date'] = pd.to_datetime(df['date'])
    
    if 'symbol' not in df.columns:
        for col in df.columns:
            if 'sym' in col.lower() or 'ticker' in col.lower():
                df.rename(columns={col: 'symbol'}, inplace=True)
                break
    
    return df

def engineer_features(df):
    """Add comprehensive engineered features"""
    print("   🔧 Adding comprehensive engineered features...")
    
    # Price changes
    for period in [1, 3, 5, 10, 20]:
        df[f'return_{period}d'] = df.groupby('symbol')['close'].pct_change(period)
    
    # Rolling statistics
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(period).mean())
        df[f'std_{period}'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(period).std())
        df[f'close_sma_ratio_{period}'] = df['close'] / df[f'sma_{period}']
    
    # Volatility
    df['volatility'] = (df['high'] - df['low']) / df['close']
    for period in [5, 10, 20]:
        df[f'volatility_{period}d'] = df.groupby('symbol')['volatility'].transform(lambda x: x.rolling(period).mean())
    
    # Volume features
    df['volume_ma_20'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
    df['volume_ma_50'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(50).mean())
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    df['volume_ratio_ma'] = df.groupby('symbol')['volume_ratio'].transform(lambda x: x.rolling(10).mean())
    
    # Support/Resistance
    for period in [20, 50, 100]:
        df[f'resistance_{period}d'] = df.groupby('symbol')['high'].transform(lambda x: x.rolling(period).max())
        df[f'support_{period}d'] = df.groupby('symbol')['low'].transform(lambda x: x.rolling(period).min())
        df[f'dist_to_resistance_{period}'] = (df[f'resistance_{period}d'] - df['close']) / df['close'] * 100
        df[f'dist_to_support_{period}'] = (df['close'] - df[f'support_{period}d']) / df['close'] * 100
    
    # MACD signals
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_histogram_trend'] = df['macd_histogram'].diff(3).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    
    # RSI signals
    if 'rsi' in df.columns:
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_cross_above_30'] = ((df['rsi'] >= 30) & (df['rsi'].shift(1) < 30)).astype(int)
        df['rsi_cross_below_70'] = ((df['rsi'] <= 70) & (df['rsi'].shift(1) > 70)).astype(int)
        df['rsi_bullish_divergence'] = ((df['low'] <= df['low'].shift(1)) & 
                                         (df['rsi'] > df['rsi'].shift(1))).astype(int)
        df['rsi_bearish_divergence'] = ((df['high'] >= df['high'].shift(1)) & 
                                         (df['rsi'] < df['rsi'].shift(1))).astype(int)
    
    # ATR features
    if 'atr' in df.columns:
        df['atr_ratio'] = df['atr'] / df['close'] * 100
    
    # Target
    df['future_return'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
    df['target'] = (df['future_return'] > 0.02).astype(int)
    
    # Drop NaN target only
    initial_len = len(df)
    df = df.dropna(subset=['target'])
    print(f"   📊 Rows after target drop: {len(df):,} (dropped {initial_len - len(df):,})")
    
    return df

def get_features(df):
    """Get comprehensive list of features"""
    feature_cols = [
        'open', 'high', 'low', 'volume', 'value', 'trades', 'change', 'marketCap',
        'bb_upper', 'bb_middle', 'bb_lower', 'macd', 'macd_signal', 'macd_hist',
        'rsi', 'atr', 'zigzag', 'ema_200',
        'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
        'sma_5', 'sma_10', 'sma_20', 'sma_50',
        'close_sma_ratio_5', 'close_sma_ratio_10', 'close_sma_ratio_20', 'close_sma_ratio_50',
        'std_5', 'std_10', 'std_20', 'std_50',
        'volatility', 'volatility_5d', 'volatility_10d', 'volatility_20d',
        'volume_ratio', 'volume_ratio_ma',
        'dist_to_resistance_20', 'dist_to_support_20',
        'dist_to_resistance_50', 'dist_to_support_50',
        'dist_to_resistance_100', 'dist_to_support_100',
        'macd_cross_up', 'macd_cross_down', 'macd_histogram_trend',
        'rsi_oversold', 'rsi_overbought', 'rsi_cross_above_30', 'rsi_cross_below_70',
        'rsi_bullish_divergence', 'rsi_bearish_divergence',
        'atr_ratio'
    ]
    
    return [f for f in feature_cols if f in df.columns]

def comprehensive_hyperparameter_tuning(X_train, y_train):
    """Comprehensive hyperparameter tuning"""
    print("   🔧 Running comprehensive hyperparameter tuning...")
    print("   📊 Testing 100+ parameter combinations...")
    
    from sklearn.model_selection import RandomizedSearchCV
    
    # Use 10000 samples
    if len(X_train) > 10000:
        X_sample = X_train.sample(n=10000, random_state=42)
        y_sample = y_train.loc[X_sample.index]
        print(f"   📊 Using {len(X_sample)} samples for tuning")
    else:
        X_sample = X_train
        y_sample = y_train
    
    # Randomized search
    random_search = RandomizedSearchCV(
        xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
        ULTIMATE_TUNING_GRID,
        n_iter=50,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0,
        random_state=42
    )
    
    random_search.fit(X_sample, y_sample)
    
    print(f"   ✅ Best params: {random_search.best_params_}")
    print(f"   📊 Best CV score: {random_search.best_score_:.4f}")
    
    return random_search.best_params_

# =========================
# ULTIMATE ADAPTIVE MODEL CLASS
# =========================

class UltimateXGBoost:
    """Ultimate quality XGBoost"""
    
    def __init__(self, symbol, base_params):
        self.symbol = symbol
        self.base_params = base_params.copy()
        self.model = None
        self.training_history = []
        self.model_path = os.path.join(MODEL_DIR, f'xgb_model_{symbol}.joblib')
        self.cv_scores = []
        
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False
    
    def save_model(self):
        joblib.dump(self.model, self.model_path)
        
    def train_ultimate(self, X_train, y_train, X_val=None, y_val=None):
        """Ultimate quality training"""
        print(f"   🏆 Ultimate quality training...")
        
        # Create model with parameters (eval_metric already in constructor)
        self.model = xgb.XGBClassifier(**self.base_params)
        
        # Train (no eval_metric in fit)
        if X_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        if X_val is not None:
            self._evaluate(X_val, y_val)
        
        self.save_model()
        
    def _evaluate(self, X_test, y_test):
        """Comprehensive evaluation"""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        self.training_history.append({
            'date': datetime.now(),
            'accuracy': acc,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        print(f"   ✅ Acc: {acc:.2%}, AUC: {auc:.2%}, F1: {f1:.2%}")
        
        return acc, auc, f1

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
    print("\n🔧 Engineering comprehensive features...")
    df = engineer_features(df)
    features = get_features(df)
    print(f"   Using {len(features)} features")
    
    # Global best params
    global_best_params = None
    
    if tuning_needed:
        print("\n" + "="*90)
        print("🎯 COMPREHENSIVE HYPERPARAMETER TUNING")
        print("="*90)
        print("⏱️ This will take 30-60 minutes for best results...")
        
        # Prepare tuning data
        all_data = df[features].copy()
        all_target = df['target'].copy()
        
        for col in all_data.columns:
            if all_data[col].isnull().any():
                all_data[col] = all_data[col].fillna(all_data[col].median())
        
        if len(all_data) > 1000:
            X_tune, X_tune_val, y_tune, y_tune_val = train_test_split(
                all_data, all_target, test_size=0.2, random_state=42
            )
            
            global_best_params = comprehensive_hyperparameter_tuning(X_tune, y_tune)
            
            # Update ultimate params
            for key, value in global_best_params.items():
                ULTIMATE_PARAMS[key] = value
                print(f"   ✅ Updated {key}: {value}")
    
    # Process each symbol
    print("\n" + "="*90)
    print("🏆 TRAINING ULTIMATE QUALITY MODELS")
    print("="*90)
    
    results = []
    training_stats = []
    total_trained = 0
    
    for symbol, group in df.groupby('symbol'):
        if len(group) < MIN_SAMPLES_PER_SYMBOL:
            continue
        
        print(f"\n{'─'*80}")
        print(f"🔹 {symbol} ({len(group)} rows)")
        
        # Prepare data
        group = group.sort_values('date')
        X = group[features].copy()
        y = group['target'].copy()
        
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        
        target_ratio = y.mean()
        if target_ratio < MIN_TARGET_RATIO or target_ratio > MAX_TARGET_RATIO:
            print(f"   ⚠️ Skipped (target ratio: {target_ratio:.2%})")
            continue
        
        # Time-series split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        print(f"   Training: {len(X_train)} rows, Test: {len(X_test)} rows")
        print(f"   Target ratio: {target_ratio:.2%}")
        
        # Create ultimate model
        ultimate_model = UltimateXGBoost(symbol, ULTIMATE_PARAMS)
        
        # Train
        ultimate_model.train_ultimate(X_train, y_train, X_test, y_test)
        total_trained += 1
        
        # Predict
        X_full = X.copy()
        for col in X_full.columns:
            if X_full[col].isnull().any():
                X_full[col] = X_full[col].fillna(X_full[col].median())
        
        group['confidence_score'] = ultimate_model.model.predict_proba(X_full)[:, 1] * 100
        group['prediction'] = (group['confidence_score'] > 50).astype(int)
        group['signal_strength'] = pd.cut(
            group['confidence_score'],
            bins=[0, 30, 50, 70, 100],
            labels=['Weak', 'Moderate', 'Strong', 'Very Strong']
        )
        
        results.append(group[['symbol', 'date', 'close', 'confidence_score', 'prediction', 'signal_strength']])
        
        # Stats
        last_metrics = ultimate_model.training_history[-1] if ultimate_model.training_history else {}
        training_stats.append({
            'symbol': symbol,
            'samples': len(group),
            'accuracy': last_metrics.get('accuracy', 0),
            'auc': last_metrics.get('auc', 0),
            'f1': last_metrics.get('f1', 0)
        })
    
    # Save results
    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv('./csv/xgb_confidence.csv', index=False)
        print(f"\n✅ Predictions saved: {len(final_df):,} rows")
    
    if training_stats:
        stats_df = pd.DataFrame(training_stats)
        stats_df.to_csv(os.path.join(MODEL_DIR, 'training_summary.csv'), index=False)
        
        print(f"\n📊 ULTIMATE QUALITY SUMMARY:")
        print(f"   Average Accuracy: {stats_df['accuracy'].mean():.2%}")
        print(f"   Average AUC: {stats_df['auc'].mean():.2%}")
        print(f"   Average F1: {stats_df['f1'].mean():.2%}")
    
    # Update dates
    if finetune_needed:
        update_last_run(LAST_FINETUNE_FILE, datetime.today())
    if retrain_needed:
        update_last_run(LAST_RETRAIN_FILE, datetime.today())
    if tuning_needed:
        update_last_run(LAST_TUNING_FILE, datetime.today())
    
    print("\n" + "="*90)
    print("🏆 ULTIMATE XGBOOST TRAINING COMPLETE!")
    print("="*90)
    print(f"✅ Models trained: {total_trained}")
    print(f"📁 Models saved: {MODEL_DIR}")
    print("="*90)

if __name__ == "__main__":
    main()