# xgboost_scheduler.py - Complete 3-in-1 System
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
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
LAST_DAILY_FILE = './csv/last_daily.txt'
LAST_WEEKLY_FILE = './csv/last_weekly.txt'
LAST_MONTHLY_FILE = './csv/last_monthly.txt'

# Schedule intervals (in days)
DAILY_INTERVAL = 1
WEEKLY_INTERVAL = 7
MONTHLY_INTERVAL = 30

# Minimum samples per symbol
MIN_SAMPLES_PER_SYMBOL = 50
MIN_TARGET_RATIO = 0.15
MAX_TARGET_RATIO = 0.85

# =========================
# MODEL PARAMETERS BY MODE
# =========================

# Fast Mode (Daily) - 1-2 minutes
FAST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.1,
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

# Balanced Mode (Weekly) - 10-15 minutes
BALANCED_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
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

# Ultimate Mode (Monthly) - 2-4 hours
ULTIMATE_PARAMS = {
    'n_estimators': 1000,
    'max_depth': 8,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'colsample_bynode': 0.7,
    'min_child_weight': 5,
    'gamma': 0.2,
    'reg_alpha': 0.5,
    'reg_lambda': 2,
    'max_leaves': 31,
    'random_state': 42,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'tree_method': 'hist',
    'grow_policy': 'lossguide'
}

# Ultimate tuning grid
ULTIMATE_TUNING_GRID = {
    'max_depth': [6, 7, 8, 9, 10],
    'min_child_weight': [3, 5, 7, 10],
    'gamma': [0, 0.1, 0.2, 0.3, 0.5],
    'learning_rate': [0.1, 0.05, 0.03, 0.01, 0.005],
    'n_estimators': [500, 750, 1000, 1500],
    'reg_alpha': [0, 0.1, 0.5, 1, 2],
    'reg_lambda': [0.5, 1, 1.5, 2, 3],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
}

print("="*80)
print("🤖 XGBOOST AUTOMATED SCHEDULER - 3 in 1 SYSTEM")
print("="*80)
print("📅 Daily   : Fast training (1-2 min)  - Keep models updated")
print("📆 Weekly  : Balanced training (10-15 min) - Improve accuracy")
print("📅 Monthly : Ultimate training (2-4 hours) - Best quality")
print("="*80)

# =========================
# HELPER FUNCTIONS
# =========================

def check_last_run(file_path, interval):
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
    needed = days_since >= interval
    
    return last_date, today, days_since, needed

def update_last_run(file_path, date):
    """Update last run date"""
    with open(file_path, 'w') as f:
        f.write(date.strftime('%Y-%m-%d'))

def fix_bom_columns(df):
    """Remove BOM characters from column names"""
    df.columns = df.columns.str.replace('ï»¿', '').str.replace('\ufeff', '').str.strip()
    return df

def load_data():
    """Load and prepare data"""
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
    """Add engineered features"""
    print("   🔧 Adding engineered features...")
    
    # Price changes
    for period in [1, 3, 5, 10]:
        df[f'return_{period}d'] = df.groupby('symbol')['close'].pct_change(period)
    
    # Volatility
    df['volatility'] = (df['high'] - df['low']) / df['close']
    df['volatility_5d'] = df.groupby('symbol')['volatility'].transform(lambda x: x.rolling(5).mean())
    
    # Volume
    df['volume_ma'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
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
    
    # Target
    df['future_return'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
    df['target'] = (df['future_return'] > 0.02).astype(int)
    
    # Drop NaN target only
    df = df.dropna(subset=['target'])
    
    return df

def get_features(df):
    """Get list of available features"""
    feature_cols = [
        'open', 'high', 'low', 'volume', 'value', 'trades', 'change', 'marketCap',
        'bb_upper', 'bb_middle', 'bb_lower', 'macd', 'macd_signal', 'macd_hist',
        'rsi', 'atr', 'zigzag', 'ema_200',
        'return_1d', 'return_3d', 'return_5d', 'return_10d',
        'volatility', 'volatility_5d',
        'volume_ratio',
        'dist_to_resistance', 'dist_to_support',
        'macd_cross_up', 'macd_cross_down',
        'rsi_oversold', 'rsi_overbought', 'rsi_cross_above_30', 'rsi_cross_below_70'
    ]
    
    return [f for f in feature_cols if f in df.columns]

def hyperparameter_tuning(X_train, y_train):
    """Comprehensive hyperparameter tuning for monthly mode"""
    print("   🔧 Running hyperparameter tuning...")
    
    from sklearn.model_selection import RandomizedSearchCV
    
    if len(X_train) > 10000:
        X_sample = X_train.sample(n=10000, random_state=42)
        y_sample = y_train.loc[X_sample.index]
    else:
        X_sample = X_train
        y_sample = y_train
    
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
# TRAINER CLASS
# =========================

class XGBoostTrainer:
    def __init__(self, symbol, params):
        self.symbol = symbol
        self.params = params
        self.model = None
        self.model_path = os.path.join(MODEL_DIR, f'xgb_model_{symbol}.joblib')
        
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False
    
    def save_model(self):
        joblib.dump(self.model, self.model_path)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train model with given parameters"""
        self.model = xgb.XGBClassifier(**self.params)
        
        if X_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X_train, y_train)
        
        self.save_model()
        
        if X_val is not None:
            y_pred = self.model.predict(X_val)
            y_proba = self.model.predict_proba(X_val)[:, 1]
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)
            return acc, auc
        
        return 0, 0

# =========================
# MAIN EXECUTION
# =========================

def main():
    # Check schedules
    _, today, _, daily_needed = check_last_run(LAST_DAILY_FILE, DAILY_INTERVAL)
    _, _, _, weekly_needed = check_last_run(LAST_WEEKLY_FILE, WEEKLY_INTERVAL)
    _, _, _, monthly_needed = check_last_run(LAST_MONTHLY_FILE, MONTHLY_INTERVAL)
    
    # Determine mode
    if monthly_needed:
        mode = "MONTHLY"
        params = ULTIMATE_PARAMS
        tuning_needed = True
        expected_time = "2-4 hours"
    elif weekly_needed:
        mode = "WEEKLY"
        params = BALANCED_PARAMS
        tuning_needed = False
        expected_time = "10-15 minutes"
    elif daily_needed:
        mode = "DAILY"
        params = FAST_PARAMS
        tuning_needed = False
        expected_time = "1-2 minutes"
    else:
        print("\n✅ No training needed today!")
        print(f"   Next Daily: {(datetime.today() + timedelta(days=DAILY_INTERVAL)).date()}")
        print(f"   Next Weekly: {(datetime.today() + timedelta(days=WEEKLY_INTERVAL - (datetime.today() - check_last_run(LAST_WEEKLY_FILE, WEEKLY_INTERVAL)[0]).days)).date() if os.path.exists(LAST_WEEKLY_FILE) else 'First run needed'}")
        print(f"   Next Monthly: {(datetime.today() + timedelta(days=MONTHLY_INTERVAL - (datetime.today() - check_last_run(LAST_MONTHLY_FILE, MONTHLY_INTERVAL)[0]).days)).date() if os.path.exists(LAST_MONTHLY_FILE) else 'First run needed'}")
        return
    
    print(f"\n{'='*80}")
    print(f"🎯 RUNNING {mode} MODE")
    print(f"⏱️ Expected time: {expected_time}")
    print(f"{'='*80}\n")
    
    # Load data
    print("📂 Loading data...")
    df = load_data()
    print(f"   Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols")
    
    # Engineer features
    print("\n🔧 Engineering features...")
    df = engineer_features(df)
    features = get_features(df)
    print(f"   Using {len(features)} features")
    
    # Global tuning for monthly mode
    global_best_params = None
    if tuning_needed:
        print("\n" + "="*80)
        print("🎯 HYPERPARAMETER TUNING (Monthly Mode)")
        print("="*80)
        
        all_data = df[features].copy()
        all_target = df['target'].copy()
        
        for col in all_data.columns:
            if all_data[col].isnull().any():
                all_data[col] = all_data[col].fillna(all_data[col].median())
        
        if len(all_data) > 1000:
            X_tune, _, y_tune, _ = train_test_split(all_data, all_target, test_size=0.2, random_state=42)
            global_best_params = hyperparameter_tuning(X_tune, y_tune)
            
            for key, value in global_best_params.items():
                params[key] = value
                print(f"   ✅ Updated {key}: {value}")
    
    # Train models
    print("\n" + "="*80)
    print(f"🏆 TRAINING MODELS ({mode} MODE)")
    print("="*80)
    
    results = []
    training_stats = []
    total_trained = 0
    
    for symbol, group in df.groupby('symbol'):
        if len(group) < MIN_SAMPLES_PER_SYMBOL:
            continue
        
        print(f"\n{'─'*50}")
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
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        print(f"   Training: {len(X_train)} rows, Test: {len(X_test)} rows")
        print(f"   Target ratio: {target_ratio:.2%}")
        
        # Train
        trainer = XGBoostTrainer(symbol, params)
        acc, auc = trainer.train(X_train, y_train, X_test, y_test)
        
        print(f"   ✅ Acc: {acc:.2%}, AUC: {auc:.2%}")
        total_trained += 1
        
        # Predict
        X_full = X.copy()
        for col in X_full.columns:
            if X_full[col].isnull().any():
                X_full[col] = X_full[col].fillna(X_full[col].median())
        
        group['confidence_score'] = trainer.model.predict_proba(X_full)[:, 1] * 100
        group['prediction'] = (group['confidence_score'] > 50).astype(int)
        
        results.append(group[['symbol', 'date', 'close', 'confidence_score', 'prediction']])
        
        training_stats.append({
            'symbol': symbol,
            'samples': len(group),
            'accuracy': acc,
            'auc': auc,
            'mode': mode
        })
    
    # Save results
    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv('./csv/xgb_confidence.csv', index=False)
        print(f"\n✅ Predictions saved: {len(final_df):,} rows")
    
    if training_stats:
        stats_df = pd.DataFrame(training_stats)
        stats_path = os.path.join(MODEL_DIR, f'training_summary_{mode.lower()}.csv')
        stats_df.to_csv(stats_path, index=False)
        
        print(f"\n📊 {mode} MODE SUMMARY:")
        print(f"   Average Accuracy: {stats_df['accuracy'].mean():.2%}")
        print(f"   Average AUC: {stats_df['auc'].mean():.2%}")
        print(f"   Models trained: {total_trained}")
    
    # Update last run dates
    if daily_needed:
        update_last_run(LAST_DAILY_FILE, datetime.today())
        print(f"✅ Daily run date updated: {datetime.today().date()}")
    
    if weekly_needed:
        update_last_run(LAST_WEEKLY_FILE, datetime.today())
        print(f"✅ Weekly run date updated: {datetime.today().date()}")
    
    if monthly_needed:
        update_last_run(LAST_MONTHLY_FILE, datetime.today())
        print(f"✅ Monthly run date updated: {datetime.today().date()}")
    
    print("\n" + "="*80)
    print(f"✅ {mode} MODE TRAINING COMPLETE!")
    print(f"📁 Models saved: {MODEL_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()