# xgboost_scheduler.py - COMPLETE VERSION (Daily/Weekly/Monthly + Feedback Learning + HF Upload)

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIG
# =========================
DATA_PATH = './csv/mongodb.csv'
MODEL_DIR = './csv/xgboost/'
PREDICTION_LOG = './csv/prediction_log.csv'
XGB_CONFIDENCE = './csv/xgb_confidence.csv'

os.makedirs(MODEL_DIR, exist_ok=True)

# Schedule tracking files
LAST_DAILY_FILE = './csv/last_daily.txt'
LAST_WEEKLY_FILE = './csv/last_weekly.txt'
LAST_MONTHLY_FILE = './csv/last_monthly.txt'

# Schedule intervals (in days)
DAILY_INTERVAL = 1
WEEKLY_INTERVAL = 7
MONTHLY_INTERVAL = 30

FEEDBACK_DAYS = 5
MIN_SAMPLES_PER_SYMBOL = 50

# =========================
# MODEL PARAMETERS BY MODE
# =========================

# Daily Mode (15 minutes)
DAILY_PARAMS = {
    'n_estimators': 1000,
    'max_depth': 8,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,
    'gamma': 0.2,
    'reg_alpha': 0.5,
    'reg_lambda': 1.5,
    'random_state': 42,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'verbosity': 0
}

# Weekly Mode (30-40 minutes)
WEEKLY_PARAMS = {
    'n_estimators': 1500,
    'max_depth': 9,
    'learning_rate': 0.007,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 6,
    'gamma': 0.25,
    'reg_alpha': 0.6,
    'reg_lambda': 2.0,
    'random_state': 42,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'verbosity': 0
}

# Monthly Mode (2-4 hours)
MONTHLY_PARAMS = {
    'n_estimators': 2000,
    'max_depth': 10,
    'learning_rate': 0.005,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'min_child_weight': 8,
    'gamma': 0.3,
    'reg_alpha': 0.8,
    'reg_lambda': 2.5,
    'random_state': 42,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'verbosity': 0
}

# =========================
# SCHEDULE CHECK FUNCTIONS
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
    
    return needed, days_since

def update_last_run(file_path, date):
    """Update last run date"""
    with open(file_path, 'w') as f:
        f.write(date.strftime('%Y-%m-%d'))

# =========================
# DATA FUNCTIONS
# =========================

def load_data():
    """Load data with proper encoding"""
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found: {DATA_PATH}")
        return pd.DataFrame()
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df.columns = df.columns.str.replace('ï»¿', '').str.replace('\ufeff', '').str.strip()
    df['date'] = pd.to_datetime(df['date'])
    return df

def engineer_features(df):
    """Add engineered features"""
    if df.empty:
        return df
    
    # Price changes
    df['return_5d'] = df.groupby('symbol')['close'].pct_change(5)
    df['return_10d'] = df.groupby('symbol')['close'].pct_change(10)
    
    # Volatility
    df['volatility'] = (df['high'] - df['low']) / df['close']
    df['volatility_5d'] = df.groupby('symbol')['volatility'].transform(lambda x: x.rolling(5).mean())
    
    # Volume
    df['volume_ma'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # RSI signals
    if 'rsi' in df.columns:
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    
    # Target
    df['future_return'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
    df['target'] = (df['future_return'] > 0.02).astype(int)
    
    return df.dropna(subset=['target'])

def get_features(df):
    """Get list of available features"""
    features = ['close', 'volume', 'return_5d', 'return_10d', 'volatility', 'volatility_5d', 'volume_ratio']
    
    if 'rsi_oversold' in df.columns:
        features.extend(['rsi_oversold', 'rsi_overbought'])
    
    return features

# =========================
# FEEDBACK SYSTEM
# =========================

def update_actual_results():
    """Update actual results after FEEDBACK_DAYS"""
    if not os.path.exists(PREDICTION_LOG):
        return None
    
    log = pd.read_csv(PREDICTION_LOG)
    log['date'] = pd.to_datetime(log['date'])
    
    df = load_data()
    if df.empty:
        return log
    
    updated = 0
    
    for i, row in log.iterrows():
        if row.get('checked', 0) == 1:
            continue
        
        future_date = row['date'] + timedelta(days=FEEDBACK_DAYS)
        
        future = df[
            (df['symbol'] == row['symbol']) &
            (df['date'] >= future_date)
        ]
        
        if len(future) > 0:
            future_price = future.iloc[0]['close']
            ret = (future_price - row['close']) / row['close']
            actual = 1 if ret > 0.02 else 0
            
            log.at[i, 'actual'] = actual
            log.at[i, 'checked'] = 1
            updated += 1
    
    if updated > 0:
        log.to_csv(PREDICTION_LOG, index=False)
        print(f"🔄 Feedback updated: {updated} rows")
    
    return log

def get_sample_weights(df, log):
    """Get sample weights based on past mistakes"""
    weights = np.ones(len(df))
    
    if log is None or log.empty:
        return weights
    
    checked_log = log[log['checked'] == 1].copy()
    
    if checked_log.empty:
        return weights
    
    merged = df.merge(
        checked_log[['symbol', 'date', 'prediction', 'actual']],
        on=['symbol', 'date'],
        how='left'
    )
    
    wrong = (merged['prediction'] != merged['actual']) & (~merged['actual'].isna())
    
    if wrong.sum() > 0:
        weights[wrong.values] = 2.0
        print(f"⚖️ Wrong samples boosted: {wrong.sum()}")
    
    return weights

def save_prediction_log(df):
    """Save prediction log"""
    if df.empty:
        return
    
    df_log = df[['symbol', 'date', 'close', 'confidence_score', 'prediction']].copy()
    df_log['actual'] = np.nan
    df_log['checked'] = 0
    
    if os.path.exists(PREDICTION_LOG):
        old = pd.read_csv(PREDICTION_LOG)
        df_log = pd.concat([old, df_log], ignore_index=True)
        df_log = df_log.drop_duplicates(subset=['symbol', 'date'], keep='last')
    
    df_log.to_csv(PREDICTION_LOG, index=False)
    print(f"📝 Log saved: {len(df_log)} rows")

# =========================
# TRAINING FUNCTION
# =========================

def train_symbol(symbol, group, features, params, feedback_log):
    """Train model for a single symbol"""
    try:
        group = group.sort_values('date')
        X = group[features]
        y = group['target']
        
        if len(X) < MIN_SAMPLES_PER_SYMBOL:
            return None
        
        split = int(len(X) * 0.8)
        
        if split < 10 or len(X) - split < 5:
            return None
        
        X_train = X.iloc[:split]
        y_train = y.iloc[:split]
        X_test = X.iloc[split:]
        y_test = y.iloc[split:]
        
        # Dynamic class weight
        target_ratio = y_train.mean()
        if target_ratio < 0.3:
            scale_pos = (1 - target_ratio) / target_ratio
            scale_pos = min(scale_pos, 10)
        elif target_ratio > 0.7:
            scale_pos = target_ratio / (1 - target_ratio)
            scale_pos = min(scale_pos, 10)
        else:
            scale_pos = 1
        
        params_copy = params.copy()
        params_copy['scale_pos_weight'] = scale_pos
        
        print(f"   Class ratio: {target_ratio:.2%} → scale_pos_weight: {scale_pos:.2f}")
        
        weights = get_sample_weights(group.iloc[:split], feedback_log)
        
        model = xgb.XGBClassifier(**params_copy)
        
        try:
            model.fit(X_train, y_train, sample_weight=weights, eval_set=[(X_test, y_test)], verbose=False)
        except:
            model.fit(X_train, y_train, sample_weight=weights, verbose=False)
        
        # ✅ SAVE MODEL
        model_path = os.path.join(MODEL_DIR, f'{symbol}.joblib')
        joblib.dump(model, model_path)
        print(f"   💾 Model saved: {model_path}")
        
        preds = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else 0.5
        
        print(f"   ✅ Acc: {acc:.2%}, AUC: {auc:.2%}")
        
        group['confidence_score'] = model.predict_proba(X)[:, 1] * 100
        group['prediction'] = (group['confidence_score'] > 50).astype(int)
        
        return group[['symbol', 'date', 'close', 'confidence_score', 'prediction']]
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

# =========================
# HF UPLOAD FUNCTION
# =========================

def upload_to_huggingface():
    """Upload models and CSV files to Hugging Face"""
    try:
        from hf_uploader import SmartDatasetUploader, REPO_ID, HF_TOKEN
        
        print("\n" + "="*70)
        print("📤 UPLOADING TO HUGGING FACE")
        print("="*70)
        
        uploader = SmartDatasetUploader(REPO_ID, HF_TOKEN)
        
        # Upload entire csv folder
        uploader.smart_upload(
            local_folder="./csv",
            unique_columns=['symbol', 'date']
        )
        
        print("✅ Upload to Hugging Face complete!")
        return True
        
    except Exception as e:
        print(f"⚠️ HF upload failed: {e}")
        return False

# =========================
# MAIN
# =========================

def main():
    print("="*70)
    print("🚀 XGBOOST SCHEDULER (Daily/Weekly/Monthly + Feedback Learning)")
    print("="*70)
    
    # Check schedule
    daily_needed, daily_days = check_last_run(LAST_DAILY_FILE, DAILY_INTERVAL)
    weekly_needed, weekly_days = check_last_run(LAST_WEEKLY_FILE, WEEKLY_INTERVAL)
    monthly_needed, monthly_days = check_last_run(LAST_MONTHLY_FILE, MONTHLY_INTERVAL)
    
    # Determine mode
    if monthly_needed:
        mode = "MONTHLY"
        params = MONTHLY_PARAMS
        expected_time = "2-4 hours"
    elif weekly_needed:
        mode = "WEEKLY"
        params = WEEKLY_PARAMS
        expected_time = "30-40 minutes"
    elif daily_needed:
        mode = "DAILY"
        params = DAILY_PARAMS
        expected_time = "15 minutes"
    else:
        print("\n✅ No training needed today!")
        print(f"   Next Daily: {(datetime.today() + timedelta(days=DAILY_INTERVAL - daily_days)).date()}")
        print(f"   Next Weekly: {(datetime.today() + timedelta(days=WEEKLY_INTERVAL - weekly_days)).date()}")
        print(f"   Next Monthly: {(datetime.today() + timedelta(days=MONTHLY_INTERVAL - monthly_days)).date()}")
        
        # Still upload existing files
        upload_to_huggingface()
        return
    
    print(f"\n{'='*70}")
    print(f"🎯 RUNNING {mode} MODE")
    print(f"⏱️ Expected time: {expected_time}")
    print(f"📊 Parameters: n_estimators={params['n_estimators']}, depth={params['max_depth']}, lr={params['learning_rate']}, gamma={params['gamma']}")
    print(f"{'='*70}\n")
    
    # Feedback update
    print("📊 Step 1: Updating feedback...")
    feedback_log = update_actual_results()
    
    # Load data
    print("\n📂 Step 2: Loading data...")
    df = load_data()
    
    if df.empty:
        print("❌ No data loaded. Exiting.")
        return
    
    print(f"   Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols")
    
    # Feature engineering
    print("\n🔧 Step 3: Feature engineering...")
    df = engineer_features(df)
    features = get_features(df)
    print(f"   Using {len(features)} features: {features}")
    
    # Train models
    print("\n🏆 Step 4: Training models...")
    print("="*70)
    
    results = []
    trained_count = 0
    skipped_count = 0
    model_files = []
    
    for symbol, group in df.groupby('symbol'):
        if len(group) < MIN_SAMPLES_PER_SYMBOL:
            skipped_count += 1
            continue
        
        print(f"\n🔹 {symbol} ({len(group)} rows)")
        
        result = train_symbol(symbol, group, features, params, feedback_log)
        
        if result is not None:
            results.append(result)
            trained_count += 1
            model_files.append(f'{symbol}.joblib')
        else:
            skipped_count += 1
    
    # Save predictions
    print("\n💾 Step 5: Saving results...")
    print("="*70)
    
    if results:
        final = pd.concat(results, ignore_index=True)
        final.to_csv(XGB_CONFIDENCE, index=False)
        print(f"✅ Predictions saved: {len(final):,} rows")
        
        save_prediction_log(final)
    
    # Update schedule dates
    if daily_needed:
        update_last_run(LAST_DAILY_FILE, datetime.today())
    if weekly_needed:
        update_last_run(LAST_WEEKLY_FILE, datetime.today())
    if monthly_needed:
        update_last_run(LAST_MONTHLY_FILE, datetime.today())
    
    # Summary
    print("\n" + "="*70)
    print(f"✅ {mode} MODE TRAINING COMPLETE!")
    print(f"📊 Models trained: {trained_count}")
    print(f"⚠️ Symbols skipped: {skipped_count}")
    print(f"📁 Models saved: {MODEL_DIR}")
    
    # Show saved models
    if model_files:
        print(f"\n📋 Model files saved:")
        for f in model_files[:5]:
            print(f"   - {f}")
        if len(model_files) > 5:
            print(f"   ... and {len(model_files)-5} more")
    
    # Upload to Hugging Face
    upload_to_huggingface()
    
    print("="*70)
    print("🎉 DONE!")

if __name__ == "__main__":
    main()