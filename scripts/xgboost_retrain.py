# xgboost_scheduler.py - COMPLETE VERSION
# Features:
# 1. Only good models (AUC >= 0.55) are saved
# 2. Bad models retry up to 3 times, then retry monthly
# 3. Already trained good models retrain with new data
# 4. Single commit upload to Hugging Face
# 5. Monthly retry for permanently failed models

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
MODEL_METADATA = './csv/model_metadata.csv'  # ট্র্যাক করতে কোন মডেল ভালো/খারাপ

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

# 🆕 Model quality threshold
AUC_THRESHOLD = 0.55  # AUC 0.55-এর নিচে হলে মডেল সেভ হবে না
RETRAIN_ATTEMPTS = 3  # কতবার চেষ্টা করবে খারাপ মডেল রিট্রেন করতে
MONTHLY_RETRY_AFTER = 30  # ব্যর্থ হওয়ার পর কত দিন পরে আবার চেষ্টা করবে

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
# MODEL METADATA MANAGEMENT
# =========================

def load_model_metadata():
    """Load model metadata (which models are good/bad)"""
    if os.path.exists(MODEL_METADATA):
        df = pd.read_csv(MODEL_METADATA)
        df['last_trained'] = pd.to_datetime(df['last_trained'])
        df['last_attempt'] = pd.to_datetime(df['last_attempt']) if 'last_attempt' in df.columns else pd.to_datetime(df['last_trained'])
        return df
    else:
        # Create empty metadata
        return pd.DataFrame(columns=['symbol', 'last_trained', 'last_attempt', 'auc', 'acc', 
                                      'failed_attempts', 'status', 'class_ratio'])

def save_model_metadata(df):
    """Save model metadata"""
    df.to_csv(MODEL_METADATA, index=False)

def should_retrain(symbol, metadata, current_date=None):
    """
    Check if a symbol should be retrained
    Rules:
    1. New symbol -> retrain
    2. GOOD model -> retrain (update with new data)
    3. BAD model with attempts < RETRAIN_ATTEMPTS -> retrain
    4. BAD model with attempts >= RETRAIN_ATTEMPTS -> retry monthly
    """
    if current_date is None:
        current_date = datetime.now()
    
    if metadata.empty or symbol not in metadata['symbol'].values:
        return True, "new_symbol"  # নতুন স্টক, ট্রেন করবে
    
    symbol_data = metadata[metadata['symbol'] == symbol].iloc[0]
    
    # Status GOOD হলে রিট্রেন করবে (নতুন ডাটা দিয়ে আপডেট)
    if symbol_data['status'] == 'GOOD':
        return True, "good_model_update"
    
    # Status BAD হলে
    if symbol_data['status'] == 'BAD':
        failed_attempts = symbol_data['failed_attempts']
        
        # যদি ৩ বারের কম চেষ্টা করে থাকে
        if failed_attempts < RETRAIN_ATTEMPTS:
            return True, f"bad_retry_{failed_attempts+1}"
        
        # ৩ বার ব্যর্থ হয়েছে, মাসিক রিট্রাই চেক করুন
        last_attempt = symbol_data.get('last_attempt', symbol_data['last_trained'])
        if isinstance(last_attempt, str):
            last_attempt = pd.to_datetime(last_attempt)
        
        days_since_last_attempt = (current_date - last_attempt).days
        
        if days_since_last_attempt >= MONTHLY_RETRY_AFTER:
            return True, f"monthly_retry_after_{days_since_last_attempt}_days"
        else:
            days_left = MONTHLY_RETRY_AFTER - days_since_last_attempt
            return False, f"monthly_wait_{days_left}_days"
    
    return True, "default"

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
# TRAINING FUNCTION (IMPROVED)
# =========================

def train_symbol(symbol, group, features, params, feedback_log, metadata):
    """Train model - only saves if AUC >= threshold"""
    try:
        group = group.sort_values('date')
        X = group[features]
        y = group['target']

        if len(X) < MIN_SAMPLES_PER_SYMBOL:
            return None, None

        split = int(len(X) * 0.8)

        if split < 10 or len(X) - split < 5:
            return None, None

        X_train = X.iloc[:split]
        y_train = y.iloc[:split]
        X_test = X.iloc[split:]
        y_test = y.iloc[split:]

        # Dynamic class weight
        target_ratio = y_train.mean()
        if target_ratio < 0.3:
            scale_pos = min((1 - target_ratio) / target_ratio, 10)
        elif target_ratio > 0.7:
            scale_pos = min(target_ratio / (1 - target_ratio), 10)
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

        preds = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else 0.5

        print(f"   ✅ Acc: {acc:.2%}, AUC: {auc:.2%}")

        # Check if model is good enough
        if auc >= AUC_THRESHOLD:
            # GOOD MODEL - Save it
            model_path = os.path.join(MODEL_DIR, f'{symbol}.joblib')
            joblib.dump(model, model_path)
            print(f"   💾 GOOD MODEL saved: {model_path}")
            
            # Generate predictions
            group['confidence_score'] = model.predict_proba(X)[:, 1] * 100
            group['prediction'] = (group['confidence_score'] > 50).astype(int)
            
            result = group[['symbol', 'date', 'close', 'confidence_score', 'prediction']]
            status = 'GOOD'
            failed_attempts = 0
            
        else:
            # BAD MODEL - Don't save
            print(f"   ⚠️ BAD MODEL (AUC {auc:.2%} < {AUC_THRESHOLD}) - Not saving")
            
            # Check previous attempts
            if not metadata.empty and symbol in metadata['symbol'].values:
                prev_data = metadata[metadata['symbol'] == symbol].iloc[0]
                failed_attempts = prev_data.get('failed_attempts', 0) + 1
            else:
                failed_attempts = 1
            
            # Use default predictions (always 0 - conservative)
            group['confidence_score'] = 50
            group['prediction'] = 0
            result = group[['symbol', 'date', 'close', 'confidence_score', 'prediction']]
            status = 'BAD'
            
            if failed_attempts >= RETRAIN_ATTEMPTS:
                print(f"   🔴 {symbol}: Failed {failed_attempts} times. Will retry monthly.")
        
        return result, {
            'symbol': symbol,
            'last_trained': datetime.now(),
            'last_attempt': datetime.now(),
            'auc': auc,
            'acc': acc,
            'failed_attempts': failed_attempts if auc < AUC_THRESHOLD else 0,
            'status': status,
            'class_ratio': target_ratio
        }

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, None

# =========================
# HF UPLOAD FUNCTION (SINGLE COMMIT)
# =========================

def upload_to_huggingface():
    """Upload all files in a single commit to Hugging Face"""
    try:
        from huggingface_hub import HfApi
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        hf_token = os.getenv("hf_token")
        
        if not hf_token:
            print("⚠️ No HF_TOKEN found")
            return False
        
        api = HfApi()
        repo_id = "ahashanahmed/csv"
        
        print("\n" + "="*70)
        print("📤 UPLOADING TO HUGGING FACE (SINGLE COMMIT)")
        print("="*70)
        
        # Upload entire folder in one commit
        api.upload_folder(
            folder_path="./csv",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Auto-update: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ignore_patterns=["*.tmp", "*.log", "__pycache__", ".DS_Store"]
        )
        
        print("✅ Upload complete! (single commit)")
        return True
        
    except Exception as e:
        print(f"⚠️ HF upload failed: {e}")
        return False

# =========================
# DOWNLOAD FUNCTION
# =========================

def download_from_huggingface():
    """Download latest data from Hugging Face"""
    try:
        from huggingface_hub import snapshot_download
        import os
        
        print("\n📥 Checking for existing data...")
        
        if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) < 1000:
            print("   Downloading from Hugging Face...")
            snapshot_download(
                repo_id="ahashanahmed/csv",
                repo_type="dataset",
                local_dir="./csv",
                local_dir_use_symlinks=False
            )
            print("   ✅ Download complete!")
            return True
        else:
            print(f"   ✅ Local data exists ({os.path.getsize(DATA_PATH)/1024:.1f} KB)")
            return True
            
    except Exception as e:
        print(f"   ⚠️ Download failed: {e}")
        return False

# =========================
# MAIN
# =========================

def main():
    print("="*70)
    print("🚀 XGBOOST SCHEDULER (Good Models Only + Monthly Retry)")
    print("="*70)
    
    # Step 0: Download latest data from HF
    download_from_huggingface()
    
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
    print(f"📈 Model threshold: AUC >= {AUC_THRESHOLD} to save")
    print(f"🔄 Bad models: {RETRAIN_ATTEMPTS} attempts, then monthly retry")
    print(f"{'='*70}\n")

    # Feedback update
    print("📊 Step 1: Updating feedback...")
    feedback_log = update_actual_results()

    # Load metadata
    metadata = load_model_metadata()
    print(f"📋 Loaded metadata: {len(metadata)} symbols tracked")

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
    print(f"🎯 Target: Save only models with AUC >= {AUC_THRESHOLD}")
    print(f"🔄 Bad models: {RETRAIN_ATTEMPTS} immediate attempts, then monthly retry")
    print("="*70)

    results = []
    updated_metadata = []
    trained_count = 0
    good_count = 0
    bad_count = 0
    monthly_retry_count = 0
    skipped_count = 0
    
    retry_reasons = {
        'new_symbol': [],
        'good_model_update': [],
        'bad_retry': [],
        'monthly_retry': []
    }

    for symbol, group in df.groupby('symbol'):
        if len(group) < MIN_SAMPLES_PER_SYMBOL:
            skipped_count += 1
            continue
        
        # Check if we should retrain this symbol
        should_train, reason = should_retrain(symbol, metadata)
        
        if not should_train:
            print(f"\n🔸 {symbol} ({len(group)} rows) - SKIPPED: {reason}")
            skipped_count += 1
            continue
        
        # Track retry reason
        if 'retry' in reason:
            if 'monthly' in reason:
                monthly_retry_count += 1
                retry_reasons['monthly_retry'].append(symbol)
            elif 'bad' in reason:
                retry_reasons['bad_retry'].append(symbol)
        elif reason == 'new_symbol':
            retry_reasons['new_symbol'].append(symbol)
        elif reason == 'good_model_update':
            retry_reasons['good_model_update'].append(symbol)

        print(f"\n🔹 {symbol} ({len(group)} rows) - Reason: {reason}")
        
        result, model_info = train_symbol(symbol, group, features, params, feedback_log, metadata)
        
        if result is not None:
            results.append(result)
            trained_count += 1
            
            if model_info:
                updated_metadata.append(model_info)
                if model_info['status'] == 'GOOD':
                    good_count += 1
                else:
                    bad_count += 1
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

    # Update metadata
    if updated_metadata:
        new_metadata = pd.DataFrame(updated_metadata)
        
        # Merge with existing metadata
        if not metadata.empty:
            # Remove old entries for symbols we just updated
            symbols_updated = new_metadata['symbol'].unique()
            metadata = metadata[~metadata['symbol'].isin(symbols_updated)]
        
        final_metadata = pd.concat([metadata, new_metadata], ignore_index=True)
        save_model_metadata(final_metadata)
        print(f"✅ Metadata updated: {len(final_metadata)} symbols tracked")
        print(f"   🟢 Good models: {len(final_metadata[final_metadata['status'] == 'GOOD'])}")
        print(f"   🔴 Bad models: {len(final_metadata[final_metadata['status'] == 'BAD'])}")

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
    print("="*70)
    print(f"📊 Models trained: {trained_count}")
    print(f"   🟢 Good models saved: {good_count} (AUC >= {AUC_THRESHOLD})")
    print(f"   🔴 Bad models (not saved): {bad_count}")
    print(f"   🔄 Monthly retry attempts: {monthly_retry_count}")
    print(f"⚠️ Symbols skipped: {skipped_count}")
    
    # Show retry statistics
    if retry_reasons['new_symbol']:
        print(f"\n🆕 New symbols trained: {len(retry_reasons['new_symbol'])}")
    if retry_reasons['good_model_update']:
        print(f"🟢 Good models updated: {len(retry_reasons['good_model_update'])}")
    if retry_reasons['bad_retry']:
        print(f"🔴 Bad models retrying: {len(retry_reasons['bad_retry'])}")
    if retry_reasons['monthly_retry']:
        print(f"📅 Monthly retry (failed models): {len(retry_reasons['monthly_retry'])}")
        print(f"   Symbols: {', '.join(retry_reasons['monthly_retry'][:10])}")
        if len(retry_reasons['monthly_retry']) > 10:
            print(f"   ... and {len(retry_reasons['monthly_retry'])-10} more")

    # Show good models
    if good_count > 0 and 'final_metadata' in locals():
        good_models = final_metadata[final_metadata['status'] == 'GOOD'].sort_values('auc', ascending=False)
        if len(good_models) > 0:
            print(f"\n🟢 TOP 10 GOOD MODELS:")
            for _, row in good_models.head(10).iterrows():
                print(f"   {row['symbol']}: AUC={row['auc']:.2%}, Acc={row['acc']:.2%}")

    # Show bad models status
    if bad_count > 0 and 'final_metadata' in locals():
        bad_models = final_metadata[final_metadata['status'] == 'BAD'].sort_values('failed_attempts', ascending=False)
        if len(bad_models) > 0:
            print(f"\n🔴 BAD MODELS STATUS:")
            for _, row in bad_models.head(10).iterrows():
                attempts = row['failed_attempts']
                if attempts >= RETRAIN_ATTEMPTS:
                    print(f"   {row['symbol']}: AUC={row['auc']:.2%}, Attempts={attempts} (Monthly retry mode)")
                else:
                    remaining = RETRAIN_ATTEMPTS - attempts
                    print(f"   {row['symbol']}: AUC={row['auc']:.2%}, Attempts={attempts}, Remaining={remaining}")

    # Upload to Hugging Face
    upload_to_huggingface()

    print("\n" + "="*70)
    print("🎉 DONE!")
    print("="*70)

if __name__ == "__main__":
    main()