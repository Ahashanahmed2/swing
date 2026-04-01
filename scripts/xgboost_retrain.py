# xgboost_retrain.py
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIG
# =========================
DATA_PATH = './csv/mongodb.csv'
MODEL_DIR = './csv/xgboost/'
os.makedirs(MODEL_DIR, exist_ok=True)

LAST_RETRAIN_FILE = './csv/last_retrain.txt'
RETRAIN_TYPE = 'weekly'  # 'weekly' or 'monthly'
RETRAIN_DAYS = 7 if RETRAIN_TYPE == 'weekly' else 30

# Updated thresholds for better coverage
MIN_SAMPLES_PER_SYMBOL = 30   # Lowered from 100 to include more symbols
MIN_TARGET_RATIO = 0.15       # Skip if target ratio < 15%
MAX_TARGET_RATIO = 0.85       # Skip if target ratio > 85%
MIN_ACCURACY_THRESHOLD = 0.45 # Minimum acceptable accuracy

# Model parameters
MODEL_PARAMS = {
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

print("="*70)
print("XGBOOST MODEL RETRAINER")
print("="*70)
print(f"Retrain Type: {RETRAIN_TYPE.upper()}")
print(f"Retrain Days: {RETRAIN_DAYS}")
print(f"Min Samples per Symbol: {MIN_SAMPLES_PER_SYMBOL}")
print("="*70)

# =========================
# CHECK RETRAIN
# =========================
def check_retrain_needed():
    if os.path.exists(LAST_RETRAIN_FILE):
        try:
            with open(LAST_RETRAIN_FILE, 'r') as f:
                last_retrain_date = datetime.strptime(f.read().strip(), '%Y-%m-%d')
        except:
            last_retrain_date = datetime(2000, 1, 1)
    else:
        last_retrain_date = datetime(2000, 1, 1)
    
    today = datetime.today()
    days_since = (today - last_retrain_date).days
    retrain_needed = days_since >= RETRAIN_DAYS
    
    return last_retrain_date, today, days_since, retrain_needed

last_retrain_date, today, days_since, retrain_needed = check_retrain_needed()

print(f"\n📅 Last retrain: {last_retrain_date.date()}")
print(f"📅 Today: {today.date()}")
print(f"📊 Days since: {days_since}")
print(f"🔄 Retrain needed: {'YES' if retrain_needed else 'NO'}")

if not retrain_needed:
    print(f"\n⏰ No retrain needed. Next retrain: {(last_retrain_date + timedelta(days=RETRAIN_DAYS)).date()}")
    print("❌ Exiting...")
    sys.exit(0)

print("\n" + "="*70)
print("STARTING RETRAINING PROCESS")
print("="*70)

# =========================
# LOAD DATA
# =========================
print("\n📂 Loading data...")
try:
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    print(f"✅ Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols")
    print(f"📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    sys.exit(1)

# =========================
# FEATURE ENGINEERING (Only additional features not in CSV)
# =========================
print("\n🔧 Engineering additional features...")

# Price changes
df['return'] = df['close'].pct_change()
df['return_3d'] = df.groupby('symbol')['close'].pct_change(3)
df['return_5d'] = df.groupby('symbol')['close'].pct_change(5)
df['return_10d'] = df.groupby('symbol')['close'].pct_change(10)

# Volatility
df['volatility'] = (df['high'] - df['low']) / df['close']
df['volatility_5d'] = df.groupby('symbol')['volatility'].transform(lambda x: x.rolling(5).mean())
df['volatility_10d'] = df.groupby('symbol')['volatility'].transform(lambda x: x.rolling(10).mean())

# Volume features
df['volume_ma'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
df['volume_ratio'] = df['volume'] / df['volume_ma']
df['volume_ratio_5d'] = df.groupby('symbol')['volume_ratio'].transform(lambda x: x.rolling(5).mean())
df['volume_spike'] = (df['volume'] > df['volume_ma'] * 1.5).astype(int)
df['volume_dry_up'] = (df['volume'] < df['volume_ma'] * 0.5).astype(int)

# MACD cross signals (if macd and macd_signal exist)
if 'macd' in df.columns and 'macd_signal' in df.columns:
    df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                           (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

# RSI signals (if rsi exists)
if 'rsi' in df.columns:
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_cross_above_30'] = ((df['rsi'] >= 30) & (df['rsi'].shift(1) < 30)).astype(int)
    df['rsi_cross_below_70'] = ((df['rsi'] <= 70) & (df['rsi'].shift(1) > 70)).astype(int)

# ATR ratio
if 'atr' in df.columns:
    df['atr_ratio'] = df['atr'] / df['close'] * 100

# Support/Resistance
df['resistance_20d'] = df.groupby('symbol')['high'].transform(lambda x: x.rolling(20).max())
df['support_20d'] = df.groupby('symbol')['low'].transform(lambda x: x.rolling(20).min())
df['distance_to_resistance'] = (df['resistance_20d'] - df['close']) / df['close'] * 100
df['distance_to_support'] = (df['close'] - df['support_20d']) / df['close'] * 100

# RSI Divergence
if 'rsi' in df.columns:
    df['rsi_bullish_divergence'] = ((df['low'] <= df['low'].shift(1)) & 
                                     (df['rsi'] > df['rsi'].shift(1))).astype(int)
    df['rsi_bearish_divergence'] = ((df['high'] >= df['high'].shift(1)) & 
                                     (df['rsi'] < df['rsi'].shift(1))).astype(int)

print(f"✅ Feature engineering complete")

# =========================
# CREATE TARGET
# =========================
print("\n🎯 Creating target variable...")

# Future return (next 5 days)
df['future_return'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
df['target'] = (df['future_return'] > 0.02).astype(int)  # 2% profit target

# IMPORTANT: Only drop rows where target is NaN (end of series)
initial_len = len(df)
df = df.dropna(subset=['target'])
print(f"✅ Dropped {initial_len - len(df):,} rows with NaN target")
print(f"✅ Remaining rows: {len(df):,}")

# =========================
# FEATURES LIST
# =========================
print("\n📋 Defining features...")

# All available features (from CSV + engineered)
base_features = [
    'open', 'high', 'low', 'volume', 'value', 'trades', 'change', 'marketCap',
    'bb_upper', 'bb_middle', 'bb_lower', 'macd', 'macd_signal', 'macd_hist',
    'rsi', 'atr', 'zigzag', 'ema_200'
]

engineered_features = [
    'return', 'return_3d', 'return_5d', 'return_10d',
    'volatility', 'volatility_5d', 'volatility_10d',
    'volume_ratio', 'volume_ratio_5d', 'volume_spike', 'volume_dry_up',
    'distance_to_resistance', 'distance_to_support',
    'atr_ratio'
]

signal_features = [
    'macd_cross_up', 'macd_cross_down',
    'rsi_oversold', 'rsi_overbought', 'rsi_cross_above_30', 'rsi_cross_below_70',
    'rsi_bullish_divergence', 'rsi_bearish_divergence'
]

# Combine all features that exist in dataframe
all_features = base_features + engineered_features + signal_features
features = [f for f in all_features if f in df.columns]

print(f"✅ Total features: {len(features)}")
print(f"   Base features: {len([f for f in base_features if f in df.columns])}")
print(f"   Engineered: {len([f for f in engineered_features if f in df.columns])}")
print(f"   Signals: {len([f for f in signal_features if f in df.columns])}")

# =========================
# SYMBOL-WISE TRAINING
# =========================
print("\n" + "="*70)
print("TRAINING MODELS BY SYMBOL")
print("="*70)

output_rows = []
training_stats = []
failed_symbols = []
skipped_symbols = []

# Get symbol counts
symbol_counts = df['symbol'].value_counts()
print(f"\n📊 Symbol distribution:")
print(f"   ≥{MIN_SAMPLES_PER_SYMBOL} rows: {(symbol_counts >= MIN_SAMPLES_PER_SYMBOL).sum()} symbols")
print(f"   <{MIN_SAMPLES_PER_SYMBOL} rows: {(symbol_counts < MIN_SAMPLES_PER_SYMBOL).sum()} symbols")

for symbol, group in df.groupby('symbol'):
    print(f"\n{'─'*50}")
    print(f"🔹 Symbol: {symbol}")
    print(f"📊 Total data points: {len(group)}")
    
    # Check minimum samples
    if len(group) < MIN_SAMPLES_PER_SYMBOL:
        print(f"⚠️ Skipped - insufficient data (need {MIN_SAMPLES_PER_SYMBOL} rows)")
        skipped_symbols.append(symbol)
        continue
    
    # Sort by date and use all data
    group = group.sort_values('date')
    X = group[features]
    y = group['target']
    
    # Check target distribution
    target_ratio = y.mean()
    print(f"🎯 Target ratio: {target_ratio:.2%}")
    
    if target_ratio < MIN_TARGET_RATIO or target_ratio > MAX_TARGET_RATIO:
        print(f"⚠️ Skipped - target ratio too extreme ({target_ratio:.2%})")
        skipped_symbols.append(symbol)
        continue
    
    model_path = os.path.join(MODEL_DIR, f'xgb_model_{symbol}.joblib')
    
    try:
        print(f"🔄 Training model...")
        
        # Use stratified split for better distribution
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Target ratio (train): {y_train.mean():.2%}")
        print(f"   Target ratio (test): {y_test.mean():.2%}")
        
        # Handle missing values - fill with median
        for col in X_train.columns:
            if X_train[col].isnull().any():
                median_val = X_train[col].median()
                X_train[col] = X_train[col].fillna(median_val)
                X_test[col] = X_test[col].fillna(median_val)
        
        # Train model
        model = xgb.XGBClassifier(**MODEL_PARAMS)
        
        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"✅ {symbol} - Accuracy: {acc:.2%}, AUC: {auc:.2%}")
        
        # Skip if model is poor
        if acc < MIN_ACCURACY_THRESHOLD:
            print(f"⚠️ Skipped - poor accuracy ({acc:.2%})")
            skipped_symbols.append(symbol)
            continue
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.head(20).to_csv(
            os.path.join(MODEL_DIR, f'feature_importance_{symbol}.csv'),
            index=False
        )
        
        # Store stats
        training_stats.append({
            'symbol': symbol,
            'samples': len(group),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'accuracy': acc,
            'auc': auc,
            'target_ratio': target_ratio
        })
        
        # Predict confidence for all data
        # Handle missing values in full X
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
        
        group['confidence_score'] = model.predict_proba(X)[:, 1] * 100
        group['prediction'] = (group['confidence_score'] > 50).astype(int)
        group['signal_strength'] = pd.cut(
            group['confidence_score'],
            bins=[0, 30, 50, 70, 100],
            labels=['Weak', 'Moderate', 'Strong', 'Very Strong']
        )
        
        output_rows.append(
            group[['symbol', 'date', 'close', 'confidence_score', 'prediction', 
                   'signal_strength']]
        )
        
    except Exception as e:
        print(f"❌ Error training {symbol}: {e}")
        failed_symbols.append(symbol)
        continue

# =========================
# SAVE OUTPUT
# =========================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

if output_rows:
    output_df = pd.concat(output_rows, ignore_index=True)
    output_path = './csv/xgb_confidence.csv'
    output_df.to_csv(output_path, index=False)
    print(f"✅ Prediction saved: {output_path}")
    print(f"📊 Total predictions: {len(output_df):,}")
    
    # Save training summary
    if training_stats:
        stats_df = pd.DataFrame(training_stats)
        stats_path = os.path.join(MODEL_DIR, 'training_summary.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"✅ Training summary saved: {stats_path}")
        
        print("\n📊 Training Summary:")
        print(stats_df.to_string(index=False))
    
    if skipped_symbols:
        print(f"\n⚠️ Skipped symbols: {len(skipped_symbols)}")
    if failed_symbols:
        print(f"❌ Failed symbols: {len(failed_symbols)}")
        
else:
    print("❌ No symbols were processed successfully!")

# =========================
# UPDATE RETRAIN DATE
# =========================
if retrain_needed and training_stats:
    with open(LAST_RETRAIN_FILE, 'w') as f:
        f.write(today.strftime('%Y-%m-%d'))
    print(f"\n✅ Retrain date updated: {today.strftime('%Y-%m-%d')}")
    print(f"📅 Next retrain: {(today + timedelta(days=RETRAIN_DAYS)).date()}")

print("\n" + "="*70)
print("RETRAINING PROCESS COMPLETED")
print("="*70)
print(f"✅ Models trained: {len(training_stats)}")
print(f"📁 Models saved in: {MODEL_DIR}")
print("🎉 Done!")