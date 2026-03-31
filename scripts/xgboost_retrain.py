# xgboost_retrain.py
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
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

# Model parameters
MODEL_PARAMS = {
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

# Minimum data requirements
MIN_SAMPLES_PER_SYMBOL = 100
TEST_SIZE = 0.2
MAX_RECENT_DAYS = 500  # Use last 500 days maximum

print("="*70)
print("XGBOOST MODEL RETRAINER")
print("="*70)
print(f"Retrain Type: {RETRAIN_TYPE.upper()}")
print(f"Retrain Days: {RETRAIN_DAYS}")
print(f"Model Directory: {MODEL_DIR}")
print("="*70)

# =========================
# CHECK RETRAIN
# =========================
def check_retrain_needed():
    """Check if retraining is needed based on last retrain date"""
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
print(f"📊 Days since last retrain: {days_since}")
print(f"🔄 Retrain needed: {'YES' if retrain_needed else 'NO'}")

if not retrain_needed:
    print(f"\n⏰ No retrain needed. Next retrain on: {(last_retrain_date + timedelta(days=RETRAIN_DAYS)).date()}")
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
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    print(f"✅ Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    print(f"📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    sys.exit(1)

# =========================
# FEATURE ENGINEERING
# =========================
print("\n🔧 Engineering features...")

# Bollinger Bands
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8).astype(int)

# MACD
df['macd_histogram'] = df['macd_hist']
df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                       (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                         (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
df['macd_hist_trend'] = df['macd_histogram'].diff(3).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

# RSI
df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
df['rsi_cross_above_30'] = ((df['rsi'] >= 30) & (df['rsi'].shift(1) < 30)).astype(int)
df['rsi_cross_below_70'] = ((df['rsi'] <= 70) & (df['rsi'].shift(1) > 70)).astype(int)
df['rsi_mid'] = ((df['rsi'] > 40) & (df['rsi'] < 60)).astype(int)

# ATR
df['atr_ratio'] = df['atr'] / df['close'] * 100
df['atr_percentile'] = df['atr_ratio'].rolling(50).rank(pct=True)

# Zigzag
df['zigzag_signal'] = df['zigzag'].fillna(0).astype(int)
df['zigzag_breakout'] = ((df['zigzag_signal'] == 1) & (df['zigzag_signal'].shift(1) == 0)).astype(int)

# Candlestick patterns
patterns = ['Hammer', 'BullishEngulfing', 'MorningStar', 'Doji', 'PiercingLine', 'ThreeWhiteSoldiers']
for p in patterns:
    if p in df.columns:
        df[p] = df[p].fillna(False).astype(int)

# EMAs
for span in [9, 12, 20, 26, 50, 100, 200]:
    df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

# EMA Crossovers
df['ema_cross_bullish'] = ((df['ema_9'] > df['ema_20']) & 
                           (df['ema_9'].shift(1) <= df['ema_20'].shift(1))).astype(int)
df['ema_cross_bearish'] = ((df['ema_9'] < df['ema_20']) & 
                           (df['ema_9'].shift(1) >= df['ema_20'].shift(1))).astype(int)

# Golden/Death Cross
df['golden_cross'] = ((df['ema_50'] > df['ema_200']) & 
                      (df['ema_50'].shift(1) <= df['ema_200'].shift(1))).astype(int)
df['death_cross'] = ((df['ema_50'] < df['ema_200']) & 
                     (df['ema_50'].shift(1) >= df['ema_200'].shift(1))).astype(int)

# EMA Alignment
df['ema_alignment'] = ((df['ema_9'] > df['ema_20']) & 
                       (df['ema_20'] > df['ema_50']) & 
                       (df['ema_50'] > df['ema_200'])).astype(int)

# Price relative to EMAs
df['price_above_ema9'] = (df['close'] > df['ema_9']).astype(int)
df['price_above_ema20'] = (df['close'] > df['ema_20']).astype(int)
df['price_above_ema50'] = (df['close'] > df['ema_50']).astype(int)
df['price_above_ema200'] = (df['close'] > df['ema_200']).astype(int)

# Distance from EMAs
df['dist_to_ema9'] = (df['close'] - df['ema_9']) / df['ema_9'] * 100
df['dist_to_ema20'] = (df['close'] - df['ema_20']) / df['ema_20'] * 100
df['dist_to_ema50'] = (df['close'] - df['ema_50']) / df['ema_50'] * 100

# EMA Slopes
df['ema9_slope'] = df['ema_9'].pct_change(3) * 100
df['ema20_slope'] = df['ema_20'].pct_change(3) * 100
df['ema50_slope'] = df['ema_50'].pct_change(3) * 100

# Returns
df['return'] = df['close'].pct_change()
df['return_3d'] = df['close'].pct_change(3)
df['return_5d'] = df['close'].pct_change(5)
df['return_10d'] = df['close'].pct_change(10)

# Volatility
df['volatility'] = (df['high'] - df['low']) / df['close']
df['volatility_5d'] = df['volatility'].rolling(5).mean()
df['volatility_10d'] = df['volatility'].rolling(10).mean()

# Volume
df['volume_ma'] = df['volume'].rolling(20).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma']
df['volume_ratio_5d'] = df['volume_ratio'].rolling(5).mean()
df['volume_spike'] = (df['volume'] > df['volume_ma'] * 1.5).astype(int)
df['volume_dry_up'] = (df['volume'] < df['volume_ma'] * 0.5).astype(int)
df['volume_trend'] = df['volume_ratio'].rolling(5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)

# VPT
df['vpt'] = (df['volume'] * df['return']).cumsum()
df['vpt_trend'] = df['vpt'].pct_change(5) * 100

# Support/Resistance
df['resistance_20d'] = df['high'].rolling(20).max()
df['support_20d'] = df['low'].rolling(20).min()
df['resistance_50d'] = df['high'].rolling(50).max()
df['support_50d'] = df['low'].rolling(50).min()
df['distance_to_resistance'] = (df['resistance_20d'] - df['close']) / df['close'] * 100
df['distance_to_support'] = (df['close'] - df['support_20d']) / df['close'] * 100

# Pivot points
df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
df['resistance_1'] = 2 * df['pivot'] - df['low']
df['support_1'] = 2 * df['pivot'] - df['high']

# RSI Divergence
df['rsi_bullish_divergence'] = ((df['low'] <= df['low'].shift(1)) & 
                                 (df['rsi'] > df['rsi'].shift(1))).astype(int)
df['rsi_bearish_divergence'] = ((df['high'] >= df['high'].shift(1)) & 
                                 (df['rsi'] < df['rsi'].shift(1))).astype(int)

# MACD Divergence
df['macd_bullish_divergence'] = ((df['low'] <= df['low'].shift(1)) & 
                                  (df['macd'] > df['macd'].shift(1))).astype(int)
df['macd_bearish_divergence'] = ((df['high'] >= df['high'].shift(1)) & 
                                  (df['macd'] < df['macd'].shift(1))).astype(int)

# Composite Scores
df['bullish_score'] = (
    df['rsi_cross_above_30'] + df['macd_cross_up'] + 
    df['Hammer'] + df['BullishEngulfing'] + df['MorningStar'] + 
    df['PiercingLine'] + df['ThreeWhiteSoldiers'] + 
    df['ema_cross_bullish'] + df['golden_cross'] + 
    df['rsi_bullish_divergence'] + df['macd_bullish_divergence']
)

df['bearish_score'] = (
    df['macd_cross_down'] + df['rsi_cross_below_70'] + 
    df['rsi_bearish_divergence'] + df['ema_cross_bearish'] + 
    df['death_cross'] + df['macd_bearish_divergence']
)

# Target
df['future_return'] = df['close'].shift(-5) / df['close'] - 1
df['target'] = (df['future_return'] > 0.02).astype(int)  # Lowered threshold for more signals

df = df.dropna()

print(f"✅ After feature engineering: {len(df)} rows")

# =========================
# FEATURES
# =========================
features = [
    'change', 'marketCap', 'bb_width', 'bb_position', 'bb_squeeze',
    'macd', 'macd_signal', 'macd_histogram', 'macd_cross_up', 'macd_cross_down', 'macd_hist_trend',
    'rsi', 'rsi_oversold', 'rsi_overbought', 'rsi_cross_above_30', 'rsi_cross_below_70', 'rsi_mid',
    'atr', 'atr_ratio', 'atr_percentile', 'zigzag_signal', 'zigzag_breakout',
    'Hammer', 'BullishEngulfing', 'MorningStar', 'Doji', 'PiercingLine', 'ThreeWhiteSoldiers',
    'ema_9', 'ema_12', 'ema_20', 'ema_26', 'ema_50', 'ema_100', 'ema_200',
    'ema_cross_bullish', 'ema_cross_bearish', 'golden_cross', 'death_cross', 'ema_alignment',
    'price_above_ema9', 'price_above_ema20', 'price_above_ema50', 'price_above_ema200',
    'dist_to_ema9', 'dist_to_ema20', 'dist_to_ema50',
    'ema9_slope', 'ema20_slope', 'ema50_slope',
    'return', 'return_3d', 'return_5d', 'return_10d',
    'volatility', 'volatility_5d', 'volatility_10d',
    'volume_ratio', 'volume_ratio_5d', 'volume_spike', 'volume_dry_up', 'volume_trend',
    'vpt', 'vpt_trend',
    'distance_to_resistance', 'distance_to_support', 'resistance_1', 'support_1',
    'rsi_bullish_divergence', 'rsi_bearish_divergence',
    'macd_bullish_divergence', 'macd_bearish_divergence',
    'bullish_score', 'bearish_score'
]

# Filter features that exist
features = [f for f in features if f in df.columns]

print(f"✅ Total features: {len(features)}")

# =========================
# SYMBOL-WISE TRAINING
# =========================
print("\n" + "="*70)
print("TRAINING MODELS BY SYMBOL")
print("="*70)

output_rows = []
training_stats = []
failed_symbols = []

for symbol, group in df.groupby('symbol'):
    print(f"\n{'─'*50}")
    print(f"🔹 Symbol: {symbol}")
    print(f"📊 Total data points: {len(group)}")
    
    if len(group) < MIN_SAMPLES_PER_SYMBOL:
        print(f"⚠️ Skipped - insufficient data (need {MIN_SAMPLES_PER_SYMBOL} rows)")
        continue
    
    # Sort by date and take recent data
    group = group.sort_values('date').tail(MAX_RECENT_DAYS)
    print(f"📈 Using last {len(group)} rows for training")
    
    X = group[features]
    y = group['target']
    
    model_path = os.path.join(MODEL_DIR, f'xgb_model_{symbol}.joblib')
    
    try:
        print(f"🔄 Training model for {symbol}...")
        
        # Split data (time series split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, shuffle=False
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Target ratio (train): {y_train.mean():.2%}")
        print(f"   Target ratio (test): {y_test.mean():.2%}")
        
        # Train model
        model = xgb.XGBClassifier(**MODEL_PARAMS)
        
        # Fit with early stopping
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
            'target_ratio': y.mean()
        })
        
    except Exception as e:
        print(f"❌ Error training {symbol}: {e}")
        failed_symbols.append(symbol)
        continue
    
    # Predict confidence for all data
    group['confidence_score'] = model.predict_proba(X)[:, 1] * 100
    group['prediction'] = (group['confidence_score'] > 50).astype(int)
    group['signal_strength'] = pd.cut(
        group['confidence_score'],
        bins=[0, 30, 50, 70, 100],
        labels=['Weak', 'Moderate', 'Strong', 'Very Strong']
    )
    
    output_rows.append(
        group[['symbol', 'date', 'close', 'confidence_score', 'prediction', 
               'signal_strength', 'bullish_score', 'bearish_score']]
    )

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
    print(f"📊 Total predictions: {len(output_df)}")
    
    # Save training summary
    if training_stats:
        stats_df = pd.DataFrame(training_stats)
        stats_path = os.path.join(MODEL_DIR, 'training_summary.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"✅ Training summary saved: {stats_path}")
        
        print("\n📊 Training Summary:")
        print(stats_df.to_string(index=False))
    
    if failed_symbols:
        print(f"\n⚠️ Failed symbols: {failed_symbols}")
else:
    print("❌ No symbols were processed successfully!")

# =========================
# UPDATE RETRAIN DATE
# =========================
if retrain_needed and output_rows:
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