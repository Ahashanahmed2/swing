# xgboost_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIG
# =========================
DATA_PATH = './csv/mongodb.csv'
MODEL_DIR = './csv/xgboost/'
OUTPUT_PATH = './csv/xgb_confidence.csv'
RETRAIN_DAYS = 7  # Retrain every 7 days
LAST_RETRAIN_FILE = './csv/last_retrain.txt'

os.makedirs(MODEL_DIR, exist_ok=True)

print("="*60)
print("XGBOOST MODEL TRAINING")
print("="*60)

# =========================
# CHECK IF RETRAIN NEEDED
# =========================
def check_retrain_needed():
    """Check if retraining is needed based on last retrain date"""
    try:
        if os.path.exists(LAST_RETRAIN_FILE):
            with open(LAST_RETRAIN_FILE, 'r') as f:
                last_date = pd.to_datetime(f.read().strip())
                days_diff = (pd.Timestamp.now() - last_date).days
                return days_diff >= RETRAIN_DAYS
    except:
        pass
    return True  # Retrain if file doesn't exist or error

retrain_needed = check_retrain_needed()
print(f"🔄 Retrain needed: {retrain_needed}")

# =========================
# LOAD DATA
# =========================
print("\n📂 Loading data...")
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
print(f"✅ Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")

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
df['macd_histogram_trend'] = df['macd_histogram'].diff(3).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

# RSI
df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
df['rsi_cross_above_30'] = ((df['rsi'] >= 30) & (df['rsi'].shift(1) < 30)).astype(int)
df['rsi_cross_below_70'] = ((df['rsi'] <= 70) & (df['rsi'].shift(1) > 70)).astype(int)
df['rsi_mid'] = (df['rsi'] > 40) & (df['rsi'] < 60).astype(int)

# ATR
df['atr_ratio'] = df['atr'] / df['close'] * 100
df['atr_percentile'] = df['atr_ratio'].rolling(50).rank(pct=True)

# Zigzag
df['zigzag_signal'] = df['zigzag'].fillna(0).astype(int)
df['zigzag_breakout'] = ((df['zigzag_signal'] == 1) & (df['zigzag_signal'].shift(1) == 0)).astype(int)

# Candlestick patterns
candle_patterns = ['Hammer', 'BullishEngulfing', 'MorningStar', 'Doji', 'PiercingLine', 'ThreeWhiteSoldiers']
for pattern in candle_patterns:
    if pattern in df.columns:
        df[pattern] = df[pattern].fillna(False).astype(int)

# EMA
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

# Scores
df['bullish_score'] = (
    df['rsi_cross_above_30'] + 
    df['macd_cross_up'] + 
    df['Hammer'] + 
    df['BullishEngulfing'] + 
    df['MorningStar'] + 
    df['PiercingLine'] + 
    df['ThreeWhiteSoldiers'] + 
    df['ema_cross_bullish'] + 
    df['golden_cross'] + 
    df['rsi_bullish_divergence'] +
    df['macd_bullish_divergence']
)

df['bearish_score'] = (
    df['macd_cross_down'] + 
    df['rsi_cross_below_70'] + 
    df['rsi_bearish_divergence'] + 
    df['ema_cross_bearish'] + 
    df['death_cross'] +
    df['macd_bearish_divergence']
)

# Target (Next 5 days return > 2%)
df['future_return'] = df['close'].shift(-5) / df['close'] - 1
df['target'] = (df['future_return'] > 0.02).astype(int)  # Lowered threshold for more signals

df = df.dropna()

print(f"✅ After feature engineering: {len(df)} rows")

# =========================
# FEATURES
# =========================
# Exclude non-feature columns
exclude_cols = ['symbol', 'date', 'future_return', 'target']
features = [col for col in df.columns if col not in exclude_cols]

print(f"✅ Total features: {len(features)}")

# =========================
# SYMBOL-WISE TRAINING
# =========================
all_outputs = []
training_stats = []

for symbol, group in df.groupby('symbol'):
    print(f"\n{'='*50}")
    print(f"🔹 Processing {symbol}")
    print(f"📊 Data points: {len(group)}")
    
    if len(group) < 100:
        print(f"⚠️ Skipped {symbol} (minimum 100 rows required)")
        continue
    
    # Sort by date
    group = group.sort_values('date')
    
    X = group[features]
    y = group['target']
    
    # Check model file
    model_path = os.path.join(MODEL_DIR, f"xgb_{symbol}.joblib")
    
    # Train if retrain needed or model doesn't exist
    if retrain_needed or not os.path.exists(model_path):
        print(f"🔄 Training new model for {symbol}")
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Use last 20% for testing
        test_size = int(len(group) * 0.2)
        X_train = X[:-test_size]
        y_train = y[:-test_size]
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        
        print(f"📈 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"🎯 Target distribution - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            eval_metric='logloss',
            random_state=42
        )
        
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
        
        print(f"✅ {symbol} - Accuracy: {acc:.2f}")
        print(f"📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Loss', 'Profit']))
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.head(10).to_csv(
            os.path.join(MODEL_DIR, f"feature_importance_{symbol}.csv"), 
            index=False
        )
        
        training_stats.append({
            'symbol': symbol,
            'samples': len(group),
            'accuracy': acc,
            'profit_ratio': y.mean()
        })
        
    else:
        print(f"📂 Loading existing model for {symbol}")
        model = joblib.load(model_path)
    
    # Predict confidence
    group['confidence_score'] = model.predict_proba(X)[:, 1] * 100
    
    # Add prediction direction
    group['prediction'] = (group['confidence_score'] > 50).astype(int)
    
    # Add signal strength
    group['signal_strength'] = pd.cut(
        group['confidence_score'],
        bins=[0, 30, 50, 70, 100],
        labels=['Weak', 'Moderate', 'Strong', 'Very Strong']
    )
    
    all_outputs.append(
        group[['symbol', 'date', 'close', 'confidence_score', 'prediction', 
               'signal_strength', 'bullish_score', 'bearish_score']]
    )

# =========================
# SAVE OUTPUT
# =========================
if all_outputs:
    final_df = pd.concat(all_outputs, ignore_index=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"📁 Models folder: {MODEL_DIR}")
    print(f"📊 Output saved: {OUTPUT_PATH}")
    print(f"📈 Total predictions: {len(final_df)}")
    
    # Print training stats
    if training_stats:
        stats_df = pd.DataFrame(training_stats)
        print(f"\n📊 Training Summary:")
        print(stats_df.to_string(index=False))
    
    # Update last retrain date
    if retrain_needed:
        with open(LAST_RETRAIN_FILE, 'w') as f:
            f.write(pd.Timestamp.now().strftime('%Y-%m-%d'))
        print(f"✅ Updated retrain date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
        
else:
    print("❌ No symbols were processed!")

print("\n🎉 Process completed successfully!")