import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# =========================
# CONFIG
# =========================
DATA_PATH = './csv/mongodb.csv'
MODEL_DIR = './csv/xgboost/'
OUTPUT_PATH = './csv/xgb_confidence.csv'

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])

# =========================
# FEATURE ENGINEERING
# =========================
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

df['macd_histogram'] = df['macd_hist']
df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                       (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                         (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
df['rsi_cross_above_30'] = ((df['rsi'] >= 30) & (df['rsi'].shift(1) < 30)).astype(int)
df['rsi_cross_below_70'] = ((df['rsi'] <= 70) & (df['rsi'].shift(1) > 70)).astype(int)

df['atr_ratio'] = df['atr'] / df['close'] * 100
df['zigzag_signal'] = df['zigzag'].fillna(0).astype(int)

candle_patterns = ['Hammer','BullishEngulfing','MorningStar','Doji','PiercingLine','ThreeWhiteSoldiers']
for pattern in candle_patterns:
    if pattern in df.columns:
        df[pattern] = df[pattern].fillna(False).astype(int)

# EMA
for span in [9,12,20,26,50,200]:
    df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

df['ema_cross_bullish'] = ((df['ema_9'] > df['ema_20']) & (df['ema_9'].shift(1) <= df['ema_20'].shift(1))).astype(int)
df['ema_cross_bearish'] = ((df['ema_9'] < df['ema_20']) & (df['ema_9'].shift(1) >= df['ema_20'].shift(1))).astype(int)

df['golden_cross'] = ((df['ema_50'] > df['ema_200']) & (df['ema_50'].shift(1) <= df['ema_200'].shift(1))).astype(int)
df['death_cross'] = ((df['ema_50'] < df['ema_200']) & (df['ema_50'].shift(1) >= df['ema_200'].shift(1))).astype(int)

df['price_above_ema9'] = (df['close'] > df['ema_9']).astype(int)
df['price_above_ema20'] = (df['close'] > df['ema_20']).astype(int)
df['price_above_ema50'] = (df['close'] > df['ema_50']).astype(int)
df['price_above_ema200'] = (df['close'] > df['ema_200']).astype(int)

df['dist_to_ema9'] = (df['close'] - df['ema_9']) / df['ema_9'] * 100
df['dist_to_ema20'] = (df['close'] - df['ema_20']) / df['ema_20'] * 100
df['dist_to_ema50'] = (df['close'] - df['ema_50']) / df['ema_50'] * 100

df['ema9_slope'] = df['ema_9'].pct_change(3) * 100
df['ema20_slope'] = df['ema_20'].pct_change(3) * 100
df['ema50_slope'] = df['ema_50'].pct_change(3) * 100

# Returns & volatility
df['return'] = df['close'].pct_change()
df['return_5d'] = df['close'].pct_change(5)
df['return_10d'] = df['close'].pct_change(10)

df['volatility'] = (df['high'] - df['low']) / df['close']
df['volatility_5d'] = df['volatility'].rolling(5).mean()
df['volatility_10d'] = df['volatility'].rolling(10).mean()

# Volume
df['volume_ma'] = df['volume'].rolling(20).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma']
df['volume_ratio_5d'] = df['volume_ratio'].rolling(5).mean()
df['volume_spike'] = (df['volume'] > df['volume_ma'] * 1.5).astype(int)
df['volume_dry_up'] = (df['volume'] < df['volume_ma'] * 0.5).astype(int)

df['vpt'] = (df['volume'] * df['return']).cumsum()

# Support/Resistance
df['resistance_20d'] = df['high'].rolling(20).max()
df['support_20d'] = df['low'].rolling(20).min()
df['distance_to_resistance'] = (df['resistance_20d'] - df['close']) / df['close'] * 100
df['distance_to_support'] = (df['close'] - df['support_20d']) / df['close'] * 100

df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
df['resistance_1'] = 2 * df['pivot'] - df['low']
df['support_1'] = 2 * df['pivot'] - df['high']

# Divergence
df['rsi_bullish_divergence'] = ((df['low'] <= df['low'].shift(1)) & (df['rsi'] > df['rsi'].shift(1))).astype(int)
df['rsi_bearish_divergence'] = ((df['high'] >= df['high'].shift(1)) & (df['rsi'] < df['rsi'].shift(1))).astype(int)

# Scores
df['bullish_score'] = df['rsi_cross_above_30'] + df['macd_cross_up'] + df['Hammer'] + df['BullishEngulfing'] + df['MorningStar'] + df['PiercingLine'] + df['ThreeWhiteSoldiers'] + df['ema_cross_bullish'] + df['golden_cross'] + df['rsi_bullish_divergence']

df['bearish_score'] = df['macd_cross_down'] + df['rsi_cross_below_70'] + df['rsi_bearish_divergence'] + df['ema_cross_bearish'] + df['death_cross']

# Target
df['future_return'] = df['close'].shift(-5) / df['close'] - 1
df['target'] = (df['future_return'] > 0.03).astype(int)

df = df.dropna()

# =========================
# FEATURES
# =========================
features = [col for col in df.columns if col not in ['symbol','date','future_return','target']]

# =========================
# SYMBOL-WISE TRAINING
# =========================
all_outputs = []

for symbol, group in df.groupby('symbol'):
    print(f"\n🔹 Processing {symbol}")

    if len(group) < 100:
        print(f"⚠️ Skipped {symbol} (not enough data)")
        continue

    X = group[features]
    y = group['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ {symbol} Accuracy: {acc:.2f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, f"xgb_{symbol}.joblib")
    joblib.dump(model, model_path)

    # Predict
    group['confidence_score'] = model.predict_proba(X)[:, 1] * 100

    all_outputs.append(
        group[['symbol','date','close','confidence_score','bullish_score','bearish_score']]
    )

# =========================
# SAVE OUTPUT
# =========================
final_df = pd.concat(all_outputs)
final_df.to_csv(OUTPUT_PATH, index=False)

print("\n✅ All models trained & saved")
print(f"📁 Models folder: {MODEL_DIR}")
print(f"📊 Output saved: {OUTPUT_PATH}")