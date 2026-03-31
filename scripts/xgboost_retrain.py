import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime

# =========================
# CONFIG
# =========================
DATA_PATH = './csv/mongodb.csv'
MODEL_DIR = './csv/xgboost/'
os.makedirs(MODEL_DIR, exist_ok=True)

LAST_RETRAIN_FILE = './csv/last_retrain.txt'
RETRAIN_TYPE = 'weekly'
RETRAIN_DAYS = 7 if RETRAIN_TYPE == 'weekly' else 30

# =========================
# CHECK RETRAIN
# =========================
if os.path.exists(LAST_RETRAIN_FILE):
    with open(LAST_RETRAIN_FILE, 'r') as f:
        last_retrain_date = datetime.strptime(f.read().strip(), '%Y-%m-%d')
else:
    last_retrain_date = datetime(2000, 1, 1)

today = datetime.today()
days_since = (today - last_retrain_date).days
retrain_needed = days_since >= RETRAIN_DAYS

print(f"Last retrain: {last_retrain_date.date()} | Days: {days_since}")
print(f"Retrain needed: {retrain_needed}")

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
df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
df['rsi_cross_above_30'] = ((df['rsi'] >= 30) & (df['rsi'].shift(1) < 30)).astype(int)
df['rsi_cross_below_70'] = ((df['rsi'] <= 70) & (df['rsi'].shift(1) > 70)).astype(int)

df['atr_ratio'] = df['atr'] / df['close'] * 100
df['zigzag_signal'] = df['zigzag'].fillna(0).astype(int)

patterns = ['Hammer','BullishEngulfing','MorningStar','Doji','PiercingLine','ThreeWhiteSoldiers']
for p in patterns:
    if p in df.columns:
        df[p] = df[p].fillna(False).astype(int)

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

df['return'] = df['close'].pct_change()
df['return_5d'] = df['close'].pct_change(5)
df['return_10d'] = df['close'].pct_change(10)

df['volatility'] = (df['high'] - df['low']) / df['close']
df['volatility_5d'] = df['volatility'].rolling(5).mean()
df['volatility_10d'] = df['volatility'].rolling(10).mean()

df['volume_ma'] = df['volume'].rolling(20).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma']
df['volume_ratio_5d'] = df['volume_ratio'].rolling(5).mean()
df['volume_spike'] = (df['volume'] > df['volume_ma'] * 1.5).astype(int)
df['volume_dry_up'] = (df['volume'] < df['volume_ma'] * 0.5).astype(int)

df['vpt'] = (df['volume'] * df['return']).cumsum()

df['resistance_20d'] = df['high'].rolling(20).max()
df['support_20d'] = df['low'].rolling(20).min()
df['distance_to_resistance'] = (df['resistance_20d'] - df['close']) / df['close'] * 100
df['distance_to_support'] = (df['close'] - df['support_20d']) / df['close'] * 100

df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
df['resistance_1'] = 2 * df['pivot'] - df['low']
df['support_1'] = 2 * df['pivot'] - df['high']

df['rsi_bullish_divergence'] = ((df['low'] <= df['low'].shift(1)) & (df['rsi'] > df['rsi'].shift(1))).astype(int)
df['rsi_bearish_divergence'] = ((df['high'] >= df['high'].shift(1)) & (df['rsi'] < df['rsi'].shift(1))).astype(int)

df['bullish_score'] = df['rsi_cross_above_30'] + df['macd_cross_up'] + df['Hammer'] + df['BullishEngulfing'] + df['MorningStar'] + df['PiercingLine'] + df['ThreeWhiteSoldiers'] + df['ema_cross_bullish'] + df['golden_cross'] + df['rsi_bullish_divergence']

df['bearish_score'] = df['macd_cross_down'] + df['rsi_cross_below_70'] + df['rsi_bearish_divergence'] + df['ema_cross_bearish'] + df['death_cross']

df['future_return'] = df['close'].shift(-5) / df['close'] - 1
df['target'] = (df['future_return'] > 0.03).astype(int)

df = df.dropna()

# =========================
# FEATURES
# =========================
features = [
    'change','marketCap','bb_width','bb_position','macd','macd_signal','macd_histogram',
    'macd_cross_up','macd_cross_down','rsi','rsi_oversold','rsi_overbought',
    'rsi_cross_above_30','rsi_cross_below_70','atr','atr_ratio','zigzag_signal',
    'Hammer','BullishEngulfing','MorningStar','Doji','PiercingLine','ThreeWhiteSoldiers',
    'ema_9','ema_12','ema_20','ema_26','ema_50','ema_200',
    'ema_cross_bullish','ema_cross_bearish','golden_cross','death_cross',
    'price_above_ema9','price_above_ema20','price_above_ema50','price_above_ema200',
    'dist_to_ema9','dist_to_ema20','dist_to_ema50','ema9_slope','ema20_slope','ema50_slope',
    'distance_to_resistance','distance_to_support','resistance_1','support_1',
    'return','return_5d','return_10d','volatility','volatility_5d','volatility_10d',
    'volume_ratio','volume_ratio_5d','volume_spike','volume_dry_up','vpt',
    'rsi_bullish_divergence','rsi_bearish_divergence','bullish_score','bearish_score'
]

# SAFE FILTER
features = [f for f in features if f in df.columns]

# =========================
# SYMBOL LOOP
# =========================
output_rows = []

for symbol, group in df.groupby('symbol'):
    group = group.sort_values('date').tail(300)  # recent data
    group = group.copy()

    X = group[features]
    y = group['target']

    model_path = os.path.join(MODEL_DIR, f'xgb_model_{symbol}.joblib')

    if retrain_needed or not os.path.exists(model_path):
        print(f"🔹 Training {symbol}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, gamma=0.1,
            reg_alpha=0.1, reg_lambda=1, random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{symbol} Accuracy: {acc:.2f}")

        joblib.dump(model, model_path)

    else:
        print(f"➡️ Using existing model for {symbol}")
        model = joblib.load(model_path)

    group['confidence_score'] = model.predict_proba(X)[:,1] * 100

    output_rows.append(
        group[['symbol','date','close','confidence_score','bullish_score','bearish_score']]
    )

# =========================
# SAVE OUTPUT
# =========================
output_df = pd.concat(output_rows)
output_df.to_csv('./csv/xgb_confidence.csv', index=False)

print("✅ Prediction saved")

# =========================
# UPDATE RETRAIN DATE
# =========================
if retrain_needed:
    with open(LAST_RETRAIN_FILE, 'w') as f:
        f.write(today.strftime('%Y-%m-%d'))
    print("✅ Retrain date updated")