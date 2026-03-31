import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('./csv/mongodb.csv')
df['date'] = pd.to_datetime(df['date'])

# =========================
# FEATURE ENGINEERING
# =========================

# RSI
delta = df['close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# Price change
df['return'] = df['close'].pct_change()

# Volatility
df['volatility'] = (df['high'] - df['low']) / df['close']

# Volume strength
df['volume_ma'] = df['volume'].rolling(20).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma']

# =========================
# ADD YOUR SIGNAL FEATURES
# =========================

# Example (replace with your real output)
df['wave_score'] = np.random.randint(1, 6, len(df))   # wave 1–5
df['divergence'] = np.random.randint(0, 2, len(df))   # 0 বা 1

# =========================
# TARGET (LABEL)
# =========================
# Future return (next 5 days)
df['future_return'] = df['close'].shift(-5) / df['close'] - 1

# Profit = 1, Loss = 0
df['target'] = (df['future_return'] > 0.03).astype(int)

df = df.dropna()

# =========================
# FEATURES
# =========================
features = [
    'rsi',
    'return',
    'volatility',
    'volume_ratio',
    'wave_score',
    'divergence'
]

X = df[features]
y = df['target']

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =========================
# TRAIN MODEL
# =========================
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")

# =========================
# PREDICT CONFIDENCE
# =========================
df['confidence_score'] = model.predict_proba(X)[:, 1] * 100

# =========================
# SAVE OUTPUT
# =========================
df[['symbol', 'date', 'close', 'confidence_score']].to_csv(
    './output/ai_signal/xgb_confidence.csv',
    index=False
)

print("✅ XGBoost confidence generated!")