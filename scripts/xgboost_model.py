# xgboost_model.py - XGBoost Training with Sector Features
# ✅ Sector momentum, relative strength, peer comparison
# ✅ Sector rotation detection
# ✅ Sector-weighted confidence adjustment
# ✅ Telegram notifications

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import requests
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =========================
# TELEGRAM NOTIFICATION (NEW)
# =========================

def send_telegram_message(message, token=None, chat_id=None):
    """Send message to Telegram"""
    token = token or os.getenv("TELEGRAM_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        return False
    
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        response = requests.post(url, json=payload, timeout=10)
        return response.json()
    except:
        return False

# =========================
# CONFIG
# =========================
DATA_PATH = './csv/mongodb.csv'
MODEL_DIR = './csv/xgboost/'
OUTPUT_PATH = './csv/xgb_confidence.csv'
RETRAIN_DAYS = 7  # Retrain every 7 days
LAST_RETRAIN_FILE = './csv/last_retrain.txt'
SECTOR_PERFORMANCE_FILE = './csv/sector_performance.csv'  # ✅ NEW

os.makedirs(MODEL_DIR, exist_ok=True)

print("="*60)
print("XGBOOST MODEL TRAINING (with Sector Features)")
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

# ✅ NEW: Check if sector column exists
if 'sector' in df.columns:
    print(f"✅ Sector column found: {df['sector'].nunique()} unique sectors")
else:
    print("⚠️ Sector column not found, sector features disabled")
    df['sector'] = 'Unknown'

# =========================
# SECTOR ANALYZER (NEW)
# =========================

class SectorAnalyzer:
    """Analyze sector performance and generate sector features"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.sector_stats = {}
        self.sector_momentum = {}
        self.sector_ranks = {}
        self._calculate_sector_stats()
    
    def _calculate_sector_stats(self):
        """Calculate sector-level statistics"""
        if 'sector' not in self.data.columns:
            return
        
        for sector in self.data['sector'].unique():
            if pd.isna(sector) or sector == 'Unknown':
                continue
            
            sector_df = self.data[self.data['sector'] == sector]
            
            # Calculate average returns by date
            sector_returns = sector_df.groupby('date')['close'].mean().pct_change()
            
            self.sector_stats[sector] = {
                'avg_return_5d': sector_df.groupby('symbol')['close'].apply(
                    lambda x: (x.iloc[-1] - x.iloc[-5]) / x.iloc[-5] if len(x) >= 5 else 0
                ).mean(),
                'avg_return_20d': sector_df.groupby('symbol')['close'].apply(
                    lambda x: (x.iloc[-1] - x.iloc[-20]) / x.iloc[-20] if len(x) >= 20 else 0
                ).mean(),
                'avg_volume_growth': sector_df.groupby('symbol')['volume'].apply(
                    lambda x: (x.iloc[-5:].mean() / x.iloc[-20:-5].mean() - 1) if len(x) >= 20 else 0
                ).mean(),
                'symbol_count': sector_df['symbol'].nunique(),
                'total_rows': len(sector_df)
            }
            
            # Sector momentum (20-day return)
            self.sector_momentum[sector] = self.sector_stats[sector]['avg_return_20d']
        
        # Calculate sector ranks
        if self.sector_momentum:
            sorted_sectors = sorted(self.sector_momentum.items(), key=lambda x: x[1], reverse=True)
            for rank, (sector, _) in enumerate(sorted_sectors, 1):
                self.sector_ranks[sector] = rank
    
    def get_sector_features(self, symbol, current_price, date):
        """Get sector-based features for a symbol"""
        if 'sector' not in self.data.columns:
            return {
                'sector_momentum': 0,
                'sector_relative_strength': 0,
                'sector_rank': 0.5,
                'sector_trend': 0,
                'sector_peer_return': 0
            }
        
        # Get symbol's sector
        symbol_data = self.data[self.data['symbol'] == symbol]
        if len(symbol_data) == 0:
            return {'sector_momentum': 0, 'sector_relative_strength': 0, 'sector_rank': 0.5, 'sector_trend': 0, 'sector_peer_return': 0}
        
        sector = symbol_data.iloc[0].get('sector', 'Unknown')
        if pd.isna(sector) or sector == 'Unknown':
            return {'sector_momentum': 0, 'sector_relative_strength': 0, 'sector_rank': 0.5, 'sector_trend': 0, 'sector_peer_return': 0}
        
        # Get sector stats
        stats = self.sector_stats.get(sector, {})
        sector_momentum = self.sector_momentum.get(sector, 0)
        
        # Calculate relative strength (symbol vs sector)
        sector_df = self.data[self.data['sector'] == sector]
        sector_avg_price = sector_df[sector_df['date'] == date]['close'].mean() if date in sector_df['date'].values else sector_df['close'].mean()
        relative_strength = (current_price / sector_avg_price - 1) if sector_avg_price > 0 else 0
        
        # Get sector rank (normalized 0-1)
        total_sectors = len(self.sector_ranks) if self.sector_ranks else 1
        rank = self.sector_ranks.get(sector, total_sectors)
        sector_rank_norm = 1 - (rank - 1) / total_sectors  # Higher is better
        
        # Sector trend
        sector_trend = 1 if sector_momentum > 0.02 else -1 if sector_momentum < -0.02 else 0
        
        # Peer average return (same sector, same date)
        peer_data = sector_df[sector_df['date'] == date]
        peer_return = peer_data['close'].pct_change().mean() if len(peer_data) > 1 else 0
        
        return {
            'sector_momentum': np.clip(sector_momentum, -0.5, 0.5),
            'sector_relative_strength': np.clip(relative_strength, -0.5, 0.5),
            'sector_rank': sector_rank_norm,
            'sector_trend': sector_trend,
            'sector_peer_return': np.clip(peer_return, -0.1, 0.1)
        }
    
    def get_sector_rotation_signal(self):
        """Detect sector rotation"""
        if len(self.sector_momentum) < 2:
            return {}
        
        # Find top and bottom sectors
        sorted_sectors = sorted(self.sector_momentum.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_sectors[:3]
        bottom_3 = sorted_sectors[-3:]
        
        return {
            'top_sectors': [s[0] for s in top_3],
            'bottom_sectors': [s[0] for s in bottom_3],
            'rotation_strength': top_3[0][1] - bottom_3[0][1]
        }
    
    def save_sector_performance(self):
        """Save sector performance to CSV"""
        if not self.sector_stats:
            return
        
        rows = []
        for sector, stats in self.sector_stats.items():
            rows.append({
                'sector': sector,
                'momentum': self.sector_momentum.get(sector, 0),
                'rank': self.sector_ranks.get(sector, 0),
                'symbol_count': stats['symbol_count'],
                'avg_return_5d': stats['avg_return_5d'],
                'avg_return_20d': stats['avg_return_20d']
            })
        
        df_sector = pd.DataFrame(rows)
        df_sector.to_csv(SECTOR_PERFORMANCE_FILE, index=False)
        print(f"📊 Sector performance saved: {SECTOR_PERFORMANCE_FILE}")

# Initialize Sector Analyzer
sector_analyzer = SectorAnalyzer(df)
print(f"📊 Sector Analyzer initialized with {len(sector_analyzer.sector_ranks)} sectors")

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

# ✅ NEW: Sector Features
print("🔧 Adding sector features...")
sector_features = []
for idx, row in df.iterrows():
    if idx % 1000 == 0:
        print(f"   Processing row {idx}/{len(df)}...")
    
    features = sector_analyzer.get_sector_features(row['symbol'], row['close'], row['date'])
    sector_features.append(features)

# Convert to DataFrame and merge
sector_df = pd.DataFrame(sector_features)
df = pd.concat([df, sector_df], axis=1)
print(f"✅ Sector features added: {list(sector_df.columns)}")

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
    df['macd_bullish_divergence'] +
    (df['sector_trend'] > 0).astype(int) * 2 +  # ✅ NEW: Sector trend bonus
    (df['sector_rank'] > 0.7).astype(int) * 2    # ✅ NEW: High sector rank bonus
)

df['bearish_score'] = (
    df['macd_cross_down'] + 
    df['rsi_cross_below_70'] + 
    df['rsi_bearish_divergence'] + 
    df['ema_cross_bearish'] + 
    df['death_cross'] +
    df['macd_bearish_divergence'] +
    (df['sector_trend'] < 0).astype(int) * 2 +  # ✅ NEW: Sector trend penalty
    (df['sector_rank'] < 0.3).astype(int) * 2    # ✅ NEW: Low sector rank penalty
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
print(f"   Including sector features: {[f for f in features if 'sector' in f]}")

# =========================
# SYMBOL-WISE TRAINING
# =========================
all_outputs = []
training_stats = []
sector_accuracy = defaultdict(list)  # ✅ NEW: Track accuracy by sector

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
        
        # ✅ NEW: Track accuracy by sector
        symbol_sector = group.iloc[0].get('sector', 'Unknown')
        sector_accuracy[symbol_sector].append(acc)

        # Save model
        joblib.dump(model, model_path)

        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        importance_df.head(15).to_csv(  # ✅ NEW: Save top 15
            os.path.join(MODEL_DIR, f"feature_importance_{symbol}.csv"), 
            index=False
        )

        training_stats.append({
            'symbol': symbol,
            'sector': symbol_sector,  # ✅ NEW
            'samples': len(group),
            'accuracy': acc,
            'profit_ratio': y.mean()
        })

    else:
        print(f"📂 Loading existing model for {symbol}")
        model = joblib.load(model_path)

    # Predict confidence
    group['confidence_score'] = model.predict_proba(X)[:, 1] * 100

    # ✅ NEW: Adjust confidence by sector momentum
    sector_momentum = group['sector_momentum'].iloc[0] if 'sector_momentum' in group.columns else 0
    sector_rank = group['sector_rank'].iloc[0] if 'sector_rank' in group.columns else 0.5
    
    # Boost confidence for symbols in strong sectors
    if sector_momentum > 0.02 and sector_rank > 0.7:
        group['confidence_score'] = group['confidence_score'] * 1.1
    elif sector_momentum < -0.02 or sector_rank < 0.3:
        group['confidence_score'] = group['confidence_score'] * 0.9
    
    group['confidence_score'] = group['confidence_score'].clip(0, 100)

    # Add prediction direction
    group['prediction'] = (group['confidence_score'] > 50).astype(int)

    # Add signal strength
    group['signal_strength'] = pd.cut(
        group['confidence_score'],
        bins=[0, 30, 50, 70, 100],
        labels=['Weak', 'Moderate', 'Strong', 'Very Strong']
    )

    # ✅ NEW: Add sector info to output
    output_cols = ['symbol', 'date', 'close', 'confidence_score', 'prediction', 
                   'signal_strength', 'bullish_score', 'bearish_score']
    
    if 'sector' in group.columns:
        output_cols.append('sector')
    if 'sector_momentum' in group.columns:
        output_cols.append('sector_momentum')
    if 'sector_rank' in group.columns:
        output_cols.append('sector_rank')
    
    all_outputs.append(group[output_cols])

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

    # ✅ NEW: Save sector performance
    sector_analyzer.save_sector_performance()
    
    # ✅ NEW: Print sector rotation signal
    rotation = sector_analyzer.get_sector_rotation_signal()
    if rotation:
        print(f"\n🔄 Sector Rotation Signal:")
        print(f"   Top Sectors: {', '.join(rotation['top_sectors'])}")
        print(f"   Bottom Sectors: {', '.join(rotation['bottom_sectors'])}")
        print(f"   Rotation Strength: {rotation['rotation_strength']:.3f}")

    # ✅ NEW: Print accuracy by sector
    if sector_accuracy:
        print(f"\n📊 Model Accuracy by Sector:")
        sector_acc_summary = []
        for sector, accs in sector_accuracy.items():
            avg_acc = np.mean(accs)
            sector_acc_summary.append({'sector': sector, 'avg_accuracy': avg_acc, 'models': len(accs)})
        
        sector_acc_df = pd.DataFrame(sector_acc_summary).sort_values('avg_accuracy', ascending=False)
        for _, row in sector_acc_df.head(5).iterrows():
            print(f"   {row['sector']}: {row['avg_accuracy']:.2%} ({row['models']} models)")

    # Print training stats
    if training_stats:
        stats_df = pd.DataFrame(training_stats)
        print(f"\n📊 Training Summary:")
        print(stats_df.to_string(index=False))
        
        # ✅ NEW: Send Telegram summary
        avg_acc = stats_df['accuracy'].mean()
        total_models = len(stats_df)
        message = f"""
✅ <b>XGBoost Training Complete</b>
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
🤖 Models trained: {total_models}
📊 Avg Accuracy: {avg_acc:.2%}
🏆 Best Model: {stats_df.loc[stats_df['accuracy'].idxmax(), 'symbol']} ({stats_df['accuracy'].max():.2%})
"""
        send_telegram_message(message)

    # Update last retrain date
    if retrain_needed:
        with open(LAST_RETRAIN_FILE, 'w') as f:
            f.write(pd.Timestamp.now().strftime('%Y-%m-%d'))
        print(f"✅ Updated retrain date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")

else:
    print("❌ No symbols were processed!")

print("\n🎉 Process completed successfully!")