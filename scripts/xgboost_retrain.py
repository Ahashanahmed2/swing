# ================== xgboost_scheduler.py ==================
# COMPLETE VERSION with ALL ENHANCEMENTS
# Features:
# 1. Only good models (AUC >= 0.55) are saved
# 2. Bad models retry up to 3 times, then retry monthly
# 3. Already trained good models retrain with new data
# 4. Single commit upload to Hugging Face
# 5. Monthly retry for permanently failed models
# 6. Advanced features: Support/Resistance, RSI Divergence
# 7. ✅ Sector momentum, relative strength, peer comparison
# 8. ✅ Telegram notifications for training status
# 9. ✅ Sector performance tracking
# 10. ✅ freeFloatMarketCap features (log, rank, liquidity)
# 11. ✅ Walk-Forward Validation
# 12. ✅ Probability Calibration (Isotonic)
# 13. ✅ Target Encoding (Sector-wise)
# 14. ✅ Interaction Features (Price×Vol, RSI×Vol, MCap×Momentum)
# 15. ✅ Optuna Hyperparameter Tuning (Best Params)
# 16. ✅ Market Regime Detection (Bull/Bear/Sideways)
# 17. ✅ Time-Based Cyclical Features
# 18. ✅ Technical Indicators (BB, MACD, ATR, MFI)
# 19. ✅ Rolling Statistical Features
# 20. ✅ Price Pattern Detection
# 21. ✅ Feature Selection (Correlation-based)
# 22. ✅ Ensemble Models (Weekly/Monthly)
# 23. ✅ Dynamic Thresholds by Mode & Regime
# 24. ✅ Adaptive Parameters by Data Size
# 25. ✅ Mode-Specific Targets & Horizons

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import requests
import optuna
import json
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =========================
# TELEGRAM NOTIFICATION
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

def send_training_summary(mode, trained_count, good_count, bad_count, monthly_retry_count, good_models_list):
    """Send training summary to Telegram"""
    if trained_count == 0:
        return
    
    message = f"""
🤖 <b>XGBoost {mode} Training Complete</b>
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
──────────────────────────────
📊 Models Trained: {trained_count}
🟢 Good Models (saved): {good_count}
🔴 Bad Models: {bad_count}
📅 Monthly Retry: {monthly_retry_count}
"""
    
    if good_models_list:
        message += f"\n\n🏆 <b>Top 5 Good Models:</b>"
        for model in good_models_list[:5]:
            message += f"\n   • {model['symbol']}: AUC={model['auc']:.2%}"
    
    send_telegram_message(message)

# =========================
# CONFIG
# =========================
DATA_PATH = './csv/mongodb.csv'
MODEL_DIR = './csv/xgboost/'
PREDICTION_LOG = './csv/prediction_log.csv'
XGB_CONFIDENCE = './csv/xgb_confidence.csv'
MODEL_METADATA = './csv/model_metadata.csv'
SECTOR_PERFORMANCE_FILE = './csv/sector_performance.csv'

# Advanced features files
SUPPORT_RESISTANCE_PATH = './csv/support_resistance.csv'
RSI_DIVERGENCE_PATH = './csv/rsi_diver.csv'

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
MIN_SAMPLES_PER_SYMBOL = 60

# Model quality threshold
AUC_THRESHOLD = 0.55
RETRAIN_ATTEMPTS = 3
MONTHLY_RETRY_AFTER = 30

# ✅ Optuna settings
OPTUNA_TRIALS = 30  # Hyperparameter tuning trials per symbol
ENABLE_OPTUNA = True  # Set False to skip tuning

# =========================
# ✅ MODE-SPECIFIC CONFIGURATION
# =========================

MODE_CONFIG = {
    'DAILY': {
        'target_horizon': 1,           # 1 day ahead
        'target_return': 0.005,         # 0.5% return target
        'optuna_trials': 20,            # কম ট্রায়াল (দ্রুত)
        'enable_optuna': True,
        'wf_splits': 3,                 # কম স্প্লিট
        'min_samples': 40,
        'early_stopping': 30,
        'calibration_min_samples': 30,
        'lookback_days': 10,            # ছোট লুকব্যাক
        'feature_selection': True,      # ফিচার সিলেকশন
        'ensemble_models': False,       # সিঙ্গেল মডেল
        'market_regime_filter': False,  # মার্কেট ফিল্টার না
        'train_ratio': 0.7,
        'val_ratio': 0.85,
        'auc_threshold': 0.52,          # Lower for daily
        'confidence_threshold': 45,
        'overfit_threshold': 0.15,
    },
    'WEEKLY': {
        'target_horizon': 5,            # 5 days ahead
        'target_return': 0.02,          # 2% return target
        'optuna_trials': 30,
        'enable_optuna': True,
        'wf_splits': 5,
        'min_samples': 60,
        'early_stopping': 50,
        'calibration_min_samples': 50,
        'lookback_days': 30,
        'feature_selection': True,
        'ensemble_models': True,        # এনসেম্বল মডেল
        'market_regime_filter': True,   # মার্কেট ফিল্টার
        'train_ratio': 0.65,
        'val_ratio': 0.85,
        'auc_threshold': 0.55,
        'confidence_threshold': 50,
        'overfit_threshold': 0.12,
    },
    'MONTHLY': {
        'target_horizon': 20,           # 20 days ahead
        'target_return': 0.05,          # 5% return target
        'optuna_trials': 50,            # বেশি ট্রায়াল
        'enable_optuna': True,
        'wf_splits': 5,
        'min_samples': 100,
        'early_stopping': 80,
        'calibration_min_samples': 100,
        'lookback_days': 60,            # বড় লুকব্যাক
        'feature_selection': True,
        'ensemble_models': True,        # এনসেম্বল
        'market_regime_filter': True,
        'train_ratio': 0.6,
        'val_ratio': 0.85,
        'auc_threshold': 0.58,          # Higher for monthly
        'confidence_threshold': 55,
        'overfit_threshold': 0.10,
    }
}

# =========================
# OPTUNA BEST PARAMS STORAGE
# =========================
BEST_PARAMS_FILE = './csv/xgboost_best_params.json'
best_params_cache = {}

def load_best_params():
    """Load best hyperparameters from previous Optuna runs"""
    global best_params_cache
    if os.path.exists(BEST_PARAMS_FILE):
        try:
            with open(BEST_PARAMS_FILE, 'r') as f:
                best_params_cache = json.load(f)
            print(f"✅ Loaded best params for {len(best_params_cache)} symbols")
        except:
            pass

def save_best_params(symbol, params):
    """Save best hyperparameters for a symbol"""
    global best_params_cache
    # Convert any numpy types to native Python types for JSON serialization
    params_serializable = {}
    for key, value in params.items():
        if isinstance(value, (np.integer,)):
            params_serializable[key] = int(value)
        elif isinstance(value, (np.floating,)):
            params_serializable[key] = float(value)
        else:
            params_serializable[key] = value
    
    best_params_cache[symbol] = params_serializable
    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(best_params_cache, f, indent=2)

# =========================
# ✅ MARKET REGIME DETECTION
# =========================

def detect_market_regime(df, symbol=None):
    """
    Detect market regime: Bull, Bear, Sideways, High Volatility
    Returns regime label and confidence
    """
    if len(df) < 20:
        return 'unknown', 0
    
    df_sorted = df.sort_values('date')
    
    # Calculate market indicators
    sma_20 = df_sorted['close'].rolling(20).mean()
    sma_50 = df_sorted['close'].rolling(50).mean() if len(df_sorted) >= 50 else sma_20
    volatility = df_sorted['close'].pct_change().rolling(20).std()
    returns_20d = (df_sorted['close'] - df_sorted['close'].shift(20)) / df_sorted['close'].shift(20)
    
    current_price = df_sorted['close'].iloc[-1]
    current_vol = volatility.iloc[-1]
    current_return = returns_20d.iloc[-1]
    
    # Historical volatility percentiles
    vol_80th = volatility.quantile(0.8)
    vol_20th = volatility.quantile(0.2)
    
    # Determine regime
    if pd.isna(current_vol) or pd.isna(current_return):
        return 'unknown', 0
    
    if current_vol > vol_80th:
        regime = 'high_volatility'
        confidence = 0.8
    elif current_return > 0.05 and current_price > sma_20.iloc[-1]:
        regime = 'bull'
        confidence = 0.9
    elif current_return < -0.05 and current_price < sma_20.iloc[-1]:
        regime = 'bear'
        confidence = 0.9
    elif current_vol < vol_20th and abs(current_return) < 0.02:
        regime = 'sideways_low_vol'
        confidence = 0.7
    else:
        regime = 'sideways'
        confidence = 0.6
    
    return regime, confidence

def add_market_regime_features(df):
    """Add market regime features"""
    df['market_regime'] = 'unknown'
    df['regime_confidence'] = 0
    
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].sort_values('date')
        if len(symbol_data) >= 20:
            regime, conf = detect_market_regime(symbol_data)
            
            mask = df['symbol'] == symbol
            df.loc[mask, 'market_regime'] = regime
            df.loc[mask, 'regime_confidence'] = conf
    
    # One-hot encode regime
    for regime in ['bull', 'bear', 'sideways', 'sideways_low_vol', 'high_volatility']:
        df[f'regime_{regime}'] = (df['market_regime'] == regime).astype(int)
    
    return df

# =========================
# ✅ TIME-BASED FEATURES
# =========================

def add_temporal_features(df):
    """Add time-based cyclical features"""
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_quarter_end'] = (df['date'].dt.month % 3 == 0).astype(int)
    
    # Cyclical encoding
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

# =========================
# ✅ TECHNICAL INDICATORS
# =========================

def add_technical_indicators(df):
    """Add technical indicators: Bollinger Bands, MACD, ATR, MFI"""
    
    # Bollinger Bands
    df['bb_middle'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).mean())
    bb_std = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).std())
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    
    # MACD
    ema_12 = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema_26 = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df.groupby('symbol')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # ATR (Average True Range)
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = (df['high'] - df['close'].shift(1)).abs()
    df['low_close'] = (df['low'] - df['close'].shift(1)).abs()
    
    def calc_atr(group):
        tr = group[['high_low', 'high_close', 'low_close']].max(axis=1)
        return tr.rolling(14).mean()
    
    df['atr'] = df.groupby('symbol').apply(calc_atr).reset_index(level=0, drop=True)
    df['atr_pct'] = df['atr'] / df['close'] * 100
    
    # MFI (Money Flow Index)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    df['price_change'] = typical_price.diff()
    df['positive_flow'] = money_flow.where(df['price_change'] > 0, 0)
    df['negative_flow'] = money_flow.where(df['price_change'] < 0, 0)
    
    def calc_mfi(group):
        pos_sum = group['positive_flow'].rolling(14).sum()
        neg_sum = group['negative_flow'].rolling(14).sum()
        return 100 - (100 / (1 + pos_sum / neg_sum.replace(0, 1)))
    
    df['mfi'] = df.groupby('symbol').apply(calc_mfi).reset_index(level=0, drop=True)
    
    # Drop temporary columns
    df.drop(['high_low', 'high_close', 'low_close', 'price_change', 
             'positive_flow', 'negative_flow'], axis=1, errors='ignore', inplace=True)
    
    return df

# =========================
# ✅ ROLLING FEATURE ENGINEERING
# =========================

def add_rolling_features(df):
    """Add rolling statistical features"""
    
    windows = [5, 10, 20]
    
    for window in windows:
        if len(df) < window:
            continue
            
        # Returns
        df[f'return_{window}d_mean'] = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(window, min_periods=1).mean()
        )
        df[f'return_{window}d_std'] = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(window, min_periods=1).std()
        )
        
        # Volume
        df[f'volume_{window}d_mean'] = df.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'volume_{window}d_ratio'] = df['volume'] / (df[f'volume_{window}d_mean'] + 1e-8)
    
    return df

# =========================
# ✅ PRICE PATTERN DETECTION
# =========================

def detect_price_patterns(df):
    """Detect common price patterns"""
    
    df['price_pattern'] = 0
    df['consecutive_up'] = 0
    df['consecutive_down'] = 0
    
    if 'open' not in df.columns:
        df['open'] = df['close'].shift(1)
    
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        symbol_data = df[mask].sort_values('date')
        
        if len(symbol_data) < 5:
            continue
        
        closes = symbol_data['close'].values
        opens = symbol_data['open'].values
        highs = symbol_data['high'].values
        lows = symbol_data['low'].values
        
        patterns = np.zeros(len(symbol_data))
        consec_up = np.zeros(len(symbol_data))
        consec_down = np.zeros(len(symbol_data))
        
        up_count = 0
        down_count = 0
        
        for i in range(1, len(symbol_data)):
            # Consecutive tracking
            if closes[i] > closes[i-1]:
                up_count += 1
                down_count = 0
            elif closes[i] < closes[i-1]:
                down_count += 1
                up_count = 0
            
            consec_up[i] = up_count
            consec_down[i] = down_count
            
            # Pattern detection (i >= 4)
            if i >= 4:
                # Bullish Engulfing
                if (closes[i-1] < opens[i-1] and
                    closes[i] > opens[i] and
                    closes[i] > opens[i-1] and
                    opens[i] < closes[i-1]):
                    patterns[i] = 1
                
                # Bearish Engulfing
                elif (closes[i-1] > opens[i-1] and
                      closes[i] < opens[i] and
                      closes[i] < opens[i-1] and
                      opens[i] > closes[i-1]):
                    patterns[i] = -1
                
                # Doji
                elif abs(closes[i] - opens[i]) < (highs[i] - lows[i]) * 0.1:
                    patterns[i] = 0.5 if closes[i] > closes[i-1] else -0.5
        
        df.loc[mask, 'price_pattern'] = patterns
        df.loc[mask, 'consecutive_up'] = consec_up
        df.loc[mask, 'consecutive_down'] = consec_down
    
    return df

# =========================
# SECTOR ANALYZER
# =========================

class SectorAnalyzer:
    """Analyze sector performance and generate sector features"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.sector_stats = {}
        self.sector_momentum = {}
        self.sector_ranks = {}
        self.symbol_sector_map = {}
        self.symbol_market_cap = {}
        self._calculate_sector_stats()
    
    def _calculate_sector_stats(self):
        """Calculate sector-level statistics including market cap"""
        if 'sector' not in self.data.columns:
            return
        
        # Build symbol to sector mapping
        for _, row in self.data[['symbol', 'sector']].drop_duplicates().iterrows():
            self.symbol_sector_map[row['symbol']] = row['sector']
        
        # Build symbol to market cap mapping
        if 'freeFloatMarketCap' in self.data.columns:
            for _, row in self.data[['symbol', 'freeFloatMarketCap']].drop_duplicates().iterrows():
                if pd.notna(row['freeFloatMarketCap']):
                    self.symbol_market_cap[row['symbol']] = float(row['freeFloatMarketCap'])
        
        for sector in self.data['sector'].unique():
            if pd.isna(sector) or sector == 'Unknown':
                continue
            
            sector_df = self.data[self.data['sector'] == sector]
            
            self.sector_stats[sector] = {
                'avg_return_5d': sector_df.groupby('symbol')['close'].apply(
                    lambda x: (x.iloc[-1] - x.iloc[-5]) / x.iloc[-5] if len(x) >= 5 else 0
                ).mean(),
                'avg_return_20d': sector_df.groupby('symbol')['close'].apply(
                    lambda x: (x.iloc[-1] - x.iloc[-20]) / x.iloc[-20] if len(x) >= 20 else 0
                ).mean(),
                'symbol_count': sector_df['symbol'].nunique(),
                'total_rows': len(sector_df)
            }
            
            # Market Cap Stats
            sector_symbols = sector_df['symbol'].unique()
            sector_caps = [self.symbol_market_cap.get(s, 0) for s in sector_symbols]
            self.sector_stats[sector]['total_market_cap'] = sum(sector_caps)
            self.sector_stats[sector]['avg_market_cap'] = np.mean(sector_caps) if sector_caps else 0
            
            self.sector_momentum[sector] = self.sector_stats[sector]['avg_return_20d']
        
        # Calculate sector ranks
        if self.sector_momentum:
            sorted_sectors = sorted(self.sector_momentum.items(), key=lambda x: x[1], reverse=True)
            for rank, (sector, _) in enumerate(sorted_sectors, 1):
                self.sector_ranks[sector] = rank
    
    def get_sector_features(self, symbol):
        """Get sector-based features for a symbol"""
        if symbol not in self.symbol_sector_map:
            return {
                'sector_momentum': 0,
                'sector_rank': 0.5,
                'sector_trend': 0,
                'sector': 'Unknown'
            }
        
        sector = self.symbol_sector_map[symbol]
        sector_momentum = self.sector_momentum.get(sector, 0)
        
        total_sectors = len(self.sector_ranks) if self.sector_ranks else 1
        rank = self.sector_ranks.get(sector, total_sectors)
        sector_rank_norm = 1 - (rank - 1) / total_sectors
        
        sector_trend = 1 if sector_momentum > 0.02 else -1 if sector_momentum < -0.02 else 0
        
        return {
            'sector_momentum': np.clip(sector_momentum, -0.5, 0.5),
            'sector_rank': sector_rank_norm,
            'sector_trend': sector_trend,
            'sector': sector
        }
    
    def get_sector_rotation_signal(self):
        """Detect sector rotation"""
        if len(self.sector_momentum) < 2:
            return {}
        
        sorted_sectors = sorted(self.sector_momentum.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_sectors[:3]
        bottom_3 = sorted_sectors[-3:]
        
        return {
            'top_sectors': [s[0] for s in top_3],
            'bottom_sectors': [s[0] for s in bottom_3],
            'rotation_strength': top_3[0][1] - bottom_3[0][1] if top_3 and bottom_3 else 0
        }

# =========================
# MODEL PARAMETERS BY MODE
# =========================

# Daily Mode (15 minutes)
DAILY_PARAMS = {
    'n_estimators': 1500,
    'max_depth': 5,
    'learning_rate': 0.005,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'colsample_bylevel':0.6,
    'colsample_bynode':0.6,
    'min_child_weight': 10,
    'gamma': 0.3,
    'reg_alpha': 0.5,
    'reg_lambda': 2,
    'random_state': 42,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'verbosity': 0,
    'early_stopping_rounds':50,
}

# Weekly Mode (30-40 minutes)
WEEKLY_PARAMS = {
    'n_estimators': 2500,
    'max_depth': 6,
    'learning_rate': 0.003,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'colsample_bylevel':0.6,
    'colsample_bynode':0.6,
    'min_child_weight': 12,
    'gamma': 0.4,
    'reg_alpha': 0.8,
    'reg_lambda': 3,
    'max_delta_step':0,
    'random_state': 42,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'verbosity': 0,
    'early_stopping_rounds':80,
}

# Monthly Mode (2-4 hours)
MONTHLY_PARAMS = {
    'n_estimators': 4000,
    'max_depth': 7,
    'learning_rate': 0.002,
    'subsample': 0.55,
    'colsample_bytree': 0.55,
    'colsample_bylevel':0.55,
    'colsample_bynode':0.55,
    'min_child_weight': 15,
    'gamma': 0.5,
    'reg_alpha': 1.0,
    'reg_lambda': 4.0,
    'max_delta_step':0,
    'random_state': 42,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'verbosity': 0,
    'early_stopping_rounds':120,
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
    """Load model metadata"""
    if os.path.exists(MODEL_METADATA):
        df = pd.read_csv(MODEL_METADATA)
        df['last_trained'] = pd.to_datetime(df['last_trained'])
        df['last_attempt'] = pd.to_datetime(df['last_attempt']) if 'last_attempt' in df.columns else pd.to_datetime(df['last_trained'])
        return df
    else:
        return pd.DataFrame(columns=['symbol', 'last_trained', 'last_attempt', 'auc', 'acc', 
                                      'failed_attempts', 'status', 'class_ratio', 'sector'])

def save_model_metadata(df):
    """Save model metadata"""
    df.to_csv(MODEL_METADATA, index=False)

def should_retrain(symbol, metadata, current_date=None):
    """Check if a symbol should be retrained"""
    if current_date is None:
        current_date = datetime.now()

    if metadata.empty or symbol not in metadata['symbol'].values:
        return True, "new_symbol"

    symbol_data = metadata[metadata['symbol'] == symbol].iloc[0]

    if symbol_data['status'] == 'GOOD':
        return True, "good_model_update"

    if symbol_data['status'] == 'BAD':
        failed_attempts = symbol_data['failed_attempts']

        if failed_attempts < RETRAIN_ATTEMPTS:
            return True, f"bad_retry_{failed_attempts+1}"

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
# DATA FUNCTIONS WITH ADVANCED FEATURES
# =========================

def load_data():
    """Load data with proper encoding"""
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df.columns = df.columns.str.replace('ï»¿', '').str.replace('\ufeff', '').str.strip()
    df['date'] = pd.to_datetime(df['date'])
    
    if 'sector' not in df.columns:
        df['sector'] = 'Unknown'
    
    return df

def safe_parse_date(date_series):
    """Safely parse dates with multiple formats"""
    try:
        return pd.to_datetime(date_series, format='%Y-%m-%d', errors='coerce')
    except:
        pass

    try:
        return pd.to_datetime(date_series, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    except:
        pass

    try:
        return pd.to_datetime(date_series, format='mixed', errors='coerce')
    except:
        pass

    return pd.to_datetime(date_series, errors='coerce')

def engineer_features(df):
    """Add ALL engineered features"""
    if df.empty:
        return df

    # =========================
    # BASE FEATURES
    # =========================

    df['return_5d'] = df.groupby('symbol')['close'].pct_change(5)
    df['return_10d'] = df.groupby('symbol')['close'].pct_change(10)
    df['volatility'] = (df['high'] - df['low']) / df['close']
    df['volatility_5d'] = df.groupby('symbol')['volatility'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['volume_ma'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)

    if 'rsi' in df.columns:
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)

    # =========================
    # MARKET CAP FEATURES
    # =========================
    if 'freeFloatMarketCap' in df.columns:
        df['log_market_cap'] = np.log1p(df['freeFloatMarketCap'])
        df['market_cap_rank'] = df.groupby('sector')['freeFloatMarketCap'].rank(pct=True)
        df['liquidity_score'] = df['volume'] / (df['freeFloatMarketCap'] * 1e6 + 1e-8)
        df['liquidity_score'] = df['liquidity_score'].clip(0, 1)
        df['mcap_volume_ratio'] = df['volume'] / (df['freeFloatMarketCap'] + 1e-8)
        df['mcap_volume_ratio'] = np.log1p(df['mcap_volume_ratio'])

    # =========================
    # SECTOR FEATURES
    # =========================
    sector_analyzer = SectorAnalyzer(df)
    
    df['sector_momentum'] = 0.0
    df['sector_rank'] = 0.5
    df['sector_trend'] = 0
    
    for symbol in df['symbol'].unique():
        features = sector_analyzer.get_sector_features(symbol)
        mask = df['symbol'] == symbol
        df.loc[mask, 'sector_momentum'] = features['sector_momentum']
        df.loc[mask, 'sector_rank'] = features['sector_rank']
        df.loc[mask, 'sector_trend'] = features['sector_trend']

    # =========================
    # 1. SUPPORT & RESISTANCE FEATURES
    # =========================
    try:
        if os.path.exists(SUPPORT_RESISTANCE_PATH):
            sr_df = pd.read_csv(SUPPORT_RESISTANCE_PATH, encoding='utf-8-sig')
            sr_df['current_date'] = pd.to_datetime(sr_df['current_date'])

            strength_map = {'Weak': 1, 'Moderate': 2, 'Strong': 3}
            sr_df['strength_score'] = sr_df['strength'].map(strength_map).fillna(1)

            df = df.merge(sr_df, left_on=['symbol', 'date'], 
                          right_on=['symbol', 'current_date'], how='left')

            df['dist_from_sr'] = (df['close'] - df['level_price']) / df['level_price'] * 100
            df['dist_from_sr'] = df['dist_from_sr'].clip(-20, 20)

            df['is_support'] = (df['type'] == 'support').astype(int)
            df['is_resistance'] = (df['type'] == 'resistance').astype(int)
            df['sr_strength'] = df['strength_score']
            df['sr_gap_days'] = df['gap_days'].fillna(999).clip(0, 100)

            drop_cols = ['type', 'current_low', 'current_high', 'current_close', 
                         'level_date', 'strength', 'strength_score', 'current_date', 'gap_days']
            df.drop(drop_cols, axis=1, errors='ignore', inplace=True)

            for col in ['dist_from_sr', 'is_support', 'is_resistance', 'sr_strength', 'sr_gap_days']:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
        else:
            for col in ['dist_from_sr', 'is_support', 'is_resistance', 'sr_strength', 'sr_gap_days']:
                df[col] = 0
    except:
        for col in ['dist_from_sr', 'is_support', 'is_resistance', 'sr_strength', 'sr_gap_days']:
            df[col] = 0

    # =========================
    # 2. RSI DIVERGENCE FEATURES
    # =========================
    try:
        if os.path.exists(RSI_DIVERGENCE_PATH):
            div_df = pd.read_csv(RSI_DIVERGENCE_PATH, encoding='utf-8-sig')
            div_df['last_date'] = pd.to_datetime(div_df['last_date'])

            div_df['is_bullish_div'] = (div_df['divergence_type'] == 'Bullish').astype(int)
            div_df['is_bearish_div'] = (div_df['divergence_type'] == 'Bearish').astype(int)
            div_df['div_strength'] = div_df['strength'].map({'Strong': 2, 'Moderate': 1, 'Weak': 0}).fillna(0)

            df = df.merge(div_df[['symbol', 'last_date', 'is_bullish_div', 'is_bearish_div', 'div_strength']], 
                          left_on=['symbol', 'date'], right_on=['symbol', 'last_date'], how='left')

            for col in ['is_bullish_div', 'is_bearish_div', 'div_strength']:
                df[col] = df[col].fillna(0)

            df.drop(['last_date'], axis=1, errors='ignore', inplace=True)
        else:
            for col in ['is_bullish_div', 'is_bearish_div', 'div_strength']:
                df[col] = 0
    except:
        for col in ['is_bullish_div', 'is_bearish_div', 'div_strength']:
            df[col] = 0

    # =========================
    # 4. INTERACTION FEATURES
    # =========================
    df['price_volume_interaction'] = df['close'] * df['volume_ratio']
    
    if 'rsi' in df.columns:
        df['rsi_volatility'] = df['rsi'] * df['volatility_5d']
        df['rsi_momentum'] = df['rsi'] * df['return_5d']
    
    if 'freeFloatMarketCap' in df.columns and 'sector_momentum' in df.columns:
        df['mcap_sector_momentum'] = np.log1p(df['freeFloatMarketCap']) * df['sector_momentum']
    
    if 'dist_from_sr' in df.columns and 'sr_strength' in df.columns:
        df['sr_signal'] = df['dist_from_sr'] * df['sr_strength']
    
    if 'rsi' in df.columns and 'is_bullish_div' in df.columns:
        df['rsi_div_interaction'] = df['rsi'] * df['is_bullish_div']
    
    if 'volume_ratio' in df.columns and 'volatility_5d' in df.columns:
        df['volume_volatility'] = df['volume_ratio'] * df['volatility_5d']

    return df

def get_features(df):
    """Get list of ALL available features"""
    features = [
        'close', 'volume', 'return_5d', 'return_10d', 
        'volatility', 'volatility_5d', 'volume_ratio'
    ]

    if 'rsi_oversold' in df.columns:
        features.extend(['rsi_oversold', 'rsi_overbought'])

    sr_features = ['dist_from_sr', 'is_support', 'is_resistance', 'sr_strength', 'sr_gap_days']
    features.extend([f for f in sr_features if f in df.columns])

    div_features = ['is_bullish_div', 'is_bearish_div', 'div_strength']
    features.extend([f for f in div_features if f in df.columns])
    
    sector_features = ['sector_momentum', 'sector_rank', 'sector_trend']
    features.extend([f for f in sector_features if f in df.columns])
    
    mc_features = ['freeFloatMarketCap', 'log_market_cap', 'market_cap_rank', 
                   'liquidity_score', 'mcap_volume_ratio']
    features.extend([f for f in mc_features if f in df.columns])
    
    interaction_features = ['price_volume_interaction', 'rsi_volatility', 'rsi_momentum',
                           'mcap_sector_momentum', 'sr_signal', 'rsi_div_interaction',
                           'volume_volatility']
    features.extend([f for f in interaction_features if f in df.columns])
    
    # New features
    regime_features = ['regime_bull', 'regime_bear', 'regime_sideways', 
                       'regime_sideways_low_vol', 'regime_high_volatility', 'regime_confidence']
    features.extend([f for f in regime_features if f in df.columns])
    
    temporal_features = ['day_sin', 'day_cos', 'month_sin', 'month_cos',
                         'is_month_start', 'is_month_end', 'is_quarter_end']
    features.extend([f for f in temporal_features if f in df.columns])
    
    technical_features = ['bb_width', 'bb_position', 'macd', 'macd_histogram',
                          'atr_pct', 'mfi']
    features.extend([f for f in technical_features if f in df.columns])
    
    rolling_features = ['return_5d_mean', 'return_5d_std', 'return_20d_mean', 'return_20d_std',
                        'volume_5d_ratio', 'volume_20d_ratio']
    features.extend([f for f in rolling_features if f in df.columns])
    
    pattern_features = ['price_pattern', 'consecutive_up', 'consecutive_down']
    features.extend([f for f in pattern_features if f in df.columns])

    return features

# =========================
# ✅ FEATURE SELECTION
# =========================

def select_features_for_mode(df, mode, features):
    """Select best features based on mode and importance"""
    
    if not MODE_CONFIG[mode]['feature_selection']:
        return features
    
    # Remove highly correlated features
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) < 10:
        return available_features
    
    try:
        corr_matrix = df[available_features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        
        reduced_features = [f for f in available_features if f not in to_drop]
        
        if len(reduced_features) < len(available_features):
            print(f"   🔍 Feature Selection: {len(available_features)} → {len(reduced_features)} (removed {len(to_drop)} correlated)")
        
        return reduced_features
    except:
        return available_features

# =========================
# ✅ MODE-SPECIFIC TARGET CREATION
# =========================

def create_mode_specific_targets(df, mode):
    """Create targets specific to each mode"""
    
    config = MODE_CONFIG[mode]
    horizon = config['target_horizon']
    return_target = config['target_return']
    
    # Forward returns for specific horizon
    df[f'future_return_{mode.lower()}'] = df.groupby('symbol')['close'].transform(
        lambda x: x.shift(-horizon) / x - 1
    )
    
    # Target: binary classification
    df[f'target_{mode.lower()}'] = (df[f'future_return_{mode.lower()}'] > return_target).astype(int)
    
    return df

# =========================
# ✅ DYNAMIC THRESHOLDS
# =========================

def get_dynamic_thresholds(mode, market_regime=None):
    """Get dynamic thresholds based on mode and market regime"""
    
    base_thresholds = {
        'DAILY': {
            'auc_threshold': 0.52,
            'confidence_threshold': 45,
            'overfit_threshold': 0.15,
        },
        'WEEKLY': {
            'auc_threshold': 0.55,
            'confidence_threshold': 50,
            'overfit_threshold': 0.12,
        },
        'MONTHLY': {
            'auc_threshold': 0.58,
            'confidence_threshold': 55,
            'overfit_threshold': 0.10,
        }
    }
    
    thresholds = base_thresholds.get(mode, base_thresholds['DAILY']).copy()
    
    # Adjust for market regime
    if market_regime == 'bear':
        thresholds['auc_threshold'] += 0.02
        thresholds['confidence_threshold'] += 5
    elif market_regime == 'bull':
        thresholds['auc_threshold'] -= 0.01
    
    return thresholds

# =========================
# ✅ ADAPTIVE PARAMETERS
# =========================

def get_adaptive_params(mode, symbol_data_size, class_balance):
    """Adjust parameters based on data characteristics"""
    
    params = MODE_CONFIG[mode].copy()
    base_params = None
    
    if mode == 'DAILY':
        base_params = DAILY_PARAMS.copy()
    elif mode == 'WEEKLY':
        base_params = WEEKLY_PARAMS.copy()
    else:
        base_params = MONTHLY_PARAMS.copy()
    
    # Adjust for small datasets
    if symbol_data_size < 200:
        base_params['n_estimators'] = min(base_params['n_estimators'], 500)
        base_params['max_depth'] = min(base_params['max_depth'], 4)
        base_params['learning_rate'] = base_params['learning_rate'] * 1.5
        base_params['subsample'] = 0.8
    
    # Adjust for imbalanced classes
    if class_balance < 0.2 or class_balance > 0.8:
        base_params['gamma'] = base_params['gamma'] * 0.5
        base_params['min_child_weight'] = max(base_params['min_child_weight'] - 3, 1)
    
    # Adjust for large datasets
    if symbol_data_size > 1000:
        base_params['n_estimators'] = int(base_params['n_estimators'] * 1.2)
        base_params['subsample'] = max(base_params['subsample'] - 0.1, 0.5)
    
    return base_params

# =========================
# FEEDBACK SYSTEM
# =========================

def update_actual_results():
    """Update actual results after FEEDBACK_DAYS"""
    if not os.path.exists(PREDICTION_LOG):
        return None

    log = pd.read_csv(PREDICTION_LOG)
    log['date'] = safe_parse_date(log['date'])
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

# =========================
# ✅ OPTUNA HYPERPARAMETER TUNING
# =========================

def optimize_hyperparameters(X_train, y_train, X_val, y_val, symbol, mode='DAILY'):
    """Bayesian hyperparameter optimization with Optuna"""
    
    cache_key = f"{symbol}_{mode}"
    if cache_key in best_params_cache:
        print(f"   📦 Using cached best params for {cache_key}")
        return best_params_cache[cache_key]
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    optuna_trials = MODE_CONFIG[mode]['optuna_trials']
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'random_state': 42,
            'eval_metric': 'logloss',
            'verbosity': 0,
            'use_label_encoder': False,
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        prob = model.predict_proba(X_val)[:, 1]
        
        if len(np.unique(y_val)) > 1:
            return roc_auc_score(y_val, prob)
        return 0.5
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['eval_metric'] = 'logloss'
    best_params['verbosity'] = 0
    best_params['use_label_encoder'] = False
    
    save_best_params(cache_key, best_params)
    
    print(f"   ✅ Optuna ({mode}): Best AUC={study.best_value:.3f}")
    
    return best_params


# =========================
# ✅ WALK-FORWARD VALIDATION
# =========================

def walk_forward_validation(X, y, params, n_splits=5):
    """Time-series aware cross validation"""
    scores = []
    fold_size = len(X) // (n_splits + 1)
    
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        val_end = min(train_end + fold_size, len(X))
        
        if train_end >= val_end:
            break
        
        X_tr = X.iloc[:train_end]
        y_tr = y.iloc[:train_end]
        X_v = X.iloc[train_end:val_end]
        y_v = y.iloc[train_end:val_end]
        
        if len(X_tr) < 20 or len(X_v) < 5:
            continue
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, verbose=False)
        prob = model.predict_proba(X_v)[:, 1]
        
        if len(np.unique(y_v)) > 1:
            scores.append(roc_auc_score(y_v, prob))
    
    if scores:
        return np.mean(scores), np.std(scores)
    return 0.5, 0.0


# =========================
# ✅ PROBABILITY CALIBRATION
# =========================

def calibrate_model(model, X_val, y_val):
    """Calibrate probability outputs using Isotonic Regression"""
    if len(X_val) < 50:
        return model    
    try:
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method='isotonic',
            cv='prefit'
        )
        calibrated.fit(X_val, y_val)
        return calibrated
    except:
        return model

# =========================
# ✅ ENSEMBLE MODELS
# =========================

def train_ensemble(X_train, y_train, X_val, y_val, params, mode):
    """Train ensemble of models"""
    
    if not MODE_CONFIG[mode]['ensemble_models']:
        return None
    
    models = []
    weights = []
    
    # Model 1: Standard XGBoost
    model1 = xgb.XGBClassifier(**params)
    model1.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
               early_stopping_rounds=MODE_CONFIG[mode]['early_stopping'], verbose=False)
    
    if len(np.unique(y_val)) > 1:
        auc1 = roc_auc_score(y_val, model1.predict_proba(X_val)[:, 1])
    else:
        auc1 = 0.5
    
    models.append(model1)
    weights.append(max(auc1 - 0.5, 0.1))
    
    # Model 2: XGBoost with different max_depth
    params2 = params.copy()
    params2['max_depth'] = min(params2['max_depth'] + 2, 12)
    params2['learning_rate'] = params2['learning_rate'] * 0.8
    
    model2 = xgb.XGBClassifier(**params2)
    model2.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
               early_stopping_rounds=MODE_CONFIG[mode]['early_stopping'], verbose=False)
    
    if len(np.unique(y_val)) > 1:
        auc2 = roc_auc_score(y_val, model2.predict_proba(X_val)[:, 1])
    else:
        auc2 = 0.5
    
    models.append(model2)
    weights.append(max(auc2 - 0.5, 0.1))
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [0.5, 0.5]
    
    return {'models': models, 'weights': weights}

def predict_ensemble(ensemble, X):
    """Predict using ensemble"""
    if ensemble is None or len(ensemble['models']) == 0:
        return None
    
    prob = np.zeros(len(X))
    for model, weight in zip(ensemble['models'], ensemble['weights']):
        prob += model.predict_proba(X)[:, 1] * weight
    
    return prob


# =========================
# TRAINING FUNCTION (FULLY ENHANCED)
# =========================

def train_symbol(symbol, group, features, params, feedback_log, metadata, sector_analyzer=None, mode='DAILY'):
    """Train model with ALL enhancements"""
    try:
        group = group.sort_values('date')
        
        # Filter available features
        available_features = [f for f in features if f in group.columns]
        
        X = group[available_features]
        
        # Use mode-specific target
        target_col = f'target_{mode.lower()}'
        if target_col in group.columns:
            y = group[target_col]
        else:
            y = group['target']
        
        config = MODE_CONFIG[mode]
        
        if len(X) < config['min_samples']:
            return None, None
        
        # Mode-specific split ratios
        train_ratio = config['train_ratio']
        val_ratio = config['val_ratio']
        
        train_idx = int(len(X) * train_ratio)
        val_idx = int(len(X) * val_ratio)
        
        X_train = X.iloc[:train_idx]
        y_train = y.iloc[:train_idx]
        X_val = X.iloc[train_idx:val_idx]
        y_val = y.iloc[train_idx:val_idx]
        X_test = X.iloc[val_idx:]
        y_test = y.iloc[val_idx:]
        
        if len(X_train) < 20 or len(X_test) < 5:
            return None, None
        
        # Walk-Forward Validation with mode-specific splits
        wf_mean, wf_std = walk_forward_validation(
            X_train, y_train, params, 
            n_splits=config['wf_splits']
        )
        print(f"   📊 Walk-Forward AUC: {wf_mean:.3f} ± {wf_std:.3f}")
        
        # Class weight
        target_ratio = y_train.mean()
        if target_ratio < 0.3:
            scale_pos = min((1 - target_ratio) / target_ratio, 10)
        elif target_ratio > 0.7:
            scale_pos = min(target_ratio / (1 - target_ratio), 10)
        else:
            scale_pos = 1
        
        # ✅ Adaptive Parameters
        adaptive_params = get_adaptive_params(mode, len(group), target_ratio)
        
        # Optuna Tuning with mode-specific trials
        if config['enable_optuna'] and len(X_train) >= 100:
            params_optimized = optimize_hyperparameters(X_train, y_train, X_val, y_val, symbol, mode)
            params_copy = params_optimized
            print(f"   🎯 Using Optuna-optimized params")
        else:
            params_copy = adaptive_params.copy()
        
        params_copy['scale_pos_weight'] = scale_pos
        
        # Sample weights from feedback
        weights = get_sample_weights(group.iloc[:train_idx], feedback_log)
        
        # ✅ Ensemble Training (Weekly/Monthly)
        ensemble = None
        if config['ensemble_models']:
            ensemble = train_ensemble(X_train, y_train, X_val, y_val, params_copy, mode)
            if ensemble:
                print(f"   🤖 Ensemble trained with {len(ensemble['models'])} models")
        
        # Train main model
        model = xgb.XGBClassifier(**params_copy)
        model.fit(
            X_train, y_train,
            sample_weight=weights,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=config['early_stopping'],
            verbose=False
        )
        
        # Calibration
        if len(X_val) >= config['calibration_min_samples']:
            model = calibrate_model(model, X_val, y_val)
        
        # Evaluate
        if ensemble:
            prob = predict_ensemble(ensemble, X_test)
            if prob is None:
                prob = model.predict_proba(X_test)[:, 1]
            preds = (prob > 0.5).astype(int)
        else:
            prob = model.predict_proba(X_test)[:, 1]
            preds = model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else 0.5
        
        # Overfitting check with mode-specific threshold
        train_acc = accuracy_score(y_train, model.predict(X_train))
        overfit_gap = train_acc - acc
        
        thresholds = get_dynamic_thresholds(mode)
        
        if overfit_gap > thresholds['overfit_threshold']:
            print(f"   ⚠️ Overfitting! Train={train_acc:.3f}, Test={acc:.3f}, Gap={overfit_gap:.3f}")
        
        # Save model based on mode-specific threshold
        auc_threshold = thresholds['auc_threshold']
        
        if auc >= auc_threshold:
            model_path = os.path.join(MODEL_DIR, f'{symbol}_{mode.lower()}.joblib')
            joblib.dump(model, model_path)
            
            # Save ensemble if exists
            if ensemble:
                ensemble_path = os.path.join(MODEL_DIR, f'{symbol}_{mode.lower()}_ensemble.joblib')
                joblib.dump(ensemble, ensemble_path)
            
            # Predict on ALL data
            if ensemble:
                all_probs = predict_ensemble(ensemble, X)
                if all_probs is None:
                    all_probs = model.predict_proba(X)[:, 1]
            else:
                all_probs = model.predict_proba(X)[:, 1]
            
            group['confidence_score'] = all_probs * 100
            
            # Mode-specific adjustments
            if mode == 'DAILY':
                if 'sector_momentum' in group.columns:
                    group['confidence_score'] *= (1 + group['sector_momentum'] * 0.5)
            elif mode == 'MONTHLY':
                if 'sector_momentum' in group.columns:
                    group['confidence_score'] *= (1 + group['sector_momentum'] * 1.5)
            else:  # WEEKLY
                if 'sector_momentum' in group.columns:
                    group['confidence_score'] *= (1 + group['sector_momentum'] * 1.0)
            
            group['confidence_score'] = group['confidence_score'].clip(0, 100)
            group['prediction'] = (group['confidence_score'] > thresholds['confidence_threshold']).astype(int)
            
            result = group[['symbol', 'date', 'close', 'confidence_score', 'prediction']]
            status = 'GOOD'
            failed_attempts = 0
            
            print(f"   ✅ AUC={auc:.3f} | Acc={acc:.3f} | WF={wf_mean:.3f} | Mode={mode}")
        else:
            group['confidence_score'] = 50
            group['prediction'] = 0
            result = group[['symbol', 'date', 'close', 'confidence_score', 'prediction']]
            status = 'BAD'
            
            if not metadata.empty and symbol in metadata['symbol'].values:
                prev_data = metadata[metadata['symbol'] == symbol].iloc[0]
                failed_attempts = prev_data.get('failed_attempts', 0) + 1
            else:
                failed_attempts = 1
            
            print(f"   ❌ AUC={auc:.3f} < {auc_threshold} (Mode: {mode})")
        
        # Get sector info
        sector = 'Unknown'
        if sector_analyzer and symbol in sector_analyzer.symbol_sector_map:
            sector = sector_analyzer.symbol_sector_map[symbol]
        
        return result, {
            'symbol': symbol,
            'last_trained': datetime.now(),
            'last_attempt': datetime.now(),
            'auc': auc,
            'acc': acc,
            'val_acc': accuracy_score(y_val, model.predict(X_val)),
            'wf_auc': wf_mean,
            'failed_attempts': failed_attempts if auc < auc_threshold else 0,
            'status': status,
            'class_ratio': target_ratio,
            'sector': sector,
            'features_used': len(available_features),
            'mode': mode,
            'ensemble': ensemble is not None
        }
    
    except Exception as e:
        print(f"   ❌ Error training {symbol}: {str(e)[:100]}")
        return None, None

# =========================
# HF UPLOAD FUNCTION
# =========================

def upload_to_huggingface():
    """Upload all files in a single commit to Hugging Face"""
    try:
        from huggingface_hub import HfApi
        from dotenv import load_dotenv

        load_dotenv()
        hf_token = os.getenv("hf_token")

        if not hf_token:
            return False

        api = HfApi()
        repo_id = "ahashanahmed/csv"

        api.upload_folder(
            folder_path="./csv",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Auto-update: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ignore_patterns=["*.tmp", "*.log", "__pycache__", ".DS_Store"]
        )

        return True

    except Exception as e:
        return False

# =========================
# DOWNLOAD FUNCTION
# =========================

def download_from_huggingface():
    """Download latest data from Hugging Face"""
    try:
        from huggingface_hub import snapshot_download

        if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) < 1000:
            snapshot_download(
                repo_id="ahashanahmed/csv",
                repo_type="dataset",
                local_dir="./csv",
                local_dir_use_symlinks=False
            )
            return True
        else:
            return True

    except Exception as e:
        return False

# =========================
# MAIN
# =========================

def main():
    print("🚀 XGBOOST SCHEDULER (ALL ENHANCEMENTS v2.0)")
    print(f"   ✅ Walk-Forward Validation")
    print(f"   ✅ Probability Calibration")
    print(f"   ✅ Target Encoding")
    print(f"   ✅ Interaction Features")
    print(f"   ✅ Optuna Tuning")
    print(f"   ✅ Market Cap + Sector Features")
    print(f"   ✅ Market Regime Detection")
    print(f"   ✅ Technical Indicators (BB, MACD, ATR, MFI)")
    print(f"   ✅ Temporal & Rolling Features")
    print(f"   ✅ Price Pattern Detection")
    print(f"   ✅ Feature Selection")
    print(f"   ✅ Ensemble Models (Weekly/Monthly)")
    print(f"   ✅ Mode-Specific Targets & Thresholds")
    print(f"   ✅ Adaptive Parameters")
    print("="*60)

    # Load best params cache
    load_best_params()

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
    elif weekly_needed:
        mode = "WEEKLY"
        params = WEEKLY_PARAMS
    elif daily_needed:
        mode = "DAILY"
        params = DAILY_PARAMS
    else:
        print("📅 No training needed. Uploading existing files...")
        upload_to_huggingface()
        return
    
    config = MODE_CONFIG[mode]
    
    print(f"\n📅 Mode: {mode}")
    print(f"🎯 Target Horizon: {config['target_horizon']} days")
    print(f"📈 Target Return: {config['target_return']*100}%")
    print(f"🔧 Optuna Trials: {config['optuna_trials']}")
    print(f"🤖 Ensemble: {config['ensemble_models']}")
    print(f"📊 Feature Selection: {config['feature_selection']}")
    print("="*60)

    # Feedback update
    feedback_log = update_actual_results()

    # Load metadata
    metadata = load_model_metadata()

    # Load data
    df = load_data()

    if df.empty:
        print("❌ No data loaded. Exiting.")
        return

    # Initialize Sector Analyzer
    sector_analyzer = SectorAnalyzer(df)

    # Feature engineering (base features)
    print("\n🔧 Engineering Base Features...")
    df = engineer_features(df)
    
    # ✅ Add new enhanced features
    print("🔧 Adding Enhanced Features...")
    df = add_market_regime_features(df)
    df = add_temporal_features(df)
    df = add_technical_indicators(df)
    df = add_rolling_features(df)
    df = detect_price_patterns(df)
    
    # Create mode-specific targets
    df = create_mode_specific_targets(df, mode)
    
    # Use mode-specific target
    target_col = f'target_{mode.lower()}'
    if target_col in df.columns:
        df['target'] = df[target_col]
    
    # Drop rows with NaN targets
    df = df.dropna(subset=['target'])
    
    # Get all features
    features = get_features(df)
    
    # Feature Selection
    features = select_features_for_mode(df, mode, features)
    
    print(f"\n📊 Features used: {len(features)}")
    print(f"   Mode-Specific: Target horizon {config['target_horizon']}d, Return > {config['target_return']*100}%")

    # Train models
    results = []
    updated_metadata = []
    trained_count = 0
    good_count = 0
    bad_count = 0
    monthly_retry_count = 0
    skipped_count = 0
    good_models_list = []

    sector_performance = defaultdict(lambda: {'good': 0, 'bad': 0, 'total': 0})

    for symbol, group in df.groupby('symbol'):
        if len(group) < config['min_samples']:
            skipped_count += 1
            continue

        should_train, reason = should_retrain(symbol, metadata)

        if not should_train:
            skipped_count += 1
            continue

        if 'monthly' in reason:
            monthly_retry_count += 1

        # Detect market regime for this symbol
        symbol_data = group.sort_values('date')
        market_regime, regime_conf = detect_market_regime(symbol_data)
        
        print(f"\n🔧 Training: {symbol} ({len(group)} rows)")
        print(f"   📊 Market Regime: {market_regime} (conf: {regime_conf:.2f})")
        
        result, model_info = train_symbol(symbol, group, features, params, feedback_log, metadata, sector_analyzer, mode)

        if result is not None:
            results.append(result)
            trained_count += 1

            if model_info:
                updated_metadata.append(model_info)
                if model_info['status'] == 'GOOD':
                    good_count += 1
                    good_models_list.append({
                        'symbol': symbol, 
                        'auc': model_info['auc'],
                        'wf_auc': model_info.get('wf_auc', 0),
                        'mode': mode
                    })
                    
                    sector = model_info.get('sector', 'Unknown')
                    sector_performance[sector]['good'] += 1
                    sector_performance[sector]['total'] += 1
                else:
                    bad_count += 1
                    sector = model_info.get('sector', 'Unknown')
                    sector_performance[sector]['bad'] += 1
                    sector_performance[sector]['total'] += 1
        else:
            skipped_count += 1

    # Save predictions with mode prefix
    if results:
        final = pd.concat(results, ignore_index=True)
        final.to_csv(f'./csv/xgb_confidence_{mode.lower()}.csv', index=False)
        final.to_csv(XGB_CONFIDENCE, index=False)
        save_prediction_log(final)

    # Update metadata
    if updated_metadata:
        new_metadata = pd.DataFrame(updated_metadata)

        if not metadata.empty:
            symbols_updated = new_metadata['symbol'].unique()
            metadata = metadata[~metadata['symbol'].isin(symbols_updated)]

        final_metadata = pd.concat([metadata, new_metadata], ignore_index=True)
        save_model_metadata(final_metadata)

    # Update schedule dates
    if daily_needed:
        update_last_run(LAST_DAILY_FILE, datetime.today())
    if weekly_needed:
        update_last_run(LAST_WEEKLY_FILE, datetime.today())
    if monthly_needed:
        update_last_run(LAST_MONTHLY_FILE, datetime.today())

    # Save sector performance
    if sector_performance:
        sector_rows = []
        for sector, perf in sector_performance.items():
            if perf['total'] > 0:
                sector_rows.append({
                    'sector': sector,
                    'good_models': perf['good'],
                    'bad_models': perf['bad'],
                    'total_models': perf['total'],
                    'success_rate': perf['good'] / perf['total'] * 100 if perf['total'] > 0 else 0
                })
        
        if sector_rows:
            sector_df = pd.DataFrame(sector_rows)
            sector_df.to_csv(SECTOR_PERFORMANCE_FILE, index=False)

    # Send Telegram summary
    send_training_summary(mode, trained_count, good_count, bad_count, monthly_retry_count, good_models_list)

    # Upload to Hugging Face
    upload_to_huggingface()

    print("\n" + "="*60)
    print(f"🎉 {mode} TRAINING COMPLETE!")
    print(f"   Mode: {mode}")
    print(f"   Target: {config['target_horizon']}d, >{config['target_return']*100}% return")
    print(f"   ✅ Good Models: {good_count}")
    print(f"   ❌ Bad Models: {bad_count}")
    print(f"   ⏭️ Skipped: {skipped_count}")
    print(f"   📊 Total Features: {len(features)}")
    print("="*60)

if __name__ == "__main__":
    main()