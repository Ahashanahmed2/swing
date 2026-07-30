# ================== xgboost_scheduler.py ==================
# COMPLETE VERSION with ALL ENHANCEMENTS
# Features:
# 1. Only good models (AUC >= 0.55) are saved
# 2. Bad models retry up to 3 times, then retry monthly
# 3. Already trained good models retrain with new data
# 4. Single commit upload to Hugging Face
# 5. Monthly retry for permanently failed models
# 6. Advanced features: Support/Resistance, RSI Divergence, EMA 200
# 7. ✅ Sector momentum, relative strength, peer comparison
# 8. ✅ Telegram notifications for training status
# 9. ✅ Sector performance tracking
# 10. ✅ freeFloatMarketCap features (log, rank, liquidity)
# 11. ✅ Walk-Forward Validation
# 12. ✅ Probability Calibration (Isotonic)
# 13. ✅ Target Encoding (Sector-wise)
# 14. ✅ Interaction Features (Price×Vol, RSI×Vol, MCap×Momentum)
# 15. ✅ Optuna Hyperparameter Tuning (Best Params)

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import requests
import optuna
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
EMA_200_PATH = './csv/ema_200.csv'

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
    best_params_cache[symbol] = params
    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(best_params_cache, f, indent=2)

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
    df['volatility_5d'] = df.groupby('symbol')['volatility'].transform(lambda x: x.rolling(5).mean())
    df['volume_ma'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
    df['volume_ratio'] = df['volume'] / df['volume_ma']

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
    # 3. EMA 200 FEATURES
    # =========================
    try:
        if os.path.exists(EMA_200_PATH):
            ema_df = pd.read_csv(EMA_200_PATH, encoding='utf-8-sig')
            ema_df.columns = ema_df.columns.str.strip()

            if 'close' in ema_df.columns:
                ema_df = ema_df.rename(columns={'close': 'ema_200'})
            else:
                numeric_cols = ema_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    ema_df = ema_df.rename(columns={numeric_cols[-1]: 'ema_200'})

            ema_df['date'] = pd.to_datetime(ema_df['date'])

            df = df.merge(ema_df[['symbol', 'date', 'ema_200']], on=['symbol', 'date'], how='left')

            df['dist_from_ema'] = (df['close'] - df['ema_200']) / df['ema_200'] * 100
            df['dist_from_ema'] = df['dist_from_ema'].clip(-30, 30)
            df['above_ema'] = (df['close'] > df['ema_200']).astype(int)

            df['ema_200'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(200).mean())
            df['dist_from_ema'] = df['dist_from_ema'].fillna(0)
            df['above_ema'] = df['above_ema'].fillna(0)
            df['ema_200'] = df['ema_200'].fillna(df['close'])
        else:
            df['dist_from_ema'] = 0
            df['above_ema'] = 0
    except:
        df['dist_from_ema'] = 0
        df['above_ema'] = 0

    # =========================
    # ✅ 4. INTERACTION FEATURES
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

    # =========================
    # ✅ 5. TARGET ENCODING
    # =========================
    df['future_return'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
    df['target'] = (df['future_return'] > 0.02).astype(int)
    
    df['sector_target_mean'] = df.groupby('sector')['target'].transform('mean')
    df['sector_target_std'] = df.groupby('sector')['target'].transform('std')
    df['sector_recent_target'] = df.groupby('sector')['target'].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    
    df['symbol_target_mean'] = df.groupby('symbol')['target'].transform(
        lambda x: x.expanding().mean()
    )

    return df.dropna(subset=['target'])

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

    ema_features = ['dist_from_ema', 'above_ema']
    features.extend([f for f in ema_features if f in df.columns])
    
    sector_features = ['sector_momentum', 'sector_rank', 'sector_trend',
                       'sector_target_mean', 'sector_target_std', 'sector_recent_target']
    features.extend([f for f in sector_features if f in df.columns])
    
    mc_features = ['freeFloatMarketCap', 'log_market_cap', 'market_cap_rank', 
                   'liquidity_score', 'mcap_volume_ratio']
    features.extend([f for f in mc_features if f in df.columns])
    
    interaction_features = ['price_volume_interaction', 'rsi_volatility', 'rsi_momentum',
                           'mcap_sector_momentum', 'sr_signal', 'rsi_div_interaction',
                           'volume_volatility']
    features.extend([f for f in interaction_features if f in df.columns])
    
    features.append('symbol_target_mean')

    return features

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

def optimize_hyperparameters(X_train, y_train, X_val, y_val, symbol):
    """Bayesian hyperparameter optimization with Optuna"""
    
    if symbol in best_params_cache:
        print(f"   📦 Using cached best params for {symbol}")
        return best_params_cache[symbol]
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
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
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['eval_metric'] = 'logloss'
    best_params['verbosity'] = 0
    best_params['use_label_encoder'] = False
    
    save_best_params(symbol, best_params)
    
    print(f"   ✅ Optuna: Best AUC={study.best_value:.3f}")
    
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
# TRAINING FUNCTION (FULLY ENHANCED)
# =========================

def train_symbol(symbol, group, features, params, feedback_log, metadata, sector_analyzer=None):
    """Train model with ALL enhancements"""
    try:
        group = group.sort_values('date')
        X = group[features]
        y = group['target']

        if len(X) < MIN_SAMPLES_PER_SYMBOL:
            return None, None

        # =========================
        # Three-way split
        # =========================
        train_idx = int(len(X) * 0.7)
        val_idx = int(len(X) * 0.85)
        
        X_train = X.iloc[:train_idx]
        y_train = y.iloc[:train_idx]
        X_val = X.iloc[train_idx:val_idx]
        y_val = y.iloc[train_idx:val_idx]
        X_test = X.iloc[val_idx:]
        y_test = y.iloc[val_idx:]

        if len(X_train) < 20 or len(X_test) < 5:
            return None, None

        # =========================
        # ✅ Walk-Forward Validation
        # =========================
        wf_mean, wf_std = walk_forward_validation(X_train, y_train, params)
        print(f"   📊 Walk-Forward AUC: {wf_mean:.3f} ± {wf_std:.3f}")

        # =========================
        # Class weight
        # =========================
        target_ratio = y_train.mean()
        if target_ratio < 0.3:
            scale_pos = min((1 - target_ratio) / target_ratio, 10)
        elif target_ratio > 0.7:
            scale_pos = min(target_ratio / (1 - target_ratio), 10)
        else:
            scale_pos = 1

        # =========================
        # ✅ Optuna Tuning
        # =========================
        if ENABLE_OPTUNA and len(X_train) >= 100:
            params_optimized = optimize_hyperparameters(X_train, y_train, X_val, y_val, symbol)
            params_copy = params_optimized
            print(f"   🎯 Using Optuna-optimized params")
        else:
            params_copy = params.copy()
        
        params_copy['scale_pos_weight'] = scale_pos

        # Sample weights from feedback
        weights = get_sample_weights(group.iloc[:train_idx], feedback_log)

        # =========================
        # Train model
        # =========================
        model = xgb.XGBClassifier(**params_copy)
        
        model.fit(
            X_train, y_train,
            sample_weight=weights,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )

        # =========================
        # ✅ Probability Calibration
        # =========================
        model = calibrate_model(model, X_val, y_val)

        # =========================
        # Evaluate on TEST set
        # =========================
        prob = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else 0.5

        # Overfitting Check
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        overfit_gap = train_acc - acc
        
        if overfit_gap > 0.12:
            print(f"   ⚠️ Overfitting! Train={train_acc:.3f}, Test={acc:.3f}, Gap={overfit_gap:.3f}")

        # Get sector info
        sector = 'Unknown'
        if sector_analyzer and symbol in sector_analyzer.symbol_sector_map:
            sector = sector_analyzer.symbol_sector_map[symbol]

        # =========================
        # Save or reject model
        # =========================
        if auc >= AUC_THRESHOLD:
            model_path = os.path.join(MODEL_DIR, f'{symbol}.joblib')
            joblib.dump(model, model_path)

            # Predict on ALL data
            group['confidence_score'] = model.predict_proba(X)[:, 1] * 100
            
            # Adjust by sector momentum
            if 'sector_momentum' in group.columns:
                sector_momentum = group['sector_momentum'].iloc[0]
                if sector_momentum > 0.02:
                    group['confidence_score'] = group['confidence_score'] * 1.1
                elif sector_momentum < -0.02:
                    group['confidence_score'] = group['confidence_score'] * 0.9
                group['confidence_score'] = group['confidence_score'].clip(0, 100)
            
            # Adjust by market cap
            if 'freeFloatMarketCap' in group.columns:
                mcap = group['freeFloatMarketCap'].iloc[-1]
                if mcap > 5000:
                    group['confidence_score'] = group['confidence_score'] * 1.05
                elif mcap < 50:
                    group['confidence_score'] = group['confidence_score'] * 0.95
                group['confidence_score'] = group['confidence_score'].clip(0, 100)
            
            group['prediction'] = (group['confidence_score'] > 50).astype(int)

            result = group[['symbol', 'date', 'close', 'confidence_score', 'prediction']]
            status = 'GOOD'
            failed_attempts = 0
            
            print(f"   ✅ AUC={auc:.3f} | Acc={acc:.3f} | WF={wf_mean:.3f}")

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
            
            print(f"   ❌ AUC={auc:.3f} < {AUC_THRESHOLD}")

        return result, {
            'symbol': symbol,
            'last_trained': datetime.now(),
            'last_attempt': datetime.now(),
            'auc': auc,
            'acc': acc,
            'val_acc': val_acc,
            'wf_auc': wf_mean,
            'failed_attempts': failed_attempts if auc < AUC_THRESHOLD else 0,
            'status': status,
            'class_ratio': target_ratio,
            'sector': sector,
            'features_used': len(features)
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
    print("🚀 XGBOOST SCHEDULER (ALL ENHANCEMENTS)")
    print(f"   ✅ Walk-Forward Validation")
    print(f"   ✅ Probability Calibration")
    print(f"   ✅ Target Encoding")
    print(f"   ✅ Interaction Features")
    print(f"   ✅ Optuna Tuning ({OPTUNA_TRIALS} trials)")
    print(f"   ✅ Market Cap + Sector Features")
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
        upload_to_huggingface()
        return

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

    # Feature engineering
    df = engineer_features(df)
    features = get_features(df)
    
    print(f"\n📊 Features used: {len(features)}")
    print(f"   Base: close, volume, return_5d, return_10d, volatility, volatility_5d, volume_ratio")
    print(f"   Market Cap: {[f for f in features if 'cap' in f.lower() or 'liquidity' in f.lower()]}")
    print(f"   Sector: {[f for f in features if 'sector' in f.lower()]}")
    print(f"   Interaction: {[f for f in features if 'interaction' in f.lower() or 'signal' in f.lower()]}")
    print(f"   Target Encoding: sector_target_mean, sector_target_std, sector_recent_target, symbol_target_mean")

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
        if len(group) < MIN_SAMPLES_PER_SYMBOL:
            skipped_count += 1
            continue

        should_train, reason = should_retrain(symbol, metadata)

        if not should_train:
            skipped_count += 1
            continue

        if 'monthly' in reason:
            monthly_retry_count += 1

        print(f"\n🔧 Training: {symbol} ({len(group)} rows)")
        result, model_info = train_symbol(symbol, group, features, params, feedback_log, metadata, sector_analyzer)

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
                        'wf_auc': model_info.get('wf_auc', 0)
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

    # Save predictions
    if results:
        final = pd.concat(results, ignore_index=True)
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
    print("🎉 ALL DONE!")
    print(f"   ✅ Good Models: {good_count}")
    print(f"   ❌ Bad Models: {bad_count}")
    print(f"   ⏭️ Skipped: {skipped_count}")
    print(f"   📊 Total Features: {len(features)}")
    print("="*60)

if __name__ == "__main__":
    main()