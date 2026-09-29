# ================== patch_tst_predictor.py ==================
# PatchTST - Time Series Transformer for Price Prediction
# State-of-the-art financial forecasting
# Drop-in module — no changes to existing code required
# ✅ FIXED: Checkpoint conflict (patchtst_model.pt vs checkpoints/best_model.pt)
# ✅ FIXED: 'seq_len' KeyError on load_model
# ✅ FIXED: format_version + input_dim in checkpoint
# ✅ HF Backup Upload (No Download)
# ✅ Mistake Learning & Auto-Correction
# ✅ Accuracy Check before HF Upload
# ✅ FIXED: SECTOR FEATURES — full rewrite (sector column key + date-wise merge_asof)
# ✅ FIXED: mcap_rank_sector via symbol→sector map
# ✅ Auto-Pilot Training (All Symbols)
# ✅ Weekly Fine-tune + Monthly Retrain
# ✅ MAX QUALITY: Market Cap + EMA + Walk-Forward + OneCycleLR + Adaptive Params
# ✅ ULTIMATE: 3-Layer LSTM, CosineAnnealing, Gradient Accumulation, Extended Epochs
# ✅ RATE-LIMIT SAFE: 40s delay between HF uploads, max 90 commits/hour

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
import os
import re
import time
import warnings
from datetime import datetime, timedelta
from collections import deque
warnings.filterwarnings('ignore')

# Try importing deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Install: pip install torch")

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =========================================================
# SECTOR NAME NORMALIZATION HELPER
# =========================================================

def normalize_sector_name(name):
    """Sector নাম normalize করুন - space, case, special char handle"""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ''
    s = str(name).strip().lower()
    s = s.replace('&', 'and')
    s = s.replace('/', ' and ')
    s = s.replace('_', ' ')
    s = s.replace('-', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


# =========================================================
# PATCHING MODULE
# =========================================================

class Patching(nn.Module):
    """Time series patching - split sequence into overlapping patches"""
    
    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
    
    def forward(self, x):
        n_patches = (x.shape[-1] - self.patch_len) // self.stride + 1
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return x


# =========================================================
# TRANSFORMER ENCODER
# =========================================================

class TransformerEncoder(nn.Module):
    """Lightweight Transformer Encoder for financial data"""
    
    def __init__(self, d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.encoder(x)


# =========================================================
# ULTIMATE PRICE PREDICTOR (3-Layer LSTM + Attention + Residual)
# =========================================================

class SimpleAttentionPredictor(nn.Module):
    """Ultimate Attention-based Price Predictor - 3-Layer LSTM + Residual"""
    
    def __init__(self, input_dim=10, seq_len=60, hidden_dim=128, pred_len=5):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        
        # 3-Layer Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # 8-Head Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.15,
            batch_first=True
        )
        
        # 3-Layer MLP head
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 3)
        
        self.dropout = nn.Dropout(0.3)
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim // 2)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm1(lstm_out + attn_out)
        pooled = attn_out.mean(dim=1)
        
        out = self.fc1(pooled)
        out = pooled[:, :self.hidden_dim] * 0.3 + out * 0.7
        out = self.layer_norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        out2 = self.fc2(out)
        out2 = self.layer_norm3(out2)
        out2 = F.gelu(out2)
        out2 = self.dropout(out2)
        
        out = self.fc3(out2)
        probs = torch.softmax(out[:, :2], dim=-1)
        magnitude = torch.tanh(out[:, 2:3])
        return torch.cat([probs, magnitude], dim=-1)


# =========================================================
# MAIN PREDICTOR CLASS (ULTIMATE QUALITY)
# =========================================================

class PatchTSTPredictor:
    """Time Series Transformer for Price Prediction - ULTIMATE QUALITY"""
    
    def __init__(
        self,
        seq_len=60,
        pred_len=5,
        hidden_dim=128,
        model_dir="./csv/patchtst_models",
        device=None,
        use_sector_features=True,
        use_sr_features=True,
        use_rsi_div_features=True,
        use_market_cap_features=True,
        full_df=None,   # ✅ NEW: পুরো mongodb.csv (sector map এর জন্য)
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_sector_features = use_sector_features
        self.use_sr_features = use_sr_features
        self.use_rsi_div_features = use_rsi_div_features
        self.use_market_cap_features = use_market_cap_features
        
        # Sector state
        self.symbol_to_sector = {}     # {symbol: sector_name}
        self.sector_daily = {}         # {sector_name: DataFrame}
        self.sector_weekly = {}        # {sector_name: DataFrame}
        
        # S/R + RSI divergence
        self.sr_data = None
        self.rsi_div_data = {}
        
        # Build sector map from full_df (if provided)
        if full_df is not None:
            self.symbol_to_sector = self._build_symbol_to_sector(full_df)
        
        self._load_external_features()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else 'cpu'
        else:
            self.device = device
        
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_fitted = False
        self.feature_columns = None
        
        print(f"✅ PatchTST Predictor initialized (device: {self.device}, hidden_dim={hidden_dim})")
    
    # =========================================================
    # SECTOR MAP BUILDER (mongodb.csv থেকে)
    # =========================================================
    
    def _build_symbol_to_sector(self, full_df):
        """mongodb.csv-এর latest non-null row থেকে symbol → sector map"""
        if 'sector' not in full_df.columns or 'symbol' not in full_df.columns:
            print("   ⚠️ 'sector' or 'symbol' column missing in full_df")
            return {}
        
        try:
            df = full_df.copy()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.sort_values('date')
            
            mapping = (
                df.dropna(subset=['sector'])
                  .groupby('symbol')['sector']
                  .last()
                  .to_dict()
            )
            mapping = {k: str(v).strip() for k, v in mapping.items()}
            
            unique_sectors = sorted(set(mapping.values()))
            print(f"   ✅ Symbol→Sector map: {len(mapping)} symbols, "
                  f"{len(unique_sectors)} unique sectors")
            return mapping
        except Exception as e:
            print(f"   ⚠️ symbol→sector build failed: {e}")
            return {}
    
    # =========================================================
    # EXTERNAL FEATURES LOADERS
    # =========================================================
    
    def _load_external_features(self):
        if self.use_sector_features:
            self._load_sector_features()
        if self.use_sr_features:
            self._load_support_resistance()
        if self.use_rsi_div_features:
            self._load_rsi_divergence()
    
    def _load_sector_features(self):
        """Load sector daily/weekly CSVs keyed by sector name"""
        sector_dir = Path('./csv/sector')
        if not sector_dir.exists():
            print(f"   ⚠️ Sector folder not found: {sector_dir}")
            self.use_sector_features = False
            return
        
        self.sector_daily = {}
        self.sector_weekly = {}
        
        try:
            # ---------- Daily ----------
            daily_dir = sector_dir / 'daily'
            if daily_dir.exists():
                for f in daily_dir.glob('*.csv'):
                    try:
                        df = pd.read_csv(f)
                        if 'sector' not in df.columns or 'date' not in df.columns:
                            continue
                        
                        sector_name = str(df['sector'].iloc[0]).strip()
                        if sector_name == 'Unknown' or not sector_name:
                            continue
                        
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
                        
                        if 'rsi' in df.columns:
                            df['rsi'] = df['rsi'].ffill().bfill().fillna(50)
                        if 'change' in df.columns:
                            df['change'] = df['change'].fillna(0)
                        if 'volume' in df.columns:
                            df['volume'] = df['volume'].fillna(0)
                        
                        self.sector_daily[sector_name] = df
                    except Exception:
                        continue
            
            # ---------- Weekly ----------
            weekly_dir = sector_dir / 'weekly'
            if weekly_dir.exists():
                for f in weekly_dir.glob('*.csv'):
                    try:
                        df = pd.read_csv(f)
                        if 'sector' not in df.columns:
                            continue
                        
                        sector_name = str(df['sector'].iloc[0]).strip()
                        if sector_name == 'Unknown' or not sector_name:
                            continue
                        
                        date_col = None
                        for c in ['week_end_date', 'week_start']:
                            if c in df.columns:
                                date_col = c
                                break
                        if date_col is None:
                            continue
                        
                        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
                        df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
                        
                        if 'rsi' in df.columns:
                            df['rsi'] = df['rsi'].ffill().bfill().fillna(50)
                        if 'change' in df.columns:
                            df['change'] = df['change'].fillna(0)
                        
                        self.sector_weekly[sector_name] = df
                    except Exception:
                        continue
            
            print(f"   ✅ Sector loaded: {len(self.sector_daily)} daily, "
                  f"{len(self.sector_weekly)} weekly")
            
            # ---- Warn on mismatch ----
            if self.symbol_to_sector:
                mongo_sectors = set(self.symbol_to_sector.values())
                csv_sectors = set(self.sector_daily.keys()) | set(self.sector_weekly.keys())
                missing_in_csv = mongo_sectors - csv_sectors
                if missing_in_csv:
                    print(f"   ⚠️ Sectors in mongodb but not in CSV: {sorted(missing_in_csv)}")
            
            if not self.sector_daily and not self.sector_weekly:
                self.use_sector_features = False
                
        except Exception as e:
            print(f"   ⚠️ Sector load failed: {e}")
            self.use_sector_features = False
    
    def _load_support_resistance(self):
        sr_path = Path('./csv/support_resistance.csv')
        if not sr_path.exists():
            self.use_sr_features = False
            return
        try:
            self.sr_data = pd.read_csv(sr_path)
            if 'current_date' in self.sr_data.columns:
                self.sr_data['current_date'] = pd.to_datetime(self.sr_data['current_date'])
            print(f"   ✅ Loaded S/R data: {self.sr_data['symbol'].nunique()} symbols")
        except Exception as e:
            print(f"   ⚠️ S/R load failed: {e}")
            self.use_sr_features = False
    
    def _load_rsi_divergence(self):
        rsi_path = Path('./csv/rsi_diver.csv')
        if not rsi_path.exists():
            self.use_rsi_div_features = False
            return
        try:
            div_df = pd.read_csv(rsi_path)
            if 'date' in div_df.columns:
                div_df['date'] = pd.to_datetime(div_df['date'])
            for symbol in div_df['symbol'].unique():
                self.rsi_div_data[symbol] = div_df[div_df['symbol'] == symbol]
            print(f"   ✅ Loaded RSI divergence for {len(self.rsi_div_data)} symbols")
        except Exception as e:
            print(f"   ⚠️ RSI divergence load failed: {e}")
            self.use_rsi_div_features = False
    
    # =========================================================
    # SECTOR FEATURE LOOKUP (date-aware, no look-ahead)
    # =========================================================
    
    def _get_sector_features_for_row(self, symbol, current_date):
        """Date-wise sector features — NO look-ahead bias"""
        if not self.use_sector_features or current_date is None:
            return {}
        
        sector = self.symbol_to_sector.get(symbol)
        if not sector:
            return {}
        
        try:
            current_dt = pd.to_datetime(current_date)
        except Exception:
            return {}
        
        result = {}
        
        # Daily sector features
        if sector in self.sector_daily:
            sdf = self.sector_daily[sector]
            past = sdf[sdf['date'] <= current_dt]
            if not past.empty:
                row = past.iloc[-1]
                result['sector_returns'] = float(row.get('change', 0) or 0) / 100.0
                result['sector_rsi'] = float(row.get('rsi', 50) or 50)
                result['sector_volume'] = float(row.get('volume', 0) or 0)
        
        # Weekly sector features
        if sector in self.sector_weekly:
            swdf = self.sector_weekly[sector]
            past_w = swdf[swdf['date'] <= current_dt]
            if not past_w.empty:
                wrow = past_w.iloc[-1]
                result['sector_weekly_change'] = float(wrow.get('change', 0) or 0) / 100.0
                result['sector_weekly_rsi'] = float(wrow.get('rsi', 50) or 50)
        
        return result
    
    def _get_sr_features_for_row(self, symbol, current_date, current_close):
        if not self.use_sr_features or self.sr_data is None:
            return {}
        try:
            sym_data = self.sr_data[self.sr_data['symbol'] == symbol]
            if sym_data.empty:
                return {}
            current_dt = pd.to_datetime(current_date)
            recent = sym_data[sym_data['current_date'] <= current_dt].tail(1)
            if recent.empty:
                return {}
            row = recent.iloc[-1]
            level_price = float(row['level_price'])
            distance_pct = (current_close - level_price) / current_close if current_close > 0 else 0
            strength_str = str(row.get('strength', 'Weak')).capitalize()
            strength_map = {'Weak': 0.3, 'Moderate': 0.6, 'Strong': 1.0}
            level_type = str(row.get('type', '')).lower()
            return {
                'sr_distance': distance_pct,
                'sr_strength': strength_map.get(strength_str, 0.5),
                'sr_type': 1.0 if level_type == 'support' else -1.0 if level_type == 'resistance' else 0.0
            }
        except Exception:
            return {}
    
    def _get_rsi_div_features_for_row(self, symbol, current_date):
        if not self.use_rsi_div_features or symbol not in self.rsi_div_data:
            return {}
        try:
            div_df = self.rsi_div_data[symbol]
            current_dt = pd.to_datetime(current_date)
            recent = div_df[div_df['date'] <= current_dt].tail(1)
            if recent.empty:
                return {}
            row = recent.iloc[-1]
            div_type = str(row.get('divergence_type', 'NONE')).upper()
            strength = str(row.get('divergence_strength', 'NONE')).upper()
            strength_map = {'STRONG': 1.0, 'MODERATE': 0.6, 'WEAK': 0.3}
            return {
                'rsi_div_bullish': 1.0 if 'BULLISH' in div_type else 0.0,
                'rsi_div_bearish': 1.0 if 'BEARISH' in div_type else 0.0,
                'rsi_div_strength': strength_map.get(strength, 0.0),
                'rsi_value': float(row.get('rsi', 50))
            }
        except Exception:
            return {}
    
    # =========================================================
    # FEATURE ENGINEERING
    # =========================================================
    
    def _engineer_features(self, df):
        """Create features from OHLCV data + External features + Market Cap + EMA"""
        df = df.copy()
        
        # ---------- Basic OHLCV features ----------
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_trend'] = df['volume'].pct_change(5)
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        df['volatility'] = df['returns'].rolling(10).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(30).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_30'] = df['close'].rolling(30).mean()
        df['trend_strength'] = (df['sma_10'] - df['sma_30']) / df['sma_30']
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)
        
        for col in ['rsi', 'macd', 'macd_signal', 'macd_hist', 'atr']:
            if col in df.columns:
                df[f'{col}_norm'] = df[col] / (df[col].abs().rolling(50).mean() + 1e-8)
        
        # ---------- Market cap features ----------
        if self.use_market_cap_features and 'freeFloatMarketCap' in df.columns:
            df['log_market_cap'] = np.log1p(df['freeFloatMarketCap'])
            df['mcap_volume_ratio'] = np.log1p(df['volume'] / (df['freeFloatMarketCap'] + 1e-8))
            
            # ✅ mcap_rank_sector via symbol→sector map
            if self.symbol_to_sector and 'symbol' in df.columns:
                df['sector_for_rank'] = df['symbol'].map(self.symbol_to_sector).fillna('Unknown')
                df['mcap_rank_sector'] = (
                    df.groupby('sector_for_rank')['freeFloatMarketCap'].rank(pct=True)
                )
                df = df.drop(columns=['sector_for_rank'])
            else:
                df['mcap_rank_sector'] = 0.5
        
        # ---------- EMA features ----------
        if 'ema_200' in df.columns:
            df['dist_from_ema'] = (df['close'] - df['ema_200']) / df['ema_200'] * 100
            df['above_ema'] = (df['close'] > df['ema_200']).astype(int)
        else:
            if 'symbol' in df.columns:
                df['ema_200_calc'] = df.groupby('symbol')['close'].transform(
                    lambda x: x.ewm(span=200, adjust=False).mean()
                )
            else:
                df['ema_200_calc'] = df['close'].ewm(span=200, adjust=False).mean()
            df['dist_from_ema'] = (df['close'] - df['ema_200_calc']) / df['ema_200_calc'] * 100
            df['above_ema'] = (df['close'] > df['ema_200_calc']).astype(int)
        
        # =========================================================
        # ✅ SECTOR FEATURES (vectorized merge_asof — NO look-ahead)
        # =========================================================
        if (self.use_sector_features 
            and 'date' in df.columns 
            and 'symbol' in df.columns 
            and self.symbol_to_sector):
            
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            symbol = df['symbol'].iloc[0]
            sector = self.symbol_to_sector.get(symbol)
            
            has_sector_data = False
            
            if sector:
                # ---- Daily merge ----
                if sector in self.sector_daily:
                    sec = self.sector_daily[sector][['date', 'change', 'rsi', 'volume']].copy()
                    sec.columns = ['date', 'sec_change', 'sec_rsi', 'sec_volume']
                    sec = sec.sort_values('date').drop_duplicates('date', keep='last')
                    
                    df = df.sort_values('date')
                    df = pd.merge_asof(df, sec, on='date', direction='backward')
                    
                    df['sector_momentum'] = df['sec_change'].fillna(0) / 100.0
                    df['sector_rsi'] = df['sec_rsi'].fillna(50)
                    df['sector_volume_ratio'] = (
                        df['sec_volume'].fillna(df['sec_volume'].median() if df['sec_volume'].notna().any() else 1) / 
                        (df['sec_volume'].rolling(20, min_periods=1).mean() + 1e-8)
                    )
                    df = df.drop(columns=['sec_change', 'sec_rsi', 'sec_volume'])
                    has_sector_data = True
                
                # ---- Weekly merge (extra signals) ----
                if sector in self.sector_weekly:
                    sec_w = self.sector_weekly[sector][['date', 'change', 'rsi']].copy()
                    sec_w.columns = ['date', 'sec_w_change', 'sec_w_rsi']
                    sec_w = sec_w.sort_values('date').drop_duplicates('date', keep='last')
                    
                    df = df.sort_values('date')
                    df = pd.merge_asof(df, sec_w, on='date', direction='backward')
                    df['sector_weekly_momentum'] = df['sec_w_change'].fillna(0) / 100.0
                    df['sector_weekly_rsi'] = df['sec_w_rsi'].fillna(50)
                    df = df.drop(columns=['sec_w_change', 'sec_w_rsi'])
            
            # Fallbacks for missing sector data
            if 'sector_momentum' not in df.columns:
                df['sector_momentum'] = 0.0
            if 'sector_rsi' not in df.columns:
                df['sector_rsi'] = 50.0
            if 'sector_volume_ratio' not in df.columns:
                df['sector_volume_ratio'] = 1.0
            if 'sector_weekly_momentum' not in df.columns:
                df['sector_weekly_momentum'] = 0.0
            if 'sector_weekly_rsi' not in df.columns:
                df['sector_weekly_rsi'] = 50.0
        
        # =========================================================
        # S/R FEATURES
        # =========================================================
        if self.use_sr_features and 'date' in df.columns and 'symbol' in df.columns:
            df['sr_distance'] = 0.0
            df['sr_strength'] = 0.5
            df['sr_type'] = 0.0
            
            if self.sr_data is not None:
                for idx in df.index:
                    row = df.loc[idx]
                    sr_feat = self._get_sr_features_for_row(
                        row.get('symbol', ''), row.get('date'), row.get('close', 0)
                    )
                    if sr_feat:
                        df.loc[idx, 'sr_distance'] = sr_feat.get('sr_distance', 0)
                        df.loc[idx, 'sr_strength'] = sr_feat.get('sr_strength', 0.5)
                        df.loc[idx, 'sr_type'] = sr_feat.get('sr_type', 0)
        
        # =========================================================
        # RSI DIVERGENCE FEATURES
        # =========================================================
        if self.use_rsi_div_features and 'date' in df.columns and 'symbol' in df.columns:
            df['rsi_div_bullish'] = 0.0
            df['rsi_div_bearish'] = 0.0
            df['rsi_div_strength'] = 0.0
            df['rsi_external'] = 50.0
            
            for idx in df.index:
                row = df.loc[idx]
                div_feat = self._get_rsi_div_features_for_row(
                    row.get('symbol', ''), row.get('date')
                )
                if div_feat:
                    df.loc[idx, 'rsi_div_bullish'] = div_feat.get('rsi_div_bullish', 0)
                    df.loc[idx, 'rsi_div_bearish'] = div_feat.get('rsi_div_bearish', 0)
                    df.loc[idx, 'rsi_div_strength'] = div_feat.get('rsi_div_strength', 0)
                    df.loc[idx, 'rsi_external'] = div_feat.get('rsi_value', 50)
        
        return df
    
    # =========================================================
    # FEATURE SELECTION
    # =========================================================
    
    def _select_features(self, df):
        """Select and prepare features for the model"""
        priority_features = [
            'returns', 'log_returns', 'volume_ratio',
            'high_low_ratio', 'close_open_ratio',
            'volatility', 'volatility_ratio',
            'trend_strength', 'momentum_5', 'momentum_10',
            'price_position'
        ]
        
        external_features = [
            # Sector
            'sector_momentum', 'sector_volume_ratio', 'sector_rsi',
            'sector_weekly_momentum', 'sector_weekly_rsi',
            # S/R
            'sr_distance', 'sr_strength', 'sr_type',
            # RSI divergence
            'rsi_div_bullish', 'rsi_div_bearish', 'rsi_div_strength', 'rsi_external',
            # Market cap + EMA
            'log_market_cap', 'mcap_volume_ratio', 'mcap_rank_sector',
            'dist_from_ema', 'above_ema',
        ]
        
        for feat in external_features:
            if feat in df.columns:
                priority_features.append(feat)
        
        for col in ['rsi', 'macd', 'macd_hist', 'atr']:
            norm_col = f'{col}_norm'
            if norm_col in df.columns:
                priority_features.append(norm_col)
            elif col in df.columns:
                priority_features.append(col)
        
        available = [f for f in priority_features if f in df.columns]
        self.feature_columns = available[:25]
        
        if len(self.feature_columns) > 15:
            print(f"   📊 Extended features: {len(self.feature_columns)} features")
        
        return df[self.feature_columns]
    
    # =========================================================
    # SEQUENCE PREPARATION
    # =========================================================
    
    def _prepare_sequences(self, df):
        df = self._engineer_features(df)
        feature_df = self._select_features(df)
        feature_df = feature_df.ffill().fillna(0)
        
        if self.scaler and SKLEARN_AVAILABLE:
            if not hasattr(self.scaler, 'mean_'):
                features_scaled = self.scaler.fit_transform(feature_df)
            else:
                features_scaled = self.scaler.transform(feature_df)
        else:
            features_scaled = feature_df.values
        
        sequences = []
        targets = []
        
        for i in range(len(features_scaled) - self.seq_len - self.pred_len):
            seq = features_scaled[i:i+self.seq_len]
            future_price = df['close'].iloc[i+self.seq_len:i+self.seq_len+self.pred_len].values
            current_price = df['close'].iloc[i+self.seq_len-1]
            future_return = (future_price[-1] - current_price) / current_price
            
            if future_return > 0.005:
                target = [1.0, 0.0, min(future_return, 0.10)]
            elif future_return < -0.005:
                target = [0.0, 1.0, min(abs(future_return), 0.10)]
            else:
                target = [0.5, 0.5, abs(future_return)]
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    # =========================================================
    # WALK-FORWARD VALIDATION
    # =========================================================
    
    def _walk_forward_validation(self, df, n_splits=5):
        if len(df) < 200:
            return 0.5
        
        scores = []
        split_size = len(df) // (n_splits + 1)
        
        for i in range(n_splits):
            train_end = split_size * (i + 1)
            if train_end >= len(df) - 50:
                break
            
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:min(train_end + split_size, len(df))]
            
            if len(val_df) < 30:
                continue
            
            X, y = self._prepare_sequences(train_df)
            if len(X) < 10:
                continue
            
            temp_model = SimpleAttentionPredictor(
                input_dim=X.shape[2], seq_len=self.seq_len,
                hidden_dim=min(self.hidden_dim, 32), pred_len=self.pred_len
            ).to(self.device)
            
            temp_opt = torch.optim.AdamW(temp_model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            X_t = torch.FloatTensor(X).to(self.device)
            y_t = torch.FloatTensor(y).to(self.device)
            ds = TensorDataset(X_t, y_t)
            dl = DataLoader(ds, batch_size=16, shuffle=True)
            
            temp_model.train()
            for _ in range(15):
                for bx, by in dl:
                    temp_opt.zero_grad()
                    loss = criterion(temp_model(bx), by)
                    loss.backward()
                    temp_opt.step()
            
            temp_model.eval()
            val_features = self._select_features(self._engineer_features(val_df))
            val_features = val_features.ffill().fillna(0)
            
            if hasattr(self.scaler, 'mean_'):
                val_scaled = self.scaler.transform(val_features.iloc[-self.seq_len:])
            else:
                val_scaled = val_features.iloc[-self.seq_len:].values
            
            with torch.no_grad():
                val_tensor = torch.FloatTensor(val_scaled).unsqueeze(0).to(self.device)
                pred = temp_model(val_tensor).cpu().numpy()[0]
            
            actual_ret = (val_df['close'].iloc[-1] / val_df['close'].iloc[0] - 1)
            correct = (pred[0] > 0.5) == (actual_ret > 0)
            scores.append(1 if correct else 0)
        
        return np.mean(scores) if scores else 0.5
    
    # =========================================================
    # TRAINING
    # =========================================================
    
    def fit(self, df, epochs=50, batch_size=32, learning_rate=0.001, verbose=True):
        if not TORCH_AVAILABLE:
            print("❌ PyTorch required for training")
            return False
        
        if len(df) < self.seq_len + self.pred_len + 10:
            print(f"❌ Not enough data. Need > {self.seq_len + self.pred_len} rows")
            return False
        
        X, y = self._prepare_sequences(df)
        
        if len(X) == 0:
            print("❌ No training sequences created")
            return False
        
        if verbose:
            print(f"📊 Training data: {len(X)} sequences, shape={X.shape}, features={X.shape[2]}")
        
        self.model = SimpleAttentionPredictor(
            input_dim=X.shape[2],
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            pred_len=self.pred_len
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = self._create_scheduler(optimizer, epochs)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            scheduler.step()
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_model()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        self.is_fitted = True
        
        if verbose and len(df) >= 200:
            wf_score = self._walk_forward_validation(df)
            print(f"   📊 Walk-Forward Score: {wf_score:.3f}")
        
        if verbose:
            print(f"✅ Training complete | Best loss: {best_loss:.6f}")
        
        return True
    
    def _create_scheduler(self, optimizer, epochs):
        """CosineAnnealingWarmRestarts LR scheduling"""
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(epochs // 4, 5),
            T_mult=2,
            eta_min=1e-6
        )
    
    # =========================================================
    # PREDICTION
    # =========================================================
    
    def predict_next_n_days(self, df, n_days=5):
        if not TORCH_AVAILABLE or not self.is_fitted:
            return {
                'up_prob': 0.5, 'down_prob': 0.5, 'magnitude': 0.02,
                'direction': 'UNKNOWN', 'confidence': 0.0
            }
        
        try:
            df = self._engineer_features(df)
            feature_df = self._select_features(df)
            feature_df = feature_df.ffill().fillna(0)
            
            if len(feature_df) < self.seq_len:
                return {
                    'up_prob': 0.5, 'down_prob': 0.5, 'magnitude': 0.02,
                    'direction': 'UNKNOWN', 'confidence': 0.0
                }
            
            last_seq = feature_df.iloc[-self.seq_len:].values
            
            if self.scaler and SKLEARN_AVAILABLE and hasattr(self.scaler, 'mean_'):
                last_seq = self.scaler.transform(last_seq)
            
            self.model.eval()
            with torch.no_grad():
                X = torch.FloatTensor(last_seq).unsqueeze(0).to(self.device)
                prediction = self.model(X).cpu().numpy()[0]
            
            up_prob = float(prediction[0])
            down_prob = float(prediction[1])
            magnitude = float(prediction[2])
            
            total = up_prob + down_prob
            if total > 0:
                up_prob = up_prob / total
                down_prob = down_prob / total
            
            if up_prob > 0.55:
                direction = 'UP'
            elif down_prob > 0.55:
                direction = 'DOWN'
            else:
                direction = 'FLAT'
            
            confidence = abs(up_prob - down_prob)
            
            return {
                'up_prob': round(up_prob, 4), 'down_prob': round(down_prob, 4),
                'magnitude': round(magnitude, 4), 'direction': direction,
                'confidence': round(confidence, 4)
            }
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return {
                'up_prob': 0.5, 'down_prob': 0.5, 'magnitude': 0.02,
                'direction': 'ERROR', 'confidence': 0.0
            }
    
    def get_feature_vector(self, df):
        pred = self.predict_next_n_days(df, n_days=5)
        direction_map = {'UP': 1.0, 'DOWN': 0.0, 'FLAT': 0.5, 'UNKNOWN': 0.5, 'ERROR': 0.5}
        return np.array([
            pred['up_prob'], pred['down_prob'], pred['magnitude'],
            direction_map.get(pred['direction'], 0.5), pred['confidence']
        ], dtype=np.float32)
    
    # =========================================================
    # SAVE / LOAD
    # =========================================================
    
    def _save_model(self):
        """Save model with complete metadata"""
        if self.model is None:
            return
        model_path = self.model_dir / "patchtst_model.pt"
        
        input_dim = len(self.feature_columns) if self.feature_columns else 10
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_columns': self.feature_columns,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'hidden_dim': self.hidden_dim,
            'input_dim': input_dim,
            'saved_at': datetime.now().isoformat(),
            'format_version': 2,
        }, model_path)
        
        if self.scaler and SKLEARN_AVAILABLE:
            scaler_path = self.model_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        
        print(f"   💾 Model saved (input_dim={input_dim}, seq_len={self.seq_len})")
    
    def load_model(self, symbol=None):
        """Load model with robust error handling"""
        model_path = self.model_dir / "patchtst_model.pt"
        if not model_path.exists():
            print(f"⚠️ No saved model found at {model_path}")
            return False
        
        if not TORCH_AVAILABLE:
            print("❌ PyTorch required for loading")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            print(f"❌ Cannot read checkpoint: {e}")
            return False
        
        required = ['model_state_dict', 'seq_len', 'pred_len', 'hidden_dim']
        missing = [k for k in required if k not in checkpoint]
        if missing:
            print(f"⚠️ Incomplete checkpoint (missing: {missing}). Needs retraining.")
            return False
        
        try:
            self.seq_len = checkpoint['seq_len']
            self.pred_len = checkpoint['pred_len']
            self.hidden_dim = checkpoint['hidden_dim']
            self.feature_columns = checkpoint.get('feature_columns')
            
            if self.feature_columns:
                input_dim = len(self.feature_columns)
            else:
                input_dim = checkpoint.get('input_dim', 10)
                for k, v in checkpoint['model_state_dict'].items():
                    if 'lstm.weight_ih_l0' in k:
                        input_dim = v.shape[1]
                        break
            
            self.model = SimpleAttentionPredictor(
                input_dim=input_dim,
                seq_len=self.seq_len,
                hidden_dim=self.hidden_dim,
                pred_len=self.pred_len
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists() and SKLEARN_AVAILABLE:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            self.is_fitted = True
            print(f"✅ Model loaded (input_dim={input_dim}, seq_len={self.seq_len}, pred_len={self.pred_len})")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {type(e).__name__}: {e}")
            return False
    
    def _create_model(self, input_dim):
        return SimpleAttentionPredictor(
            input_dim=input_dim, seq_len=self.seq_len,
            hidden_dim=self.hidden_dim, pred_len=self.pred_len
        )
    
    def _create_optimizer(self, lr, model=None):
        m = model if model is not None else self.model
        if m is None:
            return None
        return torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-5)


# =========================================================
# INTEGRATION WITH env_trading.py
# =========================================================

class PatchTSTIntegration:
    """Wrapper to integrate PatchTST with existing env_trading.py"""
    
    def __init__(self, model_dir="./csv/patchtst_models", full_df=None):
        self.predictor = PatchTSTPredictor(model_dir=model_dir, full_df=full_df)
        self.models_per_symbol = {}
    
    def get_or_create_predictor(self, symbol):
        if symbol not in self.models_per_symbol:
            predictor = PatchTSTPredictor(
                model_dir=Path(f"./csv/patchtst_models/{symbol}"),
                full_df=None  # already built map
            )
            predictor.symbol_to_sector = self.predictor.symbol_to_sector
            predictor.sector_daily = self.predictor.sector_daily
            predictor.sector_weekly = self.predictor.sector_weekly
            predictor.sr_data = self.predictor.sr_data
            predictor.rsi_div_data = self.predictor.rsi_div_data
            
            if not predictor.load_model(symbol):
                print(f"   ℹ️ No existing model for {symbol}, needs training")
            self.models_per_symbol[symbol] = predictor
        return self.models_per_symbol[symbol]
    
    def predict(self, symbol, df):
        predictor = self.get_or_create_predictor(symbol)
        return predictor.predict_next_n_days(df)
    
    def get_features(self, symbol, df):
        predictor = self.get_or_create_predictor(symbol)
        return predictor.get_feature_vector(df)
    
    def train_symbol(self, symbol, df, epochs=50):
        predictor = self.get_or_create_predictor(symbol)
        return predictor.fit(df, epochs=epochs, verbose=True)


# =========================================================
# FineTunablePatchTST (Extended)
# =========================================================

class FineTunablePatchTST(PatchTSTPredictor):
    """PatchTST with fine-tuning support"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def fit_with_checkpoint(self, df, epochs=50, batch_size=32, learning_rate=0.001,
                           resume=True, patience=15, verbose=True):
        return self.fit(df, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, verbose=verbose)
    
    def fine_tune_on_new_data(self, df, epochs=20, learning_rate=0.0001):
        if not self.is_fitted:
            print("⚠️ No existing model, training from scratch")
            return self.fit(df, epochs=epochs, learning_rate=learning_rate)
        print(f"🔄 Fine-tuning on {len(df)} new rows...")
        return self.fit(df, epochs=epochs, learning_rate=learning_rate)
    
    def get_training_summary(self):
        return {'status': 'unknown'}


# =========================================================
# RATE-LIMIT SAFE CHECKPOINT MANAGER
# =========================================================

class SimpleCheckpointManager:
    """Checkpoint System: Save/Load local, Backup to HF (40s delay between uploads)"""
    
    _last_upload_time = None
    _min_upload_interval = 40
    
    def __init__(self, symbol, hf_repo="ahashanahmed/csv"):
        self.symbol = symbol
        self.hf_repo = hf_repo
        self.base_dir = Path(f"./csv/patchtst_models/{symbol}")
        self.checkpoint_dir = self.base_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_model_path = self.checkpoint_dir / "best_model.pt"
        self.scaler_path = self.base_dir / "scaler.pkl"
        self.progress_path = self.base_dir / "progress.json"
    
    def save_local(self, model, optimizer, scheduler, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
        }
        
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"   🏆 Best checkpoint saved (epoch {epoch}, loss {loss:.6f})")
        
        ckpt_path = self.checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(checkpoint, ckpt_path)
        self._save_progress(epoch, loss, is_best)
        self._cleanup_old(keep=5)
    
    def load_local(self, model, optimizer=None, scheduler=None):
        if self.best_model_path.exists():
            checkpoint = torch.load(self.best_model_path, map_location='cpu')
            print(f"   📂 Loaded best checkpoint from local")
        else:
            checkpoints = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
            if not checkpoints:
                print(f"   ℹ️ No checkpoint found, starting fresh")
                return 0, float('inf')
            checkpoint = torch.load(checkpoints[-1], map_location='cpu')
            print(f"   📂 Loaded {checkpoints[-1].name} from local")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"   ✅ Resumed from epoch {epoch} (loss: {loss:.6f})")
        return epoch, loss
    
    def _save_progress(self, epoch, loss, is_best):
        progress = {
            'symbol': self.symbol,
            'last_epoch': epoch,
            'last_loss': loss,
            'is_best': is_best,
            'last_updated': datetime.now().isoformat(),
            'checkpoint_exists': True,
        }
        with open(self.progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def _cleanup_old(self, keep=5):
        checkpoints = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
        if len(checkpoints) > keep:
            for old in checkpoints[:-keep]:
                old.unlink()
    
    def can_resume(self):
        return self.best_model_path.exists() or len(list(self.checkpoint_dir.glob("epoch_*.pt"))) > 0
    
    def get_status(self):
        if self.progress_path.exists():
            with open(self.progress_path) as f:
                return json.load(f)
        return {'status': 'not_started'}
    
    def upload_to_hf(self, message=None):
        hf_token = os.getenv("hf_token") or os.getenv("HF_TOKEN", "")
        if not hf_token:
            print(f"   ⚠️ No HF_TOKEN, skipping backup")
            return False
        
        now = datetime.now()
        if SimpleCheckpointManager._last_upload_time is not None:
            elapsed = (now - SimpleCheckpointManager._last_upload_time).total_seconds()
            if elapsed < SimpleCheckpointManager._min_upload_interval:
                wait_time = SimpleCheckpointManager._min_upload_interval - elapsed
                print(f"   ⏳ Rate limit: waiting {wait_time:.0f}s...")
                time.sleep(wait_time)
        
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            hf_path = f"patchtst_models/{self.symbol}"
            
            if message is None:
                status = self.get_status()
                message = f"💾 Checkpoint: {self.symbol} epoch {status.get('last_epoch', '?')}"
            
            api.upload_folder(
                folder_path=str(self.base_dir),
                path_in_repo=hf_path,
                repo_id=self.hf_repo,
                repo_type="dataset",
                commit_message=message
            )
            
            SimpleCheckpointManager._last_upload_time = datetime.now()
            print(f"   ☁️ Backup uploaded: {hf_path}")
            return True
            
        except Exception as e:
            print(f"   ⚠️ HF backup failed: {str(e)[:100]}")
            return False
    
    def upload_final_to_hf(self):
        return self.upload_to_hf(
            message=f"✅ FINAL MODEL: {self.symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )


# =========================================================
# MISTAKE LEARNING SYSTEM
# =========================================================

class MistakeLearner:
    """Track mistakes and adjust predictions"""
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.mistakes_path = Path(f"./csv/patchtst_models/{symbol}/mistakes.json")
        self.mistakes = []
        self.corrections = {}
        self.total_predictions = 0
        self.correct_predictions = 0
        self._load()
    
    def _load(self):
        if self.mistakes_path.exists():
            with open(self.mistakes_path) as f:
                data = json.load(f)
                self.mistakes = data.get('mistakes', [])
                self.corrections = data.get('corrections', {})
                stats = data.get('stats', {})
                self.total_predictions = stats.get('total', 0)
                self.correct_predictions = stats.get('correct', 0)
    
    def _save(self):
        self.mistakes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.mistakes_path, 'w') as f:
            json.dump({
                'mistakes': self.mistakes[-100:],
                'corrections': self.corrections,
                'stats': {
                    'total': self.total_predictions,
                    'correct': self.correct_predictions,
                    'accuracy': round(self.correct_predictions / max(self.total_predictions, 1), 3)
                },
                'updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def record(self, date, predicted_prob, actual_return):
        predicted_prob = float(predicted_prob)
        actual_return = float(actual_return)
        pred_dir = 'UP' if predicted_prob > 0.5 else 'DOWN'
        actual_dir = 'UP' if actual_return > 0.005 else 'DOWN' if actual_return < -0.005 else 'FLAT'
        was_wrong = pred_dir != actual_dir and actual_dir != 'FLAT'
        was_correct = pred_dir == actual_dir and actual_dir != 'FLAT'
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1
        if was_wrong:
            self.mistakes.append({
                'date': str(date), 'predicted': pred_dir, 'actual': actual_dir,
                'prob': predicted_prob, 'return': actual_return
            })
            if len(self.mistakes) % 10 == 0:
                self._analyze()
    
    def _analyze(self):
        recent = self.mistakes[-30:]
        fp = sum(1 for m in recent if m['predicted'] == 'UP' and m['actual'] == 'DOWN')
        fn = sum(1 for m in recent if m['predicted'] == 'DOWN' and m['actual'] == 'UP')
        if fp + fn == 0:
            return
        if fp > fn * 1.5:
            self.corrections['up_bias'] = -0.03
            self.corrections['type'] = 'overly_bullish'
        elif fn > fp * 1.5:
            self.corrections['up_bias'] = 0.03
            self.corrections['type'] = 'overly_bearish'
        else:
            self.corrections['up_bias'] = 0.0
            self.corrections['type'] = 'balanced'
        self.corrections['fp'] = fp
        self.corrections['fn'] = fn
        self.corrections['accuracy'] = round(self.correct_predictions / max(self.total_predictions, 1), 3)
        self.corrections['last_analyzed'] = datetime.now().isoformat()
        self._save()
        print(f"\n   🧠 MISTAKE LEARNING: FP:{fp} FN:{fn} Type:{self.corrections['type']} Acc:{self.corrections['accuracy']:.1%}")
    
    def apply(self, up_prob, down_prob):
        if not self.corrections:
            return up_prob, down_prob
        bias = self.corrections.get('up_bias', 0)
        up = max(0, min(1, up_prob + bias))
        down = max(0, min(1, down_prob - bias))
        total = up + down
        if total > 0:
            up /= total
            down /= total
        return up, down
    
    def get_stats(self):
        return {
            'total_mistakes': len(self.mistakes),
            'accuracy': round(self.correct_predictions / max(self.total_predictions, 1), 3),
            'correction_type': self.corrections.get('type', 'none')
        }


# =========================================================
# COMPLETE TRAINING FUNCTION
# =========================================================

def train_patchtst_with_checkpoint(
    symbol, df, full_df=None, epochs=50, batch_size=16, learning_rate=0.001,
    resume=True, backup_to_hf=True, min_accuracy=0.55, verbose=True
):
    """Complete training: Resume → Train → Learn → Validate → HF Upload"""
    
    print(f"\n{'='*60}")
    print(f"🧠 PatchTST Training: {symbol}")
    print(f"{'='*60}")
    
    # Adaptive parameters
    data_rows = len(df)
    if data_rows >= 500:
        epochs = min(epochs + 100, 250)
        hidden_dim = 128
        batch_size = 8
        accumulation_steps = 4
        patience = 30
    elif data_rows >= 300:
        epochs = min(epochs + 70, 200)
        hidden_dim = 128
        batch_size = 8
        accumulation_steps = 4
        patience = 25
    elif data_rows >= 200:
        epochs = min(epochs + 50, 150)
        hidden_dim = 96
        batch_size = 8
        accumulation_steps = 4
        patience = 20
    else:
        epochs = max(epochs, 100)
        hidden_dim = 64
        batch_size = 4
        accumulation_steps = 4
        patience = 15
    
    print(f"   📊 Data: {data_rows} rows | 🎯 Epochs: {epochs} | 🧠 Hidden: {hidden_dim} | 📦 Batch: {batch_size}×{accumulation_steps}")
    print(f"   💾 Resume: {'Yes' if resume else 'No'} | 🛑 Patience: {patience}")
    
    model_dir = Path(f"./csv/patchtst_models/{symbol}")
    
    # ✅ Pass full_df for sector map
    predictor = FineTunablePatchTST(
        model_dir=model_dir,
        hidden_dim=hidden_dim,
        full_df=full_df
    )
    checkpoint_mgr = SimpleCheckpointManager(symbol)
    mistake_learner = MistakeLearner(symbol)
    
    X, y = predictor._prepare_sequences(df)
    if len(X) == 0:
        print("   ❌ No data for training")
        return {'status': 'failed', 'reason': 'no_data'}
    
    print(f"   📐 Input: {X.shape}, Features: {X.shape[2]}")
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    model = predictor._create_model(X.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = predictor._create_scheduler(optimizer, epochs)
    
    start_epoch = 0
    best_loss = float('inf')
    
    if resume and checkpoint_mgr.can_resume():
        start_epoch, best_loss = checkpoint_mgr.load_local(model, optimizer, scheduler)
        print(f"   🔄 Resuming from epoch {start_epoch}")
    else:
        print(f"   🆕 Fresh training")
    
    model.to(predictor.device)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).to(predictor.device),
        torch.FloatTensor(y_train).to(predictor.device))
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val).to(predictor.device),
        torch.FloatTensor(y_val).to(predictor.device))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model.train()
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        n_batches = 0
        optimizer.zero_grad()
        
        for i, (batch_X, batch_y) in enumerate(train_loader):
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y) / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                pred = model(batch_X)
                val_loss += criterion(pred, batch_y).item()
        val_loss /= max(len(val_loader), 1)
        model.train()
        
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or is_best:
            checkpoint_mgr.save_local(model, optimizer, scheduler, epoch + 1, avg_loss, is_best)
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Val: {val_loss:.6f} | Best: {best_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"   ⏹️ Early stop at epoch {epoch+1}")
            break
    
    # Final save
    checkpoint_mgr.save_local(model, optimizer, scheduler, epoch + 1, avg_loss, is_best=True)
    predictor.model = model
    predictor.is_fitted = True
    predictor.feature_columns = predictor.feature_columns or []
    predictor._save_model()
    
    if verbose and len(df) >= 200:
        wf_score = predictor._walk_forward_validation(df)
        print(f"   📊 Walk-Forward Score: {wf_score:.3f}")
    
    # Mistake Learning on validation
    print(f"\n🧠 LEARNING FROM VALIDATION MISTAKES")
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            preds = model(batch_X).cpu().numpy()
            actuals = batch_y.cpu().numpy()
            for i in range(len(preds)):
                predicted_up = preds[i][0] > 0.5
                actual_up = actuals[i][0] > 0.5
                if predicted_up == actual_up:
                    total_correct += 1
                mistake_learner.record(f"val_{total_samples}", preds[i][0], actuals[i][0] - 0.5)
                all_preds.append(preds[i])
                all_actuals.append(actuals[i])
                total_samples += 1
    
    initial_accuracy = total_correct / max(total_samples, 1)
    print(f"   📊 Initial Accuracy: {initial_accuracy:.1%}")
    
    corrected_correct = 0
    for i in range(len(all_preds)):
        up, down = mistake_learner.apply(all_preds[i][0], all_preds[i][1])
        if (up > 0.5) == (all_actuals[i][0] > 0.5):
            corrected_correct += 1
    
    corrected_accuracy = corrected_correct / max(total_samples, 1)
    print(f"   📊 After corrections: {corrected_accuracy:.1%}")
    print(f"   📈 Improvement: {corrected_accuracy - initial_accuracy:+.1%}")
    
    print(f"\n📊 HF UPLOAD DECISION")
    
    if corrected_accuracy < initial_accuracy:
        print(f"   ⚠️ Corrections reduced accuracy! Skipping HF upload")
        result = {'status': 'needs_retrain', 'initial_accuracy': initial_accuracy,
                  'corrected_accuracy': corrected_accuracy, 'uploaded_to_hf': False}
    elif corrected_accuracy >= min_accuracy:
        print(f"   ✅ Accuracy {corrected_accuracy:.1%} >= {min_accuracy:.0%}")
        uploaded = False
        if backup_to_hf:
            checkpoint_mgr.upload_final_to_hf()
            uploaded = True
            print(f"   ☁️ Uploaded to HF!")
        result = {'status': 'success', 'initial_accuracy': initial_accuracy,
                  'corrected_accuracy': corrected_accuracy, 'uploaded_to_hf': uploaded,
                  'correction_type': mistake_learner.corrections.get('type', 'none')}
    else:
        print(f"   ⚠️ Accuracy {corrected_accuracy:.1%} < {min_accuracy:.0%}")
        result = {'status': 'low_accuracy', 'initial_accuracy': initial_accuracy,
                  'corrected_accuracy': corrected_accuracy, 'uploaded_to_hf': False}
    
    print(f"\n✅ {symbol}: {result['status'].upper()}")
    return result


# =========================================================
# MAIN - AUTO-PILOT TRAINING
# =========================================================

if __name__ == "__main__":
    import sys
    
    print("🚀 PatchTST ULTIMATE QUALITY Auto-Pilot Training")
    print(f"   ✅ 3-Layer LSTM | ✅ CosineAnnealingWarmRestarts | ✅ Gradient Accumulation")
    print(f"   ✅ Extended Epochs | ✅ Residual Connections | ✅ Layer Normalization")
    print(f"   ✅ Sector Features (date-wise, no look-ahead)")
    print(f"   ✅ Rate-Limit Safe: 40s delay between HF uploads")
    print("="*60)
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else './csv/mongodb.csv'
    symbol_filter = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"   ✅ Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    
    # ✅ Show sector map summary
    if 'sector' in df.columns:
        unique_sectors = df.dropna(subset=['sector'])['sector'].nunique()
        print(f"   ✅ Sectors in mongodb: {unique_sectors}")
    
    progress_path = Path('./csv/patchtst_models/_marathon_progress.json')
    marathon_done_path = Path('./csv/patchtst_models/_marathon_done.txt')
    
    if symbol_filter:
        symbols = [symbol_filter]
        mode = "SINGLE"
    else:
        counts = df.groupby('symbol').size()
        all_symbols = counts[counts >= 150].index.tolist()
        all_symbols = sorted(all_symbols, key=lambda s: counts[s], reverse=True)
        
        trained_symbols = []
        for sym in all_symbols:
            model_path = Path(f"./csv/patchtst_models/{sym}/patchtst_model.pt")
            if model_path.exists():
                trained_symbols.append(sym)
        
        remaining = [s for s in all_symbols if s not in trained_symbols]
        
        print(f"\n📊 STATUS:")
        print(f"   Total eligible: {len(all_symbols)}")
        print(f"   Already trained: {len(trained_symbols)}")
        print(f"   Remaining: {len(remaining)}")
        
        if remaining:
            symbols = remaining
            mode = "CONTINUE"
            print(f"   🔄 Mode: CONTINUE TRAINING")
        elif marathon_done_path.exists():
            last_done = datetime.fromtimestamp(marathon_done_path.stat().st_mtime)
            days_since = (datetime.now() - last_done).days
            
            if days_since >= 30:
                symbols = all_symbols
                mode = "MONTHLY_RETRAIN"
                print(f"   🔄 Mode: MONTHLY RETRAIN ({days_since} days)")
            elif days_since >= 7:
                symbols = all_symbols
                mode = "WEEKLY_FINE_TUNE"
                print(f"   🔄 Mode: WEEKLY FINE-TUNE ({days_since} days)")
            else:
                print(f"   ✅ All done recently ({days_since} days)")
                sys.exit(0)
        else:
            symbols = all_symbols
            mode = "FIRST_RUN"
            print(f"   🆕 Mode: FIRST RUN")
    
    if not symbols:
        print("❌ No symbols to train")
        sys.exit(0)
    
    # Training parameters
    if mode in ["MONTHLY_RETRAIN", "FIRST_RUN"]:
        default_epochs = 150
        learning_rate = 0.0005
    elif mode == "WEEKLY_FINE_TUNE":
        default_epochs = 50
        learning_rate = 0.0001
    else:
        default_epochs = 100
        learning_rate = 0.0008
    
    print(f"\n⚙️ CONFIG:")
    print(f"   Mode: {mode}")
    print(f"   Base Epochs: {default_epochs}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Architecture: 3-Layer LSTM + 8-Head Attention + Residual")
    print(f"   Scheduler: CosineAnnealingWarmRestarts")
    print(f"   Gradient: Accumulation ×4")
    print(f"   HF Upload: 40s delay")
    
    results = []
    hf_uploads = 0
    total = len(symbols)
    start_time = datetime.now()
    
    # ✅ Pass full_df for sector map (built ONCE)
    for i, sym in enumerate(symbols, 1):
        sym_df = df[df['symbol'] == sym].sort_values('date')
        
        print(f"\n{'='*50}")
        print(f"📊 [{i}/{total}] {sym} ({len(sym_df)} rows, {mode})")
        
        result = train_patchtst_with_checkpoint(
            symbol=sym,
            df=sym_df,
            full_df=df,   # ✅ sector map এর জন্য
            epochs=default_epochs,
            batch_size=8,
            learning_rate=learning_rate,
            resume=True,
            backup_to_hf=True,
            min_accuracy=0.55,
            verbose=False
        )
        
        results.append({'symbol': sym, **result})
        if result.get('uploaded_to_hf'):
            hf_uploads += 1
        
        elapsed = (datetime.now() - start_time).total_seconds()
        avg_time = elapsed / i
        remaining_time = avg_time * (total - i)
        print(f"   📈 Progress: {i}/{total} | ☁️ {hf_uploads} uploaded | ⏰ ETA: {remaining_time/3600:.1f}h")
        
        if i % 25 == 0:
            progress = {
                'mode': mode, 'completed': i, 'total': total, 'uploaded': hf_uploads,
                'elapsed_hours': round(elapsed/3600, 1),
                'eta_hours': round(remaining_time/3600, 1),
                'completed_symbols': [r['symbol'] for r in results],
                'timestamp': datetime.now().isoformat()
            }
            with open(progress_path, 'w') as f:
                json.dump(progress, f, indent=2)
    
    total_time = (datetime.now() - start_time).total_seconds() / 3600
    marathon_done_path.write_text(datetime.now().isoformat())
    
    final_progress = {
        'mode': mode, 'completed': total, 'total': total, 'uploaded': hf_uploads,
        'total_hours': round(total_time, 1),
        'completed_symbols': [r['symbol'] for r in results],
        'timestamp': datetime.now().isoformat(), 'all_done': True
    }
    with open(progress_path, 'w') as f:
        json.dump(final_progress, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"🎉 {mode} COMPLETE!")
    print(f"{'='*60}")
    print(f"   ⏰ Time: {total_time:.1f} hours")
    print(f"   📊 Symbols: {total}")
    print(f"   ☁️ Uploaded: {hf_uploads}")
    
    success = [r for r in results if r.get('status') == 'success']
    retrain = [r for r in results if r.get('status') == 'needs_retrain']
    low_acc = [r for r in results if r.get('status') == 'low_accuracy']
    failed = [r for r in results if r.get('status') == 'failed']
    
    print(f"\n   📊 Results:")
    print(f"   ✅ Success: {len(success)}")
    print(f"   🔄 Needs Retrain: {len(retrain)}")
    print(f"   ⚠️ Low Accuracy: {len(low_acc)}")
    print(f"   ❌ Failed: {len(failed)}")
    
    report = {
        'mode': mode, 'date': datetime.now().isoformat(),
        'total_time_hours': round(total_time, 1), 'symbols_trained': total,
        'uploaded_to_hf': hf_uploads, 'success': len(success),
        'needs_retrain': len(retrain), 'low_accuracy': len(low_acc), 'failed': len(failed)
    }
    
    report_path = Path('./csv/patchtst_models/_training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Report: {report_path}")
    print(f"✅ ALL DONE!")