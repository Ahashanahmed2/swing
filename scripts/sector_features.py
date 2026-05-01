# ================== sector_features.py ==================
# ✅ Data from ./csv/sector/weekly/ + ./csv/sector/daily/
# ✅ Sector Map from mongodb.csv (latest date)
# ✅ RSI, Momentum, Ranking, Trend — all from sector CSVs

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os

class SectorFeatureEngine:
    """
    Weekly Sector Rotation Features
    ✅ Sector Map: mongodb.csv → latest date
    ✅ Sector Data: ./csv/sector/weekly/*.csv + ./csv/sector/daily/*.csv
    """
    
    def __init__(self, csv_market_path="./csv/mongodb.csv", 
                 weekly_dir="./csv/sector/weekly/",
                 daily_dir="./csv/sector/daily/"):
        self.csv_market_path = Path(csv_market_path)
        self.weekly_dir = Path(weekly_dir)
        self.daily_dir = Path(daily_dir)
        
        # স্টোরেজ
        self.sector_map = {}
        self.symbol_to_sector = {}
        self.all_sectors = []
        self.latest_date = None
        
        # Sector data cache
        self.sector_data = {}        # {sector: weekly_df}
        self.sector_daily_data = {} # {sector: daily_df}
        self.sector_momentum = {}
        self.sector_ranking = {}
        
        # অটো-বিল্ড
        self._build_sector_map_from_latest_date()
        self._load_sector_data()
    
    # =====================================================
    # 🔥 SECTOR MAP (unchanged)
    # =====================================================
    
    def _build_sector_map_from_latest_date(self):
        """mongodb.csv থেকে sector → symbol ম্যাপিং"""
        # ... (আপনার existing code — unchanged) ...
        pass
    
    # =====================================================
    # ✅ NEW: Load Sector Data from CSVs
    # =====================================================
    
    def _load_sector_data(self):
        """Load weekly + daily sector data from ./csv/sector/"""
        self.sector_data = {}
        self.sector_daily_data = {}
        
        # Load weekly files
        if self.weekly_dir.exists():
            for file in self.weekly_dir.glob("*_weekly.csv"):
                try:
                    df = pd.read_csv(file)
                    if 'week_start' in df.columns:
                        df['week_start'] = pd.to_datetime(df['week_start'])
                    sector_name = file.stem.replace('_weekly', '').replace('_', ' ').title()
                    self.sector_data[sector_name] = df
                except:
                    pass
        
        # Load daily files
        if self.daily_dir.exists():
            for file in self.daily_dir.glob("*_daily.csv"):
                try:
                    df = pd.read_csv(file)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    sector_name = file.stem.replace('_daily', '').replace('_', ' ').title()
                    self.sector_daily_data[sector_name] = df
                except:
                    pass
        
        if self.sector_data:
            print(f"✅ Loaded {len(self.sector_data)} sector weekly files")
        if self.sector_daily_data:
            print(f"✅ Loaded {len(self.sector_daily_data)} sector daily files")
    
    # =====================================================
    # ✅ NEW: Get Sector RSI
    # =====================================================
    
    def get_sector_rsi(self, sector, timeframe='weekly'):
        """Sector RSI from CSV"""
        data = self.sector_data if timeframe == 'weekly' else self.sector_daily_data
        
        # Try exact match
        if sector in data:
            df = data[sector]
        else:
            # Try case-insensitive
            for s, df in data.items():
                if s.lower() == sector.lower():
                    break
            else:
                return 50.0
        
        if 'rsi' in df.columns:
            rsi_values = df['rsi'].dropna()
            if len(rsi_values) > 0:
                return float(rsi_values.iloc[-1])
        
        return 50.0
    
    # =====================================================
    # ✅ NEW: Get Sector Momentum (from CSV, not calculate)
    # =====================================================
    
    def _calculate_momentum_from_csv(self):
        """CSV থেকে momentum calculate"""
        momentum = {}
        
        for sector, df in self.sector_data.items():
            if 'close' not in df.columns or len(df) < 4:
                continue
            
            closes = df['close'].dropna()
            if len(closes) < 4:
                continue
            
            # Calculate returns
            returns = closes.pct_change().dropna()
            if len(returns) < 3:
                continue
            
            mom_1w = returns.iloc[-1] if len(returns) >= 1 else 0
            mom_2w = returns.iloc[-2:].mean() if len(returns) >= 2 else mom_1w
            mom_4w = returns.iloc[-4:].mean() if len(returns) >= 4 else mom_2w
            
            momentum[sector] = (mom_1w * 0.50) + (mom_2w * 0.30) + (mom_4w * 0.20)
        
        return momentum
    
    def _calculate_ranking_from_momentum(self, momentum):
        """Z-score ranking from momentum dict"""
        if not momentum:
            return {}
        
        sectors = list(momentum.keys())
        values = np.array(list(momentum.values()))
        
        mean_v = np.mean(values)
        std_v = np.std(values) + 1e-8
        z_scores = (values - mean_v) / std_v
        
        rankings = {}
        for i, sector in enumerate(sectors):
            percentile = (z_scores[i] + 3) / 6 * 100
            rankings[sector] = np.clip(percentile, 0, 100)
        
        return rankings
    
    # =====================================================
    # SECTOR LOOKUP (unchanged)
    # =====================================================
    
    def get_sector(self, symbol):
        if symbol in self.symbol_to_sector:
            return self.symbol_to_sector[symbol]
        symbol_upper = symbol.upper() if isinstance(symbol, str) else str(symbol)
        for sym, sec in self.symbol_to_sector.items():
            if sym.upper() == symbol_upper:
                return sec
        return 'OTHER'
    
    def get_symbols_in_sector(self, sector):
        return self.sector_map.get(sector, [])
    
    def get_all_sectors(self):
        return self.all_sectors
    
    # =====================================================
    # ✅ UPDATED: Feature Generation (with RSI)
    # =====================================================
    
    def generate_features(self, df, symbol, current_date=None):
        """PPO-র জন্য 8টি সেক্টর ফিচার (RSI সহ)"""
        
        sector = self.get_sector(symbol)
        
        # 1. Sector Momentum
        sector_mom = self.sector_momentum.get(sector, 0.0)
        
        # 2. Sector Ranking (0-1)
        sector_rank = self.sector_ranking.get(sector, 50.0) / 100.0
        
        # 3. Relative Strength
        all_momentums = list(self.sector_momentum.values())
        market_avg_mom = np.mean(all_momentums) if all_momentums else 0.0
        relative_strength = sector_mom - market_avg_mom
        
        # 4. Top Sector?
        sorted_sectors = sorted(self.sector_ranking.items(), key=lambda x: x[1], reverse=True)
        top_3_sectors = [s for s, _ in sorted_sectors[:3]]
        is_top_sector = 1.0 if sector in top_3_sectors else 0.0
        
        # 5. Trend Strength
        trend_strength = 0.0
        if sector in self.sector_data:
            df_sector = self.sector_data[sector]
            if 'close' in df_sector.columns and len(df_sector) >= 4:
                closes = df_sector['close'].dropna()
                if len(closes) >= 4:
                    try:
                        trend_strength = np.corrcoef(range(len(closes)), closes)[0, 1]
                        trend_strength = 0.0 if np.isnan(trend_strength) else trend_strength
                    except:
                        pass
        
        # ✅ 6. Weekly RSI
        weekly_rsi = self.get_sector_rsi(sector, 'weekly')
        weekly_rsi_norm = (weekly_rsi - 30) / 40  # Normalize 30-70 → 0-1
        weekly_rsi_norm = np.clip(weekly_rsi_norm, 0, 1)
        
        # ✅ 7. Daily RSI
        daily_rsi = self.get_sector_rsi(sector, 'daily')
        daily_rsi_norm = (daily_rsi - 30) / 40
        daily_rsi_norm = np.clip(daily_rsi_norm, 0, 1)
        
        # ✅ 8. RSI Divergence Signal (Weekly RSI vs Price)
        rsi_divergence = 0.5  # Neutral
        if sector in self.sector_data:
            df_sector = self.sector_data[sector]
            if 'rsi' in df_sector.columns and 'close' in df_sector.columns:
                valid = df_sector.dropna(subset=['rsi', 'close'])
                if len(valid) >= 2:
                    last = valid.iloc[-1]
                    prev = valid.iloc[-2]
                    if last['close'] < prev['close'] and last['rsi'] > prev['rsi']:
                        rsi_divergence = 1.0  # Bullish divergence
                    elif last['close'] > prev['close'] and last['rsi'] < prev['rsi']:
                        rsi_divergence = 0.0  # Bearish divergence
        
        return {
            'sector': sector,
            'sector_momentum': round(sector_mom, 6),
            'sector_ranking': round(sector_rank, 4),
            'relative_strength': round(relative_strength, 6),
            'is_top_sector': is_top_sector,
            'sector_trend_strength': round(trend_strength, 4),
            'weekly_rsi': round(weekly_rsi_norm, 4),
            'daily_rsi': round(daily_rsi_norm, 4),
            'rsi_divergence': rsi_divergence
        }
    
    def get_feature_vector(self, symbol):
        """8-dim feature vector"""
        features = self.generate_features(df=None, symbol=symbol)
        return np.array([
            features['sector_momentum'],
            features['sector_ranking'],
            features['relative_strength'],
            features['is_top_sector'],
            features['sector_trend_strength'],
            features['weekly_rsi'],        # ✅ NEW
            features['daily_rsi'],         # ✅ NEW
            features['rsi_divergence']     # ✅ NEW
        ], dtype=np.float32)
    
    # =====================================================
    # UPDATE
    # =====================================================
    
    def update(self, df=None):
        """Update sector data from CSVs"""
        self._load_sector_data()
        self.sector_momentum = self._calculate_momentum_from_csv()
        self.sector_ranking = self._calculate_ranking_from_momentum(self.sector_momentum)
        return True
    
    def get_top_sectors(self, n=3):
        if not self.sector_ranking:
            return []
        return sorted(self.sector_ranking.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def get_bottom_sectors(self, n=3):
        if not self.sector_ranking:
            return []
        return sorted(self.sector_ranking.items(), key=lambda x: x[1])[:n]
    
    def get_summary(self):
        return {
            'latest_date': self.latest_date.strftime('%Y-%m-%d') if self.latest_date else 'N/A',
            'total_sectors': len(self.all_sectors),
            'total_symbols': len(self.symbol_to_sector),
            'weekly_files': len(self.sector_data),
            'daily_files': len(self.sector_daily_data),
            'top_3': self.get_top_sectors(3),
            'bottom_3': self.get_bottom_sectors(3)
        }
