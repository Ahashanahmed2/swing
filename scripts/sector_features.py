# ================== sector_features.py ==================
# লেটেস্ট date থেকে sector → symbol ম্যাপিং
# সম্পূর্ণ ডায়নামিক - mongodb.csv এর sector কলাম থেকে অটো-বিল্ড

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

class SectorFeatureEngine:
    """
    Weekly Sector Rotation Features
    mongodb.csv-এর sector কলাম থেকে অটোমেটিক ম্যাপিং
    লেটেস্ট date-এর ডেটা প্রায়োরিটি পায়
    """
    
    def __init__(self, csv_market_path="./csv/mongodb.csv", weekly_data_path="./csv/weekly_sector.csv"):
        self.csv_market_path = Path(csv_market_path)
        self.weekly_data_path = Path(weekly_data_path)
        
        # স্টোরেজ
        self.sector_map = {}           # {sector: [symbols]} — লেটেস্ট date থেকে
        self.symbol_to_sector = {}     # {symbol: sector}
        self.sector_returns = {}
        self.sector_momentum = {}
        self.sector_ranking = {}
        self.all_sectors = []
        self.latest_date = None
        
        # অটো-বিল্ড
        self._build_sector_map_from_latest_date()
    
    # =====================================================
    # 🔥 লেটেস্ট DATE থেকে সেক্টর ম্যাপিং
    # =====================================================
    
    def _build_sector_map_from_latest_date(self):
        """সবচেয়ে লেটেস্ট date থেকে sector → symbol ম্যাপিং তৈরি"""
        
        if not self.csv_market_path.exists():
            print(f"⚠️ Market data not found: {self.csv_market_path}")
            return
        
        try:
            # Load data
            df = pd.read_csv(self.csv_market_path)
            
            # Date column detect & convert
            date_col = self._detect_date_column(df)
            if date_col is None:
                print("❌ No date column found!")
                return
            
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Sector column check
            if 'sector' not in df.columns:
                print("❌ 'sector' column not found in CSV!")
                print(f"   Available columns: {list(df.columns)}")
                return
            
            # 🔥 লেটেস্ট date বের করুন
            self.latest_date = df[date_col].max()
            print(f"📅 Latest Date: {self.latest_date.strftime('%Y-%m-%d')}")
            
            # লেটেস্ট date-এর ডেটা ফিল্টার
            latest_df = df[df[date_col] == self.latest_date].copy()
            print(f"📊 Latest date records: {len(latest_df)} rows")
            
            # Sector → Symbols (লেটেস্ট date থেকে)
            sector_groups = latest_df.groupby('sector')['symbol'].apply(list).to_dict()
            
            # হিস্টোরিক্যাল ডেটাও চেক করুন (নতুন সিম্বলের জন্য)
            all_dates_df = df.groupby(['sector', 'symbol']).size().reset_index()
            historical_groups = all_dates_df.groupby('sector')['symbol'].apply(list).to_dict()
            
            # মার্জ: লেটেস্ট + হিস্টোরিক্যাল (কোনো সিম্বল মিস না হয়)
            for sector, symbols in historical_groups.items():
                if sector not in sector_groups:
                    sector_groups[sector] = []
                # Add symbols not in latest date
                existing = set(sector_groups[sector])
                for sym in symbols:
                    if sym not in existing:
                        sector_groups[sector].append(sym)
                        print(f"   📌 Added from history: {sym} → {sector}")
            
            # Store
            self.sector_map = sector_groups
            self.all_sectors = sorted(sector_groups.keys())
            
            # Reverse mapping: symbol → sector
            self.symbol_to_sector = {}
            for sector, symbols in self.sector_map.items():
                for symbol in symbols:
                    self.symbol_to_sector[symbol] = sector
            
            # Summary
            print(f"\n{'='*50}")
            print(f"📊 SECTOR MAP (from latest date)")
            print(f"{'='*50}")
            print(f"   Latest Date: {self.latest_date.strftime('%Y-%m-%d')}")
            print(f"   Total Sectors: {len(self.all_sectors)}")
            print(f"   Total Symbols: {len(self.symbol_to_sector)}")
            print(f"{'='*50}")
            
            # Top 10 sectors by symbol count
            sector_counts = {s: len(syms) for s, syms in self.sector_map.items()}
            top_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            print(f"\n   Top 10 Sectors by Symbol Count:")
            for i, (sector, count) in enumerate(top_sectors, 1):
                print(f"   {i:2d}. {sector:30s} → {count:3d} symbols")
            print(f"{'='*50}\n")
            
        except Exception as e:
            print(f"❌ Error building sector map: {e}")
            import traceback
            traceback.print_exc()
    
    def _detect_date_column(self, df):
        """Date column auto-detect"""
        possible_names = ['date', 'Date', 'DATE', 'datetime', 'timestamp', 'trade_date']
        
        for col in possible_names:
            if col in df.columns:
                return col
        
        # Fuzzy match
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                return col
        
        return None
    
    def reload_sector_map(self):
        """Re-build sector map (new data এলে কল করুন)"""
        print("🔄 Reloading sector map from latest date...")
        self.sector_map = {}
        self.symbol_to_sector = {}
        self.all_sectors = []
        self._build_sector_map_from_latest_date()
    
    # =====================================================
    # SECTOR LOOKUP
    # =====================================================
    
    def get_sector(self, symbol):
        """Symbol → Sector"""
        if symbol in self.symbol_to_sector:
            return self.symbol_to_sector[symbol]
        
        # Case-insensitive
        symbol_upper = symbol.upper() if isinstance(symbol, str) else str(symbol)
        for sym, sec in self.symbol_to_sector.items():
            if sym.upper() == symbol_upper:
                return sec
        
        return 'OTHER'
    
    def get_symbols_in_sector(self, sector):
        """একটি সেক্টরের সব সিম্বল"""
        return self.sector_map.get(sector, [])
    
    def get_all_sectors(self):
        """সব সেক্টর লিস্ট"""
        return self.all_sectors
    
    def get_symbols_for_sectors(self, sectors):
        """একাধিক সেক্টরের সিম্বল (ট্রেডিং ফিল্টারের জন্য)"""
        symbols = []
        for sector in sectors:
            symbols.extend(self.get_symbols_in_sector(sector))
        return list(set(symbols))
    
    # =====================================================
    # WEEKLY RETURNS CALCULATION
    # =====================================================
    
    def calculate_weekly_sector_returns(self, df=None):
        """Weekly returns per sector"""
        
        if df is None:
            if not self.csv_market_path.exists():
                return pd.DataFrame()
            df = pd.read_csv(self.csv_market_path)
        
        # Date column
        date_col = self._detect_date_column(df)
        if date_col is None:
            return pd.DataFrame()
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Week number
        df['week'] = df[date_col].dt.isocalendar().week.astype(int)
        df['year'] = df[date_col].dt.isocalendar().year.astype(int)
        
        # Sector from existing column
        if 'sector' not in df.columns:
            df['sector'] = df['symbol'].map(self.get_sector)
        
        # Filter valid sectors
        df_valid = df[df['sector'] != 'OTHER'].copy()
        
        if df_valid.empty:
            return pd.DataFrame()
        
        # Group by year, week, sector
        agg_dict = {'close': ['first', 'last', 'mean']}
        if 'volume' in df_valid.columns:
            agg_dict['volume'] = 'sum'
        
        sector_weekly = df_valid.groupby(['year', 'week', 'sector']).agg(agg_dict).reset_index()
        
        # Flatten columns
        if 'volume' in df_valid.columns:
            sector_weekly.columns = ['year', 'week', 'sector', 'open', 'close', 'avg_price', 'volume']
        else:
            sector_weekly.columns = ['year', 'week', 'sector', 'open', 'close', 'avg_price']
            sector_weekly['volume'] = 0
        
        # Return
        sector_weekly['return'] = np.where(
            sector_weekly['open'] > 0,
            (sector_weekly['close'] - sector_weekly['open']) / sector_weekly['open'],
            0
        )
        
        return sector_weekly
    
    def calculate_sector_momentum(self, sector_weekly, lookback_weeks=4):
        """Multi-timeframe momentum"""
        momentum = {}
        
        for sector in sector_weekly['sector'].unique():
            sector_data = sector_weekly[sector_weekly['sector'] == sector].tail(lookback_weeks)
            
            if len(sector_data) >= lookback_weeks:
                returns = sector_data['return'].values
                mom_1w = returns[-1] if len(returns) >= 1 else 0
                mom_2w = np.mean(returns[-2:]) if len(returns) >= 2 else mom_1w
                mom_4w = np.mean(returns)
                momentum[sector] = (mom_1w * 0.50) + (mom_2w * 0.30) + (mom_4w * 0.20)
            else:
                momentum[sector] = 0.0
        
        return momentum
    
    def calculate_sector_relative_strength(self, sector_momentum):
        """Z-score based ranking (0-100)"""
        if not sector_momentum:
            return {}
        
        sectors = list(sector_momentum.keys())
        momentums = np.array(list(sector_momentum.values()))
        
        mean_mom = np.mean(momentums)
        std_mom = np.std(momentums) + 1e-8
        z_scores = (momentums - mean_mom) / std_mom
        
        rankings = {}
        for i, sector in enumerate(sectors):
            percentile = (z_scores[i] + 3) / 6 * 100
            rankings[sector] = np.clip(percentile, 0, 100)
        
        return rankings
    
    # =====================================================
    # PPO FEATURE GENERATION
    # =====================================================
    
    def generate_features(self, df, symbol, current_date=None):
        """PPO-র জন্য ৫টি সেক্টর ফিচার"""
        
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
        if sector in self.sector_returns:
            sector_history = self.sector_returns[sector]
            if len(sector_history) >= 4:
                try:
                    trend_strength = np.corrcoef(range(len(sector_history)), sector_history)[0, 1]
                    trend_strength = 0.0 if np.isnan(trend_strength) else trend_strength
                except:
                    trend_strength = 0.0
            else:
                trend_strength = 0.0
        else:
            trend_strength = 0.0
        
        return {
            'sector': sector,
            'sector_momentum': round(sector_mom, 6),
            'sector_ranking': round(sector_rank, 4),
            'relative_strength': round(relative_strength, 6),
            'is_top_sector': is_top_sector,
            'sector_trend_strength': round(trend_strength, 4)
        }
    
    def get_feature_vector(self, symbol):
        """ফিচার ভেক্টর (numpy array) — env-তে ব্যবহারের জন্য"""
        features = self.generate_features(df=None, symbol=symbol)
        return np.array([
            features['sector_momentum'],
            features['sector_ranking'],
            features['relative_strength'],
            features['is_top_sector'],
            features['sector_trend_strength']
        ], dtype=np.float32)
    
    # =====================================================
    # UPDATE
    # =====================================================
    
    def update(self, df=None):
        """Weekly metrics update"""
        
        if df is None:
            if not self.csv_market_path.exists():
                return False
            df = pd.read_csv(self.csv_market_path)
        
        # Rebuild map if needed
        if len(self.symbol_to_sector) == 0:
            self._build_sector_map_from_latest_date()
        
        sector_weekly = self.calculate_weekly_sector_returns(df)
        
        if sector_weekly.empty:
            return False
        
        self.sector_momentum = self.calculate_sector_momentum(sector_weekly)
        self.sector_ranking = self.calculate_sector_relative_strength(self.sector_momentum)
        
        # Store history
        for sector in sector_weekly['sector'].unique():
            sector_data = sector_weekly[sector_weekly['sector'] == sector]
            self.sector_returns[sector] = sector_data['return'].tail(12).tolist()
        
        return True
    
    def get_top_sectors(self, n=3):
        """Top N sectors"""
        if not self.sector_ranking:
            return []
        return sorted(self.sector_ranking.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def get_bottom_sectors(self, n=3):
        """Bottom N sectors"""
        if not self.sector_ranking:
            return []
        return sorted(self.sector_ranking.items(), key=lambda x: x[1])[:n]
    
    # =====================================================
    # EXPORT
    # =====================================================
    
    def export_sector_rankings(self, path=None):
        """Export sector rankings to CSV"""
        if path is None:
            path = self.weekly_data_path
        
        if not self.sector_ranking:
            return
        
        rows = []
        for sector, rank in self.sector_ranking.items():
            rows.append({
                'sector': sector,
                'rank': round(rank, 2),
                'momentum': round(self.sector_momentum.get(sector, 0), 6),
                'symbol_count': len(self.sector_map.get(sector, [])),
                'is_top_3': sector in [s for s, _ in self.get_top_sectors(3)],
                'symbols': ', '.join(self.sector_map.get(sector, [])[:10])
            })
        
        df = pd.DataFrame(rows).sort_values('rank', ascending=False)
        df.to_csv(path, index=False)
        print(f"✅ Rankings exported to {path}")
        return df
    
    def get_summary(self):
        """Summary statistics"""
        return {
            'latest_date': self.latest_date.strftime('%Y-%m-%d') if self.latest_date else 'N/A',
            'total_sectors': len(self.all_sectors),
            'total_symbols': len(self.symbol_to_sector),
            'top_3': self.get_top_sectors(3),
            'bottom_3': self.get_bottom_sectors(3)
        }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
