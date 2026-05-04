# ================== patch_tst_predictor.py (UPDATED) ==================
# ... উপরের সব কোড অপরিবর্তিত থাকবে ...
# নিচের অংশটি _engineer_features মেথডের পরে যোগ করুন

class PatchTSTPredictor:
    # ... আগের সব কোড ...
    
    def __init__(
        self,
        seq_len=60,
        pred_len=5,
        hidden_dim=64,
        model_dir="./csv/patchtst_models",
        device=None,
        use_sector_features=True,      # ✅ Sector ফিচার
        use_sr_features=True,          # ✅ Support/Resistance
        use_rsi_div_features=True,     # ✅ RSI Divergence
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ নতুন ফিচার ফ্ল্যাগ
        self.use_sector_features = use_sector_features
        self.use_sr_features = use_sr_features
        self.use_rsi_div_features = use_rsi_div_features
        
        # ✅ ক্যাশ ফিচার ডাটা
        self.sector_data = {}
        self.sr_data = None
        self.rsi_div_data = {}
        
        # ✅ ফিচার ডাটা লোড
        self._load_external_features()
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else 'cpu'
        else:
            self.device = device
        
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_fitted = False
        self.feature_columns = None
        
        print(f"✅ PatchTST Predictor initialized (device: {self.device})")
    
    # -------------------------------------------------
    # ✅ NEW: LOAD EXTERNAL FEATURES
    # -------------------------------------------------
    def _load_external_features(self):
        """Load sector, support/resistance, RSI divergence from ./csv/"""
        
        # Load Sector Features
        if self.use_sector_features:
            self._load_sector_features()
        
        # Load Support/Resistance
        if self.use_sr_features:
            self._load_support_resistance()
        
        # Load RSI Divergence
        if self.use_rsi_div_features:
            self._load_rsi_divergence()
    
    def _load_sector_features(self):
        """Load sector features from ./csv/sector/"""
        sector_dir = Path('./csv/sector')
        
        if not sector_dir.exists():
            print("   ⚠️ Sector directory not found: ./csv/sector/")
            self.use_sector_features = False
            return
        
        try:
            # ✅ Weekly sector files
            weekly_files = list(sector_dir.glob('*weekly*.csv'))
            daily_files = list(sector_dir.glob('*daily*.csv'))
            
            for f in weekly_files + daily_files:
                try:
                    df = pd.read_csv(f)
                    
                    # RSI থাকলে ব্যবহার
                    has_rsi = 'rsi' in df.columns
                    
                    for _, row in df.iterrows():
                        symbol = row.get('symbol', '')
                        if symbol:
                            if symbol not in self.sector_data:
                                self.sector_data[symbol] = {}
                            
                            # Sector momentum features
                            self.sector_data[symbol]['sector_returns'] = float(row.get('returns', 0))
                            self.sector_data[symbol]['sector_volume'] = float(row.get('volume_ratio', 1))
                            
                            if has_rsi:
                                self.sector_data[symbol]['sector_rsi'] = float(row.get('rsi', 50))
                except:
                    pass
            
            print(f"   ✅ Loaded sector features for {len(self.sector_data)} symbols")
            
        except Exception as e:
            print(f"   ⚠️ Sector load error: {e}")
            self.use_sector_features = False
    
    def _load_support_resistance(self):
        """Load support/resistance from ./csv/support_resistance.csv"""
        sr_path = Path('./csv/support_resistance.csv')
        
        if not sr_path.exists():
            print("   ⚠️ support_resistance.csv not found")
            self.use_sr_features = False
            return
        
        try:
            self.sr_data = pd.read_csv(sr_path)
            if 'current_date' in self.sr_data.columns:
                self.sr_data['current_date'] = pd.to_datetime(self.sr_data['current_date'])
            
            print(f"   ✅ Loaded S/R data: {self.sr_data['symbol'].nunique()} symbols")
            
        except Exception as e:
            print(f"   ⚠️ S/R load error: {e}")
            self.use_sr_features = False
    
    def _load_rsi_divergence(self):
        """Load RSI divergence from ./csv/rsi_diver.csv"""
        rsi_path = Path('./csv/rsi_diver.csv')
        
        if not rsi_path.exists():
            print("   ⚠️ rsi_diver.csv not found")
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
            print(f"   ⚠️ RSI divergence load error: {e}")
            self.use_rsi_div_features = False
    
    # -------------------------------------------------
    # ✅ NEW: GET EXTERNAL FEATURES FOR A ROW
    # -------------------------------------------------
    def _get_sector_features_for_row(self, symbol, current_date):
        """Get sector features for a specific date"""
        if not self.use_sector_features or symbol not in self.sector_data:
            return {}
        
        return self.sector_data.get(symbol, {})
    
    def _get_sr_features_for_row(self, symbol, current_date, current_close):
        """Get support/resistance features for a specific date"""
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
            
        except:
            return {}
    
    def _get_rsi_div_features_for_row(self, symbol, current_date):
        """Get RSI divergence features for a specific date"""
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
            
        except:
            return {}
    
    # -------------------------------------------------
    # UPDATED: _engineer_features with NEW features
    # -------------------------------------------------
    def _engineer_features(self, df):
        """Create features from OHLCV data + External features"""
        df = df.copy()
        
        # ============================
        # ORIGINAL FEATURES (unchanged)
        # ============================
        
        # Basic returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_trend'] = df['volume'].pct_change(5)
        
        # Price features
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(10).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(30).mean()
        
        # Trend
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_30'] = df['close'].rolling(30).mean()
        df['trend_strength'] = (df['sma_10'] - df['sma_30']) / df['sma_30']
        
        # Momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Price position
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)
        
        # Existing indicators (if available)
        for col in ['rsi', 'macd', 'macd_signal', 'macd_hist', 'atr']:
            if col in df.columns:
                df[f'{col}_norm'] = df[col] / (df[col].abs().rolling(50).mean() + 1e-8)
        
        # ============================
        # ✅ NEW: EXTERNAL FEATURES
        # ============================
        
        # Sector features
        if self.use_sector_features:
            df['sector_momentum'] = 0.0
            df['sector_volume_ratio'] = 1.0
            df['sector_rsi'] = 50.0
            
            symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else ''
            sector_feat = self._get_sector_features_for_row(symbol, None)
            
            if sector_feat:
                df['sector_momentum'] = sector_feat.get('sector_returns', 0)
                df['sector_volume_ratio'] = sector_feat.get('sector_volume', 1)
                df['sector_rsi'] = sector_feat.get('sector_rsi', 50)
        
        # Support/Resistance features (per row)
        if self.use_sr_features and 'date' in df.columns:
            df['sr_distance'] = 0.0
            df['sr_strength'] = 0.5
            df['sr_type'] = 0.0
            
            for idx in df.index:
                row = df.loc[idx]
                sr_feat = self._get_sr_features_for_row(
                    row.get('symbol', ''), 
                    row.get('date'), 
                    row.get('close', 0)
                )
                if sr_feat:
                    df.loc[idx, 'sr_distance'] = sr_feat.get('sr_distance', 0)
                    df.loc[idx, 'sr_strength'] = sr_feat.get('sr_strength', 0.5)
                    df.loc[idx, 'sr_type'] = sr_feat.get('sr_type', 0)
        
        # RSI Divergence features (per row)
        if self.use_rsi_div_features and 'date' in df.columns:
            df['rsi_div_bullish'] = 0.0
            df['rsi_div_bearish'] = 0.0
            df['rsi_div_strength'] = 0.0
            df['rsi_external'] = 50.0
            
            for idx in df.index:
                row = df.loc[idx]
                div_feat = self._get_rsi_div_features_for_row(
                    row.get('symbol', ''),
                    row.get('date')
                )
                if div_feat:
                    df.loc[idx, 'rsi_div_bullish'] = div_feat.get('rsi_div_bullish', 0)
                    df.loc[idx, 'rsi_div_bearish'] = div_feat.get('rsi_div_bearish', 0)
                    df.loc[idx, 'rsi_div_strength'] = div_feat.get('rsi_div_strength', 0)
                    df.loc[idx, 'rsi_external'] = div_feat.get('rsi_value', 50)
        
        return df
    
    # -------------------------------------------------
    # UPDATED: _select_features with NEW features
    # -------------------------------------------------
    def _select_features(self, df):
        """Select and prepare features for the model"""
        
        # Priority features (original)
        priority_features = [
            'returns', 'log_returns', 'volume_ratio',
            'high_low_ratio', 'close_open_ratio',
            'volatility', 'volatility_ratio',
            'trend_strength', 'momentum_5', 'momentum_10',
            'price_position'
        ]
        
        # ✅ NEW: External features
        external_features = [
            'sector_momentum', 'sector_volume_ratio', 'sector_rsi',  # Sector (3)
            'sr_distance', 'sr_strength', 'sr_type',                # S/R (3)
            'rsi_div_bullish', 'rsi_div_bearish', 'rsi_div_strength', 'rsi_external'  # RSI Div (4)
        ]
        
        # Add available external features
        for feat in external_features:
            if feat in df.columns:
                priority_features.append(feat)
        
        # Add available indicators
        for col in ['rsi', 'macd', 'macd_hist', 'atr']:
            norm_col = f'{col}_norm'
            if norm_col in df.columns:
                priority_features.append(norm_col)
            elif col in df.columns:
                priority_features.append(col)
        
        # Select available features
        available = [f for f in priority_features if f in df.columns]
        
        # Limit to top 20 features (was 15)
        self.feature_columns = available[:20]
        
        if len(self.feature_columns) > 15:
            print(f"   📊 Extended features: {len(self.feature_columns)} features (including Sector/SR/RSI)")
        
        return df[self.feature_columns]
    
    # বাকি সব মেথড অপরিবর্তিত...
    # _prepare_sequences, fit, predict_next_n_days, get_feature_vector
    # _save_model, load_model, _create_model, _create_optimizer, _create_scheduler
    # সব আগের মতই থাকবে