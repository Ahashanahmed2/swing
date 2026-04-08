# nightly_trader.py - LLM + XGBoost + Agentic Loop Enhanced v4.0
# একটাই স্ক্রিপ্ট, সব রাতে হয়
# স্ট্রাকচার অপরিবর্তিত, সব Critical + Hidden + New Issues ফিক্স করা হয়েছে

import pandas as pd
import numpy as np
import os
import json
import re
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallback
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ Transformers not installed. LLM features disabled.")


# =========================================================
# PORTFOLIO STATE MANAGER (NEW - Drawdown Tracking)
# =========================================================

class PortfolioStateManager:
    """✅ FIX 4: Track portfolio equity and drawdown"""
    
    def __init__(self, initial_capital=500000, state_file='./csv/portfolio_state.json'):
        self.initial_capital = initial_capital
        self.current_equity = initial_capital
        self.peak_equity = initial_capital
        self.state_file = state_file
        self.trade_history = []
        self.load_state()
    
    def load_state(self):
        """Load saved portfolio state"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.current_equity = state.get('current_equity', self.initial_capital)
                    self.peak_equity = state.get('peak_equity', self.initial_capital)
                    self.trade_history = state.get('trade_history', [])
                print(f"✅ Portfolio state loaded: Equity=${self.current_equity:,.0f}")
            except:
                pass
    
    def save_state(self):
        """Save portfolio state"""
        try:
            state = {
                'current_equity': self.current_equity,
                'peak_equity': self.peak_equity,
                'trade_history': self.trade_history[-50:],  # Keep last 50 trades
                'last_updated': str(datetime.now())
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except:
            pass
    
    def update_from_trade_log(self):
        """Update equity from trade log"""
        trade_log_file = './csv/trade_log.csv'
        if not os.path.exists(trade_log_file):
            return
        
        try:
            trade_df = pd.read_csv(trade_log_file)
            if len(trade_df) == 0:
                return
            
            # Calculate cumulative PnL
            if 'pnl' in trade_df.columns:
                total_pnl = trade_df['pnl'].sum()
                self.current_equity = self.initial_capital + total_pnl
                self.peak_equity = max(self.peak_equity, self.current_equity)
                
                # Store recent trades
                recent = trade_df.tail(20)
                self.trade_history = recent.to_dict('records')
                self.save_state()
                
                print(f"✅ Portfolio updated: Equity=${self.current_equity:,.0f}, Peak=${self.peak_equity:,.0f}")
                
        except Exception as e:
            print(f"   ⚠️ Could not update from trade log: {e}")
    
    def get_drawdown_ratio(self):
        """Calculate current drawdown ratio"""
        if self.peak_equity <= 0:
            return 0
        return (self.peak_equity - self.current_equity) / self.peak_equity
    
    def get_risk_adjustment_factor(self):
        """✅ FIX 4: Adjust risk based on drawdown"""
        dd = self.get_drawdown_ratio()
        
        if dd > 0.25:
            return 0.3  # Severe drawdown - cut risk to 30%
        elif dd > 0.20:
            return 0.5  # Significant drawdown - 50% risk
        elif dd > 0.15:
            return 0.7  # Moderate drawdown - 70% risk
        elif dd > 0.10:
            return 0.85  # Mild drawdown - 85% risk
        else:
            return 1.0  # Normal risk
    
    def get_current_capital(self):
        """Get current available capital"""
        return self.current_equity
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        if len(self.trade_history) < 5:
            return {}
        
        trades = pd.DataFrame(self.trade_history)
        
        # Win rate
        if 'pnl' in trades.columns:
            wins = (trades['pnl'] > 0).sum()
            total = len(trades)
            win_rate = wins / total if total > 0 else 0
            
            # Last 20 trades win rate
            last_20 = trades.tail(20)
            wins_20 = (last_20['pnl'] > 0).sum()
            win_rate_20 = wins_20 / len(last_20) if len(last_20) > 0 else 0
            
            # Sharpe ratio (simplified)
            returns = trades['pnl'].values / self.initial_capital
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252) if len(returns) > 1 else 0
            
            # Max drawdown from trades
            cumulative = trades['pnl'].cumsum()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / self.initial_capital
            max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            return {
                'win_rate': win_rate,
                'win_rate_last_20': win_rate_20,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'total_trades': total,
                'total_pnl': trades['pnl'].sum()
            }
        
        return {}


# =========================================================
# LLM PREDICTOR CLASS (FIXED - Reliability Enhanced)
# =========================================================

class LLMPredictor:
    """LLM থেকে ট্রেডিং সিগন্যাল জেনারেটর"""
    
    def __init__(self, model_path="./llm_model"):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.performance_history = {'BUY': [], 'SELL': [], 'HOLD': []}
        self.signal_history = {}  # Store signal dates for decay
        self.trade_history_file = './csv/llm_trade_history.json'
        self.load_model()
        self.load_performance_history()
    
    def load_model(self):
        """LLM মডেল লোড করুন"""
        if not LLM_AVAILABLE:
            print("⚠️ LLM not available")
            return
        
        try:
            if os.path.exists(self.model_path):
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("✅ LLM Model loaded")
            else:
                print("⚠️ No local LLM model found")
        except Exception as e:
            print(f"⚠️ LLM load failed: {e}")
    
    def load_performance_history(self):
        """Load saved performance history"""
        if os.path.exists(self.trade_history_file):
            try:
                with open(self.trade_history_file, 'r') as f:
                    saved = json.load(f)
                    self.performance_history = saved.get('history', self.performance_history)
                    self.signal_history = saved.get('signal_history', {})
                print(f"✅ LLM performance history loaded")
            except:
                pass
    
    def save_performance_history(self):
        """Save performance history"""
        try:
            with open(self.trade_history_file, 'w') as f:
                json.dump({
                    'history': self.performance_history, 
                    'signal_history': self.signal_history,
                    'updated': str(datetime.now())
                }, f)
        except:
            pass
    
    def get_llm_signal(self, symbol, market_data, pattern_data=None, regime='UNKNOWN'):
        """LLM থেকে BUY/SELL/HOLD সিগন্যাল নিন - ✅ UPGRADE 8: Enhanced prompt"""
        if self.model is None:
            return {'signal': 'HOLD', 'confidence': 0.5, 'score': 0.5, 'structured': False, 'reliable': False}
        
        try:
            prompt = self._create_enhanced_prompt(symbol, market_data, pattern_data, regime)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else 'cpu'
            if device != 'cpu':
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            signal, confidence, reasoning, is_structured = self._parse_structured_output(generated)
            
            if not is_structured:
                return {
                    'signal': 'HOLD', 
                    'confidence': 0.3,
                    'score': 0.5, 
                    'structured': False, 
                    'reliable': False,
                    'reasoning': 'Unstructured output'
                }
            
            if signal not in ['BUY', 'SELL', 'HOLD']:
                signal = 'HOLD'
                confidence = 0.3
            
            # ✅ UPGRADE 9: Confidence decay for old signals
            confidence = self._apply_confidence_decay(symbol, confidence)
            
            score = self._calculate_score(signal, confidence)
            reliable = self._check_reliability(generated, signal, market_data)
            
            # Store signal date
            self.signal_history[symbol] = str(datetime.now())
            
            return {
                'signal': signal,
                'confidence': confidence,
                'score': score,
                'reasoning': reasoning,
                'structured': is_structured,
                'reliable': reliable
            }
            
        except Exception as e:
            print(f"   ⚠️ LLM prediction failed for {symbol}: {e}")
            return {'signal': 'HOLD', 'confidence': 0.3, 'score': 0.5, 'structured': False, 'reliable': False}
    
    def _create_enhanced_prompt(self, symbol, market_data, pattern_data, regime):
        """✅ UPGRADE 8: Enhanced prompt with market context"""
        pattern_info = ""
        if pattern_data and pattern_data.get('pattern') != 'Unknown':
            pattern_info = f"""
📐 PATTERN ANALYSIS:
Pattern: {pattern_data.get('pattern', 'Unknown')}
Strength: {pattern_data.get('strength', 'Medium')}
Breakout: {pattern_data.get('breakout', 'Pending')}
"""
        
        # ✅ UPGRADE: Microstructure signals
        micro_info = ""
        if market_data.get('volume_spike', False):
            micro_info += "⚠️ VOLUME SPIKE detected - potential breakout\n"
        if abs(market_data.get('change_1d', 0)) > 3:
            micro_info += f"⚠️ GAP detected: {market_data.get('change_1d', 0):.1f}% move\n"
        if market_data.get('liquidity_warning', False):
            micro_info += "⚠️ Low liquidity warning\n"
        
        return f"""You are a professional stock analyst. Analyze this data and provide structured response.

Symbol: {symbol}

📊 MARKET CONTEXT:
Regime: {regime}
Trend Strength: {market_data.get('trend_strength', 0):.3f}
Volatility: {market_data.get('volatility', 0.02):.3f}
{micro_info}
📊 PRICE DATA:
Close: {market_data.get('close', 0):.2f}
Change 1D: {market_data.get('change_1d', 0):.2f}%
Change 5D: {market_data.get('change_5d', 0):.2f}%
Volume: {market_data.get('volume', 0):,.0f}
Volume Ratio: {market_data.get('volume_ratio', 1.0):.2f}x

📈 INDICATORS:
RSI: {market_data.get('rsi', 50):.1f}
MACD: {market_data.get('macd', 0):.4f}
Signal: {market_data.get('macd_signal', 0):.4f}
SMA20: {market_data.get('sma_20', 0):.2f}
SMA50: {market_data.get('sma_50', 0):.2f}
ATR: {market_data.get('atr', 0):.2f}
{pattern_info}
⚠️ IMPORTANT RULES:
- Avoid buying when RSI > 70 (overbought)
- Avoid selling when RSI < 30 (oversold)
- Prefer trend continuation in strong trends
- Avoid low volume breakouts
- Be cautious in SIDEWAYS regime

Respond EXACTLY in this format:
SIGNAL: [BUY/SELL/HOLD]
CONFIDENCE: [0-100]
REASONING: [Brief reason]

Your response:
"""
    
    def _parse_structured_output(self, text):
        """Strict structured parsing with validation"""
        lines = text.strip().split('\n')
        signal = 'HOLD'
        confidence = 0.5
        reasoning = ''
        is_structured = False
        
        signal_found = False
        conf_found = False
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('SIGNAL:'):
                signal_part = line.split(':', 1)[1].strip().upper()
                if 'BUY' in signal_part:
                    signal = 'BUY'
                    signal_found = True
                elif 'SELL' in signal_part:
                    signal = 'SELL'
                    signal_found = True
                elif 'HOLD' in signal_part:
                    signal = 'HOLD'
                    signal_found = True
            elif line.upper().startswith('CONFIDENCE:'):
                conf_part = line.split(':', 1)[1].strip()
                numbers = re.findall(r'\d+', conf_part)
                if numbers:
                    conf_value = int(numbers[0])
                    confidence = max(0.1, min(1.0, conf_value / 100.0))
                    conf_found = True
            elif line.upper().startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
        
        is_structured = signal_found and conf_found
        
        # Anti-hallucination check
        text_lower = text.lower()
        if signal == 'BUY' and any(p in text_lower for p in ['not a buy', 'not buy', 'avoid buy', 'no buy', "don't buy"]):
            signal, confidence, is_structured = 'HOLD', 0.3, False
        if signal == 'SELL' and any(p in text_lower for p in ['not a sell', 'not sell', 'avoid sell', 'no sell', "don't sell"]):
            signal, confidence, is_structured = 'HOLD', 0.3, False
        
        return signal, confidence, reasoning, is_structured
    
    def _apply_confidence_decay(self, symbol, confidence):
        """✅ UPGRADE 9: Confidence decay for old signals"""
        if symbol in self.signal_history:
            try:
                signal_date = datetime.fromisoformat(self.signal_history[symbol])
                days_old = (datetime.now() - signal_date).days
                if days_old > 0:
                    decay = np.exp(-days_old / 5)  # Exponential decay
                    confidence = confidence * decay
            except:
                pass
        return confidence
    
    def _check_reliability(self, generated, signal, market_data):
        """Additional reliability checks"""
        generated_lower = generated.lower()
        
        hallucination_keywords = ['i think', 'maybe', 'possibly', 'could be', 'might be', 'not sure', 'uncertain']
        for kw in hallucination_keywords:
            if kw in generated_lower:
                return False
        
        rsi = market_data.get('rsi', 50)
        if signal == 'BUY' and rsi > 75:
            return False
        if signal == 'SELL' and rsi < 25:
            return False
        
        return True
    
    def _calculate_score(self, signal, confidence):
        """Correct score calculation"""
        if signal == 'BUY':
            return 0.5 + (confidence * 0.5)
        elif signal == 'SELL':
            return 0.5 - (confidence * 0.5)
        return 0.5
    
    def update_performance(self, signal, was_correct):
        """Update performance history with trade result"""
        if signal in self.performance_history:
            self.performance_history[signal].append(1 if was_correct else 0)
            if len(self.performance_history[signal]) > 100:
                self.performance_history[signal] = self.performance_history[signal][-100:]
            self.save_performance_history()
    
    def get_accuracy(self, signal=None):
        """Get actual LLM accuracy from tracked history"""
        if signal:
            history = self.performance_history.get(signal, [])
        else:
            history = []
            for h in self.performance_history.values():
                history.extend(h)
        
        if not history:
            return 0.5
        return sum(history) / len(history)
    
    def get_signal_counts(self):
        """Get count of predictions by signal type"""
        return {
            'BUY': len(self.performance_history.get('BUY', [])),
            'SELL': len(self.performance_history.get('SELL', [])),
            'HOLD': len(self.performance_history.get('HOLD', []))
        }


# =========================================================
# MARKET STRESS DETECTOR (NEW - Crash Protection)
# =========================================================

class MarketStressDetector:
    """✅ FIX 1: Detect market-wide stress/crash conditions"""
    
    @staticmethod
    def detect_market_stress(market_df):
        """Detect if market is under stress (crash protection)"""
        if len(market_df) < 100:
            return False, {}
        
        symbols = market_df['symbol'].unique()
        returns_5d = []
        returns_1d = []
        volume_spikes = []
        below_sma50 = []
        
        for symbol in symbols[:100]:  # Sample top 100 symbols
            sym_data = market_df[market_df['symbol'] == symbol].sort_values('date')
            if len(sym_data) >= 10:
                current = sym_data.iloc[-1]['close']
                
                # 5-day return
                if len(sym_data) >= 6:
                    close_5d_ago = sym_data.iloc[-6]['close']
                    ret_5d = (current - close_5d_ago) / close_5d_ago if close_5d_ago > 0 else 0
                    returns_5d.append(ret_5d)
                
                # 1-day return
                if len(sym_data) >= 2:
                    prev = sym_data.iloc[-2]['close']
                    ret_1d = (current - prev) / prev if prev > 0 else 0
                    returns_1d.append(ret_1d)
                
                # Volume spike check
                if len(sym_data) >= 20:
                    avg_vol = sym_data['volume'].iloc[-20:].mean()
                    current_vol = sym_data.iloc[-1]['volume']
                    if current_vol > avg_vol * 3:
                        volume_spikes.append(symbol)
                
                # Below SMA50 check
                if len(sym_data) >= 50:
                    sma50 = sym_data['close'].rolling(50).mean().iloc[-1]
                    if current < sma50 * 0.95:
                        below_sma50.append(symbol)
        
        # Stress indicators
        crash_ratio_5d = (np.array(returns_5d) < -0.05).mean() if returns_5d else 0
        crash_ratio_1d = (np.array(returns_1d) < -0.03).mean() if returns_1d else 0
        panic_volume = len(volume_spikes) > len(symbols) * 0.3
        broad_weakness = len(below_sma50) > len(symbols) * 0.5
        
        stress_detected = (
            crash_ratio_5d > 0.3 or
            crash_ratio_1d > 0.4 or
            panic_volume or
            broad_weakness
        )
        
        metrics = {
            'crash_ratio_5d': crash_ratio_5d,
            'crash_ratio_1d': crash_ratio_1d,
            'panic_volume': panic_volume,
            'broad_weakness': broad_weakness,
            'stress_level': 'HIGH' if stress_detected else 'NORMAL'
        }
        
        return stress_detected, metrics
    
    @staticmethod
    def detect_liquidity_issues(symbol, market_df):
        """Detect liquidity issues for a symbol"""
        sym_data = market_df[market_df['symbol'] == symbol].sort_values('date')
        if len(sym_data) < 20:
            return False
        
        recent = sym_data.iloc[-20:]
        
        # Low volume check
        avg_volume = recent['volume'].mean()
        if avg_volume < 100000:
            return True
        
        # Volume inconsistency
        volume_std = recent['volume'].std()
        if volume_std / avg_volume > 1.5:
            return True
        
        # Price gaps
        gaps = 0
        for i in range(1, min(10, len(recent))):
            prev_close = recent.iloc[i-1]['close']
            curr_open = recent.iloc[i]['open']
            if prev_close > 0:
                gap_pct = abs(curr_open - prev_close) / prev_close
                if gap_pct > 0.03:
                    gaps += 1
        
        return gaps >= 3


# =========================================================
# CORRELATION MANAGER (NEW - Advanced Correlation Control)
# =========================================================

class CorrelationManager:
    """✅ FIX 3: Advanced correlation control"""
    
    def __init__(self):
        self.correlation_matrix = None
        self.price_cache = {}
        self.max_correlation = 0.70
    
    def build_correlation_matrix(self, market_df, symbols):
        """Build correlation matrix from recent price data"""
        if len(symbols) < 5:
            return None
        
        try:
            # Get price data for symbols
            price_data = {}
            latest_date = market_df['date'].max()
            
            for symbol in symbols[:50]:  # Limit to 50 symbols
                sym_data = market_df[market_df['symbol'] == symbol].sort_values('date')
                if len(sym_data) >= 20:
                    # Get last 20 days of returns
                    sym_data = sym_data[sym_data['date'] <= latest_date].tail(20)
                    returns = sym_data['close'].pct_change().dropna()
                    if len(returns) >= 15:
                        price_data[symbol] = returns.values
            
            if len(price_data) >= 5:
                df = pd.DataFrame(price_data)
                self.correlation_matrix = df.corr()
                return self.correlation_matrix
                
        except Exception as e:
            print(f"   ⚠️ Could not build correlation matrix: {e}")
        
        return None
    
    def get_correlation(self, symbol1, symbol2):
        """Get correlation between two symbols"""
        if self.correlation_matrix is None:
            return 0
        
        if symbol1 in self.correlation_matrix.index and symbol2 in self.correlation_matrix.columns:
            return self.correlation_matrix.loc[symbol1, symbol2]
        return 0
    
    def filter_highly_correlated(self, trades_df):
        """✅ FIX 3: Remove trades with >0.7 correlation"""
        if len(trades_df) <= 1:
            return trades_df
        
        symbols = trades_df['symbol'].tolist()
        to_remove = set()
        
        for i, sym1 in enumerate(symbols):
            if sym1 in to_remove:
                continue
            for sym2 in symbols[i+1:]:
                if sym2 in to_remove:
                    continue
                corr = self.get_correlation(sym1, sym2)
                if abs(corr) > self.max_correlation:
                    # Keep the one with higher quality score
                    score1 = trades_df[trades_df['symbol'] == sym1]['quality_score'].values[0]
                    score2 = trades_df[trades_df['symbol'] == sym2]['quality_score'].values[0]
                    if score1 >= score2:
                        to_remove.add(sym2)
                    else:
                        to_remove.add(sym1)
        
        if to_remove:
            trades_df = trades_df[~trades_df['symbol'].isin(to_remove)]
            print(f"   🔗 Removed {len(to_remove)} highly correlated trades")
        
        return trades_df


# =========================================================
# XGBOOST PREDICTOR CLASS
# =========================================================

class XGBoostPredictor:
    """XGBoost মডেল থেকে প্রেডিকশন"""
    
    def __init__(self, model_dir='./csv/xgboost/'):
        self.model_dir = model_dir
        self.models = {}
        self.feature_info = {}
        self.load_models()
    
    def load_models(self):
        """XGBoost মডেল লোড করুন"""
        if os.path.exists(self.model_dir):
            for file in os.listdir(self.model_dir):
                if file.endswith('.joblib'):
                    symbol = file.replace('.joblib', '')
                    try:
                        model_path = os.path.join(self.model_dir, file)
                        model = joblib.load(model_path)
                        self.models[symbol] = model
                        
                        if hasattr(model, 'n_features_in_'):
                            self.feature_info[symbol] = {
                                'n_features': model.n_features_in_,
                                'feature_names': list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else None
                            }
                    except:
                        pass
        print(f"✅ XGBoost Models: {len(self.models)}")
    
    def get_xgb_score(self, symbol, market_data=None):
        """Enhanced feature extraction for XGBoost"""
        if symbol not in self.models or market_data is None:
            return 0.5, 0.5, False
        
        try:
            model = self.models[symbol]
            expected_features = self.feature_info.get(symbol, {}).get('n_features', 30)
            features = self._build_enhanced_features(market_data, expected_features)
            
            if features.shape[1] != expected_features:
                if features.shape[1] < expected_features:
                    padding = np.zeros((1, expected_features - features.shape[1]))
                    features = np.hstack([features, padding])
                else:
                    features = features[:, :expected_features]
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(features)[0, 1]
            else:
                prob = float(model.predict(features)[0])
            
            confidence = prob if prob > 0.5 else 1 - prob
            return prob, confidence, True
            
        except Exception as e:
            return 0.5, 0.5, False
    
    def _build_enhanced_features(self, market_data, target_size=30):
        """Build features to match expected size"""
        feature_values = []
        
        feature_values.append(market_data.get('close', 0))
        feature_values.append(market_data.get('open', 0))
        feature_values.append(market_data.get('high', 0))
        feature_values.append(market_data.get('low', 0))
        feature_values.append(market_data.get('typical_price', 0))
        feature_values.append(market_data.get('change_1d', 0))
        feature_values.append(market_data.get('change_5d', 0))
        feature_values.append(market_data.get('change_10d', 0))
        feature_values.append(market_data.get('change_20d', 0))
        feature_values.append(market_data.get('volume', 0))
        feature_values.append(market_data.get('volume_change', 0))
        feature_values.append(market_data.get('volume_ratio', 1.0))
        feature_values.append(market_data.get('sma_20', 0))
        feature_values.append(market_data.get('sma_50', 0))
        feature_values.append(market_data.get('ema_20', 0))
        feature_values.append(market_data.get('distance_from_sma20', 0))
        feature_values.append(market_data.get('distance_from_sma50', 0))
        feature_values.append(market_data.get('rsi', 50))
        feature_values.append(1 if market_data.get('rsi', 50) < 30 else 0)
        feature_values.append(1 if market_data.get('rsi', 50) > 70 else 0)
        feature_values.append(market_data.get('macd', 0))
        feature_values.append(market_data.get('macd_signal', 0))
        feature_values.append(market_data.get('macd_hist', 0))
        feature_values.append(market_data.get('stoch_k', 50))
        feature_values.append(market_data.get('stoch_d', 50))
        feature_values.append(market_data.get('atr', 0))
        feature_values.append(market_data.get('atr_pct', 0))
        feature_values.append(market_data.get('volatility', 0.02))
        feature_values.append(market_data.get('momentum', 0))
        feature_values.append(market_data.get('trend_strength', 0))
        feature_values.append(market_data.get('bb_upper', 0))
        feature_values.append(market_data.get('bb_middle', 0))
        feature_values.append(market_data.get('bb_lower', 0))
        feature_values.append(market_data.get('bb_position', 0))
        feature_values.append(1 if market_data.get('above_sma20', False) else 0)
        feature_values.append(1 if market_data.get('above_sma50', False) else 0)
        feature_values.append(1 if market_data.get('macd_bullish', False) else 0)
        feature_values.append(1 if market_data.get('golden_cross', False) else 0)
        
        features = [float(v) if not pd.isna(v) else 0.0 for v in feature_values]
        
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features).reshape(1, -1)


# =========================================================
# AGENTIC LOOP SCORER CLASS
# =========================================================

class AgenticLoopScorer:
    """Agentic Loop থেকে কনসেনসাস স্কোর নিন"""
    
    def __init__(self):
        self.state = None
        self.load_state()
    
    def load_state(self):
        """Agentic Loop স্টেট লোড করুন"""
        state_file = './csv/agentic_loop_state.json'
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    self.state = json.load(f)
                print(f"✅ Agentic Loop state loaded")
            except:
                self.state = None
    
    def calculate_consensus_boost(self, symbol, base_score):
        """Proper boost calculation - additive"""
        if not self.state or 'agents' not in self.state:
            return 0.0
        
        agents = self.state['agents']
        accuracies = []
        
        for info in agents.values():
            if isinstance(info, dict):
                accuracies.append(info.get('accuracy', 0.5))
        
        if not accuracies:
            return 0.0
        
        avg_accuracy = np.mean(accuracies)
        
        if avg_accuracy > 0.60:
            boost = min(0.10, (avg_accuracy - 0.60) * 0.25)
        elif avg_accuracy < 0.50:
            boost = max(-0.10, (avg_accuracy - 0.50) * 0.25)
        else:
            boost = 0.0
        
        return boost
    
    def get_dynamic_weights(self, llm_accuracy=None):
        """Dynamic weighting based on actual performance"""
        weights = {'base': 0.25, 'llm': 0.35, 'xgb': 0.25, 'agentic': 0.15}
        
        if llm_accuracy is not None:
            if llm_accuracy > 0.65:
                weights = {'base': 0.18, 'llm': 0.42, 'xgb': 0.25, 'agentic': 0.15}
            elif llm_accuracy > 0.55:
                weights = {'base': 0.22, 'llm': 0.38, 'xgb': 0.25, 'agentic': 0.15}
            elif llm_accuracy < 0.45:
                weights = {'base': 0.30, 'llm': 0.20, 'xgb': 0.35, 'agentic': 0.15}
            elif llm_accuracy < 0.50:
                weights = {'base': 0.28, 'llm': 0.22, 'xgb': 0.35, 'agentic': 0.15}
        
        return weights


# =========================================================
# SECTOR CLASSIFIER
# =========================================================

class SectorClassifier:
    """Sector classification for correlation control"""
    
    SECTOR_MAP = {
        'TECH': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ADBE', 'ORCL', 'IBM', 'CSCO', 'TSM', 'AVGO', 'TXN', 'QCOM'],
        'FINANCE': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK', 'SCHW'],
        'HEALTHCARE': ['JNJ', 'PFE', 'MRK', 'ABBV', 'BMY', 'LLY', 'UNH', 'CVS', 'AMGN', 'GILD'],
        'CONSUMER': ['AMZN', 'WMT', 'TGT', 'COST', 'HD', 'LOW', 'MCD', 'SBUX', 'NKE', 'DIS', 'TSLA'],
        'ENERGY': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY'],
        'INDUSTRIAL': ['BA', 'CAT', 'DE', 'GE', 'HON', 'LMT', 'MMM', 'RTX', 'UNP', 'UPS'],
        'TELECOM': ['T', 'VZ', 'TMUS', 'CMCSA', 'CHTR', 'DISH'],
        'REAL_ESTATE': ['PLD', 'AMT', 'CCI', 'EQIX', 'SPG', 'WELL', 'AVB', 'EQR'],
        'MATERIALS': ['LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW', 'DD'],
        'UTILITIES': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG']
    }
    
    def __init__(self):
        self.symbol_to_sector = {}
        self._build_mapping()
    
    def _build_mapping(self):
        for sector, symbols in self.SECTOR_MAP.items():
            for symbol in symbols:
                self.symbol_to_sector[symbol.upper()] = sector
    
    def get_sector(self, symbol):
        return self.symbol_to_sector.get(symbol.upper(), 'UNKNOWN')
    
    def get_sector_exposure(self, trades_df):
        if len(trades_df) == 0:
            return {}
        
        sector_investment = {}
        for _, row in trades_df.iterrows():
            sector = self.get_sector(row['symbol'])
            investment = row.get('investment', row.get('entry_price', 0) * row.get('shares', 0))
            sector_investment[sector] = sector_investment.get(sector, 0) + investment
        
        return sector_investment


# =========================================================
# PORTFOLIO RISK MANAGER CLASS (ENHANCED)
# =========================================================

class PortfolioRiskManager:
    """Enhanced portfolio risk control with drawdown adjustment"""
    
    def __init__(self, capital=500000, risk_per_trade=0.02, portfolio_state=None):
        self.base_capital = capital
        self.portfolio_state = portfolio_state
        self.risk_per_trade = risk_per_trade
        self.commission = 0.001
        self.slippage = 0.0005
        self.max_position_pct = 0.25
        self.max_portfolio_exposure = 0.80
        self.max_sector_exposure = 0.40
        self.max_trades = 8
        self.min_trades = 1
        self.sector_classifier = SectorClassifier()
    
    def get_current_capital(self):
        """Get risk-adjusted capital"""
        if self.portfolio_state:
            return self.portfolio_state.get_current_capital()
        return self.base_capital
    
    def get_risk_adjustment(self):
        """✅ FIX 4: Get drawdown-based risk adjustment"""
        if self.portfolio_state:
            return self.portfolio_state.get_risk_adjustment_factor()
        return 1.0
    
    def calculate_position_size(self, entry_price, stop_loss, confidence, volatility=0.02, rank=None, trend_strength=0):
        """Enhanced position sizing with drawdown adjustment"""
        capital = self.get_current_capital()
        risk_adj = self.get_risk_adjustment()
        
        risk_amount = capital * self.risk_per_trade * risk_adj
        confidence_multiplier = 0.5 + (confidence * 0.5)
        adjusted_risk = risk_amount * confidence_multiplier
        
        # ✅ FIX 5: Apply slippage to entry and stop
        entry_with_slippage = entry_price * (1 + self.slippage)
        stop_with_slippage = stop_loss * (1 - self.slippage)
        
        risk_per_share = abs(entry_with_slippage - stop_with_slippage)
        if risk_per_share <= 0:
            return 0, 0, entry_with_slippage, stop_with_slippage
        
        shares = int(adjusted_risk / risk_per_share)
        max_shares = int((capital * self.max_position_pct) / entry_with_slippage)
        shares = min(shares, max_shares)
        
        if rank is not None and rank <= 3:
            shares = int(shares * 1.2)
        elif rank is not None and rank <= 5:
            shares = int(shares * 1.0)
        else:
            shares = int(shares * 0.8)
        
        if volatility > 0.03:
            shares = int(shares * 0.7)
        elif volatility < 0.015:
            shares = int(shares * 1.1)
        
        if trend_strength > 0.05:
            shares = int(shares * 1.1)
        
        investment = shares * entry_with_slippage
        return shares, investment, entry_with_slippage, stop_with_slippage
    
    def get_dynamic_sl_multiplier(self, trend_strength, volatility):
        """Dynamic stop loss multiplier"""
        base_multiplier = 1.5
        
        if trend_strength > 0.05:
            base_multiplier = 2.0
        elif trend_strength < 0.02:
            base_multiplier = 1.2
        
        if volatility > 0.03:
            base_multiplier *= 1.2
        
        return base_multiplier
    
    def get_dynamic_tp_multiplier(self, trend_strength, regime):
        """✅ UPGRADE 7: Adaptive take profit"""
        if trend_strength > 0.05:
            if regime == 'BULL':
                return 2.5
            else:
                return 2.0
        elif trend_strength > 0.02:
            return 1.8
        else:
            return 1.5
    
    def calculate_trade_quality_score(self, combined_score, confidence, risk_reward, reliable, volume_spike=False, gap=False):
        """Quality score for ranking with microstructure boost"""
        if not reliable:
            combined_score *= 0.7
        
        quality = combined_score * confidence * (1 + risk_reward) if risk_reward > 0 else combined_score * confidence
        
        # ✅ UPGRADE: Microstructure boost
        if volume_spike and combined_score > 0.6:
            quality *= 1.05  # 5% boost for volume confirmation
        if gap and abs(gap) > 2:
            quality *= 0.95  # 5% penalty for gap risk
        
        return quality
    
    def validate_portfolio(self, trades_df, total_capital=None, correlation_manager=None):
        """Portfolio validation with sector control and correlation filter"""
        if len(trades_df) == 0:
            return True, trades_df, "No trades"
        
        if total_capital is None:
            total_capital = self.get_current_capital()
        
        issues = []
        
        # ✅ FIX 3: Correlation filter
        if correlation_manager is not None:
            trades_df = correlation_manager.filter_highly_correlated(trades_df)
        
        total_investment = trades_df['investment'].sum() if 'investment' in trades_df.columns else 0
        exposure_pct = total_investment / total_capital
        
        if exposure_pct > self.max_portfolio_exposure:
            reduction_factor = self.max_portfolio_exposure / exposure_pct
            trades_df['shares'] = (trades_df['shares'] * reduction_factor).astype(int)
            trades_df['investment'] = trades_df['shares'] * trades_df['entry_price']
            total_investment = trades_df['investment'].sum()
            issues.append(f"Exposure reduced to {self.max_portfolio_exposure:.0%}")
        
        sector_exposure = self.sector_classifier.get_sector_exposure(trades_df)
        sectors_to_reduce = []
        
        for sector, investment in sector_exposure.items():
            sector_pct = investment / total_capital
            if sector_pct > self.max_sector_exposure:
                sectors_to_reduce.append(sector)
                issues.append(f"⚠️ {sector} sector exposure {sector_pct:.1%} exceeds limit")
        
        if sectors_to_reduce:
            for sector in sectors_to_reduce:
                sector_trades = trades_df[trades_df['symbol'].apply(lambda x: self.sector_classifier.get_sector(x) == sector)]
                if len(sector_trades) > 1:
                    keep_indices = sector_trades.nlargest(2, 'quality_score').index
                    drop_indices = sector_trades.index.difference(keep_indices)
                    trades_df = trades_df.drop(drop_indices)
            issues.append(f"Reduced overexposed sectors")
        
        if len(trades_df) > self.max_trades:
            trades_df = trades_df.head(self.max_trades)
            issues.append(f"Limited to {self.max_trades} trades")
        
        trades_df = trades_df[trades_df['investment'] >= 5000]
        
        if len(trades_df) == 0:
            return False, trades_df, "No valid trades after filters"
        
        return True, trades_df, "; ".join(issues) if issues else "Portfolio validated"
    
    def get_portfolio_summary(self, trades_df, total_capital=None):
        """Generate portfolio summary with sector breakdown"""
        if len(trades_df) == 0:
            return {}
        
        if total_capital is None:
            total_capital = self.get_current_capital()
        
        total_investment = trades_df['investment'].sum()
        sector_exposure = self.sector_classifier.get_sector_exposure(trades_df)
        
        return {
            'total_trades': len(trades_df),
            'total_investment': total_investment,
            'exposure_pct': total_investment / total_capital,
            'avg_confidence': trades_df['confidence'].mean(),
            'avg_rrr': trades_df['risk_reward'].mean() if 'risk_reward' in trades_df.columns else 0,
            'max_single_position': trades_df['investment'].max(),
            'capital_remaining': total_capital - total_investment,
            'sector_exposure': sector_exposure
        }


# =========================================================
# ENHANCED MARKET DATA CALCULATOR
# =========================================================

class MarketDataCalculator:
    """Calculate comprehensive market features"""
    
    @staticmethod
    def calculate_enhanced_features(symbol, market_df, current_row, stress_detector=None):
        """Calculate comprehensive market features with microstructure"""
        sym_data = market_df[market_df['symbol'] == symbol].sort_values('date')
        
        if len(sym_data) < 20:
            return {}
        
        current_price = current_row['close']
        
        features = {
            'close': current_price,
            'open': current_row.get('open', current_price),
            'high': current_row.get('high', current_price),
            'low': current_row.get('low', current_price),
            'volume': current_row.get('volume', 0),
            'typical_price': (current_row.get('high', current_price) + current_row.get('low', current_price) + current_price) / 3
        }
        
        close_prices = sym_data['close']
        
        # Returns
        if len(close_prices) >= 2:
            features['change_1d'] = ((current_price - close_prices.iloc[-2]) / close_prices.iloc[-2] * 100) if close_prices.iloc[-2] > 0 else 0
        else:
            features['change_1d'] = 0
            
        if len(close_prices) >= 6:
            features['change_5d'] = ((current_price - close_prices.iloc[-6]) / close_prices.iloc[-6] * 100) if close_prices.iloc[-6] > 0 else 0
        else:
            features['change_5d'] = 0
            
        if len(close_prices) >= 11:
            features['change_10d'] = ((current_price - close_prices.iloc[-11]) / close_prices.iloc[-11] * 100) if close_prices.iloc[-11] > 0 else 0
        else:
            features['change_10d'] = 0
            
        if len(close_prices) >= 21:
            features['change_20d'] = ((current_price - close_prices.iloc[-21]) / close_prices.iloc[-21] * 100) if close_prices.iloc[-21] > 0 else 0
        else:
            features['change_20d'] = 0
        
        # Volume features
        if len(sym_data) >= 20:
            avg_volume = sym_data['volume'].iloc[-20:].mean()
            features['volume_change'] = ((features['volume'] - avg_volume) / avg_volume * 100) if avg_volume > 0 else 0
            features['volume_ratio'] = features['volume'] / avg_volume if avg_volume > 0 else 1.0
            features['volume_spike'] = features['volume_ratio'] > 2.5
        else:
            features['volume_change'] = 0
            features['volume_ratio'] = 1.0
            features['volume_spike'] = False
        
        # Gap detection
        if len(sym_data) >= 2:
            prev_close = sym_data.iloc[-2]['close']
            curr_open = features['open']
            if prev_close > 0:
                features['gap_pct'] = ((curr_open - prev_close) / prev_close) * 100
            else:
                features['gap_pct'] = 0
        else:
            features['gap_pct'] = 0
        
        # Liquidity check
        if stress_detector:
            features['liquidity_warning'] = stress_detector.detect_liquidity_issues(symbol, market_df)
        else:
            features['liquidity_warning'] = False
        
        # Moving Averages
        features['sma_20'] = close_prices.rolling(20).mean().iloc[-1] if len(close_prices) >= 20 else current_price
        features['sma_50'] = close_prices.rolling(50).mean().iloc[-1] if len(close_prices) >= 50 else current_price
        features['ema_20'] = close_prices.ewm(span=20, adjust=False).mean().iloc[-1] if len(close_prices) >= 20 else current_price
        
        if features['sma_20'] > 0:
            features['distance_from_sma20'] = ((current_price - features['sma_20']) / features['sma_20']) * 100
        else:
            features['distance_from_sma20'] = 0
            
        if features['sma_50'] > 0:
            features['distance_from_sma50'] = ((current_price - features['sma_50']) / features['sma_50']) * 100
        else:
            features['distance_from_sma50'] = 0
        
        features['above_sma20'] = current_price > features['sma_20']
        features['above_sma50'] = current_price > features['sma_50']
        features['golden_cross'] = features['sma_20'] > features['sma_50'] if features['sma_20'] > 0 else False
        
        # RSI
        features['rsi'] = MarketDataCalculator._calculate_rsi(close_prices)
        
        # MACD
        macd, signal, hist = MarketDataCalculator._calculate_macd(close_prices)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        features['macd_bullish'] = macd > signal
        
        # Stochastic
        if 'high' in sym_data.columns and 'low' in sym_data.columns:
            stoch_k, stoch_d = MarketDataCalculator._calculate_stochastic(sym_data)
            features['stoch_k'] = stoch_k
            features['stoch_d'] = stoch_d
        else:
            features['stoch_k'] = 50
            features['stoch_d'] = 50
        
        # ATR
        if 'high' in sym_data.columns and 'low' in sym_data.columns:
            features['atr'] = MarketDataCalculator._calculate_atr(sym_data)
            features['atr_pct'] = (features['atr'] / current_price * 100) if current_price > 0 else 0
        else:
            features['atr'] = current_price * 0.02
            features['atr_pct'] = 2.0
        
        # Volatility
        if len(close_prices) >= 20:
            returns = close_prices.pct_change().dropna()
            features['volatility'] = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else 0.02
        else:
            features['volatility'] = 0.02
        
        # Momentum
        if len(close_prices) >= 10:
            features['momentum'] = (close_prices.iloc[-1] - close_prices.iloc[-10]) / close_prices.iloc[-10]
        else:
            features['momentum'] = 0
        
        # Trend strength
        if features['sma_20'] > 0 and features['sma_50'] > 0:
            features['trend_strength'] = abs(features['sma_20'] - features['sma_50']) / features['sma_50']
        else:
            features['trend_strength'] = 0
        
        # Bollinger Bands
        if len(close_prices) >= 20:
            sma20 = close_prices.rolling(20).mean()
            std20 = close_prices.rolling(20).std()
            features['bb_upper'] = sma20.iloc[-1] + (2 * std20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else current_price * 1.05
            features['bb_middle'] = sma20.iloc[-1] if not pd.isna(sma20.iloc[-1]) else current_price
            features['bb_lower'] = sma20.iloc[-1] - (2 * std20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else current_price * 0.95
            
            if features['bb_upper'] > features['bb_lower']:
                features['bb_position'] = (current_price - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            else:
                features['bb_position'] = 0.5
        else:
            features['bb_upper'] = current_price * 1.05
            features['bb_middle'] = current_price
            features['bb_lower'] = current_price * 0.95
            features['bb_position'] = 0.5
        
        return features
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        if len(prices) < period:
            return 50
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        loss = loss.replace(0, 1e-10)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).iloc[-1]
    
    @staticmethod
    def _calculate_macd(prices, fast=12, slow=26, signal=9):
        if len(prices) < slow:
            return 0, 0, 0
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        return macd.iloc[-1], signal_line.iloc[-1], hist.iloc[-1]
    
    @staticmethod
    def _calculate_stochastic(sym_data, k_period=14, d_period=3):
        if len(sym_data) < k_period:
            return 50, 50
        high = sym_data['high']
        low = sym_data['low']
        close = sym_data['close']
        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()
        k = 100 * ((close - low_min) / (high_max - low_min + 1e-10))
        d = k.rolling(window=d_period).mean()
        return k.fillna(50).iloc[-1], d.fillna(50).iloc[-1]
    
    @staticmethod
    def _calculate_atr(sym_data, period=14):
        if len(sym_data) < period:
            return 0
        high = sym_data['high']
        low = sym_data['low']
        close = sym_data['close']
        tr = np.maximum(high - low, 
                       np.maximum(np.abs(high - close.shift()), 
                                np.abs(low - close.shift())))
        return tr.rolling(window=period).mean().iloc[-1]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def detect_market_regime(market_df):
    """Detect market regime"""
    if len(market_df) < 20:
        return 'SIDEWAYS'
    
    symbols = market_df['symbol'].unique()[:50]
    returns = []
    
    for symbol in symbols:
        sym_data = market_df[market_df['symbol'] == symbol].sort_values('date')
        if len(sym_data) >= 20:
            ret = (sym_data.iloc[-1]['close'] - sym_data.iloc[-20]['close']) / sym_data.iloc[-20]['close']
            returns.append(ret)
    
    if not returns:
        return 'SIDEWAYS'
    
    avg_return = np.mean(returns)
    
    if avg_return > 0.02:
        return 'BULL'
    elif avg_return < -0.03:
        return 'BEAR'
    return 'SIDEWAYS'


def save_decision_log(decisions, regime, llm_signals, portfolio_summary=None, performance_metrics=None, stress_metrics=None):
    """✅ UPGRADE 10: Enhanced logging with performance metrics"""
    log_file = './csv/trading_decisions_enhanced.csv'
    
    log_entry = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'market_regime': regime,
        'num_trades': len(decisions),
        'llm_buy': llm_signals.get('BUY', 0),
        'llm_sell': llm_signals.get('SELL', 0),
        'llm_hold': llm_signals.get('HOLD', 0),
        'symbols': ', '.join(decisions['symbol'].head(5).tolist()) if len(decisions) > 0 else 'None'
    }
    
    if portfolio_summary:
        log_entry.update({
            'total_investment': portfolio_summary.get('total_investment', 0),
            'exposure_pct': portfolio_summary.get('exposure_pct', 0),
            'avg_confidence': portfolio_summary.get('avg_confidence', 0)
        })
    
    if performance_metrics:
        log_entry.update({
            'win_rate': performance_metrics.get('win_rate', 0),
            'win_rate_last_20': performance_metrics.get('win_rate_last_20', 0),
            'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
            'max_drawdown': performance_metrics.get('max_drawdown', 0),
            'total_pnl': performance_metrics.get('total_pnl', 0)
        })
    
    if stress_metrics:
        log_entry['stress_level'] = stress_metrics.get('stress_level', 'NORMAL')
        log_entry['crash_ratio'] = stress_metrics.get('crash_ratio_5d', 0)
    
    new_entry = pd.DataFrame([log_entry])
    
    if os.path.exists(log_file):
        existing = pd.read_csv(log_file)
        updated = pd.concat([existing, new_entry], ignore_index=True)
    else:
        updated = new_entry
    
    updated.to_csv(log_file, index=False)


def update_llm_performance_from_trade_log(llm_predictor):
    """Update LLM performance from historical trade log"""
    trade_log_file = './csv/trade_log.csv'
    
    if not os.path.exists(trade_log_file):
        return
    
    try:
        trade_df = pd.read_csv(trade_log_file)
        
        if len(trade_df) == 0:
            return
        
        trade_df['date'] = pd.to_datetime(trade_df['date'], errors='coerce')
        recent_trades = trade_df[trade_df['date'] > datetime.now() - pd.Timedelta(days=30)]
        
        for _, trade in recent_trades.iterrows():
            pnl = trade.get('pnl', 0)
            was_win = pnl > 0
            
            if 'llm_signal' in trade.index:
                signal = trade['llm_signal']
                llm_predictor.update_performance(signal, was_win)
        
        print(f"✅ LLM performance updated from {len(recent_trades)} historical trades")
        
    except Exception as e:
        print(f"   ⚠️ Could not update LLM from trade log: {e}")


# =========================================================
# MAIN NIGHTLY TRADING SYSTEM
# =========================================================

def nightly_trading_system():
    """
    LLM + XGBoost + Agentic Loop - v4.0
    All Critical + Hidden + New Issues Fixed
    """
    
    print("="*70)
    print("🌙 NIGHTLY TRADING SYSTEM (LLM + Agentic Loop Enhanced v4.0)")
    print("="*70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Generating decisions for TOMORROW's trading")
    print("="*70)

    # =========================================================
    # STEP 1: LOAD DATA & INITIALIZE
    # =========================================================
    base_path = './csv/'
    
    market_df = pd.read_csv(os.path.join(base_path, 'mongodb.csv'))
    pred_df = pd.read_csv(os.path.join(base_path, 'prediction_log.csv'))
    meta_df = pd.read_csv(os.path.join(base_path, 'model_metadata.csv'))
    
    market_df['date'] = pd.to_datetime(market_df['date'])
    pred_df['date'] = pd.to_datetime(pred_df['date'], format='mixed', errors='coerce')
    
    print("\n📦 Loading Models & Managers...")
    
    # Initialize portfolio state
    portfolio_state = PortfolioStateManager(initial_capital=500000)
    portfolio_state.update_from_trade_log()
    
    # Initialize components
    llm_predictor = LLMPredictor(model_path="./llm_model")
    agentic_scorer = AgenticLoopScorer()
    xgb_predictor = XGBoostPredictor()
    stress_detector = MarketStressDetector()
    correlation_manager = CorrelationManager()
    portfolio_manager = PortfolioRiskManager(capital=500000, risk_per_trade=0.02, portfolio_state=portfolio_state)
    data_calculator = MarketDataCalculator()
    
    update_llm_performance_from_trade_log(llm_predictor)
    
    # Get performance metrics
    performance_metrics = portfolio_state.get_performance_metrics()
    
    # Latest data
    latest_date = market_df['date'].max()
    today_data = market_df[market_df['date'] == latest_date]
    
    print(f"\n📊 Market Data: {latest_date.strftime('%Y-%m-%d')}")
    print(f"   Symbols: {len(today_data['symbol'].unique())}")
    
    if performance_metrics:
        print(f"\n📈 Portfolio Performance:")
        print(f"   Current Equity: ${portfolio_state.current_equity:,.0f}")
        print(f"   Win Rate: {performance_metrics.get('win_rate', 0):.1%}")
        print(f"   Win Rate (L20): {performance_metrics.get('win_rate_last_20', 0):.1%}")
        print(f"   Sharpe: {performance_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Max DD: {performance_metrics.get('max_drawdown', 0):.1%}")
        print(f"   Risk Adj: {portfolio_state.get_risk_adjustment_factor():.0%}")

    # =========================================================
    # STEP 2: MARKET STRESS CHECK (CRASH PROTECTION)
    # =========================================================
    stress_detected, stress_metrics = stress_detector.detect_market_stress(market_df)
    
    print(f"\n🚨 Market Stress Check:")
    print(f"   Stress Level: {stress_metrics.get('stress_level', 'NORMAL')}")
    print(f"   Crash Ratio (5D): {stress_metrics.get('crash_ratio_5d', 0):.1%}")
    print(f"   Panic Volume: {stress_metrics.get('panic_volume', False)}")
    print(f"   Broad Weakness: {stress_metrics.get('broad_weakness', False)}")
    
    if stress_detected:
        print(f"\n🚨🚨🚨 MARKET STRESS DETECTED - NO TRADES TOMORROW 🚨🚨🚨")
        print(f"   Reason: High probability of market crash/panic")
        print(f"   Action: Wait for market stabilization")
        
        # Save empty trade file
        empty_df = pd.DataFrame(columns=['symbol', 'date', 'buy', 'SL', 'tp', 'confidence', 'RRR', 'shares'])
        empty_df.to_csv('./csv/trade_stock.csv', index=False)
        
        save_decision_log(pd.DataFrame(), 'STRESS', {'BUY': 0, 'SELL': 0, 'HOLD': 0}, 
                         performance_metrics=performance_metrics, stress_metrics=stress_metrics)
        
        print("\n" + "="*70)
        print("✅ NIGHTLY DECISION COMPLETE (NO TRADES - MARKET STRESS)")
        print("="*70)
        return True

    # =========================================================
    # STEP 3: MARKET REGIME
    # =========================================================
    regime = detect_market_regime(market_df)
    print(f"\n📈 Market Regime: {regime}")
    
    if regime == 'BEAR':
        max_trades = 3
        min_score = 0.75
    else:
        max_trades = 8
        min_score = 0.65 if regime == 'BULL' else 0.70

    # =========================================================
    # STEP 4: BUILD CORRELATION MATRIX
    # =========================================================
    all_symbols_list = meta_df[meta_df['auc'] >= 0.55]['symbol'].unique().tolist()
    correlation_manager.build_correlation_matrix(market_df, all_symbols_list)

    # =========================================================
    # STEP 5: GOOD SYMBOLS
    # =========================================================
    good_symbols = meta_df[meta_df['auc'] >= 0.55]['symbol'].unique()
    xgb_symbols = list(xgb_predictor.models.keys())
    all_symbols = list(set(good_symbols) | set(xgb_symbols))
    print(f"\n✅ Qualified Symbols: {len(all_symbols)}")
    
    llm_stats = llm_predictor.get_signal_counts()
    llm_accuracy = llm_predictor.get_accuracy()
    print(f"🤖 LLM Accuracy: {llm_accuracy:.1%} (from {sum(llm_stats.values())} tracked predictions)")
    
    # Show drawdown-adjusted risk
    risk_adj = portfolio_state.get_risk_adjustment_factor()
    if risk_adj < 1.0:
        print(f"⚠️ Risk Reduced to {risk_adj:.0%} due to {portfolio_state.get_drawdown_ratio():.1%} drawdown")

    # =========================================================
    # STEP 6: GENERATE SIGNALS
    # =========================================================
    print("\n🎯 Generating signals...")
    
    decisions = []
    llm_signals = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    unstructured_count = 0
    unreliable_count = 0
    
    dynamic_weights = agentic_scorer.get_dynamic_weights(llm_accuracy)
    current_capital = portfolio_state.get_current_capital()
    print(f"   Dynamic Weights: Base={dynamic_weights['base']:.2f}, LLM={dynamic_weights['llm']:.2f}, XGB={dynamic_weights['xgb']:.2f}")
    print(f"   Available Capital: ${current_capital:,.0f}")
    
    for symbol in all_symbols[:150]:
        
        symbol_today = today_data[today_data['symbol'] == symbol]
        if len(symbol_today) == 0:
            continue
        
        row = symbol_today.iloc[0]
        current_price = row['close']
        
        # Enhanced market data with microstructure
        market_data = data_calculator.calculate_enhanced_features(symbol, market_df, row, stress_detector)
        
        pattern_data = {
            'pattern': row.get('pattern', 'Unknown') if 'pattern' in row.index else 'Unknown',
            'strength': 'Medium',
            'breakout': 'Pending'
        }
        
        # Skip if liquidity warning
        if market_data.get('liquidity_warning', False):
            continue
        
        # Base score
        symbol_pred = pred_df[pred_df['symbol'] == symbol]
        if len(symbol_pred) > 0:
            symbol_pred = symbol_pred[symbol_pred['date'] <= latest_date]
            if len(symbol_pred) > 0:
                latest = symbol_pred.iloc[-1]
                base_score = latest.get('prob_up', 0.5)
                base_conf = latest.get('confidence_score', 50) / 100
            else:
                base_score, base_conf = 0.5, 0.5
        else:
            base_score, base_conf = 0.5, 0.5
        
        # LLM signal
        llm_result = llm_predictor.get_llm_signal(symbol, market_data, pattern_data, regime)
        llm_signal = llm_result['signal']
        llm_score = llm_result['score']
        llm_conf = llm_result['confidence']
        is_structured = llm_result.get('structured', False)
        is_reliable = llm_result.get('reliable', False)
        
        llm_signals[llm_signal] += 1
        if not is_structured:
            unstructured_count += 1
        if not is_reliable:
            unreliable_count += 1
        
        if not is_structured or not is_reliable:
            llm_score = 0.5
            llm_conf = 0.3
        
        # LLM as filter
        if llm_signal == 'SELL' and is_reliable:
            base_score = base_score * 0.7
        
        # XGBoost signal
        xgb_score, xgb_conf, xgb_valid = xgb_predictor.get_xgb_score(symbol, market_data)
        
        # Agentic boost
        agentic_boost = agentic_scorer.calculate_consensus_boost(symbol, base_score)
        
        # Non-linear fusion
        combined_score = (
            base_score * dynamic_weights['base'] +
            (llm_score ** 1.2) * dynamic_weights['llm'] +
            (xgb_score ** 1.1) * dynamic_weights['xgb']
        )
        combined_score += agentic_boost
        
        # Microstructure boost
        if market_data.get('volume_spike', False) and combined_score > 0.6:
            combined_score += 0.02
        if abs(market_data.get('gap_pct', 0)) > 3:
            combined_score -= 0.03
        
        if regime == 'BULL':
            combined_score = combined_score * 1.02
        elif regime == 'BEAR':
            combined_score = combined_score * 0.98
        
        combined_score = min(max(combined_score, 0.0), 1.0)
        
        # Weighted confidence
        combined_conf = (
            base_conf * dynamic_weights['base'] +
            llm_conf * dynamic_weights['llm'] +
            xgb_conf * dynamic_weights['xgb']
        )
        
        # ✅ UPGRADE 6: NO TRADE ZONE
        if 0.45 < combined_score < 0.65 and regime != 'BULL':
            decision = "SKIP"
            stop_loss = take_profit = 0
            risk_reward = 0
            atr = 0
        elif combined_score >= min_score:
            decision = "BUY"
            atr = market_data.get('atr', current_price * 0.02)
            if atr == 0:
                atr = current_price * 0.02
            
            trend_strength = market_data.get('trend_strength', 0)
            volatility = market_data.get('volatility', 0.02)
            
            sl_multiplier = portfolio_manager.get_dynamic_sl_multiplier(trend_strength, volatility)
            tp_multiplier = portfolio_manager.get_dynamic_tp_multiplier(trend_strength, regime)
            
            stop_loss = current_price - (sl_multiplier * atr)
            take_profit = current_price + (tp_multiplier * atr)
            risk_reward = round((take_profit - current_price) / (current_price - stop_loss), 2) if stop_loss < current_price else 0
        elif combined_score <= 0.30:
            decision = "SELL"
            atr = market_data.get('atr', current_price * 0.02)
            if atr == 0:
                atr = current_price * 0.02
            stop_loss = current_price + (1.5 * atr)
            take_profit = current_price - (2.0 * atr)
            risk_reward = round((current_price - take_profit) / (stop_loss - current_price), 2) if stop_loss > current_price else 0
        else:
            decision = "HOLD"
            stop_loss = take_profit = 0
            risk_reward = 0
            atr = 0
        
        decisions.append({
            'symbol': symbol,
            'decision': decision,
            'combined_score': round(combined_score, 3),
            'llm_signal': llm_signal,
            'llm_score': round(llm_score, 3),
            'llm_reliable': is_reliable,
            'base_score': round(base_score, 3),
            'xgb_score': round(xgb_score, 3),
            'agentic_boost': round(agentic_boost, 3),
            'current_price': round(current_price, 2),
            'entry_price': round(current_price, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'risk_reward': risk_reward,
            'confidence': round(combined_conf, 3),
            'volatility': market_data.get('volatility', 0.02),
            'trend_strength': market_data.get('trend_strength', 0),
            'volume_spike': market_data.get('volume_spike', False),
            'gap_pct': market_data.get('gap_pct', 0),
            'atr': atr
        })
    
    print(f"   LLM Stats: {unstructured_count} unstructured, {unreliable_count} unreliable")
    
    # =========================================================
    # STEP 7: CREATE AND VALIDATE TRADE FILE
    # =========================================================
    decisions_df = pd.DataFrame(decisions)
    buy_decisions = decisions_df[decisions_df['decision'] == 'BUY'].copy()
    
    trades_list = []
    for _, row in buy_decisions.iterrows():
        quality = portfolio_manager.calculate_trade_quality_score(
            row['combined_score'], 
            row['confidence'], 
            row['risk_reward'],
            row['llm_reliable'],
            row['volume_spike'],
            row['gap_pct']
        )
        
        trades_list.append({
            **row.to_dict(),
            'quality_score': quality
        })
    
    trades_df = pd.DataFrame(trades_list)
    if len(trades_df) > 0:
        trades_df = trades_df.sort_values('quality_score', ascending=False)
    
    final_trades_list = []
    for rank, (_, row) in enumerate(trades_df.iterrows(), 1):
        shares, investment, entry_slip, stop_slip = portfolio_manager.calculate_position_size(
            row['entry_price'], row['stop_loss'], row['confidence'], 
            row['volatility'], rank, row['trend_strength']
        )
        
        final_trades_list.append({
            'symbol': row['symbol'],
            'entry_price': entry_slip,  # With slippage
            'stop_loss': stop_slip,      # With slippage
            'take_profit': row['take_profit'],
            'combined_score': row['combined_score'],
            'quality_score': round(row['quality_score'], 3),
            'confidence': row['confidence'],
            'risk_reward': row['risk_reward'],
            'llm_signal': row['llm_signal'],
            'llm_reliable': row['llm_reliable'],
            'shares': shares,
            'investment': investment,
            'rank': rank
        })
    
    final_trades_df = pd.DataFrame(final_trades_list)
    
    # Portfolio validation with correlation filter
    is_valid, validated_trades, validation_message = portfolio_manager.validate_portfolio(
        final_trades_df, current_capital, correlation_manager
    )
    
    if is_valid and len(validated_trades) > 0:
        final_trades = validated_trades.head(max_trades)
        
        trade_output = final_trades[['symbol', 'entry_price', 'stop_loss', 'take_profit', 'quality_score', 'shares', 'risk_reward']].copy()
        trade_output.columns = ['symbol', 'buy', 'SL', 'tp', 'confidence', 'shares', 'RRR']
        trade_output['date'] = latest_date.strftime('%Y-%m-%d')
        trade_output.to_csv('./csv/trade_stock.csv', index=False)
        
        print(f"\n✅ {len(final_trades)} BUY signals for tomorrow")
        print(f"   {validation_message}")
    else:
        print(f"\n⚠️ No BUY signals for tomorrow")
        print(f"   Reason: {validation_message}")
        empty_df = pd.DataFrame(columns=['symbol', 'date', 'buy', 'SL', 'tp', 'confidence', 'RRR', 'shares'])
        empty_df.to_csv('./csv/trade_stock.csv', index=False)
        final_trades = pd.DataFrame()
    
    decisions_df.to_csv('./csv/nightly_decisions_enhanced.csv', index=False)
    
    # =========================================================
    # STEP 8: PORTFOLIO SUMMARY
    # =========================================================
    portfolio_summary = portfolio_manager.get_portfolio_summary(final_trades, current_capital) if len(final_trades) > 0 else {}
    
    # =========================================================
    # STEP 9: PRINT SUMMARY
    # =========================================================
    print("\n" + "="*70)
    print("📊 TOMORROW'S TRADING DECISIONS")
    print("="*70)
    print(f"   Market Regime: {regime}")
    print(f"   Stress Level: {stress_metrics.get('stress_level', 'NORMAL')}")
    print(f"   Total Analyzed: {len(decisions_df)}")
    print(f"   BUY: {len(decisions_df[decisions_df['decision'] == 'BUY'])}")
    print(f"   SELL: {len(decisions_df[decisions_df['decision'] == 'SELL'])}")
    print(f"   SKIP: {len(decisions_df[decisions_df['decision'] == 'SKIP'])}")
    print(f"   HOLD: {len(decisions_df[decisions_df['decision'] == 'HOLD'])}")
    print(f"\n🤖 LLM Signals: BUY={llm_signals['BUY']}, SELL={llm_signals['SELL']}, HOLD={llm_signals['HOLD']}")
    print(f"   LLM Accuracy: {llm_accuracy:.1%} | Unreliable: {unreliable_count}")
    
    if portfolio_summary:
        print(f"\n💰 PORTFOLIO SUMMARY:")
        print(f"   Available Capital: ${current_capital:,.0f}")
        print(f"   Total Investment: ${portfolio_summary['total_investment']:,.0f}")
        print(f"   Portfolio Exposure: {portfolio_summary['exposure_pct']:.1%}")
        print(f"   Capital Remaining: ${portfolio_summary['capital_remaining']:,.0f}")
        print(f"   Avg Quality Score: {final_trades['quality_score'].mean():.3f}" if len(final_trades) > 0 else "")
        
        if 'sector_exposure' in portfolio_summary and portfolio_summary['sector_exposure']:
            print(f"\n   📊 Sector Exposure:")
            for sector, amount in portfolio_summary['sector_exposure'].items():
                pct = amount / current_capital * 100
                print(f"      {sector}: ${amount:,.0f} ({pct:.1f}%)")
    
    if len(final_trades) > 0:
        print("\n" + "🔥"*35)
        print("🔥 YOUR TRADING PLAN FOR TOMORROW 🔥")
        print("🔥"*35)
        
        for _, row in final_trades.iterrows():
            reliable_mark = "✅" if row['llm_reliable'] else "⚠️"
            print(f"\n{row['rank']}. {row['symbol']} {reliable_mark}")
            print(f"   🤖 LLM: {row['llm_signal']} | Quality: {row['quality_score']:.3f}")
            print(f"   📌 BUY at ${row['entry_price']:.2f} (incl. slippage)")
            print(f"   🛑 SL: ${row['stop_loss']:.2f} | 🎯 TP: ${row['take_profit']:.2f}")
            print(f"   📊 R:R = 1:{row['risk_reward']:.2f}")
            print(f"   💰 Shares: {row['shares']} | Investment: ${row['investment']:,.0f}")
    
    else:
        print("\n" + "💤"*35)
        print("💤 NO TRADES TOMORROW")
        print("💤"*35)
    
    save_decision_log(final_trades if len(final_trades) > 0 else pd.DataFrame(), 
                     regime, llm_signals, portfolio_summary, performance_metrics, stress_metrics)
    
    print("\n" + "="*70)
    print("✅ NIGHTLY DECISION COMPLETE")
    print("📁 Check trade_stock.csv for broker orders")
    print("="*70)
    
    return True


def morning_check():
    """সকালে চেক করুন"""
    print("\n" + "="*70)
    print("🌅 MORNING TRADING CHECK (LLM Enhanced v4.0)")
    print("="*70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d')}")
    print("="*70)
    
    try:
        trade_df = pd.read_csv('./csv/trade_stock.csv')
        enhanced_df = pd.read_csv('./csv/nightly_decisions_enhanced.csv')
        
        if len(trade_df) == 0:
            print("\n📭 No trades scheduled for today")
            return
        
        total_investment = (trade_df['shares'] * trade_df['buy']).sum() if 'shares' in trade_df.columns else 0
        
        print(f"\n✅ TODAY'S TRADES:")
        print(f"   Total Investment: ${total_investment:,.0f}")
        
        for _, row in trade_df.iterrows():
            symbol = row['symbol']
            se = enhanced_df[enhanced_df['symbol'] == symbol]
            
            if len(se) > 0:
                s = se.iloc[0]
                reliable_mark = "✅" if s.get('llm_reliable', True) else "⚠️"
                print(f"\n   {symbol} {reliable_mark}")
                print(f"      Buy: ${row['buy']:.2f} | SL: ${row['SL']:.2f} | TP: ${row['tp']:.2f}")
                print(f"      LLM: {s['llm_signal']} | Quality: {row['confidence']:.3f}")
                print(f"      Shares: {row['shares']} | R:R = 1:{row['RRR']:.2f}")
        
        print("\n" + "="*70)
        print("📋 ACTION ITEMS:")
        print("="*70)
        print("   1. Place LIMIT orders at BUY price")
        print("   2. Set STOP LOSS at SL price")
        print("   3. Set TAKE PROFIT at TP price")
        print("   4. If gap down >2%, skip the trade")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--morning':
        morning_check()
    else:
        nightly_trading_system()