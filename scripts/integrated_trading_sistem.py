# integrated_trading_system.py - সেরা সমাধান

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stable_baselines3 import PPO
import os
import time

# =========================================================
# PART 1: MARKET CYCLE DETECTOR (Advanced)
# =========================================================

class MarketCycleDetector:
    """মার্কেট সাইকেল ডিটেক্টর - আপট্রেন্ড, ডাউনট্রেন্ড, কারেকশন চিহ্নিত করে"""
    
    def __init__(self, market_df):
        self.market_df = market_df
        self.cycle_status = self.detect_cycle()
        
    def detect_cycle(self):
        """বর্তমান মার্কেট সাইকেল চিহ্নিত করুন"""
        
        latest_date = self.market_df['date'].max()
        lookback_90 = latest_date - timedelta(days=90)
        data_90d = self.market_df[self.market_df['date'] >= lookback_90]
        
        daily_returns = []
        for symbol in data_90d['symbol'].unique()[:50]:
            sym_data = data_90d[data_90d['symbol'] == symbol].sort_values('date')
            if len(sym_data) > 1:
                returns = sym_data['close'].pct_change().dropna()
                daily_returns.extend(returns.tolist())
        
        if len(daily_returns) < 20:
            return {'phase': 'unknown', 'trend': 'neutral'}
        
        returns_series = pd.Series(daily_returns)
        sma_5 = returns_series.tail(5).mean()
        sma_20 = returns_series.tail(20).mean()
        sma_50 = returns_series.tail(50).mean() if len(returns_series) >= 50 else sma_20
        
        if sma_5 > sma_20 > sma_50:
            trend, phase = "strong_uptrend", "bullish"
        elif sma_5 > sma_20 and sma_20 > 0:
            trend, phase = "uptrend_starting", "accumulation"
        elif sma_5 < sma_20 < sma_50:
            trend, phase = "strong_downtrend", "bearish"
        elif sma_5 < sma_20 and sma_20 < 0:
            trend, phase = "downtrend_continuing", "distribution"
        else:
            trend, phase = "sideways", "consolidation"
        
        recent_high = returns_series.tail(30).max() if len(returns_series) >= 30 else 0
        recent_low = returns_series.tail(10).min() if len(returns_series) >= 10 else 0
        
        correction_active = (recent_high > 0.02 and recent_low < -0.03)
        correction_depth = abs(recent_low) if correction_active else 0
        
        return {
            'trend': trend, 'phase': phase,
            'correction_active': correction_active, 'correction_depth': correction_depth,
            'sma_5': sma_5, 'sma_20': sma_20, 'sma_50': sma_50
        }
    
    def should_wait_for_uptrend(self):
        if self.cycle_status['correction_active']:
            return True, f"Correction active (-{self.cycle_status['correction_depth']:.1%})"
        if self.cycle_status['trend'] in ['downtrend_continuing', 'strong_downtrend']:
            return True, f"{self.cycle_status['trend']} - wait"
        return False, "Market condition favorable"
    
    def get_cycle_advice(self):
        cycle = self.cycle_status
        advice_map = {
            'bullish': ('AGGRESSIVE', 100, 'আপট্রেন্ড চলছে - পূর্ণ পজিশন নিন'),
            'accumulation': ('CAUTIOUS_BUY', 50, 'আপট্রেন্ড শুরু হতে পারে - ধীরে ধীরে পজিশন বাড়ান'),
            'bearish': ('AVOID', 0, 'ডাউনট্রেন্ড চলছে - ট্রেড এড়িয়ে চলুন'),
            'distribution': ('EXIT', 0, 'টপ গঠন হচ্ছে - প্রফিট বুক করুন'),
        }
        action, max_pos, desc = advice_map.get(cycle['phase'], ('WAIT', 25, 'সাইডওয়েস - অপেক্ষা করুন'))
        return {'action': action, 'max_position': max_pos, 'description': desc}


# =========================================================
# PART 2: DAILY SIGNAL GENERATOR (with cycle awareness)
# =========================================================

class DailySignalGenerator:
    """প্রতিদিন চালান - প্রতিটি সিম্বলের জন্য আজকের সিদ্ধান্ত দেয়"""
    
    def __init__(self):
        self.symbol_models = {}
        self.load_models()
    
    def load_models(self):
        model_dir = './csv/ppo_models/per_symbol/'
        if not os.path.exists(model_dir):
            print("   ⚠️ No models found. Train PPO first.")
            return
        for f in os.listdir(model_dir):
            if f.startswith('ppo_') and not f.endswith('.pkl'):
                symbol = f.replace('ppo_', '')
                try:
                    self.symbol_models[symbol] = PPO.load(f"{model_dir}/{f}")
                except:
                    pass
        print(f"   ✅ Loaded {len(self.symbol_models)} symbol models")
    
    def prepare_observation(self, data):
        """Observation prepare - আপনার environment অনুযায়ী কাস্টমাইজ করুন"""
        import numpy as np
        # Placeholder - 실제 observation logic এখানে দিন
        return np.zeros(10)
    
    def get_today_signal(self, symbol, current_data):
        if symbol not in self.symbol_models:
            return 'HOLD'
        model = self.symbol_models[symbol]
        observation = self.prepare_observation(current_data)
        action, _ = model.predict(observation, deterministic=True)
        return ['HOLD', 'BUY', 'SELL'][action] if action in [0,1,2] else 'HOLD'
    
    def generate_daily_signals(self, market_df, cycle_detector, pending_signals=None):
        print("\n" + "="*70)
        print(f"📊 DAILY SIGNALS - {datetime.now().strftime('%Y-%m-%d')}")
        print("="*70)
        
        should_wait, reason = cycle_detector.should_wait_for_uptrend()
        
        if should_wait:
            print(f"⏸️ {reason}")
            if pending_signals:
                print(f"   📋 {len(pending_signals)} pending signals waiting for uptrend")
            return []
        
        cycle_advice = cycle_detector.get_cycle_advice()
        print(f"📈 Market: {cycle_advice['description']}")
        
        signals = []
        latest_date = market_df['date'].max()
        meta_df = pd.read_csv('./csv/model_metadata.csv')
        top_symbols = meta_df.nlargest(20, 'auc')['symbol'].tolist()
        
        for symbol in top_symbols:
            sym_data = market_df[market_df['symbol'] == symbol].sort_values('date')
            if len(sym_data) < 10:
                continue
            
            signal = self.get_today_signal(symbol, sym_data)
            
            if signal != 'HOLD':
                latest_price = sym_data.iloc[-1]['close']
                atr = sym_data['atr'].iloc[-1] if 'atr' in sym_data.columns else latest_price * 0.02
                
                if signal == 'BUY':
                    stop_loss = latest_price - (1.5 * atr)
                    take_profit = latest_price + (2.5 * atr)
                else:
                    stop_loss = latest_price + (1.5 * atr)
                    take_profit = latest_price - (2.5 * atr)
                
                signals.append({
                    'symbol': symbol, 'date': latest_date.strftime('%Y-%m-%d'),
                    'signal': signal, 'entry_price': round(latest_price, 2),
                    'stop_loss': round(stop_loss, 2), 'take_profit': round(take_profit, 2)
                })
        
        if signals:
            pd.DataFrame(signals).to_csv('./csv/daily_signals.csv', index=False)
            buy_cnt = sum(1 for s in signals if s['signal'] == 'BUY')
            print(f"\n✅ {len(signals)} signals (BUY: {buy_cnt}, SELL: {len(signals)-buy_cnt})")
        
        return signals


# =========================================================
# PART 3: INTEGRATED SYSTEM (Best of both)
# =========================================================

class IntegratedTradingSystem:
    """
    সম্পূর্ণ সিস্টেম যা:
    1. মার্কেট সাইকেল বুঝে (market_cycle_aware)
    2. প্রতিদিন সিগন্যাল দেয় (complete_system)
    3. পেন্ডিং সিগন্যাল স্টোর করে
    4. আপট্রেন্ড শুরু হলে অটো ট্রেড করে
    """
    
    def __init__(self):
        self.daily_generator = DailySignalGenerator()
        self.pending_signals = []
        self.cycle_detector = None
        self.last_cycle_check = None
    
    def run(self):
        """মূল ফাংশন - সবকিছু একসাথে"""
        
        print("\n" + "="*70)
        print("🏦 INTEGRATED TRADING SYSTEM")
        print("="*70)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load market data
        market_df = pd.read_csv('./csv/mongodb.csv')
        market_df['date'] = pd.to_datetime(market_df['date'])
        
        # Detect cycle
        self.cycle_detector = MarketCycleDetector(market_df)
        cycle = self.cycle_detector.cycle_status
        
        print(f"\n🔄 Market Cycle: {cycle['phase'].upper()}")
        print(f"   Trend: {cycle['trend']}")
        print(f"   Correction: {'Yes' if cycle['correction_active'] else 'No'}")
        
        # Check if we should wait
        should_wait, reason = self.cycle_detector.should_wait_for_uptrend()
        
        if should_wait:
            print(f"\n⏳ {reason}")
            
            # Store pending signals for uptrend
            if not self.pending_signals:
                self.pending_signals = self._generate_pending_signals(market_df)
                print(f"   📋 Stored {len(self.pending_signals)} pending signals")
            
            # Check if uptrend started
            if self._check_uptrend_started(market_df):
                print("\n🚀 UPTREND STARTED! Executing pending signals...")
                self._execute_pending_signals()
            else:
                print("   💡 Waiting for uptrend confirmation...")
                return self.pending_signals
        else:
            print(f"\n✅ Market favorable - Generating daily signals")
            signals = self.daily_generator.generate_daily_signals(market_df, self.cycle_detector, self.pending_signals)
            
            # Also execute any pending signals
            if self.pending_signals:
                self._execute_pending_signals()
            
            return signals
    
    def _generate_pending_signals(self, market_df):
        """আপট্রেন্ডের জন্য পেন্ডিং সিগন্যাল"""
        pending = []
        latest_date = market_df['date'].max()
        meta_df = pd.read_csv('./csv/model_metadata.csv')
        good_symbols = meta_df[meta_df['auc'] >= 0.80]['symbol'].tolist()
        
        for symbol in good_symbols[:20]:
            sym_data = market_df[market_df['symbol'] == symbol].sort_values('date')
            if len(sym_data) >= 20:
                latest_price = sym_data.iloc[-1]['close']
                support_20 = sym_data['low'].tail(20).min()
                pending.append({
                    'symbol': symbol, 'current_price': latest_price,
                    'support_level': support_20, 'status': 'pending'
                })
        return pending
    
    def _check_uptrend_started(self, market_df):
        """আপট্রেন্ড শুরু হয়েছে কিনা চেক করুন"""
        # Re-check cycle
        new_cycle = MarketCycleDetector(market_df)
        should_wait, _ = new_cycle.should_wait_for_uptrend()
        return not should_wait
    
    def _execute_pending_signals(self):
        """পেন্ডিং সিগন্যাল এক্সিকিউট করুন"""
        executed = [s for s in self.pending_signals if s.get('status') == 'pending']
        if executed:
            trade_df = pd.DataFrame([{
                'symbol': s['symbol'], 'buy': s['current_price'],
                'SL': s['current_price'] * 0.97, 'tp': s['current_price'] * 1.06,
                'confidence': 0.70, 'date': datetime.now().strftime('%Y-%m-%d'), 'RRR': 2.0
            } for s in executed])
            trade_df.to_csv('./csv/trade_stock.csv', index=False)
            print(f"✅ {len(executed)} signals executed")
            for s in executed:
                s['status'] = 'executed'


# =========================================================
# MAIN
# =========================================================

def main():
    system = IntegratedTradingSystem()
    result = system.run()
    
    print("\n" + "="*70)
    print("✅ Trading system run complete")
    print("="*70)

if __name__ == "__main__":
    main()