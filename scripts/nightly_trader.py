# nightly_trader.py - একটাই স্ক্রিপ্ট, সব রাতে হয়

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def nightly_trading_system():
    """
    একটাই ফাংশন - রাতে সব সিদ্ধান্ত নেয়
    সকালে শুধু দেখবেন আর ট্রেড করবেন
    """
    
    print("="*70)
    print("🌙 NIGHTLY TRADING DECISION SYSTEM")
    print("="*70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Generating decisions for TOMORROW's trading")
    print("="*70)
    
    # =========================================================
    # STEP 1: LOAD DATA
    # =========================================================
    base_path = './csv/'
    
    market_df = pd.read_csv(os.path.join(base_path, 'mongodb.csv'))
    pred_df = pd.read_csv(os.path.join(base_path, 'prediction_log.csv'))
    meta_df = pd.read_csv(os.path.join(base_path, 'model_metadata.csv'))
    
    market_df['date'] = pd.to_datetime(market_df['date'])
    pred_df['date'] = pd.to_datetime(pred_df['date'], format='mixed', errors='coerce')
    
    # Latest data
    latest_date = market_df['date'].max()
    today_data = market_df[market_df['date'] == latest_date]
    
    print(f"\n📊 Market Data: {latest_date.strftime('%Y-%m-%d')}")
    print(f"   Symbols: {len(today_data['symbol'].unique())}")
    
    # =========================================================
    # STEP 2: MARKET REGIME
    # =========================================================
    regime = detect_market_regime(market_df)
    print(f"\n📈 Market Regime: {regime}")
    
    if regime == 'BEAR':
        print(f"\n⚠️ BEAR MARKET - No trades tomorrow")
        print(f"   Reason: Market down, waiting for recovery")
        
        # Empty trade file
        empty_df = pd.DataFrame(columns=['symbol', 'date', 'buy', 'SL', 'tp', 'confidence', 'RRR'])
        empty_df.to_csv('./csv/trade_stock.csv', index=False)
        
        # Save decision log
        save_decision_log([], regime, "No trades - Bear market")
        return True
    
    # =========================================================
    # STEP 3: GOOD MODELS
    # =========================================================
    good_symbols = meta_df[meta_df['auc'] >= 0.55]['symbol'].unique()
    print(f"\n✅ Good Models: {len(good_symbols)} symbols")
    
    # =========================================================
    # STEP 4: CREATE PROB_UP
    # =========================================================
    if 'prob_up' not in pred_df.columns:
        if 'prediction' in pred_df.columns:
            pred_df['prob_up'] = pred_df.apply(
                lambda row: row['confidence_score'] / 100 if row['prediction'] == 1 
                else (100 - row['confidence_score']) / 100, axis=1
            )
        else:
            pred_df['prob_up'] = pred_df['confidence_score'] / 100
    
    # Latest predictions
    latest_pred = pred_df[pred_df['date'] == latest_date]
    if len(latest_pred) == 0:
        nearest_date = pred_df[pred_df['date'] <= latest_date]['date'].max()
        latest_pred = pred_df[pred_df['date'] == nearest_date]
    
    # =========================================================
    # STEP 5: GENERATE SIGNALS
    # =========================================================
    print("\n🎯 Generating tomorrow's trading signals...")
    
    decisions = []
    
    for symbol in good_symbols:
        # Get today's price
        symbol_today = today_data[today_data['symbol'] == symbol]
        if len(symbol_today) == 0:
            continue
        
        current_price = symbol_today.iloc[0]['close']
        
        # Get prediction
        symbol_pred = latest_pred[latest_pred['symbol'] == symbol]
        if len(symbol_pred) == 0:
            continue
        
        prob_up = symbol_pred.iloc[0]['prob_up']
        confidence = symbol_pred.iloc[0]['confidence_score']
        
        # Get AUC
        auc = meta_df[meta_df['symbol'] == symbol]['auc'].values
        auc_score = auc[0] if len(auc) > 0 else 0.5
        
        # Calculate score
        score = (prob_up * 0.40) + ((confidence / 100) * 0.30) + (auc_score * 0.30)
        
        # Market adjustment
        if regime == 'BULL':
            score = score * 1.1
            min_buy = 0.55
        else:  # SIDEWAYS
            min_buy = 0.60
        
        score = min(max(score, 0), 1)
        
        # Decision
        if score >= min_buy:
            decision = "BUY"
            atr = calculate_atr(symbol, market_df)
            if atr == 0:
                atr = current_price * 0.02
            
            stop_loss = current_price - (1.5 * atr)
            take_profit = current_price + (2.5 * atr)
            risk_reward = round((take_profit - current_price) / (current_price - stop_loss), 2)
            
        elif score <= 0.25:
            decision = "SELL"
            atr = calculate_atr(symbol, market_df)
            if atr == 0:
                atr = current_price * 0.02
            stop_loss = current_price + (1.5 * atr)
            take_profit = current_price - (2.5 * atr)
            risk_reward = round((current_price - take_profit) / (stop_loss - current_price), 2)
        else:
            decision = "HOLD"
            stop_loss = 0
            take_profit = 0
            risk_reward = 0
        
        decisions.append({
            'symbol': symbol,
            'decision': decision,
            'score': round(score, 3),
            'current_price': round(current_price, 2),
            'entry_price': round(current_price, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'risk_reward': risk_reward,
            'prob_up': round(prob_up, 3),
            'confidence': confidence,
            'auc': round(auc_score, 3)
        })
    
    # =========================================================
    # STEP 6: CREATE TRADE FILE (ONLY BUY)
    # =========================================================
    decisions_df = pd.DataFrame(decisions)
    buy_decisions = decisions_df[decisions_df['decision'] == 'BUY'].sort_values('score', ascending=False)
    
    # Limit to top 10
    max_trades = 10
    top_buy = buy_decisions.head(max_trades)
    
    if len(top_buy) > 0:
        trade_df = top_buy[['symbol', 'entry_price', 'stop_loss', 'take_profit', 'score']].copy()
        trade_df.columns = ['symbol', 'buy', 'SL', 'tp', 'confidence']
        trade_df['date'] = latest_date.strftime('%Y-%m-%d')
        trade_df['RRR'] = top_buy['risk_reward'].values
        
        trade_df.to_csv('./csv/trade_stock.csv', index=False)
        print(f"\n✅ {len(trade_df)} BUY signals for tomorrow")
    else:
        print(f"\n⚠️ No BUY signals for tomorrow")
        empty_df = pd.DataFrame(columns=['symbol', 'date', 'buy', 'SL', 'tp', 'confidence', 'RRR'])
        empty_df.to_csv('./csv/trade_stock.csv', index=False)
    
    # =========================================================
    # STEP 7: SAVE DETAILED DECISIONS
    # =========================================================
    decisions_df.to_csv('./csv/nightly_decisions.csv', index=False)
    
    # =========================================================
    # STEP 8: PRINT SUMMARY (WHAT YOU SEE AT NIGHT)
    # =========================================================
    print("\n" + "="*70)
    print("📊 TOMORROW'S TRADING DECISIONS")
    print("="*70)
    print(f"   Market Regime: {regime}")
    print(f"   Total Analyzed: {len(decisions_df)}")
    print(f"   BUY: {len(decisions_df[decisions_df['decision'] == 'BUY'])}")
    print(f"   SELL: {len(decisions_df[decisions_df['decision'] == 'SELL'])}")
    print(f"   HOLD: {len(decisions_df[decisions_df['decision'] == 'HOLD'])}")
    
    if len(top_buy) > 0:
        print("\n" + "🔥"*35)
        print("🔥 YOUR TRADING PLAN FOR TOMORROW 🔥")
        print("🔥"*35)
        
        for i, (_, row) in enumerate(top_buy.iterrows(), 1):
            print(f"\n{i}. {row['symbol']}")
            print(f"   📌 Action: BUY at {row['entry_price']}")
            print(f"   🛑 Stop Loss: {row['stop_loss']} (loss: {((row['stop_loss'] - row['entry_price'])/row['entry_price']*100):.1f}%)")
            print(f"   🎯 Target: {row['take_profit']} (profit: {((row['take_profit'] - row['entry_price'])/row['entry_price']*100):.1f}%)")
            print(f"   📊 Risk:Reward = 1:{row['risk_reward']}")
            print(f"   💯 Confidence: {row['score']*100:.0f}%")
        
        # Position sizing
        print("\n" + "="*70)
        print("💰 POSITION SIZING (for 5 Lac capital)")
        print("="*70)
        
        capital = 500000
        risk_per_trade = capital * 0.02  # 2% risk per trade
        
        for _, row in top_buy.iterrows():
            risk_per_share = row['entry_price'] - row['stop_loss']
            shares = int(risk_per_trade / risk_per_share) if risk_per_share > 0 else 0
            investment = shares * row['entry_price']
            print(f"   {row['symbol']}: Buy {shares} shares (${investment:,.0f})")
        
    else:
        print("\n" + "💤"*35)
        print("💤 NO TRADES TOMORROW")
        print("💤"*35)
        print("\n   Reason: No quality BUY signals")
        print("   Action: Wait for next night's signal")
    
    # Save decision log
    save_decision_log(top_buy, regime, f"{len(top_buy)} trades for tomorrow")
    
    print("\n" + "="*70)
    print("✅ NIGHTLY DECISION COMPLETE")
    print("📁 Check trade_stock.csv for broker orders")
    print("="*70)
    
    return True


def detect_market_regime(market_df):
    """Detect market regime"""
    if len(market_df) < 20:
        return 'SIDEWAYS'
    
    # Use top 50 symbols
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
    else:
        return 'SIDEWAYS'


def calculate_atr(symbol, market_df, period=14):
    """Calculate ATR"""
    sym_data = market_df[market_df['symbol'] == symbol].sort_values('date')
    if len(sym_data) < period:
        return 0
    
    high = sym_data['high'].values[-period:]
    low = sym_data['low'].values[-period:]
    close = sym_data['close'].values[-period-1:-1] if len(sym_data) > period else sym_data['close'].values[-period:]
    
    tr = np.maximum(high - low, 
                   np.maximum(np.abs(high - close), 
                            np.abs(low - close)))
    
    return np.mean(tr)


def save_decision_log(decisions, regime, summary):
    """Save nightly decision log"""
    log_file = './csv/trading_decisions_log.csv'
    
    new_entry = pd.DataFrame([{
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'market_regime': regime,
        'num_trades': len(decisions),
        'summary': summary,
        'symbols': ', '.join([d['symbol'] for d in decisions.head(5).to_dict('records')]) if len(decisions) > 0 else 'None'
    }])
    
    if os.path.exists(log_file):
        existing = pd.read_csv(log_file)
        updated = pd.concat([existing, new_entry], ignore_index=True)
    else:
        updated = new_entry
    
    updated.to_csv(log_file, index=False)


# =========================================================
# MORNING CHECK (optional - just to see what was decided)
# =========================================================

def morning_check():
    """সকালে চালান - দেখুন রাতে কী সিদ্ধান্ত নিয়েছেন"""
    print("\n" + "="*70)
    print("🌅 MORNING TRADING CHECK")
    print("="*70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d')}")
    print("="*70)
    
    try:
        trade_df = pd.read_csv('./csv/trade_stock.csv')
        
        if len(trade_df) == 0:
            print("\n📭 No trades scheduled for today")
            print("   Check nightly_decisions.csv for details")
            return
        
        print(f"\n✅ TODAY'S TRADES (Decided last night):")
        print(trade_df[['symbol', 'buy', 'SL', 'tp', 'confidence', 'RRR']].to_string(index=False))
        
        print("\n" + "="*70)
        print("📋 ACTION ITEMS FOR TODAY:")
        print("="*70)
        print("   1. Place LIMIT orders at BUY price")
        print("   2. Set STOP LOSS at SL")
        print("   3. Set TAKE PROFIT at TP")
        print("   4. Check market open (if gap down >2%, skip)")
        print("="*70)
        
    except:
        print("\n❌ No trade file found")
        print("   Run nightly_trader.py last night?")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--morning':
        # সকালে চেক করুন (optional)
        morning_check()
    else:
        # রাতে ডিসিশন নিন
        nightly_trading_system()
