# create_trade_stock_complete.py - সম্পূর্ণ ওয়ার্কিং ভার্সন

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_trade_stock_complete():
    """হাজিং ফেস ডেটাসেটের জন্য সম্পূর্ণ ট্রেডিং সিগন্যাল জেনারেটর"""
    
    print("="*70)
    print("📊 ADVANCED TRADING SIGNAL GENERATOR")
    print("="*70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # পাথ ডিফাইন
    base_path = './csv/'
    files = {
        'sr': os.path.join(base_path, 'support_resistance.csv'),
        'market': os.path.join(base_path, 'mongodb.csv'),
        'pred': os.path.join(base_path, 'prediction_log.csv'),
        'conf': os.path.join(base_path, 'xgb_confidence.csv'),
        'meta': os.path.join(base_path, 'model_metadata.csv'),
        'rsi': os.path.join(base_path, 'rsi_diver.csv')
    }
    
    # চেক ফাইল আছে কিনা
    missing = [k for k, v in files.items() if not os.path.exists(v)]
    if missing:
        print(f"\n❌ Missing files: {missing}")
        print("   Download from: https://huggingface.co/datasets/ahashanahmed/csv")
        return False
    
    # লোড ডাটা
    print("\n📂 Loading data files...")
    sr_df = pd.read_csv(files['sr'])
    market_df = pd.read_csv(files['market'])
    pred_df = pd.read_csv(files['pred'])
    conf_df = pd.read_csv(files['conf'])
    meta_df = pd.read_csv(files['meta'])
    rsi_df = pd.read_csv(files['rsi'])
    
    # কনভার্ট ডেট
    sr_df['current_date'] = pd.to_datetime(sr_df['current_date'])
    market_df['date'] = pd.to_datetime(market_df['date'])
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    conf_df['date'] = pd.to_datetime(conf_df['date'])
    
    print(f"   ✅ Support/Resistance: {len(sr_df)} records")
    print(f"   ✅ Market data: {len(market_df)} rows, {market_df['symbol'].nunique()} symbols")
    print(f"   ✅ Prediction log: {len(pred_df)} records")
    print(f"   ✅ XGBoost confidence: {len(conf_df)} records")
    print(f"   ✅ Model metadata: {len(meta_df)} records")
    print(f"   ✅ RSI divergence: {len(rsi_df)} records")
    
    # ফিল্টার গুড মডেল (AUC >= 0.55)
    good_symbols = meta_df[meta_df['auc'] >= 0.55]['symbol'].unique()
    print(f"\n🔍 Filtering GOOD models: {len(good_symbols)} symbols (AUC >= 0.55)")
    
    # মার্জ কনফিডেন্স
    pred_df = pred_df.merge(conf_df[['symbol', 'date', 'confidence_score']], 
                           on=['symbol', 'date'], how='left')
    pred_df = pred_df[pred_df['symbol'].isin(good_symbols)]
    
    # সাপোর্ট লেভেল ফিল্টার
    sr_support = sr_df[sr_df['type'] == 'support'].copy()
    print(f"   ✅ Support levels only: {len(sr_support)}")
    
    # ক্যালকুলেট এটিআর
    print("\n📊 Calculating ATR and indicators...")
    
    def calc_atr(group):
        high = group['high'].values
        low = group['low'].values
        close = group['close'].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        atr = pd.Series(tr).rolling(window=14).mean()
        return atr.fillna(tr.mean())
    
    market_df['atr'] = market_df.groupby('symbol').apply(calc_atr).reset_index(level=0, drop=True)
    market_df['atr'] = market_df['atr'].fillna(market_df['close'] * 0.02)
    
    # জেনারেট সিগন্যাল
    print("\n🎯 Generating trading signals...")
    
    signals = []
    skipped_no_data = 0
    skipped_broken = 0
    skipped_no_pred = 0
    skipped_low_score = 0
    
    for idx, row in sr_support.iterrows():
        symbol = row['symbol']
        current_date = row['current_date']
        support_level = row['current_low']
        strength = row.get('strength', 'Moderate')
        gap_days = row.get('gap_days', 0)
        
        # সিম্বলের মার্কেট ডাটা
        sym_market = market_df[market_df['symbol'] == symbol].sort_values('date')
        
        if len(sym_market) < 5:
            skipped_no_data += 1
            continue
        
        # ডেট ম্যাচ
        date_diff = (sym_market['date'] - current_date).abs()
        min_diff_idx = date_diff.idxmin()
        min_diff = date_diff.min()
        
        if min_diff.days > 5:  # 5 দিনের বেশি পুরনো
            skipped_no_data += 1
            continue
        
        market_idx = sym_market.index.get_loc(min_diff_idx)
        
        if market_idx + 1 >= len(sym_market):
            skipped_no_data += 1
            continue
        
        # চেক সাপোর্ট হোল্ড করেছে কিনা
        next_candle_low = sym_market.iloc[market_idx + 1]['low']
        if next_candle_low <= support_level:
            skipped_broken += 1
            continue
        
        # প্রেডিকশন চেক
        pred_row = pred_df[(pred_df['symbol'] == symbol) & 
                          (pred_df['date'] == current_date)]
        
        if len(pred_row) == 0:
            skipped_no_pred += 1
            continue
        
        # এক্সট্রাক্ট প্রোবাবিলিটি
        prob_up = pred_row.iloc[0].get('prob_up', 0.5)
        if pd.isna(prob_up):
            prob_up = 0.5
            
        confidence = pred_row.iloc[0].get('confidence_score', 50)
        if pd.isna(confidence):
            confidence = 50
            
        # আরএসআই ডাইভারজেন্স
        rsi_row = rsi_df[(rsi_df['symbol'] == symbol) & 
                        (rsi_df['last_date'] == current_date)]
        
        rsi_bonus = 0.15 if (len(rsi_row) > 0 and 
                            rsi_row.iloc[0].get('divergence_type') == 'Bullish') else 0
        
        # স্ট্রেন্থ মাল্টিপ্লায়ার
        strength_mult = {'Weak': 0.6, 'Moderate': 0.8, 'Strong': 1.0}.get(strength, 0.7)
        
        # ফাইনাল স্কোর
        base_score = (prob_up * 0.5) + ((confidence / 100) * 0.35) + (rsi_bonus * 0.15)
        final_score = base_score * strength_mult
        final_score = min(max(final_score, 0), 1)
        
        # সিগন্যাল ডিটারমাইন
        if final_score >= 0.75:
            signal = "STRONG BUY"
        elif final_score >= 0.60:
            signal = "BUY"
        elif final_score >= 0.45:
            signal = "NEUTRAL"
        elif final_score >= 0.25:
            signal = "SELL"
        else:
            signal = "STRONG SELL"
        
        # নিউট্রাল স্কিপ
        if signal == "NEUTRAL":
            skipped_low_score += 1
            continue
        
        # এন্ট্রি প্রাইস
        entry_price = sym_market.iloc[market_idx + 1]['close']
        atr_value = sym_market.iloc[market_idx + 1].get('atr', entry_price * 0.02)
        
        # স্টপ লস এবং টেক প্রফিট
        stop_loss = entry_price - (1.5 * atr_value)
        take_profit_1 = entry_price + (2.0 * atr_value)
        take_profit_2 = entry_price + (3.0 * atr_value)
        
        # আরআরআর ক্যালকুলেশন
        risk = entry_price - stop_loss
        rrr_1 = round((take_profit_1 - entry_price) / risk, 2) if risk > 0 else 0
        rrr_2 = round((take_profit_2 - entry_price) / risk, 2) if risk > 0 else 0
        
        # সেভ সিগন্যাল
        signals.append({
            'symbol': symbol,
            'date': current_date.strftime('%Y-%m-%d'),
            'signal': signal,
            'score': round(final_score, 4),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'tp1': round(take_profit_1, 2),
            'tp2': round(take_profit_2, 2),
            'rrr_tp1': rrr_1,
            'rrr_tp2': rrr_2,
            'support_level': round(support_level, 2),
            'strength': strength,
            'gap_days': gap_days,
            'xgb_prob': round(prob_up, 4),
            'xgb_confidence': confidence,
            'rsi_divergence': 'Bullish' if rsi_bonus > 0 else 'None',
            'atr': round(atr_value, 4),
            'risk_amount': round(risk, 2),
            'created_at': datetime.now().strftime('%Y-%Y-%m-%d %H:%M:%S')
        })
    
    # রিপোর্ট
    print(f"\n   ✅ Processed: {len(signals)} support levels")
    print(f"   ⚠️ Skipped (no data): {skipped_no_data}")
    print(f"   ⚠️ Skipped (support broken): {skipped_broken}")
    print(f"   ⚠️ Skipped (no prediction): {skipped_no_pred}")
    print(f"   ⚠️ Skipped (low score): {skipped_low_score}")
    
    if not signals:
        print("\n❌ No signals generated!")
        return False
    
    # আউটপুট ডাটাফ্রেম
    output_df = pd.DataFrame(signals)
    output_df = output_df.sort_values('score', ascending=False)
    
    # সেভ মেইন ফাইল
    output_path = './csv/trade_stock_advanced.csv'
    output_df.to_csv(output_path, index=False)
    
    # সেভ পিপিও ফাইল
    ppo_signals = output_df[output_df['signal'].isin(['STRONG BUY', 'BUY'])].copy()
    
    if len(ppo_signals) > 0:
        ppo_df = ppo_signals[['symbol', 'entry_price', 'stop_loss', 'tp1', 'score']].copy()
        ppo_df.columns = ['symbol', 'buy', 'SL', 'tp', 'confidence']
        ppo_df['date'] = datetime.now().strftime('%Y-%m-%d')
        ppo_df['RRR'] = ppo_df.apply(
            lambda x: round((x['tp'] - x['buy']) / (x['buy'] - x['SL']), 2), axis=1
        )
        ppo_df['source'] = 'Advanced_AI'
        
        ppo_path = './csv/trade_stock.csv'
        ppo_df.to_csv(ppo_path, index=False)
        print(f"\n   ✅ PPO signals: {len(ppo_df)}")
    else:
        print("\n   ⚠️ No BUY signals for PPO")
    
    # সারাংশ প্রিন্ট
    print("\n" + "="*70)
    print("📊 SIGNAL SUMMARY")
    print("="*70)
    print(output_df['signal'].value_counts().to_string())
    
    print(f"\n🔥 TOP 10 SIGNALS:")
    print("-"*70)
    top_cols = ['symbol', 'signal', 'score', 'entry_price', 'stop_loss', 'tp1', 'rrr_tp1']
    print(output_df.head(10)[top_cols].to_string())
    
    print("\n" + "="*70)
    print(f"✅ Total signals: {len(output_df)}")
    print(f"✅ Saved to: {output_path}")
    print(f"✅ PPO ready: {ppo_path}")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = create_trade_stock_complete()
    if not success:
        print("\n❌ Trade stock generation failed!")