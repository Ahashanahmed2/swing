# run_complete_system.py - সবকিছু একসাথে চালানোর জন্য

import pandas as pd
import numpy as np
import os
import subprocess
import sys
from datetime import datetime

print("="*80)
print("🏦 COMPLETE HEDGE FUND TRADING SYSTEM")
print("="*80)
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# =========================================================
# STEP 1: Update mongodb.csv (if needed)
# =========================================================
print("\n📊 STEP 1: Checking market data...")
if os.path.exists('./csv/mongodb.csv'):
    df = pd.read_csv('./csv/mongodb.csv')
    print(f"   ✅ mongodb.csv exists: {len(df)} rows, {df['symbol'].nunique()} symbols")
    print(f"   📅 Date range: {df['date'].min()} to {df['date'].max()}")
else:
    print("   ❌ mongodb.csv not found! Run mongodb.py first")
    sys.exit(1)
 =========================================================
# STEP 3: Generate trading signals with FIXED logic
# =========================================================
print("\n📊 STEP 3: Generating trading signals...")

# লোড ফাইল
sr_df = pd.read_csv('./csv/support_resistance.csv')
market_df = pd.read_csv('./csv/mongodb.csv')
pred_df = pd.read_csv('./csv/prediction_log.csv')

print(f"   Loaded: {len(sr_df)} levels, {len(market_df)} market rows, {len(pred_df)} predictions")

# ডেট কনভার্ট
sr_df['current_date'] = pd.to_datetime(sr_df['current_date'])
market_df['date'] = pd.to_datetime(market_df['date'])
pred_df['date'] = pd.to_datetime(pred_df['date'])

# শুধু সাপোর্ট টাইপ
sr_support = sr_df[sr_df['type'] == 'support'].copy()
print(f"   Support levels: {len(sr_support)}")

# প্রেডিকশন থেকে prob_up তৈরি করুন
if 'prediction' in pred_df.columns:
    pred_df['prob_up'] = pred_df.apply(
        lambda row: row['confidence_score'] / 100 if row['prediction'] == 1 
        else (100 - row['confidence_score']) / 100,
        axis=1
    )
else:
    pred_df['prob_up'] = pred_df['confidence_score'] / 100

# ফিল্টার ভালো প্রেডিকশন (confidence > 55)
good_pred_symbols = pred_df[pred_df['confidence_score'] > 55]['symbol'].unique()
print(f"   Good prediction symbols: {len(good_pred_symbols)}")

# সিগন্যাল জেনারেট
signals = []
matched = 0
not_matched = 0

for _, row in sr_support.iterrows():
    symbol = row['symbol']
    
    # চেক সিম্বল ভালো প্রেডিকশন আছে কিনা
    if symbol not in good_pred_symbols:
        not_matched += 1
        continue
    
    current_date = row['current_date']
    support_level = row['level_price']
    strength = row['strength']
    
    # মার্কেট ডাটা
    sym_market = market_df[market_df['symbol'] == symbol].sort_values('date')
    
    if len(sym_market) < 5:
        not_matched += 1
        continue
    
    # ক্লোজেস্ট ডেট
    date_diff = (sym_market['date'] - current_date).abs()
    if len(date_diff) == 0:
        not_matched += 1
        continue
    
    min_diff = date_diff.min()
    if min_diff.days > 10:
        not_matched += 1
        continue
    
    market_idx = date_diff.idxmin()
    
    # প্রেডিকশন
    pred_row = pred_df[(pred_df['symbol'] == symbol) & 
                       (pred_df['date'] == current_date)]
    
    if len(pred_row) == 0:
        not_matched += 1
        continue
    
    prob_up = pred_row.iloc[0]['prob_up']
    confidence = pred_row.iloc[0]['confidence_score']
    
    # স্কোর
    strength_mult = {'Weak': 0.6, 'Moderate': 0.8, 'Strong': 1.0}.get(strength, 0.7)
    score = prob_up * strength_mult
    
    if score >= 0.60:
        # এন্ট্রি
        entry_idx = market_df.index.get_loc(market_idx) + 1
        if entry_idx >= len(market_df):
            continue
            
        entry_price = market_df.iloc[entry_idx]['close']
        
        # সিম্পল SL/TP
        atr = entry_price * 0.02
        stop_loss = entry_price - (1.5 * atr)
        take_profit = entry_price + (2.5 * atr)
        
        signals.append({
            'symbol': symbol,
            'date': current_date.strftime('%Y-%m-%d'),
            'signal': 'STRONG BUY' if score >= 0.75 else 'BUY',
            'score': round(score, 3),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'confidence': confidence,
            'support_level': round(support_level, 2),
            'strength': strength
        })
        matched += 1

print(f"\n   ✅ Matched: {matched}")
print(f"   ⚠️ Not matched: {not_matched}")

# =========================================================
# STEP 4: Save signals for PPO
# =========================================================
if len(signals) > 0:
    output_df = pd.DataFrame(signals)
    output_df = output_df.sort_values('score', ascending=False)
    
    # PPO ফরম্যাট
    ppo_df = output_df[['symbol', 'entry_price', 'stop_loss', 'take_profit', 'score']].copy()
    ppo_df.columns = ['symbol', 'buy', 'SL', 'tp', 'confidence']
    ppo_df['date'] = datetime.now().strftime('%Y-%m-%d')
    ppo_df['RRR'] = ppo_df.apply(
        lambda x: round((x['tp'] - x['buy']) / (x['buy'] - x['SL']), 2), axis=1
    )
    
    ppo_df.to_csv('./csv/trade_stock.csv', index=False)
    
    print("\n" + "="*80)
    print("📊 SIGNALS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\n✅ Total signals: {len(signals)}")
    print(f"   STRONG BUY: {len(output_df[output_df['signal'] == 'STRONG BUY'])}")
    print(f"   BUY: {len(output_df[output_df['signal'] == 'BUY'])}")
    
    print("\n🔝 TOP 10 SIGNALS:")
    print(output_df[['symbol', 'signal', 'score', 'entry_price', 'stop_loss', 'take_profit']].head(10).to_string())
    
else:
    print("\n❌ No signals generated!")
    print("\n🛠️ Creating fallback signals for PPO training...")
    
    # ফ্যালব্যাক: সাম্প্রতিক ডাটা থেকে
    latest_date = market_df['date'].max()
    recent = market_df[market_df['date'] >= latest_date - pd.Timedelta(days=7)]
    
    fallback = []
    for symbol in recent['symbol'].unique()[:20]:
        sym_data = recent[recent['symbol'] == symbol].sort_values('date')
        if len(sym_data) >= 2:
            price_change = (sym_data.iloc[-1]['close'] - sym_data.iloc[-2]['close']) / sym_data.iloc[-2]['close']
            if price_change > 0.01:  # 1% uptrend
                entry = sym_data.iloc[-1]['close']
                fallback.append({
                    'symbol': symbol,
                    'buy': round(entry, 2),
                    'SL': round(entry * 0.97, 2),
                    'tp': round(entry * 1.05, 2),
                    'confidence': round(0.5 + price_change, 2),
                    'date': latest_date.strftime('%Y-%m-%d'),
                    'RRR': round(1.67, 2)
                })
    
    if fallback:
        fallback_df = pd.DataFrame(fallback)
        fallback_df.to_csv('./csv/trade_stock.csv', index=False)
        print(f"   ✅ {len(fallback)} fallback signals created")
    else:
        # ডামি সিগন্যাল
        dummy = pd.DataFrame({
            'symbol': ['KPCL', 'AAMRANET', 'SONALIANSH', 'SALVOCHEM'],
            'buy': [100, 150, 200, 50],
            'SL': [95, 142.5, 190, 47.5],
            'tp': [110, 165, 220, 55],
            'confidence': [0.75, 0.70, 0.65, 0.60],
            'date': [datetime.now().strftime('%Y-%m-%d')] * 4,
            'RRR': [2.0, 2.0, 2.0, 2.0]
        })
        dummy.to_csv('./csv/trade_stock.csv', index=False)
        print("   ✅ Dummy signals created")

print("\n" + "="*80)
print("✅ System ready for PPO training!")
print("="*80)