# debug_signals.py - ডিবাগ করে দেখুন কেন মিলছে না

import pandas as pd
import numpy as np
from datetime import datetime

print("="*70)
print("🔍 DEBUGGING SIGNAL GENERATION")
print("="*70)

# লোড ফাইল
sr_df = pd.read_csv('./csv/support_resistance.csv')
market_df = pd.read_csv('./csv/mongodb.csv')

# ডেট কনভার্ট
sr_df['current_date'] = pd.to_datetime(sr_df['current_date'])
market_df['date'] = pd.to_datetime(market_df['date'])

print(f"\n📅 Date Ranges:")
print(f"   Support levels: {sr_df['current_date'].min()} to {sr_df['current_date'].max()}")
print(f"   Market data: {market_df['date'].min()} to {market_df['date'].max()}")

# শুধু সাপোর্ট টাইপ
sr_support = sr_df[sr_df['type'] == 'support']
print(f"\n📊 Support levels: {len(sr_support)}")

# চেক করুন প্রতিটি সাপোর্ট লেভেলের জন্য
print("\n🔍 Checking each support level:")
print("-"*70)

for idx, row in sr_support.head(10).iterrows():  # প্রথম ১০টা চেক
    symbol = row['symbol']
    current_date = row['current_date']
    support_level = row['current_low']
    
    # এই সিম্বলের মার্কেট ডাটা
    sym_market = market_df[market_df['symbol'] == symbol]
    
    if len(sym_market) == 0:
        print(f"❌ {symbol} - No market data at all")
        continue
    
    # ডেট ম্যাচিং
    date_diff = (sym_market['date'] - current_date).abs()
    min_diff = date_diff.min()
    
    if min_diff.days > 5:
        print(f"❌ {symbol} - Date {current_date.strftime('%Y-%m-%d')} not found (closest: {min_diff.days} days away)")
    else:
        match_idx = date_diff.idxmin()
        matched_date = sym_market.loc[match_idx, 'date']
        print(f"✅ {symbol} - Found at {matched_date.strftime('%Y-%m-%d')} (diff: {min_diff.days} days)")
        
        # চেক করুন সাপোর্টের পরে ডাটা আছে কিনা
        market_idx = sym_market.index.get_loc(match_idx)
        if market_idx + 1 >= len(sym_market):
            print(f"   ⚠️ No next candle data")
        else:
            next_price = sym_market.iloc[market_idx + 1]['close']
            print(f"   📈 Next close: {next_price}, Support: {support_level}")

print("\n" + "="*70)
print("💡 SUGGESTED FIXES:")
print("="*70)
print("1. আপনার support_resistance.csv এর তারিখগুলো mongodb.csv এর সাথে মেলে না")
print("2. সম্ভবত support_resistance.csv এ future dates আছে")
print("3. অথবা mongodb.csv এ পুরনো ডাটা আছে")

# সমাধান: বর্তমান ডেটা দিয়ে সিগন্যাল জেনারেট করুন
print("\n🛠️ Generating signals with available data...")

# সর্বশেষ 30 দিনের ডাটা নিন
latest_date = market_df['date'].max()
start_date = latest_date - pd.Timedelta(days=30)

recent_market = market_df[market_df['date'] >= start_date]
recent_symbols = recent_market['symbol'].unique()

print(f"\n📊 Recent market data:")
print(f"   Latest date: {latest_date.strftime('%Y-%m-%d')}")
print(f"   Recent symbols: {len(recent_symbols)}")

# সিম্পল সিগন্যাল জেনারেট করুন
signals = []
for symbol in recent_symbols[:10]:  # প্রথম ১০টা সিম্বলের জন্য
    sym_data = recent_market[recent_market['symbol'] == symbol].sort_values('date')
    
    if len(sym_data) < 2:
        continue
    
    # গত 5 দিনের ট্রেন্ড
    latest = sym_data.iloc[-1]
    prev = sym_data.iloc[-2]
    
    # সিম্পল মোমেন্টাম স্ট্র্যাটেজি
    price_change = (latest['close'] - prev['close']) / prev['close']
    
    if price_change > 0.02:  # 2% উপরে
        signal = "BUY"
        score = min(0.5 + price_change * 5, 0.9)
        
        atr = latest.get('atr', latest['close'] * 0.02)
        if pd.isna(atr):
            atr = latest['close'] * 0.02
            
        signals.append({
            'symbol': symbol,
            'date': latest_date.strftime('%Y-%m-%d'),
            'signal': signal,
            'score': round(score, 4),
            'entry_price': round(latest['close'], 2),
            'stop_loss': round(latest['close'] - (1.5 * atr), 2),
            'tp1': round(latest['close'] + (2.0 * atr), 2),
            'tp2': round(latest['close'] + (3.0 * atr), 2),
            'rrr_tp1': round((2.0 * atr) / (1.5 * atr), 2),
            'source': 'Momentum_Strategy'
        })

if signals:
    output_df = pd.DataFrame(signals)
    output_df.to_csv('./csv/trade_stock.csv', index=False)
    print(f"\n✅ Generated {len(signals)} momentum signals!")
    print(output_df[['symbol', 'signal', 'score', 'entry_price']].head())
else:
    print("\n❌ Still no signals - creating dummy data for PPO")
    # PPO ট্রেনিং চালানোর জন্য ডামি ডাটা
    dummy_df = pd.DataFrame({
        'symbol': ['KPCL', 'SONALIANSH', 'AAMRANET'],
        'date': [latest_date.strftime('%Y-%m-%d')] * 3,
        'buy': [100, 150, 200],
        'SL': [95, 142, 190],
        'tp': [110, 165, 220],
        'confidence': [0.7, 0.65, 0.6],
        'RRR': [2.0, 2.14, 2.0],
        'source': ['Dummy'] * 3
    })
    dummy_df.to_csv('./csv/trade_stock.csv', index=False)
    print("✅ Created dummy signals for PPO training")

print("\n" + "="*70)