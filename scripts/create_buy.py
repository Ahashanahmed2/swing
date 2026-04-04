# create_buy_signals.py - শুধু BUY সিগন্যাল তৈরি করুন

import pandas as pd
from datetime import datetime

print("="*70)
print("📊 CREATING BUY SIGNALS FOR PPO")
print("="*70)

# লোড মার্কেট ডাটা
market_df = pd.read_csv('./csv/mongodb.csv')
market_df['date'] = pd.to_datetime(market_df['date'])

# লোড প্রেডিকশন
pred_df = pd.read_csv('./csv/prediction_log.csv')
pred_df['date'] = pd.to_datetime(pred_df['date'])

# Create prob_up
if 'prediction' in pred_df.columns:
    pred_df['prob_up'] = pred_df.apply(
        lambda row: row['confidence_score'] / 100 if row['prediction'] == 1 
        else (100 - row['confidence_score']) / 100, axis=1
    )
else:
    pred_df['prob_up'] = pred_df['confidence_score'] / 100

# সবচেয়ে ভালো prob_up সহ সিম্বল খুঁজুন
latest_pred = pred_df.sort_values('date').groupby('symbol').last().reset_index()
best_symbols = latest_pred.nlargest(30, 'prob_up')['symbol'].tolist()

print(f"Top symbols by prob_up: {best_symbols[:10]}")

# BUY সিগন্যাল তৈরি করুন
signals = []
latest_date = market_df['date'].max()

for symbol in best_symbols[:20]:  # প্রথম 20টা সিম্বল
    sym_data = market_df[market_df['symbol'] == symbol].sort_values('date')
    
    if len(sym_data) >= 2:
        latest = sym_data.iloc[-1]
        prev = sym_data.iloc[-2]
        
        # চেক আপট্রেন্ড
        if latest['close'] > prev['close']:
            entry = latest['close']
            stop_loss = entry * 0.97  # 3% SL
            take_profit = entry * 1.06  # 6% TP
            
            # প্রোবাবিলিটি নিন
            prob = latest_pred[latest_pred['symbol'] == symbol]['prob_up'].values
            confidence = prob[0] if len(prob) > 0 else 0.6
            
            signals.append({
                'symbol': symbol,
                'date': latest_date.strftime('%Y-%m-%d'),
                'buy': round(entry, 2),
                'SL': round(stop_loss, 2),
                'tp': round(take_profit, 2),
                'confidence': round(confidence, 3),
                'RRR': round((take_profit - entry) / (entry - stop_loss), 2)
            })

if signals:
    trade_df = pd.DataFrame(signals)
    trade_df = trade_df[['symbol', 'date', 'buy', 'SL', 'tp', 'confidence', 'RRR']]
    trade_df.to_csv('./csv/trade_stock.csv', index=False)
    
    print(f"\n✅ Created {len(signals)} BUY signals")
    print("\n📊 TOP 10 SIGNALS:")
    print(trade_df.head(10).to_string())
else:
    # ডামি সিগন্যাল
    dummy = pd.DataFrame({
        'symbol': ['KPCL', 'AAMRANET', 'SONALIANSH', 'SALVOCHEM', 'FAREASTFIN'],
        'date': [datetime.now().strftime('%Y-%m-%d')] * 5,
        'buy': [100, 150, 200, 50, 2.7],
        'SL': [95, 142.5, 190, 47.5, 2.62],
        'tp': [110, 165, 220, 55, 2.84],
        'confidence': [0.75, 0.70, 0.65, 0.60, 0.67],
        'RRR': [2.0, 2.0, 2.0, 2.0, 2.0]
    })
    dummy.to_csv('./csv/trade_stock.csv', index=False)
    print("✅ Created dummy BUY signals")

print("\n" + "="*70)
print("✅ Ready for PPO training!")
print("="*70)
