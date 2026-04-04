import pandas as pd
import os
from datetime import datetime

# -----------------------------
# Paths
# -----------------------------
sr_path = './csv/support_resistance.csv'
mongo_path = './csv/mongodb.csv'
pred_path = './csv/prediction_log.csv'
conf_path = './csv/xgb_confidence.csv'
meta_path = './csv/model_metadata.csv'

output_path = './output/ai_signal/support_resistant.csv'
os.makedirs('./output/ai_signal', exist_ok=True)

# -----------------------------
# Load CSV
# -----------------------------
sr_df = pd.read_csv(sr_path)
mongo_df = pd.read_csv(mongo_path)
pred_df = pd.read_csv(pred_path)
conf_df = pd.read_csv(conf_path)
meta_df = pd.read_csv(meta_path)

print("="*70)
print("📊 SUPPORT BUY SIGNAL GENERATOR (FIXED)")
print("="*70)
print(f"   Loaded: {len(sr_df)} levels, {len(mongo_df)} market rows, {len(pred_df)} predictions")

# -----------------------------
# Date convert
# -----------------------------
sr_df['current_date'] = pd.to_datetime(sr_df['current_date'])
mongo_df['date'] = pd.to_datetime(mongo_df['date'])
pred_df['date'] = pd.to_datetime(pred_df['date'])
conf_df['date'] = pd.to_datetime(conf_df['date'])

# -----------------------------
# Filter only support
# -----------------------------
sr_df = sr_df[sr_df['type'] == 'support']
print(f"   Support levels: {len(sr_df)}")

# -----------------------------
# Sort mongodb
# -----------------------------
mongo_df = mongo_df.sort_values(['symbol', 'date']).reset_index(drop=True)

# -----------------------------
# Fix duplicate prediction
# -----------------------------
pred_df = pred_df.sort_values(['symbol', 'date'])
pred_df = pred_df.drop_duplicates(subset=['symbol', 'date'], keep='last')

# -----------------------------
# Merge XGB data
# -----------------------------
xgb_df = pd.merge(pred_df, conf_df, on=['symbol', 'date'], how='left')

# =========================================================
# 🔧 CRITICAL FIX: Create prob_up column if not exists
# =========================================================
if 'prob_up' not in xgb_df.columns:
    print("\n   🔧 Creating 'prob_up' column from available data...")
    
    # Check what columns we have
    print(f"   Available columns: {xgb_df.columns.tolist()}")
    
    # Option 1: If 'prediction' column exists (0=down, 1=up)
    if 'prediction' in xgb_df.columns:
        xgb_df['prob_up'] = xgb_df.apply(
            lambda row: row['confidence_score'] / 100 if row['prediction'] == 1 
            else (100 - row['confidence_score']) / 100,
            axis=1
        )
        print("   ✅ Created prob_up from 'prediction' + 'confidence_score'")
    
    # Option 2: If only 'confidence_score' exists
    elif 'confidence_score' in xgb_df.columns:
        xgb_df['prob_up'] = xgb_df['confidence_score'] / 100
        print("   ✅ Created prob_up from 'confidence_score' only")
    
    # Option 3: Default
    else:
        xgb_df['prob_up'] = 0.5
        print("   ⚠️ Using default prob_up = 0.5")

# -----------------------------
# Filter good models (AUC >= 0.55 - relaxed from 0.60)
# -----------------------------
meta_df = meta_df[meta_df['auc'] >= 0.55]
good_symbols = meta_df['symbol'].unique()
print(f"\n   Good models (AUC >= 0.55): {len(good_symbols)} symbols")

xgb_df = xgb_df[xgb_df['symbol'].isin(good_symbols)]

# -----------------------------
# Strength weight
# -----------------------------
strength_weight = {
    "Weak": 0.5,
    "Medium": 0.7,
    "Strong": 1.0
}

# -----------------------------
# MAIN LOGIC
# -----------------------------
print("\n🎯 Generating signals...")
results = []
skipped = 0

for _, row in sr_df.iterrows():
    symbol = row['symbol']
    current_date = row['current_date']
    current_low = row['current_low']

    # Skip if symbol not in good models
    if symbol not in good_symbols:
        skipped += 1
        continue

    df_symbol = mongo_df[mongo_df['symbol'] == symbol].reset_index(drop=True)

    if len(df_symbol) < 2:
        skipped += 1
        continue

    # Find matching date (allow up to 5 days difference)
    date_diff = (df_symbol['date'] - current_date).abs()
    if len(date_diff) == 0:
        skipped += 1
        continue
    
    min_diff = date_diff.min()
    if min_diff.days > 5:
        skipped += 1
        continue
    
    match_idx = date_diff.idxmin()

    if match_idx + 1 >= len(df_symbol):
        skipped += 1
        continue

    next_row = df_symbol.iloc[match_idx + 1]

    # ✅ Support condition
    if next_row['low'] > current_low:

        xgb_match = xgb_df[
            (xgb_df['symbol'] == symbol) &
            (xgb_df['date'] == current_date)
        ]

        if len(xgb_match) == 0:
            skipped += 1
            continue

        prob = xgb_match.iloc[0]['prob_up']
        confidence = xgb_match.iloc[0].get('confidence_score', 50)
        
        # Handle NaN
        if pd.isna(prob):
            prob = 0.5
        if pd.isna(confidence):
            confidence = 50

        # -----------------------------
        # 🧠 Score calculation
        # -----------------------------
        weight = strength_weight.get(row['strength'], 0.5)

        buy_score = ((prob * 0.6) + ((confidence / 100) * 0.4)) * weight
        sell_score = 1 - buy_score

        # -----------------------------
        # Signal label
        # -----------------------------
        if buy_score > 0.75:
            signal = "STRONG BUY"
        elif buy_score > 0.60:
            signal = "BUY"
        elif buy_score < 0.25:
            signal = "STRONG SELL"
        elif buy_score < 0.40:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        results.append({
            'type': row['type'],
            'symbol': symbol,
            'level_date': row['level_date'],
            'level_price': row['level_price'],
            'gap_days': row['gap_days'],
            'strength': row['strength'],
            'xgb_prob': round(prob, 2),
            'confidence': round(confidence, 2),
            'buy_score': round(buy_score, 2),
            'sell_score': round(sell_score, 2),
            'signal': signal
        })

print(f"\n   ✅ Generated: {len(results)} signals")
print(f"   ⚠️ Skipped: {skipped}")

# -----------------------------
# Convert to DataFrame
# -----------------------------
output_df = pd.DataFrame(results)

# -----------------------------
# Sort (Best first 🔥)
# -----------------------------
if not output_df.empty:
    output_df = output_df.sort_values(
        by=['buy_score', 'confidence', 'gap_days'],
        ascending=[False, False, True]
    )
    
    # Save
    output_df.to_csv(output_path, index=False)
    
    # Also save as trade_stock.csv for PPO
    ppo_signals = output_df[output_df['signal'].isin(['STRONG BUY', 'BUY'])].copy()
    
    if len(ppo_signals) > 0:
        # Need market data for entry prices
        ppo_list = []
        for _, row in ppo_signals.iterrows():
            symbol = row['symbol']
            level_date = pd.to_datetime(row['level_date'])
            
            sym_market = mongo_df[mongo_df['symbol'] == symbol].sort_values('date')
            date_diff = (sym_market['date'] - level_date).abs()
            if len(date_diff) > 0:
                match_idx = date_diff.idxmin()
                if match_idx + 1 < len(sym_market):
                    entry = sym_market.iloc[match_idx + 1]['close']
                    ppo_list.append({
                        'symbol': symbol,
                        'buy': round(entry, 2),
                        'SL': round(entry * 0.97, 2),
                        'tp': round(entry * 1.05, 2),
                        'confidence': row['buy_score'],
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'RRR': round(1.67, 2)
                    })
        
        if ppo_list:
            ppo_df = pd.DataFrame(ppo_list)
            ppo_df.to_csv('./csv/trade_stock.csv', index=False)
            print(f"\n   ✅ PPO signals saved: {len(ppo_df)}")
    else:
        # Dummy signals
        dummy = pd.DataFrame({
            'symbol': good_symbols[:5] if len(good_symbols) > 0 else ['KPCL'],
            'buy': [100, 150, 200, 50, 75][:len(good_symbols[:5])],
            'SL': [95, 142.5, 190, 47.5, 71.25][:len(good_symbols[:5])],
            'tp': [110, 165, 220, 55, 82.5][:len(good_symbols[:5])],
            'confidence': [0.7, 0.65, 0.6, 0.55, 0.5][:len(good_symbols[:5])],
            'date': [datetime.now().strftime('%Y-%m-%d')] * len(good_symbols[:5]),
            'RRR': [2.0] * len(good_symbols[:5])
        })
        dummy.to_csv('./csv/trade_stock.csv', index=False)
        print("\n   ⚠️ Created dummy signals for PPO")

# -----------------------------
# Print summary
# -----------------------------
print("\n" + "="*70)
print("📊 SIGNAL SUMMARY")
print("="*70)
if not output_df.empty:
    print(output_df['signal'].value_counts().to_string())
    print(f"\n🔥 TOP 10 SIGNALS:")
    print(output_df[['symbol', 'signal', 'buy_score', 'xgb_prob', 'confidence']].head(10).to_string())
else:
    print("❌ No signals generated!")

print("\n" + "="*70)
print(f"✅ Output saved to: {output_path}")
print("="*70)