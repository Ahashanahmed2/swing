import pandas as pd
import os
from datetime import datetime

# -----------------------------
# Paths
# -----------------------------
sr_df_path = './csv/support_resistance.csv'
mongo_df_path = './csv/mongodb.csv'
pred_df_path = './csv/prediction_log.csv'
conf_df_path = './csv/xgb_confidence.csv'

output_path = './output/ai_signal/support_resistant.csv'
os.makedirs('./output/ai_signal', exist_ok=True)

# -----------------------------
# Load CSV
# -----------------------------
sr_df = pd.read_csv(sr_df_path)
mongo_df = pd.read_csv(mongo_df_path)

# -----------------------------
# Load prediction files (optional)
# -----------------------------
xgb_df = None
try:
    if os.path.exists(pred_df_path):
        pred_df = pd.read_csv(pred_df_path)
        if os.path.exists(conf_df_path):
            conf_df = pd.read_csv(conf_df_path)
            xgb_df = pd.merge(pred_df, conf_df, on=['symbol', 'date'], how='left')
        else:
            xgb_df = pred_df.copy()
        print("✅ Loaded XGB data")
    else:
        print("⚠️ No XGB data - continuing without predictions")
except Exception as e:
    print(f"⚠️ Error: {e} - continuing without predictions")

# -----------------------------
# Date convert
# -----------------------------
sr_df['current_date'] = pd.to_datetime(sr_df['current_date'], format='mixed', errors='coerce')
mongo_df['date'] = pd.to_datetime(mongo_df['date'], format='mixed', errors='coerce')
if xgb_df is not None:
    xgb_df['date'] = pd.to_datetime(xgb_df['date'], format='mixed', errors='coerce')
    
    # Create prob_up column
    if 'prob_up' not in xgb_df.columns:
        if 'prediction' in xgb_df.columns and 'confidence_score' in xgb_df.columns:
            xgb_df['prob_up'] = xgb_df.apply(
                lambda row: row['confidence_score'] / 100 if row['prediction'] == 1 
                else (100 - row['confidence_score']) / 100, axis=1
            )
        elif 'confidence_score' in xgb_df.columns:
            xgb_df['prob_up'] = xgb_df['confidence_score'] / 100
        else:
            xgb_df['prob_up'] = 0.5

# -----------------------------
# Sort mongodb
# -----------------------------
mongo_df = mongo_df.sort_values(['symbol', 'date']).reset_index(drop=True)

# -----------------------------
# Strength weight
# -----------------------------
strength_weight = {
    "Weak": 0.5,
    "Moderate": 0.7,
    "Strong": 1.0
}

# -----------------------------
# MAIN LOGIC
# -----------------------------
print("\n🎯 Generating signals...")
results = []
skipped = 0
counter = 1

# Create XGB lookup if available
xgb_lookup = {}
if xgb_df is not None:
    for _, row in xgb_df.iterrows():
        sym = row['symbol']
        dt = row['date']
        if sym not in xgb_lookup:
            xgb_lookup[sym] = {}
        xgb_lookup[sym][dt] = row

for _, row in sr_df.iterrows():
    symbol = row['symbol']
    current_date = row['current_date']
    current_low = row['current_low']
    level_price = row['level_price']
    gap_days = row['gap_days']
    strength = row['strength']
    sr_type = row['type']
    
    if pd.isna(current_date):
        skipped += 1
        continue
    
    # Get future data
    df_symbol = mongo_df[mongo_df['symbol'] == symbol].reset_index(drop=True)
    if len(df_symbol) == 0:
        skipped += 1
        continue
    
    future_rows = df_symbol[df_symbol['date'] > current_date]
    if len(future_rows) == 0:
        skipped += 1
        continue
    
    latest_row = future_rows.iloc[0]
    
    # Check condition
    condition_met = False
    if sr_type == 'support':
        if latest_row['low'] > current_low:
            condition_met = True
    else:  # resistance
        if latest_row['high'] > level_price:
            condition_met = True
    
    if not condition_met:
        skipped += 1
        continue
    
    last_high = latest_row['high']
    
    # Get XGB prediction (optional)
    prob = 0.5
    confidence = 50
    has_xgb = False
    
    if xgb_lookup and symbol in xgb_lookup:
        if current_date in xgb_lookup[symbol]:
            match = xgb_lookup[symbol][current_date]
            prob = match.get('prob_up', 0.5)
            confidence = match.get('confidence_score', 50)
            has_xgb = True
    
    if pd.isna(prob):
        prob = 0.5
    if pd.isna(confidence):
        confidence = 50
    
    # Calculate score
    weight = strength_weight.get(strength, 0.5)
    
    if has_xgb:
        buy_score = ((prob * 0.6) + ((confidence / 100) * 0.4)) * weight
    else:
        buy_score = weight
    
    sell_score = 1 - buy_score
    
    # Signal label
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
    
    # Save result (only basic columns)
    results.append({
        'no': counter,
        'symbol': symbol,
        'high': round(last_high, 2),
        'gape': gap_days,
        'xgb_prob': round(prob, 2) if has_xgb else None,
        'confidence': round(confidence, 2) if has_xgb else None,
        'buy_score': round(buy_score, 2),
        'sell_score': round(sell_score, 2),
        'signal': signal
    })
    
    counter += 1

print(f"\n✅ Generated: {len(results)} signals")
print(f"⚠️ Skipped: {skipped}")

# -----------------------------
# Save output
# -----------------------------
output_df = pd.DataFrame(results)

if not output_df.empty:
    output_df = output_df.sort_values(
        by=['buy_score', 'confidence', 'gape'],
        ascending=[False, False, True]
    )
    output_df = output_df.reset_index(drop=True)
    output_df['no'] = output_df.index + 1
    
    output_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*50}")
    print("📊 SIGNAL SUMMARY")
    print(f"{'='*50}")
    print(output_df['signal'].value_counts().to_string())
    
    print(f"\n🔥 TOP 10 SIGNALS")
    print(f"{'='*50}")
    print(output_df[['no', 'symbol', 'signal', 'buy_score', 'gape']].head(10).to_string(index=False))
else:
    print("❌ No signals generated!")

print(f"\n✅ Output saved to: {output_path}")