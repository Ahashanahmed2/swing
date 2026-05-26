import pandas as pd
import os
from datetime import datetime

# -----------------------------
# Paths
# -----------------------------
sr_df = './csv/support_resistance.csv'
mongo_df = './csv/mongodb.csv'
pred_df = './csv/prediction_log.csv'
conf_df = './csv/xgb_confidence.csv'
meta_df = './csv/model_metadata.csv'

output_path = './output/ai_signal/support_resistant.csv'
os.makedirs('./output/ai_signal', exist_ok=True)

# -----------------------------
# Load CSV
# -----------------------------
sr_df = pd.read_csv(sr_df)
mongo_df = pd.read_csv(mongo_df)
pred_df = pd.read_csv(pred_df)
conf_df = pd.read_csv(conf_df)
meta_df = pd.read_csv(meta_df)

# -----------------------------
# Date convert
# -----------------------------
sr_df['current_date'] = pd.to_datetime(sr_df['current_date'], format='mixed', errors='coerce')
mongo_df['date'] = pd.to_datetime(mongo_df['date'], format='mixed', errors='coerce')
pred_df['date'] = pd.to_datetime(pred_df['date'], format='mixed', errors='coerce')
conf_df['date'] = pd.to_datetime(conf_df['date'], format='mixed', errors='coerce')

# -----------------------------
# Filter only support
# -----------------------------
sr_df = sr_df[sr_df['type'] == 'support']

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
counter = 1

for _, row in sr_df.iterrows():
    symbol = row['symbol']
    current_date = row['current_date']
    current_low = row['current_low']
    gap_days = row['gap_days']
    strength = row['strength']

    # Get mongodb data for this symbol
    df_symbol = mongo_df[mongo_df['symbol'] == symbol].reset_index(drop=True)

    if len(df_symbol) == 0:
        skipped += 1
        continue

    # Find latest row where date > current_date
    future_rows = df_symbol[df_symbol['date'] > current_date]

    if len(future_rows) == 0:
        skipped += 1
        continue

    # Get the latest row (closest to current_date but after it)
    latest_row = future_rows.iloc[0]

    # Check support condition: latest low > current_low
    if latest_row['low'] <= current_low:
        skipped += 1
        continue

    last_high = latest_row['high']

    # Get XGB prediction for this symbol and date
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
    weight = strength_weight.get(strength, 0.5)

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
        'no': counter,
        'symbol': symbol,
        'high': last_high,
        'gape': gap_days,
        'xgb_prob': round(prob, 2),
        'confidence': round(confidence, 2),
        'buy_score': round(buy_score, 2),
        'sell_score': round(sell_score, 2),
        'signal': signal
    })
    
    counter += 1

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
        by=['buy_score', 'confidence', 'gape'],
        ascending=[False, False, True]
    )
    
    # Reset index and update serial no
    output_df = output_df.reset_index(drop=True)
    output_df['no'] = output_df.index + 1

    # Save
    output_df.to_csv(output_path, index=False)

# -----------------------------
# Print summary
# -----------------------------
print("\n" + "="*70)
print("📊 SIGNAL SUMMARY")
print("="*70)
if not output_df.empty:
    print(output_df['signal'].value_counts().to_string())
    # "TOP 10 SIGNALS" প্রিন্ট করার লজিক এখান থেকে সরিয়ে দেওয়া হয়েছে
else:
    print("❌ No signals generated!")

print("\n" + "="*70)
print(f"✅ Output saved to: {output_path}")
print("="*70)
