import pandas as pd
import os

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

# -----------------------------
# Filter good models
# -----------------------------
meta_df = meta_df[meta_df['auc'] >= 0.60]
good_symbols = meta_df['symbol'].unique()

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
results = []

for _, row in sr_df.iterrows():
    symbol = row['symbol']
    current_date = row['current_date']
    current_low = row['current_low']

    df_symbol = mongo_df[mongo_df['symbol'] == symbol].reset_index(drop=True)

    match_idx = df_symbol[df_symbol['date'] == current_date].index

    if len(match_idx) == 0:
        continue

    idx = match_idx[0]

    if idx + 1 >= len(df_symbol):
        continue

    next_row = df_symbol.iloc[idx + 1]

    # ✅ Support condition
    if next_row['low'] > current_low:

        xgb_match = xgb_df[
            (xgb_df['symbol'] == symbol) &
            (xgb_df['date'] == current_date)
        ]

        if len(xgb_match) == 0:
            continue

        prob = xgb_match.iloc[0]['prob_up']
        confidence = xgb_match.iloc[0].get('confidence_score', 0)

        # -----------------------------
        # 🧠 Score calculation
        # -----------------------------
        weight = strength_weight.get(row['strength'], 0.5)

        buy_score = ((prob * 0.6) + (confidence * 0.4)) * weight
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

# -----------------------------
# Save
# -----------------------------
output_df.to_csv(output_path, index=False)

print("✅ DONE! AI signal generated & sorted.")