import pandas as pd
import os

# ---------- 📥 Load AI Signals ----------
signal_path = "./csv/signals.csv"
if not os.path.exists(signal_path):
    print(f"❌ signals.csv not found at {signal_path}")
    exit()

signals = pd.read_csv(signal_path)

# ---------- 📥 Load Ground Truth (main_df) ----------
main_path = "./csv/mongodb.csv"
if not os.path.exists(main_path):
    print(f"❌ mongodb.csv not found at {main_path}")
    exit()

main_df = pd.read_csv(main_path)
main_df.fillna(0, inplace=True)

# ---------- 🔍 Analyze Accuracy by Symbol ----------
report = []
for symbol in signals['symbol'].unique():
    ai_data = signals[signals['symbol'] == symbol].reset_index(drop=True)
    true_data = main_df[main_df['symbol'] == symbol].reset_index(drop=True)

    correct = wrong = 0

    for i in range(len(ai_data)-1):
        ai_action = ai_data.loc[i, 'action']
        price_now = ai_data.loc[i, 'price']
        price_next = ai_data.loc[i+1, 'price']

        if pd.isna(price_now) or pd.isna(price_next):
            continue

        # actual market move
        moved_up = price_next > price_now
        moved_down = price_next < price_now

        if ai_action == "Buy" and moved_up:
            correct += 1
        elif ai_action == "Sell" and moved_down:
            correct += 1
        elif ai_action == "Hold" and abs(price_next - price_now) < 0.5:
            correct += 1
        else:
            wrong += 1

    total = correct + wrong
    accuracy = round((correct / total) * 100, 2) if total > 0 else 0.0

    report.append({
        'symbol': symbol,
        'ai_action':ai_action,
        'total_signals': total,
        'correct_action': correct,
        'wrong_action': wrong,
        'accuracy (%)': accuracy
    })

# ---------- 💾 Save CSV ----------
report_df = pd.DataFrame(report)
os.makedirs("./csv", exist_ok=True)
report_df.to_csv("./csv/accuracy_by_symbol.csv", index=False)
print("✅ Saved: ./csv/accuracy_by_symbol.csv")
