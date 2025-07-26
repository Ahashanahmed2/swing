import pandas as pd
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# 🔐 লোড করুন .env থেকে Telegram Token ও Chat ID
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN_TRADE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID_TRADE")

# 🔗 Telegram মেসেজ পাঠানোর ফাংশন
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        response = requests.post(url, json=payload)
        return response.json()
    except Exception as e:
        print(f"Telegram message sending failed: {str(e)}")
        return None

# 🔍 RSI Divergence চেক এবং রিপোর্ট পাঠানো
def check_rsi_divergence_and_send():
    try:
<<<<<<< HEAD
        # 📥 CSV লোড করুন
        df = pd.read_csv('./csv/swing/down_to_up.csv')

        # ডেটা প্রস্তুত করুন
=======
        # 📥 দুইটি CSV লোড করে source কলাম যোগ করুন
        df1 = pd.read_csv('./csv/swing/down_to_up.csv')
        df1['source'] = 'down_to_up'

        df2 = pd.read_csv('./csv/swing/up_to_down.csv')
        df2['source'] = 'up_to_down'

        # ➕ একত্রিত করুন
        df = pd.concat([df1, df2], ignore_index=True)

        # ডেটা প্রস্তুত
>>>>>>> 02b7957 (last update 26-07-2025)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['symbol', 'date'])

        results = []

        for symbol, group in df.groupby('symbol'):
<<<<<<< HEAD
            group = group.reset_index(drop=True)
            if len(group) < 2:
                continue

            last_row = group.iloc[-1]
            prev_row = group.iloc[-2]

            # ✅ শর্ত: orderblock_low কম, RSI বেশি
            if (last_row['orderblock_low'] < prev_row['orderblock_low']) and (last_row['rsi'] > prev_row['rsi']):
                results.append(last_row)
=======
            group = group.sort_values(by='date').reset_index(drop=True)
            if len(group) < 2:
                continue

            last_row = group.iloc[-1]  # সবশেষ row
            previous_rows = group.iloc[:-1]  # আগের সব row

            for prev_row in previous_rows.itertuples():
                if (last_row.orderblock_low < prev_row.orderblock_low) and (last_row.rsi > prev_row.rsi):
                    results.append(last_row)
                    break  # যেকোনো একটায় মিললে যথেষ্ট
>>>>>>> 02b7957 (last update 26-07-2025)

        today = datetime.now().strftime('%Y-%m-%d')

        # ✅ রেজাল্ট থাকলে CSV সেভ এবং Telegram-এ পাঠান
        if results:
            output_df = pd.DataFrame(results)
<<<<<<< HEAD
=======
            output_df = output_df.sort_values(by='date', ascending=False)
>>>>>>> 02b7957 (last update 26-07-2025)
            output_path = './csv/swing/rsi_divergences/rsi_divergences.csv'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_df.to_csv(output_path, index=False)

            # 📨 টেলিগ্রাম মেসেজ তৈরি
            message = f"<b>📉 RSI Bearish Divergence (Swing OB Based) - {today}</b>\n\n"
            for row in output_df.itertuples():
                message += (
<<<<<<< HEAD
                    f"<b>📌 {row.symbol}</b>\n"
=======
                    f"<b>📌 {row.symbol}</b> ({row.source})\n"
>>>>>>> 02b7957 (last update 26-07-2025)
                    f"📅 তারিখ: {row.date.strftime('%Y-%m-%d')}\n"
                    f"🔻 OB Low: {row.orderblock_low}, RSI: {row.rsi:.2f}\n\n"
                )

            send_telegram_message(message)
            send_telegram_message(f"✅ রিপোর্ট সেভ হয়েছে: {output_path}")
        else:
            send_telegram_message(f"ℹ️ {today} তারিখে কোনো RSI divergence পাওয়া যায়নি।")

    except Exception as e:
        error_msg = f"❌ প্রসেসিং中 ত্রুটি: {str(e)}"
        print(error_msg)
        send_telegram_message(error_msg)

# ▶️ ফাংশন চালান
check_rsi_divergence_and_send()
