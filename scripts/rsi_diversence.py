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
        # 📥 CSV লোড করুন
        df = pd.read_csv('./csv/swing/down_to_up.csv')

        # ডেটা প্রস্তুত করুন
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['symbol', 'date'])

        results = []

        for symbol, group in df.groupby('symbol'):
            group = group.reset_index(drop=True)
            if len(group) < 2:
                continue

            last_row = group.iloc[-1]
            prev_row = group.iloc[-2]

            # ✅ শর্ত: orderblock_low কম, RSI বেশি
            if (last_row['orderblock_low'] < prev_row['orderblock_low']) and (last_row['rsi'] > prev_row['rsi']):
                results.append(last_row)

        today = datetime.now().strftime('%Y-%m-%d')

        # ✅ রেজাল্ট থাকলে CSV সেভ এবং Telegram-এ পাঠান
        if results:
            output_df = pd.DataFrame(results)
            output_path = './csv/swing/rsi_divergences/rsi_divergences.csv'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_df.to_csv(output_path, index=False)

            # 📨 টেলিগ্রাম মেসেজ তৈরি
            message = f"<b>📉 RSI Bearish Divergence (Swing OB Based) - {today}</b>\n\n"
            for row in output_df.itertuples():
                message += (
                    f"<b>📌 {row.symbol}</b>\n"
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
