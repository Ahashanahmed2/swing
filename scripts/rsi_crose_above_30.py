import pandas as pd
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# 🔐 Load Telegram credentials
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN_TRADE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID_TRADE")

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

def filter_rsi_and_send():
    try:
        df = pd.read_csv('./csv/mongodb.csv')
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by=['symbol', 'date'], inplace=True)

        final_rows = []

        for symbol, group in df.groupby('symbol'):
            group = group.reset_index(drop=True)
            if len(group) < 2:
                continue

            last_row = group.iloc[-1]
            prev_row = group.iloc[-2]

            try:
                rsi_last = float(last_row['rsi'])
                rsi_prev = float(prev_row['rsi'])
                close = float(last_row['close'])
            except:
                continue

            if rsi_prev <= 30 and 30 < rsi_last < 60:
                final_rows.append({
                    'symbol': last_row['symbol'],
                    'date': last_row['date'].strftime('%Y-%m-%d'),
                    'close': close,
                    'rsi': rsi_last
                })

        today = datetime.now().strftime('%Y-%m-%d')

        if final_rows:
            output_df = pd.DataFrame(final_rows)
            output_df.sort_values(by='date', ascending=False, inplace=True)
            output_path = './csv/filtered_output.csv'
            output_df.to_csv(output_path, index=False)

            # 📩 Prepare Telegram message
            message = f"<b>✅ RSI Recovery Filter - {today}</b>\n\n"
            for row in output_df.itertuples():
                message += (
                    f"<b>📌 {row.symbol}</b>\n"
                    f"📅 তারিখ: {row.date}\n"
                    f"💰 Close: {row.close}, RSI: {row.rsi:.2f}\n\n"
                )

            send_telegram_message(message)
            send_telegram_message(f"✅ Filtered রিপোর্ট সেভ হয়েছে: {output_path}")
        else:
            send_telegram_message(f"ℹ️ {today} তারিখে কোনো RSI recovery মিলে নি।")

    except Exception as e:
        error_msg = f"❌ RSI Filter প্রসেসিং中 ত্রুটি: {str(e)}"
        print(error_msg)
        send_telegram_message(error_msg)

# ▶️ Run it
filter_rsi_and_send()
