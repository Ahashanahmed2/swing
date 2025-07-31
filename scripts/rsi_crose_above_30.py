import pandas as pd
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load env variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN_TRADE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID_TRADED")

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

def detect_rsi_cross_above_30():
    try:
        df = pd.read_csv('./csv/mongodb.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['symbol', 'date'])

        cross_list = []

        for symbol, group in df.groupby('symbol'):
            group = group.sort_values(by='date').reset_index(drop=True)

            if len(group) < 2:
                continue

            prev_row = group.iloc[-2]
            curr_row = group.iloc[-1]

            # RSI cross above 30 detection
            if prev_row['rsi'] <= 30 and curr_row['rsi'] > 30:
                cross_list.append({
                    "symbol": symbol,
                    "date": curr_row['date'],
                    "rsi": curr_row['rsi'],
                    "close": curr_row['close']
                })

        today = datetime.now().strftime('%Y-%m-%d')

        if cross_list:
            message = f"<b>📈 RSI Cross Above 30 Detected - {today}</b>\n\n"
            for item in cross_list:
                message += (
                    f"<b>📌 {item['symbol']}</b>\n"
                    f"📅 তারিখ: {item['date'].strftime('%Y-%m-%d')}\n"
                    f"💹 RSI: {item['rsi']:.2f}, 📉 Close: {item['close']}\n\n"
                )
            send_telegram_message(message)
        else:
            send_telegram_message(f"ℹ️ {today} তারিখে কোনো RSI > 30 cross পাওয়া যায়নি।")

    except Exception as e:
        error_msg = f"❌ ত্রুটি: {str(e)}"
        print(error_msg)
        send_telegram_message(error_msg)

# ▶️ Run
detect_rsi_cross_above_30()
