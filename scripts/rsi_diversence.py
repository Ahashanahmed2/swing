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

def check_rsi_divergence_and_send():
    try:
        df = pd.read_csv('./csv/swing/down_to_up.csv')
        df['source'] = 'down_to_up'
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['symbol', 'date'])

        mongo_df = pd.read_csv('./csv/mongodb.csv')
        mongo_df['date'] = pd.to_datetime(mongo_df['date'])

        results = []

        for symbol, group in df.groupby('symbol'):
            group = group.sort_values(by='date', ascending=True).reset_index(drop=True)

            if len(group) < 2:
                continue

            for i in range(len(group) - 1):
                for j in range(i + 1, len(group)):
                    row1 = group.iloc[i]
                    row2 = group.iloc[j]

                    # RSI divergence condition
                    if row2.orderblock_low < row1.orderblock_low and row2.rsi > row1.rsi:
                        start_date = row1.date
                        end_date = row2.date
                        start_price = row1.orderblock_low
                        end_price = row2.orderblock_low

                        days_diff = (end_date - start_date).days
                        if days_diff == 0:
                            continue

                        slope = (end_price - start_price) / days_diff

                        symbol_mongo = mongo_df[(mongo_df['symbol'] == symbol) & 
                                                (mongo_df['date'] >= start_date) & 
                                                (mongo_df['date'] <= end_date)].copy()
                        if symbol_mongo.empty:
                            continue

                        symbol_mongo['trendline'] = symbol_mongo['date'].apply(
                            lambda d: start_price + slope * (d - start_date).days
                        )

                        if all(symbol_mongo['close'] >= symbol_mongo['trendline']):
                            results.append(row2)

        today = datetime.now().strftime('%Y-%m-%d')

        if results:
            output_df = pd.DataFrame(results)
            output_df = output_df.sort_values(by='date', ascending=False)

            output_path = './csv/swing/rsi_divergences/rsi_divergences.csv'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_df.to_csv(output_path, index=False)

            message = f"<b>📉 RSI Divergence + Trendline Hold - {today}</b>\n\n"
            for row in output_df.itertuples():
                message += (
                    f"<b>📌 {row.symbol}</b> ({row.source})\n"
                    f"📅 তারিখ: {row.date.strftime('%Y-%m-%d')}\n"
                    f"🔻 OB Low: {row.orderblock_low}, RSI: {row.rsi:.2f}\n\n"
                )

            send_telegram_message(message)
            send_telegram_message(f"✅ রিপোর্ট সেভ হয়েছে: {output_path}")
        else:
            send_telegram_message(f"ℹ️ {today} তারিখে কোনো RSI divergence পাওয়া যায়নি।")

    except Exception as e:
        error_msg = f"❌ ত্রুটি: {str(e)}"
        print(error_msg)
        send_telegram_message(error_msg)

# ▶️ Run
check_rsi_divergence_and_send()
