import pandas as pd
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# 🔐 Load Telegram credentials
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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

            last_idx = len(group) - 1
            last_row = group.iloc[last_idx]

            # 👉 Loop from second last to top
            for i in range(last_idx - 1, -1, -1):
                prev_row = group.iloc[i]

                if (last_row.orderblock_low < prev_row.orderblock_low) and (last_row.rsi > prev_row.rsi):
                    # 🔍 Trendline check
                    start_date = prev_row.date
                    end_date = last_row.date
                    start_price = prev_row.orderblock_low
                    end_price = last_row.orderblock_low

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

                    # ✅ Check trendline condition
                    if all(symbol_mongo['close'] >= symbol_mongo['trendline']):
                        results.append(last_row)
                        break  # ☑️ First valid match only

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
