import pandas as pd
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# টেলিগ্রাম কনফিগারেশন
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN_TRADE")
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID_TRADE')

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

def rsiDivergenceFromUpToDown():
    os.makedirs('./csv/swing/rsi_divergences/', exist_ok=True)

    try:
        # 📥 ফাইল লোড
        df = pd.read_csv('./csv/swing/up_to_down.csv')
        df['date'] = pd.to_datetime(df['date'])
        df['orderblock_date'] = pd.to_datetime(df['orderblock_date'])
        df['orderblock_low'] = pd.to_numeric(df['orderblock_low'], errors='coerce')

        # 📥 RSI তুলতে mongodb.csv থেকে
        df_mongo = pd.read_csv('./csv/mongodb.csv', usecols=['symbol', 'date', 'rsi'])
        df_mongo['date'] = pd.to_datetime(df_mongo['date'])
        df_mongo['rsi'] = pd.to_numeric(df_mongo['rsi'], errors='coerce')

        # 🔗 Merge RSI from orderblock_date
        df = df.merge(
            df_mongo.rename(columns={'date': 'orderblock_date', 'rsi': 'rsi_ob_date'}),
            on=['symbol', 'orderblock_date'],
            how='left'
        )

        missing_rsi_count = df['rsi_ob_date'].isna().sum()
        print(f"ℹ️ RSI not found for {missing_rsi_count} rows based on orderblock_date.")

        # 🔍 Sort & prepare for analysis
        df = df.sort_values(by=['symbol', 'date']).reset_index(drop=True)
        divergences = []

        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].sort_values(by='date').reset_index(drop=True)

            for i in range(1, len(symbol_df)):
                current = symbol_df.iloc[i]
                previous = symbol_df.iloc[i - 1]

                # 📊 Divergence check using OB Low and orderblock_date RSI
                if (
                    pd.notna(previous['orderblock_low']) and pd.notna(current['orderblock_low']) and
                    pd.notna(previous['rsi_ob_date']) and pd.notna(current['rsi_ob_date']) and
                    current['orderblock_low'] < previous['orderblock_low'] and
                    current['rsi_ob_date'] > previous['rsi_ob_date']
                ):
                    divergences.append({
                        'symbol': symbol,
                        'prev_date': previous['orderblock_date'].strftime('%Y-%m-%d'),
                        'curr_date': current['orderblock_date'].strftime('%Y-%m-%d'),
                        'prev_ob_low': previous['orderblock_low'],
                        'curr_ob_low': current['orderblock_low'],
                        'prev_rsi': previous['rsi_ob_date'],
                        'curr_rsi': current['rsi_ob_date']
                    })

        today = datetime.now().strftime('%Y-%m-%d')

        # 📤 রিপোর্ট তৈরি ও পাঠানো
        if divergences:
            message = f"<b>📈 RSI Bullish Divergence (Orderblock Based) - {today}</b>\n\n"
            for item in divergences:
                message += (
                    f"<b>📌 {item['symbol']}</b>\n"
                    f"🔹 আগের OB Date: {item['prev_date']}, OB Low: {item['prev_ob_low']}, RSI: {item['prev_rsi']:.2f}\n"
                    f"🔹 বর্তমান OB Date: {item['curr_date']}, OB Low: {item['curr_ob_low']}, RSI: {item['curr_rsi']:.2f}\n\n"
                )

            # ✅ Save to CSV
            output_df = pd.DataFrame(divergences)
            output_file = f'./csv/swing/rsi_divergences.csv'
            output_df.to_csv(output_file, index=False)

            send_telegram_message(message)
            send_telegram_message(f"✅ রিপোর্ট সেভ হয়েছে: {output_file}")
        else:
            send_telegram_message(f"ℹ️ {today} তারিখে কোনো RSI divergence (orderblock ভিত্তিক) পাওয়া যায়নি।")

    except Exception as e:
        error_msg = f"❌ ডেটা প্রসেসিংয়ে সমস্যা: {str(e)}"
        print(error_msg)
        send_telegram_message(error_msg)

# ▶️ Run the function
rsiDivergenceFromUpToDown()
