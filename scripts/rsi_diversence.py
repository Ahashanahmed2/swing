import pandas as pd
import glob
import os
import requests  # Added missing import
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# টেলিগ্রাম বট কনফিগারেশন
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        response = requests.post(url, json=payload)  # Defined response
        return response.json()
    except Exception as e:
        print(f"Telegram message sending failed: {str(e)}")
        return None

def rsiDivergence():
    # ডিরেক্টরি তৈরি
    os.makedirs('./csv/swing/rsi_divergences/', exist_ok=True)
    
    # CSV ফাইল লোড
    target_dir = './csv/swing/swing_low/low_candle/'
    csv_files = glob.glob(os.path.join(target_dir, '*.csv'))
    
    if not csv_files:
        send_telegram_message("⚠️ কোনো CSV ফাইল পাওয়া যায়নি!")
        return

    rsi_divergences = []
    today = datetime.now().strftime('%Y-%m-%d')
    
    for file in csv_files:
        try:
            low_df = pd.read_csv(file)
            low_df['date'] = pd.to_datetime(low_df['date']).dt.strftime('%Y-%m-%d')
            
            for i in range(1, len(low_df)):
                current = low_df.iloc[i]
                previous = low_df.iloc[i-1]
                
                if (current['close'] < previous['close'] and 
                    current['rsi'] > previous['rsi'] and
                    current['date'] == today):
                    
                    rsi_divergences.append({
                        'symbol': current['symbol'],
                        'date': current['date'],
                        'close': current['close'],
                        'rsi': current['rsi'],
                        'prev_close': previous['close'],
                        'prev_rsi': previous['rsi'],
                        'change': f"{(current['close']-previous['close'])/previous['close']*100:.2f}%"
                    })
        
        except Exception as e:
            error_msg = f"Error processing {os.path.basename(file)}: {str(e)}"
            print(error_msg)
            send_telegram_message(f"⚠️ {error_msg}")
    
    if rsi_divergences:
        # HTML ফরম্যাটে মেসেজ তৈরি
        message = f"<b>📊 আজকের RSI ডাইভারজেন্স ({today})</b>\n\n"
        message += "<i>বুলিশ ডাইভারজেন্স (প্রাইস কমছে কিন্তু RSI বাড়ছে)</i>\n\n"
        
        for item in rsi_divergences:
            message += (
                f"<b>📌 {item['symbol']}</b>\n"
                f"🔹 বর্তমান ক্লোজ: {item['close']} (পূর্ববর্তী: {item['prev_close']})\n"
                f"🔹 বর্তমান RSI: {item['rsi']:.2f} (পূর্ববর্তী: {item['prev_rsi']:.2f})\n"
                f"🔹 পরিবর্তন: {item['change']}\n\n"
            )
        
        # CSV হিসেবে সেভ
        df = pd.DataFrame(rsi_divergences)
        output_file = f'./csv/swing/rsi_divergences/{today}_divergences.csv'
        df.to_csv(output_file, index=False)
        
        # টেলিগ্রামে পাঠান
        send_telegram_message(message)
        send_telegram_message(f"📁 সম্পূর্ণ রিপোর্ট CSV হিসেবে সেভ করা হয়েছে: {output_file}")
    else:
        send_telegram_message(f"ℹ️ আজকে ({today}) কোনো RSI ডাইভারজেন্স পাওয়া যায়নি।")

# ফাংশন কল
rsiDivergence()
