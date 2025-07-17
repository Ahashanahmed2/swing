# ./scripts/check_dqn_.py

import os
import subprocess
import requests
import smtplib
from email.message import EmailMessage
from datetime import datetime
from dotenv import load_dotenv
import sys
# Load environment variables
load_dotenv()

# Telegram Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN_TRADE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID_TRADE")

# Email Config
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_TO = os.getenv("EMAIL_TO")
EMAIL_PASS = os.getenv("EMAIL_PASS")

# File paths
zip_file_path = "./csv/dqn_retrained.zip"
log_dir = "./csv/logs"
log_file_path = os.path.join(log_dir, "check_dqn_log.txt")

os.makedirs(log_dir, exist_ok=True)

def timestamp():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

def write_log(message):
    with open(log_file_path, "a") as f:
        f.write(f"{timestamp()} {message}\n")
    print(message)

def send_telegram_alert(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": f"⚠️ DQN Script Alert:\n{message}",
        }
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            write_log(f"❌ টেলিগ্রাম মেসেজ ব্যর্থ: {response.text}")
    except Exception as e:
        write_log(f"❌ টেলিগ্রাম exception: {str(e)}")

def send_email_alert(subject, body):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_USER
        msg["To"] = EMAIL_TO

        with smtplib.SMTP_SSL("smtp.mail.yahoo.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
    except Exception as e:
        write_log(f"❌ ইমেইল exception: {str(e)}")
def main():
    try:
        scripts = [
             "scripts/mongodb.py",
    "scripts/swing_High_Low.py",
    "scripts/swing_point.py",
     "scripts/imbalance.py",
    "scripts/main_rsu_divergence.py",
    "scripts/rsi_diversence.py",
    'scripts/uptrand_downtrand.py',
    'scripts/trands.py',
    'scripts/signals.py',
    'scripts/error_analysis.py',
    "scripts/backtest_and_retrain.py",
    'scripts/generate_signal.py',
    'scripts/sort_signals.py',
    'scripts/filter_trends.py',
    'scripts/generate_pdf.py',
    'scripts/dayliMassage.py',
    'scripts/email_reports.py',
    
                ]

        for script in scripts:
            script_path=os.path.abspath(script)
            if os.path.exists(zip_file_path):
                result = subprocess.run([sys.executable,script_path ], capture_output=True, text=True)
            else:
                print(f"train_script.py চালানো হচ্ছে")
                scripts = [
                "main_train.py"
                ] 
    
            for script in scripts:
                script_path=os.path.abspath(script)
                result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)

        # Output log
        write_log(f"STDOUT:\n{result.stdout}")
        if result.stderr or result.returncode != 0:
            error_message = f"❌ স্ক্রিপ্ট চালানোর সময় ত্রুটি:\n{result.stderr}"
            send_telegram_alert(error_message)
            send_email_alert("🚨 DQN Script Error Alert", error_message)

    except Exception as e:
        error_message = f"❌ স্ক্রিপ্ট চালানোর সময় ত্রুটি:\n{str(e)}"
        send_telegram_alert(error_message)
        send_email_alert("🚨 DQN Script Error Alert", error_message)

if __name__ == "__main__":
    main()
