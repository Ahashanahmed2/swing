import os
import requests
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN_TRADE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID_TRADE")

PDF_FOLDER = "./output/ai_signal/pdfs"


def send_telegram_pdf(pdf_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    try:
        with open(pdf_path, 'rb') as f:
            files = {'document': f}
            data = {'chat_id': TELEGRAM_CHAT_ID}
            response = requests.post(url, data=data, files=files)
            if response.status_code != 200:
                print(f"Failed to send {pdf_path}: {response.text}")
            return response.json()
    except Exception as e:
        print(f"Error sending {pdf_path}: {e}")
        return None


def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    requests.post(url, data=payload)


# Prepare message
message = "<b>📁 PDF Reports Sent:</b>\n\n"

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]

if pdf_files:
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        print(f"Sending {pdf_file}...")
        send_telegram_pdf(pdf_path)
        message += f"📄 {pdf_file}\n"
else:
    message += "No PDFs found to send."

# Send summary message
send_telegram_message(message)

print("✅ All PDFs sent successfully.")