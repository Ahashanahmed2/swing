import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN_TRADE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID_TRADE")

PDF_FOLDER = "./output/ai_signal/pdfs"

def send_telegram_pdf(pdf_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF ফাইলটি নেই: {pdf_path}")
        return None

    try:
        with open(pdf_path, 'rb') as f:
            files = {'document': f}
            data = {'chat_id': TELEGRAM_CHAT_ID}
            response = requests.post(url, data=data, files=files)
            if response.status_code != 200:
                print(f"❌ Failed to send {pdf_path}: {response.text}")
            else:
                print(f"✅ Sent: {os.path.basename(pdf_path)}")
            return response.json()
    except Exception as e:
        print(f"⚠️ Error sending {pdf_path}: {e}")
        return None

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"⚠️ Failed to send message: {response.text}")
    except Exception as e:
        print(f"⚠️ Error sending message: {e}")

# Prepare message
message = "<b>📁 PDF Reports Sent:</b>\n\n"

if os.path.exists(PDF_FOLDER):
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    if pdf_files:
        print(f"pdf_folder:{PDF_FOLDER},\n pdf_file:{pdf_files}")
        for pdf_file in pdf_files:
            pdf_path = os.path.abspath(os.path.join(PDF_FOLDER, pdf_file))
            send_telegram_pdf(pdf_path)
            message += f"📄 {pdf_file}\n"
    else:
        message += "⚠️ No PDF files found to send."
else:
    message += "❌ PDF folder not found."

# Send summary message
send_telegram_message(message)

print("✅ Finished sending all available PDFs.")
