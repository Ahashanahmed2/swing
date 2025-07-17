# scripts/send_email.py

import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_TO = os.getenv("EMAIL_TO")
EMAIL_PASS = os.getenv("EMAIL_PASS")

# Directory containing PDFs
PDF_DIR = "output/ai_signal/pdfs"

def send_email():
    msg = EmailMessage()
    msg['Subject'] = "📈 Daily Stock Signal Report"
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_TO

    msg.set_content(
        """\
Hello,

Please find attached today's AI-generated stock signal report(s).
This report includes trend direction and confidence for selected DSE stocks.

Best regards,

AI Signal Bot
"""
    )

    # Attach all PDF files from the directory
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, filename)
            with open(pdf_path, "rb") as f:
                file_data = f.read()
            msg.add_attachment(file_data, maintype="application", subtype="pdf", filename=filename)

    # Send the email
    with smtplib.SMTP_SSL("smtp.mail.yahoo.com", 465) as smtp:
        smtp.login(EMAIL_USER, EMAIL_PASS)
        smtp.send_message(msg)

    print("✅ Email sent successfully with all PDFs!")

if __name__ == "__main__":
    send_email()
