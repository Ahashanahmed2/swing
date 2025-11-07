#!/usr/bin/env python3
"""
scripts/send_email.py
GitHub Actions-এ চালানোর জন্য।
PDF-গুলো $GITHUB_WORKSPACE/output/ai_signal/pdfs থেকে নিয়ে Yahoo SMTP দিয়ে মেইল করে।
"""

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import List

# ---------- ENV ----------
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO   = os.getenv("EMAIL_TO")

missing = [k for k, v in {
    "EMAIL_USER": EMAIL_USER,
    "EMAIL_PASS": EMAIL_PASS,
    "EMAIL_TO":   EMAIL_TO,
}.items() if not v]
if missing:
    raise SystemExit(f"❌ Missing secrets: {', '.join(missing)}")

# ---------- PATH ----------
PDF_FOLDER = Path(os.environ["GITHUB_WORKSPACE"]) / "output" / "ai_signal" / "pdfs"
PDF_FOLDER.mkdir(parents=True, exist_ok=True)   # যদি না-ই থাকে

def get_pdfs(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".pdf")

# ---------- MAIL ----------
def send_email() -> None:
    pdfs = get_pdfs(PDF_FOLDER)
    if not pdfs:
        print("⚠️  No PDF found; mail will have no attachment.")

    msg = EmailMessage()
    msg["Subject"] = "📈 Daily Stock Signal Report"
    msg["From"]    = EMAIL_USER
    msg["To"]      = EMAIL_TO
    msg.set_content(
        "Hello,\n\n"
        "Please find attached today's AI-generated stock signal report(s).\n\n"
        "Best regards,\nAI Signal Bot"
    )

    for pdf in pdfs:
        if pdf.stat().st_size > 25 * 1024 * 1024:   # 25 MB
            print(f"⚠️  Skipping large file: {pdf.name}")
            continue
        msg.add_attachment(
            pdf.read_bytes(),
            maintype="application",
            subtype="pdf",
            filename=pdf.name,
        )

    with smtplib.SMTP_SSL("smtp.mail.yahoo.com", 465, timeout=20) as smtp:
        smtp.login(EMAIL_USER, EMAIL_PASS)
        smtp.send_message(msg)

    print("✅ Email sent successfully!")

if __name__ == "__main__":
    send_email()
