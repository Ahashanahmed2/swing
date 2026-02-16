#!/usr/bin/env python3
"""
scripts/send_email.py
GitHub Actions-এ চালানোর জন্য।
PDF-গুলো $GITHUB_WORKSPACE/output/ai_signal/pdfs থেকে নিয়ে
emails.txt ফাইল থেকে সব ইমেইলে পাঠায়।
"""

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import List

# ---------- ENV ----------
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

missing = [k for k, v in {
    "EMAIL_USER": EMAIL_USER,
    "EMAIL_PASS": EMAIL_PASS,
}.items() if not v]
if missing:
    raise SystemExit(f"❌ Missing secrets: {', '.join(missing)}")

# ---------- PATHS ----------
WORKSPACE = Path(os.environ.get("GITHUB_WORKSPACE", "."))
PDF_FOLDER = WORKSPACE / "output" / "ai_signal" / "pdfs"
EMAILS_FILE = WORKSPACE / "csv"/ "emails.txt"  # রিপোজিটরির রুটে থাকবে

def get_pdfs(folder: Path) -> List[Path]:
    """PDF ফাইলগুলো লিস্ট করে"""
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".pdf")

def get_email_list(file_path: Path) -> List[str]:
    """emails.txt ফাইল থেকে ইমেইল লিস্ট তৈরি করে"""
    if not file_path.exists():
        print(f"⚠️  {file_path} not found! Using EMAIL_TO from secrets.")
        # ব্যাকআপ হিসেবে EMAIL_TO ব্যবহার করব (যদি থাকে)
        backup_email = os.getenv("EMAIL_TO")
        return [backup_email] if backup_email else []
    
    emails = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # খালি লাইন, কমেন্ট, এবং ইনভ্যালিড ইমেইল বাদ দিই
            if line and not line.startswith('#') and '@' in line:
                emails.append(line)
    
    return emails

def send_email_to_all() -> None:
    """সব ইমেইলে PDF পাঠায়"""
    pdfs = get_pdfs(PDF_FOLDER)
    recipients = get_email_list(EMAILS_FILE)
    
    if not recipients:
        print("❌ No recipients found! Check emails.txt or EMAIL_TO secret.")
        return
    
    if not pdfs:
        print("⚠️  No PDF found; sending mail without attachment.")
    
    print(f"📧 Sending to {len(recipients)} recipients...")
    print(f"📎 Attaching {len(pdfs)} PDF file(s)")
    
    # প্রতিটি ইমেইলের জন্য আলাদা করে পাঠাই (BCC না করে)
    # কারণ অনেক ইমেইল সার্ভার BCC তে সীমা রাখে
    success_count = 0
    failed_count = 0
    
    for recipient in recipients:
        try:
            msg = EmailMessage()
            msg["Subject"] = "📈 Daily Stock Signal Report"
            msg["From"]    = EMAIL_USER
            msg["To"]      = recipient
            msg.set_content(
                f"Hello,\n\n"
                f"Please find attached today's AI-generated stock signal report(s).\n\n"
                f"This email was sent to you as part of our daily update service.\n\n"
                f"Best regards,\nAI Signal Bot"
            )
            
            # PDF অ্যাটাচ করুন
            for pdf in pdfs:
                if pdf.stat().st_size > 25 * 1024 * 1024:   # 25 MB limit
                    print(f"⚠️  Skipping large file: {pdf.name}")
                    continue
                msg.add_attachment(
                    pdf.read_bytes(),
                    maintype="application",
                    subtype="pdf",
                    filename=pdf.name,
                )
            
            # ইমেইল পাঠান
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=60) as smtp:
                smtp.login(EMAIL_USER, EMAIL_PASS)
                smtp.send_message(msg)
            
            print(f"✅ Sent to {recipient}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Failed to send to {recipient}: {str(e)}")
            failed_count += 1
    
    print(f"\n📊 Summary: {success_count} successful, {failed_count} failed")

if __name__ == "__main__":
    send_email_to_all()
