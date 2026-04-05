# generate_pdf.py - Improved version

import os
import pandas as pd
from fpdf.enums import XPos, YPos
from fpdf import FPDF
from datetime import datetime, timedelta
import requests
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import numpy as np

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO   = os.getenv("EMAIL_TO")

# =========================
# CONFIG
# =========================
CSV_FOLDER = "./csv"
OUTPUT_FOLDER = "./output/ai_signal"
PDF_FOLDER = os.path.join(OUTPUT_FOLDER, "pdfs")

os.makedirs(CSV_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

# =========================
# PDF CLASS
# =========================
class PDF(FPDF):
    def __init__(self, csv_headers, title=""):
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=15)
        self.csv_headers = csv_headers
        self.title_text = title
        self.set_font("Helvetica", size=10)

    def header(self):
        now_plus_8 = datetime.now() + timedelta(hours=8)
        formatted_time = now_plus_8.strftime("%d/%m/%Y %I:%M %p")
        self.set_font("Helvetica", 'B', 14)
        if self.title_text:
            self.cell(0, 10, self.title_text, 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            self.cell(0, 5, formatted_time, 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("Helvetica", "", 10)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C")

    def add_table(self, data):
        if data.empty or not self.csv_headers:
            print("⚠️ No data or headers to generate PDF.")
            return

        if len(data) > 500:
            print(f"⚠️ Data has {len(data)} rows, limiting to 500 for PDF")
            data = data.head(500)

        page_width = self.w - 2 * self.l_margin
        col_width = page_width / len(self.csv_headers)

        def draw_header():
            self.set_font("Helvetica", "B", 9)
            self.set_fill_color(220, 220, 220)
            for header in self.csv_headers:
                self.cell(col_width, 8, str(header).upper(), border=1, align="C", fill=True)
            self.ln()
            self.set_font("Helvetica", "", 8)

        draw_header()

        for idx, row in data.iterrows():
            if self.get_y() > self.h - 25:
                self.add_page()
                draw_header()

            for header in self.csv_headers:
                value = row.get(header, "")
                if pd.isna(value):
                    value = ""
                else:
                    value = str(value)

                if header.lower() == 'symbol':
                    self.set_font("Helvetica", "B", 8)
                    self.cell(col_width, 8, value, border=1, align="C")
                    self.set_font("Helvetica", "", 8)
                else:
                    self.cell(col_width, 8, value, border=1, align="C")
            self.ln()

# =========================
# PDF GENERATOR
# =========================
def generate_pdf_report(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
        
        # Check if DataFrame is empty or has no columns
        if df.empty or len(df.columns) == 0:
            print(f"⚠️ Empty or invalid CSV: {csv_path}")
            return False
        
        # Check if all columns are empty
        if all(df[col].isna().all() for col in df.columns):
            print(f"⚠️ CSV has no data (all columns empty): {csv_path}")
            return False
        
        # Convert all data to string for PDF
        df = df.fillna("")
        
        # Remove columns that are completely empty
        df = df.loc[:, (df != "").any(axis=0)]
        
        if df.empty or len(df.columns) == 0:
            print(f"⚠️ No valid columns to display: {csv_path}")
            return False
            
        print(f"📊 CSV info: {len(df)} rows, {len(df.columns)} columns")

    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False

    title = os.path.splitext(os.path.basename(csv_path))[0]
    
    try:
        pdf = PDF(csv_headers=df.columns.tolist(), title=title)
        pdf.add_page()
        pdf.add_table(df)
        pdf.output(output_path)
        print(f"✅ PDF created: {output_path}")
        return True
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        return False

# =========================
# NOTIFICATION FUNCTIONS
# =========================
def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
        if r.status_code == 200:
            print("✅ Telegram alert sent.")
        else:
            print(f"❌ Telegram failed: {r.text}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

def send_email_alert(subject, body):
    if not all([EMAIL_USER, EMAIL_PASS, EMAIL_TO]):
        print("⚠️ Email credentials missing.")
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL("smtp.mail.yahoo.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
            print("✅ Email alert sent.")
    except Exception as e:
        print(f"❌ Email error: {e}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("="*60)
    print("📄 PDF GENERATOR")
    print("="*60)
    print(f"📁 PDF Source: {OUTPUT_FOLDER}")
    print("="*60)


    # =========================================================
    # 2. PDF Generation
    # =========================================================
    print("\n📄 STEP 1: Generating PDF reports...")

    pdf_generated = False
    generated_files = []
    failed_files = []

    if not os.path.exists(OUTPUT_FOLDER):
        print(f"❌ Directory not found: {OUTPUT_FOLDER}")
    else:
        csv_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".csv")]
        
        if not csv_files:
            print("⚠️ No CSV files found in OUTPUT_FOLDER.")
        else:
            print(f"\n📁 Found {len(csv_files)} CSV files")
            
            for csv_file in sorted(csv_files):
                csv_path = os.path.join(OUTPUT_FOLDER, csv_file)
                pdf_path = os.path.join(PDF_FOLDER, os.path.splitext(csv_file)[0] + ".pdf")
                print(f"\n📄 Processing: {csv_file}")
                
                if generate_pdf_report(csv_path, pdf_path):
                    pdf_generated = True
                    generated_files.append(pdf_path)
                else:
                    failed_files.append(csv_file)

    # =========================================================
    # 3. Summary
    # =========================================================
    print("\n" + "="*60)
    print("📊 GENERATION SUMMARY")
    print("="*60)
    print(f"✅ Successful: {len(generated_files)} PDFs")
    print(f"❌ Failed: {len(failed_files)} files")
    
    if failed_files:
        print(f"\n⚠️ Failed files:")
        for f in failed_files:
            print(f"   - {f}")

    # =========================================================
    # 4. Notifications
    # =========================================================
    print("\n📢 STEP 2: Sending notifications...")

    if pdf_generated:
        success_msg = f"""✅ PDF Generation Complete!
📊 PDF Files: {len(generated_files)}
📁 Location: {PDF_FOLDER}
📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""
        send_telegram_alert(success_msg)
        
        if generated_files:
            file_list = "\n".join([f"   - {os.path.basename(f)}" for f in generated_files[:10]])
            if len(generated_files) > 10:
                file_list += f"\n   ... and {len(generated_files)-10} more"
            send_email_alert("PDF Generation Complete", f"{success_msg}\n\nFiles:\n{file_list}")
    else:
        alert = "⚠️ No PDFs were generated! Check the CSV files in ./output/ai_signal/"
        send_telegram_alert(alert)
        send_email_alert("PDF Generation Failed", alert)

    # Final summary
    print("\n" + "="*60)
    print("✅ GENERATE_PDF COMPLETE!")
    print("="*60)
    print(f"📄 PDF Generated: {'✅' if pdf_generated else '❌'}")
    print(f"📁 PDF Folder: {PDF_FOLDER}")
    print("="*60)
