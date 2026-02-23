import os
import pandas as pd
from fpdf.enums import XPos, YPos
from fpdf import FPDF
from datetime import datetime, timedelta
#from hf_uploader import SmartDatasetUploader, REPO_ID, HF_TOKEN
import requests
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO   = os.getenv("EMAIL_TO")

# ✅ PDF ক্লাস
class PDF(FPDF):
    def __init__(self, csv_headers, title=""):
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=15)
        self.csv_headers = csv_headers
        self.title_text = title
        
        # ফন্ট সেটআপ
        font_dir = "./fonts"
        if not os.path.exists(font_dir):
            os.makedirs(font_dir, exist_ok=True)
            
        # ডিফল্ট ফন্ট ব্যবহার করুন (সহজে কপি করার জন্য)
        self.set_font("Helvetica", size=10)  # Helvetica ফন্ট ব্যবহার করছি যা সিলেক্ট করা যায়

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
        
        # ডাটা লিমিট করুন (যদি খুব বড় হয়)
        if len(data) > 500:
            print(f"⚠️ Data has {len(data)} rows, limiting to 500 for PDF")
            data = data.head(500)
            
        page_width = self.w - 2 * self.l_margin
        col_width = page_width / len(self.csv_headers)

        def draw_header():
            self.set_font("Helvetica", "B", 9)
            self.set_fill_color(220, 220, 220)
            for header in self.csv_headers:
                # হেডার সেন্টার করে দেখান
                self.cell(col_width, 8, str(header).upper(), border=1, align="C", fill=True)
            self.ln()
            self.set_font("Helvetica", "", 8)

        draw_header()
        
        # ডাটা রো দেখান
        for idx, row in data.iterrows():
            if self.get_y() > self.h - 25:
                self.add_page()
                draw_header()
            
            for header in self.csv_headers:
                value = row.get(header, "")
                
                # NaN/null ভ্যালু হ্যান্ডেল করুন
                if pd.isna(value):
                    value = ""
                else:
                    value = str(value)
                
                # সিম্বল কলামের জন্য বিশেষ ফরম্যাট
                if header.lower() == 'symbol':
                    # সিম্বল বোল্ড করুন
                    self.set_font("Helvetica", "B", 8)
                    self.cell(col_width, 8, value, border=1, align="C")
                    self.set_font("Helvetica", "", 8)
                else:
                    # অন্যান্য ডাটা নরমাল ফন্টে
                    self.cell(col_width, 8, value, border=1, align="C")
            
            self.ln()
            
            # প্রতি 50 রো পর একটি খালি লাইন দিন (পড়তে সুবিধা)
            if (idx + 1) % 50 == 0 and idx < len(data) - 1:
                self.ln(2)

# ✅ PDF জেনারেটর
def generate_pdf_report(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"❌ CSV ফাইল পাওয়া যায়নি: {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        
        # ডাটা ভ্যালিডেশন
        if df.empty or len(df.columns) == 0:
            print(f"⚠️ খালি বা অবৈধ CSV: {csv_path}")
            return False
        
        # NaN ভ্যালুগুলো খালি স্ট্রিং দিয়ে প্রতিস্থাপন
        df = df.fillna("")
        
        print(f"📊 CSV তথ্য: {len(df)} রো, {len(df.columns)} কলাম")
        print(f"📋 কলামসমূহ: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ CSV ফাইল পড়তে সমস্যা: {e}")
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    title = os.path.splitext(os.path.basename(csv_path))[0]
    
    pdf = PDF(csv_headers=df.columns.tolist(), title=title)
    pdf.add_page()
    pdf.add_table(df)
    pdf.output(output_path)
    
    print(f"✅ PDF তৈরি হয়েছে: {output_path}")
    return True

# ✅ PDF চেকার
def check_pdf_generation(pdf_dir):
    if not os.path.exists(pdf_dir):
        return False
    return any(f.endswith(".pdf") for f in os.listdir(pdf_dir))

# ✅ Telegram নোটিফিকেশন
def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
        print("✅ Telegram alert sent." if r.status_code == 200 else f"❌ Telegram failed: {r.text}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

# ✅ Email নোটিফিকেশন
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

# ✅ মেইন ফাংশন
if __name__ == "__main__":
    folder_path = "./output/ai_signal"
    output_pdf_dir = os.path.join(folder_path, "pdfs")
    os.makedirs(output_pdf_dir, exist_ok=True)

    # CSV ফাইল প্রসেসিং
    pdf_generated = False
    
    if not os.path.exists(folder_path):
        print(f"❌ ডিরেক্টরি পাওয়া যায়নি: {folder_path}")
    else:
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if not csv_files:
            print("⚠️ কোনো CSV ফাইল পাওয়া যায়নি।")
        else:
            print(f"\n📁 মোট {len(csv_files)} টি CSV ফাইল পাওয়া গেছে")
            for csv_file in csv_files:
                csv_path = os.path.join(folder_path, csv_file)
                pdf_path = os.path.join(output_pdf_dir, os.path.splitext(csv_file)[0] + ".pdf")
                print(f"\n📄 রিপোর্ট তৈরি হচ্ছে: {csv_file}")
                if generate_pdf_report(csv_path, pdf_path):
                    pdf_generated = True

    # -------------------------------------------------------------------
    """# Step 9: Upload updated CSV to Hugging Face (optional)
    # -------------------------------------------------------------------
    print("\n📤 Uploading CSV files to Hugging Face...")

    # csv_folder ডিফাইন করুন
    csv_folder = folder_path

    # Hugging Face আপলোড
    try:
        uploader = SmartDatasetUploader(REPO_ID, HF_TOKEN)
        uploader.smart_upload(
            local_folder=csv_folder,
            unique_columns=['symbol']
        )
        print("✅ Upload to Hugging Face complete!")
    except Exception as e:
        print(f"❌ Hugging Face upload failed: {e}")"""

    # ✅ PDF না থাকলে নোটিফিকেশন
    if not pdf_generated:
        alert = "⚠️ কোনো PDF তৈরি হয়নি। CSV ফাইল খালি বা ত্রুটিপূর্ণ হতে পারে।"
        send_telegram_alert(alert)
        send_email_alert("PDF Generation Failed", alert)
    else:
        print(f"\n✅ সমস্ত PDF তৈরি সম্পন্ন হয়েছে! লোকেশন: {output_pdf_dir}")