# generate_pdf.py - Complete Updated Version
import os
import pandas as pd
from fpdf.enums import XPos, YPos
from fpdf import FPDF
from datetime import datetime, timedelta
from hf_uploader import SmartDatasetUploader, REPO_ID, HF_TOKEN
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

            if (idx + 1) % 50 == 0 and idx < len(data) - 1:
                self.ln(2)

# ✅ PDF জেনারেটর
def generate_pdf_report(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"❌ CSV ফাইল পাওয়া যায়নি: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
        if df.empty or len(df.columns) == 0:
            print(f"⚠️ খালি বা অবৈধ CSV: {csv_path}")
            return False

        df = df.fillna("")
        print(f"📊 CSV তথ্য: {len(df)} রো, {len(df.columns)} কলাম")

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

# ✅ Hugging Face আপলোড ফাংশন
def upload_to_huggingface():
    """Upload CSV and model files to Hugging Face"""
    print("\n" + "="*60)
    print("📤 UPLOADING TO HUGGING FACE")
    print("="*60)
    
    # ✅ সঠিক ফোল্ডার: ./csv (এতে mongodb.csv, xgboost/, xgb_confidence.csv সব আছে)
    master_folder = "./csv"
    
    if not os.path.exists(master_folder):
        print(f"❌ মাস্টার ফোল্ডার নেই: {master_folder}")
        return False
    
    # Check what's in the folder
    csv_files = [f for f in os.listdir(master_folder) if f.endswith('.csv')]
    print(f"📁 মাস্টার ফোল্ডারে CSV ফাইল: {csv_files}")
    
    # Check xgboost folder
    xgboost_folder = os.path.join(master_folder, 'xgboost')
    if os.path.exists(xgboost_folder):
        model_files = [f for f in os.listdir(xgboost_folder) if f.endswith('.joblib')]
        print(f"🤖 মডেল ফোল্ডারে {len(model_files)} টি মডেল ফাইল পাওয়া গেছে")
        
        # Show some model names
        if model_files:
            print(f"   উদাহরণ: {model_files[:3]}")
    else:
        print(f"⚠️ xgboost ফোল্ডার নেই: {xgboost_folder}")
    
    # Upload everything
    try:
        uploader = SmartDatasetUploader(REPO_ID, HF_TOKEN)
        uploader.smart_upload(
            local_folder=master_folder,  # ✅ সঠিক ফোল্ডার
            unique_columns=['symbol', 'date']  # CSV merge এর জন্য
        )
        print("✅ Master CSV and models uploaded to Hugging Face!")
        return True
    except Exception as e:
        print(f"❌ Hugging Face upload failed: {e}")
        return False

# ✅ মেইন ফাংশন
if __name__ == "__main__":
    print("="*60)
    print("📄 PDF GENERATOR & HF UPLOADER")
    print("="*60)
    
    # =========================================================
    # 1. PDF তৈরি (./output/ai_signal থেকে)
    # =========================================================
    print("\n📄 STEP 1: Generating PDF reports...")
    folder_path = "./output/ai_signal"
    output_pdf_dir = os.path.join(folder_path, "pdfs")
    os.makedirs(output_pdf_dir, exist_ok=True)

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

    # =========================================================
    # 2. Hugging Face আপলোড - মাস্টার CSV ফোল্ডার
    # =========================================================
    print("\n" + "="*60)
    print("📤 STEP 2: Uploading to Hugging Face...")
    print("="*60)
    
    upload_success = upload_to_huggingface()
    
    # =========================================================
    # 3. Check if xgb_confidence.csv exists in ./csv
    # =========================================================
    confidence_file = "./csv/xgb_confidence.csv"
    if os.path.exists(confidence_file):
        df_conf = pd.read_csv(confidence_file)
        print(f"\n📊 xgb_confidence.csv তথ্য:")
        print(f"   মোট রো: {len(df_conf):,}")
        print(f"   কলাম: {list(df_conf.columns)}")
        print(f"   সর্বশেষ তারিখ: {df_conf['date'].max() if 'date' in df_conf.columns else 'N/A'}")
    else:
        print(f"\n⚠️ xgb_confidence.csv ফাইল পাওয়া যায়নি: {confidence_file}")
    
    # =========================================================
    # 4. নোটিফিকেশন
    # =========================================================
    print("\n" + "="*60)
    print("📢 STEP 3: Sending notifications...")
    print("="*60)
    
    if not pdf_generated:
        alert = "⚠️ কোনো PDF তৈরি হয়নি। CSV ফাইল খালি বা ত্রুটিপূর্ণ হতে পারে।"
        send_telegram_alert(alert)
        send_email_alert("PDF Generation Failed", alert)
    else:
        print(f"\n✅ সমস্ত PDF তৈরি সম্পন্ন হয়েছে! লোকেশন: {output_pdf_dir}")
        
        # Send success notification
        if upload_success:
            success_msg = f"✅ PDF Generation & HF Upload Complete!\n📊 Total predictions: {len(df_conf):, if 'df_conf' in dir() else 'N/A'}\n📁 Models: 387 symbols"
            send_telegram_alert(success_msg)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ GENERATE_PDF COMPLETE!")
    print("="*60)
    print(f"📄 PDF Generated: {'✅' if pdf_generated else '❌'}")
    print(f"📤 HF Upload: {'✅' if upload_success else '❌'}")
    print("="*60)