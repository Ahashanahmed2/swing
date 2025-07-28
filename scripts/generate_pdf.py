from hf_uploader import upload_to_hf
import pandas as pd
from fpdf.enums import XPos, YPos
from fpdf import FPDF
import os
from datetime import datetime, timedelta
# ✅ PDF ক্লাস (Unicode + Header/Footer + Table)
class PDF(FPDF):
    def __init__(self, csv_headers, title=""): # Added title parameter
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=15)
        self.csv_headers = csv_headers
        self.title_text = title # Store the title

        # ✅ ইউনিকোড ফন্ট যুক্ত করো
        self.add_font("NotoSans", "", "./fonts/NotoSans-Regular.ttf")
        self.add_font("NotoSans", "B", "./fonts/NotoSans-Bold.ttf")
        self.add_font("NotoSans", "I", "./fonts/NotoSans-Italic.ttf")

    def header(self):
        # বর্তমান সময়
        now = datetime.now()

        # সাথে ৮ ঘন্টা যোগ করা
        now_plus_8 = now + timedelta(hours=8)

        # ফরম্যাট করে দেখানো
        formatted_time = now_plus_8.strftime("%d/%m/%Y %I:%M %p")
        self.set_font("NotoSans",'B', 14)
        if self.title_text:
            self.cell(0, 10, self.title_text, 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

            self.cell(0, 5, formatted_time, 0, 1, "C")
        self.set_font("NotoSans","", 10)
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font("NotoSans", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def add_table(self, data):
        if data.empty:
            print("⚠️ No data to generate PDF.")
            return

        headers_to_use = self.csv_headers # Use dynamic headers from CSV

        page_width = self.w - 2 * self.l_margin
        
        # Ensure headers_to_use is not empty to avoid ZeroDivisionError
        if not headers_to_use:
            print("⚠️ No headers found in CSV, cannot draw table.")
            return

        col_width = page_width / len(headers_to_use)

        def draw_header():
            self.set_font("NotoSans", "B", 9)
            self.set_fill_color(220, 220, 220)
            self.set_text_color(0, 0, 0)
            for header in headers_to_use:
                self.cell(col_width, 8, str(header).upper(), border=1, align="C", fill=True)
            self.ln()
            self.set_font("NotoSans", "", 8)
            self.set_text_color(0)

        draw_header()

        for index, row in data.iterrows():
            if self.get_y() > self.h - 25:
                self.add_page()
                draw_header()

            for header in headers_to_use:
                value = str(row.get(header, ""))
                self.cell(col_width, 8, value, border=1, align="C")
            self.ln()

# ✅ PDF জেনারেটর ফাংশন
def generate_pdf_report(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"❌ CSV ফাইল পাওয়া যায়নি: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"⚠️ খালি CSV ফাইল, PDF তৈরি করা হচ্ছে না: {csv_path}")
        return
    except Exception as e:
        print(f"❌ CSV ফাইল পড়তে সমস্যা হয়েছে {csv_path}: {e}")
        return

    if df.empty:
        print(f"⚠️ খালি CSV ফাইল, PDF তৈরি করা হচ্ছে না: {csv_path}")
        return

    if len(df.columns) == 0:
        print(f"⚠️ CSV ফাইলে কোনো কলাম নেই, PDF তৈরি করা হচ্ছে না: {csv_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Extract title from CSV file name
    csv_file_name = os.path.splitext(os.path.basename(csv_path))[0]
    pdf = PDF(csv_headers=df.columns.tolist(), title=csv_file_name)
    pdf.add_page()
    pdf.add_table(df)
    pdf.output(output_path)
    print(f"✅ PDF রিপোর্ট তৈরি হয়েছে: {output_path}")

# ✅ মেইন ফাংশন
if __name__ == "__main__":
    # আপনার CSV ফাইলগুলো যে ফোল্ডারে আছে, সেই পাথটি এখানে দিন
    # উদাহরণ: folder_path = "/home/ubuntu/my_csv_files/"
    folder_path = "./output/ai_signal" # এটি একটি উদাহরণ, আপনার পাথ দিন

    output_pdf_dir = os.path.join(folder_path, "pdfs")
    os.makedirs(output_pdf_dir, exist_ok=True)

    # NotoSans-Regular.ttf ফন্ট ফাইলটি ./fonts/ ডিরেক্টরিতে আছে কিনা নিশ্চিত করুন
    # যদি না থাকে, তাহলে এটি ডাউনলোড করে এই ডিরেক্টরিতে রাখুন
    # উদাহরণ: mkdir -p fonts && wget -O fonts/NotoSans-Regular.ttf https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf
    # NotoSans-Bold.ttf এবং NotoSans-Italic.ttf এর জন্যও একই কাজ করুন

    if not os.path.exists(folder_path):
        print(f"❌ ডিরেক্টরিটি পাওয়া যায়নি: {folder_path}")
    else:
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

        if not csv_files:
            print("⚠️ কোনো CSV ফাইল পাওয়া যায়নি এই ডিরেক্টরিতে।")
        else:
            for csv_file in csv_files:
                full_csv_path = os.path.join(folder_path, csv_file)
                output_pdf_name = os.path.splitext(csv_file)[0] + ".pdf"
                output_pdf_path = os.path.join(output_pdf_dir, output_pdf_name)

                print(f"\n📄 রিপোর্ট তৈরি হচ্ছে: {csv_file}")
                generate_pdf_report(full_csv_path, output_pdf_path)

upload_to_hf()
