# xgboost_Elliott_support.py - Complete Trading System
# Features:
# 1. XGBoost predictions with GOOD/BAD model filtering
# 2. Elliott Wave signal filtering (AUTO from elliott_wave.py output)
# 3. Support/Resistance confirmation
# 4. Final trading signals with confidence scoring
# 5. Telegram/Email notifications
# 6. PDF generation for reports

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIGURATION
# =========================
CSV_FOLDER = "./csv"
OUTPUT_FOLDER = "./output/ai_signal"
PDF_FOLDER = os.path.join(OUTPUT_FOLDER, "pdfs")
MODEL_METADATA = os.path.join(CSV_FOLDER, "model_metadata.csv")
XGB_CONFIDENCE = os.path.join(CSV_FOLDER, "xgb_confidence.csv")
SUPPORT_RESISTANCE = os.path.join(CSV_FOLDER, "support_resistance.csv")
ELLIOTT_WAVE_OUTPUT = os.path.join(CSV_FOLDER, "Elliott_wave.csv")  # ✅ elliott_wave.py থেকে
ELLIOTT_SIGNALS = os.path.join(CSV_FOLDER, "elliott_signals.csv")   # ✅ অটো-জেনারেটেড
FINAL_SIGNALS = os.path.join(CSV_FOLDER, "final_trading_signals.csv")

os.makedirs(CSV_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

# Signal thresholds
XGB_MIN_CONFIDENCE = 6.0  # XGBoost confidence (0-10 scale)
ELLIOTT_MIN_CONFIDENCE = 50  # Elliott Wave minimum confidence
SR_DISTANCE_THRESHOLD = 5  # Support/Resistance distance threshold (%)
MAX_SIGNALS_PER_DAY = 10  # Maximum signals to generate

# =========================
# DATA LOADING FUNCTIONS
# =========================

def load_xgb_predictions():
    """Load XGBoost predictions"""
    if not os.path.exists(XGB_CONFIDENCE):
        print(f"❌ XGBoost predictions not found: {XGB_CONFIDENCE}")
        return None
    
    df = pd.read_csv(XGB_CONFIDENCE)
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_model_metadata():
    """Load model metadata (GOOD/BAD status)"""
    if not os.path.exists(MODEL_METADATA):
        print(f"⚠️ Model metadata not found")
        return None
    
    df = pd.read_csv(MODEL_METADATA)
    return df

def load_support_resistance():
    """Load support/resistance levels"""
    if not os.path.exists(SUPPORT_RESISTANCE):
        print(f"⚠️ Support/Resistance file not found")
        return None
    
    df = pd.read_csv(SUPPORT_RESISTANCE)
    df['current_date'] = pd.to_datetime(df['current_date'])
    return df

def load_elliott_signals():
    """
    ✅ AUTO: Load Elliott Wave signals from elliott_wave.py output
    যদি elliott_signals.csv থাকে সেটি ব্যবহার করবে
    নাহলে elliott_wave.py থেকে জেনারেট করবে
    """
    # প্রথমে elliott_signals.csv চেক করুন (অটো-জেনারেটেড)
    if os.path.exists(ELLIOTT_SIGNALS):
        print(f"   ✅ Loading auto-generated Elliott signals: {ELLIOTT_SIGNALS}")
        df = pd.read_csv(ELLIOTT_SIGNALS)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    
    # যদি elliott_signals.csv না থাকে, elliott_wave.py এর আউটপুট থেকে জেনারেট করুন
    if os.path.exists(ELLIOTT_WAVE_OUTPUT):
        print(f"   📂 Found elliott_wave.py output: {ELLIOTT_WAVE_OUTPUT}")
        print(f"   🔄 Generating elliott_signals.csv from Elliott Wave analysis...")
        
        wave_df = pd.read_csv(ELLIOTT_WAVE_OUTPUT)
        
        # FINAL_AI_SIGNALS থেকে বর্তমান প্রাইস নিন
        signals_file = os.path.join(CSV_FOLDER, "FINAL_AI_SIGNALS.csv")
        if os.path.exists(signals_file):
            signals = pd.read_csv(signals_file)
            if 'symbol' in signals.columns and 'current_price' in signals.columns:
                signals = signals[['symbol', 'current_price']].drop_duplicates(subset='symbol')
                wave_df = wave_df.merge(signals, left_on='SYMBOL', right_on='symbol', how='left')
        
        # ✅ FIXED: Confidence logic with proper brackets
        def get_confidence(wave_str):
            """Elliott Wave pattern থেকে confidence নির্ধারণ"""
            wave_str = str(wave_str)
            if 'Impulse' in wave_str:
                return 85
            elif 'Extension' in wave_str:
                return 75
            elif 'Flat' in wave_str:
                return 65
            elif 'Triangle' in wave_str:
                return 55
            elif 'Zigzag' in wave_str:
                return 60
            elif 'Diagonal' in wave_str:
                return 70
            else:
                return 50
        
        def get_signal(wave_str):
            """Elliott Wave pattern থেকে BUY/SELL/HOLD নির্ধারণ"""
            wave_str = str(wave_str)
            if 'Bullish' in wave_str:
                return 'BUY'
            elif 'Bearish' in wave_str:
                return 'SELL'
            else:
                return 'HOLD'
        
        # elliott_signals.csv ফরম্যাটে কনভার্ট
        elliott_df = pd.DataFrame({
            'symbol': wave_df['SYMBOL'],
            'date': pd.Timestamp.now(),
            'signal': wave_df['WAVE'].apply(get_signal),
            'confidence': wave_df['WAVE'].apply(get_confidence),
            'wave': wave_df['WAVE'].fillna('Unknown'),
            'sub_wave': wave_df.get('SUB_WAVE', 'N/A').fillna('N/A'),
            'price': wave_df['current_price'].fillna(0) if 'current_price' in wave_df.columns else 0
        })
        
        # সেভ করুন
        elliott_df.to_csv(ELLIOTT_SIGNALS, index=False)
        print(f"   ✅ Generated {len(elliott_df)} Elliott signals: {ELLIOTT_SIGNALS}")
        return elliott_df
    
    # কিছুই না পাওয়া গেলে
    print(f"   ⚠️ No Elliott Wave data found. Run elliott_wave.py first.")
    return None

# =========================
# SIGNAL FILTERING FUNCTIONS
# =========================

def filter_by_model_quality(xgb_df, metadata):
    """Filter only GOOD XGBoost models"""
    if metadata is None:
        return xgb_df
    
    # Check if 'status' or 'auc' column exists
    if 'status' in metadata.columns:
        good_symbols = metadata[metadata['status'] == 'GOOD']['symbol'].tolist()
    elif 'auc' in metadata.columns:
        good_symbols = metadata[metadata['auc'] >= 0.55]['symbol'].tolist()
    else:
        return xgb_df
    
    filtered_df = xgb_df[xgb_df['symbol'].isin(good_symbols)]
    
    print(f"   📊 Model filtering: {len(xgb_df)} → {len(filtered_df)} (GOOD models only)")
    return filtered_df

def filter_by_xgb_confidence(xgb_df, min_confidence=6.0):
    """Filter by XGBoost confidence score (auto-detect scale)"""
    
    if xgb_df.empty:
        return xgb_df
    
    # ✅ FIX: Auto-detect confidence score scale
    max_conf = xgb_df['confidence_score'].max()
    min_conf = xgb_df['confidence_score'].min()
    
    if max_conf <= 1.0:
        # 0-1 scale
        threshold = min_confidence / 10
        print(f"   📊 Detected 0-1 scale (max={max_conf:.3f}), threshold: {threshold:.3f}")
    elif max_conf <= 10.0:
        # 0-10 scale (YOUR DATA)
        threshold = min_confidence
        print(f"   📊 Detected 0-10 scale (max={max_conf:.2f}), threshold: {threshold:.2f}")
    elif max_conf <= 100.0:
        # 0-100 scale
        threshold = min_confidence * 10
        print(f"   📊 Detected 0-100 scale (max={max_conf:.1f}), threshold: {threshold:.1f}")
    else:
        # Unknown scale
        threshold = min_confidence
        print(f"   📊 Unknown scale (max={max_conf:.2f}), using raw threshold: {threshold:.2f}")
    
    filtered_df = xgb_df[xgb_df['confidence_score'] >= threshold]
    print(f"   📊 XGBoost confidence filter (>={threshold:.2f}): {len(filtered_df)} signals")
    return filtered_df

def filter_by_elliott_match(xgb_df, elliott_df):
    """Keep only stocks that have Elliott Wave BUY signals"""
    if elliott_df is None or elliott_df.empty:
        print(f"   ⚠️ No Elliott signals found - skipping Elliott filter")
        return xgb_df
    
    elliott_buy = elliott_df[elliott_df['signal'] == 'BUY']['symbol'].tolist()
    filtered_df = xgb_df[xgb_df['symbol'].isin(elliott_buy)]
    
    print(f"   📊 Elliott Wave matching: {len(xgb_df)} → {len(filtered_df)} signals")
    return filtered_df

def filter_by_support_resistance(signals_df, sr_df):
    """Check if price is near support/resistance levels"""
    if sr_df is None or sr_df.empty:
        print(f"   ⚠️ No Support/Resistance data - skipping S/R filter")
        return signals_df
    
    if signals_df.empty:
        return signals_df
    
    # ✅ FIX: Find the correct price column
    price_column = None
    for col in ['close', 'current_price', 'price', 'ltp', 'Close']:
        if col in signals_df.columns:
            price_column = col
            break
    
    if price_column is None:
        print(f"   ⚠️ No price column found in signals")
        return signals_df
    
    print(f"   📊 Using price column: '{price_column}'")
    
    # Get latest support/resistance
    if 'current_date' not in sr_df.columns:
        return signals_df
    
    latest_date = sr_df['current_date'].max()
    latest_sr = sr_df[sr_df['current_date'] == latest_date]
    
    # Create a dictionary of support/resistance levels
    sr_levels = {}
    for _, row in latest_sr.iterrows():
        symbol = row['symbol']
        sr_levels[symbol] = {
            'type': row['type'],
            'level': row['level_price'],
            'strength': row.get('strength', 'Weak')
        }
    
    # Check each signal
    signals_with_sr = []
    for _, row in signals_df.iterrows():
        symbol = row['symbol']
        price = row[price_column]
        
        row_dict = row.to_dict()
        
        if symbol in sr_levels:
            sr_info = sr_levels[symbol]
            distance = abs(price - sr_info['level']) / price * 100 if price > 0 else 100
            
            if sr_info['type'] == 'support' and price > sr_info['level'] and distance < SR_DISTANCE_THRESHOLD:
                row_dict['sr_confirmation'] = f"Near support at {sr_info['level']} ({sr_info['strength']})"
                row_dict['sr_bonus'] = 5 if sr_info['strength'] == 'Strong' else (2 if sr_info['strength'] == 'Moderate' else 1)
            elif sr_info['type'] == 'resistance' and price < sr_info['level'] and distance < SR_DISTANCE_THRESHOLD:
                row_dict['sr_confirmation'] = f"Below resistance at {sr_info['level']} ({sr_info['strength']})"
                row_dict['sr_bonus'] = 3 if sr_info['strength'] == 'Strong' else 1
            else:
                row_dict['sr_confirmation'] = "No significant S/R near"
                row_dict['sr_bonus'] = 0
        else:
            row_dict['sr_confirmation'] = "No S/R data"
            row_dict['sr_bonus'] = 0
        
        signals_with_sr.append(row_dict)
    
    result_df = pd.DataFrame(signals_with_sr)
    print(f"   📊 Support/Resistance: {len(result_df)} signals analyzed")
    return result_df

# =========================
# SIGNAL SCORING FUNCTION
# =========================

def calculate_final_score(row):
    """
    Calculate final confidence score based on:
    - XGBoost confidence (60% weight)
    - Elliott confidence (30% weight)
    - Support/Resistance bonus (10% weight)
    """
    xgb_weight = 0.60
    elliott_weight = 0.30
    sr_weight = 0.10
    
    # ✅ FIX: Normalize XGBoost score to 0-100
    xgb_raw = row.get('confidence_score', 5)
    max_conf = 10.0  # Your scale is 0-10
    
    if max_conf > 0:
        xgb_score = min(100, xgb_raw / max_conf * 100)
    else:
        xgb_score = 50
    
    # Elliott score (if available)
    if 'elliott_confidence' in row and pd.notna(row.get('elliott_confidence')):
        elliott_score = row.get('elliott_confidence', 50)
    else:
        elliott_score = 50
    
    # Support/Resistance bonus
    sr_bonus = row.get('sr_bonus', 0)  # Max 5 bonus
    
    # Calculate final score
    final_score = (xgb_score * xgb_weight + elliott_score * elliott_weight + sr_bonus * sr_weight * 10)
    
    return min(100, final_score)

# =========================
# MAIN SIGNAL GENERATION
# =========================

def generate_final_signals():
    """Main function to generate final trading signals"""
    
    print("="*70)
    print("🎯 FINAL TRADING SIGNAL GENERATOR")
    print("="*70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)
    
    # Step 1: Load all data sources
    print("\n📂 Step 1: Loading data sources...")
    
    xgb_df = load_xgb_predictions()
    if xgb_df is None:
        print("❌ Cannot proceed without XGBoost predictions")
        return None
    
    metadata = load_model_metadata()
    elliott_df = load_elliott_signals()  # ✅ AUTO from elliott_wave.py
    sr_df = load_support_resistance()
    
    # Get latest predictions
    latest_date = xgb_df['date'].max()
    latest_xgb = xgb_df[xgb_df['date'] == latest_date].copy()
    print(f"   ✅ XGBoost loaded: {len(latest_xgb)} predictions for {latest_date.date()}")
    
    # Step 2: Filter by model quality (only GOOD models)
    print("\n🔍 Step 2: Filtering by model quality...")
    filtered_df = filter_by_model_quality(latest_xgb, metadata)
    
    if filtered_df.empty:
        print("❌ No signals after model quality filter")
        return None
    
    # Step 3: Filter by XGBoost confidence
    print("\n🎯 Step 3: Filtering by XGBoost confidence...")
    filtered_df = filter_by_xgb_confidence(filtered_df, XGB_MIN_CONFIDENCE)
    
    if filtered_df.empty:
        print("❌ No signals after XGBoost confidence filter")
        return None
    
    # Step 4: Filter by Elliott Wave (only BUY signals)
    print("\n🌊 Step 4: Filtering by Elliott Wave...")
    filtered_df = filter_by_elliott_match(filtered_df, elliott_df)
    
    # Step 5: Add Elliott confidence to signals
    if elliott_df is not None and not elliott_df.empty and not filtered_df.empty:
        elliott_dict = {}
        for _, row in elliott_df.iterrows():
            sym = row['symbol']
            elliott_dict[sym] = {
                'confidence': row.get('confidence', 50),
                'wave': row.get('wave', 'Unknown'),
                'sub_wave': row.get('sub_wave', 'N/A')
            }
        
        for idx in filtered_df.index:
            symbol = filtered_df.loc[idx, 'symbol']
            if symbol in elliott_dict:
                filtered_df.loc[idx, 'elliott_confidence'] = elliott_dict[symbol]['confidence']
                filtered_df.loc[idx, 'elliott_wave'] = elliott_dict[symbol]['wave']
                filtered_df.loc[idx, 'elliott_sub_wave'] = elliott_dict[symbol]['sub_wave']
            else:
                filtered_df.loc[idx, 'elliott_confidence'] = 50
                filtered_df.loc[idx, 'elliott_wave'] = 'Unknown'
                filtered_df.loc[idx, 'elliott_sub_wave'] = 'N/A'
    
    # Step 6: Check Support/Resistance
    print("\n📊 Step 5: Checking Support/Resistance...")
    filtered_df = filter_by_support_resistance(filtered_df, sr_df)
    
    # Step 7: Calculate final scores
    print("\n📈 Step 6: Calculating final scores...")
    filtered_df['final_score'] = filtered_df.apply(calculate_final_score, axis=1)
    
    # Step 8: Sort and limit signals
    filtered_df = filtered_df.sort_values('final_score', ascending=False)
    final_signals = filtered_df.head(MAX_SIGNALS_PER_DAY).copy()
    
    # ✅ FIX: Find the correct price column for entry/SL/TP
    price_column = None
    for col in ['close', 'current_price', 'price', 'ltp', 'Close']:
        if col in final_signals.columns:
            price_column = col
            break
    
    if price_column:
        current_price = final_signals[price_column]
        print(f"   📊 Using price column '{price_column}' for Entry/SL/TP")
    else:
        print(f"   ⚠️ No price column found, using Elliott price")
        if 'price' in final_signals.columns:
            current_price = final_signals['price']
        else:
            current_price = pd.Series([0] * len(final_signals))
    
    # Add execution levels
    final_signals['entry_price'] = current_price
    final_signals['stop_loss'] = current_price * 0.97  # -3%
    final_signals['take_profit'] = current_price * 1.05  # +5%
    
    # Add signal strength
    final_signals['strength'] = final_signals['final_score'].apply(
        lambda x: 'STRONG' if x >= 80 else ('MEDIUM' if x >= 70 else 'WEAK')
    )
    
    # Add timestamp
    final_signals['generated_at'] = datetime.now()
    
    return final_signals

# =========================
# REPORT GENERATION
# =========================

def generate_pdf_report(signals_df):
    """Generate PDF report of final signals"""
    try:
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
    except ImportError:
        print("⚠️ fpdf not installed. Skipping PDF generation.")
        return None
    
    if signals_df is None or signals_df.empty:
        print("⚠️ No signals to generate PDF")
        return None
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 14)
            self.cell(0, 10, 'FINAL TRADING SIGNALS', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
            self.set_font('Helvetica', 'I', 10)
            self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
            self.ln(10)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    pdf = PDF(orientation='L')
    pdf.add_page()
    pdf.set_font('Helvetica', '', 9)
    
    # Price column
    price_col = 'entry_price'
    
    # Table headers
    headers = ['Symbol', 'Price', 'Signal', 'XGB%', 'Elliott%', 'Final%', 'Entry', 'SL', 'TP', 'Strength']
    col_widths = [30, 25, 30, 25, 25, 25, 30, 30, 30, 30]
    
    # Header
    pdf.set_fill_color(200, 200, 200)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, border=1, align='C', fill=True)
    pdf.ln()
    
    # Data rows
    for _, row in signals_df.iterrows():
        pdf.cell(col_widths[0], 8, str(row['symbol']), border=1, align='C')
        pdf.cell(col_widths[1], 8, f"{row.get(price_col, 0):.2f}", border=1, align='C')
        pdf.cell(col_widths[2], 8, 'BUY', border=1, align='C')
        pdf.cell(col_widths[3], 8, f"{row['confidence_score']:.1f}", border=1, align='C')
        pdf.cell(col_widths[4], 8, f"{row.get('elliott_confidence', 50):.0f}%", border=1, align='C')
        pdf.cell(col_widths[5], 8, f"{row['final_score']:.0f}%", border=1, align='C')
        pdf.cell(col_widths[6], 8, f"{row['entry_price']:.2f}", border=1, align='C')
        pdf.cell(col_widths[7], 8, f"{row['stop_loss']:.2f}", border=1, align='C')
        pdf.cell(col_widths[8], 8, f"{row['take_profit']:.2f}", border=1, align='C')
        pdf.cell(col_widths[9], 8, str(row['strength']), border=1, align='C')
        pdf.ln()
        
        if pdf.get_y() > 250:
            pdf.add_page()
    
    # Save PDF
    pdf_path = os.path.join(PDF_FOLDER, f"trading_signals_{datetime.now().strftime('%Y%m%d')}.pdf")
    pdf.output(pdf_path)
    print(f"   ✅ PDF saved: {pdf_path}")
    
    return pdf_path

def print_summary(signals_df):
    """Print summary of signals"""
    if signals_df is None or signals_df.empty:
        print("\n❌ No final signals generated!")
        return
    
    print("\n" + "="*70)
    print("📊 FINAL TRADING SIGNALS SUMMARY")
    print("="*70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"📈 Total signals: {len(signals_df)}")
    
    if 'strength' in signals_df.columns:
        print(f"   🔥 STRONG: {len(signals_df[signals_df['strength'] == 'STRONG'])}")
        print(f"   📊 MEDIUM: {len(signals_df[signals_df['strength'] == 'MEDIUM'])}")
        print(f"   ⚠️ WEAK: {len(signals_df[signals_df['strength'] == 'WEAK'])}")
    
    print("\n🔥 TOP 10 STRONGEST SIGNALS:")
    print("-"*70)
    
    # ✅ FIX: Use correct price column
    price_col = 'entry_price'
    
    for i, (_, row) in enumerate(signals_df.head(min(10, len(signals_df))).iterrows()):
        print(f"\n{i+1}. *{row['symbol']}* @ {row[price_col]:.2f}")
        print(f"   📊 XGBoost: {row['confidence_score']:.2f}")
        print(f"   🌊 Elliott: {row.get('elliott_confidence', 50):.0f}% ({row.get('elliott_wave', 'N/A')})")
        print(f"   🎯 Final Score: {row['final_score']:.0f}% ({row['strength']})")
        print(f"   📈 Entry: {row['entry_price']:.2f} | SL: {row['stop_loss']:.2f} | TP: {row['take_profit']:.2f}")
        if pd.notna(row.get('sr_confirmation')):
            print(f"   📊 S/R: {row['sr_confirmation']}")
    
    # Save to CSV
    signals_df.to_csv(FINAL_SIGNALS, index=False)
    print(f"\n✅ Signals saved to: {FINAL_SIGNALS}")

def send_telegram_alert(signals_df):
    """Send signals to Telegram"""
    try:
        import requests
        from dotenv import load_dotenv
        
        load_dotenv()
        telegram_token = os.getenv("TELEGRAM_TOKEN_TRADE")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID_TRADE")
        
        if not telegram_token or not telegram_chat_id:
            print("   ⚠️ Telegram credentials not found")
            return
        
        if signals_df is None or signals_df.empty:
            return
        
        message = "📊 *FINAL TRADING SIGNALS* 📊\n\n"
        message += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        message += f"📈 Total: {len(signals_df)} signals\n\n"
        message += "*TOP 5 RECOMMENDATIONS:*\n\n"
        
        for i, (_, row) in enumerate(signals_df.head(min(5, len(signals_df))).iterrows()):
            message += f"{i+1}. *{row['symbol']}* @ {row['entry_price']:.2f}\n"
            message += f"   🎯 Score: {row['final_score']:.0f}% ({row['strength']})\n"
            message += f"   📈 Entry: {row['entry_price']:.2f}\n"
            message += f"   📉 SL: {row['stop_loss']:.2f} | TP: {row['take_profit']:.2f}\n\n"
        
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        response = requests.post(url, data={
            'chat_id': telegram_chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        })
        
        if response.status_code == 200:
            print("   ✅ Telegram alert sent!")
        else:
            print(f"   ❌ Telegram failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Telegram error: {e}")

# =========================
# MAIN EXECUTION
# =========================

def main():
    """Main execution function"""
    
    # Generate final signals
    signals = generate_final_signals()
    
    if signals is not None and not signals.empty:
        # Print summary
        print_summary(signals)
        
        # Generate PDF report
        pdf_path = generate_pdf_report(signals)
        
        # Send Telegram alert
        send_telegram_alert(signals)
        
        print("\n" + "="*70)
        print("✅ FINAL TRADING SYSTEM COMPLETE!")
        print("="*70)
        print(f"📊 Signals: {len(signals)}")
        if pdf_path:
            print(f"📄 PDF: {pdf_path}")
        print(f"📁 CSV: {FINAL_SIGNALS}")
        print("="*70)
        
    else:
        print("\n❌ No signals generated. Check your data sources.")
        print("   Ensure the following files exist:")
        print(f"   - {XGB_CONFIDENCE}")
        print(f"   - {MODEL_METADATA}")
        print(f"   - {ELLIOTT_WAVE_OUTPUT} (Run elliott_wave.py first)")
        print(f"   - {SUPPORT_RESISTANCE}")

if __name__ == "__main__":
    main()
