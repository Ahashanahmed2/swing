# final_trading_system.py - Complete Trading System
# Features:
# 1. XGBoost predictions with GOOD/BAD model filtering
# 2. Elliott Wave signal filtering (only for GOOD XGBoost models)
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
FINAL_SIGNALS = os.path.join(CSV_FOLDER, "final_trading_signals.csv")

os.makedirs(CSV_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

# Signal thresholds
XGB_MIN_CONFIDENCE = 60  # XGBoost minimum confidence for BUY
ELLIOTT_MIN_CONFIDENCE = 70  # Elliott Wave minimum confidence
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
    Load Elliott Wave signals from CSV (if exists)
    Or create from your PDF data
    """
    elliott_file = os.path.join(CSV_FOLDER, "elliott_signals.csv")
    
    if os.path.exists(elliott_file):
        df = pd.read_csv(elliott_file)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    # Manual Elliott signals from your PDF (updated with XGBoost filtering)
    elliott_data = {
        '1STPRIMFMF': {'signal': 'BUY', 'confidence': 83.4, 'wave': 'Impulse Bullish', 'price': 18.4},
        'AAMRANET': {'signal': 'BUY', 'confidence': 56.0, 'wave': 'Impulse Bullish', 'price': 16.8},
        'CITYGENINS': {'signal': 'BUY', 'confidence': 85.6, 'wave': 'Impulse Bullish', 'price': 99.2},
        'HFL': {'signal': 'BUY', 'confidence': 90.3, 'wave': 'Impulse Bullish', 'price': 12.4},
        'DOMINAGE': {'signal': 'BUY', 'confidence': 80.8, 'wave': 'Impulse Bullish', 'price': 42.9},
        'ACMEPL': {'signal': 'BUY', 'confidence': 82.4, 'wave': 'Impulse Bullish', 'price': 21.2},
        'BDTHAIFOOD': {'signal': 'BUY', 'confidence': 72.2, 'wave': 'Impulse Bullish', 'price': 19.2},
        'BBSCABLES': {'signal': 'BUY', 'confidence': 72.7, 'wave': 'Impulse Bullish', 'price': 19.3},
        'YPL': {'signal': 'BUY', 'confidence': 77.2, 'wave': 'Impulse Bullish', 'price': 18.7},
        'SUNLIFEINS': {'signal': 'BUY', 'confidence': 70.6, 'wave': 'Impulse Bullish', 'price': 60.2},
        'DSSL': {'signal': 'BUY', 'confidence': 73.3, 'wave': 'Impulse Bullish', 'price': 9.7},
        'LEGACYFOOT': {'signal': 'BUY', 'confidence': 72.5, 'wave': 'Impulse Bullish', 'price': 61.0},
        'EXIM1STMF': {'signal': 'BUY', 'confidence': 80.0, 'wave': 'Impulse Bullish', 'price': 3.5},
        'SALVO': {'signal': 'BUY', 'confidence': 72.5, 'wave': 'Impulse Bullish', 'price': 34.1},
    }
    
    df = pd.DataFrame([{
        'symbol': sym,
        'date': datetime.now(),
        'signal': data['signal'],
        'confidence': data['confidence'],
        'wave': data['wave'],
        'price': data['price']
    } for sym, data in elliott_data.items()])
    
    return df

# =========================
# SIGNAL FILTERING FUNCTIONS
# =========================

def filter_by_model_quality(xgb_df, metadata):
    """Filter only GOOD XGBoost models"""
    if metadata is None:
        return xgb_df
    
    good_symbols = metadata[metadata['status'] == 'GOOD']['symbol'].tolist()
    filtered_df = xgb_df[xgb_df['symbol'].isin(good_symbols)]
    
    print(f"   📊 Model filtering: {len(xgb_df)} → {len(filtered_df)} (GOOD models only)")
    return filtered_df

def filter_by_xgb_confidence(xgb_df, min_confidence=60):
    """Filter by XGBoost confidence score"""
    filtered_df = xgb_df[xgb_df['confidence_score'] >= min_confidence]
    print(f"   📊 XGBoost confidence filter (>={min_confidence}%): {len(filtered_df)} signals")
    return filtered_df

def filter_by_elliott_match(xgb_df, elliott_df):
    """Keep only stocks that have Elliott Wave BUY signals"""
    if elliott_df is None or elliott_df.empty:
        print(f"   ⚠️ No Elliott signals found")
        return xgb_df
    
    elliott_buy = elliott_df[elliott_df['signal'] == 'BUY']['symbol'].tolist()
    filtered_df = xgb_df[xgb_df['symbol'].isin(elliott_buy)]
    
    print(f"   📊 Elliott Wave matching: {len(xgb_df)} → {len(filtered_df)} signals")
    return filtered_df

def filter_by_support_resistance(signals_df, sr_df):
    """Check if price is near support/resistance levels"""
    if sr_df is None or sr_df.empty:
        print(f"   ⚠️ No Support/Resistance data")
        return signals_df
    
    # Get latest support/resistance
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
        price = row['close']
        
        if symbol in sr_levels:
            sr_info = sr_levels[symbol]
            distance = abs(price - sr_info['level']) / price * 100
            
            if sr_info['type'] == 'support' and price > sr_info['level'] and distance < SR_DISTANCE_THRESHOLD:
                row['sr_confirmation'] = f"Near support at {sr_info['level']} ({sr_info['strength']})"
                row['sr_bonus'] = 5 if sr_info['strength'] == 'Strong' else (2 if sr_info['strength'] == 'Moderate' else 1)
                signals_with_sr.append(row)
            elif sr_info['type'] == 'resistance' and price < sr_info['level'] and distance < SR_DISTANCE_THRESHOLD:
                row['sr_confirmation'] = f"Below resistance at {sr_info['level']} ({sr_info['strength']})"
                row['sr_bonus'] = 3 if sr_info['strength'] == 'Strong' else 1
                signals_with_sr.append(row)
            else:
                row['sr_confirmation'] = "No significant S/R near"
                row['sr_bonus'] = 0
                signals_with_sr.append(row)
        else:
            row['sr_confirmation'] = "No S/R data"
            row['sr_bonus'] = 0
            signals_with_sr.append(row)
    
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
    
    # XGBoost score
    xgb_score = row.get('confidence_score', 50) / 100
    
    # Elliott score (if available)
    if 'elliott_confidence' in row and pd.notna(row.get('elliott_confidence')):
        elliott_score = row.get('elliott_confidence', 50) / 100
    else:
        elliott_score = 0.5
    
    # Support/Resistance bonus
    sr_bonus = row.get('sr_bonus', 0) / 10  # Max 0.5 bonus (5/10)
    
    # Calculate final score
    final_score = (xgb_score * xgb_weight + elliott_score * elliott_weight + sr_bonus * sr_weight) * 100
    
    return min(100, final_score)  # Cap at 100

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
    elliott_df = load_elliott_signals()
    sr_df = load_support_resistance()
    
    # Get latest predictions
    latest_date = xgb_df['date'].max()
    latest_xgb = xgb_df[xgb_df['date'] == latest_date].copy()
    print(f"   ✅ XGBoost loaded: {len(latest_xgb)} predictions for {latest_date.date()}")
    
    # Step 2: Filter by model quality (only GOOD models)
    print("\n🔍 Step 2: Filtering by model quality...")
    filtered_df = filter_by_model_quality(latest_xgb, metadata)
    
    # Step 3: Filter by XGBoost confidence
    print("\n🎯 Step 3: Filtering by XGBoost confidence...")
    filtered_df = filter_by_xgb_confidence(filtered_df, XGB_MIN_CONFIDENCE)
    
    # Step 4: Filter by Elliott Wave (only BUY signals)
    print("\n🌊 Step 4: Filtering by Elliott Wave...")
    filtered_df = filter_by_elliott_match(filtered_df, elliott_df)
    
    # Step 5: Add Elliott confidence to signals
    if elliott_df is not None:
        elliott_dict = elliott_df.set_index('symbol')[['confidence', 'wave']].to_dict('index')
        for sym, row in filtered_df.iterrows():
            symbol = row['symbol']
            if symbol in elliott_dict:
                filtered_df.loc[sym, 'elliott_confidence'] = elliott_dict[symbol]['confidence']
                filtered_df.loc[sym, 'elliott_wave'] = elliott_dict[symbol]['wave']
            else:
                filtered_df.loc[sym, 'elliott_confidence'] = 50
                filtered_df.loc[sym, 'elliott_wave'] = 'Unknown'
    
    # Step 6: Check Support/Resistance
    print("\n📊 Step 5: Checking Support/Resistance...")
    filtered_df = filter_by_support_resistance(filtered_df, sr_df)
    
    # Step 7: Calculate final scores
    print("\n📈 Step 6: Calculating final scores...")
    filtered_df['final_score'] = filtered_df.apply(calculate_final_score, axis=1)
    
    # Step 8: Sort and limit signals
    filtered_df = filtered_df.sort_values('final_score', ascending=False)
    final_signals = filtered_df.head(MAX_SIGNALS_PER_DAY).copy()
    
    # Add execution levels
    final_signals['entry_price'] = final_signals['close']
    final_signals['stop_loss'] = final_signals['close'] * 0.97  # -3%
    final_signals['take_profit'] = final_signals['close'] * 1.05  # +5%
    
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
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    
    if signals_df is None or signals_df.empty:
        print("⚠️ No signals to generate PDF")
        return
    
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
        pdf.cell(col_widths[0], 8, row['symbol'], border=1, align='C')
        pdf.cell(col_widths[1], 8, f"{row['close']:.2f}", border=1, align='C')
        pdf.cell(col_widths[2], 8, 'BUY', border=1, align='C')
        pdf.cell(col_widths[3], 8, f"{row['confidence_score']:.0f}%", border=1, align='C')
        pdf.cell(col_widths[4], 8, f"{row.get('elliott_confidence', 50):.0f}%", border=1, align='C')
        pdf.cell(col_widths[5], 8, f"{row['final_score']:.0f}%", border=1, align='C')
        pdf.cell(col_widths[6], 8, f"{row['entry_price']:.2f}", border=1, align='C')
        pdf.cell(col_widths[7], 8, f"{row['stop_loss']:.2f}", border=1, align='C')
        pdf.cell(col_widths[8], 8, f"{row['take_profit']:.2f}", border=1, align='C')
        pdf.cell(col_widths[9], 8, row['strength'], border=1, align='C')
        pdf.ln()
        
        if pdf.get_y() > 250:
            pdf.add_page()
    
    # Save PDF
    pdf_path = os.path.join(PDF_FOLDER, f"trading_signals_{datetime.now().strftime('%Y%m%d')}.pdf")
    pdf.output(pdf_path)
    print(f"✅ PDF saved: {pdf_path}")
    
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
    print(f"   🔥 STRONG: {len(signals_df[signals_df['strength'] == 'STRONG'])}")
    print(f"   📊 MEDIUM: {len(signals_df[signals_df['strength'] == 'MEDIUM'])}")
    print(f"   ⚠️ WEAK: {len(signals_df[signals_df['strength'] == 'WEAK'])}")
    
    print("\n🔥 TOP 10 STRONGEST SIGNALS:")
    print("-"*70)
    
    for i, row in signals_df.head(10).iterrows():
        print(f"\n{i+1}. *{row['symbol']}* @ {row['close']:.2f}")
        print(f"   📊 XGBoost: {row['confidence_score']:.0f}%")
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
            print("⚠️ Telegram credentials not found")
            return
        
        if signals_df is None or signals_df.empty:
            return
        
        message = "📊 *FINAL TRADING SIGNALS* 📊\n\n"
        message += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        message += f"📈 Total: {len(signals_df)} signals\n\n"
        message += "*TOP 5 RECOMMENDATIONS:*\n\n"
        
        for i, row in signals_df.head(5).iterrows():
            message += f"{i+1}. *{row['symbol']}* @ {row['close']:.2f}\n"
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
            print("✅ Telegram alert sent!")
        else:
            print(f"❌ Telegram failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Telegram error: {e}")

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
        print(f"📄 PDF: {pdf_path}")
        print(f"📁 CSV: {FINAL_SIGNALS}")
        print("="*70)
        
    else:
        print("\n❌ No signals generated. Check your data sources.")

if __name__ == "__main__":
    main()
