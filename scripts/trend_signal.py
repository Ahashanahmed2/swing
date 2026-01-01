import pandas as pd
import os
from datetime import datetime
import requests

# টেলিগ্রাম বট কনফিগারেশন
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    """টেলিগ্রামে টেক্সট মেসেজ পাঠানো"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ টেলিগ্রাম টোকেন বা চ্যাট আইডি সেট করা নেই!")
        return None
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ টেলিগ্রাম মেসেজ সফলভাবে পাঠানো হয়েছে!")
            return response.json()
        else:
            print(f"❌ টেলিগ্রাম মেসেজ পাঠানো ব্যর্থ: {response.text}")
            return None
    except Exception as e:
        print(f"⚠️ টেলিগ্রামে মেসেজ পাঠানোতে ত্রুটি: {e}")
        return None

def send_summary_to_telegram(summary_file):
    """সারাংশ ফাইল টেলিগ্রামে পাঠানো"""
    if not os.path.exists(summary_file):
        print(f"❌ সারাংশ ফাইল পাওয়া যায়নি: {summary_file}")
        return False
    
    try:
        # ফাইল পড়া
        with open(summary_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # HTML ফরম্যাটে কনভার্ট করা
        html_content = content.replace('\n', '\n')
        html_content = f"<pre>{html_content}</pre>"
        
        # টেলিগ্রামে পাঠানো
        print(f"📤 টেলিগ্রামে সারাংশ পাঠানো হচ্ছে...")
        return send_telegram_message(html_content)
        
    except Exception as e:
        print(f"⚠️ সারাংশ পাঠানোতে ত্রুটি: {e}")
        return False

def create_uptrend_downtrend_signals():
    """
    Create uptrend.csv and downtrend.csv based on price comparison
    """
    
    # Input file paths
    mongodb_csv = './csv/mongodb.csv'
    trand_base_dir = './csv/trand/'
    output_dir = './csv/'
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file paths
    uptrend_file = os.path.join(output_dir, 'uptrend.csv')
    downtrend_file = os.path.join(output_dir, 'downtrend.csv')
    
    # Read mongodb.csv
    print(f"Reading {mongodb_csv}...")
    try:
        mongodb_df = pd.read_csv(mongodb_csv)
    except FileNotFoundError:
        print(f"Error: {mongodb_csv} not found!")
        return
    
    # Check required columns
    required_columns = ['symbol', 'date', 'close']
    for col in required_columns:
        if col not in mongodb_df.columns:
            print(f"Error: Required column '{col}' not found in {mongodb_csv}")
            return
    
    # Convert date to datetime
    mongodb_df['date'] = pd.to_datetime(mongodb_df['date'])
    
    # Get latest close for each symbol
    print("Getting latest close prices for each symbol...")
    latest_data = {}
    
    for symbol in mongodb_df['symbol'].unique():
        symbol_data = mongodb_df[mongodb_df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date', ascending=False).reset_index(drop=True)
        
        if not symbol_data.empty:
            latest_close = symbol_data.iloc[0]['close']
            latest_date = symbol_data.iloc[0]['date']
            latest_data[symbol] = {
                'close': latest_close,
                'date': latest_date
            }
    
    print(f"Found {len(latest_data)} symbols with latest data")
    
    # Initialize lists for uptrend and downtrend signals
    uptrend_signals = []
    downtrend_signals = []
    
    # Process each symbol
    print("\nProcessing symbols for trend signals...")
    
    for symbol, latest_info in latest_data.items():
        symbol_dir = os.path.join(trand_base_dir, symbol)
        
        # Check if high.csv exists
        high_file = os.path.join(symbol_dir, 'high.csv')
        low_file = os.path.join(symbol_dir, 'low.csv')
        
        has_high = os.path.exists(high_file)
        has_low = os.path.exists(low_file)
        
        if not (has_high or has_low):
            continue  # Skip symbols without swing data
        
        latest_close = latest_info['close']
        latest_date = latest_info['date']
        
        # Check UPTREND condition from high.csv
        if has_high:
            try:
                high_df = pd.read_csv(high_file)
                
                # Check if high.csv has at least 2 rows
                if len(high_df) >= 2:
                    # Get the two most recent high swing prices (assuming sorted newest first)
                    row_1_price = high_df.iloc[0]['price'] if 'price' in high_df.columns else high_df.iloc[0]['price']
                    row_2_price = high_df.iloc[1]['price'] if 'price' in high_df.columns else high_df.iloc[1]['price']
                    
                    # UPTREND condition: row_1_price < close > row_2_price and row_1_price < row_2_price
                    # Note: Fixed condition based on your input
                    condition1 = row_1_price < latest_close > row_2_price
                    condition2 = row_1_price < row_2_price
                    
                    if condition1 and condition2:
                        uptrend_signals.append({
                            'no': len(uptrend_signals) + 1,
                            'date': latest_date,
                            'symbol': symbol,
                            'close': latest_close,
                            'row_1_price': row_1_price,
                            'row_2_price': row_2_price,
                            'condition': f"{row_1_price} < {latest_close} > {row_2_price} and {row_1_price} < {row_2_price}"
                        })
                        print(f"✓ UPTREND signal for {symbol}")
                        
            except Exception as e:
                print(f"  Error reading high.csv for {symbol}: {e}")
        
        # Check DOWNTREND condition from low.csv
        if has_low:
            try:
                low_df = pd.read_csv(low_file)
                
                # Check if low.csv has at least 2 rows
                if len(low_df) >= 2:
                    # Get the two most recent low swing prices (assuming sorted newest first)
                    row_1_price = low_df.iloc[0]['price'] if 'price' in low_df.columns else low_df.iloc[0]['price']
                    row_2_price = low_df.iloc[1]['price'] if 'price' in low_df.columns else low_df.iloc[1]['price']
                    
                    # DOWNTREND condition: row_1_price > close < row_2_price and row_1_price > row_2_price
                    condition1 = row_1_price > latest_close < row_2_price
                    condition2 = row_1_price > row_2_price
                    
                    if condition1 and condition2:
                        downtrend_signals.append({
                            'no': len(downtrend_signals) + 1,
                            'date': latest_date,
                            'symbol': symbol,
                            'close': latest_close,
                            'row_1_price': row_1_price,
                            'row_2_price': row_2_price,
                            'condition': f"{row_1_price} > {latest_close} < {row_2_price} and {row_1_price} > {row_2_price}"
                        })
                        print(f"✓ DOWNTREND signal for {symbol}")
                        
            except Exception as e:
                print(f"  Error reading low.csv for {symbol}: {e}")
    
    # Save uptrend signals to CSV
    if uptrend_signals:
        uptrend_df = pd.DataFrame(uptrend_signals)
        uptrend_df.to_csv(uptrend_file, index=False)
        print(f"\n✅ Saved {len(uptrend_signals)} uptrend signals to {uptrend_file}")
        
        # Display sample
        print("\nSample uptrend signals:")
        print(uptrend_df[['no', 'date', 'symbol', 'close']].head().to_string(index=False))
    else:
        print(f"\n❌ No uptrend signals found")
    
    # Save downtrend signals to CSV
    if downtrend_signals:
        downtrend_df = pd.DataFrame(downtrend_signals)
        downtrend_df.to_csv(downtrend_file, index=False)
        print(f"\n✅ Saved {len(downtrend_signals)} downtrend signals to {downtrend_file}")
        
        # Display sample
        print("\nSample downtrend signals:")
        print(downtrend_df[['no', 'date', 'symbol', 'close']].head().to_string(index=False))
    else:
        print(f"\n❌ No downtrend signals found")
    
    # Create summary report
    summary_file = os.path.join(output_dir, 'trend_signals_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("📊 TREND SIGNALS SUMMARY REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"📅 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"📁 Data Source: {mongodb_csv}\n")
        f.write(f"📈 Swing Data: {trand_base_dir}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("📊 EXECUTIVE SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"• Total symbols processed: {len(latest_data)}\n")
        f.write(f"• Uptrend signals found: {len(uptrend_signals)}\n")
        f.write(f"• Downtrend signals found: {len(downtrend_signals)}\n")
        f.write(f"• Success rate: {((len(uptrend_signals) + len(downtrend_signals)) / len(latest_data) * 100):.1f}%\n\n")
        
        if uptrend_signals:
            f.write("🟢 UPTREND SIGNALS:\n")
            f.write("=" * 30 + "\n")
            for signal in uptrend_signals:
                date_str = signal['date'].strftime('%Y-%m-%d') if hasattr(signal['date'], 'strftime') else signal['date']
                f.write(f"{signal['no']:2d}. {signal['symbol']:<8} Close: {signal['close']:>10.2f} Date: {date_str}\n")
            f.write("\n")
        
        if downtrend_signals:
            f.write("🔴 DOWNTREND SIGNALS:\n")
            f.write("=" * 30 + "\n")
            for signal in downtrend_signals:
                date_str = signal['date'].strftime('%Y-%m-%d') if hasattr(signal['date'], 'strftime') else signal['date']
                f.write(f"{signal['no']:2d}. {signal['symbol']:<8} Close: {signal['close']:>10.2f} Date: {date_str}\n")
            f.write("\n")
        
        f.write("📈 MARKET ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        
        if len(uptrend_signals) > len(downtrend_signals):
            f.write("↗️  Market is showing BULLISH bias\n")
            f.write(f"   Uptrend signals ({len(uptrend_signals)}) > Downtrend signals ({len(downtrend_signals)})\n")
        elif len(downtrend_signals) > len(uptrend_signals):
            f.write("↘️  Market is showing BEARISH bias\n")
            f.write(f"   Downtrend signals ({len(downtrend_signals)}) > Uptrend signals ({len(uptrend_signals)})\n")
        else:
            f.write("➡️  Market is showing NEUTRAL bias\n")
            f.write(f"   Equal uptrend ({len(uptrend_signals)}) and downtrend ({len(downtrend_signals)}) signals\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("📋 Files Generated:\n")
        f.write(f"• Uptrend signals: {uptrend_file}\n")
        f.write(f"• Downtrend signals: {downtrend_file}\n")
        f.write(f"• Summary report: {summary_file}\n")
        f.write("=" * 60 + "\n")
        f.write("✅ Trend signal detection completed!\n")
        f.write(f"⏰ Next analysis: {datetime.now().strftime('%Y-%m-%d')} 09:00:00\n")
    
    print(f"\n📊 Summary saved to: {summary_file}")
    
    # Send summary to Telegram
    print("\n📤 Sending summary to Telegram...")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        telegram_sent = send_summary_to_telegram(summary_file)
        if telegram_sent:
            print("✅ Summary sent to Telegram successfully!")
        else:
            print("⚠️ Failed to send summary to Telegram")
    else:
        print("ℹ️ Telegram credentials not set, skipping Telegram send")
    
    print("\n🎯 Trend signal detection completed!")

def main():
    """
    Main function to execute trend signal detection
    """
    print("=" * 60)
    print("TREND SIGNAL DETECTION SCRIPT")
    print("=" * 60)
    
    create_uptrend_downtrend_signals()

if __name__ == "__main__":
    main()