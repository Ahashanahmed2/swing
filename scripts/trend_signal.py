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
                'date': latest_date,
                'data': symbol_data  # Store full data for debugging
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
                    # Get the two most recent high swing prices
                    # row_1 = সর্বশেষ/নতুন high swing (date বেশি)
                    # row_2 = আগের high swing (date কম)
                    row_1_price = float(high_df.iloc[0]['price'])
                    row_2_price = float(high_df.iloc[1]['price'])
                    row_1_date = high_df.iloc[0]['date']
                    row_2_date = high_df.iloc[1]['date']
                    
                    print(f"\n{symbol} - High Swings Analysis:")
                    print(f"  Row 1 (newest): Date={row_1_date}, Price={row_1_price}")
                    print(f"  Row 2 (older):  Date={row_2_date}, Price={row_2_price}")
                    print(f"  Latest Close: {latest_close} on {latest_date}")
                    
                    # UPTREND condition: 
                    # 1. row_1_price < latest_close > row_2_price
                    # 2. row_1_price < row_2_price (নতুন high কম, আগের high বেশি)
                    condition1_part1 = row_1_price < latest_close
                    condition1_part2 = latest_close > row_2_price
                    condition1 = condition1_part1 and condition1_part2
                    condition2 = row_1_price < row_2_price
                    
                    print(f"  Condition 1: {row_1_price} < {latest_close} > {row_2_price} = {condition1}")
                    print(f"  Condition 2: {row_1_price} < {row_2_price} = {condition2}")
                    
                    if condition1 and condition2:
                        uptrend_signals.append({
                            'no': len(uptrend_signals) + 1,
                            'date': latest_date,
                            'symbol': symbol,
                            'close': latest_close,
                            'row_1_price': row_1_price,
                            'row_2_price': row_2_price,
                            'row_1_date': row_1_date,
                            'row_2_date': row_2_date,
                            'condition1': f"{row_1_price} < {latest_close} > {row_2_price}",
                            'condition2': f"{row_1_price} < {row_2_price}",
                            'pattern': "Descending Highs Breakout"
                        })
                        print(f"  ✓ UPTREND signal for {symbol} - DESCENDING HIGHS BREAKOUT")
                    else:
                        print(f"  ✗ No uptrend signal for {symbol}")
                        
            except Exception as e:
                print(f"  Error reading high.csv for {symbol}: {e}")
        
        # Check DOWNTREND condition from low.csv
        if has_low:
            try:
                low_df = pd.read_csv(low_file)
                
                # Check if low.csv has at least 2 rows
                if len(low_df) >= 2:
                    # Get the two most recent low swing prices
                    # row_1 = সর্বশেষ/নতুন low swing (date বেশি)
                    # row_2 = আগের low swing (date কম)
                    row_1_price = float(low_df.iloc[0]['price'])
                    row_2_price = float(low_df.iloc[1]['price'])
                    row_1_date = low_df.iloc[0]['date']
                    row_2_date = low_df.iloc[1]['date']
                    
                    print(f"\n{symbol} - Low Swings Analysis:")
                    print(f"  Row 1 (newest): Date={row_1_date}, Price={row_1_price}")
                    print(f"  Row 2 (older):  Date={row_2_date}, Price={row_2_price}")
                    print(f"  Latest Close: {latest_close} on {latest_date}")
                    
                    # DOWNTREND condition:
                    # 1. row_1_price > latest_close < row_2_price
                    # 2. row_1_price > row_2_price (নতুন low বেশি, আগের low কম)
                    condition1_part1 = row_1_price > latest_close
                    condition1_part2 = latest_close < row_2_price
                    condition1 = condition1_part1 and condition1_part2
                    condition2 = row_1_price > row_2_price
                    
                    print(f"  Condition 1: {row_1_price} > {latest_close} < {row_2_price} = {condition1}")
                    print(f"  Condition 2: {row_1_price} > {row_2_price} = {condition2}")
                    
                    if condition1 and condition2:
                        downtrend_signals.append({
                            'no': len(downtrend_signals) + 1,
                            'date': latest_date,
                            'symbol': symbol,
                            'close': latest_close,
                            'row_1_price': row_1_price,
                            'row_2_price': row_2_price,
                            'row_1_date': row_1_date,
                            'row_2_date': row_2_date,
                            'condition1': f"{row_1_price} > {latest_close} < {row_2_price}",
                            'condition2': f"{row_1_price} > {row_2_price}",
                            'pattern': "Ascending Lows Breakdown"
                        })
                        print(f"  ✓ DOWNTREND signal for {symbol} - ASCENDING LOWS BREAKDOWN")
                    else:
                        print(f"  ✗ No downtrend signal for {symbol}")
                        
            except Exception as e:
                print(f"  Error reading low.csv for {symbol}: {e}")
    
    # Save uptrend signals to CSV
    if uptrend_signals:
        uptrend_df = pd.DataFrame(uptrend_signals)
        # Keep only necessary columns for CSV
        csv_columns = ['no', 'date', 'symbol', 'close', 'row_1_price', 'row_2_price', 'pattern']
        uptrend_df[csv_columns].to_csv(uptrend_file, index=False)
        print(f"\n✅ Saved {len(uptrend_signals)} uptrend signals to {uptrend_file}")
        
        # Display sample
        print("\nSample uptrend signals (Descending Highs Breakout):")
        print(uptrend_df[['no', 'symbol', 'close', 'row_1_price', 'row_2_price', 'pattern']].head().to_string(index=False))
    else:
        print(f"\n❌ No uptrend signals found")
    
    # Save downtrend signals to CSV
    if downtrend_signals:
        downtrend_df = pd.DataFrame(downtrend_signals)
        # Keep only necessary columns for CSV
        csv_columns = ['no', 'date', 'symbol', 'close', 'row_1_price', 'row_2_price', 'pattern']
        downtrend_df[csv_columns].to_csv(downtrend_file, index=False)
        print(f"\n✅ Saved {len(downtrend_signals)} downtrend signals to {downtrend_file}")
        
        # Display sample
        print("\nSample downtrend signals (Ascending Lows Breakdown):")
        print(downtrend_df[['no', 'symbol', 'close', 'row_1_price', 'row_2_price', 'pattern']].head().to_string(index=False))
    else:
        print(f"\n❌ No downtrend signals found")
    
    # Create summary report
    summary_file = os.path.join(output_dir, 'trend_signals_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("📊 TREND BREAKOUT/BREAKDOWN SIGNALS REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"📅 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"📁 Data Source: {mongodb_csv}\n")
        f.write(f"📈 Swing Data: {trand_base_dir}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("📊 EXECUTIVE SUMMARY:\n")
        f.write("-" * 45 + "\n")
        f.write(f"• Total symbols analyzed: {len(latest_data)}\n")
        f.write(f"• Descending Highs Breakout (Uptrend): {len(uptrend_signals)}\n")
        f.write(f"• Ascending Lows Breakdown (Downtrend): {len(downtrend_signals)}\n")
        success_rate = ((len(uptrend_signals) + len(downtrend_signals)) / len(latest_data) * 100) if len(latest_data) > 0 else 0
        f.write(f"• Signal detection rate: {success_rate:.1f}%\n\n")
        
        if uptrend_signals:
            f.write("🟢 DESCENDING HIGHS BREAKOUT SIGNALS (UPTREND):\n")
            f.write("=" * 50 + "\n")
            f.write("Pattern: New High < Old High, but Close breaks above both\n")
            f.write("-" * 50 + "\n")
            for signal in uptrend_signals:
                date_str = signal['date'].strftime('%Y-%m-%d') if hasattr(signal['date'], 'strftime') else signal['date']
                f.write(f"{signal['no']:2d}. {signal['symbol']:<8} Close: {signal['close']:>10.2f}\n")
                f.write(f"     Latest High: {signal['row_1_price']:>8.2f} (New)\n")
                f.write(f"     Previous High: {signal['row_2_price']:>7.2f} (Old)\n")
                f.write(f"     Date: {date_str}\n")
            f.write("\n")
        
        if downtrend_signals:
            f.write("🔴 ASCENDING LOWS BREAKDOWN SIGNALS (DOWNTREND):\n")
            f.write("=" * 50 + "\n")
            f.write("Pattern: New Low > Old Low, but Close breaks below both\n")
            f.write("-" * 50 + "\n")
            for signal in downtrend_signals:
                date_str = signal['date'].strftime('%Y-%m-%d') if hasattr(signal['date'], 'strftime') else signal['date']
                f.write(f"{signal['no']:2d}. {signal['symbol']:<8} Close: {signal['close']:>10.2f}\n")
                f.write(f"     Latest Low: {signal['row_1_price']:>9.2f} (New)\n")
                f.write(f"     Previous Low: {signal['row_2_price']:>8.2f} (Old)\n")
                f.write(f"     Date: {date_str}\n")
            f.write("\n")
        
        f.write("📈 PATTERN EXPLANATION:\n")
        f.write("-" * 45 + "\n")
        f.write("🟢 UPTREND (Descending Highs Breakout):\n")
        f.write("   • Recent swing high is LOWER than previous swing high\n")
        f.write("   • But current price breaks ABOVE both swing highs\n")
        f.write("   • Indicates potential trend reversal to bullish\n\n")
        
        f.write("🔴 DOWNTREND (Ascending Lows Breakdown):\n")
        f.write("   • Recent swing low is HIGHER than previous swing low\n")
        f.write("   • But current price breaks BELOW both swing lows\n")
        f.write("   • Indicates potential trend reversal to bearish\n\n")
        
        f.write("📊 MARKET INSIGHTS:\n")
        f.write("-" * 45 + "\n")
        if len(uptrend_signals) > 0 or len(downtrend_signals) > 0:
            if len(uptrend_signals) > len(downtrend_signals):
                f.write("↗️  More breakout signals than breakdown signals\n")
                f.write("   Potential BULLISH bias in market\n")
            elif len(downtrend_signals) > len(uptrend_signals):
                f.write("↘️  More breakdown signals than breakout signals\n")
                f.write("   Potential BEARISH bias in market\n")
            else:
                f.write("➡️  Equal breakout and breakdown signals\n")
                f.write("   Market in BALANCE/UNCERTAINTY\n")
        else:
            f.write("📉 No clear breakout/breakdown patterns detected\n")
            f.write("   Market may be in consolidation phase\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("📋 FILES GENERATED:\n")
        f.write(f"• Descending Highs Breakouts: {uptrend_file}\n")
        f.write(f"• Ascending Lows Breakdowns: {downtrend_file}\n")
        f.write(f"• Summary report: {summary_file}\n")
        f.write("=" * 70 + "\n")
        f.write("✅ Trend breakout/breakdown detection completed!\n")
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
    
    print("\n🎯 Trend breakout/breakdown detection completed!")

def main():
    """
    Main function to execute trend signal detection
    """
    print("=" * 70)
    print("TREND BREAKOUT/BREAKDOWN DETECTION SCRIPT")
    print("=" * 70)
    
    create_uptrend_downtrend_signals()

if __name__ == "__main__":
    main()