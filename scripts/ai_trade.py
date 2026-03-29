#!/usr/bin/env python3
"""
ai_trade.py - Automated Trading Monitor
Monitors stock entries, stop losses, and take profits based on CSV data
"""

import os
import csv
import json
import time
import schedule
import pandas as pd
import smtplib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from email.message import EmailMessage
from typing import List, Dict, Tuple, Optional
from telegram import Bot
from telegram.error import TelegramError
import asyncio
import nest_asyncio

# Apply nest_asyncio for running async in sync context
nest_asyncio.apply()

# ==================== CONFIGURATION ====================

# Directories
CSV_STOCK_DIR = "./csv/stock"
CSV_DATA_DIR = "./csv"
MONGO_DB_FILE = f"{CSV_DATA_DIR}/mongodb.csv"
ED_FILE = f"{CSV_DATA_DIR}/ed.csv"
RD_FILE = f"{CSV_DATA_DIR}/rd.csv"
SL_FILE = f"{CSV_DATA_DIR}/sl.csv"
TP_FILE = f"{CSV_DATA_DIR}/tp.csv"

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Email Configuration
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", EMAIL_USER)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_trade.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== DATA CLASSES ====================

class StockData:
    """Stock data structure"""
    def __init__(self, row: List[str], date: str):
        self.symbol = row[0].strip()
        self.entry_zone = row[3].strip()
        self.stop_loss = self._parse_price(row[4].strip())
        self.tp1 = self._parse_price(row[5].strip())
        self.tp2 = self._parse_price(row[6].strip())
        self.tp3 = self._parse_price(row[7].strip())
        self.rrr = row[8].strip()
        self.score = row[9].strip()
        self.insight = row[10].strip() if len(row) > 10 else ""
        self.wave = row[1].strip() if len(row) > 1 else ""
        self.subwave = row[2].strip() if len(row) > 2 else ""
        self.date = date
        self.original_row = row[:11]  # Store original 11 columns
        
    def _parse_price(self, price_str: str) -> float:
        """Parse price string to float"""
        try:
            # Handle ranges like "13.8-14.2"
            if '-' in price_str:
                parts = price_str.split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            return float(price_str)
        except:
            return 0.0
            
    def get_entry_range(self) -> Tuple[float, float]:
        """Get entry zone range"""
        try:
            if '-' in self.entry_zone:
                parts = self.entry_zone.split('-')
                return float(parts[0]), float(parts[1])
            entry = float(self.entry_zone)
            return entry - 0.2, entry + 0.2
        except:
            return 0, 0

class MongoDBData:
    """MongoDB data structure"""
    def __init__(self, symbol: str, date: str, close: float, high: float, low: float):
        self.symbol = symbol
        self.date = date
        self.close = close
        self.high = high
        self.low = low

# ==================== FILE OPERATIONS ====================

def ensure_directories():
    """Create necessary directories if they don't exist"""
    Path(CSV_STOCK_DIR).mkdir(parents=True, exist_ok=True)
    Path(CSV_DATA_DIR).mkdir(parents=True, exist_ok=True)

def read_csv_file(filepath: str) -> Optional[List[List[str]]]:
    """Read CSV file and return data"""
    try:
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            return list(reader)
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return None

def write_csv_file(filepath: str, data: List[List[str]]):
    """Write data to CSV file"""
    try:
        with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        logger.info(f"Written to {filepath}: {len(data)} rows")
    except Exception as e:
        logger.error(f"Error writing {filepath}: {e}")

def load_mongodb_data() -> Dict[str, List[MongoDBData]]:
    """Load MongoDB CSV and organize by symbol"""
    data = read_csv_file(MONGO_DB_FILE)
    if not data:
        return {}
    
    # Assuming MongoDB CSV has headers: symbol,date,close,high,low
    result = {}
    for row in data[1:]:  # Skip header
        if len(row) >= 5:
            symbol = row[0].strip()
            date = row[1].strip()
            try:
                close = float(row[2])
                high = float(row[3])
                low = float(row[4])
            except:
                continue
            
            if symbol not in result:
                result[symbol] = []
            result[symbol].append(MongoDBData(symbol, date, close, high, low))
    
    # Sort by date for each symbol
    for symbol in result:
        result[symbol].sort(key=lambda x: x.date)
    
    return result

def get_latest_mongodb_data(mongodb_data: Dict[str, List[MongoDBData]], symbol: str) -> Optional[MongoDBData]:
    """Get latest MongoDB data for a symbol"""
    if symbol not in mongodb_data or not mongodb_data[symbol]:
        return None
    return mongodb_data[symbol][-1]

def load_stock_files() -> List[Tuple[str, List[StockData]]]:
    """Load all stock CSV files from stock directory"""
    stock_files = []
    if not os.path.exists(CSV_STOCK_DIR):
        return stock_files
    
    for filename in os.listdir(CSV_STOCK_DIR):
        if filename.endswith('.csv'):
            filepath = os.path.join(CSV_STOCK_DIR, filename)
            date = filename.replace('.csv', '')
            data = read_csv_file(filepath)
            
            if data:
                # Find header or assume first row is data
                start_idx = 0
                if data and data[0] and data[0][0].lower() == 'symbol':
                    start_idx = 1
                
                stocks = []
                for row in data[start_idx:]:
                    if row and len(row) >= 11:
                        stocks.append(StockData(row, date))
                
                if stocks:
                    stock_files.append((date, stocks))
                    logger.info(f"Loaded {len(stocks)} stocks from {filename}")
    
    return sorted(stock_files, key=lambda x: x[0])  # Sort by date

# ==================== TRADE LOGIC ====================

def check_entry_conditions(stock: StockData, latest_data: MongoDBData) -> bool:
    """Check if price is within entry zone"""
    entry_low, entry_high = stock.get_entry_range()
    return entry_low <= latest_data.close <= entry_high or entry_low <= latest_data.low <= entry_high

def check_stop_loss(stock: StockData, latest_data: MongoDBData) -> bool:
    """Check if stop loss is hit"""
    return latest_data.close <= stock.stop_loss

def check_take_profit(stock: StockData, latest_data: MongoDBData, level: int) -> bool:
    """Check if take profit level is hit"""
    if level == 1:
        return latest_data.close >= stock.tp1 or latest_data.high >= stock.tp1
    elif level == 2:
        return latest_data.close >= stock.tp2 or latest_data.high >= stock.tp2
    elif level == 3:
        return latest_data.close >= stock.tp3 or latest_data.high >= stock.tp3
    return False

def load_existing_rd() -> Dict[str, Dict]:
    """Load existing RD data"""
    data = read_csv_file(RD_FILE)
    if not data:
        return {}
    
    result = {}
    # Assuming RD file has: symbol,date,ed_date,rd_date,original_columns...
    for row in data[1:]:  # Skip header
        if row and len(row) > 0:
            symbol = row[0]
            result[symbol] = {'row': row, 'rd_date': row[2] if len(row) > 2 else ''}
    return result

def save_to_ed(stock: StockData, latest_date: str):
    """Save entry to ED file"""
    ed_data = read_csv_file(ED_FILE)
    if not ed_data:
        # Create header
        ed_data = [['symbol', 'wave', 'subwave', 'entry_zone', 'stop_loss', 
                    'tp1', 'tp2', 'tp3', 'rrr', 'score', 'insight', 'ed_date']]
    
    # Create row with original 11 columns + ed_date
    new_row = stock.original_row + [latest_date]
    ed_data.append(new_row)
    write_csv_file(ED_FILE, ed_data)
    logger.info(f"Saved to ED: {stock.symbol} on {latest_date}")

def save_to_rd(stock: StockData, entry_date: str, running_date: str):
    """Save to RD file"""
    rd_data = read_csv_file(RD_FILE)
    if not rd_data:
        # Create header with original 11 columns + ed_date + rd_date
        rd_data = [['symbol', 'wave', 'subwave', 'entry_zone', 'stop_loss', 
                    'tp1', 'tp2', 'tp3', 'rrr', 'score', 'insight', 'ed_date', 'rd_date']]
    
    new_row = stock.original_row + [entry_date, running_date]
    rd_data.append(new_row)
    write_csv_file(RD_FILE, rd_data)
    logger.info(f"Saved to RD: {stock.symbol} from {entry_date} to {running_date}")

def update_rd_with_running_date(symbol: str, running_date: str):
    """Update RD file with latest running date"""
    rd_data = read_csv_file(RD_FILE)
    if not rd_data:
        return
    
    header = rd_data[0]
    # Find symbol in RD
    for i, row in enumerate(rd_data[1:], 1):
        if row and row[0] == symbol:
            # Update rd_date column (index 12)
            if len(row) > 12:
                row[12] = running_date
            else:
                while len(row) < 13:
                    row.append('')
                row[12] = running_date
            break
    
    write_csv_file(RD_FILE, rd_data)

def save_to_sl(stock: StockData, rd_row: List[str], rd_date: str, sl_date: str, gap: int):
    """Save to SL file"""
    sl_data = read_csv_file(SL_FILE)
    if not sl_data:
        # Create header with original 11 + ed_date + rd_date + sld_date + gap
        sl_data = [['symbol', 'wave', 'subwave', 'entry_zone', 'stop_loss', 
                    'tp1', 'tp2', 'tp3', 'rrr', 'score', 'insight', 
                    'ed_date', 'rd_date', 'sld_date', 'gap']]
    
    new_row = rd_row[:11] + [rd_row[11] if len(rd_row) > 11 else '', 
                              rd_row[12] if len(rd_row) > 12 else '',
                              sl_date, str(gap)]
    sl_data.append(new_row)
    write_csv_file(SL_FILE, sl_data)
    logger.info(f"Saved to SL: {stock.symbol} at {sl_date}")

def save_to_tp(stock: StockData, rd_row: List[str], rd_date: str, tp_date: str, 
               tp_level: int, gap: int, previous_tp_date: str = ''):
    """Save to TP file"""
    tp_data = read_csv_file(TP_FILE)
    
    if not tp_data:
        # Create header
        tp_data = [['symbol', 'wave', 'subwave', 'entry_zone', 'stop_loss', 
                    'tp1', 'tp2', 'tp3', 'rrr', 'score', 'insight', 
                    'ed_date', 'rd_date', f'tpd{tp_level}', f'gap{tp_level}']]
    
    # Check if symbol already exists with lower TP
    found = False
    for i, row in enumerate(tp_data[1:], 1):
        if row and row[0] == stock.symbol:
            # Update with new TP level
            while len(row) < 15:
                row.append('')
            row[13] = tp_date if tp_level == 1 else row[13]
            row[14] = str(gap) if tp_level == 1 else row[14]
            if tp_level == 2:
                # Add tpd2 and gap2 columns if needed
                if len(row) < 16:
                    row.append('')
                if len(row) < 17:
                    row.append('')
                row[15] = tp_date
                row[16] = str(gap)
            elif tp_level == 3:
                if len(row) < 17:
                    row.append('')
                if len(row) < 18:
                    row.append('')
                row[17] = tp_date
                row[18] = str(gap)
            found = True
            break
    
    if not found:
        new_row = rd_row[:11] + [rd_row[11] if len(rd_row) > 11 else '', 
                                  rd_row[12] if len(rd_row) > 12 else '',
                                  tp_date if tp_level == 1 else '',
                                  str(gap) if tp_level == 1 else '']
        if tp_level == 2:
            new_row.append(tp_date)
            new_row.append(str(gap))
            new_row.append('')
            new_row.append('')
        elif tp_level == 3:
            new_row.append('')
            new_row.append('')
            new_row.append(tp_date)
            new_row.append(str(gap))
        else:
            new_row.append('')
            new_row.append('')
            new_row.append('')
            new_row.append('')
        tp_data.append(new_row)
    
    write_csv_file(TP_FILE, tp_data)
    logger.info(f"Saved to TP: {stock.symbol} TP{tp_level} at {tp_date}")

def remove_old_records():
    """Remove records older than 1 year"""
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%d-%m-%Y")
    
    for filepath in [ED_FILE, RD_FILE, SL_FILE, TP_FILE]:
        data = read_csv_file(filepath)
        if not data or len(data) <= 1:
            continue
        
        header = data[0]
        new_data = [header]
        
        # Find date column index
        date_col = -1
        if 'ed_date' in header:
            date_col = header.index('ed_date')
        elif 'rd_date' in header:
            date_col = header.index('rd_date')
        elif 'sld_date' in header:
            date_col = header.index('sld_date')
        elif 'tpd1' in header:
            date_col = header.index('tpd1')
        
        if date_col == -1:
            continue
        
        # Keep only records from last year
        for row in data[1:]:
            if len(row) > date_col:
                row_date = row[date_col]
                if row_date >= one_year_ago:
                    new_data.append(row)
        
        if len(new_data) != len(data):
            write_csv_file(filepath, new_data)
            logger.info(f"Removed old records from {filepath}: {len(data) - len(new_data)} rows")

# ==================== NOTIFICATION ====================

async def send_telegram_message(message: str):
    """Send message via Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set")
        return
    
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
        logger.info("Telegram message sent")
    except TelegramError as e:
        logger.error(f"Telegram error: {e}")

def send_email(subject: str, body: str):
    """Send email notification"""
    if not EMAIL_USER or not EMAIL_PASS:
        logger.warning("Email credentials not set")
        return
    
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_RECIPIENT
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
        logger.info("Email sent")
    except Exception as e:
        logger.error(f"Email error: {e}")

def generate_notification(stock: StockData, action: str, price: float, date: str, gap: int = None):
    """Generate notification message"""
    if action == "entry":
        message = f"""
🔔 <b>ENTRY SIGNAL</b> 🔔

📊 <b>Symbol:</b> {stock.symbol}
💰 <b>Entry Zone:</b> {stock.entry_zone}
🎯 <b>Current Price:</b> {price}
📅 <b>Date:</b> {date}
⭐ <b>Score:</b> {stock.score}/100

📈 <b>Wave:</b> {stock.wave} → {stock.subwave}
🎯 <b>Targets:</b> {stock.tp1} → {stock.tp2} → {stock.tp3}
🛑 <b>Stop Loss:</b> {stock.stop_loss}
📊 <b>RRR:</b> {stock.rrr}

💡 <b>Insight:</b> {stock.insight[:100]}...
"""
    elif action == "stop_loss":
        message = f"""
⚠️ <b>STOP LOSS HIT</b> ⚠️

📊 <b>Symbol:</b> {stock.symbol}
💰 <b>Stop Loss:</b> {stock.stop_loss}
🎯 <b>Exit Price:</b> {price}
📅 <b>Date:</b> {date}
📈 <b>Holding Period:</b> {gap} days

📊 <b>Wave:</b> {stock.wave}
⭐ <b>Score:</b> {stock.score}/100
"""
    elif action == "take_profit":
        message = f"""
✅ <b>TAKE PROFIT HIT - TP{action.split('_')[1] if '_' in action else '1'}</b> ✅

📊 <b>Symbol:</b> {stock.symbol}
💰 <b>Target Price:</b> {price}
🎯 <b>Exit Price:</b> {price}
📅 <b>Date:</b> {date}
📈 <b>Holding Period:</b> {gap} days

📊 <b>Wave:</b> {stock.wave}
⭐ <b>Score:</b> {stock.score}/100
"""
    else:
        return
    
    # Send both Telegram and Email
    asyncio.run(send_telegram_message(message))
    send_email(f"Trade Alert: {stock.symbol} - {action}", message)
    
    return message

# ==================== MAIN TRADE MONITOR ====================

def monitor_trades():
    """Main monitoring function"""
    logger.info("=" * 50)
    logger.info("Starting trade monitoring...")
    
    # Load all data
    mongodb_data = load_mongodb_data()
    stock_files = load_stock_files()
    existing_rd = load_existing_rd()
    
    if not mongodb_data:
        logger.warning("No MongoDB data found")
        return
    
    # Process each stock file
    for file_date, stocks in stock_files:
        logger.info(f"Processing {file_date} with {len(stocks)} stocks")
        
        for stock in stocks:
            latest_data = get_latest_mongodb_data(mongodb_data, stock.symbol)
            if not latest_data:
                continue
            
            # Check if already in RD
            in_rd = stock.symbol in existing_rd
            
            # Check entry condition
            if not in_rd and check_entry_conditions(stock, latest_data):
                logger.info(f"Entry signal for {stock.symbol} at {latest_data.close}")
                save_to_ed(stock, latest_data.date)
                save_to_rd(stock, file_date, latest_data.date)
                generate_notification(stock, "entry", latest_data.close, latest_data.date)
                existing_rd[stock.symbol] = {'row': None, 'rd_date': latest_data.date}
            
            # Check existing trades in RD
            elif in_rd:
                rd_info = existing_rd[stock.symbol]
                rd_date = rd_info['rd_date']
                
                # Calculate gap days
                try:
                    rd_dt = datetime.strptime(rd_date, "%d-%m-%Y")
                    current_dt = datetime.strptime(latest_data.date, "%d-%m-%Y")
                    gap = (current_dt - rd_dt).days
                except:
                    gap = 0
                
                # Check stop loss
                if check_stop_loss(stock, latest_data):
                    logger.info(f"Stop loss hit for {stock.symbol} at {latest_data.close}")
                    # Get RD row data
                    rd_data = read_csv_file(RD_FILE)
                    rd_row = None
                    if rd_data:
                        for row in rd_data[1:]:
                            if row and row[0] == stock.symbol:
                                rd_row = row
                                break
                    
                    if rd_row:
                        save_to_sl(stock, rd_row, rd_date, latest_data.date, gap)
                        generate_notification(stock, "stop_loss", latest_data.close, latest_data.date, gap)
                    
                    # Remove from RD
                    # Could implement removal logic here
                
                # Check take profits (in order)
                elif check_take_profit(stock, latest_data, 3):
                    logger.info(f"TP3 hit for {stock.symbol} at {latest_data.close}")
                    # Get RD row
                    rd_data = read_csv_file(RD_FILE)
                    rd_row = None
                    if rd_data:
                        for row in rd_data[1:]:
                            if row and row[0] == stock.symbol:
                                rd_row = row
                                break
                    
                    if rd_row:
                        save_to_tp(stock, rd_row, rd_date, latest_data.date, 3, gap)
                        generate_notification(stock, "take_profit_3", latest_data.close, latest_data.date, gap)
                
                elif check_take_profit(stock, latest_data, 2):
                    logger.info(f"TP2 hit for {stock.symbol} at {latest_data.close}")
                    rd_data = read_csv_file(RD_FILE)
                    rd_row = None
                    if rd_data:
                        for row in rd_data[1:]:
                            if row and row[0] == stock.symbol:
                                rd_row = row
                                break
                    
                    if rd_row:
                        save_to_tp(stock, rd_row, rd_date, latest_data.date, 2, gap)
                        generate_notification(stock, "take_profit_2", latest_data.close, latest_data.date, gap)
                
                elif check_take_profit(stock, latest_data, 1):
                    logger.info(f"TP1 hit for {stock.symbol} at {latest_data.close}")
                    rd_data = read_csv_file(RD_FILE)
                    rd_row = None
                    if rd_data:
                        for row in rd_data[1:]:
                            if row and row[0] == stock.symbol:
                                rd_row = row
                                break
                    
                    if rd_row:
                        save_to_tp(stock, rd_row, rd_date, latest_data.date, 1, gap)
                        generate_notification(stock, "take_profit_1", latest_data.close, latest_data.date, gap)
                
                # Update RD with latest running date
                else:
                    update_rd_with_running_date(stock.symbol, latest_data.date)
    
    # Remove old records
    remove_old_records()
    
    logger.info("Trade monitoring completed")
    logger.info("=" * 50)

def generate_summary_report():
    """Generate and send summary report"""
    logger.info("Generating summary report...")
    
    # Load all files
    ed_data = read_csv_file(ED_FILE)
    rd_data = read_csv_file(RD_FILE)
    sl_data = read_csv_file(SL_FILE)
    tp_data = read_csv_file(TP_FILE)
    
    # Calculate stats
    stats = {
        'total_entries': len(ed_data) - 1 if ed_data else 0,
        'active_trades': len(rd_data) - 1 if rd_data else 0,
        'stop_loss_hits': len(sl_data) - 1 if sl_data else 0,
        'take_profit_hits': len(tp_data) - 1 if tp_data else 0,
    }
    
    # Calculate win rate
    total_closed = stats['stop_loss_hits'] + stats['take_profit_hits']
    win_rate = (stats['take_profit_hits'] / total_closed * 100) if total_closed > 0 else 0
    
    # Generate message
    message = f"""
📊 <b>TRADING SUMMARY REPORT</b>
📅 <b>Date:</b> {datetime.now().strftime('%d-%m-%Y %H:%M')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 <b>PERFORMANCE METRICS</b>

• Total Entries: {stats['total_entries']}
• Active Trades: {stats['active_trades']}
• Stop Loss Hits: {stats['stop_loss_hits']}
• Take Profit Hits: {stats['take_profit_hits']}
• Win Rate: {win_rate:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 <b>TOP PERFORMERS</b>
"""
    
    # Add top performers from TP file
    if tp_data and len(tp_data) > 1:
        # Sort by highest TP level
        tp_stats = {}
        for row in tp_data[1:]:
            if row and len(row) > 0:
                symbol = row[0]
                tp_stats[symbol] = tp_stats.get(symbol, 0) + 1
        
        top_symbols = sorted(tp_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        for sym, count in top_symbols:
            message += f"• {sym}: {count} TP hits\n"
    
    message += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ <b>RISK METRICS</b>
"""
    
    # Add risk metrics from SL file
    if sl_data and len(sl_data) > 1:
        message += f"• Total Losses: {stats['stop_loss_hits']}\n"
        message += f"• Loss Rate: {(stats['stop_loss_hits'] / (stats['stop_loss_hits'] + stats['take_profit_hits']) * 100) if (stats['stop_loss_hits'] + stats['take_profit_hits']) > 0 else 0:.1f}%\n"
    
    # Send report
    asyncio.run(send_telegram_message(message))
    send_email("Trading Summary Report", message.replace('<b>', '').replace('</b>', ''))
    
    logger.info("Summary report sent")

# ==================== SCHEDULER ====================

def run_scheduler():
    """Run scheduled tasks"""
    # Run every 5 minutes during market hours
    schedule.every(5).minutes.do(monitor_trades)
    
    # Run summary report daily at 6 PM
    schedule.every().day.at("18:00").do(generate_summary_report)
    
    # Clean old records daily at midnight
    schedule.every().day.at("00:00").do(remove_old_records)
    
    logger.info("Scheduler started")
    
    while True:
        schedule.run_pending()
        time.sleep(1)

# ==================== MAIN ====================

def main():
    """Main function"""
    print("=" * 60)
    print("AI TRADE MONITOR - Automated Trading System")
    print("=" * 60)
    print(f"Stock Directory: {CSV_STOCK_DIR}")
    print(f"Data Directory: {CSV_DATA_DIR}")
    print(f"Telegram: {'✓' if TELEGRAM_BOT_TOKEN else '✗'}")
    print(f"Email: {'✓' if EMAIL_USER else '✗'}")
    print("=" * 60)
    
    # Ensure directories exist
    ensure_directories()
    
    # Run initial scan
    monitor_trades()
    
    # Start scheduler
    try:
        run_scheduler()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        print("\nShutting down AI Trade Monitor...")

if __name__ == "__main__":
    main()