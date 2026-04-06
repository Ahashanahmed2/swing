# ai_trade.py - GitHub Actions এর জন্য অপটিমাইজড (শুধু একবার রান করে)

import os
import csv
import time
import logging
import smtplib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from email.message import EmailMessage
from dataclasses import dataclass
from collections import defaultdict

# ==================== CONFIGURATION ====================

# Directories
CSV_STOCK_DIR = "./csv/stock"
CSV_DATA_DIR = "./csv"
MONGO_DB_FILE = f"{CSV_DATA_DIR}/mongodb.csv"
ED_FILE = f"{CSV_DATA_DIR}/ed.csv"
RD_FILE = f"{CSV_DATA_DIR}/rd.csv"
SL_FILE = f"{CSV_DATA_DIR}/sl.csv"
TP_FILE = f"{CSV_DATA_DIR}/tp.csv"

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

# CSV Headers
RD_HEADERS = ['symbol', 'wave', 'subwave', 'entry_zone', 'stop_loss', 
              'tp1', 'tp2', 'tp3', 'rrr', 'score', 'insight', 'ed_date', 'rd_date']

SL_HEADERS = ['symbol', 'wave', 'subwave', 'entry_zone', 'stop_loss', 
              'tp1', 'tp2', 'tp3', 'rrr', 'score', 'insight', 
              'ed_date', 'rd_date', 'sld_date', 'gap']

TP_HEADERS = ['symbol', 'wave', 'subwave', 'entry_zone', 'stop_loss', 
              'tp1', 'tp2', 'tp3', 'rrr', 'score', 'insight', 
              'ed_date', 'rd_date', 'tpd1', 'gap1', 'tpd2', 'gap2', 'tpd3', 'gap3']

ED_HEADERS = ['symbol', 'wave', 'subwave', 'entry_zone', 'stop_loss', 
              'tp1', 'tp2', 'tp3', 'rrr', 'score', 'insight', 'ed_date']


# ==================== DATA CLASSES ====================

@dataclass
class StockData:
    """Stock data structure - date ফাইলের নাম থেকে আসে"""
    symbol: str
    entry_zone: str
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    rrr: str
    score: str
    insight: str
    wave: str
    subwave: str
    date: str  # ফাইলের নাম থেকে আসা তারিখ (dd-mm-yyyy)
    original_row: List[str]
    
    @classmethod
    def from_row(cls, row: List[str], date: str):
        """Create StockData from CSV row - date প্যারামিটার ফাইলের নাম থেকে আসে"""
        def parse_price(price_str: str) -> float:
            try:
                if '-' in price_str:
                    parts = price_str.split('-')
                    return (float(parts[0]) + float(parts[1])) / 2
                return float(price_str) if price_str else 0.0
            except:
                return 0.0
        
        return cls(
            symbol=row[0].strip(),
            entry_zone=row[3].strip() if len(row) > 3 else "",
            stop_loss=parse_price(row[4].strip() if len(row) > 4 else ""),
            tp1=parse_price(row[5].strip() if len(row) > 5 else ""),
            tp2=parse_price(row[6].strip() if len(row) > 6 else ""),
            tp3=parse_price(row[7].strip() if len(row) > 7 else ""),
            rrr=row[8].strip() if len(row) > 8 else "",
            score=row[9].strip() if len(row) > 9 else "",
            insight=row[10].strip() if len(row) > 10 else "",
            wave=row[1].strip() if len(row) > 1 else "",
            subwave=row[2].strip() if len(row) > 2 else "",
            date=date,
            original_row=row[:11]
        )
    
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
    
    def get_score_emoji(self) -> str:
        """Get score emoji"""
        try:
            score_num = int(self.score)
            if score_num >= 85:
                return "💎"
            elif score_num >= 80:
                return "🔥"
            elif score_num >= 70:
                return "⭐"
            elif score_num >= 60:
                return "✅"
            elif score_num >= 50:
                return "📈"
            elif score_num >= 40:
                return "⚠️"
            else:
                return "❌"
        except:
            return "⭐"


@dataclass
class MongoDBData:
    """MongoDB data structure"""
    symbol: str
    date: str
    close: float
    high: float
    low: float


# ==================== SAFE FILE OPERATIONS ====================

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


# ==================== DATA LOADING ====================

def load_mongodb_data() -> Dict[str, List[MongoDBData]]:
    """Load MongoDB CSV and organize by symbol"""
    data = read_csv_file(MONGO_DB_FILE)
    if not data:
        logger.warning("No MongoDB data found")
        return {}

    result = defaultdict(list)
    for row in data[1:]:  # Skip header
        if len(row) >= 5:
            try:
                symbol = row[0].strip()
                date = row[1].strip()
                close = float(row[2]) if row[2] else 0
                high = float(row[3]) if row[3] else 0
                low = float(row[4]) if row[4] else 0
                
                result[symbol].append(MongoDBData(symbol, date, close, high, low))
            except (ValueError, IndexError) as e:
                logger.debug(f"Skipping row: {e}")
                continue

    # Sort by date for each symbol
    for symbol in result:
        result[symbol].sort(key=lambda x: x.date)

    logger.info(f"Loaded MongoDB data: {len(result)} symbols")
    return result


def get_latest_mongodb_data(mongodb_data: Dict[str, List[MongoDBData]], symbol: str) -> Optional[MongoDBData]:
    """Get latest MongoDB data for a symbol"""
    if symbol not in mongodb_data or not mongodb_data[symbol]:
        return None
    return mongodb_data[symbol][-1]


def load_stock_files() -> List[Tuple[str, List[StockData]]]:
    """
    Load all stock CSV files from stock directory
    ফাইলের নাম থেকে তারিখ নেয় (যেমন: 29-03-2026.csv → date = "29-03-2026")
    """
    stock_files = []
    if not os.path.exists(CSV_STOCK_DIR):
        logger.warning(f"Stock directory not found: {CSV_STOCK_DIR}")
        return stock_files

    for filename in os.listdir(CSV_STOCK_DIR):
        if filename.endswith('.csv'):
            filepath = os.path.join(CSV_STOCK_DIR, filename)
            # ফাইলের নাম থেকে তারিখ নিন
            file_date = filename.replace('.csv', '')
            
            data = read_csv_file(filepath)
            if data and len(data) > 1:
                start_idx = 0
                if data[0] and data[0][0].lower() == 'symbol':
                    start_idx = 1

                stocks = []
                for row in data[start_idx:]:
                    if row and len(row) >= 11 and row[0].strip():
                        try:
                            stocks.append(StockData.from_row(row, file_date))
                        except Exception as e:
                            logger.debug(f"Error creating StockData: {e}")
                            continue

                if stocks:
                    stock_files.append((file_date, stocks))
                    logger.info(f"Loaded {len(stocks)} stocks from {filename} (date: {file_date})")

    return sorted(stock_files, key=lambda x: x[0])


# ==================== TRADE LOGIC ====================

def check_entry_conditions(stock: StockData, latest_data: MongoDBData) -> bool:
    """Check if price is within entry zone"""
    entry_low, entry_high = stock.get_entry_range()
    if entry_low == 0 and entry_high == 0:
        return False
    return entry_low <= latest_data.close <= entry_high or entry_low <= latest_data.low <= entry_high


def check_stop_loss(stock: StockData, latest_data: MongoDBData) -> bool:
    """Check if stop loss is hit"""
    return stock.stop_loss > 0 and latest_data.close <= stock.stop_loss


def check_take_profit(stock: StockData, latest_data: MongoDBData, level: int) -> bool:
    """Check if take profit level is hit"""
    tp_value = getattr(stock, f'tp{level}', 0)
    if tp_value <= 0:
        return False
    return latest_data.close >= tp_value or latest_data.high >= tp_value


def load_existing_rd() -> Dict[str, Dict]:
    """Load existing RD data"""
    data = read_csv_file(RD_FILE)
    if not data or len(data) <= 1:
        return {}

    result = {}
    for row in data[1:]:
        if row and len(row) > 0 and row[0]:
            result[row[0]] = {
                'row': row,
                'rd_date': row[12] if len(row) > 12 else '',
                'ed_date': row[11] if len(row) > 11 else ''
            }
    return result


def is_duplicate_entry(symbol: str, date: str) -> bool:
    """একই তারিখে একই সিম্বল আগে এন্ট্রি হয়েছে কিনা চেক করে"""
    ed_data = read_csv_file(ED_FILE)
    if not ed_data or len(ed_data) <= 1:
        return False
    
    for row in ed_data[1:]:
        if len(row) > 11 and row[0] == symbol and row[11] == date:
            return True
    return False


def save_to_ed(stock: StockData) -> bool:
    """একই তারিখে একই সিম্বল ডুপ্লিকেট হবে না"""
    ed_data = read_csv_file(ED_FILE)
    if not ed_data:
        ed_data = [ED_HEADERS]
    
    if is_duplicate_entry(stock.symbol, stock.date):
        logger.warning(f"⚠️ Duplicate skipped: {stock.symbol} already entered on {stock.date}")
        return False
    
    new_row = stock.original_row + [stock.date]
    ed_data.append(new_row)
    write_csv_file(ED_FILE, ed_data)
    logger.info(f"✅ Saved to ED: {stock.symbol} on {stock.date}")
    return True


def save_to_rd(stock: StockData, running_date: str):
    """Save to RD file"""
    rd_data = read_csv_file(RD_FILE)
    if not rd_data:
        rd_data = [RD_HEADERS]

    new_row = stock.original_row + [stock.date, running_date]
    rd_data.append(new_row)
    write_csv_file(RD_FILE, rd_data)
    logger.info(f"✅ Saved to RD: {stock.symbol} (entry: {stock.date}, running: {running_date})")


def update_rd_with_running_date(symbol: str, running_date: str):
    """Update RD file with latest running date"""
    rd_data = read_csv_file(RD_FILE)
    if not rd_data or len(rd_data) <= 1:
        return

    for i, row in enumerate(rd_data[1:], 1):
        if row and row[0] == symbol:
            if len(row) > 12:
                row[12] = running_date
            else:
                while len(row) < 13:
                    row.append('')
                row[12] = running_date
            break

    write_csv_file(RD_FILE, rd_data)


def remove_from_rd(symbol: str):
    """Remove symbol from RD file"""
    rd_data = read_csv_file(RD_FILE)
    if not rd_data or len(rd_data) <= 1:
        return

    new_data = [rd_data[0]]
    for row in rd_data[1:]:
        if row and row[0] != symbol:
            new_data.append(row)

    if len(new_data) != len(rd_data):
        write_csv_file(RD_FILE, new_data)
        logger.info(f"Removed {symbol} from RD")


def save_to_sl(stock: StockData, rd_row: List[str], rd_date: str, sl_date: str, gap: int):
    """Save to SL file"""
    sl_data = read_csv_file(SL_FILE)
    if not sl_data:
        sl_data = [SL_HEADERS]

    original_data = rd_row[:11] if len(rd_row) >= 11 else stock.original_row
    
    new_row = original_data + [
        rd_row[11] if len(rd_row) > 11 else '',
        rd_date,
        sl_date,
        str(gap)
    ]
    sl_data.append(new_row)
    write_csv_file(SL_FILE, sl_data)
    logger.info(f"Saved to SL: {stock.symbol} at {sl_date}")


def save_to_tp(stock: StockData, rd_row: List[str], rd_date: str, tp_date: str, 
               tp_level: int, gap: int):
    """Save to TP file"""
    tp_data = read_csv_file(TP_FILE)
    if not tp_data:
        tp_data = [TP_HEADERS]

    original_data = rd_row[:11] if len(rd_row) >= 11 else stock.original_row
    ed_date = rd_row[11] if len(rd_row) > 11 else ''
    
    found_index = -1
    for i, row in enumerate(tp_data[1:], 1):
        if row and row[0] == stock.symbol:
            found_index = i
            break
    
    if found_index > 0:
        row = tp_data[found_index]
        while len(row) < 18:
            row.append('')
        
        if tp_level == 1:
            row[12] = tp_date
            row[13] = str(gap)
        elif tp_level == 2:
            row[14] = tp_date
            row[15] = str(gap)
        elif tp_level == 3:
            row[16] = tp_date
            row[17] = str(gap)
    else:
        new_row = original_data + [ed_date, rd_date, '', '', '', '', '', '']
        if tp_level == 1:
            new_row[12] = tp_date
            new_row[13] = str(gap)
        elif tp_level == 2:
            new_row[14] = tp_date
            new_row[15] = str(gap)
        elif tp_level == 3:
            new_row[16] = tp_date
            new_row[17] = str(gap)
        tp_data.append(new_row)

    write_csv_file(TP_FILE, tp_data)
    logger.info(f"Saved to TP: {stock.symbol} TP{tp_level} at {tp_date}")


def remove_old_records():
    """Remove records older than 1 year"""
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%d-%m-%Y")
    
    for filepath, date_col_index in [(ED_FILE, 11), (RD_FILE, 12), (SL_FILE, 12), (TP_FILE, 12)]:
        data = read_csv_file(filepath)
        if not data or len(data) <= 1:
            continue

        new_data = [data[0]]
        removed_count = 0
        
        for row in data[1:]:
            if len(row) > date_col_index and row[date_col_index]:
                if row[date_col_index] >= one_year_ago:
                    new_data.append(row)
                else:
                    removed_count += 1
            else:
                new_data.append(row)

        if removed_count > 0:
            write_csv_file(filepath, new_data)
            logger.info(f"Removed {removed_count} old records from {filepath}")


# ==================== NOTIFICATION ====================

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
    emoji = stock.get_score_emoji()
    
    if action == "entry":
        message = f"""
🔔 ENTRY SIGNAL 🔔

📊 Symbol: {stock.symbol} {emoji}
💰 Entry Zone: {stock.entry_zone}
🎯 Current Price: {price}
📅 Entry Date: {stock.date}
📅 Signal Date: {date}
⭐ Score: {stock.score}/100

📈 Wave: {stock.wave} → {stock.subwave}
🎯 Targets: {stock.tp1} → {stock.tp2} → {stock.tp3}
🛑 Stop Loss: {stock.stop_loss}
📊 RRR: {stock.rrr}

💡 Insight: {stock.insight[:100]}...
"""
    elif action == "stop_loss":
        message = f"""
⚠️ STOP LOSS HIT ⚠️

📊 Symbol: {stock.symbol} {emoji}
💰 Stop Loss: {stock.stop_loss}
🎯 Exit Price: {price}
📅 Entry Date: {stock.date}
📅 Exit Date: {date}
📈 Holding Period: {gap} days

📊 Wave: {stock.wave}
⭐ Score: {stock.score}/100
"""
    elif action.startswith("take_profit"):
        level = action.split('_')[-1] if '_' in action else '1'
        message = f"""
✅ TAKE PROFIT HIT - TP{level} ✅

📊 Symbol: {stock.symbol} {emoji}
💰 Target Price: {price}
🎯 Exit Price: {price}
📅 Entry Date: {stock.date}
📅 Exit Date: {date}
📈 Holding Period: {gap} days

📊 Wave: {stock.wave}
⭐ Score: {stock.score}/100
"""
    else:
        return

    send_email(f"Trade Alert: {stock.symbol} - {action}", message)
    logger.info(f"Notification sent for {stock.symbol}: {action}")


# ==================== MAIN TRADE MONITOR ====================

def calculate_gap_days(date1: str, date2: str) -> int:
    """Calculate gap days between two dates"""
    try:
        d1 = datetime.strptime(date1, "%d-%m-%Y")
        d2 = datetime.strptime(date2, "%d-%m-%Y")
        return abs((d2 - d1).days)
    except:
        return 0


def monitor_trades():
    """Main monitoring function - শুধু একবার রান করে"""
    logger.info("=" * 50)
    logger.info("Starting trade monitoring...")

    # Load all data
    mongodb_data = load_mongodb_data()
    stock_files = load_stock_files()
    existing_rd = load_existing_rd()

    if not mongodb_data:
        logger.warning("No MongoDB data found")
        return

    if not stock_files:
        logger.warning("No stock files found")
        return

    # Track processed symbols for this run
    processed = set()

    # Process each stock file
    for file_date, stocks in stock_files:
        logger.info(f"Processing {file_date} with {len(stocks)} stocks")

        for stock in stocks:
            if stock.symbol in processed:
                continue
                
            latest_data = get_latest_mongodb_data(mongodb_data, stock.symbol)
            if not latest_data:
                continue

            # Check if already in RD
            in_rd = stock.symbol in existing_rd

            # Check entry condition (not in RD)
            if not in_rd and check_entry_conditions(stock, latest_data):
                logger.info(f"✅ Entry signal for {stock.symbol} at {latest_data.close}")
                
                if save_to_ed(stock):
                    save_to_rd(stock, latest_data.date)
                    generate_notification(stock, "entry", latest_data.close, latest_data.date)
                    processed.add(stock.symbol)
                    existing_rd[stock.symbol] = {'row': None, 'rd_date': latest_data.date, 'ed_date': stock.date}

            # Check existing trades in RD
            elif in_rd:
                rd_info = existing_rd[stock.symbol]
                rd_date = rd_info['rd_date']
                ed_date = rd_info.get('ed_date', stock.date)
                gap = calculate_gap_days(ed_date, latest_data.date) if ed_date else 0

                # Check take profits in order (highest first)
                tp_hit = False
                for level in [3, 2, 1]:
                    if check_take_profit(stock, latest_data, level):
                        logger.info(f"✅ TP{level} hit for {stock.symbol} at {latest_data.close}")
                        save_to_tp(stock, rd_info['row'], rd_date, latest_data.date, level, gap)
                        generate_notification(stock, f"take_profit_{level}", latest_data.close, latest_data.date, gap)
                        remove_from_rd(stock.symbol)
                        tp_hit = True
                        processed.add(stock.symbol)
                        break

                # Check stop loss if no TP hit
                if not tp_hit and check_stop_loss(stock, latest_data):
                    logger.info(f"⚠️ Stop loss hit for {stock.symbol} at {latest_data.close}")
                    save_to_sl(stock, rd_info['row'], rd_date, latest_data.date, gap)
                    generate_notification(stock, "stop_loss", latest_data.close, latest_data.date, gap)
                    remove_from_rd(stock.symbol)
                    processed.add(stock.symbol)

                # Update RD with latest running date
                elif not tp_hit:
                    update_rd_with_running_date(stock.symbol, latest_data.date)

    # Remove old records
    remove_old_records()

    logger.info("Trade monitoring completed")
    logger.info("=" * 50)


# ==================== MAIN ====================

def main():
    """Main function - GitHub Actions এর জন্য (শুধু একবার রান করে)"""
    """
    print("=" * 60)
    print("🤖 AI TRADE MONITOR - Automated Trading System")
    print("=" * 60)
    print(f"📁 Stock Directory: {CSV_STOCK_DIR}")
    print(f"📁 Data Directory: {CSV_DATA_DIR}")
    print(f"📧 Email: {'✅' if EMAIL_USER else '❌'}")
    print("=" * 60)
    print("📅 Date source: Filename (stock/date.csv)")
    print("📌 Duplicate check: Same symbol + same date = blocked")
    print("▶️ Running once mode (for GitHub Actions)")
    print("=" * 60)"""

    # Ensure directories exist
    ensure_directories()

    # Run once and exit
    try:
        monitor_trades()
        print("\n✅ Script completed successfully!")
        logger.info("✅ Script completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
