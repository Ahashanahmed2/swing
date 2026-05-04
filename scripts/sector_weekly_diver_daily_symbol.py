# ================== sector_weekly_daily_symbol.py ==================
# Complete: SWRSI (Weekly + Daily Confluence) + SWD (Weekly Divergence Only)
# ✅ SWRSI → swrsi.csv (Weekly + Daily Both Bullish, Full Details)
# ✅ SWD → swd.csv (Only signal_date + sector)

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import json

# কনফিগারেশন
WEEKLY_DIR = './csv/sector/weekly/'
RSI_DIVER_FILE = './csv/rsi_diver.csv'
MONGO_CSV = './csv/mongodb.csv'
OUTPUT_DIR = './output/ai_signal/'

# ✅ দুইটি আউটপুট
SWRSI_OUTPUT = os.path.join(OUTPUT_DIR, 'swrsi.csv')  # Weekly + Daily Confluence
SWD_OUTPUT = os.path.join(OUTPUT_DIR, 'swd.csv')      # Weekly Divergence Only (signal_date + sector)

SWRSI_LOG = './csv/swrsi_log.json'
SWD_LOG = './csv/swd_log.json'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_sector_name(sector):
    """সেক্টর নাম ক্লিন"""
    if pd.isna(sector) or sector == '' or sector is None:
        return 'Unknown'
    return str(sector)

def safe_filename(sector):
    """ফাইল নাম জেনারেট"""
    name = clean_sector_name(sector)
    return name.replace(' ', '_').replace('/', '_').replace('&', 'and').replace('(', '').replace(')', '').strip().lower()

def load_weekly_sector(sector):
    """সেক্টরের উইকলি CSV লোড"""
    filepath = os.path.join(WEEKLY_DIR, f"{safe_filename(sector)}_weekly.csv")
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            if len(df) < 2:
                return None
            
            date_col = 'week_start' if 'week_start' in df.columns else 'date'
            df['date'] = pd.to_datetime(df[date_col])
            
            if 'week_end_date' in df.columns:
                df['week_end_date'] = pd.to_datetime(df['week_end_date'])
            
            df = df.sort_values('date').reset_index(drop=True)
            return df
        except Exception as e:
            print(f"  ⚠ Error loading {sector}: {e}")
            return None
    return None

def load_rsi_diver():
    """লোড Daily RSI Divergence ডাটা"""
    if not os.path.exists(RSI_DIVER_FILE):
        print(f"❌ {RSI_DIVER_FILE} পাওয়া যায়নি!")
        return None
    
    try:
        df = pd.read_csv(RSI_DIVER_FILE)
        
        date_cols = ['last_date', 'previous_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        print(f"\n📥 Daily RSI Divergence ডাটা:")
        print(f"   মোট সিম্বল: {df['symbol'].nunique()}")
        print(f"   মোট রো: {len(df)}")
        
        if 'divergence_type' in df.columns:
            bull = len(df[df['divergence_type'].str.lower() == 'bullish'])
            bear = len(df[df['divergence_type'].str.lower() == 'bearish'])
            print(f"   Bullish: {bull} 🔼 | Bearish: {bear} 🔽")
        
        if 'strength' in df.columns:
            print(f"   Strength:", end=" ")
            for s in df['strength'].unique():
                print(f"{s}: {len(df[df['strength'] == s])}", end=" | ")
            print()
        
        if 'last_date' in df.columns:
            print(f"   Latest signal: {df['last_date'].max().strftime('%Y-%m-%d')}")
        
        return df
    except Exception as e:
        print(f"⚠ Error: {e}")
        return None

def get_sector_symbols(sector_name):
    """MongoDB CSV থেকে নির্দিষ্ট সেক্টরের সব সিম্বল বের করা"""
    if not os.path.exists(MONGO_CSV):
        print(f"⚠ {MONGO_CSV} পাওয়া যায়নি")
        return []
    
    try:
        df = pd.read_csv(MONGO_CSV)
        df['sector'] = df['sector'].apply(clean_sector_name)
        symbols = df[df['sector'].str.lower() == sector_name.lower()]['symbol'].unique()
        return sorted(list(symbols))
    except Exception as e:
        print(f"⚠ MongoDB CSV Error: {e}")
        return []

def check_weekly_divergence(weekly_df):
    """
    🔍 Sector Weekly Bullish Divergence চেক করে
    CSV-তে থাকা RSI ব্যবহার করে (আলাদা ক্যালকুলেশন নেই)
    
    Condition:
    - শেষ সপ্তাহের Low < আগের সপ্তাহের Low (Price making lower low)
    - শেষ সপ্তাহের RSI > আগের সপ্তাহের RSI (RSI making higher low)
    
    = Bullish Divergence
    """
    if weekly_df is None or len(weekly_df) < 2:
        return None
    
    # ✅ RSI NaN নয় এমন শেষ ২টি row খুঁজুন
    valid_rsi_df = weekly_df.dropna(subset=['rsi'])
    
    if len(valid_rsi_df) < 2:
        return None
    
    # শেষ 2 সপ্তাহ (valid RSI সহ)
    last = valid_rsi_df.iloc[-1]
    prev = valid_rsi_df.iloc[-2]
    
    # দরকারি কলাম চেক
    if not all(col in weekly_df.columns for col in ['low', 'rsi']):
        return None
    
    # ⭐ মূল কন্ডিশন: Price Low নিচে ↓ কিন্তু RSI উপরে ↑
    price_condition = last['low'] < prev['low']       # Lower low in price
    rsi_condition = last['rsi'] > prev['rsi']          # Higher low in RSI
    
    if not (price_condition and rsi_condition):
        return None
    
    # 📊 Divergence Strength হিসাব
    price_drop_pct = ((prev['low'] - last['low']) / prev['low']) * 100
    rsi_gain = last['rsi'] - prev['rsi']
    
    # Strength Score (0-100)
    score = 0
    
    # Price drop magnitude
    if price_drop_pct >= 5:
        score += 35
    elif price_drop_pct >= 3:
        score += 25
    elif price_drop_pct >= 1:
        score += 15
    else:
        score += 5
    
    # RSI gain magnitude
    if rsi_gain >= 10:
        score += 35
    elif rsi_gain >= 5:
        score += 25
    elif rsi_gain >= 2:
        score += 15
    else:
        score += 5
    
    # RSI অবস্থান bonus (oversold zone থেকে recovery বেশি powerful)
    if prev['rsi'] < 40:
        score += 15
    if prev['rsi'] < 30:
        score += 10
    
    # Volume confirmation (optional)
    if 'volume' in weekly_df.columns:
        if last['volume'] > prev['volume']:
            score += 5
    
    # Close confirmation (bullish candle = close > open)
    if 'close' in weekly_df.columns and 'open' in weekly_df.columns:
        if last['close'] > last['open']:
            score += 5
    
    score = min(score, 100)
    
    strength = 'Strong' if score >= 70 else 'Moderate' if score >= 45 else 'Weak'
    
    return {
        'has_divergence': True,
        'strength_score': score,
        'strength_label': strength,
        'price_drop_pct': round(price_drop_pct, 2),
        'rsi_gain': round(rsi_gain, 2),
        'prev_week': {
            'date': prev['date'].strftime('%Y-%m-%d'),
            'low': round(prev['low'], 2),
            'rsi': round(prev['rsi'], 2)
        },
        'last_week': {
            'date': last['date'].strftime('%Y-%m-%d'),
            'low': round(last['low'], 2),
            'rsi': round(last['rsi'], 2)
        }
    }

def load_signal_log(log_path):
    """Signal tracking log"""
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'signals': [], 'last_run': None, 'total_signals_generated': 0}

def save_signal_log(log_data, log_path):
    """Save signal log"""
    log_data['last_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_data['total_signals_generated'] = len(log_data['signals'])
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

def save_swrsi_csv(signals_list, output_path):
    """Save SWRSI signals to CSV"""
    if not signals_list:
        return pd.DataFrame()
    
    signals_df = pd.DataFrame(signals_list)
    
    existing_df = pd.DataFrame()
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
        except:
            pass
    
    if len(existing_df) > 0:
        combined = pd.concat([existing_df, signals_df], ignore_index=True)
        if 'symbol' in combined.columns and 'weekly_curr_date' in combined.columns:
            combined = combined.drop_duplicates(subset=['symbol', 'weekly_curr_date'], keep='last')
        combined = combined.sort_values(['composite_score', 'weekly_strength_score'], 
                                        ascending=[False, False])
    else:
        combined = signals_df.sort_values(['composite_score', 'weekly_strength_score'], 
                                          ascending=[False, False])
    
    combined = combined.reset_index(drop=True)
    combined.to_csv(output_path, index=False)
    
    return combined

def save_swd_csv(signals_list, output_path):
    """Save SWD signals to CSV (only signal_date + sector)"""
    if not signals_list:
        return pd.DataFrame()
    
    signals_df = pd.DataFrame(signals_list)
    
    existing_df = pd.DataFrame()
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
        except:
            pass
    
    if len(existing_df) > 0:
        combined = pd.concat([existing_df, signals_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['sector', 'signal_date'], keep='last')
        combined = combined.sort_values('signal_date', ascending=False)
    else:
        combined = signals_df.sort_values('signal_date', ascending=False)
    
    combined = combined.reset_index(drop=True)
    combined.to_csv(output_path, index=False)
    
    return combined

def generate_all_signals():
    """
    🎯 MAIN: Generate BOTH SWRSI and SWD Signals
    
    SWRSI: Weekly + Daily Bullish Divergence Confluence (Full Details)
    SWD: Sector Weekly Divergence Only (signal_date + sector)
    """
    print("=" * 70)
    print("🔍 SECTOR WEEKLY DIVERGENCE SIGNAL GENERATOR")
    print("   📊 SWRSI: Weekly + Daily Confluence → swrsi.csv")
    print("   📊 SWD: Weekly Divergence Only → swd.csv")
    print("=" * 70)
    
    # Load Daily RSI Divergence (for SWRSI)
    rsi_diver_df = load_rsi_diver()
    diver_symbols_set = set(rsi_diver_df['symbol'].unique()) if rsi_diver_df is not None else set()
    
    # Load signal logs
    swrsi_log = load_signal_log(SWRSI_LOG)
    swd_log = load_signal_log(SWD_LOG)
    
    # Find all weekly sector files
    weekly_files = [f for f in os.listdir(WEEKLY_DIR) if f.endswith('_weekly.csv')]
    
    if not weekly_files:
        print(f"\n❌ {WEEKLY_DIR}-তে কোনো উইকলি ফাইল নেই!")
        return
    
    print(f"\n📁 {len(weekly_files)}টি সেক্টর স্ক্যান হচ্ছে...")
    
    # Signal collectors
    swrsi_signals = []  # Weekly + Daily Confluence (Full Details)
    swd_signals = []    # Weekly Divergence Only (signal_date + sector)
    
    sectors_checked = 0
    sectors_with_divergence = 0
    
    for weekly_file in sorted(weekly_files):
        sectors_checked += 1
        sector_name = weekly_file.replace('_weekly.csv', '').replace('_', ' ').title()
        
        # Load weekly data
        weekly_df = load_weekly_sector(sector_name)
        if weekly_df is None:
            continue
        
        if 'rsi' not in weekly_df.columns:
            continue
        
        valid_rsi = weekly_df['rsi'].dropna()
        if len(valid_rsi) < 2:
            continue
        
        # Check Weekly Divergence
        div_result = check_weekly_divergence(weekly_df)
        
        if div_result is None:
            continue
        
        sectors_with_divergence += 1
        
        print(f"\n{'─'*60}")
        print(f"🔔 SECTOR: {sector_name}")
        print(f"   📊 Weekly Divergence: {div_result['strength_label']} (Score: {div_result['strength_score']}/100)")
        print(f"   📅 Previous Week ({div_result['prev_week']['date']}):")
        print(f"      Low: {div_result['prev_week']['low']} | RSI: {div_result['prev_week']['rsi']}")
        print(f"   📅 Current Week  ({div_result['last_week']['date']}):")
        print(f"      Low: {div_result['last_week']['low']} | RSI: {div_result['last_week']['rsi']}")
        print(f"   📉 Price Drop: {div_result['price_drop_pct']}% | RSI Gain: +{div_result['rsi_gain']}")
        
        # ============================
        # SWD SIGNAL: Weekly Divergence Only
        # ============================
        swd_signal = {
            'signal_date': datetime.now().strftime('%Y-%m-%d'),
            'sector': sector_name,
        }
        swd_signals.append(swd_signal)
        
        swd_log['signals'].append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'sector': sector_name,
            'type': 'SWD'
        })
        
        print(f"   📊 SWD: Signal added (signal_date + sector)")
        
        # ============================
        # SWRSI SIGNAL: Weekly + Daily Confluence
        # ============================
        # Get sector symbols
        sector_symbols = get_sector_symbols(sector_name)
        
        if rsi_diver_df is not None and len(sector_symbols) > 0:
            matched_count = 0
            weekly_signal_date = pd.to_datetime(div_result['last_week']['date'])
            
            for sym in sector_symbols:
                if sym not in diver_symbols_set:
                    continue
                
                sym_div_data = rsi_diver_df[rsi_diver_df['symbol'] == sym]
                
                for _, sym_row in sym_div_data.iterrows():
                    div_type = str(sym_row.get('divergence_type', '')).strip().lower()
                    
                    if div_type != 'bullish':
                        continue
                    
                    daily_date = None
                    if 'last_date' in sym_row and pd.notna(sym_row['last_date']):
                        daily_date = pd.to_datetime(sym_row['last_date'])
                    
                    if daily_date is None:
                        continue
                    
                    # Window: Weekly date থেকে ৩ দিন পর পর্যন্ত
                    days_diff = (daily_date - weekly_signal_date).days
                    
                    if 0 <= days_diff <= 3:
                        daily_strength = str(sym_row.get('strength', 'Moderate')).strip()
                        daily_strength_bonus = {'Strong': 30, 'Moderate': 20, 'Weak': 10}.get(daily_strength, 10)
                        time_penalty = days_diff * 2
                        composite_score = min(div_result['strength_score'] + daily_strength_bonus - time_penalty, 100)
                        composite_score = max(composite_score, 0)
                        
                        swrsi_signal = {
                            'signal_date': datetime.now().strftime('%Y-%m-%d'),
                            'composite_score': composite_score,
                            'symbol': sym,
                            'sector': sector_name,
                            'weekly_divergence': 'Bullish',
                            'weekly_strength_label': div_result['strength_label'],
                            'weekly_strength_score': div_result['strength_score'],
                            'weekly_prev_low': div_result['prev_week']['low'],
                            'weekly_curr_low': div_result['last_week']['low'],
                            'weekly_prev_rsi': div_result['prev_week']['rsi'],
                            'weekly_curr_rsi': div_result['last_week']['rsi'],
                            'weekly_price_drop_pct': div_result['price_drop_pct'],
                            'weekly_rsi_gain': div_result['rsi_gain'],
                            'weekly_prev_date': div_result['prev_week']['date'],
                            'weekly_curr_date': div_result['last_week']['date'],
                            'daily_divergence_type': sym_row.get('divergence_type', ''),
                            'daily_divergence_strength': daily_strength,
                            'daily_last_date': str(sym_row.get('last_date', ''))[:10] if pd.notna(sym_row.get('last_date')) else '',
                            'daily_last_price': sym_row.get('last_price', ''),
                            'daily_last_rsi': sym_row.get('last_rsi', ''),
                            'daily_prev_date': str(sym_row.get('previous_date', ''))[:10] if pd.notna(sym_row.get('previous_date')) else '',
                            'daily_prev_price': sym_row.get('previous_price', ''),
                            'daily_prev_rsi': sym_row.get('previous_rsi', ''),
                            'daily_last_high': sym_row.get('last_high', '') if 'last_high' in sym_row else '',
                            'daily_prev_price_2': sym_row.get('previous_price', '') if 'previous_price' in sym_row else '',
                        }
                        
                        swrsi_signals.append(swrsi_signal)
                        
                        swrsi_log['signals'].append({
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'symbol': sym,
                            'sector': sector_name,
                            'composite_score': composite_score,
                            'weekly_score': div_result['strength_score'],
                            'daily_strength': daily_strength
                        })
                        
                        matched_count += 1
                        day_label = "SAME DAY" if days_diff == 0 else f"+{days_diff}d"
                        print(f"      ✅ {sym:<15} | Daily: {daily_strength:<10} | {day_label:<10} | Score: {composite_score:.0f}/100")
                        break
            
            if matched_count == 0:
                print(f"   ⊘ SWRSI: No confluence found")
        else:
            print(f"   ⊘ SWRSI: No daily divergence data available")
    
    # ============================
    # SAVE RESULTS
    # ============================
    print(f"\n{'='*70}")
    print("📊 SIGNAL GENERATION SUMMARY")
    print(f"{'='*70}")
    print(f"🔍 Sectors checked: {sectors_checked}")
    print(f"🔔 Sectors with Weekly Divergence: {sectors_with_divergence}")
    print(f"\n📊 SWD Signals: {len(swd_signals)}")
    print(f"📊 SWRSI Signals: {len(swrsi_signals)}")
    
    # Save SWD
    if swd_signals:
        swd_df = save_swd_csv(swd_signals, SWD_OUTPUT)
        print(f"\n📁 SWD saved: {SWD_OUTPUT} ({len(swd_df)} records)")
        save_signal_log(swd_log, SWD_LOG)
        
        print(f"\n📊 SWD Signals (Weekly Divergence Sectors):")
        for _, row in swd_df.iterrows():
            print(f"   📅 {row['signal_date']} | 🏭 {row['sector']}")
    
    # Save SWRSI
    if swrsi_signals:
        swrsi_df = save_swrsi_csv(swrsi_signals, SWRSI_OUTPUT)
        print(f"\n📁 SWRSI saved: {SWRSI_OUTPUT} ({len(swrsi_df)} records)")
        save_signal_log(swrsi_log, SWRSI_LOG)
        
        print(f"\n🏆 TOP SWRSI SIGNALS:")
        print(f"{'Rank':<5} {'Symbol':<15} {'Sector':<20} {'Score':<8} {'Weekly':<12} {'Daily':<12}")
        print(f"{'─'*75}")
        for i, (_, row) in enumerate(swrsi_df.head(10).iterrows(), 1):
            print(f"{i:<5} {row['symbol']:<15} {row['sector']:<20} {row['composite_score']:<8.0f} "
                  f"{row['weekly_strength_label']:<12} {row['daily_divergence_strength']:<12}")
    else:
        print("\nℹ️ No SWRSI confluence signals found")
    
    if not swd_signals and not swrsi_signals:
        print("\nℹ️ No signals generated at all")
    
    return swd_signals, swrsi_signals

if __name__ == "__main__":
    swd, swrsi = generate_all_signals()