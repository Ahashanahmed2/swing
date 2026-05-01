# sector_weekly-diver_daily_symbol.py
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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'swrsi.csv')
SIGNAL_LOG = './csv/swrsi_log.json'

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
            
            # ✅ নতুন CSV কলাম: week_start (date column)
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

def load_signal_log():
    """Signal tracking log"""
    if os.path.exists(SIGNAL_LOG):
        try:
            with open(SIGNAL_LOG, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'signals': [], 'last_run': None, 'total_signals_generated': 0}

def save_signal_log(log_data):
    """Save signal log"""
    log_data['last_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_data['total_signals_generated'] = len(log_data['signals'])
    with open(SIGNAL_LOG, 'w') as f:
        json.dump(log_data, f, indent=2)

def generate_swrsi_signals():
    """
    🎯 MAIN: Sector Weekly RSI Divergence → Daily Symbol RSI Divergence
    
    Uses RSI from sector weekly CSV (no separate calculation)
    """
    print("=" * 70)
    print("🔍 SWRSI - Sector Weekly + Daily RSI Divergence Signal Generator")
    print("   📊 Uses RSI from sector CSV (no separate calculation)")
    print("=" * 70)
    
    # 1. Load Daily RSI Divergence data
    rsi_diver_df = load_rsi_diver()
    if rsi_diver_df is None or len(rsi_diver_df) == 0:
        print("\n❌ Daily RSI Divergence ডাটা নেই। স্ক্রিপ্ট বন্ধ।")
        return pd.DataFrame()
    
    # 2. Load signal log
    signal_log = load_signal_log()
    
    # 3. Find all weekly sector files
    weekly_files = [f for f in os.listdir(WEEKLY_DIR) if f.endswith('_weekly.csv')]
    
    if not weekly_files:
        print(f"\n❌ {WEEKLY_DIR}-তে কোনো উইকলি ফাইল নেই!")
        return pd.DataFrame()
    
    print(f"\n📁 {len(weekly_files)}টি সেক্টর উইকলি ফাইল স্ক্যান হচ্ছে...")
    
    # 4. Process each sector
    signals = []
    sectors_checked = 0
    sectors_with_divergence = 0
    sectors_no_rsi = 0
    
    diver_symbols_set = set(rsi_diver_df['symbol'].unique())
    
    for weekly_file in sorted(weekly_files):
        sectors_checked += 1
        sector_name = weekly_file.replace('_weekly.csv', '').replace('_', ' ').title()
        
        # Load weekly data
        weekly_df = load_weekly_sector(sector_name)
        if weekly_df is None:
            continue
        
        # RSI available check
        if 'rsi' not in weekly_df.columns:
            sectors_no_rsi += 1
            continue
        
        valid_rsi = weekly_df['rsi'].dropna()
        if len(valid_rsi) < 2:
            sectors_no_rsi += 1
            continue
        
        # Check Weekly Divergence
        div_result = check_weekly_divergence(weekly_df)
        
        if div_result is None:
            continue
        
        sectors_with_divergence += 1
        
        # 🎉 Sector Weekly Divergence Found!
        print(f"\n{'─'*60}")
        print(f"🔔 SECTOR: {sector_name}")
        print(f"   📊 Weekly Divergence: {div_result['strength_label']} (Score: {div_result['strength_score']}/100)")
        print(f"   📅 Previous Week ({div_result['prev_week']['date']}):")
        print(f"      Low: {div_result['prev_week']['low']} | RSI: {div_result['prev_week']['rsi']}")
        print(f"   📅 Current Week  ({div_result['last_week']['date']}):")
        print(f"      Low: {div_result['last_week']['low']} | RSI: {div_result['last_week']['rsi']}")
        print(f"   📉 Price Drop: {div_result['price_drop_pct']}% | RSI Gain: +{div_result['rsi_gain']}")
        print(f"   📈 Total weeks with RSI: {len(valid_rsi)}")
        
        # Get sector symbols
        sector_symbols = get_sector_symbols(sector_name)
        
        if not sector_symbols:
            print(f"   ⚠ No symbols found for this sector")
            continue
        
        # Cross-check with Daily RSI Divergence symbols
        matched_symbols = [s for s in sector_symbols if s in diver_symbols_set]
        
        if not matched_symbols:
            print(f"   ❌ No symbol has Daily RSI Divergence (checked {len(sector_symbols)} symbols)")
            continue
        
        print(f"   ✅ {len(matched_symbols)}/{len(sector_symbols)} symbols have Daily RSI Divergence:")
        
        # Process each matched symbol
        for sym in matched_symbols:
            sym_div_data = rsi_diver_df[rsi_diver_df['symbol'] == sym]
            
            if len(sym_div_data) == 0:
                continue
            
            sym_row = sym_div_data.iloc[-1]
            
            div_type = str(sym_row.get('divergence_type', '')).strip().lower()
            
            if div_type != 'bullish':
                print(f"      ⊘ {sym}: Skipped ({div_type} daily divergence)")
                continue
            
            # ✅ CONFLUENCE FOUND!
            daily_strength = str(sym_row.get('strength', 'Moderate')).strip()
            daily_strength_bonus = {'Strong': 30, 'Moderate': 20, 'Weak': 10}.get(daily_strength, 10)
            composite_score = min(div_result['strength_score'] + daily_strength_bonus, 100)
            
            signal = {
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
            
            signals.append(signal)
            
            print(f"      ✅ {sym:<15} | Daily: {daily_strength:<10} | Composite: {composite_score:.0f}/100")
            print(f"         Weekly: Low {div_result['prev_week']['low']}→{div_result['last_week']['low']} | RSI {div_result['prev_week']['rsi']}→{div_result['last_week']['rsi']}")
            
            signal_log['signals'].append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'symbol': sym,
                'sector': sector_name,
                'composite_score': composite_score,
                'weekly_score': div_result['strength_score'],
                'daily_strength': daily_strength
            })
    
    # 5. Save Results
    print(f"\n{'='*70}")
    print("📊 SIGNAL GENERATION SUMMARY")
    print(f"{'='*70}")
    print(f"🔍 Sectors checked: {sectors_checked}")
    print(f"⚠️ Sectors without enough RSI: {sectors_no_rsi}")
    print(f"🔔 Sectors with Weekly Divergence: {sectors_with_divergence}")
    print(f"✅ Total Signals Generated: {len(signals)}")
    
    if signals:
        signals_df = pd.DataFrame(signals)
        
        existing_df = pd.DataFrame()
        if os.path.exists(OUTPUT_FILE):
            try:
                existing_df = pd.read_csv(OUTPUT_FILE)
            except:
                pass
        
        if len(existing_df) > 0:
            combined = pd.concat([existing_df, signals_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['symbol', 'weekly_curr_date'], keep='last')
            combined = combined.sort_values(['composite_score', 'weekly_strength_score'], ascending=[False, False])
        else:
            combined = signals_df.sort_values(['composite_score', 'weekly_strength_score'], ascending=[False, False])
        
        combined = combined.reset_index(drop=True)
        
        combined.to_csv(OUTPUT_FILE, index=False)
        print(f"📁 Signals saved: {OUTPUT_FILE}")
        print(f"📦 Total signals in file: {len(combined)}")
        
        save_signal_log(signal_log)
        print(f"📝 Log saved: {SIGNAL_LOG}")
        
        print(f"\n🏆 TOP SIGNALS:")
        print(f"{'Rank':<5} {'Symbol':<15} {'Sector':<20} {'Score':<8} {'Weekly':<12} {'Daily':<12}")
        print(f"{'─'*75}")
        for i, (_, row) in enumerate(combined.head(10).iterrows(), 1):
            print(f"{i:<5} {row['symbol']:<15} {row['sector']:<20} {row['composite_score']:<8.0f} "
                  f"{row['weekly_strength_label']:<12} {row['daily_divergence_strength']:<12}")
        
        return combined
    else:
        print("\nℹ️ No confluence signals found")
        save_signal_log(signal_log)
        return pd.DataFrame()

if __name__ == "__main__":
    result = generate_swrsi_signals()
