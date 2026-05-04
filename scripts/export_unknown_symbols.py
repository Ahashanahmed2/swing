# ================== export_unknown_symbols.py ==================
# MongoDB CSV থেকে Unknown sector-এর symbol বের করে unknown.csv তৈরি
# ফরম্যাট: no, symbol, date, high, low

import pandas as pd
import os
from datetime import datetime

# কনফিগারেশন
MONGO_CSV = './csv/mongodb.csv'
OUTPUT_DIR = './output/ai_signal/'
UNKNOWN_OUTPUT = os.path.join(OUTPUT_DIR, 'unknown.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_sector_name(sector):
    """সেক্টর নাম ক্লিন"""
    if pd.isna(sector) or sector == '' or sector is None:
        return 'Unknown'
    return str(sector)

def export_unknown_symbols():
    """
    MongoDB CSV থেকে Unknown sector-এর symbol বের করে unknown.csv ফাইল তৈরি
    ফরম্যাট: no, symbol, date, high, low
    """
    print("=" * 60)
    print("🔍 MongoDB → Unknown Sector → unknown.csv")
    print("=" * 60)
    
    # MongoDB CSV চেক
    if not os.path.exists(MONGO_CSV):
        print(f"\n❌ {MONGO_CSV} পাওয়া যায়নি!")
        return None
    
    try:
        # MongoDB CSV লোড
        df = pd.read_csv(MONGO_CSV)
        
        print(f"\n📁 MongoDB CSV তথ্য:")
        print(f"   • মোট রো: {len(df)}")
        print(f"   • কলাম: {list(df.columns)}")
        
        # সেক্টর ক্লিন
        df['sector'] = df['sector'].apply(clean_sector_name)
        
        # Unknown সেক্টর ফিল্টার
        unknown_df = df[df['sector'].str.lower() == 'unknown'].copy()
        
        if len(unknown_df) == 0:
            print("\n⚠️ MongoDB CSV-তে কোনো Unknown সেক্টরের সিম্বল নেই!")
            return None
        
        print(f"\n🔴 Unknown সেক্টর:")
        print(f"   • মোট সিম্বল: {unknown_df['symbol'].nunique()}")
        print(f"   • মোট রো: {len(unknown_df)}")
        
        # দরকারি কলাম খোঁজা
        symbol_col = 'symbol'
        date_col = None
        high_col = None
        low_col = None
        
        # ডেট কলাম খোঁজা
        for col in ['date', 'Date', 'DATE', 'last_date', 'signal_date']:
            if col in unknown_df.columns:
                date_col = col
                break
        
        # হাই-লো কলাম খোঁজা
        for col in ['high', 'High', 'HIGH']:
            if col in unknown_df.columns:
                high_col = col
                break
        
        for col in ['low', 'Low', 'LOW']:
            if col in unknown_df.columns:
                low_col = col
                break
        
        # চেক করুন দরকারি কলাম আছে কিনা
        if not all([symbol_col, date_col, high_col, low_col]):
            missing = []
            if not symbol_col: missing.append('symbol')
            if not date_col: missing.append('date')
            if not high_col: missing.append('high')
            if not low_col: missing.append('low')
            print(f"\n⚠️ এই কলামগুলো নেই: {missing}")
            print(f"   উপলব্ধ কলাম: {list(unknown_df.columns)}")
            
            # যা আছে তাই নিয়ে সেভ করি
            output_df = unknown_df.copy()
        else:
            print(f"\n✅ সব কলাম পাওয়া গেছে:")
            print(f"   • Symbol: {symbol_col}")
            print(f"   • Date: {date_col}")
            print(f"   • High: {high_col}")
            print(f"   • Low: {low_col}")
            
            # প্রয়োজনীয় কলাম সিলেক্ট ও রিনেম
            output_df = unknown_df[[symbol_col, date_col, high_col, low_col]].copy()
            output_df.columns = ['symbol', 'date', 'high', 'low']
        
        # নম্বর যোগ
        output_df.insert(0, 'no', range(1, len(output_df) + 1))
        
        # ডেট ফরম্যাট চেক
        if 'date' in output_df.columns:
            try:
                output_df['date'] = pd.to_datetime(output_df['date']).dt.strftime('%Y-%m-%d')
            except:
                pass  # যদি ডেট না হয় তাহলে যেমন আছে তেমন থাকবে
        
        # unknown.csv সেভ
        output_df.to_csv(UNKNOWN_OUTPUT, index=False)
        
        print(f"\n{'=' * 60}")
        print(f"✅ SAVED: {UNKNOWN_OUTPUT}")
        print(f"{'=' * 60}")
        print(f"📁 ফাইল তথ্য:")
        print(f"   • মোট রো: {len(output_df)}")
        print(f"   • কলাম: {list(output_df.columns)}")
        
        # প্রিভিউ
        print(f"\n📋 unknown.csv প্রিভিউ (প্রথম ৫টি রো):")
        print(output_df.head(5).to_string(index=False))
        
        # সিম্বল লিস্ট
        if 'symbol' in output_df.columns:
            symbols = sorted(output_df['symbol'].unique())
            print(f"\n📊 Unknown সেক্টরের সিম্বল ({len(symbols)}টি):")
            print("─" * 40)
            for i, sym in enumerate(symbols, 1):
                print(f"   {i:3d}. {sym}")
        
        return output_df
        
    except Exception as e:
        print(f"\n❌ ত্রুটি: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = export_unknown_symbols()
    
    if result is not None:
        print(f"\n🎉 সম্পন্ন! {UNKNOWN_OUTPUT} ফাইল তৈরি হয়েছে।")
    else:
        print(f"\n⚠️ unknown.csv তৈরি করা যায়নি।")
