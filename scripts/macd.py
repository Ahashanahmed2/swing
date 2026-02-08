import pandas as pd
import numpy as np
import ta
import os

def calculate_macd_for_last_35_days(group):
    """Calculate MACD for last 35 days of each symbol"""
    group = group.copy()
    
    # তারিখ অনুসারে সাজানো (সবচেয়ে পুরানো থেকে নতুন)
    group = group.sort_values('date')
    
    # শেষ ৩৫ দিনের ডেটা নিন (অথবা যত দিন আছে)
    last_35_days = group.tail(35).copy()
    
    if len(last_35_days) < 26:  # MACD এর জন্য ন্যূনতম ২৬ দিন দরকার
        # পুরো গ্রুপে NaN সেট করুন
        group['macd'] = np.nan
        group['macd_signal'] = np.nan
        group['macd_hist'] = np.nan
        return group
    
    try:
        # শেষ ৩৫ দিনের উপর MACD ক্যালকুলেশন
        macd_indicator = ta.trend.MACD(close=last_35_days['close'])
        
        # MACD ভ্যালুগুলো
        macd_values = macd_indicator.macd()
        signal_values = macd_indicator.macd_signal()
        hist_values = macd_indicator.macd_diff()
        
        # শেষ ৩৫ দিনের জন্য ভ্যালু এসাইন করুন
        last_35_days.loc[:, 'macd'] = macd_values
        last_35_days.loc[:, 'macd_signal'] = signal_values
        last_35_days.loc[:, 'macd_hist'] = hist_values
        
        # মূল গ্রুপে MACD ভ্যালু যোগ করুন (শুধুমাত্র শেষ ৩৫ দিনের জন্য)
        # প্রথমে পুরো গ্রুপে NaN সেট করুন
        group['macd'] = np.nan
        group['macd_signal'] = np.nan
        group['macd_hist'] = np.nan
        
        # তারপর শেষ ৩৫ দিনের ডেটার জন্য ভ্যালু এসাইন করুন
        last_35_indices = last_35_days.index
        group.loc[last_35_indices, 'macd'] = last_35_days['macd'].values
        group.loc[last_35_indices, 'macd_signal'] = last_35_days['macd_signal'].values
        group.loc[last_35_indices, 'macd_hist'] = last_35_days['macd_hist'].values
        
    except Exception as e:
        print(f"⚠️ MACD calculation error for {group['symbol'].iloc[0]}: {e}")
        group['macd'] = np.nan
        group['macd_signal'] = np.nan
        group['macd_hist'] = np.nan
    
    return group

def process_macd_signals():
    # ফাইল পাথ
    input_file = "./csv/mongodb.csv"
    output_dir = "./output/ai_signal"
    output_file = os.path.join(output_dir, "macd.csv")
    
    # আউটপুট ডিরেক্টরি তৈরি
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # CSV ফাইল পড়া
        print(f"📂 ফাইল পড়ছি: {input_file}")
        df = pd.read_csv(input_file)
        
        # কলাম নাম চেক
        print(f"📋 ইনপুট ফাইলের কলাম ({len(df.columns)} টি):")
        print(df.columns.tolist()[:10], "...")  # প্রথম ১০টি কলাম দেখান
        
        # তারিখ ফরম্যাট করা
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            print(f"📅 তারিখ রেঞ্জ: {df['date'].min().date()} থেকে {df['date'].max().date()}")
        else:
            print("❌ 'date' কলাম পাওয়া যায়নি!")
            return None
        
        # প্রয়োজনীয় কলামগুলো নিশ্চিত করা
        required_cols = ['symbol', 'date', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ নিম্নলিখিত কলামগুলো পাওয়া যায়নি: {missing_cols}")
            return None
        
        # সংখ্যাসূচক কলামগুলো নিশ্চিত করা
        if 'close' in df.columns:
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
        
        # প্রতিটি সিম্বলের ডেটা পর্যালোচনা
        print(f"\n📊 ডেটা পরিসংখ্যান:")
        symbol_stats = df.groupby('symbol').size().reset_index(name='total_days')
        
        # প্রতিটি সিম্বলের শেষ তারিখ
        last_dates = df.groupby('symbol')['date'].max().reset_index(name='last_date')
        symbol_stats = pd.merge(symbol_stats, last_dates, on='symbol')
        
        print(f"  - মোট সিম্বল: {len(symbol_stats)}")
        print(f"  - গড় দিন/সিম্বল: {symbol_stats['total_days'].mean():.1f}")
        
        # ৩৫ দিনের কম ডেটা আছে এমন সিম্বল
        low_data_symbols = symbol_stats[symbol_stats['total_days'] < 35]
        if len(low_data_symbols) > 0:
            print(f"  ⚠️  {len(low_data_symbols)} টি সিম্বলের ৩৫ দিনের কম ডেটা আছে")
        
        # -------------------------------------------------------------------
        # Step 1: শেষ ৩৫ দিনের উপর MACD ক্যালকুলেশন
        # -------------------------------------------------------------------
        print("\n📈 শেষ ৩৫ দিনের উপর MACD ক্যালকুলেশন করছি...")
        df = df.groupby('symbol', group_keys=False).apply(calculate_macd_for_last_35_days)
        
        # শুধুমাত্র শেষ দিনের ডেটা ফিল্টার করুন
        print("\n🎯 শুধুমাত্র শেষ দিনের ডেটা নিয়ে কাজ করছি...")
        
        # প্রতিটি সিম্বলের শেষ তারিখ বের করুন
        last_dates_df = df.groupby('symbol')['date'].max().reset_index()
        
        # শুধুমাত্র প্রতিটি সিম্বলের শেষ দিনের row নিন
        last_day_data = []
        for _, row in last_dates_df.iterrows():
            symbol = row['symbol']
            last_date = row['date']
            
            symbol_last_row = df[(df['symbol'] == symbol) & (df['date'] == last_date)]
            
            if not symbol_last_row.empty:
                last_day_data.append(symbol_last_row.iloc[0])
        
        last_day_df = pd.DataFrame(last_day_data)
        
        print(f"✅ শেষ দিনের ডেটা পাওয়া গেছে {len(last_day_df)} টি সিম্বলের")
        
        # শুধুমাত্র MACD ভ্যালু আছে এমন সিম্বল ফিল্টার করুন
        valid_macd_df = last_day_df.dropna(subset=['macd', 'macd_signal', 'macd_hist'])
        print(f"📊 MACD ভ্যালু আছে এমন সিম্বল: {len(valid_macd_df)}/{len(last_day_df)}")
        
        # প্রথম কয়েকটি সিম্বলের MACD ভ্যালু দেখান
        print(f"\n🔍 প্রথম ৫টি সিম্বলের MACD ভ্যালু:")
        print("="*70)
        for i, row in valid_macd_df.head(5).iterrows():
            print(f"{row['symbol']}: তারিখ={row['date'].date()}, "
                  f"Close={row['close']:.2f}, "
                  f"MACD={row['macd']:.6f}, "
                  f"Signal={row['macd_signal']:.6f}, "
                  f"Hist={row['macd_hist']:.6f}")
        
        # -------------------------------------------------------------------
        # Step 2: আগের দিনের MACD হিস্টোগ্রাম খুঁজে বের করা
        # -------------------------------------------------------------------
        print(f"\n🔍 প্রতিটি সিম্বলের জন্য আগের দিনের MACD হিস্টোগ্রাম খুঁজছি...")
        
        results = []
        
        for _, last_row in valid_macd_df.iterrows():
            symbol = last_row['symbol']
            last_date = last_row['date']
            
            # এই সিম্বলের সব ডেটা নিন
            symbol_data = df[df['symbol'] == symbol].sort_values('date')
            
            # শেষ দিনের আগের দিন খুঁজুন
            prev_days = symbol_data[symbol_data['date'] < last_date]
            
            if len(prev_days) == 0:
                continue  # আগের দিনের ডেটা নেই
            
            # সর্বশেষ আগের দিনের row নিন
            prev_row = prev_days.iloc[-1]
            
            # আগের দিনের MACD ভ্যালু আছে কিনা চেক করুন
            if pd.isna(prev_row['macd_hist']):
                continue  # আগের দিনের MACD হিস্টোগ্রাম নেই
            
            # শর্তগুলো চেক করুন
            prev_macd_hist = prev_row['macd_hist']
            last_macd_hist = last_row['macd_hist']
            last_macd = last_row['macd']
            last_macd_signal = last_row['macd_signal']
            
            # শর্ত ১: MACD > MACD Signal (শেষ দিনে)
            condition1 = last_macd > last_macd_signal
            
            # শর্ত ২: আগের দিনে MACD Histogram ছিল নেগেটিভ (0 এর নিচে)
            condition2 = prev_macd_hist < 0
            
            # শর্ত ৩: আজকের দিনে MACD Histogram হয়েছে পজিটিভ (0 এর উপরে)
            condition3 = last_macd_hist > 0
            
            if condition1 and condition2 and condition3:
                # ডিবাগ প্রিন্ট
                print(f"✅ {symbol}: {last_date.date()}")
                print(f"   আগের দিন ({prev_row['date'].date()}) hist: {prev_macd_hist:.6f}")
                print(f"   আজ ({last_date.date()}) hist: {last_macd_hist:.6f}")
                print(f"   MACD: {last_macd:.6f} > Signal: {last_macd_signal:.6f}")
                print(f"   ক্লোজ প্রাইস: {last_row['close']:.2f}")
                print(f"   {'-'*60}")
                
                results.append({
                    'symbol': symbol,
                    'date': last_date,
                    'close': last_row['close'],
                    'macd': last_macd,
                    'macd_signal': last_macd_signal,
                    'macd_hist': last_macd_hist,
                    'prev_macd_hist': prev_macd_hist,
                    'prev_date': prev_row['date']
                })
        
        # -------------------------------------------------------------------
        # Step 3: ফলাফল সংরক্ষণ
        # -------------------------------------------------------------------
        print("\n" + "="*80)
        
        if results:
            result_df = pd.DataFrame(results)
            
            # তারিখ অনুসারে সাজানো (নতুন থেকে পুরাতন)
            result_df = result_df.sort_values('date', ascending=False)
            
            # ক্রমিক নং যোগ করা
            result_df.insert(0, 'No', range(1, len(result_df) + 1))
            
            # কলাম অর্ডার
            column_order = ['No', 'symbol', 'date', 'close', 
                           'macd', 'macd_signal', 'macd_hist', 'prev_macd_hist']
            
            # আউটপুট ফাইল তৈরি
            output_df = result_df[column_order]
            output_df.to_csv(output_file, index=False)
            
            # সংখ্যাসূচক কলাম রাউন্ডিং
            for col in ['macd', 'macd_signal', 'macd_hist', 'prev_macd_hist']:
                if col in output_df.columns:
                    output_df[col] = output_df[col].round(6)
            
            print(f"✅ মোট {len(result_df)} টি MACD সিগনাল পাওয়া গেছে!")
            print(f"💾 ফাইল সংরক্ষিত: {output_file}")
            
            # বিস্তারিত ফলাফল
            print(f"\n📈 MACD সিগনাল সমূহ:")
            print("="*100)
            for i, row in result_df.iterrows():
                print(f"{row['No']:3d}. {row['symbol']:10} {row['date'].date()} "
                      f"Close: {row['close']:8.2f} | "
                      f"MACD: {row['macd']:7.4f} > {row['macd_signal']:7.4f} | "
                      f"Hist: {row['prev_macd_hist']:7.4f} → {row['macd_hist']:7.4f}")
        
        else:
            print("❌ কোনো MACD সিগনাল পাওয়া যায়নি!")
            
            # খালি ফাইল তৈরি
            column_order = ['No', 'symbol', 'date', 'close', 
                           'macd', 'macd_signal', 'macd_hist', 'prev_macd_hist']
            empty_df = pd.DataFrame(columns=column_order)
            empty_df.to_csv(output_file, index=False)
            print(f"💾 খালি ফাইল তৈরি করা হয়েছে: {output_file}")
        
        return results if results else None
    
    except Exception as e:
        print(f"❌ ত্রুটি: {str(e)}")
        import traceback
        print(f"ট্রেসব্যাক:\n{traceback.format_exc()}")
        return None

if __name__ == "__main__":
    process_macd_signals()