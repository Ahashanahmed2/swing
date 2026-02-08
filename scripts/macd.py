import pandas as pd
import os

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
        print(f"📋 ইনপুট ফাইলের কলাম: {df.columns.tolist()}")
        
        # তারিখ ফরম্যাট করা
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            print("❌ 'date' কলাম পাওয়া যায়নি!")
            return None
        
        # সংখ্যাসূচক কলামগুলো নিশ্চিত করা
        required_cols = ['symbol', 'date', 'macd', 'macd_signal', 'macd_hist', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ নিম্নলিখিত কলামগুলো পাওয়া যায়নি: {missing_cols}")
            return None
        
        numeric_cols = ['macd', 'macd_signal', 'macd_hist', 'close']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"{col}: {df[col].dtype}")
        
        # প্রতিটি symbol এর জন্য প্রক্রিয়া
        results = []
        match_count = 0
        total_symbols = df['symbol'].nunique()
        
        print(f"\n🔍 মোট {total_symbols} টি সিম্বল প্রক্রিয়া করা হচ্ছে...")
        print("="*80)
        
        for idx, (symbol, group) in enumerate(df.groupby('symbol'), 1):
            group = group.sort_values('date').reset_index(drop=True)
            
            # প্রোগ্রেস বার (ঐচ্ছিক)
            if idx % 50 == 0 or idx == total_symbols:
                print(f"প্রগতি: {idx}/{total_symbols}")
            
            # অন্তত ২টি row থাকতে হবে
            if len(group) >= 2:
                # শেষ দুইটি row নিন
                last_row = group.iloc[-1]
                prev_row = group.iloc[-2]
                
                # MACD এবং MACD Histogram মানগুলো
                prev_macd_hist = prev_row['macd_hist']
                last_macd_hist = last_row['macd_hist']
                last_macd = last_row['macd']
                last_macd_signal = last_row['macd_signal']
                last_close = last_row['close']
                last_date = last_row['date']
                
                # শর্তগুলো:
                # 1. MACD > MACD Signal (শেষ দিনে)
                # 2. আগের দিনে MACD Histogram ছিল নেগেটিভ (0 এর নিচে)
                # 3. আজকের দিনে MACD Histogram হয়েছে পজিটিভ (0 এর উপরে)
                condition1 = last_macd > last_macd_signal
                condition2 = prev_macd_hist < 0  # নেগেটিভ
                condition3 = last_macd_hist > 0  # পজিটিভ (0 এর উপরে)
                
                if condition1 and condition2 and condition3:
                    match_count += 1
                    
                    # ডিবাগ প্রিন্ট
                    print(f"✅ {match_count}. {symbol}: {last_date.date()}")
                    print(f"   আগের দিন hist: {prev_macd_hist:.6f} → আজ hist: {last_macd_hist:.6f}")
                    print(f"   MACD: {last_macd:.6f} > Signal: {last_macd_signal:.6f}")
                    print(f"   ক্লোজ প্রাইস: {last_close}")
                    print(f"   {'-'*60}")
                    
                    results.append({
                        'symbol': symbol,
                        'date': last_date,
                        'close': last_close,
                        'macd': last_macd,
                        'macd_signal': last_macd_signal,
                        'macd_hist': last_macd_hist,
                        'prev_macd_hist': prev_macd_hist
                    })
        
        # ফলাফল প্রক্রিয়া
        print("\n" + "="*80)
        
        if results:
            result_df = pd.DataFrame(results)
            
            # তারিখ অনুসারে সাজানো (নতুন থেকে পুরাতন)
            result_df = result_df.sort_values('date', ascending=False)
            
            # আউটপুটের জন্য নতুন ক্রমিক নং যোগ করা (1 থেকে শুরু)
            result_df.insert(0, 'No', range(1, len(result_df) + 1))
            
            # কলামের অর্ডার নির্ধারণ
            column_order = ['No', 'symbol', 'date', 'close', 
                           'macd', 'macd_signal', 'macd_hist', 'prev_macd_hist']
            
            # আউটপুট ফাইল তৈরি
            output_df = result_df[column_order]
            output_df.to_csv(output_file, index=False)
            
            # সংখ্যাসূচক কলামগুলোর ফরম্যাট ঠিক করা
            for col in ['macd', 'macd_signal', 'macd_hist', 'prev_macd_hist', 'close']:
                if col in output_df.columns:
                    output_df[col] = output_df[col].round(6)
            
            print(f"✅ মোট {len(result_df)} টি সিগনাল পাওয়া গেছে!")
            print(f"💾 ফাইল সংরক্ষিত: {output_file}")
            
            # ফাইল স্ট্যাটিস্টিক্স
            file_size = os.path.getsize(output_file) / 1024  # KB তে
            print(f"📊 ফাইল সাইজ: {file_size:.2f} KB")
            
            # বিস্তারিত ফলাফল দেখান
            print(f"\n📈 প্রথম 10টি ফলাফল:")
            print("="*100)
            print(f"{'No':<4} {'Symbol':<8} {'Date':<12} {'Close':<10} {'MACD':<10} {'Signal':<10} {'Hist':<10} {'Prev Hist':<10}")
            print("-"*100)
            
            for i, row in result_df.head(10).iterrows():
                print(f"{row['No']:<4} {row['symbol']:<8} {row['date'].date():<12} "
                      f"{row['close']:<10.2f} {row['macd']:<10.4f} {row['macd_signal']:<10.4f} "
                      f"{row['macd_hist']:<10.4f} {row['prev_macd_hist']:<10.4f}")
            
            # CSV ফাইলের কলাম চেক
            print(f"\n📋 আউটপুট ফাইলের কলাম ({len(column_order)} টি):")
            for i, col in enumerate(column_order, 1):
                print(f"  {i}. {col}")
            
        else:
            print("❌ কোনো সিগনাল পাওয়া যায়নি!")
            
            # খালি ফাইল তৈরি (সমস্ত কলামসহ)
            column_order = ['No', 'symbol', 'date', 'close', 
                           'macd', 'macd_signal', 'macd_hist', 'prev_macd_hist']
            empty_df = pd.DataFrame(columns=column_order)
            empty_df.to_csv(output_file, index=False)
            print(f"💾 খালি ফাইল তৈরি করা হয়েছে: {output_file}")
        
        return results
    
    except FileNotFoundError:
        print(f"❌ ইনপুট ফাইল পাওয়া যায়নি: {input_file}")
        return None
    except Exception as e:
        print(f"❌ ত্রুটি: {str(e)}")
        import traceback
        print(f"ট্রেসব্যাক:\n{traceback.format_exc()}")
        return None

if __name__ == "__main__":
    process_macd_signals()