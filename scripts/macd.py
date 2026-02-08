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
        print(f"ফাইল পড়ছি: {input_file}")
        df = pd.read_csv(input_file)
        
        # কলাম নাম চেক
        print(f"কলামগুলো: {df.columns.tolist()}")
        
        # তারিখ ফরম্যাট করা
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            print("❌ 'date' কলাম পাওয়া যায়নি!")
            return None
        
        # সংখ্যাসূচক কলামগুলো নিশ্চিত করা
        numeric_cols = ['macd', 'macd_signal', 'macd_hist', 'close']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"{col}: {df[col].dtype}")
            else:
                print(f"❌ '{col}' কলাম পাওয়া যায়নি!")
                return None
        
        # প্রতিটি symbol এর জন্য প্রক্রিয়া
        results = []
        
        for symbol, group in df.groupby('symbol'):
            group = group.sort_values('date').reset_index(drop=True)
            
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
                
                # শর্তগুলো:
                # 1. MACD > MACD Signal (শেষ দিনে)
                # 2. আগের দিনে MACD Histogram ছিল নেগেটিভ (0 এর নিচে)
                # 3. আজকের দিনে MACD Histogram হয়েছে পজিটিভ (0 এর উপরে)
                condition1 = last_macd > last_macd_signal
                condition2 = prev_macd_hist < 0  # নেগেটিভ
                condition3 = last_macd_hist > 0  # পজিটিভ (0 এর উপরে)
                
                # ডিবাগ প্রিন্ট
                debug_msg = f"\n{symbol}: "
                debug_msg += f"আগের দিন hist={prev_macd_hist:.4f}, "
                debug_msg += f"আজ hist={last_macd_hist:.4f}, "
                debug_msg += f"MACD={last_macd:.4f}, Signal={last_macd_signal:.4f}"
                debug_msg += f" | শর্ত: {condition1} & {condition2} & {condition3}"
                
                if condition1 and condition2 and condition3:
                    debug_msg += " ✅ MATCH"
                    print(debug_msg)
                    
                    results.append({
                        'symbol': symbol,
                        'date': last_row['date'],
                        'close': last_row['close'],
                        'macd': last_macd,
                        'macd_signal': last_macd_signal,
                        'macd_hist': last_macd_hist,
                        'prev_macd_hist': prev_macd_hist
                    })
                else:
                    print(debug_msg)
        
        # ফলাফল প্রক্রিয়া
        if results:
            result_df = pd.DataFrame(results)
            result_df.insert(0, 'No', range(1, len(result_df) + 1))
            
            # আউটপুট ফাইল তৈরি
            output_df = result_df[['No', 'symbol', 'date', 'close']]
            output_df.to_csv(output_file, index=False)
            
            print(f"\n{'='*50}")
            print(f"✅ মোট {len(result_df)} টি সিগনাল পাওয়া গেছে!")
            print(f"💾 ফাইল সংরক্ষিত: {output_file}")
            print(f"{'='*50}")
            
            # বিস্তারিত ফলাফল দেখান
            print("\n📊 বিস্তারিত ফলাফল:")
            print(result_df[['No', 'symbol', 'date', 'close', 'prev_macd_hist', 'macd_hist', 'macd', 'macd_signal']].to_string(index=False))
            
        else:
            print(f"\n{'='*50}")
            print("❌ কোনো সিগনাল পাওয়া যায়নি!")
            print(f"{'='*50}")
            
            # খালি ফাইল তৈরি
            pd.DataFrame(columns=['No', 'symbol', 'date', 'close']).to_csv(output_file, index=False)
            print(f"💾 খালি ফাইল তৈরি করা হয়েছে: {output_file}")
        
        return results
    
    except FileNotFoundError:
        print(f"❌ ফাইল পাওয়া যায়নি: {input_file}")
        return None
    except Exception as e:
        print(f"❌ ত্রুটি: {str(e)}")
        import traceback
        print(f"ট্রেসব্যাক:\n{traceback.format_exc()}")
        return None

if __name__ == "__main__":
    process_macd_signals()