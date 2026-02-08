import pandas as pd
import os

def process_macd_signals():
    # ফাইল পাথ
    input_file = "./csv/mongodb.csv"
    output_dir = "./output/ai_signal"
    output_file = os.path.join(output_dir, "macd.csv")
    
    # আউটপুট ডিরেক্টরি তৈরি (যদি না থাকে)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # CSV ফাইল পড়া
        df = pd.read_csv(input_file)
        
        # তারিখ ফরম্যাট করো (যদি স্ট্রিং থাকে)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # প্রতিটি symbol এর জন্য গ্রুপ করে প্রক্রিয়া করা
        results = []
        
        for symbol, group in df.groupby('symbol'):
            # তারিখ অনুসারে সাজানো
            group = group.sort_values('date')
            
            # অন্তত ২টি row থাকতে হবে
            if len(group) >= 2:
                # শেষ দুইটি row
                last_row = group.iloc[-1]
                prev_row = group.iloc[-2]
                
                # শর্ত পরীক্ষা
                condition1 = last_row['macd'] > last_row['macd_signal']
                condition2 = prev_row['macd_hist'] < 0  # আগের row এর macd_hist negative
                condition3 = last_row['macd_hist'] >= 0  # শেষ row এর macd_hist 0 বা বেশি
                
                if condition1 and condition2 and condition3:
                    results.append({
                        'symbol': symbol,
                        'date': last_row['date'],
                        'close': last_row['close']
                    })
        
        # ফলাফল DataFrame তৈরি
        result_df = pd.DataFrame(results)
        
        # আউটপুটের জন্য নতুন ক্রমিক নং যোগ করা (1 থেকে শুরু)
        result_df.insert(0, 'No', range(1, len(result_df) + 1))
        
        # আউটপুট CSV ফাইলে সংরক্ষণ
        result_df.to_csv(output_file, index=False)
        
        print(f"প্রক্রিয়া সম্পন্ন হয়েছে! মোট {len(result_df)} টি সিগনাল পাওয়া গেছে।")
        print(f"ফলাফল সংরক্ষিত হয়েছে: {output_file}")
        
        # ফলাফল প্রিন্ট করা (ঐচ্ছিক)
        if not result_df.empty:
            print("\nপাওয়া সিগনালগুলো:")
            print(result_df.to_string(index=False))
        else:
            print("কোনো সিগনাল পাওয়া যায়নি।")
            
        return result_df
    
    except FileNotFoundError:
        print(f"ইনপুট ফাইল পাওয়া যায়নি: {input_file}")
        return None
    except KeyError as e:
        print(f"কলাম পাওয়া যায়নি: {e}")
        print("দয়া করে নিশ্চিত করুন যে CSV ফাইলে নিম্নোক্ত কলামগুলো আছে:")
        print("- symbol, date, macd, macd_signal, macd_hist, close")
        return None
    except Exception as e:
        print(f"ত্রুটি হয়েছে: {e}")
        return None

if __name__ == "__main__":
    process_macd_signals()