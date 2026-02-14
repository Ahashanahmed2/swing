import pandas as pd
import os

def merge_symbols():
    # ফাইল পাথ চেক করা
    macd_file = './output/ai_signal/macd.csv'
    daily_buy_file = './output/ai_signal/daily_buy.csv'
    output_file = './output/ai_signal/macd_daily.csv'
    
    # চেক করা যে ফাইল দুটি আছে কিনা
    if not os.path.exists(macd_file):
        print(f"Error: {macd_file} ফাইলটি পাওয়া যায়নি!")
        return
    
    if not os.path.exists(daily_buy_file):
        print(f"Error: {daily_buy_file} ফাইলটি পাওয়া যায়নি!")
        return
    
    try:
        # CSV ফাইল দুটি পড়া
        df_macd = pd.read_csv(macd_file)
        df_daily = pd.read_csv(daily_buy_file)
        
        print("daily_buy.csv এর কলামসমূহ:", list(df_daily.columns))
        
        # কলাম নাম চেক করা (uppercase/lowercase issue সমাধান)
        df_daily.columns = df_daily.columns.str.upper()  # সব কলামকে uppercase এ রূপান্তর
        
        # 'SYMBOL' কলাম আছে কিনা চেক করা
        if 'SYMBOL' not in df_daily.columns:
            print("Error: daily_buy.csv এ 'SYMBOL' কলাম নেই!")
            print("উপলব্ধ কলামসমূহ:", list(df_daily.columns))
            return
        
        # macd.csv এর জন্য কলাম চেক
        df_macd.columns = df_macd.columns.str.upper()
        if 'SYMBOL' not in df_macd.columns:
            print("Error: macd.csv এ 'SYMBOL' কলাম নেই!")
            print("উপলব্ধ কলামসমূহ:", list(df_macd.columns))
            return
        
        # symbol কলাম ট্রিম করা (extra spaces সরানো)
        df_macd['SYMBOL'] = df_macd['SYMBOL'].astype(str).str.strip()
        df_daily['SYMBOL'] = df_daily['SYMBOL'].astype(str).str.strip()
        
        # common_symbols বের করা
        common_symbols = set(df_macd['SYMBOL']).intersection(set(df_daily['SYMBOL']))
        
        if len(common_symbols) == 0:
            print("কোন কমন symbol পাওয়া যায়নি!")
            return
        
        print(f"মোট {len(common_symbols)} টি কমন symbol পাওয়া গেছে")
        
        # রেজাল্ট ডাটাফ্রেম তৈরি করা
        result_data = []
        
        for i, symbol in enumerate(sorted(common_symbols), 1):  # 1 থেকে সিরিয়াল শুরু
            # daily_buy.csv থেকে symbol এর জন্য buy এবং file কলামের মান নেওয়া
            symbol_data = df_daily[df_daily['SYMBOL'] == symbol]
            
            if len(symbol_data) > 0:
                # BUY ভ্যালু নেওয়া
                buy_value = symbol_data['BUY'].iloc[0] if 'BUY' in symbol_data.columns else 'N/A'
                
                # FILE কলামের মান নেওয়া
                if 'FILE' in symbol_data.columns:
                    file_value = symbol_data['FILE'].iloc[0]
                else:
                    file_value = ''
            else:
                buy_value = 'N/A'
                file_value = ''
            
            result_data.append({
                'No': i,
                'symbol': symbol,
                'buy': buy_value,
                'file': file_value
            })
        
        # রেজাল্ট ডাটাফ্রেম তৈরি
        df_result = pd.DataFrame(result_data)
        
        # কলামের অর্ডার ঠিক করা (No, symbol, buy, file)
        df_result = df_result[['No', 'symbol', 'buy', 'file']]
        
        # আউটপুট ডিরেক্টরি চেক করা
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # CSV ফাইল হিসেবে সংরক্ষণ
        df_result.to_csv(output_file, index=False)
        
        print(f"\nসফলভাবে {output_file} ফাইলটি তৈরি করা হয়েছে!")
        print(f"মোট {len(result_data)} টি রেকর্ড সংরক্ষণ করা হয়েছে")
        
        # প্রথম ১০ টি রেকর্ড দেখানো
        print("\nপ্রথম ১০ টি রেকর্ড:")
        print(df_result.head(10))
        
        # স্ট্যাটিস্টিক্স দেখানো
        print(f"\nস্ট্যাটিস্টিক্স:")
        print(f"মোট কমন সিম্বল: {len(result_data)}")
        print(f"কলামসমূহ: {', '.join(df_result.columns)}")
        
    except Exception as e:
        print(f"একটি ত্রুটি হয়েছে: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    merge_symbols()