import pandas as pd
import os

def process_macd_signals():
    input_file = "./csv/mongodb.csv"
    output_dir = "./output/ai_signal"
    output_file = os.path.join(output_dir, "macd.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # CSV পড়া
        df = pd.read_csv(input_file)
        
        # ডেটা প্রিপ্রোসেসিং
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # সংখ্যাসূচক কলাম নিশ্চিত করা
        numeric_cols = ['macd', 'macd_signal', 'macd_hist', 'close']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # ফলাফল সংগ্রহ
        results = []
        
        for symbol, group in df.groupby('symbol'):
            group = group.sort_values('date')
            
            if len(group) >= 2:
                last_row = group.iloc[-1]
                prev_row = group.iloc[-2]
                
                # শর্ত ৩টি:
                # 1. macd > macd_signal (শেষ দিনে)
                # 2. আগের দিনের macd_hist < 0 (negative)
                # 3. শেষ দিনের macd_hist >= 0 (0 বা positive)
                if (last_row['macd'] > last_row['macd_signal'] and 
                    prev_row['macd_hist'] < 0 and 
                    last_row['macd_hist'] >= 0):
                    
                    results.append({
                        'symbol': symbol,
                        'date': last_row['date'],
                        'close': last_row['close']
                    })
        
        # ফলাফল সংরক্ষণ
        result_df = pd.DataFrame(results)
        
        if not result_df.empty:
            # নতুন ক্রমিক নং
            result_df.insert(0, 'No', range(1, len(result_df) + 1))
            
            # CSV তে সংরক্ষণ
            result_df.to_csv(output_file, index=False)
            print(f"✅ {len(result_df)} টি সিগনাল পাওয়া গেছে। ফাইল: {output_file}")
        else:
            # খালি ফাইল
            pd.DataFrame(columns=['No', 'symbol', 'date', 'close']).to_csv(output_file, index=False)
            print("⚠️ কোনো সিগনাল পাওয়া যায়নি। খালি ফাইল তৈরি করা হয়েছে।")
            
        return result_df
    
    except Exception as e:
        print(f"❌ ত্রুটি: {e}")
        return None

if __name__ == "__main__":
    process_macd_signals()