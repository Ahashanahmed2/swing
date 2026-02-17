import pandas as pd
import os
from pathlib import Path

def process_macd_signal():
    """
    daily_buy.csv থেকে symbol এবং latest_date পড়ে,
    mongodb.csv থেকে symbol এবং date অনুযায়ী ডেটা বিশ্লেষণ করে
    macd > macd_signal (বর্তমান) এবং macd < macd_signal (পূর্ববর্তী) শর্ত পূরণ করলে
    down_macd.csv ফাইল তৈরি করে
    """
    
    # ফাইল পাথ নির্ধারণ
    daily_buy_path = Path('./output/ai_signal/daily_buy.csv')
    mongodb_path = Path('./csv/mongodb.csv')
    output_path = Path('./output/ai_signal/down_macd.csv')
    
    # চেক করা ফাইল আছে কিনা
    if not daily_buy_path.exists():
        print(f"Error: {daily_buy_path} ফাইলটি পাওয়া যায়নি")
        return
    
    if not mongodb_path.exists():
        print(f"Error: {mongodb_path} ফাইলটি পাওয়া যায়নি")
        return
    
    try:
        # daily_buy.csv পড়া
        daily_buy_df = pd.read_csv(daily_buy_path)
        
        # প্রয়োজনীয় কলাম আছে কিনা চেক করা
        if 'symbol' not in daily_buy_df.columns or 'latest_date' not in daily_buy_df.columns:
            print("Error: daily_buy.csv তে 'symbol' এবং 'latest_date' কলাম প্রয়োজন")
            return
        
        # mongodb.csv পড়া
        mongodb_df = pd.read_csv(mongodb_path)
        
        # mongodb.csv তে প্রয়োজনীয় কলাম আছে কিনা চেক করা
        required_columns = ['symbol', 'date', 'macd', 'macd_signal']
        for col in required_columns:
            if col not in mongodb_df.columns:
                print(f"Error: mongodb.csv তে '{col}' কলামটি প্রয়োজন")
                return
        
        # mongodb.csv ডেটা তারিখ অনুযায়ী সাজানো
        mongodb_df['date'] = pd.to_datetime(mongodb_df['date'])
        mongodb_df = mongodb_df.sort_values(['symbol', 'date'])
        
        # ফলাফল সংরক্ষণের জন্য খালি লিস্ট
        result_rows = []
        
        # প্রতিটি symbol এর জন্য প্রসেস করা
        for _, row in daily_buy_df.iterrows():
            symbol = row['symbol']
            latest_date = pd.to_datetime(row['latest_date'])
            
            # symbol এবং date অনুযায়ী mongodb_df থেকে ডেটা ফিল্টার করা
            symbol_data = mongodb_df[mongodb_df['symbol'] == symbol].copy()
            
            if symbol_data.empty:
                print(f"Warning: {symbol} এর জন্য mongodb.csv এ কোন ডেটা পাওয়া যায়নি")
                continue
            
            # নির্দিষ্ট তারিখের ডেটা খোঁজা
            date_data = symbol_data[symbol_data['date'] == latest_date]
            
            if date_data.empty:
                print(f"Warning: {symbol} এর জন্য {latest_date} তারিখের ডেটা mongodb.csv এ পাওয়া যায়নি")
                continue
            
            # বর্তমান রো এর ইনডেক্স পাওয়া
            current_idx = date_data.index[0]
            
            # চেক করা আগের রো আছে কিনা
            if current_idx > 0 and symbol_data.iloc[current_idx - 1]['symbol'] == symbol:
                current_row = symbol_data.loc[current_idx]
                prev_row = symbol_data.iloc[current_idx - 1]
                
                # শর্ত চেক করা
                if (current_row['macd'] >= current_row['macd_signal'] and 
                    prev_row['macd'] < prev_row['macd_signal']):
                    
                    # daily_buy.csv থেকে সম্পূর্ণ রো যোগ করা
                    result_rows.append(row.to_dict())
                    print(f"Found: {symbol} - {latest_date}")
            else:
                print(f"Info: {symbol} এর জন্য পর্যাপ্ত ডেটা নেই বা এটি প্রথম রো")
        
        if result_rows:
            # DataFrame তৈরি করা
            result_df = pd.DataFrame(result_rows)
            
            # 'No' কলাম যোগ করা (প্রথম কলাম হিসেবে)
            result_df.insert(0, 'No', range(1, len(result_df) + 1))
            
            # আউটপুট ডিরেক্টরি তৈরি করা (যদি না থাকে)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # CSV ফাইল হিসেবে সংরক্ষণ
            result_df.to_csv(output_path, index=False)
            print(f"\nSuccess: {len(result_rows)} টি রো {output_path} এ সংরক্ষণ করা হয়েছে")
            print("\nফলাফলের সারাংশ:")
            print(result_df[['No', 'symbol', 'latest_date']].to_string(index=False))
        else:
            print("\nNo results: কোন ডেটা শর্ত পূরণ করেনি")
            
    except Exception as e:
        print(f"Error: প্রসেসিং এর সময় সমস্যা হয়েছে - {str(e)}")

if __name__ == "__main__":
    process_macd_signal()