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
        df_macd.columns = df_macd.columns.str.upper()    # macd.csv এর কলামগুলোও uppercase

        # 'SYMBOL' কলাম আছে কিনা চেক করা
        if 'SYMBOL' not in df_daily.columns:
            print("Error: daily_buy.csv এ 'SYMBOL' কলাম নেই!")
            print("উপলব্ধ কলামসমূহ:", list(df_daily.columns))
            return

        # macd.csv এর জন্য কলাম চেক
        if 'SYMBOL' not in df_macd.columns:
            print("Error: macd.csv এ 'SYMBOL' কলাম নেই!")
            print("উপলব্ধ কলামসমূহ:", list(df_macd.columns))
            return

        # RSI কলাম আছে কিনা চেক করা
        if 'RSI' not in df_macd.columns:
            print("Warning: macd.csv এ 'RSI' কলাম নেই! RSI ছাড়া প্রসেস চলবে।")
            # RSI কলাম না থাকলে ডামি মান সেট করা
            df_macd['RSI'] = 0

        # symbol কলাম ট্রিম করা (extra spaces সরানো)
        df_macd['SYMBOL'] = df_macd['SYMBOL'].astype(str).str.strip()
        df_daily['SYMBOL'] = df_daily['SYMBOL'].astype(str).str.strip()

        # macd.csv থেকে symbol এবং rsi নেওয়া
        df_macd_unique = df_macd[['SYMBOL', 'RSI']].copy()
        
        # যদি একই symbol এর জন্য একাধিক এন্ট্রি থাকে, তাহলে গড় RSI নেওয়া
        df_macd_unique = df_macd_unique.groupby('SYMBOL')['RSI'].mean().reset_index()

        # common_symbols বের করা
        common_symbols = set(df_macd_unique['SYMBOL']).intersection(set(df_daily['SYMBOL']))

        if len(common_symbols) == 0:
            print("কোন কমন symbol পাওয়া যায়নি!")
            return

        print(f"মোট {len(common_symbols)} টি কমন symbol পাওয়া গেছে")

        # রেজাল্ট ডাটাফ্রেম তৈরি করা
        result_data = []

        for symbol in common_symbols:
            # macd_unique থেকে RSI মান নেওয়া
            rsi_value = df_macd_unique[df_macd_unique['SYMBOL'] == symbol]['RSI'].iloc[0]
            
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
                'symbol': symbol,
                'rsi': round(rsi_value, 2),  # RSI মান যোগ করা
                'buy': buy_value,
                'file': file_value
            })

        # ডাটাফ্রেম তৈরি করা
        df_result = pd.DataFrame(result_data)

        # RSI অনুযায়ী সর্ট করা (ছোট RSI প্রথমে)
        df_result = df_result.sort_values('rsi', ascending=True)

        # সিরিয়াল নম্বর যোগ করা (সর্ট করার পর)
        df_result.insert(0, 'No', range(1, len(df_result) + 1))

        # কলামের অর্ডার ঠিক করা (No, symbol, rsi, buy, file)
        df_result = df_result[['No', 'symbol', 'rsi', 'buy', 'file']]

        # আউটপুট ডিরেক্টরি চেক করা
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # CSV ফাইল হিসেবে সংরক্ষণ
        df_result.to_csv(output_file, index=False)

        print(f"\nসফলভাবে {output_file} ফাইলটি তৈরি করা হয়েছে!")
        print(f"মোট {len(result_data)} টি রেকর্ড সংরক্ষণ করা হয়েছে")

        # প্রথম ১০ টি রেকর্ড দেখানো
        print("\nপ্রথম ১০ টি রেকর্ড (RSI অনুযায়ী সর্টেড):")
        print(df_result.head(10))

        # স্ট্যাটিস্টিক্স দেখানো
        print(f"\nস্ট্যাটিস্টিক্স:")
        print(f"মোট কমন সিম্বল: {len(result_data)}")
        print(f"RSI রেঞ্জ: {df_result['rsi'].min():.2f} - {df_result['rsi'].max():.2f}")
        print(f"কলামসমূহ: {', '.join(df_result.columns)}")

    except Exception as e:
        print(f"একটি ত্রুটি হয়েছে: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    merge_symbols()