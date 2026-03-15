import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SwingLowRayLineAnalyzer:
    def __init__(self, left_bars=5, right_bars=5, tolerance_percent=0.5):
        """
        সুইং লো এবং রে লাইন বিশ্লেষক
        
        Parameters:
        -----------
        left_bars : int
            সুইং লো শনাক্ত করতে বামের বার সংখ্যা
        right_bars : int
            সুইং লো শনাক্ত করতে ডানের বার সংখ্যা
        tolerance_percent : float
            সাপোর্ট লেভেল চেক করার জন্য টলারেন্স (%)
        """
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.tolerance_percent = tolerance_percent
    
    def find_swing_lows(self, data):
        """প্রদত্ত ডাটা থেকে সুইং লো চিহ্নিত করে"""
        swing_lows = []
        
        for i in range(self.left_bars, len(data) - self.right_bars):
            current_low = data['low'].iloc[i]
            
            # বামের বারগুলো চেক করা
            left_condition = True
            for j in range(1, self.left_bars + 1):
                if data['low'].iloc[i - j] <= current_low:
                    left_condition = False
                    break
            
            # ডানের বারগুলো চেক করা
            right_condition = True
            for j in range(1, self.right_bars + 1):
                if data['low'].iloc[i + j] <= current_low:
                    right_condition = False
                    break
            
            if left_condition and right_condition:
                swing_lows.append({
                    'index': i,
                    'date': data['date'].iloc[i] if 'date' in data.columns else i,
                    'price': current_low,
                    'type': 'swing_low'
                })
        
        return pd.DataFrame(swing_lows) if swing_lows else pd.DataFrame()
    
    def analyze_symbol(self, symbol_data):
        """
        একটি নির্দিষ্ট সিম্বলের জন্য সম্পূর্ণ বিশ্লেষণ করে
        - প্রথম ও দ্বিতীয় সুইং লো দেখায়
        """
        if len(symbol_data) < (self.left_bars + self.right_bars + 1):
            return {
                'symbol': symbol_data['symbol'].iloc[0] if 'symbol' in symbol_data.columns else 'Unknown',
                'date': symbol_data['date'].iloc[-1] if 'date' in symbol_data.columns else 'Unknown',
                'low': round(symbol_data['low'].iloc[-1], 2) if len(symbol_data) > 0 else 0,
                'close': round(symbol_data['close'].iloc[-1], 2) if len(symbol_data) > 0 else 0,
                'ray_1_date': 'N/A',
                'ray_1_price': 0,
                'ray_2_date': 'N/A',
                'ray_2_price': 0,
                'nearest_support_percent': 999
            }
        
        # সুইং লো সনাক্ত
        swing_lows = self.find_swing_lows(symbol_data)
        
        low = symbol_data['low'].iloc[-1]
        close = symbol_data['close'].iloc[-1]
        latest_date = symbol_data['date'].iloc[-1] if 'date' in symbol_data.columns else 'Unknown'
        
        result = {
            'symbol': symbol_data['symbol'].iloc[0] if 'symbol' in symbol_data.columns else 'Unknown',
            'date': latest_date,
            'low': round(low, 2),
            'close': round(close, 2),
            'ray_1_date': 'N/A',
            'ray_1_price': 0,
            'ray_2_date': 'N/A',
            'ray_2_price': 0,
            'nearest_support_percent': 999
        }
        
        if not swing_lows.empty:
            # সুইং লো গুলোকে ইনডেক্স অনুযায়ী সাজানো (সবচেয়ে পুরনো প্রথম)
            swing_lows_sorted = swing_lows.sort_values('index')
            
            # প্রথম সুইং লো (সবচেয়ে পুরনো)
            if len(swing_lows_sorted) >= 1:
                ray1 = swing_lows_sorted.iloc[0]
                ray1_price = ray1['price']
                
                result['ray_1_date'] = ray1['date']
                result['ray_1_price'] = round(ray1_price, 2)
            
            # দ্বিতীয় সুইং লো
            if len(swing_lows_sorted) >= 2:
                ray2 = swing_lows_sorted.iloc[1]
                ray2_price = ray2['price']
                
                result['ray_2_date'] = ray2['date']
                result['ray_2_price'] = round(ray2_price, 2)
            
            # নিকটতম সাপোর্ট বের করা (সবগুলো সুইং লো থেকে)
            nearest_support = 999
            for _, swing_low in swing_lows.iterrows():
                # চেক করা যে সাপোর্ট এখনও ভ্যালিড কিনা (ব্রোকেন হয়নি)
                if not self.is_support_broken(symbol_data, swing_low):
                    percent = ((close - swing_low['price']) / swing_low['price']) * 100
                    if percent < nearest_support:
                        nearest_support = percent
            
            result['nearest_support_percent'] = round(nearest_support, 2) if nearest_support != 999 else 999
        
        return result
    
    def is_support_broken(self, data, swing_low):
        """
        সুইং লো ব্রোকেন হয়েছে কিনা চেক করে
        """
        start_idx = swing_low['index']
        subsequent_data = data.iloc[start_idx:]
        
        # যদি কোন পরবর্তী লো সুইং লো এর চেয়ে কম হয়, তাহলে সাপোর্ট ব্রোকেন
        if (subsequent_data['low'] < swing_low['price'] * 0.995).any():  # 0.5% টলারেন্স
            return True
        return False
    
    def analyze_all_symbols(self, file_path):
        """
        CSV ফাইল থেকে সব সিম্বল বিশ্লেষণ করে
        """
        # CSV ফাইল পড়া
        print(f"ফাইল পড়া হচ্ছে: {file_path}")
        df = pd.read_csv(file_path)
        
        # ডাটা চেক
        print(f"কলামসমূহ: {list(df.columns)}")
        
        required_columns = ['symbol', 'open', 'high', 'low', 'close', 'date']
        missing_columns = []
        for col in required_columns:
            if col not in df.columns:
                missing_columns.append(col)
                print(f"সতর্কতা: {col} কলাম পাওয়া যায়নি")
        
        if missing_columns:
            print(f"❗ অনুপস্থিত কলাম: {missing_columns}")
        
        # সিম্বল অনুযায়ী গ্রুপ
        symbols = df['symbol'].unique() if 'symbol' in df.columns else ['Unknown']
        print(f"\nমোট {len(symbols)} টি সিম্বল পাওয়া গেছে")
        
        all_results = []
        
        for symbol in symbols:
            print(f"\n🔄 বিশ্লেষণ করা হচ্ছে: {symbol}")
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # ডাটা সাজানো (ধরে নিচ্ছি ডাটা তারিখ অনুযায়ী সাজানো আছে)
            if 'date' in symbol_data.columns:
                symbol_data['date'] = pd.to_datetime(symbol_data['date'])
                symbol_data = symbol_data.sort_values('date')
            
            print(f"  মোট {len(symbol_data)} টি রেকর্ড")
            
            if len(symbol_data) < 20:
                print(f"  ⚠️  পর্যাপ্ত ডাটা নেই ({len(symbol_data)} বার), ন্যূনতম ২০ প্রয়োজন")
                # তবুও বিশ্লেষণ চেষ্টা করা
                result = self.analyze_symbol(symbol_data)
                all_results.append(result)
                continue
            
            result = self.analyze_symbol(symbol_data)
            all_results.append(result)
            
            # রেজাল্ট প্রিন্ট
            print(f"  📅 তারিখ: {result['date']}")
            print(f"  💰 লো: {result['low']}, ক্লোজ: {result['close']}")
            
            if result['ray_1_price'] > 0:
                print(f"  🔵 রে-১: {result['ray_1_date']} - {result['ray_1_price']}")
            if result['ray_2_price'] > 0:
                print(f"  🔵 রে-২: {result['ray_2_date']} - {result['ray_2_price']}")
            
            if result['nearest_support_percent'] != 999:
                print(f"  🎯 নিকটতম সাপোর্ট: {result['nearest_support_percent']}%")
        
        return all_results

def save_results_to_csv(results, output_dir='./output/ai_signal'):
    """
    ফলাফল CSV ফাইলে সংরক্ষণ করে
    """
    # আউটপুট ডিরেক্টরি তৈরি
    os.makedirs(output_dir, exist_ok=True)
    
    # রেজাল্ট টেবিল
    summary_data = []
    for r in results:
        summary_data.append({
            'symbol': r['symbol'],
            'date': r['date'],
            'low': r['low'],
            'close': r['close'],
            'ray_1_date': r['ray_1_date'],
            'ray_1_price': r['ray_1_price'],
            'ray_2_date': r['ray_2_date'],
            'ray_2_price': r['ray_2_price'],
            'nearest_support_percent': r['nearest_support_percent']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # nearest_support_percent এর ভিত্তিতে sorting (কম % সিম্বল উপরে)
    summary_df = summary_df.sort_values('nearest_support_percent', ascending=True)
    
    # মূল রিপোর্ট সংরক্ষণ
    output_file = os.path.join(output_dir, 'ray_support.csv')
    summary_df.to_csv(output_file, index=False)
    print(f"\n✅ রিপোর্ট সংরক্ষণ করা হয়েছে: {output_file}")
    
    return summary_df

def print_summary_table(summary_df):
    """
    সংক্ষিপ্ত ফলাফল টেবিল প্রিন্ট করে
    """
    print("\n" + "="*110)
    print("📊 রে লাইন সাপোর্ট অ্যানালাইসিস রিপোর্ট")
    print("="*110)
    
    # টেবিল হেডার
    print(f"{'Rank':<5} {'Symbol':<10} {'Date':<12} {'Low':<8} {'Close':<8} {'Ray-1 Date':<12} {'Ray-1':<8} {'Ray-2 Date':<12} {'Ray-2':<8} {'Nearest':<8}")
    print("-"*110)
    
    for idx, (_, row) in enumerate(summary_df.iterrows(), 1):
        # nearest_support_percent এর ভিত্তিতে ranking
        if row['nearest_support_percent'] == 999:
            signal = "🚫"
        elif row['nearest_support_percent'] < 3:
            signal = "🟢"
        elif row['nearest_support_percent'] < 7:
            signal = "🟡"
        else:
            signal = "🔴"
        
        # Date ফরম্যাটিং
        date_str = str(row['date'])[:10] if row['date'] != 'Unknown' else 'N/A'
        ray1_date = str(row['ray_1_date'])[:10] if row['ray_1_date'] != 'N/A' else 'N/A'
        ray2_date = str(row['ray_2_date'])[:10] if row['ray_2_date'] != 'N/A' else 'N/A'
        
        print(f"{idx:<5} {row['symbol']:<10} {date_str:<12} {row['low']:<8.2f} {row['close']:<8.2f} "
              f"{ray1_date:<12} {row['ray_1_price']:<8.2f} "
              f"{ray2_date:<12} {row['ray_2_price']:<8.2f} "
              f"{row['nearest_support_percent'] if row['nearest_support_percent'] != 999 else 'N/A':<8}")
    
    print("="*110)
    
    # সারাংশ পরিসংখ্যান
    strong_count = len(summary_df[summary_df['nearest_support_percent'] < 3])
    moderate_count = len(summary_df[(summary_df['nearest_support_percent'] >= 3) & (summary_df['nearest_support_percent'] < 7)])
    weak_count = len(summary_df[(summary_df['nearest_support_percent'] >= 7) & (summary_df['nearest_support_percent'] < 999)])
    no_support_count = len(summary_df[summary_df['nearest_support_percent'] == 999])
    
    print(f"\n📈 সারাংশ:")
    print(f"মোট সিম্বল: {len(summary_df)}")
    print(f"🟢 স্ট্রং ({strong_count}): nearest_support < 3%")
    print(f"🟡 মডারেট ({moderate_count}): nearest_support 3-7%")
    print(f"🔴 উইক ({weak_count}): nearest_support > 7%")
    print(f"⚪ সাপোর্ট নেই ({no_support_count})")
    
    # টপ ৫ সিম্বল
    top_5 = summary_df[summary_df['nearest_support_percent'] < 999].head(5)
    if not top_5.empty:
        print("\n🏆 টপ ৫ সিম্বল (সবচেয়ে কাছের সাপোর্ট):")
        for _, row in top_5.iterrows():
            print(f"   {row['symbol']}: {row['nearest_support_percent']:.2f}% (রে১: {row['ray_1_price']:.2f}, রে২: {row['ray_2_price']:.2f})")

# মূল প্রোগ্রাম
def main():
    # ইনপুট এবং আউটপুট ফাইল পাথ
    input_file = './csv/mongodb.csv'
    output_dir = './output/ai_signal'
    
    print("="*80)
    print("🚀 রে লাইন সাপোর্ট অ্যানালাইজার")
    print("="*80)
    
    # চেক করা যে ইনপুট ফাইল আছে কিনা
    if not os.path.exists(input_file):
        print(f"❌ ত্রুটি: {input_file} ফাইলটি পাওয়া যায়নি!")
        print("দয়া করে নিশ্চিত করুন যে ফাইলটি সঠিক পাথে আছে।")
        return
    
    try:
        # অ্যানালাইজার তৈরি
        analyzer = SwingLowRayLineAnalyzer(
            left_bars=5,
            right_bars=5,
            tolerance_percent=0.5
        )
        
        # সব সিম্বল বিশ্লেষণ
        print("\n🔄 সিম্বল বিশ্লেষণ শুরু হচ্ছে...")
        results = analyzer.analyze_all_symbols(input_file)
        
        if not results:
            print("❌ কোনো বৈধ ফলাফল পাওয়া যায়নি!")
            return
        
        print(f"\n✅ মোট {len(results)} টি সিম্বল সফলভাবে বিশ্লেষণ করা হয়েছে")
        
        # ফলাফল সংরক্ষণ
        print("\n💾 ফলাফল সংরক্ষণ করা হচ্ছে...")
        summary_df = save_results_to_csv(results, output_dir)
        
        # সংক্ষিপ্ত ফলাফল প্রিন্ট
        print_summary_table(summary_df)
        
        print(f"\n✅ বিশ্লেষণ সম্পন্ন! ফলাফল {output_dir}/ray_support.csv ফাইলে সংরক্ষণ করা হয়েছে।")
        
    except Exception as e:
        print(f"❌ ত্রুটি: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()