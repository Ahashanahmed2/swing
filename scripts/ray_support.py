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
    
    def calculate_support_percentage(self, data, swing_lows_df):
        """
        প্রতিটি সুইং লো থেকে বর্তমান প্রাইস কত % উপরে আছে তা গণনা করে
        """
        if swing_lows_df.empty:
            return []
        
        current_price = data['close'].iloc[-1]
        support_analysis = []
        
        for _, swing_low in swing_lows_df.iterrows():
            # সুইং লো থেকে বর্তমান প্রাইসের পার্থক্য %
            price_diff_percent = ((current_price - swing_low['price']) / swing_low['price']) * 100
            
            # সুইং লো টি কি এখনও সাপোর্ট হিসেবে কাজ করছে?
            is_support = self.check_support_level(data, swing_low)
            
            support_analysis.append({
                'swing_low_index': swing_low['index'],
                'swing_low_date': swing_low['date'],
                'swing_low_price': round(swing_low['price'], 2),
                'current_price': round(current_price, 2),
                'percent_above_support': round(price_diff_percent, 2),
                'is_active_support': is_support,
                'support_status': 'ACTIVE' if is_support else 'BROKEN'
            })
        
        return support_analysis
    
    def check_support_level(self, data, swing_low):
        """
        চেক করে যে সুইং লো টি এখনও সাপোর্ট হিসেবে কাজ করছে কিনা
        (অর্থাৎ, সুইং লো এর পর থেকে কি কখনো এর নিচে প্রাইস গেছে?)
        """
        start_idx = swing_low['index']
        subsequent_data = data.iloc[start_idx:]
        
        # যদি কোন পরবর্তী লো সুইং লো এর চেয়ে কম হয়, তাহলে সাপোর্ট ব্রোকেন
        if (subsequent_data['low'] < swing_low['price'] * 0.995).any():  # 0.5% টলারেন্স
            return False
        return True
    
    def analyze_symbol(self, symbol_data):
        """
        একটি নির্দিষ্ট সিম্বলের জন্য সম্পূর্ণ বিশ্লেষণ করে
        """
        if len(symbol_data) < (self.left_bars + self.right_bars + 1):
            return {
                'symbol': symbol_data['symbol'].iloc[0] if 'symbol' in symbol_data.columns else 'Unknown',
                'total_swing_lows': 0,
                'active_supports': 0,
                'inactive_supports': 0,
                'avg_percent_above': 0,
                'min_percent_above': 0,
                'max_percent_above': 0,
                'median_percent_above': 0,
                'nearest_support_percent': 999,  # বড় সংখ্যা যাতে নিচে থাকে
                'farthest_support_percent': 0,
                'support_distribution': {},
                'detailed_supports': [],
                'current_price': symbol_data['close'].iloc[-1] if len(symbol_data) > 0 else 0,
                'current_date': symbol_data['date'].iloc[-1] if 'date' in symbol_data.columns else 'Unknown',
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # সুইং লো সনাক্ত
        swing_lows = self.find_swing_lows(symbol_data)
        
        if swing_lows.empty:
            return {
                'symbol': symbol_data['symbol'].iloc[0] if 'symbol' in symbol_data.columns else 'Unknown',
                'total_swing_lows': 0,
                'active_supports': 0,
                'inactive_supports': 0,
                'avg_percent_above': 0,
                'min_percent_above': 0,
                'max_percent_above': 0,
                'median_percent_above': 0,
                'nearest_support_percent': 999,
                'farthest_support_percent': 0,
                'support_distribution': {},
                'detailed_supports': [],
                'current_price': symbol_data['close'].iloc[-1],
                'current_date': symbol_data['date'].iloc[-1] if 'date' in symbol_data.columns else 'Unknown',
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # সাপোর্ট বিশ্লেষণ
        support_analysis = self.calculate_support_percentage(symbol_data, swing_lows)
        
        if not support_analysis:
            return {
                'symbol': symbol_data['symbol'].iloc[0] if 'symbol' in symbol_data.columns else 'Unknown',
                'total_swing_lows': len(swing_lows),
                'active_supports': 0,
                'inactive_supports': len(swing_lows),
                'avg_percent_above': 0,
                'min_percent_above': 0,
                'max_percent_above': 0,
                'median_percent_above': 0,
                'nearest_support_percent': 999,
                'farthest_support_percent': 0,
                'support_distribution': {},
                'detailed_supports': support_analysis,
                'current_price': symbol_data['close'].iloc[-1],
                'current_date': symbol_data['date'].iloc[-1] if 'date' in symbol_data.columns else 'Unknown',
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # পরিসংখ্যান তৈরি
        df_support = pd.DataFrame(support_analysis)
        active_supports = df_support[df_support['is_active_support'] == True]
        
        # পার্সেন্টাইল ডিস্ট্রিবিউশন
        percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        distribution = {}
        
        if not active_supports.empty:
            for p in percentiles:
                percentile_value = np.percentile(active_supports['percent_above_support'], p)
                distribution[f'{p}th_percentile'] = round(percentile_value, 2)
        
        # nearest_support_percent হল min_percent_above (সবচেয়ে কাছের সাপোর্ট)
        nearest_support = round(active_supports['percent_above_support'].min(), 2) if not active_supports.empty else 999
        
        result = {
            'symbol': symbol_data['symbol'].iloc[0] if 'symbol' in symbol_data.columns else 'Unknown',
            'total_swing_lows': len(swing_lows),
            'active_supports': len(active_supports),
            'inactive_supports': len(df_support) - len(active_supports),
            'avg_percent_above': round(active_supports['percent_above_support'].mean(), 2) if not active_supports.empty else 0,
            'min_percent_above': round(active_supports['percent_above_support'].min(), 2) if not active_supports.empty else 0,
            'max_percent_above': round(active_supports['percent_above_support'].max(), 2) if not active_supports.empty else 0,
            'median_percent_above': round(active_supports['percent_above_support'].median(), 2) if not active_supports.empty else 0,
            'nearest_support_percent': nearest_support,  # এই ভ্যালুর উপর ভিত্তি করে sorting হবে
            'farthest_support_percent': round(active_supports['percent_above_support'].max(), 2) if not active_supports.empty else 0,
            'support_distribution': distribution,
            'detailed_supports': support_analysis,
            'current_price': symbol_data['close'].iloc[-1],
            'current_date': symbol_data['date'].iloc[-1] if 'date' in symbol_data.columns else 'Unknown',
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
    
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
        
        # ডাটা প্রিভিউ
        print(f"\nডাটার প্রথম ৫ সারি:")
        print(df.head())
        
        # ডাটার ধরণ চেক
        print(f"\nডাটার ধরণ:")
        print(df.dtypes)
        
        # সিম্বল অনুযায়ী গ্রুপ
        symbols = df['symbol'].unique() if 'symbol' in df.columns else ['Unknown']
        print(f"\nমোট {len(symbols)} টি সিম্বল পাওয়া গেছে")
        print(f"সিম্বল লিস্ট: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
        
        all_results = []
        all_detailed_supports = []
        
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
                continue
            
            result = self.analyze_symbol(symbol_data)
            all_results.append(result)
            
            # ডিটেইলড সাপোর্ট ডাটা সংরক্ষণ
            for support in result['detailed_supports']:
                support['symbol'] = symbol
                support['current_price'] = result['current_price']
                support['current_date'] = result['current_date']
                support['analysis_date'] = result['analysis_date']
                all_detailed_supports.append(support)
            
            print(f"  ✅ সুইং লো: {result['total_swing_lows']}, একটিভ: {result['active_supports']}")
            if result['active_supports'] > 0:
                print(f"  📊 গড় পার্সেন্ট: {result['avg_percent_above']}%, নিকটতম: {result['nearest_support_percent']}%")
            else:
                print(f"  ❌ কোনো সক্রিয় সাপোর্ট নেই")
        
        return all_results, all_detailed_supports

def save_results_to_csv(results, detailed_supports, output_dir='./output/ai_signal'):
    """
    ফলাফল CSV ফাইলে সংরক্ষণ করে
    """
    # আউটপুট ডিরেক্টরি তৈরি
    os.makedirs(output_dir, exist_ok=True)
    
    # মূল রেজাল্ট টেবিল
    summary_data = []
    for r in results:
        summary_data.append({
            'symbol': r['symbol'],
            'current_date': r['current_date'],
            'current_price': r['current_price'],
            'total_swing_lows': r['total_swing_lows'],
            'active_supports': r['active_supports'],
            'inactive_supports': r['inactive_supports'],
            'avg_percent_above': r['avg_percent_above'],
            'min_percent_above': r['min_percent_above'],
            'max_percent_above': r['max_percent_above'],
            'median_percent_above': r['median_percent_above'],
            'nearest_support_percent': r['nearest_support_percent'],
            'farthest_support_percent': r['farthest_support_percent'],
            'analysis_date': r['analysis_date']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # nearest_support_percent এর ভিত্তিতে sorting (যাদের মান সবচেয়ে কম তারা উপরে)
    # 999 মানে কোনো সক্রিয় সাপোর্ট নেই, তারা সবার নিচে থাকবে
    summary_df = summary_df.sort_values('nearest_support_percent', ascending=True)
    
    # পার্সেন্টাইল ডিস্ট্রিবিউশন যোগ করা
    for r in results:
        if r['support_distribution']:
            for percentile, value in r['support_distribution'].items():
                col_name = f"support_{percentile}"
                if col_name not in summary_df.columns:
                    summary_df[col_name] = None
                summary_df.loc[summary_df['symbol'] == r['symbol'], col_name] = value
    
    # মূল রিপোর্ট সংরক্ষণ
    output_file = os.path.join(output_dir, 'ray_support.csv')
    summary_df.to_csv(output_file, index=False)
    print(f"\n✅ মূল রিপোর্ট সংরক্ষণ করা হয়েছে: {output_file}")
    
    # ডিটেইলড সাপোর্ট ডাটা সংরক্ষণ
    detailed_df = pd.DataFrame(detailed_supports)
    if not detailed_df.empty:
        # percent_above_support অনুযায়ী sorting (কম % উপরে আছে তাদের আগে)
        detailed_df = detailed_df.sort_values('percent_above_support', ascending=True)
        detailed_file = os.path.join(output_dir, 'ray_support_detailed.csv')
        detailed_df.to_csv(detailed_file, index=False)
        print(f"✅ ডিটেইলড রিপোর্ট সংরক্ষণ করা হয়েছে: {detailed_file}")
        print(f"   মোট {len(detailed_df)} টি সুইং লো রেকর্ড")
    
    # স্ট্যাটিস্টিক্স রিপোর্ট
    stats = []
    for r in results:
        if r['active_supports'] > 0:
            # nearest_support_percent এর ভিত্তিতে risk level
            if r['nearest_support_percent'] < 2:
                risk = "VERY LOW"
                quality = "EXCELLENT"
            elif r['nearest_support_percent'] < 5:
                risk = "LOW"
                quality = "GOOD"
            elif r['nearest_support_percent'] < 10:
                risk = "MEDIUM"
                quality = "AVERAGE"
            else:
                risk = "HIGH"
                quality = "WEAK"
            
            stats.append({
                'symbol': r['symbol'],
                'current_price': r['current_price'],
                'nearest_support': r['nearest_support_percent'],
                'support_count': r['active_supports'],
                'avg_support_distance': r['avg_percent_above'],
                'risk_level': risk,
                'support_quality': quality,
                'signal': 'STRONG BUY' if risk == 'VERY LOW' else 'BUY' if risk == 'LOW' else 'HOLD' if risk == 'MEDIUM' else 'CAUTION'
            })
    
    if stats:
        stats_df = pd.DataFrame(stats)
        # nearest_support_distance এর ভিত্তিতে sorting
        stats_df = stats_df.sort_values('nearest_support', ascending=True)
        stats_file = os.path.join(output_dir, 'ray_support_stats.csv')
        stats_df.to_csv(stats_file, index=False)
        print(f"✅ স্ট্যাটিস্টিক্স রিপোর্ট সংরক্ষণ করা হয়েছে: {stats_file}")
    
    return summary_df

def print_summary_table(summary_df):
    """
    সংক্ষিপ্ত ফলাফল টেবিল প্রিন্ট করে (কম % সিম্বলগুলো উপরে)
    """
    print("\n" + "="*130)
    print("📊 সুইং লো সাপোর্ট অ্যানালাইসিস রিপোর্ট (কম % সিম্বল উপরে)")
    print("="*130)
    
    # টেবিল হেডার
    print(f"{'Rank':<5} {'Symbol':<12} {'Date':<12} {'Price':<8} {'Active':<6} {'Avg %':<8} {'Nearest':<8} {'Risk':<10} {'Signal':<12}")
    print("-"*130)
    
    strong_buy_count = 0
    buy_count = 0
    hold_count = 0
    caution_count = 0
    no_support_count = 0
    
    for idx, (_, row) in enumerate(summary_df.iterrows(), 1):
        # nearest_support_percent এর ভিত্তিতে ranking এবং signal
        if row['nearest_support_percent'] == 999:
            status = "🚫 NO SUPPORT"
            risk = "HIGH"
            signal = "WAIT"
            no_support_count += 1
        elif row['nearest_support_percent'] < 2:
            status = "🟢 VERY STRONG"
            risk = "VERY LOW"
            signal = "STRONG BUY"
            strong_buy_count += 1
        elif row['nearest_support_percent'] < 5:
            status = "🔵 STRONG"
            risk = "LOW"
            signal = "BUY"
            buy_count += 1
        elif row['nearest_support_percent'] < 10:
            status = "🟡 MODERATE"
            risk = "MEDIUM"
            signal = "HOLD"
            hold_count += 1
        else:
            status = "🔴 WEAK"
            risk = "HIGH"
            signal = "CAUTION"
            caution_count += 1
        
        # Date ফরম্যাটিং
        date_str = str(row['current_date'])[:10] if row['current_date'] != 'Unknown' else 'N/A'
        
        print(f"{idx:<5} {row['symbol']:<12} {date_str:<12} {row['current_price']:<8.2f} "
              f"{row['active_supports']:<6} {row['avg_percent_above']:<8.2f} "
              f"{row['nearest_support_percent']:<8.2f} {risk:<10} {signal:<12}")
    
    print("="*130)
    
    # সারাংশ পরিসংখ্যান
    print("\n📈 সারাংশ:")
    print(f"মোট সিম্বল: {len(summary_df)}")
    print(f"🟢 স্ট্রং বাই ({strong_buy_count}): nearest_support < 2%")
    print(f"🔵 বাই ({buy_count}): nearest_support 2-5%")
    print(f"🟡 হোল্ড ({hold_count}): nearest_support 5-10%")
    print(f"🔴 সতর্কতা ({caution_count}): nearest_support > 10%")
    print(f"🚫 সাপোর্ট নেই ({no_support_count}): কোনো সক্রিয় সাপোর্ট নেই")
    
    # টপ ৫ সিম্বল (নিকটতম সাপোর্ট)
    top_5 = summary_df[summary_df['nearest_support_percent'] < 999].head(5)
    if not top_5.empty:
        print("\n🏆 টপ ৫ সিম্বল (সবচেয়ে কাছের সাপোর্ট):")
        for _, row in top_5.iterrows():
            print(f"   {row['symbol']}: {row['nearest_support_percent']:.2f}% (প্রাইস: {row['current_price']:.2f})")

# মূল প্রোগ্রাম
def main():
    # ইনপুট এবং আউটপুট ফাইল পাথ
    input_file = './csv/mongodb.csv'
    output_dir = './output/ai_signal'
    
    print("="*80)
    print("🚀 সুইং লো রে লাইন সাপোর্ট অ্যানালাইজার")
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
        results, detailed_supports = analyzer.analyze_all_symbols(input_file)
        
        if not results:
            print("❌ কোনো বৈধ ফলাফল পাওয়া যায়নি!")
            return
        
        print(f"\n✅ মোট {len(results)} টি সিম্বল সফলভাবে বিশ্লেষণ করা হয়েছে")
        
        # ফলাফল সংরক্ষণ
        print("\n💾 ফলাফল সংরক্ষণ করা হচ্ছে...")
        summary_df = save_results_to_csv(results, detailed_supports, output_dir)
        
        # সংক্ষিপ্ত ফলাফল প্রিন্ট
        print_summary_table(summary_df)
        
        print(f"\n✅ বিশ্লেষণ সম্পন্ন! ফলাফল {output_dir} ডিরেক্টরিতে সংরক্ষণ করা হয়েছে।")
        print(f"   - ray_support.csv: মূল রিপোর্ট")
        print(f"   - ray_support_detailed.csv: বিস্তারিত সুইং লো ডাটা")
        print(f"   - ray_support_stats.csv: ট্রেডিং সিগনাল সহ স্ট্যাটিস্টিক্স")
        
    except Exception as e:
        print(f"❌ ত্রুটি: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()