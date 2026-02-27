import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ChartPatternDetector:
    def __init__(self, csv_path):
        """
        চার্ট প্যাটার্ন ডিটেক্টর ক্লাস
        """
        self.df = pd.read_csv(csv_path)
        self.df['date'] = pd.to_datetime(self.df['date']) if 'date' in self.df.columns else range(len(self.df))
        self.results = []
        
    def detect_cup_and_handle(self, prices, window=20):
        """
        কাপ এন্ড হ্যান্ডেল প্যাটার্ন ডিটেক্ট করুন
        """
        if len(prices) < window:
            return False
            
        # কাপের আকৃতি চেক করুন (U-shaped)
        recent_prices = prices[-window:]
        mid_point = window // 2
        
        left_side = recent_prices[:mid_point]
        right_side = recent_prices[mid_point:]
        bottom = min(recent_prices)
        
        # কাপের শর্ত চেক করুন
        left_peak = max(left_side)
        right_peak = max(right_side)
        
        # বটম থেকে পিকের অনুপাত
        left_drop = (left_peak - bottom) / left_peak
        right_rise = (right_peak - bottom) / right_peak
        
        # হ্যান্ডেল চেক করুন (শেষের দিকে ছোট ডিপ)
        handle = recent_prices[-5:] if len(recent_prices) >= 5 else recent_prices
        handle_drop = (max(handle) - min(handle)) / max(handle)
        
        if (left_drop > 0.05 and right_rise > 0.05 and 
            left_drop < 0.3 and right_rise < 0.3 and
            handle_drop < 0.1):
            return True
        return False
    
    def detect_bullish_flag(self, prices, window=15):
        """
        বুলিশ ফ্ল্যাগ প্যাটার্ন ডিটেক্ট করুন
        """
        if len(prices) < window:
            return False
            
        # ফ্ল্যাগপোল চেক করুন (দ্রুত বাড়া)
        flagpole = prices[-window:-window//2] if window > 2 else prices[:-1]
        flag = prices[-window//2:]
        
        if len(flagpole) < 2 or len(flag) < 2:
            return False
            
        flagpole_rise = (flagpole[-1] - flagpole[0]) / flagpole[0]
        
        # ফ্ল্যাগ চেক করুন (কনসলিডেশন)
        flag_high = max(flag)
        flag_low = min(flag)
        flag_range = (flag_high - flag_low) / flag_low
        
        if flagpole_rise > 0.03 and flag_range < 0.02:
            return True
        return False
    
    def detect_bearish_flag(self, prices, window=15):
        """
        বিয়ারিশ ফ্ল্যাগ প্যাটার্ন ডিটেক্ট করুন
        """
        if len(prices) < window:
            return False
            
        # ফ্ল্যাগপোল চেক করুন (দ্রুত কমা)
        flagpole = prices[-window:-window//2] if window > 2 else prices[:-1]
        flag = prices[-window//2:]
        
        if len(flagpole) < 2 or len(flag) < 2:
            return False
            
        flagpole_drop = (flagpole[0] - flagpole[-1]) / flagpole[0]
        
        # ফ্ল্যাগ চেক করুন (কনসলিডেশন)
        flag_high = max(flag)
        flag_low = min(flag)
        flag_range = (flag_high - flag_low) / flag_low
        
        if flagpole_drop > 0.03 and flag_range < 0.02:
            return True
        return False
    
    def detect_double_bottom(self, prices, window=20):
        """
        ডাবল বটম প্যাটার্ন ডিটেক্ট করুন (W-shaped)
        """
        if len(prices) < window:
            return False
            
        recent = prices[-window:]
        
        # দুইটি বটম খুঁজুন
        bottoms = []
        for i in range(1, len(recent)-1):
            if recent[i] < recent[i-1] and recent[i] < recent[i+1]:
                bottoms.append((i, recent[i]))
        
        if len(bottoms) < 2:
            return False
            
        # প্রথম এবং শেষ বটম চেক করুন
        first_bottom = bottoms[0][1]
        last_bottom = bottoms[-1][1]
        
        # বটমের মধ্যে দূরত্ব
        bottom_diff = abs(first_bottom - last_bottom) / first_bottom
        
        # মাঝের পিক চেক করুন
        between_prices = recent[bottoms[0][0]:bottoms[-1][0]]
        middle_peak = max(between_prices) if len(between_prices) > 0 else 0
        peak_height = (middle_peak - first_bottom) / first_bottom if first_bottom > 0 else 0
        
        if bottom_diff < 0.02 and peak_height > 0.02:
            return True
        return False
    
    def detect_head_and_shoulders(self, prices, window=30):
        """
        হেড এন্ড শোল্ডার্স প্যাটার্ন ডিটেক্ট করুন
        """
        if len(prices) < window:
            return False
            
        recent = prices[-window:]
        
        # পিক পয়েন্ট খুঁজুন
        peaks = []
        for i in range(1, len(recent)-1):
            if recent[i] > recent[i-1] and recent[i] > recent[i+1]:
                peaks.append((i, recent[i]))
        
        if len(peaks) < 3:
            return False
            
        # তিনটি পিক চেক করুন (বাম কাঁধ, মাথা, ডান কাঁধ)
        left_shoulder = peaks[0][1]
        head = peaks[1][1] if len(peaks) > 1 else 0
        right_shoulder = peaks[2][1] if len(peaks) > 2 else 0
        
        if head > left_shoulder and head > right_shoulder:
            return True
        return False
    
    def detect_rounding_bottom(self, prices, window=20):
        """
        রাউন্ডিং বটম (সসার) প্যাটার্ন ডিটেক্ট করুন
        """
        if len(prices) < window:
            return False
            
        recent = prices[-window:]
        mid_point = window // 2
        
        left_side = recent[:mid_point]
        right_side = recent[mid_point:]
        bottom = min(recent)
        
        left_trend = left_side[-1] - left_side[0] if len(left_side) > 1 else 0
        right_trend = right_side[-1] - right_side[0] if len(right_side) > 1 else 0
        
        # বাম পাশে ডাউনট্রেন্ড, ডান পাশে আপট্রেন্ড
        if left_trend < 0 and right_trend > 0:
            return True
        return False
    
    def analyze_symbol(self, symbol_data):
        """
        প্রতিটি সিম্বলের জন্য প্যাটার্ন বিশ্লেষণ করুন
        """
        prices = symbol_data['close'].values
        
        patterns = []
        
        # বিভিন্ন প্যাটার্ন ডিটেক্ট করুন
        if self.detect_cup_and_handle(prices):
            patterns.append('cup_and_handle')
            
        if self.detect_bullish_flag(prices):
            patterns.append('bullish_flag')
            
        if self.detect_bearish_flag(prices):
            patterns.append('bearish_flag')
            
        if self.detect_double_bottom(prices):
            patterns.append('double_bottom')
            
        if self.detect_head_and_shoulders(prices):
            patterns.append('head_and_shoulders')
            
        if self.detect_rounding_bottom(prices):
            patterns.append('rounding_bottom')
        
        # ডিফল্ট প্যাটার্ন যদি কিছু না পাওয়া যায়
        if not patterns:
            # ট্রেন্ড চেক করুন
            if len(prices) > 5:
                short_trend = (prices[-1] - prices[-5]) / prices[-5]
                if short_trend > 0.02:
                    patterns.append('uptrend')
                elif short_trend < -0.02:
                    patterns.append('downtrend')
                else:
                    patterns.append('sideways')
            else:
                patterns.append('insufficient_data')
        
        return patterns
    
    def process_all_symbols(self):
        """
        সব সিম্বল প্রসেস করুন
        """
        symbols = self.df['symbol'].unique()
        
        for symbol in symbols:
            symbol_data = self.df[self.df['symbol'] == symbol].sort_values('date')
            
            if len(symbol_data) >= 20:  # মিনিমাম 20টি ক্যান্ডেল লাগবে
                patterns = self.analyze_symbol(symbol_data)
                
                # প্রতিটি প্যাটার্নের জন্য আলাদা রো তৈরি করুন
                for pattern in patterns:
                    self.results.append({
                        'symbol': symbol,
                        'pattern': pattern,
                        'last_close': symbol_data['close'].iloc[-1],
                        'last_open': symbol_data['open'].iloc[-1],
                        'last_high': symbol_data['high'].iloc[-1],
                        'last_low': symbol_data['low'].iloc[-1],
                        'volume': symbol_data.get('volume', pd.Series([0])).iloc[-1] if 'volume' in symbol_data.columns else 0,
                        'date': symbol_data['date'].iloc[-1]
                    })
    
    def save_results(self, output_path):
        """
        রেজাল্ট CSV ফাইলে সেভ করুন
        """
        if self.results:
            result_df = pd.DataFrame(self.results)
            result_df.to_csv(output_path, index=False)
            print(f"✅ ফলাফল সেভ করা হয়েছে: {output_path}")
            print(f"   মোট {len(result_df)}টি প্যাটার্ন পাওয়া গেছে")
            print(f"\nপ্যাটার্ন সমূহের পরিসংখ্যান:")
            print(result_df['pattern'].value_counts())
            return result_df
        else:
            print("❌ কোন প্যাটার্ন পাওয়া যায়নি!")
            return None

# মেইন ফাংশন
def main():
    # ইনপুট এবং আউটপুট ফাইল পাথ
    input_file = "./csv/mongodb.csv"
    output_file = "./csv/paratn.csv"  # pattern এর বানান ভুল হয়েছে, আপনি চাইলে ঠিক করতে পারেন
    
    # চেক করুন ইনপুট ফাইল আছে কিনা
    if not Path(input_file).exists():
        print(f"❌ এরর: {input_file} ফাইলটি পাওয়া যায়নি!")
        return
    
    # প্যাটার্ন ডিটেক্টর তৈরি করুন
    detector = ChartPatternDetector(input_file)
    
    # সব সিম্বল প্রসেস করুন
    print("🔄 প্যাটার্ন বিশ্লেষণ চলছে...")
    detector.process_all_symbols()
    
    # রেজাল্ট সেভ করুন
    result = detector.save_results(output_file)
    
    if result is not None:
        print(f"\n📊 স্যাম্পল ডাটা:")
        print(result[['symbol', 'pattern', 'last_close']].head(10))

if __name__ == "__main__":
    main()