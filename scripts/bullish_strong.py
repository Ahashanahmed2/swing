#bullish_strong.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os

def process_bullish_strong(input_file='./csv/rsi_diver.csv', output_dir='./output/ai_signal/'):
    """
    RSI Divergence CSV থেকে Bullish Strong সিগন্যাল ফিল্টার করে
    """
    
    # আউটপুট ডিরেক্টরি তৈরি
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # CSV ফাইল পড়া
        df = pd.read_csv(input_file)
        
        # প্রয়োজনীয় কলাম সিলেক্ট করা (lr, pr যোগ করা হয়েছে)
        df_filtered = df[['symbol', 'divergence_type', 'strength', 'last_date', 'last_high', 'lr', 'pr', 'last_price', 'pp', 'pd', 'gape']].copy()
        
        # কলাম rename করে বড় হাতের করা
        df_filtered.columns = ['SYMBOL', 'DIVERGENCE_TYPE', 'STRENGTH', 'LAST_DATE', 'LAST_HIGH', 
                               'LAST_RSI', 'PREVIOUS_RSI', 'LAST_PRICE', 'PREVIOUS_PRICE', 'PREVIOUS_DATE', 'GAPE']
        
        # DIVERGENCE_TYPE বিশ্লেষণ
        df_filtered['DIVERGENCE_TYPE_CLEAN'] = df_filtered['DIVERGENCE_TYPE'].str.strip()
        df_filtered['STRENGTH_CLEAN'] = df_filtered['STRENGTH'].str.strip()
        
        total = len(df_filtered)
        bullish_count = len(df_filtered[df_filtered['DIVERGENCE_TYPE_CLEAN'] == 'Bullish'])
        bearish_count = len(df_filtered[df_filtered['DIVERGENCE_TYPE_CLEAN'] == 'Bearish'])
        
        bullish_pct = round((bullish_count / total * 100), 2) if total > 0 else 0
        bearish_pct = round((bearish_count / total * 100), 2) if total > 0 else 0
        
        # Bullish:Bearish Ratio calculation
        if bearish_count > 0:
            bull_bear_ratio = round(bullish_count / bearish_count, 2)
            ratio_text = f"{bullish_count}:{bearish_count}"
        elif bullish_count > 0:
            bull_bear_ratio = float('inf')
            ratio_text = f"{bullish_count}:0 (All Bullish)"
        else:
            bull_bear_ratio = 0
            ratio_text = "0:0 (No signals)"
        
        print("=" * 60)
        print("DIVERGENCE_TYPE বিশ্লেষণ (১০০% এর মধ্যে):")
        print("=" * 60)
        print(f"  Bullish : {bullish_count}টি ({bullish_pct}%)")
        print(f"  Bearish : {bearish_count}টি ({bearish_pct}%)")
        print(f"  মোট    : {total}টি (১০০%)")
        print(f"  Bullish:Bearish Ratio : {ratio_text} ({bull_bear_ratio}x)")
        print("=" * 60)
        
        # Market Bias Analysis based on ratio
        if bull_bear_ratio >= 2.0:
            market_bias = "STRONG_BULLISH"
            bias_comment = "Bullish signals 2x+ more than Bearish"
        elif bull_bear_ratio >= 1.5:
            market_bias = "BULLISH"
            bias_comment = "Bullish signals significantly higher"
        elif bull_bear_ratio > 1.0:
            market_bias = "SLIGHTLY_BULLISH"
            bias_comment = "Bullish signals slightly higher"
        elif bull_bear_ratio == 1.0:
            market_bias = "NEUTRAL"
            bias_comment = "Equal Bullish and Bearish signals"
        elif bull_bear_ratio >= 0.5:
            market_bias = "SLIGHTLY_BEARISH"
            bias_comment = "Bearish signals slightly higher"
        elif bull_bear_ratio > 0:
            market_bias = "BEARISH"
            bias_comment = "Bearish signals significantly higher"
        else:
            market_bias = "STRONG_BEARISH"
            bias_comment = "All Bearish signals"
        
        print(f"  Market Bias : {market_bias}")
        print(f"  Comment     : {bias_comment}")
        print("=" * 60)
        
        # STRENGTH বিশ্লেষণ
        bullish_strong_count = len(df_filtered[(df_filtered['DIVERGENCE_TYPE_CLEAN'] == 'Bullish') & (df_filtered['STRENGTH_CLEAN'] == 'Strong')])
        bullish_moderate_count = len(df_filtered[(df_filtered['DIVERGENCE_TYPE_CLEAN'] == 'Bullish') & (df_filtered['STRENGTH_CLEAN'] == 'Moderate')])
        bearish_strong_count = len(df_filtered[(df_filtered['DIVERGENCE_TYPE_CLEAN'] == 'Bearish') & (df_filtered['STRENGTH_CLEAN'] == 'Strong')])
        bearish_moderate_count = len(df_filtered[(df_filtered['DIVERGENCE_TYPE_CLEAN'] == 'Bearish') & (df_filtered['STRENGTH_CLEAN'] == 'Moderate')])
        
        # Strong Ratio
        if bearish_strong_count > 0:
            strong_ratio = round(bullish_strong_count / bearish_strong_count, 2)
            strong_ratio_text = f"{bullish_strong_count}:{bearish_strong_count}"
        elif bullish_strong_count > 0:
            strong_ratio = float('inf')
            strong_ratio_text = f"{bullish_strong_count}:0 (All Bullish Strong)"
        else:
            strong_ratio = 0
            strong_ratio_text = "0:0 (No Strong signals)"
        
        print("\nSTRENGTH অনুযায়ী বিশ্লেষণ:")
        print("=" * 60)
        print(f"  Bullish Strong   : {bullish_strong_count}টি")
        print(f"  Bullish Moderate  : {bullish_moderate_count}টি")
        print(f"  Bearish Strong   : {bearish_strong_count}টি")
        print(f"  Bearish Moderate  : {bearish_moderate_count}টি")
        print(f"  Strong Ratio (B:B) : {strong_ratio_text} ({strong_ratio}x)")
        print("=" * 60)
        
        # Bullish Strong ফিল্টার (STRENGTH = Strong এবং LAST_HIGH >= 10)
        bullish_strong = df_filtered[
            (df_filtered['DIVERGENCE_TYPE_CLEAN'] == 'Bullish') & 
            (df_filtered['STRENGTH_CLEAN'] == 'Strong') &
            (df_filtered['LAST_HIGH'] >= 10)
        ].copy()
        
        # শুধু প্রয়োজনীয় কলাম রাখা (LAST_DATE বাদ, RSI কলাম যোগ)
        bullish_strong = bullish_strong[['SYMBOL', 'LAST_HIGH', 'LAST_RSI', 'PREVIOUS_RSI', 'LAST_PRICE', 'PREVIOUS_PRICE', 'PREVIOUS_DATE', 'GAPE']].copy()
        
        # সিরিয়াল নং যোগ করা
        bullish_strong.reset_index(drop=True, inplace=True)
        bullish_strong.index = bullish_strong.index + 1  # 1 থেকে শুরু
        bullish_strong.insert(0, 'NO', bullish_strong.index)
        
        # Market statistics কলাম যোগ করা (Ratio সহ)
        bullish_strong['BULLISH_COUNT'] = bullish_count
        bullish_strong['BULLISH_PCT'] = bullish_pct
        bullish_strong['BEARISH_COUNT'] = bearish_count
        bullish_strong['BEARISH_PCT'] = bearish_pct
        bullish_strong['BULL_BEAR_RATIO'] = bull_bear_ratio
        bullish_strong['RATIO_TEXT'] = ratio_text
        bullish_strong['MARKET_BIAS'] = market_bias
        
        # CSV ফাইল সংরক্ষণ
        output_file = os.path.join(output_dir, 'bullish_strong.csv')
        
        if len(bullish_strong) > 0:
            bullish_strong.to_csv(output_file, index=False)
            print(f"\n✓ Bullish Strong সিগন্যাল পাওয়া গেছে: {len(bullish_strong)}টি")
            print(f"\nফিল্টারকৃত ডেটা:")
            print(bullish_strong.to_string(index=False))
            print(f"\n✓ ফাইল সংরক্ষিত: {output_file}")
        else:
            # খালি ফাইল তৈরি (শুধু হেডার সহ)
            bullish_strong.to_csv(output_file, index=False)
            print(f"\n✗ কোনো Bullish Strong সিগন্যাল পাওয়া যায়নি")
            print(f"  (শর্ত: STRENGTH='Strong' এবং LAST_HIGH>=10)")
            print(f"✓ খালি ফাইল তৈরি করা হয়েছে: {output_file}")
            print(f"  মোট Row: 0টি")
        
        return bullish_strong
        
    except FileNotFoundError:
        print(f"✗ ত্রুটি: {input_file} ফাইলটি পাওয়া যায়নি")
        return pd.DataFrame()
    except Exception as e:
        print(f"✗ ত্রুটি: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # স্ক্রিপ্ট রান করা
    result = process_bullish_strong()
    
    print("\n" + "=" * 60)
    print("আউটপুট ফাইল: ./output/ai_signal/bullish_strong.csv")
    print("কলাম: NO, SYMBOL, LAST_HIGH, LAST_RSI, PREVIOUS_RSI,")
    print("       LAST_PRICE, PREVIOUS_PRICE, PREVIOUS_DATE, GAPE,")
    print("       BULLISH_COUNT, BULLISH_PCT, BEARISH_COUNT, BEARISH_PCT,")
    print("       BULL_BEAR_RATIO, RATIO_TEXT, MARKET_BIAS")
    print(f"মোট Row: {len(result)}টি")
    print("=" * 60)
