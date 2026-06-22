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
        
        # প্রয়োজনীয় কলাম সিলেক্ট করা
        df_filtered = df[['symbol', 'divergence_type', 'strength', 'last_date', 'last_high', 'lr', 'pr', 'last_price', 'pp', 'pd', 'gape']].copy()
        
        # কলাম rename করা
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
            ratio_text = f"{bullish_count}:0"
        else:
            bull_bear_ratio = 0
            ratio_text = "0:0"
        
        print("=" * 60)
        print("DIVERGENCE_TYPE বিশ্লেষণ (১০০% এর মধ্যে):")
        print("=" * 60)
        print(f"  Bullish : {bullish_count}টি ({bullish_pct}%)")
        print(f"  Bearish : {bearish_count}টি ({bearish_pct}%)")
        print(f"  মোট    : {total}টি (১০০%)")
        print(f"  Ratio : {ratio_text} ({bull_bear_ratio}x)")
        print("=" * 60)
        
        # Market Bias Analysis (শুধু প্রিন্টের জন্য)
        if bull_bear_ratio >= 2.0:
            market_bias = "STRONG_BULLISH"
        elif bull_bear_ratio >= 1.5:
            market_bias = "BULLISH"
        elif bull_bear_ratio > 1.0:
            market_bias = "SLIGHTLY_BULLISH"
        elif bull_bear_ratio == 1.0:
            market_bias = "NEUTRAL"
        elif bull_bear_ratio >= 0.5:
            market_bias = "SLIGHTLY_BEARISH"
        elif bull_bear_ratio > 0:
            market_bias = "BEARISH"
        else:
            market_bias = "STRONG_BEARISH"
        
        print(f"  Market Bias : {market_bias}")
        print("=" * 60)
        
        # STRENGTH বিশ্লেষণ
        bullish_strong_count = len(df_filtered[(df_filtered['DIVERGENCE_TYPE_CLEAN'] == 'Bullish') & (df_filtered['STRENGTH_CLEAN'] == 'Strong')])
        bullish_moderate_count = len(df_filtered[(df_filtered['DIVERGENCE_TYPE_CLEAN'] == 'Bullish') & (df_filtered['STRENGTH_CLEAN'] == 'Moderate')])
        bearish_strong_count = len(df_filtered[(df_filtered['DIVERGENCE_TYPE_CLEAN'] == 'Bearish') & (df_filtered['STRENGTH_CLEAN'] == 'Strong')])
        bearish_moderate_count = len(df_filtered[(df_filtered['DIVERGENCE_TYPE_CLEAN'] == 'Bearish') & (df_filtered['STRENGTH_CLEAN'] == 'Moderate')])
        
        print("\nSTRENGTH অনুযায়ী বিশ্লেষণ:")
        print("=" * 60)
        print(f"  Bullish Strong   : {bullish_strong_count}টি")
        print(f"  Bullish Moderate  : {bullish_moderate_count}টি")
        print(f"  Bearish Strong   : {bearish_strong_count}টি")
        print(f"  Bearish Moderate  : {bearish_moderate_count}টি")
        print("=" * 60)
        
        # Bullish Strong ফিল্টার (STRENGTH = Strong এবং LAST_HIGH >= 10)
        bullish_strong = df_filtered[
            (df_filtered['DIVERGENCE_TYPE_CLEAN'] == 'Bullish') & 
            (df_filtered['STRENGTH_CLEAN'] == 'Strong') &
            (df_filtered['LAST_HIGH'] >= 10)
        ].copy()
        
        # শুধু প্রয়োজনীয় কলাম রাখা
        bullish_strong = bullish_strong[['SYMBOL', 'LAST_HIGH', 'GAPE']].copy()
        
        # কলামের নাম ছোট হাতের করা
        bullish_strong.rename(columns={'SYMBOL': 'symbol', 'LAST_HIGH': 'high', 'GAPE': 'gape'}, inplace=True)
        
        # ============= IMPORTANT FIX =============
        # rt: Use RSI ratio (LAST_RSI/PREVIOUS_RSI) or default
        if 'LAST_RSI' in df_filtered.columns and 'PREVIOUS_RSI' in df_filtered.columns:
            # Merge RSI data back to bullish_strong
            rsi_data = df_filtered[['SYMBOL', 'LAST_RSI', 'PREVIOUS_RSI']].copy()
            rsi_data.rename(columns={'SYMBOL': 'symbol'}, inplace=True)
            bullish_strong = bullish_strong.merge(rsi_data, on='symbol', how='left')
            
            # Calculate rt as RSI ratio
            bullish_strong['rt'] = (bullish_strong['LAST_RSI'] / bullish_strong['PREVIOUS_RSI']).round(2)
            # Fill NaN with default
            bullish_strong['rt'] = bullish_strong['rt'].fillna(1.0)
        else:
            # Use ratio_text as string for rt
            bullish_strong['rt'] = bull_bear_ratio  # Numeric value
            
        # bbr: Use Bull/Bear ratio (numeric)
        bullish_strong['bbr'] = bull_bear_ratio
        
        # strong: Use bullish_strong_count as numeric (not text)
        # This is the key fix - strong should be a number
        bullish_strong['strong'] = bullish_strong_count  # Numeric value
        
        # Also keep ratio_text as separate column if needed
        # bullish_strong['strong_ratio'] = ratio_text  # Optional
        
        # ============= END FIX =============
        
        # সিরিয়াল নং বাদ
        bullish_strong.reset_index(drop=True, inplace=True)
        
        # Reorder columns - rt, bbr, strong should be last
        cols = ['symbol', 'high', 'gape', 'rt', 'bbr', 'strong']
        bullish_strong = bullish_strong[cols]
        
        # CSV ফাইল সংরক্ষণ
        output_file = os.path.join(output_dir, 'bullish_strong.csv')
        
        if len(bullish_strong) > 0:
            bullish_strong.to_csv(output_file, index=False)
            print(f"\n✓ Bullish Strong সিগন্যাল পাওয়া গেছে: {len(bullish_strong)}টি")
            print(f"\nফিল্টারকৃত ডেটা (প্রথম 5টি):")
            print(bullish_strong.head().to_string(index=False))
            print(f"\n✓ ফাইল সংরক্ষিত: {output_file}")
            
            # Show column info
            print(f"\n📊 কলাম সমূহ: {bullish_strong.columns.tolist()}")
            print(f"📊 ডেটা টাইপ: {bullish_strong.dtypes}")
        else:
            # খালি ফাইল তৈরি (শুধু হেডার সহ)
            pd.DataFrame(columns=['symbol', 'high', 'gape', 'rt', 'bbr', 'strong']).to_csv(output_file, index=False)
            print(f"\n✗ কোনো Bullish Strong সিগন্যাল পাওয়া যায়নি")
            print(f"  (শর্ত: STRENGTH='Strong' এবং LAST_HIGH>=10)")
            print(f"✓ খালি ফাইল তৈরি করা হয়েছে: {output_file}")
        
        return bullish_strong
        
    except FileNotFoundError:
        print(f"✗ ত্রুটি: {input_file} ফাইলটি পাওয়া যায়নি")
        return pd.DataFrame()
    except Exception as e:
        print(f"✗ ত্রুটি: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    result = process_bullish_strong()
    
    print("\n" + "=" * 60)
    print("আউটপুট ফাইল: ./output/ai_signal/bullish_strong.csv")
    print("কলাম: symbol, high, gape, rt, bbr, strong")
    print(f"মোট Row: {len(result)}টি")
    
    if len(result) > 0:
        print("\nনমুনা ডেটা:")
        print(result[['symbol', 'rt', 'bbr', 'strong']].head())
    print("=" * 60)
