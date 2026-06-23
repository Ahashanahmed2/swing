#!/usr/bin/env python3
import csv
import os
from datetime import datetime
import sys
import pandas as pd
import re

def main():
    input_file = "./output/ai_signal/daily_buy.csv"
    final_output = "./output/ai_signal/strong_ratio.csv"

    os.makedirs(os.path.dirname(final_output), exist_ok=True)

    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        create_default_output(final_output)
        return

    try:
        df = pd.read_csv(input_file)
        
        print(f"📊 Input columns: {df.columns.tolist()}")
        print(f"📊 Input shape: {df.shape}")
        
        # Check for rt, bbr, strong columns
        missing = []
        for col in ['rt', 'bbr', 'strong']:
            if col not in df.columns:
                missing.append(col)
                # Create column with default values
                if col == 'rt':
                    df[col] = 0.0
                elif col == 'bbr':
                    df[col] = 0.0
                elif col == 'strong':
                    df[col] = 0
        
        if missing:
            print(f"⚠️ Created missing columns: {missing}")
        
        # Get first row values
        first_row = df.iloc[0]
        
        # ============= উন্নত টেক্সট হ্যান্ডলিং =============
        
        def extract_numeric(value, default="0.0"):
            """যেকোনো টেক্সট থেকে সংখ্যা বের করে"""
            if pd.isna(value):
                return default
            
            value_str = str(value).strip()
            
            # যদি খালি হয়
            if not value_str:
                return default
            
            # যদি ":" থাকে (যেমন "5:2")
            if ':' in value_str:
                parts = value_str.split(':')
                # প্রথম অংশ নেওয়ার চেষ্টা
                if parts and parts[0].strip():
                    value_str = parts[0].strip()
                else:
                    return default
            
            # সংখ্যা বের করার চেষ্টা (regex)
            match = re.search(r'[-+]?\d*\.?\d+', value_str)
            if match:
                return match.group()
            
            # যদি কোনো সংখ্যা না পাওয়া যায়
            return default
        
        def extract_rt_value(value):
            """RT ভ্যালু এক্সট্রাক্ট (শতকরা বা অনুপাত)"""
            if pd.isna(value):
                return "0.0"
            
            value_str = str(value).strip()
            
            # যদি "bullish_strong_count:bearish_strong_count" ফরম্যাট হয়
            if ':' in value_str:
                parts = value_str.split(':')
                if len(parts) >= 2:
                    try:
                        bullish = float(parts[0].strip())
                        bearish = float(parts[1].strip())
                        if bearish > 0:
                            ratio = bullish / bearish
                            return f"{ratio:.2f}"
                        else:
                            return "0.0"
                    except:
                        pass
            
            # সাধারণ সংখ্যা বের করা
            return extract_numeric(value, "0.0")
        
        def extract_strong_value(value):
            """Strong ভ্যালু এক্সট্রাক্ট (শুধু সংখ্যা)"""
            if pd.isna(value):
                return "0"
            
            value_str = str(value).strip()
            
            # যদি "5:2" ফরম্যাট হয়
            if ':' in value_str:
                parts = value_str.split(':')
                if parts and parts[0].strip():
                    try:
                        # প্রথম অংশকে integer এ কনভার্ট
                        return str(int(float(parts[0].strip())))
                    except:
                        pass
            
            # সাধারণ সংখ্যা বের করা
            num = extract_numeric(value, "0")
            try:
                return str(int(float(num)))
            except:
                return "0"
        
        # ভ্যালু এক্সট্রাক্ট করা
        rt_val = first_row.get('rt', 0)
        bbr_val = first_row.get('bbr', 0)
        strong_val = first_row.get('strong', 0)
        
        # প্রক্রিয়াকরণ
        rt_value = extract_rt_value(rt_val)
        bbr_value = extract_numeric(bbr_val, "0.0")
        strong_value = extract_strong_value(strong_val)
        
        # ============= ডিবাগ ইনফো =============
        print(f"\n📊 Original values from first row:")
        print(f"   RT (raw): {rt_val} (type: {type(rt_val)})")
        print(f"   BBR (raw): {bbr_val} (type: {type(bbr_val)})")
        print(f"   STRONG (raw): {strong_val} (type: {type(strong_val)})")
        print(f"\n📊 Extracted values:")
        print(f"   RT: {rt_value}")
        print(f"   BBR: {bbr_value}")
        print(f"   STRONG: {strong_value}")
        
        # নতুন row তৈরি
        new_row = {
            'rt': rt_value,
            'bbr': bbr_value,
            'strong': strong_value,
            'date': datetime.now().strftime('%Y-%m-%d')
        }

        # CSV তে লেখা
        with open(final_output, 'w', newline='', encoding='utf-8') as outfile:
            fieldnames = ['rt', 'bbr', 'strong', 'date']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(new_row)

        print(f"\n✅ Data saved successfully to: {final_output}")
        print(f"   RT: {rt_value}")
        print(f"   BBR: {bbr_value}")
        print(f"   STRONG: {strong_value}")
        print(f"   Date: {new_row['date']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        create_default_output(final_output)

def create_default_output(output_file):
    """Create default output file"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rt', 'bbr', 'strong', 'date'])
        writer.writerow(['0.0', '0.0', '0', datetime.now().strftime('%Y-%m-%d')])
    print(f"✅ Created default output: {output_file}")

if __name__ == "__main__":
    main()
