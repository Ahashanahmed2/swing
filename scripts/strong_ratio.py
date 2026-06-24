#!/usr/bin/env python3
import csv
import os
from datetime import datetime
import sys
import pandas as pd

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
        print(f"📊 Total rows: {len(df)}")
        print(f"\n📊 First 5 rows (relevant columns):")
        
        # Show only rt, bbr, strong columns if they exist
        show_cols = []
        for col in ['rt', 'bbr', 'strong']:
            if col in df.columns:
                show_cols.append(col)
        if show_cols:
            print(df[show_cols].head())
        else:
            print(df.head())
        
        # ============= প্রথম নন-খালি মান খুঁজে বের করা =============
        
        def get_first_non_empty_value(df, column_name, default="0.0"):
            """একটি কলাম থেকে প্রথম নন-খালি (non-NaN, non-empty) মান রিটার্ন করে"""
            if column_name not in df.columns:
                print(f"⚠️ কলাম '{column_name}' পাওয়া যায়নি")
                return default
            
            for idx, value in enumerate(df[column_name]):
                if pd.isna(value):
                    continue
                
                value_str = str(value).strip()
                if value_str == '' or value_str == 'nan' or value_str == 'None':
                    continue
                
                print(f"✅ {column_name}: প্রথম নন-খালি মান পেয়েছি row {idx+1} → '{value_str}'")
                return value_str
            
            print(f"⚠️ {column_name}: কোনো নন-খালি মান পাওয়া যায়নি")
            return default
        
        # প্রথমে raw মানগুলো নিই
        rt_raw = get_first_non_empty_value(df, 'rt', '0.0')
        bbr_raw = get_first_non_empty_value(df, 'bbr', '0.0')
        strong_raw = get_first_non_empty_value(df, 'strong', '0')
        
        # ============= ভ্যালু প্রসেস করা =============
        # অপশন ৩: যেভাবে আছে সেভাবেই রাখা, কোনো পরিবর্তন করা হবে না
        
        def keep_as_is(value, default="0.0"):
            """ভ্যালু যেভাবে আছে সেভাবেই রাখা"""
            if not value or value == '0.0' or value == '0':
                return default
            return str(value).strip()
        
        # প্রসেস করা মান (কোনো পরিবর্তন ছাড়া)
        rt_value = keep_as_is(rt_raw, "0.0")
        bbr_value = keep_as_is(bbr_raw, "0.0")
        strong_value = keep_as_is(strong_raw, "0")
        
        # ============= ডিবাগ ইনফো =============
        print(f"\n📊 Raw values found:")
        print(f"   RT (raw): {rt_raw}")
        print(f"   BBR (raw): {bbr_raw}")
        print(f"   STRONG (raw): {strong_raw}")
        
        print(f"\n📊 Final values (kept as-is):")
        print(f"   RT: {rt_value}")
        print(f"   BBR: {bbr_value}")
        print(f"   STRONG: {strong_value}")
        
        # নতুন row তৈরি (শুধু ৪টি কলাম)
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
        print(f"   Columns: {fieldnames}")
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
