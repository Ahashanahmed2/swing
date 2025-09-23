import pandas as pd
import numpy as np
import re

# CSV ফাইল লোড করুন এবং ডাটা টাইপ কনভার্ট করুন
df = pd.read_csv('./../csv/dse_financial_data.csv')

# LTP কলামটি numeric এ কনভার্ট করুন
df['LTP'] = pd.to_numeric(df['LTP'], errors='coerce')

# ডাইনামিকভাবে Div এবং EPS কলাম আইডেন্টিফাই করুন
def get_dynamic_columns(df):
    # Div কলামগুলো (যেগুলো 'Div' দিয়ে শুরু হয়)
    div_cols = [col for col in df.columns if col.startswith('Div')]
    # EPS কলামগুলো (যেগুলো 'EPS' দিয়ে শুরু হয়)
    eps_cols = [col for col in df.columns if col.startswith('EPS')]
    
    # সাল অনুযায়ী সাজান (নতুন থেকে পুরানো)
    def sort_by_year(cols):
        year_cols = {}
        for col in cols:
            # সাল extract করুন
            year_match = re.search(r'\d{4}', col)
            if year_match:
                year = int(year_match.group())
                year_cols[col] = year
        
        # সাল অনুযায়ী descending order এ সাজান
        sorted_cols = sorted(year_cols.keys(), key=lambda x: year_cols[x], reverse=True)
        return sorted_cols
    
    div_columns = sort_by_year(div_cols)
   
    eps_columns = sort_by_year(eps_cols)
  
    
    return div_columns, eps_columns

# ডাইনামিক কলাম পেতে ফাংশন কল করুন
div_columns, eps_columns = get_dynamic_columns(df)

print(f"পাওয়া Div কলামগুলো: {div_columns}")
print(f"পাওয়া EPS কলামগুলো: {eps_columns}")

def get_valid_values(row, columns, num_values=5, is_dividend=False):
    """প্রথম থেকে num_values টি ভেলু নিন, যদি প্রথমটি খালি থাকে তবে দ্বিতীয় থেকে শুরু করুন"""
    values = []
    start_index = 0
    
    # প্রথম ভেলু খালি কিনা চেক করুন
    if len(columns) > 0:
        first_val = row[columns[0]]
        if pd.isna(first_val) or first_val == '' or str(first_val).strip() == 'nan':
            start_index = 1  # প্রথম ভেলু খালি হলে দ্বিতীয় থেকে শুরু করুন
    
    # নির্দিষ্ট সংখ্যক ভেলু নিন
    for i in range(start_index, min(start_index + num_values, len(columns))):
        col = columns[i]
        val = row[col]
        if pd.notna(val) and val != '' and str(val).strip() != 'nan':
            try:
                # ভেলুটি numeric এ কনভার্ট করুন
                numeric_val = pd.to_numeric(val, errors='coerce')
                if not pd.isna(numeric_val):
                    # যদি ডিভিডেন্ট ভেলু হয়, তাহলে ১০০ দ্বারা ভাগ করুন
                    if is_dividend:
                        numeric_val = numeric_val / 100
                    values.append(float(numeric_val))
            except (ValueError, TypeError):
                continue
    
    return values

def calculate_metrics(row):
    # LTP নিন (ইতিমধ্যে numeric এ কনভার্ট করা হয়েছে)
    ltp = row['LTP']
    sector = row['Sector']
    
    # LTP ভেলিড কিনা চেক করুন
    if pd.isna(ltp) or ltp <= 0:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
    
    # DIV ভেলু প্রসেসিং - প্রথম ৫টি ভেলু নিন (প্রথমটি খালি হলে দ্বিতীয় থেকে)
    # is_dividend=True পাস করুন যাতে ডিভিডেন্ট ভেলুগুলো ১০০ দ্বারা ভাগ হয়
    div_values = get_valid_values(row, div_columns, 5, is_dividend=True)
    
    # EPS ভেলু প্রসেসিং - প্রথম ৫টি ভেলু নিন (প্রথমটি খালি হলে দ্বিতীয় থেকে)
    eps_values = get_valid_values(row, eps_columns, 5, is_dividend=False)
    
    # যদি পর্যাপ্ত ডাটা না থাকে, NaN রিটার্ন করুন
    if len(eps_values) < 2 or len(div_values) < 2:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
    
    # PE রেশিও ক্যালকুলেশন (সাম্প্রতিক EPS ব্যবহার করে)
    recent_eps = eps_values[0] if len(eps_values) > 0 else np.nan
    if pd.isna(recent_eps) or recent_eps <= 0:
        pe = np.nan
    else:
        pe = ltp / recent_eps
    
    # EPS Growth Rate (ECG) - CAGR ক্যালকুলেশন
    if len(eps_values) >= 2:
        start_eps = eps_values[-1]  # প্রথম ভেলু (সবচেয়ে পুরানো)
        end_eps = eps_values[0]     # শেষ ভেলু (সবচেয়ে সাম্প্রতিক)
        n_years = len(eps_values) - 1
        if start_eps > 0 and n_years > 0 and end_eps > 0:
            cagr_eps = (end_eps / start_eps) ** (1/n_years) - 1
            ecg = cagr_eps * 100  # Percentage হিসেবে
        else:
            ecg = np.nan
    else:
        ecg = np.nan
    
    # Dividend Yield (DY) - CORRECTED CALCULATION
    recent_div = div_values[0] if len(div_values) > 0 else np.nan
    if pd.isna(recent_div) or recent_div <= 0:
        dy = np.nan
    else:
        # Correction: DY = (Dividend / LTP) * 100
        dy = ((recent_div / ltp) * 100) * 10
        
        # DY কে reasonable limit-এ bound করুন (0% থেকে 20% পর্যন্ত)
        if dy > 20:  # 20% এর বেশি DY unrealistic
            print(f"Warning: {row['Code']} has high DY: {dy:.2f}%")
        if dy < 0:   # Negative DY impossible
            dy = np.nan
    
    # Dividend Growth Rate (DCG) - CAGR ক্যালকুলেশন
    if len(div_values) >= 2:
        start_div = div_values[-1]  # প্রথম ভেলু (সবচেয়ে পুরানো)
        end_div = div_values[0]     # শেষ ভেলু (সবচেয়ে সাম্প্রতিক)
        n_years_div = len(div_values) - 1
        if start_div > 0 and n_years_div > 0 and end_div > 0:
            cagr_div = (end_div / start_div) ** (1/n_years_div) - 1
            dcg = cagr_div * 100  # Percentage হিসেবে
            
            # DCG কেও reasonable limit-এ bound করুন
            if abs(dcg) > 100:  # 100% এর বেশি growth/decline unrealistic
                dcg = np.nan
        else:
            dcg = np.nan
    else:
        dcg = np.nan
    
    # PEGY Ratio - additional validation সহ
    if (not np.isnan(pe) and not np.isnan(ecg) and not np.isnan(dy) and 
        (ecg + dy) != 0 and abs(ecg + dy) < 1000):  # Avoid extreme values
        pegy = pe / (ecg + dy)
        
        # PEGY কেও reasonable range-এ bound করুন
        if abs(pegy) > 100:  # Very high/low PEGY likely data error
            pegy = np.nan
    else:
        pegy = np.nan
    
    return pd.Series([pe, ecg, dy, dcg, pegy])

# মেট্রিক্স ক্যালকুলেশন অ্যাপ্লাই করুন
metrics = df.apply(calculate_metrics, axis=1)

# সংক্ষিপ্ত কলাম নাম ব্যবহার করুন
metrics.columns = ['PE', 'ECG', 'DY', 'DCG', 'PEGY']

# রেজাল্ট ডাটাফ্রেম তৈরি করুন
result_df = pd.concat([df[['Code', 'LTP', 'Sector']], metrics], axis=1)

# PEGY Ratio অনুসারে সাজান (কম PEGY উপরে, বেশি PEGY নিচে)
valid_pegy_df = result_df.dropna(subset=['PEGY']).sort_values(by='PEGY', ascending=True)
invalid_pegy_df = result_df[result_df['PEGY'].isna()]

# Valid এবং invalid ডাটা একত্রিত করুন
result_df_sorted = pd.concat([valid_pegy_df, invalid_pegy_df])

# CSV ফাইলে সেভ করুন - দশমিকের পর ২টি সংখ্যা সহ
result_df_sorted.to_csv('./../csv/financial_ratios_results.csv', index=False, 
                 float_format='%.2f')

print(f"\nResults saved to 'financial_ratios_results.csv'")
print(f"Total companies processed: {len(result_df_sorted)}")
print(f"Successful PEGY calculations: {len(valid_pegy_df)}")

# Sample verification - কিছু স্টকের জন্য ডিভিডেন্ট correction verify করুন
print("\nDividend Correction Verification:")
print("="*50)
sample_stocks = ['RECKITTBEN', 'MARICO', 'PADMAOIL', 'BATBC', 'LINDEBD']

for code in sample_stocks:
    if code in result_df_sorted['Code'].values:
        stock_data = result_df_sorted[result_df_sorted['Code'] == code].iloc[0]
        original_div = df[df['Code'] == code][div_columns[0]].iloc[0] if not df[df['Code'] == code][div_columns[0]].isna().iloc[0] else "N/A"
        corrected_div = original_div / 100 if original_div != "N/A" else "N/A"
        
        print(f"{code}:")
        print(f"  Original Div: {original_div} -> Corrected: {corrected_div}")
        print(f"  LTP: {stock_data['LTP']}")
        print(f"  DY: {stock_data['DY']:.2f}%")
        print(f"  PEGY: {stock_data['PEGY']:.2f}" if not pd.isna(stock_data['PEGY']) else "  PEGY: N/A")
        print()

# Top 20 Valid PEGY Stocks
print("\nTop 20 Stocks by PEGY Ratio (After Dividend Correction):")
print("="*80)
top_20_valid = valid_pegy_df.head(20)
display_columns = ['Code', 'LTP', 'Sector', 'PE', 'ECG', 'DY', 'DCG', 'PEGY']
print(top_20_valid[display_columns].to_string(index=False))

# DY Statistics
print("\nDY Statistics (After Correction):")
print("="*40)
dy_stats = result_df_sorted['DY'].dropna()
if len(dy_stats) > 0:
    print(f"Average DY: {dy_stats.mean():.2f}%")
    print(f"Min DY: {dy_stats.min():.2f}%")
    print(f"Max DY: {dy_stats.max():.2f}%")
    print(f"Median DY: {dy_stats.median():.2f}%")
    
    # DY distribution
    print("\nDY Distribution:")
    dy_0_5 = len(dy_stats[dy_stats <= 5])
    dy_5_10 = len(dy_stats[(dy_stats > 5) & (dy_stats <= 10)])
    dy_10_20 = len(dy_stats[(dy_stats > 10) & (dy_stats <= 20)])
    dy_20_plus = len(dy_stats[dy_stats > 20])
    
    print(f"DY ≤ 5%: {dy_0_5} stocks")
    print(f"5% < DY ≤ 10%: {dy_5_10} stocks")
    print(f"10% < DY ≤ 20%: {dy_10_20} stocks")
    print(f"DY > 20%: {dy_20_plus} stocks")

print("\nরেজাল্ট সফলভাবে CSV ফাইলে সেভ করা হয়েছে")
print("ডিভিডেন্ট ভেলুগুলো ১০০ দ্বারা ভাগ করে realistic DY পেতে correction করা হয়েছে")