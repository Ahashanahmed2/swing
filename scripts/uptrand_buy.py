#uptrand_buy.py
import pandas as pd
import os
from datetime import datetime


def detect_swing_highs(df):
    """
    Swing high detection logic:
    একটি row swing high হবে যদি:
    - তার high, আগের row-এর high থেকে বড় হয়
    - তার high, তার আগের row-এর high থেকেও বড় হয়
    - তার high, পরের row-এর high থেকে বড় হয়
    - তার high, তার পরের row-এর high থেকেও বড় হয়
    """
    swing_highs = []
    
    if len(df) < 5:
        return swing_highs
    
    for i in range(2, len(df) - 2):
        current_high = df.iloc[i]['high']
        prev1_high = df.iloc[i-1]['high']
        prev2_high = df.iloc[i-2]['high']
        next1_high = df.iloc[i+1]['high']
        next2_high = df.iloc[i+2]['high']
        
        # Swing high condition
        if (current_high > prev1_high and 
            current_high > prev2_high and 
            current_high > next1_high and 
            current_high > next2_high):
            
            swing_highs.append({
                'date': df.iloc[i]['date'],
                'high': current_high
            })
    
    return swing_highs


def validate_uptrand_entry(historical_df, uptrand_entry):
    """
    একটি uptrand entry ভ্যালিড কিনা চেক করে
    """
    uptrand_date = pd.to_datetime(uptrand_entry['date'])
    symbol = uptrand_entry['symbol']
    
    # symbol এর ডেটা ফিল্টার করে date অনুযায়ী সাজানো
    symbol_df = historical_df[historical_df['symbol'] == symbol].copy()
    symbol_df = symbol_df.sort_values('date').reset_index(drop=True)
    
    if len(symbol_df) < 5:
        return False, None
    
    # latest row এবং তার আগের row খুঁজে বের করা
    latest_row = symbol_df.iloc[-1]
    prev_row = symbol_df.iloc[-2]
    
    # চেক ১: লেটেস্ট row এর close > আগের row এর high
    if not (latest_row['close'] > prev_row['high']):
        return False, None
    
    # uptrand.csv এর date এবং latest date এর মধ্যে row গুলো filter করা
    mask = (symbol_df['date'] > uptrand_date) & (symbol_df['date'] <= latest_row['date'])
    in_between_df = symbol_df[mask].copy()
    
    if len(in_between_df) < 5:
        return False, None
    
    # swing high খোঁজা
    extended_mask = (symbol_df['date'] >= uptrand_date - pd.Timedelta(days=10)) & (symbol_df['date'] <= latest_row['date'])
    analysis_df = symbol_df[extended_mask].copy()
    
    swing_highs = detect_swing_highs(analysis_df)
    
    if len(swing_highs) == 0:
        return False, None
    
    # in-between-এ অন্তত একটি swing high আছে কিনা
    in_between_swing_highs = [
        sh for sh in swing_highs 
        if sh['date'] > uptrand_date and sh['date'] < latest_row['date']
    ]
    
    if len(in_between_swing_highs) == 0:
        return False, None
    
    return True, {
        'symbol': symbol,
        'date': latest_row['date'],
        'close': latest_row['close'],
        'high': latest_row['high'],
        'prev_high': prev_row['high'],
        'swing_highs': in_between_swing_highs,
        'uptrand_date': uptrand_date
    }


def create_uptrand_buy_signals():
    """
    Main function to create uptrand buy signals
    """
    mongodb_csv = './csv/mongodb.csv'
    uptrand_csv = './csv/uptrand.csv'
    output_dir = './output/ai_signal/'
    output_file = os.path.join(output_dir, 'uptrand_buy.csv')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ফাইল চেক
    if not os.path.exists(mongodb_csv):
        print(f"❌ Error: {mongodb_csv} not found!")
        return
    
    if not os.path.exists(uptrand_csv):
        print(f"❌ Error: {uptrand_csv} not found!")
        return
    
    # mongodb.csv-তে high কলাম চেক
    mongodb_df = pd.read_csv(mongodb_csv)
    required_columns = ['symbol', 'date', 'close', 'high']
    for col in required_columns:
        if col not in mongodb_df.columns:
            print(f"❌ Missing column in mongodb.csv: {col}")
            return
    
    mongodb_df['date'] = pd.to_datetime(mongodb_df['date'])
    
    print(f"📖 Reading {uptrand_csv}...")
    try:
        uptrand_df = pd.read_csv(uptrand_csv)
        uptrand_df['date'] = pd.to_datetime(uptrand_df['date'])
    except pd.errors.EmptyDataError:
        print("⚠️ uptrand.csv is empty")
        return
    
    if uptrand_df.empty:
        print("⚠️ No symbols in uptrand.csv")
        return
    
    print(f"✅ Found {len(uptrand_df)} uptrand entries")
    
    valid_signals = []
    processed_symbols = []  # ✅ প্রসেস করা symbol গুলো ট্র্যাক করা
    
    for idx, uptrand_entry in uptrand_df.iterrows():
        symbol = uptrand_entry['symbol']
        uptrand_date = uptrand_entry['date']
        
        print(f"🔍 Checking {symbol} (uptrand date: {uptrand_date.date()})...")
        
        if symbol not in mongodb_df['symbol'].values:
            print(f"  ⚠️ {symbol} not found in mongodb.csv")
            continue
        
        is_valid, signal_details = validate_uptrand_entry(mongodb_df, uptrand_entry)
        
        if is_valid:
            print(f"  ✅ Valid buy signal for {symbol}")
            valid_signals.append(signal_details)
            processed_symbols.append(symbol)  # ✅ প্রসেস করা symbol লিস্টে যোগ
        else:
            print(f"  ❌ Not valid for {symbol}")
    
    # ✅ ./csv/uptrand.csv থেকে প্রসেস করা symbol গুলো ডিলিট
    if processed_symbols:
        print(f"\n🗑️ Removing {len(processed_symbols)} processed symbols from {uptrand_csv}...")
        
        # যে symbol গুলো প্রসেস করা হয়েছে সেগুলো বাদ দিয়ে বাকিগুলো রাখা
        remaining_uptrand_df = uptrand_df[~uptrand_df['symbol'].isin(processed_symbols)]
        
        if remaining_uptrand_df.empty:
            # সব symbol ডিলিট হয়ে গেলে খালি ফাইল সেভ
            pd.DataFrame(columns=uptrand_df.columns).to_csv(uptrand_csv, index=False)
            print(f"✅ All symbols processed. {uptrand_csv} is now empty.")
        else:
            # বাকি symbol গুলো সেভ
            remaining_uptrand_df.to_csv(uptrand_csv, index=False)
            print(f"✅ {len(remaining_uptrand_df)} symbols remaining in {uptrand_csv}")
            print(f"   Remaining symbols: {remaining_uptrand_df['symbol'].tolist()}")
    
    # ফলাফল সেভ
    if valid_signals:
        result_df = pd.DataFrame(valid_signals)
        
        result_df['swing_highs_count'] = result_df['swing_highs'].apply(len)
        result_df['swing_highs_details'] = result_df['swing_highs'].apply(
            lambda x: ' | '.join([f"{sh['date'].date()}: {sh['high']}" for sh in x])
        )
        
        output_df = result_df[['symbol', 'date', 'close', 'high', 'prev_high', 'uptrand_date', 'swing_highs_count', 'swing_highs_details']]
        
        output_df.to_csv(output_file, index=False)
        print(f"\n✅ Saved {len(output_df)} valid signals to {output_file}")
        
        print("\n" + "=" * 60)
        print("VALID BUY SIGNALS:")
        print("=" * 60)
        for _, row in output_df.iterrows():
            print(f"📊 {row['symbol']}")
            print(f"   Date: {row['date'].date()}")
            print(f"   Close: {row['close']}")
            print(f"   High: {row['high']}")
            print(f"   Previous High: {row['prev_high']}")
            print(f"   Swing Highs Found: {row['swing_highs_count']}")
            print(f"   Details: {row['swing_highs_details']}")
            print("-" * 40)
    else:
        pd.DataFrame(columns=['symbol', 'date', 'close', 'high', 'prev_high', 'uptrand_date', 'swing_highs_count', 'swing_highs_details']).to_csv(output_file, index=False)
        print(f"\n❌ No valid buy signals found. Empty file saved to {output_file}")
    
    print("\n🎯 Uptrand buy signal detection completed!")


def main():
    print("=" * 60)
    print("UPTRAND BUY SIGNAL DETECTION")
    print("=" * 60)
    create_uptrand_buy_signals()


if __name__ == "__main__":
    main()
