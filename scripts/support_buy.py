import pandas as pd
import os
from datetime import datetime

def compare_and_filter_support():
    """
    Compare support levels and update files based on conditions:
    1. Keep symbols in ./output/ai_signal/support_buy.csv where latest low from mongodb.csv > low from support.csv
    2. Delete all data from ./csv/support.csv
    """
    
    # Define file paths
    mongodb_file = './csv/mongodb.csv'
    support_file = './csv/support.csv'
    ai_signal_buy_file = './output/ai_signal/support_buy.csv'
    
    print("=" * 60)
    print("Starting Support Level Comparison for Buy Signals")
    print("=" * 60)
    
    # Check if input files exist
    if not os.path.exists(mongodb_file):
        print(f"❌ Error: Input file {mongodb_file} not found!")
        return False
    
    if not os.path.exists(support_file):
        print(f"⚠️ Warning: {support_file} not found. Creating empty file.")
        # Create empty support.csv if it doesn't exist
        os.makedirs(os.path.dirname(support_file), exist_ok=True)
        empty_df = pd.DataFrame(columns=['date', 'support_date', 'symbol', 'close', 'low', 'gap'])
        empty_df.to_csv(support_file, index=False)
    
    # Read mongodb.csv
    print(f"\n📖 Reading {mongodb_file}...")
    mongodb_df = pd.read_csv(mongodb_file)
    mongodb_df['date'] = pd.to_datetime(mongodb_df['date'])
    
    # Get latest low for each symbol from mongodb.csv
    print("🔄 Getting latest low for each symbol from mongodb.csv...")
    latest_data = []
    for symbol in mongodb_df['symbol'].unique():
        symbol_data = mongodb_df[mongodb_df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date', ascending=False)
        latest_row = symbol_data.iloc[0]
        latest_data.append({
            'symbol': symbol,
            'latest_low': latest_row['low'],
            'latest_date': latest_row['date'],
            'latest_close': latest_row['close']
        })
    
    mongodb_latest_df = pd.DataFrame(latest_data)
    print(f"✅ Found {len(mongodb_latest_df)} symbols in mongodb.csv")
    
    # Read support.csv
    print(f"\n📖 Reading {support_file}...")
    if os.path.exists(support_file):
        support_df = pd.read_csv(support_file)
        print(f"✅ Found {len(support_df)} records in support.csv")
    else:
        print(f"⚠️ {support_file} not found. Creating empty DataFrame.")
        support_df = pd.DataFrame(columns=['date', 'support_date', 'symbol', 'close', 'low', 'gap'])
    
    # Read existing ai_signal_buy file
    print(f"\n📖 Reading {ai_signal_buy_file}...")
    if os.path.exists(ai_signal_buy_file):
        ai_signal_buy_df = pd.read_csv(ai_signal_buy_file)
        print(f"✅ Found {len(ai_signal_buy_df)} records in support_buy.csv")
    else:
        print(f"⚠️ {ai_signal_buy_file} not found. Creating new file.")
        ai_signal_buy_df = pd.DataFrame(columns=['date', 'support_date', 'symbol', 'close', 'low', 'gap'])
    
    # Compare and filter - Keep symbols where latest low > support low
    print("\n🔍 Comparing support levels (Condition: Latest Low > Support Low)...")
    
    buy_signals = []
    
    if len(support_df) > 0 and len(mongodb_latest_df) > 0:
        # Merge support data with mongodb latest data
        comparison_df = support_df.merge(
            mongodb_latest_df[['symbol', 'latest_low']], 
            on='symbol', 
            how='inner'
        )
        
        # Filter where latest low > support low (BUY SIGNAL condition)
        buy_condition = comparison_df['latest_low'] > comparison_df['low']
        buy_signals_df = comparison_df[buy_condition]
        
        if len(buy_signals_df) > 0:
            print(f"\n📊 BUY SIGNALS FOUND for {len(buy_signals_df)} symbols:")
            for _, row in buy_signals_df.iterrows():
                print(f"  ✅ {row['symbol']}: Support Low = {row['low']} < Latest Low = {row['latest_low']} (BUY SIGNAL)")
                buy_signals.append(row['symbol'])
            
            # Create buy signals dataframe with original support data
            buy_signals_data = support_df[support_df['symbol'].isin(buy_signals)]
            
            # Save to support_buy.csv
            os.makedirs(os.path.dirname(ai_signal_buy_file), exist_ok=True)
            buy_signals_data.to_csv(ai_signal_buy_file, index=False)
            
            print(f"\n📝 Updated {ai_signal_buy_file}:")
            print(f"  • Total buy signals: {len(buy_signals_data)}")
            print(f"  • Symbols with buy signals: {', '.join(buy_signals)}")
            
        else:
            print("\n⚠️ No buy signals found. No symbols meet the condition.")
            # Create empty file with headers
            empty_df = pd.DataFrame(columns=['date', 'support_date', 'symbol', 'close', 'low', 'gap'])
            os.makedirs(os.path.dirname(ai_signal_buy_file), exist_ok=True)
            empty_df.to_csv(ai_signal_buy_file, index=False)
            print(f"📁 Created empty {ai_signal_buy_file}")
    
    else:
        print("⚠️ Not enough data for comparison:")
        print(f"  • Support.csv records: {len(support_df)}")
        print(f"  • MongoDB symbols: {len(mongodb_latest_df)}")
        
        if len(support_df) > 0:
            print("\n📋 Support.csv contents:")
            print(support_df[['symbol', 'low', 'date']].head())
        
        # Create empty buy signals file
        empty_df = pd.DataFrame(columns=['date', 'support_date', 'symbol', 'close', 'low', 'gap'])
        os.makedirs(os.path.dirname(ai_signal_buy_file), exist_ok=True)
        empty_df.to_csv(ai_signal_buy_file, index=False)
        print(f"\n📁 Created empty {ai_signal_buy_file}")
    
    # Clear support.csv data (keep headers)
    print(f"\n🗑️ Clearing all data from {support_file}...")
    empty_df = pd.DataFrame(columns=['date', 'support_date', 'symbol', 'close', 'low', 'gap'])
    empty_df.to_csv(support_file, index=False)
    print(f"✅ {support_file} cleared successfully (headers preserved)")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    
    # Read final files for summary
    if os.path.exists(ai_signal_buy_file):
        final_buy_df = pd.read_csv(ai_signal_buy_file)
        print(f"✅ {ai_signal_buy_file}: {len(final_buy_df)} records")
        if len(final_buy_df) > 0:
            print(f"   🟢 BUY SIGNALS for: {', '.join(final_buy_df['symbol'].unique())}")
            print(f"\n   Buy Signal Details:")
            for _, row in final_buy_df.iterrows():
                print(f"     • {row['symbol']}: Date={row['date']}, Support Date={row['support_date']}, Low={row['low']}")
    
    final_support_df = pd.read_csv(support_file)
    print(f"✅ {support_file}: {len(final_support_df)} records (CLEARED)")
    
    return True

def main():
    """
    Main function to run the comparison
    """
    try:
        success = compare_and_filter_support()
        
        if success:
            print("\n✨ Operation completed successfully!")
            print("\n📌 Summary:")
            print("  1. Buy signals saved to: ./output/ai_signal/support_buy.csv")
            print("  2. support.csv has been cleared")
        else:
            print("\n⚠️ Operation completed with warnings!")
            
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()