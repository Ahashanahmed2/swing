# scripts/hf_data.py
import pandas as pd
import requests
from io import StringIO
import sys

def fetch_and_analyze_hf_data():
    """
    Fetch and analyze data from Hugging Face
    Handles BOM and encoding issues automatically
    """
    
    url = "https://huggingface.co/datasets/ahashanahmed/csv/resolve/main/mongodb.csv"
    
    print("="*70)
    print("HUGGING FACE DATA ANALYZER")
    print("="*70)
    print(f"📡 Fetching data from: {url}")
    
    try:
        # Download data
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Download failed: HTTP {response.status_code}")
            return None
        
        print(f"✅ Download successful ({len(response.text)} bytes)")
        
        # Read CSV with BOM handling
        # Use utf-8-sig to automatically remove BOM (ï»¿)
        df = pd.read_csv(StringIO(response.text), encoding='utf-8-sig')
        
        # Clean column names (remove any hidden characters)
        original_columns = df.columns.tolist()
        df.columns = df.columns.str.strip().str.replace('ï»¿', '').str.replace('\ufeff', '')
        
        print(f"\n📋 Columns found: {df.columns.tolist()}")
        
        # Check and fix symbol column
        if 'symbol' not in df.columns:
            print("⚠️ 'symbol' column not found. Searching for alternative...")
            
            # Try to find symbol column (case insensitive)
            for col in df.columns:
                if 'sym' in col.lower() or 'ticker' in col.lower() or 'stock' in col.lower():
                    df.rename(columns={col: 'symbol'}, inplace=True)
                    print(f"✅ Renamed '{col}' → 'symbol'")
                    break
            
            # If still not found, use first column as symbol
            if 'symbol' not in df.columns:
                print(f"⚠️ Using first column '{df.columns[0]}' as symbol")
                df.rename(columns={df.columns[0]: 'symbol'}, inplace=True)
        
        # Convert date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            print(f"📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        print("\n" + "="*70)
        print("DATA SUMMARY")
        print("="*70)
        print(f"📊 Total rows: {len(df):,}")
        print(f"📈 Unique symbols: {df['symbol'].nunique():,}")
        print(f"📋 Columns: {len(df.columns)}")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️ Missing values found:")
            for col, count in missing[missing > 0].items():
                print(f"   {col}: {count} ({count/len(df)*100:.1f}%)")
        
        print("\n" + "="*70)
        print("SYMBOL DISTRIBUTION")
        print("="*70)
        
        # Symbol counts
        symbol_counts = df['symbol'].value_counts().sort_values(ascending=False)
        
        # Categorize symbols
        high_data = symbol_counts[symbol_counts >= 100]
        medium_data = symbol_counts[(symbol_counts >= 50) & (symbol_counts < 100)]
        low_data = symbol_counts[symbol_counts < 50]
        
        print(f"\n✅ High data (≥100 rows): {len(high_data)} symbols")
        print(f"⚠️ Medium data (50-99 rows): {len(medium_data)} symbols")
        print(f"❌ Low data (<50 rows): {len(low_data)} symbols")
        
        # Show top 20 symbols
        print("\n📊 Top 20 symbols by row count:")
        print("-"*50)
        for i, (symbol, count) in enumerate(symbol_counts.head(20).items(), 1):
            status = "✅" if count >= 100 else "⚠️" if count >= 50 else "❌"
            print(f"{i:2}. {status} {symbol:20} {count:5} rows")
        
        # Show bottom 10 symbols (with data)
        print("\n📊 Bottom 10 symbols (with at least 1 row):")
        print("-"*50)
        for i, (symbol, count) in enumerate(symbol_counts.tail(10).items(), 1):
            status = "✅" if count >= 100 else "⚠️" if count >= 50 else "❌"
            print(f"{i:2}. {status} {symbol:20} {count:5} rows")
        
        # Data quality metrics
        print("\n" + "="*70)
        print("DATA QUALITY METRICS")
        print("="*70)
        
        # Check date coverage
        if 'date' in df.columns:
            date_range = (df['date'].max() - df['date'].min()).days
            print(f"📅 Date coverage: {date_range} days")
            
            # Group by symbol to see data distribution
            symbol_date_counts = df.groupby('symbol')['date'].nunique()
            avg_dates = symbol_date_counts.mean()
            print(f"📊 Average unique dates per symbol: {avg_dates:.1f}")
            
            symbols_with_good_dates = (symbol_date_counts >= 100).sum()
            print(f"✅ Symbols with 100+ dates: {symbols_with_good_dates}")
        
        # Price and volume stats
        print("\n📈 Price Statistics:")
        print(f"   Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
        print(f"   Average close: {df['close'].mean():.2f}")
        
        if 'volume' in df.columns:
            print(f"📊 Volume Statistics:")
            print(f"   Average daily volume: {df['volume'].mean():,.0f}")
            print(f"   Max daily volume: {df['volume'].max():,.0f}")
        
        # Recommendations
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        print("\n📋 For XGBoost Training:")
        if len(high_data) >= 10:
            print(f"✅ Good! {len(high_data)} symbols have sufficient data (≥100 rows)")
            print(f"   Recommended MIN_SAMPLES_PER_SYMBOL = 100")
        elif len(medium_data) >= 5:
            print(f"⚠️ Acceptable: {len(medium_data)} symbols have 50-99 rows")
            print(f"   Recommended MIN_SAMPLES_PER_SYMBOL = 50")
        else:
            print(f"❌ Limited data: Only {len(high_data)} symbols have ≥100 rows")
            print(f"   Recommended MIN_SAMPLES_PER_SYMBOL = 40")
        
        print(f"\n🎯 Target threshold recommendation:")
        avg_return = df['close'].pct_change().mean()
        print(f"   Average daily return: {avg_return:.2%}")
        print(f"   Recommended target: >2% or >3% based on volatility")
        
        # Save analysis to file
        analysis_file = './csv/hf_data_analysis.csv'
        
        # Create summary dataframe
        summary_df = pd.DataFrame({
            'symbol': symbol_counts.index,
            'row_count': symbol_counts.values,
            'category': ['high' if c >= 100 else 'medium' if c >= 50 else 'low' for c in symbol_counts.values]
        })
        
        summary_df.to_csv(analysis_file, index=False)
        print(f"\n💾 Analysis saved to: {analysis_file}")
        
        # Save filtered data for training
        training_data = df[df['symbol'].isin(high_data.index.tolist() + medium_data.index.tolist())]
        training_file = './csv/training_ready.csv'
        training_data.to_csv(training_file, index=False, encoding='utf-8-sig')
        print(f"💾 Training-ready data saved to: {training_file}")
        print(f"   (Symbols with ≥50 rows: {training_data['symbol'].nunique()})")
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        
        return df
        
    except requests.exceptions.Timeout:
        print("❌ Request timeout - Hugging Face might be slow")
        return None
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - Check your internet")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run the analysis"""
    df = fetch_and_analyze_hf_data()
    
    if df is not None:
        print("\n✅ Data successfully processed!")
        print(f"📊 Ready for XGBoost training with {df['symbol'].nunique()} symbols")
    else:
        print("\n❌ Failed to process data")
        sys.exit(1)

if __name__ == "__main__":
    main()
