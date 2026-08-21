import pandas as pd
import os
from datetime import datetime

def extract_rsi_below_30_final():
    """
    শুধু RSI <= 30 এর সিম্বলগুলো rsi.csv তে সেভ করবে
    """
    
    # File paths
    input_file = './csv/mongodb.csv'
    output_dir = './output/ai_signal'
    output_file = os.path.join(output_dir, 'rsi.csv')
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ Output directory: {output_dir}")
    except Exception as e:
        print(f"❌ Error creating directory: {e}")
        return
    
    try:
        # Read CSV
        print(f"\n📖 Reading file: {input_file}")
        df = pd.read_csv(input_file)
        print(f"📊 Total rows: {len(df)}")
        print(f"📊 Columns: {', '.join(df.columns)}")
        
        # Check required columns
        required_cols = ['date', 'symbol', 'rsi', 'high']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        
        # Get latest date for each symbol
        print(f"\n🔄 Finding latest date for each symbol...")
        latest_data = df.sort_values('date', ascending=False).groupby('symbol').first().reset_index()
        print(f"✅ Total symbols: {len(latest_data)}")
        
        # Filter RSI <= 30
        filtered_df = latest_data[latest_data['rsi'] <= 30].copy()
        print(f"✅ Symbols with RSI <= 30: {len(filtered_df)}")
        
        if len(filtered_df) == 0:
            print("❌ No symbols found with RSI <= 30")
            return
        
        # Prepare result
        result_df = filtered_df[['symbol', 'date', 'high', 'rsi']].copy()
        result_df = result_df.sort_values('rsi', ascending=True)
        result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')
        result_df.reset_index(drop=True, inplace=True)
        result_df.insert(0, 'sl', range(1, len(result_df) + 1))
        result_df = result_df[['sl', 'symbol', 'date', 'high', 'rsi']]
        
        # ============ FORCE SAVE ============
        print(f"\n💾 Saving to: {output_file}")
        
        # Try different methods to save
        try:
            # Method 1: Direct save
            result_df.to_csv(output_file, index=False)
            print(f"✅ File saved successfully!")
            
            # Verify file exists
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"📁 File size: {file_size} bytes")
                print(f"📁 Absolute path: {os.path.abspath(output_file)}")
            else:
                print(f"❌ File not found after save!")
                
        except Exception as e:
            print(f"❌ Save error: {e}")
            
            # Method 2: Save with full path
            try:
                full_path = os.path.abspath(output_file)
                result_df.to_csv(full_path, index=False)
                print(f"✅ Saved with full path: {full_path}")
            except Exception as e2:
                print(f"❌ Second save attempt failed: {e2}")
        
        # ============ SHOW RESULTS ============
        print("\n" + "="*60)
        print("📊 RSI <= 30 SYMBOLS")
        print("="*60)
        print(result_df.to_string(index=False))
        
        # ============ VERIFY ============
        print("\n" + "="*60)
        print("🔍 VERIFYING OUTPUT")
        print("="*60)
        
        # Check if file can be read
        try:
            verify_df = pd.read_csv(output_file)
            print(f"✅ File read successful!")
            print(f"📊 Rows: {len(verify_df)}")
            print(f"📊 Columns: {', '.join(verify_df.columns)}")
            print(f"\n📄 First 5 rows:")
            print(verify_df.head().to_string(index=False))
        except Exception as e:
            print(f"❌ Cannot read file: {e}")
        
        # ============ SHOW FILE LOCATION ============
        print("\n" + "="*60)
        print("📍 FILE LOCATION")
        print("="*60)
        print(f"File: {output_file}")
        print(f"Absolute: {os.path.abspath(output_file)}")
        print(f"Directory: {os.path.dirname(os.path.abspath(output_file))}")
        
        # Show directory contents
        print(f"\n📁 Contents of {output_dir}:")
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  - {file} ({size} bytes)")
                else:
                    print(f"  - {file}/")
        
        print("="*60)
        
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("="*60)
    print("🚀 RSI EXTRACTION SCRIPT")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    extract_rsi_below_30_final()
    print("\n✅ Script completed!")