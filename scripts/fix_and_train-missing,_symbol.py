# scripts/fix_and_train_missing_symbols.py
"""
লোকাল ./csv/ ফাইল থেকে untrained symbols বের করে,
ডেটা ভ্যালিডিটি চেক করে, সমস্যা ফিক্স করে ট্রেনিং করবে
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# =========================================================
# CONFIGURATION
# =========================================================
LOCAL_MONGODB = "./csv/mongodb.csv"
LOCAL_TRAINED = "./csv/trained_symbols.json"
LOCAL_BATCH_TRACKING = "./csv/batch_tracking.json"
BATCH_SIZE = 40

# =========================================================
# STEP 1: Trained Symbols লোড
# =========================================================

def load_trained_symbols():
    """লোকাল trained_symbols.json থেকে ট্রেইনড symbols লোড"""
    if os.path.exists(LOCAL_TRAINED):
        with open(LOCAL_TRAINED, 'r') as f:
            data = json.load(f)
            return set(data.get('symbols', []))
    return set()

# =========================================================
# STEP 2: All Symbols লোড
# =========================================================

def load_all_symbols():
    """লোকাল mongodb.csv থেকে সব symbols লোড"""
    df = pd.read_csv(LOCAL_MONGODB)
    return sorted(df['symbol'].unique().tolist())

# =========================================================
# STEP 3: ব্যাচ ট্র্যাকিং থেকে ব্যাচ-wise symbols
# =========================================================

def load_batch_tracking():
    """batch_tracking.json থেকে ব্যাচ-wise symbols লোড"""
    if os.path.exists(LOCAL_BATCH_TRACKING):
        with open(LOCAL_BATCH_TRACKING, 'r') as f:
            return json.load(f)
    return {'batch_symbols': {}, 'completed_batches': []}

# =========================================================
# STEP 4: Untrained Symbols চিহ্নিত করা (ব্যাচ অনুযায়ী)
# =========================================================

def find_untrained_by_batch():
    """প্রতিটি ব্যাচের কোন symbols train হয়নি তা বের করা"""
    
    all_symbols = load_all_symbols()
    trained_symbols = load_trained_symbols()
    batch_data = load_batch_tracking()
    
    print("="*70)
    print("🔍 FINDING UNTRAINED SYMBOLS BY BATCH")
    print("="*70)
    print(f"📊 Total symbols in mongodb.csv: {len(all_symbols)}")
    print(f"✅ Trained symbols: {len(trained_symbols)}")
    print(f"❌ Total untrained: {len(all_symbols) - len(trained_symbols)}")
    print()
    
    # ব্যাচ-wise untrained বের করা
    batch_untrained = {}
    batch_symbols_dict = batch_data.get('batch_symbols', {})
    
    if batch_symbols_dict:
        for batch_num, symbols in batch_symbols_dict.items():
            trained_in_batch = [s for s in symbols if s in trained_symbols]
            untrained_in_batch = [s for s in symbols if s not in trained_symbols]
            
            batch_untrained[batch_num] = {
                'total': len(symbols),
                'trained': len(trained_in_batch),
                'untrained': len(untrained_in_batch),
                'symbols': untrained_in_batch
            }
            
            status = "✅ COMPLETE" if len(untrained_in_batch) == 0 else f"⚠️ {len(untrained_in_batch)} UNTRAINED"
            print(f"📦 Batch {batch_num}: {len(symbols)} symbols → {status}")
    
    # ব্যাচ ট্র্যাকিং না থাকলে সব untrained একসাথে
    all_untrained = [s for s in all_symbols if s not in trained_symbols]
    
    return batch_untrained, all_untrained

# =========================================================
# STEP 5: Symbol এর ডেটা ভ্যালিডিটি চেক
# =========================================================

def check_symbol_data_validity(symbol):
    """Check if symbol has valid data for training"""
    df = pd.read_csv(LOCAL_MONGODB)
    symbol_df = df[df['symbol'] == symbol]
    
    issues = []
    
    # Check if data exists
    if len(symbol_df) == 0:
        issues.append("NO_DATA")
        return False, issues
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in symbol_df.columns:
            issues.append(f"MISSING_COLUMN_{col}")
            continue
        
        # Check for NaN/None/Null values
        null_count = symbol_df[col].isna().sum()
        if null_count > 0:
            issues.append(f"NULL_VALUES_IN_{col}:{null_count}")
        
        # Check for zero values where shouldn't be zero
        if col in ['high', 'low', 'close']:
            zero_count = (symbol_df[col] == 0).sum()
            if zero_count > 0:
                issues.append(f"ZERO_VALUES_IN_{col}:{zero_count}")
    
    # Check volume
    if 'volume' in symbol_df.columns:
        zero_volume = (symbol_df['volume'] == 0).sum()
        if zero_volume > len(symbol_df) * 0.9:  # 90% zero volume
            issues.append(f"LOW_VOLUME:{zero_volume}/{len(symbol_df)}")
    
    # Check if enough data points
    if len(symbol_df) < 20:
        issues.append(f"INSUFFICIENT_DATA:{len(symbol_df)}")
    
    return len(issues) == 0, issues

# =========================================================
# STEP 6: ডেটা ফিক্স করা
# =========================================================

def fix_symbol_data(symbol):
    """Fix data issues for a symbol"""
    df = pd.read_csv(LOCAL_MONGODB)
    symbol_mask = df['symbol'] == symbol
    
    fixed = False
    
    # Fix NaN values
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            # Forward fill then backward fill
            df.loc[symbol_mask, col] = df.loc[symbol_mask, col].fillna(method='ffill').fillna(method='bfill')
            # Still NaN? Fill with 0
            df.loc[symbol_mask, col] = df.loc[symbol_mask, col].fillna(0)
            fixed = True
    
    # Fix volume
    if 'volume' in df.columns:
        df.loc[symbol_mask, 'volume'] = df.loc[symbol_mask, 'volume'].fillna(0)
        fixed = True
    
    # Fix value column
    if 'value' in df.columns:
        df.loc[symbol_mask, 'value'] = df.loc[symbol_mask, 'value'].fillna(0)
        fixed = True
    
    # Fix trades column
    if 'trades' in df.columns:
        df.loc[symbol_mask, 'trades'] = df.loc[symbol_mask, 'trades'].fillna(0)
        fixed = True
    
    if fixed:
        df.to_csv(LOCAL_MONGODB, index=False)
        print(f"   🔧 Fixed data for {symbol}")
    
    return fixed

# =========================================================
# STEP 7: সব Untrained Symbols চেক ও ফিক্স
# =========================================================

def analyze_and_fix_untrained_symbols(untrained_symbols):
    """প্রতিটি untrained symbol চেক করে সমস্যা চিহ্নিত ও ফিক্স করা"""
    
    print("\n" + "="*70)
    print("🔬 ANALYZING UNTRAINED SYMBOLS")
    print("="*70)
    
    analysis = {
        'valid': [],
        'fixed': [],
        'unfixable': [],
        'issues_summary': defaultdict(int)
    }
    
    for i, symbol in enumerate(untrained_symbols, 1):
        print(f"\n[{i}/{len(untrained_symbols)}] Checking {symbol}...")
        
        is_valid, issues = check_symbol_data_validity(symbol)
        
        if is_valid:
            print(f"   ✅ Valid data")
            analysis['valid'].append(symbol)
        else:
            print(f"   ⚠️ Issues found: {', '.join(issues)}")
            
            # Record issues
            for issue in issues:
                analysis['issues_summary'][issue.split(':')[0]] += 1
            
            # Try to fix
            fixed = fix_symbol_data(symbol)
            if fixed:
                # Re-check after fix
                is_valid_now, remaining_issues = check_symbol_data_validity(symbol)
                if is_valid_now:
                    print(f"   ✅ Fixed successfully")
                    analysis['fixed'].append(symbol)
                else:
                    print(f"   ❌ Still has issues: {remaining_issues}")
                    analysis['unfixable'].append({'symbol': symbol, 'issues': remaining_issues})
            else:
                analysis['unfixable'].append({'symbol': symbol, 'issues': issues})
    
    return analysis

# =========================================================
# STEP 8: ট্রেনিং ডেটা জেনারেট
# =========================================================

def generate_training_data(symbols, batch_name="untrained"):
    """Symbols এর জন্য ট্রেনিং ডেটা জেনারেট"""
    if not symbols:
        return True
    
    print(f"\n📝 Generating training data for {len(symbols)} symbols...")
    
    import subprocess
    
    # ব্যাচ করে জেনারেট
    batches = [symbols[i:i+BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
    
    for i, batch in enumerate(batches, 1):
        symbols_str = ",".join(batch)
        print(f"   Batch {i}: {len(batch)} symbols")
        
        result = subprocess.run(
            ["python", "scripts/generate_pattern_training_data_complete.py",
             "--symbols", symbols_str],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"   ❌ Failed: {result.stderr[:300]}")
            return False
    
    print("   ✅ Training data generated")
    return True

# =========================================================
# STEP 9: ট্রেনিং রান
# =========================================================

def train_symbols(symbols, mode="incremental"):
    """LLM Trainer দিয়ে symbols ট্রেন করা"""
    if not symbols:
        return True
    
    print(f"\n🚀 Training {len(symbols)} symbols (mode: {mode})...")
    
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from scripts.llm_train import AutoLLMTrainer
    
    trainer = AutoLLMTrainer()
    trainer.load_model_with_lora()
    
    # ব্যাচ করে ট্রেন
    batches = [symbols[i:i+BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
    
    for i, batch in enumerate(batches, 1):
        print(f"\n   📦 Training Batch {i}/{len(batches)}: {len(batch)} symbols")
        
        success = trainer.train(mode=mode, symbols_batch=batch)
        
        if success:
            trainer.trained_symbols.extend(batch)
            trainer.save_trained_symbols()
            print(f"   ✅ Batch {i} complete!")
        else:
            print(f"   ❌ Batch {i} failed!")
            return False
    
    return True

# =========================================================
# STEP 10: রিপোর্ট জেনারেট
# =========================================================

def generate_report(batch_untrained, analysis, final_untrained):
    """বিস্তারিত রিপোর্ট জেনারেট"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'batch_summary': {},
        'analysis': {
            'valid': len(analysis['valid']),
            'fixed': len(analysis['fixed']),
            'unfixable': len(analysis['unfixable']),
            'issues_summary': dict(analysis['issues_summary'])
        },
        'unfixable_details': analysis['unfixable'],
        'final_untrained': final_untrained
    }
    
    for batch_num, data in batch_untrained.items():
        report['batch_summary'][batch_num] = {
            'total': data['total'],
            'trained': data['trained'],
            'untrained': data['untrained']
        }
    
    # Save report
    report_path = f"./csv/untrained_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved: {report_path}")
    
    return report

# =========================================================
# MAIN
# =========================================================

def main():
    print("="*70)
    print("🔧 UNTRAINED SYMBOLS DETECTOR & FIXER (LOCAL ONLY)")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Data: {LOCAL_MONGODB}")
    print("="*70)
    
    # Step 1: ব্যাচ-wise untrained symbols খুঁজুন
    batch_untrained, all_untrained = find_untrained_by_batch()
    
    if not all_untrained:
        print("\n✅ সব symbol train করা আছে!")
        return
    
    # Step 2: Untrained symbols বিশ্লেষণ ও ফিক্স
    analysis = analyze_and_fix_untrained_symbols(all_untrained)
    
    # Step 3: Trainable symbols প্রস্তুত
    trainable_symbols = analysis['valid'] + analysis['fixed']
    
    print("\n" + "="*70)
    print("📊 ANALYSIS SUMMARY")
    print("="*70)
    print(f"   ✅ Valid symbols: {len(analysis['valid'])}")
    print(f"   🔧 Fixed symbols: {len(analysis['fixed'])}")
    print(f"   ❌ Unfixable symbols: {len(analysis['unfixable'])}")
    print(f"   🎯 Trainable total: {len(trainable_symbols)}")
    
    if analysis['issues_summary']:
        print(f"\n   Issues found:")
        for issue, count in analysis['issues_summary'].items():
            print(f"      - {issue}: {count}")
    
    # Step 4: Unfixable symbols দেখান
    if analysis['unfixable']:
        print(f"\n   ❌ Unfixable Symbols:")
        for item in analysis['unfixable']:
            print(f"      - {item['symbol']}: {', '.join(item['issues'])}")
    
    # Step 5: Trainable symbols ট্রেন করুন
    if trainable_symbols:
        print("\n" + "="*70)
        print("🚀 STARTING TRAINING FOR FIXED SYMBOLS")
        print("="*70)
        
        # ডেটা জেনারেট
        if generate_training_data(trainable_symbols):
            # ট্রেনিং মোড নির্ধারণ
            trained_count = len(load_trained_symbols())
            mode = "first_train" if trained_count == 0 else "incremental"
            
            # ট্রেনিং
            train_symbols(trainable_symbols, mode)
    
    # Step 6: ফাইনাল রিপোর্ট
    trained_after = load_trained_symbols()
    still_untrained = [s for s in all_untrained if s not in trained_after]
    
    generate_report(batch_untrained, analysis, still_untrained)
    
    print("\n" + "="*70)
    print("✅ PROCESS COMPLETED")
    print("="*70)
    print(f"   Initially untrained: {len(all_untrained)}")
    print(f"   Now trained: {len(all_untrained) - len(still_untrained)}")
    print(f"   Still untrained: {len(still_untrained)}")
    print("="*70)

if __name__ == "__main__":
    main()