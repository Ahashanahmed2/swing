"""
scripts/add_ai_values_to_support_resistance.py
support_resistant.csv-তে থাকা সিম্বলগুলোর জন্য LLM, XGBoost, PPO রেজাল্ট যোগ করে
"""

import pandas as pd
import torch
import os
import re
import json
import joblib
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================================================
# CONFIGURATION
# =========================================================
LLM_MODEL_DIR = "./csv/llm_model"
SUPPORT_RESISTANCE_PATH = "./output/ai_signal/support_resistant.csv"
MONGO_PATH = "./csv/mongodb.csv"
XGBOOST_DIR = "./csv/xgboost"
PPO_MODELS_DIR = "./csv/ppo_models/per_symbol"
MODEL_METADATA_PATH = "./csv/model_metadata.csv"
PREDICTION_LOG_PATH = "./csv/prediction_log.csv"
XGB_CONFIDENCE_PATH = "./csv/xgb_confidence.csv"

OUTPUT_PATH = "./output/ai_signal/support_resistant_with_ai.csv"
os.makedirs("./output/ai_signal", exist_ok=True)

# =========================================================
# ১. support_resistant.csv লোড করুন
# =========================================================
print("📂 Loading support_resistant.csv...")
sr_df = pd.read_csv(SUPPORT_RESISTANCE_PATH)
print(f"   ✅ Loaded {len(sr_df)} signals for {sr_df['symbol'].nunique()} symbols")

# ইউনিক সিম্বলের তালিকা
target_symbols = sr_df['symbol'].unique().tolist()
print(f"   🎯 Target symbols: {target_symbols}")

# =========================================================
# ২. mongodb.csv থেকে সর্বশেষ ডেটা লোড করুন
# =========================================================
print("\n📂 Loading market data...")
mongo_df = pd.read_csv(MONGO_PATH)
mongo_df['date'] = pd.to_datetime(mongo_df['date'], format='mixed', errors='coerce')
mongo_df = mongo_df.sort_values(['symbol', 'date'])

# প্রতি সিম্বলের সর্বশেষ রো
latest_data = mongo_df.groupby('symbol').tail(1).set_index('symbol')

# =========================================================
# ৩. XGBoost ডেটা লোড করুন
# =========================================================
print("\n📂 Loading XGBoost data...")

# মডেল মেটাডেটা
meta_df = pd.read_csv(MODEL_METADATA_PATH)
meta_df = meta_df[meta_df['auc'] >= 0.55]  # ভালো মডেল ফিল্টার

# প্রেডিকশন লগ
pred_df = pd.read_csv(PREDICTION_LOG_PATH)
pred_df['date'] = pd.to_datetime(pred_df['date'], format='mixed', errors='coerce')
pred_df = pred_df.sort_values(['symbol', 'date'])
pred_df = pred_df.drop_duplicates(subset=['symbol', 'date'], keep='last')

# XGB কনফিডেন্স
conf_df = pd.read_csv(XGB_CONFIDENCE_PATH)
conf_df['date'] = pd.to_datetime(conf_df['date'], format='mixed', errors='coerce')

# XGB ডেটা মার্জ
xgb_df = pd.merge(pred_df, conf_df, on=['symbol', 'date'], how='left')

# prob_up কলাম তৈরি (না থাকলে)
if 'prob_up' not in xgb_df.columns:
    if 'prediction' in xgb_df.columns and 'confidence_score' in xgb_df.columns:
        xgb_df['prob_up'] = xgb_df.apply(
            lambda row: row['confidence_score'] / 100 if row['prediction'] == 1 
            else (100 - row['confidence_score']) / 100,
            axis=1
        )
    else:
        xgb_df['prob_up'] = 0.5

# ভালো সিম্বল ফিল্টার
good_symbols = meta_df['symbol'].unique()
xgb_df = xgb_df[xgb_df['symbol'].isin(good_symbols)]

# প্রতি সিম্বলের সর্বশেষ XGB ডেটা
xgb_latest = xgb_df.sort_values(['symbol', 'date']).groupby('symbol').tail(1).set_index('symbol')

print(f"   ✅ XGBoost data loaded for {len(good_symbols)} good models")

# =========================================================
# ৪. PPO মডেল স্ট্যাটাস চেক করুন
# =========================================================
print("\n📂 Checking PPO models...")
ppo_status = {}

for symbol in target_symbols:
    ppo_path = os.path.join(PPO_MODELS_DIR, f"ppo_{symbol}.zip")
    if os.path.exists(ppo_path):
        ppo_status[symbol] = {'has_ppo': True, 'ppo_path': ppo_path}
    else:
        ppo_status[symbol] = {'has_ppo': False, 'ppo_path': None}

ppo_available = sum(1 for v in ppo_status.values() if v['has_ppo'])
print(f"   ✅ PPO models available: {ppo_available}/{len(target_symbols)}")

# =========================================================
# ৫. LLM মডেল লোড করুন
# =========================================================
print("\n🤖 Loading LLM Model...")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_DIR,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"   ✅ LLM loaded on {device}")

# =========================================================
# ৬. LLM ইনফারেন্স ফাংশন
# =========================================================
def get_llm_signal(symbol, row):
    """LLM থেকে সিগন্যাল ও ভ্যালু জেনারেট করে"""
    
    prompt = f"""================================================================================
Pattern: TECHNICAL ANALYSIS
Symbol: {symbol}
Sector: {row.get('sector', 'Unknown')}
Date: {row['date']}

📊 PRICE DATA:
Open: {row['open']:.2f} | High: {row['high']:.2f} | Low: {row['low']:.2f}
Close: {row['close']:.2f} | Volume: {int(row['volume']):,}

📈 TECHNICAL INDICATORS:
RSI: {row.get('rsi', 50):.1f} | MACD: {row.get('macd', 0):.4f}

🎯 PATTERN ANALYSIS:
Based on the above, provide trading signal (BUY/SELL/HOLD), confidence%, entry, stop loss, target.

📝 RECOMMENDATION:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # ভ্যালু এক্সট্রাক্ট
    result = {
        'llm_signal': 'UNKNOWN',
        'llm_confidence': 0.0,
        'llm_entry': row['close'],
        'llm_stop_loss': row['close'] * 0.98,
        'llm_target': row['close'] * 1.05,
        'llm_strength': 'UNKNOWN',
        'llm_response': response[:200]
    }
    
    # সিগন্যাল
    if re.search(r'✅ BUY|Signal:?\s*BUY', response, re.IGNORECASE):
        result['llm_signal'] = 'BUY'
    elif re.search(r'❌ SELL|Signal:?\s*SELL', response, re.IGNORECASE):
        result['llm_signal'] = 'SELL'
    else:
        result['llm_signal'] = 'HOLD'
    
    # কনফিডেন্স
    conf_match = re.search(r'(\d+(?:\.\d+)?)%', response)
    if conf_match:
        result['llm_confidence'] = float(conf_match.group(1))
    
    # এন্ট্রি
    entry_match = re.search(r'Entry:?\s*([\d.]+)', response, re.IGNORECASE)
    if entry_match:
        result['llm_entry'] = float(entry_match.group(1))
    
    # স্টপ লস
    sl_match = re.search(r'Stop(?:\s*Loss)?:?\s*([\d.]+)', response, re.IGNORECASE)
    if sl_match:
        result['llm_stop_loss'] = float(sl_match.group(1))
    
    # টার্গেট
    tp_match = re.search(r'Target:?\s*([\d.]+)', response, re.IGNORECASE)
    if tp_match:
        result['llm_target'] = float(tp_match.group(1))
    
    # স্ট্রেন্থ
    if re.search(r'STRONG|HIGH', response, re.IGNORECASE):
        result['llm_strength'] = 'STRONG'
    elif re.search(r'MEDIUM|MODERATE', response, re.IGNORECASE):
        result['llm_strength'] = 'MEDIUM'
    elif re.search(r'WEAK|LOW', response, re.IGNORECASE):
        result['llm_strength'] = 'WEAK'
    
    return result

# =========================================================
# ৭. XGBoost ডেটা ফেচ ফাংশন
# =========================================================
def get_xgb_data(symbol):
    """XGBoost ডেটা রিটার্ন করে"""
    if symbol in xgb_latest.index:
        row = xgb_latest.loc[symbol]
        return {
            'xgb_prob_up': round(row.get('prob_up', 0.5), 3),
            'xgb_confidence': round(row.get('confidence_score', 50), 1),
            'xgb_prediction': row.get('prediction', -1),
            'xgb_auc': meta_df[meta_df['symbol'] == symbol]['auc'].values[0] if symbol in meta_df['symbol'].values else 0
        }
    else:
        return {
            'xgb_prob_up': 0.5,
            'xgb_confidence': 50.0,
            'xgb_prediction': -1,
            'xgb_auc': 0
        }

# =========================================================
# ৮. PPO ডেটা ফেচ ফাংশন
# =========================================================
def get_ppo_data(symbol):
    """PPO ডেটা রিটার্ন করে (যদি থাকে)"""
    if symbol in ppo_status and ppo_status[symbol]['has_ppo']:
        return {
            'ppo_available': True,
            'ppo_signal': 'AVAILABLE',  # PPO মডেল থেকে লোড করা সম্ভব
            'ppo_weight': 1.0
        }
    else:
        return {
            'ppo_available': False,
            'ppo_signal': 'NOT_AVAILABLE',
            'ppo_weight': 0.0
        }

# =========================================================
# ৯. মূল লুপ - প্রতিটি সিম্বলের জন্য ডেটা যোগ করুন
# =========================================================
print("\n🎯 Generating AI values for each symbol...")

llm_results = {}
xgb_results = {}
ppo_results = {}

for symbol in target_symbols:
    print(f"\n   🔍 Processing {symbol}...")
    
    # LLM ইনফারেন্স (শুধু যদি ডেটা থাকে)
    if symbol in latest_data.index:
        row = latest_data.loc[symbol]
        llm_results[symbol] = get_llm_signal(symbol, row)
        print(f"      LLM: {llm_results[symbol]['llm_signal']} ({llm_results[symbol]['llm_confidence']:.1f}%)")
    else:
        llm_results[symbol] = {
            'llm_signal': 'NO_DATA',
            'llm_confidence': 0,
            'llm_entry': 0,
            'llm_stop_loss': 0,
            'llm_target': 0,
            'llm_strength': 'NO_DATA',
            'llm_response': ''
        }
        print(f"      LLM: No market data available")
    
    # XGBoost ডেটা
    xgb_results[symbol] = get_xgb_data(symbol)
    print(f"      XGB: prob={xgb_results[symbol]['xgb_prob_up']:.3f}, conf={xgb_results[symbol]['xgb_confidence']:.1f}%")
    
    # PPO ডেটা
    ppo_results[symbol] = get_ppo_data(symbol)
    print(f"      PPO: {'Available' if ppo_results[symbol]['ppo_available'] else 'Not available'}")

# =========================================================
# ১০. support_resistant.csv-তে নতুন কলাম যোগ করুন
# =========================================================
print("\n📊 Adding AI columns to dataframe...")

# LLM কলাম
sr_df['llm_signal'] = sr_df['symbol'].map(lambda s: llm_results.get(s, {}).get('llm_signal', 'UNKNOWN'))
sr_df['llm_confidence'] = sr_df['symbol'].map(lambda s: llm_results.get(s, {}).get('llm_confidence', 0))
sr_df['llm_entry'] = sr_df['symbol'].map(lambda s: llm_results.get(s, {}).get('llm_entry', 0))
sr_df['llm_stop_loss'] = sr_df['symbol'].map(lambda s: llm_results.get(s, {}).get('llm_stop_loss', 0))
sr_df['llm_target'] = sr_df['symbol'].map(lambda s: llm_results.get(s, {}).get('llm_target', 0))
sr_df['llm_strength'] = sr_df['symbol'].map(lambda s: llm_results.get(s, {}).get('llm_strength', 'UNKNOWN'))

# XGBoost কলাম
sr_df['xgb_prob_up'] = sr_df['symbol'].map(lambda s: xgb_results.get(s, {}).get('xgb_prob_up', 0.5))
sr_df['xgb_confidence'] = sr_df['symbol'].map(lambda s: xgb_results.get(s, {}).get('xgb_confidence', 50))
sr_df['xgb_prediction'] = sr_df['symbol'].map(lambda s: xgb_results.get(s, {}).get('xgb_prediction', -1))
sr_df['xgb_auc'] = sr_df['symbol'].map(lambda s: xgb_results.get(s, {}).get('xgb_auc', 0))

# PPO কলাম
sr_df['ppo_available'] = sr_df['symbol'].map(lambda s: ppo_results.get(s, {}).get('ppo_available', False))

# =========================================================
# ১১. কম্বাইন্ড স্কোর তৈরি করুন (LLM + XGBoost + PPO)
# =========================================================
def calculate_combined_score(row):
    """তিনটি মডেলের কম্বাইন্ড স্কোর"""
    score = 0
    weight_sum = 0
    
    # LLM স্কোর (ওজন: ৪০%)
    if row['llm_signal'] == 'BUY':
        score += row['llm_confidence'] * 0.4
    elif row['llm_signal'] == 'SELL':
        score += (100 - row['llm_confidence']) * 0.4
    else:
        score += 50 * 0.4
    weight_sum += 0.4
    
    # XGBoost স্কোর (ওজন: ৪০%)
    score += row['xgb_prob_up'] * 100 * 0.4
    weight_sum += 0.4
    
    # PPO স্কোর (ওজন: ২০%, যদি থাকে)
    if row['ppo_available']:
        score += 50 * 0.2  # PPO সিগন্যাল না থাকলে নিউট্রাল
        weight_sum += 0.2
    
    return score / weight_sum if weight_sum > 0 else 50

sr_df['combined_score'] = sr_df.apply(calculate_combined_score, axis=1)

# কম্বাইন্ড সিগন্যাল
def get_combined_signal(score):
    if score >= 70:
        return 'STRONG BUY'
    elif score >= 60:
        return 'BUY'
    elif score <= 30:
        return 'STRONG SELL'
    elif score <= 40:
        return 'SELL'
    else:
        return 'NEUTRAL'

sr_df['combined_signal'] = sr_df['combined_score'].apply(get_combined_signal)

# =========================================================
# ১২. সেভ করুন
# =========================================================
sr_df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "="*70)
print("📊 FINAL SUMMARY")
print("="*70)
print(f"   Total signals: {len(sr_df)}")
print(f"   Unique symbols: {sr_df['symbol'].nunique()}")
print("\n📈 Combined Signal Distribution:")
print(sr_df['combined_signal'].value_counts().to_string())
print("\n🤖 LLM Signal Distribution:")
print(sr_df['llm_signal'].value_counts().to_string())
print("\n" + "="*70)
print(f"✅ Output saved to: {OUTPUT_PATH}")
print("="*70)

# =========================================================
# ১৩. ডেমো - প্রথম ৫টি রো দেখান
# =========================================================
print("\n📋 SAMPLE OUTPUT (First 5 rows):")
print("="*70)
display_cols = ['symbol', 'combined_signal', 'combined_score', 'llm_signal', 
                'llm_confidence', 'xgb_prob_up', 'xgb_confidence', 'ppo_available']
print(sr_df[display_cols].head(5).to_string())
print("="*70)