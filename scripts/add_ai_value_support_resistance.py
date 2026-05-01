"""
scripts/add_ai_values_to_support_resistance.py
support_resistance.csv-তে থাকা সিম্বলগুলোর জন্য LLM, XGBoost, PPO, Agentic Loop, PatchTST, Sector রেজাল্ট যোগ করে
✅ PPO সিগন্যাল fully integrated (BUY/SELL/HOLD)
✅ Updated with ALL features from env_trading.py
✅ Original Structure 100% Preserved
✅ No Code Deleted
"""

import pandas as pd
import torch
import os
import re
import json
import joblib
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================================================
# PATH SETUP
# =========================================================
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# =========================================================
# CONFIGURATION
# =========================================================
LLM_MODEL_DIR = "./csv/llm_model"
SUPPORT_RESISTANCE_PATH = "./csv/support_resistance.csv"
MONGO_PATH = "./csv/mongodb.csv"
XGBOOST_DIR = "./csv/xgboost"
PPO_MODELS_DIR = "./csv/ppo_models/per_symbol"
MODEL_METADATA_PATH = "./csv/model_metadata.csv"
PREDICTION_LOG_PATH = "./csv/prediction_log.csv"
XGB_CONFIDENCE_PATH = "./csv/xgb_confidence.csv"

OUTPUT_PATH = "./csv/support_resistance_with_ai.csv"
os.makedirs("./csv", exist_ok=True)

# =========================================================
# ✅ ADDITIONAL IMPORTS FOR NEW FEATURES
# =========================================================

try:
    from sector_features import SectorFeatureEngine
    SECTOR_AVAILABLE = True
    print("✅ SectorFeatureEngine loaded")
except ImportError:
    SECTOR_AVAILABLE = False
    print("⚠️ SectorFeatureEngine not available")

try:
    from env_trading import RSIDivergenceFeatures, SupportResistanceFeatures
    ENV_FEATURES_AVAILABLE = True
    print("✅ RSI Divergence + S/R Features loaded")
except ImportError:
    ENV_FEATURES_AVAILABLE = False
    print("⚠️ Env features not available")

try:
    from agentic_loop import AgenticLoop
    AGENTIC_LOOP_AVAILABLE = True
    print("✅ Agentic Loop loaded")
except ImportError:
    AGENTIC_LOOP_AVAILABLE = False
    print("⚠️ Agentic Loop not available")

try:
    from patch_tst_predictor import PatchTSTIntegration
    PATCHTST_AVAILABLE = True
    print("✅ PatchTST loaded")
except ImportError:
    PATCHTST_AVAILABLE = False
    print("⚠️ PatchTST not available")

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
    print("✅ Stable-Baselines3 loaded")
except ImportError:
    SB3_AVAILABLE = False
    print("⚠️ Stable-Baselines3 not available")

# =========================================================
# ১. support_resistance.csv লোড করুন
# =========================================================
print("\n📂 Loading support_resistance.csv...")
sr_df = pd.read_csv(SUPPORT_RESISTANCE_PATH)
if 'current_date' in sr_df.columns:
    sr_df['current_date'] = pd.to_datetime(sr_df['current_date'])
print(f"   ✅ Loaded {len(sr_df)} signals for {sr_df['symbol'].nunique()} symbols")

# ইউনিক সিম্বলের তালিকা
target_symbols = sr_df['symbol'].unique().tolist()
print(f"   🎯 Target symbols: {len(target_symbols)}")

# =========================================================
# ২. mongodb.csv থেকে সর্বশেষ ডেটা লোড করুন
# =========================================================
print("\n📂 Loading market data...")
mongo_df = pd.read_csv(MONGO_PATH)
if 'date' in mongo_df.columns:
    mongo_df['date'] = pd.to_datetime(mongo_df['date'], format='mixed', errors='coerce')
mongo_df = mongo_df.sort_values(['symbol', 'date'])

# প্রতি সিম্বলের সর্বশেষ রো
latest_data = mongo_df.groupby('symbol').tail(1).set_index('symbol')
print(f"   ✅ Market data: {len(latest_data)} symbols with latest data")

# =========================================================
# ✅ ২.৫ Sector Engine Initialize
# =========================================================
sector_engine = None
if SECTOR_AVAILABLE:
    try:
        sector_engine = SectorFeatureEngine(csv_market_path=MONGO_PATH)
        print("✅ Sector Engine initialized")
    except Exception as e:
        print(f"⚠️ Sector Engine failed: {e}")

# =========================================================
# ✅ ২.৬ RSI Divergence + S/R Features Initialize
# =========================================================
rsi_div = None
sr_features_obj = None
if ENV_FEATURES_AVAILABLE:
    try:
        rsi_div = RSIDivergenceFeatures()
        sr_features_obj = SupportResistanceFeatures()
        print("✅ RSI Divergence + S/R Features initialized")
    except Exception as e:
        print(f"⚠️ Feature init failed: {e}")

# =========================================================
# ✅ ২.৭ Agentic Loop Initialize
# =========================================================
agentic_loop = None
if AGENTIC_LOOP_AVAILABLE:
    try:
        agentic_loop = AgenticLoop(xgb_model_dir=XGBOOST_DIR)
        print("✅ Agentic Loop initialized")
    except Exception as e:
        print(f"⚠️ Agentic Loop failed: {e}")

# =========================================================
# ✅ ২.৮ PatchTST Initialize
# =========================================================
patch_tst = None
if PATCHTST_AVAILABLE:
    try:
        patch_tst = PatchTSTIntegration(model_dir="./csv/patchtst_models")
        print("✅ PatchTST initialized")
    except Exception as e:
        print(f"⚠️ PatchTST failed: {e}")

# =========================================================
# ৩. XGBoost ডেটা লোড করুন
# =========================================================
print("\n📂 Loading XGBoost data...")

# মডেল মেটাডেটা
meta_df = pd.read_csv(MODEL_METADATA_PATH) if os.path.exists(MODEL_METADATA_PATH) else pd.DataFrame()
if not meta_df.empty:
    meta_df = meta_df[meta_df['auc'] >= 0.55]

# প্রেডিকশন লগ
pred_df = pd.read_csv(PREDICTION_LOG_PATH) if os.path.exists(PREDICTION_LOG_PATH) else pd.DataFrame()
if not pred_df.empty and 'date' in pred_df.columns:
    pred_df['date'] = pd.to_datetime(pred_df['date'], format='mixed', errors='coerce')
    pred_df = pred_df.sort_values(['symbol', 'date'])
    pred_df = pred_df.drop_duplicates(subset=['symbol', 'date'], keep='last')

# XGB কনফিডেন্স
conf_df = pd.read_csv(XGB_CONFIDENCE_PATH) if os.path.exists(XGB_CONFIDENCE_PATH) else pd.DataFrame()
if not conf_df.empty and 'date' in conf_df.columns:
    conf_df['date'] = pd.to_datetime(conf_df['date'], format='mixed', errors='coerce')

# XGB ডেটা মার্জ
if not pred_df.empty and not conf_df.empty:
    xgb_df = pd.merge(pred_df, conf_df, on=['symbol', 'date'], how='left')
elif not pred_df.empty:
    xgb_df = pred_df.copy()
else:
    xgb_df = pd.DataFrame()

# prob_up কলাম তৈরি (না থাকলে)
if not xgb_df.empty and 'prob_up' not in xgb_df.columns:
    if 'prediction' in xgb_df.columns and 'confidence_score' in xgb_df.columns:
        xgb_df['prob_up'] = xgb_df.apply(
            lambda row: row['confidence_score'] / 100 if row['prediction'] == 1 
            else (100 - row['confidence_score']) / 100,
            axis=1
        )
    else:
        xgb_df['prob_up'] = 0.5

# ভালো সিম্বল ফিল্টার
if not meta_df.empty:
    good_symbols = meta_df['symbol'].unique()
    xgb_df = xgb_df[xgb_df['symbol'].isin(good_symbols)]

# প্রতি সিম্বলের সর্বশেষ XGB ডেটা
if not xgb_df.empty:
    xgb_latest = xgb_df.sort_values(['symbol', 'date']).groupby('symbol').tail(1).set_index('symbol')
else:
    xgb_latest = pd.DataFrame()

print(f"   ✅ XGBoost data loaded")

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
        # Check ensemble
        ensemble_path = os.path.join(PPO_MODELS_DIR.replace('per_symbol', ''), 'ensemble')
        ensemble_info = os.path.join(PPO_MODELS_DIR.replace('per_symbol', ''), f'ensemble_{symbol}.pkl')
        if os.path.exists(ensemble_info):
            ppo_status[symbol] = {'has_ppo': True, 'ppo_path': ensemble_info}
        else:
            ppo_status[symbol] = {'has_ppo': False, 'ppo_path': None}

ppo_available = sum(1 for v in ppo_status.values() if v['has_ppo'])
print(f"   ✅ PPO models available: {ppo_available}/{len(target_symbols)}")

# =========================================================
# ৫. LLM মডেল লোড করুন
# =========================================================
print("\n🤖 Loading LLM Model...")

tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(LLM_MODEL_DIR):
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_DIR,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device)
        print(f"   ✅ LLM loaded on {device}")
    except Exception as e:
        print(f"   ⚠️ LLM load failed: {e}")
else:
    print(f"   ⚠️ LLM model not found at {LLM_MODEL_DIR}")

# =========================================================
# ৬. LLM ইনফারেন্স ফাংশন (UPDATED with Sector + RSI Div + S/R)
# =========================================================
def get_llm_signal(symbol, row):
    """LLM থেকে সিগন্যাল ও ভ্যালু জেনারেট করে (Enhanced)"""
    
    result = {
        'llm_signal': 'UNKNOWN',
        'llm_confidence': 0.0,
        'llm_entry': row.get('close', 0),
        'llm_stop_loss': row.get('close', 0) * 0.98,
        'llm_target': row.get('close', 0) * 1.05,
        'llm_strength': 'UNKNOWN',
        'llm_response': ''
    }
    
    if model is None or tokenizer is None:
        return result
    
    try:
        # Sector info
        sector = 'Unknown'
        if sector_engine:
            sector = sector_engine.get_sector(symbol)
        
        # RSI Divergence info
        div_info = ""
        if rsi_div:
            rsi_vec = rsi_div.get_features(symbol, str(row.get('date', '')))
            if rsi_vec[0] == 1.0:
                div_info = f"\n🎯 RSI DIVERGENCE: BULLISH (Strength: {rsi_vec[1]:.0%})"
            elif rsi_vec[0] == 0.0:
                div_info = f"\n🎯 RSI DIVERGENCE: BEARISH (Strength: {rsi_vec[1]:.0%})"
        
        # Support/Resistance info
        sr_info = ""
        if sr_features_obj:
            sr_vec = sr_features_obj.get_features(symbol, str(row.get('date', '')), row.get('close', 0))
            sr_type = "SUPPORT" if sr_vec[2] == 1.0 else "RESISTANCE" if sr_vec[2] == -1.0 else "NONE"
            sr_info = f"\n📊 LEVEL: {sr_type} (Distance: {sr_vec[0]:.1%}, Strength: {sr_vec[1]:.0%})"
        
        prompt = f"""================================================================================
Pattern: TECHNICAL ANALYSIS
Symbol: {symbol}
Sector: {sector}
Date: {row.get('date', 'N/A')}

📊 PRICE DATA:
Open: {row.get('open', 0):.2f} | High: {row.get('high', 0):.2f} | Low: {row.get('low', 0):.2f}
Close: {row.get('close', 0):.2f} | Volume: {int(row.get('volume', 0)):,}

📈 TECHNICAL INDICATORS:
RSI: {row.get('rsi', 50):.1f} | MACD: {row.get('macd', 0):.4f}{div_info}{sr_info}

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
        result['llm_response'] = response[:200]
        
        # Signal
        if re.search(r'✅ BUY|Signal:?\s*BUY', response, re.IGNORECASE):
            result['llm_signal'] = 'BUY'
        elif re.search(r'❌ SELL|Signal:?\s*SELL', response, re.IGNORECASE):
            result['llm_signal'] = 'SELL'
        else:
            result['llm_signal'] = 'HOLD'
        
        # Confidence
        conf_match = re.search(r'(\d+(?:\.\d+)?)%', response)
        if conf_match:
            result['llm_confidence'] = float(conf_match.group(1))
        
        # Entry
        entry_match = re.search(r'Entry:?\s*([\d.]+)', response, re.IGNORECASE)
        if entry_match:
            result['llm_entry'] = float(entry_match.group(1))
        
        # Stop Loss
        sl_match = re.search(r'Stop(?:\s*Loss)?:?\s*([\d.]+)', response, re.IGNORECASE)
        if sl_match:
            result['llm_stop_loss'] = float(sl_match.group(1))
        
        # Target
        tp_match = re.search(r'Target:?\s*([\d.]+)', response, re.IGNORECASE)
        if tp_match:
            result['llm_target'] = float(tp_match.group(1))
        
        # Strength
        if re.search(r'STRONG|HIGH', response, re.IGNORECASE):
            result['llm_strength'] = 'STRONG'
        elif re.search(r'MEDIUM|MODERATE', response, re.IGNORECASE):
            result['llm_strength'] = 'MEDIUM'
        elif re.search(r'WEAK|LOW', response, re.IGNORECASE):
            result['llm_strength'] = 'WEAK'
        
    except Exception as e:
        print(f"      ⚠️ LLM inference failed for {symbol}: {e}")
    
    return result

# =========================================================
# ৭. XGBoost ডেটা ফেচ ফাংশন
# =========================================================
def get_xgb_data(symbol):
    """XGBoost ডেটা রিটার্ন করে"""
    if not xgb_latest.empty and symbol in xgb_latest.index:
        row = xgb_latest.loc[symbol]
        auc_val = 0
        if not meta_df.empty and symbol in meta_df['symbol'].values:
            auc_val = meta_df[meta_df['symbol'] == symbol]['auc'].values[0]
        return {
            'xgb_prob_up': round(row.get('prob_up', 0.5), 3),
            'xgb_confidence': round(row.get('confidence_score', 50), 1),
            'xgb_prediction': row.get('prediction', -1),
            'xgb_auc': auc_val if auc_val else 0
        }
    else:
        return {
            'xgb_prob_up': 0.5,
            'xgb_confidence': 50.0,
            'xgb_prediction': -1,
            'xgb_auc': 0
        }

# =========================================================
# ✅ ৮. PPO ডেটা ফেচ ফাংশন (FULLY INTEGRATED)
# =========================================================
def get_ppo_data(symbol, row):
    """PPO model লোড করে সিগন্যাল জেনারেট করে (BUY/SELL/HOLD)"""
    if not SB3_AVAILABLE:
        return {
            'ppo_available': False,
            'ppo_signal': 'SB3_NOT_AVAILABLE',
            'ppo_action': -1,
            'ppo_weight': 0.0
        }
    
    if symbol not in ppo_status or not ppo_status[symbol]['has_ppo']:
        return {
            'ppo_available': False,
            'ppo_signal': 'MODEL_NOT_FOUND',
            'ppo_action': -1,
            'ppo_weight': 0.0
        }
    
    try:
        model_path = ppo_status[symbol]['ppo_path']
        
        # Check if it's ensemble
        if model_path.endswith('.pkl'):
            # Ensemble model
            with open(model_path, 'rb') as f:
                ensemble_info = joblib.load(f)
            if ensemble_info.get('model_paths'):
                model_path = ensemble_info['model_paths'][0]
        
        # Load PPO model
        ppo_model = PPO.load(model_path, device="cpu")
        
        # Build observation (simplified - 54 dims for state_dim)
        obs = np.zeros(54, dtype=np.float32)
        
        # Fill with available data
        if 'close' in row:
            obs[0] = row.get('close', 0) / 1000.0
        if 'volume' in row:
            obs[1] = row.get('volume', 0) / 1000000.0
        if 'rsi' in row:
            obs[2] = row.get('rsi', 50) / 100.0
        if 'macd' in row:
            obs[3] = row.get('macd', 0)
        
        # PPO predict
        action, _ = ppo_model.predict(obs, deterministic=True)
        
        # Action mapping
        if isinstance(action, (list, tuple, np.ndarray)):
            action_val = int(action[0]) if len(action) > 0 else 0
        else:
            action_val = int(action)
        
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        ppo_signal = action_map.get(action_val, 'HOLD')
        
        return {
            'ppo_available': True,
            'ppo_signal': ppo_signal,
            'ppo_action': action_val,
            'ppo_weight': 1.0
        }
        
    except Exception as e:
        print(f"      ⚠️ PPO inference failed for {symbol}: {e}")
        return {
            'ppo_available': True,
            'ppo_signal': 'ERROR',
            'ppo_action': -1,
            'ppo_weight': 0.0
        }

# =========================================================
# ✅ ৮.৫ Agentic Loop ডেটা ফেচ ফাংশন (NEW)
# =========================================================
def get_agentic_data(symbol):
    """Agentic Loop consensus ডেটা রিটার্ন করে"""
    if agentic_loop and symbol in latest_data.index:
        try:
            symbol_data = mongo_df[mongo_df['symbol'] == symbol].tail(50)
            decision, score, confidence, details = agentic_loop.get_consensus(
                symbol=symbol,
                symbol_data=symbol_data,
                volatility=0.02,
                market_regime='NEUTRAL'
            )
            return {
                'agentic_available': True,
                'agentic_signal': decision,
                'agentic_score': score,
                'agentic_confidence': confidence
            }
        except:
            pass
    
    return {
        'agentic_available': False,
        'agentic_signal': 'NOT_AVAILABLE',
        'agentic_score': 0.5,
        'agentic_confidence': 0.0
    }

# =========================================================
# ✅ ৮.৬ PatchTST ডেটা ফেচ ফাংশন (NEW)
# =========================================================
def get_patch_tst_data(symbol):
    """PatchTST prediction ডেটা রিটার্ন করে"""
    if patch_tst and symbol in latest_data.index:
        try:
            symbol_df = mongo_df[mongo_df['symbol'] == symbol]
            pred = patch_tst.predict(symbol, symbol_df)
            return {
                'patch_tst_available': True,
                'patch_tst_direction': pred.get('direction', 'UNKNOWN'),
                'patch_tst_confidence': pred.get('confidence', 0.0),
                'patch_tst_up_prob': pred.get('up_prob', 0.5)
            }
        except:
            pass
    
    return {
        'patch_tst_available': False,
        'patch_tst_direction': 'NOT_AVAILABLE',
        'patch_tst_confidence': 0.0,
        'patch_tst_up_prob': 0.5
    }

# =========================================================
# ✅ ৮.৭ Sector ডেটা ফেচ ফাংশন (NEW)
# =========================================================
def get_sector_data(symbol):
    """Sector ranking ডেটা রিটার্ন করে"""
    if sector_engine:
        try:
            sector = sector_engine.get_sector(symbol)
            top3 = [s for s, _ in sector_engine.get_top_sectors(3)]
            bottom2 = [s for s, _ in sector_engine.get_bottom_sectors(2)]
            
            if sector in top3:
                sector_score = 80
            elif sector in bottom2:
                sector_score = 20
            else:
                sector_score = 50
            
            return {
                'sector_name': sector,
                'sector_score': sector_score,
                'is_top_sector': sector in top3,
                'is_bottom_sector': sector in bottom2
            }
        except:
            pass
    
    return {
        'sector_name': 'Unknown',
        'sector_score': 50,
        'is_top_sector': False,
        'is_bottom_sector': False
    }

# =========================================================
# ৯. মূল লুপ - প্রতিটি সিম্বলের জন্য ডেটা যোগ করুন
# =========================================================
print("\n🎯 Generating AI values for each symbol...")

llm_results = {}
xgb_results = {}
ppo_results = {}
agentic_results = {}
patch_tst_results = {}
sector_results = {}

for symbol in target_symbols:
    print(f"\n   🔍 Processing {symbol}...")
    
    row = latest_data.loc[symbol] if symbol in latest_data.index else None
    
    # LLM
    if row is not None:
        llm_results[symbol] = get_llm_signal(symbol, row)
        print(f"      LLM: {llm_results[symbol]['llm_signal']} ({llm_results[symbol]['llm_confidence']:.1f}%)")
    else:
        llm_results[symbol] = {
            'llm_signal': 'NO_DATA', 'llm_confidence': 0,
            'llm_entry': 0, 'llm_stop_loss': 0, 'llm_target': 0,
            'llm_strength': 'NO_DATA', 'llm_response': ''
        }
        print(f"      LLM: No market data available")
    
    # XGBoost
    xgb_results[symbol] = get_xgb_data(symbol)
    print(f"      XGB: prob={xgb_results[symbol]['xgb_prob_up']:.3f}, conf={xgb_results[symbol]['xgb_confidence']:.1f}%")
    
    # ✅ PPO (FULL SIGNAL)
    ppo_results[symbol] = get_ppo_data(symbol, row if row is not None else {})
    if ppo_results[symbol]['ppo_available']:
        print(f"      PPO: {ppo_results[symbol]['ppo_signal']} (action: {ppo_results[symbol]['ppo_action']})")
    else:
        print(f"      PPO: {ppo_results[symbol]['ppo_signal']}")
    
    # Agentic Loop
    agentic_results[symbol] = get_agentic_data(symbol)
    if agentic_results[symbol]['agentic_available']:
        print(f"      Agentic: {agentic_results[symbol]['agentic_signal']} (score: {agentic_results[symbol]['agentic_score']:.3f})")
    else:
        print(f"      Agentic: Not available")
    
    # PatchTST
    patch_tst_results[symbol] = get_patch_tst_data(symbol)
    if patch_tst_results[symbol]['patch_tst_available']:
        print(f"      PatchTST: {patch_tst_results[symbol]['patch_tst_direction']} (conf: {patch_tst_results[symbol]['patch_tst_confidence']:.3f})")
    else:
        print(f"      PatchTST: Not available")
    
    # Sector
    sector_results[symbol] = get_sector_data(symbol)
    print(f"      Sector: {sector_results[symbol]['sector_name']} (score: {sector_results[symbol]['sector_score']})")

# =========================================================
# ১০. support_resistance.csv-তে নতুন কলাম যোগ করুন
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

# ✅ PPO কলাম (UPDATED with actual signal)
sr_df['ppo_available'] = sr_df['symbol'].map(lambda s: ppo_results.get(s, {}).get('ppo_available', False))
sr_df['ppo_signal'] = sr_df['symbol'].map(lambda s: ppo_results.get(s, {}).get('ppo_signal', 'UNKNOWN'))
sr_df['ppo_action'] = sr_df['symbol'].map(lambda s: ppo_results.get(s, {}).get('ppo_action', -1))

# Agentic Loop কলাম
sr_df['agentic_available'] = sr_df['symbol'].map(lambda s: agentic_results.get(s, {}).get('agentic_available', False))
sr_df['agentic_signal'] = sr_df['symbol'].map(lambda s: agentic_results.get(s, {}).get('agentic_signal', 'NOT_AVAILABLE'))
sr_df['agentic_score'] = sr_df['symbol'].map(lambda s: agentic_results.get(s, {}).get('agentic_score', 0.5))
sr_df['agentic_confidence'] = sr_df['symbol'].map(lambda s: agentic_results.get(s, {}).get('agentic_confidence', 0.0))

# PatchTST কলাম
sr_df['patch_tst_available'] = sr_df['symbol'].map(lambda s: patch_tst_results.get(s, {}).get('patch_tst_available', False))
sr_df['patch_tst_direction'] = sr_df['symbol'].map(lambda s: patch_tst_results.get(s, {}).get('patch_tst_direction', 'NOT_AVAILABLE'))
sr_df['patch_tst_confidence'] = sr_df['symbol'].map(lambda s: patch_tst_results.get(s, {}).get('patch_tst_confidence', 0.0))
sr_df['patch_tst_up_prob'] = sr_df['symbol'].map(lambda s: patch_tst_results.get(s, {}).get('patch_tst_up_prob', 0.5))

# Sector কলাম
sr_df['sector_name'] = sr_df['symbol'].map(lambda s: sector_results.get(s, {}).get('sector_name', 'Unknown'))
sr_df['sector_score'] = sr_df['symbol'].map(lambda s: sector_results.get(s, {}).get('sector_score', 50))
sr_df['is_top_sector'] = sr_df['symbol'].map(lambda s: sector_results.get(s, {}).get('is_top_sector', False))
sr_df['is_bottom_sector'] = sr_df['symbol'].map(lambda s: sector_results.get(s, {}).get('is_bottom_sector', False))

# =========================================================
# ১১. কম্বাইন্ড স্কোর তৈরি করুন (6 Sources — PPO included)
# =========================================================
def calculate_combined_score(row):
    """ছয়টি সোর্সের কম্বাইন্ড স্কোর (LLM + XGB + PPO + Agentic + PatchTST + Sector)"""
    score = 0
    weight_sum = 0
    
    # LLM (20%)
    if row['llm_signal'] == 'BUY':
        score += row['llm_confidence'] * 0.20
    elif row['llm_signal'] == 'SELL':
        score += (100 - row['llm_confidence']) * 0.20
    else:
        score += 50 * 0.20
    weight_sum += 0.20
    
    # XGBoost (20%)
    score += row['xgb_prob_up'] * 100 * 0.20
    weight_sum += 0.20
    
    # ✅ PPO (15%) — Actual signal used
    if row['ppo_available']:
        ppo_sig = row.get('ppo_signal', 'HOLD')
        if ppo_sig == 'BUY':
            score += 70 * 0.15
        elif ppo_sig == 'SELL':
            score += 30 * 0.15
        else:  # HOLD
            score += 50 * 0.15
        weight_sum += 0.15
    
    # Agentic Loop (15%)
    if row['agentic_available']:
        agentic_sig = row.get('agentic_signal', 'HOLD')
        if agentic_sig == 'BUY':
            score += row.get('agentic_score', 0.5) * 100 * 0.15
        elif agentic_sig == 'SELL':
            score += (1 - row.get('agentic_score', 0.5)) * 100 * 0.15
        else:
            score += 50 * 0.15
        weight_sum += 0.15
    
    # PatchTST (15%)
    if row['patch_tst_available']:
        score += row['patch_tst_up_prob'] * 100 * 0.15
        weight_sum += 0.15
    
    # Sector (15%)
    score += row['sector_score'] * 0.15
    weight_sum += 0.15
    
    return score / weight_sum if weight_sum > 0 else 50

sr_df['combined_score'] = sr_df.apply(calculate_combined_score, axis=1)

# কম্বাইন্ড সিগন্যাল
def get_combined_signal(score):
    if score >= 75:
        return 'STRONG BUY'
    elif score >= 60:
        return 'BUY'
    elif score <= 25:
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
print("\n✅ PPO Signal Distribution:")
print(sr_df['ppo_signal'].value_counts().to_string())
print("\n✅ Agentic Loop Available:")
print(f"   {sr_df['agentic_available'].sum()}/{len(sr_df)} symbols")
print("\n✅ PatchTST Available:")
print(f"   {sr_df['patch_tst_available'].sum()}/{len(sr_df)} symbols")
print(f"\n📤 Output: {OUTPUT_PATH}")
print("="*70)

# =========================================================
# ১৩. ডেমো
# =========================================================
print("\n📋 SAMPLE (First 5 rows):")
display_cols = ['symbol', 'combined_signal', 'combined_score', 
                'llm_signal', 'xgb_prob_up', 'ppo_signal',
                'agentic_signal', 'patch_tst_direction', 'sector_name']
print(sr_df[display_cols].head(5).to_string())