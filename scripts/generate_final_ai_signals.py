"""
scripts/generate_final_ai_signals.py
সমস্ত AI মডেল (GPT-2 + Qwen3 + DeepSeek + XGBoost + PPO + Agentic Loop + PatchTST) একত্রে
কম্বাইন্ড ফাইনাল ট্রেডিং সিগন্যাল জেনারেটর

✅ LLM: GPT-2 (legacy) + Qwen3-0.6B + DeepSeek-1.5B
✅ Elliott Wave: REMOVED
✅ Bullish Strong: REMOVED
✅ Sector Analysis: REMOVED
✅ PPO: Real model load with proper obs reshape (1, 50)
✅ PatchTST: Prediction included
✅ Simplified 33-column output
"""

import pandas as pd
import numpy as np
import torch
import os
import re
import json
import joblib
import sys
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================================================
# PATH SETUP
# =========================================================
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# =========================================================
# কনফিগারেশন
# =========================================================

# ✅ LLM models (all 3)
GPT2_MODEL_DIR = "./csv/llm_model"                     # ← GPT-2 (legacy)
QWEN3_MODEL_DIR = "./csv/llm_model_qwen3"              # ← Qwen3
DEEPSEEK_MODEL_DIR = "./csv/llm_model_deepseek"        # ← DeepSeek

MONGO_PATH = "./csv/mongodb.csv"
XGBOOST_DIR = "./csv/xgboost"
PPO_MODELS_DIR = "./csv/ppo_models/per_symbol"
MODEL_METADATA_PATH = "./csv/model_metadata.csv"
PREDICTION_LOG_PATH = "./csv/prediction_log.csv"
XGB_CONFIDENCE_PATH = "./csv/xgb_confidence.csv"
AGENTIC_LOOP_STATE = "./csv/agentic_loop_state.json"

FINAL_OUTPUT_PATH = "./output/ai_signal/FINAL_AI_SIGNALS.csv"
os.makedirs("./output/ai_signal", exist_ok=True)

# ✅ PPO observation config — MUST match ppo_train.py
WINDOW = 10
DEFAULT_MARKET_COLS = ["open", "high", "low", "close", "volume"]
try:
    from env_trading import MARKET_COLS
except ImportError:
    MARKET_COLS = DEFAULT_MARKET_COLS

STATE_DIM = len(MARKET_COLS) * WINDOW  # = 50

# =========================================================
# ✅ ADDITIONAL IMPORTS
# =========================================================

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
    print("✅ Stable-Baselines3 loaded")
except ImportError:
    SB3_AVAILABLE = False
    print("⚠️ Stable-Baselines3 not available")

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

# =========================================================
# ✅ INITIALIZE COMPONENTS
# =========================================================

agentic_loop = None
if AGENTIC_LOOP_AVAILABLE:
    try:
        agentic_loop = AgenticLoop(xgb_model_dir=XGBOOST_DIR)
        print("✅ Agentic Loop initialized")
    except Exception as e:
        print(f"⚠️ Agentic Loop failed: {e}")

patch_tst = None
if PATCHTST_AVAILABLE:
    try:
        patch_tst = PatchTSTIntegration(model_dir="./csv/patchtst_models")
        print("✅ PatchTST initialized")
    except Exception as e:
        print(f"⚠️ PatchTST failed: {e}")

# =========================================================
# ✅ AI মডেল ওজন (Weights)
# =========================================================
AI_WEIGHTS = {
    'gpt2': 0.10,         # GPT-2 (legacy)
    'qwen3': 0.15,        # Qwen3-0.6B
    'deepseek': 0.10,     # DeepSeek-1.5B
    'xgb': 0.25,          # XGBoost
    'ppo': 0.15,          # PPO
    'agentic': 0.15,      # Agentic Loop
    'patch_tst': 0.10,    # PatchTST
}
# Total = 1.00 ✅

print("="*70)
print("🤖 COMBINED AI TRADING SIGNAL GENERATOR")
print("="*70)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📊 Weights: GPT2={AI_WEIGHTS['gpt2']*100:.0f}% | Qwen3={AI_WEIGHTS['qwen3']*100:.0f}% | DeepSeek={AI_WEIGHTS['deepseek']*100:.0f}%")
print(f"          XGB={AI_WEIGHTS['xgb']*100:.0f}% | PPO={AI_WEIGHTS['ppo']*100:.0f}% | Agentic={AI_WEIGHTS['agentic']*100:.0f}% | PatchTST={AI_WEIGHTS['patch_tst']*100:.0f}%")
print(f"🔧 PPO STATE_DIM={STATE_DIM} | WINDOW={WINDOW}")
print("="*70)

# =========================================================
# ১. mongodb — ONCE লোড
# =========================================================
print("\n📂 Loading symbols from mongodb...")
if not os.path.exists(MONGO_PATH):
    print(f"   ❌ mongodb.csv not found: {MONGO_PATH}")
    sys.exit(1)

mongo_df = pd.read_csv(MONGO_PATH)
mongo_df['date'] = pd.to_datetime(mongo_df['date'], format='mixed', errors='coerce')
mongo_df = mongo_df.sort_values(['symbol', 'date'])
target_symbols = mongo_df['symbol'].unique().tolist()
print(f"   ✅ Loaded {len(target_symbols)} symbols")

# Per-symbol cache
mongo_by_symbol = {sym: grp for sym, grp in mongo_df.groupby('symbol')}
latest_market = mongo_df.groupby('symbol').tail(1).set_index('symbol')

# =========================================================
# ✅ PPO OBSERVATION BUILDER
# =========================================================
def build_ppo_observation(symbol, symbol_df=None, window=WINDOW):
    """Build 50-dim PPO observation from market features."""
    try:
        if symbol_df is None:
            symbol_df = mongo_by_symbol.get(symbol)

        if symbol_df is None or len(symbol_df) == 0:
            return np.zeros(STATE_DIM, dtype=np.float32)

        sym_tail = symbol_df.tail(window)

        available_cols = [c for c in MARKET_COLS if c in sym_tail.columns]
        if not available_cols:
            available_cols = ['close', 'volume'] if 'close' in sym_tail.columns else ['close']

        seg = sym_tail[available_cols].values.astype(np.float32)

        if len(seg) < window:
            pad = window - len(seg)
            seg = np.pad(seg, ((pad, 0), (0, 0)), mode="edge")

        market_vec = seg.flatten()

        if len(market_vec) < STATE_DIM:
            market_vec = np.pad(market_vec, (0, STATE_DIM - len(market_vec)))
        elif len(market_vec) > STATE_DIM:
            market_vec = market_vec[:STATE_DIM]

        return np.nan_to_num(market_vec).astype(np.float32)
    except Exception as e:
        return np.zeros(STATE_DIM, dtype=np.float32)

# =========================================================
# ✅ SAFE ACTION EXTRACTOR
# =========================================================
def extract_ppo_action(action):
    """Extract scalar action value from PPO predict output."""
    try:
        action_arr = np.asarray(action).flatten()
        if len(action_arr) > 0:
            return int(action_arr[0])
    except:
        pass
    return 0

# =========================================================
# ২. XGBoost ডেটা লোড
# =========================================================
print("\n📂 Loading XGBoost data...")

meta_df = pd.DataFrame()
good_xgb_symbols = np.array([])
if os.path.exists(MODEL_METADATA_PATH):
    try:
        _meta = pd.read_csv(MODEL_METADATA_PATH)
        if 'auc' in _meta.columns and 'symbol' in _meta.columns:
            meta_df = _meta[_meta['auc'] >= 0.55]
            good_xgb_symbols = meta_df['symbol'].unique()
            print(f"   ✅ model_metadata.csv loaded ({len(good_xgb_symbols)} good models)")
    except Exception as e:
        print(f"   ⚠️ Failed to load model_metadata: {e}")

pred_df = pd.DataFrame()
if os.path.exists(PREDICTION_LOG_PATH):
    try:
        pred_df = pd.read_csv(PREDICTION_LOG_PATH)
        pred_df['date'] = pd.to_datetime(pred_df['date'], format='mixed', errors='coerce')
        pred_df = pred_df.sort_values(['symbol', 'date'])
        pred_df = pred_df.drop_duplicates(subset=['symbol', 'date'], keep='last')
        print(f"   ✅ prediction_log.csv loaded ({len(pred_df)} rows)")
    except Exception as e:
        print(f"   ⚠️ Failed to load prediction_log: {e}")
        pred_df = pd.DataFrame()

conf_df = pd.DataFrame()
if os.path.exists(XGB_CONFIDENCE_PATH):
    try:
        conf_df = pd.read_csv(XGB_CONFIDENCE_PATH)
        conf_df['date'] = pd.to_datetime(conf_df['date'], format='mixed', errors='coerce')
        print(f"   ✅ xgb_confidence.csv loaded ({len(conf_df)} rows)")
    except Exception as e:
        print(f"   ⚠️ Failed to load xgb_confidence: {e}")
        conf_df = pd.DataFrame()

if not pred_df.empty and not conf_df.empty:
    try:
        xgb_df = pd.merge(pred_df, conf_df, on=['symbol', 'date'], how='left')
    except Exception as e:
        xgb_df = pred_df.copy()
elif not pred_df.empty:
    xgb_df = pred_df.copy()
else:
    xgb_df = pd.DataFrame()

if not xgb_df.empty and len(good_xgb_symbols) > 0:
    xgb_df = xgb_df[xgb_df['symbol'].isin(good_xgb_symbols)]

if not xgb_df.empty:
    if 'prob_up' not in xgb_df.columns:
        if 'prediction' in xgb_df.columns and 'confidence_score' in xgb_df.columns:
            xgb_df['prob_up'] = xgb_df.apply(
                lambda row: row['confidence_score'] / 100 if row['prediction'] == 1
                else (100 - row['confidence_score']) / 100,
                axis=1
            )
        else:
            xgb_df['prob_up'] = 0.5

    xgb_latest = xgb_df.sort_values(['symbol', 'date']).groupby('symbol').tail(1).set_index('symbol')
else:
    xgb_latest = pd.DataFrame()

print(f"   ✅ XGBoost: {len(good_xgb_symbols)} good models, {len(xgb_latest)} symbols ready")

# =========================================================
# ৩. PPO মডেল লোড
# =========================================================
print("\n📂 Loading PPO models with REAL market observations...")
ppo_data = {}

for symbol in target_symbols:
    ppo_path = os.path.join(PPO_MODELS_DIR, f"ppo_{symbol}.zip")
    ensemble_path = os.path.join(PPO_MODELS_DIR, f"ensemble_{symbol}.pkl")

    obs = build_ppo_observation(symbol, mongo_by_symbol.get(symbol))
    obs_2d = np.asarray(obs, dtype=np.float32).reshape(1, -1)

    if os.path.exists(ppo_path) and SB3_AVAILABLE:
        try:
            ppo_model = PPO.load(ppo_path, device="cpu")
            action, _ = ppo_model.predict(obs_2d, deterministic=True)
            action_val = extract_ppo_action(action)
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            ppo_data[symbol] = {
                'available': True,
                'signal': action_map.get(action_val, 'HOLD'),
                'confidence': 65.0,
                'action': action_val
            }
        except Exception as e:
            ppo_data[symbol] = {'available': False, 'signal': 'ERROR', 'confidence': 0, 'action': -1}
    elif os.path.exists(ensemble_path) and SB3_AVAILABLE:
        try:
            with open(ensemble_path, 'rb') as f:
                ensemble_info = joblib.load(f)
            if ensemble_info.get('model_paths'):
                all_actions = []
                for mp in ensemble_info['model_paths']:
                    try:
                        m = PPO.load(mp, device="cpu")
                        a, _ = m.predict(obs_2d, deterministic=True)
                        all_actions.append(extract_ppo_action(a))
                    except:
                        continue

                if all_actions:
                    action_val = int(pd.Series(all_actions).mode().iloc[0])
                else:
                    action_val = 0

                action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
                ppo_data[symbol] = {
                    'available': True,
                    'signal': action_map.get(action_val, 'HOLD'),
                    'confidence': 65.0,
                    'action': action_val
                }
            else:
                ppo_data[symbol] = {'available': False, 'signal': 'N/A', 'confidence': 0, 'action': -1}
        except:
            ppo_data[symbol] = {'available': False, 'signal': 'N/A', 'confidence': 0, 'action': -1}
    else:
        ppo_data[symbol] = {'available': False, 'signal': 'N/A', 'confidence': 0, 'action': -1}

ppo_available = sum(1 for v in ppo_data.values() if v['available'])
print(f"   ✅ PPO: {ppo_available}/{len(target_symbols)} models loaded")

# =========================================================
# ৪. Agentic Loop
# =========================================================
print("\n📂 Initializing Agentic Loop...")

agentic_state_data = {}
if os.path.exists(AGENTIC_LOOP_STATE):
    try:
        with open(AGENTIC_LOOP_STATE, 'r') as f:
            agentic_state_data = json.load(f)
        print(f"   ✅ Agentic Loop state loaded")
    except:
        print(f"   ⚠️ Agentic Loop state file corrupted")

agentic_available = agentic_loop is not None
print(f"   {'✅' if agentic_available else '⚠️'} Agentic Loop: {'Live consensus ready' if agentic_available else 'not available'}")

# =========================================================
# ৫. GPT-2 LOAD
# =========================================================
print("\n🤖 Loading GPT-2 (legacy)...")
gpt2_available = False
gpt2_tokenizer = None
gpt2_model = None
gpt2_device = "cpu"

if os.path.exists(os.path.join(GPT2_MODEL_DIR, "config.json")):
    try:
        gpt2_tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_DIR)
        gpt2_model = AutoModelForCausalLM.from_pretrained(
            GPT2_MODEL_DIR,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model.config.pad_token_id = gpt2_tokenizer.pad_token_id
        gpt2_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpt2_model.to(gpt2_device)
        gpt2_available = True
        print(f"   ✅ GPT-2 loaded on {gpt2_device}")
    except Exception as e:
        print(f"   ⚠️ GPT-2 load failed: {e}")
else:
    print(f"   ⚠️ GPT-2 not found: {GPT2_MODEL_DIR}")

# =========================================================
# ৬. QWEN3 LOAD
# =========================================================
print("\n🤖 Loading Qwen3...")
qwen3_available = False
qwen3_tokenizer = None
qwen3_model = None
qwen3_device = "cpu"

if os.path.exists(os.path.join(QWEN3_MODEL_DIR, "config.json")):
    try:
        qwen3_tokenizer = AutoTokenizer.from_pretrained(
            QWEN3_MODEL_DIR,
            trust_remote_code=True,
        )
        qwen3_model = AutoModelForCausalLM.from_pretrained(
            QWEN3_MODEL_DIR,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        if qwen3_tokenizer.pad_token is None:
            qwen3_tokenizer.pad_token = qwen3_tokenizer.eos_token
        if qwen3_model.config.pad_token_id is None:
            qwen3_model.config.pad_token_id = qwen3_tokenizer.pad_token_id

        qwen3_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qwen3_model.to(qwen3_device)
        qwen3_available = True
        print(f"   ✅ Qwen3 loaded on {qwen3_device}")
    except Exception as e:
        print(f"   ⚠️ Qwen3 load failed: {e}")
else:
    print(f"   ⚠️ Qwen3 not found: {QWEN3_MODEL_DIR}")

# =========================================================
# ৭. DEEPSEEK LOAD
# =========================================================
print("\n🤖 Loading DeepSeek...")
deepseek_available = False
deepseek_tokenizer = None
deepseek_model = None
deepseek_device = "cpu"

if os.path.exists(os.path.join(DEEPSEEK_MODEL_DIR, "config.json")):
    try:
        deepseek_tokenizer = AutoTokenizer.from_pretrained(
            DEEPSEEK_MODEL_DIR,
            trust_remote_code=True,
        )
        deepseek_model = AutoModelForCausalLM.from_pretrained(
            DEEPSEEK_MODEL_DIR,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        if deepseek_tokenizer.pad_token is None:
            deepseek_tokenizer.pad_token = deepseek_tokenizer.eos_token
        if deepseek_model.config.pad_token_id is None:
            deepseek_model.config.pad_token_id = deepseek_tokenizer.pad_token_id

        deepseek_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        deepseek_model.to(deepseek_device)
        deepseek_available = True
        print(f"   ✅ DeepSeek loaded on {deepseek_device}")
    except Exception as e:
        print(f"   ⚠️ DeepSeek load failed: {e}")
else:
    print(f"   ⚠️ DeepSeek not found: {DEEPSEEK_MODEL_DIR}")

# =========================================================
# ✅ LLM INFERENCE FUNCTION (Generic)
# =========================================================
def _llm_inference(tokenizer, model, device, symbol, row, model_name="LLM"):
    """Generic LLM inference"""
    prompt = f"""Symbol: {symbol} | Price: {row.get('close', 0):.2f}
RSI: {row.get('rsi', 50):.1f} | MACD: {row.get('macd', 0):.4f}

Provide trading signal (BUY/SELL/HOLD), confidence%, entry, stop loss, target.

RECOMMENDATION:"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        result = {
            'signal': 'HOLD', 'confidence': 50, 'strength': 'MEDIUM',
            'bias': 'NEUTRAL', 'entry': row.get('close', 0),
            'stop_loss': row.get('close', 0) * 0.98, 'target': row.get('close', 0) * 1.05
        }

        if re.search(r'BUY', response, re.IGNORECASE):
            result['signal'] = 'BUY'; result['bias'] = 'BULLISH'
        elif re.search(r'SELL', response, re.IGNORECASE):
            result['signal'] = 'SELL'; result['bias'] = 'BEARISH'

        conf_match = re.search(r'(\d+)%', response)
        if conf_match: result['confidence'] = float(conf_match.group(1))

        if re.search(r'STRONG', response, re.IGNORECASE): result['strength'] = 'STRONG'
        elif re.search(r'WEAK', response, re.IGNORECASE): result['strength'] = 'WEAK'

        return result
    except Exception as e:
        return {
            'signal': 'ERROR', 'confidence': 0, 'strength': 'N/A',
            'bias': 'NEUTRAL', 'entry': 0, 'stop_loss': 0, 'target': 0
        }

# =========================================================
# ✅ LLM SIGNAL FETCHERS
# =========================================================
def get_gpt2_signal(symbol, row):
    if not gpt2_available:
        return {
            'signal': 'MODEL_NOT_READY', 'confidence': 0, 'strength': 'N/A',
            'bias': 'NEUTRAL', 'entry': 0, 'stop_loss': 0, 'target': 0
        }
    return _llm_inference(gpt2_tokenizer, gpt2_model, gpt2_device, symbol, row, "GPT-2")

def get_qwen3_signal(symbol, row):
    if not qwen3_available:
        return {
            'signal': 'MODEL_NOT_READY', 'confidence': 0, 'strength': 'N/A',
            'bias': 'NEUTRAL', 'entry': 0, 'stop_loss': 0, 'target': 0
        }
    return _llm_inference(qwen3_tokenizer, qwen3_model, qwen3_device, symbol, row, "Qwen3")

def get_deepseek_signal(symbol, row):
    if not deepseek_available:
        return {
            'signal': 'MODEL_NOT_READY', 'confidence': 0, 'strength': 'N/A',
            'bias': 'NEUTRAL', 'entry': 0, 'stop_loss': 0, 'target': 0
        }
    return _llm_inference(deepseek_tokenizer, deepseek_model, deepseek_device, symbol, row, "DeepSeek")

# =========================================================
# ✅ OTHER SIGNAL FETCHERS
# =========================================================
def get_xgb_data(symbol):
    if not xgb_latest.empty and symbol in xgb_latest.index:
        row = xgb_latest.loc[symbol]
        prob = row.get('prob_up', 0.5)
        conf = row.get('confidence_score', 50)

        if prob > 0.60: signal = 'BUY'
        elif prob < 0.40: signal = 'SELL'
        else: signal = 'HOLD'

        return {
            'signal': signal, 'confidence': conf
        }
    return {'signal': 'N/A', 'confidence': 0}

def get_ppo_data(symbol):
    if symbol in ppo_data and ppo_data[symbol]['available']:
        return ppo_data[symbol]
    return {'available': False, 'signal': 'N/A', 'confidence': 0, 'action': -1}

def get_agentic_signal(symbol):
    if not agentic_available or agentic_loop is None:
        return {'bias': 'NEUTRAL', 'available': False, 'confidence': 0}

    try:
        symbol_data = mongo_by_symbol.get(symbol)
        if symbol_data is not None:
            symbol_data = symbol_data.tail(50)

        decision, score, confidence, details = agentic_loop.get_consensus(
            symbol=symbol,
            symbol_data=symbol_data,
            volatility=0.02,
            market_regime='NEUTRAL'
        )
        return {
            'bias': decision,
            'confidence': confidence,
            'available': True
        }
    except:
        return {'bias': 'NEUTRAL', 'available': False, 'confidence': 0}

def get_patch_tst_signal(symbol):
    if not PATCHTST_AVAILABLE or patch_tst is None:
        return {'available': False, 'direction': 'N/A', 'confidence': 0}

    try:
        symbol_df = mongo_by_symbol.get(symbol)
        pred = patch_tst.predict(symbol, symbol_df)
        return {
            'available': True,
            'direction': pred.get('direction', 'UNKNOWN'),
            'confidence': pred.get('confidence', 0)
        }
    except:
        return {'available': False, 'direction': 'N/A', 'confidence': 0}

# =========================================================
# ✅ FINAL SCORE CALCULATION (7 models)
# =========================================================
def calculate_final_combined_score(gpt2_sig, qwen3_sig, deepseek_sig, xgb_sig, ppo_sig, agentic_sig, patch_tst_sig):
    final_score = 0

    # GPT-2
    if gpt2_sig['signal'] == 'BUY': final_score += gpt2_sig['confidence'] * AI_WEIGHTS['gpt2']
    elif gpt2_sig['signal'] == 'SELL': final_score += (100 - gpt2_sig['confidence']) * AI_WEIGHTS['gpt2']
    else: final_score += 50 * AI_WEIGHTS['gpt2']

    # Qwen3
    if qwen3_sig['signal'] == 'BUY': final_score += qwen3_sig['confidence'] * AI_WEIGHTS['qwen3']
    elif qwen3_sig['signal'] == 'SELL': final_score += (100 - qwen3_sig['confidence']) * AI_WEIGHTS['qwen3']
    else: final_score += 50 * AI_WEIGHTS['qwen3']

    # DeepSeek
    if deepseek_sig['signal'] == 'BUY': final_score += deepseek_sig['confidence'] * AI_WEIGHTS['deepseek']
    elif deepseek_sig['signal'] == 'SELL': final_score += (100 - deepseek_sig['confidence']) * AI_WEIGHTS['deepseek']
    else: final_score += 50 * AI_WEIGHTS['deepseek']

    # XGBoost
    if xgb_sig['signal'] == 'BUY': final_score += xgb_sig['confidence'] * AI_WEIGHTS['xgb']
    elif xgb_sig['signal'] == 'SELL': final_score += (100 - xgb_sig['confidence']) * AI_WEIGHTS['xgb']
    else: final_score += 50 * AI_WEIGHTS['xgb']

    # PPO
    if ppo_sig['available']:
        if ppo_sig['signal'] == 'BUY': final_score += ppo_sig['confidence'] * AI_WEIGHTS['ppo']
        elif ppo_sig['signal'] == 'SELL': final_score += (100 - ppo_sig['confidence']) * AI_WEIGHTS['ppo']
        else: final_score += 50 * AI_WEIGHTS['ppo']
    else: final_score += 50 * AI_WEIGHTS['ppo']

    # Agentic
    if agentic_sig['available']:
        if agentic_sig['bias'] in ['BUY', 'STRONG_BUY']:
            final_score += agentic_sig['confidence'] * 100 * AI_WEIGHTS['agentic']
        elif agentic_sig['bias'] in ['SELL', 'STRONG_SELL']:
            final_score += (1 - agentic_sig['confidence']) * 100 * AI_WEIGHTS['agentic']
        else:
            final_score += 50 * AI_WEIGHTS['agentic']
    else: final_score += 50 * AI_WEIGHTS['agentic']

    # PatchTST
    if patch_tst_sig['available']:
        if patch_tst_sig['direction'] == 'UP':
            final_score += (50 + patch_tst_sig['confidence'] * 50) * AI_WEIGHTS['patch_tst']
        elif patch_tst_sig['direction'] == 'DOWN':
            final_score += (50 - patch_tst_sig['confidence'] * 50) * AI_WEIGHTS['patch_tst']
        else:
            final_score += 50 * AI_WEIGHTS['patch_tst']
    else: final_score += 50 * AI_WEIGHTS['patch_tst']

    return min(100, final_score)

def get_final_signal_label(score):
    if score >= 80: return '🔥 STRONG BUY'
    elif score >= 65: return '✅ BUY'
    elif score >= 55: return '👀 WATCH (Near BUY)'
    elif score >= 45: return '⏳ HOLD'
    elif score >= 35: return '⚠️ WATCH (Near SELL)'
    elif score >= 20: return '❌ SELL'
    else: return '💀 STRONG SELL'

def get_model_availability(gpt2_avail, qwen3_avail, deepseek_avail, xgb_avail, ppo_avail, agentic_avail, patch_tst_avail):
    count = sum([gpt2_avail, qwen3_avail, deepseek_avail, xgb_avail, ppo_avail, agentic_avail, patch_tst_avail])
    total = 7
    if count >= 7: return f'FULL ({count}/{total})'
    elif count >= 5: return f'GOOD ({count}/{total})'
    elif count >= 2: return f'BASIC ({count}/{total})'
    else: return f'NONE ({count}/{total})'

# =========================================================
# ৮. Main loop
# =========================================================
print("\n🎯 Generating FINAL AI signals...")
print("-"*70)

results = []

for i, symbol in enumerate(target_symbols):
    print(f"\r   🔍 Processing {i+1}/{len(target_symbols)}: {symbol}...", end='')

    if symbol in latest_market.index:
        market_row = latest_market.loc[symbol]
        current_high = market_row.get('high', 0) if hasattr(market_row, 'get') else 0
    else:
        market_row = pd.Series({'close': 0, 'rsi': 50, 'macd': 0})

    # ✅ All 3 LLM signals
    gpt2_sig = get_gpt2_signal(symbol, market_row)
    qwen3_sig = get_qwen3_signal(symbol, market_row)
    deepseek_sig = get_deepseek_signal(symbol, market_row)

    # ✅ Other models
    xgb_sig = get_xgb_data(symbol)
    ppo_sig = get_ppo_data(symbol)
    agentic_sig = get_agentic_signal(symbol)
    patch_tst_sig = get_patch_tst_signal(symbol)

    model_avail = get_model_availability(
        gpt2_available,
        qwen3_available,
        deepseek_available,
        symbol in good_xgb_symbols,
        ppo_sig['available'],
        agentic_sig['available'],
        patch_tst_sig['available']
    )

    final_score = calculate_final_combined_score(
        gpt2_sig, qwen3_sig, deepseek_sig,
        xgb_sig, ppo_sig, agentic_sig, patch_tst_sig
    )
    final_signal = get_final_signal_label(final_score)

    # ✅ 33-column output
    results.append({
        'symbol': symbol,
        'current_price': market_row.get('close', 0) if hasattr(market_row, 'get') else 0,
        'high': current_high,

        # GPT-2
        'gpt2_signal': gpt2_sig['signal'],
        'gpt2_confidence': round(gpt2_sig['confidence'], 1),
        'gpt2_strength': gpt2_sig['strength'],
        'gpt2_bias': gpt2_sig['bias'],
        'gpt2_available': gpt2_available,

        # Qwen3
        'qwen3_signal': qwen3_sig['signal'],
        'qwen3_confidence': round(qwen3_sig['confidence'], 1),
        'qwen3_strength': qwen3_sig['strength'],
        'qwen3_bias': qwen3_sig['bias'],
        'qwen3_available': qwen3_available,

        # DeepSeek
        'deepseek_signal': deepseek_sig['signal'],
        'deepseek_confidence': round(deepseek_sig['confidence'], 1),
        'deepseek_strength': deepseek_sig['strength'],
        'deepseek_bias': deepseek_sig['bias'],
        'deepseek_available': deepseek_available,

        # XGBoost
        'xgb_signal': xgb_sig['signal'],
        'xgb_confidence': round(xgb_sig['confidence'], 1),
        'xgb_available': symbol in good_xgb_symbols,

        # PPO
        'ppo_signal': ppo_sig['signal'],
        'ppo_confidence': round(ppo_sig['confidence'], 1),
        'ppo_available': ppo_sig['available'],

        # Agentic
        'agentic_signal': agentic_sig['bias'],
        'agentic_confidence': round(agentic_sig.get('confidence', 0), 3),
        'agentic_available': agentic_sig['available'],

        # PatchTST
        'patch_tst_direction': patch_tst_sig['direction'],
        'patch_tst_confidence': round(patch_tst_sig['confidence'], 3),
        'patch_tst_available': patch_tst_sig['available'],

        # Final
        'model_availability': model_avail,
        'final_combined_score': round(final_score, 1),
        'final_signal': final_signal,
    })

print("\n")

# =========================================================
# ৯. সেভ
# =========================================================
output_df = pd.DataFrame(results)
output_df = output_df.sort_values('final_combined_score', ascending=False)
output_df = output_df[output_df['final_combined_score'] >= 55]
output_df.to_csv(FINAL_OUTPUT_PATH, index=False)

# =========================================================
# ১০. রিপোর্ট
# =========================================================
print("="*70)
print("📊 FINAL AI TRADING SIGNALS REPORT (GPT-2 + Qwen3 + DeepSeek)")
print("="*70)
print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📊 Total Signals: {len(output_df)}")

print(f"\n🤖 AI Models Available:")
print(f"   GPT-2: {'✅ Available' if gpt2_available else '❌ Not Ready'}")
print(f"   Qwen3: {'✅ Available' if qwen3_available else '❌ Not Ready'}")
print(f"   DeepSeek: {'✅ Available' if deepseek_available else '❌ Not Ready'}")
print(f"   XGBoost: ✅ Available ({len(good_xgb_symbols)} models)")
print(f"   PPO: {'✅' if ppo_available > 0 else '❌'} Available ({ppo_available} models)")
print(f"   Agentic Loop: {'✅' if agentic_available else '❌'} Available")
print(f"   PatchTST: {'✅' if PATCHTST_AVAILABLE else '❌'} Available")

print(f"\n📈 SIGNAL DISTRIBUTION:")
if len(output_df) > 0:
    print(output_df['final_signal'].value_counts().to_string())
else:
    print("   (No signals)")

print(f"\n📊 PPO SIGNAL DISTRIBUTION:")
if len(output_df) > 0:
    print(output_df['ppo_signal'].value_counts().to_string())
else:
    print("   (No signals)")

print(f"\n📊 MODEL AVAILABILITY:")
if len(output_df) > 0:
    print(output_df['model_availability'].value_counts().to_string())
else:
    print("   (No signals)")

print(f"\n🔥 TOP 10 BUY SIGNALS:")
if len(output_df) > 0:
    buy_signals = output_df[output_df['final_signal'].str.contains('BUY', na=False)].head(10)
    if len(buy_signals) > 0:
        print(buy_signals[['symbol', 'final_signal', 'final_combined_score',
                            'gpt2_signal', 'qwen3_signal', 'deepseek_signal',
                            'xgb_signal', 'ppo_signal', 'patch_tst_direction']].to_string())
    else:
        print("   (No BUY signals)")

print(f"\n💀 TOP 5 SELL SIGNALS:")
if len(output_df) > 0:
    sell_signals = output_df[output_df['final_signal'].str.contains('SELL', na=False)].head(5)
    if len(sell_signals) > 0:
        print(sell_signals[['symbol', 'final_signal', 'final_combined_score',
                             'gpt2_signal', 'qwen3_signal', 'deepseek_signal',
                             'ppo_signal']].to_string())
    else:
        print("   (No SELL signals)")

print(f"\n" + "="*70)
print(f"✅ FINAL OUTPUT: {FINAL_OUTPUT_PATH}")
print(f"📊 Total Columns: {len(output_df.columns)}")
print("="*70)

print(f"\n📋 ALL COLUMNS ({len(output_df.columns)}):")
for i, col in enumerate(output_df.columns, 1):
    print(f"   {i:2d}. {col}")

print("="*70)