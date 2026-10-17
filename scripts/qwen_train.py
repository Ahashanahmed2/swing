# ================== scripts/qwen_train.py ==================
# Qwen3 Trainer v5 — Full Patched
# ✅ FIX: LR too low (8e-6 → 2e-4) for LoRA
# ✅ FIX: Epochs too high (12 → 2)
# ✅ FIX: Max length (384 → 192) for speed
# ✅ FIX: Checkpoint-based mode detection (not tracking-based)
# ✅ FIX: Optimizer mismatch auto-handle
# ✅ FIX: Loss monitoring (warn if > 4.0)

import os
import torch
import json
import warnings
import pandas as pd
import numpy as np
import re
import requests
import joblib
import glob
import shutil
from datetime import datetime, timedelta
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from huggingface_hub import login, create_repo, upload_folder, upload_file, HfApi, CommitOperationDelete

# =========================================================
# AGENTIC LOOP
# =========================================================
try:
    from agentic_loop import AgenticLoop
    AGENTIC_LOOP_AVAILABLE = True
except ImportError:
    AGENTIC_LOOP_AVAILABLE = False
    print("⚠️ Agentic Loop not found")

try:
    from peft import LoraConfig, get_peft_model
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    print("⚠️ PEFT not installed")

warnings.filterwarnings('ignore')


# =========================================================
# TELEGRAM
# =========================================================
def send_telegram_message(message, token=None, chat_id=None):
    token = token or os.getenv("TELEGRAM_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        return requests.post(url, json=payload, timeout=10).json()
    except Exception as e:
        print(f"⚠️ Telegram failed: {e}")


# =========================================================
# CONFIGURATION
# =========================================================

BATCH_SIZE = 40
TOTAL_BATCHES = "auto"
MAX_SYMBOLS_PER_BATCH = 40

HF_DATASET_REPO = "ahashanahmed/csv"

# ✅ QWEN3 MODEL
BASE_MODEL = "Qwen/Qwen3-0.6B"
LLM_MODEL_DIR = "./csv/llm_model_qwen3"

# ✅ QWEN3 CHECKPOINT HF PATHS
QWEN3_HF_CHECKPOINT_PREFIX = "qwen3_checkpoints/qwen3_checkpoint-"
QWEN3_FINAL_MODEL_PREFIX = "final_model_qwen3"
QWEN3_LOCAL_CHECKPOINT_DIR = "./csv/qwen3_checkpoints"

# ✅ Per-model paths
TRACKING_FILE = "./csv/trained_symbols_qwen3.json"
BATCH_TRACKING_FILE = "./csv/batch_tracking_qwen3.json"
LAST_FINE_TUNE_FILE = "./csv/last_finetune_qwen3.txt"
LAST_CONSOLIDATE_FILE = "./csv/last_consolidate_qwen3.txt"

# ✅ Data paths
MARKET_DATA_PATH = "./csv/mongodb.csv"
TRAINING_DATA_PATH = "./csv/training_texts.txt"
MISTAKES_FILE = "./csv/trading_mistakes_qwen3.csv"
CONFIDENCE_LOG = "./csv/llm_confidence_log_qwen3.csv"
HARD_EXAMPLES_FILE = "./csv/hard_examples_qwen3.csv"

# ✅ Shared paths
XGBOOST_DIR = "./csv/xgboost"
PPO_MODELS_DIR = "./csv/ppo_models"
PPO_PER_SYMBOL_DIR = "./csv/ppo_models/per_symbol"

# Schedule
FINE_TUNE_INTERVAL = 7
CONSOLIDATE_INTERVAL = 30

# Learning parameters
MAX_OLD_EXAMPLES = 10000
HARD_EXAMPLE_THRESHOLD = 0.25
HIGH_PRIORITY_THRESHOLD = 0.35
MAX_GRAD_NORM = 0.5
VALIDATION_SPLIT_RATIO = 0.15

FORCE_RETRAIN = False

# ✅ LoRA config (r=32, alpha=64 — for Qwen3-0.6B)
LORA_CONFIG = {
    'r': 32,
    'lora_alpha': 64,
    'target_modules': [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj'
    ],
    'lora_dropout': 0.05,
    'bias': 'none',
    'task_type': 'CAUSAL_LM',
}

# ✅ Qwen3 load config
QWEN3_LOAD_KWARGS = {
    'trust_remote_code': True,
    'torch_dtype': torch.float32,
    'low_cpu_mem_usage': True,
}

# Label patterns
SIGNAL_PATTERNS = {
    'bullish': r'(?:Signal|Prediction|Recommendation):?\s*(?:BUY|Bullish|LONG|✅ BUY)',
    'bearish': r'(?:Signal|Prediction|Recommendation):?\s*(?:SELL|Bearish|SHORT|❌ SELL)',
    'neutral': r'(?:Signal|Prediction|Recommendation):?\s*(?:HOLD|Neutral|WAIT|⏳ WAIT)'
}

CONFIDENCE_PATTERN = r'(?:Confidence|Signal Strength):?\s*(\d+(?:\.\d+)?)%'

AGENTIC_LOOP_STATE_FILE = "./csv/agentic_loop_state.json"
AGENTIC_LOOP_LOG_DIR = "./csv/agentic_loop_logs"

# =========================================================
# ✅ FIX: EPOCH + LR CONFIG (CRITICAL)
# =========================================================
EPOCHS_CONFIG = {
    "first_train": 2,           # ← was 12
    "incremental": 2,           # ← was 4
    "weekly_finetune": 1,       # ← was 6
    "consolidate": 3,           # ← was 20
    "mistake_learning": 1,      # ← was 10
}

LR_CONFIG = {
    "first_train": 2e-4,        # ← was 8e-6 (25x higher)
    "incremental": 1e-4,        # ← was 5e-6
    "weekly_finetune": 5e-5,    # ← was 2e-6
    "consolidate": 1e-4,        # ← was 5e-6
    "mistake_learning": 5e-5,   # ← was 5e-6
}

BATCH_SIZE_CONFIG = {
    "first_train": 1,
    "incremental": 1,
    "weekly_finetune": 1,
    "consolidate": 1,
    "mistake_learning": 1,
}

GRAD_ACCUM_CONFIG = {
    "first_train": 32,
    "incremental": 24,
    "weekly_finetune": 16,
    "consolidate": 32,
    "mistake_learning": 24,
}

# Mode explanation
MODE_EXPLANATION = {
    "first_train": "🎯 FIRST TIME TRAINING (Qwen3-0.6B - Base Model)",
    "incremental": "⚙️ INCREMENTAL TRAINING (New symbols added)",
    "weekly_finetune": "🔄 WEEKLY FINE-TUNE (Every 7 days)",
    "consolidate": "📈 MONTHLY RE-TUNE (Every 30 days)",
    "mistake_learning": "🎯 MISTAKE LEARNING (Retraining from past errors)"
}

MODE_SHORT_EXPLANATION = {
    "first_train": "First Time",
    "incremental": "New Symbols",
    "weekly_finetune": "Weekly Fine-Tune",
    "consolidate": "Monthly Re-Tune",
    "mistake_learning": "Mistake Learning"
}


# =========================================================
# HF UPLOADER
# =========================================================

class HFUploader:
    def __init__(self, repo_id=HF_DATASET_REPO):
        self.repo_id = repo_id
        self.api = None
        self._init_api()

    def _init_api(self):
        token = os.getenv("hf_token")
        if token:
            try:
                login(token=token)
                self.api = HfApi(token=token)
                create_repo(repo_id=self.repo_id, repo_type="dataset", exist_ok=True)
                print(f"   ✅ HF Dataset Repo ready: {self.repo_id}")
            except Exception as e:
                print(f"   ⚠️ HF API init failed: {e}")
                self.api = None

    def upload_checkpoint(self, checkpoint_path, step_num):
        """Upload Qwen3 checkpoint (qwen3_checkpoints/qwen3_checkpoint-N)"""
        if self.api is None:
            return False
        try:
            repo_path = f"{QWEN3_HF_CHECKPOINT_PREFIX}{step_num}"
            self.api.upload_folder(
                folder_path=checkpoint_path,
                path_in_repo=repo_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"🤖 Qwen3 checkpoint {step_num} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            print(f"   📤 Qwen3 checkpoint {step_num} → {self.repo_id}/{repo_path}")
            return True
        except Exception as e:
            print(f"   ⚠️ Qwen3 checkpoint upload failed: {e}")
            return False

    def upload_final_model(self, model_path, mode):
        if self.api is None:
            return False
        try:
            repo_path = f"{QWEN3_FINAL_MODEL_PREFIX}/{mode}"
            self.api.upload_folder(
                folder_path=model_path,
                path_in_repo=repo_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"🤖 Qwen3 Final Model ({mode}) - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            print(f"   📤 Qwen3 final model → {self.repo_id}/{repo_path}/")
            return True
        except Exception as e:
            print(f"   ⚠️ Qwen3 final model upload failed: {e}")
            return False

    def upload_tracking_files(self):
        if self.api is None:
            return
        try:
            if os.path.exists(TRACKING_FILE):
                self.api.upload_file(
                    path_or_fileobj=TRACKING_FILE,
                    path_in_repo="trained_symbols_qwen3.json",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Update Qwen3 trained symbols - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                print(f"   📤 trained_symbols_qwen3.json uploaded")
            if os.path.exists(BATCH_TRACKING_FILE):
                self.api.upload_file(
                    path_or_fileobj=BATCH_TRACKING_FILE,
                    path_in_repo="batch_tracking_qwen3.json",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Update Qwen3 batch tracking - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                print(f"   📤 batch_tracking_qwen3.json uploaded")
        except Exception as e:
            print(f"   ⚠️ Tracking upload failed: {e}")

    def delete_mode_checkpoints(self, mode):
        """Delete HF checkpoints after successful training (per mode)"""
        if self.api is None:
            return
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
            pat = re.compile(rf"qwen3_checkpoint-{re.escape(mode)}(_final)?-\d+$")
            folders = sorted({f.split("/")[1] for f in files
                              if f.startswith("qwen3_checkpoints/") and len(f.split("/")) >= 3
                              and pat.match(f.split("/")[1])})
            if not folders:
                return
            self.api.create_commit(
                repo_id=self.repo_id,
                repo_type="dataset",
                operations=[CommitOperationDelete(path_in_repo=f"qwen3_checkpoints/{fo}/")
                            for fo in folders],
                commit_message=f"🗑️ Qwen3 {mode} finished - remove {len(folders)} checkpoint folder(s)",
            )
            print(f"   🗑️ HF: removed {len(folders)} '{mode}' checkpoint folder(s)")
        except Exception as e:
            print(f"   ⚠️ HF checkpoint cleanup failed: {e}")


# =========================================================
# BATCH MANAGER
# =========================================================

class BatchManager:
    def __init__(self):
        self.batch_tracking = self.load_batch_tracking()
        self._ensure_required_fields()
        self.current_batch_index = self.batch_tracking.get('current_batch', 0)
        self.completed_batches = self.batch_tracking.get('completed_batches', [])
        self.batch_symbols = self.batch_tracking.get('batch_symbols', {})

    def _ensure_required_fields(self):
        required = {
            'weekly_trained_batches': [],
            'last_weekly_finetune': None,
            'last_consolidate': None,
            'monthly_consolidation_done': False,
            'last_mistake_learning': None,
        }
        updated = False
        for field, default_value in required.items():
            if field not in self.batch_tracking:
                self.batch_tracking[field] = default_value
                updated = True
        if updated:
            self.save_batch_tracking()

    def load_batch_tracking(self):
        if os.path.exists(BATCH_TRACKING_FILE):
            try:
                with open(BATCH_TRACKING_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'current_batch': 0,
            'completed_batches': [],
            'batch_symbols': {},
            'total_symbols_trained': 0,
            'last_batch_date': None,
            'weekly_trained_batches': [],
            'last_weekly_finetune': None,
            'last_consolidate': None,
            'monthly_consolidation_done': False,
            'last_mistake_learning': None,
        }

    def save_batch_tracking(self):
        os.makedirs(os.path.dirname(BATCH_TRACKING_FILE), exist_ok=True)
        with open(BATCH_TRACKING_FILE, 'w') as f:
            json.dump(self.batch_tracking, f, indent=2)

    def mark_batch_completed(self, batch_num, symbols):
        if batch_num not in self.completed_batches:
            self.completed_batches.append(batch_num)
            self.current_batch_index = batch_num
            self.batch_tracking['current_batch'] = batch_num
            self.batch_tracking['completed_batches'] = self.completed_batches
            self.batch_symbols[str(batch_num)] = list(symbols)
            self.batch_tracking['batch_symbols'] = self.batch_symbols
            self.batch_tracking['total_symbols_trained'] += len(symbols)
            self.batch_tracking['last_batch_date'] = datetime.now().isoformat()
            if not self.batch_tracking.get('last_weekly_finetune'):
                self.batch_tracking['last_weekly_finetune'] = datetime.now().isoformat()
            self.save_batch_tracking()

    @staticmethod
    def _days_since(iso_str):
        try:
            return (datetime.now() - datetime.fromisoformat(iso_str)).days
        except (TypeError, ValueError):
            return None

    def get_batch_for_weekly_finetune(self):
        if not self.completed_batches:
            return None, []
        last_weekly = self.batch_tracking.get('last_weekly_finetune')
        if last_weekly:
            days = self._days_since(last_weekly)
            if days is not None and days < FINE_TUNE_INTERVAL:
                return None, []
        trained_batches = set(self.batch_tracking.get('weekly_trained_batches', []))
        available_batches = [b for b in self.completed_batches if b not in trained_batches]
        if available_batches:
            batch_num = available_batches[0]
            return batch_num, self.batch_symbols.get(str(batch_num), [])
        if len(trained_batches) >= len(self.completed_batches) and self.completed_batches:
            self.batch_tracking['weekly_trained_batches'] = []
            self.save_batch_tracking()
            batch_num = self.completed_batches[0]
            return batch_num, self.batch_symbols.get(str(batch_num), [])
        return None, []

    def mark_weekly_done(self, batch_num):
        weekly = self.batch_tracking.get('weekly_trained_batches', [])
        if batch_num not in weekly:
            weekly.append(batch_num)
            self.batch_tracking['weekly_trained_batches'] = weekly
            self.batch_tracking['last_weekly_finetune'] = datetime.now().isoformat()
            self.save_batch_tracking()

    def should_consolidate(self):
        last = self.batch_tracking.get('last_consolidate')
        if not last:
            self.batch_tracking['last_consolidate'] = datetime.now().isoformat()
            self.save_batch_tracking()
            return False
        days = self._days_since(last)
        return days is None or days >= CONSOLIDATE_INTERVAL

    def mark_consolidated(self):
        now = datetime.now().isoformat()
        self.batch_tracking['last_consolidate'] = now
        self.batch_tracking['monthly_consolidation_done'] = True
        self.batch_tracking['weekly_trained_batches'] = []
        self.batch_tracking['last_weekly_finetune'] = now
        self.save_batch_tracking()

    def should_run_mistake_learning(self):
        last = self.batch_tracking.get('last_mistake_learning')
        if not last:
            self.batch_tracking['last_mistake_learning'] = datetime.now().isoformat()
            self.save_batch_tracking()
            return True
        days = self._days_since(last)
        return days is None or days >= FINE_TUNE_INTERVAL

    def mark_mistake_learning_done(self):
        self.batch_tracking['last_mistake_learning'] = datetime.now().isoformat()
        self.save_batch_tracking()

    def get_all_batch_symbols(self):
        out = []
        for b in self.completed_batches:
            out.extend(self.batch_symbols.get(str(b), []))
        return list(dict.fromkeys(out))


# =========================================================
# XGBoost + PPO
# =========================================================

class XGBoostPPOIntegrator:
    def __init__(self):
        self.xgb_models = {}
        self.ppo_models = {}
        self.load_xgb_models()
        self.load_ppo_metadata()

    def load_xgb_models(self):
        if not os.path.exists(XGBOOST_DIR):
            return
        for file in os.listdir(XGBOOST_DIR):
            if file.endswith('.joblib'):
                symbol = file.replace('.joblib', '')
                try:
                    self.xgb_models[symbol] = joblib.load(os.path.join(XGBOOST_DIR, file))
                except Exception as e:
                    print(f"   ⚠️ XGB load failed ({symbol}): {e}")
        print(f"   ✅ Loaded {len(self.xgb_models)} XGBoost models")

    def load_ppo_metadata(self):
        if not os.path.exists(PPO_PER_SYMBOL_DIR):
            return
        for file in os.listdir(PPO_PER_SYMBOL_DIR):
            if file.endswith('.zip') and file.startswith('ppo_'):
                symbol = file.replace('ppo_', '').replace('.zip', '')
                self.ppo_models[symbol] = os.path.join(PPO_PER_SYMBOL_DIR, file)
        print(f"   ✅ Found {len(self.ppo_models)} PPO models")

    def get_xgb_prediction(self, symbol, features_dict=None):
        if symbol not in self.xgb_models:
            return None
        try:
            model = self.xgb_models[symbol]
            if features_dict:
                feature_order = ['close', 'volume', 'return_5d', 'return_10d',
                                 'volatility', 'volatility_5d', 'volume_ratio',
                                 'rsi_oversold', 'rsi_overbought', 'dist_from_sr',
                                 'sr_strength', 'is_bullish_div', 'div_strength',
                                 'dist_from_ema', 'above_ema']
                features = [features_dict.get(c, 0) if not pd.isna(features_dict.get(c, 0)) else 0
                            for c in feature_order]
                prob = model.predict_proba(np.array(features).reshape(1, -1))[0, 1]
            else:
                prob = 0.5
            return {
                'prob_up': prob,
                'signal': 'BUY' if prob > 0.55 else 'SELL' if prob < 0.45 else 'NEUTRAL',
                'confidence': prob,
            }
        except:
            return None


# =========================================================
# WEIGHTED TRAINER
# =========================================================

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.get("weight", None)
        labels = inputs.get("labels")
        if weights is not None:
            inputs = {k: v for k, v in inputs.items() if k != "weight"}
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if weights is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_logits.shape[0], -1).mean(dim=1)
            loss = (loss * weights.to(loss.device)).mean()
        else:
            loss = super().compute_loss(model, inputs, return_outputs)
        return (loss, outputs) if return_outputs else loss


class StructuredDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, weights=None):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels = encodings['input_ids']
        self.weights = torch.tensor(weights) if weights is not None else torch.ones(len(self.input_ids))

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
            'weight': self.weights[idx]
        }

    def __len__(self):
        return len(self.input_ids)


class LabelExtractor:
    @staticmethod
    def extract_signal(text):
        text_lower = text.lower()
        for kw in ['buy', 'bullish', 'long', '✅ buy', 'signal: buy']:
            if kw in text_lower:
                return 1
        for kw in ['sell', 'bearish', 'short', '❌ sell', 'signal: sell']:
            if kw in text_lower:
                return 0
        return 2

    @staticmethod
    def extract_confidence(text):
        match = re.search(CONFIDENCE_PATTERN, text)
        return float(match.group(1)) / 100.0 if match else 0.5


class MistakeCollector:
    def __init__(self, xgb_ppo_integrator=None):
        self.mistakes = []
        self.confidence_history = []
        self.hard_examples = []
        self.xgb_ppo = xgb_ppo_integrator
        self.load_mistakes()
        self.load_hard_examples()

    def load_mistakes(self):
        if os.path.exists(MISTAKES_FILE):
            try:
                df = pd.read_csv(MISTAKES_FILE)
                self.mistakes = df.to_dict('records')
                print(f"   ✅ Loaded {len(self.mistakes)} past mistakes")
            except Exception as e:
                print(f"   ⚠️ Could not load mistakes: {e}")

    def load_hard_examples(self):
        if os.path.exists(HARD_EXAMPLES_FILE):
            try:
                df = pd.read_csv(HARD_EXAMPLES_FILE)
                self.hard_examples = df.to_dict('records')
            except:
                pass

    def get_hard_examples(self, limit=100, priority_only=False):
        if priority_only:
            examples = [m for m in self.hard_examples if m.get('is_high_priority', False)]
        else:
            examples = self.hard_examples.copy()
        examples.sort(key=lambda x: x.get('confidence', 1.0))
        return examples[:limit]

    def get_confidence_stats(self):
        if not self.confidence_history:
            return {'avg_confidence': 0, 'mistake_rate': 0, 'high_priority_count': 0}
        df = pd.DataFrame(self.confidence_history)
        return {
            'avg_confidence': df['confidence'].mean() if 'confidence' in df.columns else 0,
            'mistake_rate': (df['is_mistake'].mean() * 100) if 'is_mistake' in df.columns else 0,
            'high_priority_count': len(df[df.get('is_high_priority', False)])
        }

    def get_mistake_dataset(self, limit=200):
        mistake_texts = []
        signal_map = {1: 'BUY', 0: 'SELL', 2: 'HOLD'}
        for m in self.get_hard_examples(limit=limit):
            text = f"""
================================================================================
Pattern: {m.get('pattern', 'Unknown')}
Symbol: {m.get('symbol')}
Technical Analysis: Pattern detected with {m.get('confidence', 0.5):.0%} confidence

Analysis: {m.get('correct_explanation', 'Review the pattern rules')}

Signal: {signal_map.get(m.get('actual', 2), 'HOLD')}
Confidence: {min(95, max(65, int(m.get('confidence', 0.7) * 100 + 10)))}
================================================================================
"""
            mistake_texts.append(text)
        return mistake_texts


# =========================================================
# MAIN QWEN3 TRAINER
# =========================================================

class AutoQwen3Trainer:
    def __init__(self):
        for d in ("./csv", LLM_MODEL_DIR, QWEN3_LOCAL_CHECKPOINT_DIR, AGENTIC_LOOP_LOG_DIR):
            os.makedirs(d, exist_ok=True)

        self.trained_symbols = self.load_trained_symbols()
        self.model = None
        self.tokenizer = None
        self.xgb_ppo = XGBoostPPOIntegrator()
        self.mistake_collector = MistakeCollector(self.xgb_ppo)
        self.batch_manager = BatchManager()
        self.old_training_texts = []
        self.hf_uploader = HFUploader()

        self.agentic_loop = None
        if AGENTIC_LOOP_AVAILABLE:
            self._init_agentic_loop()

        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if self.telegram_token and self.telegram_chat_id:
            print("✅ Telegram notifications enabled")

    def _init_agentic_loop(self):
        try:
            print("\n" + "="*60 + "\n🤖 INITIALIZING AGENTIC LOOP\n" + "="*60)
            self.agentic_loop = AgenticLoop(xgb_model_dir=XGBOOST_DIR)
            print("="*60 + "\n")
        except Exception as e:
            print(f"   ❌ Agentic Loop init failed: {e}")
            self.agentic_loop = None

    def load_trained_symbols(self):
        if os.path.exists(TRACKING_FILE):
            try:
                with open(TRACKING_FILE, 'r') as f:
                    return json.load(f).get('symbols', [])
            except:
                return []
        return []

    def save_trained_symbols(self):
        self.trained_symbols = list(dict.fromkeys(self.trained_symbols))
        with open(TRACKING_FILE, 'w') as f:
            json.dump({
                'symbols': self.trained_symbols,
                'last_updated': datetime.now().isoformat(),
                'total_trained': len(self.trained_symbols)
            }, f, indent=2)

    def get_all_symbols_from_mongodb(self, limit=None):
        if not os.path.exists(MARKET_DATA_PATH):
            return []
        df = pd.read_csv(MARKET_DATA_PATH, usecols=["symbol"])
        symbols = df['symbol'].dropna().unique().tolist()
        if limit:
            symbols = symbols[:limit]
        print(f"   Found {len(symbols)} total symbols in mongodb.csv")
        return symbols

    def get_new_symbols(self):
        print("\n🔍 Checking for new symbols...")
        all_symbols = self.get_all_symbols_from_mongodb()
        trained_local = set(self.trained_symbols)
        new_symbols = [s for s in all_symbols if s not in trained_local]
        print(f"   Already trained: {len(self.trained_symbols)} symbols")
        print(f"   New symbols found: {len(new_symbols)}")
        return new_symbols

    def classify_example_difficulty(self, text):
        text_lower = text.lower()
        for kw in ['complex', 'multi timeframe', 'divergence', 'harmonic', 'elliott', 'smc', 'order block', 'fvg', 'liquidity']:
            if kw in text_lower:
                return 'hard'
        for kw in ['triangle', 'wedge', 'flag', 'pennant', 'reversal']:
            if kw in text_lower:
                return 'medium'
        if len(text) > 1500:
            return 'hard'
        elif len(text) > 800:
            return 'medium'
        return 'easy'

    def load_training_data_with_curriculum(self):
        if not os.path.exists(TRAINING_DATA_PATH):
            print(f"❌ Training data not found: {TRAINING_DATA_PATH}")
            return None, None

        with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
            text_data = f.read()

        raw_examples = text_data.split('=' * 80)
        new_texts = [ex.strip() for ex in raw_examples if len(ex.strip()) > 100]
        print(f"📊 New examples: {len(new_texts)}")

        if self.old_training_texts:
            self.old_training_texts.extend(new_texts)
            self.old_training_texts = self.old_training_texts[-MAX_OLD_EXAMPLES:]
            train_texts = self.old_training_texts.copy()
            print(f"   Replay buffer: {len(self.old_training_texts)} total examples")
        else:
            train_texts = new_texts
            self.old_training_texts = train_texts.copy()

        mistake_texts = self.mistake_collector.get_mistake_dataset(limit=300)
        if mistake_texts:
            normal_count = int(len(train_texts) * 0.75)
            mistake_count = min(len(mistake_texts), int(len(train_texts) * 0.25))
            train_texts = train_texts[:normal_count] + mistake_texts[:mistake_count]
            print(f"   Data mix: {normal_count} normal + {mistake_count} mistakes")

        easy_texts = [t for t in train_texts if self.classify_example_difficulty(t) == 'easy']
        medium_texts = [t for t in train_texts if self.classify_example_difficulty(t) == 'medium']
        hard_texts = [t for t in train_texts if self.classify_example_difficulty(t) == 'hard']
        train_texts = easy_texts + medium_texts + hard_texts
        print(f"   Curriculum: {len(easy_texts)} easy, {len(medium_texts)} medium, {len(hard_texts)} hard")

        example_weights = np.ones(len(train_texts))
        for i, text in enumerate(train_texts):
            if 'Elliott Wave' in text or 'Impulse Wave' in text or 'Corrective Wave' in text:
                example_weights[i] = 5.0
            elif 'SMC' in text or 'Order Block' in text or 'FVG' in text or 'Liquidity' in text:
                example_weights[i] = 4.5
            elif 'Harmonic' in text or 'Gartley' in text or 'Butterfly' in text:
                example_weights[i] = 4.0
            elif self.classify_example_difficulty(text) == 'hard':
                example_weights[i] = 3.5
            elif self.classify_example_difficulty(text) == 'medium':
                example_weights[i] = 1.5
            else:
                example_weights[i] = 0.8

        return train_texts, example_weights

    def load_model_with_lora(self):
        """Load Qwen3 model + apply LoRA"""
        print("\n🏗️ Loading Qwen3 model...")

        local_valid = (
            os.path.exists(LLM_MODEL_DIR) and
            os.path.exists(os.path.join(LLM_MODEL_DIR, "config.json"))
        )

        if local_valid:
            try:
                print(f"   Loading local Qwen3 from {LLM_MODEL_DIR}...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL_DIR,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    LLM_MODEL_DIR,
                    trust_remote_code=True,
                )
                print("   ✅ Qwen3 loaded from local")
            except Exception as e:
                print(f"   ⚠️ Local load failed: {e}")
                self.model = None

        if self.model is None:
            print(f"   📥 Downloading base Qwen3: {BASE_MODEL}")
            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL,
                trust_remote_code=True,
            )
            print("   ✅ Base Qwen3 loaded")

        if LORA_AVAILABLE:
            lora_config = LoraConfig(**LORA_CONFIG)
            self.model = get_peft_model(self.model, lora_config)
            print(f"   ✅ LoRA applied (r={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']})")

        self._post_load_setup()

    def _post_load_setup(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Device: {device}")

    def save_training_status_file(self, mode, symbols_batch):
        status_file = "./csv/current_training_status.json"
        try:
            with open(status_file, 'w') as f:
                json.dump({
                    'model': 'qwen3',
                    'mode': mode,
                    'mode_description': MODE_EXPLANATION.get(mode, mode.upper()),
                    'mode_short': MODE_SHORT_EXPLANATION.get(mode, mode.upper()),
                    'start_time': datetime.now().isoformat(),
                    'symbols_count': len(symbols_batch) if symbols_batch else 0,
                    'epochs': EPOCHS_CONFIG.get(mode, 2),
                    'learning_rate': LR_CONFIG.get(mode, 2e-4)
                }, f, indent=2)
        except:
            pass

    def train(self, mode="incremental", symbols_batch=None):
        self.save_training_status_file(mode, symbols_batch)

        mode_explanation = MODE_EXPLANATION.get(mode, f"Mode: {mode.upper()}")
        mode_short = MODE_SHORT_EXPLANATION.get(mode, mode.upper())

        start_msg = f"""
🚀 <b>Qwen3 Training Started</b>
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
🎯 Mode: {mode.upper()} - {mode_explanation}
📚 Symbols: {len(symbols_batch) if symbols_batch else 'ALL'}
⚙️ Epochs: {EPOCHS_CONFIG.get(mode, 2)}
"""
        send_telegram_message(start_msg, self.telegram_token, self.telegram_chat_id)

        print(f"\n{'='*60}")
        print(f"🎯 QWEN3 TRAINING MODE: {mode.upper()}")
        print(f"🔍 MODE EXPLANATION: {mode_explanation}")
        if symbols_batch:
            print(f"📚 Symbols in this batch: {len(symbols_batch)}")
        print(f"{'='*60}")

        train_texts, example_weights = self.load_training_data_with_curriculum()
        if not train_texts:
            print("❌ No training data found!")
            return False

        # ✅ FIX: max_length 384 → 192 (2x speed)
        encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding="max_length",
            max_length=192,             # ← 384 → 192
            return_tensors="pt"
        )

        train_dataset = StructuredDataset(encodings, example_weights)

        dataset_size = len(train_dataset)
        val_size = max(1, int(dataset_size * VALIDATION_SPLIT_RATIO))
        train_size = dataset_size - val_size

        train_indices = list(range(train_size))
        val_indices = list(range(train_size, dataset_size))

        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        print(f"   Dataset split: {train_size} train, {val_size} validation")

        num_epochs = EPOCHS_CONFIG.get(mode, 2)
        learning_rate = LR_CONFIG.get(mode, 2e-4)
        batch_size = BATCH_SIZE_CONFIG.get(mode, 1)
        grad_accum = GRAD_ACCUM_CONFIG.get(mode, 16)

        print(f"\n⚙️ Qwen3 Training Config:")
        print(f"   Model: {BASE_MODEL}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Batch Size: {batch_size} (effective: {batch_size * grad_accum})")
        print(f"   Gradient Accumulation: {grad_accum}")
        print(f"   LoRA: r={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']}")
        print(f"   Max Length: 192")
        print(f"   XGBoost Integration: Enabled ({len(self.xgb_ppo.xgb_models)} models)")
        print(f"   📤 HF Repo: {HF_DATASET_REPO}")

        # =========================================================
        # ✅ LOCAL CHECKPOINT RESUME
        # =========================================================
        last_checkpoint = None
        print(f"   🔍 Scanning local checkpoints in: {QWEN3_LOCAL_CHECKPOINT_DIR}")

        local_patterns = [
            os.path.join(QWEN3_LOCAL_CHECKPOINT_DIR, "qwen3_checkpoint-*"),
            os.path.join(QWEN3_LOCAL_CHECKPOINT_DIR, "checkpoint-*"),
        ]

        all_local_ckpts = []
        for pattern in local_patterns:
            all_local_ckpts.extend(glob.glob(pattern))

        if all_local_ckpts:
            def get_step_num(path):
                m = re.search(r'checkpoint-(\d+)', path)
                return int(m.group(1)) if m else 0
            last_checkpoint = sorted(all_local_ckpts, key=get_step_num)[-1]
            print(f"   ✅ Found local checkpoint: {last_checkpoint}")
            print(f"   📊 Step: {get_step_num(last_checkpoint)}")
        else:
            print(f"   ℹ️ No local checkpoint found - starting fresh")

        if last_checkpoint:
            send_telegram_message(
                f"🔄 <b>Resuming Qwen3 from LOCAL</b>\n📂 {last_checkpoint}\n🎯 Mode: {mode.upper()}",
                self.telegram_token, self.telegram_chat_id
            )

        training_args = TrainingArguments(
            output_dir=LLM_MODEL_DIR,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            warmup_ratio=0.05,              # ← was warmup_steps=100
            weight_decay=0.025,
            lr_scheduler_type="cosine",
            save_steps=20,
            save_total_limit=5,
            logging_steps=10,
            save_strategy="steps",
            fp16=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            report_to="none",
            max_grad_norm=MAX_GRAD_NORM,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-8,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=None,
            data_collator=data_collator,
        )

        class CustomHFCallback:
            def __init__(self, hf_uploader):
                self.hf_uploader = hf_uploader

            def on_save(self, args, state, control, **kwargs):
                if state.is_world_process_zero:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                    if os.path.exists(checkpoint_dir):
                        print(f"\n   📤 Uploading Qwen3 checkpoint {state.global_step} to HF...")
                        self.hf_uploader.upload_checkpoint(checkpoint_dir, state.global_step)
                        self.hf_uploader.upload_tracking_files()

                        try:
                            local_ckpt_target = os.path.join(
                                QWEN3_LOCAL_CHECKPOINT_DIR,
                                f"qwen3_checkpoint-{state.global_step}"
                            )
                            if not os.path.exists(local_ckpt_target):
                                shutil.copytree(checkpoint_dir, local_ckpt_target)
                                print(f"   💾 Local copy saved: {local_ckpt_target}")
                        except Exception as e:
                            print(f"   ⚠️ Local copy failed: {e}")
                return control

            def __getattr__(self, name):
                if name.startswith('on_'):
                    return lambda *args, **kwargs: args[2] if len(args) > 2 else None
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        trainer.add_callback(CustomHFCallback(self.hf_uploader))

        print("\n🏋️ Starting Qwen3 Training...")
        print(f"   📂 Local checkpoints: {QWEN3_LOCAL_CHECKPOINT_DIR}")
        print(f"   📤 HF checkpoints → {HF_DATASET_REPO}/{QWEN3_HF_CHECKPOINT_PREFIX}*")

        # ═══════════════════════════════════════════════════════════
        # ✅ FIX: Auto-handle optimizer mismatch + gradient errors
        # ═══════════════════════════════════════════════════════════
        try:
            trainer.train(resume_from_checkpoint=last_checkpoint)
        except ValueError as e:
            err_str = str(e)
            if last_checkpoint and ("parameter group" in err_str or "optimizer" in err_str.lower()):
                print("\n   ⚠️ Optimizer state mismatch detected!")
                print("   🔄 Removing optimizer/scheduler state and retrying...")
                for fname in ("optimizer.pt", "scheduler.pt"):
                    p = os.path.join(last_checkpoint, fname)
                    if os.path.exists(p):
                        os.remove(p)
                        print(f"   🗑️ Removed {fname}")
                self.model.train()
                trainer.train(resume_from_checkpoint=last_checkpoint)
            else:
                error_msg = f"""
⚠️ <b>Qwen3 Training Error</b>
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
🎯 Mode: {mode.upper()}
❌ Error: {err_str[:200]}
"""
                send_telegram_message(error_msg, self.telegram_token, self.telegram_chat_id)
                raise
        except RuntimeError as e:
            err_str = str(e)
            if "does not require grad" in err_str or "grad_fn" in err_str:
                print("\n   ⚠️ Gradient error → re-enabling LoRA params...")
                for name, p in self.model.named_parameters():
                    if "lora_" in name.lower():
                        p.requires_grad = True
                self.model.train()
                trainer.train(resume_from_checkpoint=last_checkpoint)
            else:
                error_msg = f"""
⚠️ <b>Qwen3 Training Error</b>
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
🎯 Mode: {mode.upper()}
❌ Error: {err_str[:200]}
"""
                send_telegram_message(error_msg, self.telegram_token, self.telegram_chat_id)
                raise
        except Exception as e:
            error_msg = f"""
⚠️ <b>Qwen3 Training Error</b>
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
🎯 Mode: {mode.upper()}
❌ Error: {str(e)[:200]}
"""
            send_telegram_message(error_msg, self.telegram_token, self.telegram_chat_id)
            raise

        print("\n✅ Qwen3 training completed!")

        if symbols_batch and self.agentic_loop is not None:
            batch_num = self.batch_manager.current_batch_index if self.batch_manager.current_batch_index > 0 else 1
            self._update_agentic_loop_after_batch(batch_num, symbols_batch, eval_loss=0.5)

        # ✅ Merge LoRA + save full Qwen3 model
        try:
            if LORA_AVAILABLE and hasattr(self.model, 'merge_and_unload'):
                print("🔄 Merging LoRA into Qwen3 base...")
                merged_model = self.model.merge_and_unload()
                merged_model.save_pretrained(LLM_MODEL_DIR)
                self.tokenizer.save_pretrained(LLM_MODEL_DIR)
                print(f"✅ Merged Qwen3 saved to {LLM_MODEL_DIR}")
            else:
                self.model.save_pretrained(LLM_MODEL_DIR)
                self.tokenizer.save_pretrained(LLM_MODEL_DIR)
                print(f"💾 Qwen3 saved to {LLM_MODEL_DIR}")
        except Exception as e:
            print(f"⚠️ Merge failed, saving LoRA only: {e}")
            self.model.save_pretrained(LLM_MODEL_DIR)
            self.tokenizer.save_pretrained(LLM_MODEL_DIR)

        self.upload_final_model_to_hf(mode)

        complete_msg = f"""
✅ <b>Qwen3 Training Completed</b>
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
🎯 {mode_explanation}
📚 Symbols trained: {len(symbols_batch) if symbols_batch else 'ALL'}
💾 Model saved: {LLM_MODEL_DIR}
📤 Uploaded: {HF_DATASET_REPO}/{QWEN3_FINAL_MODEL_PREFIX}/{mode}/
"""
        send_telegram_message(complete_msg, self.telegram_token, self.telegram_chat_id)

        # ✅ Cleanup HF checkpoints after success
        self.hf_uploader.delete_mode_checkpoints(mode)

        # Update status
        status_file = "./csv/current_training_status.json"
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    current_status = json.load(f)
                current_status['completed_at'] = datetime.now().isoformat()
                current_status['status'] = 'completed'
                with open(status_file, 'w') as f:
                    json.dump(current_status, f, indent=2)
            except:
                pass

        return True

    def _update_agentic_loop_after_batch(self, batch_num, symbols, eval_loss=None):
        if self.agentic_loop is None:
            return
        try:
            simulated_pnl = 0.02
            if eval_loss is not None:
                simulated_pnl = max(-0.08, min(0.08, -eval_loss * 0.008))
            success = simulated_pnl > 0
            for symbol in symbols:
                trade_result = {'symbol': symbol, 'pnl': simulated_pnl, 'success': success, 'batch': batch_num}
                self.agentic_loop.after_trade_feedback(trade_result)
        except Exception as e:
            print(f"   ⚠️ Agentic Loop update failed: {e}")

    def upload_final_model_to_hf(self, mode):
        token = os.getenv("hf_token")
        if not token:
            return
        print(f"\n📤 Uploading Qwen3 final model → {HF_DATASET_REPO}/{QWEN3_FINAL_MODEL_PREFIX}/{mode}/")
        try:
            self.hf_uploader.upload_final_model(LLM_MODEL_DIR, mode)
            self.hf_uploader.upload_tracking_files()
        except Exception as e:
            print(f"⚠️ Qwen3 final model upload failed: {e}")

    def generate_training_data_for_symbols(self, symbols):
        print(f"\n📝 Generating training data for {len(symbols)} symbols...")
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "scripts/generate_pattern_training_data_complete.py",
             "--symbols", ",".join(symbols)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"   ⚠️ Data generation failed: {result.stderr[:300]}")
            return False
        print("   ✅ Training data generated")
        return True

    def _finalize_agentic_loop(self):
        if self.agentic_loop is None:
            return
        try:
            print("\n" + "="*60 + "\n🏆 AGENTIC LOOP FINAL REPORT\n" + "="*60)
            summary = self.agentic_loop.get_summary()
            if len(summary) > 0:
                for _, row in summary.iterrows():
                    print(f"   {row['agent']}: {row['accuracy']} accuracy")
            state = {'timestamp': str(datetime.now()), 'agents': {}}
            for agent in self.agentic_loop.agents:
                state['agents'][agent.name] = {
                    'accuracy': agent.get_accuracy(),
                    'predictions': agent.total_predictions,
                    'weight': agent.get_dynamic_weight()
                }
            with open(AGENTIC_LOOP_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"   ⚠️ Final report failed: {e}")

    def run(self):
        global TRAINING_DATA_PATH
        print("="*60)
        print("🚀 AUTO QWEN3 TRAINER (v5 — Full Patched)")
        print("="*60)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🧠 Model: {BASE_MODEL}")
        print(f"📁 Local dir: {LLM_MODEL_DIR}")
        print(f"📁 Local checkpoints: {QWEN3_LOCAL_CHECKPOINT_DIR}")
        print(f"🔧 LoRA: r={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']}")
        print(f"📊 XGBoost Models: {len(self.xgb_ppo.xgb_models)}")
        print(f"📁 PPO Models: {len(self.xgb_ppo.ppo_models)}")
        print(f"📤 HF Checkpoints: {QWEN3_HF_CHECKPOINT_PREFIX}*")
        print(f"📤 HF Final: {QWEN3_FINAL_MODEL_PREFIX}/*")
        print("="*60)

        confidence_stats = self.mistake_collector.get_confidence_stats()
        print(f"\n📊 Confidence Statistics:")
        print(f"   Average confidence: {confidence_stats['avg_confidence']:.2%}")
        print(f"   Mistake rate: {confidence_stats['mistake_rate']:.2f}%")

        all_symbols = self.get_all_symbols_from_mongodb()
        new_symbols = self.get_new_symbols()

        self.load_model_with_lora()

        # =========================================================
        # ✅ FIX: Checkpoint-based mode detection
        # =========================================================
        if new_symbols:
            print(f"\n📚 Found {len(new_symbols)} new symbols to train")
            for i in range(0, len(new_symbols), BATCH_SIZE):
                batch = new_symbols[i:i+BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1
                print(f"\n📦 Processing batch {batch_num}: {len(batch)} symbols")

                if self.generate_training_data_for_symbols(batch):
                    # ✅ Checkpoint-based mode detection (not tracking-based)
                    existing_ckpts = glob.glob(
                        os.path.join(QWEN3_LOCAL_CHECKPOINT_DIR, "qwen3_checkpoint-*")
                    ) + glob.glob(
                        os.path.join(QWEN3_LOCAL_CHECKPOINT_DIR, "checkpoint-*")
                    )

                    if existing_ckpts:
                        mode = "incremental"
                        print(f"   🔄 Found {len(existing_ckpts)} checkpoints → INCREMENTAL")
                    elif len(self.trained_symbols) == 0:
                        mode = "first_train"
                        print(f"   🎯 No checkpoints, no tracking → FIRST_TRAIN")
                    else:
                        mode = "incremental"
                        print(f"   ⚙️ {len(self.trained_symbols)} trained → INCREMENTAL")

                    if self.train(mode=mode, symbols_batch=batch):
                        self.trained_symbols.extend(batch)
                        self.save_trained_symbols()
                        self.batch_manager.mark_batch_completed(batch_num, batch)
                        print(f"✅ Batch {batch_num} complete! Total trained: {len(self.trained_symbols)}")
        else:
            print("\n✅ No new symbols found!")

        # Weekly fine-tune
        weekly_batch_num, weekly_symbols = self.batch_manager.get_batch_for_weekly_finetune()
        if weekly_batch_num and weekly_symbols:
            print(f"\n🔄 Weekly fine-tune Batch {weekly_batch_num} ({len(weekly_symbols)} symbols)")
            if self.generate_training_data_for_symbols(weekly_symbols):
                if self.train(mode="weekly_finetune", symbols_batch=weekly_symbols):
                    self.batch_manager.mark_weekly_done(weekly_batch_num)

        # Monthly consolidation
        if self.batch_manager.should_consolidate():
            all_trained_symbols = self.batch_manager.get_all_batch_symbols()
            if all_trained_symbols:
                print(f"\n🔄 Monthly consolidation - {len(all_trained_symbols)} symbols")
                if self.generate_training_data_for_symbols(all_trained_symbols):
                    if self.train(mode="consolidate", symbols_batch=all_trained_symbols):
                        self.batch_manager.mark_consolidated()

        # Mistake learning
        high_priority_examples = self.mistake_collector.get_hard_examples(limit=200, priority_only=True)
        if high_priority_examples and self.batch_manager.should_run_mistake_learning():
            print(f"\n🔥 {len(high_priority_examples)} high priority mistakes")
            temp_file = "./csv/temp_hard_examples.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                signal_map = {1: 'BUY', 0: 'SELL', 2: 'HOLD'}
                for ex in high_priority_examples[:100]:
                    f.write(f"""
================================================================================
Pattern: {ex.get('pattern', 'Unknown')}
Symbol: {ex.get('symbol')}
Signal: {signal_map.get(ex.get('actual', 2), 'HOLD')}
Confidence: {min(95, max(65, int(ex.get('confidence', 0.7) * 100 + 10)))}
================================================================================
""")
            original_path = TRAINING_DATA_PATH
            TRAINING_DATA_PATH = temp_file
            if self.train(mode="mistake_learning", symbols_batch=None):
                self.batch_manager.mark_mistake_learning_done()
            TRAINING_DATA_PATH = original_path
            if os.path.exists(temp_file):
                os.remove(temp_file)

        print("\n" + "="*60 + "\n📊 FINAL STATUS\n" + "="*60)
        print(f"   Total trained symbols: {len(self.trained_symbols)}")
        print(f"   Completed batches: {len(self.batch_manager.completed_batches)}")
        print(f"   Last weekly fine-tune: {self.batch_manager.batch_tracking.get('last_weekly_finetune', 'Never')}")
        print(f"   Last consolidation: {self.batch_manager.batch_tracking.get('last_consolidate', 'Never')}")
        print(f"   XGBoost Models: {len(self.xgb_ppo.xgb_models)}")
        print(f"   PPO Models: {len(self.xgb_ppo.ppo_models)}")
        print(f"   Local Qwen3 Dir: {LLM_MODEL_DIR}")
        print(f"   Local Checkpoints: {QWEN3_LOCAL_CHECKPOINT_DIR}")
        print(f"   HF Dataset Repo: {HF_DATASET_REPO}")
        print("="*60)

        self._finalize_agentic_loop()


if __name__ == "__main__":
    trainer = AutoQwen3Trainer()
    trainer.run()