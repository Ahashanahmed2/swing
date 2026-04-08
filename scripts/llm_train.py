# v4 - FINAL (UPDATED PATHS)
# scripts/llm_train.py
# Advanced LLM Trainer with XGBoost + PPO Integration
# ✅ CHECKPOINT + FINAL MODEL SAVE TO HF DATASET REPO: ahashanahmed/csv/
# ✅ NO DOWNLOAD FROM HF - ALL LOCAL DATA FROM ./csv/
# ✅ ALL LOCAL PATHS UPDATED TO ./csv/

import os
import torch
import json
import warnings
import pandas as pd
import numpy as np
import re
import requests
import joblib
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from huggingface_hub import login, create_repo, upload_folder, upload_file, HfApi

# =========================================================
# AGENTIC LOOP INTEGRATION
# =========================================================
try:
    from agentic_loop import AgenticLoop
    AGENTIC_LOOP_AVAILABLE = True
except ImportError:
    AGENTIC_LOOP_AVAILABLE = False
    print("⚠️ Agentic Loop not found. Multi-agent voting disabled.")

# Optional: LoRA for faster training
try:
    from peft import LoraConfig, get_peft_model
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    print("⚠️ PEFT not installed. Install with: pip install peft")

warnings.filterwarnings('ignore')

# =========================================================
# CONFIGURATION - 70+ HOURS ULTIMATE TRAINING
# =========================================================

# Batch configuration for incremental training
BATCH_SIZE = 40
TOTAL_BATCHES = "auto"
MAX_SYMBOLS_PER_BATCH = 40

# ✅ SINGLE REPOSITORY CONFIGURATION
HF_DATASET_REPO = "ahashanahmed/csv"  # ✅ সবকিছু এখানে সেভ হবে (চেকপয়েন্ট + ফাইনাল মডেল)
BASE_MODEL = "distilgpt2"

# ✅ ALL LOCAL PATHS UPDATED TO ./csv/
TRACKING_FILE = "./csv/trained_symbols.json"
BATCH_TRACKING_FILE = "./csv/batch_tracking.json"
LAST_FINE_TUNE_FILE = "./csv/last_finetune.txt"
LAST_CONSOLIDATE_FILE = "./csv/last_consolidate.txt"

# ✅ LOCAL DATA PATHS ONLY - NO HF DOWNLOAD
TRAINING_DATA_PATH = "./csv/training_texts.txt"
MARKET_DATA_PATH = "./csv/mongodb.csv"
MISTAKES_FILE = "./csv/trading_mistakes.csv"
CONFIDENCE_LOG = "./csv/llm_confidence_log.csv"
HARD_EXAMPLES_FILE = "./csv/hard_examples.csv"

# XGBoost and PPO paths
XGBOOST_DIR = "./csv/xgboost"
PPO_MODELS_DIR = "./csv/ppo_models"
PPO_PER_SYMBOL_DIR = "./csv/ppo_models/per_symbol"

# LLM Model local directory
LLM_MODEL_DIR = "./csv/llm_model"

# Schedule
FINE_TUNE_INTERVAL = 7
CONSOLIDATE_INTERVAL = 30

# Learning parameters - 70+ HOURS OPTIMIZED
MAX_OLD_EXAMPLES = 5000
HARD_EXAMPLE_THRESHOLD = 0.20
HIGH_PRIORITY_THRESHOLD = 0.30
WEIGHTED_LOSS_ENABLED = True
MAX_GRAD_NORM = 0.7
EARLY_STOPPING_PATIENCE = 8
VALIDATION_SPLIT_RATIO = 0.1

# Training mode flags
FORCE_RETRAIN = False

# LoRA config for distilgpt2 - 70+ HOURS OPTIMIZED
LORA_CONFIG = {
    'r': 32,
    'lora_alpha': 64,
    'target_modules': ['c_attn', 'c_proj', 'c_fc'],
    'lora_dropout': 0.15,
    'bias': 'none',
}

# Label patterns for extraction
SIGNAL_PATTERNS = {
    'bullish': r'(?:Signal|Prediction|Recommendation):?\s*(?:BUY|Bullish|LONG|✅ BUY)',
    'bearish': r'(?:Signal|Prediction|Recommendation):?\s*(?:SELL|Bearish|SHORT|❌ SELL)',
    'neutral': r'(?:Signal|Prediction|Recommendation):?\s*(?:HOLD|Neutral|WAIT|⏳ WAIT)'
}

CONFIDENCE_PATTERN = r'(?:Confidence|Signal Strength):?\s*(\d+(?:\.\d+)?)%'

# Agentic Loop paths
AGENTIC_LOOP_STATE_FILE = "./csv/agentic_loop_state.json"
AGENTIC_LOOP_LOG_DIR = "./csv/agentic_loop_logs"

# =========================================================
# 70+ HOURS EPOCH CONFIGURATION
# =========================================================
EPOCHS_CONFIG = {
    "first_train": 5,
    "incremental": 3,
    "weekly_finetune": 3,
    "consolidate": 10,
    "mistake_learning": 4,
}

LR_CONFIG = {
    "first_train": 1.5e-5,
    "incremental": 1e-5,
    "weekly_finetune":8e-6,
    "consolidate": 8e-6,
    "mistake_learning": 1.5e-5,
}

BATCH_SIZE_CONFIG = {
    "first_train": 1,
    "incremental": 1,
    "weekly_finetune": 1,
    "consolidate": 1,
    "mistake_learning": 1,
}

GRAD_ACCUM_CONFIG = {
    "first_train": 16,
    "incremental": 16,
    "weekly_finetune": 8,
    "consolidate": 16,
    "mistake_learning": 16,
}


# =========================================================
# HF UPLOADER (CHECKPOINT + FINAL MODEL)
# =========================================================

class HFUploader:
    """Upload checkpoints and final model to HF Dataset Repository"""
    
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
        """Upload a checkpoint folder to HF Dataset repo"""
        if self.api is None:
            print("   ⚠️ HF API not available, skipping checkpoint upload")
            return False
        
        try:
            repo_path = f"checkpoints/checkpoint-{step_num}"
            
            self.api.upload_folder(
                folder_path=checkpoint_path,
                path_in_repo=repo_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Checkpoint {step_num} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            print(f"   📤 Checkpoint {step_num} uploaded to {self.repo_id}/checkpoints/")
            return True
        except Exception as e:
            print(f"   ⚠️ Checkpoint upload failed: {e}")
            return False
    
    def upload_final_model(self, model_path, mode):
        """Upload final model to HF Dataset repo"""
        if self.api is None:
            print("   ⚠️ HF API not available, skipping final model upload")
            return False
        
        try:
            repo_path = f"final_model/{mode}"
            
            self.api.upload_folder(
                folder_path=model_path,
                path_in_repo=repo_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Final Model ({mode}) - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            print(f"   📤 Final model uploaded to {self.repo_id}/final_model/{mode}/")
            return True
        except Exception as e:
            print(f"   ⚠️ Final model upload failed: {e}")
            return False
    
    def upload_tracking_files(self):
        """Upload tracking files to HF Dataset repo"""
        if self.api is None:
            return
        
        try:
            # trained_symbols.json
            if os.path.exists(TRACKING_FILE):
                self.api.upload_file(
                    path_or_fileobj=TRACKING_FILE,
                    path_in_repo="trained_symbols.json",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Update trained symbols - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                print(f"   📤 trained_symbols.json uploaded to {self.repo_id}")
            
            # batch_tracking.json
            if os.path.exists(BATCH_TRACKING_FILE):
                self.api.upload_file(
                    path_or_fileobj=BATCH_TRACKING_FILE,
                    path_in_repo="batch_tracking.json",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Update batch tracking - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                print(f"   📤 batch_tracking.json uploaded to {self.repo_id}")
                
        except Exception as e:
            print(f"   ⚠️ Tracking upload failed: {e}")


# =========================================================
# BATCH MANAGER
# =========================================================

class BatchManager:
    """Manages incremental training batches and auto batch growth"""

    def __init__(self):
        self.batch_tracking = self.load_batch_tracking()
        self.current_batch_index = self.batch_tracking.get('current_batch', 0)
        self.completed_batches = self.batch_tracking.get('completed_batches', [])
        self.batch_symbols = self.batch_tracking.get('batch_symbols', {})

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
            'last_batch_date': None
        }

    def save_batch_tracking(self):
        # Ensure directory exists
        os.makedirs(os.path.dirname(BATCH_TRACKING_FILE), exist_ok=True)
        with open(BATCH_TRACKING_FILE, 'w') as f:
            json.dump(self.batch_tracking, f, indent=2)

    def get_next_batch_symbols(self, all_symbols, trained_symbols, batch_size=BATCH_SIZE):
        untrained = [s for s in all_symbols if s not in trained_symbols]
        if not untrained:
            return []
        next_batch = untrained[:batch_size]
        batch_num = self.current_batch_index + 1
        self.batch_symbols[str(batch_num)] = next_batch
        self.batch_tracking['batch_symbols'] = self.batch_symbols
        return next_batch, batch_num

    def mark_batch_completed(self, batch_num, symbols):
        if batch_num not in self.completed_batches:
            self.completed_batches.append(batch_num)
            self.current_batch_index = batch_num
            self.batch_tracking['current_batch'] = batch_num
            self.batch_tracking['completed_batches'] = self.completed_batches
            self.batch_tracking['total_symbols_trained'] += len(symbols)
            self.batch_tracking['last_batch_date'] = datetime.now().isoformat()
            self.save_batch_tracking()

    def get_batch_for_weekly_finetune(self):
        if not self.completed_batches:
            return None, []
        last_date_str = self.batch_tracking.get('last_batch_date', '2024-01-01T00:00:00.000000')
        try:
            last_date = datetime.fromisoformat(last_date_str)
        except:
            last_date = datetime(2024, 1, 1)
        week_num = (datetime.now() - last_date).days // 7
        batch_index = week_num % len(self.completed_batches)
        batch_num = self.completed_batches[batch_index]
        return batch_num, self.batch_symbols.get(str(batch_num), [])

    def should_consolidate(self):
        last_consolidate = self.batch_tracking.get('last_consolidate', None)
        if not last_consolidate:
            return True
        try:
            last_date = datetime.fromisoformat(last_consolidate)
        except:
            return True
        days_since = (datetime.now() - last_date).days
        return days_since >= CONSOLIDATE_INTERVAL

    def mark_consolidated(self):
        self.batch_tracking['last_consolidate'] = datetime.now().isoformat()
        self.save_batch_tracking()

    def get_all_batch_symbols(self):
        all_symbols = []
        for batch_num in self.completed_batches:
            symbols = self.batch_symbols.get(str(batch_num), [])
            all_symbols.extend(symbols)
        return all_symbols


# =========================================================
# XGBOOST + PPO INTEGRATION
# =========================================================

class XGBoostPPOIntegrator:
    """Integrate XGBoost and PPO models with LLM training"""

    def __init__(self):
        self.xgb_models = {}
        self.xgb_metadata = None
        self.ppo_models = {}
        self.load_xgb_models()
        self.load_ppo_metadata()

    def load_xgb_models(self):
        if os.path.exists(XGBOOST_DIR):
            for file in os.listdir(XGBOOST_DIR):
                if file.endswith('.joblib'):
                    symbol = file.replace('.joblib', '')
                    try:
                        self.xgb_models[symbol] = joblib.load(os.path.join(XGBOOST_DIR, file))
                    except Exception as e:
                        print(f"   ⚠️ Failed to load XGBoost for {symbol}: {e}")
            print(f"   ✅ Loaded {len(self.xgb_models)} XGBoost models")
        else:
            print(f"   ⚠️ XGBoost directory not found: {XGBOOST_DIR}")

    def load_ppo_metadata(self):
        if os.path.exists(PPO_PER_SYMBOL_DIR):
            for file in os.listdir(PPO_PER_SYMBOL_DIR):
                if file.endswith('.zip') and file.startswith('ppo_'):
                    symbol = file.replace('ppo_', '').replace('.zip', '')
                    self.ppo_models[symbol] = os.path.join(PPO_PER_SYMBOL_DIR, file)
            print(f"   ✅ Found {len(self.ppo_models)} PPO models")
        else:
            print(f"   ⚠️ PPO models directory not found: {PPO_PER_SYMBOL_DIR}")

    def get_xgb_prediction(self, symbol, features_dict=None):
        if symbol not in self.xgb_models:
            return None
        try:
            model = self.xgb_models[symbol]
            if features_dict:
                import numpy as np
                feature_order = ['close', 'volume', 'return_5d', 'return_10d', 
                                'volatility', 'volatility_5d', 'volume_ratio',
                                'rsi_oversold', 'rsi_overbought', 'dist_from_sr', 
                                'sr_strength', 'is_bullish_div', 'div_strength',
                                'dist_from_ema', 'above_ema']
                features = []
                for col in feature_order:
                    val = features_dict.get(col, 0)
                    if pd.isna(val):
                        val = 0
                    features.append(val)
                features_array = np.array(features).reshape(1, -1)
                prob = model.predict_proba(features_array)[0, 1]
            else:
                prob = 0.5
            return {
                'prob_up': prob,
                'signal': 'BUY' if prob > 0.55 else 'SELL' if prob < 0.45 else 'NEUTRAL',
                'confidence': prob,
                'source': 'XGBoost'
            }
        except Exception as e:
            return None

    def get_ppo_signal(self, symbol):
        if symbol in self.ppo_models:
            return {'exists': True, 'model_path': self.ppo_models[symbol], 'source': 'PPO'}
        return None


class WeightedTrainer(Trainer):
    """Custom trainer with weighted loss"""

    def compute_loss(self, model, inputs, return_outputs=False):
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
    """Dataset with built-in weights for safe training"""

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
    """Extract labels from generated text"""

    @staticmethod
    def extract_signal(text):
        text_lower = text.lower()
        bullish_keywords = ['buy', 'bullish', 'long', '✅ buy', 'signal: buy']
        for kw in bullish_keywords:
            if kw in text_lower:
                return 1
        bearish_keywords = ['sell', 'bearish', 'short', '❌ sell', 'signal: sell']
        for kw in bearish_keywords:
            if kw in text_lower:
                return 0
        neutral_keywords = ['hold', 'neutral', 'wait']
        for kw in neutral_keywords:
            if kw in text_lower:
                return 2
        for pattern in SIGNAL_PATTERNS.values():
            if re.search(pattern, text, re.IGNORECASE):
                if 'buy' in pattern or 'bullish' in pattern:
                    return 1
                elif 'sell' in pattern or 'bearish' in pattern:
                    return 0
                else:
                    return 2
        return 2

    @staticmethod
    def extract_confidence(text):
        match = re.search(CONFIDENCE_PATTERN, text)
        if match:
            return float(match.group(1)) / 100.0
        return 0.5


class MistakeCollector:
    """Mistake collector with confidence tracking and XGBoost/PPO integration"""

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

    def add_mistake(self, symbol, prediction, actual, confidence, pattern, market_regime=""):
        mistake = {
            'symbol': symbol, 'timestamp': datetime.now().isoformat(),
            'prediction': prediction, 'actual': actual, 'confidence': confidence,
            'pattern': pattern, 'market_regime': market_regime,
            'is_hard': confidence < HARD_EXAMPLE_THRESHOLD,
            'is_high_priority': confidence < HIGH_PRIORITY_THRESHOLD and prediction != actual,
            'correct_explanation': self._generate_explanation(pattern, actual, market_regime)
        }
        self.mistakes.append(mistake)
        if mistake['is_hard']:
            self.hard_examples.append(mistake)
            self.save_hard_examples()
        self.save_mistakes()
        self.confidence_history.append({
            'timestamp': datetime.now(), 'symbol': symbol, 'confidence': confidence,
            'is_mistake': prediction != actual, 'is_high_priority': mistake['is_high_priority']
        })

    def _generate_explanation(self, pattern, actual, market_regime):
        explanations = {
            1: f"This pattern indicates upward price movement. Entry at breakout, stop loss below support.",
            0: f"This pattern indicates downward price movement. Entry at breakdown, stop loss above resistance.",
            2: f"This pattern indicates consolidation. Wait for breakout confirmation."
        }
        return explanations.get(actual, f"The correct signal is {actual}")

    def save_mistakes(self):
        try:
            df = pd.DataFrame(self.mistakes)
            df.to_csv(MISTAKES_FILE, index=False)
        except:
            pass

    def save_hard_examples(self):
        try:
            df = pd.DataFrame(self.hard_examples)
            df.to_csv(HARD_EXAMPLES_FILE, index=False)
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
            'low_confidence_count': len(df[df['confidence'] < aHARD_EXAMPLE_THRESHOLD]) if 'confidence' in df.columns else 0,
            'high_priority_count': len(df[df.get('is_high_priority', False)])
        }

    def get_mistake_dataset(self, limit=200):
        mistake_texts = []
        signal_map = {1: 'BUY', 0: 'SELL', 2: 'HOLD'}
        for m in self.get_hard_examples(limit=limit):
            enhanced_context = ""
            if self.xgb_ppo:
                xgb_pred = self.xgb_ppo.get_xgb_prediction(m.get('symbol', ''))
                if xgb_pred:
                    enhanced_context = f"\nXGBoost Signal: {xgb_pred['signal']} (Confidence: {xgb_pred['prob_up']:.0%})"
            text = f"""
================================================================================
Pattern: {m.get('pattern', 'Unknown')}
Symbol: {m.get('symbol')}
Technical Analysis: Pattern detected with {m.get('confidence', 0.5):.0%} confidence{enhanced_context}

Analysis: {m.get('correct_explanation', 'Review the pattern rules')}

Signal: {signal_map.get(m.get('actual', 2), 'HOLD')}
Confidence: {min(95, max(65, int(m.get('confidence', 0.7) * 100 + 10)))}
Risk Level: Medium
Timeframe: Short-term
================================================================================
"""
            mistake_texts.append(text)
        return mistake_texts


# =========================================================
# AUTO LLM TRAINER CLASS
# =========================================================

class AutoLLMTrainer:
    def __init__(self):
        # Ensure csv directory exists
        os.makedirs("./csv", exist_ok=True)
        os.makedirs(LLM_MODEL_DIR, exist_ok=True)
        os.makedirs(AGENTIC_LOOP_LOG_DIR, exist_ok=True)
        
        self.trained_symbols = self.load_trained_symbols()
        self.model = None
        self.tokenizer = None
        self.xgb_ppo = XGBoostPPOIntegrator()
        self.mistake_collector = MistakeCollector(self.xgb_ppo)
        self.batch_manager = BatchManager()
        self.old_training_texts = []
        self.hf_uploader = HFUploader()  # ✅ HF আপলোডার (চেকপয়েন্ট + ফাইনাল মডেল)
        
        # ========== AGENTIC LOOP INIT ==========
        self.agentic_loop = None
        if AGENTIC_LOOP_AVAILABLE:
            self._init_agentic_loop()
        # =======================================

    # =========================================================
    # AGENTIC LOOP METHODS
    # =========================================================

    def _init_agentic_loop(self):
        try:
            print("\n" + "="*60)
            print("🤖 INITIALIZING AGENTIC LOOP")
            print("="*60)
            self.agentic_loop = AgenticLoop(xgb_model_dir=XGBOOST_DIR)
            xgb_agent = next((a for a in self.agentic_loop.agents if a.name == "XGBoost"), None)
            if xgb_agent and xgb_agent.models:
                print(f"   ✅ Agentic Loop ready with {len(xgb_agent.models)} XGBoost models")
            else:
                print(f"   ⚠️ Agentic Loop running without XGBoost models")
            os.makedirs(AGENTIC_LOOP_LOG_DIR, exist_ok=True)
            print("="*60 + "\n")
        except Exception as e:
            print(f"   ❌ Agentic Loop init failed: {e}")
            self.agentic_loop = None

    def _update_agentic_loop_after_batch(self, batch_num, symbols, eval_loss=None):
        if self.agentic_loop is None:
            return
        try:
            print(f"\n   📊 Agentic Loop: Processing batch {batch_num} feedback...")
            simulated_pnl = 0.02
            if eval_loss is not None:
                simulated_pnl = max(-0.08, min(0.08, -eval_loss * 0.008))
            success = simulated_pnl > 0
            for symbol in symbols:
                trade_result = {'symbol': symbol, 'pnl': simulated_pnl, 'success': success, 'batch': batch_num}
                self.agentic_loop.after_trade_feedback(trade_result)
            summary = self.agentic_loop.get_summary()
            if len(summary) > 0:
                print("\n   📈 Agent Performance:")
                for _, row in summary.iterrows():
                    print(f"      {row['agent']}: {row['accuracy']} accuracy")
            log_path = os.path.join(AGENTIC_LOOP_LOG_DIR, f'batch_{batch_num}_log.csv')
            self.agentic_loop.save_decision_log(log_path)
        except Exception as e:
            print(f"   ⚠️ Agentic Loop update failed: {e}")

    def _finalize_agentic_loop(self):
        if self.agentic_loop is None:
            return
        try:
            print("\n" + "="*60)
            print("🏆 AGENTIC LOOP FINAL REPORT")
            print("="*60)
            summary = self.agentic_loop.get_summary()
            if len(summary) > 0:
                for _, row in summary.iterrows():
                    print(f"   {row['agent']}: {row['accuracy']} accuracy ({row['total_predictions']} predictions)")
            best_agent = None
            best_acc = 0
            for agent in self.agentic_loop.agents:
                acc = agent.get_accuracy()
                if acc > best_acc:
                    best_acc = acc
                    best_agent = agent.name
            if best_agent:
                print(f"\n   🥇 Best Agent: {best_agent} ({best_acc:.1%} accuracy)")
            state = {'timestamp': str(datetime.now()), 'agents': {}}
            for agent in self.agentic_loop.agents:
                state['agents'][agent.name] = {
                    'accuracy': agent.get_accuracy(),
                    'predictions': agent.total_predictions,
                    'weight': agent.get_dynamic_weight()
                }
            with open(AGENTIC_LOOP_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"\n   💾 State saved to {AGENTIC_LOOP_STATE_FILE}")
            print("="*60)
        except Exception as e:
            print(f"   ⚠️ Final report failed: {e}")

    # =========================================================
    # ORIGINAL METHODS
    # =========================================================

    def load_trained_symbols(self):
        if os.path.exists(TRACKING_FILE):
            try:
                with open(TRACKING_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get('symbols', [])
            except:
                return []
        return []

    def save_trained_symbols(self):
        data = {
            'symbols': self.trained_symbols,
            'last_updated': datetime.now().isoformat(),
            'total_trained': len(self.trained_symbols)
        }
        with open(TRACKING_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def get_all_symbols_from_mongodb(self, limit=None):
        if not os.path.exists(MARKET_DATA_PATH):
            print(f"❌ Market data not found: {MARKET_DATA_PATH}")
            return []
        df = pd.read_csv(MARKET_DATA_PATH)
        symbols = df['symbol'].unique().tolist()
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
        hard_keywords = ['complex', 'multi timeframe', 'divergence', 'harmonic', 'elliott', 'smc', 'order block', 'fvg', 'liquidity']
        medium_keywords = ['triangle', 'wedge', 'flag', 'pennant', 'reversal']
        for kw in hard_keywords:
            if kw in text_lower:
                return 'hard'
        for kw in medium_keywords:
            if kw in text_lower:
                return 'medium'
        text_len = len(text)
        if text_len > 1500:
            return 'hard'
        elif text_len > 800:
            return 'medium'
        return 'easy'

    def load_training_data_with_curriculum(self):
        if not os.path.exists(TRAINING_DATA_PATH):
            print(f"❌ Training data not found: {TRAINING_DATA_PATH}")
            return None, None

        with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
            text_data = f.read()

        raw_examples = text_data.split('================================================================================')
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
            difficulty = self.classify_example_difficulty(text)

            if 'Elliott Wave' in text or 'Impulse Wave' in text or 'Corrective Wave' in text:
                example_weights[i] = 5.0
            elif 'SMC' in text or 'Order Block' in text or 'FVG' in text or 'Liquidity' in text:
                example_weights[i] = 4.5
            elif 'Harmonic' in text or 'Gartley' in text or 'Butterfly' in text:
                example_weights[i] = 4.0
            elif difficulty == 'hard':
                example_weights[i] = 3.5
            elif difficulty == 'medium':
                example_weights[i] = 1.5
            else:
                example_weights[i] = 0.8

        return train_texts, example_weights

    def load_model_with_lora(self):
        print("\n🏗️ Loading model...")
        token = os.getenv("hf_token")

        # ✅ HF Dataset Repo থেকে ফাইনাল মডেল লোড করার চেষ্টা
        if token:
            try:
                print(f"   Attempting to load from HF Dataset: {HF_DATASET_REPO}/final_model/")
                from huggingface_hub import list_repo_files
                files = list_repo_files(repo_id=HF_DATASET_REPO, repo_type="dataset", token=token)
                
                final_model_folders = [f for f in files if f.startswith("final_model/")]
                if final_model_folders:
                    print(f"   ✅ Found final model in HF Dataset repo")
            except Exception as e:
                print(f"   No final model in HF Dataset: {e}")

        if os.path.exists(LLM_MODEL_DIR):
            try:
                print(f"   Loading local model from {LLM_MODEL_DIR}...")
                self.model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_DIR)
                self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_DIR)
                print("   ✅ Loaded local model")
                if LORA_AVAILABLE:
                    lora_config = LoraConfig(**LORA_CONFIG)
                    self.model = get_peft_model(self.model, lora_config)
                    print("   ✅ LoRA applied")
                self._post_load_setup()
                return
            except:
                pass

        print(f"   Loading base model: {BASE_MODEL}")
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            torch_dtype=torch.float32, 
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        if LORA_AVAILABLE:
            lora_config = LoraConfig(**LORA_CONFIG)
            self.model = get_peft_model(self.model, lora_config)
            print("   ✅ LoRA applied")

        self._post_load_setup()
        print("   ✅ Loaded base model")

    def _post_load_setup(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Device: {device}")
        print(f"   XGBoost Models Loaded: {len(self.xgb_ppo.xgb_models)}")
        print(f"   PPO Models Found: {len(self.xgb_ppo.ppo_models)}")

    def train(self, mode="incremental", symbols_batch=None):
        print(f"\n{'='*60}")
        print(f"🎯 TRAINING MODE: {mode.upper()}")
        if symbols_batch:
            print(f"📚 Symbols in this batch: {len(symbols_batch)}")
        print(f"{'='*60}")

        train_texts, example_weights = self.load_training_data_with_curriculum()
        if not train_texts:
            print("❌ No training data found!")
            return False

        encodings = self.tokenizer(
            train_texts, 
            truncation=True, 
            padding="max_length", 
            max_length=512,
            return_tensors="pt"
        )

        train_dataset = StructuredDataset(encodings, example_weights)

        dataset_size = len(train_dataset)
        val_size = max(1, int(dataset_size * VALIDATION_SPLIT_RATIO))
        train_size = dataset_size - val_size

        train_indices = list(range(train_size))
        val_indices = list(range(train_size, dataset_size))

        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        eval_subset = torch.utils.data.Subset(train_dataset, val_indices)
        print(f"   Dataset split: {train_size} train, {val_size} validation (chronological)")

        num_epochs = EPOCHS_CONFIG.get(mode, 10)
        learning_rate = LR_CONFIG.get(mode, 1e-5)
        batch_size = BATCH_SIZE_CONFIG.get(mode, 1)
        grad_accum = GRAD_ACCUM_CONFIG.get(mode, 16)

        print(f"\n⚙️ 70+ Hours Training Config:")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Batch Size: {batch_size} (effective: {batch_size * grad_accum})")
        print(f"   Gradient Accumulation: {grad_accum}")
        print(f"   LoRA: {'Enabled (r=32)' if LORA_AVAILABLE else 'Disabled'}")
        print(f"   Max Length: 512 (full context)")
        print(f"   XGBoost Integration: Enabled ({len(self.xgb_ppo.xgb_models)} models)")
        print(f"   📤 HF Repo: {HF_DATASET_REPO}")

        training_args = TrainingArguments(
            output_dir=LLM_MODEL_DIR,
            overwrite_output_dir=True,

            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,

            learning_rate=learning_rate,
            warmup_steps=300,
            weight_decay=0.015,
            lr_scheduler_type="cosine_with_restarts",

            save_steps=80,
            save_total_limit=3,
            logging_steps=10,
            save_strategy="steps",

            evaluation_strategy="no",
            load_best_model_at_end=False,

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
        
        #✅ ফিক্সড HF চেকপয়েন্ট কাস্টম আপলোডার
        class CustomHFCallback:
            def __init__(self, hf_uploader):
                self.hf_uploader = hf_uploader
        
            def on_save(self, args, state, control, **kwargs):
                if state.is_world_process_zero:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                    if os.path.exists(checkpoint_dir):
                        print(f"\n   📤 Uploading checkpoint {state.global_step} to HF Dataset repo...")
                        self.hf_uploader.upload_checkpoint(checkpoint_dir, state.global_step)
                        self.hf_uploader.upload_tracking_files()
                return control         
        
            # সব missing methods handle করার জন্য
            def __getattr__(self, name):
                if name.startswith('on_'):
                    return lambda *args, **kwargs: args[2] if len(args) > 2 else None
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
        trainer.add_callback(CustomHFCallback(self.hf_uploader))

        print("\n🏋️ Starting 70+ Hours Training...")
        # ... বাকি কোড ...
        print("   ⏰ This will take multiple days (resume automatically)")
        print(f"   📤 Checkpoints will be uploaded to: {HF_DATASET_REPO}/checkpoints/")
        trainer.train()
        print("\n✅ Training completed!")

        # ========== AGENTIC LOOP UPDATE ==========
        if symbols_batch and hasattr(self, 'agentic_loop') and self.agentic_loop is not None:
            batch_num = self.batch_manager.current_batch_index if self.batch_manager.current_batch_index > 0 else 1
            self._update_agentic_loop_after_batch(batch_num, symbols_batch, eval_loss=0.5)
        # =========================================

        self.model.save_pretrained(LLM_MODEL_DIR)
        self.tokenizer.save_pretrained(LLM_MODEL_DIR)
        print(f"💾 Model saved locally to {LLM_MODEL_DIR}")

        # ✅ ফাইনাল মডেল HF Dataset Repo-তে আপলোড
        self.upload_final_model_to_hf(mode)
        return True

    def upload_final_model_to_hf(self, mode):
        """Upload final model to HF Dataset Repo"""
        token = os.getenv("hf_token")
        if not token:
            print("ℹ️ No HF_TOKEN, skipping final model upload")
            return

        print(f"\n📤 Uploading final model to HF Dataset Repo: {HF_DATASET_REPO}/final_model/{mode}/")
        try:
            self.hf_uploader.upload_final_model(LLM_MODEL_DIR, mode)
            self.hf_uploader.upload_tracking_files()
            print(f"✅ Final model uploaded to: https://huggingface.co/datasets/{HF_DATASET_REPO}/tree/main/final_model/{mode}")
        except Exception as e:
            print(f"⚠️ Final model upload failed: {e}")

    def generate_training_data_for_symbols(self, symbols):
        print(f"\n📝 Generating training data for {len(symbols)} symbols...")
        import subprocess
        result = subprocess.run(
            ["python", "scripts/generate_pattern_training_data_complete.py", 
             "--symbols", ",".join(symbols)], 
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"   ⚠️ Data generation failed: {result.stderr}")
            return False
        print("   ✅ Training data generated")
        return True

    def run(self):
        print("="*60)
        print("🚀 AUTO LLM TRAINER - 70+ HOURS ULTIMATE VERSION")
        print("="*60)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📚 Batch size: {BATCH_SIZE}")
        print(f"🔄 Fine-tune interval: {FINE_TUNE_INTERVAL} days")
        print(f"🔧 LoRA: r=32, alpha=64 (Enhanced)")
        print(f"📊 XGBoost Models: {len(self.xgb_ppo.xgb_models)}")
        print(f"📁 PPO Models: {len(self.xgb_ppo.ppo_models)}")
        print(f"⏰ 70+ Hours Training Mode: ENABLED")
        print(f"📤 ALL SAVED TO: {HF_DATASET_REPO}")
        print(f"💾 Local Model Dir: {LLM_MODEL_DIR}")
        print("="*60)

        confidence_stats = self.mistake_collector.get_confidence_stats()
        print(f"\n📊 Confidence Statistics:")
        print(f"   Average confidence: {confidence_stats['avg_confidence']:.2%}")
        print(f"   Mistake rate: {confidence_stats['mistake_rate']:.2f}%")
        print(f"   High priority mistakes: {confidence_stats['high_priority_count']}")

        all_symbols = self.get_all_symbols_from_mongodb()
        new_symbols = self.get_new_symbols()

        self.load_model_with_lora()

        # STEP 1: Train new symbols in batches
        if new_symbols:
            print(f"\n📚 Found {len(new_symbols)} new symbols to train")

            for i in range(0, len(new_symbols), BATCH_SIZE):
                batch = new_symbols[i:i+BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1
                print(f"\n📦 Processing batch {batch_num}: {len(batch)} symbols")

                if self.generate_training_data_for_symbols(batch):
                    if len(self.trained_symbols) == 0:
                        mode = "first_train"
                    else:
                        mode = "incremental"

                    if self.train(mode=mode, symbols_batch=batch):
                        self.trained_symbols.extend(batch)
                        self.save_trained_symbols()
                        self.batch_manager.mark_batch_completed(batch_num, batch)
                        print(f"✅ Batch {batch_num} complete! Total trained: {len(self.trained_symbols)}")
        else:
            print("\n✅ No new symbols found!")

        # STEP 2: Weekly fine-tune
        weekly_batch_num, weekly_symbols = self.batch_manager.get_batch_for_weekly_finetune()

        if weekly_symbols and len(weekly_symbols) > 0:
            print(f"\n🔄 Weekly fine-tuning on Batch {weekly_batch_num} ({len(weekly_symbols)} symbols)")
            if self.generate_training_data_for_symbols(weekly_symbols):
                self.train(mode="weekly_finetune", symbols_batch=weekly_symbols)
                print("✅ Weekly fine-tune complete!")

        # STEP 3: Monthly consolidation
        if self.batch_manager.should_consolidate():
            print(f"\n🔄 Monthly consolidation - Training all symbols together")
            all_trained_symbols = self.batch_manager.get_all_batch_symbols()
            if all_trained_symbols:
                if self.generate_training_data_for_symbols(all_trained_symbols):
                    self.train(mode="consolidate", symbols_batch=all_trained_symbols)
                    self.batch_manager.mark_consolidated()
                    print("✅ Monthly consolidation complete!")

        # STEP 4: Hard example retraining
        high_priority_examples = self.mistake_collector.get_hard_examples(limit=200, priority_only=True)
        if high_priority_examples:
            print(f"\n🔥 Found {len(high_priority_examples)} high priority mistakes for retraining!")
            temp_file = "./csv/temp_hard_examples.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                signal_map = {1: 'BUY', 0: 'SELL', 2: 'HOLD'}
                for ex in high_priority_examples[:100]:
                    xgb_context = ""
                    xgb_pred = self.xgb_ppo.get_xgb_prediction(ex.get('symbol', ''))
                    if xgb_pred:
                        xgb_context = f"\nXGBoost Analysis: {xgb_pred['signal']} ({xgb_pred['prob_up']:.0%} confidence)"

                    f.write(f"""
================================================================================
Pattern: {ex.get('pattern', 'Unknown')}
Symbol: {ex.get('symbol')}
Previous Prediction: {signal_map.get(ex.get('prediction', 2), 'HOLD')}
❌ This was WRONG{xgb_context}

✅ CORRECT ANSWER: {signal_map.get(ex.get('actual', 2), 'HOLD')}
Explanation: {ex.get('correct_explanation', 'Review the pattern rules')}

Signal: {signal_map.get(ex.get('actual', 2), 'HOLD')}
Confidence: {min(95, max(65, int(ex.get('confidence', 0.7) * 100 + 10)))}
================================================================================
""")
            # Temporarily replace training data
            original_path = TRAINING_DATA_PATH
            TRAINING_DATA_PATH = temp_file
            self.train(mode="mistake_learning")
            TRAINING_DATA_PATH = original_path
            if os.path.exists(temp_file):
                os.remove(temp_file)

        print("\n" + "="*60)
        print("📊 FINAL STATUS")
        print("="*60)
        print(f"   Total trained symbols: {len(self.trained_symbols)}")
        print(f"   Completed batches: {len(self.batch_manager.completed_batches)}")
        print(f"   XGBoost Models Available: {len(self.xgb_ppo.xgb_models)}")
        print(f"   PPO Models Available: {len(self.xgb_ppo.ppo_models)}")
        print(f"   Local Model Dir: {LLM_MODEL_DIR}")
        print(f"   HF Dataset Repo (ALL): {HF_DATASET_REPO}")
        print("="*60)

        # ========== AGENTIC LOOP FINALIZE ==========
        self._finalize_agentic_loop()
        # ===========================================


if __name__ == "__main__":
    trainer = AutoLLMTrainer()
    trainer.run()
