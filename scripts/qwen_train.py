# ================== scripts/qwen_train.py ==================
# Qwen3 Trainer v6 — TIME-SAFE + FULL PATCHED
# ✅ ADDED: TimeLimitCallback (stop at 5.5h → save before 6h timeout)
# ✅ ADDED: Eval step control (avoid 18-min eval overhead)
# ✅ ADDED: Reduce epochs to fit time budget
# ✅ PRESERVED: All Script 2 fixes (marker, weighted loss, random split)

import os
import re
import gc
import sys
import json
import glob
import html
import time
import shutil
import hashlib
import inspect
import warnings
import subprocess
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from huggingface_hub import login, create_repo, HfApi

# =========================================================
# OPTIONAL IMPORTS
# =========================================================
try:
    from agentic_loop import AgenticLoop
    AGENTIC_LOOP_AVAILABLE = True
except ImportError:
    AGENTIC_LOOP_AVAILABLE = False
    print("⚠️ Agentic Loop not found. Multi-agent voting disabled.")

try:
    from peft import LoraConfig, get_peft_model
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    print("⚠️ PEFT not installed. Install with: pip install peft")

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================
BATCH_SIZE = 40

HF_DATASET_REPO = "ahashanahmed/csv"

BASE_MODEL = "Qwen/Qwen3-0.6B"
LLM_MODEL_DIR = "./csv/llm_model_qwen3"

# HF paths (fixed, overwritten each time)
QWEN3_HF_CHECKPOINT_PATH = "qwen3_checkpoints/current"
QWEN3_HF_FINAL_PATH = "final_model_qwen3/latest"

# Local checkpoint dir
QWEN3_LOCAL_CHECKPOINT_DIR = "./csv/qwen3_checkpoints"
MARKER_NAME = "run_marker.json"

TRACKING_FILE = "./csv/trained_symbols_qwen3.json"
BATCH_TRACKING_FILE = "./csv/batch_tracking_qwen3.json"

MARKET_DATA_PATH = "./csv/mongodb.csv"
TRAINING_DATA_PATH = "./csv/training_texts.txt"
MISTAKES_FILE = "./csv/trading_mistakes_qwen3.csv"
HARD_EXAMPLES_FILE = "./csv/hard_examples_qwen3.csv"
STATUS_FILE = "./csv/current_training_status.json"

XGBOOST_DIR = "./csv/xgboost"
PPO_PER_SYMBOL_DIR = "./csv/ppo_models/per_symbol"

AGENTIC_LOOP_STATE_FILE = "./csv/agentic_loop_state.json"
AGENTIC_LOOP_LOG_DIR = "./csv/agentic_loop_logs"

FINE_TUNE_INTERVAL = 7
CONSOLIDATE_INTERVAL = 30

MAX_OLD_EXAMPLES = 10000
HARD_EXAMPLE_THRESHOLD = 0.25
HIGH_PRIORITY_THRESHOLD = 0.35
MAX_GRAD_NORM = 0.5
VALIDATION_SPLIT_RATIO = 0.15
SPLIT_SEED = 42
MAX_LENGTH = 192

SAVE_STEPS = 20
HF_CKPT_MIN_INTERVAL_SEC = 600

# ✅ TIME LIMIT (GitHub 6h hard limit → stop at 5.5h)
MAX_TRAINING_HOURS = 5.5        # Stop training at this point

# ✅ EVAL control (avoid slow evaluation)
MAX_EVAL_SAMPLES = 100          # Cap val dataset size (was 603 → 18 min)
EVAL_STEPS = 100                # Eval every 100 optimizer steps (was save_steps)

LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

CONFIDENCE_PATTERN = r"(?:Confidence|Signal Strength):?\s*(\d+(?:\.\d+)?)%"
DELIM = "=" * 80

# ✅ REDUCED EPOCHS (fit in 5.5h budget)
EPOCHS_CONFIG = {
    "first_train": 1,           # was 2
    "incremental": 1,           # was 2
    "weekly_finetune": 1,
    "consolidate": 2,           # was 3
    "mistake_learning": 1,
}
LR_CONFIG = {
    "first_train": 2e-4, "incremental": 1e-4, "weekly_finetune": 5e-5,
    "consolidate": 1e-4, "mistake_learning": 5e-5,
}
BATCH_SIZE_CONFIG = {
    "first_train": 1, "incremental": 1, "weekly_finetune": 1,
    "consolidate": 1, "mistake_learning": 1,
}
GRAD_ACCUM_CONFIG = {
    "first_train": 32, "incremental": 24, "weekly_finetune": 16,
    "consolidate": 32, "mistake_learning": 24,
}

MODE_EXPLANATION = {
    "first_train": "🎯 FIRST TIME TRAINING (Qwen3-0.6B - Base Model)",
    "incremental": "⚙️ INCREMENTAL TRAINING (New symbols added)",
    "weekly_finetune": "🔄 WEEKLY FINE-TUNE (Every 7 days)",
    "consolidate": "📈 MONTHLY RE-TUNE (Every 30 days)",
    "mistake_learning": "🎯 MISTAKE LEARNING (Retraining from past errors)",
}
MODE_SHORT_EXPLANATION = {
    "first_train": "First Time", "incremental": "New Symbols",
    "weekly_finetune": "Weekly Fine-Tune", "consolidate": "Monthly Re-Tune",
    "mistake_learning": "Mistake Learning",
}

# =========================================================
# HELPERS
# =========================================================

def get_hf_token():
    return os.getenv("HF_TOKEN") or os.getenv("hf_token")


def esc(x, limit=300):
    return html.escape(str(x)[:limit])


def send_telegram_message(message, token=None, chat_id=None):
    token = token or os.getenv("TELEGRAM_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("⚠️ Telegram credentials not found")
        return None
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        return requests.post(url, json=payload, timeout=10).json()
    except Exception as e:
        print(f"⚠️ Telegram send failed: {e}")
        return None


def load_causal_lm(path):
    """Load model in float32 (works on old and new transformers)."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, low_cpu_mem_usage=True)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32, low_cpu_mem_usage=True)


def write_marker(folder, marker):
    try:
        with open(os.path.join(folder, MARKER_NAME), "w") as f:
            json.dump(marker, f)
    except Exception as e:
        print(f"   ⚠️ Could not write marker in {folder}: {e}")


def read_marker(folder):
    try:
        with open(os.path.join(folder, MARKER_NAME), "r") as f:
            return json.load(f)
    except Exception:
        return None


def get_base_tag():
    try:
        with open(os.path.join(LLM_MODEL_DIR, "train_info.json"), "r") as f:
            return str(json.load(f).get("trained_at", "base"))
    except Exception:
        return "base"


def make_marker(mode, symbols_batch):
    sig_src = mode + "|" + ",".join(sorted(symbols_batch or [])) + "|" + get_base_tag()
    return {"mode": mode,
            "signature": hashlib.md5(sig_src.encode("utf-8")).hexdigest()}


def clear_dir_contents(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


# =========================================================
# ✅ NEW: TIME LIMIT CALLBACK (GitHub 6h protection)
# =========================================================

class TimeLimitCallback(TrainerCallback):
    """
    Stops training at MAX_TRAINING_HOURS to avoid GitHub 6-hour timeout.
    Saves checkpoint before stopping so next run can resume.
    """

    def __init__(self, max_hours=5.5):
        self.max_seconds = max_hours * 3600
        self.start_time = None
        self.stopped = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"   ⏰ Time limit: {self.max_seconds / 3600:.1f}h (GitHub 6h safety)")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.stopped:
            return control
        if self.start_time is None:
            return control
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_seconds:
            elapsed_h = elapsed / 3600
            print(f"\n   ⏰ TIME LIMIT REACHED ({elapsed_h:.2f}h) at step {state.global_step}")
            print(f"   🛑 Stopping training + saving checkpoint")
            control.should_training_stop = True
            control.should_save = True
            self.stopped = True
        elif elapsed >= self.max_seconds - 300:  # warn 5 min before
            if state.global_step % 10 == 0:
                remaining = (self.max_seconds - elapsed) / 60
                print(f"   ⚠️ ~{remaining:.0f} min left before time limit (step {state.global_step})")
        return control


# =========================================================
# HF UPLOADER
# =========================================================

class HFUploader:
    def __init__(self, repo_id=HF_DATASET_REPO):
        self.repo_id = repo_id
        self.api = None
        self._init_api()

    def _init_api(self):
        token = get_hf_token()
        if not token:
            print("   ℹ️ No HF token, HF upload disabled")
            return
        try:
            login(token=token)
            self.api = HfApi(token=token)
            create_repo(repo_id=self.repo_id, repo_type="dataset",
                        exist_ok=True, token=token)
            print(f"   ✅ HF Dataset Repo ready: {self.repo_id}")
        except Exception as e:
            print(f"   ⚠️ HF API init failed: {e}")
            self.api = None

    def _upload_folder(self, folder, repo_path, message):
        self.api.upload_folder(
            folder_path=folder,
            path_in_repo=repo_path,
            repo_id=self.repo_id,
            repo_type="dataset",
            delete_patterns="*",
            commit_message=message,
        )

    def upload_checkpoint(self, checkpoint_path):
        if self.api is None:
            return False
        try:
            self._upload_folder(
                checkpoint_path, QWEN3_HF_CHECKPOINT_PATH,
                f"🤖 Qwen3 checkpoint - {datetime.now():%Y-%m-%d %H:%M}")
            print(f"   📤 checkpoint → {self.repo_id}/{QWEN3_HF_CHECKPOINT_PATH}")
            return True
        except Exception as e:
            print(f"   ⚠️ Checkpoint upload failed: {e}")
            return False

    def upload_final_model(self, model_path):
        if self.api is None:
            return False
        try:
            self._upload_folder(
                model_path, QWEN3_HF_FINAL_PATH,
                f"🤖 Qwen3 final model - {datetime.now():%Y-%m-%d %H:%M}")
            print(f"   📤 final model → {self.repo_id}/{QWEN3_HF_FINAL_PATH}/")
            return True
        except Exception as e:
            print(f"   ⚠️ Final model upload failed: {e}")
            return False

    def clear_hf_checkpoints(self):
        if self.api is None:
            return
        try:
            self.api.delete_folder(
                path_in_repo="qwen3_checkpoints",
                repo_id=self.repo_id, repo_type="dataset",
                commit_message="🗑️ Qwen3 training finished - clear checkpoints")
            print("   🗑️ HF checkpoints cleared")
        except Exception as e:
            print(f"   ℹ️ HF checkpoint clear skipped: {str(e)[:120]}")

    def upload_tracking_files(self):
        if self.api is None:
            return
        for local, remote in ((TRACKING_FILE, "trained_symbols_qwen3.json"),
                              (BATCH_TRACKING_FILE, "batch_tracking_qwen3.json")):
            if not os.path.exists(local):
                continue
            try:
                self.api.upload_file(
                    path_or_fileobj=local, path_in_repo=remote,
                    repo_id=self.repo_id, repo_type="dataset",
                    commit_message=f"Update {remote} - {datetime.now():%Y-%m-%d %H:%M}")
                print(f"   📤 {remote} uploaded")
            except Exception as e:
                print(f"   ⚠️ Tracking upload failed ({remote}): {e}")


# =========================================================
# BATCH MANAGER
# =========================================================

class BatchManager:
    def __init__(self):
        self.batch_tracking = self.load_batch_tracking()
        self._ensure_required_fields()
        self.completed_batches = self.batch_tracking["completed_batches"]
        self.batch_symbols = self.batch_tracking["batch_symbols"]
        self.current_batch_index = self.batch_tracking.get("current_batch", 0)

    @staticmethod
    def _defaults():
        return {
            "current_batch": 0, "completed_batches": [], "batch_symbols": {},
            "total_symbols_trained": 0, "last_batch_date": None,
            "weekly_trained_batches": [], "last_weekly_finetune": None,
            "last_consolidate": None, "monthly_consolidation_done": False,
            "last_mistake_learning": None,
        }

    def _ensure_required_fields(self):
        updated = False
        for k, v in self._defaults().items():
            if k not in self.batch_tracking:
                self.batch_tracking[k] = v
                updated = True
        if updated:
            self.save_batch_tracking()

    def load_batch_tracking(self):
        if os.path.exists(BATCH_TRACKING_FILE):
            try:
                with open(BATCH_TRACKING_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"   ⚠️ Could not read batch tracking: {e}")
        return self._defaults()

    def save_batch_tracking(self):
        os.makedirs(os.path.dirname(BATCH_TRACKING_FILE), exist_ok=True)
        with open(BATCH_TRACKING_FILE, "w") as f:
            json.dump(self.batch_tracking, f, indent=2)

    def next_batch_number(self):
        return (max(self.completed_batches) + 1) if self.completed_batches else 1

    def mark_batch_completed(self, batch_num, symbols):
        if batch_num not in self.completed_batches:
            self.completed_batches.append(batch_num)
        self.batch_symbols[str(batch_num)] = list(symbols)
        self.current_batch_index = batch_num
        self.batch_tracking["current_batch"] = batch_num
        self.batch_tracking["completed_batches"] = self.completed_batches
        self.batch_tracking["batch_symbols"] = self.batch_symbols
        self.batch_tracking["total_symbols_trained"] = (
            self.batch_tracking.get("total_symbols_trained", 0) + len(symbols))
        self.batch_tracking["last_batch_date"] = datetime.now().isoformat()
        self.save_batch_tracking()

    def init_schedule_baselines(self):
        now = datetime.now().isoformat()
        changed = False
        if not self.batch_tracking.get("last_weekly_finetune"):
            self.batch_tracking["last_weekly_finetune"] = now
            changed = True
        if not self.batch_tracking.get("last_consolidate"):
            self.batch_tracking["last_consolidate"] = now
            changed = True
        if changed:
            self.save_batch_tracking()

    def get_batch_for_weekly_finetune(self):
        if not self.completed_batches:
            return None, []

        last_weekly = self.batch_tracking.get("last_weekly_finetune")
        if last_weekly:
            try:
                days_passed = (datetime.now() - datetime.fromisoformat(last_weekly)).days
                print(f"   📅 Days since last weekly fine-tune: {days_passed}")
                if days_passed < FINE_TUNE_INTERVAL:
                    print(f"   ⏳ Waiting {FINE_TUNE_INTERVAL - days_passed} more days")
                    return None, []
            except Exception:
                pass

        trained = set(self.batch_tracking.get("weekly_trained_batches", []))
        available = [b for b in self.completed_batches if b not in trained]

        if not available:
            print("   🔄 All batches trained, resetting weekly cycle")
            self.batch_tracking["weekly_trained_batches"] = []
            self.save_batch_tracking()
            available = list(self.completed_batches)

        batch_num = available[0]
        symbols = self.batch_symbols.get(str(batch_num), [])
        if not symbols:
            print(f"   ⚠️ Batch {batch_num} has no stored symbols, skipping")
            return None, []
        print(f"   ✅ Weekly batch {batch_num} ({len(symbols)} symbols)")
        return batch_num, symbols

    def mark_weekly_done(self, batch_num):
        weekly = self.batch_tracking.get("weekly_trained_batches", [])
        if batch_num not in weekly:
            weekly.append(batch_num)
        self.batch_tracking["weekly_trained_batches"] = weekly
        self.batch_tracking["last_weekly_finetune"] = datetime.now().isoformat()
        self.save_batch_tracking()
        print(f"   ✅ Batch {batch_num} marked as weekly trained")

    def should_consolidate(self):
        last = self.batch_tracking.get("last_consolidate")
        if not last:
            return bool(self.completed_batches)
        try:
            days = (datetime.now() - datetime.fromisoformat(last)).days
            print(f"   📅 Days since last consolidation: {days}")
            if days >= CONSOLIDATE_INTERVAL:
                return True
            print(f"   ⏳ {CONSOLIDATE_INTERVAL - days} days until next consolidation")
            return False
        except Exception:
            return True

    def mark_consolidated(self):
        self.batch_tracking["last_consolidate"] = datetime.now().isoformat()
        self.batch_tracking["monthly_consolidation_done"] = True
        self.batch_tracking["weekly_trained_batches"] = []
        self.batch_tracking["last_weekly_finetune"] = datetime.now().isoformat()
        self.save_batch_tracking()
        print(f"   ✅ Consolidation marked on {datetime.now():%Y-%m-%d}")

    def should_run_mistake_learning(self):
        last = self.batch_tracking.get("last_mistake_learning")
        if not last:
            return True
        try:
            return (datetime.now() - datetime.fromisoformat(last)).days >= FINE_TUNE_INTERVAL
        except Exception:
            return True

    def mark_mistake_learning_done(self):
        self.batch_tracking["last_mistake_learning"] = datetime.now().isoformat()
        self.save_batch_tracking()

    def get_all_batch_symbols(self):
        out = []
        for b in self.completed_batches:
            out.extend(self.batch_symbols.get(str(b), []))
        return list(dict.fromkeys(out))


# =========================================================
# XGBOOST + PPO INTEGRATION
# =========================================================

class XGBoostPPOIntegrator:
    FEATURE_ORDER = ["close", "volume", "return_5d", "return_10d",
                     "volatility", "volatility_5d", "volume_ratio",
                     "rsi_oversold", "rsi_overbought", "dist_from_sr",
                     "sr_strength", "is_bullish_div", "div_strength",
                     "dist_from_ema", "above_ema"]

    def __init__(self):
        self.xgb_models = {}
        self.ppo_models = {}
        self.load_xgb_models()
        self.load_ppo_metadata()

    def load_xgb_models(self):
        if not os.path.exists(XGBOOST_DIR):
            print(f"   ⚠️ XGBoost directory not found: {XGBOOST_DIR}")
            return
        for file in os.listdir(XGBOOST_DIR):
            if file.endswith(".joblib"):
                symbol = file[:-len(".joblib")]
                try:
                    self.xgb_models[symbol] = joblib.load(os.path.join(XGBOOST_DIR, file))
                except Exception as e:
                    print(f"   ⚠️ Failed to load XGBoost for {symbol}: {e}")
        print(f"   ✅ Loaded {len(self.xgb_models)} XGBoost models")

    def load_ppo_metadata(self):
        if not os.path.exists(PPO_PER_SYMBOL_DIR):
            print(f"   ⚠️ PPO models directory not found: {PPO_PER_SYMBOL_DIR}")
            return
        for file in os.listdir(PPO_PER_SYMBOL_DIR):
            if file.endswith(".zip") and file.startswith("ppo_"):
                symbol = file[len("ppo_"):-len(".zip")]
                self.ppo_models[symbol] = os.path.join(PPO_PER_SYMBOL_DIR, file)
        print(f"   ✅ Found {len(self.ppo_models)} PPO models")

    def get_xgb_prediction(self, symbol, features_dict=None):
        if symbol not in self.xgb_models or not features_dict:
            return None
        try:
            feats = []
            for col in self.FEATURE_ORDER:
                val = features_dict.get(col, 0)
                feats.append(0 if pd.isna(val) else val)
            prob = float(self.xgb_models[symbol].predict_proba(
                np.array(feats).reshape(1, -1))[0, 1])
            return {
                "prob_up": prob,
                "signal": "BUY" if prob > 0.55 else "SELL" if prob < 0.45 else "NEUTRAL",
                "confidence": prob, "source": "XGBoost",
            }
        except Exception:
            return None

    def get_ppo_signal(self, symbol):
        if symbol in self.ppo_models:
            return {"exists": True, "model_path": self.ppo_models[symbol], "source": "PPO"}
        return None


# =========================================================
# TRAINER / DATASET / COLLATOR
# =========================================================

class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        weights = inputs.pop("weight", None)

        if weights is None:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch)

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_tok = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        mask = (shift_labels != -100).float()
        per_seq = (per_tok * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        loss = (per_seq * weights.to(per_seq.device, per_seq.dtype)).mean()

        return (loss, outputs) if return_outputs else loss


class StructuredDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, weights=None):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        n = len(self.input_ids)
        self.weights = (torch.tensor(weights, dtype=torch.float32)
                        if weights is not None else torch.ones(n))

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "weight": self.weights[idx],
        }

    def __len__(self):
        return len(self.input_ids)


def lm_collate(features):
    input_ids = torch.stack([f["input_ids"] for f in features])
    attention_mask = torch.stack([f["attention_mask"] for f in features])
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    weight = torch.stack([f["weight"] for f in features]).float()
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "labels": labels, "weight": weight}


class HFCheckpointCallback(TrainerCallback):
    def __init__(self, hf_uploader, marker):
        self.hf_uploader = hf_uploader
        self.marker = marker
        self._last_upload = 0.0

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(ckpt):
                write_marker(ckpt, self.marker)
                if time.time() - self._last_upload >= HF_CKPT_MIN_INTERVAL_SEC:
                    print(f"\n   📤 Uploading checkpoint {state.global_step} to HF...")
                    if self.hf_uploader.upload_checkpoint(ckpt):
                        self._last_upload = time.time()
                        self.hf_uploader.upload_tracking_files()
        return control


class LossMonitorCallback(TrainerCallback):
    def __init__(self, warn_above=4.0):
        self.warn_above = warn_above

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            if loss != loss:
                print(f"   🚨 Loss is NaN at step {state.global_step}")
            elif state.global_step >= 10 and loss > self.warn_above:
                print(f"   ⚠️ High loss {loss:.3f} at step {state.global_step}")
        return control


# =========================================================
# LABELS / MISTAKES
# =========================================================

class LabelExtractor:
    @staticmethod
    def extract_signal(text):
        t = text.lower()
        for kw in ("buy", "bullish", "long"):
            if kw in t:
                return 1
        for kw in ("sell", "bearish", "short"):
            if kw in t:
                return 0
        return 2

    @staticmethod
    def extract_confidence(text):
        m = re.search(CONFIDENCE_PATTERN, text)
        return float(m.group(1)) / 100.0 if m else 0.5


class MistakeCollector:
    SIGNAL_MAP = {1: "BUY", 0: "SELL", 2: "HOLD"}

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
                self.mistakes = pd.read_csv(MISTAKES_FILE).to_dict("records")
                print(f"   ✅ Loaded {len(self.mistakes)} past mistakes")
            except Exception as e:
                print(f"   ⚠️ Could not load mistakes: {e}")

    def load_hard_examples(self):
        if os.path.exists(HARD_EXAMPLES_FILE):
            try:
                self.hard_examples = pd.read_csv(HARD_EXAMPLES_FILE).to_dict("records")
            except Exception as e:
                print(f"   ⚠️ Could not load hard examples: {e}")

    def add_mistake(self, symbol, prediction, actual, confidence, pattern, market_regime=""):
        mistake = {
            "symbol": symbol, "timestamp": datetime.now().isoformat(),
            "prediction": prediction, "actual": actual, "confidence": confidence,
            "pattern": pattern, "market_regime": market_regime,
            "is_hard": confidence < HARD_EXAMPLE_THRESHOLD,
            "is_high_priority": confidence < HIGH_PRIORITY_THRESHOLD and prediction != actual,
            "correct_explanation": self._generate_explanation(pattern, actual, market_regime),
        }
        self.mistakes.append(mistake)
        self.confidence_history.append({"confidence": confidence,
                                        "is_mistake": prediction != actual,
                                        "is_high_priority": mistake["is_high_priority"]})
        if mistake["is_hard"]:
            self.hard_examples.append(mistake)
            self.save_hard_examples()
        self.save_mistakes()

    @staticmethod
    def _generate_explanation(pattern, actual, market_regime):
        explanations = {
            1: "This pattern indicates upward price movement. Entry at breakout, stop loss below support.",
            0: "This pattern indicates downward price movement. Entry at breakdown, stop loss above resistance.",
            2: "This pattern indicates consolidation. Wait for breakout confirmation.",
        }
        return explanations.get(actual, f"The correct signal is {actual}")

    def save_mistakes(self):
        try:
            pd.DataFrame(self.mistakes).to_csv(MISTAKES_FILE, index=False)
        except Exception as e:
            print(f"   ⚠️ Could not save mistakes: {e}")

    def save_hard_examples(self):
        try:
            pd.DataFrame(self.hard_examples).to_csv(HARD_EXAMPLES_FILE, index=False)
        except Exception as e:
            print(f"   ⚠️ Could not save hard examples: {e}")

    def get_hard_examples(self, limit=100, priority_only=False):
        if priority_only:
            examples = [m for m in self.hard_examples if m.get("is_high_priority", False) is True
                        or str(m.get("is_high_priority")).lower() == "true"]
        else:
            examples = list(self.hard_examples)
        examples.sort(key=lambda x: x.get("confidence", 1.0))
        return examples[:limit]

    def get_confidence_stats(self):
        stats = {"avg_confidence": 0.0, "mistake_rate": 0.0,
                 "low_confidence_count": 0, "high_priority_count": 0}
        if self.confidence_history:
            df = pd.DataFrame(self.confidence_history)
            if "confidence" in df.columns:
                stats["avg_confidence"] = float(df["confidence"].mean())
                stats["low_confidence_count"] = int((df["confidence"] < HARD_EXAMPLE_THRESHOLD).sum())
            if "is_mistake" in df.columns:
                stats["mistake_rate"] = float(df["is_mistake"].mean() * 100)
        if self.mistakes:
            stats["high_priority_count"] = sum(
                1 for m in self.mistakes
                if str(m.get("is_high_priority")).lower() == "true")
            if not self.confidence_history:
                confs = [m.get("confidence", 0) for m in self.mistakes
                         if isinstance(m.get("confidence", None), (int, float))]
                if confs:
                    stats["avg_confidence"] = float(np.mean(confs))
        return stats

    def _xgb_line(self, symbol, label="XGBoost Signal"):
        if not self.xgb_ppo:
            return ""
        pred = self.xgb_ppo.get_xgb_prediction(symbol)
        if pred:
            return f"\n{label}: {pred['signal']} (Confidence: {pred['prob_up']:.0%})"
        return ""

    def get_mistake_dataset(self, limit=200):
        texts = []
        for m in self.get_hard_examples(limit=limit):
            ctx = self._xgb_line(m.get("symbol", ""))
            actual = self.SIGNAL_MAP.get(m.get("actual", 2), "HOLD")
            conf = min(95, max(65, int(m.get("confidence", 0.7) * 100 + 10)))
            texts.append(f"""
{DELIM}
Pattern: {m.get('pattern', 'Unknown')}
Symbol: {m.get('symbol')}
Technical Analysis: Pattern detected with {m.get('confidence', 0.5):.0%} confidence{ctx}

Analysis: {m.get('correct_explanation', 'Review the pattern rules')}

Signal: {actual}
Confidence: {conf}
Risk Level: Medium
Timeframe: Short-term
{DELIM}
""")
        return texts


# =========================================================
# AUTO QWEN3 TRAINER
# =========================================================

class AutoQwen3Trainer:
    def __init__(self):
        os.makedirs("./csv", exist_ok=True)
        os.makedirs(LLM_MODEL_DIR, exist_ok=True)
        os.makedirs(QWEN3_LOCAL_CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(AGENTIC_LOOP_LOG_DIR, exist_ok=True)

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

    def notify(self, msg):
        send_telegram_message(msg, self.telegram_token, self.telegram_chat_id)

    def _init_agentic_loop(self):
        try:
            print("\n" + "=" * 60 + "\n🤖 INITIALIZING AGENTIC LOOP\n" + "=" * 60)
            self.agentic_loop = AgenticLoop(xgb_model_dir=XGBOOST_DIR)
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"   ❌ Agentic Loop init failed: {e}")
            self.agentic_loop = None

    def _update_agentic_loop_after_batch(self, batch_num, symbols, eval_loss=None):
        if self.agentic_loop is None:
            return
        try:
            print(f"\n   📊 Agentic Loop: batch {batch_num} feedback...")
            simulated_pnl = 0.02
            if eval_loss is not None:
                simulated_pnl = max(-0.08, min(0.08, -eval_loss * 0.008))
            success = simulated_pnl > 0
            for symbol in symbols:
                self.agentic_loop.after_trade_feedback(
                    {"symbol": symbol, "pnl": simulated_pnl,
                     "success": success, "batch": batch_num})
            summary = self.agentic_loop.get_summary()
            if len(summary) > 0:
                for _, row in summary.iterrows():
                    print(f"      {row['agent']}: {row['accuracy']} accuracy")
            self.agentic_loop.save_decision_log(
                os.path.join(AGENTIC_LOOP_LOG_DIR, f"batch_{batch_num}_log.csv"))
        except Exception as e:
            print(f"   ⚠️ Agentic Loop update failed: {e}")

    def _finalize_agentic_loop(self):
        if self.agentic_loop is None:
            return
        try:
            print("\n" + "=" * 60 + "\n🏆 AGENTIC LOOP FINAL REPORT\n" + "=" * 60)
            summary = self.agentic_loop.get_summary()
            if len(summary) > 0:
                for _, row in summary.iterrows():
                    print(f"   {row['agent']}: {row['accuracy']} accuracy")
            state = {"timestamp": str(datetime.now()), "agents": {}}
            for agent in self.agentic_loop.agents:
                state["agents"][agent.name] = {
                    "accuracy": agent.get_accuracy(),
                    "predictions": agent.total_predictions,
                    "weight": agent.get_dynamic_weight(),
                }
            with open(AGENTIC_LOOP_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"   ⚠️ Final report failed: {e}")

    def load_trained_symbols(self):
        if os.path.exists(TRACKING_FILE):
            try:
                with open(TRACKING_FILE, "r") as f:
                    return json.load(f).get("symbols", [])
            except Exception as e:
                print(f"   ⚠️ Could not read {TRACKING_FILE}: {e}")
        return []

    def save_trained_symbols(self):
        self.trained_symbols = list(dict.fromkeys(self.trained_symbols))
        with open(TRACKING_FILE, "w") as f:
            json.dump({"symbols": self.trained_symbols,
                       "last_updated": datetime.now().isoformat(),
                       "total_trained": len(self.trained_symbols)}, f, indent=2)

    def get_all_symbols_from_mongodb(self, limit=None):
        if not os.path.exists(MARKET_DATA_PATH):
            print(f"❌ Market data not found: {MARKET_DATA_PATH}")
            return []
        df = pd.read_csv(MARKET_DATA_PATH, usecols=["symbol"])
        symbols = df["symbol"].dropna().unique().tolist()
        if limit:
            symbols = symbols[:limit]
        print(f"   Found {len(symbols)} total symbols in mongodb.csv")
        return symbols

    def get_new_symbols(self):
        print("\n🔍 Checking for new symbols...")
        all_symbols = self.get_all_symbols_from_mongodb()
        trained = set(self.trained_symbols)
        new_symbols = [s for s in all_symbols if s not in trained]
        print(f"   Already trained: {len(trained)} | New: {len(new_symbols)}")
        return new_symbols

    @staticmethod
    def classify_example_difficulty(text):
        t = text.lower()
        for kw in ("complex", "multi timeframe", "divergence", "harmonic",
                   "elliott", "smc", "order block", "fvg", "liquidity"):
            if kw in t:
                return "hard"
        for kw in ("triangle", "wedge", "flag", "pennant", "reversal"):
            if kw in t:
                return "medium"
        if len(text) > 1500:
            return "hard"
        if len(text) > 800:
            return "medium"
        return "easy"

    def load_training_data_with_curriculum(self, data_path=None, use_replay=True):
        data_path = data_path or TRAINING_DATA_PATH
        if not os.path.exists(data_path):
            print(f"❌ Training data not found: {data_path}")
            return None, None

        with open(data_path, "r", encoding="utf-8") as f:
            raw = f.read()
        new_texts = [ex.strip() for ex in raw.split(DELIM) if len(ex.strip()) > 100]
        print(f"📊 New examples: {len(new_texts)}")

        if use_replay:
            self.old_training_texts.extend(new_texts)
            self.old_training_texts = self.old_training_texts[-MAX_OLD_EXAMPLES:]
            train_texts = list(self.old_training_texts)
            print(f"   Replay buffer: {len(train_texts)} examples")

            mistake_texts = [t.strip() for t in self.mistake_collector.get_mistake_dataset(limit=300)]
            if mistake_texts:
                mistake_count = min(len(mistake_texts), int(len(train_texts) * 0.25))
                train_texts += mistake_texts[:mistake_count]
                print(f"   Data mix: {len(train_texts) - mistake_count} normal + {mistake_count} mistakes")
        else:
            train_texts = new_texts

        if not train_texts:
            return None, None

        weights = np.ones(len(train_texts))
        counts = {"easy": 0, "medium": 0, "hard": 0}
        for i, text in enumerate(train_texts):
            diff = self.classify_example_difficulty(text)
            counts[diff] += 1
            if "Elliott Wave" in text or "Impulse Wave" in text or "Corrective Wave" in text:
                weights[i] = 5.0
            elif "SMC" in text or "Order Block" in text or "FVG" in text or "Liquidity" in text:
                weights[i] = 4.5
            elif "Harmonic" in text or "Gartley" in text or "Butterfly" in text:
                weights[i] = 4.0
            elif diff == "hard":
                weights[i] = 3.5
            elif diff == "medium":
                weights[i] = 1.5
            else:
                weights[i] = 0.8
        print(f"   Difficulty: {counts['easy']} easy, {counts['medium']} medium, {counts['hard']} hard")
        return train_texts, weights

    def load_model_with_lora(self):
        print("\n🏗️ Loading Qwen3 model...")
        self.model = None
        local_valid = os.path.exists(os.path.join(LLM_MODEL_DIR, "config.json"))

        if local_valid:
            try:
                print(f"   Loading local Qwen3 from {LLM_MODEL_DIR}...")
                self.model = load_causal_lm(LLM_MODEL_DIR)
                self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_DIR)
                print("   ✅ Qwen3 loaded from local")
            except Exception as e:
                print(f"   ⚠️ Local load failed: {e}")
                self.model = None

        if self.model is None:
            print(f"   📥 Loading base Qwen3: {BASE_MODEL}")
            self.model = load_causal_lm(BASE_MODEL)
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            print("   ✅ Base Qwen3 loaded")

        if LORA_AVAILABLE:
            self.model = get_peft_model(self.model, LoraConfig(**LORA_CONFIG))
            print(f"   ✅ LoRA applied (r={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']})")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        base_cfg = self.model.config
        if getattr(base_cfg, "pad_token_id", None) is None:
            base_cfg.pad_token_id = self.tokenizer.pad_token_id
        base_cfg.use_cache = False

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total: {total:,} | Trainable: {trainable:,}")

    def _release_model(self):
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _save_merged_model(self):
        model = self.model
        if LORA_AVAILABLE and hasattr(model, "merge_and_unload"):
            print("🔄 Merging LoRA into Qwen3 base...")
            model = model.merge_and_unload()
            self.model = model

        tmp_dir = LLM_MODEL_DIR + "_tmp"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
        model.config.use_cache = True
        model.save_pretrained(tmp_dir)
        self.tokenizer.save_pretrained(tmp_dir)

        # ✅ Verify
        required = ["config.json", "tokenizer_config.json"]
        missing = [r for r in required if not os.path.exists(os.path.join(tmp_dir, r))]
        weight_files = ["model.safetensors", "pytorch_model.bin"]
        if not any(os.path.exists(os.path.join(tmp_dir, w)) for w in weight_files):
            missing.append("weights")
        if missing:
            raise RuntimeError(f"Merge save incomplete: missing {missing}")

        shutil.rmtree(LLM_MODEL_DIR, ignore_errors=True)
        os.replace(tmp_dir, LLM_MODEL_DIR)
        print(f"✅ Merged Qwen3 saved to {LLM_MODEL_DIR}")

    @staticmethod
    def _ckpt_step(path):
        try:
            with open(os.path.join(path, "trainer_state.json"), "r") as f:
                return int(json.load(f).get("global_step", 0))
        except Exception:
            return None

    def find_matching_checkpoint(self, marker):
        candidates = (glob.glob(os.path.join(QWEN3_LOCAL_CHECKPOINT_DIR, "checkpoint-*")) +
                      [os.path.join(QWEN3_LOCAL_CHECKPOINT_DIR, "current")])
        best, best_step = None, -1
        for p in candidates:
            if not os.path.isdir(p):
                continue
            if read_marker(p) != marker:
                continue
            step = self._ckpt_step(p)
            if step is None:
                continue
            if step > best_step:
                best, best_step = p, step
        return best

    def save_training_status_file(self, mode, symbols_batch):
        try:
            with open(STATUS_FILE, "w") as f:
                json.dump({
                    "model": "qwen3", "mode": mode,
                    "mode_description": MODE_EXPLANATION.get(mode, mode.upper()),
                    "mode_short": MODE_SHORT_EXPLANATION.get(mode, mode.upper()),
                    "start_time": datetime.now().isoformat(),
                    "symbols_count": len(symbols_batch) if symbols_batch else 0,
                    "epochs": EPOCHS_CONFIG.get(mode, 1),
                    "learning_rate": LR_CONFIG.get(mode, 2e-4),
                }, f, indent=2)
        except Exception as e:
            print(f"   ⚠️ Could not save status file: {e}")

    def _mark_status_completed(self):
        try:
            with open(STATUS_FILE, "r") as f:
                st = json.load(f)
            st["completed_at"] = datetime.now().isoformat()
            st["status"] = "completed"
            with open(STATUS_FILE, "w") as f:
                json.dump(st, f, indent=2)
        except Exception:
            pass

    def train(self, mode="incremental", symbols_batch=None, data_path=None, use_replay=True):
        """Returns (success: bool, eval_loss: float | None)."""
        self.save_training_status_file(mode, symbols_batch)
        mode_explanation = MODE_EXPLANATION.get(mode, f"Mode: {mode.upper()}")

        self.notify(f"""
🚀 <b>Qwen3 Training Started</b>
📅 {datetime.now():%Y-%m-%d %H:%M}
🎯 Mode: {esc(mode.upper())} - {esc(mode_explanation)}
📚 Symbols: {len(symbols_batch) if symbols_batch else 'ALL'}
⚙️ Epochs: {EPOCHS_CONFIG.get(mode, 1)}
""")

        print(f"\n{'=' * 60}\n🎯 QWEN3 TRAINING MODE: {mode.upper()}\n🔍 {mode_explanation}")
        if symbols_batch:
            print(f"📚 Symbols in this batch: {len(symbols_batch)}")
        print("=" * 60)

        train_texts, example_weights = self.load_training_data_with_curriculum(
            data_path, use_replay)
        if not train_texts:
            print("❌ No training data found!")
            return False, None

        self.load_model_with_lora()

        encodings = self.tokenizer(
            train_texts, truncation=True, padding="max_length",
            max_length=MAX_LENGTH, return_tensors="pt")
        full_dataset = StructuredDataset(encodings, example_weights)

        # ✅ Random split + CAP val size
        n = len(full_dataset)
        val_size = max(1, int(n * VALIDATION_SPLIT_RATIO)) if n >= 10 else 0
        val_size = min(val_size, MAX_EVAL_SAMPLES)   # ✅ cap at 100
        perm = np.random.RandomState(SPLIT_SEED).permutation(n).tolist()
        val_idx, train_idx = perm[:val_size], perm[val_size:]
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx) if val_size else None
        print(f"   Dataset split: {len(train_idx)} train, {val_size} validation")

        num_epochs = EPOCHS_CONFIG.get(mode, 1)
        learning_rate = LR_CONFIG.get(mode, 2e-4)
        batch_size = BATCH_SIZE_CONFIG.get(mode, 1)
        grad_accum = GRAD_ACCUM_CONFIG.get(mode, 24)

        steps_per_epoch = max(1, -(-len(train_idx) // (batch_size * grad_accum)))
        total_opt_steps = steps_per_epoch * num_epochs
        warmup_steps = max(1, min(100, int(total_opt_steps * 0.05)))
        save_steps = max(1, min(SAVE_STEPS, total_opt_steps // 3))

        # ✅ Eval every EVAL_STEPS (not too frequent)
        eval_steps = max(EVAL_STEPS, min(save_steps * 2, total_opt_steps // 2))
        print(f"   Optimizer steps: ~{total_opt_steps} (warmup {warmup_steps}, save/eval every {save_steps}/{eval_steps})")

        print(f"\n⚙️ Config: epochs={num_epochs}, lr={learning_rate}, "
              f"batch={batch_size}x{grad_accum}, max_len={MAX_LENGTH}")

        marker = make_marker(mode, symbols_batch)
        last_checkpoint = self.find_matching_checkpoint(marker)
        if last_checkpoint:
            print(f"   ✅ Resuming from matching checkpoint: {last_checkpoint}")
            self.notify(f"🔄 <b>Resuming Qwen3</b>\n📂 {esc(last_checkpoint)}\n🎯 {esc(mode.upper())}")
        else:
            print("   ℹ️ No matching checkpoint - starting fresh (stale checkpoints cleared)")
            clear_dir_contents(QWEN3_LOCAL_CHECKPOINT_DIR)

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

        ta_kwargs = dict(
            output_dir=QWEN3_LOCAL_CHECKPOINT_DIR,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=0.025,
            lr_scheduler_type="cosine",
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,
            logging_steps=10,
            fp16=False,
            bf16=use_bf16,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="none",
            max_grad_norm=MAX_GRAD_NORM,
            optim="adamw_torch",
            adam_beta1=0.9, adam_beta2=0.98, adam_epsilon=1e-8,
            seed=SPLIT_SEED,
        )
        if val_subset is not None:
            ta_kwargs[eval_key] = "steps"
            ta_kwargs["eval_steps"] = eval_steps
        training_args = TrainingArguments(**ta_kwargs)

        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            data_collator=lm_collate,
        )
        trainer.add_callback(HFCheckpointCallback(self.hf_uploader, marker))
        trainer.add_callback(LossMonitorCallback())
        # ✅ CRITICAL: Stop at 5.5h to avoid GitHub 6h timeout
        trainer.add_callback(TimeLimitCallback(max_hours=MAX_TRAINING_HOURS))

        print("\n🏋️ Starting Qwen3 Training...")
        print(f"   ⏰ Auto-stop at {MAX_TRAINING_HOURS}h (GitHub 6h limit safety)")
        train_success = False
        try:
            try:
                trainer.train(resume_from_checkpoint=last_checkpoint)
                train_success = True
            except ValueError as e:
                err = str(e)
                if last_checkpoint and ("parameter group" in err or "optimizer" in err.lower()):
                    print("\n   ⚠️ Optimizer mismatch - removing optimizer/scheduler and retrying")
                    for fname in ("optimizer.pt", "scheduler.pt"):
                        p = os.path.join(last_checkpoint, fname)
                        if os.path.exists(p):
                            os.remove(p)
                    trainer.train(resume_from_checkpoint=last_checkpoint)
                    train_success = True
                else:
                    raise
        except Exception as e:
            self.notify(f"""
⚠️ <b>Qwen3 Training Error</b>
📅 {datetime.now():%Y-%m-%d %H:%M}
🎯 Mode: {esc(mode.upper())}
❌ Error: {esc(e, 200)}
""")
            self._release_model()
            raise

        print("\n✅ Qwen3 training loop ended!")

        # ✅ Check if time-stopped (partial training)
        final_step = trainer.state.global_step
        max_steps = trainer.state.max_steps
        partial = final_step < max_steps
        if partial:
            print(f"\n   ⏸️ PARTIAL training: step {final_step}/{max_steps}")
            print(f"   💾 Checkpoint saved for next run (resume marker in checkpoint)")
            self.notify(f"""
⏸️ <b>Qwen3 Training Paused (Time Limit)</b>
📅 {datetime.now():%Y-%m-%d %H:%M}
🎯 Mode: {esc(mode.upper())}
📊 Step: {final_step}/{max_steps}
💾 Next run will resume automatically
""")
            self._release_model()
            return True, None   # partial success (don't save merged model yet)

        # Full training complete
        eval_loss = None
        if val_subset is not None:
            try:
                eval_loss = float(trainer.evaluate().get("eval_loss"))
                print(f"   📉 Final eval loss: {eval_loss:.4f}")
            except Exception as e:
                print(f"   ⚠️ Final evaluation failed: {e}")

        try:
            self._save_merged_model()
        except Exception as e:
            self.notify(f"⚠️ <b>Qwen3 save failed</b>\n❌ {esc(e, 200)}")
            self._release_model()
            raise

        try:
            with open(os.path.join(LLM_MODEL_DIR, "train_info.json"), "w") as f:
                json.dump({"mode": mode, "eval_loss": eval_loss,
                           "trained_at": datetime.now().isoformat()}, f, indent=2)
        except Exception:
            pass

        self.upload_final_model_to_hf()

        clear_dir_contents(QWEN3_LOCAL_CHECKPOINT_DIR)
        self.hf_uploader.clear_hf_checkpoints()

        self._release_model()

        self.notify(f"""
✅ <b>Qwen3 Training Completed</b>
📅 {datetime.now():%Y-%m-%d %H:%M}
🎯 {esc(mode_explanation)}
📚 Symbols: {len(symbols_batch) if symbols_batch else 'ALL'}
📉 Eval loss: {f'{eval_loss:.4f}' if eval_loss is not None else 'n/a'}
💾 Model: {esc(LLM_MODEL_DIR)}
""")
        self._mark_status_completed()
        return True, eval_loss

    def upload_final_model_to_hf(self):
        if not get_hf_token():
            print("ℹ️ No HF token, skipping final model upload")
            return
        print(f"\n📤 Uploading final model → {HF_DATASET_REPO}/{QWEN3_HF_FINAL_PATH}/")
        if self.hf_uploader.upload_final_model(LLM_MODEL_DIR):
            self.hf_uploader.upload_tracking_files()

    def generate_training_data_for_symbols(self, symbols):
        print(f"\n📝 Generating training data for {len(symbols)} symbols...")
        result = subprocess.run(
            [sys.executable, "scripts/generate_pattern_training_data_complete.py",
             "--symbols", ",".join(symbols)],
            capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ⚠️ Data generation failed: {result.stderr[-500:]}")
            return False
        print("   ✅ Training data generated")
        return True

    def run(self):
        print("=" * 60 + "\n🚀 AUTO QWEN3 TRAINER (v6 — TIME SAFE)\n" + "=" * 60)
        print(f"📅 {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"🧠 Model: {BASE_MODEL}")
        print(f"⏰ Auto-stop: {MAX_TRAINING_HOURS}h")
        print("=" * 60)

        stats = self.mistake_collector.get_confidence_stats()
        print(f"\n📊 Confidence: avg={stats['avg_confidence']:.2%}, "
              f"mistake rate={stats['mistake_rate']:.2f}%")

        new_symbols = self.get_new_symbols()
        trained_new = False

        if new_symbols:
            print(f"\n📚 {len(new_symbols)} new symbols to train")
            for i in range(0, len(new_symbols), BATCH_SIZE):
                batch = new_symbols[i:i + BATCH_SIZE]
                batch_num = self.batch_manager.next_batch_number()
                print(f"\n📦 Batch {batch_num}: {len(batch)} symbols")

                if not self.generate_training_data_for_symbols(batch):
                    continue
                mode = "first_train" if len(self.trained_symbols) == 0 else "incremental"
                ok, eval_loss = self.train(mode=mode, symbols_batch=batch)
                if ok and eval_loss is not None:
                    # Full training completed
                    self.trained_symbols.extend(batch)
                    self.save_trained_symbols()
                    self.batch_manager.mark_batch_completed(batch_num, batch)
                    self._update_agentic_loop_after_batch(batch_num, batch, eval_loss)
                    trained_new = True
                    print(f"✅ Batch {batch_num} complete! Total: {len(self.trained_symbols)}")
                elif ok and eval_loss is None:
                    # Time stopped - DO NOT mark batch complete
                    print(f"⏸️ Batch {batch_num} partial - will resume next run")
                    break   # Stop processing more batches this run
            if trained_new:
                self.batch_manager.init_schedule_baselines()
        else:
            print("\n✅ No new symbols found!")

        # Weekly fine-tune (skip if just trained new)
        if not trained_new:
            weekly_num, weekly_symbols = self.batch_manager.get_batch_for_weekly_finetune()
            if weekly_num and weekly_symbols:
                print(f"\n🔄 Weekly fine-tune batch {weekly_num}")
                if self.generate_training_data_for_symbols(weekly_symbols):
                    ok, _ = self.train(mode="weekly_finetune", symbols_batch=weekly_symbols)
                    if ok:
                        self.batch_manager.mark_weekly_done(weekly_num)

        # Monthly consolidation
        if not trained_new and self.batch_manager.should_consolidate():
            all_syms = self.batch_manager.get_all_batch_symbols()
            if all_syms:
                print(f"\n🔄 Monthly consolidation - {len(all_syms)} symbols")
                if self.generate_training_data_for_symbols(all_syms):
                    ok, _ = self.train(mode="consolidate", symbols_batch=all_syms)
                    if ok:
                        self.batch_manager.mark_consolidated()

        # Mistake learning
        high_priority = self.mistake_collector.get_hard_examples(limit=200, priority_only=True)
        if high_priority and self.batch_manager.should_run_mistake_learning():
            print(f"\n🔥 {len(high_priority)} high priority mistakes")
            temp_file = "./csv/temp_hard_examples.txt"
            sig = MistakeCollector.SIGNAL_MAP
            with open(temp_file, "w", encoding="utf-8") as f:
                for ex in high_priority[:100]:
                    ctx = self.mistake_collector._xgb_line(ex.get("symbol", ""), "XGBoost Analysis")
                    actual = sig.get(ex.get("actual", 2), "HOLD")
                    conf = min(95, max(65, int(ex.get("confidence", 0.7) * 100 + 10)))
                    f.write(f"""
{DELIM}
Pattern: {ex.get('pattern', 'Unknown')}
Symbol: {ex.get('symbol')}
Previous Prediction: {sig.get(ex.get('prediction', 2), 'HOLD')}
❌ This was WRONG{ctx}

✅ CORRECT ANSWER: {actual}
Explanation: {ex.get('correct_explanation', 'Review the pattern rules')}

Signal: {actual}
Confidence: {conf}
{DELIM}
""")
            try:
                ok, _ = self.train(mode="mistake_learning", data_path=temp_file, use_replay=False)
                if ok:
                    self.batch_manager.mark_mistake_learning_done()
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        print("\n" + "=" * 60 + "\n📊 FINAL STATUS\n" + "=" * 60)
        print(f"   Total trained: {len(self.trained_symbols)}")
        print(f"   Completed batches: {len(self.batch_manager.completed_batches)}")
        print("=" * 60)

        self._finalize_agentic_loop()


if __name__ == "__main__":
    AutoQwen3Trainer().run()