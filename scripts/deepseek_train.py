# ================== scripts/deepseek_train.py ==================
# DeepSeek-R1-Distill-Qwen-1.5B LoRA Trainer (v4.1)
#
# v4 পরিবর্তন:
#  1. Trainer/dataset স্পষ্টভাবে ছেড়ে GPU মেমরি খালি করা
#  2. সত্যিকারের fp32 merge (আলাদা fp32 base লোড করে, ট্রেনিং মডেল ফ্রি করার পর)
#  3. Hard/mistake example normal ডেটার মতো একই ফরম্যাট (শুধু সঠিক উত্তর, হার্ডকোড ফিল্ড নেই)
#  4. Replay buffer থেকে শুধু নমুনা (ছোট ব্যাচ = দ্রুত weekly)
#  5. mistake_learning-এ cooldown (সপ্তাহে একবার)
#  6. অকেজো on_train_end কলব্যাক বাদ
#  7. pad_token_id `is None` চেক, অব্যবহৃত import বাদ, dtype/torch_dtype ভার্সন-সেফ
#  8. merge ব্যর্থ হলে টেলিগ্রামে আলাদা সতর্কতা

import os
import gc
import sys
import json
import html
import random
import shutil
import warnings
import subprocess
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from huggingface_hub import HfApi, create_repo, login

warnings.filterwarnings("ignore")

# =========================================================
# OPTIONAL DEPENDENCIES
# =========================================================
try:
    from agentic_loop import AgenticLoop
    AGENTIC_LOOP_AVAILABLE = True
except ImportError:
    AGENTIC_LOOP_AVAILABLE = False
    print("⚠️ Agentic Loop not found")

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    print("⚠️ PEFT not installed")


# =========================================================
# CONFIGURATION
# =========================================================
BATCH_SIZE = 40
HF_DATASET_REPO = "ahashanahmed/csv"

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LLM_MODEL_DIR = "./csv/llm_model_deepseek"                 # merged full model
LORA_ADAPTER_DIR = "./csv/llm_model_deepseek_lora"         # continuing adapter
DEEPSEEK_LOCAL_CHECKPOINT_DIR = "./csv/deepseek_checkpoints"

DEEPSEEK_HF_CHECKPOINT_PREFIX = "deepseek_checkpoints/deepseek_checkpoint-"
DEEPSEEK_FINAL_MODEL_PREFIX = "final_model_deepseek"

TRACKING_FILE = "./csv/trained_symbols_deepseek.json"
BATCH_TRACKING_FILE = "./csv/batch_tracking_deepseek.json"
REPLAY_BUFFER_FILE = "./csv/replay_buffer_deepseek.json"

MARKET_DATA_PATH = "./csv/mongodb.csv"
TRAINING_DATA_PATH = "./csv/training_texts.txt"
MISTAKES_FILE = "./csv/trading_mistakes_deepseek.csv"
HARD_EXAMPLES_FILE = "./csv/hard_examples_deepseek.csv"

XGBOOST_DIR = "./csv/xgboost"
PPO_PER_SYMBOL_DIR = "./csv/ppo_models/per_symbol"

AGENTIC_LOOP_LOG_DIR = "./csv/agentic_loop_logs"

CONSOLIDATE_INTERVAL = 30       # days
WEEKLY_INTERVAL = 7             # days
MISTAKE_LEARNING_INTERVAL = 7   # days (cooldown)

MAX_OLD_EXAMPLES = 10000
MAX_VAL_EXAMPLES = 200
MAX_SEQ_LEN = 384
MAX_GRAD_NORM = 1.0
VALIDATION_SPLIT_RATIO = 0.10
MISTAKE_MIX_RATIO = 0.25
MIN_TEXT_LEN = 100              # normal ডেটার ন্যূনতম দৈর্ঘ্য
MIN_OVERRIDE_LEN = 30           # mistake উদাহরণ ছোট, তাই আলাদা সীমা
SEED = 42
SAVE_STEPS = 50

# প্রতি মোডে replay buffer থেকে কতগুলো পুরনো উদাহরণ নমুনা হিসেবে নেওয়া হবে
REPLAY_SAMPLE_SIZE = {
    "first_train": 0,
    "incremental": 1000,
    "weekly_finetune": 1000,
    "consolidate": 3000,
    "mistake_learning": 0,   # texts_override ব্যবহার হয়, তবু স্পষ্টতার জন্য
}

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

EPOCHS_CONFIG = {
    "first_train": 3,
    "incremental": 2,
    "weekly_finetune": 2,
    "consolidate": 3,
    "mistake_learning": 2,
}
LR_CONFIG = {
    "first_train": 2e-4,
    "incremental": 1e-4,
    "weekly_finetune": 5e-5,
    "consolidate": 1e-4,
    "mistake_learning": 5e-5,
}
BATCH_SIZE_CONFIG = {k: 1 for k in EPOCHS_CONFIG}
GRAD_ACCUM_CONFIG = {
    "first_train": 32,
    "incremental": 24,
    "weekly_finetune": 16,
    "consolidate": 32,
    "mistake_learning": 16,
}

MODE_EXPLANATION = {
    "first_train": "🎯 FIRST TIME TRAINING (DeepSeek 1.5B)",
    "incremental": "⚙️ INCREMENTAL TRAINING (New symbols)",
    "weekly_finetune": "🔄 WEEKLY FINE-TUNE",
    "consolidate": "📈 MONTHLY RE-TUNE",
    "mistake_learning": "🎯 MISTAKE LEARNING",
}

SIGNAL_MAP = {1: "BUY", 0: "SELL", 2: "HOLD"}

# ✅ Version-safe argument names
_TF_VERSION = tuple(int(x) for x in transformers.__version__.split(".")[:2])
_EVAL_ARG = "eval_strategy" if _TF_VERSION >= (4, 46) else "evaluation_strategy"
_DTYPE_ARG = "dtype" if _TF_VERSION >= (4, 56) else "torch_dtype"
print(f"ℹ️ Transformers {transformers.__version__} → '{_EVAL_ARG}', '{_DTYPE_ARG}'")


# =========================================================
# TELEGRAM
# =========================================================
def send_telegram_message(message, token=None, chat_id=None):
    token = token or os.getenv("TELEGRAM_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return None
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        return requests.post(url, json=payload, timeout=10).json()
    except Exception as e:
        print(f"⚠️ Telegram failed: {e}")
        return None


def esc(text, limit=200):
    """Telegram HTML-safe text."""
    return html.escape(str(text)[:limit])


# =========================================================
# HF UPLOADER
# =========================================================
class HFUploader:
    def __init__(self, repo_id=HF_DATASET_REPO):
        self.repo_id = repo_id
        self.api = None
        self._init_api()

    def _init_api(self):
        token = os.getenv("HF_TOKEN") or os.getenv("hf_token")
        if not token:
            print("   ℹ️ HF token not set - uploads disabled")
            return
        try:
            login(token=token)
            self.api = HfApi(token=token)
            create_repo(repo_id=self.repo_id, repo_type="dataset", exist_ok=True)
            print(f"   ✅ HF Dataset Repo ready: {self.repo_id}")
        except Exception as e:
            print(f"   ⚠️ HF API init failed: {e}")
            self.api = None

    def _upload_folder(self, folder, repo_path, message, ignore=None):
        if self.api is None:
            return False
        try:
            self.api.upload_folder(
                folder_path=folder,
                path_in_repo=repo_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=message,
                ignore_patterns=ignore,
            )
            print(f"   📤 {folder} → {self.repo_id}/{repo_path}")
            return True
        except Exception as e:
            print(f"   ⚠️ Upload failed ({repo_path}): {e}")
            return False

    def upload_checkpoint(self, checkpoint_path, step_num, mode):
        return self._upload_folder(
            checkpoint_path,
            f"{DEEPSEEK_HF_CHECKPOINT_PREFIX}{mode}-{step_num}",
            f"🧠 DeepSeek {mode} checkpoint {step_num} - {datetime.now():%Y-%m-%d %H:%M}",
            ignore=["optimizer.pt", "scheduler.pt", "rng_state*.pth"],
        )

    def upload_final_model(self, model_path, mode):
        return self._upload_folder(
            model_path,
            f"{DEEPSEEK_FINAL_MODEL_PREFIX}/{mode}",
            f"🧠 DeepSeek Final ({mode}) - {datetime.now():%Y-%m-%d %H:%M}",
        )

    def upload_adapter(self, adapter_path, mode):
        return self._upload_folder(
            adapter_path,
            f"{DEEPSEEK_FINAL_MODEL_PREFIX}/{mode}_adapter",
            f"🧠 DeepSeek adapter ({mode}) - {datetime.now():%Y-%m-%d %H:%M}",
        )

    def upload_adapter_latest(self, adapter_path):
        """নির্দিষ্ট পাথ (মোড ছাড়া) — CI-তে পরের রানে সহজে ডাউনলোডের জন্য।"""
        return self._upload_folder(
            adapter_path,
            f"{DEEPSEEK_FINAL_MODEL_PREFIX}/adapter_latest",
            f"🧠 DeepSeek adapter latest - {datetime.now():%Y-%m-%d %H:%M}",
        )

    def upload_tracking_files(self):
        if self.api is None:
            return
        for local, remote in [
            (TRACKING_FILE, "trained_symbols_deepseek.json"),
            (BATCH_TRACKING_FILE, "batch_tracking_deepseek.json"),
            (REPLAY_BUFFER_FILE, "replay_buffer_deepseek.json"),
        ]:
            if not os.path.exists(local):
                continue
            try:
                self.api.upload_file(
                    path_or_fileobj=local,
                    path_in_repo=remote,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"Update {remote} - {datetime.now():%Y-%m-%d %H:%M}",
                )
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

    def _default(self):
        return {
            "current_batch": 0,
            "completed_batches": [],
            "batch_symbols": {},
            "total_symbols_trained": 0,
            "last_batch_date": None,
            "weekly_trained_batches": [],
            "last_weekly_finetune": None,
            "last_consolidate": None,
            "last_mistake_learning": None,
            "monthly_consolidation_done": False,
        }

    def _ensure_required_fields(self):
        updated = False
        for k, v in self._default().items():
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
            except (OSError, json.JSONDecodeError) as e:
                print(f"⚠️ Batch tracking unreadable, resetting: {e}")
        return self._default()

    def save_batch_tracking(self):
        os.makedirs(os.path.dirname(BATCH_TRACKING_FILE), exist_ok=True)
        with open(BATCH_TRACKING_FILE, "w") as f:
            json.dump(self.batch_tracking, f, indent=2)

    def next_batch_number(self):
        return (max(self.completed_batches) + 1) if self.completed_batches else 1

    def mark_batch_completed(self, batch_num, symbols):
        if batch_num in self.completed_batches:
            return
        self.completed_batches.append(batch_num)
        self.batch_symbols[str(batch_num)] = list(symbols)
        self.batch_tracking["current_batch"] = batch_num
        self.batch_tracking["total_symbols_trained"] += len(symbols)
        self.batch_tracking["last_batch_date"] = datetime.now().isoformat()
        if not self.batch_tracking.get("last_weekly_finetune"):
            self.batch_tracking["last_weekly_finetune"] = datetime.now().isoformat()
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

        last = self.batch_tracking.get("last_weekly_finetune")
        days = self._days_since(last) if last else None
        if days is not None and days < WEEKLY_INTERVAL:
            return None, []

        trained = set(self.batch_tracking.get("weekly_trained_batches", []))
        available = [b for b in self.completed_batches if b not in trained]
        if not available:
            self.batch_tracking["weekly_trained_batches"] = []
            self.save_batch_tracking()
            available = list(self.completed_batches)

        batch_num = available[0]
        return batch_num, self.batch_symbols.get(str(batch_num), [])

    def mark_weekly_done(self, batch_num):
        weekly = self.batch_tracking.get("weekly_trained_batches", [])
        if batch_num not in weekly:
            weekly.append(batch_num)
        self.batch_tracking["weekly_trained_batches"] = weekly
        self.batch_tracking["last_weekly_finetune"] = datetime.now().isoformat()
        self.save_batch_tracking()

    def should_consolidate(self):
        last = self.batch_tracking.get("last_consolidate")
        if not last:
            self.batch_tracking["last_consolidate"] = datetime.now().isoformat()
            self.save_batch_tracking()
            return False
        days = self._days_since(last)
        return days is None or days >= CONSOLIDATE_INTERVAL

    def mark_consolidated(self):
        now = datetime.now().isoformat()
        self.batch_tracking["last_consolidate"] = now
        self.batch_tracking["monthly_consolidation_done"] = True
        self.batch_tracking["weekly_trained_batches"] = []
        self.batch_tracking["last_weekly_finetune"] = now
        self.save_batch_tracking()

    def should_run_mistake_learning(self):
        last = self.batch_tracking.get("last_mistake_learning")
        if not last:
            return True
        days = self._days_since(last)
        return days is None or days >= MISTAKE_LEARNING_INTERVAL

    def mark_mistake_learning_done(self):
        self.batch_tracking["last_mistake_learning"] = datetime.now().isoformat()
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
            return
        for file in os.listdir(XGBOOST_DIR):
            if file.endswith(".joblib"):
                try:
                    self.xgb_models[file[:-7]] = joblib.load(os.path.join(XGBOOST_DIR, file))
                except Exception as e:
                    print(f"   ⚠️ XGB load failed ({file}): {e}")
        print(f"   ✅ Loaded {len(self.xgb_models)} XGBoost models")

    def load_ppo_metadata(self):
        if not os.path.exists(PPO_PER_SYMBOL_DIR):
            return
        for file in os.listdir(PPO_PER_SYMBOL_DIR):
            if file.endswith(".zip") and file.startswith("ppo_"):
                self.ppo_models[file[4:-4]] = os.path.join(PPO_PER_SYMBOL_DIR, file)
        print(f"   ✅ Found {len(self.ppo_models)} PPO models")

    def get_xgb_prediction(self, symbol, features_dict=None):
        model = self.xgb_models.get(symbol)
        if model is None:
            return None
        try:
            if features_dict:
                feats = []
                for c in self.FEATURE_ORDER:
                    v = features_dict.get(c, 0)
                    feats.append(0 if pd.isna(v) else v)
                prob = float(model.predict_proba(np.array(feats).reshape(1, -1))[0, 1])
            else:
                prob = 0.5
            return {
                "prob_up": prob,
                "signal": "BUY" if prob > 0.55 else "SELL" if prob < 0.45 else "NEUTRAL",
                "confidence": prob,
                "source": "XGBoost",
            }
        except Exception as e:
            print(f"   ⚠️ XGB predict failed ({symbol}): {e}")
            return None


# =========================================================
# DATASET / COLLATOR / TRAINER
# =========================================================
class TokenDataset(torch.utils.data.Dataset):
    """Unpadded token ids + per-example weight."""

    def __init__(self, input_ids, weights):
        self.input_ids = input_ids
        self.weights = [float(w) for w in weights]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "weight": self.weights[idx]}


class PadCollator:
    """Dynamic padding; pad positions get label -100."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        ids, mask, labels = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            ids.append(f["input_ids"] + [self.pad_id] * pad)
            mask.append([1] * n + [0] * pad)
            labels.append(f["input_ids"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "weight": torch.tensor([f["weight"] for f in features], dtype=torch.float32),
        }


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        weights = inputs.pop("weight", None)

        outputs = model(**inputs)
        logits = outputs.logits[..., :-1, :].float()
        shift_labels = labels[..., 1:].to(logits.device)

        tok_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)

        mask = (shift_labels != -100).float()
        per_sample = (tok_loss * mask).sum(1) / mask.sum(1).clamp(min=1)

        if weights is None:
            loss = per_sample.mean()
        else:
            w = weights.to(per_sample.device, per_sample.dtype)
            loss = (per_sample * w).sum() / w.sum().clamp(min=1e-8)

        return (loss, outputs) if return_outputs else loss


class HFUploadCallback(TrainerCallback):
    """প্রতি save_steps-এ checkpoint HF-এ ব্যাকআপ (optimizer ছাড়া)।"""

    def __init__(self, uploader, mode):
        self.uploader = uploader
        self.mode = mode

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.exists(ckpt):
                print(f"\n   📤 Uploading checkpoint {state.global_step} to HF...")
                self.uploader.upload_checkpoint(ckpt, state.global_step, self.mode)
        return control


# =========================================================
# EXAMPLE FORMAT + MISTAKE COLLECTOR
# =========================================================
def _present(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip() != ""


def _clean(v, default):
    """None / NaN / 'nan' / খালি স্ট্রিং → default।"""
    if not _present(v) or str(v).strip().lower() in ("none", "nan"):
        return default
    return str(v).strip()


def format_example(m):
    """Normal ডেটার মতো একই ধরন: শুধু সঠিক উত্তর, কোনো 'WRONG' লেবেল বা হার্ডকোড ফিল্ড নেই।
    risk_level/timeframe থাকলে তবেই যোগ হয়।"""
    try:
        actual = int(m.get("actual", 2))
    except (TypeError, ValueError):
        actual = 2
    try:
        conf = float(m.get("confidence", 0.7))
    except (TypeError, ValueError):
        conf = 0.7

    lines = [
        f"Pattern: {_clean(m.get('pattern'), 'Unknown')}",
        f"Symbol: {_clean(m.get('symbol'), 'UNKNOWN')}",
        f"Signal: {SIGNAL_MAP.get(actual, 'HOLD')}",
        f"Confidence: {min(95, max(65, int(conf * 100 + 10)))}",
    ]
    if _present(m.get("risk_level")):
        lines.append(f"Risk Level: {m['risk_level']}")
    if _present(m.get("timeframe")):
        lines.append(f"Timeframe: {m['timeframe']}")
    return "\n".join(lines)


class MistakeCollector:
    def __init__(self, xgb_ppo_integrator=None):
        self.mistakes = []
        self.confidence_history = []
        self.hard_examples = []
        self.xgb_ppo = xgb_ppo_integrator
        self.load_mistakes()
        self.load_hard_examples()

    def _read_csv(self, path):
        if not os.path.exists(path):
            return []
        try:
            return pd.read_csv(path).to_dict("records")
        except Exception as e:
            print(f"   ⚠️ Cannot read {path}: {e}")
            return []

    def load_mistakes(self):
        self.mistakes = self._read_csv(MISTAKES_FILE)
        if self.mistakes:
            print(f"   ✅ Loaded {len(self.mistakes)} past mistakes")

    def load_hard_examples(self):
        self.hard_examples = self._read_csv(HARD_EXAMPLES_FILE)

    def get_hard_examples(self, limit=100, priority_only=False):
        if priority_only:
            ex = [m for m in self.hard_examples if m.get("is_high_priority", False)]
        else:
            ex = list(self.hard_examples)
        ex.sort(key=lambda x: x.get("confidence", 1.0))
        return ex[:limit]

    def get_confidence_stats(self):
        empty = {"avg_confidence": 0, "mistake_rate": 0, "high_priority_count": 0}
        if not self.confidence_history:
            return empty
        df = pd.DataFrame(self.confidence_history)
        hp = int(df["is_high_priority"].sum()) if "is_high_priority" in df.columns else 0
        return {
            "avg_confidence": df["confidence"].mean() if "confidence" in df.columns else 0,
            "mistake_rate": df["is_mistake"].mean() * 100 if "is_mistake" in df.columns else 0,
            "high_priority_count": hp,
        }

    def get_mistake_dataset(self, limit=200, priority_only=False):
        return [format_example(m) for m in self.get_hard_examples(limit, priority_only)]


# =========================================================
# MAIN TRAINER
# =========================================================
class AutoDeepSeekTrainer:
    def __init__(self):
        for d in ("./csv", LLM_MODEL_DIR, LORA_ADAPTER_DIR,
                  DEEPSEEK_LOCAL_CHECKPOINT_DIR, AGENTIC_LOOP_LOG_DIR):
            os.makedirs(d, exist_ok=True)

        self.trained_symbols = self.load_trained_symbols()
        self.model = None
        self.tokenizer = None
        self.xgb_ppo = XGBoostPPOIntegrator()
        self.mistake_collector = MistakeCollector(self.xgb_ppo)
        self.batch_manager = BatchManager()
        self.replay_buffer = self.load_replay_buffer()
        self.hf_uploader = HFUploader()

        self.agentic_loop = None
        if AGENTIC_LOOP_AVAILABLE:
            self._init_agentic_loop()

        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if self.telegram_token and self.telegram_chat_id:
            print("✅ Telegram notifications enabled")

    # ---------- helpers ----------
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

    def load_trained_symbols(self):
        if os.path.exists(TRACKING_FILE):
            try:
                with open(TRACKING_FILE, "r") as f:
                    return json.load(f).get("symbols", [])
            except (OSError, json.JSONDecodeError):
                return []
        return []

    def save_trained_symbols(self):
        self.trained_symbols = list(dict.fromkeys(self.trained_symbols))
        with open(TRACKING_FILE, "w") as f:
            json.dump({
                "symbols": self.trained_symbols,
                "last_updated": datetime.now().isoformat(),
                "total_trained": len(self.trained_symbols),
            }, f, indent=2)

    def load_replay_buffer(self):
        if os.path.exists(REPLAY_BUFFER_FILE):
            try:
                with open(REPLAY_BUFFER_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        return []

    def save_replay_buffer(self):
        with open(REPLAY_BUFFER_FILE, "w", encoding="utf-8") as f:
            json.dump(self.replay_buffer, f, ensure_ascii=False)

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

    def get_new_symbols(self, all_symbols=None):
        print("\n🔍 Checking for new symbols...")
        all_symbols = all_symbols if all_symbols is not None else self.get_all_symbols_from_mongodb()
        trained = set(self.trained_symbols)
        new = [s for s in all_symbols if s not in trained]
        print(f"   Already trained: {len(trained)} | New: {len(new)}")
        return new

    # ---------- data ----------
    @staticmethod
    def classify_example_difficulty(text):
        low = text.lower()
        for kw in ("complex", "multi timeframe", "divergence", "harmonic",
                   "elliott", "smc", "order block", "fvg", "liquidity"):
            if kw in low:
                return "hard"
        for kw in ("triangle", "wedge", "flag", "pennant", "reversal"):
            if kw in low:
                return "medium"
        n = len(text)
        return "hard" if n > 1500 else "medium" if n > 800 else "easy"

    def example_weight(self, text):
        if "Elliott Wave" in text or "Impulse Wave" in text:
            return 5.0
        if "SMC" in text or "Order Block" in text or "FVG" in text:
            return 4.5
        if "Harmonic" in text or "Gartley" in text:
            return 4.0
        d = self.classify_example_difficulty(text)
        return {"hard": 3.5, "medium": 1.5, "easy": 0.8}[d]

    def read_new_examples(self, path):
        if not os.path.exists(path):
            print(f"❌ Training data not found: {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        return [ex.strip() for ex in raw.split("=" * 80) if len(ex.strip()) > MIN_TEXT_LEN]

    def build_texts(self, mode, data_path, texts_override=None):
        """এই রানের ট্রেনিং টেক্সট: নতুন ডেটা + replay নমুনা + mistake মিক্স।"""
        if texts_override is not None:
            filtered = [t for t in texts_override if len(t) >= MIN_OVERRIDE_LEN]
            dropped = len(texts_override) - len(filtered)
            if dropped:
                print(f"   ⚠️ Filtered {dropped} too-short override texts")
            return filtered

        new_texts = self.read_new_examples(data_path)
        print(f"📊 New examples: {len(new_texts)}")
        if not new_texts:
            return []

        # Replay buffer আপডেট (dedup + cap)
        merged = list(dict.fromkeys(self.replay_buffer + new_texts))
        self.replay_buffer = merged[-MAX_OLD_EXAMPLES:]
        self.save_replay_buffer()

        # নতুন ডেটা + পুরনো থেকে ছোট নমুনা (পুরো buffer নয়)
        new_set = set(new_texts)
        old_pool = [t for t in self.replay_buffer if t not in new_set]
        k_old = min(len(old_pool), REPLAY_SAMPLE_SIZE.get(mode, 1000))
        old_sample = random.Random(SEED).sample(old_pool, k_old) if k_old else []
        texts = new_texts + old_sample
        print(f"   Replay sample: {len(old_sample)} old + {len(new_texts)} new")

        # Mistakes মেশানো
        mistakes = self.mistake_collector.get_mistake_dataset(limit=300)
        if mistakes:
            k = min(len(mistakes), int(len(texts) * MISTAKE_MIX_RATIO))
            texts += mistakes[:k]
        return texts

    def make_datasets(self, texts):
        rng = random.Random(SEED)
        texts = list(dict.fromkeys(texts))
        rng.shuffle(texts)

        weights = [self.example_weight(t) for t in texts]
        enc = self.tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN, padding=False)
        ids = enc["input_ids"]

        n = len(ids)
        val_n = min(MAX_VAL_EXAMPLES, int(n * VALIDATION_SPLIT_RATIO))
        if n < 20:
            val_n = 0
        train_ds = TokenDataset(ids[val_n:], weights[val_n:])
        val_ds = TokenDataset(ids[:val_n], weights[:val_n]) if val_n > 0 else None
        print(f"   Train: {len(train_ds)} | Val: {len(val_ds) if val_ds else 0}")
        return train_ds, val_ds

    # ---------- model ----------
    @staticmethod
    def _dtype():
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32

    def load_model_with_lora(self, mode):
        """Base + LoRA লোড। first_train = fresh; বাকি = আগের adapter থেকে চালিয়ে যায়।"""
        if not LORA_AVAILABLE:
            raise RuntimeError("PEFT not installed")

        self.free_model()
        print(f"\n🏗️ Loading DeepSeek base: {BASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **{_DTYPE_ARG: self._dtype()},
        )
        model.config.use_cache = False

        adapter_exists = os.path.exists(os.path.join(LORA_ADAPTER_DIR, "adapter_config.json"))
        if mode != "first_train" and adapter_exists:
            print(f"   🔁 Continuing from adapter: {LORA_ADAPTER_DIR}")
            try:
                model = PeftModel.from_pretrained(model, LORA_ADAPTER_DIR, is_trainable=True)
                n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
                if n_train == 0:
                    print("   ⚠️ is_trainable did not enable grads → enabling LoRA params manually")
                    for name, p in model.named_parameters():
                        if "lora_" in name.lower():
                            p.requires_grad = True
                    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    if n_train == 0:
                        raise RuntimeError("adapter has no trainable params")
                print(f"   ✅ Adapter loaded ({n_train:,} trainable)")
            except Exception as e:
                print(f"   ⚠️ Adapter load failed, using fresh LoRA: {e}")
                model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
        else:
            print("   🆕 Fresh LoRA adapter")
            model = get_peft_model(model, LoraConfig(**LORA_CONFIG))

        if torch.cuda.is_available():
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()

        # pad token (id 0 হলেও ঠিক থাকে)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        model.config.pad_token_id = self.tokenizer.pad_token_id

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if trainable == 0:
            raise RuntimeError("LoRA applied but 0 trainable params")
        print(f"   📊 Trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

        model.train()
        self.model = model

    def free_model(self):
        self.model = None
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ---------- training ----------
    def save_training_status_file(self, mode, symbols_batch):
        try:
            with open("./csv/current_training_status.json", "w") as f:
                json.dump({
                    "model": "deepseek",
                    "mode": mode,
                    "mode_description": MODE_EXPLANATION.get(mode, mode.upper()),
                    "start_time": datetime.now().isoformat(),
                    "symbols_count": len(symbols_batch) if symbols_batch else 0,
                    "epochs": EPOCHS_CONFIG.get(mode),
                    "learning_rate": LR_CONFIG.get(mode),
                }, f, indent=2)
        except OSError:
            pass

    def _prepare_ckpt_dir(self, mode):
        """একই মোডের ক্র্যাশ-হওয়া রান থাকলে resume, নইলে পরিষ্কার শুরু।"""
        ckpt_dir = os.path.join(DEEPSEEK_LOCAL_CHECKPOINT_DIR, mode)
        marker = os.path.join(ckpt_dir, "IN_PROGRESS")
        resume = None
        if os.path.exists(marker):
            resume = get_last_checkpoint(ckpt_dir)
            if resume:
                print(f"   ✅ Resuming interrupted '{mode}' run from {resume}")
        if resume is None:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(marker, "w") as f:
                f.write(datetime.now().isoformat())
            print("   ℹ️ Fresh run (no interrupted checkpoint)")
        return ckpt_dir, resume

    def _run_trainer(self, trainer, resume):
        try:
            return trainer.train(resume_from_checkpoint=resume)
        except ValueError as e:
            if resume and ("parameter group" in str(e) or "optimizer" in str(e).lower()):
                print("\n   ⚠️ Optimizer state mismatch → removing optimizer/scheduler state")
                for fname in ("optimizer.pt", "scheduler.pt"):
                    p = os.path.join(resume, fname)
                    if os.path.exists(p):
                        os.remove(p)
                self.model.train()
                return trainer.train(resume_from_checkpoint=resume)
            else:
                raise

    def train(self, mode="incremental", symbols_batch=None,
              data_path=TRAINING_DATA_PATH, texts_override=None):
        self.save_training_status_file(mode, symbols_batch)
        explanation = MODE_EXPLANATION.get(mode, mode.upper())

        self.notify(f"""
🚀 <b>DeepSeek Training Started</b>
📅 {datetime.now():%Y-%m-%d %H:%M}
🎯 {esc(mode.upper())} - {esc(explanation)}
📚 Symbols: {len(symbols_batch) if symbols_batch else 'ALL'}
⚙️ Epochs: {EPOCHS_CONFIG.get(mode)}
""")
        print(f"\n{'=' * 60}\n🎯 DEEPSEEK TRAINING MODE: {mode.upper()}\n🔍 {explanation}\n{'=' * 60}")

        trainer = train_ds = val_ds = None
        ckpt_dir = None
        merged_ok = False
        try:
            self.load_model_with_lora(mode)

            texts = self.build_texts(mode, data_path, texts_override)
            if not texts:
                print("❌ No training data found!")
                return False
            train_ds, val_ds = self.make_datasets(texts)

            epochs = EPOCHS_CONFIG[mode]
            lr = LR_CONFIG[mode]
            bs = BATCH_SIZE_CONFIG[mode]
            ga = GRAD_ACCUM_CONFIG[mode]
            use_bf16 = self._dtype() == torch.bfloat16
            print(f"\n⚙️ Config: epochs={epochs} lr={lr} batch={bs}x{ga} "
                  f"bf16={use_bf16} max_len={MAX_SEQ_LEN}")

            ckpt_dir, resume = self._prepare_ckpt_dir(mode)

            args = TrainingArguments(
                output_dir=ckpt_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=bs,
                gradient_accumulation_steps=ga,
                learning_rate=lr,
                warmup_ratio=0.05,
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                save_strategy="steps",
                save_steps=SAVE_STEPS,
                save_total_limit=2,
                **{_EVAL_ARG: "epoch" if val_ds else "no"},
                prediction_loss_only=True,
                logging_steps=10,
                bf16=use_bf16,
                fp16=False,
                dataloader_num_workers=0,
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                report_to="none",
                max_grad_norm=MAX_GRAD_NORM,
                optim="adamw_torch",
                adam_beta2=0.98,
                seed=SEED,
            )

            trainer = WeightedTrainer(
                model=self.model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=PadCollator(self.tokenizer.pad_token_id),
            )
            trainer.add_callback(HFUploadCallback(self.hf_uploader, mode))

            print("\n🏋️ Starting DeepSeek Training...")
            result = self._run_trainer(trainer, resume)
            final_loss = getattr(result, "training_loss", None)
            if final_loss is not None:
                print(f"   📊 Final training loss: {final_loss:.4f}")
            print("\n✅ DeepSeek training completed!")

            # 1) Adapter সেভ (ট্রেনিং মডেল থেকে) + আপলোড
            self.save_adapter(mode)

            # 2) ট্রেনিং-এর সব কিছু ছেড়ে মেমরি খালি করা
            trainer = train_ds = val_ds = None
            self.free_model()

            # 3) আলাদা fp32 base-এ সত্যিকারের merge
            merged_ok = self.merge_and_save(mode)

            shutil.rmtree(ckpt_dir, ignore_errors=True)
            self.hf_uploader.upload_tracking_files()

            merge_line = (f"💾 Merged model: {esc(LLM_MODEL_DIR)}" if merged_ok
                          else "⚠️ Merge FAILED - adapter saved, merged model is OLD")
            self.notify(f"""
✅ <b>DeepSeek Training Completed</b>
📅 {datetime.now():%Y-%m-%d %H:%M}
🎯 {esc(explanation)}
📚 Symbols trained: {len(symbols_batch) if symbols_batch else 'ALL'}
{merge_line}
""")
            return True

        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            self.notify(f"""
⚠️ <b>DeepSeek Training Error</b>
📅 {datetime.now():%Y-%m-%d %H:%M}
🎯 {esc(mode)}
❌ {esc(e)}
""")
            return False
        finally:
            trainer = train_ds = val_ds = None  # optimizer states ইত্যাদি ছাড়ো
            self.free_model()

    # ---------- saving ----------
    def save_adapter(self, mode):
        self.model.save_pretrained(LORA_ADAPTER_DIR)
        self.tokenizer.save_pretrained(LORA_ADAPTER_DIR)
        print(f"💾 LoRA adapter saved: {LORA_ADAPTER_DIR}")
        self.hf_uploader.upload_adapter(LORA_ADAPTER_DIR, mode)
        self.hf_uploader.upload_adapter_latest(LORA_ADAPTER_DIR)

    def merge_and_save(self, mode):
        """সত্যিকারের fp32 merge: আলাদা fp32 base + সদ্য সেভ হওয়া adapter।"""
        merged = base = peft_model = None
        try:
            print("🔄 Merging adapter into fp32 base...")
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="cpu",   # merge সবসময় CPU-তে (GPU মেমরি ছোঁয়া হয় না)
                **{_DTYPE_ARG: torch.float32},
            )
            peft_model = PeftModel.from_pretrained(base, LORA_ADAPTER_DIR)
            merged = peft_model.merge_and_unload()
            merged.config.use_cache = True
            merged.save_pretrained(LLM_MODEL_DIR, safe_serialization=True)
            tok = AutoTokenizer.from_pretrained(LORA_ADAPTER_DIR, trust_remote_code=True)
            tok.save_pretrained(LLM_MODEL_DIR)
            print(f"✅ Merged model saved (fp32): {LLM_MODEL_DIR}")
            self.hf_uploader.upload_final_model(LLM_MODEL_DIR, mode)
            return True
        except Exception as e:
            print(f"⚠️ Merge/save failed (adapter is safe): {e}")
            return False
        finally:
            merged = base = peft_model = None
            self.free_model()

    # ---------- data generation ----------
    def generate_training_data_for_symbols(self, symbols):
        print(f"\n📝 Generating training data for {len(symbols)} symbols...")
        result = subprocess.run(
            [sys.executable, "scripts/generate_pattern_training_data_complete.py",
             "--symbols", ",".join(symbols)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"   ⚠️ Data generation failed: {result.stderr[:300]}")
            return False
        print("   ✅ Training data generated")
        return True

    # ---------- main flow ----------
    def run(self):
        print("=" * 60)
        print("🚀 AUTO DEEPSEEK TRAINER (v4.1)")
        print("=" * 60)
        print(f"📅 {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"🧠 Model: {BASE_MODEL}")
        print(f"📁 Merged: {LLM_MODEL_DIR} | Adapter: {LORA_ADAPTER_DIR}")
        print(f"🔧 LoRA: r={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']}")
        print(f"📊 XGBoost: {len(self.xgb_ppo.xgb_models)} models")
        print("=" * 60)

        stats = self.mistake_collector.get_confidence_stats()
        print(f"\n📊 Confidence: avg={stats['avg_confidence']:.2%} "
              f"mistake_rate={stats['mistake_rate']:.2f}%")

        all_symbols = self.get_all_symbols_from_mongodb()
        new_symbols = self.get_new_symbols(all_symbols)

        # STEP 1: new symbols
        if new_symbols:
            print(f"\n📚 {len(new_symbols)} new symbols to train")
            for i in range(0, len(new_symbols), BATCH_SIZE):
                batch = new_symbols[i:i + BATCH_SIZE]
                batch_num = self.batch_manager.next_batch_number()
                print(f"\n📦 Batch {batch_num}: {len(batch)} symbols")

                if not self.generate_training_data_for_symbols(batch):
                    continue

                has_adapter = os.path.exists(os.path.join(LORA_ADAPTER_DIR, "adapter_config.json"))
                mode = "incremental" if (has_adapter or self.trained_symbols) else "first_train"
                print(f"   ➡️ Mode: {mode}")

                if self.train(mode=mode, symbols_batch=batch):
                    self.trained_symbols.extend(batch)
                    self.save_trained_symbols()
                    self.batch_manager.mark_batch_completed(batch_num, batch)
                    print(f"✅ Batch {batch_num} complete!")
                else:
                    print(f"❌ Batch {batch_num} failed - stopping new-symbol loop")
                    break
        else:
            print("\n✅ No new symbols found!")

        # STEP 2: weekly fine-tune
        wb, wsyms = self.batch_manager.get_batch_for_weekly_finetune()
        if wb and wsyms:
            print(f"\n🔄 Weekly fine-tune Batch {wb}")
            if self.generate_training_data_for_symbols(wsyms):
                if self.train(mode="weekly_finetune", symbols_batch=wsyms):
                    self.batch_manager.mark_weekly_done(wb)

        # STEP 3: monthly consolidation
        if self.batch_manager.should_consolidate():
            all_trained = self.batch_manager.get_all_batch_symbols()
            if all_trained:
                print(f"\n🔄 Monthly consolidation - {len(all_trained)} symbols")
                if self.generate_training_data_for_symbols(all_trained):
                    if self.train(mode="consolidate", symbols_batch=all_trained):
                        self.batch_manager.mark_consolidated()

        # STEP 4: hard-example retraining (cooldown সহ)
        hp = self.mistake_collector.get_hard_examples(limit=100, priority_only=True)
        if hp and self.batch_manager.should_run_mistake_learning():
            print(f"\n🔥 {len(hp)} high priority mistakes")
            texts = [format_example(ex) for ex in hp]
            texts = [t for t in texts if len(t) >= MIN_OVERRIDE_LEN]
            if texts:
                print(f"   ✅ Prepared {len(texts)} hard example texts")
                if self.train(mode="mistake_learning", texts_override=texts):
                    self.batch_manager.mark_mistake_learning_done()
        elif hp:
            print("\n⏳ Mistake learning on cooldown - skipping")

        print("\n" + "=" * 60 + "\n📊 FINAL STATUS\n" + "=" * 60)
        print(f"   Total trained symbols: {len(self.trained_symbols)}")
        print(f"   XGBoost Models: {len(self.xgb_ppo.xgb_models)}")
        print(f"   Merged model: {LLM_MODEL_DIR}")
        print(f"   Adapter: {LORA_ADAPTER_DIR}")
        print(f"   HF Repo: {HF_DATASET_REPO}")
        print("=" * 60)


if __name__ == "__main__":
    AutoDeepSeekTrainer().run()
