# scripts/llm_train.py
# Advanced LLM Trainer with XGBoost + PPO Integration, Proper Label Prediction, Confidence Learning
# Fully Updated with Auto Batch Management, Incremental Learning, and Mistake Learning

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
from huggingface_hub import login, create_repo, upload_folder, list_repo_files

# Optional: LoRA for faster training
try:
    from peft import LoraConfig, get_peft_model
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    print("⚠️ PEFT not installed. Install with: pip install peft")

warnings.filterwarnings('ignore')

# =========================================================
# CONFIGURATION
# =========================================================

# Batch configuration for incremental training
BATCH_SIZE = 5  # Number of symbols per training batch
TOTAL_BATCHES = 4  # Will be auto-calculated
MAX_SYMBOLS_PER_BATCH = 100  # Maximum symbols to train in one batch

HF_REPO_ID = "ahashanahmed/llm-stock-model"
BASE_MODEL = "distilgpt2"
TRACKING_FILE = "./trained_symbols.json"
BATCH_TRACKING_FILE = "./batch_tracking.json"
TRAINING_DATA_PATH = "./csv/training_texts.txt"
MARKET_DATA_PATH = "./csv/mongodb.csv"
MISTAKES_FILE = "./csv/trading_mistakes.csv"
CONFIDENCE_LOG = "./csv/llm_confidence_log.csv"
HARD_EXAMPLES_FILE = "./csv/hard_examples.csv"

# XGBoost and PPO paths
XGBOOST_DIR = "./csv/xgboost"
PPO_MODELS_DIR = "./csv/ppo_models"
PPO_PER_SYMBOL_DIR = "./csv/ppo_models/per_symbol"

# Schedule
FINE_TUNE_INTERVAL = 7  # Days between fine-tuning
LAST_FINE_TUNE_FILE = "./last_finetune.txt"
LAST_CONSOLIDATE_FILE = "./last_consolidate.txt"
CONSOLIDATE_INTERVAL = 30  # Days between full consolidation

# Learning parameters
MAX_OLD_EXAMPLES = 1000
HARD_EXAMPLE_THRESHOLD = 0.3
HIGH_PRIORITY_THRESHOLD = 0.4
WEIGHTED_LOSS_ENABLED = True
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 3
VALIDATION_SPLIT_RATIO = 0.1

# Training mode flags
FORCE_RETRAIN = False  # Set to True to retrain all symbols

# LoRA config for distilgpt2
LORA_CONFIG = {
    'r': 8,
    'lora_alpha': 32,
    'target_modules': ['c_attn'],
    'lora_dropout': 0.1,
    'bias': 'none',
}

# Label patterns for extraction
SIGNAL_PATTERNS = {
    'bullish': r'(?:Signal|Prediction|Recommendation):?\s*(?:BUY|Bullish|LONG|✅ BUY)',
    'bearish': r'(?:Signal|Prediction|Recommendation):?\s*(?:SELL|Bearish|SHORT|❌ SELL)',
    'neutral': r'(?:Signal|Prediction|Recommendation):?\s*(?:HOLD|Neutral|WAIT|⏳ WAIT)'
}

CONFIDENCE_PATTERN = r'(?:Confidence|Signal Strength):?\s*(\d+(?:\.\d+)?)%'


# =========================================================
# BATCH MANAGER (NEW)
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
        with open(BATCH_TRACKING_FILE, 'w') as f:
            json.dump(self.batch_tracking, f, indent=2)
    
    def get_next_batch_symbols(self, all_symbols, trained_symbols, batch_size=BATCH_SIZE):
        """Get next batch of symbols to train (auto batch creation)"""
        untrained = [s for s in all_symbols if s not in trained_symbols]
        
        if not untrained:
            return []
        
        # Get next batch
        next_batch = untrained[:batch_size]
        
        # Store batch info
        batch_num = self.current_batch_index + 1
        self.batch_symbols[str(batch_num)] = next_batch
        self.batch_tracking['batch_symbols'] = self.batch_symbols
        
        return next_batch, batch_num
    
    def mark_batch_completed(self, batch_num, symbols):
        """Mark a batch as completed"""
        if batch_num not in self.completed_batches:
            self.completed_batches.append(batch_num)
            self.current_batch_index = batch_num
            self.batch_tracking['current_batch'] = batch_num
            self.batch_tracking['completed_batches'] = self.completed_batches
            self.batch_tracking['total_symbols_trained'] += len(symbols)
            self.batch_tracking['last_batch_date'] = datetime.now().isoformat()
            self.save_batch_tracking()
    
    def get_batch_for_weekly_finetune(self):
        """Get which batch to fine-tune this week (rotation)"""
        if not self.completed_batches:
            return None
        
        # Get current week number
        week_num = (datetime.now() - datetime.strptime(self.batch_tracking.get('last_batch_date', '2024-01-01'), '%Y-%m-%dT%H:%M:%S.%f').date()).days // 7
        
        # Rotate through completed batches
        batch_index = week_num % len(self.completed_batches)
        batch_num = self.completed_batches[batch_index]
        
        return batch_num, self.batch_symbols.get(str(batch_num), [])
    
    def should_consolidate(self):
        """Check if it's time for full consolidation"""
        last_consolidate = self.batch_tracking.get('last_consolidate', None)
        if not last_consolidate:
            return True
        
        last_date = datetime.fromisoformat(last_consolidate)
        days_since = (datetime.now() - last_date).days
        return days_since >= CONSOLIDATE_INTERVAL
    
    def mark_consolidated(self):
        """Mark that consolidation has been performed"""
        self.batch_tracking['last_consolidate'] = datetime.now().isoformat()
        self.save_batch_tracking()
    
    def get_all_batch_symbols(self):
        """Get all symbols from all completed batches"""
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
        """Load all XGBoost models from directory"""
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
        """Load PPO model metadata"""
        if os.path.exists(PPO_PER_SYMBOL_DIR):
            for file in os.listdir(PPO_PER_SYMBOL_DIR):
                if file.endswith('.zip') and file.startswith('ppo_'):
                    symbol = file.replace('ppo_', '').replace('.zip', '')
                    self.ppo_models[symbol] = os.path.join(PPO_PER_SYMBOL_DIR, file)
            print(f"   ✅ Found {len(self.ppo_models)} PPO models")
        else:
            print(f"   ⚠️ PPO models directory not found: {PPO_PER_SYMBOL_DIR}")
    
    def get_xgb_prediction(self, symbol, features_dict=None):
        """Get XGBoost prediction for a symbol"""
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
        """Check if PPO model exists for symbol"""
        if symbol in self.ppo_models:
            return {
                'exists': True,
                'model_path': self.ppo_models[symbol],
                'source': 'PPO'
            }
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


class MetricsCalculator:
    """Evaluation metrics with proper label extraction"""
    
    @staticmethod
    def evaluate_model(model, tokenizer, eval_texts, true_labels):
        model.eval()
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for text in eval_texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                pred = LabelExtractor.extract_signal(generated)
                conf = LabelExtractor.extract_confidence(generated)
                predictions.append(pred)
                confidences.append(conf)
        
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        profit = 0
        correct = 0
        for pred, true, conf in zip(predictions, true_labels, confidences):
            if conf < 0.5:
                continue
            if pred == true:
                correct += 1
                profit += 0.02 if true == 1 else 0.01
            else:
                profit -= 0.01 if pred == 1 else 0.005
        
        win_rate = correct / len([c for c in confidences if c >= 0.5]) if any(c >= 0.5 for c in confidences) else 0
        
        print(f"\n📊 MODEL EVALUATION:")
        print(f"   Accuracy: {acc:.2%}")
        print(f"   F1 Score: {f1:.3f}")
        print(f"   Simulated Profit: {profit:.2%}")
        print(f"   Win Rate: {win_rate:.2%}")
        
        return {'accuracy': acc, 'f1_score': f1, 'profit': profit, 'win_rate': win_rate}


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
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'actual': actual,
            'confidence': confidence,
            'pattern': pattern,
            'market_regime': market_regime,
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
            'timestamp': datetime.now(),
            'symbol': symbol,
            'confidence': confidence,
            'is_mistake': prediction != actual,
            'is_high_priority': mistake['is_high_priority']
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
            'avg_confidence': df['confidence'].mean(),
            'mistake_rate': (df['prediction'] != df['actual']).mean() if 'actual' in df.columns else 0,
            'low_confidence_count': len(df[df['confidence'] < HARD_EXAMPLE_THRESHOLD]),
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


class AutoLLMTrainer:
    def __init__(self):
        self.trained_symbols = self.load_trained_symbols()
        self.model = None
        self.tokenizer = None
        self.xgb_ppo = XGBoostPPOIntegrator()
        self.mistake_collector = MistakeCollector(self.xgb_ppo)
        self.batch_manager = BatchManager()
        self.old_training_texts = []
        
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
    
    def get_hf_trained_symbols(self):
        try:
            token = os.getenv("hf_token")
            if not token:
                return []
            files = list_repo_files(repo_id=HF_REPO_ID, repo_type="model", token=token)
            if "trained_symbols.json" in files:
                url = f"https://huggingface.co/{HF_REPO_ID}/raw/main/trained_symbols.json"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    return data.get('symbols', [])
            return []
        except:
            return []
    
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
        trained_hf = set(self.get_hf_trained_symbols())
        all_trained = trained_local.union(trained_hf)
        new_symbols = [s for s in all_symbols if s not in all_trained]
        print(f"   Already trained: {len(all_trained)} symbols")
        print(f"   New symbols found: {len(new_symbols)}")
        return new_symbols
    
    def classify_example_difficulty(self, text):
        text_lower = text.lower()
        hard_keywords = ['complex', 'multi timeframe', 'divergence', 'harmonic', 'elliott']
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
        
        mistake_texts = self.mistake_collector.get_mistake_dataset(limit=200)
        
        if mistake_texts:
            normal_count = int(len(train_texts) * 0.8)
            mistake_count = min(len(mistake_texts), int(len(train_texts) * 0.2))
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
            if difficulty == 'hard':
                example_weights[i] = 2.0
            elif difficulty == 'medium':
                example_weights[i] = 1.0
            else:
                example_weights[i] = 0.5
        
        return train_texts, example_weights
    
    def load_model_with_lora(self):
        print("\n🏗️ Loading model...")
        token = os.getenv("hf_token")
        
        if token:
            try:
                print(f"   Attempting to load from HF: {HF_REPO_ID}")
                self.model = AutoModelForCausalLM.from_pretrained(HF_REPO_ID)
                self.tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
                print("   ✅ Loaded existing model from Hugging Face")
                if LORA_AVAILABLE:
                    lora_config = LoraConfig(**LORA_CONFIG)
                    self.model = get_peft_model(self.model, lora_config)
                    print("   ✅ LoRA applied")
                self._post_load_setup()
                return
            except Exception as e:
                print(f"   No existing model on HF: {e}")
        
        if os.path.exists("./llm_model"):
            try:
                print("   Loading local model...")
                self.model = AutoModelForCausalLM.from_pretrained("./llm_model")
                self.tokenizer = AutoTokenizer.from_pretrained("./llm_model")
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
            padding=True, 
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
        
        if mode == "first_train":
            num_epochs = 2
            learning_rate = 5e-5
            batch_size = 4
        elif mode == "incremental":
            num_epochs = 2
            learning_rate = 3e-5
            batch_size = 4
        elif mode == "weekly_finetune":
            num_epochs = 1
            learning_rate = 1e-5
            batch_size = 4
        else:
            num_epochs = 2
            learning_rate = 1e-5
            batch_size = 4
        
        print(f"\n⚙️ Training Config:")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Batch Size: {batch_size}")
        print(f"   LoRA: {'Enabled' if LORA_AVAILABLE else 'Disabled'}")
        print(f"   XGBoost Integration: Enabled ({len(self.xgb_ppo.xgb_models)} models)")
        
        training_args = TrainingArguments(
            output_dir="./llm_model",
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            save_steps=200,
            save_total_limit=2,
            logging_steps=20,
            evaluation_strategy="steps",
            eval_steps=200,
            learning_rate=learning_rate,
            warmup_steps=50,
            weight_decay=0.01,
            fp16=False,
            report_to="none",
            max_grad_norm=MAX_GRAD_NORM,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False
        )
        early_stop_callback = EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE
        )
        
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=eval_subset,
            data_collator=data_collator,
            callbacks=[early_stop_callback]
        )
        
        print("\n🏋️ Starting training...")
        trainer.train()
        print("\n✅ Training completed!")
        
        print("\n📊 Running evaluation...")
        eval_results = trainer.evaluate()
        print(f"   Evaluation loss: {eval_results.get('eval_loss', 0):.4f}")
        
        self.model.save_pretrained("./llm_model")
        self.tokenizer.save_pretrained("./llm_model")
        print("💾 Model saved locally")
        
        self.upload_to_huggingface(mode)
        return True
    
    def upload_to_huggingface(self, mode):
        token = os.getenv("hf_token")
        if not token:
            print("ℹ️ No HF_TOKEN, skipping upload")
            return
        
        print("\n📤 Uploading to Hugging Face...")
        try:
            login(token=token)
            create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
            
            with open(TRACKING_FILE, 'w') as f:
                json.dump({
                    'symbols': self.trained_symbols,
                    'last_updated': datetime.now().isoformat(),
                    'total_trained': len(self.trained_symbols)
                }, f)
            
            upload_folder(
                folder_path="./llm_model", 
                repo_id=HF_REPO_ID, 
                repo_type="model", 
                commit_message=f"{mode}: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            upload_folder(
                folder_path=".", 
                repo_id=HF_REPO_ID, 
                repo_type="model", 
                path_in_repo="trained_symbols.json", 
                commit_message="Update trained symbols"
            )
            print(f"✅ Uploaded to: https://huggingface.co/{HF_REPO_ID}")
        except Exception as e:
            print(f"⚠️ Upload failed: {e}")
    
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
        print("🚀 AUTO LLM TRAINER (Pro Version v6.0 - Auto Batch Management)")
        print("="*60)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📚 Batch size: {BATCH_SIZE}")
        print(f"🔄 Fine-tune interval: {FINE_TUNE_INTERVAL} days")
        print(f"🔧 LoRA: {'Enabled' if LORA_AVAILABLE else 'Disabled'}")
        print(f"📊 XGBoost Models: {len(self.xgb_ppo.xgb_models)}")
        print(f"📁 PPO Models: {len(self.xgb_ppo.ppo_models)}")
        print("="*60)
        
        confidence_stats = self.mistake_collector.get_confidence_stats()
        print(f"\n📊 Confidence Statistics:")
        print(f"   Average confidence: {confidence_stats['avg_confidence']:.2%}")
        print(f"   High priority mistakes: {confidence_stats['high_priority_count']}")
        
        all_symbols = self.get_all_symbols_from_mongodb()
        new_symbols = self.get_new_symbols()
        
        self.load_model_with_lora()
        
        # STEP 1: Train new symbols in batches
        if new_symbols:
            print(f"\n📚 Found {len(new_symbols)} new symbols to train")
            
            # Process in batches
            for i in range(0, len(new_symbols), BATCH_SIZE):
                batch = new_symbols[i:i+BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1
                print(f"\n📦 Processing batch {batch_num}: {len(batch)} symbols")
                
                if self.generate_training_data_for_symbols(batch):
                    # Determine training mode
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
        
        # STEP 2: Weekly fine-tune (rotate through batches)
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
        high_priority_examples = self.mistake_collector.get_hard_examples(limit=100, priority_only=True)
        if high_priority_examples:
            print(f"\n🔥 Found {len(high_priority_examples)} high priority mistakes for retraining!")
            temp_file = "./temp_hard_examples.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                signal_map = {1: 'BUY', 0: 'SELL', 2: 'HOLD'}
                for ex in high_priority_examples[:50]:
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
            self.train(mode="mistake_learning")
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print("\n" + "="*60)
        print("📊 FINAL STATUS")
        print("="*60)
        print(f"   Total trained symbols: {len(self.trained_symbols)}")
        print(f"   Completed batches: {len(self.batch_manager.completed_batches)}")
        print(f"   XGBoost Models Available: {len(self.xgb_ppo.xgb_models)}")
        print(f"   PPO Models Available: {len(self.xgb_ppo.ppo_models)}")
        print(f"   HF Repository: {HF_REPO_ID}")
        print("="*60)


if __name__ == "__main__":
    trainer = AutoLLMTrainer()
    trainer.run()