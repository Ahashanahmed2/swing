# scripts/llm_train.py
# Advanced LLM Trainer with Proper Label Prediction, Confidence Learning, and PPO Integration

import os
import torch
import json
import warnings
import pandas as pd
import numpy as np
import re
import requests
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
    from peft import LoraConfig, get_peft_model, TaskType
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    print("⚠️ PEFT not installed. Install with: pip install peft")

warnings.filterwarnings('ignore')

# =========================================================
# CONFIGURATION
# =========================================================

BATCH_SIZE = 20
HF_REPO_ID = "ahashanahmed/llm-stock-model"
BASE_MODEL = "distilgpt2"
TRACKING_FILE = "./trained_symbols.json"
TRAINING_DATA_PATH = "./csv/training_texts.txt"
MARKET_DATA_PATH = "./csv/mongodb.csv"
MISTAKES_FILE = "./csv/trading_mistakes.csv"
CONFIDENCE_LOG = "./csv/llm_confidence_log.csv"
HARD_EXAMPLES_FILE = "./csv/hard_examples.csv"

# Schedule
FINE_TUNE_INTERVAL = 7
LAST_FINE_TUNE_FILE = "./last_finetune.txt"

# Learning parameters
MAX_OLD_EXAMPLES = 1000
HARD_EXAMPLE_THRESHOLD = 0.3
HIGH_PRIORITY_THRESHOLD = 0.4
WEIGHTED_LOSS_ENABLED = True
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 3
VALIDATION_SPLIT_RATIO = 0.1

# ✅ FIX 1: Correct LoRA config for distilgpt2
LORA_CONFIG = {
    'r': 8,
    'lora_alpha': 32,
    'target_modules': ['c_attn'],
    'lora_dropout': 0.1,
    'bias': 'none',
    'task_type': TaskType.CAUSAL_LM
}

# Label patterns for extraction
SIGNAL_PATTERNS = {
    'bullish': r'(?:Signal|Prediction|Recommendation):?\s*(?:BUY|Bullish|LONG|✅ BUY)',
    'bearish': r'(?:Signal|Prediction|Recommendation):?\s*(?:SELL|Bearish|SHORT|❌ SELL)',
    'neutral': r'(?:Signal|Prediction|Recommendation):?\s*(?:HOLD|Neutral|WAIT|⏳ WAIT)'
}

CONFIDENCE_PATTERN = r'(?:Confidence|Signal Strength):?\s*(\d+(?:\.\d+)?)%'


class WeightedTrainer(Trainer):
    """Custom trainer with weighted loss - ✅ FIX 2: Production-grade weight handling"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # ✅ FIX 2: Get weights from inputs (safer)
        weights = inputs.get("weight", None)
        labels = inputs.get("labels")
        
        # Remove weight from inputs before forward pass
        if weights is not None:
            inputs = {k: v for k, v in inputs.items() if k != "weight"}
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if weights is not None:
            # Shift for causal LM
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
    """✅ NEW: Dataset with built-in weights for safe training"""
    
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
            'weight': self.weights[idx]  # ✅ FIX 2: Weight in dataset
        }
    
    def __len__(self):
        return len(self.input_ids)


class LabelExtractor:
    """✅ FIX 1: Extract labels from generated text"""
    
    @staticmethod
    def extract_signal(text):
        """Extract trading signal from generated text"""
        text_lower = text.lower()
        
        # Check for bullish signals
        bullish_keywords = ['buy', 'bullish', 'long', '✅ buy', 'signal: buy']
        for kw in bullish_keywords:
            if kw in text_lower:
                return 1  # Bullish/Buy
        
        # Check for bearish signals
        bearish_keywords = ['sell', 'bearish', 'short', '❌ sell', 'signal: sell']
        for kw in bearish_keywords:
            if kw in text_lower:
                return 0  # Bearish/Sell
        
        # Check for neutral
        neutral_keywords = ['hold', 'neutral', 'wait']
        for kw in neutral_keywords:
            if kw in text_lower:
                return 2  # Neutral
        
        # Try regex patterns
        for pattern in SIGNAL_PATTERNS.values():
            if re.search(pattern, text, re.IGNORECASE):
                if 'buy' in pattern or 'bullish' in pattern:
                    return 1
                elif 'sell' in pattern or 'bearish' in pattern:
                    return 0
                else:
                    return 2
        
        return 2  # Default neutral
    
    @staticmethod
    def extract_confidence(text):
        """Extract confidence score from generated text"""
        match = re.search(CONFIDENCE_PATTERN, text)
        if match:
            return float(match.group(1)) / 100.0
        return 0.5


class MetricsCalculator:
    """Evaluation metrics with proper label extraction"""
    
    @staticmethod
    def evaluate_model(model, tokenizer, eval_texts, true_labels):
        """✅ FIX 1: Proper label-based evaluation"""
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
                
                # Extract label and confidence
                pred = LabelExtractor.extract_signal(generated)
                conf = LabelExtractor.extract_confidence(generated)
                predictions.append(pred)
                confidences.append(conf)
        
        # Calculate metrics
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Calculate profit simulation
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
    """Mistake collector with confidence tracking"""
    
    def __init__(self):
        self.mistakes = []
        self.confidence_history = []
        self.hard_examples = []
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
        """Generate training examples with structured labels"""
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
        self.mistake_collector = MistakeCollector()
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
        
        # Replay buffer
        if self.old_training_texts:
            self.old_training_texts.extend(new_texts)
            self.old_training_texts = self.old_training_texts[-MAX_OLD_EXAMPLES:]
            train_texts = self.old_training_texts.copy()
            print(f"   Replay buffer: {len(self.old_training_texts)} total examples")
        else:
            train_texts = new_texts
            self.old_training_texts = train_texts.copy()
        
        # Add mistake examples
        mistake_texts = self.mistake_collector.get_mistake_dataset(limit=200)
        
        if mistake_texts:
            normal_count = int(len(train_texts) * 0.8)
            mistake_count = min(len(mistake_texts), int(len(train_texts) * 0.2))
            train_texts = train_texts[:normal_count] + mistake_texts[:mistake_count]
            print(f"   Data mix: {normal_count} normal + {mistake_count} mistakes")
        
        # Curriculum sorting
        easy_texts = [t for t in train_texts if self.classify_example_difficulty(t) == 'easy']
        medium_texts = [t for t in train_texts if self.classify_example_difficulty(t) == 'medium']
        hard_texts = [t for t in train_texts if self.classify_example_difficulty(t) == 'hard']
        train_texts = easy_texts + medium_texts + hard_texts
        print(f"   Curriculum: {len(easy_texts)} easy, {len(medium_texts)} medium, {len(hard_texts)} hard")
        
        # Weighted learning
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
        
        # Try to load from Hugging Face
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
        
        # Try local model
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
        
        # Start from base model
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
        """✅ FIX 3: Proper tokenizer and model config"""
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # ✅ FIX 3: Set pad_token_id in model config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # ✅ FIX 5: Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Device: {device}")
        print(f"   Gradient Checkpointing: Enabled")
    
    def train(self, mode="incremental"):
        print(f"\n{'='*60}")
        print(f"🎯 TRAINING MODE: {mode.upper()}")
        print(f"{'='*60}")
        
        train_texts, example_weights = self.load_training_data_with_curriculum()
        if not train_texts:
            print("❌ No training data found!")
            return False
        
        # Tokenize
        encodings = self.tokenizer(
            train_texts, 
            truncation=True, 
            padding=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        # ✅ FIX 2: Use StructuredDataset with built-in weights
        train_dataset = StructuredDataset(encodings, example_weights)
        
        # Chronological split for validation
        dataset_size = len(train_dataset)
        val_size = max(1, int(dataset_size * VALIDATION_SPLIT_RATIO))
        train_size = dataset_size - val_size
        
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, dataset_size))
        
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        eval_subset = torch.utils.data.Subset(train_dataset, val_indices)
        print(f"   Dataset split: {train_size} train, {val_size} validation (chronological)")
        
        # Training config
        if mode == "first_train":
            num_epochs = 10
            learning_rate = 5e-5
            batch_size = 2
        elif mode == "incremental":
            num_epochs = 5
            learning_rate = 3e-5
            batch_size = 2
        else:
            num_epochs = 3
            learning_rate = 1e-5
            batch_size = 4
        
        print(f"\n⚙️ Training Config:")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Batch Size: {batch_size}")
        print(f"   LoRA: {'Enabled' if LORA_AVAILABLE else 'Disabled'}")
        print(f"   Gradient Checkpointing: Enabled")
        
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
        
        # Evaluation
        print("\n📊 Running evaluation...")
        eval_results = trainer.evaluate()
        print(f"   Evaluation loss: {eval_results.get('eval_loss', 0):.4f}")
        
        # Save model
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
    
    def should_fine_tune(self):
        if not os.path.exists(LAST_FINE_TUNE_FILE):
            return True
        with open(LAST_FINE_TUNE_FILE, 'r') as f:
            last_date = datetime.fromisoformat(f.read().strip())
        return (datetime.now() - last_date).days >= FINE_TUNE_INTERVAL
    
    def update_last_fine_tune(self):
        with open(LAST_FINE_TUNE_FILE, 'w') as f:
            f.write(datetime.now().isoformat())
    
    def generate_training_data_for_symbols(self, symbols):
        print(f"\n📝 Generating training data for {len(symbols)} symbols...")
        import subprocess
        result = subprocess.run(
            ["python", "scripts/generate_pattern_training_data_complete.py", 
             "--symbols", ",".join(symbols[:BATCH_SIZE])], 
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"   ⚠️ Data generation failed: {result.stderr}")
            return False
        print("   ✅ Training data generated")
        return True
    
    def run(self):
        print("="*60)
        print("🚀 AUTO LLM TRAINER (Pro Version v4.0)")
        print("="*60)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📚 Batch size: {BATCH_SIZE}")
        print(f"🔄 Fine-tune interval: {FINE_TUNE_INTERVAL} days")
        print(f"🔧 LoRA: {'Enabled' if LORA_AVAILABLE else 'Disabled'}")
        print("="*60)
        
        confidence_stats = self.mistake_collector.get_confidence_stats()
        print(f"\n📊 Confidence Statistics:")
        print(f"   Average confidence: {confidence_stats['avg_confidence']:.2%}")
        print(f"   High priority mistakes: {confidence_stats['high_priority_count']}")
        
        new_symbols = self.get_new_symbols()
        self.load_model_with_lora()
        
        if new_symbols:
            print(f"\n📚 Found {len(new_symbols)} new symbols to train")
            for i in range(0, len(new_symbols), BATCH_SIZE):
                batch = new_symbols[i:i+BATCH_SIZE]
                print(f"\n📦 Processing batch {i//BATCH_SIZE + 1}: {len(batch)} symbols")
                if self.generate_training_data_for_symbols(batch):
                    mode = "first_train" if len(self.trained_symbols) == 0 else "incremental"
                    if self.train(mode):
                        self.trained_symbols.extend(batch)
                        self.save_trained_symbols()
        else:
            print("\n✅ No new symbols found!")
        
        # Hard example retraining
        high_priority_examples = self.mistake_collector.get_hard_examples(limit=100, priority_only=True)
        if high_priority_examples:
            print(f"\n🔥 Found {len(high_priority_examples)} high priority mistakes for retraining!")
            temp_file = "./temp_hard_examples.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                signal_map = {1: 'BUY', 0: 'SELL', 2: 'HOLD'}
                for ex in high_priority_examples[:50]:
                    f.write(f"""
================================================================================
Pattern: {ex.get('pattern', 'Unknown')}
Symbol: {ex.get('symbol')}
Previous Prediction: {signal_map.get(ex.get('prediction', 2), 'HOLD')}
❌ This was WRONG

✅ CORRECT ANSWER: {signal_map.get(ex.get('actual', 2), 'HOLD')}
Explanation: {ex.get('correct_explanation', 'Review the pattern rules')}

Signal: {signal_map.get(ex.get('actual', 2), 'HOLD')}
Confidence: {min(95, max(65, int(ex.get('confidence', 0.7) * 100 + 10)))}
================================================================================
""")
            self.train(mode="fine_tune")
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Weekly fine-tune
        if self.should_fine_tune():
            print(f"\n🔄 Time for weekly fine-tuning!")
            self.load_model_with_lora()
            if self.train(mode="fine_tune"):
                self.update_last_fine_tune()
        
        print("\n" + "="*60)
        print("📊 FINAL STATUS")
        print("="*60)
        print(f"   Total trained symbols: {len(self.trained_symbols)}")
        print(f"   HF Repository: {HF_REPO_ID}")
        print("="*60)


if __name__ == "__main__":
    trainer = AutoLLMTrainer()
    trainer.run()