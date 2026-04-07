# scripts/llm_train.py
# Hugging Face ডেটাসেট থেকে ডাটা নিয়ে ছোট LLM ট্রেনিং করার স্ক্রিপ্ট

import os
import torch

import sys
print("Python executable:", sys.executable)
try:
    print("PyTorch version:", torch.__version__)
except ImportError as e:
    print("PyTorch import failed:", e)
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("🚀 TRAINING SMALL LLM FOR STOCK PATTERN RECOGNITION")
    print("="*60)
    
    # =========================================================
    # 1. টোকেন চেক করুন
    # =========================================================
    token = os.getenv("hf_token")
    if not token:
        print("❌ HF_TOKEN not found in environment variables!")
        return
    
    try:
        login(token=token)
        print("✅ Logged in to Hugging Face")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return
    
    # =========================================================
    ✅ লোকাল ফাইল থেকে ডাটা লোড করুন
     csv_path = "./csv/training_texts.txt"
    
    if not os.path.exists(csv_path):
        print(f"❌ Training file not found: {csv_path}")
        print("   Please run generate_pattern_training_data_complete.py first")
        return
    
    with open(csv_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    
    print(f"✅ Loaded {len(text_data)} characters from {csv_path}")
    
    # টেক্সটকে লাইনে ভাগ করুন
    texts = [t.strip() for t in text_data.split('================================================================================') if len(t.strip()) > 100]
    print(f"📊 Training examples: {len(texts)}")
    
    # =========================================================
    # 3. টোকেনাইজার সেটআপ করুন
    # =========================================================
    print("\n🔧 Setting up tokenizer...")
    
    # ছোট মডেলের জন্য GPT-2 টোকেনাইজার ব্যবহার করুন
    model_name = "distilgpt2"  # ছোট মডেল (CPU-তে দ্রুত কাজ করে)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # ডাটা টোকেনাইজ করুন
    encodings = tokenizer(texts,truncation=True, padding=True, max_length=512, return_tensors="pt")
    
    print(f"✅ Tokenizer ready (vocab size: {tokenizer.vocab_size})")
    
    # =========================================================
    # 4. মডেল সেটআপ করুন (CPU-তে চলবে)
    # =========================================================
    print("\n🏗️ Setting up model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU-তে float32 ভালো
        low_cpu_mem_usage=True
      
    )
    
    # CPU-তে চলার জন্য
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {model_name}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Device: {device}")
    
    # =========================================================
    # 5. ডাটা প্রস্তুত করুন
    # =========================================================
    printextstr📊 Preparing dataset...")
    
    # টেক্সটকে ট্রেনিং ডাটাতে রূপান্তর
    block_size = 128
    texts = text_data.split('\n\n')  # প্যারাগ্রাফ দিয়ে ভাগ করুন
    
    train_texts = []
    for text in texts:
        if len(text.strip()) > 50:  # খুব ছোট টেক্সট বাদ দিন
            train_texts.append(text.strip())
    
    print(f"   Training examples: {len(train_texts)}")
    
    # টোকেনাইজড ডাটা তৈরি করুন
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=block_size,
        return_tensors="pt"
    )
    
    # PyTorch ডেটাসেট তৈরি করুন
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.input_ids = encodings['input_ids']
            self.attention_mask = encodings['attention_mask']
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.input_ids[idx]  # causal LM এর জন্য
            }
        
        def __len__(self):
            return len(self.input_ids)
    
    train_dataset = TextDataset(train_encodings)
    
    # =========================================================
    # 6. ট্রেনিং আর্গুমেন্টস
    # =========================================================
    print("\n⚙️ Setting up training arguments...")
    
    output_dir = "./llm_stock_model"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=10,  # কম epoch (CPU-তে দ্রুত)
        per_device_train_batch_size=2,  # ছোট batch (CPU memory)
        gradient_accumulation_steps=4,
        save_steps=100,
        save_total_limit=2,
        logging_steps=20,
        prediction_loss_only=True,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=False,  # CPU-তে fp16 বন্ধ
        dataloader_drop_last=False,
        report_to="none",  # TensorBoard বন্ধ (optional)
    )
    
    print(f"   Output directory: {output_dir}")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    
    # =========================================================
    # 7. ট্রেনিং শুরু করুন
    # =========================================================
    print("\n🏋️ Starting training...")
    print("-" * 40)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM (GPT-style)
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    try:
        trainer.train()
        print("\n✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return
    
    # =========================================================
    # 8. মডেল সেভ করুন
    # =========================================================
    print("\n💾 Saving model...")
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to: {output_dir}")
    
    # =========================================================
    # 9. Hugging Face-এ আপলোড করুন
    # =========================================================
    print("\n📤 Uploading model to Hugging Face...")
    
    model_repo_id = "ahashanahmed/llm-stock-model"
    
    try:
        from huggingface_hub import create_repo, upload_folder
        
        # রিপোজিটরি তৈরি করুন (যদি না থাকে)
        create_repo(
            repo_id=model_repo_id,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"✅ Repository ready: {model_repo_id}")
        
        # মডেল আপলোড করুন
        upload_folder(
            folder_path=output_dir,
            repo_id=model_repo_id,
            repo_type="model",
            commit_message="Upload trained LLM for stock pattern recognition"
        )
        print(f"✅ Model uploaded to: https://huggingface.co/{model_repo_id}")
        
    except Exception as e:
        print(f"⚠️ Upload failed: {e}")
        print("   You can manually upload the folder:", output_dir)
    
    # =========================================================
    # 10. টেস্ট করুন
    # =========================================================
    print("\n🧪 Testing the model...")
    
    model.eval()
    test_prompt = "Stock pattern: Cup and Handle"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n📝 Test generation:")
    print(f"   Prompt: {test_prompt}")
    print(f"   Generated: {generated_text}")
    
    # =========================================================
    # 11. সারাংশ
    # =========================================================
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    print(f"   ✅ Model trained successfully!")
    print(f"   📁 Local model: {output_dir}")
    print(f"   🤗 Hugging Face: https://huggingface.co/{model_repo_id}")
    print(f"   📚 Dataset: https://huggingface.co/datasets/ahashanahmed/LLM_model_stock")
    print("="*60)

if __name__ == "__main__":
    main()
