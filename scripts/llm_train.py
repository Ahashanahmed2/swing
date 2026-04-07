# scripts/llm_train.py
# লোকাল ./csv/training_texts.txt ফাইল থেকে ডাটা নিয়ে ছোট LLM ট্রেনিং করার স্ক্রিপ্ট

import os
import torch
import sys
import warnings
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from huggingface_hub import login
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("🚀 TRAINING SMALL LLM FOR STOCK PATTERN RECOGNITION")
    print("="*60)
    
    # 1. টোকেন চেক (আপলোডের জন্য প্রয়োজন, লোকাল ট্রেনিংয়ের জন্য না)
    token = os.getenv("hf_token")
    if token:
        try:
            login(token=token)
            print("✅ Logged in to Hugging Face (for upload)")
        except Exception as e:
            print(f"⚠️ Login failed, but training will continue: {e}")
    else:
        print("ℹ️ No HF_TOKEN found. Model will be saved locally only.")
    
    # =========================================================
    # 2. লোকাল ফাইল থেকে ডাটা লোড করুন (সংশোধিত অংশ)
    # =========================================================
    csv_path = "./csv/training_texts.txt"
    
    if not os.path.exists(csv_path):
        print(f"❌ Training file not found: {csv_path}")
        print("   Please run generate_pattern_training_data_complete.py first")
        return
    
    with open(csv_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    
    print(f"✅ Loaded {len(text_data)} characters from {csv_path}")
    
    # টেক্সটকে ট্রেনিং উদাহরণে ভাগ করুন (সংশোধিত অংশ)
    # ('===' ডিলিমিটার ব্যবহার করে বিভক্ত করুন এবং খালি/ছোট উদাহরণ বাদ দিন)
    raw_examples = text_data.split('================================================================================')
    train_texts = []
    for ex in raw_examples:
        ex = ex.strip()
        if len(ex) > 100:  # যেসব উদাহরণ ১০০ ক্যারেক্টারের বেশি, সেগুলো নিন
            train_texts.append(ex)
    
    print(f"📊 Total training examples found: {len(train_texts)}")
    
    if not train_texts:
        print("❌ No valid training examples found in the file.")
        return
    
    # 3. টোকেনাইজার সেটআপ
    print("\n🔧 Setting up tokenizer...")
    model_name = "distilgpt2"  # CPU-তে দ্রুত কাজ করে
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # ডাটা টোকেনাইজ করুন
    encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    print(f"✅ Tokenizer ready (vocab size: {tokenizer.vocab_size})")
    
    # 4. মডেল সেটআপ
    print("\n🏗️ Setting up model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {model_name}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Device: {device}")
    
    # 5. PyTorch ডেটাসেট তৈরি করুন
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.input_ids = encodings['input_ids']
            self.attention_mask = encodings['attention_mask']
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.input_ids[idx]
            }
        
        def __len__(self):
            return len(self.input_ids)
    
    train_dataset = TextDataset(encodings)
    
    # 6. ট্রেনিং আর্গুমেন্টস
    print("\n⚙️ Setting up training arguments...")
    output_dir = "./llm_stock_model"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_steps=200,          # প্রতি ২০০ স্টেপে সেভ করবে (CPU-তে কম সেভ করা ভালো)
        save_total_limit=2,
        logging_steps=20,
        prediction_loss_only=True,
        learning_rate=5e-5,
        warmup_steps=50,
        weight_decay=0.01,
        fp16=False,
        dataloader_drop_last=False,
        report_to="none",
    )
    
    print(f"   Output directory: {output_dir}")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    
    # 7. ট্রেনিং শুরু
    print("\n🏋️ Starting training...")
    print("-" * 40)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
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
    
    # 8. মডেল সেভ
    print("\n💾 Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to: {output_dir}")
    
    # 9. Hugging Face-এ আপলোড (শুধুমাত্র টোকেন থাকলেই)
    if token:
        print("\n📤 Uploading model to Hugging Face...")
        model_repo_id = "ahashanahmed/llm-stock-model"
        try:
            from huggingface_hub import create_repo, upload_folder
            create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True, private=False)
            upload_folder(
                folder_path=output_dir,
                repo_id=model_repo_id,
                repo_type="model",
                commit_message="Upload trained LLM from local training_texts.txt"
            )
            print(f"✅ Model uploaded to: https://huggingface.co/{model_repo_id}")
        except Exception as e:
            print(f"⚠️ Upload failed: {e}")
            print("   You can manually upload the folder:", output_dir)
    else:
        print("\nℹ️ Skipping upload to Hugging Face (no token provided).")
    
    # 10. টেস্ট
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
    
    # 11. সারাংশ
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    print(f"   ✅ Model trained successfully!")
    print(f"   📁 Local model: {output_dir}")
    if token:
        print(f"   🤗 Hugging Face: https://huggingface.co/ahashanahmed/llm-stock-model")
    print(f"   📚 Training data source: {csv_path}")
    print("="*60)

if __name__ == "__main__":
    main()
