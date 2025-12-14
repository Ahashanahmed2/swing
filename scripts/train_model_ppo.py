# train_all_sb3.py
import pandas as pd
import numpy as np
import os
import warnings
import time
from datetime import datetime
from typing import Dict, List, Tuple

# Suppress warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from envs.trading_env import TradingEnv


def load_data(data_dir: str = "./csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """লোড এবং প্রিপ্রসেস ডাটা"""
    print("📦 trade_stock.csv এবং mongodb.csv লোড করা হচ্ছে...")
    
    try:
        signals = pd.read_csv(f"{data_dir}/trade_stock.csv")
        market = pd.read_csv(f"{data_dir}/mongodb.csv")
        
        # তারিখ কনভার্ট করুন
        signals['date'] = pd.to_datetime(signals['date'])
        market['date'] = pd.to_datetime(market['date'])
        
        # তারিখ অনুযায়ী সর্ট করুন
        signals = signals.sort_values(['symbol', 'date'])
        market = market.sort_values(['symbol', 'date'])
        
        print(f"✅ সিগনাল ডাটা: {signals.shape}, মার্কেট ডাটা: {market.shape}")
        return signals, market
        
    except Exception as e:
        print(f"❌ ডাটা লোড করতে সমস্যা: {e}")
        raise


def check_data_for_symbol(signals: pd.DataFrame, market: pd.DataFrame, symbol: str) -> bool:
    """একটি সিম্বলের জন্য ডাটা আছে কিনা চেক করুন"""
    symbol_signals = signals[signals['symbol'] == symbol]
    symbol_market = market[market['symbol'] == symbol]
    
    if len(symbol_signals) == 0:
        print(f"  ⚠️ {symbol} এর জন্য সিগনাল ডাটা নেই")
        return False
        
    if len(symbol_market) == 0:
        print(f"  ⚠️ {symbol} এর জন্য মার্কেট ডাটা নেই")
        return False
    
    # মিনিমাম ডাটা পয়েন্ট চেক করুন
    if len(symbol_market) < 50:
        print(f"  ⚠️ {symbol} এর জন্য পর্যাপ্ত ডাটা নেই: {len(symbol_market)} রেকর্ড")
        return False
    
    return True


def train_symbol(signals: pd.DataFrame, market: pd.DataFrame, symbol: str, 
                total_timesteps: int = 50000) -> bool:
    """একটি সিম্বলের জন্য PPO মডেল ট্রেন করুন"""
    print(f"\n📊 {symbol} ট্রেনিং শুরু...")
    
    # প্রথমে ডাটা চেক করুন
    if not check_data_for_symbol(signals, market, symbol):
        return False
    
    try:
        # এনভায়রনমেন্ট তৈরি করুন
        env = TradingEnv(signals, market, symbol=symbol)
        print(f"  ✅ এনভায়রনমেন্ট তৈরি হয়েছে")
        
        # PPO মডেল তৈরি করুন
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            verbose=0,
            device="cpu",
            seed=42
        )
        print(f"  ✅ PPO মডেল তৈরি হয়েছে")
        
        # ট্রেনিং শুরু
        start_time = time.time()
        print(f"  ⏳ {total_timesteps:,} টাইমস্টেপ ট্রেনিং চলছে...")
        
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True  # প্রোগ্রেস বার দেখান
        )
        
        training_time = time.time() - start_time
        print(f"  ✅ ট্রেনিং সম্পূর্ণ! সময় লেগেছে: {training_time:.1f} সেকেন্ড")
        
        # মডেল সেভ করুন
        model_path = f"./models/ppo_{symbol}.zip"
        os.makedirs("models", exist_ok=True)
        
        model.save(model_path)
        print(f"  💾 মডেল সেভ হয়েছে: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ট্রেনিং ব্যর্থ: {str(e)}")
        return False


def main():
    """মেইন ট্রেনিং পাইপলাইন"""
    print("=" * 60)
    print("🤖 PPO ট্রেডিং মডেল ট্রেনিং")
    print("=" * 60)
    
    # ডিরেক্টরি তৈরি করুন
    os.makedirs("models", exist_ok=True)
    
    # ডাটা লোড করুন
    try:
        signals, market = load_data()
    except Exception as e:
        print(f"❌ ডাটা লোড করতে পারিনি: {e}")
        return
    
    # কমন সিম্বল খুঁজুন
    symbols = sorted(set(signals['symbol']) & set(market['symbol']))
    
    if not symbols:
        print("❌ দুটি ফাইলের মধ্যে কোন কমন সিম্বল নেই!")
        return
    
    print(f"\n✅ {len(symbols)} টি সিম্বল পাওয়া গেছে")
    
    # যেসব সিম্বল ট্রেন করতে হবে (প্রথম ১০টা, অথবা সবগুলো)
    # symbols_to_train = symbols[:10]  # প্রথম ১০টা
    symbols_to_train = symbols  # সবগুলো
    
    print(f"🎯 {len(symbols_to_train)} টি সিম্বল ট্রেন করা হবে")
    
    # ট্রেনিং শুরু
    results = []
    
    for i, symbol in enumerate(symbols_to_train, 1):
        print(f"\n[{i}/{len(symbols_to_train)}] {'='*40}")
        
        success = train_symbol(
            signals=signals,
            market=market,
            symbol=symbol,
            total_timesteps=50000  # টাইমস্টেপ কমিয়েছি, বাড়াতে পারেন
        )
        
        results.append({
            'symbol': symbol,
            'success': success,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # রেজাল্ট সেভ করুন
    results_df = pd.DataFrame(results)
    results_df.to_csv("./models/training_results.csv", index=False)
    
    # সামারি দেখান
    print(f"\n{'='*60}")
    print("📊 ট্রেনিং সামারি")
    print(f"{'='*60}")
    
    success_count = results_df['success'].sum()
    total_count = len(results_df)
    
    print(f"✅ সফল: {success_count} / {total_count}")
    
    if success_count < total_count:
        failed = results_df[results_df['success'] == False]['symbol'].tolist()
        print(f"❌ ব্যর্থ: {failed}")
    
    print(f"\n📁 মডেলগুলো সেভ হয়েছে: ./models/ ফোল্ডারে")
    print(f"📄 রেজাল্টস সেভ হয়েছে: ./models/training_results.csv")
    
    # মডেল লোড করার উদাহরণ
    print(f"\n🔧 মডেল লোড করার উদাহরণ:")
    print(f'''
from stable_baselines3 import PPO
from envs.trading_env import TradingEnv

# মডেল লোড করুন
model = PPO.load("models/ppo_YOUR_SYMBOL.zip", device="cpu")

# ব্যবহার করুন
obs = env.reset()
action, _states = model.predict(obs, deterministic=True)
    ''')


if __name__ == "__main__":
    main()