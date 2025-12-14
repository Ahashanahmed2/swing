# evaluate_ppo.py
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Suppress warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.trading_env import TradingEnv  # আপনার এনভায়রনমেন্ট মডিউল


def load_data(data_dir: str = "./csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """লোড এবং প্রিপ্রসেস ডাটা"""
    print("📦 trade_stock.csv এবং mongodb.csv লোড করা হচ্ছে...")
    
    try:
        signals = pd.read_csv(f"{data_dir}/trade_stock.csv")
        market = pd.read_csv(f"{data_dir}/mongodb.csv")
        
        # তারিখ কনভার্ট করুন
        signals['date'] = pd.to_datetime(signals['date'])
        market['date'] = pd.to_datetime(market['date'])
        
        # সঠিক কলাম নাম চেক করুন
        required_cols = ['buy', 'SL', 'tp', 'diff']
        for col in required_cols:
            if col not in signals.columns:
                print(f"⚠️ {col} কলাম signals-এ নেই")
        
        print(f"✅ সিগনাল ডাটা: {signals.shape}, মার্কেট ডাটা: {market.shape}")
        return signals, market
        
    except Exception as e:
        print(f"❌ ডাটা লোড করতে সমস্যা: {e}")
        raise


def estimate_win_probability(row: Dict) -> float:
    """Win probability estimation based on technical factors"""
    base = 0.5  # Base probability
    
    # RRR based adjustment
    if row['final_RRR'] >= 2.0:
        base += 0.15
    elif row['final_RRR'] >= 1.5:
        base += 0.08
    elif row['final_RRR'] >= 1.0:
        base += 0.03
    else:
        base -= 0.05
    
    # Price difference based adjustment
    if row['buy'] > 0:
        diff_pct = abs(row['final_diff']) / row['buy']
        if diff_pct <= 0.02:  # Tight stop loss
            base += 0.10
        elif diff_pct <= 0.05:
            base += 0.05
    
    # RSI based adjustment (if available)
    if 'rsi' in row and not pd.isna(row['rsi']):
        rsi = row['rsi']
        if 30 <= rsi <= 70:  # Neutral zone
            base += 0.05
        elif rsi < 30:  # Oversold
            base += 0.10
        elif rsi > 70:  # Overbought
            base -= 0.05
    
    # Volume confirmation (if available)
    if 'volume' in row and not pd.isna(row['volume']):
        # You can add volume-based logic here
        pass
    
    # Ensure probability is within reasonable bounds
    return np.clip(base, 0.15, 0.90)


def calculate_position_size(row: Dict, initial_capital: float = 500000, 
                           risk_per_trade: float = 0.02) -> Tuple[int, float, float]:
    """Calculate position size based on risk management"""
    buy_price = row['buy']
    stop_loss = row['final_SL']
    
    # Calculate risk per share
    risk_per_share = buy_price - stop_loss
    if risk_per_share <= 0:
        return 0, 0.0, 0.0
    
    # Maximum risk per trade
    max_risk_amount = initial_capital * risk_per_trade
    
    # Calculate maximum shares based on risk
    max_shares_by_risk = int(max_risk_amount / risk_per_share)
    
    # Calculate maximum shares based on capital
    max_shares_by_capital = int(initial_capital / buy_price)
    
    # Use position ratio from model
    pos_ratio = np.clip(row.get('pos_ratio', 1.0), 0.0, 2.0)
    base_shares = min(max_shares_by_risk, max_shares_by_capital)
    
    # Apply position ratio
    final_shares = int(base_shares * pos_ratio)
    final_shares = max(100, final_shares)  # Minimum 100 shares
    
    # Calculate exposure and risk
    exposure = final_shares * buy_price
    risk_amount = final_shares * risk_per_share
    
    return final_shares, exposure, risk_amount


def process_symbol(signals: pd.DataFrame, market: pd.DataFrame, 
                  symbol: str, initial_capital: float = 500000) -> Optional[Dict]:
    """একটি সিম্বলের জন্য PPO মডেল দিয়ে সিগন্যাল জেনারেট করুন"""
    print(f"  📊 {symbol} প্রসেস করা হচ্ছে...")
    
    try:
        # প্রথমে মডেল চেক করুন
        model_path = f"./models/ppo_{symbol}.zip"
        if not os.path.exists(model_path):
            print(f"  ⚠️ {symbol} এর মডেল নেই: {model_path}")
            return None
        
        # এনভায়রনমেন্ট তৈরি করুন
        env = TradingEnv(signals, market, symbol=symbol, initial_capital=initial_capital)
        
        # মডেল লোড করুন
        print(f"  🔄 {symbol} মডেল লোড করা হচ্ছে...")
        model = PPO.load(model_path, env=env, device="cpu")
        
        # স্টেপ বাই স্টেপ প্রেডিকশন
        obs = env.reset()
        done = False
        enhanced_rows = []
        step_count = 0
        max_steps = min(1000, len(env.data))  # ম্যাক্সিমাম স্টেপ লিমিট
        
        while not done and step_count < max_steps:
            # মডেল থেকে একশন নিন
            action, _states = model.predict(obs, deterministic=True)
            
            # একশন ভ্যালু নিষ্কাশন
            # ধরে নিচ্ছি action = [pos_ratio, sl_mult, tp_mult, ...]
            if isinstance(action, np.ndarray):
                if len(action) >= 4:
                    pos_ratio, sl_mult, tp_mult, _ = action[:4]
                elif len(action) >= 3:
                    pos_ratio, sl_mult, tp_mult = action[:3]
                elif len(action) >= 2:
                    pos_ratio, sl_mult = action[:2]
                    tp_mult = 2.0  # ডিফল্ট
                else:
                    pos_ratio = action[0] if len(action) > 0 else 1.0
                    sl_mult = 1.0
                    tp_mult = 2.0
            else:
                pos_ratio = float(action)
                sl_mult = 1.0
                tp_mult = 2.0
            
            # ক্লিপ করা
            pos_ratio = np.clip(pos_ratio, 0.1, 2.0)
            sl_mult = np.clip(sl_mult, 0.5, 3.0)
            tp_mult = np.clip(tp_mult, 1.0, 4.0)
            
            # কারেন্ট মার্কেট ডাটা
            current_data = env.data.iloc[env.current_step] if env.current_step < len(env.data) else None
            
            if current_data is not None and 'buy' in current_data:
                buy_price = current_data['buy']
                
                # ATR বা volatility measure
                atr = current_data.get('atr', buy_price * 0.02)  # 2% ডিফল্ট
                
                # স্টপ লস এবং টেক প্রফিট ক্যালকুলেট
                final_SL = buy_price - (sl_mult * atr)
                final_TP = buy_price + (tp_mult * atr)
                
                # মিনিমাম/ম্যাক্সিমাম লিমিট
                final_SL = max(final_SL, buy_price * 0.85)  # ম্যাক্সিমাম 15% লস
                final_SL = min(final_SL, buy_price * 0.98)  # মিনিমাম 2% লস
                final_TP = min(final_TP, buy_price * 1.20)  # ম্যাক্সিমাম 20% প্রফিট
                
                # ডিফারেন্স এবং RRR
                final_diff = buy_price - final_SL
                if final_diff > 0:
                    final_RRR = (final_TP - buy_price) / final_diff
                else:
                    final_RRR = 1.0
                
                # পজিশন সাইজ ক্যালকুলেট
                final_shares, exposure, risk_amount = calculate_position_size(
                    {
                        'buy': buy_price,
                        'final_SL': final_SL,
                        'final_diff': final_diff,
                        'pos_ratio': pos_ratio
                    },
                    initial_capital=initial_capital
                )
                
                # রেজাল্ট ডিকশনারি তৈরি
                result_row = {
                    'symbol': symbol,
                    'date': current_data.get('date', ''),
                    'buy': buy_price,
                    'final_SL': round(final_SL, 2),
                    'final_tp': round(final_TP, 2),
                    'final_diff': round(final_diff, 2),
                    'final_RRR': round(final_RRR, 2),
                    'pos_ratio': round(pos_ratio, 2),
                    'sl_mult': round(sl_mult, 2),
                    'tp_mult': round(tp_mult, 2),
                    'final_position_size': final_shares,
                    'exposure_bdt': round(exposure, 2),
                    'actual_risk_bdt': round(risk_amount, 2)
                }
                
                # টেকনিক্যাল ইন্ডিকেটরস যোগ করুন (যদি থাকে)
                for indicator in ['rsi', 'volume', 'atr', 'macd', 'bb_width']:
                    if indicator in current_data:
                        result_row[indicator] = current_data[indicator]
                
                enhanced_rows.append(result_row)
            
            # পরবর্তী স্টেপ
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
        
        if not enhanced_rows:
            print(f"  ⚠️ {symbol} এর জন্য কোন সিগন্যাল জেনারেট হয়নি")
            return None
        
        # সবচেয়ে ভালো সিগন্যাল বাছাই করুন
        df_symbol = pd.DataFrame(enhanced_rows)
        
        # Win probability ক্যালকুলেট
        df_symbol['win%'] = df_symbol.apply(estimate_win_probability, axis=1) * 100
        df_symbol['win%'] = df_symbol['win%'].round(1)
        
        # সর্টিং: win% (উচ্চ থেকে নিম্ন), তারপর RRR (উচ্চ থেকে নিম্ন)
        df_sorted = df_symbol.sort_values(
            by=['win%', 'final_RRR', 'final_diff'],
            ascending=[False, False, True]  # final_diff ছোট থেকে বড়
        )
        
        # সেরা সিগন্যাল নিন
        best_signal = df_sorted.iloc[0].to_dict()
        
        print(f"  ✅ {symbol}: Win%={best_signal['win%']}%, RRR={best_signal['final_RRR']:.2f}")
        return best_signal
        
    except Exception as e:
        print(f"  ❌ {symbol} প্রসেস করতে সমস্যা: {str(e)[:100]}")
        return None


def main():
    """মেইন ইভ্যালুয়েশন পাইপলাইন"""
    print("=" * 60)
    print("🤖 PPO মডেল ইভ্যালুয়েশন এবং সিগন্যাল জেনারেশন")
    print("=" * 60)
    
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
    
    # মডেল ফোল্ডার চেক
    models_dir = "./models"
    if not os.path.exists(models_dir):
        print(f"❌ মডেল ফোল্ডার নেই: {models_dir}")
        print(f"   প্রথমে train_all_sb3.py রান করুন")
        return
    
    # শুধু যেসব সিম্বলের মডেল আছে
    available_models = []
    for symbol in symbols:
        model_path = f"{models_dir}/ppo_{symbol}.zip"
        if os.path.exists(model_path):
            available_models.append(symbol)
    
    print(f"📁 {len(available_models)} টি সিম্বলের মডেল পাওয়া গেছে")
    
    if len(available_models) == 0:
        print("❌ কোন মডেল পাওয়া যায়নি!")
        return
    
    # প্রতিটি সিম্বল প্রসেস করুন
    best_signals = []
    
    print(f"\n🚀 {len(available_models)} টি মডেল ইভ্যালুয়েট করা হচ্ছে...")
    
    for i, symbol in enumerate(available_models, 1):
        print(f"\n[{i}/{len(available_models)}] ", end="")
        
        signal = process_symbol(signals, market, symbol)
        if signal:
            best_signals.append(signal)
    
    if not best_signals:
        print("\n❌ কোন ভ্যালিড সিগন্যাল জেনারেট হয়নি")
        return
    
    # ফাইনাল DataFrame তৈরি
    df_final = pd.DataFrame(best_signals)
    
    # কলাম অর্ডার
    column_order = [
        'symbol', 'date', 'buy', 'final_SL', 'final_tp',
        'final_diff', 'final_RRR', 'win%', 'final_position_size',
        'exposure_bdt', 'actual_risk_bdt',
        'pos_ratio', 'sl_mult', 'tp_mult'
    ]
    
    # শুধু যে কলামগুলো আছে
    existing_cols = [col for col in column_order if col in df_final.columns]
    df_final = df_final[existing_cols].copy()
    
    # সর্টিং: final_diff (ছোট থেকে বড়) - রিস্ক কম এমন সিগন্যাল প্রথমে
    df_final = df_final.sort_values('final_diff', ascending=True).reset_index(drop=True)
    
    # ইনডেক্স নম্বর
    df_final.insert(0, 'Rank', range(1, len(df_final) + 1))
    
    # CSV তে সেভ
    output_path = "./csv/enhanced_signals.csv"
    df_final.to_csv(output_path, index=False)
    
    # রেজাল্ট প্রিন্ট
    print(f"\n{'='*60}")
    print("✅ ইভ্যালুয়েশন সম্পূর্ণ!")
    print(f"{'='*60}")
    print(f"📊 সামারি:")
    print(f"   • মোট সিম্বল: {len(symbols)}")
    print(f"   • মডেল পাওয়া গেছে: {len(available_models)}")
    print(f"   • সিগন্যাল জেনারেট হয়েছে: {len(best_signals)}")
    print(f"   • আউটপুট ফাইল: {output_path}")
    
    # ডিসপ্লে টেবিল
    display_cols = ['Rank', 'symbol', 'buy', 'final_SL', 'final_diff', 
                   'final_RRR', 'win%', 'final_position_size']
    display_cols = [col for col in display_cols if col in df_final.columns]
    
    print(f"\n📈 টপ সিগন্যালস:")
    print(df_final[display_cols].head(20).to_string(index=False))
    
    # স্ট্যাটিস্টিক্স
    if len(df_final) > 0:
        print(f"\n📊 স্ট্যাটিস্টিক্স:")
        print(f"   গড় Win%       : {df_final['win%'].mean():.1f}%")
        print(f"   গড় RRR        : {df_final['final_RRR'].mean():.2f}")
        print(f"   গড় Risk (BDT) : {df_final['final_diff'].mean():.2f}")
        print(f"   মোট এক্সপোজার : {df_final['exposure_bdt'].sum():,.0f} BDT")
        
        # রিস্ক-রিওয়ার্ড অ্যানালাইসিস
        high_quality = df_final[df_final['win%'] >= 60]
        if len(high_quality) > 0:
            print(f"\n🎯 High Quality Signals (Win% ≥ 60%): {len(high_quality)}")
            print(high_quality[['Rank', 'symbol', 'win%', 'final_RRR']].to_string(index=False))


if __name__ == "__main__":
    main()