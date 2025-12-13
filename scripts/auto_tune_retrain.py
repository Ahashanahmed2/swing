import os
import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env import TradeEnv
import json

# 📂 Paths
main_df_path = './csv/mongodb.csv'
backtest_dir = './csv/backtest_result'
model_path = './csv/dqn_retrained'
liquidity_path = './csv/liquidity_system.csv'

# ---------------------------------------------------------
# 🔧 Load config (for risk context)
# ---------------------------------------------------------
CONFIG_PATH = "./config.json"
TOTAL_CAPITAL = 500000
RISK_PERCENT = 0.01
TEST_MODE = False  # ✅ NEW: Set to True for safe first run

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
        TOTAL_CAPITAL = cfg.get("total_capital", 500000)
        RISK_PERCENT = cfg.get("risk_percent", 0.01)
        TEST_MODE = cfg.get("test_mode", False)  # ✅ Enable test mode from config

print(f"✅ Config: Capital = {TOTAL_CAPITAL:,.00f} BDT, Risk = {RISK_PERCENT*100:.1f}%")
if TEST_MODE:
    print("🧪 TEST MODE: Training on top 5 symbols only")

# ---------------------------------------------------------
# 📊 Load critical system metrics
# ---------------------------------------------------------
strategy_metrics_path = "./output/ai_signal/strategy_metrics.csv"
symbol_ref_path = "./output/ai_signal/symbol_reference_metrics.csv"

strategy_metrics = pd.read_csv(strategy_metrics_path) if os.path.exists(strategy_metrics_path) else pd.DataFrame()
symbol_ref_metrics = pd.read_csv(symbol_ref_path) if os.path.exists(symbol_ref_path) else pd.DataFrame()

# ---------------------------------------------------------
# SAFE: Ensure folders exist
# ---------------------------------------------------------
os.makedirs(backtest_dir, exist_ok=True)

# ---------------------------------------------------------
# Load MongoDB main_df safely — ✅ WITH NaN HANDLING
# ---------------------------------------------------------
if not os.path.isfile(main_df_path):
    print(f"❌ MongoDB CSV missing → {main_df_path}")
    exit()

try:
    main_df = pd.read_csv(main_df_path)
    print(f"✅ Loaded {len(main_df)} rows from {main_df_path}")
except Exception as e:
    print(f"❌ Failed to load {main_df_path}: {e}")
    exit()

# ✅ CRITICAL: Fill NaN BEFORE processing
main_df = main_df.fillna({
    'open': main_df['close'],
    'high': main_df['close'],
    'low': main_df['close'],
    'volume': 0,
    'value': 0,
    'trades': 0,
    'change': 0,
    'marketCap': 0,
    'RSI': 50,
    'bb_upper': main_df['close'],
    'bb_middle': main_df['close'],
    'bb_lower': main_df['close'],
    'macd': 0,
    'macd_signal': 0,
    'macd_hist': 0,
    'zigzag': 0,
    'Hammer': 'FALSE',
    'BullishEngulfing': 'FALSE',
    'MorningStar': 'FALSE'
})

main_df['date'] = pd.to_datetime(main_df['date'], errors='coerce').dt.date
main_df['symbol'] = main_df['symbol'].str.upper()
main_df = main_df.dropna(subset=['date', 'symbol', 'close'])

print(f"🧹 Cleaned to {len(main_df)} rows (NaN removed)")

# ---------------------------------------------------------
# Load backtest files safely
# ---------------------------------------------------------
if not os.path.isdir(backtest_dir):
    print(f"⚠️ Backtest folder missing → {backtest_dir}")
    exit()

backtest_files = [f for f in os.listdir(backtest_dir) if f.endswith('.csv')]

if not backtest_files:
    print("⚠️ No backtest files found → skipping retrain.")
    exit()

# Concatenate all backtests
bt_df_list = []
for f in backtest_files:
    try:
        df = pd.read_csv(os.path.join(backtest_dir, f))
        df['symbol'] = df['symbol'].str.upper()
        bt_df_list.append(df)
    except Exception as e:
        print(f"⚠️ Failed to load backtest file {f}: {e}")

if not bt_df_list:
    print("⚠️ No valid backtest files → retrain aborted.")
    exit()

bt_df = pd.concat(bt_df_list, ignore_index=True)
bt_df = bt_df.dropna(subset=['symbol', 'outcome'])
print(f"📈 Loaded {len(bt_df)} clean backtest trades")

# ---------------------------------------------------------
# ✅ ENHANCED SYMBOL SELECTION
# ---------------------------------------------------------
tp_symbols = bt_df[bt_df['outcome'] == 'TP']['symbol'].value_counts()
sl_symbols = bt_df[bt_df['outcome'] == 'SL']['symbol'].value_counts()

high_expectancy_symbols = []
if not symbol_ref_metrics.empty:
    good_syms = symbol_ref_metrics[symbol_ref_metrics['Expectancy (BDT)'] > 100]
    high_expectancy_symbols = good_syms['Symbol'].str.upper().unique().tolist()

high_win_symbols = []
if not symbol_ref_metrics.empty:
    good_win = symbol_ref_metrics[symbol_ref_metrics['Win%'] > 65]
    high_win_symbols = good_win['Symbol'].str.upper().unique().tolist()

high_liquidity_symbols = []
if os.path.exists(liquidity_path):
    try:
        liquidity_df = pd.read_csv(liquidity_path)
        liquidity_df['symbol'] = liquidity_df['symbol'].str.upper()
        good_liq = liquidity_df[liquidity_df['liquidity_score'] >= 0.4]
        high_liquidity_symbols = good_liq['symbol'].unique().tolist()
    except Exception as e:
        print(f"⚠️ Failed to load liquidity: {e}")

quality_symbols = set(high_expectancy_symbols) & set(high_win_symbols)
if high_liquidity_symbols:
    quality_symbols = quality_symbols & set(high_liquidity_symbols)

# ✅ TEST MODE: Use only top 5 symbols for first run
if TEST_MODE and len(quality_symbols) > 5:
    top_syms = symbol_ref_metrics.sort_values('Expectancy (BDT)', ascending=False)['Symbol'].str.upper().head(5)
    quality_symbols = set([s for s in top_syms if s in quality_symbols])
    print(f"🧪 TEST MODE: Using {len(quality_symbols)} symbols: {list(quality_symbols)}")

print(f"🌟 {len(quality_symbols)} high-quality symbols selected")

# ---------------------------------------------------------
# Filter main_df
# ---------------------------------------------------------
filtered_df = main_df.copy()

if quality_symbols:
    filtered_df = filtered_df[filtered_df['symbol'].isin(quality_symbols)]

blacklist = sl_symbols[sl_symbols >= 3].index.tolist()
filtered_df = filtered_df[~filtered_df['symbol'].isin(blacklist)]

fast_tp = bt_df[(bt_df['outcome'] == 'TP') & (bt_df['duration_days'] <= 3)]
fast_tp_symbols = fast_tp['symbol'].unique().tolist()
if fast_tp_symbols:
    filtered_df = filtered_df[filtered_df['symbol'].isin(fast_tp_symbols)]

# Add system features
if not filtered_df.empty and not symbol_ref_metrics.empty:
    def add_system_features(row):
        sym = row['symbol'].upper()
        win_pct = 50.0
        expectancy = 0.0
        ref_row = symbol_ref_metrics[
            (symbol_ref_metrics['Symbol'].str.upper() == sym) &
            (symbol_ref_metrics['Reference'] == 'SWING')
        ]
        if not ref_row.empty:
            win_pct = float(ref_row.iloc[0].get('Win%', 50.0))
            expectancy = float(ref_row.iloc[0].get('Expectancy (BDT)', 0.0))
        return pd.Series([win_pct, expectancy], index=['system_win_pct', 'system_expectancy_bdt'])

    filtered_df[['system_win_pct', 'system_expectancy_bdt']] = filtered_df.apply(add_system_features, axis=1)

print(f"🔍 Final training set: {len(filtered_df)} rows across {filtered_df['symbol'].nunique()} symbols")

# ---------------------------------------------------------
# Empty dataset check
# ---------------------------------------------------------
if filtered_df.empty:
    print("⚠️ Filtered training data is empty → Retrain skipped.")
    exit()

# ---------------------------------------------------------
# ✅ ROBUST ENVIRONMENT CREATION (with error fallback)
# ---------------------------------------------------------
env = None
try:
    env = TradeEnv(
        maindf=filtered_df,
        gape_path="./csv/gape.csv",
        gapebuy_path="./csv/gape_buy.csv",
        shortbuy_path="./csv/short_buy.csv",
        rsi_diver_path="./csv/rsi_diver.csv",
        rsi_diver_retest_path="./csv/rsi_diver_retest.csv",
        trade_stock_path="./csv/trade_stock.csv",
        metrics_path=strategy_metrics_path,
        symbol_ref_path=symbol_ref_path,
        config_path=CONFIG_PATH,
        liquidity_path=liquidity_path
    )
    env = DummyVecEnv([lambda: env])
    print("✅ Environment created successfully")
except Exception as e:
    print(f"❌ Environment creation failed: {e}")
    print("🔄 Trying with minimal dataset...")
    # Fallback: Use just 1000 rows of POWERGRID
    fallback_df = filtered_df[filtered_df['symbol'] == 'POWERGRID'].head(1000)
    if len(fallback_df) < 100:
        fallback_df = filtered_df.head(1000)
    env = TradeEnv(maindf=fallback_df, config_path=CONFIG_PATH)
    env = DummyVecEnv([lambda: env])
    print("✅ Fallback environment created")

# ---------------------------------------------------------
# Load or create model — DQN-OPTIMIZED
# ---------------------------------------------------------
if os.path.exists(model_path + ".zip"):
    try:
        model = DQN.load(model_path, env=env, device='cpu')
        print("✅ Loaded existing model for fine-tuning")
    except Exception as e:
        print(f"⚠️ Model load failed, creating new: {e}")
        model = None
else:
    model = None

if model is None:
    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,      # ↓ Reduced for stability
        learning_starts=1000,   # ↓ Faster start
        batch_size=128,         # ↓ More stable
        tau=0.01,               # ↑ More stable target update
        gamma=0.99,             # ↑ Longer horizon
        train_freq=4,           # ↑ More frequent updates
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,  # ↑ More exploration
        device='cpu'
    )
    print("🆕 Created new DQN model with STABLE parameters")

# ---------------------------------------------------------
# 🚀 TRAINING with ERROR HANDLING
# ---------------------------------------------------------
print(f"\n🚀 Retraining DQN with {len(filtered_df)} rows...")
print("   🔁 100,000 timesteps (≈ 10-15 mins)")

try:
    model.learn(
        total_timesteps=100_000,  # ↓ Reduced for stability
        log_interval=1000,
        progress_bar=True
    )
    print("✅ Training completed successfully")
except Exception as e:
    print(f"❌ Training failed: {e}")
    print("🔄 Trying with shorter timesteps...")
    try:
        model.learn(total_timesteps=50_000, log_interval=500)
        print("✅ Short training completed")
    except Exception as e2:
        print(f"❌ Short training also failed: {e2}")
        exit(1)

# ---------------------------------------------------------
# ✅ Save & Report
# ---------------------------------------------------------
try:
    model.save(model_path)
    print(f"\n✅ Model saved at {model_path}.zip")
    
    # Simple report
    print("\n" + "="*50)
    print("📊 DQN TRAINING SUMMARY")
    print("="*50)
    if hasattr(model, 'ep_info_buffer') and model.ep_info_buffer:
        rewards = [ep['r'] for ep in model.ep_info_buffer]
        print(f"📈 Final Avg Reward: {np.mean(rewards[-10:]):+.3f}")
    print(f"🧮 Trained on: {filtered_df['symbol'].nunique()} symbols")
    print("="*50)
    
except Exception as e:
    print(f"❌ Model save failed: {e}")

# 🔁 Optional: Generate signals
auto_signal = input("\n🔄 Generate signals with new model? (y/n): ").strip().lower()
if auto_signal == 'y':
    try:
        from generate_signals import generate_signals
        generate_signals()
    except Exception as e:
        print(f"⚠️ Signal generation failed: {e}")
        print("💡 Tip: Check if TradeEnv.get_obs() returns correct shape")