# create_trade_stock_ai_advanced.py - Complete Advanced Trading Signal Generator
# Features:
# 1. Multi-source integration (XGBoost, Support/Resistance, RSI Divergence)
# 2. ATR-based dynamic stop loss
# 3. Fibonacci retracement take profit levels
# 4. ZigZag + Bollinger Bands confirmation
# 5. Smart scoring system with configurable weights
# 6. Complete error handling
# 7. Professional output formatting

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# CONFIGURATION
# =========================================================

# Scoring weights
WEIGHTS = {
    'xgb_prob': 0.50,      # XGBoost probability weight
    'xgb_conf': 0.30,      # XGBoost confidence weight
    'rsi_div': 0.20,       # RSI divergence weight
}

# Strength multipliers
STRENGTH_MULTIPLIER = {
    'Weak': 0.5,
    'Moderate': 0.7,
    'Strong': 1.0
}

# Signal thresholds
SIGNAL_THRESHOLDS = {
    'STRONG_BUY': 0.75,
    'BUY': 0.60,
    'NEUTRAL': 0.40,
    'SELL': 0.25,
    'STRONG_SELL': 0.00
}

# Fibonacci levels for TP
FIB_LEVELS = [0.382, 0.618, 1.0, 1.618]  # Added 161.8%

# ATR multiplier for SL
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 3.0

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_float(value, default=0.0):
    """Safely convert to float"""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except:
        return default

def safe_get(df, col, idx, default=0):
    """Safely get value from dataframe"""
    try:
        if col in df.columns and idx < len(df):
            val = df.iloc[idx][col]
            if pd.isna(val):
                return default
            return val
        return default
    except:
        return default

def calculate_atr(df, period=14):
    """Calculate ATR if not present"""
    if 'atr' in df.columns:
        return df['atr']
    
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    return atr.fillna(tr.mean())

def calculate_zigzag(df, price, default_low, default_high):
    """Calculate ZigZag levels"""
    zigzag_col = 'zigzag' if 'zigzag' in df.columns else None
    
    if zigzag_col:
        zigzag_values = df[zigzag_col].dropna()
        zigzag_low = zigzag_values[zigzag_values < price].min() if len(zigzag_values[zigzag_values < price]) > 0 else default_low
        zigzag_high = zigzag_values[zigzag_values > price].max() if len(zigzag_values[zigzag_values > price]) > 0 else default_high
    else:
        zigzag_low = default_low
        zigzag_high = default_high
    
    return zigzag_low, zigzag_high

# =========================================================
# MAIN FUNCTION
# =========================================================

def create_trade_stock_ai_advanced():
    """Create advanced trade_stock.csv with multiple technical factors"""
    
    print("="*70)
    print("📊 CREATING ADVANCED TRADING SIGNALS")
    print("="*70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)

    # -----------------------------
    # Paths
    # -----------------------------
    sr_path = './csv/support_resistance.csv'
    mongo_path = './csv/mongodb.csv'
    pred_path = './csv/prediction_log.csv'
    conf_path = './csv/xgb_confidence.csv'
    meta_path = './csv/model_metadata.csv'
    rsi_path = './csv/rsi_diver.csv'
    output_path = './csv/trade_stock_advanced.csv'

    # Check all required files
    missing_files = []
    for path, name in [(sr_path, 'support_resistance'), (mongo_path, 'mongodb'), 
                       (pred_path, 'prediction_log'), (conf_path, 'xgb_confidence'),
                       (meta_path, 'model_metadata'), (rsi_path, 'rsi_diver')]:
        if not os.path.exists(path):
            missing_files.append(name)
            print(f"❌ {name}.csv not found!")
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        print("   Please run required scripts first.")
        return False

    # -----------------------------
    # Load CSVs
    # -----------------------------
    print("\n📂 Loading data files...")
    
    sr_df = pd.read_csv(sr_path)
    mongo_df = pd.read_csv(mongo_path)
    pred_df = pd.read_csv(pred_path)
    conf_df = pd.read_csv(conf_path)
    meta_df = pd.read_csv(meta_path)
    rsi_df = pd.read_csv(rsi_path)

    # Convert dates
    sr_df['current_date'] = pd.to_datetime(sr_df['current_date'])
    mongo_df['date'] = pd.to_datetime(mongo_df['date'])
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    conf_df['date'] = pd.to_datetime(conf_df['date'])
    rsi_df['last_date'] = pd.to_datetime(rsi_df['last_date'])

    print(f"   ✅ Support/Resistance: {len(sr_df)} records")
    print(f"   ✅ Market data: {len(mongo_df)} rows")
    print(f"   ✅ Prediction log: {len(pred_df)} records")
    print(f"   ✅ XGBoost confidence: {len(conf_df)} records")
    print(f"   ✅ Model metadata: {len(meta_df)} records")
    print(f"   ✅ RSI divergence: {len(rsi_df)} records")

    # -----------------------------
    # Filter Data
    # -----------------------------
    print("\n🔍 Filtering data...")
    
    # Keep only support levels
    sr_df = sr_df[sr_df['type'] == 'support']
    print(f"   ✅ Support levels only: {len(sr_df)}")
    
    # Sort and deduplicate
    mongo_df = mongo_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    pred_df = pred_df.sort_values(['symbol', 'date']).drop_duplicates(subset=['symbol', 'date'], keep='last')
    
    # Merge XGBoost data
    xgb_df = pd.merge(pred_df, conf_df, on=['symbol', 'date'], how='left')
    
    # Filter GOOD models only (AUC >= 0.55 or 0.6)
    meta_df = meta_df[meta_df['auc'] >= 0.55]
    good_symbols = meta_df['symbol'].unique()
    xgb_df = xgb_df[xgb_df['symbol'].isin(good_symbols)]
    print(f"   ✅ GOOD models: {len(good_symbols)} symbols")
    
    # Calculate ATR for all symbols
    print("\n📊 Calculating technical indicators...")
    mongo_df['atr'] = mongo_df.groupby('symbol').apply(
        lambda x: calculate_atr(x)
    ).reset_index(level=0, drop=True)
    
    # Calculate RSI if not present
    if 'rsi' not in mongo_df.columns:
        print("   ⚠️ RSI not found, calculating...")
        def calc_rsi(group, period=14):
            delta = group['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        
        mongo_df['rsi'] = mongo_df.groupby('symbol')['close'].transform(
            lambda x: calc_rsi(pd.DataFrame({'close': x}))
        )
    
    print("   ✅ Technical indicators ready")

    # -----------------------------
    # Process Each Support Level
    # -----------------------------
    print("\n🎯 Generating trading signals...")
    
    results = []
    skipped_count = 0
    processed_count = 0
    
    for _, row in sr_df.iterrows():
        symbol = row['symbol']
        current_date = row['current_date']
        current_low = row['current_low']
        strength = row.get('strength', 'Weak')
        
        # Get symbol data
        df_symbol = mongo_df[mongo_df['symbol'] == symbol].reset_index(drop=True)
        if len(df_symbol) < 2:
            skipped_count += 1
            continue
        
        # Find matching index
        match_idx = df_symbol[df_symbol['date'] == current_date].index
        if len(match_idx) == 0:
            continue
        
        idx = match_idx[0]
        if idx + 1 >= len(df_symbol):
            continue
        
        # Check if support held (next candle low > support level)
        next_row = df_symbol.iloc[idx + 1]
        
        if next_row['low'] <= current_low:
            continue  # Support broken, skip
        
        processed_count += 1
        
        # Get XGBoost data
        xgb_match = xgb_df[(xgb_df['symbol'] == symbol) & (xgb_df['date'] == current_date)]
        
        if len(xgb_match) == 0:
            continue
        
        xgb_row = xgb_match.iloc[0]
        
        # Get probability (handle different column names)
        if 'prob_up' in xgb_row and pd.notna(xgb_row['prob_up']):
            prob = safe_float(xgb_row['prob_up'], 0.5)
        elif 'confidence_score' in xgb_row and pd.notna(xgb_row['confidence_score']):
            prob = safe_float(xgb_row['confidence_score'], 50) / 100
        else:
            prob = 0.5
        
        confidence = safe_float(xgb_row.get('confidence_score', 50), 50)
        
        # RSI divergence check
        rsi_match = rsi_df[(rsi_df['symbol'] == symbol) & (rsi_df['last_date'] == current_date)]
        rsi_weight = 0.2 if (not rsi_match.empty and rsi_match.iloc[0]['divergence_type'] == 'Bullish') else 0
        rsi_type = rsi_match.iloc[0]['divergence_type'] if not rsi_match.empty else ''
        
        # Calculate buy score
        strength_mult = STRENGTH_MULTIPLIER.get(strength, 0.5)
        buy_score = ((prob * WEIGHTS['xgb_prob'] + 
                     (confidence/100) * WEIGHTS['xgb_conf'] + 
                     rsi_weight) * strength_mult)
        
        buy_score = min(max(buy_score, 0), 1)  # Clamp between 0 and 1
        
        # Determine signal
        if buy_score >= SIGNAL_THRESHOLDS['STRONG_BUY']:
            signal = "STRONG BUY"
        elif buy_score >= SIGNAL_THRESHOLDS['BUY']:
            signal = "BUY"
        elif buy_score >= SIGNAL_THRESHOLDS['NEUTRAL']:
            signal = "NEUTRAL"
        elif buy_score >= SIGNAL_THRESHOLDS['SELL']:
            signal = "SELL"
        else:
            signal = "STRONG SELL"
        
        # Skip neutral signals
        if signal == "NEUTRAL":
            continue
        
        # Get price and ATR
        price = next_row['close']
        atr = safe_get(df_symbol, 'atr', idx + 1, price * 0.02)
        
        # Calculate SL using ATR
        sl_atr = price - (ATR_SL_MULTIPLIER * atr)
        
        # Get ZigZag and Bollinger levels
        zigzag_low, zigzag_high = calculate_zigzag(df_symbol, price, current_low, price)
        
        bb_lower = safe_get(df_symbol, 'bb_lower', idx, current_low)
        bb_upper = safe_get(df_symbol, 'bb_upper', idx, price)
        
        # Final SL (most conservative)
        sl_final = min(sl_atr, zigzag_low, bb_lower)
        
        # Calculate TP levels using Fibonacci
        risk = price - sl_final
        if risk <= 0:
            risk = price * 0.03
        
        tp_levels = [price + (risk * fib) for fib in FIB_LEVELS]
        
        # Calculate RRR for each TP
        rrr_levels = [round((tp - price) / risk, 2) for tp in tp_levels]
        
        # Final TP (most aggressive)
        tp_final = max(tp_levels + [zigzag_high, bb_upper])
        rrr_final = round((tp_final - price) / risk, 2) if risk > 0 else 0
        
        # Get RSI value
        rsi_value = safe_get(df_symbol, 'rsi', idx + 1, 50)
        
        # Create record
        results.append({
            'type': row['type'],
            'symbol': symbol,
            'level_date': row['level_date'],
            'level_price': round(row['level_price'], 2),
            'gap_days': row['gap_days'],
            'strength': strength,
            'xgb_prob': round(prob, 4),
            'xgb_confidence': round(confidence, 2),
            'rsi_divergence': rsi_type,
            'buy_score': round(buy_score, 4),
            'signal': signal,
            # Entry and risk management
            'entry_price': round(price, 2),
            'stop_loss': round(sl_final, 2),
            'tp1': round(tp_levels[0], 2),
            'tp2': round(tp_levels[1], 2),
            'tp3': round(tp_levels[2], 2),
            'tp4': round(tp_levels[3], 2) if len(tp_levels) > 3 else 0,
            'rrr_tp1': rrr_levels[0],
            'rrr_tp2': rrr_levels[1],
            'rrr_tp3': rrr_levels[2],
            'rrr_final': rrr_final,
            # Technical levels
            'zigzag_low': round(zigzag_low, 2),
            'zigzag_high': round(zigzag_high, 2),
            'bb_lower': round(bb_lower, 2),
            'bb_upper': round(bb_upper, 2),
            'atr': round(atr, 4),
            'rsi': round(rsi_value, 2),
            'risk_amount': round(risk, 2),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    print(f"\n   ✅ Processed: {processed_count} support levels")
    print(f"   ⚠️ Skipped: {skipped_count} (insufficient data)")
    print(f"   🎯 Generated: {len(results)} trading signals")
    
    # -----------------------------
    # Create Output DataFrame
    # -----------------------------
    if not results:
        print("\n❌ No signals generated!")
        return False
    
    output_df = pd.DataFrame(results)
    
    # Sort by score and confidence
    output_df = output_df.sort_values(
        by=['buy_score', 'xgb_confidence', 'gap_days'], 
        ascending=[False, False, True]
    )
    
    # Add position sizing recommendation
    output_df['position_size'] = output_df['buy_score'].apply(
        lambda x: 'Large' if x > 0.8 else ('Medium' if x > 0.65 else 'Small')
    )
    
    output_df['risk_level'] = output_df['buy_score'].apply(
        lambda x: 'Low' if x > 0.75 else ('Medium' if x > 0.6 else 'High')
    )
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    
    # -----------------------------
    # Print Summary
    # -----------------------------
    print("\n" + "="*70)
    print("📊 SIGNAL SUMMARY")
    print("="*70)
    
    signal_counts = output_df['signal'].value_counts()
    for signal, count in signal_counts.items():
        print(f"   {signal}: {count}")
    
    print(f"\n📈 Position Sizing Guide:")
    print(f"   • Large position (80%+ score): High confidence trades")
    print(f"   • Medium position (65-80%): Normal trades")
    print(f"   • Small position (60-65%): Low confidence trades")
    
    print(f"\n🔥 TOP 10 STRONGEST SIGNALS:")
    print("-"*70)
    
    display_cols = ['symbol', 'entry_price', 'stop_loss', 'tp1', 'tp2', 'tp3', 
                    'rrr_final', 'buy_score', 'signal', 'position_size']
    
    print(output_df.head(10)[display_cols].to_string())
    
    print("\n" + "="*70)
    print(f"✅ Trade signals saved to: {output_path}")
    print(f"   Total signals: {len(output_df)}")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # -----------------------------
    # Create Simplified Version for PPO
    # -----------------------------
    print("\n📝 Creating simplified version for PPO training...")
    
    ppo_df = output_df[output_df['signal'].isin(['STRONG BUY', 'BUY'])].copy()
    
    if not ppo_df.empty:
        ppo_trade = ppo_df[['symbol', 'entry_price', 'stop_loss', 'tp1', 'buy_score']].copy()
        ppo_trade.columns = ['symbol', 'buy', 'SL', 'tp', 'confidence']
        ppo_trade['date'] = datetime.now().strftime('%Y-%m-%d')
        ppo_trade['RRR'] = ppo_trade.apply(
            lambda x: round((x['tp'] - x['buy']) / (x['buy'] - x['SL']), 2), axis=1
        )
        ppo_trade['source'] = 'Advanced_AI'
        
        ppo_path = './csv/trade_stock.csv'
        ppo_trade.to_csv(ppo_path, index=False)
        print(f"   ✅ PPO-ready signals saved to: {ppo_path}")
        print(f"   📊 PPO signals: {len(ppo_trade)}")
    else:
        print("   ⚠️ No BUY signals for PPO training")
    
    return True

# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    success = create_trade_stock_ai_advanced()
    
    if success:
        print("\n🎉 Trade stock generation complete!")
    else:
        print("\n❌ Trade stock generation failed!")