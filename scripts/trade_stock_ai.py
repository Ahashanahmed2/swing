# ================== create_trade_stock_complete.py ==================
# DSEX MARKET REGIME AWARE VERSION (BANGLADESH)
# ✅ DSEX-specific features: Circuit Breaker, Thin Market, Sector Rotation
# ✅ Original Structure 100% Preserved
# ✅ No Code Deleted
# ✅ All Previous Features + DSEX Enhancements

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# DSEX-SPECIFIC CONFIGURATION
# =========================================================

DSEX_CIRCUIT_BREAKERS = {
    'individual': 0.10,  # ১০% individual stock circuit
    'index_high': 0.05,  # ৫% index up → halt
    'index_low': 0.05,   # ৫% index down → halt
}

DSEX_SECTORS = [
    'Bank', 'Pharmaceuticals & Chemicals', 'Fuel & Power', 'Engineering', 
    'Textile', 'Financial Institutions', 'Insurance', 'Telecommunication', 
    'Food & Allied', 'Cement', 'IT', 'Services & Real Estate',
    'Travel & Leisure', 'Paper & Printing', 'Tannery Industries',
    'Ceramics Sector', 'Jute', 'Mutual Funds', 'Miscellaneous'
]

THIN_MARKET_THRESHOLD = 10000    # ১০ হাজারের কম = thin
LOW_LIQUIDITY_THRESHOLD = 50000  # ৫০ হাজারের কম = low liquidity


def detect_market_regime(market_df):
    """
    Detect current market regime with DSEX-specific logic
    Returns: 'BULL', 'BEAR', 'SIDEWAYS', 'PANIC'
    """
    # Make sure date is datetime
    if 'date' in market_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(market_df['date']):
            market_df['date'] = pd.to_datetime(market_df['date'], format='mixed', errors='coerce')
            market_df = market_df.dropna(subset=['date'])
    
    # Get DSEX or broader market index
    if 'DSEX' in market_df['symbol'].values:
        index_data = market_df[market_df['symbol'] == 'DSEX'].sort_values('date')
    elif 'DS30' in market_df['symbol'].values:
        index_data = market_df[market_df['symbol'] == 'DS30'].sort_values('date')
    elif 'freeFloatMarketCap' in market_df.columns:
        # Top 10 large cap symbols as market proxy
        latest = market_df.sort_values('date').groupby('symbol').tail(1)
        top10 = latest.nlargest(10, 'freeFloatMarketCap')['symbol'].tolist()
        index_df = market_df[market_df['symbol'].isin(top10)]
        index_data = index_df.groupby('date')['close'].mean().reset_index()
        index_data['date'] = pd.to_datetime(index_data['date'])
    else:
        # Use average of all symbols
        all_prices = market_df.pivot(index='date', columns='symbol', values='close')
        index_data = pd.DataFrame({'close': all_prices.mean(axis=1)}).reset_index()
        index_data['date'] = pd.to_datetime(index_data['date'], format='mixed', errors='coerce')
        index_data = index_data.dropna(subset=['date']).sort_values('date')

    if len(index_data) < 20:
        return 'SIDEWAYS', 0

    # Calculate returns
    index_data['returns'] = index_data['close'].pct_change()
    index_data['sma_20'] = index_data['close'].rolling(20).mean()
    index_data['sma_50'] = index_data['close'].rolling(50).mean()

    latest = index_data.iloc[-1]
    prev_20 = index_data.iloc[-20] if len(index_data) >= 20 else index_data.iloc[0]
    prev_50 = index_data.iloc[-50] if len(index_data) >= 50 else index_data.iloc[0]

    # ✅ DSEX Panic Detection (৩+ দিন ৫% drop)
    daily_returns = index_data['returns'].dropna().tail(20)
    panic_days = (daily_returns < -0.03).sum()  # ৩%+ drop
    severe_panic = (daily_returns < -0.05).sum()  # ৫%+ drop
    
    if severe_panic >= 3 or (panic_days >= 5 and latest['close'] < prev_20['close'] * 0.90):
        regime = 'PANIC'
        strength = (prev_20['close'] - latest['close']) / latest['close'] * 100
    elif latest['close'] > latest['sma_50'] and latest['close'] > latest['sma_20']:
        regime = 'BULL'
        strength = (latest['close'] - latest['sma_50']) / latest['sma_50'] * 100
    elif latest['close'] < latest['sma_50'] and latest['close'] < latest['sma_20']:
        regime = 'BEAR'
        strength = (latest['sma_50'] - latest['close']) / latest['close'] * 100
    else:
        regime = 'SIDEWAYS'
        strength = 0

    # 20-day return
    twenty_day_return = (latest['close'] - prev_20['close']) / prev_20['close'] * 100

    return regime, {
        'strength': round(strength, 2),
        'twenty_day_return': round(twenty_day_return, 2),
        'current_price': round(latest['close'], 2),
        'sma_20': round(latest['sma_20'], 2),
        'sma_50': round(latest['sma_50'], 2),
        'panic_days': panic_days,
        'severe_panic_days': severe_panic
    }


def get_market_aware_thresholds(regime):
    """
    Return different thresholds based on market regime (DSEX-aware)
    """
    if regime == 'BULL':
        return {
            'STRONG_BUY': 0.65,
            'BUY': 0.55,
            'NEUTRAL': 0.40,
            'SELL': 0.25,
            'min_score_for_buy': 0.55,
            'description': 'Bull market - normal thresholds'
        }
    elif regime == 'BEAR':
        return {
            'STRONG_BUY': 0.80,
            'BUY': 0.70,
            'NEUTRAL': 0.50,
            'SELL': 0.30,
            'min_score_for_buy': 0.70,
            'description': 'Bear market - very selective'
        }
    elif regime == 'PANIC':
        return {
            'STRONG_BUY': 0.90,  # Almost impossible
            'BUY': 0.85,         # Only absolutely best
            'NEUTRAL': 0.60,
            'SELL': 0.35,
            'min_score_for_buy': 0.85,
            'description': 'PANIC market - almost no BUY signals'
        }
    else:  # SIDEWAYS
        return {
            'STRONG_BUY': 0.70,
            'BUY': 0.60,
            'NEUTRAL': 0.45,
            'SELL': 0.30,
            'min_score_for_buy': 0.60,
            'description': 'Sideways market - selective'
        }


def detect_thin_market(symbol_data):
    """DSEX Thin Market Detection - avoid illiquid stocks"""
    if len(symbol_data) < 10:
        return True, 'NO_DATA'
    
    avg_volume = symbol_data['volume'].tail(20).mean()
    avg_trades = symbol_data.get('trades', pd.Series([0]*len(symbol_data))).tail(20).mean() if 'trades' in symbol_data.columns else 0
    avg_value = symbol_data.get('value', pd.Series([0]*len(symbol_data))).tail(20).mean() if 'value' in symbol_data.columns else 0
    
    if avg_volume < THIN_MARKET_THRESHOLD:
        return True, 'THIN'
    elif avg_volume < LOW_LIQUIDITY_THRESHOLD:
        return True, 'LOW_LIQUIDITY'
    elif avg_trades < 10:  # ১০-এর কম trades/day
        return True, 'LOW_ACTIVITY'
    
    return False, 'NORMAL'


def detect_dsex_sector_rotation(market_df):
    """DSEX-specific sector rotation detection"""
    sector_returns = {}
    
    for sector in DSEX_SECTORS:
        sector_data = market_df[market_df['sector'].str.lower() == sector.lower()]
        if len(sector_data) > 50:
            avg_return = sector_data.groupby('date')['close'].mean().pct_change(20).iloc[-1] if len(sector_data) >= 20 else 0
            avg_volume = sector_data.groupby('date')['volume'].mean().tail(20).mean()
            sector_returns[sector] = {
                'return': avg_return,
                'volume': avg_volume,
                'symbol_count': sector_data['symbol'].nunique()
            }
    
    if sector_returns:
        return sorted(sector_returns.items(), key=lambda x: x[1]['return'], reverse=True)
    return []


def get_dsex_market_session(date):
    """DSEX trading session (Sunday-Thursday)"""
    weekday = pd.to_datetime(date).weekday()
    if weekday in [4, 5, 6]:  # Friday=4, Saturday=5 (holidays)
        return 'HOLIDAY'
    elif weekday == 0:  # Sunday
        return 'WEEK_START'
    elif weekday == 3:  # Wednesday
        return 'MID_WEEK'
    elif weekday == 4:  # Thursday (last trading day)
        return 'WEEK_END'
    return 'NORMAL'


def adjust_for_circuit_breaker(entry_price, stop_loss, atr_value):
    """DSEX Circuit Breaker-aware position sizing"""
    circuit_level_down = entry_price * (1 - DSEX_CIRCUIT_BREAKERS['individual'])
    
    if stop_loss < circuit_level_down * 0.95:
        return 0.5  # 50% position (too risky)
    elif stop_loss < circuit_level_down:
        return 0.7  # 70% position
    
    return 1.0  # Full position


def get_relative_strength_score(symbol_data, market_df):
    """
    Calculate relative strength vs market
    Higher score means stock is stronger than market
    """
    if len(symbol_data) < 20:
        return 0.5

    # Symbol returns
    symbol_returns = symbol_data['close'].pct_change().tail(20).mean()

    # Market returns (using all symbols average)
    market_avg = market_df.groupby('date')['close'].mean().pct_change().tail(20).mean()

    # Relative strength
    if market_avg < 0:
        relative = (symbol_returns - market_avg) / abs(market_avg) if market_avg != 0 else 0
    else:
        relative = symbol_returns / market_avg if market_avg > 0 else 0

    # Convert to 0-1 score
    score = min(max(relative + 0.5, 0), 1)
    return score


def create_trade_stock_complete():
    """DSEX Market regime aware trading signal generator"""

    print("="*70)
    print("📊 DSEX MARKET REGIME AWARE TRADING SIGNAL GENERATOR")
    print("="*70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏛️ Market: Bangladesh (DSEX)")
    print(f"⚡ Circuit Breaker: Individual {DSEX_CIRCUIT_BREAKERS['individual']*100:.0f}%")
    print("="*70)

    # Paths
    base_path = './csv/'
    files = {
        'sr': os.path.join(base_path, 'support_resistance.csv'),
        'market': os.path.join(base_path, 'mongodb.csv'),
        'pred': os.path.join(base_path, 'prediction_log.csv'),
        'conf': os.path.join(base_path, 'xgb_confidence.csv'),
        'meta': os.path.join(base_path, 'model_metadata.csv'),
        'rsi': os.path.join(base_path, 'rsi_diver.csv')
    }

    missing = [k for k, v in files.items() if not os.path.exists(v)]
    if missing:
        print(f"\n❌ Missing files: {missing}")
        return False

    # Load data
    print("\n📂 Loading data files...")
    sr_df = pd.read_csv(files['sr'])
    market_df = pd.read_csv(files['market'])
    pred_df = pd.read_csv(files['pred'])
    conf_df = pd.read_csv(files['conf'])
    meta_df = pd.read_csv(files['meta'])
    rsi_df = pd.read_csv(files['rsi'])

    # Convert dates with mixed format support
    sr_df['current_date'] = pd.to_datetime(sr_df['current_date'], format='mixed', errors='coerce')
    market_df['date'] = pd.to_datetime(market_df['date'], format='mixed', errors='coerce')
    pred_df['date'] = pd.to_datetime(pred_df['date'], format='mixed', errors='coerce')
    conf_df['date'] = pd.to_datetime(conf_df['date'], format='mixed', errors='coerce')
    
    # Remove rows with invalid dates
    pred_df = pred_df.dropna(subset=['date'])
    conf_df = conf_df.dropna(subset=['date'])
    sr_df = sr_df.dropna(subset=['current_date'])
    market_df = market_df.dropna(subset=['date'])

    print(f"   ✅ Support/Resistance: {len(sr_df)} records")
    print(f"   ✅ Market data: {len(market_df)} rows, {market_df['symbol'].nunique()} symbols")

    # =========================================================
    # DETECT MARKET REGIME (DSEX-Aware)
    # =========================================================
    print("\n📈 Detecting DSEX market regime...")
    regime, regime_stats = detect_market_regime(market_df)
    thresholds = get_market_aware_thresholds(regime)

    print(f"   🎯 Market Regime: {regime}")
    print(f"   📊 {regime_stats}")
    print(f"   🎚️ Thresholds: {thresholds['description']}")
    print(f"      STRONG BUY: ≥{thresholds['STRONG_BUY']}, BUY: ≥{thresholds['BUY']}")

    # ✅ DSEX Sector Rotation
    print("\n🔄 DSEX Sector Rotation Analysis...")
    sector_rotation = detect_dsex_sector_rotation(market_df)
    if sector_rotation:
        print(f"   Top 3 Sectors:")
        for sector, info in sector_rotation[:3]:
            print(f"      • {sector}: {info['return']*100:+.1f}% ({info['symbol_count']} symbols)")
        print(f"   Bottom 3 Sectors:")
        for sector, info in sector_rotation[-3:]:
            print(f"      • {sector}: {info['return']*100:+.1f}%")

    # In PANIC market
    if regime == 'PANIC':
        print(f"\n   🚨 PANIC MARKET DETECTED!")
        print(f"   💡 Strategy: Extremely selective - almost no BUY signals")
        print(f"   💡 Consider: Wait for market stabilization")

        # Create empty trade_stock.csv (no signals)
        empty_df = pd.DataFrame(columns=['symbol', 'date', 'buy', 'SL', 'tp', 'confidence', 'RRR'])
        empty_df.to_csv('./csv/trade_stock.csv', index=False)
        print(f"\n   ✅ Empty trade_stock.csv created (no BUY signals in PANIC market)")
        return True

    # In bear market, check severity
    if regime == 'BEAR':
        print(f"\n   ⚠️ BEAR MARKET DETECTED!")
        print(f"   💡 Strategy: Very selective - only high confidence signals")

        if regime_stats['twenty_day_return'] < -10:
            print(f"\n   🛑 Severe bear market ({regime_stats['twenty_day_return']:.1f}% down in 20 days)")
            print(f"   📝 No BUY signals will be generated - market too risky")

            empty_df = pd.DataFrame(columns=['symbol', 'date', 'buy', 'SL', 'tp', 'confidence', 'RRR'])
            empty_df.to_csv('./csv/trade_stock.csv', index=False)
            print(f"\n   ✅ Empty trade_stock.csv created (no BUY signals in bear market)")
            return True

    # Filter good models
    good_symbols = meta_df[meta_df['auc'] >= 0.55]['symbol'].unique()
    print(f"\n🔍 GOOD models: {len(good_symbols)} symbols")

    # Create prob_up
    if 'prob_up' not in pred_df.columns:
        if 'prediction' in pred_df.columns:
            pred_df['prob_up'] = pred_df.apply(
                lambda row: row['confidence_score'] / 100 if row['prediction'] == 1 
                else (100 - row['confidence_score']) / 100, axis=1
            )
        else:
            pred_df['prob_up'] = pred_df['confidence_score'] / 100

    pred_df = pred_df.merge(conf_df[['symbol', 'date', 'confidence_score']], 
                           on=['symbol', 'date'], how='left')
    pred_df = pred_df[pred_df['symbol'].isin(good_symbols)]

    # Support levels
    sr_support = sr_df[sr_df['type'] == 'support'].copy()
    print(f"   ✅ Support levels: {len(sr_support)}")

    # Calculate ATR
    print("\n📊 Calculating ATR...")

    def calc_atr(group):
        group = group.copy().reset_index(drop=True)
        high = group['high'].values
        low = group['low'].values
        close = group['close'].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(np.maximum(high - low, np.abs(high - prev_close)), np.abs(low - prev_close))
        atr = pd.Series(tr).rolling(window=14).mean()
        return atr.fillna(tr.mean())

    market_df['atr'] = 0.0
    for symbol in market_df['symbol'].unique():
        mask = market_df['symbol'] == symbol
        market_df.loc[mask, 'atr'] = calc_atr(market_df[mask]).values
    market_df['atr'] = market_df['atr'].fillna(market_df['close'] * 0.02)

    # Generate signals
    print("\n🎯 Generating DSEX-aware trading signals...")

    signals = []
    buy_signals = []

    for idx, row in sr_support.iterrows():
        symbol = row['symbol']
        current_date = row['current_date']
        support_level = row['current_low']
        strength = row.get('strength', 'Moderate')

        # Skip if symbol not in good models
        if symbol not in good_symbols:
            continue

        # Get market data
        sym_market = market_df[market_df['symbol'] == symbol].sort_values('date').reset_index(drop=True)

        if len(sym_market) < 10:
            continue

        # ✅ DSEX Thin Market Check
        is_thin, thin_type = detect_thin_market(sym_market)
        if is_thin and regime != 'BULL':
            continue  # Skip illiquid stocks in non-bull market
        
        thin_penalty = 0.7 if is_thin else 1.0

        # Find closest date
        date_diff = (sym_market['date'] - current_date).abs()
        if len(date_diff) == 0:
            continue

        min_diff_idx = date_diff.idxmin()
        min_diff = date_diff.min()

        if min_diff.days > 10:
            continue

        if min_diff_idx + 1 >= len(sym_market):
            continue

        # Check support held
        if sym_market.iloc[min_diff_idx + 1]['low'] <= support_level:
            continue

        # Get prediction
        pred_row = pred_df[(pred_df['symbol'] == symbol) & (pred_df['date'] == current_date)]

        if len(pred_row) == 0:
            # Try nearest date
            pred_dates = pred_df[pred_df['symbol'] == symbol]['date']
            if len(pred_dates) > 0:
                pred_date_diff = (pred_dates - current_date).abs()
                if len(pred_date_diff) > 0:
                    nearest_idx = pred_date_diff.idxmin()
                    pred_row = pred_df.loc[[nearest_idx]]

            if len(pred_row) == 0:
                continue

        prob_up = pred_row.iloc[0].get('prob_up', 0.5)
        if pd.isna(prob_up):
            prob_up = 0.5

        confidence = pred_row.iloc[0].get('confidence_score', 50)
        if pd.isna(confidence):
            confidence = 50

        # Calculate relative strength
        rel_strength = get_relative_strength_score(sym_market, market_df)

        # RSI divergence
        rsi_row = rsi_df[(rsi_df['symbol'] == symbol) & (rsi_df['last_date'] == current_date)]
        rsi_bonus = 0.10 if (len(rsi_row) > 0 and rsi_row.iloc[0].get('divergence_type') == 'Bullish') else 0

        # ✅ DSEX Market Session Adjustment
        market_session = get_dsex_market_session(current_date)
        session_mult = 0.95 if market_session in ['WEEK_START', 'WEEK_END'] else 1.0

        # Strength multiplier
        if regime == 'BEAR':
            strength_mult = {'Weak': 0.4, 'Moderate': 0.6, 'Strong': 0.8}.get(strength, 0.5)
        else:
            strength_mult = {'Weak': 0.6, 'Moderate': 0.8, 'Strong': 1.0}.get(strength, 0.7)

        # Final score
        base_score = (prob_up * 0.40) + ((confidence / 100) * 0.25) + (rel_strength * 0.20) + (rsi_bonus * 0.15)
        final_score = base_score * strength_mult * thin_penalty * session_mult
        final_score = min(max(final_score, 0), 1)

        # Determine signal
        if final_score >= thresholds['STRONG_BUY']:
            signal = "STRONG BUY"
        elif final_score >= thresholds['BUY']:
            signal = "BUY"
        elif final_score >= thresholds['NEUTRAL']:
            signal = "NEUTRAL"
        elif final_score >= thresholds['SELL']:
            signal = "SELL"
        else:
            signal = "STRONG SELL"

        # Calculate entry and risk
        entry_price = sym_market.iloc[min_diff_idx + 1]['close']
        atr_value = sym_market.iloc[min_diff_idx + 1].get('atr', entry_price * 0.02)
        if pd.isna(atr_value):
            atr_value = entry_price * 0.02

        # ✅ DSEX Circuit Breaker-aware stops
        if regime == 'BEAR':
            stop_loss = entry_price - (2.0 * atr_value)
            take_profit = entry_price + (1.5 * atr_value)
        else:
            stop_loss = entry_price - (1.5 * atr_value)
            take_profit = entry_price + (2.5 * atr_value)

        # ✅ Circuit Breaker Adjustment
        cb_mult = adjust_for_circuit_breaker(entry_price, stop_loss, atr_value)
        stop_loss = stop_loss * cb_mult if cb_mult < 1.0 else stop_loss

        risk = entry_price - stop_loss
        rrr = round((take_profit - entry_price) / risk, 2) if risk > 0 else 0

        signal_record = {
            'symbol': symbol,
            'date': current_date.strftime('%Y-%m-%d'),
            'signal': signal,
            'score': round(final_score, 4),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'rrr': rrr,
            'support_level': round(support_level, 2),
            'strength': strength,
            'xgb_prob': round(prob_up, 4),
            'xgb_confidence': confidence,
            'rel_strength': round(rel_strength, 3),
            'rsi_divergence': 'Bullish' if rsi_bonus > 0 else 'None',
            'market_regime': regime,
            'market_session': market_session,
            'thin_market': thin_type if is_thin else 'NORMAL'
        }

        signals.append(signal_record)

        if signal in ['STRONG BUY', 'BUY'] and final_score >= thresholds['min_score_for_buy']:
            buy_signals.append(signal_record)

    print(f"\n   📊 Results:")
    print(f"      Total signals: {len(signals)}")
    print(f"      BUY signals: {len(buy_signals)}")
    print(f"      SELL/HOLD signals: {len(signals) - len(buy_signals)}")

    # Create PPO file
    if len(buy_signals) > 0:
        ppo_df = pd.DataFrame([{
            'symbol': s['symbol'],
            'date': s['date'],
            'buy': s['entry_price'],
            'SL': s['stop_loss'],
            'tp': s['take_profit'],
            'confidence': s['score'],
            'RRR': s['rrr']
        } for s in buy_signals])

        ppo_df.to_csv('./csv/trade_stock.csv', index=False)
        print(f"\n   ✅ {len(buy_signals)} BUY signals saved to trade_stock.csv")
    else:
        print(f"\n   ⚠️ No BUY signals in {regime} market")
        print(f"   💡 This is CORRECT - market conditions don't favor buying")

        empty_df = pd.DataFrame(columns=['symbol', 'date', 'buy', 'SL', 'tp', 'confidence', 'RRR'])
        empty_df.to_csv('./csv/trade_stock.csv', index=False)
        print(f"   ✅ Empty trade_stock.csv created")

    # Save detailed signals
    if signals:
        output_df = pd.DataFrame(signals)
        output_df.to_csv('./csv/trade_stock_advanced.csv', index=False)

        print("\n" + "="*70)
        print("📊 DSEX SIGNAL SUMMARY")
        print("="*70)
        print(output_df['signal'].value_counts().to_string())
        
        if 'market_regime' in output_df.columns:
            print(f"\n📈 Market Regime Distribution:")
            print(output_df['market_regime'].value_counts().to_string())
        
        if 'thin_market' in output_df.columns:
            print(f"\n💧 Liquidity Distribution:")
            print(output_df['thin_market'].value_counts().to_string())

        if len(buy_signals) > 0:
            print(f"\n🔥 BUY SIGNALS ({len(buy_signals)}):")
            buy_df = pd.DataFrame(buy_signals)
            print(buy_df[['symbol', 'signal', 'score', 'entry_price', 'rrr', 'market_regime']].head(10).to_string())

    print("\n" + "="*70)
    print(f"✅ DSEX Market Regime: {regime}")
    print(f"✅ BUY signals generated: {len(buy_signals)}")
    print(f"✅ DSEX Circuit Breaker: Active")
    print(f"✅ Thin Market Filter: Active")
    print("="*70)

    return True


if __name__ == "__main__":
    success = create_trade_stock_complete()
    if not success:
        print("\n❌ Trade stock generation failed!")
    else:
        print("\n🎉 DSEX Trade stock generation complete!")