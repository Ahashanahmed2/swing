# evaluate_ppo.py
import pandas as pd
import numpy as np
import torch
import os
from envs.trading_env import TradingEnv
from ppo_agent import PPOAgent

def load_data():
    signals = pd.read_csv("./csv/trade_stock.csv")
    market = pd.read_csv("./csv/mongodb.csv")
    signals['date'] = pd.to_datetime(signals['date'])
    market['date'] = pd.to_datetime(market['date'])
    return signals, market

def estimate_win_probability(row):
    """
    আপনার পছন্দমতো 'win%' কলাম — PPO একশন ও মার্কেট কনটেক্সট থেকে হিউরিস্টিক:
    - ছোট diff (tight SL) → higher win%
    - বড় RRR → higher quality, but lower win% (trade-off)
    - ভলিউম ↑, RSI in zone → boost
    """
    base = 0.5  # baseline
    
    # 1. Risk-reward impact
    if row['final_RRR'] >= 2.0:
        base += 0.10
    elif row['final_RRR'] >= 1.5:
        base += 0.05
    else:
        base -= 0.05
    
    # 2. SL tightness (diff %)
    diff_pct = row['final_diff'] / row['buy'] if row['buy'] > 0 else 0.01
    if diff_pct <= 0.02:   # ≤2% risk
        base += 0.10
    elif diff_pct <= 0.03:
        base += 0.05
    
    # 3. Volume & RSI (if available)
    if 'volume' in row and 'rsi' in row:
        vol_ratio = row['volume'] / row.get('avg_volume', 1e-5) if 'avg_volume' in row else 1.0
        if vol_ratio > 1.2:
            base += 0.05
        rsi = row['rsi']
        if 30 <= rsi <= 60:  # healthy zone
            base += 0.05
        elif rsi < 25 or rsi > 75:
            base -= 0.05
    
    return np.clip(base, 0.1, 0.85)

def main():
    print("🔍 Loading data...")
    signals, market = load_data()
    symbol = "POWERGRID"
    
    # Filter for target symbol
    signals = signals[signals['symbol'] == symbol].copy()
    market = market[market['symbol'] == symbol].copy()
    
    if len(signals) == 0:
        print(f"❌ No signals found for symbol: {symbol}")
        return
    
    print(f"✅ Found {len(signals)} signals for {symbol}")

    # Initialize environment (for obs generation & simulation)
    try:
        env = TradingEnv(signals, market, symbol=symbol)
    except Exception as e:
        print(f"❌ Env error: {e}")
        return

    # Load trained PPO agent
    model_path = f"models/ppo_{symbol}_best.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("👉 Run `train_ppo.py` first.")
        return

    print(f"📥 Loading PPO model: {model_path}")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, action_dim, device="cpu")
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor.eval()

    # Prepare enhanced rows
    enhanced_rows = []

    print("🤖 Generating PPO-enhanced signals...")
    obs, _ = env.reset()
    step = 0

    while step < len(env.data):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # shape: [1, obs_dim]
            mean, _ = agent.actor(obs_tensor)
            action = mean.squeeze(0).cpu().numpy()  # shape: [4]

        # Unpack action
        pos_ratio, sl_mult, tp_mult, close_ratio = action
        pos_ratio = np.clip(pos_ratio, 0.0, 2.0)
        sl_mult = np.clip(sl_mult, 0.5, 3.0)
        tp_mult = np.clip(tp_mult, 1.0, 4.0)

        row = env.data.iloc[step]
        
        # Compute final SL/TP using ATR (as in env)
        atr = row['atr'] if 'atr' in row and row['atr'] > 0 else 0.01 * row['buy']
        final_SL = row['buy'] - sl_mult * atr
        final_TP = row['buy'] + tp_mult * atr
        
        # Ensure SL < buy < TP
        final_SL = min(final_SL, row['buy'] * 0.99)
        final_TP = max(final_TP, row['buy'] * 1.01)
        
        final_diff = row['buy'] - final_SL
        final_RRR = (final_TP - row['buy']) / final_diff if final_diff > 1e-5 else 0.0

        # Position size: respect risk (1% of capital)
        risk_per_share = final_diff
        max_risk_bdt = env.initial_capital * env.risk_per_trade  # e.g., 5000 BDT
        max_shares_by_risk = int(max_risk_bdt / risk_per_share) if risk_per_share > 0 else 0
        base_shares = int(row['position_size']) if 'position_size' in row else 100
        final_shares = int(base_shares * pos_ratio)
        final_shares = min(final_shares, max_shares_by_risk, int(env.initial_capital / row['buy']))

        # Exposure & risk
        exposure = final_shares * row['buy']
        actual_risk = final_shares * final_diff

        # Simulate next OHLC to estimate if SL/TP likely to hit (for win%)
        next_idx = step + 1
        hit_sl, hit_tp = False, False
        if next_idx < len(env.data):
            next_row = env.data.iloc[next_idx]
            hit_sl = next_row['low'] <= final_SL
            hit_tp = next_row['high'] >= final_TP

        # Build output row
        out_row = {
            'No': step + 1,
            'symbol': row['symbol'],
            'date': row['date'],
            'buy': row['buy'],
            'SL': row['SL'],
            'final_SL': final_SL,
            'tp': row.get('tp', np.nan),
            'final_tp': final_TP,
            'position_size': row.get('position_size', 0),
            'final_position_size': final_shares,
            'exposure_bdt': exposure,
            'actual_risk_bdt': actual_risk,
            'diff': row.get('diff', row['buy'] - row['SL']),
            'final_diff': final_diff,
            'RRR1': row.get('RRR1', np.nan),
            'final_RRR': final_RRR,
            'sl_mult': sl_mult,
            'tp_mult': tp_mult,
            'pos_ratio': pos_ratio,
            'hit_SL_next': hit_sl,
            'hit_TP_next': hit_tp,
        }

        # Add market features if available (for win% estimation)
        for feat in ['rsi', 'volume', 'atr']:
            if feat in row:
                out_row[feat] = row[feat]

        enhanced_rows.append(out_row)
        step += 1

        # Get next obs (for sequence) — but we don't execute trades here
        if step < len(env.data):
            # Simulate a "no-op" step to advance env.obs
            dummy_action = np.array([0.0, 1.0, 2.0, 0.0])  # hold
            obs, _, _, _, _ = env.step(dummy_action)

    # Create DataFrame
    df = pd.DataFrame(enhanced_rows)
    
    # Add win% column — using your preferred logic
    df['win%'] = df.apply(estimate_win_probability, axis=1) * 100  # convert to %
    df['win%'] = df['win%'].round(1)

    # Sort by diff (ascending) — as per your preference
    df = df.sort_values('final_diff', ascending=True).reset_index(drop=True)
    df['No'] = range(1, len(df) + 1)

    # Select & order columns (retain your naming: 'buy', 'SL', etc.)
    output_cols = [
        'No', 'symbol', 'date', 'buy', 'SL', 'final_SL',
        'tp', 'final_tp', 'position_size', 'final_position_size',
        'exposure_bdt', 'actual_risk_bdt',
        'diff', 'final_diff',
        'RRR1', 'final_RRR',
        'win%', 'sl_mult', 'tp_mult', 'pos_ratio'
    ]
    output_cols = [c for c in output_cols if c in df.columns]

    # Save to CSV (OVERWRITE)
    output_path = "./csv/enhanced_signals.csv"
    df[output_cols].to_csv(output_path, index=False)
    
    print("\n✅ PPO Evaluation Complete!")
    print(f"   ➤ Output saved to: {output_path}")
    print(f"   ➤ Total signals: {len(df)}")
    print(f"\n📊 Top 5 signals (sorted by buy - SL diff ↑):")
    print(df[['symbol', 'buy', 'final_SL', 'final_diff', 'final_RRR', 'win%', 'final_position_size']].head().to_string(index=False))

    # Optional: Show summary stats
    print("\n📈 Summary:")
    print(f"   Avg win%     : {df['win%'].mean():.1f}%")
    print(f"   Avg RRR      : {df['final_RRR'].mean():.2f}")
    print(f"   Avg diff (BDT): {df['final_diff'].mean():.2f}")
    print(f"   Max position : {df['final_position_size'].max()} shares")

if __name__ == "__main__":
    main()