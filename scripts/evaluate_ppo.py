# evaluate_ppo.py
import pandas as pd
import numpy as np
import torch
import os
from env import TradeEnv
from ppo_agent import PPOAgent

def load_data():
    signals = pd.read_csv("./csv/trade_stock.csv")
    market = pd.read_csv("./csv/mongodb.csv")
    signals['date'] = pd.to_datetime(signals['date'])
    market['date'] = pd.to_datetime(market['date'])
    return signals, market

def estimate_win_probability(row):
    base = 0.5
    if row['final_RRR'] >= 2.0:
        base += 0.10
    elif row['final_RRR'] >= 1.5:
        base += 0.05
    else:
        base -= 0.05
    
    diff_pct = row['final_diff'] / row['buy'] if row['buy'] > 0 else 0.01
    if diff_pct <= 0.02:
        base += 0.10
    elif diff_pct <= 0.03:
        base += 0.05
    
    if 'rsi' in row:
        rsi = row['rsi']
        if 30 <= rsi <= 60:
            base += 0.05
        elif rsi < 25 or rsi > 75:
            base -= 0.05
    return np.clip(base, 0.1, 0.85)

def process_symbol(signals, market, symbol):
    try:
        env = TradeEnv(signals, market, symbol=symbol, initial_capital=500_000)
    except Exception as e:
        print(f"   ⚠️ Skip {symbol}: {str(e)[:50]}")
        return None

    model_path = f"models/ppo_{symbol}_best.pth"
    if not os.path.exists(model_path):
        print(f"   ⚠️ Model missing for {symbol} → skipping")
        return None

    # Load agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, action_dim, device="cpu")
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        agent.actor.load_state_dict(ckpt['actor_state_dict'])
        agent.actor.eval()
    except Exception as e:
        print(f"   ⚠️ Load error {symbol}: {str(e)[:50]}")
        return None

    enhanced_rows = []
    obs, _ = env.reset()
    step = 0

    while step < len(env.data):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            mean, _ = agent.actor(obs_tensor)
            action = mean.squeeze(0).cpu().numpy()

        pos_ratio, sl_mult, tp_mult, _ = action
        pos_ratio = np.clip(pos_ratio, 0.0, 2.0)
        sl_mult = np.clip(sl_mult, 0.5, 3.0)
        tp_mult = np.clip(tp_mult, 1.0, 4.0)

        row = env.data.iloc[step]
        atr = row.get('atr', 0.01 * row['buy'])
        final_SL = row['buy'] - sl_mult * atr
        final_TP = row['buy'] + tp_mult * atr
        final_SL = min(final_SL, row['buy'] * 0.99)
        final_TP = max(final_TP, row['buy'] * 1.01)
        final_diff = row['buy'] - final_SL
        final_RRR = (final_TP - row['buy']) / final_diff if final_diff > 1e-5 else 0.0

        risk_per_share = final_diff
        max_risk_bdt = env.initial_capital * env.risk_per_trade
        max_shares = int(max_risk_bdt / risk_per_share) if risk_per_share > 0 else 0
        base_shares = int(row.get('position_size', 100))
        final_shares = int(base_shares * pos_ratio)
        final_shares = min(final_shares, max_shares, int(env.initial_capital / row['buy']))

        out_row = {
            'symbol': symbol,
            'date': row['date'],
            'buy': row['buy'],
            'SL': row.get('SL', np.nan),
            'final_SL': final_SL,
            'tp': row.get('tp', np.nan),
            'final_tp': final_TP,
            'position_size': base_shares,
            'final_position_size': final_shares,
            'exposure_bdt': final_shares * row['buy'],
            'actual_risk_bdt': final_shares * final_diff,
            'diff': row.get('diff', row['buy'] - row.get('SL', row['buy']*0.95)),
            'final_diff': final_diff,
            'RRR1': row.get('RRR1', np.nan),
            'final_RRR': final_RRR,
            'sl_mult': sl_mult,
            'tp_mult': tp_mult,
            'pos_ratio': pos_ratio,
        }
        
        for feat in ['rsi', 'volume', 'atr']:
            if feat in row:
                out_row[feat] = row[feat]

        enhanced_rows.append(out_row)
        step += 1

        if step < len(env.data):
            obs, _, _, _, _ = env.step(np.array([0.0, 1.0, 2.0, 0.0]))

    if not enhanced_rows:
        return None

    df = pd.DataFrame(enhanced_rows)
    df['win%'] = df.apply(estimate_win_probability, axis=1) * 100
    df['win%'] = df['win%'].round(1)

    # ✅ **প্রতি সিম্বলের শীর্ষ 1টি সিগন্যাল বাছাই**
    # সর্টিং ক্রাইটেরিয়া: 1. win% (desc), 2. final_RRR (desc), 3. final_diff (asc)
    df = df.sort_values(
        by=['win%', 'final_RRR', 'final_diff'],
        ascending=[False, False, True]
    ).head(1)

    return df.iloc[0].to_dict()

def main():
    print("🔍 Loading multi-symbol data...")
    signals, market = load_data()
    
    all_symbols = sorted(set(signals['symbol']) & set(market['symbol']))
    print(f"✅ Found {len(all_symbols)} common symbols: {', '.join(all_symbols[:5])}{'...' if len(all_symbols)>5 else ''}")

    best_signals = []
    print("\n🧠 Evaluating PPO models per symbol...")
    
    for i, symbol in enumerate(all_symbols, 1):
        print(f"  [{i}/{len(all_symbols)}] Processing {symbol}...")
        result = process_symbol(signals, market, symbol)
        if result:
            best_signals.append(result)

    if not best_signals:
        print("❌ No valid signals generated.")
        return

    # Create final DataFrame
    df_final = pd.DataFrame(best_signals)
    
    # Reorder & clean columns
    cols = [
        'symbol', 'date', 'buy', 'SL', 'final_SL', 
        'tp', 'final_tp', 'final_position_size',
        'final_diff', 'final_RRR', 'win%',
        'exposure_bdt', 'actual_risk_bdt',
        'sl_mult', 'tp_mult', 'pos_ratio'
    ]
    cols = [c for c in cols if c in df_final.columns]
    df_final = df_final[cols].copy()
    
    # Sort final output by final_diff (ascending) — as per your preference
    df_final = df_final.sort_values('final_diff', ascending=True).reset_index(drop=True)
    df_final.index = df_final.index + 1  # 1-based index
    df_final.insert(0, 'No', range(1, len(df_final)+1))

    # Save (OVERWRITE)
    output_path = "./csv/enhanced_signals.csv"
    df_final.to_csv(output_path, index=False)
    
    print("\n✅ Done! Final signal summary:")
    print(f"   ➤ Total symbols processed: {len(all_symbols)}")
    print(f"   ➤ Valid signals generated: {len(df_final)}")
    print(f"   ➤ Output saved to: {output_path}\n")

    print(df_final[['No','symbol','buy','final_SL','final_diff','final_RRR','win%','final_position_size']].to_string(index=False))
    
    # Summary stats
    print(f"\n📈 Overall:")
    print(f"   Avg win%     : {df_final['win%'].mean():.1f}%")
    print(f"   Avg RRR      : {df_final['final_RRR'].mean():.2f}")
    print(f"   Avg diff (BDT): {df_final['final_diff'].mean():.2f}")

if __name__ == "__main__":
    main()