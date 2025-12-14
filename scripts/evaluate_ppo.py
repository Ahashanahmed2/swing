# evaluate_ppo.py
import pandas as pd
import numpy as np
import torch
from envs.trading_env import TradingEnv
from ppo_agent import PPOAgent

def load_data():
    signals = pd.read_csv("./csv/trade_stock.csv")
    market = pd.read_csv("./csv/mongodb.csv")
    signals['date'] = pd.to_datetime(signals['date'])
    market['date'] = pd.to_datetime(market['date'])
    return signals, market

def main():
    signals, market = load_data()
    symbol = "POWERGRID"
    
    env = TradingEnv(signals, market, symbol=symbol)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(obs_dim, action_dim)
    checkpoint = torch.load(f"models/ppo_{symbol}_best.pth", map_location="cpu")
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor.eval()
    
    obs, _ = env.reset()
    done = False
    
    enhanced_rows = []
    
    while not done:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            mean, _ = agent.actor(obs_t)
            action = mean.squeeze(0).cpu().numpy()
        
        # Unpack
        pos_ratio, sl_mult, tp_mult, close_ratio = action
        row = env.data.iloc[env.current_step]
        
        # Compute final decisions
        risk_per_share = row['buy'] - (row['buy'] - sl_mult * row['atr'])
        max_shares = int((env.initial_capital * env.risk_per_trade) / (risk_per_share + 1e-5))
        final_shares = int(row['position_size'] * np.clip(pos_ratio, 0, 2))
        final_shares = min(final_shares, max_shares)
        
        final_SL = row['buy'] - sl_mult * row['atr']
        final_TP = row['buy'] + tp_mult * row['atr']
        final_diff = row['buy'] - final_SL
        final_RRR = (final_TP - row['buy']) / final_diff if final_diff > 0 else 0
        
        enhanced_rows.append({
            'No': len(enhanced_rows) + 1,
            'symbol': row['symbol'],
            'date': row['date'],
            'buy': row['buy'],
            'SL': row['SL'],
            'final_SL': final_SL,
            'tp': row['tp'],
            'final_tp': final_TP,
            'position_size': row['position_size'],
            'final_position_size': final_shares,
            'exposure_bdt': row['position_size'] * row['buy'],
            'final_exposure_bdt': final_shares * row['buy'],
            'actual_risk_bdt': final_shares * (row['buy'] - final_SL),
            'diff': row['diff'],
            'final_diff': final_diff,
            'RRR1': row['RRR1'],
            'final_RRR': final_RRR,
            'ppo_action': action.tolist(),
            'sl_mult': sl_mult,
            'tp_mult': tp_mult,
            'pos_ratio': pos_ratio
        })
        
        obs, _, done, _, _ = env.step(action)
    
    # Create DataFrame
    df = pd.DataFrame(enhanced_rows)
    
    # Add win% & metrics (simplified: use PPO's implied success)
    df['win_prob'] = np.clip(df['final_RRR'] * 0.3 + df['pos_ratio'] * 0.2, 0, 1)
    
    # Sort by diff (ascending) — your preference
    df = df.sort_values('final_diff', ascending=True).reset_index(drop=True)
    df['No'] = range(1, len(df)+1)
    
    # Save
    df.to_csv("./csv/enhanced_signals.csv", index=False)
    print(f"✅ PPO-enhanced signals saved for {symbol} → {len(df)} rows")
    print(df.head()[['symbol','buy','final_SL','final_diff','final_RRR','final_position_size']])

if __name__ == "__main__":
    main()