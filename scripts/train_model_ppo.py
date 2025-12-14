# train_ppo.py
import numpy as np
import pandas as pd
import torch
import os
import warnings
warnings.filterwarnings("ignore")

# Import your modules
from envs.trading_env import TradingEnv
from ppo_agent import PPOAgent

def load_data():
    signals = pd.read_csv("./csv/trade_stock.csv")
    market = pd.read_csv("./csv/mongodb.csv")
    signals['date'] = pd.to_datetime(signals['date'])
    market['date'] = pd.to_datetime(market['date'])
    return signals, market

def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    if len(rewards) == 0:
        return np.array([]), np.array([])
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    advantages = np.array(advantages)
    returns = advantages + values[:-1]  # values has one extra (final)
    return advantages, returns

def main():
    print("📦 Loading data...")
    signals, market = load_data()
    symbol = "POWERGRID"  # Change or loop as needed
    
    try:
        env = TradingEnv(signals, market, symbol=symbol)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"✅ Environment ready for {symbol}")
    print(f"   Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Force CPU (PyTorch 2.3+ compatible)
    agent = PPOAgent(obs_dim, action_dim, device="cpu")
    
    total_timesteps = 0
    best_reward = -np.inf
    no_improve = 0
    patience = 100  # stop if no improvement in 100 episodes

    print("\n🚀 Starting PPO Training (CPU-only)...")
    print("-" * 60)

    # Training loop
    for episode in range(1000):  # adjust as needed
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        batch = {
            'obs': [], 'actions': [], 'log_probs': [], 
            'rewards': [], 'values': [], 'dones': []
        }
        
        while not done:
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            
            # Store experience
            batch['obs'].append(obs.copy())
            batch['actions'].append(action.copy())
            batch['log_probs'].append(log_prob)
            batch['rewards'].append(reward)
            batch['values'].append(value)
            batch['dones'].append(done)
            
            obs = next_obs
            episode_reward += reward
            total_timesteps += 1

        # Add final state value
        _, _, final_value = agent.get_action(obs)
        batch['values'].append(final_value)

        # Compute GAE & returns
        advantages, returns = compute_gae(
            batch['rewards'],
            np.array(batch['values'][:-1]),
            np.array(batch['values'][1:]),
            np.array(batch['dones']),
            gamma=0.99,
            lam=0.95
        )

        # Skip update if no steps (edge case)
        if len(advantages) == 0:
            continue

        # Prepare batch
        batch_data = (
            np.array(batch['obs'], dtype=np.float32),
            np.array(batch['actions'], dtype=np.float32),
            np.array(batch['log_probs'], dtype=np.float32),
            returns.astype(np.float32),
            advantages.astype(np.float32)
        )

        # Update agent
        actor_loss, critic_loss = agent.update(batch_data)

        # Logging (every 50 episodes)
        if episode % 50 == 0 or episode == 0:
            print(f"Ep {episode:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Balance: {info['balance']:8.0f} | "
                  f"Trades: {info['trades']:3d} | "
                  f"A-Loss: {actor_loss:.4f} | "
                  f"C-Loss: {critic_loss:.4f}")

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            no_improve = 0
            os.makedirs("models", exist_ok=True)
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'episode': episode,
                'reward': episode_reward
            }, f"models/ppo_{symbol}_best.pth")
            print(f"   🎯 New best reward: {best_reward:.2f} → Model saved!")
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= patience:
            print(f"\n⏹️  Early stopping: no improvement in {patience} episodes.")
            break

    print("-" * 60)
    print(f"✅ Training completed!")
    print(f"   Total episodes: {episode + 1}")
    print(f"   Best reward: {best_reward:.2f}")
    print(f"   Model saved at: ./models/ppo_{symbol}_best.pth")

if __name__ == "__main__":
    main()