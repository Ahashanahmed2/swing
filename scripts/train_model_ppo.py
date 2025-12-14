# train_ppo.py
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import os
from envs.trading_env import TradingEnv
from ppo_agent import PPOAgent
from torch.utils.tensorboard import SummaryWriter

def load_data():
    signals = pd.read_csv("./csv/trade_stock.csv")
    market = pd.read_csv("./csv/mongodb.csv")
    signals['date'] = pd.to_datetime(signals['date'])
    market['date'] = pd.to_datetime(market['date'])
    return signals, market

def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    returns = np.array(advantages) + values
    return np.array(advantages), returns

def main():
    signals, market = load_data()
    symbol = "POWERGRID"  # or loop over symbols
    
    env = TradingEnv(signals, market, symbol=symbol)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(obs_dim, action_dim, device="cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=f"./runs/ppo_{symbol}")
    
    total_timesteps = 0
    best_reward = -np.inf
    
    # Training loop
    for episode in range(1000):  # adjust as needed
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        batch = {
            'obs': [], 'actions': [], 'log_probs': [], 
            'rewards': [], 'values': [], 'dones': []
        }
        
        while not done:
            action, log_prob, value = agent.get_action(obs)
            
            next_obs, reward, done, _, info = env.step(action)
            
            # Store
            batch['obs'].append(obs)
            batch['actions'].append(action)
            batch['log_probs'].append(log_prob)
            batch['rewards'].append(reward)
            batch['values'].append(value)
            batch['dones'].append(done)
            
            obs = next_obs
            episode_reward += reward
            total_timesteps += 1
        
        # Add final value
        _, _, final_value = agent.get_action(obs)
        batch['values'].append(final_value)
        
        # Compute GAE
        advantages, returns = compute_gae(
            batch['rewards'],
            batch['values'][:-1],
            batch['values'][1:],
            batch['dones'],
            gamma=0.99, lam=0.95
        )
        
        # Update agent
        batch_data = (
            np.array(batch['obs']),
            np.array(batch['actions']),
            np.array(batch['log_probs']),
            returns,
            advantages
        )
        actor_loss, critic_loss = agent.update(batch_data)
        
        # Logging
        writer.add_scalar("Episode/Reward", episode_reward, episode)
        writer.add_scalar("Episode/Trades", info['trades'], episode)
        writer.add_scalar("Episode/Balance", info['balance'], episode)
        writer.add_scalar("Loss/Actor", actor_loss, episode)
        writer.add_scalar("Loss/Critic", critic_loss, episode)
        
        if episode % 50 == 0:
            print(f"Episode {episode} | Reward: {episode_reward:.2f} | Trades: {info['trades']} | Balance: {info['balance']:.0f}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            os.makedirs("models", exist_ok=True)
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'episode': episode,
                'reward': episode_reward
            }, f"models/ppo_{symbol}_best.pth")
    
    writer.close()
    print("✅ PPO Training completed.")

if __name__ == "__main__":
    main()