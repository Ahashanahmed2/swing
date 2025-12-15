# src/ppo_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MONGODB_PATH, PPO_MODEL_PATH

class PPOTradingAgent:
    def __init__(self, state_dim=110, action_dim=3, learning_rate=0.0003, 
                 gamma=0.99, epsilon=0.2, epochs=10, batch_size=64):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=learning_rate
        )
        
        self.memory = []
        
    def get_state(self, market_data, idx):
        """কনভার্ট মার্কেট ডেটা টু স্টেট ভেক্টর"""
        if idx >= len(market_data):
            return None
        
        state = []
        window_size = 10
        
        # Price features
        for i in range(max(0, idx-window_size+1), idx+1):
            if i < len(market_data):
                row = market_data.iloc[i]
                state.extend([
                    row['open'], row['close'], row['high'], row['low'],
                    row['volume'], row['rsi'], row['macd'], row['macd_signal'],
                    row['bb_upper'], row['bb_middle'], row['bb_lower']
                ])
        
        # Pad if needed
        while len(state) < self.state_dim:
            state.append(0)
        
        return np.array(state[:self.state_dim])
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), probs.detach().numpy()[0]
    
    def train_episode(self, market_data, initial_balance=100000):
        """ট্রেইন ওয়ান এপিসোড"""
        balance = initial_balance
        position = 0
        entry_price = 0
        
        for idx in range(len(market_data)):
            state = self.get_state(market_data, idx)
            if state is None:
                continue
            
            # Select action (0: hold, 1: buy, 2: sell)
            action, log_prob, _ = self.select_action(state)
            
            # Execute action
            current_price = market_data.iloc[idx]['close']
            reward = 0
            
            if action == 1 and position == 0:  # Buy
                position = balance / current_price
                entry_price = current_price
                balance = 0
                
            elif action == 2 and position > 0:  # Sell
                balance = position * current_price
                profit = balance - (position * entry_price)
                position = 0
                reward = profit / entry_price
            
            # Store experience
            next_state = self.get_state(market_data, idx + 1)
            
            self.memory.append((state, action, log_prob.item(), reward, 
                               next_state if next_state is not None else state, 
                               1 if idx == len(market_data)-1 else 0))
            
            # Update if batch is complete
            if len(self.memory) >= self.batch_size:
                self._update_model()
        
        return balance + (position * market_data.iloc[-1]['close'] if position > 0 else 0)
    
    def _update_model(self):
        """ইন্টারনাল আপডেট মডেল"""
        states, actions, log_probs, rewards, next_states, dones = zip(*self.memory)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs)
        rewards = torch.FloatTensor(rewards)
        
        # Compute returns
        returns = self._compute_returns(rewards.numpy())
        
        # Compute advantages
        values = self.value_net(states).squeeze()
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.epochs):
            probs = self.policy_net(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns)
            
            loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.memory = []
    
    def _compute_returns(self, rewards):
        """কম্পিউট রিটার্নস"""
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        return torch.FloatTensor(returns)
    
    def save_model(self):
        """সেভ মডেল"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PPO_MODEL_PATH)
        print(f"PPO model saved to {PPO_MODEL_PATH}")
    
    def load_model(self):
        """লোড মডেল"""
        checkpoint = torch.load(PPO_MODEL_PATH)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"PPO model loaded from {PPO_MODEL_PATH}")

if __name__ == "__main__":
    # Standalone execution
    market_data = pd.read_csv(MONGODB_PATH)
    
    agent = PPOTradingAgent(state_dim=110, action_dim=3)
    
    print("Training PPO agent...")
    for episode in range(10):  # Reduced for testing
        final_balance = agent.train_episode(market_data)
        print(f"Episode {episode+1}: Final Balance: {final_balance:.2f}")
    
    agent.save_model()