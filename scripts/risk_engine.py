# src/risk_engine.py
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_POSITION_SIZE, MAX_PORTFOLIO_RISK, STOP_LOSS_PCT

class RiskEngine:
    def __init__(self):
        self.max_position_size = MAX_POSITION_SIZE
        self.max_portfolio_risk = MAX_PORTFOLIO_RISK
        self.stop_loss_pct = STOP_LOSS_PCT
        self.max_drawdown_limit = 0.15
        self.positions = {}
        
    def calculate_position_size(self, account_balance, entry_price, stop_loss_price):
        """ক্যালকুলেট পজিশন সাইজ"""
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return 0
            
        max_risk_amount = account_balance * self.max_portfolio_risk
        position_size = max_risk_amount / risk_per_share
        
        position_value = position_size * entry_price
        max_position_value = account_balance * self.max_position_size
        
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
        
        return int(position_size)
    
    def validate_trade(self, trade_data, account_balance, portfolio_value):
        """ভ্যালিডেট ট্রেড"""
        symbol = trade_data['symbol']
        entry_price = trade_data['buy']
        stop_loss = trade_data.get('SL', entry_price * (1 - self.stop_loss_pct))
        take_profit = trade_data.get('tp', entry_price * (1 + 2 * self.stop_loss_pct))
        
        # Calculate risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return False, "Zero risk detected"
            
        risk_reward_ratio = reward / risk
        
        # Minimum risk-reward ratio
        if risk_reward_ratio < 1:
            return False, f"Risk-reward ratio too low: {risk_reward_ratio:.2f}"
        
        # Calculate position size
        position_size = self.calculate_position_size(account_balance, entry_price, stop_loss)
        
        if position_size == 0:
            return False, "Position size is zero"
            
        position_value = position_size * entry_price
        
        # Check position size limit
        if position_value > account_balance * self.max_position_size:
            return False, f"Exceeds maximum position size: {position_value:.2f} > {account_balance * self.max_position_size:.2f}"
        
        # Check portfolio concentration
        current_exposure = sum(pos['value'] for pos in self.positions.values())
        max_exposure = portfolio_value * 0.3
        
        if current_exposure + position_value > max_exposure:
            return False, f"Exceeds maximum portfolio exposure: {current_exposure + position_value:.2f} > {max_exposure:.2f}"
        
        return True, {
            'position_size': position_size,
            'position_value': position_value,
            'risk_amount': position_size * risk,
            'risk_reward_ratio': risk_reward_ratio,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def calculate_portfolio_risk_metrics(self, portfolio, market_data):
        """ক্যালকুলেট পোর্টফোলিও রিস্ক মেট্রিক্স"""
        if not portfolio:
            return {}
        
        # Calculate portfolio returns
        portfolio_values = []
        dates = sorted(market_data['date'].unique())[-30:]  # Last 30 days
        
        for date in dates:
            daily_value = 0
            daily_data = market_data[market_data['date'] == date]
            
            for symbol, position in portfolio.items():
                symbol_data = daily_data[daily_data['symbol'] == symbol]
                if not symbol_data.empty:
                    current_price = symbol_data['close'].iloc[0]
                    daily_value += position['shares'] * current_price
            
            portfolio_values.append(daily_value)
        
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            metrics = {
                'volatility': np.std(returns) * np.sqrt(252),
                'sharpe_ratio': (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0,
                'max_drawdown': self._calculate_drawdown(portfolio_values),
                'var_95': np.percentile(returns, 5) * portfolio_values[-1]
            }
        else:
            metrics = {
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'var_95': 0
            }
        
        return metrics
    
    def _calculate_drawdown(self, values):
        """ক্যালকুলেট ড্রঅডাউন"""
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return np.min(drawdown)
    
    def update_position(self, symbol, position_info):
        """আপডেট পজিশন"""
        self.positions[symbol] = position_info

if __name__ == "__main__":
    # Standalone test
    risk_engine = RiskEngine()
    
    sample_trade = {
        'symbol': 'GSPFINANCE',
        'buy': 1.8,
        'SL': 1.4,
        'tp': 2.3
    }
    
    is_valid, result = risk_engine.validate_trade(
        sample_trade,
        account_balance=100000,
        portfolio_value=150000
    )
    
    print(f"Trade valid: {is_valid}")
    if is_valid:
        print(f"Position size: {result['position_size']}")
        print(f"Risk amount: {result['risk_amount']:.2f}")
        print(f"Risk-reward ratio: {result['risk_reward_ratio']:.2f}")