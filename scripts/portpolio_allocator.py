# src/portfolio_allocator.py
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PortfolioAllocator:
    def __init__(self, risk_aversion=1.0):
        self.risk_aversion = risk_aversion
        self.portfolio = {}
        
    def calculate_allocation(self, market_data, trade_signals, account_balance):
        """ক্যালকুলেট অ্যালোকেশন"""
        symbols = list(trade_signals.keys())
        
        if not symbols:
            return {}
        
        # Prepare returns data
        returns_data = pd.DataFrame()
        
        for symbol in symbols:
            symbol_data = market_data[market_data['symbol'] == symbol]
            if len(symbol_data) > 1:
                returns_data[symbol] = symbol_data['close'].pct_change().dropna()
        
        if returns_data.empty:
            return {}
        
        # Calculate optimal weights
        if len(symbols) > 1:
            try:
                weights = self._markowitz_optimization(returns_data)
            except:
                # Equal weighting if optimization fails
                weights = np.array([1/len(symbols)] * len(symbols))
        else:
            weights = np.array([1.0])
        
        # Apply trade signal weights
        final_weights = {}
        total_signal_weight = 0
        
        for idx, symbol in enumerate(symbols):
            if idx < len(weights):
                signal_strength = trade_signals[symbol]
                adjusted_weight = weights[idx] * signal_strength
                final_weights[symbol] = adjusted_weight
                total_signal_weight += adjusted_weight
        
        # Normalize weights
        if total_signal_weight > 0:
            for symbol in final_weights:
                final_weights[symbol] /= total_signal_weight
        
        # Calculate position sizes
        allocations = {}
        total_allocated = 0
        
        for symbol, weight in final_weights.items():
            if weight > 0.01:  # Only allocate if weight > 1%
                symbol_data = market_data[market_data['symbol'] == symbol]
                if not symbol_data.empty:
                    current_price = symbol_data['close'].iloc[-1]
                    position_value = account_balance * weight
                    
                    # Ensure we don't exceed account balance
                    if total_allocated + position_value > account_balance:
                        position_value = account_balance - total_allocated
                        if position_value <= 0:
                            break
                    
                    position_size = int(position_value / current_price)
                    
                    if position_size > 0:
                        allocations[symbol] = {
                            'weight': weight,
                            'position_size': position_size,
                            'position_value': position_value,
                            'current_price': current_price
                        }
                        total_allocated += position_value
        
        return allocations
    
    def _markowitz_optimization(self, returns_data):
        """মার্কোভিটজ অপ্টিমাইজেশন"""
        expected_returns = returns_data.mean()
        cov_matrix = returns_data.cov()
        
        n_assets = len(expected_returns)
        
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights
        
        def portfolio_return(weights):
            return weights.T @ expected_returns
        
        # Objective function
        def objective(weights):
            port_return = portfolio_return(weights)
            port_variance = portfolio_variance(weights)
            return - (port_return - self.risk_aversion * port_variance)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        init_guess = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def rebalance_portfolio(self, current_portfolio, target_allocations, transaction_cost=0.001):
        """রিব্যালেন্স পোর্টফোলিও"""
        rebalance_orders = []
        
        # Calculate total portfolio value
        total_value = 0
        for symbol, position in current_portfolio.items():
            if 'current_value' in position:
                total_value += position['current_value']
            else:
                total_value += position['value']
        
        for symbol, target in target_allocations.items():
            target_value = total_value * target['weight']
            
            if symbol in current_portfolio:
                # Get current value
                if 'current_value' in current_portfolio[symbol]:
                    current_value = current_portfolio[symbol]['current_value']
                else:
                    current_value = current_portfolio[symbol]['value']
                
                difference = target_value - current_value
                
                # Only rebalance if difference > 1%
                if abs(difference) > total_value * 0.01:
                    order_type = 'BUY' if difference > 0 else 'SELL'
                    order_value = abs(difference) * (1 - transaction_cost)
                    
                    rebalance_orders.append({
                        'symbol': symbol,
                        'type': order_type,
                        'value': order_value,
                        'shares': int(order_value / target['current_price'])
                    })
            else:
                # New position
                if target_value > 0:
                    rebalance_orders.append({
                        'symbol': symbol,
                        'type': 'BUY',
                        'value': target_value,
                        'shares': int(target_value / target['current_price'])
                    })
        
        return rebalance_orders

if __name__ == "__main__":
    # Standalone test
    allocator = PortfolioAllocator()
    
    # Load sample data
    market_data = pd.read_csv("../csv/mongodb.csv")
    
    # Create sample signals
    trade_signals = {
        'GSPFINANCE': 0.8,
        '其他symbol': 0.6
    }
    
    allocations = allocator.calculate_allocation(
        market_data,
        trade_signals,
        account_balance=100000
    )
    
    print("Portfolio Allocation:")
    for symbol, alloc in allocations.items():
        print(f"{symbol}:")
        print(f"  Weight: {alloc['weight']:.2%}")
        print(f"  Position Size: {alloc['position_size']}")
        print(f"  Position Value: {alloc['position_value']:.2f}")