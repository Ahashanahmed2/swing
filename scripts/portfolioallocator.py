import pandas as pd
import numpy as np
from scipy.optimize import minimize

class PortfolioAllocator:
    def __init__(self):
        self.portfolio = {}
        
    def markowitz_optimization(self, returns_data, target_return=None, risk_free_rate=0.05):
        """Modern Portfolio Theory optimization"""
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns_data.mean()
        cov_matrix = returns_data.cov()
        
        n_assets = len(expected_returns)
        
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights
        
        def portfolio_return(weights):
            return weights.T @ expected_returns
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # weights sum to 1
        ]
        
        if target_return is not None:
            constraints.append(
                {'type': 'eq', 'fun': lambda w: portfolio_return(w) - target_return}
            )
        
        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        init_guess = np.array([1/n_assets] * n_assets)
        
        # Optimize for minimum variance
        result = minimize(
            portfolio_variance,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def risk_parity_allocation(self, cov_matrix):
        """Risk Parity allocation"""
        
        n_assets = cov_matrix.shape[0]
        
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_risk = cov_matrix @ weights / portfolio_vol
            risk_contributions = weights * marginal_risk
            
            # Target equal risk contribution
            target_risk = portfolio_vol / n_assets
            
            # Calculate deviation from equal risk contribution
            deviation = np.sum((risk_contributions - target_risk) ** 2)
            return deviation
        
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
            risk_parity_objective,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def calculate_allocation(self, market_data, trade_signals, account_balance):
        """Calculate optimal portfolio allocation based on signals"""
        
        # Get unique symbols
        symbols = market_data['symbol'].unique()
        
        # Prepare returns data
        returns_data = pd.DataFrame()
        
        for symbol in symbols:
            symbol_data = market_data[market_data['symbol'] == symbol]
            returns_data[symbol] = symbol_data['close'].pct_change()
        
        returns_data = returns_data.dropna()
        
        # Calculate weights using Markowitz
        weights = self.markowitz_optimization(returns_data)
        
        # Apply trade signals
        final_weights = {}
        total_signal_strength = 0
        
        for idx, symbol in enumerate(symbols):
            # Get latest signal for this symbol
            symbol_signals = trade_signals.get(symbol, [])
            if symbol_signals:
                # Use average of recent signals
                signal_strength = np.mean(symbol_signals[-5:]) if len(symbol_signals) >= 5 else symbol_signals[-1]
            else:
                signal_strength = 0.5  # Neutral signal
            
            # Adjust weight based on signal
            adjusted_weight = weights[idx] * signal_strength
            final_weights[symbol] = adjusted_weight
            total_signal_strength += adjusted_weight
        
        # Normalize weights
        if total_signal_strength > 0:
            for symbol in final_weights:
                final_weights[symbol] /= total_signal_strength
        
        # Calculate position sizes
        allocations = {}
        for symbol, weight in final_weights.items():
            if weight > 0.01:  # Only allocate if weight > 1%
                symbol_data = market_data[market_data['symbol'] == symbol]
                if not symbol_data.empty:
                    current_price = symbol_data['close'].iloc[-1]
                    position_value = account_balance * weight
                    position_size = position_value / current_price
                    
                    allocations[symbol] = {
                        'weight': weight,
                        'position_size': int(position_size),
                        'position_value': position_value,
                        'current_price': current_price
                    }
        
        return allocations
    
    def rebalance_portfolio(self, current_portfolio, target_allocations, transaction_cost=0.001):
        """Calculate rebalancing trades"""
        
        rebalance_orders = []
        
        total_value = sum(pos['value'] for pos in current_portfolio.values())
        
        for symbol, target in target_allocations.items():
            target_value = total_value * target['weight']
            
            if symbol in current_portfolio:
                current_value = current_portfolio[symbol]['value']
                difference = target_value - current_value
                
                if abs(difference) > total_value * 0.01:  # Only rebalance if difference > 1%
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
                rebalance_orders.append({
                    'symbol': symbol,
                    'type': 'BUY',
                    'value': target_value,
                    'shares': int(target_value / target['current_price'])
                })
        
        return rebalance_orders

# Usage
if __name__ == "__main__":
    allocator = PortfolioAllocator()
    
    # Load data
    market_data = pd.read_csv("./csv/mongodb.csv")
    trade_data = pd.read_csv("./csv/trade_stock.csv")
    
    # Prepare trade signals (example)
    trade_signals = {}
    for symbol in market_data['symbol'].unique():
        symbol_trades = trade_data[trade_data['symbol'] == symbol]
        if not symbol_trades.empty:
            # Normalize buy signals to 0-1 range
            signals = symbol_trades['buy'].values / symbol_trades['buy'].max()
            trade_signals[symbol] = signals.tolist()
    
    # Calculate allocation
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