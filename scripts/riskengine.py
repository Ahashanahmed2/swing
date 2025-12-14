import pandas as pd
import numpy as np
from scipy import stats

class RiskEngine:
    def __init__(self, max_position_size=0.1, max_portfolio_risk=0.02, 
                 stop_loss_pct=0.05, max_drawdown_limit=0.15):
        self.max_position_size = max_position_size  # Max 10% per position
        self.max_portfolio_risk = max_portfolio_risk  # Max 2% portfolio risk
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.positions = {}
        
    def calculate_position_size(self, account_balance, entry_price, stop_loss_price):
        """Calculate position size using Kelly Criterion or fixed fraction"""
        
        # Calculate risk per share
        risk_per_share = entry_price - stop_loss_price
        
        # Maximum risk per trade
        max_risk_amount = account_balance * self.max_portfolio_risk
        
        # Calculate position size
        position_size = max_risk_amount / abs(risk_per_share)
        
        # Calculate position value
        position_value = position_size * entry_price
        
        # Apply max position size constraint
        max_position_value = account_balance * self.max_position_size
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
        
        return int(position_size)
    
    def calculate_var(self, returns, confidence_level=0.95):
        """Calculate Value at Risk"""
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var
    
    def calculate_expected_shortfall(self, returns, confidence_level=0.95):
        """Calculate Expected Shortfall (CVaR)"""
        var = self.calculate_var(returns, confidence_level)
        es = returns[returns <= var].mean()
        return es
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.05):
        """Calculate Sharpe Ratio"""
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        sharpe = np.sqrt(252) * excess_returns.mean() / (excess_returns.std() + 1e-8)
        return sharpe
    
    def calculate_max_drawdown(self, equity_curve):
        """Calculate Maximum Drawdown"""
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        return max_drawdown
    
    def check_market_conditions(self, market_data):
        """Assess current market conditions for risk adjustment"""
        
        recent_data = market_data.tail(20)
        
        # Calculate volatility
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Check for extreme conditions
        extreme_volatility = volatility > 0.03  # 3% daily volatility
        
        # Check volume spikes
        avg_volume = recent_data['volume'].mean()
        current_volume = recent_data['volume'].iloc[-1]
        volume_spike = current_volume > 2 * avg_volume
        
        # Check RSI extremes
        current_rsi = recent_data['rsi'].iloc[-1]
        rsi_extreme = current_rsi > 80 or current_rsi < 20
        
        risk_multiplier = 1.0
        
        if extreme_volatility or volume_spike or rsi_extreme:
            risk_multiplier = 0.5  # Reduce risk by 50%
        
        return {
            'volatility': volatility,
            'extreme_volatility': extreme_volatility,
            'volume_spike': volume_spike,
            'rsi_extreme': rsi_extreme,
            'risk_multiplier': risk_multiplier
        }
    
    def validate_trade(self, trade_data, account_balance, portfolio_value):
        """Validate if a trade meets risk criteria"""
        
        symbol = trade_data['symbol']
        entry_price = trade_data['buy']
        stop_loss = trade_data['SL']
        take_profit = trade_data['tp']
        
        # Calculate risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Minimum risk-reward ratio
        if risk_reward_ratio < 1:
            return False, "Risk-reward ratio too low"
        
        # Calculate position size
        position_size = self.calculate_position_size(account_balance, entry_price, stop_loss)
        position_value = position_size * entry_price
        
        # Check if exceeds max position size
        if position_value > account_balance * self.max_position_size:
            return False, "Exceeds maximum position size"
        
        # Check portfolio concentration
        current_exposure = sum(pos['value'] for pos in self.positions.values())
        if current_exposure + position_value > portfolio_value * 0.3:  # Max 30% exposure
            return False, "Exceeds maximum portfolio exposure"
        
        return True, {
            'position_size': position_size,
            'position_value': position_value,
            'risk_amount': position_size * risk,
            'risk_reward_ratio': risk_reward_ratio
        }

# Usage
if __name__ == "__main__":
    risk_engine = RiskEngine()
    
    # Load trade data
    trade_data = pd.read_csv("./csv/trade_stock.csv")
    
    # Example validation
    sample_trade = trade_data.iloc[0]
    is_valid, result = risk_engine.validate_trade(
        sample_trade, 
        account_balance=100000,
        portfolio_value=150000
    )
    
    print(f"Trade valid: {is_valid}")
    if is_valid:
        print(f"Recommended position size: {result['position_size']}")
        print(f"Risk amount: {result['risk_amount']:.2f}")