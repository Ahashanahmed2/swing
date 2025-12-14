import pandas as pd
import numpy as np
from datetime import datetime

class TradingSystem:
    def __init__(self):
        self.xgb_model = XGBoostTradingModel()
        self.ppo_agent = None
        self.risk_engine = RiskEngine()
        self.portfolio_allocator = PortfolioAllocator()
        
        self.portfolio = {}
        self.account_balance = 100000
        self.portfolio_value = self.account_balance
        self.trade_history = []
        
    def initialize_system(self):
        """Initialize all components"""
        print("Initializing Trading System...")
        
        # Load and train XGBoost model
        print("Training XGBoost model...")
        X, y = self.xgb_model.load_and_prepare_data(
            "./csv/mongodb.csv", 
            "./csv/trade_stock.csv"
        )
        self.xgb_model.train(X, y)
        
        # Initialize PPO agent
        print("Initializing PPO agent...")
        state_dim = 110
        action_dim = 3
        self.ppo_agent = PPOTradingAgent(state_dim, action_dim)
        
        print("Trading System initialized successfully!")
    
    def run_daily_analysis(self, date):
        """Run daily trading analysis"""
        
        # Load market data for the date
        market_data = pd.read_csv("./csv/mongodb.csv")
        market_data['date'] = pd.to_datetime(market_data['date'])
        daily_data = market_data[market_data['date'] == date]
        
        if daily_data.empty:
            print(f"No data available for {date}")
            return
        
        # Get XGBoost predictions
        predictions, probabilities = self.xgb_model.predict_signals(daily_data)
        
        # Get PPO agent action
        # (In production, this would use the current state)
        
        # Prepare trade signals
        trade_signals = {}
        for idx, row in daily_data.iterrows():
            symbol = row['symbol']
            if predictions[idx] == 1 and probabilities[idx] > 0.7:  # Strong buy signal
                trade_signals[symbol] = probabilities[idx]
        
        # Calculate portfolio allocation
        target_allocations = self.portfolio_allocator.calculate_allocation(
            market_data,
            trade_signals,
            self.account_balance
        )
        
        # Generate rebalancing orders
        rebalance_orders = self.portfolio_allocator.rebalance_portfolio(
            self.portfolio,
            target_allocations
        )
        
        # Execute orders with risk validation
        executed_orders = []
        for order in rebalance_orders:
            # Simulate trade execution
            symbol = order['symbol']
            order_type = order['type']
            shares = order['shares']
            
            # Get current price
            symbol_data = daily_data[daily_data['symbol'] == symbol]
            if not symbol_data.empty:
                current_price = symbol_data['close'].iloc[0]
                
                # Create trade data for risk validation
                trade_data = {
                    'symbol': symbol,
                    'buy': current_price,
                    'SL': current_price * 0.95,  # 5% stop loss
                    'tp': current_price * 1.1,   # 10% take profit
                    'date': date
                }
                
                # Validate trade
                is_valid, risk_info = self.risk_engine.validate_trade(
                    trade_data,
                    self.account_balance,
                    self.portfolio_value
                )
                
                if is_valid:
                    # Execute trade
                    if order_type == 'BUY':
                        cost = shares * current_price
                        if cost <= self.account_balance:
                            self.account_balance -= cost
                            if symbol in self.portfolio:
                                self.portfolio[symbol]['shares'] += shares
                                self.portfolio[symbol]['value'] += cost
                            else:
                                self.portfolio[symbol] = {
                                    'shares': shares,
                                    'value': cost,
                                    'entry_price': current_price
                                }
                    else:  # SELL
                        if symbol in self.portfolio:
                            proceeds = shares * current_price
                            self.account_balance += proceeds
                            self.portfolio[symbol]['shares'] -= shares
                            self.portfolio[symbol]['value'] -= shares * current_price
                            
                            if self.portfolio[symbol]['shares'] == 0:
                                del self.portfolio[symbol]
                    
                    executed_orders.append({
                        'date': date,
                        'symbol': symbol,
                        'type': order_type,
                        'shares': shares,
                        'price': current_price,
                        'risk_info': risk_info
                    })
        
        # Update portfolio value
        self.update_portfolio_value(daily_data)
        
        # Log daily results
        self.log_daily_results(date, executed_orders)
        
        return executed_orders
    
    def update_portfolio_value(self, market_data):
        """Update portfolio value based on current prices"""
        total_value = self.account_balance
        
        for symbol, position in self.portfolio.items():
            symbol_data = market_data[market_data['symbol'] == symbol]
            if not symbol_data.empty:
                current_price = symbol_data['close'].iloc[0]
                position['current_value'] = position['shares'] * current_price
                position['pnl'] = position['current_value'] - position['value']
                total_value += position['current_value']
        
        self.portfolio_value = total_value
        return total_value
    
    def log_daily_results(self, date, executed_orders):
        """Log daily trading results"""
        
        log_entry = {
            'date': date,
            'portfolio_value': self.portfolio_value,
            'account_balance': self.account_balance,
            'number_of_positions': len(self.portfolio),
            'executed_orders': len(executed_orders),
            'orders': executed_orders
        }
        
        self.trade_history.append(log_entry)
        
        print(f"\n=== Daily Results for {date} ===")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        print(f"Account Balance: ${self.account_balance:.2f}")
        print(f"Number of Positions: {len(self.portfolio)}")
        print(f"Orders Executed: {len(executed_orders)}")
        
        if executed_orders:
            print("\nExecuted Orders:")
            for order in executed_orders:
                print(f"  {order['type']} {order['shares']} shares of {order['symbol']} "
                      f"at ${order['price']:.2f}")
    
    def generate_performance_report(self):
        """Generate performance report"""
        
        if not self.trade_history:
            print("No trading history available")
            return
        
        # Create DataFrame from trade history
        history_df = pd.DataFrame(self.trade_history)
        
        # Calculate performance metrics
        initial_value = 100000
        final_value = self.portfolio_value
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        daily_values = [h['portfolio_value'] for h in self.trade_history]
        daily_returns = np.diff(daily_values) / daily_values[:-1]
        
        if len(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-8)
            max_drawdown = self.calculate_max_drawdown(daily_values)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        print("\n" + "="*50)
        print("PERFORMANCE REPORT")
        print("="*50)
        print(f"Initial Capital: ${initial_value:.2f}")
        print(f"Final Portfolio Value: ${final_value:.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Total Trades: {sum(h['executed_orders'] for h in self.trade_history)}")
        print(f"Trading Days: {len(self.trade_history)}")
        
        # Calculate win rate if we have trade details
        winning_trades = 0
        total_trades = 0
        
        for day in self.trade_history:
            for order in day['orders']:
                total_trades += 1
                if 'risk_info' in order:
                    if order['risk_info'].get('pnl', 0) > 0:
                        winning_trades += 1
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            print(f"Win Rate: {win_rate:.2%}")
        
        print("="*50)
    
    def calculate_max_drawdown(self, values):
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return np.min(drawdown)

# Main execution
if __name__ == "__main__":
    # Create trading system
    trading_system = TradingSystem()
    
    # Initialize system
    trading_system.initialize_system()
    
    # Get unique dates from data
    market_data = pd.read_csv("./csv/mongodb.csv")
    market_data['date'] = pd.to_datetime(market_data['date'])
    unique_dates = market_data['date'].unique()
    
    # Run simulation for each date
    print("\nStarting Trading Simulation...")
    for date in sorted(unique_dates)[:30]:  # Run for first 30 days
        date_str = date.strftime('%Y-%m-%d')
        print(f"\nProcessing {date_str}...")
        trading_system.run_daily_analysis(date)
    
    # Generate performance report
    trading_system.generate_performance_report()
    
    # Save results
    results_df = pd.DataFrame(trading_system.trade_history)
    results_df.to_csv('trading_results.csv', index=False)
    print("\nResults saved to 'trading_results.csv'")