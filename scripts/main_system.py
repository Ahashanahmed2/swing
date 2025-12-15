# src/main_system.py
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all components
from src.xgboost_model import XGBoostTradingModel
from src.ppo_agent import PPOTradingAgent
from src.risk_engine import RiskEngine
from src.portfolio_allocator import PortfolioAllocator
from config import *

class TradingSystem:
    def __init__(self):
        print("Initializing Trading System...")
        
        # Initialize components
        self.xgb_model = XGBoostTradingModel()
        self.ppo_agent = PPOTradingAgent()
        self.risk_engine = RiskEngine()
        self.portfolio_allocator = PortfolioAllocator()
        
        # Initialize portfolio
        self.portfolio = {}
        self.account_balance = INITIAL_BALANCE
        self.portfolio_value = self.account_balance
        self.trade_history = []
        
        # Load data
        self.market_data = pd.read_csv(MONGODB_PATH)
        self.market_data['date'] = pd.to_datetime(self.market_data['date'])
        
        self.trade_data = pd.read_csv(TRADE_STOCK_PATH)
        self.trade_data['date'] = pd.to_datetime(self.trade_data['date'])
        
        print("Trading System initialized successfully!")
    
    def initialize_models(self):
        """ইনিশিয়ালাইজ মডেল"""
        print("\nInitializing models...")
        
        # Train or load XGBoost
        try:
            self.xgb_model.load_model()
            print("XGBoost model loaded from file")
        except:
            print("Training new XGBoost model...")
            X, y = self.xgb_model.load_and_prepare_data()
            self.xgb_model.train(X, y)
            self.xgb_model.save_model()
        
        # Load PPO if exists
        try:
            self.ppo_agent.load_model()
            print("PPO model loaded from file")
        except:
            print("PPO model will be trained during simulation")
        
        print("Models initialized!")
    
    def run_daily_trading(self, date):
        """রান ডেইলি ট্রেডিং"""
        date_str = date.strftime('%Y-%m-%d')
        print(f"\nProcessing {date_str}...")
        
        # Get daily data
        daily_data = self.market_data[self.market_data['date'] == date]
        
        if daily_data.empty:
            print(f"No data available for {date_str}")
            return []
        
        # Get XGBoost predictions
        predictions, probabilities = self.xgb_model.predict(daily_data)
        
        # Prepare trade signals
        trade_signals = {}
        strong_signals = []
        
        for idx, row in daily_data.iterrows():
            symbol = row['symbol']
            if predictions[idx] == 1 and probabilities[idx] > 0.7:
                trade_signals[symbol] = probabilities[idx]
                strong_signals.append(symbol)
        
        if not strong_signals:
            print("No strong trading signals today")
            return []
        
        print(f"Strong signals detected for: {strong_signals}")
        
        # Calculate portfolio allocation
        target_allocations = self.portfolio_allocator.calculate_allocation(
            self.market_data,
            trade_signals,
            self.account_balance
        )
        
        if not target_allocations:
            print("No valid allocations calculated")
            return []
        
        # Generate rebalance orders
        rebalance_orders = self.portfolio_allocator.rebalance_portfolio(
            self.portfolio,
            target_allocations
        )
        
        # Execute orders
        executed_orders = []
        for order in rebalance_orders:
            order_result = self._execute_order(order, date)
            if order_result:
                executed_orders.append(order_result)
        
        # Update portfolio value
        self._update_portfolio_value(daily_data)
        
        # Log results
        self._log_daily_results(date, executed_orders)
        
        return executed_orders
    
    def _execute_order(self, order, date):
        """এক্সিকিউট অর্ডার"""
        symbol = order['symbol']
        order_type = order['type']
        shares = order['shares']
        
        # Get current price
        symbol_data = self.market_data[
            (self.market_data['symbol'] == symbol) & 
            (self.market_data['date'] == date)
        ]
        
        if symbol_data.empty:
            return None
        
        current_price = symbol_data['close'].iloc[0]
        
        # Create trade data for risk validation
        trade_data = {
            'symbol': symbol,
            'buy': current_price,
            'SL': current_price * 0.95,
            'tp': current_price * 1.1,
            'date': date
        }
        
        # Validate trade
        is_valid, risk_info = self.risk_engine.validate_trade(
            trade_data,
            self.account_balance,
            self.portfolio_value
        )
        
        if not is_valid:
            print(f"Trade rejected for {symbol}: {risk_info}")
            return None
        
        # Execute trade
        trade_value = shares * current_price
        
        if order_type == 'BUY':
            if trade_value > self.account_balance:
                print(f"Insufficient balance for {symbol}: {trade_value:.2f} > {self.account_balance:.2f}")
                return None
            
            self.account_balance -= trade_value
            
            if symbol in self.portfolio:
                self.portfolio[symbol]['shares'] += shares
                self.portfolio[symbol]['value'] += trade_value
                self.portfolio[symbol]['entry_price'] = (
                    (self.portfolio[symbol]['entry_price'] * (self.portfolio[symbol]['shares'] - shares) + 
                     current_price * shares) / self.portfolio[symbol]['shares']
                )
            else:
                self.portfolio[symbol] = {
                    'shares': shares,
                    'value': trade_value,
                    'entry_price': current_price,
                    'entry_date': date
                }
            
            print(f"Bought {shares} shares of {symbol} at {current_price:.2f}")
            
        else:  # SELL
            if symbol not in self.portfolio or self.portfolio[symbol]['shares'] < shares:
                print(f"Insufficient shares to sell {symbol}")
                return None
            
            self.account_balance += trade_value
            self.portfolio[symbol]['shares'] -= shares
            self.portfolio[symbol]['value'] -= shares * current_price
            
            if self.portfolio[symbol]['shares'] == 0:
                del self.portfolio[symbol]
            
            print(f"Sold {shares} shares of {symbol} at {current_price:.2f}")
        
        # Update risk engine position
        if symbol in self.portfolio:
            self.risk_engine.update_position(symbol, self.portfolio[symbol])
        
        # Return order result
        return {
            'date': date,
            'symbol': symbol,
            'type': order_type,
            'shares': shares,
            'price': current_price,
            'value': trade_value,
            'risk_info': risk_info
        }
    
    def _update_portfolio_value(self, daily_data):
        """আপডেট পোর্টফোলিও ভ্যালু"""
        total_value = self.account_balance
        
        for symbol, position in self.portfolio.items():
            symbol_data = daily_data[daily_data['symbol'] == symbol]
            if not symbol_data.empty:
                current_price = symbol_data['close'].iloc[0]
                current_value = position['shares'] * current_price
                position['current_value'] = current_value
                position['current_price'] = current_price
                position['pnl'] = current_value - position['value']
                position['pnl_pct'] = (current_price - position['entry_price']) / position['entry_price']
                total_value += current_value
        
        self.portfolio_value = total_value
        return total_value
    
    def _log_daily_results(self, date, executed_orders):
        """লগ ডেইলি রেজাল্ট"""
        log_entry = {
            'date': date,
            'portfolio_value': self.portfolio_value,
            'account_balance': self.account_balance,
            'number_of_positions': len(self.portfolio),
            'executed_orders': len(executed_orders),
            'orders': executed_orders
        }
        
        self.trade_history.append(log_entry)
        
        print(f"\n=== Daily Summary for {date} ===")
        print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"Account Balance: ${self.account_balance:,.2f}")
        print(f"Number of Positions: {len(self.portfolio)}")
        print(f"Orders Executed: {len(executed_orders)}")
        
        if self.portfolio:
            print("\nCurrent Positions:")
            for symbol, position in self.portfolio.items():
                pnl_pct = position.get('pnl_pct', 0) * 100
                print(f"  {symbol}: {position['shares']} shares, "
                      f"P&L: {pnl_pct:+.2f}%")
    
    def generate_report(self):
        """জেনারেট রিপোর্ট"""
        if not self.trade_history:
            print("No trading history available")
            return
        
        print("\n" + "="*60)
        print("TRADING SYSTEM PERFORMANCE REPORT")
        print("="*60)
        
        # Calculate performance metrics
        initial_value = INITIAL_BALANCE
        final_value = self.portfolio_value
        total_return_pct = (final_value - initial_value) / initial_value * 100
        
        # Get daily values
        daily_values = [h['portfolio_value'] for h in self.trade_history]
        
        if len(daily_values) > 1:
            daily_returns = np.diff(daily_values) / daily_values[:-1]
            
            if len(daily_returns) > 0 and daily_returns.std() > 0:
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            else:
                sharpe_ratio = 0
            
            # Calculate max drawdown
            peak = np.maximum.accumulate(daily_values)
            drawdown = (daily_values - peak) / peak
            max_drawdown = np.min(drawdown) * 100
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        total_trades = sum(h['executed_orders'] for h in self.trade_history)
        
        print(f"\nPerformance Metrics:")
        print(f"Initial Capital: ${initial_value:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return_pct:+.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Total Trading Days: {len(self.trade_history)}")
        print(f"Total Trades Executed: {total_trades}")
        
        # Calculate win rate
        winning_trades = 0
        for day in self.trade_history:
            for order in day['orders']:
                if order['type'] == 'SELL':
                    winning_trades += 1  # Simplified
        
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            print(f"Win Rate: {win_rate:.1f}%")
        
        print("\n" + "="*60)
        
        # Save report to file
        report_data = {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return_pct': total_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'total_trades': total_trades,
            'trading_days': len(self.trade_history)
        }
        
        report_df = pd.DataFrame([report_data])
        report_df.to_csv('performance_report.csv', index=False)
        print("Performance report saved to 'performance_report.csv'")
        
        # Save trade history
        history_df = pd.DataFrame(self.trade_history)
        history_df.to_csv('trade_history.csv', index=False)
        print("Trade history saved to 'trade_history.csv'")

def main():
    """মেইন এক্সিকিউশন ফাংশন"""
    print("="*60)
    print("AUTOMATED TRADING SYSTEM")
    print("="*60)
    
    # Create trading system
    trading_system = TradingSystem()
    
    # Initialize models
    trading_system.initialize_models()
    
    # Get unique dates
    unique_dates = trading_system.market_data['date'].unique()
    
    print(f"\nFound {len(unique_dates)} trading days")
    print("Starting simulation...")
    
    # Run simulation for each date
    for i, date in enumerate(sorted(unique_dates)[:50]):  # Limit to 50 days for testing
        print(f"\nDay {i+1}/{min(50, len(unique_dates))}")
        trading_system.run_daily_trading(date)
    
    # Generate final report
    trading_system.generate_report()
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()