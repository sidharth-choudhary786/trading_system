# trading_system/analysis/backtester.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..core.base_system import TradingSystem
from ..core.exceptions import TradingSystemError
from ..core.types import Order, Trade, OrderSide, OrderType
from ..portfolio.portfolio import Portfolio
from ..execution.backtest_execution import BacktestExecution

class Backtester:
    """
    Backtesting engine for strategy validation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.initial_capital = config.get('initial_capital', 100000)
        self.commission = config.get('commission', 0.001)
        self.slippage = config.get('slippage', 0.001)
        self.portfolio = Portfolio(self.initial_capital)
        self.execution = BacktestExecution(commission=self.commission, slippage=self.slippage)
        
        # Results storage
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        
    def run_backtest(
        self, 
        data: Dict[str, pd.DataFrame],
        strategy,
        start_date: str,
        end_date: str,
        rebalance_frequency: str = 'fortnightly'
    ) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            data: Dictionary of symbol -> DataFrame with historical data
            strategy: Strategy instance with generate_signals method
            start_date: Backtest start date
            end_date: Backtest end date
            rebalance_frequency: 'daily', 'weekly', 'fortnightly', 'monthly'
        """
        print(f"Running backtest from {start_date} to {end_date}")
        
        # Convert to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Generate date range
        dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        # Initialize tracking
        self._initialize_backtest()
        
        # Main backtest loop
        for current_date in dates:
            if current_date.weekday() >= 5:  # Skip weekends
                continue
                
            self._process_day(current_date, data, strategy, rebalance_frequency)
        
        # Calculate final results
        results = self._calculate_performance_metrics()
        return results
    
    def _initialize_backtest(self):
        """Initialize backtest tracking"""
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        self.portfolio = Portfolio(self.initial_capital)
        
    def _process_day(self, current_date: datetime, data: Dict, strategy, rebalance_frequency: str):
        """
        Process a single day in backtest
        """
        # Get current prices for all symbols
        current_prices = {}
        for symbol, df in data.items():
            if current_date in df.index:
                current_prices[symbol] = df.loc[current_date, 'close']
        
        # Update portfolio value
        self.portfolio.update_portfolio(current_date, current_prices)
        
        # Check if it's rebalance day
        if self._is_rebalance_day(current_date, rebalance_frequency):
            # Generate signals
            signals = strategy.generate_signals(data, current_date)
            
            # Generate orders based on signals
            orders = self._generate_orders(signals, current_prices)
            
            # Execute orders
            trades = self.execution.execute_orders(orders, current_prices, current_date)
            
            # Update portfolio with trades
            for trade in trades:
                self.portfolio.execute_trade(trade)
                self.trade_history.append(trade)
        
        # Record portfolio snapshot
        self._record_daily_snapshot(current_date, current_prices)
    
    def _is_rebalance_day(self, current_date: datetime, frequency: str) -> bool:
        """
        Check if current date is a rebalance day
        """
        if frequency == 'daily':
            return True
        elif frequency == 'weekly':
            return current_date.weekday() == 0  # Monday
        elif frequency == 'fortnightly':
            # Every 2 weeks (10 business days)
            days_since_start = (current_date - pd.Timestamp('2000-01-01')).days
            return days_since_start % 14 == 0
        elif frequency == 'monthly':
            return current_date.day == 1
        else:
            return False
    
    def _generate_orders(self, signals: Dict, current_prices: Dict) -> List[Order]:
        """
        Generate orders based on signals
        """
        orders = []
        
        for symbol, signal in signals.items():
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            if signal == 'BUY' and symbol not in self.portfolio.positions:
                # Calculate position size (simplified)
                position_size = self._calculate_position_size(symbol, current_price)
                
                order = Order(
                    order_id=f"order_{len(orders)}_{symbol}",
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY,
                    quantity=position_size,
                    timestamp=datetime.now()
                )
                orders.append(order)
                
            elif signal == 'SELL' and symbol in self.portfolio.positions:
                current_position = self.portfolio.positions[symbol]
                
                order = Order(
                    order_id=f"order_{len(orders)}_{symbol}",
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL,
                    quantity=current_position,
                    timestamp=datetime.now()
                )
                orders.append(order)
        
        return orders
    
    def _calculate_position_size(self, symbol: str, current_price: float) -> int:
        """
        Calculate position size based on risk management
        """
        # Simple position sizing - 10% of portfolio per stock
        portfolio_value = self.portfolio.portfolio_value
        max_position_value = portfolio_value * 0.1
        position_size = max_position_value / current_price
        
        # Round down to integer shares
        return int(position_size)
    
    def _record_daily_snapshot(self, date: datetime, prices: Dict):
        """
        Record daily portfolio snapshot
        """
        snapshot = {
            'date': date,
            'portfolio_value': self.portfolio.portfolio_value,
            'cash': self.portfolio.current_cash,
            'positions': self.portfolio.positions.copy(),
            'daily_return': self.portfolio.daily_returns.get(date, 0)
        }
        self.portfolio_history.append(snapshot)
        
        # Record daily return
        if len(self.portfolio_history) > 1:
            prev_value = self.portfolio_history[-2]['portfolio_value']
            current_value = self.portfolio.portfolio_value
            daily_return = (current_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if not self.portfolio_history:
            return {}
            
        # Extract portfolio values and returns
        portfolio_values = [snapshot['portfolio_value'] for snapshot in self.portfolio_history]
        dates = [snapshot['date'] for snapshot in self.portfolio_history]
        
        returns = pd.Series(self.daily_returns, index=dates[:len(self.daily_returns)])
        
        # Calculate metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
        if len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        portfolio_series = pd.Series(portfolio_values, index=dates)
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate from trades
        if self.trade_history:
            profitable_trades = [t for t in self.trade_history 
                               if (t.side == OrderSide.SELL and 
                                   t.price > self._get_average_buy_price(t.symbol))]
            win_rate = len(profitable_trades) / len(self.trade_history) if self.trade_history else 0
        else:
            win_rate = 0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': portfolio_values[-1],
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate': win_rate,
            'total_trades': len(self.trade_history),
            'portfolio_history': self.portfolio_history,
            'trade_history': self.trade_history,
            'daily_returns': returns.tolist()
        }
        
        return results
    
    def _get_average_buy_price(self, symbol: str) -> float:
        """
        Calculate average buy price for a symbol
        """
        buy_trades = [t for t in self.trade_history 
                     if t.symbol == symbol and t.side == OrderSide.BUY]
        
        if not buy_trades:
            return 0
        
        total_cost = sum(t.quantity * t.price for t in buy_trades)
        total_quantity = sum(t.quantity for t in buy_trades)
        
        return total_cost / total_quantity if total_quantity > 0 else 0
    
    def get_backtest_results(self) -> Dict:
        """Get backtest results"""
        return self._calculate_performance_metrics()
