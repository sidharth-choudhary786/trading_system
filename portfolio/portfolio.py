# trading_system/portfolio/portfolio.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from ..core.types import Trade, OrderSide
from ..core.exceptions import PortfolioError

@dataclass
class Position:
    """Represents a single position in the portfolio"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    weight: float

class Portfolio:
    """
    Manages portfolio state, positions, and performance tracking
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = Decimal(str(initial_capital))
        self.current_cash = Decimal(str(initial_capital))
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_value = Decimal(str(initial_capital))
        
        # Performance tracking
        self.portfolio_history = []
        self.daily_returns = []
        self.daily_pnl = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_portfolio_value = Decimal(str(initial_capital))
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Portfolio initialized with initial capital: ₹{initial_capital:,.2f}")
    
    def execute_trade(self, trade: Trade) -> bool:
        """
        Execute a trade and update portfolio positions
        
        Args:
            trade: Trade object to execute
            
        Returns:
            True if successful
        """
        try:
            # Convert to Decimal for precise calculations
            trade_quantity = Decimal(str(trade.quantity))
            trade_price = Decimal(str(trade.price))
            commission = Decimal(str(trade.commission))
            
            # Calculate trade value
            trade_value = trade_quantity * trade_price
            
            if trade.side == OrderSide.BUY:
                return self._execute_buy_trade(trade, trade_quantity, trade_price, trade_value, commission)
            else:
                return self._execute_sell_trade(trade, trade_quantity, trade_price, trade_value, commission)
                
        except Exception as e:
            self.logger.error(f"Error executing trade {trade.trade_id}: {e}")
            return False
    
    def _execute_buy_trade(self, trade: Trade, quantity: Decimal, price: Decimal, 
                          value: Decimal, commission: Decimal) -> bool:
        """Execute a buy trade"""
        total_cost = value + commission
        
        # Check if we have sufficient cash
        if self.current_cash < total_cost:
            self.logger.error(f"Insufficient cash for buy trade. Available: ₹{self.current_cash:.2f}, Required: ₹{total_cost:.2f}")
            return False
        
        # Update cash
        self.current_cash -= total_cost
        
        # Update or create position
        if trade.symbol in self.positions:
            position = self.positions[trade.symbol]
            # Calculate new average price
            total_quantity = Decimal(str(position.quantity)) + quantity
            total_value = (Decimal(str(position.quantity)) * Decimal(str(position.avg_price))) + value
            new_avg_price = total_value / total_quantity
            
            position.quantity = float(total_quantity)
            position.avg_price = float(new_avg_price)
            
        else:
            # Create new position
            self.positions[trade.symbol] = Position(
                symbol=trade.symbol,
                quantity=float(quantity),
                avg_price=float(price),
                current_price=float(price),
                market_value=float(value),
                unrealized_pnl=0.0,
                unrealized_pnl_percent=0.0,
                weight=0.0
            )
        
        # Record trade
        self.trades.append(trade)
        self.logger.info(f"Executed BUY trade: {trade.symbol} {quantity} @ ₹{price:.2f}")
        
        return True
    
    def _execute_sell_trade(self, trade: Trade, quantity: Decimal, price: Decimal, 
                           value: Decimal, commission: Decimal) -> bool:
        """Execute a sell trade"""
        # Check if we have sufficient position
        if trade.symbol not in self.positions:
            self.logger.error(f"No position found for symbol: {trade.symbol}")
            return False
        
        position = self.positions[trade.symbol]
        current_quantity = Decimal(str(position.quantity))
        
        if current_quantity < quantity:
            self.logger.error(f"Insufficient position. Available: {current_quantity}, Trying to sell: {quantity}")
            return False
        
        # Calculate P&L
        buy_value = quantity * Decimal(str(position.avg_price))
        sell_value = value - commission
        realized_pnl = sell_value - buy_value
        
        # Update cash
        self.current_cash += sell_value
        
        # Update position
        if current_quantity == quantity:
            # Close entire position
            del self.positions[trade.symbol]
        else:
            # Reduce position
            position.quantity = float(current_quantity - quantity)
        
        # Record trade with realized P&L
        trade.realized_pnl = float(realized_pnl)
        self.trades.append(trade)
        
        self.logger.info(f"Executed SELL trade: {trade.symbol} {quantity} @ ₹{price:.2f}, P&L: ₹{realized_pnl:.2f}")
        
        return True
    
    def update_portfolio(self, current_date: datetime, current_prices: Dict[str, float]):
        """
        Update portfolio valuation with current prices
        
        Args:
            current_date: Current date for valuation
            current_prices: Dictionary of symbol -> current price
        """
        try:
            total_market_value = Decimal('0.0')
            
            # Update positions with current prices
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    current_price = Decimal(str(current_prices[symbol]))
                    position.current_price = float(current_price)
                    position.market_value = float(Decimal(str(position.quantity)) * current_price)
                    
                    # Calculate unrealized P&L
                    cost_basis = Decimal(str(position.quantity)) * Decimal(str(position.avg_price))
                    current_value = Decimal(str(position.quantity)) * current_price
                    position.unrealized_pnl = float(current_value - cost_basis)
                    position.unrealized_pnl_percent = float((current_value - cost_basis) / cost_basis * 100)
                    
                    total_market_value += current_value
            
            # Calculate total portfolio value
            self.portfolio_value = self.current_cash + total_market_value
            
            # Update position weights
            if self.portfolio_value > 0:
                for symbol, position in self.positions.items():
                    position.weight = float(Decimal(str(position.market_value)) / self.portfolio_value * 100)
            
            # Update peak portfolio value and drawdown
            if self.portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = self.portfolio_value
            
            current_drawdown = float((self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value * 100)
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # Record portfolio snapshot
            snapshot = {
                'date': current_date,
                'portfolio_value': float(self.portfolio_value),
                'cash': float(self.current_cash),
                'market_value': float(total_market_value),
                'positions': self.get_positions_summary(),
                'daily_return': self._calculate_daily_return(),
                'drawdown': current_drawdown
            }
            self.portfolio_history.append(snapshot)
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
    
    def _calculate_daily_return(self) -> float:
        """Calculate daily return"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        previous_value = self.portfolio_history[-2]['portfolio_value']
        current_value = float(self.portfolio_value)
        
        if previous_value > 0:
            daily_return = (current_value - previous_value) / previous_value
            self.daily_returns.append(daily_return)
            return daily_return
        
        return 0.0
    
    def get_positions_summary(self) -> Dict[str, Any]:
        """Get summary of all positions"""
        return {
            symbol: {
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_percent': position.unrealized_pnl_percent,
                'weight': position.weight
            }
            for symbol, position in self.positions.items()
        }
    
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """Get comprehensive portfolio statistics"""
        if not self.portfolio_history:
            return {}
        
        # Basic stats
        total_return = (float(self.portfolio_value) - float(self.initial_capital)) / float(self.initial_capital) * 100
        
        # Calculate performance metrics
        returns_series = pd.Series(self.daily_returns)
        
        if len(returns_series) > 1:
            volatility = returns_series.std() * np.sqrt(252) * 100  # Annualized
            sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
        
        # Trade statistics
        if self.trades:
            winning_trades = [t for t in self.trades if getattr(t, 'realized_pnl', 0) > 0]
            losing_trades = [t for t in self.trades if getattr(t, 'realized_pnl', 0) < 0]
            win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
            avg_win = np.mean([t.realized_pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.realized_pnl for t in losing_trades]) if losing_trades else 0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
        
        stats = {
            'initial_capital': float(self.initial_capital),
            'current_portfolio_value': float(self.portfolio_value),
            'current_cash': float(self.current_cash),
            'total_market_value': float(self.portfolio_value - self.current_cash),
            'total_return_pct': total_return,
            'total_return_abs': float(self.portfolio_value - self.initial_capital),
            'max_drawdown_pct': self.max_drawdown,
            'volatility_pct': volatility,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trades),
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'current_positions': len(self.positions),
            'diversification_score': self._calculate_diversification_score()
        }
        
        return stats
    
    def _calculate_diversification_score(self) -> float:
        """Calculate portfolio diversification score (0-1)"""
        if not self.positions:
            return 0.0
        
        weights = [position.weight for position in self.positions.values()]
        if sum(weights) == 0:
            return 0.0
        
        # Herfindahl index - lower is more diversified
        herfindahl = sum((w / 100) ** 2 for w in weights)
        # Convert to diversification score (0-1)
        diversification = 1 - herfindahl
        
        return max(0.0, min(1.0, diversification))
    
    def get_sector_exposure(self, sector_mapping: Dict[str, str]) -> Dict[str, float]:
        """Calculate sector exposure"""
        sector_exposure = {}
        
        for symbol, position in self.positions.items():
            sector = sector_mapping.get(symbol, 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + position.weight
        
        return sector_exposure
    
    def rebalance_portfolio(self, target_weights: Dict[str, float], current_prices: Dict[str, float]) -> List[Trade]:
        """
        Generate trades to rebalance portfolio to target weights
        
        Args:
            target_weights: Dictionary of symbol -> target weight (0-1)
            current_prices: Current prices for calculation
            
        Returns:
            List of rebalancing trades
        """
        rebalancing_trades = []
        
        try:
            total_value = float(self.portfolio_value)
            
            for symbol, target_weight in target_weights.items():
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                target_value = total_value * target_weight
                
                if symbol in self.positions:
                    current_position = self.positions[symbol]
                    current_value = current_position.market_value
                else:
                    current_value = 0.0
                
                # Calculate value difference
                value_diff = target_value - current_value
                
                if abs(value_diff) < total_value * 0.001:  # Ignore small differences (0.1%)
                    continue
                
                # Calculate quantity to trade
                quantity = value_diff / current_price
                
                if quantity > 0:
                    # Buy order
                    order_side = OrderSide.BUY
                else:
                    # Sell order
                    order_side = OrderSide.SELL
                    quantity = abs(quantity)
                
                # Round quantity to avoid fractional shares
                quantity = int(quantity)
                
                if quantity > 0:
                    # Create trade (actual execution would happen elsewhere)
                    trade = Trade(
                        trade_id=f"rebalance_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        order_id=f"rebalance_order_{symbol}",
                        symbol=symbol,
                        side=order_side,
                        quantity=quantity,
                        price=current_price,
                        timestamp=datetime.now()
                    )
                    rebalancing_trades.append(trade)
            
            self.logger.info(f"Generated {len(rebalancing_trades)} rebalancing trades")
            
        except Exception as e:
            self.logger.error(f"Error in portfolio rebalancing: {e}")
        
        return rebalancing_trades
    
    def reset(self):
        """Reset portfolio to initial state"""
        self.current_cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.portfolio_value = self.initial_capital
        self.portfolio_history.clear()
        self.daily_returns.clear()
        self.daily_pnl.clear()
        self.max_drawdown = 0.0
        self.peak_portfolio_value = self.initial_capital
        
        self.logger.info("Portfolio reset to initial state")
