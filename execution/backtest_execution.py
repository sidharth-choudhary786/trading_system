# trading_system/execution/backtest_execution.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from .base import BaseExecutionHandler
from .slippage_models import SlippageModel
from ..core.types import Order, Trade, OrderType, OrderSide
from ..core.exceptions import ExecutionError

class BacktestExecution(BaseExecutionHandler):
    """
    Backtest execution handler - simulates order execution
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Execution configuration
        self.use_next_open = config.get('use_next_open', True)
        self.partial_fills = config.get('partial_fills', True)
        self.fill_probability = config.get('fill_probability', 0.95)  # 95% fill rate
        
        # Initialize slippage model
        slippage_config = config.get('slippage', {})
        self.slippage_model = SlippageModel(slippage_config)
        
        # Order tracking
        self.pending_orders = {}
        self.executed_trades = []
        self.failed_orders = []
        
        self.logger.info("Backtest execution handler initialized")
    
    def execute_orders(
        self, 
        orders: List[Order],
        current_prices: Dict[str, float],
        timestamp: datetime
    ) -> List[Trade]:
        """
        Execute orders in backtest environment
        """
        executed_trades = []
        
        for order in orders:
            try:
                # Validate order
                is_valid, error_msg = self.validate_order(order)
                if not is_valid:
                    self.logger.warning(f"Invalid order {order.order_id}: {error_msg}")
                    self.failed_orders.append({
                        'order': order,
                        'error': error_msg,
                        'timestamp': timestamp
                    })
                    continue
                
                # Check if symbol has price data
                if order.symbol not in current_prices:
                    self.logger.warning(f"No price data for {order.symbol}")
                    continue
                
                # Execute order based on type
                if order.order_type == OrderType.MARKET:
                    trades = self._execute_market_order(order, current_prices[order.symbol], timestamp)
                elif order.order_type == OrderType.LIMIT:
                    trades = self._execute_limit_order(order, current_prices[order.symbol], timestamp)
                elif order.order_type == OrderType.STOP:
                    trades = self._execute_stop_order(order, current_prices[order.symbol], timestamp)
                else:
                    self.logger.warning(f"Unsupported order type: {order.order_type}")
                    continue
                
                executed_trades.extend(trades)
                
            except Exception as e:
                self.logger.error(f"Error executing order {order.order_id}: {e}")
                self.failed_orders.append({
                    'order': order,
                    'error': str(e),
                    'timestamp': timestamp
                })
        
        # Store executed trades
        self.executed_trades.extend(executed_trades)
        
        self.logger.info(f"Executed {len(executed_trades)} trades from {len(orders)} orders")
        return executed_trades
    
    def _execute_market_order(
        self, 
        order: Order, 
        current_price: float, 
        timestamp: datetime
    ) -> List[Trade]:
        """
        Execute market order
        """
        # Apply slippage
        execution_price = self.apply_slippage(order, current_price)
        
        # Calculate commission
        commission = self.calculate_commission(order, execution_price)
        
        # Create trade
        trade = Trade(
            trade_id=self.generate_trade_id(order),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=timestamp,
            commission=commission,
            slippage=execution_price - current_price
        )
        
        return [trade]
    
    def _execute_limit_order(
        self, 
        order: Order, 
        current_price: float, 
        timestamp: datetime
    ) -> List[Trade]:
        """
        Execute limit order
        """
        # Check if limit price condition is met
        if (order.side == OrderSide.BUY and current_price <= order.price) or \
           (order.side == OrderSide.SELL and current_price >= order.price):
            
            # Limit order can be executed
            execution_price = order.price  # Limit orders execute at limit price
            
            # Apply additional slippage for limit orders
            execution_price = self.apply_slippage(order, execution_price)
            
            commission = self.calculate_commission(order, execution_price)
            
            trade = Trade(
                trade_id=self.generate_trade_id(order),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                timestamp=timestamp,
                commission=commission,
                slippage=execution_price - order.price
            )
            
            return [trade]
        
        else:
            # Limit order not executed, add to pending orders
            self.pending_orders[order.order_id] = {
                'order': order,
                'timestamp': timestamp,
                'current_price': current_price
            }
            return []
    
    def _execute_stop_order(
        self, 
        order: Order, 
        current_price: float, 
        timestamp: datetime
    ) -> List[Trade]:
        """
        Execute stop order
        """
        # Check if stop condition is met
        if (order.side == OrderSide.BUY and current_price >= order.price) or \
           (order.side == OrderSide.SELL and current_price <= order.price):
            
            # Stop order triggered, becomes market order
            execution_price = self.apply_slippage(order, current_price)
            commission = self.calculate_commission(order, execution_price)
            
            trade = Trade(
                trade_id=self.generate_trade_id(order),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                timestamp=timestamp,
                commission=commission,
                slippage=execution_price - current_price
            )
            
            return [trade]
        
        else:
            # Stop order not triggered
            self.pending_orders[order.order_id] = {
                'order': order,
                'timestamp': timestamp,
                'current_price': current_price
            }
            return []
    
    def process_pending_orders(
        self, 
        current_prices: Dict[str, float], 
        timestamp: datetime
    ) -> List[Trade]:
        """
        Process pending orders (limit and stop orders)
        """
        executed_trades = []
        remaining_orders = {}
        
        for order_id, order_info in self.pending_orders.items():
            order = order_info['order']
            
            if order.symbol not in current_prices:
                remaining_orders[order_id] = order_info
                continue
            
            current_price = current_prices[order.symbol]
            trades = []
            
            if order.order_type == OrderType.LIMIT:
                trades = self._execute_limit_order(order, current_price, timestamp)
            elif order.order_type == OrderType.STOP:
                trades = self._execute_stop_order(order, current_price, timestamp)
            
            if trades:
                executed_trades.extend(trades)
            else:
                remaining_orders[order_id] = order_info
        
        # Update pending orders
        self.pending_orders = remaining_orders
        
        # Store executed trades
        self.executed_trades.extend(executed_trades)
        
        if executed_trades:
            self.logger.info(f"Processed {len(executed_trades)} pending orders")
        
        return executed_trades
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get status of an order"""
        # Check if order is pending
        if order_id in self.pending_orders:
            return {
                'order_id': order_id,
                'status': 'PENDING',
                'timestamp': self.pending_orders[order_id]['timestamp']
            }
        
        # Check if order was executed
        executed_trades = [t for t in self.executed_trades if t.order_id == order_id]
        if executed_trades:
            return {
                'order_id': order_id,
                'status': 'FILLED',
                'fill_timestamp': executed_trades[0].timestamp,
                'fill_price': executed_trades[0].price,
                'quantity': executed_trades[0].quantity
            }
        
        # Check if order failed
        failed_order = next((fo for fo in self.failed_orders if fo['order'].order_id == order_id), None)
        if failed_order:
            return {
                'order_id': order_id,
                'status': 'FAILED',
                'error': failed_order['error'],
                'timestamp': failed_order['timestamp']
            }
        
        return {
            'order_id': order_id,
            'status': 'UNKNOWN',
            'message': 'Order not found'
        }
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            self.logger.info(f"Cancelled order {order_id}")
            return True
        
        self.logger.warning(f"Order {order_id} not found or not cancellable")
        return False
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        total_trades = len(self.executed_trades)
        total_orders = total_trades + len(self.pending_orders) + len(self.failed_orders)
        
        if total_orders == 0:
            return {
                'total_orders': 0,
                'executed_orders': 0,
                'pending_orders': 0,
                'failed_orders': 0,
                'execution_rate': 0.0,
                'avg_slippage': 0.0,
                'avg_commission': 0.0
            }
        
        # Calculate average metrics
        if total_trades > 0:
            avg_slippage = np.mean([t.slippage for t in self.executed_trades])
            avg_commission = np.mean([t.commission for t in self.executed_trades])
        else:
            avg_slippage = 0.0
            avg_commission = 0.0
        
        return {
            'total_orders': total_orders,
            'executed_orders': total_trades,
            'pending_orders': len(self.pending_orders),
            'failed_orders': len(self.failed_orders),
            'execution_rate': total_trades / total_orders,
            'avg_slippage': avg_slippage,
            'avg_commission': avg_commission,
            'total_commission': sum(t.commission for t in self.executed_trades)
        }
    
    def reset(self):
        """Reset execution handler state"""
        self.pending_orders.clear()
        self.executed_trades.clear()
        self.failed_orders.clear()
        self.logger.info("Execution handler reset")
