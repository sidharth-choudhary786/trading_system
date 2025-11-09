# trading_system/execution/live_execution.py
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging
import time

from .base import BaseExecutionHandler
from ..core.types import Order, Trade, OrderType, OrderSide
from ..core.exceptions import ExecutionError

class LiveExecution(BaseExecutionHandler):
    """
    Live execution handler - interfaces with real brokers
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Broker configuration
        self.broker = config.get('broker', 'zerodha')
        self.paper_trading = config.get('paper_trading', True)
        
        # Execution settings
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1)  # seconds
        
        # Initialize broker connection
        self.broker_api = self._initialize_broker_connection()
        
        # Order tracking
        self.active_orders = {}
        self.executed_trades = []
        
        self.logger.info(f"Live execution handler initialized for {self.broker}")
    
    def _initialize_broker_connection(self):
        """Initialize connection to broker API"""
        try:
            if self.broker == 'zerodha':
                from ..production.broker_apis.zerodha import ZerodhaAPI
                return ZerodhaAPI(self.config)
            elif self.broker == 'interactive_brokers':
                from ..production.broker_apis.interactive_brokers import InteractiveBrokersAPI
                return InteractiveBrokersAPI(self.config)
            elif self.broker == 'alpaca':
                from ..production.broker_apis.alpaca import AlpacaAPI
                return AlpacaAPI(self.config)
            else:
                raise ExecutionError(f"Unsupported broker: {self.broker}")
                
        except ImportError as e:
            raise ExecutionError(f"Failed to import broker API for {self.broker}: {e}")
        except Exception as e:
            raise ExecutionError(f"Failed to initialize broker connection: {e}")
    
    def execute_orders(
        self, 
        orders: List[Order],
        current_prices: Dict[str, float],
        timestamp: datetime
    ) -> List[Trade]:
        """
        Execute orders with live broker
        """
        executed_trades = []
        
        for order in orders:
            try:
                # Validate order
                is_valid, error_msg = self.validate_order(order)
                if not is_valid:
                    self.logger.error(f"Invalid order {order.order_id}: {error_msg}")
                    continue
                
                # Execute order with retry logic
                trade = self._execute_with_retry(order, timestamp)
                if trade:
                    executed_trades.append(trade)
                    self.executed_trades.append(trade)
                
            except Exception as e:
                self.logger.error(f"Failed to execute order {order.order_id}: {e}")
        
        self.logger.info(f"Executed {len(executed_trades)} live trades")
        return executed_trades
    
    def _execute_with_retry(self, order: Order, timestamp: datetime) -> Optional[Trade]:
        """Execute order with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Convert to broker-specific order format
                broker_order = self._convert_to_broker_order(order)
                
                # Place order through broker API
                broker_response = self.broker_api.place_order(broker_order)
                
                if broker_response.get('status') == 'success':
                    # Order placed successfully
                    broker_order_id = broker_response['order_id']
                    self.active_orders[order.order_id] = {
                        'broker_order_id': broker_order_id,
                        'order': order,
                        'timestamp': timestamp,
                        'status': 'PENDING'
                    }
                    
                    # For immediate execution brokers, check if filled
                    if self.broker_api.supports_immediate_execution():
                        fill_status = self.broker_api.get_order_status(broker_order_id)
                        if fill_status.get('status') == 'FILLED':
                            return self._create_trade_from_fill(order, fill_status, timestamp)
                    
                    return None
                
                else:
                    raise ExecutionError(f"Broker rejected order: {broker_response.get('error', 'Unknown error')}")
            
            except Exception as e:
                self.logger.warning(f"Order execution attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise ExecutionError(f"Order execution failed after {self.max_retries} attempts: {e}")
        
        return None
    
    def _convert_to_broker_order(self, order: Order) -> Dict:
        """Convert internal order to broker-specific format"""
        broker_order = {
            'symbol': order.symbol,
            'quantity': order.quantity,
            'side': order.side.value.upper(),
            'order_type': order.order_type.value.upper(),
            'product': 'MIS' if self.config.get('intraday', True) else 'CNC'
        }
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP]:
            broker_order['price'] = order.price
        
        if order.order_type == OrderType.STOP:
            broker_order['trigger_price'] = order.price
        
        return broker_order
    
    def _create_trade_from_fill(self, order: Order, fill_status: Dict, timestamp: datetime) -> Trade:
        """Create trade object from broker fill information"""
        fill_price = fill_status.get('average_price', fill_status.get('price', 0))
        fill_quantity = fill_status.get('filled_quantity', order.quantity)
        
        # Calculate commission and slippage
        commission = self.calculate_commission(order, fill_price)
        # Note: Slippage calculation would require reference price
        
        trade = Trade(
            trade_id=self.generate_trade_id(order),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            timestamp=timestamp,
            commission=commission,
            slippage=0.0  # Would need market data to calculate
        )
        
        return trade
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get status of an order from broker"""
        if order_id not in self.active_orders:
            return {'status': 'UNKNOWN', 'message': 'Order not found'}
        
        active_order = self.active_orders[order_id]
        broker_order_id = active_order['broker_order_id']
        
        try:
            broker_status = self.broker_api.get_order_status(broker_order_id)
            
            # Update local status
            active_order['status'] = broker_status.get('status', 'UNKNOWN')
            active_order['last_checked'] = datetime.now()
            
            # If order is filled, create trade record
            if broker_status.get('status') == 'FILLED' and order_id not in [t.order_id for t in self.executed_trades]:
                trade = self._create_trade_from_fill(
                    active_order['order'], 
                    broker_status, 
                    active_order['timestamp']
                )
                self.executed_trades.append(trade)
            
            return broker_status
            
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        if order_id not in self.active_orders:
            self.logger.warning(f"Order {order_id} not found")
            return False
        
        try:
            broker_order_id = self.active_orders[order_id]['broker_order_id']
            result = self.broker_api.cancel_order(broker_order_id)
            
            if result.get('status') == 'CANCELLED':
                del self.active_orders[order_id]
                self.logger.info(f"Successfully cancelled order {order_id}")
                return True
            else:
                self.logger.error(f"Failed to cancel order {order_id}: {result.get('error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def sync_orders(self) -> List[Trade]:
        """Sync order status with broker and return new fills"""
        new_trades = []
        
        for order_id in list(self.active_orders.keys()):
            try:
                status = self.get_order_status(order_id)
                
                # Remove completed orders from active tracking
                if status.get('status') in ['FILLED', 'CANCELLED', 'REJECTED']:
                    if status.get('status') == 'FILLED':
                        # Trade should already be created in get_order_status
                        pass
                    del self.active_orders[order_id]
                    
            except Exception as e:
                self.logger.error(f"Error syncing order {order_id}: {e}")
        
        return new_trades
    
    def get_account_info(self) -> Dict:
        """Get account information from broker"""
        try:
            return self.broker_api.get_account_info()
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_positions(self) -> Dict:
        """Get current positions from broker"""
        try:
            return self.broker_api.get_positions()
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        total_trades = len(self.executed_trades)
        active_orders = len(self.active_orders)
        
        if total_trades > 0:
            avg_commission = sum(t.commission for t in self.executed_trades) / total_trades
        else:
            avg_commission = 0.0
        
        return {
            'broker': self.broker,
            'paper_trading': self.paper_trading,
            'total_trades': total_trades,
            'active_orders': active_orders,
            'avg_commission': avg_commission,
            'total_commission': sum(t.commission for t in self.executed_trades)
        }
