# trading_system/execution/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

from ..core.types import Order, Trade, OrderType, OrderSide
from ..core.exceptions import ExecutionError

class BaseExecutionHandler(ABC):
    """
    Abstract base class for all execution handlers
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.commission = config.get('commission', 0.001)  # 0.1% default
        self.slippage_model = None
        
    @abstractmethod
    def execute_orders(
        self, 
        orders: List[Order],
        current_prices: Dict[str, float],
        timestamp: datetime
    ) -> List[Trade]:
        """
        Execute a list of orders
        
        Args:
            orders: List of orders to execute
            current_prices: Current market prices for symbols
            timestamp: Execution timestamp
            
        Returns:
            List of executed trades
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get status of a specific order
        
        Args:
            order_id: Order identifier
            
        Returns:
            Order status information
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order
        
        Args:
            order_id: Order identifier
            
        Returns:
            True if successful
        """
        pass
    
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """
        Calculate commission for an order
        
        Args:
            order: Order object
            fill_price: Execution price
            
        Returns:
            Commission amount
        """
        notional_value = order.quantity * fill_price
        return notional_value * self.commission
    
    def apply_slippage(self, order: Order, current_price: float) -> float:
        """
        Apply slippage to execution price
        
        Args:
            order: Order object
            current_price: Current market price
            
        Returns:
            Slippage-adjusted price
        """
        if self.slippage_model:
            return self.slippage_model.calculate_execution_price(order, current_price)
        return current_price
    
    def validate_order(self, order: Order) -> Tuple[bool, str]:
        """
        Validate order parameters
        
        Args:
            order: Order to validate
            
        Returns:
            (is_valid, error_message)
        """
        if order.quantity <= 0:
            return False, "Order quantity must be positive"
        
        if order.order_type == OrderType.LIMIT and order.price is None:
            return False, "Limit orders require a price"
        
        if order.order_type == OrderType.STOP and order.price is None:
            return False, "Stop orders require a price"
        
        return True, ""
    
    def generate_trade_id(self, order: Order) -> str:
        """
        Generate unique trade ID
        
        Args:
            order: Source order
            
        Returns:
            Unique trade identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"TRADE_{order.symbol}_{timestamp}_{hash(order.order_id) % 10000:04d}"
