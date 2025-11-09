# trading_system/execution/slippage_models.py
import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging
from abc import ABC, abstractmethod

from ..core.types import Order, OrderSide
from ..core.exceptions import ExecutionError

class BaseSlippageModel(ABC):
    """
    Abstract base class for slippage models
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def calculate_execution_price(self, order: Order, current_price: float) -> float:
        """
        Calculate execution price with slippage
        
        Args:
            order: Order to execute
            current_price: Current market price
            
        Returns:
            Execution price with slippage
        """
        pass

class SlippageModel(BaseSlippageModel):
    """
    Comprehensive slippage model with multiple methods
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.method = config.get('method', 'constant')
        
        # Method configurations
        self.method_configs = {
            'constant': {
                'name': 'Constant Slippage',
                'buy_slippage': config.get('buy_slippage', 0.001),  # 0.1% for buys
                'sell_slippage': config.get('sell_slippage', 0.001)  # 0.1% for sells
            },
            'volume_based': {
                'name': 'Volume-Based Slippage',
                'base_slippage': config.get('base_slippage', 0.0005),
                'volume_impact': config.get('volume_impact', 0.0001)
            },
            'volatility_based': {
                'name': 'Volatility-Based Slippage', 
                'base_slippage': config.get('base_slippage', 0.0005),
                'volatility_multiplier': config.get('volatility_multiplier', 2.0)
            },
            'spread_based': {
                'name': 'Spread-Based Slippage',
                'spread_multiplier': config.get('spread_multiplier', 0.5)
            },
            'hybrid': {
                'name': 'Hybrid Slippage Model',
                'weights': config.get('weights', [0.3, 0.3, 0.4])  # volume, volatility, spread
            }
        }
        
        # Market data for advanced models
        self.volume_data = {}
        self.volatility_data = {}
        self.spread_data = {}
    
    def calculate_execution_price(self, order: Order, current_price: float) -> float:
        """
        Calculate execution price using selected slippage method
        """
        if current_price <= 0:
            return current_price
        
        if self.method == 'constant':
            return self._constant_slippage(order, current_price)
        elif self.method == 'volume_based':
            return self._volume_based_slippage(order, current_price)
        elif self.method == 'volatility_based':
            return self._volatility_based_slippage(order, current_price)
        elif self.method == 'spread_based':
            return self._spread_based_slippage(order, current_price)
        elif self.method == 'hybrid':
            return self._hybrid_slippage(order, current_price)
        else:
            self.logger.warning(f"Unknown slippage method: {self.method}. Using constant.")
            return self._constant_slippage(order, current_price)
    
    def _constant_slippage(self, order: Order, current_price: float) -> float:
        """Constant percentage slippage"""
        config = self.method_configs['constant']
        
        if order.side == OrderSide.BUY:
            slippage_pct = config['buy_slippage']
            # Buy orders execute at higher price
            execution_price = current_price * (1 + slippage_pct)
        else:  # SELL
            slippage_pct = config['sell_slippage']  
            # Sell orders execute at lower price
            execution_price = current_price * (1 - slippage_pct)
        
        return execution_price
    
    def _volume_based_slippage(self, order: Order, current_price: float) -> float:
        """Volume-based slippage - higher volume = more slippage"""
        config = self.method_configs['volume_based']
        base_slippage = config['base_slippage']
        volume_impact = config['volume_impact']
        
        # Get volume information for the symbol
        symbol = order.symbol
        avg_daily_volume = self.volume_data.get(symbol, {}).get('avg_daily_volume', 1000000)
        current_volume = self.volume_data.get(symbol, {}).get('current_volume', avg_daily_volume)
        
        # Calculate order size relative to average volume
        order_value = order.quantity * current_price
        volume_ratio = min(1.0, order_value / (avg_daily_volume * current_price))
        
        # Slippage increases with order size relative to volume
        slippage_pct = base_slippage + (volume_impact * volume_ratio * 10)  # Scale factor
        
        if order.side == OrderSide.BUY:
            execution_price = current_price * (1 + slippage_pct)
        else:
            execution_price = current_price * (1 - slippage_pct)
        
        return execution_price
    
    def _volatility_based_slippage(self, order: Order, current_price: float) -> float:
        """Volatility-based slippage - higher volatility = more slippage"""
        config = self.method_configs['volatility_based']
        base_slippage = config['base_slippage']
        volatility_multiplier = config['volatility_multiplier']
        
        # Get volatility information for the symbol
        symbol = order.symbol
        volatility = self.volatility_data.get(symbol, {}).get('volatility', 0.02)  # 2% default
        
        # Slippage increases with volatility
        slippage_pct = base_slippage + (volatility * volatility_multiplier)
        
        if order.side == OrderSide.BUY:
            execution_price = current_price * (1 + slippage_pct)
        else:
            execution_price = current_price * (1 - slippage_pct)
        
        return execution_price
    
    def _spread_based_slippage(self, order: Order, current_price: float) -> float:
        """Spread-based slippage - wider spreads = more slippage"""
        config = self.method_configs['spread_based']
        spread_multiplier = config['spread_multiplier']
        
        # Get spread information for the symbol
        symbol = order.symbol
        spread_pct = self.spread_data.get(symbol, {}).get('spread_pct', 0.001)  # 0.1% default spread
        
        # Slippage is proportional to spread
        slippage_pct = spread_pct * spread_multiplier
        
        if order.side == OrderSide.BUY:
            # Buy at ask price (current_price + half spread)
            execution_price = current_price * (1 + (spread_pct / 2) + slippage_pct)
        else:
            # Sell at bid price (current_price - half spread)  
            execution_price = current_price * (1 - (spread_pct / 2) - slippage_pct)
        
        return execution_price
    
    def _hybrid_slippage(self, order: Order, current_price: float) -> float:
        """Hybrid slippage model combining multiple factors"""
        config = self.method_configs['hybrid']
        weights = config['weights']  # [volume_weight, volatility_weight, spread_weight]
        
        # Calculate slippage from each component
        volume_price = self._volume_based_slippage(order, current_price)
        volatility_price = self._volatility_based_slippage(order, current_price) 
        spread_price = self._spread_based_slippage(order, current_price)
        
        # Calculate weighted average
        volume_slippage = abs(volume_price - current_price) / current_price
        volatility_slippage = abs(volatility_price - current_price) / current_price
        spread_slippage = abs(spread_price - current_price) / current_price
        
        # Apply weights
        total_slippage = (volume_slippage * weights[0] + 
                         volatility_slippage * weights[1] + 
                         spread_slippage * weights[2])
        
        # Apply direction
        if order.side == OrderSide.BUY:
            execution_price = current_price * (1 + total_slippage)
        else:
            execution_price = current_price * (1 - total_slippage)
        
        return execution_price
    
    def update_market_data(self, symbol: str, market_data: Dict):
        """
        Update market data for slippage calculations
        
        Args:
            symbol: Instrument symbol
            market_data: Dictionary with market data
        """
        if 'volume' in market_data:
            if symbol not in self.volume_data:
                self.volume_data[symbol] = {}
            self.volume_data[symbol].update(market_data['volume'])
        
        if 'volatility' in market_data:
            if symbol not in self.volatility_data:
                self.volatility_data[symbol] = {}
            self.volatility_data[symbol].update(market_data['volatility'])
        
        if 'spread' in market_data:
            if symbol not in self.spread_data:
                self.spread_data[symbol] = {}
            self.spread_data[symbol].update(market_data['spread'])
    
    def calculate_slippage_stats(self, orders: list, execution_prices: dict) -> Dict:
        """
        Calculate slippage statistics for executed orders
        
        Args:
            orders: List of orders executed
            execution_prices: Dict of order_id -> execution_price
            
        Returns:
            Slippage statistics
        """
        if not orders:
            return {}
        
        slippages = []
        buy_slippages = []
        sell_slippages = []
        
        for order in orders:
            if order.order_id in execution_prices:
                exec_price = execution_prices[order.order_id]
                # Reference price would typically be mid-price or previous close
                # For simplicity, we'll use a placeholder
                reference_price = exec_price  # In practice, this would be different
                
                if order.side == OrderSide.BUY:
                    slippage = (exec_price - reference_price) / reference_price
                    buy_slippages.append(slippage)
                else:
                    slippage = (reference_price - exec_price) / reference_price  
                    sell_slippages.append(slippage)
                
                slippages.append(slippage)
        
        if not slippages:
            return {}
        
        stats = {
            'total_orders': len(orders),
            'avg_slippage': np.mean(slippages),
            'std_slippage': np.std(slippages),
            'max_slippage': np.max(slippages),
            'min_slippage': np.min(slippages),
            'buy_orders': len(buy_slippages),
            'sell_orders': len(sell_slippages)
        }
        
        if buy_slippages:
            stats.update({
                'avg_buy_slippage': np.mean(buy_slippages),
                'avg_sell_slippage': np.mean(sell_slippages) if sell_slippages else 0
            })
        
        return stats
    
    def get_model_info(self) -> Dict:
        """Get information about the slippage model"""
        config = self.method_configs.get(self.method, {})
        
        return {
            'method': self.method,
            'name': config.get('name', 'Unknown'),
            'description': self._get_method_description(self.method),
            'parameters': config
        }
    
    def _get_method_description(self, method: str) -> str:
        """Get description for slippage method"""
        descriptions = {
            'constant': 'Fixed percentage slippage for buy and sell orders',
            'volume_based': 'Slippage varies with order size relative to trading volume',
            'volatility_based': 'Slippage increases with market volatility',
            'spread_based': 'Slippage based on bid-ask spread',
            'hybrid': 'Combines volume, volatility, and spread factors'
        }
        return descriptions.get(method, 'Unknown method')
