# trading_system/portfolio/allocator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

from ..core.types import Order, OrderSide
from ..core.exceptions import TradingSystemError

class CapitalAllocator:
    """
    Handles dynamic capital allocation and position sizing
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Allocation configuration
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% per stock
        self.max_sector_exposure = config.get('max_sector_exposure', 0.3)  # 30% per sector
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade
        self.min_position_value = config.get('min_position_value', 1000)  # Minimum position size
        
        # Sizing methods
        self.sizing_method = config.get('sizing_method', 'risk_based')  # risk_based, equal_weight, volatility_adjusted
    
    def calculate_position_size(
        self,
        symbol: str,
        target_weight: float,
        portfolio_value: float,
        current_prices: Dict[str, float],
        risk_data: Optional[Dict] = None
    ) -> Tuple[float, float]:
        """
        Calculate position size considering constraints and risk
        
        Args:
            symbol: Instrument symbol
            target_weight: Target portfolio weight
            portfolio_value: Current portfolio value
            current_prices: Current market prices
            risk_data: Optional risk metrics
            
        Returns:
            (position_value, quantity)
        """
        # Basic position value calculation
        raw_position_value = target_weight * portfolio_value
        
        # Apply position size constraints
        constrained_value = self._apply_position_constraints(
            symbol, raw_position_value, portfolio_value, risk_data
        )
        
        # Get current price
        if symbol not in current_prices:
            raise TradingSystemError(f"No price data for {symbol}")
        
        current_price = current_prices[symbol]
        
        # Calculate quantity
        quantity = self._calculate_quantity(constrained_value, current_price)
        
        # Adjust for minimum position value
        final_position_value = quantity * current_price
        if final_position_value < self.min_position_value:
            self.logger.warning(f"Position value {final_position_value:.2f} below minimum for {symbol}")
            return 0.0, 0.0
        
        self.logger.debug(f"Position size for {symbol}: {quantity} shares (${final_position_value:.2f})")
        return final_position_value, quantity
    
    def _apply_position_constraints(
        self,
        symbol: str,
        position_value: float,
        portfolio_value: float,
        risk_data: Optional[Dict]
    ) -> float:
        """Apply position sizing constraints"""
        constrained_value = position_value
        
        # Maximum position size constraint
        max_position_value = self.max_position_size * portfolio_value
        constrained_value = min(constrained_value, max_position_value)
        
        # Risk-based sizing
        if self.sizing_method == 'risk_based' and risk_data:
            constrained_value = self._risk_based_sizing(constrained_value, risk_data)
        
        # Volatility-adjusted sizing
        elif self.sizing_method == 'volatility_adjusted' and risk_data:
            constrained_value = self._volatility_adjusted_sizing(constrained_value, risk_data)
        
        return constrained_value
    
    def _risk_based_sizing(self, position_value: float, risk_data: Dict) -> float:
        """Risk-based position sizing"""
        if 'volatility' not in risk_data:
            return position_value
        
        volatility = risk_data['volatility']
        risk_budget = self.risk_per_trade * position_value
        
        # Adjust position size based on volatility
        # Higher volatility = smaller position size
        volatility_factor = 0.02 / max(volatility, 0.01)  # Normalize to 2% volatility
        volatility_factor = min(volatility_factor, 2.0)  # Cap at 2x
        
        risk_adjusted_value = position_value * volatility_factor
        return min(risk_adjusted_value, position_value * 2)  # Cap at 2x original
    
    def _volatility_adjusted_sizing(self, position_value: float, risk_data: Dict) -> float:
        """Volatility-adjusted position sizing"""
        if 'volatility' not in risk_data:
            return position_value
        
        volatility = risk_data['volatility']
        
        # Inverse volatility weighting
        # Lower volatility = larger position size
        if volatility > 0:
            volatility_weight = 1.0 / volatility
            # Normalize (assuming average volatility around 0.2)
            normalized_weight = volatility_weight * 0.2
            adjusted_value = position_value * min(normalized_weight, 2.0)
            return adjusted_value
        
        return position_value
    
    def _calculate_quantity(self, position_value: float, current_price: float) -> int:
        """Calculate integer quantity from position value"""
        if current_price <= 0:
            return 0
        
        raw_quantity = position_value / current_price
        
        # Round down to integer shares
        quantity = int(raw_quantity)
        
        # Ensure minimum 1 share if position value is significant
        if quantity == 0 and position_value >= current_price:
            quantity = 1
        
        return quantity
    
    def generate_orders_from_weights(
        self,
        target_weights: Dict[str, float],
        current_positions: Dict[str, float],
        portfolio_value: float,
        current_prices: Dict[str, float],
        risk_data: Optional[Dict[str, Dict]] = None
    ) -> List[Order]:
        """
        Generate orders to achieve target weights
        
        Args:
            target_weights: Dictionary of symbol -> target weight
            current_positions: Dictionary of symbol -> current quantity
            portfolio_value: Current portfolio value
            current_prices: Current market prices
            risk_data: Risk data by symbol
            
        Returns:
            List of orders to execute
        """
        orders = []
        
        for symbol, target_weight in target_weights.items():
            if symbol not in current_prices:
                self.logger.warning(f"No price data for {symbol}, skipping")
                continue
            
            # Calculate target position
            target_value = target_weight * portfolio_value
            current_quantity = current_positions.get(symbol, 0)
            current_value = current_quantity * current_prices[symbol]
            
            # Calculate position size with constraints
            symbol_risk_data = risk_data.get(symbol, {}) if risk_data else {}
            position_value, target_quantity = self.calculate_position_size(
                symbol, target_weight, portfolio_value, current_prices, symbol_risk_data
            )
            
            # Determine order side and quantity
            quantity_diff = target_quantity - current_quantity
            
            if abs(quantity_diff) < 1:  # Ignore small differences
                continue
            
            if quantity_diff > 0:
                # Buy order
                order = Order(
                    order_id=f"BUY_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY,
                    quantity=quantity_diff,
                    timestamp=datetime.now()
                )
                orders.append(order)
                
            elif quantity_diff < 0:
                # Sell order
                order = Order(
                    order_id=f"SELL_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL,
                    quantity=abs(quantity_diff),
                    timestamp=datetime.now()
                )
                orders.append(order)
        
        self.logger.info(f"Generated {len(orders)} orders from target weights")
        return orders
    
    def calculate_sector_allocation(
        self,
        positions: Dict[str, float],
        sector_mapping: Dict[str, str],
        current_prices: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculate current sector allocation
        
        Args:
            positions: Current positions
            sector_mapping: Symbol -> sector mapping
            current_prices: Current prices
            portfolio_value: Portfolio value
            
        Returns:
            Dictionary of sector -> allocation percentage
        """
        sector_values = {}
        
        for symbol, quantity in positions.items():
            if symbol not in current_prices or symbol not in sector_mapping:
                continue
            
            sector = sector_mapping[symbol]
            position_value = quantity * current_prices[symbol]
            
            if sector not in sector_values:
                sector_values[sector] = 0.0
            
            sector_values[sector] += position_value
        
        # Convert to percentages
        sector_allocations = {}
        for sector, value in sector_values.items():
            sector_allocations[sector] = value / portfolio_value
        
        return sector_allocations
    
    def check_sector_constraints(
        self,
        target_weights: Dict[str, float],
        sector_mapping: Dict[str, str],
        portfolio_value: float
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if target weights violate sector constraints
        
        Args:
            target_weights: Target weights by symbol
            sector_mapping: Symbol -> sector mapping
            portfolio_value: Portfolio value
            
        Returns:
            (is_valid, sector_allocations)
        """
        sector_values = {}
        
        for symbol, weight in target_weights.items():
            if symbol not in sector_mapping:
                continue
            
            sector = sector_mapping[symbol]
            position_value = weight * portfolio_value
            
            if sector not in sector_values:
                sector_values[sector] = 0.0
            
            sector_values[sector] += position_value
        
        # Calculate sector allocations
        sector_allocations = {}
        for sector, value in sector_values.items():
            allocation = value / portfolio_value
            sector_allocations[sector] = allocation
            
            # Check constraint
            if allocation > self.max_sector_exposure:
                self.logger.warning(f"Sector {sector} allocation {allocation:.2%} exceeds limit {self.max_sector_exposure:.2%}")
                return False, sector_allocations
        
        return True, sector_allocations
    
    def get_allocation_stats(
        self,
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float
    ) -> Dict:
        """Get allocation statistics"""
        stats = {
            'total_positions': len(positions),
            'position_breakdown': {},
            'concentration_metrics': {}
        }
        
        # Calculate individual position allocations
        position_allocations = {}
        for symbol, quantity in positions.items():
            if symbol in current_prices:
                value = quantity * current_prices[symbol]
                allocation = value / portfolio_value
                position_allocations[symbol] = {
                    'quantity': quantity,
                    'value': value,
                    'allocation': allocation
                }
        
        stats['position_breakdown'] = position_allocations
        
        # Concentration metrics
        if position_allocations:
            allocations = list(position_allocations.values())
            allocations.sort(key=lambda x: x['value'], reverse=True)
            
            # Top 5 concentration
            top_5_value = sum(x['value'] for x in allocations[:5])
            stats['concentration_metrics']['top_5_allocation'] = top_5_value / portfolio_value
            
            # Herfindahl index (concentration measure)
            herfindahl = sum((x['allocation'] ** 2 for x in allocations))
            stats['concentration_metrics']['herfindahl_index'] = herfindahl
            
            # Effective number of positions
            stats['concentration_metrics']['effective_positions'] = 1 / herfindahl if herfindahl > 0 else 0
        
        return stats
