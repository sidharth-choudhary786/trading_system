# trading_system/risk/compliance.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..core.types import Order, OrderSide, OrderType
from ..core.exceptions import RiskError

class ComplianceEngine:
    """
    Pre-trade compliance checks and validation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Compliance rules configuration
        self.rules = config.get('compliance_rules', {})
        self.max_position_size = self.rules.get('max_position_size', 0.1)  # 10% per stock
        self.max_sector_exposure = self.rules.get('max_sector_exposure', 0.3)  # 30% per sector
        self.max_leverage = self.rules.get('max_leverage', 1.0)  # No leverage by default
        self.restricted_symbols = self.rules.get('restricted_symbols', [])
        self.trading_hours_only = self.rules.get('trading_hours_only', True)
        
        # Market data for compliance checks
        self.sector_data = {}
        self.market_cap_data = {}
        
        self.logger.info("Compliance Engine initialized")
    
    def check_order_compliance(self, order: Order, portfolio: Dict, market_data: Dict) -> Tuple[bool, str]:
        """
        Comprehensive pre-trade compliance check
        
        Args:
            order: Order to validate
            portfolio: Current portfolio state
            market_data: Current market data
            
        Returns:
            (is_compliant, rejection_reason)
        """
        checks = [
            self._check_restricted_symbols(order),
            self._check_trading_hours(),
            self._check_position_limits(order, portfolio, market_data),
            self._check_sector_exposure(order, portfolio, market_data),
            self._check_leverage_limits(order, portfolio),
            self._check_liquidity(order, market_data),
            self._check_volatility_limits(order, market_data)
        ]
        
        for is_compliant, reason in checks:
            if not is_compliant:
                self.logger.warning(f"Order {order.order_id} failed compliance: {reason}")
                return False, reason
        
        return True, "Order compliant with all rules"
    
    def _check_restricted_symbols(self, order: Order) -> Tuple[bool, str]:
        """Check if symbol is restricted"""
        if order.symbol in self.restricted_symbols:
            return False, f"Symbol {order.symbol} is restricted from trading"
        return True, ""
    
    def _check_trading_hours(self) -> Tuple[bool, str]:
        """Check if current time is within trading hours"""
        if not self.trading_hours_only:
            return True, ""
        
        now = datetime.now()
        # Indian market hours: 9:15 AM to 3:30 PM
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # Check if weekday (Monday to Friday)
        if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False, "Trading not allowed on weekends"
        
        # Check if within market hours
        if not (market_open <= now <= market_close):
            return False, "Trading not allowed outside market hours"
        
        return True, ""
    
    def _check_position_limits(self, order: Order, portfolio: Dict, market_data: Dict) -> Tuple[bool, str]:
        """Check position size limits"""
        portfolio_value = portfolio.get('portfolio_value', 0)
        if portfolio_value <= 0:
            return True, ""  # No portfolio value, skip check
        
        # Calculate proposed position value
        current_price = market_data.get(order.symbol, {}).get('price', 0)
        if current_price <= 0:
            return False, f"No valid price data for {order.symbol}"
        
        proposed_position_value = order.quantity * current_price
        
        # Check maximum position size
        max_position_value = portfolio_value * self.max_position_size
        if proposed_position_value > max_position_value:
            return False, f"Position size {proposed_position_value:.0f} exceeds limit {max_position_value:.0f}"
        
        # Check existing positions
        current_positions = portfolio.get('positions', {})
        current_position = current_positions.get(order.symbol, 0)
        
        if order.side == OrderSide.BUY:
            new_position_value = (current_position + order.quantity) * current_price
            if new_position_value > max_position_value:
                return False, f"New position size would exceed limit"
        
        return True, ""
    
    def _check_sector_exposure(self, order: Order, portfolio: Dict, market_data: Dict) -> Tuple[bool, str]:
        """Check sector exposure limits"""
        symbol_sector = self.sector_data.get(order.symbol)
        if not symbol_sector:
            return True, ""  # No sector data, skip check
        
        portfolio_value = portfolio.get('portfolio_value', 0)
        if portfolio_value <= 0:
            return True, ""
        
        # Calculate current sector exposure
        current_positions = portfolio.get('positions', {})
        sector_exposure = {}
        
        for symbol, quantity in current_positions.items():
            sector = self.sector_data.get(symbol)
            if sector and symbol in market_data:
                price = market_data[symbol].get('price', 0)
                position_value = quantity * price
                sector_exposure[sector] = sector_exposure.get(sector, 0) + position_value
        
        # Calculate proposed new exposure
        current_price = market_data.get(order.symbol, {}).get('price', 0)
        proposed_addition = order.quantity * current_price if order.side == OrderSide.BUY else 0
        
        new_sector_exposure = sector_exposure.get(symbol_sector, 0) + proposed_addition
        max_sector_value = portfolio_value * self.max_sector_exposure
        
        if new_sector_exposure > max_sector_value:
            return False, f"Sector {symbol_sector} exposure would exceed {self.max_sector_exposure:.1%} limit"
        
        return True, ""
    
    def _check_leverage_limits(self, order: Order, portfolio: Dict) -> Tuple[bool, str]:
        """Check leverage limits"""
        portfolio_value = portfolio.get('portfolio_value', 0)
        cash = portfolio.get('cash', 0)
        
        if portfolio_value <= 0:
            return True, ""
        
        # Calculate current leverage
        invested_value = portfolio_value - cash
        current_leverage = invested_value / portfolio_value if portfolio_value > 0 else 0
        
        # For buy orders, check if new leverage would exceed limit
        if order.side == OrderSide.BUY:
            # Simplified: assume order uses cash
            order_value = order.quantity * (order.price or 0)
            if order_value > 0:
                new_invested = invested_value + order_value
                new_leverage = new_invested / (portfolio_value + order_value)
                
                if new_leverage > self.max_leverage:
                    return False, f"Leverage would exceed {self.max_leverage:.1%} limit"
        
        return True, ""
    
    def _check_liquidity(self, order: Order, market_data: Dict) -> Tuple[bool, str]:
        """Check liquidity constraints"""
        symbol_data = market_data.get(order.symbol, {})
        avg_volume = symbol_data.get('average_volume', 0)
        current_volume = symbol_data.get('volume', 0)
        
        if avg_volume > 0 and current_volume > 0:
            # Check if order size is reasonable compared to average volume
            order_to_avg_ratio = order.quantity / avg_volume
            
            if order_to_avg_ratio > 0.01:  # More than 1% of average volume
                return False, f"Order size {order.quantity} is large relative to average volume {avg_volume:.0f}"
            
            # Check if current volume is sufficient
            if current_volume < avg_volume * 0.1:  # Less than 10% of average volume
                return False, "Current trading volume is too low"
        
        return True, ""
    
    def _check_volatility_limits(self, order: Order, market_data: Dict) -> Tuple[bool, str]:
        """Check volatility-based limits"""
        symbol_data = market_data.get(order.symbol, {})
        volatility = symbol_data.get('volatility', 0)
        
        if volatility > 0.5:  # 50% annual volatility threshold
            return False, f"Volatility {volatility:.1%} exceeds safety limit"
        
        return True, ""
    
    def validate_portfolio_compliance(self, portfolio: Dict, market_data: Dict) -> Dict:
        """
        Validate entire portfolio compliance
        
        Args:
            portfolio: Current portfolio state
            market_data: Current market data
            
        Returns:
            Compliance report
        """
        report = {
            'timestamp': datetime.now(),
            'overall_compliant': True,
            'violations': [],
            'warnings': [],
            'exposure_analysis': {}
        }
        
        portfolio_value = portfolio.get('portfolio_value', 0)
        if portfolio_value <= 0:
            return report
        
        # Check position limits
        current_positions = portfolio.get('positions', {})
        for symbol, quantity in current_positions.items():
            if symbol in market_data:
                price = market_data[symbol].get('price', 0)
                position_value = quantity * price
                max_allowed = portfolio_value * self.max_position_size
                
                if position_value > max_allowed:
                    report['violations'].append(
                        f"Position {symbol} exceeds size limit: {position_value:.0f} > {max_allowed:.0f}"
                    )
                    report['overall_compliant'] = False
        
        # Check sector exposure
        sector_exposure = self._calculate_sector_exposure(portfolio, market_data)
        report['exposure_analysis']['sectors'] = sector_exposure
        
        for sector, exposure in sector_exposure.items():
            exposure_pct = exposure / portfolio_value
            max_sector_pct = self.max_sector_exposure
            
            if exposure_pct > max_sector_pct:
                report['violations'].append(
                    f"Sector {sector} exposure {exposure_pct:.1%} exceeds limit {max_sector_pct:.1%}"
                )
                report['overall_compliant'] = False
            elif exposure_pct > max_sector_pct * 0.8:  # Warning at 80% of limit
                report['warnings'].append(
                    f"Sector {sector} exposure {exposure_pct:.1%} approaching limit {max_sector_pct:.1%}"
                )
        
        # Check leverage
        cash = portfolio.get('cash', 0)
        invested_value = portfolio_value - cash
        leverage = invested_value / portfolio_value
        
        if leverage > self.max_leverage:
            report['violations'].append(
                f"Leverage {leverage:.1%} exceeds limit {self.max_leverage:.1%}"
            )
            report['overall_compliant'] = False
        
        return report
    
    def _calculate_sector_exposure(self, portfolio: Dict, market_data: Dict) -> Dict:
        """Calculate current sector exposure"""
        sector_exposure = {}
        current_positions = portfolio.get('positions', {})
        
        for symbol, quantity in current_positions.items():
            sector = self.sector_data.get(symbol)
            if sector and symbol in market_data:
                price = market_data[symbol].get('price', 0)
                position_value = quantity * price
                sector_exposure[sector] = sector_exposure.get(sector, 0) + position_value
        
        return sector_exposure
    
    def update_sector_data(self, sector_data: Dict):
        """Update sector classification data"""
        self.sector_data.update(sector_data)
    
    def update_market_cap_data(self, market_cap_data: Dict):
        """Update market capitalization data"""
        self.market_cap_data.update(market_cap_data)
    
    def get_compliance_rules(self) -> Dict:
        """Get current compliance rules"""
        return {
            'max_position_size': self.max_position_size,
            'max_sector_exposure': self.max_sector_exposure,
            'max_leverage': self.max_leverage,
            'restricted_symbols': self.restricted_symbols,
            'trading_hours_only': self.trading_hours_only
        }
