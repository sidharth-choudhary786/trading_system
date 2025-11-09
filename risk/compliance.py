# trading_system/risk/compliance.py
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..core.exceptions import RiskError
from ..core.types import Order, OrderSide, OrderType

class ComplianceEngine:
    """
    Pre-trade and post-trade compliance checking
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Compliance rules configuration
        self.compliance_rules = {
            'position_limits': {
                'enabled': config.get('position_limits_enabled', True),
                'max_single_position': config.get('max_single_position', 0.1),  # 10%
                'max_sector_exposure': config.get('max_sector_exposure', 0.3),  # 30%
                'max_leverage': config.get('max_leverage', 1.0)  # 100%
            },
            'concentration_limits': {
                'enabled': config.get('concentration_limits_enabled', True),
                'max_top_5_concentration': config.get('max_top_5_concentration', 0.6),  # 60%
                'max_single_sector': config.get('max_single_sector', 0.4)  # 40%
            },
            'trading_limits': {
                'enabled': config.get('trading_limits_enabled', True),
                'max_daily_trades': config.get('max_daily_trades', 50),
                'max_order_size': config.get('max_order_size', 1000000),  # ₹10L
                'min_holding_period': config.get('min_holding_period', 1)  # days
            },
            'regulatory_limits': {
                'enabled': config.get('regulatory_limits_enabled', True),
                'sebi_limits': config.get('sebi_limits', {}),
                'prevent_wash_sales': config.get('prevent_wash_sales', True),
                'prevent_insider_trading': config.get('prevent_insider_trading', True)
            }
        }
        
        # Trading history for compliance checks
        self.trade_history = []
        self.order_history = []
        self.holdings_history = []
        
        # Market data for compliance
        self.market_data = {}
        self.sector_data = {}
        
        # Integration with other risk modules
        self.position_limit_manager = None  # Would be initialized with PositionLimitManager
        self.circuit_breaker = None  # Would be initialized with CircuitBreaker
        
        self.logger.info("Compliance Engine initialized")
    
    def pre_trade_compliance_check(self, order: Order, portfolio: Dict, market_data: Dict) -> Tuple[bool, str]:
        """
        Pre-trade compliance check for an order
        
        Returns:
            (is_compliant, rejection_reason)
        """
        checks = [
            self._check_position_limits(order, portfolio),
            self._check_concentration_limits(order, portfolio),
            self._check_trading_limits(order),
            self._check_regulatory_limits(order, portfolio),
            self._check_market_conditions(order, market_data)
        ]
        
        for is_compliant, reason in checks:
            if not is_compliant:
                self.logger.warning(f"Pre-trade compliance failed: {reason}")
                return False, reason
        
        return True, "Order compliant"
    
    def post_trade_compliance_check(self, trade: Dict, portfolio: Dict) -> Tuple[bool, str]:
        """
        Post-trade compliance check
        """
        checks = [
            self._check_holdings_concentration(portfolio),
            self._check_leverage_limits(portfolio),
            self._check_wash_sale(trade, portfolio),
            self._check_holding_period(trade)
        ]
        
        for is_compliant, reason in checks:
            if not is_compliant:
                self.logger.warning(f"Post-trade compliance failed: {reason}")
                return False, reason
        
        return True, "Trade compliant"
