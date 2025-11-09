# trading_system/portfolio/risk_manager.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

from ..core.types import Order, Trade, OrderSide
from ..core.exceptions import RiskError

class RiskManager:
    """
    Comprehensive risk management engine with real-time monitoring
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk limits
        self.max_daily_loss = config.get('max_daily_loss', 0.02)  # 2% daily loss limit
        self.max_drawdown = config.get('max_drawdown', 0.15)  # 15% maximum drawdown
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% per position
        self.max_sector_exposure = config.get('max_sector_exposure', 0.3)  # 30% per sector
        self.var_confidence = config.get('var_confidence', 0.95)  # 95% VaR confidence
        
        # Circuit breaker settings
        self.circuit_breakers_enabled = config.get('enable_circuit_breakers', True)
        self.emergency_stop_loss = config.get('emergency_stop_loss', 0.05)  # 5% emergency stop
        self.cooldown_period = config.get('cooldown_period', 1)  # 1 day cooldown
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.max_portfolio_value = 0.0
        self.risk_violations = []
        self.circuit_breaker_triggered = False
        self.last_violation_date = None
        
        self.logger.info("Risk Manager initialized")
    
    def check_order(
        self, 
        order: Order, 
        portfolio: 'Portfolio', 
        current_prices: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Pre-trade risk check for an order
        
        Args:
            order: Order to check
            portfolio: Current portfolio state
            current_prices: Current market prices
            
        Returns:
            (is_approved, rejection_reason)
        """
        # Check if circuit breaker is active
        if self.circuit_breaker_triggered:
            return False, "Circuit breaker active - trading suspended"
        
        # Check daily loss limit
        if self._check_daily_loss_limit(portfolio):
            return False, "Daily loss limit exceeded"
        
        # Check maximum drawdown
        if self._check_max_drawdown(portfolio):
            return False, "Maximum drawdown exceeded"
        
        # Check position size limits
        if not self._check_position_size(order, portfolio, current_prices):
            return False, "Position size limit exceeded"
        
        # Check sector exposure limits
        if not self._check_sector_exposure(order, portfolio, current_prices):
            return False, "Sector exposure limit exceeded"
        
        # Check liquidity constraints
        if not self._check_liquidity(order, current_prices):
            return False, "Liquidity constraints not met"
        
        return True, "Order approved"
    
    def _check_daily_loss_limit(self, portfolio: 'Portfolio') -> bool:
        """Check if daily loss limit is exceeded"""
        if hasattr(portfolio, 'daily_pnl'):
            daily_loss_pct = abs(min(0, portfolio.daily_pnl)) / portfolio.initial_capital
            if daily_loss_pct > self.max_daily_loss:
                self._record_violation("DAILY_LOSS_LIMIT", f"Daily loss: {daily_loss_pct:.2%}")
                return True
        return False
    
    def _check_max_drawdown(self, portfolio: 'Portfolio') -> bool:
        """Check if maximum drawdown is exceeded"""
        if hasattr(portfolio, 'portfolio_value') and hasattr(portfolio, 'max_portfolio_value'):
            if portfolio.portfolio_value > portfolio.max_portfolio_value:
                portfolio.max_portfolio_value = portfolio.portfolio_value
            
            drawdown = (portfolio.max_portfolio_value - portfolio.portfolio_value) / portfolio.max_portfolio_value
            if drawdown > self.max_drawdown:
                self._record_violation("MAX_DRAWDOWN", f"Drawdown: {drawdown:.2%}")
                return True
        return False
    
    def _check_position_size(
        self, 
        order: Order, 
        portfolio: 'Portfolio', 
        current_prices: Dict[str, float]
    ) -> bool:
        """Check position size limits"""
        if order.symbol not in current_prices:
            return True  # Can't check without price
        
        # Calculate proposed position value
        current_position = portfolio.positions.get(order.symbol, 0)
        if order.side == OrderSide.BUY:
            proposed_quantity = current_position + order.quantity
        else:
            proposed_quantity = current_position - order.quantity
        
        proposed_value = abs(proposed_quantity) * current_prices[order.symbol]
        position_size = proposed_value / portfolio.portfolio_value
        
        if position_size > self.max_position_size:
            self._record_violation(
                "POSITION_SIZE", 
                f"Position {order.symbol} would be {position_size:.2%}"
            )
            return False
        
        return True
    
    def _check_sector_exposure(
        self, 
        order: Order, 
        portfolio: 'Portfolio', 
        current_prices: Dict[str, float]
    ) -> bool:
        """Check sector exposure limits"""
        # This would require sector mapping data
        # For now, return True (implementation would need sector data)
        return True
    
    def _check_liquidity(self, order: Order, current_prices: Dict[str, float]) -> bool:
        """Check liquidity constraints"""
        # Basic liquidity check - order size relative to typical volume
        # In practice, this would use volume data
        order_value = order.quantity * current_prices.get(order.symbol, 0)
        
        # Simple check: reject very large orders (> $1M) without additional logic
        if order_value > 1000000:  # $1M
            self.logger.warning(f"Large order detected: ${order_value:,.2f}")
            # Would typically check against average daily volume
        
        return True
    
    def calculate_portfolio_risk(
        self, 
        portfolio: 'Portfolio', 
        historical_returns: pd.DataFrame,
        current_prices: Dict[str, float]
    ) -> Dict:
        """
        Calculate comprehensive portfolio risk metrics
        """
        risk_metrics = {}
        
        # Value at Risk (VaR)
        risk_metrics['var'] = self._calculate_var(portfolio, historical_returns)
        
        # Conditional VaR (Expected Shortfall)
        risk_metrics['cvar'] = self._calculate_cvar(portfolio, historical_returns)
        
        # Portfolio volatility
        risk_metrics['volatility'] = self._calculate_portfolio_volatility(portfolio, historical_returns)
        
        # Beta (market exposure)
        risk_metrics['beta'] = self._calculate_beta(portfolio, historical_returns)
        
        # Drawdown analysis
        risk_metrics['drawdown'] = self._calculate_current_drawdown(portfolio)
        
        # Concentration risk
        risk_metrics['concentration'] = self._calculate_concentration_risk(portfolio, current_prices)
        
        # Liquidity risk
        risk_metrics['liquidity'] = self._calculate_liquidity_risk(portfolio, current_prices)
        
        return risk_metrics
    
    def _calculate_var(
        self, 
        portfolio: 'Portfolio', 
        historical_returns: pd.DataFrame
    ) -> float:
        """Calculate Value at Risk"""
        if historical_returns.empty:
            return 0.0
        
        # Simple historical VaR
        portfolio_returns = self._calculate_portfolio_returns(portfolio, historical_returns)
        if portfolio_returns.empty:
            return 0.0
        
        var = np.percentile(portfolio_returns, (1 - self.var_confidence) * 100)
        return abs(var)
    
    def _calculate_cvar(
        self, 
        portfolio: 'Portfolio', 
        historical_returns: pd.DataFrame
    ) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if historical_returns.empty:
            return 0.0
        
        portfolio_returns = self._calculate_portfolio_returns(portfolio, historical_returns)
        if portfolio_returns.empty:
            return 0.0
        
        var_threshold = np.percentile(portfolio_returns, (1 - self.var_confidence) * 100)
        tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
        
        if len(tail_returns) > 0:
            cvar = tail_returns.mean()
            return abs(cvar)
        
        return 0.0
    
    def _calculate_portfolio_returns(
        self, 
        portfolio: 'Portfolio', 
        historical_returns: pd.DataFrame
    ) -> pd.Series:
        """Calculate historical portfolio returns"""
        # This would require the portfolio's historical weights
        # Simplified implementation
        if hasattr(portfolio, 'returns_history'):
            return portfolio.returns_history
        return pd.Series()
    
    def _calculate_portfolio_volatility(
        self, 
        portfolio: 'Portfolio', 
        historical_returns: pd.DataFrame
    ) -> float:
        """Calculate portfolio volatility"""
        portfolio_returns = self._calculate_portfolio_returns(portfolio, historical_returns)
        if len(portfolio_returns) > 1:
            return portfolio_returns.std() * np.sqrt(252)  # Annualized
        return 0.0
    
    def _calculate_beta(
        self, 
        portfolio: 'Portfolio', 
        historical_returns: pd.DataFrame
    ) -> float:
        """Calculate portfolio beta relative to market"""
        # Simplified - would need market returns data
        return 1.0  # Placeholder
    
    def _calculate_current_drawdown(self, portfolio: 'Portfolio') -> float:
        """Calculate current drawdown from peak"""
        if (hasattr(portfolio, 'portfolio_value') and 
            hasattr(portfolio, 'max_portfolio_value') and
            portfolio.max_portfolio_value > 0):
            
            drawdown = (portfolio.max_portfolio_value - portfolio.portfolio_value) / portfolio.max_portfolio_value
            return max(0.0, drawdown)
        
        return 0.0
    
    def _calculate_concentration_risk(
        self, 
        portfolio: 'Portfolio', 
        current_prices: Dict[str, float]
    ) -> Dict:
        """Calculate concentration risk metrics"""
        if not hasattr(portfolio, 'positions'):
            return {}
        
        position_values = {}
        total_value = 0.0
        
        for symbol, quantity in portfolio.positions.items():
            if symbol in current_prices:
                value = quantity * current_prices[symbol]
                position_values[symbol] = value
                total_value += value
        
        if total_value == 0:
            return {}
        
        # Herfindahl-Hirschman Index (HHI)
        hhi = sum((value / total_value) ** 2 for value in position_values.values())
        
        # Top positions concentration
        sorted_positions = sorted(position_values.values(), reverse=True)
        top_5_concentration = sum(sorted_positions[:5]) / total_value if len(sorted_positions) >= 5 else 1.0
        
        return {
            'hhi': hhi,
            'top_5_concentration': top_5_concentration,
            'effective_n': 1 / hhi if hhi > 0 else 0,
            'max_single_position': max(position_values.values()) / total_value if position_values else 0
        }
    
    def _calculate_liquidity_risk(
        self, 
        portfolio: 'Portfolio', 
        current_prices: Dict[str, float]
    ) -> Dict:
        """Calculate liquidity risk metrics"""
        # Simplified liquidity risk assessment
        # In practice, would use volume data, bid-ask spreads, etc.
        return {
            'liquidity_score': 0.8,  # Placeholder
            'estimated_slippage': 0.001,  # 0.1% estimated slippage
            'days_to_liquidate': 1.0  # Estimated days to liquidate portfolio
        }
    
    def trigger_circuit_breaker(self, reason: str):
        """Trigger circuit breaker to stop trading"""
        if self.circuit_breakers_enabled and not self.circuit_breaker_triggered:
            self.circuit_breaker_triggered = True
            self.last_violation_date = datetime.now()
            self._record_violation("CIRCUIT_BREAKER", reason)
            self.logger.critical(f"Circuit breaker triggered: {reason}")
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker after cooldown period"""
        if (self.circuit_breaker_triggered and 
            self.last_violation_date and
            datetime.now() >= self.last_violation_date + timedelta(days=self.cooldown_period)):
            
            self.circuit_breaker_triggered = False
            self.logger.info("Circuit breaker reset")
    
    def _record_violation(self, violation_type: str, details: str):
        """Record risk violation"""
        violation = {
            'timestamp': datetime.now(),
            'type': violation_type,
            'details': details,
            'portfolio_value': getattr(self, 'portfolio_value', 0)
        }
        self.risk_violations.append(violation)
        
        self.logger.warning(f"Risk violation: {violation_type} - {details}")
    
    def get_risk_report(self, portfolio: 'Portfolio') -> Dict:
        """Generate comprehensive risk report"""
        return {
            'timestamp': datetime.now(),
            'portfolio_value': getattr(portfolio, 'portfolio_value', 0),
            'risk_limits': {
                'max_daily_loss': self.max_daily_loss,
                'max_drawdown': self.max_drawdown,
                'max_position_size': self.max_position_size,
                'max_sector_exposure': self.max_sector_exposure
            },
            'current_metrics': {
                'daily_pnl': getattr(portfolio, 'daily_pnl', 0),
                'drawdown': self._calculate_current_drawdown(portfolio),
                'circuit_breaker_active': self.circuit_breaker_triggered
            },
            'violations': self.risk_violations[-10:],  # Last 10 violations
            'recommendations': self._generate_risk_recommendations(portfolio)
        }
    
    def _generate_risk_recommendations(self, portfolio: 'Portfolio') -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Check drawdown
        drawdown = self._calculate_current_drawdown(portfolio)
        if drawdown > self.max_drawdown * 0.8:  # 80% of max drawdown
            recommendations.append("Consider reducing position sizes due to high drawdown")
        
        # Check concentration
        if hasattr(portfolio, 'positions') and len(portfolio.positions) < 10:
            recommendations.append("Consider diversifying portfolio with more positions")
        
        # Check circuit breaker
        if self.circuit_breaker_triggered:
            recommendations.append("Trading suspended due to risk limits. Review positions.")
        
        return recommendations
