# trading_system/risk/risk_engine.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..core.types import Order, Trade, OrderSide
from ..core.exceptions import RiskError
from .compliance import ComplianceEngine
from .circuit_breakers import CircuitBreaker
from .position_limits import PositionLimitManager

class RiskEngine:
    """
    Comprehensive risk management engine
    Real-time risk monitoring and control
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize risk components
        self.compliance_engine = ComplianceEngine(config)
        self.circuit_breaker = CircuitBreaker(config)
        self.position_limits = PositionLimitManager(config)
        
        # Risk thresholds
        self.risk_config = config.get('risk', {})
        self.max_daily_loss = self.risk_config.get('max_daily_loss', 0.02)  # 2%
        self.max_drawdown = self.risk_config.get('max_drawdown', 0.15)      # 15%
        self.var_confidence = self.risk_config.get('var_confidence', 0.95)  # 95%
        
        # Risk metrics tracking
        self.portfolio_history = []
        self.daily_pnl = {}
        self.risk_metrics = {}
        self.violations = []
        
        self.logger.info("Risk Engine initialized")
    
    def pre_trade_check(self, order: Order, portfolio: Dict, market_data: Dict) -> Tuple[bool, str]:
        """
        Comprehensive pre-trade risk check
        
        Args:
            order: Order to check
            portfolio: Current portfolio state
            market_data: Current market data
            
        Returns:
            (is_approved, reason)
        """
        checks = []
        
        # 1. Compliance check
        compliant, compliance_reason = self.compliance_engine.check_order(order, portfolio)
        checks.append((compliant, f"Compliance: {compliance_reason}"))
        
        # 2. Position limit check
        within_limits, limit_reason = self.position_limits.check_position(order, portfolio)
        checks.append((within_limits, f"Position Limits: {limit_reason}"))
        
        # 3. Circuit breaker check
        circuit_ok, circuit_reason = self.circuit_breaker.check_circuit_breakers(portfolio)
        checks.append((circuit_ok, f"Circuit Breakers: {circuit_reason}"))
        
        # 4. Concentration risk check
        concentration_ok, concentration_reason = self._check_concentration_risk(order, portfolio)
        checks.append((concentration_ok, f"Concentration: {concentration_reason}"))
        
        # 5. Liquidity risk check
        liquidity_ok, liquidity_reason = self._check_liquidity_risk(order, market_data)
        checks.append((liquidity_ok, f"Liquidity: {liquidity_reason}"))
        
        # 6. VaR check
        var_ok, var_reason = self._check_var_limit(order, portfolio, market_data)
        checks.append((var_ok, f"VaR: {var_reason}"))
        
        # Check all conditions
        all_checks_passed = all(check[0] for check in checks)
        
        if not all_checks_passed:
            failed_checks = [reason for passed, reason in checks if not passed]
            reason = " | ".join(failed_checks)
            
            # Log violation
            self.violations.append({
                'timestamp': datetime.now(),
                'order_id': order.order_id,
                'symbol': order.symbol,
                'reason': reason,
                'order_details': {
                    'side': order.side.value,
                    'quantity': order.quantity,
                    'price': order.price
                }
            })
            
            return False, reason
        
        return True, "All risk checks passed"
    
    def post_trade_monitoring(self, trade: Trade, portfolio: Dict, market_data: Dict):
        """
        Post-trade risk monitoring and updates
        """
        # Update position limits
        self.position_limits.update_positions(trade, portfolio)
        
        # Update circuit breakers
        self.circuit_breaker.update_metrics(trade, portfolio)
        
        # Update risk metrics
        self._update_risk_metrics(portfolio, market_data)
        
        # Check for risk threshold breaches
        self._check_risk_thresholds(portfolio)
    
    def _check_concentration_risk(self, order: Order, portfolio: Dict) -> Tuple[bool, str]:
        """Check concentration risk"""
        symbol = order.symbol
        
        # Get current portfolio value
        portfolio_value = portfolio.get('portfolio_value', 0)
        if portfolio_value <= 0:
            return True, "No portfolio value"
        
        # Calculate order value
        order_value = order.quantity * (order.price or portfolio.get('current_prices', {}).get(symbol, 0))
        if order_value <= 0:
            return True, "Zero order value"
        
        # Single stock concentration
        single_stock_limit = self.risk_config.get('max_single_stock', 0.1)  # 10%
        order_weight = order_value / portfolio_value
        
        if order_weight > single_stock_limit:
            return False, f"Single stock concentration {order_weight:.2%} > {single_stock_limit:.2%}"
        
        # Sector concentration (simplified)
        sector_limit = self.risk_config.get('max_sector', 0.3)  # 30%
        # In practice, you'd map symbol to sector and calculate sector exposure
        
        return True, "Concentration OK"
    
    def _check_liquidity_risk(self, order: Order, market_data: Dict) -> Tuple[bool, str]:
        """Check liquidity risk"""
        symbol = order.symbol
        
        if symbol not in market_data:
            return True, "No market data"
        
        # Get average daily volume
        avg_volume = market_data[symbol].get('avg_daily_volume', 0)
        if avg_volume <= 0:
            return True, "No volume data"
        
        # Check if order size is too large relative to average volume
        volume_ratio = order.quantity / avg_volume
        volume_threshold = self.risk_config.get('max_volume_ratio', 0.1)  # 10%
        
        if volume_ratio > volume_threshold:
            return False, f"Order size {volume_ratio:.2%} of average volume > {volume_threshold:.2%}"
        
        return True, "Liquidity OK"
    
    def _check_var_limit(self, order: Order, portfolio: Dict, market_data: Dict) -> Tuple[bool, str]:
        """Check Value at Risk limits"""
        # Calculate portfolio VaR
        portfolio_var = self.calculate_portfolio_var(portfolio, market_data)
        var_limit = self.risk_config.get('var_limit', 0.05)  # 5% of portfolio
        
        if portfolio_var > var_limit:
            return False, f"Portfolio VaR {portfolio_var:.2%} > limit {var_limit:.2%}"
        
        return True, f"VaR {portfolio_var:.2%} within limits"
    
    def calculate_portfolio_var(self, portfolio: Dict, market_data: Dict, horizon: int = 1) -> float:
        """
        Calculate portfolio Value at Risk
        """
        try:
            # Simplified VaR calculation using historical method
            positions = portfolio.get('positions', {})
            if not positions:
                return 0.0
            
            # Get historical returns for all positions
            portfolio_returns = []
            
            for symbol, quantity in positions.items():
                if symbol in market_data and 'returns' in market_data[symbol]:
                    returns = market_data[symbol]['returns']
                    position_returns = returns * (quantity / portfolio.get('portfolio_value', 1))
                    portfolio_returns.append(position_returns)
            
            if not portfolio_returns:
                return 0.0
            
            # Calculate portfolio returns (simplified - no correlation)
            if len(portfolio_returns) == 1:
                total_returns = portfolio_returns[0]
            else:
                total_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)
            
            # Calculate VaR
            var = np.percentile(total_returns.dropna(), (1 - self.var_confidence) * 100)
            return abs(var) * np.sqrt(horizon)
            
        except Exception as e:
            self.logger.warning(f"VaR calculation failed: {e}")
            return 0.0
    
    def _update_risk_metrics(self, portfolio: Dict, market_data: Dict):
        """Update risk metrics"""
        current_time = datetime.now()
        
        # Portfolio metrics
        portfolio_value = portfolio.get('portfolio_value', 0)
        self.portfolio_history.append({
            'timestamp': current_time,
            'portfolio_value': portfolio_value,
            'cash': portfolio.get('cash', 0),
            'positions': portfolio.get('positions', {}).copy()
        })
        
        # Keep only recent history (e.g., 30 days)
        cutoff_time = current_time - timedelta(days=30)
        self.portfolio_history = [
            h for h in self.portfolio_history 
            if h['timestamp'] > cutoff_time
        ]
        
        # Calculate current risk metrics
        self.risk_metrics = {
            'timestamp': current_time,
            'portfolio_value': portfolio_value,
            'var_95': self.calculate_portfolio_var(portfolio, market_data),
            'expected_shortfall': self.calculate_expected_shortfall(portfolio, market_data),
            'max_drawdown': self.calculate_max_drawdown(),
            'volatility': self.calculate_portfolio_volatility(),
            'beta': self.calculate_portfolio_beta(portfolio, market_data),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'position_concentration': self.calculate_position_concentration(portfolio)
        }
    
    def calculate_expected_shortfall(self, portfolio: Dict, market_data: Dict) -> float:
        """Calculate Expected Shortfall/CVaR"""
        # Simplified implementation
        var = self.calculate_portfolio_var(portfolio, market_data)
        return var * 1.3  # CVaR is typically higher than VaR
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio history"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        if len(returns) < 2:
            return 0.0
        
        return returns.std() * np.sqrt(252)  # Annualized
    
    def calculate_portfolio_beta(self, portfolio: Dict, market_data: Dict) -> float:
        """Calculate portfolio beta to market"""
        # Simplified beta calculation
        # In practice, you'd need benchmark returns
        return 1.0  # Placeholder
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)
    
    def calculate_position_concentration(self, portfolio: Dict) -> Dict:
        """Calculate position concentration metrics"""
        positions = portfolio.get('positions', {})
        portfolio_value = portfolio.get('portfolio_value', 0)
        
        if portfolio_value <= 0:
            return {}
        
        position_values = {}
        for symbol, quantity in positions.items():
            # Simplified - would need current prices
            position_values[symbol] = quantity  # Placeholder
        
        total_position_value = sum(position_values.values())
        if total_position_value <= 0:
            return {}
        
        concentrations = {}
        for symbol, value in position_values.items():
            concentrations[symbol] = value / total_position_value
        
        # Top positions concentration
        sorted_concentrations = sorted(concentrations.values(), reverse=True)
        top_5_concentration = sum(sorted_concentrations[:5])
        
        return {
            'top_position': max(concentrations.values()) if concentrations else 0,
            'top_5_concentration': top_5_concentration,
            'herfindahl_index': sum(c ** 2 for c in concentrations.values())
        }
    
    def _check_risk_thresholds(self, portfolio: Dict):
        """Check for risk threshold breaches"""
        current_pnl = portfolio.get('daily_pnl', 0)
        portfolio_value = portfolio.get('portfolio_value', 0)
        
        # Daily loss check
        if portfolio_value > 0:
            daily_return = current_pnl / portfolio_value
            if daily_return < -self.max_daily_loss:
                self.circuit_breaker.trigger_circuit_breaker(
                    'daily_loss_breach',
                    f"Daily loss {daily_return:.2%} exceeded limit {-self.max_daily_loss:.2%}"
                )
        
        # Drawdown check
        current_drawdown = self.calculate_max_drawdown()
        if current_drawdown > self.max_drawdown:
            self.circuit_breaker.trigger_circuit_breaker(
                'drawdown_breach',
                f"Drawdown {current_drawdown:.2%} exceeded limit {self.max_drawdown:.2%}"
            )
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        return {
            'timestamp': datetime.now(),
            'risk_metrics': self.risk_metrics,
            'violations': self.violations[-10:],  # Last 10 violations
            'circuit_breaker_status': self.circuit_breaker.get_status(),
            'position_limits_status': self.position_limits.get_status(),
            'compliance_status': self.compliance_engine.get_status(),
            'recommendations': self._generate_risk_recommendations()
        }
    
    def _generate_risk_recommendations(self) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Example recommendations based on risk metrics
        if self.risk_metrics.get('max_drawdown', 0) > 0.1:  # 10%
            recommendations.append("Consider reducing position sizes due to high drawdown")
        
        if self.risk_metrics.get('var_95', 0) > 0.03:  # 3%
            recommendations.append("Portfolio VaR is elevated - consider diversification")
        
        if self.risk_metrics.get('top_position', 0) > 0.15:  # 15%
            recommendations.append("High concentration in single position - consider rebalancing")
        
        return recommendations
    
    def reset(self):
        """Reset risk engine state"""
        self.portfolio_history.clear()
        self.daily_pnl.clear()
        self.risk_metrics.clear()
        self.violations.clear()
        self.circuit_breaker.reset()
        self.position_limits.reset()
        
        self.logger.info("Risk engine reset")
