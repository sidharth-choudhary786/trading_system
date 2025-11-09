# trading_system/risk/position_limits.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..core.types import Order, OrderSide
from ..core.exceptions import RiskError

class PositionLimits:
    """
    Dynamic position sizing and limit management
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Position limit configuration
        self.limits_config = config.get('position_limits', {})
        self.max_portfolio_risk = self.limits_config.get('max_portfolio_risk', 0.02)  # 2% max portfolio risk
        self.max_drawdown = self.limits_config.get('max_drawdown', 0.15)  # 15% max drawdown
        self.volatility_scaling = self.limits_config.get('volatility_scaling', True)
        self.correlation_adjustment = self.limits_config.get('correlation_adjustment', True)
        
        # Risk parameters
        self.risk_free_rate = self.limits_config.get('risk_free_rate', 0.05)  # 5% risk-free rate
        self.confidence_level = self.limits_config.get('confidence_level', 0.95)  # 95% VaR confidence
        
        # Historical data for risk calculations
        self.returns_data = {}
        self.correlation_matrix = {}
        self.volatility_data = {}
        
        self.logger.info("Position Limits manager initialized")
    
    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        portfolio: Dict,
        market_data: Dict,
        method: str = 'kelly'
    ) -> float:
        """
        Calculate optimal position size based on risk management
        
        Args:
            symbol: Trading symbol
            signal_strength: Signal confidence (0-1)
            portfolio: Current portfolio state
            market_data: Current market data
            method: Sizing method ('kelly', 'risk_parity', 'equal_risk')
            
        Returns:
            Position size in units
        """
        portfolio_value = portfolio.get('portfolio_value', 0)
        if portfolio_value <= 0:
            return 0
        
        current_price = market_data.get(symbol, {}).get('price', 0)
        if current_price <= 0:
            return 0
        
        # Get risk parameters for the symbol
        volatility = self.volatility_data.get(symbol, {}).get('volatility', 0.02)  # 2% default
        expected_return = self.volatility_data.get(symbol, {}).get('expected_return', 0)
        
        if method == 'kelly':
            position_value = self._kelly_criterion(
                expected_return, volatility, signal_strength, portfolio_value
            )
        elif method == 'risk_parity':
            position_value = self._risk_parity_method(symbol, portfolio_value, market_data)
        elif method == 'equal_risk':
            position_value = self._equal_risk_contribution(symbol, portfolio_value, market_data)
        else:
            position_value = self._simple_fractional(signal_strength, portfolio_value)
        
        # Apply volatility scaling
        if self.volatility_scaling:
            position_value = self._apply_volatility_scaling(position_value, volatility)
        
        # Apply correlation adjustment
        if self.correlation_adjustment:
            position_value = self._apply_correlation_adjustment(symbol, position_value, portfolio)
        
        # Convert to units
        position_size = position_value / current_price
        
        # Apply hard limits
        position_size = self._apply_hard_limits(symbol, position_size, portfolio_value, current_price)
        
        self.logger.debug(f"Calculated position size for {symbol}: {position_size:.0f} units")
        return position_size
    
    def _kelly_criterion(
        self, 
        expected_return: float, 
        volatility: float, 
        signal_strength: float, 
        portfolio_value: float
    ) -> float:
        """Kelly Criterion for position sizing"""
        if volatility <= 0:
            return 0
        
        # Simplified Kelly formula: f = (mu - r) / sigma^2
        kelly_fraction = (expected_return - self.risk_free_rate) / (volatility ** 2)
        
        # Adjust for signal strength
        kelly_fraction *= signal_strength
        
        # Conservative Kelly (half Kelly)
        kelly_fraction *= 0.5
        
        # Cap at maximum portfolio risk
        max_fraction = self.max_portfolio_risk
        kelly_fraction = min(kelly_fraction, max_fraction)
        
        return kelly_fraction * portfolio_value
    
    def _risk_parity_method(self, symbol: str, portfolio_value: float, market_data: Dict) -> float:
        """Risk Parity position sizing"""
        # Calculate inverse volatility weighting
        all_volatilities = {}
        
        for sym, data in self.volatility_data.items():
            if sym in market_data:
                all_volatilities[sym] = data.get('volatility', 0.02)
        
        if not all_volatilities:
            return portfolio_value * 0.1  # Default 10%
        
        # Inverse volatility weighting
        inv_vol_sum = sum(1 / max(vol, 0.001) for vol in all_volatilities.values())
        symbol_inv_vol = 1 / max(all_volatilities.get(symbol, 0.02), 0.001)
        
        risk_parity_weight = symbol_inv_vol / inv_vol_sum
        return risk_parity_weight * portfolio_value
    
    def _equal_risk_contribution(self, symbol: str, portfolio_value: float, market_data: Dict) -> float:
        """Equal Risk Contribution position sizing"""
        # This is a simplified implementation
        # In practice, this would use full covariance matrix
        
        symbol_volatility = self.volatility_data.get(symbol, {}).get('volatility', 0.02)
        
        # Simple equal risk contribution
        n_assets = len([s for s in self.volatility_data if s in market_data])
        if n_assets == 0:
            n_assets = 1
        
        risk_weight = 1 / n_assets
        position_value = risk_weight * portfolio_value
        
        # Adjust for individual volatility
        avg_volatility = np.mean([d.get('volatility', 0.02) for d in self.volatility_data.values()])
        if avg_volatility > 0:
            volatility_ratio = symbol_volatility / avg_volatility
            position_value /= volatility_ratio
        
        return position_value
    
    def _simple_fractional(self, signal_strength: float, portfolio_value: float) -> float:
        """Simple fractional position sizing"""
        base_fraction = 0.1  # 10% base allocation
        position_value = base_fraction * signal_strength * portfolio_value
        return min(position_value, portfolio_value * self.max_portfolio_risk)
    
    def _apply_volatility_scaling(self, position_value: float, volatility: float) -> float:
        """Scale position based on volatility"""
        if not self.volatility_scaling or volatility <= 0:
            return position_value
        
        # Scale inversely with volatility
        base_volatility = 0.02  # 2% base volatility
        scaling_factor = base_volatility / volatility
        scaling_factor = np.clip(scaling_factor, 0.5, 2.0)  # Limit scaling
        
        return position_value * scaling_factor
    
    def _apply_correlation_adjustment(
        self, 
        symbol: str, 
        position_value: float, 
        portfolio: Dict
    ) -> float:
        """Adjust position based on portfolio correlation"""
        if not self.correlation_adjustment:
            return position_value
        
        current_positions = portfolio.get('positions', {})
        if not current_positions:
            return position_value
        
        # Calculate correlation with existing positions
        avg_correlation = 0
        correlation_count = 0
        
        for existing_symbol in current_positions:
            correlation = self.correlation_matrix.get(symbol, {}).get(existing_symbol, 0)
            if correlation is not None:
                avg_correlation += abs(correlation)  # Use absolute correlation
                correlation_count += 1
        
        if correlation_count > 0:
            avg_correlation /= correlation_count
            
            # Reduce position for high correlation
            if avg_correlation > 0.5:
                reduction_factor = 1 - (avg_correlation - 0.5) * 2  # Reduce up to 100% for perfect correlation
                position_value *= max(reduction_factor, 0.1)  # Keep at least 10%
        
        return position_value
    
    def _apply_hard_limits(
        self, 
        symbol: str, 
        position_size: float, 
        portfolio_value: float, 
        current_price: float
    ) -> float:
        """Apply hard position limits"""
        if current_price <= 0:
            return 0
        
        position_value = position_size * current_price
        
        # Maximum position value limit
        max_position_value = portfolio_value * self.max_portfolio_risk
        max_position_size = max_position_value / current_price
        
        # Maximum absolute position size (liquidity consideration)
        max_absolute_size = 10000  # Example limit
        
        position_size = min(position_size, max_position_size, max_absolute_size)
        position_size = max(position_size, 0)  # No short positions
        
        return position_size
    
    def calculate_portfolio_risk(self, portfolio: Dict, market_data: Dict) -> Dict:
        """
        Calculate comprehensive portfolio risk metrics
        
        Args:
            portfolio: Current portfolio state
            market_data: Current market data
            
        Returns:
            Portfolio risk analysis
        """
        risk_report = {
            'timestamp': datetime.now(),
            'portfolio_value': portfolio.get('portfolio_value', 0),
            'risk_metrics': {},
            'concentration_metrics': {},
            'stress_scenarios': {}
        }
        
        portfolio_value = portfolio.get('portfolio_value', 0)
        if portfolio_value <= 0:
            return risk_report
        
        current_positions = portfolio.get('positions', {})
        
        # Calculate basic risk metrics
        risk_report['risk_metrics'] = self._calculate_basic_risk_metrics(
            current_positions, market_data, portfolio_value
        )
        
        # Calculate concentration metrics
        risk_report['concentration_metrics'] = self._calculate_concentration_metrics(
            current_positions, market_data, portfolio_value
        )
        
        # Run stress tests
        risk_report['stress_scenarios'] = self._run_stress_tests(
            current_positions, market_data, portfolio_value
        )
        
        return risk_report
    
    def _calculate_basic_risk_metrics(
        self, 
        positions: Dict, 
        market_data: Dict, 
        portfolio_value: float
    ) -> Dict:
        """Calculate basic portfolio risk metrics"""
        if not positions:
            return {}
        
        # Calculate portfolio returns (simplified)
        portfolio_returns = self._calculate_portfolio_returns(positions, market_data)
        
        if len(portfolio_returns) < 2:
            return {}
        
        risk_metrics = {
            'volatility': np.std(portfolio_returns) * np.sqrt(252),  # Annualized
            'var_95': np.percentile(portfolio_returns, 5) * portfolio_value,
            'cvar_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * portfolio_value,
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'sharpe_ratio': (np.mean(portfolio_returns) * 252 - self.risk_free_rate) / (np.std(portfolio_returns) * np.sqrt(252))
        }
        
        return risk_metrics
    
    def _calculate_portfolio_returns(self, positions: Dict, market_data: Dict) -> np.ndarray:
        """Calculate historical portfolio returns (simplified)"""
        # This is a simplified implementation
        # In practice, you would use actual historical returns
        
        returns = []
        for symbol, quantity in positions.items():
            if symbol in self.returns_data:
                symbol_returns = self.returns_data[symbol]
                if len(returns) == 0:
                    returns = symbol_returns * (quantity / len(positions))
                else:
                    returns += symbol_returns * (quantity / len(positions))
        
        if len(returns) == 0:
            # Generate random returns for demonstration
            returns = np.random.normal(0.001, 0.015, 100)  # 0.1% mean, 1.5% std
        
        return np.array(returns)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_concentration_metrics(
        self, 
        positions: Dict, 
        market_data: Dict, 
        portfolio_value: float
    ) -> Dict:
        """Calculate portfolio concentration metrics"""
        if not positions:
            return {}
        
        position_values = {}
        for symbol, quantity in positions.items():
            if symbol in market_data:
                price = market_data[symbol].get('price', 0)
                position_values[symbol] = quantity * price
        
        total_invested = sum(position_values.values())
        if total_invested <= 0:
            return {}
        
        # Herfindahl-Hirschman Index (HHI)
        weights = [value / total_invested for value in position_values.values()]
        hhi = sum(w ** 2 for w in weights) * 10000  # Scale to 0-10000
        
        # Top positions concentration
        sorted_weights = sorted(weights, reverse=True)
        top_5_concentration = sum(sorted_weights[:5])
        
        return {
            'hhi_index': hhi,
            'top_5_concentration': top_5_concentration,
            'effective_n': 1 / sum(w ** 2 for w in weights),  # Effective number of positions
            'largest_position': max(weights) if weights else 0
        }
    
    def _run_stress_tests(
        self, 
        positions: Dict, 
        market_data: Dict, 
        portfolio_value: float
    ) -> Dict:
        """Run portfolio stress tests"""
        stress_scenarios = {
            'market_crash': -0.20,  # 20% market decline
            'high_volatility': 0.03,  # 3% daily volatility
            'sector_shock': -0.15,   # 15% sector-specific shock
            'liquidity_crisis': -0.10  # 10% liquidity impact
        }
        
        results = {}
        
        for scenario, shock in stress_scenarios.items():
            scenario_loss = 0
            
            for symbol, quantity in positions.items():
                if symbol in market_data:
                    price = market_data[symbol].get('price', 0)
                    position_value = quantity * price
                    
                    # Apply scenario-specific shock
                    if scenario == 'market_crash':
                        scenario_loss += position_value * shock
                    elif scenario == 'sector_shock':
                        # Apply shock only to certain sectors
                        sector = self.sector_data.get(symbol, '')
                        if sector in ['Technology', 'Financial']:  # Example vulnerable sectors
                            scenario_loss += position_value * shock
                    elif scenario == 'liquidity_crisis':
                        # Liquidity crisis affects all positions
                        scenario_loss += position_value * shock
            
            results[scenario] = {
                'loss_amount': abs(scenario_loss),
                'loss_percentage': abs(scenario_loss) / portfolio_value,
                'impact': 'High' if abs(scenario_loss) / portfolio_value > 0.1 else 'Medium'
            }
        
        return results
    
    def update_returns_data(self, returns_data: Dict):
        """Update historical returns data"""
        self.returns_data.update(returns_data)
    
    def update_correlation_matrix(self, correlation_matrix: Dict):
        """Update correlation matrix"""
        self.correlation_matrix.update(correlation_matrix)
    
    def update_volatility_data(self, volatility_data: Dict):
        """Update volatility data"""
        self.volatility_data.update(volatility_data)
