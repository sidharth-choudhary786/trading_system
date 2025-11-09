# tests/integration/test_portfolio_optimization.py
"""
Integration tests for portfolio optimization
Tests signal generation -> optimization -> allocation flow
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_system.portfolio.optimizer import PortfolioOptimizer
from trading_system.portfolio.allocator import CapitalAllocator
from trading_system.portfolio.risk import RiskManager

class TestPortfolioOptimization:
    """Test portfolio optimization integration"""
    
    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data for multiple symbols"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        n_periods = len(dates)
        
        symbols = ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E']
        
        # Generate correlated returns
        np.random.seed(42)
        base_returns = np.random.normal(0.001, 0.02, (n_periods, len(symbols)))
        
        # Add some correlation
        correlation = 0.3
        correlated_returns = base_returns * (1 - correlation) + np.random.normal(0, 0.02, n_periods).reshape(-1, 1) * correlation
        
        returns_data = pd.DataFrame(
            correlated_returns,
            index=dates,
            columns=symbols
        )
        
        return returns_data
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample trading signals"""
        symbols = ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E']
        
        # Generate random signals (-1, 0, 1)
        signals = {symbol: np.random.choice([-1, 0, 1]) for symbol in symbols}
        
        return signals
    
    @pytest.fixture
    def config(self):
        """Test configuration for portfolio optimization"""
        return {
            'portfolio': {
                'optimization_method': 'sharpe_maximization',
                'constraints': {
                    'max_position_size': 0.3,
                    'max_sector_exposure': 0.5,
                    'min_diversification': 3,
                    'long_only': True
                },
                'rebalance_frequency': 'fortnightly',
                'lookback_period': 252
            },
            'risk': {
                'max_drawdown': 0.15,
                'max_position_size': 0.2,
                'var_confidence': 0.95
            }
        }
    
    def test_complete_optimization_flow(self, sample_returns_data, sample_signals, config):
        """Test complete portfolio optimization flow"""
        # Initialize components
        optimizer = PortfolioOptimizer(config)
        allocator = CapitalAllocator(config)
        risk_manager = RiskManager(config)
        
        # Test portfolio optimization
        current_weights = {symbol: 0.0 for symbol in sample_returns_data.columns}
        current_prices = {symbol: 100.0 for symbol in sample_returns_data.columns}
        
        # Get optimal weights
        optimal_weights = optimizer.optimize_portfolio(
            returns_data=sample_returns_data,
            signals=sample_signals,
            current_weights=current_weights,
            current_prices=current_prices
        )
        
        assert optimal_weights is not None, "Portfolio optimization failed"
        assert len(optimal_weights) == len(sample_returns_data.columns), "Weights length mismatch"
        
        # Check weight constraints
        total_weight = sum(optimal_weights.values())
        assert abs(total_weight - 1.0) < 1e-6, f"Weights don't sum to 1: {total_weight}"
        
        max_weight = max(optimal_weights.values())
        assert max_weight <= config['portfolio']['constraints']['max_position_size'], "Max position size violated"
        
        # Test capital allocation
        portfolio_value = 1000000  # 1 million
        allocations = allocator.calculate_allocations(
            weights=optimal_weights,
            portfolio_value=portfolio_value,
            current_prices=current_prices
        )
        
        assert allocations is not None, "Capital allocation failed"
        assert len(allocations) == len(optimal_weights), "Allocations length mismatch"
        
        # Test risk management
        risk_report = risk_manager.assess_portfolio_risk(
            weights=optimal_weights,
            returns_data=sample_returns_data,
            portfolio_value=portfolio_value
        )
        
        assert risk_report is not None, "Risk assessment failed"
        assert 'var' in risk_report, "VaR missing from risk report"
        assert 'expected_shortfall' in risk_report, "Expected shortfall missing"
        assert 'volatility' in risk_report, "Volatility missing"
    
    def test_rebalancing_logic(self, sample_returns_data, config):
        """Test portfolio rebalancing logic"""
        optimizer = PortfolioOptimizer(config)
        
        # Initial weights
        initial_weights = {
            'STOCK_A': 0.2, 'STOCK_B': 0.2, 'STOCK_C': 0.2, 
            'STOCK_D': 0.2, 'STOCK_E': 0.2
        }
        
        # Simulate price changes
        current_prices = {
            'STOCK_A': 110,  # +10%
            'STOCK_B': 95,   # -5%
            'STOCK_C': 105,  # +5%
            'STOCK_D': 100,  # No change
            'STOCK_E': 90    # -10%
        }
        
        # Calculate current weights after price changes
        current_values = {symbol: initial_weights[symbol] * current_prices[symbol] / 100 
                         for symbol in initial_weights}
        total_value = sum(current_values.values())
        current_weights = {symbol: value / total_value for symbol, value in current_values.items()}
        
        # Test if rebalancing is needed
        signals = {symbol: 1 for symbol in initial_weights}  # Buy signals for all
        
        needs_rebalance = optimizer.needs_rebalancing(
            current_weights=current_weights,
            target_weights=initial_weights,
            signals=signals,
            threshold=0.05  # 5% threshold
        )
        
        assert isinstance(needs_rebalance, bool), "Rebalancing check should return boolean"
    
    def test_risk_constraints(self, sample_returns_data, config):
        """Test risk constraint enforcement"""
        risk_manager = RiskManager(config)
        
        # Test risky weights
        risky_weights = {
            'STOCK_A': 0.8,  # Too concentrated
            'STOCK_B': 0.1,
            'STOCK_C': 0.1,
            'STOCK_D': 0.0,
            'STOCK_E': 0.0
        }
        
        # Check if constraints are violated
        constraint_checks = risk_manager.check_constraints(
            weights=risky_weights,
            returns_data=sample_returns_data
        )
        
        assert 'position_limits' in constraint_checks, "Position limits check missing"
        assert 'diversification' in constraint_checks, "Diversification check missing"
        
        # Should have violations
        assert not all(constraint_checks.values()), "Risk constraints not properly enforced"
    
    def test_performance_metrics(self, sample_returns_data, config):
        """Test portfolio performance metrics calculation"""
        optimizer = PortfolioOptimizer(config)
        
        # Equal weights portfolio
        weights = {symbol: 0.2 for symbol in sample_returns_data.columns}
        
        metrics = optimizer.calculate_performance_metrics(
            weights=weights,
            returns_data=sample_returns_data
        )
        
        expected_metrics = [
            'expected_return', 'volatility', 'sharpe_ratio', 
            'max_drawdown', 'var_95'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Performance metric {metric} missing"
            assert isinstance(metrics[metric], (int, float)), f"Metric {metric} should be numeric"
    
    def test_transaction_cost_impact(self, sample_returns_data, sample_signals, config):
        """Test transaction cost impact on optimization"""
        allocator = CapitalAllocator(config)
        
        # Initial and target weights
        initial_weights = {symbol: 0.2 for symbol in sample_returns_data.columns}
        target_weights = {symbol: 0.25 if i < 2 else 0.1 for i, symbol in enumerate(sample_returns_data.columns)}
        
        current_prices = {symbol: 100.0 for symbol in sample_returns_data.columns}
        portfolio_value = 1000000
        
        # Calculate allocations with transaction costs
        allocations = allocator.calculate_allocations(
            weights=target_weights,
            portfolio_value=portfolio_value,
            current_prices=current_prices,
            current_weights=initial_weights,
            transaction_cost=0.001  # 0.1% transaction cost
        )
        
        assert allocations is not None, "Allocation with transaction costs failed"
        
        # Check that transaction costs are considered
        total_trades = sum(abs(allocations[symbol]['shares_to_trade']) for symbol in allocations)
        assert total_trades > 0, "No trades calculated"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
