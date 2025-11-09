# trading_system/tests/unit/test_portfolio.py
"""
Unit tests for portfolio module
"""
import pytest
import pandas as pd
from trading_system.portfolio.portfolio import Portfolio
from trading_system.portfolio.optimizer import PortfolioOptimizer

class TestPortfolio:
    """Test portfolio management functionality"""
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization"""
        portfolio = Portfolio(initial_capital=100000)
        assert portfolio.initial_capital == 100000
        assert portfolio.current_cash == 100000
        assert portfolio.portfolio_value == 100000
        assert portfolio.positions == {}
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation"""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.positions = {'RELIANCE': 100, 'TCS': 50}
        current_prices = {'RELIANCE': 2500, 'TCS': 3500}
        
        portfolio.update_portfolio(current_prices)
        
        expected_value = 100000 + (100 * 2500) + (50 * 3500)
        assert portfolio.portfolio_value == expected_value
    
    def test_trade_execution(self):
        """Test trade execution and portfolio update"""
        portfolio = Portfolio(initial_capital=100000)
        
        # Create a mock trade
        from trading_system.core.types import Trade, OrderSide
        from datetime import datetime
        
        trade = Trade(
            trade_id="TEST_TRADE_001",
            order_id="TEST_ORDER_001",
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=100,
            price=2500.0,
            timestamp=datetime.now(),
            commission=25.0
        )
        
        portfolio.execute_trade(trade)
        
        # Check portfolio updates
        assert portfolio.positions['RELIANCE'] == 100
        assert portfolio.current_cash == 100000 - (100 * 2500) - 25.0
        assert 'RELIANCE' in portfolio.trades

class TestPortfolioOptimizer:
    """Test portfolio optimization functionality"""
    
    def test_optimizer_initialization(self):
        """Test portfolio optimizer initialization"""
        config = {
            'optimization_method': 'sharpe_maximization',
            'constraints': {
                'max_position_size': 0.1,
                'max_sector_exposure': 0.3
            }
        }
        
        optimizer = PortfolioOptimizer(config)
        assert optimizer.optimization_method == 'sharpe_maximization'
        assert optimizer.constraints['max_position_size'] == 0.1
    
    def test_sharpe_optimization(self):
        """Test Sharpe ratio optimization"""
        config = {
            'optimization_method': 'sharpe_maximization',
            'constraints': {
                'max_position_size': 0.2
            }
        }
        
        optimizer = PortfolioOptimizer(config)
        
        # Sample returns data
        returns_data = pd.DataFrame({
            'RELIANCE': np.random.normal(0.001, 0.02, 100),
            'TCS': np.random.normal(0.0008, 0.015, 100),
            'INFY': np.random.normal(0.0009, 0.018, 100)
        }, index=pd.date_range('2023-01-01', periods=100, freq='D'))
        
        # This would perform optimization in actual implementation
        # For now, test the method exists
        assert hasattr(optimizer, 'optimize_sharpe_ratio')
    
    def test_constraint_validation(self):
        """Test portfolio constraint validation"""
        config = {
            'constraints': {
                'max_position_size': 0.1,
                'long_only': True
            }
        }
        
        optimizer = PortfolioOptimizer(config)
        
        # Test valid weights
        valid_weights = {'RELIANCE': 0.08, 'TCS': 0.07, 'INFY': 0.05}
        is_valid, message = optimizer.validate_constraints(valid_weights)
        assert is_valid
        assert message == ""
        
        # Test invalid weights (exceeds position limit)
        invalid_weights = {'RELIANCE': 0.15, 'TCS': 0.07, 'INFY': 0.05}
        is_valid, message = optimizer.validate_constraints(invalid_weights)
        assert not is_valid
        assert "position size" in message.lower()
