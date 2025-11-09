# trading_system/tests/unit/test_execution.py
"""
Unit tests for execution module
"""
import pytest
import pandas as pd
from datetime import datetime
from trading_system.execution.backtest_execution import BacktestExecution
from trading_system.execution.slippage_models import SlippageModel
from trading_system.core.types import Order, OrderType, OrderSide

class TestBacktestExecution:
    """Test backtest execution functionality"""
    
    def test_execution_initialization(self, sample_config):
        """Test execution handler initialization"""
        execution = BacktestExecution(sample_config)
        assert execution is not None
        assert execution.commission == 0.001
    
    def test_market_order_execution(self, sample_order):
        """Test market order execution"""
        config = {'commission': 0.001, 'slippage': 0.001}
        execution = BacktestExecution(config)
        current_prices = {'RELIANCE': 2500.0}
        
        trades = execution.execute_orders([sample_order], current_prices, datetime.now())
        
        assert len(trades) == 1
        assert trades[0].symbol == 'RELIANCE'
        assert trades[0].quantity == 100
        assert trades[0].commission > 0
    
    def test_order_validation(self):
        """Test order validation"""
        config = {'commission': 0.001}
        execution = BacktestExecution(config)
        
        # Valid order
        valid_order = Order(
            order_id="TEST_001",
            symbol="TCS",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=50
        )
        is_valid, message = execution.validate_order(valid_order)
        assert is_valid
        assert message == ""
        
        # Invalid order (zero quantity)
        invalid_order = Order(
            order_id="TEST_002", 
            symbol="TCS",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=0
        )
        is_valid, message = execution.validate_order(invalid_order)
        assert not is_valid
        assert "quantity must be positive" in message

class TestSlippageModels:
    """Test slippage model functionality"""
    
    def test_constant_slippage(self, sample_order):
        """Test constant slippage model"""
        config = {'method': 'constant', 'buy_slippage': 0.001, 'sell_slippage': 0.001}
        slippage_model = SlippageModel(config)
        
        # Test buy order
        buy_order = sample_order
        execution_price = slippage_model.calculate_execution_price(buy_order, 2500.0)
        expected_price = 2500.0 * (1 + 0.001)
        assert execution_price == pytest.approx(expected_price)
        
        # Test sell order
        sell_order = Order(
            order_id="TEST_SELL",
            symbol="RELIANCE",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=100,
            price=2500.0
        )
        execution_price = slippage_model.calculate_execution_price(sell_order, 2500.0)
        expected_price = 2500.0 * (1 - 0.001)
        assert execution_price == pytest.approx(expected_price)
    
    def test_volume_based_slippage(self, sample_order):
        """Test volume-based slippage"""
        config = {
            'method': 'volume_based',
            'base_slippage': 0.0005,
            'volume_impact': 0.0001
        }
        slippage_model = SlippageModel(config)
        
        # Add market data for volume-based calculation
        market_data = {
            'volume': {
                'avg_daily_volume': 1000000,
                'current_volume': 500000
            }
        }
        slippage_model.update_market_data('RELIANCE', market_data)
        
        execution_price = slippage_model.calculate_execution_price(sample_order, 2500.0)
        assert execution_price != 2500.0  # Should have slippage
