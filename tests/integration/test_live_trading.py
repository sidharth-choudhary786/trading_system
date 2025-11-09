# tests/integration/test_live_trading.py
"""
Integration tests for live trading components
Tests market data -> signals -> execution flow
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_system.production.production_manager import ProductionManager
from trading_system.execution.live_execution import LiveExecution

class TestLiveTrading:
    """Test live trading integration with mocked components"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample live market data"""
        return {
            'STOCK_A': {
                'last_price': 150.25,
                'bid': 150.20,
                'ask': 150.30,
                'volume': 10000,
                'timestamp': datetime.now()
            },
            'STOCK_B': {
                'last_price': 275.80,
                'bid': 275.75,
                'ask': 275.85,
                'volume': 15000,
                'timestamp': datetime.now()
            }
        }
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample trading signals"""
        return {
            'STOCK_A': {
                'action': 'BUY',
                'confidence': 0.85,
                'quantity': 100,
                'limit_price': 150.30
            },
            'STOCK_B': {
                'action': 'SELL', 
                'confidence': 0.72,
                'quantity': 50,
                'limit_price': 275.70
            }
        }
    
    @pytest.fixture
    def config(self):
        """Test configuration for live trading"""
        return {
            'production': {
                'mode': 'paper_trading',
                'broker': 'zerodha',
                'initial_capital': 1000000,
                'max_position_size': 0.1
            },
            'risk': {
                'max_daily_loss': 0.02,
                'max_position_size': 0.1,
                'enable_circuit_breakers': True
            },
            'execution': {
                'order_type': 'LIMIT',
                'max_slippage': 0.002,
                'retry_failed_orders': True
            }
        }
    
    @patch('trading_system.production.broker_apis.zerodha.ZerodhaAPI')
    def test_live_trading_flow(self, mock_broker, sample_market_data, sample_signals, config):
        """Test complete live trading flow with mocked broker"""
        # Setup mock broker
        mock_broker_instance = Mock()
        mock_broker.return_value = mock_broker_instance
        
        # Mock broker responses
        mock_broker_instance.place_order.return_value = {
            'status': 'success',
            'order_id': 'TEST_ORDER_123',
            'message': 'Order placed successfully'
        }
        
        mock_broker_instance.get_order_status.return_value = {
            'status': 'COMPLETE',
            'filled_quantity': 100,
            'average_price': 150.25
        }
        
        mock_broker_instance.get_positions.return_value = {
            'STOCK_A': {'quantity': 100, 'average_price': 150.25}
        }
        
        mock_broker_instance.get_account_info.return_value = {
            'buying_power': 500000,
            'cash': 500000,
            'portfolio_value': 1000000
        }
        
        # Initialize production manager
        production_manager = ProductionManager(config)
        
        # Test signal processing
        orders = production_manager.process_signals(sample_signals, sample_market_data)
        
        assert len(orders) > 0, "No orders generated from signals"
        
        # Test order execution
        execution_results = production_manager.execute_orders(orders)
        
        assert execution_results is not None, "Order execution failed"
        assert len(execution_results['executed_orders']) > 0, "No orders executed"
        
        # Test portfolio update
        portfolio = production_manager.get_portfolio_status()
        
        assert portfolio is not None, "Portfolio status failed"
        assert 'positions' in portfolio, "Positions missing from portfolio"
        assert 'cash' in portfolio, "Cash balance missing"
        assert 'total_value' in portfolio, "Total value missing"
    
    @patch('trading_system.production.broker_apis.zerodha.ZerodhaAPI')
    def test_risk_management_live(self, mock_broker, config):
        """Test risk management in live trading"""
        production_manager = ProductionManager(config)
        
        # Test position limit enforcement
        large_order = {
            'symbol': 'STOCK_A',
            'action': 'BUY',
            'quantity': 10000,  # Very large quantity
            'limit_price': 150.30
        }
        
        # This should be blocked by risk management
        risk_check = production_manager.check_order_risk(large_order)
        
        assert not risk_check['approved'], "Large order should be rejected"
        assert 'position_limit' in risk_check['reasons'], "Position limit not enforced"
    
    @patch('trading_system.production.broker_apis.zerodha.ZerodhaAPI')
    def test_market_hours_validation(self, mock_broker, config):
        """Test market hours validation"""
        production_manager = ProductionManager(config)
        
        # Test during market hours
        market_hours_check = production_manager.check_market_hours()
        
        assert isinstance(market_hours_check, bool), "Market hours check should return boolean"
        
        # Test order placement outside market hours
        if not market_hours_check:
            orders = [{'symbol': 'STOCK_A', 'action': 'BUY', 'quantity': 100}]
            result = production_manager.execute_orders(orders)
            assert result['status'] == 'market_closed', "Should not execute when market closed"
    
    @patch('trading_system.production.broker_apis.zerodha.ZerodhaAPI')
    def test_error_handling(self, mock_broker, config):
        """Test error handling in live trading"""
        mock_broker_instance = Mock()
        mock_broker.return_value = mock_broker_instance
        
        # Mock broker failure
        mock_broker_instance.place_order.side_effect = Exception("Broker API error")
        
        production_manager = ProductionManager(config)
        
        orders = [{'symbol': 'STOCK_A', 'action': 'BUY', 'quantity': 100}]
        
        # Should handle broker errors gracefully
        result = production_manager.execute_orders(orders)
        
        assert result['status'] == 'error', "Should handle broker errors"
        assert 'errors' in result, "Error details should be included"
    
    @patch('trading_system.production.broker_apis.zerodha.ZerodhaAPI')
    def test_performance_monitoring(self, mock_broker, config):
        """Test performance monitoring in live trading"""
        production_manager = ProductionManager(config)
        
        # Get performance metrics
        performance = production_manager.get_performance_metrics()
        
        expected_metrics = [
            'total_trades', 'win_rate', 'total_pnl', 
            'sharpe_ratio', 'max_drawdown'
        ]
        
        for metric in expected_metrics:
            assert metric in performance, f"Performance metric {metric} missing"
    
    @patch('trading_system.production.broker_apis.zerodha.ZerodhaAPI')
    def test_circuit_breakers(self, mock_broker, config):
        """Test circuit breaker functionality"""
        production_manager = ProductionManager(config)
        
        # Simulate large loss
        production_manager.daily_pnl = -20000  # 2% loss on 1M capital
        
        # Check if circuit breaker triggers
        circuit_breaker_check = production_manager.check_circuit_breakers()
        
        assert isinstance(circuit_breaker_check, dict), "Circuit breaker check should return dict"
        assert 'triggered' in circuit_breaker_check, "Circuit breaker status missing"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
