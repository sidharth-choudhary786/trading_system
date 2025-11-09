# trading_system/tests/integration/test_backtest_engine.py
"""
Integration tests for backtesting engine
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading_system.analysis.backtester import Backtester

class TestBacktestEngine:
    """Integration tests for backtesting engine"""
    
    def test_complete_backtest_flow(self, sample_price_data):
        """Test complete backtest flow"""
        config = {
            'initial_capital': 100000,
            'commission': 0.001,
            'slippage': 0.001
        }
        
        backtester = Backtester(config)
        
        # Create sample data for multiple symbols
        data = {
            'RELIANCE': sample_price_data.set_index('date'),
            'TCS': sample_price_data.set_index('date') * 1.1  # Slightly different prices
        }
        
        # Mock strategy
        class MockStrategy:
            def generate_signals(self, data, current_date):
                # Simple buy/hold strategy
                return {'RELIANCE': 'BUY', 'TCS': 'BUY'}
        
        strategy = MockStrategy()
        
        # Run backtest
        results = backtester.run_backtest(
            data=data,
            strategy=strategy,
            start_date='2023-01-01',
            end_date='2023-12-31',
            rebalance_frequency='monthly'
        )
        
        # Verify results structure
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'portfolio_history' in results
        
        # Verify portfolio history
        assert len(results['portfolio_history']) > 0
        assert 'portfolio_value' in results['portfolio_history'][0]
    
    def test_backtest_with_different_rebalance_frequencies(self, sample_price_data):
        """Test backtest with different rebalance frequencies"""
        config = {'initial_capital': 100000}
        backtester = Backtester(config)
        
        data = {'RELIANCE': sample_price_data.set_index('date')}
        
        class MockStrategy:
            def generate_signals(self, data, current_date):
                return {'RELIANCE': 'BUY'}
        
        strategy = MockStrategy()
        
        # Test weekly rebalancing
        weekly_results = backtester.run_backtest(
            data=data,
            strategy=strategy,
            start_date='2023-01-01',
            end_date='2023-03-31',
            rebalance_frequency='weekly'
        )
        
        # Test monthly rebalancing
        monthly_results = backtester.run_backtest(
            data=data,
            strategy=strategy,
            start_date='2023-01-01',
            end_date='2023-03-31', 
            rebalance_frequency='monthly'
        )
        
        # Should have different trade counts
        assert weekly_results['total_trades'] != monthly_results['total_trades']
