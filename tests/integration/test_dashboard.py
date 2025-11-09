# trading_system/tests/integration/test_dashboard.py
import pytest
import pandas as pd
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestDashboard:
    """Integration tests for Dashboard module"""
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization"""
        from trading_system.dashboard.app import TradingDashboard
        
        config = {
            'mode': 'backtest',
            'initial_capital': 100000
        }
        
        dashboard = TradingDashboard(config)
        assert dashboard is not None
        assert dashboard.config == config
    
    def test_dashboard_components(self):
        """Test dashboard components initialization"""
        from trading_system.dashboard.components.portfolio_view import PortfolioView
        from trading_system.dashboard.components.performance_charts import PerformanceCharts
        from trading_system.dashboard.components.trade_journal import TradeJournal
        
        # Test portfolio view
        portfolio_data = {
            'total_value': 100000,
            'cash': 20000,
            'holdings': []
        }
        portfolio_view = PortfolioView(portfolio_data)
        assert portfolio_view is not None
        
        # Test performance charts
        performance_data = {
            'total_return': 10.5,
            'sharpe_ratio': 1.2
        }
        performance_charts = PerformanceCharts(performance_data)
        assert performance_charts is not None
        
        # Test trade journal
        trade_data = {
            'trades': []
        }
        trade_journal = TradeJournal(trade_data)
        assert trade_journal is not None
    
    def test_dashboard_data_flow(self):
        """Test data flow through dashboard components"""
        from trading_system.dashboard.app import TradingDashboard
        
        config = {'mode': 'backtest', 'initial_capital': 100000}
        dashboard = TradingDashboard(config)
        
        # Test sample data generation
        portfolio_data = dashboard._get_sample_portfolio_data()
        assert portfolio_data['total_value'] == 1012450
        assert 'holdings' in portfolio_data
        
        performance_data = dashboard._get_sample_performance_data()
        assert 'total_return' in performance_data
        
        trade_data = dashboard._get_sample_trade_data()
        assert 'trades' in trade_data
    
    @pytest.mark.skip(reason="Requires Streamlit environment")
    def test_dashboard_render(self):
        """Test dashboard rendering (requires Streamlit)"""
        # This test would require Streamlit testing framework
        pass
