# trading_system/tests/unit/test_utils.py
"""
Unit tests for utility functions
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from trading_system.utils.config_loader import ConfigLoader
from trading_system.utils.calendar import MarketCalendar

class TestConfigLoader:
    """Test configuration loading functionality"""
    
    def test_config_loading(self, tmp_path):
        """Test configuration file loading"""
        # Create temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
        trading:
            mode: "backtest"
            initial_capital: 100000
        data:
            sources: ["yahoo", "twelvedata"]
        """)
        
        config_loader = ConfigLoader(str(tmp_path))
        config = config_loader.load_config()
        
        assert config['trading']['mode'] == "backtest"
        assert config['trading']['initial_capital'] == 100000
        assert "yahoo" in config['data']['sources']
    
    def test_config_value_retrieval(self, tmp_path):
        """Test configuration value retrieval"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
        database:
            host: "localhost"
            port: 5432
        """)
        
        config_loader = ConfigLoader(str(tmp_path))
        config = config_loader.load_config()
        
        # Test dot notation retrieval
        host = config_loader.get_config_value(config, 'database.host')
        port = config_loader.get_config_value(config, 'database.port')
        
        assert host == "localhost"
        assert port == 5432

class TestMarketCalendar:
    """Test market calendar functionality"""
    
    def test_market_calendar_initialization(self):
        """Test market calendar initialization"""
        calendar = MarketCalendar()
        assert calendar is not None
        assert hasattr(calendar, 'holidays')
    
    def test_trading_day_check(self):
        """Test trading day verification"""
        calendar = MarketCalendar()
        
        # Test a known business day (Monday)
        business_day = datetime(2023, 12, 18)  # Monday
        assert calendar.is_trading_day(business_day)
        
        # Test a weekend
        weekend_day = datetime(2023, 12, 17)  # Sunday
        assert not calendar.is_trading_day(weekend_day)
    
    def test_next_trading_day(self):
        """Test next trading day calculation"""
        calendar = MarketCalendar()
        
        # Friday to Monday
        friday = datetime(2023, 12, 15)  # Friday
        next_day = calendar.get_next_trading_day(friday)
        expected_monday = datetime(2023, 12, 18)  # Monday
        
        assert next_day == expected_monday
    
    def test_market_hours_check(self):
        """Test market hours verification"""
        calendar = MarketCalendar()
        
        # During market hours
        market_time = datetime(2023, 12, 18, 10, 30)  ##### 10:30 AM
        assert calendar.is_market_open(market_time)
        
        # Before market hours
        pre_market = datetime(2023, 12, 18, 9, 0)  ##### 9:00 AM
        assert not calendar.is_market_open(pre_market)
        
        # After market hours  
        post_market = datetime(2023, 12, 18, 16, 0)  ##### 4:00 PM
        assert not calendar.is_market_open(post_market)
