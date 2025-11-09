# trading_system/tests/unit/test_data_sources.py
import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from trading_system.data.sources.yahoo_finance import YahooFinanceDataSource
from trading_system.core.exceptions import DataSourceError

class TestDataSources:
    """Test data sources"""
    
    def test_yahoo_finance_initialization(self):
        """Test Yahoo Finance data source initialization"""
        config = {'rate_limit': 1000}
        source = YahooFinanceDataSource(config)
        assert source is not None
        assert source.rate_limit == 1000
    
    def test_data_source_validation(self):
        """Test data source symbol validation"""
        config = {'rate_limit': 1000}
        source = YahooFinanceDataSource(config)
        
        # Test valid symbols
        assert source.validate_symbol("RELIANCE.NS") == True
        assert source.validate_symbol("TCS.BO") == True
        
        # Test invalid symbols
        assert source.validate_symbol("") == False
    
    @pytest.mark.skip(reason="Requires internet connection")
    def test_data_download(self):
        """Test data download (requires internet)"""
        config = {'rate_limit': 1000}
        source = YahooFinanceDataSource(config)
        
        # This test would actually download data
        # data = source.download_data("RELIANCE.NS", "2023-01-01", "2023-01-10")
        # assert not data.empty
        # assert 'close' in data.columns
