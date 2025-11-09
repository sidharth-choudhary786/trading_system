# trading_system/tests/fixtures/mock_objects/mock_data_source.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MockDataSource:
    """Mock data source for testing"""
    
    def __init__(self):
        self.data = {}
    
    def get_historical_data(self, symbol, start_date, end_date, interval='1day'):
        """Mock historical data"""
        dates = pd.date_range(start_date, end_date, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(150, 250, len(dates)),
            'low': np.random.uniform(50, 150, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        self.data[symbol] = data
        return data
    
    def get_dividends(self, symbol, start_date, end_date):
        """Mock dividends"""
        return pd.DataFrame()
    
    def get_splits(self, symbol, start_date, end_date):
        """Mock splits"""
        return pd.DataFrame()
