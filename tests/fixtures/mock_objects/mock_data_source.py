# trading_system/tests/fixtures/mock_objects/mock_data_source.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MockDataSource:
    """Mock data source for testing"""
    
    def __init__(self):
        self.data = self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample OHLCV data"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        # Generate realistic price data with trends and volatility
        returns = np.random.normal(0.0005, 0.015, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.normal(0, 0.002, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0.01, 0.005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0.01, 0.005, len(dates)))),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, len(dates)),
            'adj_close': prices
        })
        
        return data
    
    def get_historical_data(self, symbol, start_date, end_date):
        """Get historical data for symbol"""
        mask = (self.data['date'] >= start_date) & (self.data['date'] <= end_date)
        return self.data[mask].copy()
    
    def get_dividends(self, symbol, start_date, end_date):
        """Get dividend data"""
        return pd.DataFrame(columns=['date', 'dividend'])
    
    def get_splits(self, symbol, start_date, end_date):
        """Get split data"""
        return pd.DataFrame(columns=['date', 'split'])
