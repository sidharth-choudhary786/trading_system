# trading_system/tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def sample_price_data():
    """Sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
        'high': 105 + np.cumsum(np.random.normal(0, 1, len(dates))),
        'low': 95 + np.cumsum(np.random.normal(0, 1, len(dates))),
        'close': 102 + np.cumsum(np.random.normal(0, 1, len(dates))),
        'volume': np.random.randint(1000, 10000, len(dates)),
        'adj_close': 102 + np.cumsum(np.random.normal(0, 1, len(dates)))
    })
    return data

@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing"""
    return {
        'initial_capital': 100000,
        'positions': {'RELIANCE': 100, 'TCS': 50},
        'current_cash': 50000,
        'portfolio_value': 150000
    }

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'mode': 'backtest',
        'initial_capital': 100000,
        'commission': 0.001,
        'slippage': 0.001,
        'data': {
            'sources': ['mock'],
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        }
    }
