# trading_system/tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the trading_system package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def sample_price_data():
    """Sample price data for testing"""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(150, 250, 100),
        'low': np.random.uniform(50, 150, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.randint(1000, 10000, 100),
        'adj_close': np.random.uniform(100, 200, 100)
    })
    return data

@pytest.fixture
def sample_portfolio():
    """Sample portfolio for testing"""
    return {
        'portfolio_value': 1000000,
        'cash': 100000,
        'positions': {'RELIANCE': 100, 'TCS': 50},
        'current_prices': {'RELIANCE': 1500, 'TCS': 3200}
    }

@pytest.fixture
def sample_order():
    """Sample order for testing"""
    from trading_system.core.types import Order, OrderSide, OrderType
    
    return Order(
        order_id="TEST_ORDER_001",
        symbol="RELIANCE",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=10
    )

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'mode': 'backtest',
        'initial_capital': 1000000,
        'commission': 0.001,
        'slippage': 0.001,
        'data': {
            'sources': ['yahoo'],
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        }
    }

@pytest.fixture
def mock_market_data():
    """Mock market data for testing"""
    return {
        'RELIANCE': {
            'current_price': 1500,
            'avg_daily_volume': 1000000,
            'returns': pd.Series(np.random.normal(0, 0.02, 100))
        },
        'TCS': {
            'current_price': 3200,
            'avg_daily_volume': 500000,
            'returns': pd.Series(np.random.normal(0, 0.015, 100))
        }
    }
