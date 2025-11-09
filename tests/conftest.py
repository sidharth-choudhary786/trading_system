# trading_system/tests/conftest.py
"""
Pytest configuration and fixtures for trading system tests
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_price_data():
    """Sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 200, len(dates)),
        'high': np.random.uniform(110, 210, len(dates)),
        'low': np.random.uniform(90, 190, len(dates)),
        'close': np.random.uniform(95, 205, len(dates)),
        'volume': np.random.randint(1000, 100000, len(dates)),
        'adj_close': np.random.uniform(95, 205, len(dates))
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
def sample_order():
    """Sample order for testing"""
    from trading_system.core.types import Order, OrderType, OrderSide
    
    return Order(
        order_id="TEST_ORDER_001",
        symbol="RELIANCE",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=100,
        price=2500.0
    )

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'mode': 'backtest',
        'initial_capital': 100000,
        'commission': 0.001,
        'slippage': 0.001,
        'data_sources': ['yahoo'],
        'universe': ['RELIANCE', 'TCS', 'INFY']
    }
