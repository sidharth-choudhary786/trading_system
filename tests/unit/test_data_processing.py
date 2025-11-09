# trading_system/tests/unit/test_features.py
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from trading_system.features.technical_indicators import TechnicalIndicators
from trading_system.features.feature_engineer import FeatureEngineer

class TestFeatures:
    """Test feature engineering"""
    
    @pytest.fixture
    def sample_price_data(self):
        """Sample price data for feature testing"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'open': 100 + np.cumsum(np.random.normal(0, 1, 100)),
            'high': 105 + np.cumsum(np.random.normal(0, 1, 100)),
            'low': 95 + np.cumsum(np.random.normal(0, 1, 100)),
            'close': 102 + np.cumsum(np.random.normal(0, 1, 100)),
            'volume': np.random.randint(1000, 10000, 100)
        })
        return data
    
    def test_sma_calculation(self, sample_price_data):
        """Test SMA calculation"""
        indicators = TechnicalIndicators({})
        data = sample_price_data.set_index('date')
        
        result = indicators.add_sma(data, period=20)
        
        assert 'sma_20' in result.columns
        assert not result['sma_20'].isna().all()
        
        # SMA should be less volatile than price
        assert result['sma_20'].std() <= result['close'].std()
    
    def test_rsi_calculation(self, sample_price_data):
        """Test RSI calculation"""
        indicators = TechnicalIndicators({})
        data = sample_price_data.set_index('date')
        
        result = indicators.add_rsi(data, period=14)
        
        assert 'rsi_14' in result.columns
        # RSI should be between 0 and 100
        assert result['rsi_14'].max() <= 100
        assert result['rsi_14'].min() >= 0
    
    def test_feature_engineer_integration(self, sample_price_data):
        """Test complete feature engineering pipeline"""
        config = {
            'features': {
                'technical_indicators': ['sma', 'rsi']
            }
        }
        engineer = FeatureEngineer(config)
        
        result = engineer.generate_features(sample_price_data)
        
        # Should have original columns plus features
        assert 'close' in result.columns
        assert 'sma_20' in result.columns
        assert 'rsi_14' in result.columns
        
        # Should handle missing values from indicators
        assert not result['sma_20'].isna().all()
