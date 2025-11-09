# trading_system/tests/unit/test_features.py
"""
Unit tests for feature engineering module
"""
import pytest
import pandas as pd
import numpy as np
from trading_system.features.feature_engineer import FeatureEngineer
from trading_system.features.technical_indicators import TechnicalIndicators

class TestFeatureEngineer:
    """Test feature engineering functionality"""
    
    def test_feature_engineer_initialization(self):
        """Test feature engineer initialization"""
        config = {
            'features': {
                'technical_indicators': ['sma', 'rsi', 'macd']
            }
        }
        feature_engineer = FeatureEngineer(config)
        assert feature_engineer is not None
        assert 'sma' in feature_engineer.enabled_features
    
    def test_technical_feature_generation(self, sample_price_data):
        """Test technical feature generation"""
        config = {'features': {'technical_indicators': ['sma', 'rsi']}}
        feature_engineer = FeatureEngineer(config)
        
        features = feature_engineer._generate_technical_features(
            sample_price_data.set_index('date'), 
            'TEST'
        )
        
        # Check if SMA and RSI columns are added
        assert 'sma_20' in features.columns
        assert 'rsi_14' in features.columns
        assert len(features.columns) > len(sample_price_data.columns)
    
    def test_time_feature_generation(self, sample_price_data):
        """Test time-based feature generation"""
        config = {'features': {}}
        feature_engineer = FeatureEngineer(config)
        
        features = feature_engineer._generate_time_features(
            sample_price_data.set_index('date')
        )
        
        # Check time-based features
        assert 'day_of_week' in features.columns
        assert 'month' in features.columns
        assert 'is_weekend' in features.columns
    
    def test_feature_cleaning(self, sample_price_data):
        """Test feature cleaning functionality"""
        config = {'features': {'technical_indicators': ['sma_50']}}  # Will create NaNs
        feature_engineer = FeatureEngineer(config)
        
        # Add technical features that create NaNs
        features = feature_engineer._generate_technical_features(
            sample_price_data.set_index('date'),
            'TEST'
        )
        
        # Clean features
        cleaned_features = feature_engineer._clean_features(features)
        
        # Should have fewer rows due to NaN removal
        assert len(cleaned_features) <= len(features)
        assert cleaned_features.isna().sum().sum() == 0

class TestTechnicalIndicators:
    """Test technical indicators calculation"""
    
    def test_sma_calculation(self, sample_price_data):
        """Test SMA calculation"""
        tech_indicators = TechnicalIndicators({})
        data = sample_price_data.set_index('date')
        
        result = tech_indicators.add_sma(data)
        assert 'sma_20' in result.columns
        assert 'sma_50' in result.columns
        
        # Check SMA values
        assert not result['sma_20'].isna().all()
        assert result['sma_20'].iloc[-1] == pytest.approx(result['close'].iloc[-20:].mean())
    
    def test_rsi_calculation(self, sample_price_data):
        """Test RSI calculation"""
        tech_indicators = TechnicalIndicators({})
        data = sample_price_data.set_index('date')
        
        result = tech_indicators.add_rsi(data)
        assert 'rsi_14' in result.columns
        
        # RSI should be between 0 and 100
        rsi_values = result['rsi_14'].dropna()
        assert rsi_values.min() >= 0
        assert rsi_values.max() <= 100
    
    def test_macd_calculation(self, sample_price_data):
        """Test MACD calculation"""
        tech_indicators = TechnicalIndicators({})
        data = sample_price_data.set_index('date')
        
        result = tech_indicators.add_macd(data)
        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_histogram' in result.columns
