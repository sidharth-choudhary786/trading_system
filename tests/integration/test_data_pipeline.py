# tests/integration/test_data_pipeline.py
"""
Integration tests for complete data pipeline
Tests data ingestion -> processing -> feature engineering flow
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_system.data.data_manager import DataManager
from trading_system.data.processing.imputer import DataImputer
from trading_system.data.processing.cleaner import DataCleaner
from trading_system.features.feature_engineer import FeatureEngineer

class TestDataPipeline:
    """Test complete data pipeline integration"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        data = pd.DataFrame({
            'date': dates,
            'open': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            'high': 105 + np.cumsum(np.random.normal(0, 1, len(dates))),
            'low': 95 + np.cumsum(np.random.normal(0, 1, len(dates))),
            'close': 102 + np.cumsum(np.random.normal(0, 1, len(dates))),
            'volume': np.random.randint(1000, 10000, len(dates)),
            'symbol': 'TEST'
        })
        return data
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            'data': {
                'sources': ['yahoo_finance'],
                'start_date': '2023-01-01',
                'end_date': '2023-12-31'
            },
            'features': {
                'technical_indicators': ['sma_20', 'rsi_14', 'macd']
            },
            'processing': {
                'imputation_method': 'linear',
                'cleaning_method': 'iqr'
            }
        }
    
    def test_complete_data_processing_flow(self, sample_data, config):
        """Test complete data processing pipeline"""
        # Initialize components
        imputer = DataImputer(config)
        cleaner = DataCleaner(config)
        feature_engineer = FeatureEngineer(config)
        
        # Test imputation
        data_with_missing = sample_data.copy()
        data_with_missing.loc[10:15, 'close'] = np.nan  # Add missing values
        
        imputed_data = imputer.impute(data_with_missing)
        assert imputed_data.isnull().sum().sum() == 0, "Imputation failed"
        
        # Test cleaning
        cleaned_data = cleaner.clean(imputed_data)
        assert len(cleaned_data) > 0, "Cleaning failed"
        
        # Test feature engineering
        features_data = feature_engineer.generate_features(cleaned_data)
        
        # Verify features were added
        original_columns = set(sample_data.columns)
        feature_columns = set(features_data.columns)
        new_features = feature_columns - original_columns
        
        assert len(new_features) > 0, "No features were generated"
        assert any('sma' in col for col in new_features), "SMA features missing"
        assert any('rsi' in col for col in new_features), "RSI features missing"
    
    def test_data_quality_through_pipeline(self, sample_data, config):
        """Test data quality metrics through pipeline"""
        feature_engineer = FeatureEngineer(config)
        
        # Generate features
        features_data = feature_engineer.generate_features(sample_data)
        
        # Check data quality
        assert len(features_data) > 0, "Pipeline produced empty data"
        assert 'close' in features_data.columns, "Close price missing"
        assert features_data['close'].isnull().sum() == 0, "Null values in close price"
        
        # Check feature statistics
        numeric_columns = features_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            assert features_data[col].notna().all(), f"Null values in {col}"
    
    def test_feature_consistency(self, sample_data, config):
        """Test feature calculation consistency"""
        feature_engineer = FeatureEngineer(config)
        
        # Generate features multiple times
        features1 = feature_engineer.generate_features(sample_data)
        features2 = feature_engineer.generate_features(sample_data)
        
        # Should produce identical results
        pd.testing.assert_frame_equal(features1, features2)
        
    def test_memory_usage(self, sample_data, config):
        """Test memory usage of feature engineering"""
        feature_engineer = FeatureEngineer(config)
        
        # Generate features for large dataset
        large_data = pd.concat([sample_data] * 10, ignore_index=True)
        
        features_data = feature_engineer.generate_features(large_data)
        
        # Check memory usage is reasonable
        memory_usage_mb = features_data.memory_usage(deep=True).sum() / 1024 / 1024
        assert memory_usage_mb < 100, f"High memory usage: {memory_usage_mb:.2f} MB"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
