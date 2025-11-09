# trading_system/tests/integration/test_data_pipeline.py
"""
Integration tests for data pipeline
"""
import pytest
import pandas as pd
from trading_system.data.data_manager import DataManager
from trading_system.data.processing.imputer import DataImputer
from trading_system.data.processing.cleaner import DataCleaner

class TestDataPipeline:
    """Integration tests for data pipeline"""
    
    def test_complete_data_processing_flow(self, sample_price_data):
        """Test complete data processing pipeline"""
        # Create data with some missing values and outliers
        data = sample_price_data.copy()
        
        # Introduce some missing values
        data.loc[10:15, 'volume'] = None
        data.loc[20, 'close'] = None
        
        # Introduce an outlier
        data.loc[30, 'close'] = data['close'].mean() + 10 * data['close'].std()
        
        # Initialize processors
        imputer = DataImputer({'imputation_method': 'linear'})
        cleaner = DataCleaner({'cleaning_method': 'iqr'})
        
        # Apply processing pipeline
        imputed_data = imputer.impute(data)
        cleaned_data = cleaner.clean(imputed_data)
        
        # Verify results
        assert cleaned_data.isnull().sum().sum() == 0
        assert len(cleaned_data) <= len(data)  # Some outliers might be removed
    
    def test_data_quality_assessment_integration(self, sample_price_data):
        """Test data quality assessment integration"""
        from trading_system.data.processing.quality_assessor import DataQualityAssessor
        
        quality_assessor = DataQualityAssessor({})
        
        quality_report = quality_assessor.assess_quality(sample_price_data, 'TEST_SYMBOL')
        
        # Verify report structure
        assert 'quality_scores' in quality_report
        assert 'overall_score' in quality_report
        assert 'quality_grade' in quality_report
        assert 'issues_found' in quality_report
        
        # Scores should be between 0 and 1
        assert 0 <= quality_report['overall_score'] <= 1
        assert quality_report['quality_grade'] in ['A', 'B', 'C', 'D', 'F']
