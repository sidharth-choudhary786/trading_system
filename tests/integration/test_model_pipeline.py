# tests/integration/test_model_pipeline.py
"""
Integration tests for complete model pipeline
Tests feature engineering -> model training -> prediction flow
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_system.features.feature_engineer import FeatureEngineer
from trading_system.models.model_manager import ModelManager
from trading_system.models.ml_models.xgboost_model import XGBoostModel

class TestModelPipeline:
    """Test complete model pipeline integration"""
    
    @pytest.fixture
    def sample_features_data(self):
        """Create sample features data for model testing"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        n_samples = len(dates)
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, n_samples)
        prices = 100 * np.cumprod(1 + returns)
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_samples),
            'symbol': 'TEST'
        })
        
        # Generate features
        config = {
            'features': {
                'technical_indicators': ['sma_20', 'sma_50', 'rsi_14', 'macd']
            }
        }
        
        feature_engineer = FeatureEngineer(config)
        features_data = feature_engineer.generate_features(data)
        
        return features_data
    
    @pytest.fixture
    def config(self):
        """Test configuration for models"""
        return {
            'models': {
                'active_models': ['xgboost'],
                'walk_forward': {
                    'window_size': 252,
                    'step_size': 63
                },
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            },
            'features': {
                'technical_indicators': ['sma_20', 'sma_50', 'rsi_14']
            }
        }
    
    def test_model_training_pipeline(self, sample_features_data, config):
        """Test complete model training pipeline"""
        # Initialize model manager
        model_manager = ModelManager(config)
        
        # Prepare target variable (next day return)
        sample_features_data['target'] = sample_features_data['close'].pct_change().shift(-1)
        sample_features_data = sample_features_data.dropna()
        
        # Test model training
        model_type = 'xgboost'
        model_id = model_manager.train_model(
            data=sample_features_data,
            model_type=model_type,
            target_column='target',
            feature_columns=['sma_20', 'sma_50', 'rsi_14', 'volume']
        )
        
        assert model_id is not None, "Model training failed"
        assert model_id in model_manager.models, "Model not stored in manager"
        
        # Test model prediction
        latest_data = sample_features_data.tail(10)
        predictions = model_manager.predict(
            model_id=model_id,
            data=latest_data,
            feature_columns=['sma_20', 'sma_50', 'rsi_14', 'volume']
        )
        
        assert len(predictions) == len(latest_data), "Prediction length mismatch"
        assert not np.any(np.isnan(predictions)), "Predictions contain NaN values"
    
    def test_walk_forward_analysis(self, sample_features_data, config):
        """Test walk-forward analysis integration"""
        model_manager = ModelManager(config)
        
        # Prepare data
        sample_features_data['target'] = sample_features_data['close'].pct_change().shift(-1)
        sample_features_data = sample_features_data.dropna()
        
        # Perform walk-forward analysis
        wf_results = model_manager.walk_forward_analysis(
            data=sample_features_data,
            model_type='xgboost',
            target_column='target',
            feature_columns=['sma_20', 'sma_50', 'rsi_14']
        )
        
        assert 'results' in wf_results, "Walk-forward results missing"
        assert 'metrics' in wf_results, "Walk-forward metrics missing"
        assert len(wf_results['results']) > 0, "No walk-forward results"
        
        # Check metrics
        metrics = wf_results['metrics']
        assert 'mean_accuracy' in metrics, "Accuracy metric missing"
        assert 'mean_mse' in metrics, "MSE metric missing"
    
    def test_model_persistence(self, sample_features_data, config):
        """Test model save/load functionality"""
        model_manager = ModelManager(config)
        
        # Prepare data and train model
        sample_features_data['target'] = sample_features_data['close'].pct_change().shift(-1)
        sample_features_data = sample_features_data.dropna()
        
        model_id = model_manager.train_model(
            data=sample_features_data,
            model_type='xgboost',
            target_column='target',
            feature_columns=['sma_20', 'sma_50', 'rsi_14']
        )
        
        # Save model
        save_path = f"/tmp/test_model_{model_id}.pkl"
        success = model_manager.save_model(model_id, save_path)
        assert success, "Model save failed"
        
        # Load model
        loaded_model_id = model_manager.load_model(save_path, 'xgboost')
        assert loaded_model_id is not None, "Model load failed"
        
        # Test loaded model predictions
        test_data = sample_features_data.tail(5)
        predictions = model_manager.predict(
            model_id=loaded_model_id,
            data=test_data,
            feature_columns=['sma_20', 'sma_50', 'rsi_14']
        )
        
        assert len(predictions) == len(test_data), "Loaded model prediction failed"
        
        # Cleanup
        import os
        if os.path.exists(save_path):
            os.remove(save_path)
    
    def test_multiple_model_types(self, sample_features_data, config):
        """Test integration with multiple model types"""
        model_manager = ModelManager(config)
        
        # Prepare data
        sample_features_data['target'] = sample_features_data['close'].pct_change().shift(-1)
        sample_features_data = sample_features_data.dropna()
        
        feature_columns = ['sma_20', 'sma_50', 'rsi_14', 'volume']
        
        # Test different model types
        model_types = ['xgboost']  # Add more as implemented
        
        for model_type in model_types:
            model_id = model_manager.train_model(
                data=sample_features_data,
                model_type=model_type,
                target_column='target',
                feature_columns=feature_columns
            )
            
            assert model_id is not None, f"{model_type} training failed"
            
            # Test prediction
            test_data = sample_features_data.tail(10)
            predictions = model_manager.predict(
                model_id=model_id,
                data=test_data,
                feature_columns=feature_columns
            )
            
            assert len(predictions) == len(test_data), f"{model_type} prediction failed"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
