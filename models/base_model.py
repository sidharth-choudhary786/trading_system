# trading_system/models/base_model.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import logging
from datetime import datetime

from ..core.exceptions import ModelError

class BaseModel(ABC):
    """
    Abstract base class for all trading models
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.is_trained = False
        self.training_date = None
        self.feature_columns = []
        self.target_column = 'target'
        
        # Model metadata
        self.model_id = f"{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_version = "1.0"
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """
        Train the model
        
        Args:
            X: Feature dataframe
            y: Target series
            **kwargs: Additional training parameters
            
        Returns:
            Training results and metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            X: Feature dataframe
            
        Returns:
            Predictions array
        """
        pass
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training/prediction
        
        Args:
            data: Raw data with features
            
        Returns:
            Processed feature dataframe
        """
        if data.empty:
            return data
        
        # Select only feature columns if specified
        if self.feature_columns:
            available_features = [col for col in self.feature_columns if col in data.columns]
            if available_features:
                return data[available_features]
        
        # If no specific features, use all numeric columns except target
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != self.target_column]
        
        return data[feature_cols]
    
    def create_target(self, data: pd.DataFrame, prediction_horizon: int = 1) -> pd.Series:
        """
        Create target variable for prediction
        
        Args:
            data: Data with price information
            prediction_horizon: Number of periods ahead to predict
            
        Returns:
            Target series
        """
        if 'close' not in data.columns:
            raise ModelError("Close price column required for target creation")
        
        # Calculate future returns
        future_returns = data['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
        
        # Create binary target (1 if positive return, 0 otherwise)
        target = (future_returns > 0).astype(int)
        
        return target
    
    def validate_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> bool:
        """
        Validate input data
        
        Args:
            X: Feature data
            y: Optional target data
            
        Returns:
            True if data is valid
        """
        if X.empty:
            raise ModelError("Feature data is empty")
        
        if X.isnull().any().any():
            self.logger.warning("Feature data contains missing values")
        
        if y is not None:
            if len(X) != len(y):
                raise ModelError("Feature and target data have different lengths")
            
            if y.isnull().any():
                self.logger.warning("Target data contains missing values")
        
        return True
    
    def save_model(self, filepath: Path) -> bool:
        """
        Save model to file
        
        Args:
            filepath: Path to save model
            
        Returns:
            True if successful
        """
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'config': self.config,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'is_trained': self.is_trained,
                'training_date': self.training_date,
                'model_id': self.model_id,
                'model_version': self.model_version
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: Path) -> bool:
        """
        Load model from file
        
        Args:
            filepath: Path to model file
            
        Returns:
            True if successful
        """
        try:
            if not filepath.exists():
                raise ModelError(f"Model file not found: {filepath}")
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.config = model_data['config']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            self.is_trained = model_data['is_trained']
            self.training_date = model_data['training_date']
            self.model_id = model_data['model_id']
            self.model_version = model_data['model_version']
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dictionary of feature names and importance scores
        """
        if not self.is_trained:
            raise ModelError("Model must be trained before getting feature importance")
        
        # Default implementation - should be overridden by subclasses
        return {f"feature_{i}": 0.0 for i in range(len(self.feature_columns))}
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ModelError("Model must be trained before evaluation")
        
        try:
            predictions = self.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.metrics import confusion_matrix, classification_report
            
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, predictions)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm.tolist(),
                'predictions_count': len(predictions),
                'positive_predictions': int(predictions.sum()),
                'actual_positives': int(y_test.sum())
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {e}")
            return {}
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_type': self.__class__.__name__,
            'model_id': self.model_id,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'training_date': self.training_date,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'feature_count': len(self.feature_columns)
        }
