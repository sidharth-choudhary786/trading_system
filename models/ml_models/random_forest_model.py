# trading_system/models/ml_models/random_forest_model.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

from ...core.base_model import BaseModel
from ...core.exceptions import ModelError

class RandomForestModel(BaseModel):
    """
    Random Forest implementation for trading signals
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.model_type = config.get('model_type', 'regressor')  # 'regressor' or 'classifier'
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', None)
        self.min_samples_split = config.get('min_samples_split', 2)
        self.min_samples_leaf = config.get('min_samples_leaf', 1)
        self.random_state = config.get('random_state', 42)
        
        # Initialize model
        if self.model_type == 'classifier':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        
        self.feature_importance_ = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """
        Train Random Forest model
        """
        try:
            self.logger.info(f"Training Random Forest {self.model_type} with {len(X)} samples")
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill')
            y = y.fillna(method='ffill').fillna(method='bfill')
            
            # Split data
            test_size = kwargs.get('test_size', 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, shuffle=False
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            train_predictions = self.model.predict(X_train)
            test_predictions = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_train, y_test, train_predictions, test_predictions)
            
            # Feature importance
            self.feature_importance_ = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.is_trained = True
            self.logger.info(f"Random Forest training completed. Test score: {metrics['test_score']:.4f}")
            
            return metrics
            
        except Exception as e:
            raise ModelError(f"Random Forest training failed: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        """
        if not self.is_trained:
            raise ModelError("Model not trained. Call train() first.")
        
        try:
            X = X.fillna(method='ffill').fillna(method='bfill')
            predictions = self.model.predict(X)
            return predictions
            
        except Exception as e:
            raise ModelError(f"Prediction failed: {e}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (for classifier only)
        """
        if self.model_type != 'classifier':
            raise ModelError("Probability prediction only available for classifiers")
        
        if not self.is_trained:
            raise ModelError("Model not trained. Call train() first.")
        
        try:
            X = X.fillna(method='ffill').fillna(method='bfill')
            probabilities = self.model.predict_proba(X)
            return probabilities
            
        except Exception as e:
            raise ModelError(f"Probability prediction failed: {e}")
    
    def _calculate_metrics(self, y_train: pd.Series, y_test: pd.Series, 
                         train_pred: np.ndarray, test_pred: np.ndarray) -> Dict:
        """Calculate model performance metrics"""
        if self.model_type == 'regressor':
            train_score = self.model.score(pd.DataFrame(y_train), train_pred)
            test_score = self.model.score(pd.DataFrame(y_test), test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            metrics = {
                'train_score': train_score,
                'test_score': test_score,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'model_type': 'regressor'
            }
        else:
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            metrics = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'model_type': 'classifier'
            }
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance"""
        if self.feature_importance_ is None:
            raise ModelError("Feature importance not available. Train model first.")
        
        return self.feature_importance_.head(top_n)
    
    def save_model(self, path: str):
        """Save model to disk"""
        if not self.is_trained:
            raise ModelError("No trained model to save")
        
        try:
            model_data = {
                'model': self.model,
                'config': {
                    'model_type': self.model_type,
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'min_samples_split': self.min_samples_split,
                    'min_samples_leaf': self.min_samples_leaf,
                    'random_state': self.random_state
                },
                'feature_importance': self.feature_importance_,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, path)
            self.logger.info(f"Model saved to {path}")
            
        except Exception as e:
            raise ModelError(f"Failed to save model: {e}")
    
    def load_model(self, path: str):
        """Load model from disk"""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.feature_importance_ = model_data['feature_importance']
            self.is_trained = model_data['is_trained']
            
            # Update config
            config = model_data['config']
            self.model_type = config['model_type']
            self.n_estimators = config['n_estimators']
            self.max_depth = config['max_depth']
            self.min_samples_split = config['min_samples_split']
            self.min_samples_leaf = config['min_samples_leaf']
            self.random_state = config['random_state']
            
            self.logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            raise ModelError(f"Failed to load model: {e}")
