# trading_system/models/ml_models/catboost_model.py
from catboost import CatBoostRegressor, CatBoostClassifier
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

from ...core.exceptions import ModelError
from ..base_model import BaseModel

class CatBoostModel(BaseModel):
    """
    CatBoost implementation for trading predictions
    Excellent for handling categorical features
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.model_config = config.get('catboost', {})
        self.model = None
        self.feature_importance_ = None
        self.categorical_features = []
        
        # Training configuration
        self.early_stopping_rounds = self.model_config.get('early_stopping_rounds', 50)
        self.verbose = self.model_config.get('verbose', False)
        
        self.logger.info("CatBoost model initialized")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict:
        """
        Train CatBoost model
        """
        try:
            self.logger.info("Training CatBoost model...")
            
            # Prepare data and identify categorical features
            X_train_clean, cat_features = self._prepare_features(X_train)
            self.categorical_features = cat_features
            
            # Get model parameters
            params = self._get_model_params()
            
            # Initialize model based on problem type
            problem_type = self.model_config.get('problem_type', 'regression')
            if problem_type == 'regression':
                self.model = CatBoostRegressor(**params)
            else:
                self.model = CatBoostClassifier(**params)
            
            # Fit model
            if X_val is not None and y_val is not None:
                X_val_clean, _ = self._prepare_features(X_val)
                self.model.fit(
                    X_train_clean, y_train,
                    cat_features=cat_features,
                    eval_set=(X_val_clean, y_val),
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=self.verbose
                )
            else:
                self.model.fit(
                    X_train_clean, y_train,
                    cat_features=cat_features,
                    verbose=self.verbose
                )
            
            # Calculate feature importance
            self.feature_importance_ = dict(zip(
                X_train_clean.columns, 
                self.model.get_feature_importance()
            ))
            
            # Calculate training metrics
            train_predictions = self.predict(X_train)
            metrics = self._calculate_metrics(y_train, train_predictions)
            
            self.is_trained = True
            self.logger.info("CatBoost model training completed")
            
            return metrics
            
        except Exception as e:
            raise ModelError(f"CatBoost training failed: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        """
        if not self.is_trained or self.model is None:
            raise ModelError("Model not trained. Call train() first.")
        
        try:
            X_clean, _ = self._prepare_features(X)
            predictions = self.model.predict(X_clean)
            return predictions
            
        except Exception as e:
            raise ModelError(f"CatBoost prediction failed: {e}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (for classification)
        """
        if not self.is_trained or self.model is None:
            raise ModelError("Model not trained. Call train() first.")
        
        try:
            X_clean, _ = self._prepare_features(X)
            
            # Check if model supports probability prediction
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_clean)
                return probabilities
            else:
                raise ModelError("Probability prediction not available for this model type")
                
        except Exception as e:
            raise ModelError(f"CatBoost probability prediction failed: {e}")
    
    def _get_model_params(self) -> Dict:
        """Get CatBoost model parameters"""
        base_params = {
            'iterations': self.model_config.get('n_estimators', 1000),
            'depth': self.model_config.get('max_depth', 6),
            'learning_rate': self.model_config.get('learning_rate', 0.1),
            'l2_leaf_reg': self.model_config.get('reg_lambda', 3),
            'random_seed': self.model_config.get('random_state', 42),
            'loss_function': self._get_loss_function(),
            'verbose': self.verbose,
            'thread_count': self.model_config.get('n_jobs', -1)
        }
        
        # Remove None values
        base_params = {k: v for k, v in base_params.items() if v is not None}
        
        return base_params
    
    def _get_loss_function(self) -> str:
        """Get appropriate loss function based on problem type"""
        problem_type = self.model_config.get('problem_type', 'regression')
        
        if problem_type == 'regression':
            return 'RMSE'
        elif problem_type == 'binary_classification':
            return 'Logloss'
        elif problem_type == 'multiclass_classification':
            return 'MultiClass'
        else:
            return 'RMSE'
    
    def _prepare_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """Prepare features and identify categorical features"""
        X_clean = X.copy()
        
        # Identify categorical features
        categorical_features = []
        for col in X_clean.columns:
            # Check if column is categorical (object type or low cardinality)
            if (X_clean[col].dtype == 'object' or 
                (X_clean[col].nunique() <= 50 and X_clean[col].dtype != float)):
                categorical_features.append(col)
                # Convert to string for CatBoost
                X_clean[col] = X_clean[col].astype(str)
        
        # Fill missing values
        for col in X_clean.columns:
            if X_clean[col].isnull().any():
                if col in categorical_features:
                    X_clean[col] = X_clean[col].fillna('missing')
                else:
                    X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        return X_clean, categorical_features
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """Calculate performance metrics"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # For classification tasks
        problem_type = self.model_config.get('problem_type', 'regression')
        if problem_type in ['binary_classification', 'multiclass_classification']:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            y_pred_class = (y_pred > 0.5).astype(int) if problem_type == 'binary_classification' else np.argmax(y_pred, axis=1)
            
            metrics.update({
                'accuracy': accuracy_score(y_true, y_pred_class),
                'precision': precision_score(y_true, y_pred_class, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred_class, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred_class, average='weighted', zero_division=0)
            })
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance"""
        if self.feature_importance_ is None:
            raise ModelError("Feature importance not available. Train model first.")
        
        importance_df = pd.DataFrame({
            'feature': list(self.feature_importance_.keys()),
            'importance': list(self.feature_importance_.values())
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if not self.is_trained or self.model is None:
            raise ModelError("No trained model to save")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            self.model.save_model(filepath)
            
            # Save metadata
            metadata = {
                'feature_importance': self.feature_importance_,
                'categorical_features': self.categorical_features,
                'model_config': self.model_config,
                'is_trained': self.is_trained
            }
            
            metadata_path = filepath.replace('.cbm', '_metadata.pkl')
            joblib.dump(metadata, metadata_path)
            
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            raise ModelError(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        try:
            # Determine model type from config or file
            problem_type = self.model_config.get('problem_type', 'regression')
            
            if problem_type == 'regression':
                self.model = CatBoostRegressor()
            else:
                self.model = CatBoostClassifier()
            
            # Load model
            self.model.load_model(filepath)
            
            # Load metadata
            metadata_path = filepath.replace('.cbm', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.feature_importance_ = metadata.get('feature_importance')
                self.categorical_features = metadata.get('categorical_features', [])
                self.model_config.update(metadata.get('model_config', {}))
                self.is_trained = metadata.get('is_trained', False)
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            raise ModelError(f"Failed to load model: {e}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance with feature names"""
        if self.feature_importance_ is None:
            raise ModelError("Feature importance not available. Train model first.")
        
        importance_df = pd.DataFrame({
            'feature': list(self.feature_importance_.keys()),
            'importance': list(self.feature_importance_.values())
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
