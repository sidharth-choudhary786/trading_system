# trading_system/models/ml_models/lightgbm_model.py
import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

from ...core.exceptions import ModelError
from ..base_model import BaseModel

class LightGBMModel(BaseModel):
    """
    LightGBM implementation for trading predictions
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.model_config = config.get('lightgbm', {})
        self.model = None
        self.feature_importance_ = None
        
        # Training configuration
        self.early_stopping_rounds = self.model_config.get('early_stopping_rounds', 50)
        self.verbose = self.model_config.get('verbose', False)
        
        self.logger.info("LightGBM model initialized")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict:
        """
        Train LightGBM model
        """
        try:
            self.logger.info("Training LightGBM model...")
            
            # Prepare data
            X_train_clean = self._prepare_features(X_train)
            y_train_clean = y_train.values
            
            # Get model parameters
            params = self._get_model_params()
            
            # Create Dataset for LightGBM
            train_data = lgb.Dataset(X_train_clean, label=y_train_clean)
            
            # Validation data
            valid_sets = [train_data]
            valid_names = ['train']
            
            if X_val is not None and y_val is not None:
                X_val_clean = self._prepare_features(X_val)
                val_data = lgb.Dataset(X_val_clean, label=y_val.values, reference=train_data)
                valid_sets.append(val_data)
                valid_names.append('eval')
            
            # Train model
            self.model = lgb.train(
                params=params,
                train_set=train_data,
                num_boost_round=self.model_config.get('n_estimators', 1000),
                valid_sets=valid_sets,
                valid_names=valid_names,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=self.verbose
            )
            
            # Calculate feature importance
            self.feature_importance_ = dict(zip(
                X_train_clean.columns, 
                self.model.feature_importance(importance_type='split')
            ))
            
            # Calculate training metrics
            train_predictions = self.predict(X_train)
            metrics = self._calculate_metrics(y_train, train_predictions)
            
            self.is_trained = True
            self.logger.info("LightGBM model training completed")
            
            return metrics
            
        except Exception as e:
            raise ModelError(f"LightGBM training failed: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        """
        if not self.is_trained or self.model is None:
            raise ModelError("Model not trained. Call train() first.")
        
        try:
            X_clean = self._prepare_features(X)
            predictions = self.model.predict(X_clean)
            return predictions
            
        except Exception as e:
            raise ModelError(f"LightGBM prediction failed: {e}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (for classification)
        """
        if not self.is_trained or self.model is None:
            raise ModelError("Model not trained. Call train() first.")
        
        try:
            X_clean = self._prepare_features(X)
            
            # For binary classification, return probability of positive class
            if self.model_config.get('objective') in ['binary', 'binary_logloss']:
                probabilities = self.model.predict(X_clean)
                return probabilities
            else:
                raise ModelError("Probability prediction only available for classification tasks")
                
        except Exception as e:
            raise ModelError(f"LightGBM probability prediction failed: {e}")
    
    def _get_model_params(self) -> Dict:
        """Get LightGBM model parameters"""
        base_params = {
            'max_depth': self.model_config.get('max_depth', -1),  # -1 for no limit
            'learning_rate': self.model_config.get('learning_rate', 0.1),
            'num_leaves': self.model_config.get('num_leaves', 31),
            'subsample': self.model_config.get('subsample', 1.0),
            'colsample_bytree': self.model_config.get('colsample_bytree', 1.0),
            'reg_alpha': self.model_config.get('reg_alpha', 0),
            'reg_lambda': self.model_config.get('reg_lambda', 0),
            'min_child_samples': self.model_config.get('min_child_samples', 20),
            'min_child_weight': self.model_config.get('min_child_weight', 1e-3),
            'min_split_gain': self.model_config.get('min_split_gain', 0),
            'random_state': self.model_config.get('random_state', 42),
            'n_jobs': self.model_config.get('n_jobs', -1)
        }
        
        # Set objective based on problem type
        problem_type = self.model_config.get('problem_type', 'regression')
        if problem_type == 'regression':
            base_params['objective'] = 'regression'
            base_params['metric'] = 'rmse'
        elif problem_type == 'binary_classification':
            base_params['objective'] = 'binary'
            base_params['metric'] = 'binary_logloss'
        elif problem_type == 'multiclass_classification':
            base_params['objective'] = 'multiclass'
            base_params['metric'] = 'multi_logloss'
        
        return base_params
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction"""
        # Handle missing values
        X_clean = X.copy()
        
        # Select numeric columns only
        numeric_columns = X_clean.select_dtypes(include=[np.number]).columns
        X_clean = X_clean[numeric_columns]
        
        # Fill missing values with median
        for col in X_clean.columns:
            if X_clean[col].isnull().any():
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        return X_clean
    
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
                'model_config': self.model_config,
                'is_trained': self.is_trained
            }
            
            metadata_path = filepath.replace('.txt', '_metadata.pkl')
            joblib.dump(metadata, metadata_path)
            
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            raise ModelError(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        try:
            # Load model
            self.model = lgb.Booster(model_file=filepath)
            
            # Load metadata
            metadata_path = filepath.replace('.txt', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.feature_importance_ = metadata.get('feature_importance')
                self.model_config.update(metadata.get('model_config', {}))
                self.is_trained = metadata.get('is_trained', False)
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            raise ModelError(f"Failed to load model: {e}")
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """Perform cross-validation"""
        try:
            X_clean = self._prepare_features(X)
            params = self._get_model_params()
            
            # Create Dataset
            data = lgb.Dataset(X_clean, label=y.values)
            
            # Perform cross-validation
            cv_results = lgb.cv(
                params=params,
                train_set=data,
                num_boost_round=self.model_config.get('n_estimators', 1000),
                nfold=cv,
                early_stopping_rounds=self.early_stopping_rounds,
                stratified=False,
                verbose_eval=False,
                return_cvbooster=False
            )
            
            best_score = list(cv_results.values())[0][-1] if cv_results else 0
            
            return {
                'cv_results': cv_results,
                'best_iteration': len(list(cv_results.values())[0]) if cv_results else 0,
                'best_score': best_score
            }
            
        except Exception as e:
            raise ModelError(f"Cross-validation failed: {e}")
