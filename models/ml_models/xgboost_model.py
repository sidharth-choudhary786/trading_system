# trading_system/models/ml_models/xgboost_model.py
import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

from ...core.exceptions import ModelError
from ..base_model import BaseModel

class XGBoostModel(BaseModel):
    """
    XGBoost implementation for trading predictions
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.model_config = config.get('xgboost', {})
        self.model = None
        self.feature_importance_ = None
        
        # Training configuration
        self.early_stopping_rounds = self.model_config.get('early_stopping_rounds', 50)
        self.verbose = self.model_config.get('verbose', False)
        
        self.logger.info("XGBoost model initialized")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict:
        """
        Train XGBoost model
        """
        try:
            self.logger.info("Training XGBoost model...")
            
            # Prepare data
            X_train_clean = self._prepare_features(X_train)
            y_train_clean = y_train.values
            
            # Get model parameters
            params = self._get_model_params()
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train_clean, label=y_train_clean)
            
            # Validation data
            eval_set = []
            if X_val is not None and y_val is not None:
                X_val_clean = self._prepare_features(X_val)
                dval = xgb.DMatrix(X_val_clean, label=y_val.values)
                eval_set = [(dtrain, 'train'), (dval, 'eval')]
            else:
                eval_set = [(dtrain, 'train')]
            
            # Train model
            self.model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=self.model_config.get('n_estimators', 1000),
                evals=eval_set,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=self.verbose
            )
            
            # Calculate feature importance
            self.feature_importance_ = self.model.get_score(importance_type='weight')
            
            # Calculate training metrics
            train_predictions = self.predict(X_train)
            metrics = self._calculate_metrics(y_train, train_predictions)
            
            self.is_trained = True
            self.logger.info("XGBoost model training completed")
            
            return metrics
            
        except Exception as e:
            raise ModelError(f"XGBoost training failed: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        """
        if not self.is_trained or self.model is None:
            raise ModelError("Model not trained. Call train() first.")
        
        try:
            X_clean = self._prepare_features(X)
            dtest = xgb.DMatrix(X_clean)
            predictions = self.model.predict(dtest)
            return predictions
            
        except Exception as e:
            raise ModelError(f"XGBoost prediction failed: {e}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (for classification)
        """
        if not self.is_trained or self.model is None:
            raise ModelError("Model not trained. Call train() first.")
        
        try:
            X_clean = self._prepare_features(X)
            dtest = xgb.DMatrix(X_clean)
            
            # For binary classification, return probability of positive class
            if self.model_config.get('objective') in ['binary:logistic', 'binary:logitraw']:
                probabilities = self.model.predict(dtest, output_margin=False)
                return probabilities
            else:
                raise ModelError("Probability prediction only available for classification tasks")
                
        except Exception as e:
            raise ModelError(f"XGBoost probability prediction failed: {e}")
    
    def _get_model_params(self) -> Dict:
        """Get XGBoost model parameters"""
        base_params = {
            'max_depth': self.model_config.get('max_depth', 6),
            'learning_rate': self.model_config.get('learning_rate', 0.1),
            'subsample': self.model_config.get('subsample', 0.8),
            'colsample_bytree': self.model_config.get('colsample_bytree', 0.8),
            'reg_alpha': self.model_config.get('reg_alpha', 0),
            'reg_lambda': self.model_config.get('reg_lambda', 1),
            'min_child_weight': self.model_config.get('min_child_weight', 1),
            'gamma': self.model_config.get('gamma', 0),
            'random_state': self.model_config.get('random_state', 42)
        }
        
        # Set objective based on problem type
        problem_type = self.model_config.get('problem_type', 'regression')
        if problem_type == 'regression':
            base_params['objective'] = 'reg:squarederror'
            base_params['eval_metric'] = 'rmse'
        elif problem_type == 'binary_classification':
            base_params['objective'] = 'binary:logistic'
            base_params['eval_metric'] = 'logloss'
        elif problem_type == 'multiclass_classification':
            base_params['objective'] = 'multi:softprob'
            base_params['eval_metric'] = 'mlogloss'
        
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
            
            metadata_path = filepath.replace('.model', '_metadata.pkl')
            joblib.dump(metadata, metadata_path)
            
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            raise ModelError(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        try:
            # Load model
            self.model = xgb.Booster()
            self.model.load_model(filepath)
            
            # Load metadata
            metadata_path = filepath.replace('.model', '_metadata.pkl')
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
            
            # Create DMatrix
            data_dmatrix = xgb.DMatrix(X_clean, label=y.values)
            
            # Perform cross-validation
            cv_results = xgb.cv(
                params=params,
                dtrain=data_dmatrix,
                num_boost_round=self.model_config.get('n_estimators', 1000),
                nfold=cv,
                early_stopping_rounds=self.early_stopping_rounds,
                metrics=self._get_model_params().get('eval_metric', 'rmse'),
                as_pandas=True,
                verbose_eval=False
            )
            
            return {
                'cv_results': cv_results,
                'best_iteration': len(cv_results),
                'best_score': cv_results.iloc[-1][f'test-{params["eval_metric"]}-mean']
            }
            
        except Exception as e:
            raise ModelError(f"Cross-validation failed: {e}")
