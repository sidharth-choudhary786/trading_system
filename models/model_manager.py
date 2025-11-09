# trading_system/models/model_manager.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json

from .base_model import BaseModel
from .ml_models.xgboost_model import XGBoostModel
from .ml_models.lightgbm_model import LightGBMModel
from .ml_models.catboost_model import CatBoostModel
from .ml_models.random_forest_model import RandomForestModel
from .statistical_models.arima_model import ARIMAModel
from .statistical_models.garch_model import GARCHModel
from .statistical_models.prophet_model import ProphetModel
from ..core.exceptions import ModelError

class ModelManager:
    """
    Manages multiple models and orchestrates training, prediction, and evaluation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model registry
        self.models = {}
        self.active_models = []
        self.model_performance = {}
        
        # Initialize models from config
        self._initialize_models()
        
        self.logger.info("Model Manager initialized")
    
    def _initialize_models(self):
        """Initialize models based on configuration"""
        model_configs = self.config.get('models', {})
        active_models = model_configs.get('active_models', [])
        
        model_constructors = {
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'catboost': CatBoostModel,
            'random_forest': RandomForestModel,
            'arima': ARIMAModel,
            'garch': GARCHModel,
            'prophet': ProphetModel
        }
        
        for model_name in active_models:
            if model_name in model_constructors:
                try:
                    model_config = model_configs.get(model_name, {})
                    model = model_constructors[model_name](model_config)
                    self.models[model_name] = model
                    self.active_models.append(model_name)
                    self.logger.info(f"Initialized {model_name} model")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {model_name}: {e}")
            else:
                self.logger.warning(f"Unknown model type: {model_name}")
    
    def train_models(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        models: Optional[List[str]] = None,
        validation_data: Optional[Tuple] = None
    ) -> Dict:
        """
        Train multiple models
        
        Args:
            X: Training features
            y: Training targets
            models: List of model names to train (None for all active models)
            validation_data: Optional (X_val, y_val) for validation
            
        Returns:
            Dictionary of training results
        """
        if models is None:
            models = self.active_models
        
        training_results = {}
        
        for model_name in models:
            if model_name not in self.models:
                self.logger.warning(f"Model {model_name} not found")
                continue
            
            try:
                self.logger.info(f"Training {model_name} model...")
                model = self.models[model_name]
                
                # Prepare features
                X_processed = model.prepare_features(X)
                
                # Train model
                train_results = model.train(X_processed, y)
                
                # Validate if validation data provided
                if validation_data:
                    X_val, y_val = validation_data
                    X_val_processed = model.prepare_features(X_val)
                    val_metrics = model.evaluate(X_val_processed, y_val)
                    train_results['validation_metrics'] = val_metrics
                
                training_results[model_name] = train_results
                self.logger.info(f"Completed training {model_name}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                training_results[model_name] = {'error': str(e)}
        
        return training_results
    
    def predict_ensemble(
        self, 
        X: pd.DataFrame,
        models: Optional[List[str]] = None,
        method: str = 'average'
    ) -> Dict:
        """
        Generate ensemble predictions from multiple models
        
        Args:
            X: Feature data
            models: List of model names to use
            method: Ensemble method ('average', 'weighted', 'voting')
            
        Returns:
            Dictionary with ensemble predictions and individual model predictions
        """
        if models is None:
            models = [m for m in self.active_models if self.models[m].is_trained]
        
        if not models:
            raise ModelError("No trained models available for prediction")
        
        individual_predictions = {}
        model_weights = {}
        
        for model_name in models:
            try:
                model = self.models[model_name]
                X_processed = model.prepare_features(X)
                predictions = model.predict(X_processed)
                individual_predictions[model_name] = predictions
                
                # Set weights based on method
                if method == 'weighted':
                    # Use model performance as weights (simplified)
                    model_weights[model_name] = 1.0 / len(models)
                else:
                    model_weights[model_name] = 1.0
                    
            except Exception as e:
                self.logger.error(f"Error getting predictions from {model_name}: {e}")
        
        if not individual_predictions:
            raise ModelError("No models produced valid predictions")
        
        # Combine predictions based on method
        if method in ['average', 'weighted']:
            # Weighted average of probabilities
            ensemble_pred = np.zeros_like(next(iter(individual_predictions.values())))
            total_weight = 0
            
            for model_name, pred in individual_predictions.items():
                weight = model_weights[model_name]
                ensemble_pred += pred * weight
                total_weight += weight
            
            ensemble_pred /= total_weight
            
        elif method == 'voting':
            # Majority voting for classification
            all_predictions = np.array(list(individual_predictions.values()))
            ensemble_pred = np.round(np.mean(all_predictions, axis=0))
        
        else:
            raise ModelError(f"Unknown ensemble method: {method}")
        
        return {
            'ensemble_predictions': ensemble_pred,
            'individual_predictions': individual_predictions,
            'model_weights': model_weights,
            'ensemble_method': method
        }
    
    def evaluate_models(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        models: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate multiple models on test data
        
        Args:
            X_test: Test features
            y_test: Test targets
            models: List of model names to evaluate
            
        Returns:
            Dictionary of evaluation results
        """
        if models is None:
            models = [m for m in self.active_models if self.models[m].is_trained]
        
        evaluation_results = {}
        
        for model_name in models:
            try:
                model = self.models[model_name]
                X_processed = model.prepare_features(X_test)
                metrics = model.evaluate(X_processed, y_test)
                
                evaluation_results[model_name] = metrics
                self.model_performance[model_name] = metrics
                
                self.logger.info(f"Evaluation for {model_name}: Accuracy = {metrics.get('accuracy', 0):.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                evaluation_results[model_name] = {'error': str(e)}
        
        return evaluation_results
    
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, BaseModel]:
        """
        Get the best performing model based on specified metric
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, model_instance)
        """
        best_model_name = None
        best_score = -np.inf
        
        for model_name, performance in self.model_performance.items():
            score = performance.get(metric, -np.inf)
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        if best_model_name is None:
            raise ModelError("No model performance data available")
        
        return best_model_name, self.models[best_model_name]
    
    def save_models(self, directory: Path) -> bool:
        """
        Save all models to directory
        
        Args:
            directory: Directory to save models
            
        Returns:
            True if successful
        """
        try:
            directory.mkdir(parents=True, exist_ok=True)
            
            for model_name, model in self.models.items():
                if model.is_trained:
                    model_file = directory / f"{model_name}_{model.model_id}.pkl"
                    model.save_model(model_file)
            
            # Save model manager state
            state = {
                'active_models': self.active_models,
                'model_performance': self.model_performance,
                'saved_date': datetime.now().isoformat()
            }
            
            state_file = directory / "model_manager_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"All models saved to {directory}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self, directory: Path) -> bool:
        """
        Load models from directory
        
        Args:
            directory: Directory containing saved models
            
        Returns:
            True if successful
        """
        try:
            if not directory.exists():
                raise ModelError(f"Model directory not found: {directory}")
            
            # Load model files
            model_files = list(directory.glob("*.pkl"))
            
            for model_file in model_files:
                try:
                    # Extract model name from filename
                    model_name = model_file.stem.split('_')[0]
                    
                    if model_name in self.models:
                        self.models[model_name].load_model(model_file)
                        self.logger.info(f"Loaded {model_name} from {model_file}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading model from {model_file}: {e}")
            
            # Load manager state
            state_file = directory / "model_manager_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.active_models = state.get('active_models', [])
                self.model_performance = state.get('model_performance', {})
            
            self.logger.info(f"Models loaded from {directory}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of all models
        
        Returns:
            Dictionary with model summaries
        """
        summary = {
            'total_models': len(self.models),
            'active_models': self.active_models,
            'trained_models': [],
            'model_details': {}
        }
        
        for model_name, model in self.models.items():
            model_info = model.get_model_info()
            model_info['performance'] = self.model_performance.get(model_name, {})
            
            summary['model_details'][model_name] = model_info
            
            if model.is_trained:
                summary['trained_models'].append(model_name)
        
        return summary
    
    def retrain_models(
        self,
        new_data: pd.DataFrame,
        retrain_threshold: int = 100,
        models: Optional[List[str]] = None
    ) -> Dict:
        """
        Retrain models with new data if sufficient new samples available
        
        Args:
            new_data: New data for retraining
            retrain_threshold: Minimum new samples required for retraining
            models: List of models to retrain
            
        Returns:
            Dictionary of retraining results
        """
        if len(new_data) < retrain_threshold:
            self.logger.info(f"Insufficient new data ({len(new_data)} samples) for retraining")
            return {}
        
        if models is None:
            models = self.active_models
        
        # Create target from new data
        target = None
        for model_name in models:
            if self.models[model_name].is_trained:
                try:
                    target = self.models[model_name].create_target(new_data)
                    break
                except Exception:
                    continue
        
        if target is None or target.isnull().all():
            self.logger.warning("Could not create target from new data")
            return {}
        
        # Retrain models
        retraining_results = {}
        
        for model_name in models:
            try:
                model = self.models[model_name]
                if model.is_trained:
                    X_processed = model.prepare_features(new_data)
                    
                    # Combine with existing training logic if needed
                    # For simplicity, we'll do a full retrain here
                    train_results = model.train(X_processed, target)
                    retraining_results[model_name] = train_results
                    
                    self.logger.info(f"Retrained {model_name} with {len(new_data)} new samples")
                
            except Exception as e:
                self.logger.error(f"Error retraining {model_name}: {e}")
                retraining_results[model_name] = {'error': str(e)}
        
        return retraining_results
