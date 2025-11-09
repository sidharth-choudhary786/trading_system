# trading_system/models/statistical_models/arima_model.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

from ...core.base_model import BaseModel
from ...core.exceptions import ModelError

class ARIMAModel(BaseModel):
    """
    ARIMA model for time series forecasting
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # ARIMA configuration
        self.order = config.get('order', (1, 1, 1))  # (p, d, q)
        self.seasonal_order = config.get('seasonal_order', (0, 0, 0, 0))  # (P, D, Q, s)
        
        # Model components
        self.model = None
        self.model_fitted = None
        self.is_stationary = False
        
        self.logger.info(f"ARIMA model initialized with order {self.order}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """
        Train ARIMA model
        """
        try:
            self.logger.info(f"Training ARIMA model with {len(y)} samples")
            
            # Use y series for ARIMA (univariate)
            series = y.copy()
            series = series.fillna(method='ffill').fillna(method='bfill')
            
            # Check stationarity
            self.is_stationary = self._check_stationarity(series)
            self.logger.info(f"Series stationarity: {self.is_stationary}")
            
            # Determine optimal differencing if not specified
            if self.order[1] == 'auto':
                optimal_d = self._find_optimal_difference(series)
                self.order = (self.order[0], optimal_d, self.order[2])
                self.logger.info(f"Auto-determined d={optimal_d}")
            
            # Fit ARIMA model
            self.model = ARIMA(series, order=self.order, seasonal_order=self.seasonal_order)
            self.model_fitted = self.model.fit()
            
            # Generate in-sample predictions
            train_predictions = self.model_fitted.predict()
            
            # Calculate metrics
            metrics = self._calculate_metrics(series, train_predictions)
            
            self.is_trained = True
            self.logger.info(f"ARIMA training completed. AIC: {metrics['aic']:.2f}")
            
            return metrics
            
        except Exception as e:
            raise ModelError(f"ARIMA training failed: {e}")
    
    def predict(self, X: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """
        Make predictions with ARIMA
        """
        if not self.is_trained:
            raise ModelError("Model not trained. Call train() first.")
        
        try:
            # ARIMA doesn't use X features for prediction
            forecast = self.model_fitted.forecast(steps=steps)
            return forecast.values
            
        except Exception as e:
            raise ModelError(f"ARIMA prediction failed: {e}")
    
    def _check_stationarity(self, series: pd.Series) -> bool:
        """Check if time series is stationary using Augmented Dickey-Fuller test"""
        result = adfuller(series.dropna())
        p_value = result[1]
        return p_value < 0.05  # Stationary if p-value < 0.05
    
    def _find_optimal_difference(self, series: pd.Series, max_d: int = 5) -> int:
        """Find optimal differencing order"""
        original_series = series.copy()
        
        for d in range(max_d + 1):
            if d == 0:
                diff_series = original_series
            else:
                diff_series = original_series.diff(d).dropna()
            
            if len(diff_series) < 10:  # Need sufficient data
                return max(0, d - 1)
            
            result = adfuller(diff_series)
            p_value = result[1]
            
            if p_value < 0.05:
                return d
        
        return max_d  # Return maximum if no stationary series found
    
    def _calculate_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict:
        """Calculate ARIMA model metrics"""
        # Align actual and predicted
        aligned_actual = actual.iloc[len(actual) - len(predicted):]
        aligned_predicted = predicted
        
        mse = np.mean((aligned_actual - aligned_predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(aligned_actual - aligned_predicted))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'aic': self.model_fitted.aic,
            'bic': self.model_fitted.bic,
            'is_stationary': self.is_stationary,
            'order': self.order
        }
        
        return metrics
    
    def get_residuals(self) -> pd.Series:
        """Get model residuals"""
        if not self.is_trained:
            raise ModelError("Model not trained")
        
        return self.model_fitted.resid
    
    def get_summary(self) -> str:
        """Get model summary"""
        if not self.is_trained:
            raise ModelError("Model not trained")
        
        return str(self.model_fitted.summary())
    
    def save_model(self, path: str):
        """Save ARIMA model"""
        if not self.is_trained:
            raise ModelError("No trained model to save")
        
        try:
            import joblib
            model_data = {
                'model_fitted': self.model_fitted,
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'is_trained': self.is_trained,
                'is_stationary': self.is_stationary
            }
            joblib.dump(model_data, path)
            self.logger.info(f"ARIMA model saved to {path}")
            
        except Exception as e:
            raise ModelError(f"Failed to save ARIMA model: {e}")
    
    def load_model(self, path: str):
        """Load ARIMA model"""
        try:
            import joblib
            model_data = joblib.load(path)
            self.model_fitted = model_data['model_fitted']
            self.order = model_data['order']
            self.seasonal_order = model_data['seasonal_order']
            self.is_trained = model_data['is_trained']
            self.is_stationary = model_data['is_stationary']
            
            self.logger.info(f"ARIMA model loaded from {path}")
            
        except Exception as e:
            raise ModelError(f"Failed to load ARIMA model: {e}")
