# trading_system/models/statistical_models/garch_model.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

from ...core.base_model import BaseModel
from ...core.exceptions import ModelError

class GARCHModel(BaseModel):
    """
    GARCH model for volatility forecasting
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # GARCH configuration
        self.order = config.get('order', (1, 1))  # (p, q)
        self.dist = config.get('dist', 'normal')  # 'normal', 't', 'skewt'
        self.vol = config.get('vol', 'GARCH')  # 'GARCH', 'EGARCH', 'TARCH'
        
        # Model components
        self.model = None
        self.model_fitted = None
        
        self.logger.info(f"GARCH model initialized with order {self.order}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """
        Train GARCH model on returns data
        """
        try:
            self.logger.info(f"Training GARCH model with {len(y)} samples")
            
            # Calculate returns if price data is provided
            if y.name == 'close' or 'price' in str(y.name).lower():
                returns = y.pct_change().dropna() * 100  # Percentage returns
                self.logger.info("Converted price data to returns for GARCH")
            else:
                returns = y.copy()
            
            returns = returns.fillna(method='ffill').fillna(0)
            
            # Remove zeros to avoid numerical issues
            returns = returns[returns != 0]
            
            if len(returns) < 50:
                raise ModelError("Insufficient data for GARCH modeling")
            
            # Fit GARCH model
            self.model = arch_model(
                returns, 
                vol=self.vol,
                p=self.order[0], 
                q=self.order[1],
                dist=self.dist
            )
            
            self.model_fitted = self.model.fit(disp='off')
            
            # Get conditional volatility
            conditional_vol = self.model_fitted.conditional_volatility
            
            # Calculate metrics
            metrics = self._calculate_metrics(returns, conditional_vol)
            
            self.is_trained = True
            self.logger.info(f"GARCH training completed. Log Likelihood: {metrics['log_likelihood']:.2f}")
            
            return metrics
            
        except Exception as e:
            raise ModelError(f"GARCH training failed: {e}")
    
    def predict(self, X: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """
        Predict volatility
        """
        if not self.is_trained:
            raise ModelError("Model not trained. Call train() first.")
        
        try:
            # GARCH volatility forecast
            forecast = self.model_fitted.forecast(horizon=horizon)
            volatility_forecast = np.sqrt(forecast.variance.values[-1, :])
            
            return volatility_forecast
            
        except Exception as e:
            raise ModelError(f"GARCH prediction failed: {e}")
    
    def get_conditional_volatility(self) -> pd.Series:
        """Get conditional volatility series"""
        if not self.is_trained:
            raise ModelError("Model not trained")
        
        return self.model_fitted.conditional_volatility
    
    def get_residuals(self) -> pd.Series:
        """Get model residuals"""
        if not self.is_trained:
            raise ModelError("Model not trained")
        
        return self.model_fitted.resid
    
    def get_standardized_residuals(self) -> pd.Series:
        """Get standardized residuals"""
        if not self.is_trained:
            raise ModelError("Model not trained")
        
        return self.model_fitted.std_resid
    
    def _calculate_metrics(self, returns: pd.Series, conditional_vol: pd.Series) -> Dict:
        """Calculate GARCH model metrics"""
        metrics = {
            'log_likelihood': self.model_fitted.loglikelihood,
            'aic': self.model_fitted.aic,
            'bic': self.model_fitted.bic,
            'order': self.order,
            'distribution': self.dist,
            'volatility_type': self.vol,
            'mean_volatility': conditional_vol.mean(),
            'volatility_std': conditional_vol.std()
        }
        
        return metrics
    
    def calculate_var(self, confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR)
        """
        if not self.is_trained:
            raise ModelError("Model not trained")
        
        try:
            # Get latest conditional volatility
            latest_vol = self.model_fitted.conditional_volatility.iloc[-1]
            
            # Calculate VaR based on distribution
            if self.dist == 'normal':
                from scipy.stats import norm
                var = norm.ppf(confidence_level) * latest_vol
            elif self.dist == 't':
                # Use student-t distribution
                from scipy.stats import t
                df = self.model_fitted.params['nu']  # degrees of freedom
                var = t.ppf(confidence_level, df) * latest_vol
            else:
                # Default to normal
                from scipy.stats import norm
                var = norm.ppf(confidence_level) * latest_vol
            
            return var
            
        except Exception as e:
            self.logger.warning(f"VaR calculation failed: {e}")
            return 0.0
    
    def get_summary(self) -> str:
        """Get model summary"""
        if not self.is_trained:
            raise ModelError("Model not trained")
        
        return str(self.model_fitted.summary())
    
    def save_model(self, path: str):
        """Save GARCH model"""
        if not self.is_trained:
            raise ModelError("No trained model to save")
        
        try:
            import joblib
            model_data = {
                'model_fitted': self.model_fitted,
                'order': self.order,
                'dist': self.dist,
                'vol': self.vol,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, path)
            self.logger.info(f"GARCH model saved to {path}")
            
        except Exception as e:
            raise ModelError(f"Failed to save GARCH model: {e}")
    
    def load_model(self, path: str):
        """Load GARCH model"""
        try:
            import joblib
            model_data = joblib.load(path)
            self.model_fitted = model_data['model_fitted']
            self.order = model_data['order']
            self.dist = model_data['dist']
            self.vol = model_data['vol']
            self.is_trained = model_data['is_trained']
            
            self.logger.info(f"GARCH model loaded from {path}")
            
        except Exception as e:
            raise ModelError(f"Failed to load GARCH model: {e}")
