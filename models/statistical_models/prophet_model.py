# trading_system/models/statistical_models/prophet_model.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

from ...core.base_model import BaseModel
from ...core.exceptions import ModelError

class ProphetModel(BaseModel):
    """
    Facebook Prophet model for time series forecasting
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Prophet configuration
        self.growth = config.get('growth', 'linear')
        self.seasonality_mode = config.get('seasonality_mode', 'additive')
        self.yearly_seasonality = config.get('yearly_seasonality', True)
        self.weekly_seasonality = config.get('weekly_seasonality', True)
        self.daily_seasonality = config.get('daily_seasonality', False)
        self.changepoint_prior_scale = config.get('changepoint_prior_scale', 0.05)
        self.seasonality_prior_scale = config.get('seasonality_prior_scale', 10.0)
        
        # Model components
        self.model = None
        self.forecast = None
        
        self.logger.info("Prophet model initialized")
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """
        Train Prophet model
        """
        try:
            self.logger.info(f"Training Prophet model with {len(y)} samples")
            
            # Prepare data for Prophet
            prophet_data = self._prepare_prophet_data(y, X)
            
            # Initialize and configure Prophet model
            self.model = Prophet(
                growth=self.growth,
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale
            )
            
            # Add additional regressors if provided
            if X is not None and not X.empty:
                for column in X.columns:
                    self.model.add_regressor(column)
            
            # Fit model
            self.model.fit(prophet_data)
            
            # Generate in-sample forecast
            future = self.model.make_future_dataframe(periods=0)
            self.forecast = self.model.predict(future)
            
            # Calculate metrics
            metrics = self._calculate_metrics(prophet_data, self.forecast)
            
            self.is_trained = True
            self.logger.info(f"Prophet training completed. RMSE: {metrics['rmse']:.4f}")
            
            return metrics
            
        except Exception as e:
            raise ModelError(f"Prophet training failed: {e}")
    
    def predict(self, X: pd.DataFrame, periods: int = 30) -> np.ndarray:
        """
        Make predictions with Prophet
        """
        if not self.is_trained:
            raise ModelError("Model not trained. Call train() first.")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods)
            
            # Add regressors to future dataframe if provided
            if X is not None and not X.empty:
                # For simplicity, use the last available values for future regressors
                last_values = X.iloc[-1:]
                for column in X.columns:
                    future[column] = last_values[column].values[0]
            
            # Make forecast
            forecast = self.model.predict(future)
            
            # Return only the future predictions
            future_predictions = forecast[['ds', 'yhat']].tail(periods)
            return future_predictions['yhat'].values
            
        except Exception as e:
            raise ModelError(f"Prophet prediction failed: {e}")
    
    def _prepare_prophet_data(self, y: pd.Series, X: pd.DataFrame = None) -> pd.DataFrame:
        """Prepare data for Prophet format"""
        # Create base dataframe
        if isinstance(y.index, pd.DatetimeIndex):
            dates = y.index
        else:
            # Generate dates if not available
            start_date = pd.Timestamp('2000-01-01')
            dates = pd.date_range(start=start_date, periods=len(y), freq='D')
        
        prophet_data = pd.DataFrame({
            'ds': dates,
            'y': y.values
        })
        
        # Add additional regressors
        if X is not None and not X.empty:
            for column in X.columns:
                prophet_data[column] = X[column].values
        
        return prophet_data
    
    def _calculate_metrics(self, actual_data: pd.DataFrame, forecast_data: pd.DataFrame) -> Dict:
        """Calculate Prophet model metrics"""
        # Merge actual and forecast
        merged = actual_data.merge(forecast_data[['ds', 'yhat']], on='ds', how='left')
        
        # Calculate error metrics
        y_true = merged['y']
        y_pred = merged['yhat']
        
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'growth': self.growth,
            'seasonality_mode': self.seasonality_mode
        }
        
        return metrics
    
    def plot_components(self):
        """Plot forecast components"""
        if not self.is_trained:
            raise ModelError("Model not trained")
        
        try:
            fig = self.model.plot_components(self.forecast)
            return fig
        except Exception as e:
            self.logger.warning(f"Component plotting failed: {e}")
            return None
    
    def plot_forecast(self):
        """Plot forecast"""
        if not self.is_trained:
            raise ModelError("Model not trained")
        
        try:
            fig = self.model.plot(self.forecast)
            return fig
        except Exception as e:
            self.logger.warning(f"Forecast plotting failed: {e}")
            return None
    
    def get_changepoints(self) -> pd.DataFrame:
        """Get changepoints detected by Prophet"""
        if not self.is_trained:
            raise ModelError("Model not trained")
        
        changepoints = self.model.changepoints
        trends = self.forecast.loc[self.forecast['ds'].isin(changepoints), 'trend']
        
        return pd.DataFrame({
            'changepoint': changepoints,
            'trend': trends.values
        })
    
    def save_model(self, path: str):
        """Save Prophet model"""
        if not self.is_trained:
            raise ModelError("No trained model to save")
        
        try:
            import joblib
            model_data = {
                'model': self.model,
                'forecast': self.forecast,
                'config': {
                    'growth': self.growth,
                    'seasonality_mode': self.seasonality_mode,
                    'yearly_seasonality': self.yearly_seasonality,
                    'weekly_seasonality': self.weekly_seasonality,
                    'daily_seasonality': self.daily_seasonality,
                    'changepoint_prior_scale': self.changepoint_prior_scale,
                    'seasonality_prior_scale': self.seasonality_prior_scale
                },
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, path)
            self.logger.info(f"Prophet model saved to {path}")
            
        except Exception as e:
            raise ModelError(f"Failed to save Prophet model: {e}")
    
    def load_model(self, path: str):
        """Load Prophet model"""
        try:
            import joblib
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.forecast = model_data['forecast']
            self.is_trained = model_data['is_trained']
            
            # Update config
            config = model_data['config']
            self.growth = config['growth']
            self.seasonality_mode = config['seasonality_mode']
            self.yearly_seasonality = config['yearly_seasonality']
            self.weekly_seasonality = config['weekly_seasonality']
            self.daily_seasonality = config['daily_seasonality']
            self.changepoint_prior_scale = config['changepoint_prior_scale']
            self.seasonality_prior_scale = config['seasonality_prior_scale']
            
            self.logger.info(f"Prophet model loaded from {path}")
            
        except Exception as e:
            raise ModelError(f"Failed to load Prophet model: {e}")
