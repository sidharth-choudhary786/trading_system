# trading_system/features/feature_engineer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta

from ..core.exceptions import TradingSystemError

class FeatureEngineer:
    """
    Main feature engineering class - orchestrates all feature generation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature generators
        self.technical_indicators = TechnicalIndicators(config)
        self.regime_detector = RegimeDetector(config)
        self.spline_features = SplineFeatures(config)
        
        # Feature configuration
        self.feature_config = config.get('features', {})
        self.enabled_features = self.feature_config.get('technical_indicators', [])
        
        self.logger.info("Feature Engineer initialized")
    
    def generate_features(
        self, 
        data: pd.DataFrame, 
        symbol: str = None,
        feature_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive features for trading
        
        Args:
            data: OHLCV data with date index
            symbol: Optional symbol for context
            feature_types: Types of features to generate
            
        Returns:
            DataFrame with original data and features
        """
        if data.empty:
            return data
        
        if feature_types is None:
            feature_types = ['technical', 'regime', 'spline']
        
        self.logger.info(f"Generating features for {symbol or 'data'} - Types: {feature_types}")
        
        # Create a copy to avoid modifying original data
        feature_data = data.copy()
        
        # Ensure date is index for time series operations
        if 'date' in feature_data.columns:
            feature_data = feature_data.set_index('date')
        
        # Generate different types of features
        if 'technical' in feature_types:
            feature_data = self._generate_technical_features(feature_data, symbol)
        
        if 'regime' in feature_types:
            feature_data = self._generate_regime_features(feature_data, symbol)
        
        if 'spline' in feature_types:
            feature_data = self._generate_spline_features(feature_data, symbol)
        
        # Generate time-based features
        feature_data = self._generate_time_features(feature_data)
        
        # Generate lagged features
        feature_data = self._generate_lagged_features(feature_data)
        
        # Generate rolling statistics
        feature_data = self._generate_rolling_features(feature_data)
        
        # Clean features (remove NaN rows created by indicators)
        feature_data = self._clean_features(feature_data)
        
        self.logger.info(f"Generated {len(feature_data.columns) - len(data.columns)} features")
        return feature_data.reset_index()
    
    def _generate_technical_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate technical indicator features"""
        feature_data = data.copy()
        
        # Get enabled technical indicators from config
        enabled_indicators = self.feature_config.get('technical_indicators', [
            'sma', 'ema', 'rsi', 'macd', 'bollinger_bands', 'atr'
        ])
        
        # Generate each enabled indicator
        for indicator in enabled_indicators:
            try:
                if indicator == 'sma':
                    feature_data = self.technical_indicators.add_sma(feature_data)
                elif indicator == 'ema':
                    feature_data = self.technical_indicators.add_ema(feature_data)
                elif indicator == 'rsi':
                    feature_data = self.technical_indicators.add_rsi(feature_data)
                elif indicator == 'macd':
                    feature_data = self.technical_indicators.add_macd(feature_data)
                elif indicator == 'bollinger_bands':
                    feature_data = self.technical_indicators.add_bollinger_bands(feature_data)
                elif indicator == 'atr':
                    feature_data = self.technical_indicators.add_atr(feature_data)
                elif indicator == 'stochastic':
                    feature_data = self.technical_indicators.add_stochastic(feature_data)
                elif indicator == 'williams_r':
                    feature_data = self.technical_indicators.add_williams_r(feature_data)
                elif indicator == 'cci':
                    feature_data = self.technical_indicators.add_cci(feature_data)
                elif indicator == 'obv':
                    feature_data = self.technical_indicators.add_obv(feature_data)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate {indicator} for {symbol}: {e}")
        
        return feature_data
    
    def _generate_regime_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate market regime features"""
        feature_data = data.copy()
        
        try:
            feature_data = self.regime_detector.detect_regimes(feature_data)
        except Exception as e:
            self.logger.warning(f"Failed to generate regime features for {symbol}: {e}")
        
        return feature_data
    
    def _generate_spline_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate spline-based features"""
        feature_data = data.copy()
        
        try:
            feature_data = self.spline_features.generate_spline_indicators(feature_data)
        except Exception as e:
            self.logger.warning(f"Failed to generate spline features for {symbol}: {e}")
        
        return feature_data
    
    def _generate_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate time-based features"""
        feature_data = data.copy()
        
        if not isinstance(feature_data.index, pd.DatetimeIndex):
            return feature_data
        
        # Day of week (0=Monday, 6=Sunday)
        feature_data['day_of_week'] = feature_data.index.dayofweek
        
        # Day of month
        feature_data['day_of_month'] = feature_data.index.day
        
        # Week of year
        feature_data['week_of_year'] = feature_data.index.isocalendar().week
        
        # Month
        feature_data['month'] = feature_data.index.month
        
        # Quarter
        feature_data['quarter'] = feature_data.index.quarter
        
        # Year
        feature_data['year'] = feature_data.index.year
        
        # Is weekend?
        feature_data['is_weekend'] = (feature_data['day_of_week'] >= 5).astype(int)
        
        # Days from month end
        next_month = feature_data.index + pd.offsets.MonthBegin(1)
        feature_data['days_to_month_end'] = (next_month - feature_data.index).days
        
        return feature_data
    
    def _generate_lagged_features(self, data: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """Generate lagged price and volume features"""
        if lags is None:
            lags = [1, 2, 3, 5, 10]  # 1 day, 2 days, etc.
        
        feature_data = data.copy()
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in price_columns:
            if col in feature_data.columns:
                for lag in lags:
                    feature_data[f'{col}_lag_{lag}'] = feature_data[col].shift(lag)
        
        # Calculate lagged returns
        if 'close' in feature_data.columns:
            for lag in [1, 2, 3, 5]:
                feature_data[f'return_{lag}d'] = feature_data['close'].pct_change(lag)
        
        return feature_data
    
    def _generate_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling window statistics"""
        feature_data = data.copy()
        
        if 'close' in feature_data.columns:
            # Rolling means
            for window in [5, 10, 20, 50]:
                feature_data[f'rolling_mean_{window}'] = feature_data['close'].rolling(window=window).mean()
                feature_data[f'rolling_std_{window}'] = feature_data['close'].rolling(window=window).std()
                feature_data[f'rolling_min_{window}'] = feature_data['close'].rolling(window=window).min()
                feature_data[f'rolling_max_{window}'] = feature_data['close'].rolling(window=window).max()
            
            # Rolling volatility (annualized)
            for window in [10, 20, 50]:
                returns = feature_data['close'].pct_change()
                feature_data[f'volatility_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)
            
            # Rolling Sharpe ratio (simplified)
            for window in [20, 50]:
                returns = feature_data['close'].pct_change()
                rolling_mean = returns.rolling(window=window).mean()
                rolling_std = returns.rolling(window=window).std()
                feature_data[f'sharpe_{window}'] = (rolling_mean / rolling_std) * np.sqrt(252)
        
        if 'volume' in feature_data.columns:
            # Volume features
            for window in [5, 10, 20]:
                feature_data[f'volume_ma_{window}'] = feature_data['volume'].rolling(window=window).mean()
                feature_data[f'volume_ratio_{window}'] = feature_data['volume'] / feature_data[f'volume_ma_{window}']
        
        return feature_data
    
    def _clean_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean feature data by removing NaN rows"""
        # Remove rows where all feature columns are NaN
        feature_columns = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'adj_close']]
        
        if feature_columns:
            # Keep rows where at least one feature is not NaN
            feature_mask = data[feature_columns].notna().any(axis=1)
            cleaned_data = data[feature_mask].copy()
            
            rows_removed = len(data) - len(cleaned_data)
            if rows_removed > 0:
                self.logger.info(f"Removed {rows_removed} rows with missing features")
            
            return cleaned_data
        
        return data
    
    def get_feature_info(self, feature_data: pd.DataFrame) -> Dict:
        """Get information about generated features"""
        original_columns = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'date']
        feature_columns = [col for col in feature_data.columns if col not in original_columns]
        
        feature_info = {
            'total_features': len(feature_columns),
            'feature_columns': feature_columns,
            'feature_categories': {},
            'missing_values': {},
            'feature_stats': {}
        }
        
        # Categorize features
        categories = {
            'technical': ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'stochastic', 'williams', 'cci', 'obv'],
            'regime': ['regime'],
            'spline': ['spline'],
            'time': ['day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter', 'year', 'is_weekend'],
            'lagged': ['lag_', 'return_'],
            'rolling': ['rolling_', 'volatility_', 'sharpe_', 'volume_ma_', 'volume_ratio_']
        }
        
        for feature in feature_columns:
            # Count missing values
            feature_info['missing_values'][feature] = feature_data[feature].isna().sum()
            
            # Calculate basic statistics for numeric features
            if pd.api.types.is_numeric_dtype(feature_data[feature]):
                feature_info['feature_stats'][feature] = {
                    'mean': feature_data[feature].mean(),
                    'std': feature_data[feature].std(),
                    'min': feature_data[feature].min(),
                    'max': feature_data[feature].max()
                }
            
            # Categorize feature
            for category, keywords in categories.items():
                if any(keyword in feature.lower() for keyword in keywords):
                    if category not in feature_info['feature_categories']:
                        feature_info['feature_categories'][category] = []
                    feature_info['feature_categories'][category].append(feature)
                    break
            else:
                if 'other' not in feature_info['feature_categories']:
                    feature_info['feature_categories']['other'] = []
                feature_info['feature_categories']['other'].append(feature)
        
        return feature_info
    
    def select_features(
        self, 
        feature_data: pd.DataFrame, 
        method: str = 'all',
        top_k: int = 50
    ) -> pd.DataFrame:
        """
        Select subset of features based on method
        
        Args:
            feature_data: DataFrame with all features
            method: Selection method ('all', 'technical', 'non_technical', 'top_correlated')
            top_k: Number of features to select for top_correlated method
            
        Returns:
            DataFrame with selected features
        """
        original_columns = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'date']
        all_features = [col for col in feature_data.columns if col not in original_columns]
        
        if method == 'all':
            selected_features = all_features
        elif method == 'technical':
            selected_features = [f for f in all_features if any(x in f for x in ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr'])]
        elif method == 'non_technical':
            selected_features = [f for f in all_features if not any(x in f for x in ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr'])]
        elif method == 'top_correlated':
            # Select features most correlated with future returns
            if 'close' in feature_data.columns:
                # Calculate future returns (next day)
                future_returns = feature_data['close'].pct_change().shift(-1)
                
                # Calculate correlations
                correlations = {}
                for feature in all_features:
                    if pd.api.types.is_numeric_dtype(feature_data[feature]):
                        corr = feature_data[feature].corr(future_returns)
                        if not pd.isna(corr):
                            correlations[feature] = abs(corr)  # Use absolute correlation
                
                # Select top k features
                sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:top_k]]
            else:
                selected_features = all_features[:top_k]
        else:
            selected_features = all_features
        
        # Always include original price columns
        result_columns = original_columns + selected_features
        result_columns = [col for col in result_columns if col in feature_data.columns]
        
        self.logger.info(f"Selected {len(selected_features)} features using {method} method")
        return feature_data[result_columns]
