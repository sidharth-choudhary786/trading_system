# trading_system/features/spline_features.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

from ..core.exceptions import TradingSystemError

class SplineFeatures:
    """
    Spline-based technical indicators and feature generation
    Provides smooth, continuous representations of price data
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Spline configuration
        self.spline_config = config.get('spline_features', {})
        self.spline_order = self.spline_config.get('spline_order', 3)
        self.smoothing_factor = self.spline_config.get('smoothing_factor', 0.1)
        
        # Feature parameters
        self.derivative_windows = self.spline_config.get('derivative_windows', [5, 10, 20])
        self.curvature_threshold = self.spline_config.get('curvature_threshold', 0.01)
        
        self.logger.info("Spline Features initialized")
    
    def generate_spline_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive spline-based indicators
        
        Args:
            data: OHLCV data with date index
            
        Returns:
            DataFrame with spline indicators
        """
        if data.empty:
            return data
        
        feature_data = data.copy()
        
        # Ensure we have numeric index for spline fitting
        if isinstance(feature_data.index, pd.DatetimeIndex):
            # Convert datetime index to numeric for spline fitting
            numeric_index = np.arange(len(feature_data))
        else:
            numeric_index = np.arange(len(feature_data))
        
        # Generate spline-based features for different price series
        price_columns = ['close', 'high', 'low']
        for price_col in price_columns:
            if price_col in feature_data.columns:
                try:
                    feature_data = self._add_spline_features_for_column(
                        feature_data, numeric_index, price_col
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to generate spline features for {price_col}: {e}")
        
        # Generate volume spline features
        if 'volume' in feature_data.columns:
            try:
                feature_data = self._add_volume_spline_features(feature_data, numeric_index)
            except Exception as e:
                self.logger.warning(f"Failed to generate volume spline features: {e}")
        
        # Generate combined spline signals
        feature_data = self._generate_combined_signals(feature_data)
        
        self.logger.info(f"Generated {len([col for col in feature_data.columns if 'spline' in col])} spline indicators")
        return feature_data
    
    def _add_spline_features_for_column(self, data: pd.DataFrame, numeric_index: np.ndarray, column: str) -> pd.DataFrame:
        """Generate spline features for a specific price column"""
        feature_data = data.copy()
        prices = feature_data[column].values
        
        # Remove NaN values for spline fitting
        valid_mask = ~np.isnan(prices)
        if np.sum(valid_mask) < 10:  # Need minimum data points
            return feature_data
        
        valid_index = numeric_index[valid_mask]
        valid_prices = prices[valid_mask]
        
        try:
            # Fit cubic spline
            spline = CubicSpline(valid_index, valid_prices)
            
            # Fit smoothing spline
            smoothing_spline = UnivariateSpline(valid_index, valid_prices, s=self.smoothing_factor * len(valid_prices))
            
            # Calculate spline values for all points
            feature_data[f'{column}_spline'] = spline(numeric_index)
            feature_data[f'{column}_smooth_spline'] = smoothing_spline(numeric_index)
            
            # Calculate first derivative (instantaneous rate of change)
            first_deriv = spline(numeric_index, 1)
            feature_data[f'{column}_spline_deriv1'] = first_deriv
            
            # Calculate second derivative (acceleration/curvature)
            second_deriv = spline(numeric_index, 2)
            feature_data[f'{column}_spline_deriv2'] = second_deriv
            
            # Calculate third derivative (jerk)
            third_deriv = spline(numeric_index, 3)
            feature_data[f'{column}_spline_deriv3'] = third_deriv
            
            # Generate derived features
            feature_data = self._generate_derived_spline_features(feature_data, column, first_deriv, second_deriv)
            
        except Exception as e:
            self.logger.warning(f"Spline fitting failed for {column}: {e}")
        
        return feature_data
    
    def _generate_derived_spline_features(self, data: pd.DataFrame, column: str, 
                                        first_deriv: np.ndarray, second_deriv: np.ndarray) -> pd.DataFrame:
        """Generate derived features from spline derivatives"""
        feature_data = data.copy()
        
        # Trend strength based on first derivative
        feature_data[f'{column}_spline_trend_strength'] = np.abs(first_deriv)
        
        # Curvature strength based on second derivative
        feature_data[f'{column}_spline_curvature'] = np.abs(second_deriv)
        
        # Inflection points (where second derivative changes sign)
        inflection_points = np.diff(np.sign(second_deriv)) != 0
        feature_data[f'{column}_spline_inflection'] = False
        feature_data.iloc[1:, feature_data.columns.get_loc(f'{column}_spline_inflection')] = inflection_points
        
        # Trend direction (1 for up, -1 for down, 0 for flat)
        trend_direction = np.sign(first_deriv)
        feature_data[f'{column}_spline_trend_direction'] = trend_direction
        
        # Volatility of derivatives
        for window in self.derivative_windows:
            if len(first_deriv) > window:
                feature_data[f'{column}_spline_deriv1_vol_{window}'] = (
                    pd.Series(first_deriv).rolling(window=window).std()
                )
                feature_data[f'{column}_spline_deriv2_vol_{window}'] = (
                    pd.Series(second_deriv).rolling(window=window).std()
                )
        
        # Momentum indicators based on derivatives
        feature_data[f'{column}_spline_momentum'] = first_deriv
        feature_data[f'{column}_spline_acceleration'] = second_deriv
        
        # Extreme curvature detection
        curvature_threshold = np.percentile(np.abs(second_deriv), 95)  # Top 5%
        feature_data[f'{column}_spline_extreme_curvature'] = np.abs(second_deriv) > curvature_threshold
        
        return feature_data
    
    def _add_volume_spline_features(self, data: pd.DataFrame, numeric_index: np.ndarray) -> pd.DataFrame:
        """Generate spline features for volume data"""
        feature_data = data.copy()
        volumes = feature_data['volume'].values
        
        # Remove NaN and zero values
        valid_mask = ~(np.isnan(volumes) | (volumes == 0))
        if np.sum(valid_mask) < 10:
            return feature_data
        
        valid_index = numeric_index[valid_mask]
        valid_volumes = volumes[valid_mask]
        
        try:
            # Use log volume for better spline fitting
            log_volumes = np.log(valid_volumes)
            
            # Fit spline to log volumes
            volume_spline = CubicSpline(valid_index, log_volumes)
            
            # Calculate spline values
            feature_data['volume_spline'] = np.exp(volume_spline(numeric_index))
            
            # Volume derivatives
            volume_deriv1 = volume_spline(numeric_index, 1)
            volume_deriv2 = volume_spline(numeric_index, 2)
            
            feature_data['volume_spline_deriv1'] = volume_deriv1
            feature_data['volume_spline_deriv2'] = volume_deriv2
            
            # Volume trend features
            feature_data['volume_trend_strength'] = np.abs(volume_deriv1)
            feature_data['volume_trend_direction'] = np.sign(volume_deriv1)
            
            # Volume-price relationship features
            if 'close_spline_deriv1' in feature_data.columns:
                # Correlation between price and volume trends
                feature_data['price_volume_trend_correlation'] = (
                    feature_data['close_spline_deriv1'] * volume_deriv1
                )
                
                # Volume confirmation indicator
                feature_data['volume_confirmation'] = (
                    (feature_data['close_spline_trend_direction'] == feature_data['volume_trend_direction']).astype(int)
                )
            
        except Exception as e:
            self.logger.warning(f"Volume spline fitting failed: {e}")
        
        return feature_data
    
    def _generate_combined_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate combined trading signals from multiple spline features"""
        feature_data = data.copy()
        
        # Check if required columns exist
        required_cols = ['close_spline_deriv1', 'close_spline_deriv2', 'close_spline_curvature']
        if not all(col in feature_data.columns for col in required_cols):
            return feature_data
        
        try:
            # 1. Trend Following Signal
            # Strong trend + low curvature = good trend following
            trend_strength = np.abs(feature_data['close_spline_deriv1'])
            curvature = np.abs(feature_data['close_spline_curvature'])
            
            # Normalize signals
            trend_strength_norm = (trend_strength - trend_strength.mean()) / trend_strength.std()
            curvature_norm = (curvature - curvature.mean()) / curvature.std()
            
            feature_data['spline_trend_following_signal'] = trend_strength_norm - curvature_norm
            
            # 2. Mean Reversion Signal
            # High curvature + inflection points = potential reversal
            inflection_strength = feature_data['close_spline_inflection'].astype(int) * curvature
            feature_data['spline_mean_reversion_signal'] = inflection_strength
            
            # 3. Momentum Signal
            # Positive acceleration + strong trend = momentum
            acceleration = feature_data['close_spline_deriv2']
            momentum_signal = trend_strength * np.maximum(acceleration, 0)
            feature_data['spline_momentum_signal'] = momentum_signal
            
            # 4. Volatility Signal
            # Based on derivative volatility
            if 'close_spline_deriv1_vol_10' in feature_data.columns:
                vol_signal = feature_data['close_spline_deriv1_vol_10']
                feature_data['spline_volatility_signal'] = vol_signal
            
            # 5. Composite Signal
            # Weighted combination of all signals
            signals = []
            weights = []
            
            if 'spline_trend_following_signal' in feature_data.columns:
                signals.append(feature_data['spline_trend_following_signal'])
                weights.append(0.3)
            
            if 'spline_mean_reversion_signal' in feature_data.columns:
                signals.append(feature_data['spline_mean_reversion_signal'])
                weights.append(0.25)
            
            if 'spline_momentum_signal' in feature_data.columns:
                signals.append(feature_data['spline_momentum_signal'])
                weights.append(0.25)
            
            if 'spline_volatility_signal' in feature_data.columns:
                signals.append(feature_data['spline_volatility_signal'])
                weights.append(0.2)
            
            if signals:
                # Normalize each signal
                normalized_signals = []
                for signal in signals:
                    signal_normalized = (signal - signal.mean()) / signal.std()
                    normalized_signals.append(signal_normalized)
                
                # Create composite signal
                composite = np.zeros_like(normalized_signals[0])
                for i, signal in enumerate(normalized_signals):
                    composite += weights[i] * signal
                
                feature_data['spline_composite_signal'] = composite
            
            # 6. Regime Detection based on spline characteristics
            feature_data = self._detect_spline_regimes(feature_data)
            
        except Exception as e:
            self.logger.warning(f"Combined signal generation failed: {e}")
        
        return feature_data
    
    def _detect_spline_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect market regimes based on spline characteristics"""
        feature_data = data.copy()
        
        try:
            # Features for regime detection
            regime_features = []
            
            if 'close_spline_trend_strength' in feature_data.columns:
                regime_features.append(feature_data['close_spline_trend_strength'])
            
            if 'close_spline_curvature' in feature_data.columns:
                regime_features.append(feature_data['close_spline_curvature'])
            
            if 'close_spline_deriv1_vol_10' in feature_data.columns:
                regime_features.append(feature_data['close_spline_deriv1_vol_10'])
            
            if len(regime_features) >= 2:
                # Simple regime detection based on percentiles
                trend_strength = regime_features[0]
                curvature = regime_features[1]
                
                # Define regime thresholds
                trend_threshold = np.percentile(trend_strength, 70)
                curvature_threshold = np.percentile(curvature, 70)
                vol_threshold = np.percentile(regime_features[2], 70) if len(regime_features) > 2 else 0
                
                # Initialize regime column
                feature_data['spline_regime'] = 'neutral'
                
                # High trend, low curvature = trending regime
                trending_mask = (trend_strength > trend_threshold) & (curvature < curvature_threshold)
                feature_data.loc[trending_mask, 'spline_regime'] = 'trending'
                
                # High curvature, low trend = mean reversion regime
                mean_reversion_mask = (curvature > curvature_threshold) & (trend_strength < trend_threshold)
                feature_data.loc[mean_reversion_mask, 'spline_regime'] = 'mean_reversion'
                
                # High volatility = volatile regime
                if len(regime_features) > 2:
                    volatile_mask = regime_features[2] > vol_threshold
                    feature_data.loc[volatile_mask, 'spline_regime'] = 'volatile'
                
                # Encode regimes numerically
                regime_mapping = {
                    'neutral': 0,
                    'trending': 1,
                    'mean_reversion': 2,
                    'volatile': 3
                }
                feature_data['spline_regime_encoded'] = feature_data['spline_regime'].map(regime_mapping)
        
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
        
        return feature_data
    
    def generate_trading_signals(self, data: pd.DataFrame, method: str = 'composite') -> pd.DataFrame:
        """
        Generate explicit trading signals from spline features
        
        Args:
            data: DataFrame with spline features
            method: Signal generation method
            
        Returns:
            DataFrame with trading signals
        """
        feature_data = data.copy()
        
        if method == 'composite' and 'spline_composite_signal' in feature_data.columns:
            # Use composite signal with thresholding
            signal = feature_data['spline_composite_signal']
            
            # Dynamic thresholds based on recent distribution
            upper_threshold = signal.rolling(50).quantile(0.7).fillna(signal.quantile(0.7))
            lower_threshold = signal.rolling(50).quantile(0.3).fillna(signal.quantile(0.3))
            
            # Generate signals
            feature_data['spline_signal'] = 0
            feature_data.loc[signal > upper_threshold, 'spline_signal'] = 1  # Buy
            feature_data.loc[signal < lower_threshold, 'spline_signal'] = -1  # Sell
            
        elif method == 'trend_following' and 'spline_trend_following_signal' in feature_data.columns:
            # Trend following signals
            signal = feature_data['spline_trend_following_signal']
            threshold = signal.rolling(50).quantile(0.6).fillna(signal.quantile(0.6))
            
            feature_data['spline_signal'] = 0
            feature_data.loc[signal > threshold, 'spline_signal'] = 1
            
        elif method == 'mean_reversion' and 'spline_mean_reversion_signal' in feature_data.columns:
            # Mean reversion signals
            signal = feature_data['spline_mean_reversion_signal']
            threshold = signal.rolling(50).quantile(0.7).fillna(signal.quantile(0.7))
            
            feature_data['spline_signal'] = 0
            feature_data.loc[signal > threshold, 'spline_signal'] = -1  # Sell on high mean reversion
            
        # Add signal strength and confidence
        if 'spline_signal' in feature_data.columns:
            feature_data['spline_signal_strength'] = np.abs(feature_data['spline_composite_signal'])
            
            # Simple confidence based on regime
            if 'spline_regime' in feature_data.columns:
                confidence_map = {
                    'trending': 0.8,
                    'mean_reversion': 0.7,
                    'volatile': 0.4,
                    'neutral': 0.5
                }
                feature_data['spline_signal_confidence'] = feature_data['spline_regime'].map(confidence_map)
        
        return feature_data
    
    def get_spline_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate statistics for spline features"""
        spline_columns = [col for col in data.columns if 'spline' in col]
        
        if not spline_columns:
            return {}
        
        stats = {
            'total_spline_features': len(spline_columns),
            'spline_feature_categories': {},
            'feature_correlations': {},
            'signal_statistics': {}
        }
        
        # Categorize spline features
        categories = {
            'spline_values': ['spline', 'smooth_spline'],
            'derivatives': ['deriv1', 'deriv2', 'deriv3'],
            'trend': ['trend_strength', 'trend_direction'],
            'curvature': ['curvature', 'inflection'],
            'volatility': ['vol_', 'deriv1_vol', 'deriv2_vol'],
            'signals': ['signal', 'composite', 'momentum', 'mean_reversion'],
            'regime': ['regime']
        }
        
        for col in spline_columns:
            for category, keywords in categories.items():
                if any(keyword in col for keyword in keywords):
                    if category not in stats['spline_feature_categories']:
                        stats['spline_feature_categories'][category] = []
                    stats['spline_feature_categories'][category].append(col)
                    break
        
        # Calculate correlations with price changes if available
        if 'close' in data.columns:
            price_changes = data['close'].pct_change()
            for col in spline_columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    correlation = data[col].corr(price_changes)
                    if not pd.isna(correlation):
                        stats['feature_correlations'][col] = correlation
        
        # Signal statistics
        signal_cols = [col for col in spline_columns if 'signal' in col and col != 'spline_signal_strength']
        for col in signal_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                signal_data = data[col].dropna()
                if len(signal_data) > 0:
                    stats['signal_statistics'][col] = {
                        'mean': signal_data.mean(),
                        'std': signal_data.std(),
                        'min': signal_data.min(),
                        'max': signal_data.max(),
                        'positive_signals': (signal_data > 0).sum(),
                        'negative_signals': (signal_data < 0).sum()
                    }
        
        return stats
