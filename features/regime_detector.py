# trading_system/features/regime_detector.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

from ..core.exceptions import TradingSystemError

class RegimeDetector:
    """
    Market regime detection using multiple methods
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Regime detection configuration
        self.regime_config = config.get('regime_detection', {})
        self.method = self.regime_config.get('method', 'hmm')
        self.n_regimes = self.regime_config.get('n_regimes', 3)
        
        # Store trained models
        self.models = {}
        self.regime_labels = {}
    
    def detect_regimes(self, data: pd.DataFrame, method: str = None, n_regimes: int = None) -> pd.DataFrame:
        """
        Detect market regimes in data
        
        Args:
            data: OHLCV data
            method: Detection method ('hmm', 'kmeans', 'gmm')
            n_regimes: Number of regimes to detect
            
        Returns:
            DataFrame with regime labels
        """
        if method is None:
            method = self.method
        if n_regimes is None:
            n_regimes = self.n_regimes
        
        result_data = data.copy()
        
        # Extract features for regime detection
        regime_features = self._extract_regime_features(result_data)
        
        if regime_features.empty:
            self.logger.warning("No features available for regime detection")
            return result_data
        
        # Detect regimes using selected method
        if method == 'hmm':
            regime_labels = self._hmm_regimes(regime_features, n_regimes)
        elif method == 'kmeans':
            regime_labels = self._kmeans_regimes(regime_features, n_regimes)
        elif method == 'gmm':
            regime_labels = self._gmm_regimes(regime_features, n_regimes)
        else:
            self.logger.warning(f"Unknown regime detection method: {method}. Using HMM.")
            regime_labels = self._hmm_regimes(regime_features, n_regimes)
        
        # Add regime labels to data
        result_data['regime'] = regime_labels
        result_data['regime_label'] = result_data['regime'].apply(
            lambda x: self._get_regime_name(x, n_regimes)
        )
        
        # Add regime transitions
        result_data = self._add_regime_transitions(result_data)
        
        self.logger.info(f"Detected {n_regimes} market regimes using {method}")
        return result_data
    
    def _extract_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime detection"""
        features = pd.DataFrame(index=data.index)
        
        if 'close' not in data.columns:
            return features
        
        # Price-based features
        returns = data['close'].pct_change().dropna()
        
        if len(returns) < 50:  # Need sufficient data
            return features
        
        # Volatility features
        features['volatility_20d'] = returns.rolling(20).std()
        features['volatility_50d'] = returns.rolling(50).std()
        features['volatility_ratio'] = features['volatility_20d'] / features['volatility_50d']
        
        # Return features
        features['return_1d'] = returns
        features['return_5d'] = data['close'].pct_change(5)
        features['return_20d'] = data['close'].pct_change(20)
        
        # Trend features
        if 'sma_20' in data.columns and 'sma_50' in data.columns:
            features['trend_strength'] = (data['sma_20'] - data['sma_50']) / data['sma_50']
        
        # Volume features (if available)
        if 'volume' in data.columns:
            features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Market regime specific features
        features['high_low_range'] = (data['high'] - data['low']) / data['close'] if all(col in data.columns for col in ['high', 'low'] else 0
        features['price_vs_ma'] = data['close'] / data['close'].rolling(20).mean() - 1
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def _hmm_regimes(self, features: pd.DataFrame, n_regimes: int) -> pd.Series:
        """Detect regimes using Hidden Markov Model"""
        try:
            # Standardize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Fit HMM
            model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=1000,
                random_state=42
            )
            
            # Reshape for HMM (samples, features)
            if len(scaled_features) > 0:
                regime_labels = model.fit_predict(scaled_features)
                
                # Store model
                self.models['hmm'] = model
                self.regime_labels['hmm'] = regime_labels
                
                # Create series with original index
                return pd.Series(regime_labels, index=features.index)
            else:
                return pd.Series([0] * len(features), index=features.index)
                
        except Exception as e:
            self.logger.warning(f"HMM regime detection failed: {e}")
            return pd.Series([0] * len(features), index=features.index)
    
    def _kmeans_regimes(self, features: pd.DataFrame, n_regimes: int) -> pd.Series:
        """Detect regimes using K-Means clustering"""
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Fit K-Means
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            regime_labels = kmeans.fit_predict(scaled_features)
            
            # Store model
            self.models['kmeans'] = kmeans
            self.regime_labels['kmeans'] = regime_labels
            
            return pd.Series(regime_labels, index=features.index)
            
        except Exception as e:
            self.logger.warning(f"K-Means regime detection failed: {e}")
            return pd.Series([0] * len(features), index=features.index)
    
    def _gmm_regimes(self, features: pd.DataFrame, n_regimes: int) -> pd.Series:
        """Detect regimes using Gaussian Mixture Model"""
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.mixture import GaussianMixture
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Fit GMM
            gmm = GaussianMixture(n_components=n_regimes, random_state=42)
            regime_labels = gmm.fit_predict(scaled_features)
            
            # Store model
            self.models['gmm'] = gmm
            self.regime_labels['gmm'] = regime_labels
            
            return pd.Series(regime_labels, index=features.index)
            
        except Exception as e:
            self.logger.warning(f"GMM regime detection failed: {e}")
            return pd.Series([0] * len(features), index=features.index)
    
    def _get_regime_name(self, regime_id: int, n_regimes: int) -> str:
        """Get descriptive name for regime"""
        if n_regimes == 2:
            names = {0: 'Bear', 1: 'Bull'}
        elif n_regimes == 3:
            names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        elif n_regimes == 4:
            names = {0: 'Crash', 1: 'Bear', 2: 'Bull', 3: 'Bubble'}
        else:
            names = {i: f'Regime_{i}' for i in range(n_regimes)}
        
        return names.get(regime_id, f'Regime_{regime_id}')
    
    def _add_regime_transitions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add regime transition features"""
        result_data = data.copy()
        
        if 'regime' not in result_data.columns:
            return result_data
        
        # Regime duration
        result_data['regime_duration'] = 1
        current_regime = result_data['regime'].iloc[0]
        duration = 1
        
        for i in range(1, len(result_data)):
            if result_data['regime'].iloc[i] == current_regime:
                duration += 1
            else:
                current_regime = result_data['regime'].iloc[i]
                duration = 1
            result_data.iloc[i, result_data.columns.get_loc('regime_duration')] = duration
        
        # Regime changes
        result_data['regime_change'] = result_data['regime'].diff().fillna(0).astype(int)
        result_data['regime_change_flag'] = (result_data['regime_change'] != 0).astype(int)
        
        # Regime stability (rolling regime consistency)
        result_data['regime_stability'] = result_data['regime_change_flag'].rolling(10).mean()
        
        return result_data
    
    def get_regime_statistics(self, data: pd.DataFrame) -> Dict:
        """Get statistics for detected regimes"""
        if 'regime' not in data.columns:
            return {}
        
        stats = {}
        regimes = data['regime'].unique()
        
        for regime in regimes:
            regime_data = data[data['regime'] == regime]
            regime_stats = {
                'count': len(regime_data),
                'percentage': len(regime_data) / len(data) * 100,
                'avg_duration': regime_data['regime_duration'].mean() if 'regime_duration' in regime_data.columns else 0,
                'max_duration': regime_data['regime_duration'].max() if 'regime_duration' in regime_data.columns else 0,
            }
            
            # Price statistics for regime
            if 'close' in regime_data.columns:
                regime_stats['avg_return'] = regime_data['close'].pct_change().mean()
                regime_stats['volatility'] = regime_data['close'].pct_change().std()
                regime_stats['sharpe'] = regime_stats['avg_return'] / regime_stats['volatility'] if regime_stats['volatility'] > 0 else 0
            
            stats[f'regime_{regime}'] = regime_stats
        
        return stats
    
    def predict_future_regime(self, data: pd.DataFrame, steps: int = 1) -> pd.DataFrame:
        """Predict future regime probabilities"""
        result_data = data.copy()
        
        if 'hmm' not in self.models:
            self.logger.warning("No trained HMM model found for prediction")
            return result_data
        
        # Extract features for prediction
        features = self._extract_regime_features(data)
        if features.empty:
            return result_data
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Predict future regimes
        try:
            hmm_model = self.models['hmm']
            
            # Get current regime
            current_regime = hmm_model.predict(scaled_features[-1:])[0]
            
            # Predict next regime (simplified)
            # In practice, this would use proper HMM prediction methods
            transition_matrix = hmm_model.transmat_
            next_regime_probs = transition_matrix[current_regime]
            predicted_regime = np.argmax(next_regime_probs)
            
            # Add predictions to data
            result_data['predicted_regime'] = current_regime
            result_data['next_regime_prob'] = next_regime_probs[predicted_regime]
            result_data['predicted_regime_label'] = self._get_regime_name(
                predicted_regime, self.n_regimes
            )
            
        except Exception as e:
            self.logger.warning(f"Regime prediction failed: {e}")
        
        return result_data
    
    def get_regime_transition_matrix(self) -> np.ndarray:
        """Get regime transition probability matrix"""
        if 'hmm' not in self.models:
            return np.array([])
        
        return self.models['hmm'].transmat_
