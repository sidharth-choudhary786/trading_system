# trading_system/data/processing/cleaner.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ...core.exceptions import DataSourceError

class DataCleaner:
    """
    Handle outlier detection and cleaning using multiple methods
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.method = config.get('cleaning_method', 'iqr')
        
        # Method configurations
        self.method_configs = {
            'iqr': {'name': 'Interquartile Range', 'threshold': 1.5},
            'zscore': {'name': 'Z-Score', 'threshold': 3},
            'winsorize': {'name': 'Winsorization', 'limits': [0.05, 0.05]},
            'isolation_forest': {'name': 'Isolation Forest', 'contamination': 0.1},
            'lof': {'name': 'Local Outlier Factor', 'contamination': 0.1},
            'boudt': {'name': 'Boudt Method'},
            'geltner': {'name': 'Geltner Method'},
            'evt': {'name': 'Extreme Value Theory', 'threshold': 0.95}
        }
    
    def clean(self, data: pd.DataFrame, method: Optional[str] = None) -> pd.DataFrame:
        """
        Clean outliers from data
        
        Args:
            data: DataFrame with potential outliers
            method: Cleaning method to use
            
        Returns:
            Cleaned DataFrame
        """
        if method is None:
            method = self.method
        
        if data.empty:
            return data
        
        # Create a copy to avoid modifying original
        cleaned_data = data.copy()
        
        # Detect outliers
        outlier_report = self.detect_outliers(cleaned_data, method)
        
        if not outlier_report['outliers_found']:
            self.logger.debug("No outliers detected")
            return cleaned_data
        
        self.logger.info(f"Cleaning {outlier_report['total_outliers']} outliers using {method} method")
        
        # Apply cleaning based on method
        if method == 'winsorize':
            cleaned_data = self._winsorize_clean(cleaned_data)
        else:
            # For other methods, replace outliers with NaN and then impute
            for col, outliers in outlier_report['column_outliers'].items():
                if outliers:
                    cleaned_data.loc[outliers, col] = np.nan
        
        # Impute the NaN values created by outlier removal
        from .imputer import DataImputer
        imputer = DataImputer(self.config)
        cleaned_data = imputer.impute(cleaned_data)
        
        self.logger.info("Outlier cleaning complete")
        return cleaned_data
    
    def detect_outliers(self, data: pd.DataFrame, method: Optional[str] = None) -> Dict:
        """
        Detect outliers in data
        
        Args:
            data: DataFrame to analyze
            method: Detection method to use
            
        Returns:
            Dictionary with outlier information
        """
        if method is None:
            method = self.method
        
        numeric_columns = self._get_numeric_columns(data)
        outliers_report = {
            'method_used': method,
            'columns_analyzed': numeric_columns,
            'column_outliers': {},
            'total_outliers': 0,
            'outliers_found': False
        }
        
        for col in numeric_columns:
            if method == 'iqr':
                outliers = self._iqr_detect(data[col])
            elif method == 'zscore':
                outliers = self._zscore_detect(data[col])
            elif method == 'isolation_forest':
                outliers = self._isolation_forest_detect(data[[col]])
            elif method == 'lof':
                outliers = self._lof_detect(data[[col]])
            elif method == 'boudt':
                outliers = self._boudt_detect(data[col])
            elif method == 'geltner':
                outliers = self._geltner_detect(data[col])
            elif method == 'evt':
                outliers = self._evt_detect(data[col])
            else:
                outliers = self._iqr_detect(data[col])  # Default to IQR
            
            outliers_report['column_outliers'][col] = outliers
            outliers_report['total_outliers'] += len(outliers)
        
        outliers_report['outliers_found'] = outliers_report['total_outliers'] > 0
        return outliers_report
    
    def _iqr_detect(self, series: pd.Series) -> pd.Index:
        """IQR method for outlier detection"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        threshold = self.method_configs['iqr'].get('threshold', 1.5)
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return outliers.index
    
    def _zscore_detect(self, series: pd.Series) -> pd.Index:
        """Z-Score method for outlier detection"""
        threshold = self.method_configs['zscore'].get('threshold', 3)
        z_scores = np.abs(stats.zscore(series.dropna()))
        
        # Get indices of outliers
        outlier_indices = series.dropna().index[z_scores > threshold]
        return outlier_indices
    
    def _winsorize_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Winsorization method for outlier treatment"""
        cleaned_data = data.copy()
        numeric_columns = self._get_numeric_columns(data)
        limits = self.method_configs['winsorize'].get('limits', [0.05, 0.05])
        
        for col in numeric_columns:
            cleaned_data[col] = stats.mstats.winsorize(
                cleaned_data[col], 
                limits=limits
            )
        
        return cleaned_data
    
    def _isolation_forest_detect(self, data: pd.DataFrame) -> pd.Index:
        """Isolation Forest for outlier detection"""
        from sklearn.ensemble import IsolationForest
        
        contamination = self.method_configs['isolation_forest'].get('contamination', 0.1)
        model = IsolationForest(contamination=contamination, random_state=42)
        
        # Fit and predict
        predictions = model.fit_predict(data)
        
        # Outliers are marked as -1
        outlier_indices = data.index[predictions == -1]
        return outlier_indices
    
    def _lof_detect(self, data: pd.DataFrame) -> pd.Index:
        """Local Outlier Factor for outlier detection"""
        from sklearn.neighbors import LocalOutlierFactor
        
        contamination = self.method_configs['lof'].get('contamination', 0.1)
        model = LocalOutlierFactor(contamination=contamination)
        
        predictions = model.fit_predict(data)
        outlier_indices = data.index[predictions == -1]
        return outlier_indices
    
    def _boudt_detect(self, series: pd.Series) -> pd.Index:
        """Boudt method for financial data outlier detection"""
        # Boudt method uses robust measures for financial data
        returns = series.pct_change().dropna()
        
        if len(returns) < 10:  # Need sufficient data
            return pd.Index([])
        
        # Use median and MAD (Median Absolute Deviation)
        median_return = returns.median()
        mad_return = (returns - median_return).abs().median()
        
        # Robust z-score
        robust_z = (returns - median_return) / (1.4826 * mad_return)  # 1.4826 for consistency with std
        
        threshold = 3  # Conservative threshold for financial data
        outlier_indices = returns.index[np.abs(robust_z) > threshold]
        
        # Map back to original series indices
        original_indices = series.index[series.index.isin(outlier_indices)]
        return original_indices
    
    def _geltner_detect(self, series: pd.Series) -> pd.Index:
        """Geltner method for real estate/financial data"""
        # Geltner method for smoothing financial time series
        returns = series.pct_change().dropna()
        
        if len(returns) < 3:
            return pd.Index([])
        
        # Simple implementation: detect large single-period returns
        threshold = returns.std() * 3
        large_returns = returns[np.abs(returns) > threshold]
        
        # Map back to original series
        original_indices = series.index[series.index.isin(large_returns.index)]
        return original_indices
    
    def _evt_detect(self, series: pd.Series) -> pd.Index:
        """Extreme Value Theory for outlier detection"""
        threshold = self.method_configs['evt'].get('threshold', 0.95)
        
        # Use quantile-based threshold
        lower_threshold = series.quantile(1 - threshold)
        upper_threshold = series.quantile(threshold)
        
        outliers = series[(series < lower_threshold) | (series > upper_threshold)]
        return outliers.index
    
    def _get_numeric_columns(self, data: pd.DataFrame) -> List[str]:
        """Get numeric columns from DataFrame"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols
    
    def get_cleaning_report(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame) -> Dict:
        """Generate cleaning report"""
        outlier_report = self.detect_outliers(original_data)
        
        report = {
            'method_used': self.method,
            'outliers_detected': outlier_report,
            'data_statistics': {}
        }
        
        # Compare statistics before and after cleaning
        numeric_columns = self._get_numeric_columns(original_data)
        
        for col in numeric_columns:
            original_stats = {
                'mean': original_data[col].mean(),
                'std': original_data[col].std(),
                'skewness': original_data[col].skew(),
                'kurtosis': original_data[col].kurtosis()
            }
            
            cleaned_stats = {
                'mean': cleaned_data[col].mean(),
                'std': cleaned_data[col].std(),
                'skewness': cleaned_data[col].skew(),
                'kurtosis': cleaned_data[col].kurtosis()
            }
            
            report['data_statistics'][col] = {
                'before_cleaning': original_stats,
                'after_cleaning': cleaned_stats,
                'improvement': {
                    'std_reduction': (original_stats['std'] - cleaned_stats['std']) / original_stats['std'] * 100,
                    'kurtosis_reduction': (original_stats['kurtosis'] - cleaned_stats['kurtosis']) / abs(original_stats['kurtosis']) * 100
                }
            }
        
        return report
