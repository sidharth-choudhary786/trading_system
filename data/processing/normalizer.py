# trading_system/data/processing/normalizer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

from ...core.exceptions import DataSourceError

class DataNormalizer:
    """
    Handle data normalization and standardization using multiple methods
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.method = config.get('normalization_method', 'zscore')
        
        # Method configurations
        self.method_configs = {
            'zscore': {'name': 'Z-Score Standardization'},
            'minmax': {'name': 'Min-Max Normalization', 'feature_range': (0, 1)},
            'robust': {'name': 'Robust Scaling'},
            'log': {'name': 'Log Transformation'},
            'decimal': {'name': 'Decimal Scaling'},
            'maxabs': {'name': 'Max Absolute Scaling'}
        }
        
        # Store scalers for inverse transformation
        self.scalers = {}
    
    def normalize(self, data: pd.DataFrame, method: Optional[str] = None) -> pd.DataFrame:
        """
        Normalize data using specified method
        
        Args:
            data: DataFrame to normalize
            method: Normalization method to use
            
        Returns:
            Normalized DataFrame
        """
        if method is None:
            method = self.method
        
        if data.empty:
            return data
        
        # Create a copy to avoid modifying original
        normalized_data = data.copy()
        
        numeric_columns = self._get_numeric_columns(data)
        if not numeric_columns:
            self.logger.warning("No numeric columns found for normalization")
            return normalized_data
        
        self.logger.info(f"Normalizing {len(numeric_columns)} columns using {method} method")
        
        # Apply normalization
        if method == 'zscore':
            normalized_data = self._zscore_normalize(normalized_data, numeric_columns)
        elif method == 'minmax':
            normalized_data = self._minmax_normalize(normalized_data, numeric_columns)
        elif method == 'robust':
            normalized_data = self._robust_normalize(normalized_data, numeric_columns)
        elif method == 'log':
            normalized_data = self._log_normalize(normalized_data, numeric_columns)
        elif method == 'decimal':
            normalized_data = self._decimal_normalize(normalized_data, numeric_columns)
        elif method == 'maxabs':
            normalized_data = self._maxabs_normalize(normalized_data, numeric_columns)
        else:
            self.logger.warning(f"Unknown normalization method: {method}. Using Z-Score.")
            normalized_data = self._zscore_normalize(normalized_data, numeric_columns)
        
        self.logger.info("Data normalization complete")
        return normalized_data
    
    def _zscore_normalize(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Z-Score standardization"""
        self.scalers['zscore'] = {}
        
        for col in columns:
            scaler = StandardScaler()
            # Reshape for sklearn
            values = data[col].values.reshape(-1, 1)
            normalized_values = scaler.fit_transform(values)
            data[col] = normalized_values.flatten()
            
            # Store scaler for inverse transformation
            self.scalers['zscore'][col] = scaler
        
        return data
    
    def _minmax_normalize(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Min-Max normalization"""
        feature_range = self.method_configs['minmax'].get('feature_range', (0, 1))
        self.scalers['minmax'] = {}
        
        for col in columns:
            scaler = MinMaxScaler(feature_range=feature_range)
            values = data[col].values.reshape(-1, 1)
            normalized_values = scaler.fit_transform(values)
            data[col] = normalized_values.flatten()
            
            self.scalers['minmax'][col] = scaler
        
        return data
    
    def _robust_normalize(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Robust scaling (using median and IQR)"""
        self.scalers['robust'] = {}
        
        for col in columns:
            scaler = RobustScaler()
            values = data[col].values.reshape(-1, 1)
            normalized_values = scaler.fit_transform(values)
            data[col] = normalized_values.flatten()
            
            self.scalers['robust'][col] = scaler
        
        return data
    
    def _log_normalize(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Log transformation"""
        self.scalers['log'] = {}
        
        for col in columns:
            # Add small constant to avoid log(0)
            min_val = data[col].min()
            constant = 1 - min_val if min_val <= 0 else 0
            
            # Store parameters for inverse transformation
            self.scalers['log'][col] = {'constant': constant}
            
            # Apply log transformation
            data[col] = np.log(data[col] + constant)
        
        return data
    
    def _decimal_normalize(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Decimal scaling normalization"""
        self.scalers['decimal'] = {}
        
        for col in columns:
            max_abs = np.max(np.abs(data[col]))
            if max_abs == 0:
                continue
                
            # Find the number of decimal places needed
            decimal_places = np.ceil(np.log10(max_abs))
            scaling_factor = 10 ** decimal_places
            
            # Store scaling factor for inverse transformation
            self.scalers['decimal'][col] = {'scaling_factor': scaling_factor}
            
            # Apply decimal scaling
            data[col] = data[col] / scaling_factor
        
        return data
    
    def _maxabs_normalize(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Max absolute scaling"""
        self.scalers['maxabs'] = {}
        
        for col in columns:
            max_abs = np.max(np.abs(data[col]))
            if max_abs == 0:
                continue
                
            # Store max absolute value for inverse transformation
            self.scalers['maxabs'][col] = {'max_abs': max_abs}
            
            # Apply max absolute scaling
            data[col] = data[col] / max_abs
        
        return data
    
    def inverse_transform(self, data: pd.DataFrame, method: Optional[str] = None) -> pd.DataFrame:
        """
        Inverse transform normalized data back to original scale
        
        Args:
            data: Normalized DataFrame
            method: Normalization method used originally
            
        Returns:
            DataFrame in original scale
        """
        if method is None:
            method = self.method
        
        if data.empty or method not in self.scalers:
            return data
        
        # Create a copy to avoid modifying original
        original_scale_data = data.copy()
        
        numeric_columns = self._get_numeric_columns(data)
        scaler_info = self.scalers[method]
        
        self.logger.info(f"Applying inverse transformation for {method} method")
        
        if method == 'zscore':
            for col in numeric_columns:
                if col in scaler_info:
                    scaler = scaler_info[col]
                    values = data[col].values.reshape(-1, 1)
                    original_values = scaler.inverse_transform(values)
                    original_scale_data[col] = original_values.flatten()
        
        elif method == 'minmax':
            for col in numeric_columns:
                if col in scaler_info:
                    scaler = scaler_info[col]
                    values = data[col].values.reshape(-1, 1)
                    original_values = scaler.inverse_transform(values)
                    original_scale_data[col] = original_values.flatten()
        
        elif method == 'robust':
            for col in numeric_columns:
                if col in scaler_info:
                    scaler = scaler_info[col]
                    values = data[col].values.reshape(-1, 1)
                    original_values = scaler.inverse_transform(values)
                    original_scale_data[col] = original_values.flatten()
        
        elif method == 'log':
            for col in numeric_columns:
                if col in scaler_info:
                    constant = scaler_info[col]['constant']
                    original_scale_data[col] = np.exp(data[col]) - constant
        
        elif method == 'decimal':
            for col in numeric_columns:
                if col in scaler_info:
                    scaling_factor = scaler_info[col]['scaling_factor']
                    original_scale_data[col] = data[col] * scaling_factor
        
        elif method == 'maxabs':
            for col in numeric_columns:
                if col in scaler_info:
                    max_abs = scaler_info[col]['max_abs']
                    original_scale_data[col] = data[col] * max_abs
        
        return original_scale_data
    
    def _get_numeric_columns(self, data: pd.DataFrame) -> List[str]:
        """Get numeric columns from DataFrame"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols
    
    def get_normalization_report(self, original_data: pd.DataFrame, normalized_data: pd.DataFrame) -> Dict:
        """Generate normalization report"""
        report = {
            'method_used': self.method,
            'columns_normalized': self._get_numeric_columns(original_data),
            'statistics_comparison': {}
        }
        
        numeric_columns = self._get_numeric_columns(original_data)
        
        for col in numeric_columns:
            original_stats = {
                'mean': original_data[col].mean(),
                'std': original_data[col].std(),
                'min': original_data[col].min(),
                'max': original_data[col].max(),
                'range': original_data[col].max() - original_data[col].min()
            }
            
            normalized_stats = {
                'mean': normalized_data[col].mean(),
                'std': normalized_data[col].std(),
                'min': normalized_data[col].min(),
                'max': normalized_data[col].max(),
                'range': normalized_data[col].max() - normalized_data[col].min()
            }
            
            report['statistics_comparison'][col] = {
                'original': original_stats,
                'normalized': normalized_stats
            }
        
        return report
