# trading_system/data/processing/imputer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

from ...core.exceptions import DataSourceError

class DataImputer:
    """
    Handle missing data imputation using multiple methods
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.method = config.get('imputation_method', 'locf')
        
        # Method configurations
        self.method_configs = {
            'locf': {'name': 'Last Observation Carried Forward'},
            'linear': {'name': 'Linear Interpolation'},
            'spline': {'name': 'Spline Interpolation', 'order': 3},
            'knn': {'name': 'K-Nearest Neighbors', 'n_neighbors': 5},
            'mice': {'name': 'Multiple Imputation by Chained Equations', 'max_iter': 10},
            'mean': {'name': 'Mean Imputation'},
            'median': {'name': 'Median Imputation'}
        }
    
    def impute(self, data: pd.DataFrame, method: Optional[str] = None) -> pd.DataFrame:
        """
        Impute missing values in data
        
        Args:
            data: DataFrame with potential missing values
            method: Imputation method to use
            
        Returns:
            DataFrame with imputed values
        """
        if method is None:
            method = self.method
        
        if data.empty:
            return data
        
        # Create a copy to avoid modifying original
        imputed_data = data.copy()
        
        # Check for missing values
        missing_before = imputed_data.isnull().sum().sum()
        if missing_before == 0:
            self.logger.debug("No missing values found for imputation")
            return imputed_data
        
        self.logger.info(f"Imputing {missing_before} missing values using {method} method")
        
        # Apply selected imputation method
        if method == 'locf':
            imputed_data = self._locf_impute(imputed_data)
        elif method == 'linear':
            imputed_data = self._linear_impute(imputed_data)
        elif method == 'spline':
            imputed_data = self._spline_impute(imputed_data)
        elif method == 'knn':
            imputed_data = self._knn_impute(imputed_data)
        elif method == 'mice':
            imputed_data = self._mice_impute(imputed_data)
        elif method == 'mean':
            imputed_data = self._mean_impute(imputed_data)
        elif method == 'median':
            imputed_data = self._median_impute(imputed_data)
        else:
            self.logger.warning(f"Unknown imputation method: {method}. Using LOCF.")
            imputed_data = self._locf_impute(imputed_data)
        
        # Check results
        missing_after = imputed_data.isnull().sum().sum()
        self.logger.info(f"Imputation complete: {missing_before} -> {missing_after} missing values")
        
        return imputed_data
    
    def _locf_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Last Observation Carried Forward imputation"""
        # Forward fill then backward fill
        imputed = data.ffill().bfill()
        return imputed
    
    def _linear_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Linear interpolation imputation"""
        numeric_columns = self._get_numeric_columns(data)
        
        for col in numeric_columns:
            if data[col].isnull().any():
                data[col] = data[col].interpolate(
                    method='linear', 
                    limit_direction='both'
                )
        
        # Fill any remaining missing values with LOCF
        return self._locf_impute(data)
    
    def _spline_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Spline interpolation imputation"""
        numeric_columns = self._get_numeric_columns(data)
        order = self.method_configs['spline'].get('order', 3)
        
        for col in numeric_columns:
            if data[col].isnull().any():
                try:
                    data[col] = data[col].interpolate(
                        method='spline', 
                        order=order,
                        limit_direction='both'
                    )
                except:
                    # Fallback to linear if spline fails
                    data[col] = data[col].interpolate(
                        method='linear', 
                        limit_direction='both'
                    )
        
        return self._locf_impute(data)
    
    def _knn_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """K-Nearest Neighbors imputation"""
        numeric_columns = self._get_numeric_columns(data)
        
        if len(numeric_columns) < 2:
            self.logger.warning("KNN requires at least 2 numeric columns. Using linear imputation.")
            return self._linear_impute(data)
        
        # Extract numeric data for imputation
        numeric_data = data[numeric_columns].copy()
        
        # Use KNN imputer
        n_neighbors = self.method_configs['knn'].get('n_neighbors', 5)
        imputer = KNNImputer(n_neighbors=n_neighbors)
        
        imputed_values = imputer.fit_transform(numeric_data)
        data[numeric_columns] = imputed_values
        
        return data
    
    def _mice_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Multiple Imputation by Chained Equations"""
        numeric_columns = self._get_numeric_columns(data)
        
        if len(numeric_columns) < 2:
            self.logger.warning("MICE requires at least 2 numeric columns. Using linear imputation.")
            return self._linear_impute(data)
        
        # Extract numeric data
        numeric_data = data[numeric_columns].copy()
        
        # Use IterativeImputer (MICE implementation in sklearn)
        max_iter = self.method_configs['mice'].get('max_iter', 10)
        imputer = IterativeImputer(max_iter=max_iter, random_state=42)
        
        imputed_values = imputer.fit_transform(numeric_data)
        data[numeric_columns] = imputed_values
        
        return data
    
    def _mean_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mean value imputation"""
        numeric_columns = self._get_numeric_columns(data)
        
        for col in numeric_columns:
            if data[col].isnull().any():
                mean_val = data[col].mean()
                data[col] = data[col].fillna(mean_val)
        
        return data
    
    def _median_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Median value imputation"""
        numeric_columns = self._get_numeric_columns(data)
        
        for col in numeric_columns:
            if data[col].isnull().any():
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val)
        
        return data
    
    def _get_numeric_columns(self, data: pd.DataFrame) -> List[str]:
        """Get numeric columns from DataFrame"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols
    
    def get_imputation_report(self, original_data: pd.DataFrame, imputed_data: pd.DataFrame) -> Dict:
        """Generate imputation report"""
        report = {
            'method_used': self.method,
            'missing_before': original_data.isnull().sum().to_dict(),
            'missing_after': imputed_data.isnull().sum().to_dict(),
            'total_missing_before': original_data.isnull().sum().sum(),
            'total_missing_after': imputed_data.isnull().sum().sum(),
            'columns_imputed': [],
            'imputation_effect': {}
        }
        
        # Calculate imputation effect on statistics
        numeric_columns = self._get_numeric_columns(original_data)
        
        for col in numeric_columns:
            if original_data[col].isnull().any():
                report['columns_imputed'].append(col)
                
                # Calculate statistical changes
                original_stats = {
                    'mean': original_data[col].mean(),
                    'std': original_data[col].std(),
                    'min': original_data[col].min(),
                    'max': original_data[col].max()
                }
                
                imputed_stats = {
                    'mean': imputed_data[col].mean(),
                    'std': imputed_data[col].std(),
                    'min': imputed_data[col].min(),
                    'max': imputed_data[col].max()
                }
                
                report['imputation_effect'][col] = {
                    'original': original_stats,
                    'imputed': imputed_stats,
                    'change': {
                        'mean_change': imputed_stats['mean'] - original_stats['mean'],
                        'std_change': imputed_stats['std'] - original_stats['std']
                    }
                }
        
        return report
