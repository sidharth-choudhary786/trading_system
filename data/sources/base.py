# trading_system/data/sources/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

from ...core.exceptions import DataSourceError

class BaseDataSource(ABC):
    """
    Abstract base class for all data sources
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.name = self.__class__.__name__
        self.rate_limit = config.get('rate_limit', 1)
        self.timeout = config.get('timeout', 30)
    
    @abstractmethod
    def download_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1day"
    ) -> pd.DataFrame:
        """
        Download OHLCV data for symbol
        
        Args:
            symbol: Instrument symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1day, 1hour, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def download_dividends(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download dividend data
        
        Args:
            symbol: Instrument symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with dividend data
        """
        pass
    
    @abstractmethod
    def download_splits(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download stock split data
        
        Args:
            symbol: Instrument symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with split data
        """
        pass
    
    def get_data_quality(self, data: pd.DataFrame) -> Dict:
        """
        Assess data quality
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary with quality metrics
        """
        if data.empty:
            return {
                'completeness': 0.0,
                'consistency': 0.0,
                'freshness': 0.0,
                'overall_score': 0.0
            }
        
        metrics = {}
        
        # Completeness - percentage of non-null values
        metrics['completeness'] = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        
        # Consistency - check for data anomalies
        price_columns = ['open', 'high', 'low', 'close']
        if all(col in data.columns for col in price_columns):
            anomalies = 0
            for _, row in data.iterrows():
                if not (row['low'] <= row['open'] <= row['high'] and 
                        row['low'] <= row['close'] <= row['high']):
                    anomalies += 1
            metrics['consistency'] = 1.0 - (anomalies / len(data))
        else:
            metrics['consistency'] = 0.0
        
        # Freshness - how recent is the data
        if 'date' in data.columns:
            latest_date = pd.to_datetime(data['date']).max()
            days_old = (datetime.now() - latest_date).days
            metrics['freshness'] = max(0.0, 1.0 - (days_old / 30))  # 30 days threshold
        else:
            metrics['freshness'] = 0.0
        
        # Overall score (weighted average)
        weights = {'completeness': 0.4, 'consistency': 0.4, 'freshness': 0.2}
        metrics['overall_score'] = sum(metrics[k] * weights[k] for k in weights)
        
        return metrics
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate symbol format for this data source
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            True if valid
        """
        # Basic validation - can be overridden by specific sources
        return isinstance(symbol, str) and len(symbol) > 0
    
    def get_source_info(self) -> Dict:
        """
        Get information about this data source
        
        Returns:
            Dictionary with source information
        """
        return {
            'name': self.name,
            'rate_limit': self.rate_limit,
            'timeout': self.timeout,
            'supported_intervals': ['1day'],  # Default, can be overridden
            'supported_exchanges': []  # To be overridden
        }
