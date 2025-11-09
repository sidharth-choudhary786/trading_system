# trading_system/data/sources/investpy.py
import investpy
import pandas as pd
from typing import Dict
import time

from .base import BaseDataSource
from ...core.exceptions import DataSourceError

class InvestPyDataSource(BaseDataSource):
    """
    InvestPy (Investing.com) data source implementation
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.country = config.get('country', 'india')
        self.rate_limit = 1  # Conservative rate limit
    
    def download_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1day"
    ) -> pd.DataFrame:
        """
        Download OHLCV data from InvestPy
        """
        try:
            # Rate limiting
            time.sleep(60 / self.rate_limit)
            
            # Convert date format for InvestPy (DD/MM/YYYY)
            start_date_str = self._convert_date_format(start_date)
            end_date_str = self._convert_date_format(end_date)
            
            # Download data
            df = investpy.get_stock_historical_data(
                stock=symbol,
                country=self.country,
                from_date=start_date_str,
                to_date=end_date_str
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # Reset index to get Date as column
            df = df.reset_index()
            
            # Standardize column names
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close', 
                'Volume': 'volume'
            })
            
            return df
            
        except Exception as e:
            raise DataSourceError(f"InvestPy error for {symbol}: {e}")
    
    def download_dividends(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download dividends from InvestPy
        """
        try:
            # Convert date format
            start_date_str = self._convert_date_format(start_date)
            end_date_str = self._convert_date_format(end_date)
            
            # Get dividends
            dividends = investpy.get_stock_dividends(
                stock=symbol,
                country=self.country,
                from_date=start_date_str,
                to_date=end_date_str
            )
            
            if dividends.empty:
                return pd.DataFrame()
            
            # Standardize column names
            dividends = dividends.rename(columns={
                'Date': 'date',
                'Dividend': 'dividend'
            })
            
            return dividends
            
        except Exception as e:
            # InvestPy may not have dividend data for all stocks
            return pd.DataFrame(columns=['date', 'dividend'])
    
    def download_splits(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download stock splits from InvestPy
        """
        try:
            # Convert date format
            start_date_str = self._convert_date_format(start_date)
            end_date_str = self._convert_date_format(end_date)
            
            # Get splits
            splits = investpy.get_stock_splits(
                stock=symbol,
                country=self.country,
                from_date=start_date_str,
                to_date=end_date_str
            )
            
            if splits.empty:
                return pd.DataFrame()
            
            # Standardize column names
            splits = splits.rename(columns={
                'Date': 'date',
                'Split': 'split'
            })
            
            return splits
            
        except Exception as e:
            # InvestPy may not have split data for all stocks
            return pd.DataFrame(columns=['date', 'split'])
    
    def _convert_date_format(self, date_str: str) -> str:
        """Convert YYYY-MM-DD to DD/MM/YYYY"""
        try:
            date_obj = pd.to_datetime(date_str)
            return date_obj.strftime('%d/%m/%Y')
        except:
            return date_str
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate InvestPy symbol format"""
        # InvestPy uses plain symbols without exchange suffixes
        return isinstance(symbol, str) and len(symbol) > 0 and '.' not in symbol
    
    def get_source_info(self) -> Dict:
        """Get InvestPy source information"""
        info = super().get_source_info()
        info.update({
            'name': 'Investing.com (InvestPy)',
            'supported_intervals': ['1day'],
            'supported_exchanges': ['NSE', 'BSE'],
            'has_dividends': True,
            'has_splits': True,
            'has_adj_close': False
        })
        return info
