# trading_system/data/sources/yahoo_finance.py
import yfinance as yf
import pandas as pd
from typing import Dict
import time

from .base import BaseDataSource
from ...core.exceptions import DataSourceError

class YahooFinanceDataSource(BaseDataSource):
    """
    Yahoo Finance data source implementation
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.rate_limit = 2000  # Very high rate limit
    
    def download_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1day"
    ) -> pd.DataFrame:
        """
        Download OHLCV data from Yahoo Finance
        """
        try:
            # Rate limiting (minimal for yfinance)
            time.sleep(0.1)
            
            # Convert interval to yfinance format
            yf_interval = self._convert_interval(interval)
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
                auto_adjust=False  # We want raw data to adjust ourselves
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
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'splits'
            })
            
            # Ensure all required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None
            
            return df
            
        except Exception as e:
            raise DataSourceError(f"Yahoo Finance error for {symbol}: {e}")
    
    def download_dividends(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download dividends from Yahoo Finance
        """
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends
            
            if dividends.empty:
                return pd.DataFrame()
            
            # Filter by date range
            dividends = dividends.loc[start_date:end_date]
            
            # Convert to DataFrame
            df = dividends.reset_index()
            df = df.rename(columns={'Date': 'date', 'Dividends': 'dividend'})
            
            return df
            
        except Exception as e:
            raise DataSourceError(f"Yahoo Finance dividends error for {symbol}: {e}")
    
    def download_splits(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download stock splits from Yahoo Finance
        """
        try:
            ticker = yf.Ticker(symbol)
            splits = ticker.splits
            
            if splits.empty:
                return pd.DataFrame()
            
            # Filter by date range
            splits = splits.loc[start_date:end_date]
            
            # Convert to DataFrame
            df = splits.reset_index()
            df = df.rename(columns={'Date': 'date', 'Stock Splits': 'split'})
            
            return df
            
        except Exception as e:
            raise DataSourceError(f"Yahoo Finance splits error for {symbol}: {e}")
    
    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to yfinance interval"""
        interval_map = {
            '1day': '1d',
            '1hour': '1h',
            '1min': '1m',
            '5min': '5m',
            '15min': '15m',
            '30min': '30m',
            '1week': '1wk',
            '1month': '1mo'
        }
        return interval_map.get(interval, '1d')
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate Yahoo Finance symbol format"""
        return symbol.endswith(('.NS', '.BO'))
    
    def get_source_info(self) -> Dict:
        """Get Yahoo Finance source information"""
        info = super().get_source_info()
        info.update({
            'name': 'Yahoo Finance',
            'supported_intervals': ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'],
            'supported_exchanges': ['NSE', 'BSE'],
            'has_dividends': True,
            'has_splits': True,
            'has_adj_close': True
        })
        return info
