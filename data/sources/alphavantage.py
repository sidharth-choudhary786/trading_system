# trading_system/data/sources/alphavantage.py
import requests
import pandas as pd
from typing import Dict, Optional
import time
import os
from datetime import datetime

from .base import BaseDataSource
from ...core.exceptions import DataSourceError

class AlphaVantageDataSource(BaseDataSource):
    """
    Alpha Vantage data source implementation
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = config.get('alphavantage_api_key') or os.getenv('ALPHAVANTAGE_API_KEY')
        if not self.api_key:
            raise DataSourceError("Alpha Vantage API key not provided")
        
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit = 5  # Free tier rate limit
    
    def download_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1day"
    ) -> pd.DataFrame:
        """
        Download OHLCV data from Alpha Vantage
        """
        try:
            # Rate limiting
            time.sleep(60 / self.rate_limit)
            
            # Alpha Vantage uses different function for intraday vs daily
            if interval == "1day":
                return self._download_daily_data(symbol)
            else:
                return self._download_intraday_data(symbol, interval)
                
        except Exception as e:
            raise DataSourceError(f"Alpha Vantage error for {symbol}: {e}")
    
    def _download_daily_data(self, symbol: str) -> pd.DataFrame:
        """Download daily data from Alpha Vantage"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full',
            'datatype': 'json'
        }
        
        response = requests.get(self.base_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            error_msg = data.get('Error Message') or data.get('Note', 'Unknown error')
            raise DataSourceError(f"Alpha Vantage API error: {error_msg}")
        
        time_series = data['Time Series (Daily)']
        
        # Convert to DataFrame
        records = []
        for date, values in time_series.items():
            records.append({
                'date': date,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': float(values['5. volume'])
            })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        return df
    
    def _download_intraday_data(self, symbol: str, interval: str) -> pd.DataFrame:
        """Download intraday data from Alpha Vantage"""
        # Map interval to Alpha Vantage format
        interval_map = {
            '1min': '1min',
            '5min': '5min', 
            '15min': '15min',
            '30min': '30min',
            '1hour': '60min'
        }
        
        av_interval = interval_map.get(interval, '5min')
        
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': av_interval,
            'apikey': self.api_key,
            'outputsize': 'full',
            'datatype': 'json'
        }
        
        response = requests.get(self.base_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        data = response.json()
        
        time_series_key = f'Time Series ({av_interval})'
        if time_series_key not in data:
            error_msg = data.get('Error Message') or data.get('Note', 'Unknown error')
            raise DataSourceError(f"Alpha Vantage API error: {error_msg}")
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        records = []
        for datetime_str, values in time_series.items():
            records.append({
                'date': datetime_str,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': float(values['5. volume'])
            })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        return df
    
    def download_dividends(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download dividends from Alpha Vantage
        """
        try:
            # Rate limiting
            time.sleep(60 / self.rate_limit)
            
            params = {
                'function': 'DIVIDENDS',
                'symbol': symbol,
                'apikey': self.api_key,
                'datatype': 'json'
            }
            
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if 'dividends' not in data:
                return pd.DataFrame()
            
            dividends = data['dividends']
            
            # Convert to DataFrame
            records = []
            for dividend in dividends:
                records.append({
                    'date': dividend['date'],
                    'dividend': float(dividend['dividend'])
                })
            
            df = pd.DataFrame(records)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                # Filter by date range
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            return df
            
        except Exception as e:
            return pd.DataFrame(columns=['date', 'dividend'])
    
    def download_splits(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download stock splits from Alpha Vantage
        """
        try:
            # Rate limiting
            time.sleep(60 / self.rate_limit)
            
            params = {
                'function': 'SPLITS',
                'symbol': symbol,
                'apikey': self.api_key,
                'datatype': 'json'
            }
            
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if 'splits' not in data:
                return pd.DataFrame()
            
            splits = data['splits']
            
            # Convert to DataFrame
            records = []
            for split in splits:
                records.append({
                    'date': split['date'],
                    'split': split['splitFactor']
                })
            
            df = pd.DataFrame(records)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                # Filter by date range
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            return df
            
        except Exception as e:
            return pd.DataFrame(columns=['date', 'split'])
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate Alpha Vantage symbol format"""
        # Alpha Vantage uses plain symbols for Indian stocks
        return isinstance(symbol, str) and len(symbol) > 0
    
    def get_source_info(self) -> Dict:
        """Get Alpha Vantage source information"""
        info = super().get_source_info()
        info.update({
            'name': 'Alpha Vantage',
            'supported_intervals': ['1min', '5min', '15min', '30min', '60min', '1day'],
            'supported_exchanges': ['NSE', 'BSE'],
            'has_dividends': True,
            'has_splits': True,
            'has_adj_close': False,
            'notes': 'Free tier available, good for fundamental data'
        })
        return info
