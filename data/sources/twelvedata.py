# trading_system/data/sources/twelvedata.py
import pandas as pd
import requests
from typing import Dict, Optional
import time
import os
from datetime import datetime

from .base import BaseDataSource
from ...core.exceptions import DataSourceError

class TwelveDataDataSource(BaseDataSource):
    """
    TwelveData data source implementation
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = config.get('twelvedata_api_key') or os.getenv('TWELVEDATA_API_KEY')
        if not self.api_key:
            raise DataSourceError("TwelveData API key not provided")
        
        self.base_url = "https://api.twelvedata.com"
        self.rate_limit = 8  # requests per minute
    
    def download_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1day"
    ) -> pd.DataFrame:
        """
        Download OHLCV data from TwelveData
        """
        try:
            # Rate limiting
            time.sleep(60 / self.rate_limit)
            
            url = f"{self.base_url}/time_series"
            params = {
                'symbol': symbol,
                'interval': interval,
                'start_date': start_date,
                'end_date': end_date,
                'apikey': self.api_key,
                'outputsize': 5000
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'ok':
                raise DataSourceError(f"TwelveData API error: {data.get('message', 'Unknown error')}")
            
            if 'values' not in data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            
            # Standardize column names
            df = df.rename(columns={
                'datetime': 'date',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Convert data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert date
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise DataSourceError(f"TwelveData network error: {e}")
        except Exception as e:
            raise DataSourceError(f"TwelveData data error: {e}")
    
    def download_dividends(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download dividends from TwelveData
        """
        # TwelveData doesn't provide dividends in basic plan
        # Return empty DataFrame
        return pd.DataFrame(columns=['date', 'dividend'])
    
    def download_splits(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download stock splits from TwelveData
        """
        # TwelveData doesn't provide splits in basic plan
        # Return empty DataFrame  
        return pd.DataFrame(columns=['date', 'split'])
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate TwelveData symbol format"""
        return symbol.endswith(('.NS', '.BO'))
    
    def get_source_info(self) -> Dict:
        """Get TwelveData source information"""
        info = super().get_source_info()
        info.update({
            'name': 'TwelveData',
            'supported_intervals': ['1min', '5min', '15min', '30min', '45min', '1h', '2h', '4h', '1day', '1week', '1month'],
            'supported_exchanges': ['NSE', 'BSE'],
            'has_dividends': False,
            'has_splits': False,
            'has_adj_close': False
        })
        return info
