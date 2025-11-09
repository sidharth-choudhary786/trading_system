# trading_system/data/data_manager.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import logging
from pathlib import Path

from ..core.exceptions import DataSourceError
from .mapping.master_mapping import MasterMapping
from .mapping.ticker_resolver import TickerResolver
from .sources.twelvedata import TwelveDataDataSource
from .sources.yahoo_finance import YahooFinanceDataSource
from .sources.investpy import InvestPyDataSource
from .sources.alphavantage import AlphaVantageDataSource
from .processing.imputer import DataImputer
from .processing.cleaner import DataCleaner
from .processing.normalizer import DataNormalizer
from .processing.quality_assessor import DataQualityAssessor
from .storage.file_store import FileStore

class DataManager:
    """
    Main data orchestration class - handles complete data pipeline
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Data storage
        self.data_directory = Path(config.get('data_directory', 'data'))
        self.file_store = FileStore(self.data_directory)
        
        self.logger.info("DataManager initialized successfully")
    
    def _initialize_components(self):
        """Initialize all data components"""
        # Mapping
        self.master_mapping = MasterMapping(self.config)
        self.ticker_resolver = TickerResolver(self.config)
        
        # Data sources
        self.data_sources = {
            'twelvedata': TwelveDataDataSource(self.config),
            'yfinance': YahooFinanceDataSource(self.config),
            'investpy': InvestPyDataSource(self.config),
            'alphavantage': AlphaVantageDataSource(self.config)
        }
        
        # Data processing
        self.imputer = DataImputer(self.config)
        self.cleaner = DataCleaner(self.config)
        self.normalizer = DataNormalizer(self.config)
        self.quality_assessor = DataQualityAssessor(self.config)
    
    def download_instrument_data(
        self, 
        symbol: str,
        start_date: str = "2005-01-01",
        end_date: Optional[str] = None,
        force_redownload: bool = False
    ) -> pd.DataFrame:
        """
        Download data for a single instrument with fallback logic
        
        Args:
            symbol: Instrument symbol
            start_date: Start date for data
            end_date: End date for data (default: today)
            force_redownload: Force redownload even if data exists
        
        Returns:
            DataFrame with standardized OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        self.logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
        
        # Check if data already exists
        if not force_redownload:
            existing_data = self.file_store.load_data(symbol, 'processed')
            if existing_data is not None:
                self.logger.info(f"Found existing data for {symbol}")
                return existing_data
        
        # Get instrument mapping
        instrument_info = self.master_mapping.get_instrument(symbol)
        if not instrument_info:
            raise DataSourceError(f"No mapping found for symbol: {symbol}")
        
        # Try different exchanges and sources
        raw_data = self._download_with_fallback(instrument_info, start_date, end_date)
        
        if raw_data.empty:
            raise DataSourceError(f"Could not download data for {symbol} from any source")
        
        # Process the data
        processed_data = self._process_data(raw_data, symbol)
        
        # Save processed data
        self.file_store.save_data(symbol, processed_data, 'processed')
        
        self.logger.info(f"Successfully processed data for {symbol}")
        return processed_data
    
    def _download_with_fallback(
        self, 
        instrument_info: Dict, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Download data with fallback logic (NSE -> BSE, multiple sources)
        """
        exchanges = ['NSE', 'BSE']
        sources_order = ['twelvedata', 'yfinance', 'investpy', 'alphavantage']
        
        for exchange in exchanges:
            exchange_symbol = instrument_info.get(f'{exchange.lower()}_symbol')
            if not exchange_symbol:
                continue
                
            for source_name in sources_order:
                try:
                    self.logger.info(f"Trying {source_name} for {exchange_symbol} on {exchange}")
                    
                    source = self.data_sources[source_name]
                    raw_data = source.download_data(exchange_symbol, start_date, end_date)
                    
                    if not raw_data.empty:
                        raw_data['source'] = source_name
                        raw_data['exchange'] = exchange
                        self.logger.info(f"Successfully downloaded from {source_name}")
                        return raw_data
                        
                except Exception as e:
                    self.logger.warning(f"Failed to download from {source_name}: {e}")
                    continue
        
        return pd.DataFrame()
    
    def _process_data(self, raw_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Process raw data through complete pipeline
        """
        self.logger.info(f"Processing data for {symbol}")
        
        # Standardize column names
        processed_data = self._standardize_columns(raw_data)
        
        # Handle missing data
        processed_data = self.imputer.impute(processed_data)
        
        # Clean outliers
        processed_data = self.cleaner.clean(processed_data)
        
        # Normalize data
        processed_data = self.normalizer.normalize(processed_data)
        
        # Add metadata
        processed_data['symbol'] = symbol
        processed_data['currency'] = 'INR'
        processed_data['timezone'] = 'Asia/Kolkata'
        
        # Ensure adjusted close exists
        if 'adj_close' not in processed_data.columns:
            processed_data['adj_close'] = processed_data['close']
        
        # Standardize date format
        if 'date' in processed_data.columns:
            processed_data['date'] = pd.to_datetime(processed_data['date'])
            processed_data['date'] = processed_data['date'].dt.strftime('%d/%m/%Y %H:%M:%S IST')
        
        return processed_data
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across all sources"""
        column_mapping = {
            # TwelveData
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume',
            # Yahoo Finance
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            'Adj Close': 'adj_close',
            # InvestPy
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            # Alpha Vantage
            '1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'
        }
        
        standardized_df = df.rename(columns=column_mapping)
        
        # Ensure all required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in standardized_df.columns:
                standardized_df[col] = np.nan
        
        return standardized_df
    
    def download_universe_data(
        self,
        symbols: List[str],
        start_date: str = "2005-01-01",
        end_date: Optional[str] = None,
        force_redownload: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple symbols
        
        Args:
            symbols: List of symbols to download
            start_date: Start date for data
            end_date: End date for data
            force_redownload: Force redownload
        
        Returns:
            Dictionary of symbol -> DataFrame
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.download_instrument_data(symbol, start_date, end_date, force_redownload)
                results[symbol] = data
                self.logger.info(f"Downloaded data for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Failed to download data for {symbol}: {e}")
                continue
        
        return results
    
    def get_data_quality_report(self, symbol: str) -> Dict:
        """
        Generate data quality report for a symbol
        """
        data = self.file_store.load_data(symbol, 'processed')
        if data is None:
            raise DataSourceError(f"No data found for {symbol}")
        
        return self.quality_assessor.assess_quality(data, symbol)
    
    def update_data_incrementally(self, symbol: str) -> pd.DataFrame:
        """
        Update data for symbol incrementally
        """
        # Load existing data
        existing_data = self.file_store.load_data(symbol, 'processed')
        
        if existing_data is None:
            return self.download_instrument_data(symbol)
        
        # Find last available date
        if 'date' in existing_data.columns:
            last_date = pd.to_datetime(existing_data['date']).max()
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start_date = "2005-01-01"
        
        # Download new data
        new_data = self.download_instrument_data(symbol, start_date=start_date)
        
        if not new_data.empty:
            # Combine with existing data
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data = combined_data.drop_duplicates(subset=['date'], keep='last')
            
            # Save updated data
            self.file_store.save_data(symbol, combined_data, 'processed')
            return combined_data
        
        return existing_data
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with available data"""
        return self.file_store.get_available_symbols('processed')
