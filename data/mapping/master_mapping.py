# trading_system/data/mapping/master_mapping.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import yaml

from ...core.exceptions import DataSourceError

class MasterMapping:
    """
    Master mapping table for NSE/BSE instruments with multi-source ticker mapping
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.mapping_file = Path(config.get('mapping_file', 'data/mapping/instrument_master_mapping.csv'))
        self.mapping_df = None
        
        # Initialize mapping
        self._initialize_mapping()
    
    def _initialize_mapping(self):
        """Initialize or load master mapping"""
        if self.mapping_file.exists():
            self.logger.info(f"Loading existing mapping from {self.mapping_file}")
            self.mapping_df = pd.read_csv(self.mapping_file)
        else:
            self.logger.info("Creating new master mapping")
            self.mapping_df = self._create_empty_mapping()
            self._save_mapping()
    
    def _create_empty_mapping(self) -> pd.DataFrame:
        """Create empty mapping dataframe with correct structure"""
        columns = [
            'name', 'symbol', 'nse_symbol', 'bse_symbol', 'type', 
            'exchange_preferred', 'twelvedata_ticker', 'yfinance_ticker', 
            'investpy_ticker', 'alphavantage_ticker', 'sector_raw', 
            'sector_standard', 'currency', 'timezone', 'max_alloc_pct', 
            'source', 'remarks', 'data_quality_score', 'last_updated'
        ]
        
        return pd.DataFrame(columns=columns)
    
    def create_master_mapping(self, nse_file: Path, bse_file: Path) -> pd.DataFrame:
        """
        Create master mapping table from NSE and BSE files
        
        Args:
            nse_file: Path to NSE instruments file
            bse_file: Path to BSE instruments file
            
        Returns:
            Master mapping DataFrame
        """
        self.logger.info("Creating master mapping from NSE and BSE files")
        
        # Load NSE and BSE data
        nse_df = self._load_exchange_file(nse_file, 'NSE')
        bse_df = self._load_exchange_file(bse_file, 'BSE')
        
        # Combine and remove duplicates
        combined_df = self._combine_exchanges(nse_df, bse_df)
        
        # Classify instrument types
        combined_df['type'] = combined_df['name'].apply(self._classify_instrument_type)
        
        # Generate ticker mappings
        combined_df = self._generate_ticker_mappings(combined_df)
        
        # Set standard metadata
        combined_df['currency'] = 'INR'
        combined_df['timezone'] = 'Asia/Kolkata'
        combined_df['exchange_preferred'] = combined_df.apply(
            self._determine_preferred_exchange, axis=1
        )
        
        # Initialize other columns
        combined_df['sector_raw'] = ''
        combined_df['sector_standard'] = ''
        combined_df['max_alloc_pct'] = None
        combined_df['source'] = ''
        combined_df['remarks'] = ''
        combined_df['data_quality_score'] = 0.0
        combined_df['last_updated'] = pd.Timestamp.now()
        
        self.mapping_df = combined_df
        self._save_mapping()
        
        self.logger.info(f"Master mapping created with {len(combined_df)} instruments")
        return combined_df
    
    def _load_exchange_file(self, file_path: Path, exchange: str) -> pd.DataFrame:
        """Load exchange instrument file"""
        if not file_path.exists():
            raise DataSourceError(f"Exchange file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            df['exchange'] = exchange
            return df
        except Exception as e:
            raise DataSourceError(f"Error loading {exchange} file: {e}")
    
    def _combine_exchanges(self, nse_df: pd.DataFrame, bse_df: pd.DataFrame) -> pd.DataFrame:
        """Combine NSE and BSE data, removing duplicates"""
        # Basic cleaning
        nse_df = self._clean_exchange_data(nse_df)
        bse_df = self._clean_exchange_data(bse_df)
        
        # Combine both exchanges
        combined = pd.concat([nse_df, bse_df], ignore_index=True)
        
        # Remove duplicates - prefer NSE over BSE for same company
        combined = combined.sort_values(['name', 'exchange'], ascending=[True, True])
        combined = combined.drop_duplicates(subset=['name'], keep='first')
        
        # Create separate columns for NSE and BSE symbols
        combined['nse_symbol'] = combined.apply(
            lambda x: x['symbol'] if x['exchange'] == 'NSE' else None, axis=1
        )
        combined['bse_symbol'] = combined.apply(
            lambda x: x['symbol'] if x['exchange'] == 'BSE' else None, axis=1
        )
        
        # Set preferred symbol (NSE first, then BSE)
        combined['symbol'] = combined['nse_symbol'].combine_first(combined['bse_symbol'])
        
        return combined
    
    def _clean_exchange_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean exchange data"""
        # Ensure required columns exist
        required_cols = ['symbol', 'name']
        for col in required_cols:
            if col not in df.columns:
                raise DataSourceError(f"Required column '{col}' missing in exchange data")
        
        # Remove rows with missing essential data
        df = df.dropna(subset=['symbol', 'name'])
        
        # Clean symbol and name
        df['symbol'] = df['symbol'].astype(str).str.strip()
        df['name'] = df['name'].astype(str).str.strip()
        
        return df
    
    def _classify_instrument_type(self, name: str) -> str:
        """Classify instrument type from name"""
        name_upper = name.upper()
        
        if any(etf in name_upper for etf in ['ETF', 'EXCHANGE TRADED FUND']):
            return 'ETF'
        elif any(idx in name_upper for idx in ['INDEX', 'INDICES', 'SENSEX', 'NIFTY']):
            return 'INDEX'
        elif any(bond in name_upper for bond in ['BOND', 'DEBT', 'GILT']):
            return 'BOND'
        else:
            return 'STOCK'
    
    def _generate_ticker_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ticker mappings for all data sources"""
        # TwelveData mapping
        df['twelvedata_ticker'] = df.apply(self._generate_twelvedata_ticker, axis=1)
        
        # Yahoo Finance mapping
        df['yfinance_ticker'] = df.apply(self._generate_yfinance_ticker, axis=1)
        
        # InvestPy mapping
        df['investpy_ticker'] = df.apply(self._generate_investpy_ticker, axis=1)
        
        # Alpha Vantage mapping
        df['alphavantage_ticker'] = df.apply(self._generate_alphavantage_ticker, axis=1)
        
        return df
    
    def _generate_twelvedata_ticker(self, row) -> str:
        """Generate TwelveData ticker"""
        if pd.notna(row['nse_symbol']):
            return f"{row['nse_symbol']}.NS"
        elif pd.notna(row['bse_symbol']):
            return f"{row['bse_symbol']}.BO"
        return ""
    
    def _generate_yfinance_ticker(self, row) -> str:
        """Generate Yahoo Finance ticker"""
        if pd.notna(row['nse_symbol']):
            return f"{row['nse_symbol']}.NS"
        elif pd.notna(row['bse_symbol']):
            return f"{row['bse_symbol']}.BO"
        return ""
    
    def _generate_investpy_ticker(self, row) -> str:
        """Generate InvestPy ticker"""
        if pd.notna(row['nse_symbol']):
            return row['nse_symbol']
        elif pd.notna(row['bse_symbol']):
            return row['bse_symbol']
        return ""
    
    def _generate_alphavantage_ticker(self, row) -> str:
        """Generate Alpha Vantage ticker"""
        if pd.notna(row['nse_symbol']):
            return f"{row['nse_symbol']}.BSE"  # Alpha Vantage uses .BSE for NSE
        elif pd.notna(row['bse_symbol']):
            return row['bse_symbol']
        return ""
    
    def _determine_preferred_exchange(self, row) -> str:
        """Determine preferred exchange (NSE first)"""
        if pd.notna(row['nse_symbol']):
            return 'NSE'
        elif pd.notna(row['bse_symbol']):
            return 'BSE'
        return ''
    
    def get_instrument(self, symbol: str) -> Optional[Dict]:
        """Get instrument details by symbol"""
        if self.mapping_df is None:
            return None
        
        instrument = self.mapping_df[self.mapping_df['symbol'] == symbol]
        if not instrument.empty:
            return instrument.iloc[0].to_dict()
        return None
    
    def get_ticker(self, symbol: str, source: str) -> Optional[str]:
        """Get source-specific ticker for symbol"""
        instrument = self.get_instrument(symbol)
        if not instrument:
            return None
        
        ticker_column = f"{source}_ticker"
        return instrument.get(ticker_column)
    
    def get_all_symbols(self, instrument_type: Optional[str] = None) -> List[str]:
        """Get all symbols, optionally filtered by type"""
        if self.mapping_df is None:
            return []
        
        if instrument_type:
            filtered_df = self.mapping_df[self.mapping_df['type'] == instrument_type]
            return filtered_df['symbol'].tolist()
        else:
            return self.mapping_df['symbol'].tolist()
    
    def update_instrument(self, symbol: str, updates: Dict):
        """Update instrument details"""
        if self.mapping_df is None:
            return
        
        mask = self.mapping_df['symbol'] == symbol
        if mask.any():
            for key, value in updates.items():
                if key in self.mapping_df.columns:
                    self.mapping_df.loc[mask, key] = value
            
            self.mapping_df.loc[mask, 'last_updated'] = pd.Timestamp.now()
            self._save_mapping()
    
    def add_instrument(self, instrument_data: Dict):
        """Add new instrument to mapping"""
        if self.mapping_df is None:
            self.mapping_df = self._create_empty_mapping()
        
        # Ensure required fields
        required_fields = ['name', 'symbol', 'type']
        for field in required_fields:
            if field not in instrument_data:
                raise DataSourceError(f"Required field '{field}' missing")
        
        # Check if symbol already exists
        if instrument_data['symbol'] in self.mapping_df['symbol'].values:
            raise DataSourceError(f"Symbol {instrument_data['symbol']} already exists")
        
        # Add new instrument
        new_row = {col: instrument_data.get(col, '') for col in self.mapping_df.columns}
        new_row['last_updated'] = pd.Timestamp.now()
        
        self.mapping_df = pd.concat([self.mapping_df, pd.DataFrame([new_row])], ignore_index=True)
        self._save_mapping()
    
    def _save_mapping(self):
        """Save mapping to file"""
        if self.mapping_df is not None:
            self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
            self.mapping_df.to_csv(self.mapping_file, index=False)
            self.logger.info(f"Mapping saved to {self.mapping_file}")
    
    def get_mapping_stats(self) -> Dict:
        """Get mapping statistics"""
        if self.mapping_df is None:
            return {}
        
        stats = {
            'total_instruments': len(self.mapping_df),
            'by_type': self.mapping_df['type'].value_counts().to_dict(),
            'by_exchange': self.mapping_df['exchange_preferred'].value_counts().to_dict(),
            'nse_count': self.mapping_df['nse_symbol'].notna().sum(),
            'bse_count': self.mapping_df['bse_symbol'].notna().sum(),
            'last_updated': self.mapping_df['last_updated'].max() if 'last_updated' in self.mapping_df.columns else None
        }
        
        return stats
