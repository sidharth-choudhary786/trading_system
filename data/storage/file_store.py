# trading_system/data/storage/file_store.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import os
import json
import pickle
from pathlib import Path
import logging
from datetime import datetime

from ...core.exceptions import DataSourceError

class FileStore:
    """
    File-based data storage for trading system
    Supports CSV, Parquet, and Pickle formats
    """
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        
        # Create directory structure
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            'raw',
            'processed', 
            'metadata',
            'reports',
            'temp'
        ]
        
        for directory in directories:
            path = self.base_path / directory
            path.mkdir(parents=True, exist_ok=True)
    
    def save_data(
        self, 
        symbol: str, 
        data: pd.DataFrame, 
        data_type: str = 'processed',
        format: str = 'csv'
    ) -> bool:
        """
        Save data to file
        
        Args:
            symbol: Instrument symbol
            data: DataFrame to save
            data_type: Type of data (raw, processed, etc.)
            format: File format (csv, parquet, pickle)
            
        Returns:
            True if successful
        """
        try:
            file_path = self._get_file_path(symbol, data_type, format)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'csv':
                data.to_csv(file_path, index=False)
            elif format == 'parquet':
                data.to_parquet(file_path, index=False)
            elif format == 'pickle':
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise DataSourceError(f"Unsupported format: {format}")
            
            self.logger.info(f"Data saved for {symbol} at {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data for {symbol}: {e}")
            return False
    
    def load_data(
        self, 
        symbol: str, 
        data_type: str = 'processed',
        format: str = 'csv'
    ) -> Optional[pd.DataFrame]:
        """
        Load data from file
        
        Args:
            symbol: Instrument symbol
            data_type: Type of data
            format: File format
            
        Returns:
            DataFrame or None if not found
        """
        try:
            file_path = self._get_file_path(symbol, data_type, format)
            
            if not file_path.exists():
                return None
            
            if format == 'csv':
                data = pd.read_csv(file_path)
            elif format == 'parquet':
                data = pd.read_parquet(file_path)
            elif format == 'pickle':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise DataSourceError(f"Unsupported format: {format}")
            
            # Convert date column if exists
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            self.logger.debug(f"Data loaded for {symbol} from {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def _get_file_path(self, symbol: str, data_type: str, format: str) -> Path:
        """Get file path for symbol and data type"""
        safe_symbol = symbol.replace('/', '_').replace('\\', '_')
        filename = f"{safe_symbol}.{format}"
        return self.base_path / data_type / filename
    
    def get_available_symbols(self, data_type: str = 'processed') -> List[str]:
        """Get list of symbols with available data"""
        data_path = self.base_path / data_type
        symbols = []
        
        if data_path.exists():
            for file_path in data_path.iterdir():
                if file_path.is_file():
                    # Extract symbol from filename
                    symbol = file_path.stem
                    symbols.append(symbol)
        
        return sorted(symbols)
    
    def get_data_info(self, symbol: str, data_type: str = 'processed') -> Dict:
        """Get information about stored data"""
        info = {
            'symbol': symbol,
            'data_type': data_type,
            'exists': False,
            'file_size': 0,
            'last_modified': None,
            'data_points': 0,
            'date_range': None
        }
        
        # Check for different formats
        for format in ['csv', 'parquet', 'pickle']:
            file_path = self._get_file_path(symbol, data_type, format)
            if file_path.exists():
                info['exists'] = True
                info['format'] = format
                info['file_size'] = file_path.stat().st_size
                info['last_modified'] = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                # Load data to get stats
                data = self.load_data(symbol, data_type, format)
                if data is not None:
                    info['data_points'] = len(data)
                    if 'date' in data.columns:
                        dates = pd.to_datetime(data['date'])
                        info['date_range'] = {
                            'start': dates.min().strftime('%Y-%m-%d'),
                            'end': dates.max().strftime('%Y-%m-%d')
                        }
                break
        
        return info
    
    def delete_data(self, symbol: str, data_type: str = 'processed') -> bool:
        """Delete data for symbol"""
        try:
            deleted = False
            for format in ['csv', 'parquet', 'pickle']:
                file_path = self._get_file_path(symbol, data_type, format)
                if file_path.exists():
                    file_path.unlink()
                    deleted = True
                    self.logger.info(f"Deleted {file_path}")
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Error deleting data for {symbol}: {e}")
            return False
    
    def save_metadata(self, symbol: str, metadata: Dict):
        """Save metadata for symbol"""
        try:
            metadata_file = self.base_path / 'metadata' / f"{symbol}.json"
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Add timestamp
            metadata['last_updated'] = datetime.now().isoformat()
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.debug(f"Metadata saved for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error saving metadata for {symbol}: {e}")
    
    def load_metadata(self, symbol: str) -> Optional[Dict]:
        """Load metadata for symbol"""
        try:
            metadata_file = self.base_path / 'metadata' / f"{symbol}.json"
            
            if not metadata_file.exists():
                return None
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error loading metadata for {symbol}: {e}")
            return None
    
    def backup_data(self, symbol: str, backup_suffix: str = None) -> bool:
        """Create backup of data"""
        try:
            if backup_suffix is None:
                backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Copy all data types and formats
            for data_type in ['raw', 'processed']:
                for format in ['csv', 'parquet', 'pickle']:
                    source_path = self._get_file_path(symbol, data_type, format)
                    if source_path.exists():
                        backup_path = source_path.with_suffix(f".{backup_suffix}.{format}")
                        import shutil
                        shutil.copy2(source_path, backup_path)
            
            # Backup metadata
            metadata_source = self.base_path / 'metadata' / f"{symbol}.json"
            if metadata_source.exists():
                metadata_backup = metadata_source.with_suffix(f".{backup_suffix}.json")
                import shutil
                shutil.copy2(metadata_source, metadata_backup)
            
            self.logger.info(f"Backup created for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating backup for {symbol}: {e}")
            return False
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        stats = {
            'total_symbols': 0,
            'total_size_bytes': 0,
            'data_types': {},
            'formats': {}
        }
        
        for data_type in ['raw', 'processed']:
            data_path = self.base_path / data_type
            if data_path.exists():
                type_stats = {
                    'symbol_count': 0,
                    'total_size': 0,
                    'formats': {}
                }
                
                for file_path in data_path.iterdir():
                    if file_path.is_file():
                        type_stats['symbol_count'] += 1
                        type_stats['total_size'] += file_path.stat().st_size
                        
                        # Count by format
                        format = file_path.suffix[1:]  # Remove dot
                        type_stats['formats'][format] = type_stats['formats'].get(format, 0) + 1
                
                stats['data_types'][data_type] = type_stats
                stats['total_symbols'] += type_stats['symbol_count']
                stats['total_size_bytes'] += type_stats['total_size']
        
        return stats
