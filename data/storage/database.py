# trading_system/data/storage/database.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import sqlite3
from datetime import datetime
import logging
import json
from pathlib import Path

from ...core.exceptions import DataSourceError

class DatabaseStorage:
    """
    Database storage for trading system data
    Uses SQLite for simplicity, can be extended to other databases
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database configuration
        self.db_path = Path(config.get('database_path', 'data/trading_system.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database tables"""
        try:
            with self._get_connection() as conn:
                # Price data table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS price_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TIMESTAMP NOT NULL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL,
                        adj_close REAL,
                        source TEXT,
                        exchange TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date)
                    )
                ''')
                
                # Dividend data table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS dividend_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TIMESTAMP NOT NULL,
                        dividend REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date)
                    )
                ''')
                
                # Split data table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS split_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TIMESTAMP NOT NULL,
                        split REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date)
                    )
                ''')
                
                # Metadata table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS symbol_metadata (
                        symbol TEXT PRIMARY KEY,
                        name TEXT,
                        type TEXT,
                        exchange TEXT,
                        sector TEXT,
                        currency TEXT DEFAULT 'INR',
                        timezone TEXT DEFAULT 'Asia/Kolkata',
                        metadata_json TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_price_symbol_date ON price_data(symbol, date)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_dividend_symbol_date ON dividend_data(symbol, date)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_split_symbol_date ON split_data(symbol, date)')
                
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            raise DataSourceError(f"Database initialization failed: {e}")
    
    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path, timeout=30)
    
    def save_price_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Save price data to database
        
        Args:
            symbol: Instrument symbol
            data: DataFrame with price data
            
        Returns:
            True if successful
        """
        try:
            with self._get_connection() as conn:
                # Prepare data for insertion
                records = []
                for _, row in data.iterrows():
                    record = {
                        'symbol': symbol,
                        'date': pd.to_datetime(row.get('date')).strftime('%Y-%m-%d %H:%M:%S'),
                        'open': row.get('open'),
                        'high': row.get('high'),
                        'low': row.get('low'),
                        'close': row.get('close'),
                        'volume': row.get('volume'),
                        'adj_close': row.get('adj_close', row.get('close')),
                        'source': row.get('source', 'unknown'),
                        'exchange': row.get('exchange', 'NSE')
                    }
                    records.append(record)
                
                # Insert or replace data
                for record in records:
                    conn.execute('''
                        INSERT OR REPLACE INTO price_data 
                        (symbol, date, open, high, low, close, volume, adj_close, source, exchange)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        record['symbol'], record['date'], record['open'], record['high'],
                        record['low'], record['close'], record['volume'], record['adj_close'],
                        record['source'], record['exchange']
                    ))
                
                conn.commit()
                self.logger.info(f"Price data saved for {symbol} ({len(records)} records)")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving price data for {symbol}: {e}")
            return False
    
    def load_price_data(
        self, 
        symbol: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load price data from database
        
        Args:
            symbol: Instrument symbol
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            DataFrame with price data
        """
        try:
            with self._get_connection() as conn:
                query = '''
                    SELECT date, open, high, low, close, volume, adj_close, source, exchange
                    FROM price_data 
                    WHERE symbol = ?
                '''
                params = [symbol]
                
                if start_date:
                    query += ' AND date >= ?'
                    params.append(start_date)
                
                if end_date:
                    query += ' AND date <= ?'
                    params.append(end_date)
                
                query += ' ORDER BY date'
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    return None
                
                # Convert date column
                df['date'] = pd.to_datetime(df['date'])
                
                self.logger.debug(f"Price data loaded for {symbol} ({len(df)} records)")
                return df
                
        except Exception as e:
            self.logger.error(f"Error loading price data for {symbol}: {e}")
            return None
    
    def save_dividend_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """Save dividend data to database"""
        try:
            with self._get_connection() as conn:
                for _, row in data.iterrows():
                    conn.execute('''
                        INSERT OR REPLACE INTO dividend_data (symbol, date, dividend)
                        VALUES (?, ?, ?)
                    ''', (symbol, row['date'], row['dividend']))
                
                conn.commit()
                self.logger.info(f"Dividend data saved for {symbol}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving dividend data for {symbol}: {e}")
            return False
    
    def save_split_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """Save split data to database"""
        try:
            with self._get_connection() as conn:
                for _, row in data.iterrows():
                    conn.execute('''
                        INSERT OR REPLACE INTO split_data (symbol, date, split)
                        VALUES (?, ?, ?)
                    ''', (symbol, row['date'], row['split']))
                
                conn.commit()
                self.logger.info(f"Split data saved for {symbol}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving split data for {symbol}: {e}")
            return False
    
    def save_symbol_metadata(self, symbol: str, metadata: Dict):
        """Save symbol metadata to database"""
        try:
            with self._get_connection() as conn:
                metadata_json = json.dumps(metadata)
                
                conn.execute('''
                    INSERT OR REPLACE INTO symbol_metadata 
                    (symbol, name, type, exchange, sector, currency, timezone, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    metadata.get('name', ''),
                    metadata.get('type', 'STOCK'),
                    metadata.get('exchange', 'NSE'),
                    metadata.get('sector', ''),
                    metadata.get('currency', 'INR'),
                    metadata.get('timezone', 'Asia/Kolkata'),
                    metadata_json
                ))
                
                conn.commit()
                self.logger.debug(f"Metadata saved for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error saving metadata for {symbol}: {e}")
    
    def load_symbol_metadata(self, symbol: str) -> Optional[Dict]:
        """Load symbol metadata from database"""
        try:
            with self._get_connection() as conn:
                result = conn.execute(
                    'SELECT * FROM symbol_metadata WHERE symbol = ?', 
                    (symbol,)
                ).fetchone()
                
                if not result:
                    return None
                
                # Convert to dictionary
                metadata = {
                    'symbol': result[0],
                    'name': result[1],
                    'type': result[2],
                    'exchange': result[3],
                    'sector': result[4],
                    'currency': result[5],
                    'timezone': result[6],
                    'last_updated': result[8],
                    'created_at': result[9]
                }
                
                # Parse JSON metadata
                if result[7]:
                    metadata.update(json.loads(result[7]))
                
                return metadata
                
        except Exception as e:
            self.logger.error(f"Error loading metadata for {symbol}: {e}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with data in database"""
        try:
            with self._get_connection() as conn:
                result = conn.execute(
                    'SELECT DISTINCT symbol FROM price_data ORDER BY symbol'
                ).fetchall()
                
                return [row[0] for row in result]
                
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
    
    def get_data_stats(self, symbol: str) -> Dict:
        """Get statistics for symbol data"""
        try:
            with self._get_connection() as conn:
                # Price data stats
                price_stats = conn.execute('''
                    SELECT 
                        COUNT(*) as record_count,
                        MIN(date) as start_date,
                        MAX(date) as end_date,
                        COUNT(DISTINCT source) as source_count
                    FROM price_data 
                    WHERE symbol = ?
                ''', (symbol,)).fetchone()
                
                # Dividend stats
                dividend_stats = conn.execute('''
                    SELECT COUNT(*) FROM dividend_data WHERE symbol = ?
                ''', (symbol,)).fetchone()
                
                # Split stats
                split_stats = conn.execute('''
                    SELECT COUNT(*) FROM split_data WHERE symbol = ?
                ''', (symbol,)).fetchone()
                
                return {
                    'symbol': symbol,
                    'price_records': price_stats[0] if price_stats else 0,
                    'start_date': price_stats[1] if price_stats else None,
                    'end_date': price_stats[2] if price_stats else None,
                    'sources': price_stats[3] if price_stats else 0,
                    'dividend_records': dividend_stats[0] if dividend_stats else 0,
                    'split_records': split_stats[0] if split_stats else 0
                }
                
        except Exception as e:
            self.logger.error(f"Error getting stats for {symbol}: {e}")
            return {}
    
    def get_database_stats(self) -> Dict:
        """Get overall database statistics"""
        try:
            with self._get_connection() as conn:
                stats = {}
                
                # Table counts
                tables = ['price_data', 'dividend_data', 'split_data', 'symbol_metadata']
                for table in tables:
                    count = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                    stats[f'{table}_count'] = count
                
                # Unique symbols
                unique_symbols = conn.execute(
                    'SELECT COUNT(DISTINCT symbol) FROM price_data'
                ).fetchone()[0]
                stats['unique_symbols'] = unique_symbols
                
                # Date range
                date_range = conn.execute(
                    'SELECT MIN(date), MAX(date) FROM price_data'
                ).fetchone()
                stats['overall_start_date'] = date_range[0]
                stats['overall_end_date'] = date_range[1]
                
                # Database size
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                stats['database_size_bytes'] = db_size
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}
