# trading_system/data/mapping/ticker_resolver.py
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

from ...core.exceptions import DataSourceError

class TickerResolver:
    """
    Resolve tickers across different data sources and handle fallback logic
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Source preference order
        self.source_preference = config.get('data_sources', ['twelvedata', 'yfinance', 'investpy', 'alphavantage'])
        
        # Exchange preference
        self.exchange_preference = config.get('exchange_preference', ['NSE', 'BSE'])
    
    def resolve_ticker(
        self, 
        symbol: str, 
        source: str, 
        exchange: Optional[str] = None
    ) -> Optional[str]:
        """
        Resolve symbol to source-specific ticker
        
        Args:
            symbol: Base symbol
            source: Data source name
            exchange: Preferred exchange
            
        Returns:
            Source-specific ticker or None if not found
        """
        # This would typically use the MasterMapping, but for now we'll use simple mapping
        ticker_map = self._get_ticker_mapping(symbol, source, exchange)
        return ticker_map.get('ticker')
    
    def resolve_ticker_with_fallback(
        self, 
        symbol: str, 
        preferred_source: str = None,
        preferred_exchange: str = None
    ) -> Dict:
        """
        Resolve ticker with fallback logic
        
        Args:
            symbol: Base symbol
            preferred_source: Preferred data source
            preferred_exchange: Preferred exchange
            
        Returns:
            Dictionary with resolved ticker information
        """
        if preferred_source is None:
            preferred_source = self.source_preference[0]
        
        if preferred_exchange is None:
            preferred_exchange = self.exchange_preference[0]
        
        # Try preferred source and exchange first
        result = self._try_resolve(symbol, preferred_source, preferred_exchange)
        if result['success']:
            return result
        
        # Try other exchanges for preferred source
        for exchange in self.exchange_preference:
            if exchange != preferred_exchange:
                result = self._try_resolve(symbol, preferred_source, exchange)
                if result['success']:
                    return result
        
        # Try other sources
        for source in self.source_preference:
            if source != preferred_source:
                for exchange in self.exchange_preference:
                    result = self._try_resolve(symbol, source, exchange)
                    if result['success']:
                        return result
        
        # No resolution found
        return {
            'success': False,
            'symbol': symbol,
            'ticker': None,
            'source': None,
            'exchange': None,
            'message': f"Could not resolve ticker for {symbol} from any source"
        }
    
    def _try_resolve(self, symbol: str, source: str, exchange: str) -> Dict:
        """Try to resolve ticker for specific source and exchange"""
        try:
            ticker = self.resolve_ticker(symbol, source, exchange)
            if ticker:
                return {
                    'success': True,
                    'symbol': symbol,
                    'ticker': ticker,
                    'source': source,
                    'exchange': exchange,
                    'message': f"Resolved {symbol} to {ticker} on {source} ({exchange})"
                }
            else:
                return {
                    'success': False,
                    'symbol': symbol,
                    'ticker': None,
                    'source': source,
                    'exchange': exchange,
                    'message': f"No mapping found for {symbol} on {source} ({exchange})"
                }
        except Exception as e:
            return {
                'success': False,
                'symbol': symbol,
                'ticker': None,
                'source': source,
                'exchange': exchange,
                'message': f"Error resolving {symbol} on {source} ({exchange}): {e}"
            }
    
    def _get_ticker_mapping(self, symbol: str, source: str, exchange: str) -> Dict:
        """Get ticker mapping for symbol, source, and exchange"""
        # Simple mapping logic - in real implementation, this would use MasterMapping
        mappings = {
            'twelvedata': {
                'NSE': f"{symbol}.NS",
                'BSE': f"{symbol}.BO"
            },
            'yfinance': {
                'NSE': f"{symbol}.NS", 
                'BSE': f"{symbol}.BO"
            },
            'investpy': {
                'NSE': symbol,
                'BSE': symbol
            },
            'alphavantage': {
                'NSE': f"{symbol}.BSE",  # Alpha Vantage uses .BSE for NSE
                'BSE': symbol
            }
        }
        
        source_map = mappings.get(source, {})
        ticker = source_map.get(exchange)
        
        return {
            'ticker': ticker,
            'symbol': symbol,
            'source': source,
            'exchange': exchange
        }
    
    def get_available_sources(self, symbol: str) -> List[Dict]:
        """Get available data sources for symbol"""
        available_sources = []
        
        for source in self.source_preference:
            for exchange in self.exchange_preference:
                ticker = self.resolve_ticker(symbol, source, exchange)
                if ticker:
                    available_sources.append({
                        'source': source,
                        'exchange': exchange,
                        'ticker': ticker,
                        'preference_score': self._calculate_preference_score(source, exchange)
                    })
        
        # Sort by preference score
        available_sources.sort(key=lambda x: x['preference_score'], reverse=True)
        return available_sources
    
    def _calculate_preference_score(self, source: str, exchange: str) -> float:
        """Calculate preference score for source-exchange combination"""
        source_score = self.source_preference.index(source) if source in self.source_preference else len(self.source_preference)
        exchange_score = self.exchange_preference.index(exchange) if exchange in self.exchange_preference else len(self.exchange_preference)
        
        # Lower index = higher preference
        source_pref = 1.0 - (source_score / len(self.source_preference))
        exchange_pref = 1.0 - (exchange_score / len(self.exchange_preference))
        
        return (source_pref * 0.6) + (exchange_pref * 0.4)  # Weight source more heavily
    
    def validate_ticker(self, ticker: str, source: str) -> bool:
        """Validate if ticker format is correct for source"""
        validation_rules = {
            'twelvedata': lambda t: t.endswith(('.NS', '.BO')),
            'yfinance': lambda t: t.endswith(('.NS', '.BO')),
            'investpy': lambda t: isinstance(t, str) and len(t) > 0,
            'alphavantage': lambda t: isinstance(t, str) and len(t) > 0
        }
        
        validator = validation_rules.get(source)
        if validator:
            return validator(ticker)
        
        return False
    
    def get_source_config(self, source: str) -> Dict:
        """Get configuration for specific data source"""
        source_configs = {
            'twelvedata': {
                'name': 'Twelve Data',
                'api_docs': 'https://twelvedata.com/docs',
                'rate_limit': 8,
                'supported_exchanges': ['NSE', 'BSE'],
                'notes': 'Provides real-time and historical data'
            },
            'yfinance': {
                'name': 'Yahoo Finance',
                'api_docs': 'https://pypi.org/project/yfinance/',
                'rate_limit': 2000,
                'supported_exchanges': ['NSE', 'BSE'],
                'notes': 'Free, good for historical data'
            },
            'investpy': {
                'name': 'Investing.com',
                'api_docs': 'https://pypi.org/project/investpy/',
                'rate_limit': 1,
                'supported_exchanges': ['NSE', 'BSE'],
                'notes': 'Free alternative, may have rate limits'
            },
            'alphavantage': {
                'name': 'Alpha Vantage',
                'api_docs': 'https://www.alphavantage.co/documentation/',
                'rate_limit': 5,
                'supported_exchanges': ['NSE', 'BSE'],
                'notes': 'Free tier available, good for fundamental data'
            }
        }
        
        return source_configs.get(source, {})
