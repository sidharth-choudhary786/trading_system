# trading_system/production/broker_apis/alpaca.py
import alpaca_trade_api as tradeapi
from typing import Dict, List, Optional
import logging
from datetime import datetime

from ...core.exceptions import ExecutionError
from ...core.types import Order, OrderType, OrderSide

class AlpacaAPI:
    """
    Alpaca Broker API Implementation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Alpaca configuration
        self.api_key = config.get('alpaca_api_key')
        self.api_secret = config.get('alpaca_api_secret')
        self.base_url = config.get('alpaca_base_url', 'https://paper-api.alpaca.markets')
        
        if not self.api_key or not self.api_secret:
            raise ExecutionError("Alpaca API credentials not provided")
        
        # Initialize Alpaca client
        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret,
            self.base_url,
            api_version='v2'
        )
        
        self.logger.info("Alpaca API initialized successfully")
    
    def place_order(self, order_data: Dict) -> Dict:
        """Place order through Alpaca"""
        try:
            # Convert to Alpaca order format
            alpaca_order = self._convert_to_alpaca_order(order_data)
            
            # Submit order
            order = self.api.submit_order(**alpaca_order)
            
            return {
                'status': 'success',
                'order_id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'status': order.status
            }
            
        except Exception as e:
            self.logger.error(f"Alpaca order placement failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get order status from Alpaca"""
        try:
            order = self.api.get_order(order_id)
            
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'status': order.status,
                'side': order.side,
                'order_type': order.type,
                'quantity': float(order.qty),
                'filled_quantity': float(order.filled_qty),
                'price': float(order.limit_price) if order.limit_price else None,
                'average_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'timestamp': order.submitted_at.isoformat() if order.submitted_at else None
            }
            
        except Exception as e:
            self.logger.error(f"Alpaca order status check failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel order through Alpaca"""
        try:
            self.api.cancel_order(order_id)
            
            return {
                'status': 'success',
                'message': f'Order {order_id} cancelled'
            }
            
        except Exception as e:
            self.logger.error(f"Alpaca order cancellation failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_account_info(self) -> Dict:
        """Get account information from Alpaca"""
        try:
            account = self.api.get_account()
            
            return {
                'account_number': account.id,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'day_trading_buying_power': float(account.day_trading_buying_power),
                'regt_buying_power': float(account.regt_buying_power),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'status': account.status,
                'currency': account.currency,
                'pattern_day_trader': account.pattern_day_trader
            }
            
        except Exception as e:
            self.logger.error(f"Alpaca account info fetch failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_positions(self) -> Dict:
        """Get current positions from Alpaca"""
        try:
            positions = self.api.list_positions()
            
            position_data = {}
            for position in positions:
                position_data[position.symbol] = {
                    'symbol': position.symbol,
                    'quantity': float(position.qty),
                    'average_price': float(position.avg_entry_price),
                    'market_value': float(position.market_value),
                    'cost_basis': float(position.cost_basis),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'current_price': float(position.current_price),
                    'side': position.side
                }
            
            return {
                'status': 'success',
                'positions': position_data
            }
            
        except Exception as e:
            self.logger.error(f"Alpaca positions fetch failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get market data from Alpaca"""
        try:
            # Get latest trade
            trade = self.api.get_latest_trade(symbol)
            
            # Get latest quote
            quote = self.api.get_latest_quote(symbol)
            
            return {
                'symbol': symbol,
                'last_price': float(trade.price) if trade else None,
                'timestamp': trade.timestamp.isoformat() if trade and trade.timestamp else None,
                'bid_price': float(quote.bidprice) if quote else None,
                'ask_price': float(quote.askprice) if quote else None,
                'bid_size': float(quote.bidsize) if quote else None,
                'ask_size': float(quote.asksize) if quote else None
            }
            
        except Exception as e:
            self.logger.error(f"Alpaca market data fetch failed for {symbol}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _convert_to_alpaca_order(self, order_data: Dict) -> Dict:
        """Convert internal order format to Alpaca format"""
        alpaca_order = {
            'symbol': order_data['symbol'],
            'qty': order_data['quantity'],
            'side': order_data['side'].lower(),
            'type': order_data['order_type'].lower(),
            'time_in_force': 'day'
        }
        
        if order_data['order_type'] in ['LIMIT', 'STOP_LIMIT']:
            alpaca_order['limit_price'] = str(order_data['price'])
        
        if order_data['order_type'] in ['STOP', 'STOP_LIMIT']:
            alpaca_order['stop_price'] = str(order_data.get('trigger_price', order_data['price']))
        
        return alpaca_order
    
    def supports_immediate_execution(self) -> bool:
        """Check if broker supports immediate execution status"""
        return True
    
    def close_connection(self):
        """Close API connection"""
        # Alpaca REST API doesn't require explicit connection closing
        pass
