# trading_system/production/broker_apis/zerodha.py
from typing import Dict, List, Optional
import logging
from datetime import datetime
import requests
import json

from ...core.exceptions import ExecutionError

class ZerodhaAPI:
    """
    Zerodha Kite Connect API Implementation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Zerodha configuration
        self.api_key = config.get('zerodha_api_key')
        self.access_token = config.get('zerodha_access_token')
        self.root = config.get('zerodha_root', 'https://api.kite.trade')
        
        if not self.api_key or not self.access_token:
            raise ExecutionError("Zerodha API credentials not provided")
        
        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({
            'X-Kite-Version': '3',
            'Authorization': f'token {self.api_key}:{self.access_token}'
        })
        
        self.logger.info("Zerodha API initialized successfully")
    
    def place_order(self, order_data: Dict) -> Dict:
        """Place order through Zerodha"""
        try:
            # Convert to Zerodha order format
            kite_order = self._convert_to_kite_order(order_data)
            
            # Place order
            response = self.session.post(
                f"{self.root}/orders/{kite_order['variety']}",
                data=kite_order
            )
            
            if response.status_code == 200:
                order_result = response.json()
                return {
                    'status': 'success',
                    'order_id': order_result['data']['order_id'],
                    'symbol': order_data['symbol'],
                    'status': 'ORDER PLACED'
                }
            else:
                error_data = response.json()
                raise ExecutionError(f"Zerodha order failed: {error_data.get('message', 'Unknown error')}")
                
        except Exception as e:
            self.logger.error(f"Zerodha order placement failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get order status from Zerodha"""
        try:
            response = self.session.get(f"{self.root}/orders/{order_id}")
            
            if response.status_code == 200:
                order_data = response.json()['data']
                
                return {
                    'order_id': order_data['order_id'],
                    'symbol': order_data['tradingsymbol'],
                    'status': order_data['status'],
                    'side': 'BUY' if order_data['transaction_type'] == 'BUY' else 'SELL',
                    'order_type': order_data['order_type'],
                    'quantity': order_data['quantity'],
                    'filled_quantity': order_data['filled_quantity'],
                    'price': float(order_data['price']) if order_data['price'] else None,
                    'average_price': float(order_data['average_price']) if order_data['average_price'] else None,
                    'timestamp': order_data['order_timestamp']
                }
            else:
                raise ExecutionError(f"Failed to fetch order status: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Zerodha order status check failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel order through Zerodha"""
        try:
            # For Zerodha, we need variety (regular, amo, etc.)
            # We'll assume regular variety for simplicity
            response = self.session.delete(
                f"{self.root}/orders/regular/{order_id}"
            )
            
            if response.status_code == 200:
                return {
                    'status': 'success',
                    'message': f'Order {order_id} cancelled successfully'
                }
            else:
                error_data = response.json()
                raise ExecutionError(f"Zerodha order cancellation failed: {error_data.get('message', 'Unknown error')}")
                
        except Exception as e:
            self.logger.error(f"Zerodha order cancellation failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_account_info(self) -> Dict:
        """Get account information from Zerodha"""
        try:
            response = self.session.get(f"{self.root}/user/margins")
            
            if response.status_code == 200:
                margins_data = response.json()['data']
                equity_margins = margins_data['equity']
                
                return {
                    'available_cash': float(equity_margins['available']['cash']),
                    'available_margin': float(equity_margins['available']['margin']),
                    'used_margin': float(equity_margins['used']['debits']),
                    'net_balance': float(equity_margins['available']['cash']) + float(equity_margins['available']['margin']),
                    'currency': 'INR'
                }
            else:
                raise ExecutionError(f"Failed to fetch account info: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Zerodha account info fetch failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_positions(self) -> Dict:
        """Get current positions from Zerodha"""
        try:
            response = self.session.get(f"{self.root}/portfolio/positions")
            
            if response.status_code == 200:
                positions_data = response.json()['data']
                
                position_dict = {}
                for position in positions_data:
                    if float(position['quantity']) != 0:
                        position_dict[position['tradingsymbol']] = {
                            'symbol': position['tradingsymbol'],
                            'quantity': float(position['quantity']),
                            'average_price': float(position['average_price']),
                            'market_value': float(position['value']),
                            'unrealized_pnl': float(position['unrealised']),
                            'realized_pnl': float(position['realised']),
                            'day_change': float(position['day_change']),
                            'day_change_percentage': float(position['day_change_percentage'])
                        }
                
                return {
                    'status': 'success',
                    'positions': position_dict
                }
            else:
                raise ExecutionError(f"Failed to fetch positions: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Zerodha positions fetch failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get market data from Zerodha"""
        try:
            # Get quote data
            response = self.session.get(f"{self.root}/quote", params={'i': symbol})
            
            if response.status_code == 200:
                quote_data = response.json()['data'][symbol]
                
                return {
                    'symbol': symbol,
                    'last_price': float(quote_data['last_price']),
                    'timestamp': datetime.now().isoformat(),
                    'bid_price': float(quote_data['depth']['buy'][0]['price']),
                    'ask_price': float(quote_data['depth']['sell'][0]['price']),
                    'bid_size': float(quote_data['depth']['buy'][0]['quantity']),
                    'ask_size': float(quote_data['depth']['sell'][0]['quantity']),
                    'open': float(quote_data['ohlc']['open']),
                    'high': float(quote_data['ohlc']['high']),
                    'low': float(quote_data['ohlc']['low']),
                    'close': float(quote_data['ohlc']['close']),
                    'volume': float(quote_data['volume'])
                }
            else:
                raise ExecutionError(f"Failed to fetch market data: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Zerodha market data fetch failed for {symbol}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _convert_to_kite_order(self, order_data: Dict) -> Dict:
        """Convert internal order format to Zerodha Kite format"""
        kite_order = {
            'tradingsymbol': order_data['symbol'],
            'quantity': order_data['quantity'],
            'transaction_type': order_data['side'],
            'order_type': order_data['order_type'],
            'product': order_data.get('product', 'MIS'),
            'variety': 'regular'
        }
        
        if order_data['order_type'] in ['LIMIT', 'SL', 'SL-M']:
            kite_order['price'] = order_data['price']
        
        if order_data['order_type'] in ['SL', 'SL-M']:
            kite_order['trigger_price'] = order_data.get('trigger_price', order_data['price'])
        
        return kite_order
    
    def supports_immediate_execution(self) -> bool:
        """Check if broker supports immediate execution status"""
        return True
    
    def close_connection(self):
        """Close Zerodha session"""
        self.session.close()
        self.logger.info("Zerodha session closed")
