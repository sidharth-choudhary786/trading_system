# trading_system/tests/fixtures/mock_objects/mock_broker.py
class MockBroker:
    """Mock broker for testing"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.orders = {}
        self.positions = {}
        self.account_value = 1000000
        self.commission = 0.001
    
    def place_order(self, order):
        """Mock order placement"""
        order_id = f"ORDER_{len(self.orders) + 1}"
        self.orders[order_id] = {
            'order_id': order_id,
            'symbol': order.get('symbol'),
            'quantity': order.get('quantity'),
            'side': order.get('side'),
            'status': 'FILLED',
            'fill_price': order.get('price', 100),
            'timestamp': '2023-01-01 10:00:00'
        }
        return {'status': 'success', 'order_id': order_id}
    
    def get_order_status(self, order_id):
        """Mock order status"""
        return self.orders.get(order_id, {'status': 'UNKNOWN'})
    
    def cancel_order(self, order_id):
        """Mock order cancellation"""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'CANCELLED'
            return {'status': 'CANCELLED'}
        return {'status': 'NOT_FOUND'}
    
    def get_account_info(self):
        """Mock account info"""
        return {
            'account_value': self.account_value,
            'cash': 100000,
            'positions': self.positions
        }
    
    def get_positions(self):
        """Mock positions"""
        return self.positions
