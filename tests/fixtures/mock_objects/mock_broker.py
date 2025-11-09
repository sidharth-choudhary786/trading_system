# trading_system/tests/fixtures/mock_objects/mock_broker.py
class MockBroker:
    """Mock broker for testing"""
    
    def __init__(self):
        self.orders = {}
        self.positions = {}
        self.account_balance = 100000
        self.order_id_counter = 1
    
    def place_order(self, order_data):
        """Place a new order"""
        order_id = f"ORDER_{self.order_id_counter}"
        self.order_id_counter += 1
        
        self.orders[order_id] = {
            **order_data,
            'order_id': order_id,
            'status': 'PENDING',
            'timestamp': '2023-01-01 10:00:00'
        }
        
        # Simulate immediate execution for market orders
        if order_data.get('order_type') == 'MARKET':
            self.orders[order_id]['status'] = 'FILLED'
            self.orders[order_id]['filled_quantity'] = order_data['quantity']
            self.orders[order_id]['average_price'] = 100.0  # Mock price
        
        return {'status': 'success', 'order_id': order_id}
    
    def get_order_status(self, order_id):
        """Get order status"""
        return self.orders.get(order_id, {'status': 'UNKNOWN'})
    
    def cancel_order(self, order_id):
        """Cancel an order"""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'CANCELLED'
            return {'status': 'success'}
        return {'status': 'error', 'message': 'Order not found'}
    
    def get_account_info(self):
        """Get account information"""
        return {
            'balance': self.account_balance,
            'available_cash': self.account_balance,
            'positions': self.positions
        }
