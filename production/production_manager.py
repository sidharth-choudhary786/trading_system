# trading_system/production/production_manager.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import time
import threading
from abc import ABC, abstractmethod

from ..core.base_system import BaseTradingSystem
from ..core.types import Order, Trade, OrderSide, OrderType
from ..core.exceptions import TradingSystemError
from ..execution.live_execution import LiveExecution
from ..data.data_manager import DataManager
from ..models.model_manager import ModelManager
from ..portfolio.portfolio import Portfolio
from ..risk.risk_engine import RiskEngine
from .monitoring.health_monitor import HealthMonitor
from .monitoring.alert_manager import AlertManager

class ProductionManager(BaseTradingSystem):
    """
    Production trading manager - handles live trading operations
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Production-specific configuration
        self.broker = config.get('broker', 'zerodha')
        self.paper_trading = config.get('paper_trading', True)
        self.trading_hours = config.get('trading_hours', {})
        self.max_position_size = config.get('max_position_size', 0.1)
        
        # Initialize components
        self._initialize_production_components()
        
        # Trading state
        self.is_market_open = False
        self.last_signal_time = None
        self.heartbeat_interval = 60  # seconds
        
        # Threading
        self.trading_thread = None
        self.stop_event = threading.Event()
        
        self.logger.info(f"Production Manager initialized for {self.broker}")
    
    def _initialize_production_components(self):
        """Initialize production-specific components"""
        # Live execution handler
        execution_config = self.config.get('execution', {})
        self.execution_handler = LiveExecution(execution_config)
        
        # Data manager for real-time data
        data_config = self.config.get('data', {})
        self.data_manager = DataManager(data_config)
        
        # Model manager for live predictions
        model_config = self.config.get('models', {})
        self.model_manager = ModelManager(model_config)
        
        # Portfolio tracking
        self.portfolio = Portfolio(self.initial_capital)
        
        # Risk engine
        risk_config = self.config.get('risk', {})
        self.risk_engine = RiskEngine(risk_config)
        
        # Monitoring
        self.health_monitor = HealthMonitor(self.config)
        self.alert_manager = AlertManager(self.config)
    
    def initialize(self):
        """Initialize production system"""
        try:
            self.logger.info("Initializing Production Trading System...")
            
            # Connect to broker
            self._connect_to_broker()
            
            # Initialize data feeds
            self._initialize_data_feeds()
            
            # Load models
            self._load_trading_models()
            
            # Start monitoring
            self.health_monitor.start()
            self.alert_manager.start()
            
            # Sync with broker
            self._sync_with_broker()
            
            self.is_running = True
            self.logger.info("Production Trading System initialized successfully")
            
            # Send startup alert
            self.alert_manager.send_alert(
                "SYSTEM_STARTUP",
                "Trading system started successfully",
                "INFO"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize production system: {e}")
            self.alert_manager.send_alert(
                "SYSTEM_ERROR",
                f"Initialization failed: {e}",
                "ERROR"
            )
            raise TradingSystemError(f"Production initialization failed: {e}")
    
    def _connect_to_broker(self):
        """Connect to broker API"""
        try:
            # Test broker connection
            account_info = self.execution_handler.get_account_info()
            self.logger.info(f"Connected to {self.broker}. Account: {account_info.get('account_id', 'Unknown')}")
            
        except Exception as e:
            raise TradingSystemError(f"Broker connection failed: {e}")
    
    def _initialize_data_feeds(self):
        """Initialize real-time data feeds"""
        try:
            # Subscribe to real-time data for trading universe
            symbols = self.config.get('trading_universe', [])
            self.logger.info(f"Subscribing to real-time data for {len(symbols)} symbols")
            
            # Initialize real-time data handlers
            # This would typically involve WebSocket connections or similar
            
        except Exception as e:
            raise TradingSystemError(f"Data feed initialization failed: {e}")
    
    def _load_trading_models(self):
        """Load trading models for live predictions"""
        try:
            self.model_manager.load_models()
            self.logger.info("Trading models loaded successfully")
            
        except Exception as e:
            raise TradingSystemError(f"Model loading failed: {e}")
    
    def _sync_with_broker(self):
        """Sync with broker state"""
        try:
            # Sync positions
            broker_positions = self.execution_handler.get_positions()
            self.portfolio.sync_positions(broker_positions)
            
            # Sync orders
            self.execution_handler.sync_orders()
            
            self.logger.info("Synchronized with broker state")
            
        except Exception as e:
            self.logger.error(f"Broker sync failed: {e}")
    
    def run(self):
        """Start production trading"""
        if not self.is_running:
            raise TradingSystemError("System not initialized")
        
        self.logger.info("Starting production trading...")
        
        # Start trading thread
        self.stop_event.clear()
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        # Start health monitoring thread
        health_thread = threading.Thread(target=self._health_monitoring_loop)
        health_thread.daemon = True
        health_thread.start()
        
        self.logger.info("Production trading started")
    
    def _trading_loop(self):
        """Main trading loop"""
        while not self.stop_event.is_set() and self.is_running:
            try:
                # Check if market is open
                if not self._is_market_open():
                    if self.is_market_open:
                        self._on_market_close()
                    self.is_market_open = False
                    time.sleep(60)  # Check every minute when market closed
                    continue
                
                # Market just opened
                if not self.is_market_open:
                    self._on_market_open()
                    self.is_market_open = True
                
                # Execute trading cycle
                self._trading_cycle()
                
                # Sleep until next cycle
                time.sleep(self._get_cycle_interval())
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                self.alert_manager.send_alert(
                    "TRADING_ERROR",
                    f"Trading loop error: {e}",
                    "ERROR"
                )
                time.sleep(30)  # Wait before retrying
    
    def _trading_cycle(self):
        """Single trading cycle"""
        try:
            # 1. Update market data
            market_data = self._update_market_data()
            
            # 2. Generate signals
            signals = self._generate_signals(market_data)
            
            # 3. Check risk limits
            if not self.risk_engine.check_limits(self.portfolio, signals):
                self.logger.warning("Risk limits exceeded, skipping trading cycle")
                return
            
            # 4. Generate orders
            orders = self._generate_orders(signals, market_data)
            
            # 5. Execute orders
            if orders:
                trades = self.execution_handler.execute_orders(orders, market_data, datetime.now())
                self._process_executed_trades(trades)
            
            # 6. Update portfolio
            self.portfolio.update_portfolio(datetime.now(), market_data)
            
            # 7. Log cycle completion
            self._log_trading_cycle(signals, orders)
            
        except Exception as e:
            self.logger.error(f"Trading cycle error: {e}")
            raise
    
    def _update_market_data(self) -> Dict[str, float]:
        """Update real-time market data"""
        # This would fetch real-time prices from data feeds
        # For now, return mock data
        symbols = self.config.get('trading_universe', [])
        market_data = {}
        
        for symbol in symbols:
            # In real implementation, this would come from real-time data feed
            market_data[symbol] = np.random.uniform(100, 5000)
        
        return market_data
    
    def _generate_signals(self, market_data: Dict) -> Dict[str, str]:
        """Generate trading signals"""
        try:
            # Prepare data for model prediction
            features = self._prepare_live_features(market_data)
            
            # Get model predictions
            predictions = self.model_manager.predict(features)
            
            # Convert predictions to signals
            signals = {}
            for symbol, prediction in predictions.items():
                if prediction > 0.6:  # Buy threshold
                    signals[symbol] = 'BUY'
                elif prediction < 0.4:  # Sell threshold
                    signals[symbol] = 'SELL'
                else:
                    signals[symbol] = 'HOLD'
            
            self.last_signal_time = datetime.now()
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return {}
    
    def _prepare_live_features(self, market_data: Dict) -> pd.DataFrame:
        """Prepare features for live prediction"""
        # This would use the feature engineering pipeline
        # For now, create mock features
        features_data = []
        
        for symbol, price in market_data.items():
            features = {
                'symbol': symbol,
                'price': price,
                'volume': np.random.uniform(1000, 100000),
                'timestamp': datetime.now()
            }
            features_data.append(features)
        
        return pd.DataFrame(features_data)
    
    def _generate_orders(self, signals: Dict, market_data: Dict) -> List[Order]:
        """Generate orders from signals"""
        orders = []
        
        for symbol, signal in signals.items():
            if signal == 'HOLD':
                continue
            
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol]
            
            # Calculate position size
            position_size = self._calculate_position_size(symbol, current_price, signal)
            if position_size <= 0:
                continue
            
            # Create order
            order = Order(
                order_id=f"LIVE_{symbol}_{datetime.now().strftime('%H%M%S')}",
                symbol=symbol,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY if signal == 'BUY' else OrderSide.SELL,
                quantity=position_size,
                timestamp=datetime.now()
            )
            
            orders.append(order)
        
        return orders
    
    def _calculate_position_size(self, symbol: str, current_price: float, signal: str) -> float:
        """Calculate position size for order"""
        portfolio_value = self.portfolio.portfolio_value
        max_position_value = portfolio_value * self.max_position_size
        
        # Simple position sizing - 2% of portfolio per trade
        position_value = portfolio_value * 0.02
        position_size = position_value / current_price
        
        # Round down to integer shares
        return int(position_size)
    
    def _process_executed_trades(self, trades: List[Trade]):
        """Process executed trades"""
        for trade in trades:
            # Update portfolio
            self.portfolio.execute_trade(trade)
            
            # Log trade
            self.logger.info(
                f"Trade executed: {trade.side} {trade.quantity} {trade.symbol} "
                f"@ {trade.price:.2f} (P&L: {trade.commission:.2f})"
            )
            
            # Send trade alert
            self.alert_manager.send_alert(
                "TRADE_EXECUTED",
                f"{trade.side} {trade.quantity} {trade.symbol} @ {trade.price:.2f}",
                "INFO"
            )
    
    def _log_trading_cycle(self, signals: Dict, orders: List[Order]):
        """Log trading cycle information"""
        buy_signals = sum(1 for s in signals.values() if s == 'BUY')
        sell_signals = sum(1 for s in signals.values() if s == 'SELL')
        
        self.logger.info(
            f"Trading cycle: {buy_signals} BUY, {sell_signals} SELL, "
            f"{len(orders)} orders generated"
        )
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        
        # Simple time-based check
        market_open = datetime.now().replace(hour=9, minute=15, second=0)
        market_close = datetime.now().replace(hour=15, minute=30, second=0)
        
        return market_open <= now <= market_close and now.weekday() < 5
    
    def _on_market_open(self):
        """Handle market open"""
        self.logger.info("Market opened")
        self.alert_manager.send_alert("MARKET_OPEN", "Market has opened", "INFO")
        
        # Sync with broker at market open
        self._sync_with_broker()
    
    def _on_market_close(self):
        """Handle market close"""
        self.logger.info("Market closed")
        self.alert_manager.send_alert("MARKET_CLOSE", "Market has closed", "INFO")
        
        # Generate end-of-day report
        self._generate_eod_report()
    
    def _get_cycle_interval(self) -> int:
        """Get trading cycle interval in seconds"""
        # Adjust frequency based on market conditions
        return 300  # 5 minutes
    
    def _health_monitoring_loop(self):
        """Health monitoring loop"""
        while not self.stop_event.is_set() and self.is_running:
            try:
                # Check system health
                health_status = self.health_monitor.check_health()
                
                if not health_status['is_healthy']:
                    self.alert_manager.send_alert(
                        "SYSTEM_HEALTH",
                        f"Health check failed: {health_status['issues']}",
                        "WARNING"
                    )
                
                # Check connection health
                self._check_connections()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(60)
    
    def _check_connections(self):
        """Check critical connections"""
        try:
            # Check broker connection
            self.execution_handler.get_account_info()
            
            # Check data feed connection
            # This would verify real-time data feeds
            
        except Exception as e:
            self.logger.error(f"Connection check failed: {e}")
            self.alert_manager.send_alert(
                "CONNECTION_ERROR",
                f"Connection issue: {e}",
                "ERROR"
            )
    
    def _generate_eod_report(self):
        """Generate end-of-day report"""
        try:
            report = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'portfolio_value': self.portfolio.portfolio_value,
                'daily_pnl': self.portfolio.daily_pnl,
                'positions': self.portfolio.positions,
                'trades_today': len([t for t in self.portfolio.trades 
                                   if t.timestamp.date() == datetime.now().date()])
            }
            
            self.logger.info(f"EOD Report: {report}")
            
        except Exception as e:
            self.logger.error(f"EOD report generation failed: {e}")
    
    def stop(self):
        """Stop production trading"""
        self.logger.info("Stopping production trading...")
        
        self.is_running = False
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=30)
        
        # Stop monitoring
        self.health_monitor.stop()
        self.alert_manager.stop()
        
        # Generate shutdown report
        self._generate_shutdown_report()
        
        self.logger.info("Production trading stopped")
        
        # Send shutdown alert
        self.alert_manager.send_alert(
            "SYSTEM_SHUTDOWN",
            "Trading system stopped",
            "INFO"
        )
    
    def _generate_shutdown_report(self):
        """Generate shutdown report"""
        try:
            report = {
                'shutdown_time': datetime.now().isoformat(),
                'final_portfolio_value': self.portfolio.portfolio_value,
                'total_trades': len(self.portfolio.trades),
                'active_positions': len(self.portfolio.positions)
            }
            
            self.logger.info(f"Shutdown Report: {report}")
            
        except Exception as e:
            self.logger.error(f"Shutdown report generation failed: {e}")
    
    def get_production_stats(self) -> Dict:
        """Get production statistics"""
        return {
            'broker': self.broker,
            'paper_trading': self.paper_trading,
            'is_running': self.is_running,
            'is_market_open': self.is_market_open,
            'portfolio_value': self.portfolio.portfolio_value,
            'active_positions': len(self.portfolio.positions),
            'total_trades': len(self.portfolio.trades),
            'last_signal_time': self.last_signal_time
        }
    
    def emergency_stop(self):
        """Emergency stop - immediately cancel all orders and close positions"""
        self.logger.warning("EMERGENCY STOP ACTIVATED")
        
        try:
            # Cancel all pending orders
            self.execution_handler.sync_orders()
            
            # Close all positions (implementation depends on broker)
            self._close_all_positions()
            
            # Stop trading
            self.stop()
            
            self.alert_manager.send_alert(
                "EMERGENCY_STOP",
                "Emergency stop activated - all positions closed",
                "CRITICAL"
            )
            
        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")
            self.alert_manager.send_alert(
                "EMERGENCY_STOP_ERROR",
                f"Emergency stop failed: {e}",
                "CRITICAL"
            )
    
    def _close_all_positions(self):
        """Close all open positions"""
        # This would implement position closing logic
        # Specific implementation depends on broker API
        pass
