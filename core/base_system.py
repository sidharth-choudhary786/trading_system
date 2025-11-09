# trading_system/core/base_system.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
from .exceptions import TradingSystemError
from .types import MarketData, Order, Trade

class BaseTradingSystem(ABC):
    """
    Abstract base class for trading system.
    Both backtesting and live trading will inherit from this.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.initial_capital = config.get('initial_capital', 100000)
        self.current_cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_value = self.initial_capital
        self.is_running = False
        
    @abstractmethod
    def initialize(self):
        """Initialize the trading system"""
        pass
    
    @abstractmethod
    def run(self):
        """Run the trading system"""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop the trading system"""
        pass
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        return self.portfolio_value
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        return self.positions.copy()
    
    def get_trades(self) -> List[Trade]:
        """Get trade history"""
        return self.trades.copy()
    
    def calculate_pnl(self) -> float:
        """Calculate current P&L"""
        return self.portfolio_value - self.initial_capital

class TradingSystem(BaseTradingSystem):
    """
    Main trading system implementation
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.mode = config.get('mode', 'backtest')  # 'backtest' or 'live'
        
    def initialize(self):
        """Initialize all components"""
        print(f"Initializing Trading System in {self.mode} mode...")
        
        # Initialize components based on config
        self._initialize_data_manager()
        self._initialize_models()
        self._initialize_portfolio()
        self._initialize_risk_manager()
        
        self.is_running = True
        print("Trading System initialized successfully!")
    
    def _initialize_data_manager(self):
        """Initialize data management components"""
        from data.data_manager import DataManager
        self.data_manager = DataManager(self.config)
        
    def _initialize_models(self):
        """Initialize model components"""
        from models.model_manager import ModelManager
        self.model_manager = ModelManager(self.config)
        
    def _initialize_portfolio(self):
        """Initialize portfolio components"""
        from portfolio.portfolio import Portfolio
        self.portfolio = Portfolio(self.initial_capital)
        
    def _initialize_risk_manager(self):
        """Initialize risk management components"""
        from risk.risk_engine import RiskEngine
        self.risk_engine = RiskEngine(self.config)
    
    def run(self):
        """Main run loop"""
        if not self.is_running:
            raise TradingSystemError("System not initialized. Call initialize() first.")
            
        print("Starting Trading System...")
        
        if self.mode == 'backtest':
            self._run_backtest()
        else:
            self._run_live()
    
    def _run_backtest(self):
        """Run backtesting mode"""
        print("Running in backtest mode...")
        # Backtest logic will be implemented in backtester.py
        
    def _run_live(self):
        """Run live trading mode"""
        print("Running in live mode...")
        # Live trading logic will be implemented in production_manager.py
    
    def stop(self):
        """Stop the trading system"""
        self.is_running = False
        print("Trading System stopped.")
