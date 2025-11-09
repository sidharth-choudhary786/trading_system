# trading_system/core/exceptions.py

class TradingSystemError(Exception):
    """Base exception for all trading system errors."""
    pass

class DataSourceError(TradingSystemError):
    """Exception raised for data source related errors."""
    pass

class ModelError(TradingSystemError):
    """Exception raised for model related errors."""
    pass

class PortfolioError(TradingSystemError):
    """Exception raised for portfolio related errors."""
    pass

class RiskError(TradingSystemError):
    """Exception raised for risk management errors."""
    pass

class ExecutionError(TradingSystemError):
    """Exception raised for order execution errors."""
    pass
