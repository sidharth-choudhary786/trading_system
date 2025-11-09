# trading_system/utils/logger.py
"""
Comprehensive logging setup for the trading system.
Provides structured logging with different handlers and formats.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import colorama
from colorama import Fore, Style

# Initialize colorama for colored console output
colorama.init()

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
    
    def format(self, record):
        """Format log record with colors and structure"""
        # Add custom fields
        if not hasattr(record, 'module'):
            record.module = record.name
        
        # Color coding based on level
        level_colors = {
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Style.BRIGHT
        }
        
        record.levelcolor = level_colors.get(record.levelname, '')
        record.resetcolor = Style.RESET_ALL
        
        return super().format(record)

def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = False
) -> logging.Logger:
    """
    Setup comprehensive logging configuration for trading system
    
    Args:
        config: Configuration dictionary
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_json: Enable JSON formatted logging
    
    Returns:
        Configured logger instance
    """
    if config:
        log_level = config.get('log_level', log_level)
        log_file = config.get('log_file', log_file)
        enable_console = config.get('enable_console_logging', enable_console)
        enable_file = config.get('enable_file_logging', enable_file)
        enable_json = config.get('enable_json_logging', enable_json)
    
    # Create logs directory if it doesn't exist
    if log_file and enable_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console formatter
    console_formatter = StructuredFormatter(
        fmt='%(asctime)s %(levelcolor)s[%(levelname)8s]%(resetcolor)s %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File formatter
    file_formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)8s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # JSON formatter for structured logging
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            
            return json.dumps(log_entry)
    
    json_formatter = JSONFormatter()
    
    # Add console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if enable_file and log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # File gets all levels
        file_handler.setFormatter(file_formatter if not enable_json else json_formatter)
        logger.addHandler(file_handler)
    
    # Add specific loggers for different components
    component_loggers = [
        'trading_system.data',
        'trading_system.models', 
        'trading_system.portfolio',
        'trading_system.execution',
        'trading_system.risk',
        'trading_system.production'
    ]
    
    for component in component_loggers:
        comp_logger = logging.getLogger(component)
        comp_logger.setLevel(getattr(logging, log_level.upper()))
    
    logger.info(f"Logging setup completed - Level: {log_level}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get logger for specific component
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class PerformanceLogger:
    """Specialized logger for performance tracking"""
    
    def __init__(self, name: str = 'performance'):
        self.logger = get_logger(name)
        self.performance_data = {}
    
    def log_operation_start(self, operation: str, **kwargs):
        """Log start of an operation"""
        self.performance_data[operation] = {
            'start_time': datetime.now(),
            'kwargs': kwargs
        }
        self.logger.info(f"START {operation} - {kwargs}")
    
    def log_operation_end(self, operation: str, result: Any = None, **kwargs):
        """Log end of an operation with timing"""
        if operation in self.performance_data:
            start_data = self.performance_data[operation]
            duration = (datetime.now() - start_data['start_time']).total_seconds()
            
            log_data = {
                'operation': operation,
                'duration_seconds': duration,
                'result_type': type(result).__name__,
                **kwargs
            }
            
            self.logger.info(f"END {operation} - {duration:.3f}s - {kwargs}")
            
            # Clean up
            del self.performance_data[operation]
            
            return duration
        return 0
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        self.logger.info("PERFORMANCE_METRICS", extra={'metrics': metrics})
    
    def log_memory_usage(self):
        """Log current memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        self.logger.info(f"MEMORY_USAGE - {memory_mb:.2f} MB")

class TradeLogger:
    """Specialized logger for trade-related events"""
    
    def __init__(self, name: str = 'trades'):
        self.logger = get_logger(name)
    
    def log_order_placed(self, order_id: str, symbol: str, side: str, 
                        quantity: float, price: float, order_type: str):
        """Log order placement"""
        self.logger.info(
            f"ORDER_PLACED - {order_id} - {symbol} {side} {quantity} @ {price} ({order_type})"
        )
    
    def log_order_filled(self, trade_id: str, order_id: str, symbol: str,
                        side: str, quantity: float, price: float, 
                        commission: float, slippage: float):
        """Log order fill"""
        self.logger.info(
            f"ORDER_FILLED - {trade_id} - {order_id} - {symbol} {side} {quantity} @ {price} "
            f"(Commission: {commission:.2f}, Slippage: {slippage:.4f})"
        )
    
    def log_position_update(self, symbol: str, quantity: float, 
                          avg_price: float, unrealized_pnl: float):
        """Log position update"""
        self.logger.info(
            f"POSITION_UPDATE - {symbol} - Qty: {quantity}, Avg Price: {avg_price:.2f}, "
            f"Unrealized PnL: {unrealized_pnl:.2f}"
        )

def log_execution_time(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        logger = get_logger('performance')
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"EXECUTION_TIME - {func.__name__} - {duration:.3f}s")
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"EXECUTION_TIME_ERROR - {func.__name__} - {duration:.3f}s - {e}")
            raise
    
    return wrapper

# Example usage when module is run directly
if __name__ == "__main__":
    # Test logging setup
    setup_logging(log_level='DEBUG', log_file='logs/test.log')
    logger = get_logger(__name__)
    
    logger.debug("Debug message")
    logger.info("Info message") 
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    
    # Test performance logger
    perf_logger = PerformanceLogger()
    perf_logger.log_operation_start('data_download', symbols=['RELIANCE', 'TCS'])
    perf_logger.log_operation_end('data_download', records_downloaded=1000)
    
    print("Logging test completed!")
