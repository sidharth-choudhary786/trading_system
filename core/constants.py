# trading_system/core/constants.py
from enum import Enum

# Time constants
MARKET_OPEN_TIME = "09:15:00"
MARKET_CLOSE_TIME = "15:30:00"
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Indian market constants
INR_CURRENCY = "INR"
INDIAN_TIMEZONE = "Asia/Kolkata"
NSE_EXCHANGE = "NSE"
BSE_EXCHANGE = "BSE"

# Risk constants
MAX_POSITION_SIZE = 0.1  # 10% per stock
MAX_SECTOR_EXPOSURE = 0.3  # 30% per sector
MAX_DAILY_LOSS = 0.02  # 2% daily loss limit
MAX_DRAWDOWN = 0.15  # 15% maximum drawdown

# Trading constants
DEFAULT_COMMISSION = 0.001  # 0.1% commission
DEFAULT_SLIPPAGE = 0.001   # 0.1% slippage
MINIMUM_QUANTITY = 1

# Data constants
DEFAULT_START_DATE = "2005-01-01"
DATA_SOURCES = ["twelvedata", "yfinance", "investpy", "alphavantage"]
