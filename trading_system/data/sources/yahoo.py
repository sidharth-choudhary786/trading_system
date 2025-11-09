import pandas as pd
from .base import DataSource

class YahooDataSource(DataSource):
    def get_historical_data(self, symbol, start_date, end_date, interval='1day'):
        # Implement using yahooquery or yfinance
        # Return OHLCV data as DataFrame with Adjusted Close
        pass

    def get_dividends(self, symbol, start_date, end_date):
        # Get dividends from yahoo
        pass

    def get_splits(self, symbol, start_date, end_date):
        # Get splits from yahoo
        pass
