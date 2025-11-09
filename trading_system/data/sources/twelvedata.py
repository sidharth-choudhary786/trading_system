import pandas as pd
from .base import DataSource

class TwelveData(DataSource):
    def __init__(self, api_key):
        self.api_key = api_key

    def get_historical_data(self, symbol, start_date, end_date, interval='1day'):
        # Implement using twelvedata API
        # Return OHLCV data as DataFrame
        pass

    def get_dividends(self, symbol, start_date, end_date):
        # Twelvedata may not provide dividends, so we return empty DataFrame
        return pd.DataFrame()

    def get_splits(self, symbol, start_date, end_date):
        # Twelvedata may not provide splits, so we return empty DataFrame
        return pd.DataFrame()
