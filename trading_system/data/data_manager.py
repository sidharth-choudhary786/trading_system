import pandas as pd
from .sources.twelvedata import TwelveData
from .sources.yahoo import YahooDataSource

class DataManager:
    def __init__(self, config):
        self.config = config
        self.sources = []
        self._initialize_sources()

    def _initialize_sources(self):
        # Based on config, initialize data sources in order of preference
        if 'twelvedata' in self.config['data_sources']:
            self.sources.append(TwelveData(self.config['twelvedata_api_key']))
        if 'yahoo' in self.config['data_sources']:
            self.sources.append(YahooDataSource())

    def get_historical_data(self, symbol, start_date, end_date, interval='1day'):
        # Try each source until we get the data
        for source in self.sources:
            try:
                data = source.get_historical_data(symbol, start_date, end_date, interval)
                if data is not None and not data.empty:
                    # If we got data from twelvedata, we may need to adjust for splits and dividends
                    if isinstance(source, TwelveData):
                        # Adjust the data
                        dividends = source.get_dividends(symbol, start_date, end_date)
                        splits = source.get_splits(symbol, start_date, end_date)
                        data = self._adjust_data(data, dividends, splits)
                    return data
            except Exception as e:
                print(f"Error fetching data from {source.__class__.__name__} for {symbol}: {e}")
        return pd.DataFrame()

    def _adjust_data(self, data, dividends, splits):
        # Adjust the OHLC data for splits and dividends
        # This is a simplified version, in reality, we need to adjust each column appropriately
        # We'll implement the adjustment logic here
        return data
