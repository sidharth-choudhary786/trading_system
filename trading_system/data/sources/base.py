from abc import ABC, abstractmethod
import pandas as pd

class DataSource(ABC):
    @abstractmethod
    def get_historical_data(self, symbol, start_date, end_date, interval='1day'):
        pass

    @abstractmethod
    def get_dividends(self, symbol, start_date, end_date):
        pass

    @abstractmethod
    def get_splits(self, symbol, start_date, end_date):
        pass
