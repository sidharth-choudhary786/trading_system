# trading_system/utils/calendar.py
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, DateOffset, MO
from pandas.tseries.offsets import CustomBusinessDay

class IndianHolidayCalendar(AbstractHolidayCalendar):
    """
    Indian Market Holiday Calendar for NSE/BSE
    """
    rules = [
        Holiday('Republic Day', month=1, day=26),
        Holiday('Holi', month=3, day=7),  # Example date
        Holiday('Good Friday', month=4, day=2),  # Example date
        Holiday('Ram Navami', month=4, day=10),  # Example date
        Holiday('Maharashtra Day', month=5, day=1),
        Holiday('Bakri Id', month=6, day=29),  # Example date
        Holiday('Independence Day', month=8, day=15),
        Holiday('Ganesh Chaturthi', month=9, day=10),  # Example date
        Holiday('Gandhi Jayanti', month=10, day=2),
        Holiday('Dussehra', month=10, day=15),  # Example date
        Holiday('Diwali', month=11, day=14),  # Example date
        Holiday('Gurunanak Jayanti', month=11, day=27),  # Example date
        Holiday('Christmas', month=12, day=25)
    ]

class MarketCalendar:
    """
    Market calendar utility for Indian exchanges
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Indian business day (Monday to Friday)
        self.indian_bday = CustomBusinessDay(calendar=IndianHolidayCalendar())
        
        # Market hours
        self.market_open = pd.Timestamp("09:15:00").time()
        self.market_close = pd.Timestamp("15:30:00").time()
        
        # Pre-calculate holidays for performance
        self._precalculate_holidays()
    
    def _precalculate_holidays(self):
        """Pre-calculate holidays for quick lookup"""
        start_date = datetime(2000, 1, 1)
        end_date = datetime(2030, 12, 31)
        
        self.holidays = IndianHolidayCalendar().holidays(
            start=start_date, 
            end=end_date
        )
    
    def is_trading_day(self, date: datetime) -> bool:
        """
        Check if date is a trading day
        
        Args:
            date: Date to check
            
        Returns:
            True if trading day
        """
        # Check if weekend
        if date.weekday() >= 5:  # Saturday (5) or Sunday (6)
            return False
        
        # Check if holiday
        date_only = date.date()
        if date_only in [h.date() for h in self.holidays]:
            return False
        
        return True
    
    def is_market_hours(self, datetime_obj: datetime) -> bool:
        """
        Check if current time is within market hours
        
        Args:
            datetime_obj: Datetime to check
            
        Returns:
            True if market is open
        """
        if not self.is_trading_day(datetime_obj):
            return False
        
        current_time = datetime_obj.time()
        return self.market_open <= current_time <= self.market_close
    
    def get_previous_trading_day(self, date: datetime) -> datetime:
        """
        Get previous trading day
        
        Args:
            date: Reference date
            
        Returns:
            Previous trading day
        """
        current_date = pd.Timestamp(date)
        prev_date = current_date - self.indian_bday
        
        # Ensure we get a valid trading day
        while not self.is_trading_day(prev_date):
            prev_date -= self.indian_bday
        
        return prev_date.to_pydatetime()
    
    def get_next_trading_day(self, date: datetime) -> datetime:
        """
        Get next trading day
        
        Args:
            date: Reference date
            
        Returns:
            Next trading day
        """
        current_date = pd.Timestamp(date)
        next_date = current_date + self.indian_bday
        
        # Ensure we get a valid trading day
        while not self.is_trading_day(next_date):
            next_date += self.indian_bday
        
        return next_date.to_pydatetime()
    
    def get_trading_days_range(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        Get all trading days between start and end dates
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trading days
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = [date for date in dates if self.is_trading_day(date)]
        return [date.to_pydatetime() for date in trading_days]
    
    def get_trading_days_count(self, start_date: datetime, end_date: datetime) -> int:
        """
        Count number of trading days between dates
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Number of trading days
        """
        trading_days = self.get_trading_days_range(start_date, end_date)
        return len(trading_days)
    
    def add_trading_days(self, date: datetime, n_days: int) -> datetime:
        """
        Add n trading days to date
        
        Args:
            date: Start date
            n_days: Number of trading days to add
            
        Returns:
            Result date
        """
        current_date = pd.Timestamp(date)
        
        if n_days > 0:
            for _ in range(n_days):
                current_date += self.indian_bday
                while not self.is_trading_day(current_date):
                    current_date += self.indian_bday
        else:
            for _ in range(abs(n_days)):
                current_date -= self.indian_bday
                while not self.is_trading_day(current_date):
                    current_date -= self.indian_bday
        
        return current_date.to_pydatetime()
    
    def get_fortnightly_rebalance_dates(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        Get fortnightly rebalance dates (every 10 trading days)
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of rebalance dates
        """
        trading_days = self.get_trading_days_range(start_date, end_date)
        rebalance_days = trading_days[::10]  # Every 10 trading days
        
        return rebalance_days
    
    def get_market_holidays(self, year: int) -> List[datetime]:
        """
        Get market holidays for a specific year
        
        Args:
            year: Year to get holidays for
            
        Returns:
            List of holiday dates
        """
        year_holidays = [h for h in self.holidays if h.year == year]
        return [h.to_pydatetime() for h in year_holidays]
    
    def get_trading_calendar(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get trading calendar between dates
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with calendar information
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        calendar_data = []
        for date in dates:
            calendar_data.append({
                'date': date,
                'is_trading_day': self.is_trading_day(date),
                'day_of_week': date.strftime('%A'),
                'is_weekend': date.weekday() >= 5,
                'is_holiday': date in self.holidays
            })
        
        return pd.DataFrame(calendar_data)
