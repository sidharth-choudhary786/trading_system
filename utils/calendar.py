# trading_system/utils/calendar.py
"""
Market calendar utilities for Indian stock markets (NSE/BSE).
Handles trading days, holidays, and market hours.
"""

import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Set
import holidays
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class MarketCalendar:
    """
    Indian Stock Market Calendar for NSE and BSE
    Handles trading days, holidays, and market timing
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.country = 'India'
        self.exchange = 'NSE'  # Default exchange
        
        # Market hours
        self.market_open = self.config.get('market_open', '09:15:00')
        self.market_close = self.config.get('market_close', '15:30:00')
        self.pre_open_start = self.config.get('pre_open_start', '09:00:00')
        self.pre_open_end = self.config.get('pre_open_end', '09:08:00')
        
        # Initialize holidays
        self.holidays = self._load_holidays()
        
        # Trading session cache
        self._trading_days_cache = {}
        
        logger.info(f"Market Calendar initialized for {self.exchange}")
    
    def _load_holidays(self) -> Set[date]:
        """Load market holidays for Indian exchanges"""
        # Use python-holidays for India public holidays
        india_holidays = holidays.India(years=range(2000, 2030))
        
        # Add exchange-specific holidays
        nse_holidays = set()
        
        # Major Indian holidays (these are common market holidays)
        major_holidays = [
            # Republic Day
            lambda year: date(year, 1, 26),
            # Mahashivratri (approximate - varies by year)
            lambda year: self._get_holiday_date(year, 2, 'maha_shivratri'),
            # Holi (approximate - varies by year)
            lambda year: self._get_holiday_date(year, 3, 'holi'),
            # Good Friday (varies by year)
            lambda year: self._get_good_friday(year),
            # Ram Navami (approximate - varies by year)
            lambda year: self._get_holiday_date(year, 4, 'ram_navami'),
            # Mahavir Jayanti (approximate)
            lambda year: self._get_holiday_date(year, 4, 'mahavir_jayanti'),
            # Independence Day
            lambda year: date(year, 8, 15),
            # Gandhi Jayanti
            lambda year: date(year, 10, 2),
            # Dussehra (approximate - varies by year)
            lambda year: self._get_holiday_date(year, 10, 'dussehra'),
            # Diwali (approximate - varies by year)
            lambda year: self._get_diwali_date(year),
            # Gurunanak Jayanti (approximate)
            lambda year: self._get_holiday_date(year, 11, 'gurunanak_jayanti'),
            # Christmas
            lambda year: date(year, 12, 25)
        ]
        
        # Generate holidays for next 30 years
        current_year = datetime.now().year
        for year in range(current_year - 10, current_year + 20):
            for holiday_func in major_holidays:
                try:
                    holiday_date = holiday_func(year)
                    if holiday_date:
                        nse_holidays.add(holiday_date)
                except Exception as e:
                    logger.warning(f"Error calculating holiday for {year}: {e}")
        
        # Add to India holidays
        for holiday in nse_holidays:
            india_holidays.append(holiday)
        
        return set(india_holidays.keys())
    
    def _get_good_friday(self, year: int) -> Optional[date]:
        """Calculate Good Friday date (varies each year)"""
        # Simple approximation - usually in March or April
        # In practice, you'd use proper Easter calculation
        if year == 2023:
            return date(2023, 4, 7)
        elif year == 2024:
            return date(2024, 3, 29)
        elif year == 2025:
            return date(2025, 4, 18)
        else:
            # Fallback: first Friday of April
            april_first = date(year, 4, 1)
            days_to_friday = (4 - april_first.weekday()) % 7
            return april_first + timedelta(days=days_to_friday - 2)  # Good Friday is 2 days before Easter
    
    def _get_diwali_date(self, year: int) -> Optional[date]:
        """Calculate Diwali date (approximate)"""
        # Diwali usually in October or November
        if year == 2023:
            return date(2023, 11, 12)
        elif year == 2024:
            return date(2024, 10, 31)
        elif year == 2025:
            return date(2025, 10, 20)
        else:
            # Fallback: late October
            return date(year, 10, 25)
    
    def _get_holiday_date(self, year: int, month: int, holiday_name: str) -> Optional[date]:
        """Get approximate date for floating holidays"""
        # Simple implementation - in production, use proper lunar calendar
        holiday_approximations = {
            'maha_shivratri': date(year, 2, 25),
            'holi': date(year, 3, 8),
            'ram_navami': date(year, 4, 2),
            'mahavir_jayanti': date(year, 4, 14),
            'dussehra': date(year, 10, 24),
            'gurunanak_jayanti': date(year, 11, 27)
        }
        return holiday_approximations.get(holiday_name)
    
    def is_trading_day(self, date_obj: date) -> bool:
        """
        Check if given date is a trading day
        
        Args:
            date_obj: Date to check
            
        Returns:
            True if trading day, False otherwise
        """
        # Check if weekend
        if date_obj.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False
        
        # Check if holiday
        if date_obj in self.holidays:
            return False
        
        return True
    
    def get_previous_trading_day(self, date_obj: date) -> date:
        """
        Get previous trading day
        
        Args:
            date_obj: Reference date
            
        Returns:
            Previous trading day
        """
        current_date = date_obj
        while True:
            current_date -= timedelta(days=1)
            if self.is_trading_day(current_date):
                return current_date
    
    def get_next_trading_day(self, date_obj: date) -> date:
        """
        Get next trading day
        
        Args:
            date_obj: Reference date
            
        Returns:
            Next trading day
        """
        current_date = date_obj
        while True:
            current_date += timedelta(days=1)
            if self.is_trading_day(current_date):
                return current_date
    
    def get_trading_days(self, start_date: date, end_date: date) -> List[date]:
        """
        Get all trading days between start and end date
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trading days
        """
        cache_key = f"{start_date}_{end_date}"
        if cache_key in self._trading_days_cache:
            return self._trading_days_cache[cache_key]
        
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if self.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        self._trading_days_cache[cache_key] = trading_days
        return trading_days
    
    def get_trading_days_series(self, start_date: date, end_date: date) -> pd.DatetimeIndex:
        """
        Get trading days as pandas DatetimeIndex
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DatetimeIndex of trading days
        """
        trading_days = self.get_trading_days(start_date, end_date)
        return pd.DatetimeIndex(trading_days)
    
    def get_holidays(self, start_date: date, end_date: date) -> List[date]:
        """
        Get holidays between start and end date
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of holidays
        """
        holidays_list = []
        current_date = start_date
        
        while current_date <= end_date:
            if current_date in self.holidays:
                holidays_list.append(current_date)
            current_date += timedelta(days=1)
        
        return holidays_list
    
    def is_market_open(self, datetime_obj: datetime) -> bool:
        """
        Check if market is open at given datetime
        
        Args:
            datetime_obj: Datetime to check
            
        Returns:
            True if market is open
        """
        if not self.is_trading_day(datetime_obj.date()):
            return False
        
        time_str = datetime_obj.strftime('%H:%M:%S')
        return self.market_open <= time_str <= self.market_close
    
    def is_pre_open_session(self, datetime_obj: datetime) -> bool:
        """
        Check if it's pre-open session
        
        Args:
            datetime_obj: Datetime to check
            
        Returns:
            True if pre-open session
        """
        if not self.is_trading_day(datetime_obj.date()):
            return False
        
        time_str = datetime_obj.strftime('%H:%M:%S')
        return self.pre_open_start <= time_str <= self.pre_open_end
    
    def get_market_hours(self) -> Dict[str, str]:
        """Get market timing information"""
        return {
            'pre_open_start': self.pre_open_start,
            'pre_open_end': self.pre_open_end,
            'market_open': self.market_open,
            'market_close': self.market_close
        }
    
    def get_trading_day_range(self, date_obj: date, days: int = 30) -> Dict[str, date]:
        """
        Get trading day range around given date
        
        Args:
            date_obj: Reference date
            days: Number of trading days to look back/forward
            
        Returns:
            Dictionary with start and end dates
        """
        end_date = date_obj
        start_date = end_date
        
        # Go back specified number of trading days
        for _ in range(days):
            start_date = self.get_previous_trading_day(start_date)
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'trading_days_count': days
        }
    
    def get_fortnightly_rebalance_dates(self, start_date: date, end_date: date) -> List[date]:
        """
        Get fortnightly rebalance dates (every 10 trading days)
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of rebalance dates
        """
        trading_days = self.get_trading_days(start_date, end_date)
        rebalance_dates = []
        
        for i, trading_day in enumerate(trading_days):
            if i % 10 == 0:  # Every 10 trading days
                rebalance_dates.append(trading_day)
        
        return rebalance_dates
    
    def validate_date_range(self, start_date: date, end_date: date) -> bool:
        """
        Validate date range for trading
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            True if valid date range
        """
        if start_date > end_date:
            logger.error("Start date cannot be after end date")
            return False
        
        trading_days = self.get_trading_days(start_date, end_date)
        if len(trading_days) == 0:
            logger.warning("No trading days in specified date range")
            return False
        
        return True

# Utility functions
def create_market_calendar(config: Optional[Dict] = None) -> MarketCalendar:
    """Factory function to create market calendar"""
    return MarketCalendar(config)

def is_weekend(date_obj: date) -> bool:
    """Check if date is weekend"""
    return date_obj.weekday() >= 5

def add_trading_days(start_date: date, num_days: int, calendar: MarketCalendar) -> date:
    """Add trading days to date"""
    current_date = start_date
    days_added = 0
    
    while days_added < num_days:
        current_date += timedelta(days=1)
        if calendar.is_trading_day(current_date):
            days_added += 1
    
    return current_date

# Example usage
if __name__ == "__main__":
    # Test market calendar
    calendar = MarketCalendar()
    
    test_date = date(2023, 12, 15)
    print(f"Is {test_date} a trading day? {calendar.is_trading_day(test_date)}")
    
    # Get trading days for a month
    start_date = date(2023, 12, 1)
    end_date = date(2023, 12, 31)
    trading_days = calendar.get_trading_days(start_date, end_date)
    print(f"Trading days in December 2023: {len(trading_days)}")
    
    # Get market hours
    market_hours = calendar.get_market_hours()
    print(f"Market hours: {market_hours}")
