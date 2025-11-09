# trading_system/utils/helpers.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import json
import hashlib
import re

def validate_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format
    
    Args:
        symbol: Symbol to validate
        
    Returns:
        True if valid
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic symbol validation
    pattern = r'^[A-Z0-9.-]+$'
    return bool(re.match(pattern, symbol))

def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate returns from price series
    
    Args:
        prices: Price series
        
    Returns:
        Returns series
    """
    return prices.pct_change()

def calculate_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """
    Calculate rolling volatility
    
    Args:
        returns: Returns series
        window: Rolling window
        
    Returns:
        Volatility series
    """
    return returns.rolling(window=window).std() * np.sqrt(252)

def normalize_series(series: pd.Series) -> pd.Series:
    """
    Normalize series to 0-1 range
    
    Args:
        series: Series to normalize
        
    Returns:
        Normalized series
    """
    return (series - series.min()) / (series.max() - series.min())

def standardize_series(series: pd.Series) -> pd.Series:
    """
    Standardize series (z-score normalization)
    
    Args:
        series: Series to standardize
        
    Returns:
        Standardized series
    """
    return (series - series.mean()) / series.std()

def generate_trade_id(symbol: str, timestamp: datetime) -> str:
    """
    Generate unique trade ID
    
    Args:
        symbol: Trade symbol
        timestamp: Trade timestamp
        
    Returns:
        Unique trade ID
    """
    base_string = f"{symbol}_{timestamp.strftime('%Y%m%d%H%M%S')}"
    return hashlib.md5(base_string.encode()).hexdigest()[:12]

def calculate_portfolio_weights(returns: pd.DataFrame, method: str = 'equal') -> Dict[str, float]:
    """
    Calculate portfolio weights
    
    Args:
        returns: Returns DataFrame
        method: Weighting method
        
    Returns:
        Dictionary of symbol -> weight
    """
    symbols = returns.columns.tolist()
    
    if method == 'equal':
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}
    
    elif method == 'volatility':
        # Inverse volatility weighting
        volatilities = returns.std()
        inverse_vol = 1.0 / volatilities
        total_inverse_vol = inverse_vol.sum()
        return {symbol: inverse_vol[symbol] / total_inverse_vol for symbol in symbols}
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")

def format_currency(amount: float) -> str:
    """
    Format amount as Indian currency
    
    Args:
        amount: Amount to format
        
    Returns:
        Formatted currency string
    """
    if amount >= 1e7:  >= 1 crore
        return f"₹{amount/1e7:.2f}Cr"
    elif amount >= 1e5:  >= 1 lakh
        return f"₹{amount/1e5:.2f}L"
    else:
        return f"₹{amount:,.0f}"

def calculate_drawdown(portfolio_values: pd.Series) -> pd.Series:
    """
    Calculate portfolio drawdown
    
    Args:
        portfolio_values: Portfolio value series
        
    Returns:
        Drawdown series
    """
    rolling_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - rolling_max) / rolling_max
    return drawdown

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with zero handling
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Division result
    """
    if denominator == 0:
        return default
    return numerator / denominator

def parse_date(date_string: str) -> datetime:
    """
    Parse date string with multiple format support
    
    Args:
        date_string: Date string to parse
        
    Returns:
        Datetime object
    """
    formats = [
        '%Y-%m-%d',
        '%d/%m/%Y', 
        '%d-%m-%Y',
        '%Y%m%d',
        '%d %b %Y',
        '%d %B %Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse date: {date_string}")

def create_date_range(start_date: datetime, end_date: datetime, freq: str = 'D') -> List[datetime]:
    """
    Create date range
    
    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency
        
    Returns:
        List of dates
    """
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    return [date.to_pydatetime() for date in dates]

def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix
    
    Args:
        returns: Returns DataFrame
        
    Returns:
        Correlation matrix
    """
    return returns.corr()

def detect_anomalies(series: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """
    Detect anomalies in series
    
    Args:
        series: Input series
        method: Detection method
        threshold: Threshold for detection
        
    Returns:
        Boolean series indicating anomalies
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        zscore = (series - series.mean()) / series.std()
        return np.abs(zscore) > threshold
    
    else:
        raise ValueError(f"Unknown anomaly detection method: {method}")

def calculate_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate rolling mean
    
    Args:
        series: Input series
        window: Rolling window
        
    Returns:
        Rolling mean series
    """
    return series.rolling(window=window).mean()

def calculate_rolling_std(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate rolling standard deviation
    
    Args:
        series: Input series
        window: Rolling window
        
    Returns:
        Rolling std series
    """
    return series.rolling(window=window).std()

def serialize_to_json(data: Any) -> str:
    """
    Serialize data to JSON string
    
    Args:
        data: Data to serialize
        
    Returns:
        JSON string
    """
    def json_serializer(obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    return json.dumps(data, default=json_serializer, indent=2)

def deserialize_from_json(json_string: str) -> Any:
    """
    Deserialize JSON string to Python object
    
    Args:
        json_string: JSON string
        
    Returns:
        Python object
    """
    return json.loads(json_string)

def calculate_compound_annual_growth_rate(start_value: float, end_value: float, years: float) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR)
    
    Args:
        start_value: Starting value
        end_value: Ending value
        years: Number of years
        
    Returns:
        CAGR
    """
    if years <= 0 or start_value <= 0:
        return 0.0
    return (end_value / start_value) ** (1 / years) - 1

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage
    
    Args:
        value: Value to format
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}f}%"

def calculate_risk_adjusted_return(return_value: float, risk_value: float) -> float:
    """
    Calculate risk-adjusted return
    
    Args:
        return_value: Return value
        risk_value: Risk value (e.g., volatility)
        
    Returns:
        Risk-adjusted return
    """
    if risk_value == 0:
        return 0.0
    return return_value / risk_value
