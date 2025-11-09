# trading_system/features/technical_indicators.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import talib
from scipy import stats

from ..core.exceptions import TradingSystemError

class TechnicalIndicators:
    """
    Comprehensive technical indicators generator
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Indicator parameters from config
        self.indicator_params = config.get('technical_indicators', {})
        
    def add_sma(self, data: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Add Simple Moving Averages"""
        if periods is None:
            periods = self.indicator_params.get('sma_periods', [5, 10, 20, 50, 200])
        
        result_data = data.copy()
        
        for period in periods:
            if 'close' in result_data.columns:
                result_data[f'sma_{period}'] = talib.SMA(result_data['close'], timeperiod=period)
                
                # Price relative to SMA
                result_data[f'price_vs_sma_{period}'] = (
                    result_data['close'] / result_data[f'sma_{period}'] - 1
                )
        
        return result_data
    
    def add_ema(self, data: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Add Exponential Moving Averages"""
        if periods is None:
            periods = self.indicator_params.get('ema_periods', [12, 26])
        
        result_data = data.copy()
        
        for period in periods:
            if 'close' in result_data.columns:
                result_data[f'ema_{period}'] = talib.EMA(result_data['close'], timeperiod=period)
        
        # Add EMA crossovers
        if all(f'ema_{p}' in result_data.columns for p in [12, 26]):
            result_data['ema_12_26_diff'] = result_data['ema_12'] - result_data['ema_26']
            result_data['ema_12_26_signal'] = np.where(
                result_data['ema_12_26_diff'] > 0, 1, -1
            )
        
        return result_data
    
    def add_rsi(self, data: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """Add Relative Strength Index"""
        if period is None:
            period = self.indicator_params.get('rsi_period', 14)
        
        result_data = data.copy()
        
        if 'close' in result_data.columns:
            result_data['rsi'] = talib.RSI(result_data['close'], timeperiod=period)
            
            # RSI-based signals
            result_data['rsi_overbought'] = (result_data['rsi'] > 70).astype(int)
            result_data['rsi_oversold'] = (result_data['rsi'] < 30).astype(int)
            result_data['rsi_neutral'] = ((result_data['rsi'] >= 30) & (result_data['rsi'] <= 70)).astype(int)
        
        return result_data
    
    def add_macd(self, data: pd.DataFrame, fast: int = None, slow: int = None, signal: int = None) -> pd.DataFrame:
        """Add Moving Average Convergence Divergence"""
        if fast is None:
            fast = self.indicator_params.get('macd_fast', 12)
        if slow is None:
            slow = self.indicator_params.get('macd_slow', 26)
        if signal is None:
            signal = self.indicator_params.get('macd_signal', 9)
        
        result_data = data.copy()
        
        if 'close' in result_data.columns:
            macd, macd_signal, macd_hist = talib.MACD(
                result_data['close'], 
                fastperiod=fast, 
                slowperiod=slow, 
                signalperiod=signal
            )
            
            result_data['macd'] = macd
            result_data['macd_signal'] = macd_signal
            result_data['macd_histogram'] = macd_hist
            
            # MACD signals
            result_data['macd_signal_cross'] = np.where(macd > macd_signal, 1, -1)
            result_data['macd_histogram_trend'] = np.where(macd_hist > 0, 1, -1)
        
        return result_data
    
    def add_bollinger_bands(self, data: pd.DataFrame, period: int = None, std: float = None) -> pd.DataFrame:
        """Add Bollinger Bands"""
        if period is None:
            period = self.indicator_params.get('bollinger_period', 20)
        if std is None:
            std = self.indicator_params.get('bollinger_std', 2)
        
        result_data = data.copy()
        
        if 'close' in result_data.columns:
            upper, middle, lower = talib.BBANDS(
                result_data['close'], 
                timeperiod=period, 
                nbdevup=std, 
                nbdevdn=std
            )
            
            result_data['bb_upper'] = upper
            result_data['bb_middle'] = middle
            result_data['bb_lower'] = lower
            
            # Bollinger Band indicators
            result_data['bb_position'] = (result_data['close'] - result_data['bb_lower']) / (
                result_data['bb_upper'] - result_data['bb_lower']
            )
            result_data['bb_width'] = (
                result_data['bb_upper'] - result_data['bb_lower']
            ) / result_data['bb_middle']
            
            # Band touch indicators
            result_data['bb_touch_upper'] = (
                result_data['high'] >= result_data['bb_upper']
            ).astype(int)
            result_data['bb_touch_lower'] = (
                result_data['low'] <= result_data['bb_lower']
            ).astype(int)
        
        return result_data
    
    def add_atr(self, data: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """Add Average True Range"""
        if period is None:
            period = self.indicator_params.get('atr_period', 14)
        
        result_data = data.copy()
        
        if all(col in result_data.columns for col in ['high', 'low', 'close']):
            result_data['atr'] = talib.ATR(
                result_data['high'], 
                result_data['low'], 
                result_data['close'], 
                timeperiod=period
            )
            
            # Normalized ATR
            result_data['atr_pct'] = result_data['atr'] / result_data['close']
            
            # ATR-based volatility
            result_data['atr_ratio'] = result_data['atr'] / result_data['atr'].rolling(period).mean()
        
        return result_data
    
    def add_stochastic(self, data: pd.DataFrame, k_period: int = None, d_period: int = None) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        if k_period is None:
            k_period = 14
        if d_period is None:
            d_period = 3
        
        result_data = data.copy()
        
        if all(col in result_data.columns for col in ['high', 'low', 'close']):
            slowk, slowd = talib.STOCH(
                result_data['high'], 
                result_data['low'], 
                result_data['close'],
                fastk_period=k_period,
                slowk_period=d_period,
                slowd_period=d_period
            )
            
            result_data['stoch_k'] = slowk
            result_data['stoch_d'] = slowd
            
            # Stochastic signals
            result_data['stoch_overbought'] = ((slowk > 80) & (slowd > 80)).astype(int)
            result_data['stoch_oversold'] = ((slowk < 20) & (slowd < 20)).astype(int)
            result_data['stoch_cross'] = np.where(slowk > slowd, 1, -1)
        
        return result_data
    
    def add_williams_r(self, data: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """Add Williams %R"""
        if period is None:
            period = 14
        
        result_data = data.copy()
        
        if all(col in result_data.columns for col in ['high', 'low', 'close']):
            williams_r = talib.WILLR(
                result_data['high'], 
                result_data['low'], 
                result_data['close'], 
                timeperiod=period
            )
            
            result_data['williams_r'] = williams_r
            
            # Williams %R signals
            result_data['williams_overbought'] = (williams_r > -20).astype(int)
            result_data['williams_oversold'] = (williams_r < -80).astype(int)
        
        return result_data
    
    def add_cci(self, data: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """Add Commodity Channel Index"""
        if period is None:
            period = 20
        
        result_data = data.copy()
        
        if all(col in result_data.columns for col in ['high', 'low', 'close']):
            cci = talib.CCI(
                result_data['high'], 
                result_data['low'], 
                result_data['close'], 
                timeperiod=period
            )
            
            result_data['cci'] = cci
            
            # CCI signals
            result_data['cci_overbought'] = (cci > 100).astype(int)
            result_data['cci_oversold'] = (cci < -100).astype(int)
            result_data['cci_trend'] = np.where(cci > 0, 1, -1)
        
        return result_data
    
    def add_obv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume"""
        result_data = data.copy()
        
        if all(col in result_data.columns for col in ['close', 'volume']):
            obv = talib.OBV(result_data['close'], result_data['volume'])
            result_data['obv'] = obv
            
            # OBV trends
            result_data['obv_trend'] = np.where(obv.diff() > 0, 1, -1)
            result_data['obv_ma_20'] = obv.rolling(20).mean()
            result_data['obv_vs_ma'] = (obv / result_data['obv_ma_20'] - 1)
        
        return result_data
    
    def add_trailing_stops(self, data: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
        """Add trailing stop indicators"""
        result_data = data.copy()
        
        if 'close' in result_data.columns:
            # Calculate ATR for trailing stop
            if 'atr' not in result_data.columns:
                result_data = self.add_atr(result_data, period)
            
            # Trailing stop based on ATR
            result_data['trailing_stop_buy'] = (
                result_data['close'].rolling(period).max() - result_data['atr'] * multiplier
            )
            result_data['trailing_stop_sell'] = (
                result_data['close'].rolling(period).min() + result_data['atr'] * multiplier
            )
            
            # Stop signals
            result_data['buy_signal_trailing'] = (
                result_data['close'] > result_data['trailing_stop_buy']
            ).astype(int)
            result_data['sell_signal_trailing'] = (
                result_data['close'] < result_data['trailing_stop_sell']
            ).astype(int)
        
        return result_data
    
    def add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add various momentum indicators"""
        result_data = data.copy()
        
        if 'close' in result_data.columns:
            # Rate of Change
            for period in [5, 10, 20]:
                result_data[f'roc_{period}'] = talib.ROC(result_data['close'], timeperiod=period)
            
            # Momentum
            for period in [5, 10]:
                result_data[f'momentum_{period}'] = talib.MOM(result_data['close'], timeperiod=period)
            
            # Price Rate of Change
            result_data['price_roc_1d'] = result_data['close'].pct_change()
            result_data['price_roc_5d'] = result_data['close'].pct_change(5)
            result_data['price_roc_20d'] = result_data['close'].pct_change(20)
        
        return result_data
    
    def add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators"""
        result_data = data.copy()
        
        if 'close' in result_data.columns:
            returns = result_data['close'].pct_change()
            
            # Historical volatility
            for period in [10, 20, 50]:
                result_data[f'hist_vol_{period}'] = (
                    returns.rolling(period).std() * np.sqrt(252)
                )
            
            # Parkinson volatility (using high-low range)
            if all(col in result_data.columns for col in ['high', 'low']):
                for period in [10, 20]:
                    log_hl = np.log(result_data['high'] / result_data['low'])
                    result_data[f'parkinson_vol_{period}'] = (
                        np.sqrt(1/(4 * period * np.log(2)) * (log_hl**2).rolling(period).sum()) * np.sqrt(252)
                    )
        
        return result_data
    
    def get_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all technical indicators"""
        result_data = data.copy()
        
        # Add all indicator categories
        result_data = self.add_sma(result_data)
        result_data = self.add_ema(result_data)
        result_data = self.add_rsi(result_data)
        result_data = self.add_macd(result_data)
        result_data = self.add_bollinger_bands(result_data)
        result_data = self.add_atr(result_data)
        result_data = self.add_stochastic(result_data)
        result_data = self.add_williams_r(result_data)
        result_data = self.add_cci(result_data)
        result_data = self.add_obv(result_data)
        result_data = self.add_trailing_stops(result_data)
        result_data = self.add_momentum_indicators(result_data)
        result_data = self.add_volatility_indicators(result_data)
        
        return result_data
    
    def get_indicator_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate consolidated trading signals from all indicators"""
        result_data = data.copy()
        
        # Initialize signal columns
        result_data['signal_strength'] = 0
        result_data['buy_signals'] = 0
        result_data['sell_signals'] = 0
        
        # RSI signals
        if 'rsi' in result_data.columns:
            result_data['signal_strength'] += np.where(
                result_data['rsi'] < 30, 1, 0  # Oversold -> buy
            )
            result_data['signal_strength'] += np.where(
                result_data['rsi'] > 70, -1, 0  # Overbought -> sell
            )
        
        # MACD signals
        if 'macd_signal_cross' in result_data.columns:
            result_data['signal_strength'] += result_data['macd_signal_cross']
        
        # Bollinger Band signals
        if 'bb_position' in result_data.columns:
            result_data['signal_strength'] += np.where(
                result_data['bb_position'] < 0.1, 1, 0  # Near lower band -> buy
            )
            result_data['signal_strength'] += np.where(
                result_data['bb_position'] > 0.9, -1, 0  # Near upper band -> sell
            )
        
        # Stochastic signals
        if 'stoch_oversold' in result_data.columns:
            result_data['signal_strength'] += result_data['stoch_oversold']
            result_data['signal_strength'] -= result_data['stoch_overbought']
        
        # Count individual signals
        buy_conditions = [
            'rsi_oversold', 'stoch_oversold', 'williams_oversold', 
            'cci_oversold', 'buy_signal_trailing'
        ]
        
        sell_conditions = [
            'rsi_overbought', 'stoch_overbought', 'williams_overbought',
            'cci_overbought', 'sell_signal_trailing'
        ]
        
        for condition in buy_conditions:
            if condition in result_data.columns:
                result_data['buy_signals'] += result_data[condition]
        
        for condition in sell_conditions:
            if condition in result_data.columns:
                result_data['sell_signals'] += result_data[condition]
        
        # Final consolidated signal
        result_data['consolidated_signal'] = np.where(
            result_data['buy_signals'] > result_data['sell_signals'], 1,
            np.where(result_data['sell_signals'] > result_data['buy_signals'], -1, 0)
        )
        
        return result_data
