# trading_system/analysis/performance.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis and metrics calculation
    """
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_all_metrics(
        self, 
        portfolio_values: List[float],
        dates: List[datetime],
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0
    ) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if len(portfolio_values) < 2:
            return {}
        
        # Convert to pandas series
        portfolio_series = pd.Series(portfolio_values, index=dates)
        returns = portfolio_series.pct_change().dropna()
        
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = self._calculate_total_return(portfolio_series)
        metrics['cagr'] = self._calculate_cagr(portfolio_series)
        metrics['volatility'] = self._calculate_volatility(returns)
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns, risk_free_rate)
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns, risk_free_rate)
        
        # Risk metrics
        metrics['max_drawdown'] = self._calculate_max_drawdown(portfolio_series)
        metrics['calmar_ratio'] = self._calculate_calmar_ratio(metrics['cagr'], metrics['max_drawdown'])
        metrics['var_95'] = self._calculate_var(returns, 0.95)
        metrics['cvar_95'] = self._calculate_cvar(returns, 0.95)
        
        # Benchmark comparison
        if benchmark_returns is not None:
            metrics['alpha'] = self._calculate_alpha(returns, benchmark_returns, risk_free_rate)
            metrics['beta'] = self._calculate_beta(returns, benchmark_returns)
            metrics['tracking_error'] = self._calculate_tracking_error(returns, benchmark_returns)
            metrics['information_ratio'] = self._calculate_information_ratio(returns, benchmark_returns)
        
        # Additional metrics
        metrics['win_rate'] = self._calculate_win_rate(returns)
        metrics['profit_factor'] = self._calculate_profit_factor(returns)
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        self.metrics = metrics
        return metrics
    
    def _calculate_total_return(self, portfolio_series: pd.Series) -> float:
        """Calculate total return"""
        return (portfolio_series.iloc[-1] - portfolio_series.iloc[0]) / portfolio_series.iloc[0]
    
    def _calculate_cagr(self, portfolio_series: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate"""
        days = (portfolio_series.index[-1] - portfolio_series.index[0]).days
        years = days / 365.25
        total_return = self._calculate_total_return(portfolio_series)
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(252)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        return excess_returns.mean() * np.sqrt(252) / downside_deviation if downside_deviation > 0 else 0
    
    def _calculate_max_drawdown(self, portfolio_series: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        return drawdown.min()
    
    def _calculate_calmar_ratio(self, cagr: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        return cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk"""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def _calculate_alpha(self, returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Alpha"""
        excess_returns = returns - risk_free_rate / 252
        excess_benchmark = benchmark_returns - risk_free_rate / 252
        beta = self._calculate_beta(returns, benchmark_returns)
        alpha = excess_returns.mean() - beta * excess_benchmark.mean()
        return alpha * 252  # Annualize
    
    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Beta"""
        covariance = returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        return covariance / benchmark_variance if benchmark_variance > 0 else 0
    
    def _calculate_tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Tracking Error"""
        active_returns = returns - benchmark_returns
        return active_returns.std() * np.sqrt(252)
    
    def _calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio"""
        active_returns = returns - benchmark_returns
        tracking_error = self._calculate_tracking_error(returns, benchmark_returns)
        return active_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate"""
        return (returns > 0).mean()
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor"""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def generate_performance_report(self, metrics: Dict) -> str:
        """Generate formatted performance report"""
        report = []
        report.append("=" * 50)
        report.append("PORTFOLIO PERFORMANCE REPORT")
        report.append("=" * 50)
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                if 'return' in metric or 'ratio' in metric or 'alpha' in metric:
                    formatted_value = f"{value:.2%}"
                elif 'drawdown' in metric or 'var' in metric or 'cvar' in metric:
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.4f}"
                
                report.append(f"{metric.replace('_', ' ').title():<25}: {formatted_value}")
        
        return "\n".join(report)
