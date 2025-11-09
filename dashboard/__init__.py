# trading_system/dashboard/__init__.py
from .app import TradingDashboard
from .components.portfolio_view import PortfolioView
from .components.performance_charts import PerformanceCharts
from .components.trade_journal import TradeJournal

__all__ = [
    'TradingDashboard',
    'PortfolioView', 
    'PerformanceCharts',
    'TradeJournal'
]
