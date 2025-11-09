# trading_system/dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..analysis.performance import PerformanceAnalyzer
from ..analysis.reporting import ReportGenerator
from ..portfolio.portfolio import Portfolio

class TradingDashboard:
    """
    Main Streamlit dashboard for trading system
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.performance_analyzer = PerformanceAnalyzer()
        self.report_generator = ReportGenerator()
        self.setup_page()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Algorithmic Trading System",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        .positive {
            color: #00cc96;
        }
        .negative {
            color: #ef553b;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Run the main dashboard"""
        st.markdown('<div class="main-header">🚀 Algorithmic Trading System Dashboard</div>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content based on selection
        page = st.session_state.get('page', 'overview')
        
        if page == 'overview':
            self.render_overview()
        elif page == 'portfolio':
            self.render_portfolio_analysis()
        elif page == 'performance':
            self.render_performance_analysis()
        elif page == 'trades':
            self.render_trade_analysis()
        elif page == 'risk':
            self.render_risk_analysis()
        elif page == 'reports':
            self.render_reports()
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/stock-share.png", 
                    width=80)
            st.title("Navigation")
            
            # Page selection
            page = st.radio(
                "Go to",
                ["Overview", "Portfolio", "Performance", "Trades", "Risk", "Reports"],
                key="page_selector"
            )
            
            st.session_state.page = page.lower()
            
            # System status
            st.subheader("System Status")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mode", self.config.get('mode', 'Backtest'))
                st.metric("Status", "🟢 Running")
            
            with col2:
                st.metric("Capital", f"₹{self.config.get('initial_capital', 0):,}")
                st.metric("Trades", "42")
            
            # Date range selector
            st.subheader("Date Range")
            date_range = st.selectbox(
                "Select Period",
                ["1M", "3M", "6M", "1Y", "YTD", "All"],
                index=4
            )
            st.session_state.date_range = date_range
            
            # Refresh button
            if st.button("🔄 Refresh Data"):
                st.rerun()
    
    def render_overview(self):
        """Render overview page"""
        st.header("📊 System Overview")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return", 
                "+15.2%", 
                "+2.1%",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio", 
                "1.85", 
                "+0.15"
            )
        
        with col3:
            st.metric(
                "Max Drawdown", 
                "-8.3%", 
                "-0.5%",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Win Rate", 
                "64.2%", 
                "+3.2%"
            )
        
        # Portfolio value chart
        st.subheader("Portfolio Performance")
        self.render_portfolio_chart()
        
        # Recent activity
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Recent Trades")
            self.render_recent_trades()
        
        with col2:
            st.subheader("Top Holdings")
            self.render_top_holdings()
    
    def render_portfolio_chart(self):
        """Render portfolio value chart"""
        # Generate sample data (in real app, this would come from database)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        portfolio_values = [1000000 * (1 + 0.001 * i + 0.002 * np.random.randn()) 
                          for i in range(len(dates))]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=3),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)'
        ))
        
        # Add benchmark (Nifty 50)
        benchmark_values = [1000000 * (1 + 0.0008 * i + 0.0015 * np.random.randn()) 
                          for i in range(len(dates))]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_values,
            mode='lines',
            name='Nifty 50',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Portfolio Value vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (₹)",
            template="plotly_white",
            height=400,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_recent_trades(self):
        """Render recent trades table"""
        # Sample trades data
        trades_data = {
            'Date': ['2023-12-15', '2023-12-14', '2023-12-13', '2023-12-12'],
            'Symbol': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK'],
            'Action': ['BUY', 'SELL', 'BUY', 'SELL'],
            'Quantity': [100, 50, 75, 25],
            'Price': [2456.75, 3450.20, 1589.30, 1625.80],
            'P&L': ['+2,450', '+1,225', '-875', '+312']
        }
        
        df = pd.DataFrame(trades_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def render_top_holdings(self):
        """Render top holdings"""
        holdings_data = {
            'Symbol': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR'],
            'Weight': ['24.5%', '18.2%', '15.8%', '12.3%', '9.7%'],
            'Value': ['₹2,45,000', '₹1,82,000', '₹1,58,000', '₹1,23,000', '₹97,000']
        }
        
        df = pd.DataFrame(holdings_data)
        
        # Create a bar chart for holdings
        fig = go.Figure(data=[
            go.Bar(
                x=df['Symbol'],
                y=[float(x.strip('%')) for x in df['Weight']],
                text=df['Weight'],
                textposition='auto',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            )
        ])
        
        fig.update_layout(
            title="Portfolio Allocation",
            xaxis_title="Stocks",
            yaxis_title="Weight (%)",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_portfolio_analysis(self):
        """Render portfolio analysis page"""
        st.header("📈 Portfolio Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Portfolio composition
            st.subheader("Portfolio Composition")
            self.render_portfolio_composition()
            
            # Sector allocation
            st.subheader("Sector Allocation")
            self.render_sector_allocation()
        
        with col2:
            # Portfolio statistics
            st.subheader("Portfolio Statistics")
            self.render_portfolio_stats()
            
            # Risk metrics
            st.subheader("Risk Metrics")
            self.render_risk_metrics()
    
    def render_portfolio_composition(self):
        """Render portfolio composition chart"""
        # Sample data
        symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'Others']
        weights = [24.5, 18.2, 15.8, 12.3, 9.7, 19.5]
        
        fig = go.Figure(data=[go.Pie(
            labels=symbols,
            values=weights,
            hole=0.4,
            marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        )])
        
        fig.update_layout(
            title="Portfolio Composition",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_sector_allocation(self):
        """Render sector allocation"""
        sectors = ['Technology', 'Financial', 'Energy', 'Consumer', 'Healthcare', 'Industrial']
        allocation = [32.5, 28.3, 15.2, 12.8, 6.5, 4.7]
        
        fig = go.Figure(data=[
            go.Bar(
                x=allocation,
                y=sectors,
                orientation='h',
                marker_color='#1f77b4'
            )
        ])
        
        fig.update_layout(
            title="Sector Allocation",
            xaxis_title="Weight (%)",
            yaxis_title="Sector",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_portfolio_stats(self):
        """Render portfolio statistics"""
        stats = {
            'Total Value': '₹10,12,450',
            'Cash': '₹1,25,680',
            'Invested': '₹8,86,770',
            'Number of Stocks': '15',
            'Diversification Score': '8.2/10',
            'Turnover Ratio': '12.5%'
        }
        
        for key, value in stats.items():
            st.metric(key, value)
    
    def render_risk_metrics(self):
        """Render risk metrics"""
        risk_metrics = {
            'Beta': '0.95',
            'Alpha': '2.3%',
            'Sharpe Ratio': '1.85',
            'Sortino Ratio': '2.42',
            'VaR (95%)': '-2.1%',
            'Max Drawdown': '-8.3%'
        }
        
        for key, value in risk_metrics.items():
            st.metric(key, value)
    
    def render_performance_analysis(self):
        """Render performance analysis page"""
        st.header("📊 Performance Analysis")
        
        # Performance metrics over time
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cumulative Returns")
            self.render_cumulative_returns()
        
        with col2:
            st.subheader("Drawdown Analysis")
            self.render_drawdown_chart()
        
        # Rolling performance
        st.subheader("Rolling Performance")
        self.render_rolling_performance()
        
        # Performance metrics table
        st.subheader("Detailed Performance Metrics")
        self.render_performance_table()
    
    def render_cumulative_returns(self):
        """Render cumulative returns chart"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        portfolio_returns = np.random.normal(0.001, 0.015, len(dates))
        benchmark_returns = np.random.normal(0.0008, 0.012, len(dates))
        
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=(portfolio_cumulative - 1) * 100,
            mode='lines',
            name='Portfolio',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=(benchmark_cumulative - 1) * 100,
            mode='lines',
            name='Nifty 50',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_drawdown_chart(self):
        """Render drawdown chart"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        portfolio_values = [1000000 * (1 + 0.001 * i + 0.002 * np.random.randn()) 
                          for i in range(len(dates))]
        
        portfolio_series = pd.Series(portfolio_values, index=dates)
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='#ef553b', width=2),
            fill='tozeroy',
            fillcolor='rgba(239, 85, 59, 0.3)'
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_rolling_performance(self):
        """Render rolling performance metrics"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Create tabs for different rolling periods
        tab1, tab2, tab3 = st.tabs(["30-day", "90-day", "252-day"])
        
        with tab1:
            st.plotly_chart(self._create_rolling_chart(dates, 30), use_container_width=True)
        with tab2:
            st.plotly_chart(self._create_rolling_chart(dates, 90), use_container_width=True)
        with tab3:
            st.plotly_chart(self._create_rolling_chart(dates, 252), use_container_width=True)
    
    def _create_rolling_chart(self, dates, window):
        """Create rolling performance chart"""
        returns = np.random.normal(0.001, 0.015, len(dates))
        rolling_sharpe = pd.Series(returns).rolling(window).mean() / pd.Series(returns).rolling(window).std() * np.sqrt(252)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates[window:],
            y=rolling_sharpe[window:],
            mode='lines',
            name=f'Rolling Sharpe ({window}D)',
            line=dict(color='#2ca02c', width=2)
        ))
        
        fig.update_layout(
            title=f"Rolling {window}-Day Sharpe Ratio",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            template="plotly_white",
            height=300
        )
        
        return fig
    
    def render_performance_table(self):
        """Render performance metrics table"""
        periods = ['1M', '3M', '6M', '1Y', 'YTD', 'Since Inception']
        portfolio_returns = [2.1, 5.8, 9.3, 15.2, 12.8, 45.6]
        benchmark_returns = [1.5, 4.2, 7.1, 11.8, 10.2, 38.9]
        alpha = [0.6, 1.6, 2.2, 3.4, 2.6, 6.7]
        
        data = {
            'Period': periods,
            'Portfolio Return (%)': portfolio_returns,
            'Benchmark Return (%)': benchmark_returns,
            'Alpha (%)': alpha
        }
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def render_trade_analysis(self):
        """Render trade analysis page"""
        st.header("💼 Trade Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Trade History")
            self.render_trade_history()
            
            st.subheader("Trade Performance Over Time")
            self.render_trade_performance()
        
        with col2:
            st.subheader("Trade Statistics")
            self.render_trade_statistics()
            
            st.subheader("Profit Distribution")
            self.render_profit_distribution()
    
    def render_trade_history(self):
        """Render detailed trade history"""
        # Sample trade data
        trades = []
        symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'HINDUNILVR']
        
        for i in range(50):
            trade = {
                'Date': f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                'Symbol': symbols[i % len(symbols)],
                'Action': 'BUY' if i % 3 != 0 else 'SELL',
                'Quantity': np.random.randint(10, 100),
                'Price': np.random.uniform(1000, 5000),
                'P&L': np.random.uniform(-5000, 8000)
            }
            trades.append(trade)
        
        df = pd.DataFrame(trades)
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_symbol = st.selectbox("Filter by Symbol", ['All'] + symbols)
        
        with col2:
            selected_action = st.selectbox("Filter by Action", ['All', 'BUY', 'SELL'])
        
        with col3:
            date_range = st.date_input("Date Range", [])
        
        # Apply filters
        if selected_symbol != 'All':
            df = df[df['Symbol'] == selected_symbol]
        
        if selected_action != 'All':
            df = df[df['Action'] == selected_action]
        
        st.dataframe(df, use_container_width=True, height=400)
    
    def render_trade_performance(self):
        """Render trade performance over time"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        cumulative_pnl = np.cumsum(np.random.normal(1000, 3000, len(dates)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_pnl,
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='#00cc96', width=3)
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative P&L (₹)",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_trade_statistics(self):
        """Render trade statistics"""
        stats = {
            'Total Trades': '142',
            'Win Rate': '64.2%',
            'Avg Win': '₹2,845',
            'Avg Loss': '₹-1,235',
            'Profit Factor': '2.35',
            'Avg Trade Duration': '5.2 days'
        }
        
        for key, value in stats.items():
            st.metric(key, value)
    
    def render_profit_distribution(self):
        """Render profit distribution"""
        profits = np.random.normal(500, 2000, 1000)
        
        fig = go.Figure(data=[
            go.Histogram(
                x=profits,
                nbinsx=30,
                marker_color='#1f77b4',
                opacity=0.7
            )
        ])
        
        fig.update_layout(
            title="Profit Distribution",
            xaxis_title="P&L (₹)",
            yaxis_title="Frequency",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_analysis(self):
        """Render risk analysis page"""
        st.header("🛡️ Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Value at Risk (VaR)")
            self.render_var_analysis()
            
            st.subheader("Stress Testing")
            self.render_stress_testing()
        
        with col2:
            st.subheader("Correlation Matrix")
            self.render_correlation_matrix()
            
            st.subheader("Risk Metrics")
            self.render_detailed_risk_metrics()
    
    def render_var_analysis(self):
        """Render VaR analysis"""
        # VaR at different confidence levels
        confidence_levels = [90, 95, 99]
        var_values = [-1.2, -2.1, -3.8]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[f"{cl}%" for cl in confidence_levels],
                y=var_values,
                marker_color=['#ff7f0e', '#ef553b', '#d62728']
            )
        ])
        
        fig.update_layout(
            title="Value at Risk by Confidence Level",
            xaxis_title="Confidence Level",
            yaxis_title="VaR (%)",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_stress_testing(self):
        """Render stress testing results"""
        scenarios = ['Market Crash', 'High Volatility', 'Rate Hike', 'Recession']
        impacts = [-15.2, -8.7, -6.3, -12.8]
        
        fig = go.Figure(data=[
            go.Bar(
                x=impacts,
                y=scenarios,
                orientation='h',
                marker_color='#ef553b'
            )
        ])
        
        fig.update_layout(
            title="Stress Test Scenarios",
            xaxis_title="Portfolio Impact (%)",
            yaxis_title="Scenario",
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_correlation_matrix(self):
        """Render correlation matrix"""
        symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR']
        correlations = np.random.uniform(-0.2, 0.8, (5, 5))
        np.fill_diagonal(correlations, 1.0)
        
        fig = go.Figure(data=go.Heatmap(
            z=correlations,
            x=symbols,
            y=symbols,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Stock Correlation Matrix",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_detailed_risk_metrics(self):
        """Render detailed risk metrics"""
        risk_data = {
            'Metric': ['Volatility', 'Beta', 'Alpha', 'Sharpe', 'Sortino', 'VaR (95%)', 'CVaR (95%)'],
            'Value': ['15.2%', '0.95', '2.3%', '1.85', '2.42', '-2.1%', '-3.2%']
        }
        
        df = pd.DataFrame(risk_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def render_reports(self):
        """Render reports page"""
        st.header("📋 Reports & Exports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Generate Reports")
            
            report_type = st.selectbox(
                "Select Report Type",
                ["Performance Report", "Risk Report", "Trade Analysis", "Comprehensive"]
            )
            
            date_range = st.date_input(
                "Report Period",
                [datetime(2023, 1, 1), datetime(2023, 12, 31)]
            )
            
            if st.button("📄 Generate Report", type="primary"):
                with st.spinner("Generating report..."):
                    # Simulate report generation
                    import time
                    time.sleep(2)
                    st.success("Report generated successfully!")
                    
                    # Show preview
                    st.subheader("Report Preview")
                    st.text_area("Report Content", self._generate_sample_report(), height=300)
        
        with col2:
            st.subheader("Export Data")
            
            export_format = st.radio(
                "Export Format",
                ["CSV", "Excel", "PDF", "JSON"]
            )
            
            data_types = st.multiselect(
                "Select Data to Export",
                ["Portfolio History", "Trade History", "Performance Metrics", "Risk Metrics"]
            )
            
            if st.button("💾 Export Data"):
                with st.spinner("Exporting data..."):
                    import time
                    time.sleep(1)
                    st.success(f"Data exported as {export_format}!")
    
    def _generate_sample_report(self):
        """Generate sample report content"""
        return """
ALGORITHMIC TRADING SYSTEM - PERFORMANCE REPORT
Generated on: 2023-12-15

EXECUTIVE SUMMARY:
- Total Return: +15.2%
- Sharpe Ratio: 1.85
- Max Drawdown: -8.3%
- Win Rate: 64.2%

PERFORMANCE METRICS:
Total Return: +15.2%
Annualized Return: +16.8%
Volatility: 15.2%
Sharpe Ratio: 1.85
Sortino Ratio: 2.42
Max Drawdown: -8.3%
Calmar Ratio: 2.02

RISK METRICS:
Beta: 0.95
Alpha: +2.3%
VaR (95%): -2.1%
CVaR (95%): -3.2%

TRADING ACTIVITY:
Total Trades: 142
Win Rate: 64.2%
Profit Factor: 2.35
Average Trade Duration: 5.2 days

CONCLUSION:
The strategy has demonstrated consistent outperformance with controlled risk.
Recommendation: Continue current strategy with regular monitoring.
"""
