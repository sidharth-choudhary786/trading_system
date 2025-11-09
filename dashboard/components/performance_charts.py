# trading_system/dashboard/components/performance_charts.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class PerformanceCharts:
    """Performance visualization components"""
    
    def __init__(self, performance_data: Dict):
        self.performance_data = performance_data
    
    def render_returns_comparison(self):
        """Render returns comparison charts"""
        st.subheader("Returns Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Cumulative Returns", "Rolling Returns", "Returns Distribution"])
        
        with tab1:
            self.render_cumulative_returns()
        
        with tab2:
            self.render_rolling_returns()
        
        with tab3:
            self.render_returns_distribution()
    
    def render_cumulative_returns(self):
        """Render cumulative returns chart"""
        returns_data = self.performance_data.get('returns_history', [])
        
        if not returns_data:
            st.info("No returns data available")
            return
        
        dates = [pd.to_datetime(r['date']) for r in returns_data]
        portfolio_returns = [r['portfolio_return'] for r in returns_data]
        benchmark_returns = [r.get('benchmark_return', 0) for r in returns_data]
        
        # Calculate cumulative returns
        portfolio_cumulative = (1 + pd.Series(portfolio_returns)).cumprod()
        benchmark_cumulative = (1 + pd.Series(benchmark_returns)).cumprod()
        
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
            name='Benchmark',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_rolling_returns(self):
        """Render rolling returns chart"""
        returns_data = self.performance_data.get('returns_history', [])
        
        if not returns_data:
            st.info("No returns data available")
            return
        
        dates = [pd.to_datetime(r['date']) for r in returns_data]
        portfolio_returns = pd.Series([r['portfolio_return'] for r in returns_data])
        
        # Calculate rolling returns for different periods
        rolling_30d = portfolio_returns.rolling(30).mean() * 252 * 100
        rolling_90d = portfolio_returns.rolling(90).mean() * 252 * 100
        rolling_252d = portfolio_returns.rolling(252).mean() * 252 * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates[29:],
            y=rolling_30d[29:],
            mode='lines',
            name='30-day Rolling',
            line=dict(color='#2ca02c', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates[89:],
            y=rolling_90d[89:],
            mode='lines',
            name='90-day Rolling',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates[251:],
            y=rolling_252d[251:],
            mode='lines',
            name='252-day Rolling',
            line=dict(color='#d62728', width=2)
        ))
        
        fig.update_layout(
            title="Rolling Annualized Returns",
            xaxis_title="Date",
            yaxis_title="Annualized Return (%)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_returns_distribution(self):
        """Render returns distribution histogram"""
        returns_data = self.performance_data.get('returns_history', [])
        
        if not returns_data:
            st.info("No returns data available")
            return
        
        portfolio_returns = [r['portfolio_return'] * 100 for r in returns_data]  # Convert to percentage
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=portfolio_returns,
            nbinsx=50,
            name='Returns Distribution',
            marker_color='#1f77b4',
            opacity=0.7
        ))
        
        # Add mean line
        mean_return = np.mean(portfolio_returns)
        fig.add_vline(
            x=mean_return, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Mean: {mean_return:.2f}%"
        )
        
        fig.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_drawdown_analysis(self):
        """Render drawdown analysis"""
        st.subheader("Drawdown Analysis")
        
        performance_history = self.performance_data.get('performance_history', [])
        
        if not performance_history:
            st.info("No performance history data")
            return
        
        dates = [pd.to_datetime(p['date']) for p in performance_history]
        portfolio_values = [p['portfolio_value'] for p in performance_history]
        
        # Calculate drawdown
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
        
        # Highlight maximum drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        fig.add_annotation(
            x=max_dd_date,
            y=max_dd,
            text=f"Max Drawdown: {max_dd:.1f}%",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40
        )
        
        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_metrics(self):
        """Render detailed performance metrics table"""
        st.subheader("Performance Metrics")
        
        metrics = self.performance_data.get('metrics', {})
        
        if not metrics:
            st.info("No performance metrics available")
            return
        
        # Create metrics table
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
            st.metric("Annualized Return", f"{metrics.get('annualized_return', 0):.2f}%")
            st.metric("Volatility", f"{metrics.get('volatility', 0):.2f}%")
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
            st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
        
        with col2:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
            st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}")
            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2f}%")
            st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
            st.metric("Alpha", f"{metrics.get('alpha', 0):.2f}%")
