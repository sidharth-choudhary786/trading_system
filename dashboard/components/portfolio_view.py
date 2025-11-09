# trading_system/dashboard/components/portfolio_view.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

class PortfolioView:
    """Portfolio visualization components"""
    
    def __init__(self, portfolio_data: Dict):
        self.portfolio_data = portfolio_data
    
    def render_portfolio_summary(self):
        """Render portfolio summary cards"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Value",
                f"₹{self.portfolio_data.get('total_value', 0):,.0f}",
                delta=f"₹{self.portfolio_data.get('daily_change', 0):,.0f}"
            )
        
        with col2:
            st.metric(
                "Cash Balance", 
                f"₹{self.portfolio_data.get('cash', 0):,.0f}",
                f"{self.portfolio_data.get('cash_percentage', 0):.1f}%"
            )
        
        with col3:
            st.metric(
                "Today's P&L",
                f"₹{self.portfolio_data.get('daily_pnl', 0):,.0f}",
                f"{self.portfolio_data.get('daily_return', 0):.2f}%",
                delta_color="normal" if self.portfolio_data.get('daily_pnl', 0) >= 0 else "inverse"
            )
        
        with col4:
            st.metric(
                "Unrealized P&L",
                f"₹{self.portfolio_data.get('unrealized_pnl', 0):,.0f}",
                f"{self.portfolio_data.get('unrealized_return', 0):.2f}%",
                delta_color="normal" if self.portfolio_data.get('unrealized_pnl', 0) >= 0 else "inverse"
            )
    
    def render_holdings_table(self):
        """Render detailed holdings table"""
        st.subheader("Current Holdings")
        
        holdings = self.portfolio_data.get('holdings', [])
        
        if not holdings:
            st.info("No current holdings")
            return
        
        # Create holdings dataframe
        data = []
        for holding in holdings:
            data.append({
                'Symbol': holding.get('symbol', ''),
                'Quantity': holding.get('quantity', 0),
                'Avg Price': f"₹{holding.get('avg_price', 0):.2f}",
                'Current Price': f"₹{holding.get('current_price', 0):.2f}",
                'Market Value': f"₹{holding.get('market_value', 0):,.0f}",
                'P&L': f"₹{holding.get('pnl', 0):,.0f}",
                'P&L %': f"{holding.get('pnl_percentage', 0):.2f}%",
                'Weight': f"{holding.get('weight', 0):.1f}%"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def render_allocation_charts(self):
        """Render allocation charts"""
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_sector_allocation()
        
        with col2:
            self.render_stock_allocation()
    
    def render_sector_allocation(self):
        """Render sector allocation pie chart"""
        sector_data = self.portfolio_data.get('sector_allocation', {})
        
        if not sector_data:
            st.info("No sector allocation data")
            return
        
        sectors = list(sector_data.keys())
        weights = list(sector_data.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=sectors,
            values=weights,
            hole=0.4,
            textinfo='label+percent',
            insidetextorientation='radial'
        )])
        
        fig.update_layout(
            title="Sector Allocation",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_stock_allocation(self):
        """Render stock allocation bar chart"""
        holdings = self.portfolio_data.get('holdings', [])
        
        if not holdings:
            st.info("No holdings data")
            return
        
        symbols = [h.get('symbol', '') for h in holdings]
        weights = [h.get('weight', 0) for h in holdings]
        
        # Sort by weight
        sorted_data = sorted(zip(symbols, weights), key=lambda x: x[1], reverse=True)
        symbols, weights = zip(*sorted_data) if sorted_data else ([], [])
        
        fig = go.Figure(data=[
            go.Bar(
                x=weights,
                y=symbols,
                orientation='h',
                marker_color='#1f77b4'
            )
        ])
        
        fig.update_layout(
            title="Stock Allocation",
            xaxis_title="Weight (%)",
            yaxis_title="Stocks",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_chart(self):
        """Render portfolio performance chart"""
        performance_data = self.portfolio_data.get('performance_history', [])
        
        if not performance_data:
            st.info("No performance history data")
            return
        
        dates = [pd.to_datetime(p['date']) for p in performance_data]
        portfolio_values = [p['portfolio_value'] for p in performance_data]
        benchmark_values = [p.get('benchmark_value', p['portfolio_value'] * 0.95) for p in performance_data]
        
        fig = go.Figure()
        
        # Portfolio line
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio',
            line=dict(color='#1f77b4', width=3)
        ))
        
        # Benchmark line
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_values,
            mode='lines',
            name='Benchmark',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Portfolio Performance vs Benchmark",
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
