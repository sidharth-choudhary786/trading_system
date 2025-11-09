# trading_system/dashboard/components/trade_journal.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class TradeJournal:
    """Trade journal and analysis components"""
    
    def __init__(self, trade_data: Dict):
        self.trade_data = trade_data
    
    def render_trade_overview(self):
        """Render trade overview statistics"""
        st.subheader("Trade Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        trades = self.trade_data.get('trades', [])
        
        with col1:
            total_trades = len(trades)
            st.metric("Total Trades", total_trades)
        
        with col2:
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col3:
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            st.metric("Total P&L", f"₹{total_pnl:,.0f}")
        
        with col4:
            avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
            st.metric("Avg Trade P&L", f"₹{avg_trade_pnl:,.0f}")
    
    def render_trade_history(self):
        """Render interactive trade history table"""
        st.subheader("Trade History")
        
        trades = self.trade_data.get('trades', [])
        
        if not trades:
            st.info("No trade history available")
            return
        
        # Create filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbols = list(set(t.get('symbol', '') for t in trades))
            selected_symbol = st.selectbox("Symbol", ["All"] + symbols)
        
        with col2:
            sides = list(set(t.get('side', '') for t in trades))
            selected_side = st.selectbox("Side", ["All"] + sides)
        
        with col3:
            date_range = st.date_input(
                "Date Range",
                [datetime.now() - timedelta(days=30), datetime.now()]
            )
        
        with col4:
            pnl_filter = st.selectbox("P&L Filter", ["All", "Profitable", "Loss Making"])
        
        # Filter trades
        filtered_trades = trades
        
        if selected_symbol != "All":
            filtered_trades = [t for t in filtered_trades if t.get('symbol') == selected_symbol]
        
        if selected_side != "All":
            filtered_trades = [t for t in filtered_trades if t.get('side') == selected_side]
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_trades = [
                t for t in filtered_trades 
                if start_date <= pd.to_datetime(t.get('timestamp')).date() <= end_date
            ]
        
        if pnl_filter == "Profitable":
            filtered_trades = [t for t in filtered_trades if t.get('pnl', 0) > 0]
        elif pnl_filter == "Loss Making":
            filtered_trades = [t for t in filtered_trades if t.get('pnl', 0) < 0]
        
        # Create dataframe
        trade_list = []
        for trade in filtered_trades:
            trade_list.append({
                'Date': pd.to_datetime(trade.get('timestamp')).strftime('%Y-%m-%d'),
                'Symbol': trade.get('symbol', ''),
                'Side': trade.get('side', ''),
                'Quantity': trade.get('quantity', 0),
                'Price': f"₹{trade.get('price', 0):.2f}",
                'Value': f"₹{trade.get('quantity', 0) * trade.get('price', 0):,.0f}",
                'P&L': f"₹{trade.get('pnl', 0):,.0f}",
                'Commission': f"₹{trade.get('commission', 0):.2f}",
                'Slippage': f"₹{trade.get('slippage', 0):.2f}"
            })
        
        df = pd.DataFrame(trade_list)
        
        if not df.empty:
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.info("No trades match the selected filters")
    
    def render_trade_analysis(self):
        """Render trade analysis charts"""
        st.subheader("Trade Analysis")
        
        tab1, tab2, tab3 = st.tabs(["P&L Over Time", "Trade Duration", "Symbol Performance"])
        
        with tab1:
            self.render_pnl_over_time()
        
        with tab2:
            self.render_trade_duration()
        
        with tab3:
            self.render_symbol_performance()
    
    def render_pnl_over_time(self):
        """Render P&L over time chart"""
        trades = self.trade_data.get('trades', [])
        
        if not trades:
            st.info("No trade data available")
            return
        
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', ''))
        
        dates = []
        cumulative_pnl = []
        current_pnl = 0
        
        for trade in sorted_trades:
            dates.append(pd.to_datetime(trade.get('timestamp')))
            current_pnl += trade.get('pnl', 0)
            cumulative_pnl.append(current_pnl)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_pnl,
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='#00cc96', width=3),
            marker=dict(size=4)
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Cumulative P&L Over Time",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L (₹)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_trade_duration(self):
        """Render trade duration analysis"""
        trades = self.trade_data.get('trades', [])
        
        if not trades:
            st.info("No trade data available")
            return
        
        # Calculate trade durations (simplified)
        durations = []
        for trade in trades:
            if trade.get('entry_time') and trade.get('exit_time'):
                entry = pd.to_datetime(trade['entry_time'])
                exit = pd.to_datetime(trade['exit_time'])
                duration = (exit - entry).days
                durations.append(duration)
        
        if not durations:
            st.info("No duration data available")
            return
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=durations,
            nbinsx=20,
            name='Trade Duration',
            marker_color='#ff7f0e'
        ))
        
        fig.update_layout(
            title="Trade Duration Distribution",
            xaxis_title="Duration (Days)",
            yaxis_title="Number of Trades",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_symbol_performance(self):
        """Render symbol-wise performance"""
        trades = self.trade_data.get('trades', [])
        
        if not trades:
            st.info("No trade data available")
            return
        
        # Group trades by symbol
        symbol_data = {}
        for trade in trades:
            symbol = trade.get('symbol', 'Unknown')
            if symbol not in symbol_data:
                symbol_data[symbol] = {
                    'trades': 0,
                    'total_pnl': 0,
                    'winning_trades': 0
                }
            
            symbol_data[symbol]['trades'] += 1
            symbol_data[symbol]['total_pnl'] += trade.get('pnl', 0)
            if trade.get('pnl', 0) > 0:
                symbol_data[symbol]['winning_trades'] += 1
        
        # Create bar chart
        symbols = list(symbol_data.keys())
        total_pnl = [symbol_data[s]['total_pnl'] for s in symbols]
        win_rates = [symbol_data[s]['winning_trades'] / symbol_data[s]['trades'] * 100 
                    for s in symbols]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # P&L bars
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=total_pnl,
                name="Total P&L",
                marker_color=['#00cc96' if pnl >= 0 else '#ef553b' for pnl in total_pnl]
            ),
            secondary_y=False
        )
        
        # Win rate line
        fig.add_trace(
            go.Scatter(
                x=symbols,
                y=win_rates,
                name="Win Rate",
                line=dict(color='#1f77b4', width=3),
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title="Symbol Performance Analysis",
            xaxis_title="Symbol",
            template="plotly_white",
            height=400
        )
        
        fig.update_yaxes(title_text="Total P&L (₹)", secondary_y=False)
        fig.update_yaxes(title_text="Win Rate (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
