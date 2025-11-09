# trading_system/analysis/reporting.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ReportGenerator:
    """
    Generate comprehensive trading reports and visualizations
    """
    
    def __init__(self):
        self.figures = {}
    
    def generate_backtest_report(
        self, 
        backtest_results: Dict,
        strategy_name: str = "Trading Strategy"
    ) -> Dict:
        """
        Generate comprehensive backtest report
        """
        report = {
            'strategy_name': strategy_name,
            'generated_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'performance_metrics': backtest_results,
            'charts': self._generate_performance_charts(backtest_results),
            'trades_analysis': self._analyze_trades(backtest_results.get('trade_history', [])),
            'risk_analysis': self._analyze_risk(backtest_results)
        }
        
        return report
    
    def _generate_performance_charts(self, results: Dict) -> Dict:
        """
        Generate performance charts
        """
        charts = {}
        
        # Portfolio value over time
        if 'portfolio_history' in results:
            portfolio_data = results['portfolio_history']
            dates = [p['date'] for p in portfolio_data]
            values = [p['portfolio_value'] for p in portfolio_data]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title='Portfolio Value Over Time',
                xaxis_title='Date',
                yaxis_title='Portfolio Value (₹)',
                template='plotly_white'
            )
            
            charts['portfolio_value'] = fig
        
        # Drawdown chart
        if 'portfolio_history' in results:
            portfolio_values = [p['portfolio_value'] for p in portfolio_data]
            portfolio_series = pd.Series(portfolio_values, index=dates)
            rolling_max = portfolio_series.expanding().max()
            drawdown = (portfolio_series - rolling_max) / rolling_max
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=2),
                fill='tozeroy'
            ))
            
            fig.update_layout(
                title='Portfolio Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                template='plotly_white',
                yaxis=dict(tickformat='.1%')
            )
            
            charts['drawdown'] = fig
        
        # Returns distribution
        if 'daily_returns' in results and results['daily_returns']:
            returns = pd.Series(results['daily_returns'])
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns Distribution',
                marker_color='green',
                opacity=0.7
            ))
            
            fig.update_layout(
                title='Daily Returns Distribution',
                xaxis_title='Daily Return',
                yaxis_title='Frequency',
                template='plotly_white',
                xaxis=dict(tickformat='.1%')
            )
            
            charts['returns_distribution'] = fig
        
        return charts
    
    def _analyze_trades(self, trade_history: List) -> Dict:
        """
        Analyze trade performance
        """
        if not trade_history:
            return {}
        
        trades_df = pd.DataFrame([t.__dict__ for t in trade_history])
        
        analysis = {
            'total_trades': len(trade_history),
            'buy_trades': len([t for t in trade_history if t.side.value == 'buy']),
            'sell_trades': len([t for t in trade_history if t.side.value == 'sell']),
            'avg_trade_duration': 0,  # Would need timestamps
            'win_rate': 0,  # Would need P&L calculation
            'avg_winning_trade': 0,
            'avg_losing_trade': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'profit_factor': 0
        }
        
        return analysis
    
    def _analyze_risk(self, results: Dict) -> Dict:
        """
        Analyze risk metrics
        """
        risk_analysis = {}
        
        if 'max_drawdown' in results:
            risk_analysis['max_drawdown'] = results['max_drawdown']
        
        if 'daily_returns' in results and results['daily_returns']:
            returns = pd.Series(results['daily_returns'])
            risk_analysis['volatility'] = returns.std() * np.sqrt(252)
            risk_analysis['var_95'] = np.percentile(returns, 5)
            risk_analysis['cvar_95'] = returns[returns <= risk_analysis['var_95']].mean()
            risk_analysis['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
        
        return risk_analysis
    
    def generate_html_report(self, report_data: Dict) -> str:
        """
        Generate HTML formatted report
        """
        html_content = []
        html_content.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f4f4f4; padding: 20px; border-radius: 5px; }
                .metric { margin: 10px 0; padding: 10px; background: #f9f9f9; border-left: 4px solid #007bff; }
                .chart { margin: 20px 0; }
            </style>
        </head>
        <body>
        """)
        
        # Header
        html_content.append(f"""
        <div class="header">
            <h1>{report_data['strategy_name']} - Performance Report</h1>
            <p>Generated on: {report_data['generated_date']}</p>
        </div>
        """)
        
        # Performance Metrics
        html_content.append("<h2>Performance Metrics</h2>")
        metrics = report_data['performance_metrics']
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if 'return' in key or 'ratio' in key or 'drawdown' in key:
                    display_value = f"{value:.2%}"
                else:
                    display_value = f"{value:.2f}"
                html_content.append(f"""
                <div class="metric">
                    <strong>{key.replace('_', ' ').title()}:</strong> {display_value}
                </div>
                """)
        
        # Charts would be embedded here in actual implementation
        
        html_content.append("""
        </body>
        </html>
        """)
        
        return "\n".join(html_content)
    
    def save_report(self, report_data: Dict, filepath: str):
        """
        Save report to file
        """
        html_report = self.generate_html_report(report_data)
        
        with open(filepath, 'w', encoding='utf-8') f:
            f.write(html_report)
        
        print(f"Report saved to: {filepath}")
