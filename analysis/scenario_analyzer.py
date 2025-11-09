# trading_system/analysis/scenario_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class ScenarioAnalyzer:
    """
    Scenario analysis and stress testing
    """
    
    def __init__(self, config: Dict):
        self.config = config
    
    def monte_carlo_simulation(
        self, 
        expected_return: float, 
        volatility: float,
        time_horizon: int = 252,
        num_simulations: int = 10000,
        initial_portfolio: float = 100000
    ) -> Dict:
        """
        Run Monte Carlo simulation for portfolio returns
        
        Args:
            expected_return: Expected annual return
            volatility: Annual volatility
            time_horizon: Time horizon in days
            num_simulations: Number of simulations
            initial_portfolio: Initial portfolio value
        """
        print(f"Running Monte Carlo simulation with {num_simulations} simulations...")
        
        # Daily returns parameters
        daily_return = expected_return / 252
        daily_volatility = volatility / np.sqrt(252)
        
        # Generate random returns
        random_returns = np.random.normal(
            daily_return, 
            daily_volatility, 
            (num_simulations, time_horizon)
        )
        
        # Calculate portfolio paths
        portfolio_paths = np.zeros((num_simulations, time_horizon + 1))
        portfolio_paths[:, 0] = initial_portfolio
        
        for i in range(time_horizon):
            portfolio_paths[:, i + 1] = portfolio_paths[:, i] * (1 + random_returns[:, i])
        
        # Calculate statistics
        final_values = portfolio_paths[:, -1]
        
        results = {
            'expected_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'var_95': np.percentile(final_values, 5),
            'cvar_95': final_values[final_values <= np.percentile(final_values, 5)].mean(),
            'best_case': np.max(final_values),
            'worst_case': np.min(final_values),
            'probability_of_loss': np.mean(final_values < initial_portfolio),
            'portfolio_paths': portfolio_paths,
            'final_values': final_values
        }
        
        return results
    
    def stress_test(
        self,
        portfolio_returns: pd.Series,
        stress_scenarios: Dict[str, Dict]
    ) -> Dict:
        """
        Stress test portfolio under different scenarios
        
        Args:
            portfolio_returns: Historical portfolio returns
            stress_scenarios: Dictionary of stress scenarios
                Example: {
                    'market_crash': {'return_shock': -0.20, 'volatility_increase': 2.0},
                    'high_volatility': {'return_shock': -0.05, 'volatility_increase': 3.0}
                }
        """
        results = {}
        
        base_return = portfolio_returns.mean()
        base_volatility = portfolio_returns.std()
        
        for scenario_name, scenario_params in stress_scenarios.items():
            return_shock = scenario_params.get('return_shock', 0)
            vol_multiplier = scenario_params.get('volatility_increase', 1.0)
            
            # Apply stress to returns
            stressed_return = base_return + return_shock / 252  # Convert to daily
            stressed_volatility = base_volatility * vol_multiplier
            
            # Run Monte Carlo with stressed parameters
            scenario_results = self.monte_carlo_simulation(
                expected_return=stressed_return * 252,
                volatility=stressed_volatility * np.sqrt(252),
                num_simulations=5000  # Fewer sims for speed
            )
            
            results[scenario_name] = scenario_results
        
        return results
    
    def what_if_analysis(
        self,
        base_portfolio_value: float,
        what_if_scenarios: Dict[str, Dict]
    ) -> Dict:
        """
        What-if analysis for different market conditions
        
        Args:
            base_portfolio_value: Current portfolio value
            what_if_scenarios: Dictionary of what-if scenarios
                Example: {
                    'interest_rate_rise': {'equity_return_impact': -0.15},
                    'recession': {'equity_return_impact': -0.25, 'duration': 180}
                }
        """
        results = {}
        
        for scenario_name, scenario_params in what_if_scenarios.items():
            return_impact = scenario_params.get('equity_return_impact', 0)
            duration = scenario_params.get('duration', 90)  # days
            
            # Calculate impact
            daily_impact = return_impact / duration
            scenario_value = base_portfolio_value * (1 + return_impact)
            
            results[scenario_name] = {
                'scenario_impact_pct': return_impact * 100,
                'projected_portfolio_value': scenario_value,
                'absolute_change': scenario_value - base_portfolio_value,
                'duration_days': duration,
                'daily_impact_pct': daily_impact * 100
            }
        
        return results
    
    def correlation_analysis(
        self, 
        returns_data: pd.DataFrame,
        shock_assets: List[str],
        shock_magnitude: float = -0.10
    ) -> Dict:
        """
        Analyze impact of correlation changes
        
        Args:
            returns_data: DataFrame with asset returns
            shock_assets: Assets to shock
            shock_magnitude: Shock magnitude
        """
        # Calculate current correlations
        current_corr = returns_data.corr()
        
        # Create shocked correlation matrix
        shocked_corr = current_corr.copy()
        
        for asset in shock_assets:
            if asset in shocked_corr.columns:
                # Increase correlations with shocked assets
                for other_asset in shocked_corr.columns:
                    if other_asset != asset:
                        current_corr_val = shocked_corr.loc[asset, other_asset]
                        shocked_corr.loc[asset, other_asset] = min(0.9, current_corr_val + 0.3)
                        shocked_corr.loc[other_asset, asset] = shocked_corr.loc[asset, other_asset]
        
        analysis_results = {
            'current_correlations': current_corr,
            'shocked_correlations': shocked_corr,
            'correlation_changes': shocked_corr - current_corr,
            'shock_assets': shock_assets,
            'shock_magnitude': shock_magnitude
        }
        
        return analysis_results
