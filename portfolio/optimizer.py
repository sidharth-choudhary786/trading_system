# trading_system/portfolio/optimizer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy.optimize import minimize
import cvxpy as cp
import warnings
warnings.filterwarnings('ignore')

from ..core.exceptions import PortfolioError

class PortfolioOptimizer:
    """
    Portfolio optimization using various methods (Sharpe ratio maximization, risk parity, etc.)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Optimization configuration
        self.optimization_method = config.get('optimization_method', 'sharpe_maximization')
        self.risk_free_rate = config.get('risk_free_rate', 0.0)
        self.max_iterations = config.get('max_iterations', 1000)
        
        # Constraints
        self.constraints = config.get('constraints', {})
        self.max_position_size = self.constraints.get('max_position_size', 0.1)  # 10%
        self.max_sector_exposure = self.constraints.get('max_sector_exposure', 0.3)  # 30%
        self.min_diversification = self.constraints.get('min_diversification', 5)  # min 5 assets
        
        self.logger.info("Portfolio Optimizer initialized")
    
    def optimize(
        self, 
        expected_returns: pd.Series, 
        covariance_matrix: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            current_weights: Current portfolio weights for turnover control
            
        Returns:
            Dictionary of optimized weights
        """
        try:
            # Validate inputs
            self._validate_optimization_inputs(expected_returns, covariance_matrix)
            
            # Get symbols
            symbols = expected_returns.index.tolist()
            n_assets = len(symbols)
            
            if n_assets < self.min_diversification:
                self.logger.warning(f"Only {n_assets} assets available, less than minimum {self.min_diversification}")
            
            # Convert to numpy arrays for optimization
            mu = expected_returns.values
            Sigma = covariance_matrix.values
            
            # Run optimization based on selected method
            if self.optimization_method == 'sharpe_maximization':
                weights = self._maximize_sharpe_ratio(mu, Sigma, symbols)
            elif self.optimization_method == 'min_variance':
                weights = self._minimize_variance(mu, Sigma, symbols)
            elif self.optimization_method == 'risk_parity':
                weights = self._risk_parity_optimization(mu, Sigma, symbols)
            elif self.optimization_method == 'max_return':
                weights = self._maximize_return(mu, Sigma, symbols)
            elif self.optimization_method == 'equal_weight':
                weights = self._equal_weight_optimization(symbols)
            else:
                self.logger.warning(f"Unknown optimization method: {self.optimization_method}. Using Sharpe maximization.")
                weights = self._maximize_sharpe_ratio(mu, Sigma, symbols)
            
            # Apply constraints and normalize
            weights = self._apply_constraints(weights, symbols, current_weights)
            
            # Convert to dictionary
            optimized_weights = {symbol: weight for symbol, weight in zip(symbols, weights)}
            
            self.logger.info("Portfolio optimization completed successfully")
            return optimized_weights
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            # Fallback to equal weighting
            return self._equal_weight_fallback(expected_returns.index.tolist())
    
    def _maximize_sharpe_ratio(self, mu: np.ndarray, Sigma: np.ndarray, symbols: List[str]) -> np.ndarray:
        """Maximize Sharpe ratio using quadratic programming"""
        n_assets = len(symbols)
        
        def sharpe_objective(weights):
            portfolio_return = np.dot(weights, mu)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
            # Negative Sharpe ratio for minimization
            return - (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # weights sum to 1
        ]
        
        # Bounds
        bounds = [(0.0, self.max_position_size) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            sharpe_objective, x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': self.max_iterations}
        )
        
        if result.success:
            return result.x
        else:
            raise PortfolioError(f"Sharpe ratio optimization failed: {result.message}")
    
    def _minimize_variance(self, mu: np.ndarray, Sigma: np.ndarray, symbols: List[str]) -> np.ndarray:
        """Minimize portfolio variance"""
        n_assets = len(symbols)
        
        def variance_objective(weights):
            return np.dot(weights.T, np.dot(Sigma, weights))
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
        ]
        
        bounds = [(0.0, self.max_position_size) for _ in range(n_assets)]
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        result = minimize(
            variance_objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations}
        )
        
        if result.success:
            return result.x
        else:
            raise PortfolioError(f"Variance minimization failed: {result.message}")
    
    def _risk_parity_optimization(self, mu: np.ndarray, Sigma: np.ndarray, symbols: List[str]) -> np.ndarray:
        """Risk parity optimization - equal risk contribution"""
        n_assets = len(symbols)
        
        def risk_parity_objective(weights):
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
            marginal_risk = np.dot(Sigma, weights) / portfolio_risk
            risk_contributions = weights * marginal_risk
            
            # Objective: minimize difference between risk contributions
            target_risk_contribution = portfolio_risk / n_assets
            deviation = np.sum((risk_contributions - target_risk_contribution) ** 2)
            return deviation
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
        ]
        
        bounds = [(0.0, self.max_position_size) for _ in range(n_assets)]
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        result = minimize(
            risk_parity_objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations}
        )
        
        if result.success:
            return result.x
        else:
            raise PortfolioError(f"Risk parity optimization failed: {result.message}")
    
    def _maximize_return(self, mu: np.ndarray, Sigma: np.ndarray, symbols: List[str]) -> np.ndarray:
        """Maximize expected return with risk constraint"""
        n_assets = len(symbols)
        
        # Use convex optimization for better stability
        weights = cp.Variable(n_assets)
        
        # Objective: maximize return
        objective = cp.Maximize(weights @ mu)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            weights <= self.max_position_size,
            cp.quad_form(weights, Sigma) <= self.constraints.get('max_risk', 0.04)  # Max 20% annual volatility
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == cp.OPTIMAL:
            return weights.value
        else:
            raise PortfolioError("Return maximization failed")
    
    def _equal_weight_optimization(self, symbols: List[str]) -> np.ndarray:
        """Equal weight portfolio"""
        n_assets = len(symbols)
        return np.array([1.0 / n_assets] * n_assets)
    
    def _apply_constraints(
        self, 
        weights: np.ndarray, 
        symbols: List[str], 
        current_weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Apply portfolio constraints"""
        # Ensure weights are non-negative and normalized
        weights = np.maximum(weights, 0.0)
        weights = weights / np.sum(weights)
        
        # Apply position size constraints
        max_weight = self.max_position_size
        weights = np.minimum(weights, max_weight)
        weights = weights / np.sum(weights)  # Renormalize
        
        # Apply turnover constraints if current weights provided
        if current_weights:
            current_weights_array = np.array([current_weights.get(symbol, 0.0) for symbol in symbols])
            max_turnover = self.constraints.get('max_turnover', 0.2)  # 20% max turnover
            
            # Calculate required trades
            trades = np.abs(weights - current_weights_array)
            total_turnover = np.sum(trades) / 2  # Divide by 2 because each trade affects both buy and sell
            
            if total_turnover > max_turnover:
                # Scale down changes to meet turnover constraint
                scale_factor = max_turnover / total_turnover
                weights = current_weights_array + (weights - current_weights_array) * scale_factor
                weights = weights / np.sum(weights)  # Renormalize
        
        return weights
    
    def _validate_optimization_inputs(self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame):
        """Validate optimization inputs"""
        if len(expected_returns) == 0:
            raise PortfolioError("Expected returns cannot be empty")
        
        if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
            raise PortfolioError("Covariance matrix must be square")
        
        if len(expected_returns) != covariance_matrix.shape[0]:
            raise PortfolioError("Expected returns and covariance matrix dimensions must match")
        
        # Check for positive definiteness
        try:
            np.linalg.cholesky(covariance_matrix.values)
        except np.linalg.LinAlgError:
            self.logger.warning("Covariance matrix is not positive definite, adding regularization")
            # Add small regularization to make matrix positive definite
            n_assets = covariance_matrix.shape[0]
            covariance_matrix.values += np.eye(n_assets) * 1e-6
    
    def _equal_weight_fallback(self, symbols: List[str]) -> Dict[str, float]:
        """Fallback to equal weights if optimization fails"""
        n_assets = len(symbols)
        equal_weight = 1.0 / n_assets
        return {symbol: equal_weight for symbol in symbols}
    
    def calculate_efficient_frontier(
        self, 
        expected_returns: pd.Series, 
        covariance_matrix: pd.DataFrame,
        target_returns: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier
        
        Args:
            expected_returns: Expected returns for assets
            covariance_matrix: Covariance matrix
            target_returns: List of target returns for frontier
            
        Returns:
            DataFrame with efficient frontier points
        """
        if target_returns is None:
            min_return = expected_returns.min()
            max_return = expected_returns.max()
            target_returns = np.linspace(min_return, max_return, 50)
        
        frontier_data = []
        symbols = expected_returns.index.tolist()
        n_assets = len(symbols)
        mu = expected_returns.values
        Sigma = covariance_matrix.values
        
        for target_return in target_returns:
            try:
                # Minimize variance for given target return
                def variance_objective(weights):
                    return np.dot(weights.T, np.dot(Sigma, weights))
                
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                    {'type': 'eq', 'fun': lambda x: np.dot(x, mu) - target_return}
                ]
                
                bounds = [(0.0, self.max_position_size) for _ in range(n_assets)]
                x0 = np.array([1.0 / n_assets] * n_assets)
                
                result = minimize(
                    variance_objective, x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                if result.success:
                    portfolio_risk = np.sqrt(result.fun)
                    frontier_data.append({
                        'target_return': target_return,
                        'portfolio_risk': portfolio_risk,
                        'sharpe_ratio': (target_return - self.risk_free_rate) / portfolio_risk,
                        'weights': result.x
                    })
                    
            except Exception as e:
                self.logger.warning(f"Failed to calculate efficient frontier point for return {target_return}: {e}")
                continue
        
        return pd.DataFrame(frontier_data)
    
    def get_optimization_report(
        self, 
        expected_returns: pd.Series, 
        covariance_matrix: pd.DataFrame,
        optimized_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate optimization report"""
        symbols = expected_returns.index.tolist()
        weights_array = np.array([optimized_weights.get(symbol, 0.0) for symbol in symbols])
        mu = expected_returns.values
        Sigma = covariance_matrix.values
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights_array, mu)
        portfolio_risk = np.sqrt(np.dot(weights_array.T, np.dot(Sigma, weights_array)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        # Calculate risk contributions
        marginal_risk = np.dot(Sigma, weights_array) / portfolio_risk
        risk_contributions = weights_array * marginal_risk
        risk_contribution_pct = risk_contributions / portfolio_risk * 100
        
        report = {
            'optimization_method': self.optimization_method,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'number_of_assets': len(symbols),
            'effective_number_assets': 1 / np.sum(weights_array ** 2),  # Herfindahl index inverse
            'max_weight': np.max(weights_array),
            'min_weight': np.min(weights_array[weights_array > 0]) if np.any(weights_array > 0) else 0,
            'risk_contributions': {
                symbol: {
                    'weight': optimized_weights[symbol],
                    'risk_contribution': risk_contribution_pct[i],
                    'marginal_risk': marginal_risk[i]
                }
                for i, symbol in enumerate(symbols) if optimized_weights[symbol] > 0
            }
        }
        
        return report
