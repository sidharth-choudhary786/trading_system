# trading_system/models/walk_forward.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .model_manager import ModelManager
from ..core.exceptions import ModelError

class WalkForwardAnalyzer:
    """
    Walk-forward analysis for time series models
    Implements rolling window training and testing
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Walk-forward configuration
        wf_config = config.get('walk_forward', {})
        self.window_size = wf_config.get('window_size', 252)  # 1 year of trading days
        self.step_size = wf_config.get('step_size', 21)       # 1 month step
        self.min_training_period = wf_config.get('min_training_period', 100)
        
        # Results storage
        self.walk_forward_results = {}
        self.model_performance_history = {}
        
        self.logger.info("WalkForward Analyzer initialized")
    
    def run_walk_forward_analysis(
        self,
        data: pd.DataFrame,
        model_manager: ModelManager,
        target_column: str = 'target',
        models: Optional[List[str]] = None
    ) -> Dict:
        """
        Run walk-forward analysis on data
        
        Args:
            data: Time series data with features and target
            model_manager: ModelManager instance
            target_column: Name of target column
            models: List of model names to analyze
            
        Returns:
            Dictionary with walk-forward results
        """
        if 'date' not in data.columns:
            raise ModelError("Data must contain 'date' column for walk-forward analysis")
        
        # Ensure data is sorted by date
        data = data.sort_values('date').reset_index(drop=True)
        dates = pd.to_datetime(data['date'])
        
        # Create walk-forward windows
        windows = self._create_walk_forward_windows(dates)
        
        if not windows:
            raise ModelError("No valid walk-forward windows created")
        
        self.logger.info(f"Created {len(windows)} walk-forward windows")
        
        # Initialize results storage
        self.walk_forward_results = {
            'windows': [],
            'model_performance': {},
            'predictions': {},
            'summary': {}
        }
        
        # Run analysis for each window
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            self.logger.info(f"Processing window {i+1}/{len(windows)}: "
                           f"Train {train_start.date()} to {train_end.date()}, "
                           f"Test {test_start.date()} to {test_end.date()}")
            
            try:
                window_results = self._process_window(
                    data, dates, train_start, train_end, test_start, test_end,
                    model_manager, target_column, models, i
                )
                
                self.walk_forward_results['windows'].append(window_results)
                
            except Exception as e:
                self.logger.error(f"Error processing window {i+1}: {e}")
                continue
        
        # Generate summary statistics
        self._generate_summary_statistics()
        
        return self.walk_forward_results
    
    def _create_walk_forward_windows(self, dates: pd.Series) -> List[Tuple]:
        """Create walk-forward training and testing windows"""
        windows = []
        start_date = dates.min()
        end_date = dates.max()
        
        current_start = start_date
        
        while current_start < end_date:
            # Training period end
            train_end = current_start + timedelta(days=self.window_size)
            
            # Testing period
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.step_size - 1)
            
            # Ensure we have enough data
            if train_end > end_date or test_end > end_date:
                break
            
            # Check if we have sufficient training data
            train_dates = dates[(dates >= current_start) & (dates <= train_end)]
            if len(train_dates) < self.min_training_period:
                current_start += timedelta(days=self.step_size)
                continue
            
            windows.append((current_start, train_end, test_start, test_end))
            current_start += timedelta(days=self.step_size)
        
        return windows
    
    def _process_window(
        self,
        data: pd.DataFrame,
        dates: pd.Series,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
        model_manager: ModelManager,
        target_column: str,
        models: Optional[List[str]],
        window_index: int
    ) -> Dict:
        """Process a single walk-forward window"""
        # Split data into train and test
        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start) & (dates <= test_end)
        
        train_data = data[train_mask].copy()
        test_data = data[test_mask].copy()
        
        if train_data.empty or test_data.empty:
            raise ModelError("Empty train or test data in window")
        
        # Prepare features and target
        X_train = train_data.drop(columns=[target_column, 'date'], errors='ignore')
        y_train = train_data[target_column]
        
        X_test = test_data.drop(columns=[target_column, 'date'], errors='ignore')
        y_test = test_data[target_column]
        
        # Train models
        training_results = model_manager.train_models(X_train, y_train, models)
        
        # Evaluate models
        evaluation_results = model_manager.evaluate_models(X_test, y_test, models)
        
        # Get predictions
        predictions = {}
        for model_name in models or model_manager.active_models:
            if model_name in model_manager.models and model_manager.models[model_name].is_trained:
                try:
                    model = model_manager.models[model_name]
                    X_processed = model.prepare_features(X_test)
                    pred = model.predict(X_processed)
                    predictions[model_name] = pred.tolist()
                except Exception as e:
                    self.logger.warning(f"Error getting predictions from {model_name}: {e}")
        
        # Store window results
        window_results = {
            'window_index': window_index,
            'train_start': train_start.isoformat(),
            'train_end': train_end.isoformat(),
            'test_start': test_start.isoformat(),
            'test_end': test_end.isoformat(),
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'predictions': predictions
        }
        
        # Update performance history
        for model_name, metrics in evaluation_results.items():
            if model_name not in self.model_performance_history:
                self.model_performance_history[model_name] = []
            
            self.model_performance_history[model_name].append({
                'window_index': window_index,
                'test_period': f"{test_start.date()} to {test_end.date()}",
                'metrics': metrics
            })
        
        return window_results
    
    def _generate_summary_statistics(self):
        """Generate summary statistics from walk-forward results"""
        if not self.walk_forward_results['windows']:
            return
        
        summary = {
            'total_windows': len(self.walk_forward_results['windows']),
            'model_performance_summary': {},
            'stability_analysis': {},
            'best_performing_model': None
        }
        
        # Calculate performance statistics for each model
        for model_name, performance_history in self.model_performance_history.items():
            if not performance_history:
                continue
            
            # Extract metrics across all windows
            accuracies = [p['metrics'].get('accuracy', 0) for p in performance_history]
            f1_scores = [p['metrics'].get('f1_score', 0) for p in performance_history]
            
            if accuracies:
                summary['model_performance_summary'][model_name] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'min_accuracy': np.min(accuracies),
                    'max_accuracy': np.max(accuracies),
                    'mean_f1_score': np.mean(f1_scores) if f1_scores else 0,
                    'stability_score': 1 - (np.std(accuracies) / np.mean(accuracies)) if np.mean(accuracies) > 0 else 0
                }
        
        # Find best performing model
        if summary['model_performance_summary']:
            best_model = max(
                summary['model_performance_summary'].items(),
                key=lambda x: x[1]['mean_accuracy']
            )
            summary['best_performing_model'] = best_model[0]
        
        self.walk_forward_results['summary'] = summary
    
    def get_performance_trends(self) -> Dict:
        """Analyze performance trends across windows"""
        if not self.model_performance_history:
            return {}
        
        trends = {}
        
        for model_name, performance_history in self.model_performance_history.items():
            if len(performance_history) < 2:
                continue
            
            # Extract performance over time
            windows = [p['window_index'] for p in performance_history]
            accuracies = [p['metrics'].get('accuracy', 0) for p in performance_history]
            
            # Calculate trend (slope of linear regression)
            if len(accuracies) > 1:
                x = np.array(windows)
                y = np.array(accuracies)
                
                # Linear regression
                slope = np.polyfit(x, y, 1)[0]
                
                trends[model_name] = {
                    'performance_trend': slope,
                    'trend_direction': 'improving' if slope > 0.001 else 'declining' if slope < -0.001 else 'stable',
                    'consistency': np.std(accuracies),
                    'last_window_accuracy': accuracies[-1] if accuracies else 0
                }
        
        return trends
    
    def plot_performance(self, save_path: Optional[Path] = None):
        """Generate performance plots (placeholder for actual plotting)"""
        # This would typically use matplotlib or plotly
        # For now, we'll just log the summary
        
        summary = self.walk_forward_results.get('summary', {})
        self.logger.info("Walk-Forward Analysis Summary:")
        self.logger.info(f"Total Windows: {summary.get('total_windows', 0)}")
        
        performance_summary = summary.get('model_performance_summary', {})
        for model_name, stats in performance_summary.items():
            self.logger.info(f"{model_name}: Mean Accuracy = {stats['mean_accuracy']:.4f} "
                           f"(±{stats['std_accuracy']:.4f})")
    
    def save_results(self, filepath: Path) -> bool:
        """Save walk-forward results to file"""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert datetime objects to strings for JSON serialization
            results_to_save = self._prepare_results_for_serialization(self.walk_forward_results)
            
            with open(filepath, 'w') as f:
                import json
                json.dump(results_to_save, f, indent=2, default=str)
            
            self.logger.info(f"Walk-forward results saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving walk-forward results: {e}")
            return False
    
    def _prepare_results_for_serialization(self, results: Dict) -> Dict:
        """Prepare results for JSON serialization"""
        import copy
        
        results_copy = copy.deepcopy(results)
        
        # Convert any remaining non-serializable objects
        for window in results_copy.get('windows', []):
            # Ensure all values are JSON serializable
            for key, value in window.items():
                if isinstance(value, (np.integer, np.floating)):
                    window[key] = float(value)
                elif isinstance(value, np.ndarray):
                    window[key] = value.tolist()
        
        return results_copy
    
    def get_model_stability_ranking(self) -> List[Tuple[str, float]]:
        """Rank models by performance stability"""
        summary = self.walk_forward_results.get('summary', {})
        performance_summary = summary.get('model_performance_summary', {})
        
        if not performance_summary:
            return []
        
        # Rank by stability score (higher is better)
        stability_ranking = sorted(
            performance_summary.items(),
            key=lambda x: x[1]['stability_score'],
            reverse=True
        )
        
        return stability_ranking
