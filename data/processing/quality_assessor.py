# trading_system/data/processing/quality_assessor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ...core.exceptions import DataSourceError

class DataQualityAssessor:
    """
    Assess data quality and generate quality reports
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.quality_thresholds = {
            'completeness': 0.95,  # 95% data completeness
            'consistency': 0.90,   # 90% data consistency
            'freshness': 0.80,     # Data should be reasonably fresh
            'accuracy': 0.85,      # 85% data accuracy
            'overall': 0.85        # Overall quality threshold
        }
    
    def assess_quality(self, data: pd.DataFrame, symbol: str = None) -> Dict:
        """
        Comprehensive data quality assessment
        
        Args:
            data: DataFrame to assess
            symbol: Optional symbol for reporting
            
        Returns:
            Dictionary with quality metrics and scores
        """
        if data.empty:
            return self._empty_quality_report(symbol)
        
        quality_report = {
            'symbol': symbol,
            'assessment_date': datetime.now().isoformat(),
            'data_points': len(data),
            'columns_analyzed': list(data.columns),
            'quality_scores': {},
            'issues_found': [],
            'recommendations': [],
            'overall_score': 0.0,
            'quality_grade': 'F'
        }
        
        # Calculate individual quality metrics
        completeness_score = self._assess_completeness(data)
        consistency_score = self._assess_consistency(data)
        freshness_score = self._assess_freshness(data)
        accuracy_score = self._assess_accuracy(data)
        
        # Store individual scores
        quality_report['quality_scores'] = {
            'completeness': completeness_score,
            'consistency': consistency_score,
            'freshness': freshness_score,
            'accuracy': accuracy_score
        }
        
        # Calculate overall score (weighted average)
        weights = {
            'completeness': 0.3,
            'consistency': 0.3,
            'freshness': 0.2,
            'accuracy': 0.2
        }
        
        overall_score = (
            completeness_score * weights['completeness'] +
            consistency_score * weights['consistency'] +
            freshness_score * weights['freshness'] +
            accuracy_score * weights['accuracy']
        )
        
        quality_report['overall_score'] = overall_score
        quality_report['quality_grade'] = self._calculate_quality_grade(overall_score)
        
        # Identify issues and generate recommendations
        quality_report['issues_found'] = self._identify_issues(data, quality_report['quality_scores'])
        quality_report['recommendations'] = self._generate_recommendations(quality_report)
        
        self.logger.info(f"Quality assessment for {symbol}: {quality_report['quality_grade']} ({overall_score:.2f})")
        
        return quality_report
    
    def _assess_completeness(self, data: pd.DataFrame) -> float:
        """Assess data completeness"""
        if data.empty:
            return 0.0
        
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        completeness_ratio = 1.0 - (missing_cells / total_cells)
        
        # Penalize columns with high missingness
        column_completeness = data.notnull().mean()
        high_missing_penalty = (column_completeness < 0.5).sum() / len(data.columns) * 0.2
        
        final_score = max(0.0, completeness_ratio - high_missing_penalty)
        return final_score
    
    def _assess_consistency(self, data: pd.DataFrame) -> float:
        """Assess data consistency and validity"""
        if data.empty:
            return 0.0
        
        consistency_checks = []
        
        # Check for basic data type consistency
        dtype_consistency = self._check_dtype_consistency(data)
        consistency_checks.append(dtype_consistency)
        
        # Check for price data consistency (if available)
        price_consistency = self._check_price_consistency(data)
        consistency_checks.append(price_consistency)
        
        # Check for date consistency (if available)
        date_consistency = self._check_date_consistency(data)
        consistency_checks.append(date_consistency)
        
        # Check for volume consistency (if available)
        volume_consistency = self._check_volume_consistency(data)
        consistency_checks.append(volume_consistency)
        
        # Average all consistency checks
        consistency_score = np.mean(consistency_checks)
        return consistency_score
    
    def _check_dtype_consistency(self, data: pd.DataFrame) -> float:
        """Check data type consistency"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        expected_numeric = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        
        # Check if expected numeric columns are actually numeric
        numeric_consistency = sum(1 for col in expected_numeric if col in numeric_columns) / len(expected_numeric)
        
        return numeric_consistency
    
    def _check_price_consistency(self, data: pd.DataFrame) -> float:
        """Check price data consistency"""
        price_columns = ['open', 'high', 'low', 'close']
        available_price_cols = [col for col in price_columns if col in data.columns]
        
        if not available_price_cols:
            return 0.5  # Neutral score if no price data
        
        consistency_errors = 0
        total_checks = 0
        
        for _, row in data.iterrows():
            if all(pd.notna(row[col]) for col in available_price_cols):
                total_checks += 1
                
                # Check basic price relationships
                if 'low' in available_price_cols and 'high' in available_price_cols:
                    if row['low'] > row['high']:
                        consistency_errors += 1
                
                if all(col in available_price_cols for col in ['open', 'high', 'low', 'close']):
                    if not (row['low'] <= row['open'] <= row['high'] and 
                            row['low'] <= row['close'] <= row['high']):
                        consistency_errors += 1
        
        if total_checks == 0:
            return 0.5
        
        consistency_ratio = 1.0 - (consistency_errors / total_checks)
        return consistency_ratio
    
    def _check_date_consistency(self, data: pd.DataFrame) -> float:
        """Check date consistency"""
        if 'date' not in data.columns:
            return 0.5
        
        try:
            dates = pd.to_datetime(data['date'])
            
            # Check for duplicates
            duplicate_dates = dates.duplicated().sum()
            duplicate_ratio = duplicate_dates / len(dates)
            
            # Check for gaps (simplified)
            date_diffs = dates.sort_values().diff().dropna()
            if len(date_diffs) > 0:
                large_gaps = (date_diffs > timedelta(days=7)).sum()
                gap_ratio = large_gaps / len(date_diffs)
            else:
                gap_ratio = 0
            
            consistency_score = 1.0 - (duplicate_ratio + gap_ratio) / 2
            return max(0.0, consistency_score)
            
        except:
            return 0.3  # Low score if date parsing fails
    
    def _check_volume_consistency(self, data: pd.DataFrame) -> float:
        """Check volume data consistency"""
        if 'volume' not in data.columns:
            return 0.5
        
        volume_data = data['volume'].dropna()
        
        if len(volume_data) == 0:
            return 0.3
        
        # Check for negative volumes
        negative_volumes = (volume_data < 0).sum()
        negative_ratio = negative_volumes / len(volume_data)
        
        # Check for extreme outliers (beyond 10 standard deviations)
        if len(volume_data) > 10:
            z_scores = np.abs((volume_data - volume_data.mean()) / volume_data.std())
            extreme_outliers = (z_scores > 10).sum()
            outlier_ratio = extreme_outliers / len(volume_data)
        else:
            outlier_ratio = 0
        
        consistency_score = 1.0 - (negative_ratio + outlier_ratio) / 2
        return max(0.0, consistency_score)
    
    def _assess_freshness(self, data: pd.DataFrame) -> float:
        """Assess data freshness"""
        if 'date' not in data.columns:
            return 0.5
        
        try:
            dates = pd.to_datetime(data['date'])
            latest_date = dates.max()
            
            # Calculate how old the data is
            days_old = (datetime.now() - latest_date).days
            
            # Score based on data age (lower score for older data)
            if days_old <= 1:
                freshness_score = 1.0
            elif days_old <= 7:
                freshness_score = 0.8
            elif days_old <= 30:
                freshness_score = 0.6
            elif days_old <= 90:
                freshness_score = 0.4
            else:
                freshness_score = 0.2
            
            return freshness_score
            
        except:
            return 0.3
    
    def _assess_accuracy(self, data: pd.DataFrame) -> float:
        """Assess data accuracy (simplified)"""
        # This is a simplified accuracy assessment
        # In practice, this would involve comparison with trusted sources
        
        accuracy_indicators = []
        
        # Check for reasonable price values
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            price_data = data[['open', 'high', 'low', 'close']].dropna()
            if len(price_data) > 0:
                # Check if prices are within reasonable bounds (e.g., not zero or negative for stocks)
                reasonable_prices = ((price_data > 0) & (price_data < 1000000)).all(axis=1).mean()
                accuracy_indicators.append(reasonable_prices)
        
        # Check for reasonable volume values
        if 'volume' in data.columns:
            volume_data = data['volume'].dropna()
            if len(volume_data) > 0:
                reasonable_volumes = (volume_data >= 0).mean()
                accuracy_indicators.append(reasonable_volumes)
        
        if accuracy_indicators:
            accuracy_score = np.mean(accuracy_indicators)
        else:
            accuracy_score = 0.5  # Neutral score
        
        return accuracy_score
    
    def _identify_issues(self, data: pd.DataFrame, quality_scores: Dict) -> List[str]:
        """Identify specific data quality issues"""
        issues = []
        
        # Check completeness issues
        if quality_scores['completeness'] < self.quality_thresholds['completeness']:
            missing_cols = data.isnull().sum()
            high_missing_cols = missing_cols[missing_cols > 0].index.tolist()
            issues.append(f"High missing values in columns: {high_missing_cols}")
        
        # Check consistency issues
        if quality_scores['consistency'] < self.quality_thresholds['consistency']:
            issues.append("Data consistency issues detected (price relationships, duplicates, etc.)")
        
        # Check freshness issues
        if quality_scores['freshness'] < self.quality_thresholds['freshness']:
            issues.append("Data may be stale or outdated")
        
        # Check accuracy issues
        if quality_scores['accuracy'] < self.quality_thresholds['accuracy']:
            issues.append("Potential data accuracy issues")
        
        return issues
    
    def _generate_recommendations(self, quality_report: Dict) -> List[str]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        scores = quality_report['quality_scores']
        
        if scores['completeness'] < self.quality_thresholds['completeness']:
            recommendations.append("Consider data imputation for missing values")
        
        if scores['consistency'] < self.quality_thresholds['consistency']:
            recommendations.append("Review data for consistency errors and outliers")
        
        if scores['freshness'] < self.quality_thresholds['freshness']:
            recommendations.append("Update data with more recent observations")
        
        if scores['accuracy'] < self.quality_thresholds['accuracy']:
            recommendations.append("Validate data against trusted sources")
        
        if quality_report['overall_score'] < self.quality_thresholds['overall']:
            recommendations.append("Consider using alternative data sources")
        
        return recommendations
    
    def _calculate_quality_grade(self, score: float) -> str:
        """Calculate quality grade from score"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _empty_quality_report(self, symbol: str = None) -> Dict:
        """Generate report for empty data"""
        return {
            'symbol': symbol,
            'assessment_date': datetime.now().isoformat(),
            'data_points': 0,
            'columns_analyzed': [],
            'quality_scores': {
                'completeness': 0.0,
                'consistency': 0.0,
                'freshness': 0.0,
                'accuracy': 0.0
            },
            'issues_found': ['Empty dataset'],
            'recommendations': ['Provide valid data for assessment'],
            'overall_score': 0.0,
            'quality_grade': 'F'
        }
    
    def compare_quality(self, reports: List[Dict]) -> Dict:
        """Compare quality across multiple datasets"""
        comparison = {
            'compared_datasets': len(reports),
            'summary_stats': {},
            'ranking': [],
            'best_quality': None,
            'worst_quality': None
        }
        
        if not reports:
            return comparison
        
        # Calculate summary statistics
        overall_scores = [report['overall_score'] for report in reports]
        comparison['summary_stats'] = {
            'mean_score': np.mean(overall_scores),
            'std_score': np.std(overall_scores),
            'min_score': np.min(overall_scores),
            'max_score': np.max(overall_scores),
            'median_score': np.median(overall_scores)
        }
        
        # Create ranking
        ranked_reports = sorted(reports, key=lambda x: x['overall_score'], reverse=True)
        comparison['ranking'] = [
            {
                'symbol': report['symbol'],
                'overall_score': report['overall_score'],
                'quality_grade': report['quality_grade'],
                'rank': i + 1
            }
            for i, report in enumerate(ranked_reports)
        ]
        
        # Identify best and worst
        comparison['best_quality'] = comparison['ranking'][0] if comparison['ranking'] else None
        comparison['worst_quality'] = comparison['ranking'][-1] if comparison['ranking'] else None
        
        return comparison
