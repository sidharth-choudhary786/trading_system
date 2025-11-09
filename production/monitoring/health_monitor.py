# trading_system/production/monitoring/health_monitor.py
import time
import threading
import psutil
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import smtplib
from email.mime.text import MIMEText
import json

from ....core.exceptions import TradingSystemError

class HealthMonitor:
    """
    Real-time health monitoring for production trading system
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Monitoring configuration
        self.monitoring_interval = config.get('health_check_interval', 60)  # seconds
        self.performance_tracking = config.get('performance_tracking', True)
        
        # System metrics
        self.system_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': [],
            'network_io': [],
            'process_count': []
        }
        
        # Trading metrics
        self.trading_metrics = {
            'orders_sent': 0,
            'orders_filled': 0,
            'orders_failed': 0,
            'total_pnl': 0.0,
            'current_positions': 0
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_usage': 80.0,  # %
            'memory_usage': 85.0,  # %
            'disk_usage': 90.0,  # %
            'order_failure_rate': 10.0,  # %
            'max_drawdown': -5.0,  # %
            'latency_threshold': 5.0  # seconds
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.health_check_history = []
        
        # Alert manager
        from .alert_manager import AlertManager
        self.alert_manager = AlertManager(config)
        
        self.logger.info("Health Monitor initialized")
    
    def start_monitoring(self):
        """Start health monitoring"""
        if self.is_monitoring:
            self.logger.warning("Health monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_health = self._check_system_health()
                
                # Collect trading metrics
                trading_health = self._check_trading_health()
                
                # Combined health assessment
                health_status = self._assess_health_status(system_health, trading_health)
                
                # Store health check
                self.health_check_history.append({
                    'timestamp': datetime.now(),
                    'system_health': system_health,
                    'trading_health': trading_health,
                    'overall_status': health_status
                })
                
                # Keep only last 24 hours of history
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.health_check_history = [
                    h for h in self.health_check_history 
                    if h['timestamp'] > cutoff_time
                ]
                
                # Check for alerts
                self._check_alerts(system_health, trading_health)
                
                # Log health status
                if health_status['status'] != 'HEALTHY':
                    self.logger.warning(f"System health issue: {health_status}")
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
            
            # Wait for next check
            time.sleep(self.monitoring_interval)
    
    def _check_system_health(self) -> Dict:
        """Check system health metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_metrics = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            # Process count
            process_count = len(list(psutil.process_iter()))
            
            # Update metrics history
            self._update_system_metrics({
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'disk_usage': disk_percent,
                'network_io': network_metrics,
                'process_count': process_count
            })
            
            system_health = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'disk_usage': disk_percent,
                'process_count': process_count,
                'network_io': network_metrics,
                'timestamp': datetime.now()
            }
            
            return system_health
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            return {'error': str(e)}
    
    def _check_trading_health(self) -> Dict:
        """Check trading system health"""
        try:
            # Calculate order failure rate
            total_orders = self.trading_metrics['orders_sent']
            failed_orders = self.trading_metrics['orders_failed']
            failure_rate = (failed_orders / total_orders * 100) if total_orders > 0 else 0
            
            # Calculate fill rate
            filled_orders = self.trading_metrics['orders_filled']
            fill_rate = (filled_orders / total_orders * 100) if total_orders > 0 else 0
            
            # Get current positions
            current_positions = self.trading_metrics['current_positions']
            
            # Calculate PnL metrics (simplified)
            total_pnl = self.trading_metrics['total_pnl']
            
            trading_health = {
                'order_failure_rate': failure_rate,
                'order_fill_rate': fill_rate,
                'current_positions': current_positions,
                'total_pnl': total_pnl,
                'total_orders': total_orders,
                'failed_orders': failed_orders,
                'filled_orders': filled_orders,
                'timestamp': datetime.now()
            }
            
            return trading_health
            
        except Exception as e:
            self.logger.error(f"Error checking trading health: {e}")
            return {'error': str(e)}
    
    def _assess_health_status(self, system_health: Dict, trading_health: Dict) -> Dict:
        """Assess overall health status"""
        issues = []
        status = 'HEALTHY'
        
        # Check system health thresholds
        if system_health.get('cpu_usage', 0) > self.alert_thresholds['cpu_usage']:
            issues.append(f"High CPU usage: {system_health['cpu_usage']}%")
            status = 'WARNING'
        
        if system_health.get('memory_usage', 0) > self.alert_thresholds['memory_usage']:
            issues.append(f"High memory usage: {system_health['memory_usage']}%")
            status = 'WARNING'
        
        if system_health.get('disk_usage', 0) > self.alert_thresholds['disk_usage']:
            issues.append(f"High disk usage: {system_health['disk_usage']}%")
            status = 'WARNING'
        
        # Check trading health thresholds
        if trading_health.get('order_failure_rate', 0) > self.alert_thresholds['order_failure_rate']:
            issues.append(f"High order failure rate: {trading_health['order_failure_rate']}%")
            status = 'CRITICAL'
        
        if trading_health.get('total_pnl', 0) < self.alert_thresholds['max_drawdown']:
            issues.append(f"Max drawdown exceeded: {trading_health['total_pnl']}%")
            status = 'CRITICAL'
        
        return {
            'status': status,
            'issues': issues,
            'timestamp': datetime.now()
        }
    
    def _check_alerts(self, system_health: Dict, trading_health: Dict):
        """Check for alert conditions"""
        alerts = []
        
        # System alerts
        if system_health.get('cpu_usage', 0) > self.alert_thresholds['cpu_usage']:
            alerts.append({
                'type': 'SYSTEM',
                'severity': 'HIGH',
                'message': f"High CPU usage: {system_health['cpu_usage']}%",
                'timestamp': datetime.now()
            })
        
        if system_health.get('memory_usage', 0) > self.alert_thresholds['memory_usage']:
            alerts.append({
                'type': 'SYSTEM', 
                'severity': 'HIGH',
                'message': f"High memory usage: {system_health['memory_usage']}%",
                'timestamp': datetime.now()
            })
        
        # Trading alerts
        if trading_health.get('order_failure_rate', 0) > self.alert_thresholds['order_failure_rate']:
            alerts.append({
                'type': 'TRADING',
                'severity': 'CRITICAL',
                'message': f"High order failure rate: {trading_health['order_failure_rate']}%",
                'timestamp': datetime.now()
            })
        
        # Send alerts
        for alert in alerts:
            self.alert_manager.send_alert(alert)
    
    def _update_system_metrics(self, metrics: Dict):
        """Update system metrics history"""
        for key, value in metrics.items():
            if key in self.system_metrics:
                self.system_metrics[key].append({
                    'timestamp': datetime.now(),
                    'value': value
                })
                
                # Keep only last 1000 data points
                if len(self.system_metrics[key]) > 1000:
                    self.system_metrics[key] = self.system_metrics[key][-1000:]
    
    def update_trading_metrics(self, metrics: Dict):
        """Update trading metrics"""
        for key, value in metrics.items():
            if key in self.trading_metrics:
                if isinstance(value, (int, float)):
                    self.trading_metrics[key] += value
                else:
                    self.trading_metrics[key] = value
    
    def get_health_report(self) -> Dict:
        """Generate comprehensive health report"""
        if not self.health_check_history:
            return {'status': 'UNKNOWN', 'message': 'No health data available'}
        
        latest_health = self.health_check_history[-1]
        
        # Calculate system trends
        system_trends = self._calculate_system_trends()
        
        # Calculate trading trends
        trading_trends = self._calculate_trading_trends()
        
        report = {
            'timestamp': datetime.now(),
            'overall_status': latest_health['overall_status']['status'],
            'current_issues': latest_health['overall_status']['issues'],
            'system_health': latest_health['system_health'],
            'trading_health': latest_health['trading_health'],
            'system_trends': system_trends,
            'trading_trends': trading_trends,
            'alert_thresholds': self.alert_thresholds,
            'recommendations': self._generate_recommendations(latest_health)
        }
        
        return report
    
    def _calculate_system_trends(self) -> Dict:
        """Calculate system metric trends"""
        trends = {}
        
        for metric_name, metric_data in self.system_metrics.items():
            if len(metric_data) < 2:
                continue
            
            recent_values = [m['value'] for m in metric_data[-10:]]  # Last 10 readings
            if len(recent_values) >= 2:
                # Simple trend calculation
                first_value = recent_values[0]
                last_value = recent_values[-1]
                change = last_value - first_value
                change_pct = (change / first_value * 100) if first_value != 0 else 0
                
                trends[metric_name] = {
                    'current_value': last_value,
                    'trend': 'INCREASING' if change > 0 else 'DECREASING' if change < 0 else 'STABLE',
                    'change_amount': change,
                    'change_percentage': change_pct
                }
        
        return trends
    
    def _calculate_trading_trends(self) -> Dict:
        """Calculate trading metric trends"""
        # Use health check history for trading trends
        if len(self.health_check_history) < 2:
            return {}
        
        recent_checks = self.health_check_history[-10:]  # Last 10 checks
        
        failure_rates = [check['trading_health'].get('order_failure_rate', 0) for check in recent_checks]
        fill_rates = [check['trading_health'].get('order_fill_rate', 0) for check in recent_checks]
        
        trends = {
            'failure_rate': {
                'current': failure_rates[-1] if failure_rates else 0,
                'trend': self._calculate_trend_direction(failure_rates),
                'average': sum(failure_rates) / len(failure_rates) if failure_rates else 0
            },
            'fill_rate': {
                'current': fill_rates[-1] if fill_rates else 0,
                'trend': self._calculate_trend_direction(fill_rates),
                'average': sum(fill_rates) / len(fill_rates) if fill_rates else 0
            }
        }
        
        return trends
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return 'STABLE'
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        if avg_second > avg_first * 1.05:  # 5% increase
            return 'INCREASING'
        elif avg_second < avg_first * 0.95:  # 5% decrease
            return 'DECREASING'
        else:
            return 'STABLE'
    
    def _generate_recommendations(self, health_data: Dict) -> List[str]:
        """Generate recommendations based on health data"""
        recommendations = []
        
        system_health = health_data['system_health']
        trading_health = health_data['trading_health']
        
        # System recommendations
        if system_health.get('cpu_usage', 0) > 70:
            recommendations.append("Consider optimizing CPU-intensive processes")
        
        if system_health.get('memory_usage', 0) > 75:
            recommendations.append("Monitor memory usage and consider adding more RAM")
        
        if system_health.get('disk_usage', 0) > 80:
            recommendations.append("Consider cleaning up disk space or adding storage")
        
        # Trading recommendations
        if trading_health.get('order_failure_rate', 0) > 5:
            recommendations.append("Review order execution logic and broker connectivity")
        
        if trading_health.get('order_fill_rate', 0) < 80:
            recommendations.append("Consider adjusting order types or execution timing")
        
        return recommendations
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.trading_metrics = {
            'orders_sent': 0,
            'orders_filled': 0,
            'orders_failed': 0,
            'total_pnl': 0.0,
            'current_positions': 0
        }
        
        for key in self.system_metrics:
            self.system_metrics[key] = []
        
        self.health_check_history = []
        self.logger.info("All health metrics reset")
