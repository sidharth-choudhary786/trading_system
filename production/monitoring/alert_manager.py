# trading_system/production/monitoring/alert_manager.py
import smtplib
import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from ...core.exceptions import TradingSystemError

class AlertManager:
    """
    Manages alerts and notifications for the trading system
    Supports email, SMS, and webhook notifications
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Alert configuration
        self.alert_config = config.get('alerts', {})
        self.enabled_channels = self.alert_config.get('channels', [])
        
        # Initialize channels
        self._initialize_channels()
        
        # Alert history
        self.alert_history = []
        
        self.logger.info("Alert Manager initialized")
    
    def _initialize_channels(self):
        """Initialize alert channels"""
        self.channels = {}
        
        if 'email' in self.enabled_channels:
            self.channels['email'] = self._setup_email_channel()
        
        if 'webhook' in self.enabled_channels:
            self.channels['webhook'] = self._setup_webhook_channel()
        
        if 'sms' in self.enabled_channels:
            self.channels['sms'] = self._setup_sms_channel()
    
    def _setup_email_channel(self) -> Dict:
        """Setup email alert channel"""
        email_config = self.alert_config.get('email', {})
        return {
            'smtp_server': email_config.get('smtp_server', 'smtp.gmail.com'),
            'smtp_port': email_config.get('smtp_port', 587),
            'username': email_config.get('username'),
            'password': email_config.get('password'),
            'from_email': email_config.get('from_email'),
            'to_emails': email_config.get('to_emails', [])
        }
    
    def _setup_webhook_channel(self) -> Dict:
        """Setup webhook alert channel"""
        webhook_config = self.alert_config.get('webhook', {})
        return {
            'url': webhook_config.get('url'),
            'headers': webhook_config.get('headers', {'Content-Type': 'application/json'})
        }
    
    def _setup_sms_channel(self) -> Dict:
        """Setup SMS alert channel"""
        sms_config = self.alert_config.get('sms', {})
        return {
            'provider': sms_config.get('provider', 'twilio'),
            'account_sid': sms_config.get('account_sid'),
            'auth_token': sms_config.get('auth_token'),
            'from_number': sms_config.get('from_number'),
            'to_numbers': sms_config.get('to_numbers', [])
        }
    
    def send_alert(
        self, 
        message: str, 
        level: str = 'INFO',
        channels: Optional[List[str]] = None,
        subject: Optional[str] = None
    ) -> bool:
        """
        Send alert through specified channels
        
        Args:
            message: Alert message
            level: Alert level (INFO, WARNING, ERROR, CRITICAL)
            channels: Channels to use (default: all enabled channels)
            subject: Optional subject for email alerts
            
        Returns:
            True if alert was sent successfully
        """
        if channels is None:
            channels = self.enabled_channels
        
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'channels': channels,
            'sent_successfully': False
        }
        
        success = True
        
        # Send through each channel
        for channel in channels:
            if channel in self.channels:
                try:
                    if channel == 'email':
                        self._send_email_alert(message, level, subject)
                    elif channel == 'webhook':
                        self._send_webhook_alert(message, level)
                    elif channel == 'sms':
                        self._send_sms_alert(message, level)
                    
                    self.logger.info(f"Alert sent via {channel}: {message}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to send alert via {channel}: {e}")
                    success = False
            else:
                self.logger.warning(f"Alert channel not configured: {channel}")
                success = False
        
        alert_data['sent_successfully'] = success
        self.alert_history.append(alert_data)
        
        return success
    
    def _send_email_alert(self, message: str, level: str, subject: Optional[str] = None):
        """Send email alert"""
        email_config = self.channels['email']
        
        if subject is None:
            subject = f"Trading System Alert - {level}"
        
        # Create message
        msg = MimeMultipart()
        msg['From'] = email_config['from_email']
        msg['To'] = ', '.join(email_config['to_emails'])
        msg['Subject'] = subject
        
        body = f"""
        Trading System Alert
        
        Level: {level}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Message:
        {message}
        
        ---
        Automated Alert from Trading System
        """
        
        msg.attach(MimeText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
    
    def _send_webhook_alert(self, message: str, level: str):
        """Send webhook alert"""
        webhook_config = self.channels['webhook']
        
        payload = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'system': 'trading_system'
        }
        
        response = requests.post(
            webhook_config['url'],
            json=payload,
            headers=webhook_config['headers']
        )
        
        response.raise_for_status()
    
    def _send_sms_alert(self, message: str, level: str):
        """Send SMS alert"""
        sms_config = self.channels['sms']
        
        # This is a simplified implementation
        # In practice, you would use Twilio, AWS SNS, or similar service
        self.logger.info(f"SMS Alert ({level}): {message}")
        # Implementation would go here for actual SMS service
    
    def send_trade_alert(self, trade_data: Dict):
        """Send trade execution alert"""
        message = f"""
        Trade Executed:
        Symbol: {trade_data.get('symbol')}
        Side: {trade_data.get('side')}
        Quantity: {trade_data.get('quantity')}
        Price: {trade_data.get('price')}
        P&L: {trade_data.get('pnl', 'N/A')}
        """
        
        self.send_alert(
            message=message,
            level='INFO',
            subject=f"Trade Executed - {trade_data.get('symbol')}"
        )
    
    def send_error_alert(self, error: Exception, context: str = ""):
        """Send error alert"""
        message = f"""
        System Error:
        Context: {context}
        Error Type: {type(error).__name__}
        Error Message: {str(error)}
        """
        
        self.send_alert(
            message=message,
            level='ERROR',
            subject="Trading System Error"
        )
    
    def send_performance_alert(self, performance_data: Dict):
        """Send performance alert"""
        message = f"""
        Performance Update:
        Total Return: {performance_data.get('total_return', 'N/A')}
        Sharpe Ratio: {performance_data.get('sharpe_ratio', 'N/A')}
        Max Drawdown: {performance_data.get('max_drawdown', 'N/A')}
        Win Rate: {performance_data.get('win_rate', 'N/A')}
        Total Trades: {performance_data.get('total_trades', 'N/A')}
        """
        
        self.send_alert(
            message=message,
            level='INFO',
            subject="Trading Performance Update"
        )
    
    def send_risk_alert(self, risk_data: Dict):
        """Send risk threshold breach alert"""
        message = f"""
        Risk Threshold Breached:
        Metric: {risk_data.get('metric')}
        Current Value: {risk_data.get('current_value')}
        Threshold: {risk_data.get('threshold')}
        Breach Type: {risk_data.get('breach_type')}
        """
        
        self.send_alert(
            message=message,
            level='WARNING',
            subject="Risk Threshold Breach"
        )
    
    def get_alert_history(
        self, 
        level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """Get alert history with optional filtering"""
        filtered_alerts = self.alert_history
        
        if level:
            filtered_alerts = [a for a in filtered_alerts if a['level'] == level]
        
        if start_time:
            filtered_alerts = [a for a in filtered_alerts 
                             if datetime.fromisoformat(a['timestamp']) >= start_time]
        
        if end_time:
            filtered_alerts = [a for a in filtered_alerts 
                             if datetime.fromisoformat(a['timestamp']) <= end_time]
        
        return filtered_alerts
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics"""
        if not self.alert_history:
            return {}
        
        total_alerts = len(self.alert_history)
        successful_alerts = len([a for a in self.alert_history if a['sent_successfully']])
        
        level_counts = {}
        for alert in self.alert_history:
            level = alert['level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'successful_alerts': successful_alerts,
            'success_rate': successful_alerts / total_alerts * 100,
            'level_distribution': level_counts,
            'first_alert': min([a['timestamp'] for a in self.alert_history]) if self.alert_history else None,
            'last_alert': max([a['timestamp'] for a in self.alert_history]) if self.alert_history else None
        }
