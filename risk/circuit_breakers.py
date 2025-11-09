# trading_system/risk/circuit_breakers.py
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import threading
import time

from ..core.exceptions import RiskError
from ..core.types import Order, OrderSide

class CircuitBreaker:
    """
    Circuit breaker system for emergency trading halts
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Circuit breaker configuration
        self.breakers = {
            'max_daily_loss': {
                'enabled': config.get('max_daily_loss_enabled', True),
                'threshold': config.get('max_daily_loss_threshold', -0.05),  # -5%
                'triggered': False,
                'last_triggered': None
            },
            'max_drawdown': {
                'enabled': config.get('max_drawdown_enabled', True), 
                'threshold': config.get('max_drawdown_threshold', -0.15),  # -15%
                'triggered': False,
                'last_triggered': None
            },
            'position_breach': {
                'enabled': config.get('position_breach_enabled', True),
                'triggered': False,
                'last_triggered': None
            },
            'volatility_breaker': {
                'enabled': config.get('volatility_breaker_enabled', True),
                'threshold': config.get('volatility_threshold', 0.10),  # 10%
                'triggered': False,
                'last_triggered': None
            },
            'data_feed_breaker': {
                'enabled': config.get('data_feed_breaker_enabled', True),
                'stale_threshold': config.get('data_stale_threshold', 300),  # 5 minutes
                'triggered': False,
                'last_triggered': None
            }
        }
        
        # Trading state
        self.trading_halted = False
        self.halt_reason = None
        self.halt_start_time = None
        self.auto_resume_delay = config.get('auto_resume_delay', 300)  # 5 minutes
        
        # Monitoring state
        self.portfolio_values = []
        self.daily_pnl = 0.0
        self.max_portfolio_value = 0.0
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        self.logger.info("Circuit Breaker initialized")
    
    def check_circuit_breakers(
        self, 
        current_portfolio_value: float,
        current_positions: Dict,
        market_data: Dict,
        orders: List[Order] = None
    ) -> Dict:
        """
        Check all circuit breakers
        
        Returns:
            Dictionary with breaker status and any triggers
        """
        with self._lock:
            triggers = {}
            any_triggered = False
            
            # Update portfolio tracking
            self._update_portfolio_tracking(current_portfolio_value)
            
            # Check each circuit breaker
            if self.breakers['max_daily_loss']['enabled']:
                if self._check_max_daily_loss():
                    triggers['max_daily_loss'] = {
                        'threshold': self.breakers['max_daily_loss']['threshold'],
                        'current_pnl': self.daily_pnl
                    }
                    any_triggered = True
            
            if self.breakers['max_drawdown']['enabled']:
                if self._check_max_drawdown(current_portfolio_value):
                    triggers['max_drawdown'] = {
                        'threshold': self.breakers['max_drawdown']['threshold'],
                        'current_drawdown': self._calculate_current_drawdown(current_portfolio_value)
                    }
                    any_triggered = True
            
            if self.breakers['position_breach']['enabled']:
                if self._check_position_breach(current_positions):
                    triggers['position_breach'] = {
                        'positions': current_positions
                    }
                    any_triggered = True
            
            if self.breakers['volatility_breaker']['enabled']:
                if self._check_volatility_breaker(market_data):
                    triggers['volatility_breaker'] = {
                        'threshold': self.breakers['volatility_breaker']['threshold'],
                        'current_volatility': market_data.get('volatility', 0)
                    }
                    any_triggered = True
            
            if self.breakers['data_feed_breaker']['enabled']:
                if self._check_data_feed_breaker(market_data):
                    triggers['data_feed_breaker'] = {
                        'stale_threshold': self.breakers['data_feed_breaker']['stale_threshold'],
                        'last_update': market_data.get('last_update')
                    }
                    any_triggered = True
            
            # Handle triggers
            if any_triggered and not self.trading_halted:
                self._halt_trading(triggers)
            
            # Check for auto-resume
            elif self.trading_halted and self._should_auto_resume():
                self._resume_trading()
            
            return {
                'trading_halted': self.trading_halted,
                'halt_reason': self.halt_reason,
                'triggers': triggers,
                'breakers_status': self._get_breakers_status()
            }
    
    def _check_max_daily_loss(self) -> bool:
        """Check maximum daily loss breaker"""
        threshold = self.breakers['max_daily_loss']['threshold']
        
        if self.daily_pnl <= threshold:
            if not self.breakers['max_daily_loss']['triggered']:
                self.breakers['max_daily_loss']['triggered'] = True
                self.breakers['max_daily_loss']['last_triggered'] = datetime.now()
                self.logger.warning(f"Max daily loss circuit breaker triggered: {self.daily_pnl:.2%}")
            return True
        
        return False
    
    def _check_max_drawdown(self, current_value: float) -> bool:
        """Check maximum drawdown breaker"""
        if self.max_portfolio_value == 0:
            return False
        
        current_drawdown = (current_value - self.max_portfolio_value) / self.max_portfolio_value
        threshold = self.breakers['max_drawdown']['threshold']
        
        if current_drawdown <= threshold:
            if not self.breakers['max_drawdown']['triggered']:
                self.breakers['max_drawdown']['triggered'] = True
                self.breakers['max_drawdown']['last_triggered'] = datetime.now()
                self.logger.warning(f"Max drawdown circuit breaker triggered: {current_drawdown:.2%}")
            return True
        
        return False
    
    def _check_position_breach(self, positions: Dict) -> bool:
        """Check position limit breaches"""
        # This would integrate with PositionLimitManager
        # For now, using simplified checks
        
        # Check for excessive concentration
        if positions:
            total_value = sum(positions.values())
            if total_value > 0:
                max_position = max(positions.values())
                concentration = max_position / total_value
                
                if concentration > 0.5:  # 50% concentration
                    if not self.breakers['position_breach']['triggered']:
                        self.breakers['position_breach']['triggered'] = True
                        self.breakers['position_breach']['last_triggered'] = datetime.now()
                        self.logger.warning(f"Position concentration breach: {concentration:.2%}")
                    return True
        
        return False
    
    def _check_volatility_breaker(self, market_data: Dict) -> bool:
        """Check volatility circuit breaker"""
        current_volatility = market_data.get('volatility', 0)
        threshold = self.breakers['volatility_breaker']['threshold']
        
        if current_volatility >= threshold:
            if not self.breakers['volatility_breaker']['triggered']:
                self.breakers['volatility_breaker']['triggered'] = True
                self.breakers['volatility_breaker']['last_triggered'] = datetime.now()
                self.logger.warning(f"Volatility circuit breaker triggered: {current_volatility:.2%}")
            return True
        
        return False
    
    def _check_data_feed_breaker(self, market_data: Dict) -> bool:
        """Check data feed staleness"""
        last_update = market_data.get('last_update')
        if not last_update:
            return True
        
        if isinstance(last_update, str):
            last_update = pd.to_datetime(last_update)
        
        stale_threshold = self.breakers['data_feed_breaker']['stale_threshold']
        time_since_update = (datetime.now() - last_update).total_seconds()
        
        if time_since_update > stale_threshold:
            if not self.breakers['data_feed_breaker']['triggered']:
                self.breakers['data_feed_breaker']['triggered'] = True
                self.breakers['data_feed_breaker']['last_triggered'] = datetime.now()
                self.logger.warning(f"Data feed circuit breaker triggered: {time_since_update:.0f}s stale")
            return True
        
        return False
    
    def _update_portfolio_tracking(self, current_value: float):
        """Update portfolio value tracking"""
        current_time = datetime.now()
        
        # Reset daily PnL at market open (9:15 AM)
        if current_time.hour == 9 and current_time.minute == 15:
            self.daily_pnl = 0.0
            self.portfolio_values = []
        
        # Track portfolio values
        self.portfolio_values.append({
            'timestamp': current_time,
            'value': current_value
        })
        
        # Keep only last 24 hours
        cutoff_time = current_time - timedelta(hours=24)
        self.portfolio_values = [
            pv for pv in self.portfolio_values 
            if pv['timestamp'] > cutoff_time
        ]
        
        # Update max portfolio value
        if current_value > self.max_portfolio_value:
            self.max_portfolio_value = current_value
        
        # Calculate daily PnL (simplified)
        if len(self.portfolio_values) >= 2:
            first_value_today = self._get_first_value_today()
            if first_value_today is not None:
                self.daily_pnl = (current_value - first_value_today) / first_value_today
    
    def _get_first_value_today(self) -> Optional[float]:
        """Get first portfolio value of the day"""
        today = datetime.now().date()
        today_values = [
            pv for pv in self.portfolio_values 
            if pv['timestamp'].date() == today
        ]
        
        if today_values:
            return today_values[0]['value']
        return None
    
    def _calculate_current_drawdown(self, current_value: float) -> float:
        """Calculate current drawdown from peak"""
        if self.max_portfolio_value == 0:
            return 0.0
        return (current_value - self.max_portfolio_value) / self.max_portfolio_value
    
    def _halt_trading(self, triggers: Dict):
        """Halt trading due to circuit breaker triggers"""
        self.trading_halted = True
        self.halt_reason = triggers
        self.halt_start_time = datetime.now()
        
        # Log the halt
        trigger_names = list(triggers.keys())
        self.logger.critical(f"TRADING HALTED due to: {', '.join(trigger_names)}")
        
        # Send alerts (would integrate with alert system)
        self._send_halt_alerts(triggers)
    
    def _resume_trading(self):
        """Resume trading after halt"""
        self.trading_halted = False
        self.halt_reason = None
        
        # Reset triggered breakers (except those that need manual reset)
        for breaker_name, breaker in self.breakers.items():
            if breaker_name not in ['max_daily_loss', 'max_drawdown']:
                breaker['triggered'] = False
        
        self.logger.info("Trading resumed after circuit breaker halt")
    
    def _should_auto_resume(self) -> bool:
        """Check if trading should auto-resume"""
        if not self.halt_start_time:
            return False
        
        time_in_halt = (datetime.now() - self.halt_start_time).total_seconds()
        return time_in_halt >= self.auto_resume_delay
    
    def _send_halt_alerts(self, triggers: Dict):
        """Send alerts for trading halt"""
        # This would integrate with the alert system
        alert_message = f"TRADING HALTED - Circuit breakers triggered:\n"
        
        for trigger_name, trigger_data in triggers.items():
            alert_message += f"- {trigger_name}: {trigger_data}\n"
        
        self.logger.critical(alert_message)
    
    def _get_breakers_status(self) -> Dict:
        """Get status of all circuit breakers"""
        status = {}
        
        for breaker_name, breaker in self.breakers.items():
            status[breaker_name] = {
                'enabled': breaker['enabled'],
                'triggered': breaker['triggered'],
                'last_triggered': breaker['last_triggered']
            }
        
        return status
    
    def manual_override(self, action: str, breaker_name: Optional[str] = None):
        """
        Manual override of circuit breakers
        
        Args:
            action: 'halt', 'resume', 'reset'
            breaker_name: Specific breaker to override (None for all)
        """
        with self._lock:
            if action == 'halt':
                self.trading_halted = True
                self.halt_reason = {'manual_override': True}
                self.halt_start_time = datetime.now()
                self.logger.warning("Manual trading halt initiated")
            
            elif action == 'resume':
                self.trading_halted = False
                self.halt_reason = None
                self.logger.warning("Manual trading resume initiated")
            
            elif action == 'reset':
                if breaker_name:
                    if breaker_name in self.breakers:
                        self.breakers[breaker_name]['triggered'] = False
                        self.logger.info(f"Reset circuit breaker: {breaker_name}")
                else:
                    for breaker in self.breakers.values():
                        breaker['triggered'] = False
                    self.logger.info("Reset all circuit breakers")
    
    def get_circuit_breaker_report(self) -> Dict:
        """Generate circuit breaker status report"""
        return {
            'timestamp': datetime.now(),
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'halt_duration': self._get_halt_duration(),
            'breakers_status': self._get_breakers_status(),
            'portfolio_metrics': {
                'daily_pnl': self.daily_pnl,
                'max_drawdown': self._calculate_current_drawdown(
                    self.portfolio_values[-1]['value'] if self.portfolio_values else 0
                ),
                'max_portfolio_value': self.max_portfolio_value
            }
        }
    
    def _get_halt_duration(self) -> Optional[float]:
        """Get current halt duration in seconds"""
        if self.trading_halted and self.halt_start_time:
            return (datetime.now() - self.halt_start_time).total_seconds()
        return None
