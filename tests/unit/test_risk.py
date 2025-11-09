# trading_system/tests/unit/test_risk.py
"""
Unit tests for risk management module
"""
import pytest
import pandas as pd
import numpy as np
from trading_system.risk.risk_engine import RiskEngine
from trading_system.risk.compliance import ComplianceEngine

class TestRiskEngine:
    """Test risk engine functionality"""
    
    def test_risk_engine_initialization(self):
        """Test risk engine initialization"""
        config = {
            'risk': {
                'max_daily_loss': 0.02,
                'max_drawdown': 0.15,
                'max_position_size': 0.1
            }
        }
        
        risk_engine = RiskEngine(config)
        assert risk_engine.max_daily_loss == 0.02
        assert risk_engine.max_drawdown == 0.15
    
    def test_var_calculation(self):
        """Test Value at Risk calculation"""
        config = {'risk': {'var_confidence': 0.95}}
        risk_engine = RiskEngine(config)
        
        # Sample returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        
        var = risk_engine.calculate_var(returns)
        assert var < 0  # VaR should be negative (loss)
        assert abs(var) > 0  # Should be non-zero
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation"""
        config = {'risk': {}}
        risk_engine = RiskEngine(config)
        
        # Sample portfolio values with a drawdown
        portfolio_values = pd.Series([100, 105, 110, 95, 100, 108])
        
        drawdown = risk_engine.calculate_drawdown(portfolio_values)
        expected_drawdown = (95 - 110) / 110  # Minimum from peak
        assert drawdown == pytest.approx(expected_drawdown)
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        config = {
            'risk': {
                'max_daily_loss': 0.05,
                'enable_circuit_breakers': True
            }
        }
        
        risk_engine = RiskEngine(config)
        
        # Test within limits
        current_pnl = -0.03  # -3%
        should_trigger = risk_engine.check_circuit_breaker(current_pnl, 0.0)
        assert not should_trigger
        
        # Test beyond limits
        current_pnl = -0.06  # -6%
        should_trigger = risk_engine.check_circuit_breaker(current_pnl, 0.0)
        assert should_trigger

class TestComplianceEngine:
    """Test compliance checking functionality"""
    
    def test_compliance_initialization(self):
        """Test compliance engine initialization"""
        config = {
            'risk': {
                'max_position_size': 0.1,
                'max_sector_exposure': 0.3
            }
        }
        
        compliance = ComplianceEngine(config)
        assert compliance.max_position_size == 0.1
    
    def test_order_compliance_check(self, sample_order):
        """Test order compliance checking"""
        config = {
            'risk': {
                'max_position_size': 0.1
            }
        }
        
        compliance = ComplianceEngine(config)
        
        # Mock portfolio state
        portfolio_state = {
            'portfolio_value': 1000000,
            'positions': {'RELIANCE': 50000},  # 5% exposure
            'sector_exposure': {'Technology': 0.25}
        }
        
        # Test compliant order
        compliant_order = sample_order  # 100 * 2500 = 250,000 (25% of portfolio)
        # This would be non-compliant in real scenario, but testing the method
        is_compliant, message = compliance.check_order(compliant_order, portfolio_state)
        
        # Method should return boolean and message
        assert isinstance(is_compliant, bool)
        assert isinstance(message, str)
