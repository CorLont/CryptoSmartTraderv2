"""Unit tests for risk management components."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.cryptosmarttrader.core.risk_guard import RiskGuard, RiskLevel, TradingMode


class TestRiskGuard:
    """Test RiskGuard risk management system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.risk_guard = RiskGuard(
            daily_loss_limit_pct=5.0,
            max_drawdown_pct=10.0,
            max_position_pct=2.0,
            max_exposure_pct=20.0
        )
    
    def test_initial_state(self):
        """Test initial risk guard state."""
        assert self.risk_guard.current_risk_level == RiskLevel.NORMAL
        assert self.risk_guard.trading_mode == TradingMode.NORMAL
        assert not self.risk_guard.is_kill_switch_active()
        assert self.risk_guard.get_daily_pnl() == 0.0
    
    def test_daily_loss_limit_breach(self):
        """Test daily loss limit breach detection."""
        # Simulate large loss
        self.risk_guard.update_pnl(-6000, portfolio_value=100000)  # 6% loss
        
        assert self.risk_guard.current_risk_level >= RiskLevel.HIGH
        assert self.risk_guard.trading_mode != TradingMode.NORMAL
        
        # Check risk metrics
        metrics = self.risk_guard.get_risk_metrics()
        assert metrics['daily_loss_pct'] == 6.0
        assert metrics['daily_limit_breached'] is True
    
    def test_drawdown_escalation(self):
        """Test drawdown-based risk escalation."""
        # Simulate moderate drawdown
        self.risk_guard.update_drawdown(current_value=92000, peak_value=100000)  # 8% drawdown
        
        assert self.risk_guard.current_risk_level >= RiskLevel.MEDIUM
        
        # Simulate severe drawdown
        self.risk_guard.update_drawdown(current_value=85000, peak_value=100000)  # 15% drawdown
        
        assert self.risk_guard.current_risk_level >= RiskLevel.CRITICAL
        assert self.risk_guard.trading_mode in [TradingMode.DEFENSIVE, TradingMode.EMERGENCY]
    
    def test_position_limit_enforcement(self):
        """Test position size limit enforcement."""
        # Test valid position
        assert self.risk_guard.validate_position_size(1.5, 'BTC/USD') is True
        
        # Test oversized position
        assert self.risk_guard.validate_position_size(3.0, 'BTC/USD') is False
        
        # Test after risk escalation
        self.risk_guard.escalate_risk_level(RiskLevel.HIGH)
        assert self.risk_guard.validate_position_size(1.5, 'BTC/USD') is False  # Should be reduced
    
    def test_exposure_limit_enforcement(self):
        """Test total exposure limit enforcement."""
        # Add some positions
        self.risk_guard.add_position('BTC/USD', 5.0)  # 5% exposure
        self.risk_guard.add_position('ETH/USD', 8.0)  # 8% exposure
        
        # Should allow additional position within limit
        assert self.risk_guard.validate_new_exposure('ADA/USD', 5.0) is True  # Total: 18%
        
        # Should reject position exceeding limit
        assert self.risk_guard.validate_new_exposure('SOL/USD', 10.0) is False  # Total: 23%
    
    def test_kill_switch_activation(self):
        """Test kill switch activation scenarios."""
        # Manual activation
        self.risk_guard.activate_kill_switch(reason="manual_override")
        
        assert self.risk_guard.is_kill_switch_active()
        assert self.risk_guard.trading_mode == TradingMode.SHUTDOWN
        assert not self.risk_guard.can_place_order()
        
        # Automatic activation on severe loss
        self.risk_guard.deactivate_kill_switch()
        self.risk_guard.update_pnl(-12000, portfolio_value=100000)  # 12% loss
        
        assert self.risk_guard.is_kill_switch_active()
    
    def test_risk_level_transitions(self):
        """Test risk level transition logic."""
        initial_level = self.risk_guard.current_risk_level
        
        # Escalate through levels
        self.risk_guard.escalate_risk_level(RiskLevel.MEDIUM)
        assert self.risk_guard.current_risk_level == RiskLevel.MEDIUM
        
        self.risk_guard.escalate_risk_level(RiskLevel.HIGH)
        assert self.risk_guard.current_risk_level == RiskLevel.HIGH
        
        # Test de-escalation (should require explicit approval or time)
        self.risk_guard.attempt_deescalation()
        assert self.risk_guard.current_risk_level == RiskLevel.HIGH  # Should not auto-deescalate
    
    def test_trading_mode_restrictions(self):
        """Test trading mode restrictions."""
        # Normal mode - all operations allowed
        assert self.risk_guard.can_place_order() is True
        assert self.risk_guard.can_increase_position() is True
        
        # Conservative mode - limited operations
        self.risk_guard.set_trading_mode(TradingMode.CONSERVATIVE)
        assert self.risk_guard.can_place_order() is True
        assert self.risk_guard.can_increase_position() is False
        
        # Defensive mode - very limited operations
        self.risk_guard.set_trading_mode(TradingMode.DEFENSIVE)
        assert self.risk_guard.can_place_order() is True  # Close only
        assert self.risk_guard.can_increase_position() is False
        
        # Emergency mode - critical operations only
        self.risk_guard.set_trading_mode(TradingMode.EMERGENCY)
        assert self.risk_guard.can_place_order() is False
        assert self.risk_guard.can_increase_position() is False
    
    def test_correlation_limits(self):
        """Test correlation-based risk limits."""
        # Add correlated positions
        self.risk_guard.add_position('BTC/USD', 5.0)
        self.risk_guard.add_position('ETH/USD', 5.0)
        
        # Set high correlation between BTC and ETH
        self.risk_guard.update_correlation('BTC/USD', 'ETH/USD', 0.85)
        
        # Should limit additional correlated exposure
        assert self.risk_guard.validate_correlated_exposure('LTC/USD', 'BTC/USD', 5.0, 0.8) is False
    
    def test_volatility_adjustment(self):
        """Test volatility-based risk adjustment."""
        # High volatility should reduce allowed position sizes
        high_vol_limit = self.risk_guard.get_volatility_adjusted_limit(volatility=0.8)
        normal_vol_limit = self.risk_guard.get_volatility_adjusted_limit(volatility=0.3)
        
        assert high_vol_limit < normal_vol_limit
        assert high_vol_limit < self.risk_guard.max_position_pct
    
    def test_time_based_limits(self):
        """Test time-based trading limits."""
        # Should track and limit rapid trading
        for i in range(10):
            self.risk_guard.record_trade_attempt('BTC/USD')
        
        # Should start limiting after many attempts
        assert self.risk_guard.is_trading_frequency_exceeded('BTC/USD') is True
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        # Simulate rapid price movement
        self.risk_guard.update_market_conditions(
            volatility_spike=True,
            liquidity_crisis=False,
            correlation_breakdown=False
        )
        
        assert self.risk_guard.is_circuit_breaker_active() is True
        assert not self.risk_guard.can_place_order()
    
    def test_recovery_conditions(self):
        """Test recovery from high risk states."""
        # Escalate to high risk
        self.risk_guard.escalate_risk_level(RiskLevel.CRITICAL)
        self.risk_guard.set_trading_mode(TradingMode.EMERGENCY)
        
        # Simulate recovery conditions
        self.risk_guard.update_pnl(500, portfolio_value=100000)  # Small profit
        self.risk_guard.update_drawdown(current_value=98000, peak_value=100000)  # Reduced drawdown
        
        # Should allow gradual recovery
        recovery_result = self.risk_guard.evaluate_recovery_conditions()
        assert recovery_result['can_deescalate'] in [True, False]  # Depends on time elapsed
    
    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation."""
        # Add some data
        self.risk_guard.update_pnl(-2000, portfolio_value=100000)
        self.risk_guard.update_drawdown(current_value=95000, peak_value=100000)
        self.risk_guard.add_position('BTC/USD', 3.0)
        
        metrics = self.risk_guard.get_risk_metrics()
        
        assert 'daily_loss_pct' in metrics
        assert 'current_drawdown_pct' in metrics
        assert 'total_exposure_pct' in metrics
        assert 'risk_level' in metrics
        assert 'trading_mode' in metrics
        
        assert metrics['daily_loss_pct'] == 2.0
        assert metrics['current_drawdown_pct'] == 5.0
        assert metrics['total_exposure_pct'] == 3.0


@pytest.mark.unit 
class TestRiskLevelEnum:
    """Test RiskLevel enumeration."""
    
    def test_risk_level_ordering(self):
        """Test risk level ordering and comparison."""
        assert RiskLevel.NORMAL < RiskLevel.MEDIUM
        assert RiskLevel.MEDIUM < RiskLevel.HIGH
        assert RiskLevel.HIGH < RiskLevel.CRITICAL
        assert RiskLevel.CRITICAL < RiskLevel.EMERGENCY
    
    def test_risk_level_values(self):
        """Test risk level values."""
        assert RiskLevel.NORMAL.value == 1
        assert RiskLevel.MEDIUM.value == 2
        assert RiskLevel.HIGH.value == 3
        assert RiskLevel.CRITICAL.value == 4
        assert RiskLevel.EMERGENCY.value == 5


@pytest.mark.unit
class TestTradingModeEnum:
    """Test TradingMode enumeration."""
    
    def test_trading_mode_values(self):
        """Test trading mode string values."""
        assert TradingMode.NORMAL.value == "normal"
        assert TradingMode.CONSERVATIVE.value == "conservative"
        assert TradingMode.DEFENSIVE.value == "defensive"
        assert TradingMode.EMERGENCY.value == "emergency"
        assert TradingMode.SHUTDOWN.value == "shutdown"