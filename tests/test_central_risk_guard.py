#!/usr/bin/env python3
"""
Critical Tests for Central Risk Guard
Ensures zero-bypass architecture en mandatory risk enforcement
"""

import pytest
import time
from decimal import Decimal
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch

import sys
sys.path.append('.')

from src.cryptosmarttrader.risk.central_risk_guard import (
    CentralRiskGuard, 
    OrderRequest, 
    RiskDecision, 
    RiskLimits,
    PortfolioState
)


class TestCentralRiskGuard:
    """Critical tests voor Central Risk Guard functionality"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Temporary config directory voor testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def risk_guard(self, temp_config_dir):
        """Fresh risk guard instance voor each test"""
        # Reset singleton
        CentralRiskGuard._instance = None
        
        config_path = temp_config_dir / "risk_limits.json"
        risk_guard = CentralRiskGuard(str(config_path))
        
        # Set test limits
        risk_guard.limits = RiskLimits(
            kill_switch_active=False,
            max_daily_loss_usd=1000.0,
            max_daily_loss_percent=2.0,
            max_drawdown_percent=5.0,
            max_position_count=5,
            max_single_position_usd=10000.0,
            max_total_exposure_usd=25000.0
        )
        
        # Set initial portfolio state
        risk_guard.portfolio_state = PortfolioState(
            total_value_usd=50000.0,
            daily_pnl_usd=0.0,
            max_drawdown_from_peak=0.0,
            position_count=2,
            total_exposure_usd=15000.0
        )
        
        return risk_guard
    
    @pytest.fixture
    def test_order(self):
        """Standard test order"""
        return OrderRequest(
            symbol="ETH",
            side="buy",
            size=1.0,
            price=2000.0,
            order_type="limit"
        )
    
    def test_kill_switch_blocks_all_orders(self, risk_guard, test_order):
        """CRITICAL: Kill switch moet ALL orders blokkeren"""
        # Activate kill switch
        risk_guard.limits.kill_switch_active = True
        
        # Evaluate order
        decision, reason, adjusted_size = risk_guard.evaluate_order(test_order)
        
        # Verify immediate rejection
        assert decision == RiskDecision.EMERGENCY_STOP
        assert "KILL_SWITCH_ACTIVE" in reason
        assert adjusted_size is None
    
    def test_daily_loss_limits_enforcement(self, risk_guard, test_order):
        """CRITICAL: Daily loss limits moeten hard enforced worden"""
        # Set portfolio to exceed daily loss limit
        risk_guard.portfolio_state.daily_pnl_usd = -1500.0  # Exceeds -1000 limit
        
        decision, reason, adjusted_size = risk_guard.evaluate_order(test_order)
        
        assert decision == RiskDecision.REJECT
        assert "DAILY_LOSS_LIMIT" in reason
    
    def test_position_count_limits(self, risk_guard, test_order):
        """CRITICAL: Position count limits enforcement"""
        # Set position count at limit
        risk_guard.portfolio_state.position_count = 5  # At limit
        
        # Order for new symbol should be rejected
        test_order.symbol = "BTC"  # New position
        
        decision, reason, adjusted_size = risk_guard.evaluate_order(test_order)
        
        assert decision == RiskDecision.REJECT
        assert "POSITION_LIMIT" in reason
    
    def test_exposure_limits_with_size_reduction(self, risk_guard, test_order):
        """CRITICAL: Exposure limits met automatic size reduction"""
        # Set exposure near limit
        risk_guard.portfolio_state.total_exposure_usd = 24000.0  # Near 25k limit
        
        # Large order that would exceed limit
        test_order.size = 2.0  # 4000 USD value would exceed limit
        
        decision, reason, adjusted_size = risk_guard.evaluate_order(test_order)
        
        # Should suggest size reduction, not reject
        assert decision == RiskDecision.REDUCE_SIZE
        assert adjusted_size is not None
        assert adjusted_size < test_order.size
    
    def test_data_quality_gate(self, risk_guard, test_order):
        """CRITICAL: Data quality gate enforcement"""
        # Test with stale market data
        stale_market_data = {
            'timestamp': time.time() - 600,  # 10 minutes old
            'price': 2000.0,
            'completeness': 0.8  # Below threshold
        }
        
        decision, reason, adjusted_size = risk_guard.evaluate_order(test_order, stale_market_data)
        
        assert decision == RiskDecision.REJECT
        assert "DATA_QUALITY_FAIL" in reason
    
    def test_drawdown_limits_enforcement(self, risk_guard, test_order):
        """CRITICAL: Drawdown limits enforcement"""
        # Set drawdown at limit
        risk_guard.portfolio_state.max_drawdown_from_peak = 6.0  # Exceeds 5% limit
        
        decision, reason, adjusted_size = risk_guard.evaluate_order(test_order)
        
        assert decision == RiskDecision.REJECT
        assert "DRAWDOWN_LIMIT" in reason
    
    def test_approved_order_conditions(self, risk_guard, test_order):
        """Test conditions waar order approved wordt"""
        # Ensure all limits are within bounds
        risk_guard.portfolio_state = PortfolioState(
            total_value_usd=50000.0,
            daily_pnl_usd=100.0,  # Positive
            max_drawdown_from_peak=1.0,  # Low
            position_count=2,  # Below limit
            total_exposure_usd=5000.0  # Well below limit
        )
        
        # Provide good market data
        good_market_data = {
            'timestamp': time.time(),
            'price': 2000.0,
            'completeness': 1.0,
            'bid_ask_spread': 0.001
        }
        
        decision, reason, adjusted_size = risk_guard.evaluate_order(test_order, good_market_data)
        
        assert decision == RiskDecision.APPROVE
        assert "APPROVED" in reason
        assert adjusted_size is None
    
    def test_emergency_state_persistence(self, risk_guard, test_order):
        """CRITICAL: Emergency state moet persistent zijn"""
        # Trigger emergency stop
        risk_guard.emergency_stop("TEST_EMERGENCY")
        
        # Verify state persists
        assert risk_guard.limits.kill_switch_active is True
        assert risk_guard.kill_switch_reason == "TEST_EMERGENCY"
        
        # Verify emergency state file is created
        assert risk_guard.emergency_state_file.exists()
    
    def test_audit_trail_logging(self, risk_guard, test_order):
        """CRITICAL: Comprehensive audit trail logging"""
        # Make decision that should be logged
        decision, reason, adjusted_size = risk_guard.evaluate_order(test_order)
        
        # Verify audit log file exists en is not empty
        assert risk_guard.audit_log_path.exists()
        
        # Read last log entry
        with open(risk_guard.audit_log_path, 'r') as f:
            log_lines = f.readlines()
            assert len(log_lines) > 0
            
            last_entry = json.loads(log_lines[-1])
            assert last_entry['order_symbol'] == test_order.symbol
            assert last_entry['decision'] == decision.value
            assert last_entry['reason'] == reason
    
    def test_performance_metrics_tracking(self, risk_guard, test_order):
        """Test performance metrics tracking"""
        initial_count = risk_guard.evaluation_count
        
        # Execute multiple evaluations
        for _ in range(5):
            risk_guard.evaluate_order(test_order)
        
        # Verify metrics updated
        assert risk_guard.evaluation_count == initial_count + 5
        assert risk_guard.total_evaluation_time > 0
    
    def test_singleton_enforcement(self, temp_config_dir):
        """CRITICAL: Singleton pattern enforcement"""
        # Reset singleton
        CentralRiskGuard._instance = None
        
        config_path = temp_config_dir / "risk_limits.json"
        
        # Create two instances
        guard1 = CentralRiskGuard(str(config_path))
        guard2 = CentralRiskGuard(str(config_path))
        
        # Verify they are the same instance
        assert guard1 is guard2
    
    def test_concurrent_access_thread_safety(self, risk_guard, test_order):
        """CRITICAL: Thread safety voor concurrent access"""
        import threading
        
        results = []
        errors = []
        
        def evaluate_order_thread():
            try:
                decision, reason, size = risk_guard.evaluate_order(test_order)
                results.append((decision, reason, size))
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=evaluate_order_thread)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify no errors and consistent results
        assert len(errors) == 0
        assert len(results) == 10
        
        # All results should be consistent (same decision)
        first_decision = results[0][0]
        for decision, _, _ in results:
            assert decision == first_decision


class TestRiskLimitsConfiguration:
    """Test risk limits configuration en persistence"""
    
    def test_default_limits_creation(self, temp_config_dir):
        """Test creation of default limits"""
        CentralRiskGuard._instance = None
        config_path = temp_config_dir / "new_config.json"
        
        guard = CentralRiskGuard(str(config_path))
        
        # Verify config file was created
        assert config_path.exists()
        
        # Verify default values
        assert guard.limits.max_daily_loss_usd == 5000.0
        assert guard.limits.max_position_count == 10
    
    def test_limits_persistence(self, temp_config_dir):
        """Test limits persistence across instances"""
        CentralRiskGuard._instance = None
        config_path = temp_config_dir / "persist_config.json"
        
        # Create initial instance met custom limits
        guard1 = CentralRiskGuard(str(config_path))
        guard1.limits.max_daily_loss_usd = 2500.0
        guard1._save_risk_limits(guard1.limits)
        
        # Create new instance
        CentralRiskGuard._instance = None
        guard2 = CentralRiskGuard(str(config_path))
        
        # Verify limits persisted
        assert guard2.limits.max_daily_loss_usd == 2500.0


class TestDataQualityValidation:
    """Test data quality validation logic"""
    
    def test_data_age_validation(self):
        """Test data age validation"""
        guard = CentralRiskGuard()
        
        # Test fresh data
        fresh_data = {'timestamp': time.time()}
        result = guard._check_data_quality(fresh_data)
        assert result[0] is True
        
        # Test stale data
        stale_data = {'timestamp': time.time() - 400}  # ~7 minutes old
        result = guard._check_data_quality(stale_data)
        assert result[0] is False
        assert "DATA_AGE" in result[1]
    
    def test_data_completeness_validation(self):
        """Test data completeness validation"""
        guard = CentralRiskGuard()
        
        # Test complete data
        complete_data = {
            'timestamp': time.time(),
            'completeness': 0.98
        }
        result = guard._check_data_quality(complete_data)
        assert result[0] is True
        
        # Test incomplete data
        incomplete_data = {
            'timestamp': time.time(),
            'completeness': 0.90  # Below 0.95 threshold
        }
        result = guard._check_data_quality(incomplete_data)
        assert result[0] is False
        assert "DATA_COMPLETENESS" in result[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])