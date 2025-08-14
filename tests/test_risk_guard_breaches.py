"""
Unit tests for RiskGuard breach simulation - FASE C compliance testing
Tests kill-switch, daily loss limits, data gaps, and duplicate order detection
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import tempfile
import os

from src.cryptosmarttrader.risk.central_risk_guard import (
    CentralRiskGuard, 
    RiskDecision, 
    OrderRequest, 
    RiskLimits, 
    PortfolioState
)


class TestRiskGuardBreaches:
    """Test breach simulation scenarios for RiskGuard enforcement"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture  
    def risk_guard(self, temp_config_dir):
        """Create RiskGuard instance with test configuration"""
        config_path = os.path.join(temp_config_dir, "risk_limits.json")
        
        # Reset singleton for each test
        CentralRiskGuard._instance = None
        
        guard = CentralRiskGuard(config_path)
        
        # Set test limits
        guard.limits = RiskLimits(
            kill_switch_active=False,
            max_daily_loss_usd=1000.0,
            max_daily_loss_percent=2.0, 
            max_drawdown_percent=5.0,
            max_position_count=3,
            max_single_position_usd=5000.0,
            max_total_exposure_usd=10000.0,
            min_data_completeness=0.95,
            max_data_age_minutes=5
        )
        
        # Set portfolio state
        guard.portfolio_state = PortfolioState(
            total_value_usd=50000.0,
            daily_pnl_usd=0.0,
            max_drawdown_from_peak=0.0,
            position_count=0,
            total_exposure_usd=0.0
        )
        
        return guard
    
    @pytest.fixture
    def sample_order(self):
        """Create sample order for testing"""
        return OrderRequest(
            symbol="BTC/USD",
            side="buy",
            size=0.1,
            price=50000.0,
            client_order_id="test_order_001"
        )

    def test_kill_switch_blocks_all_orders(self, risk_guard, sample_order):
        """Test that kill switch blocks ALL orders"""
        # Activate kill switch
        risk_guard.trigger_kill_switch("Test emergency stop")
        
        # Attempt to place order
        decision, reason, adjusted_size = risk_guard.evaluate_order(sample_order)
        
        assert decision == RiskDecision.EMERGENCY_STOP
        assert "KILL_SWITCH_ACTIVE" in reason
        assert adjusted_size is None
        assert risk_guard.limits.kill_switch_active is True
        
        # Verify emergency state file was created
        assert risk_guard.emergency_state_file.exists()
        
        with open(risk_guard.emergency_state_file, 'r') as f:
            emergency_state = json.load(f)
            assert emergency_state['reason'] == "Test emergency stop"
            assert 'triggered_at' in emergency_state
    
    def test_daily_loss_limit_breach(self, risk_guard, sample_order):
        """Test daily loss limit enforcement"""
        # Set portfolio to exceed daily loss limit
        risk_guard.portfolio_state.daily_pnl_usd = -1500.0  # Exceeds $1000 limit
        
        decision, reason, adjusted_size = risk_guard.evaluate_order(sample_order)
        
        assert decision == RiskDecision.REJECT
        assert "DAILY_LOSS_LIMIT" in reason
        assert "-1500.0" in reason
        assert adjusted_size is None
    
    def test_daily_loss_percentage_breach(self, risk_guard, sample_order):
        """Test daily loss percentage limit enforcement"""
        # Set portfolio to exceed 2% daily loss
        risk_guard.portfolio_state.daily_pnl_usd = -1200.0  # 2.4% of $50k
        
        decision, reason, adjusted_size = risk_guard.evaluate_order(sample_order)
        
        assert decision == RiskDecision.REJECT
        assert "DAILY_LOSS_LIMIT" in reason
        assert "2.4%" in reason
    
    def test_max_drawdown_breach(self, risk_guard, sample_order):
        """Test maximum drawdown limit enforcement"""
        # Set portfolio to exceed 5% max drawdown
        risk_guard.portfolio_state.max_drawdown_from_peak = 6.0  # 6% drawdown
        
        decision, reason, adjusted_size = risk_guard.evaluate_order(sample_order)
        
        assert decision == RiskDecision.REJECT
        assert "DRAWDOWN_LIMIT" in reason
        assert "6.0%" in reason
    
    def test_position_count_breach(self, risk_guard, sample_order):
        """Test position count limit enforcement"""
        # Set portfolio to max position count
        risk_guard.portfolio_state.position_count = 3  # At limit
        risk_guard.portfolio_state.positions = {
            "BTC/USD": {"size": 0.1},
            "ETH/USD": {"size": 1.0}, 
            "ADA/USD": {"size": 100.0}
        }
        
        # Try to open new position
        new_order = OrderRequest(
            symbol="DOT/USD",  # New symbol
            side="buy",
            size=10.0,
            price=30.0,
            client_order_id="test_order_002"
        )
        
        decision, reason, adjusted_size = risk_guard.evaluate_order(new_order)
        
        assert decision == RiskDecision.REJECT
        assert "POSITION_LIMIT" in reason
        assert "3" in reason
    
    def test_total_exposure_breach_with_size_reduction(self, risk_guard, sample_order):
        """Test total exposure limit with automatic size reduction"""
        # Set current exposure near limit
        risk_guard.portfolio_state.total_exposure_usd = 8000.0  # $8k of $10k limit
        
        # Order would exceed limit (0.1 * $50k = $5k, total would be $13k)
        decision, reason, adjusted_size = risk_guard.evaluate_order(sample_order)
        
        # Should offer size reduction
        if decision == RiskDecision.REDUCE_SIZE:
            assert adjusted_size is not None
            assert adjusted_size < sample_order.size
            assert "EXPOSURE_LIMIT" in reason
        else:
            # Might reject if reduction not viable
            assert decision == RiskDecision.REJECT
    
    def test_single_position_size_breach(self, risk_guard, sample_order):
        """Test single position size limit enforcement"""
        # Create order exceeding single position limit
        large_order = OrderRequest(
            symbol="BTC/USD",
            side="buy", 
            size=0.15,  # 0.15 * $50k = $7.5k > $5k limit
            price=50000.0,
            client_order_id="test_order_large"
        )
        
        decision, reason, adjusted_size = risk_guard.evaluate_order(large_order)
        
        if decision == RiskDecision.REDUCE_SIZE:
            assert adjusted_size is not None
            assert adjusted_size < large_order.size
            estimated_value = adjusted_size * large_order.price
            assert estimated_value <= risk_guard.limits.max_single_position_usd
        else:
            assert decision == RiskDecision.REJECT
            assert "POSITION_SIZE_LIMIT" in reason
    
    def test_data_gap_breach(self, risk_guard, sample_order):
        """Test data gap detection and order blocking"""
        # Simulate stale market data
        stale_market_data = {
            'timestamp': time.time() - 400,  # 6+ minutes old (> 5 min limit)
            'completeness': 0.98,
            'bid': 49000.0,
            'ask': 50000.0
        }
        
        decision, reason, adjusted_size = risk_guard.evaluate_order(
            sample_order, 
            market_data=stale_market_data
        )
        
        assert decision == RiskDecision.REJECT
        assert "DATA_QUALITY_FAIL" in reason
        assert "stale" in reason.lower() or "age" in reason.lower()
    
    def test_data_completeness_breach(self, risk_guard, sample_order):
        """Test data completeness threshold enforcement"""
        # Simulate incomplete market data
        incomplete_market_data = {
            'timestamp': time.time(),
            'completeness': 0.90,  # Below 95% requirement
            'bid': 49000.0,
            'ask': 50000.0
        }
        
        decision, reason, adjusted_size = risk_guard.evaluate_order(
            sample_order,
            market_data=incomplete_market_data
        )
        
        assert decision == RiskDecision.REJECT
        assert "DATA_QUALITY_FAIL" in reason
        assert "completeness" in reason.lower()
    
    def test_duplicate_order_detection(self, risk_guard):
        """Test duplicate order blocking"""
        order1 = OrderRequest(
            symbol="BTC/USD",
            side="buy",
            size=0.1,
            price=50000.0,
            client_order_id="duplicate_test_001"
        )
        
        order2 = OrderRequest(
            symbol="BTC/USD", 
            side="buy",
            size=0.2,  # Different size
            price=51000.0,  # Different price
            client_order_id="duplicate_test_001"  # Same COID - this is the key
        )
        
        # First order should be approved (assuming no other breaches)
        decision1, reason1, _ = risk_guard.evaluate_order(order1)
        
        # Second order with same COID should be rejected
        decision2, reason2, _ = risk_guard.evaluate_order(order2)
        
        assert decision2 == RiskDecision.REJECT
        assert "duplicate" in reason2.lower() or "duplicate" in reason2
    
    def test_multiple_breach_conditions(self, risk_guard, sample_order):
        """Test behavior when multiple breach conditions exist"""
        # Set multiple breach conditions
        risk_guard.portfolio_state.daily_pnl_usd = -1200.0  # Loss limit breach
        risk_guard.portfolio_state.position_count = 3  # Position count at limit
        risk_guard.portfolio_state.total_exposure_usd = 9500.0  # Near exposure limit
        
        decision, reason, adjusted_size = risk_guard.evaluate_order(sample_order)
        
        # Should fail on first breach encountered (likely daily loss)
        assert decision == RiskDecision.REJECT
        assert "DAILY_LOSS_LIMIT" in reason  # First gate to fail
    
    def test_kill_switch_auto_trigger_on_critical_error(self, risk_guard, sample_order):
        """Test that critical errors auto-trigger kill switch"""
        # Mock a critical error in risk evaluation
        with patch.object(risk_guard, '_check_data_quality', side_effect=Exception("Critical system error")):
            decision, reason, adjusted_size = risk_guard.evaluate_order(sample_order)
            
            assert decision == RiskDecision.EMERGENCY_STOP
            assert "RISK_SYSTEM_ERROR" in reason
            assert risk_guard.limits.kill_switch_active is True
            assert risk_guard.kill_switch_reason == "Risk system error: Critical system error"
    
    def test_breach_audit_logging(self, risk_guard, sample_order):
        """Test that breaches are properly logged for audit"""
        # Set up breach condition
        risk_guard.portfolio_state.daily_pnl_usd = -1500.0
        
        # Ensure audit log directory exists
        risk_guard.audit_log_path.parent.mkdir(exist_ok=True)
        
        # Execute order that will be rejected
        decision, reason, adjusted_size = risk_guard.evaluate_order(sample_order)
        
        assert decision == RiskDecision.REJECT
        
        # Verify audit log was written
        assert risk_guard.audit_log_path.exists()
        
        # Read and verify log content
        with open(risk_guard.audit_log_path, 'r') as f:
            log_lines = f.readlines()
            
        assert len(log_lines) > 0
        
        last_log = json.loads(log_lines[-1])
        assert last_log['decision'] == 'reject'
        assert last_log['symbol'] == sample_order.symbol
        assert last_log['client_order_id'] == sample_order.client_order_id
        assert 'DAILY_LOSS_LIMIT' in last_log['reason']
    
    def test_risk_metrics_tracking(self, risk_guard, sample_order):
        """Test that risk metrics are properly tracked"""
        initial_metrics = risk_guard.get_risk_metrics()
        initial_count = initial_metrics['evaluation_count']
        initial_rejections = initial_metrics['rejection_count']
        
        # Set up breach to force rejection
        risk_guard.portfolio_state.daily_pnl_usd = -1500.0
        
        # Execute order
        decision, reason, adjusted_size = risk_guard.evaluate_order(sample_order)
        
        assert decision == RiskDecision.REJECT
        
        # Check updated metrics
        updated_metrics = risk_guard.get_risk_metrics()
        
        assert updated_metrics['evaluation_count'] == initial_count + 1
        assert updated_metrics['rejection_count'] == initial_rejections + 1
        assert updated_metrics['rejection_rate'] > initial_metrics['rejection_rate']
        assert updated_metrics['avg_evaluation_time_ms'] > 0
    
    def test_singleton_enforcement(self, temp_config_dir):
        """Test that RiskGuard maintains singleton pattern"""
        config_path = os.path.join(temp_config_dir, "risk_limits.json")
        
        # Reset singleton
        CentralRiskGuard._instance = None
        
        guard1 = CentralRiskGuard(config_path)
        guard2 = CentralRiskGuard(config_path)
        
        assert guard1 is guard2
        assert id(guard1) == id(guard2)
        
        # Test that state changes are shared
        guard1.trigger_kill_switch("Test singleton")
        assert guard2.limits.kill_switch_active is True


@pytest.mark.integration 
class TestRiskGuardIntegration:
    """Integration tests for RiskGuard with other systems"""
    
    def test_risk_guard_execution_policy_integration(self):
        """Test RiskGuard integration with ExecutionPolicy"""
        # This would test the integration between RiskGuard and ExecutionPolicy
        # to ensure orders are properly blocked at the execution level
        pass
    
    def test_risk_guard_portfolio_update_integration(self):
        """Test RiskGuard integration with portfolio state updates"""
        # This would test real-time portfolio state updates
        # and their impact on risk evaluations
        pass


if __name__ == "__main__":
    # Run specific breach tests
    pytest.main([__file__, "-v", "--tb=short"])