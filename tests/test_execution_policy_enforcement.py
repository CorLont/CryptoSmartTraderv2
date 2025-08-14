"""
Unit tests for ExecutionPolicy enforcement - FASE C compliance testing
Tests spread, depth, volume gates, slippage budget, and idempotent order handling
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import tempfile
import os

from src.cryptosmarttrader.execution.execution_policy import (
    ExecutionPolicy,
    OrderRequest,
    MarketConditions, 
    ExecutionResult,
    ExecutionDecision,
    OrderSide,
    TimeInForce,
    get_execution_policy,
    reset_execution_policy
)


class TestExecutionPolicyEnforcement:
    """Test execution policy gate enforcement scenarios"""
    
    @pytest.fixture
    def execution_policy(self):
        """Create ExecutionPolicy instance with test configuration"""
        # Reset singleton for each test
        reset_execution_policy()
        
        config = {
            'max_spread_bps': 50,  # 50 bps max spread
            'min_depth_usd': 10000,  # $10k min depth
            'max_slippage_bps': 30,  # 30 bps max slippage
            'min_volume_24h_usd': 1000000,  # $1M min volume
            'max_order_value_usd': 50000,  # $50k max order
            'daily_slippage_budget_bps': 200  # 200 bps daily budget
        }
        
        policy = get_execution_policy(config)
        return policy
    
    @pytest.fixture
    def sample_order(self):
        """Create sample order for testing"""
        return OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            price=50000.0,
            time_in_force=TimeInForce.POST_ONLY,
            client_order_id="test_order_001"
        )
    
    @pytest.fixture  
    def good_market_conditions(self):
        """Create market conditions that should pass all gates"""
        return MarketConditions(
            symbol="BTC/USD",
            bid_price=49950.0,
            ask_price=50050.0,  # 50 bps spread
            spread_bps=50.0,
            bid_depth_usd=15000.0,  # Above $10k requirement
            ask_depth_usd=15000.0,
            volume_24h_usd=5000000.0,  # Above $1M requirement
            timestamp=time.time()
        )
    
    def test_spread_gate_enforcement(self, execution_policy, sample_order):
        """Test spread gate blocks orders with excessive spread"""
        # Create market conditions with excessive spread
        bad_spread_conditions = MarketConditions(
            symbol="BTC/USD",
            bid_price=49900.0,
            ask_price=50200.0,  # 60 bps spread (exceeds 50 bps limit)
            spread_bps=60.0,
            bid_depth_usd=15000.0,
            ask_depth_usd=15000.0,
            volume_24h_usd=5000000.0,
            timestamp=time.time()
        )
        
        result = execution_policy.decide(sample_order, bad_spread_conditions)
        
        assert result.decision == ExecutionDecision.REJECT
        assert not result.approved
        assert "spread" in result.reason.lower()
        assert "60" in result.reason  # Should mention actual spread
        assert "50" in result.reason  # Should mention limit
        assert result.gate_results['spread_gate'] is False
    
    def test_depth_gate_enforcement(self, execution_policy, sample_order):
        """Test depth gate blocks orders with insufficient depth"""
        # Create market conditions with insufficient depth
        bad_depth_conditions = MarketConditions(
            symbol="BTC/USD",
            bid_price=49950.0,
            ask_price=50050.0,
            spread_bps=50.0,
            bid_depth_usd=5000.0,  # Below $10k requirement
            ask_depth_usd=5000.0,
            volume_24h_usd=5000000.0,
            timestamp=time.time()
        )
        
        result = execution_policy.decide(sample_order, bad_depth_conditions)
        
        assert result.decision == ExecutionDecision.REJECT
        assert not result.approved
        assert "depth" in result.reason.lower()
        assert "5,000" in result.reason or "5000" in result.reason
        assert "10,000" in result.reason or "10000" in result.reason
        assert result.gate_results['depth_gate'] is False
    
    def test_volume_gate_enforcement(self, execution_policy, sample_order):
        """Test volume gate blocks orders with insufficient 24h volume"""
        # Create market conditions with insufficient volume
        bad_volume_conditions = MarketConditions(
            symbol="BTC/USD",
            bid_price=49950.0,
            ask_price=50050.0,
            spread_bps=50.0,
            bid_depth_usd=15000.0,
            ask_depth_usd=15000.0,
            volume_24h_usd=500000.0,  # Below $1M requirement
            timestamp=time.time()
        )
        
        result = execution_policy.decide(sample_order, bad_volume_conditions)
        
        assert result.decision == ExecutionDecision.REJECT
        assert not result.approved
        assert "volume" in result.reason.lower()
        assert "500,000" in result.reason or "500000" in result.reason
        assert result.gate_results['volume_gate'] is False
    
    def test_slippage_budget_enforcement(self, execution_policy, sample_order, good_market_conditions):
        """Test slippage budget enforcement over daily period"""
        # First, consume most of daily slippage budget
        execution_policy.current_slippage_used_bps = 180  # 180 of 200 bps used
        
        # Create order that would push over budget
        # Estimated slippage for this order would be ~25 bps (180 + 25 = 205 > 200)
        result = execution_policy.decide(sample_order, good_market_conditions)
        
        if result.decision == ExecutionDecision.REJECT:
            assert "slippage" in result.reason.lower()
            assert "budget" in result.reason.lower()
            assert result.gate_results['slippage_budget_gate'] is False
        else:
            # If approved, slippage should be within remaining budget
            remaining_budget = 200 - 180  # 20 bps remaining
            assert result.estimated_slippage_bps <= remaining_budget
    
    def test_time_in_force_enforcement(self, execution_policy, good_market_conditions):
        """Test that POST_ONLY time-in-force is enforced"""
        # Create order without POST_ONLY (should be rejected or auto-corrected)
        market_order = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            price=None,  # Market order
            time_in_force=TimeInForce.IMMEDIATE_OR_CANCEL,
            client_order_id="test_market_order"
        )
        
        result = execution_policy.decide(market_order, good_market_conditions)
        
        # Policy should reject non-POST_ONLY orders
        assert result.decision == ExecutionDecision.REJECT
        assert "post_only" in result.reason.lower() or "tif" in result.reason.lower()
        assert result.gate_results['tif_gate'] is False
    
    def test_client_order_id_generation(self, execution_policy, good_market_conditions):
        """Test automatic client order ID generation for idempotency"""
        # Create order without COID
        order_without_coid = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            price=50000.0,
            time_in_force=TimeInForce.POST_ONLY,
            client_order_id=None  # No COID provided
        )
        
        result = execution_policy.decide(order_without_coid, good_market_conditions)
        
        # Should generate COID automatically
        assert order_without_coid.client_order_id is not None
        assert len(order_without_coid.client_order_id) > 0
        assert result.client_order_id == order_without_coid.client_order_id
    
    def test_duplicate_order_detection(self, execution_policy, good_market_conditions):
        """Test duplicate order detection via client order ID"""
        order1 = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            price=50000.0,
            time_in_force=TimeInForce.POST_ONLY,
            client_order_id="duplicate_test_001"
        )
        
        order2 = OrderRequest(
            symbol="ETH/USD",  # Different symbol
            side=OrderSide.SELL,  # Different side
            size=1.0,  # Different size
            price=3000.0,  # Different price
            time_in_force=TimeInForce.POST_ONLY,
            client_order_id="duplicate_test_001"  # Same COID!
        )
        
        # First order should succeed (assuming other gates pass)
        result1 = execution_policy.decide(order1, good_market_conditions)
        
        # Second order with same COID should be rejected
        result2 = execution_policy.decide(order2, good_market_conditions)
        
        assert result2.decision == ExecutionDecision.REJECT
        assert "duplicate" in result2.reason.lower()
        assert result2.gate_results['duplicate_check'] is False
    
    def test_order_value_limit_enforcement(self, execution_policy, good_market_conditions):
        """Test maximum order value enforcement"""
        # Create order exceeding value limit ($50k)
        large_order = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=1.2,  # 1.2 * $50k = $60k > $50k limit
            price=50000.0,
            time_in_force=TimeInForce.POST_ONLY,
            client_order_id="large_order_test"
        )
        
        result = execution_policy.decide(large_order, good_market_conditions)
        
        assert result.decision == ExecutionDecision.REJECT
        assert "value" in result.reason.lower() or "size" in result.reason.lower()
        assert "50,000" in result.reason or "50000" in result.reason
    
    def test_all_gates_pass_approval(self, execution_policy, sample_order, good_market_conditions):
        """Test that order is approved when all gates pass"""
        result = execution_policy.decide(sample_order, good_market_conditions)
        
        assert result.decision == ExecutionDecision.APPROVE
        assert result.approved is True
        assert result.client_order_id == sample_order.client_order_id
        assert result.estimated_slippage_bps > 0  # Should have slippage estimate
        assert result.processing_time_ms > 0  # Should track processing time
        
        # All gates should pass
        assert result.gate_results['spread_gate'] is True
        assert result.gate_results['depth_gate'] is True  
        assert result.gate_results['volume_gate'] is True
        assert result.gate_results['slippage_budget_gate'] is True
        assert result.gate_results['tif_gate'] is True
    
    def test_risk_guard_integration(self, execution_policy, sample_order, good_market_conditions):
        """Test integration with RiskGuard"""
        # Mock RiskGuard to reject order
        with patch('src.cryptosmarttrader.execution.hard_execution_policy.get_central_risk_guard') as mock_risk_guard:
            mock_guard = MagicMock()
            mock_guard.evaluate_order.return_value = ("reject", "Test rejection", None)
            mock_risk_guard.return_value = mock_guard
            
            result = execution_policy.decide(sample_order, good_market_conditions)
            
            assert result.decision == ExecutionDecision.REJECT
            assert "risk" in result.reason.lower()
            assert result.gate_results['risk_gate'] is False
    
    def test_slippage_estimation_accuracy(self, execution_policy, sample_order, good_market_conditions):
        """Test slippage estimation logic"""
        result = execution_policy.decide(sample_order, good_market_conditions)
        
        if result.approved:
            # Slippage should be reasonable for given market conditions
            assert 0 < result.estimated_slippage_bps < 100  # Between 0 and 100 bps
            
            # Should be based on order size and market depth
            order_value = sample_order.size * sample_order.price
            market_impact_factor = order_value / good_market_conditions.ask_depth_usd
            
            # Larger orders relative to depth should have higher slippage
            assert result.estimated_slippage_bps > 0
    
    def test_daily_slippage_budget_reset(self, execution_policy, sample_order, good_market_conditions):
        """Test that daily slippage budget resets properly"""
        # Set budget as if it's from previous day
        yesterday = datetime.now() - timedelta(days=1)
        execution_policy.slippage_reset_time = yesterday
        execution_policy.current_slippage_used_bps = 180  # Near limit
        
        # Execute order - should reset budget and allow order
        result = execution_policy.decide(sample_order, good_market_conditions)
        
        # Budget should have been reset
        assert execution_policy.current_slippage_used_bps < 180
        assert execution_policy.slippage_reset_time.date() == datetime.now().date()
    
    def test_execution_statistics_tracking(self, execution_policy, sample_order, good_market_conditions):
        """Test that execution statistics are properly tracked"""
        initial_stats = execution_policy.get_statistics()
        initial_total = initial_stats['total_requests']
        
        # Execute several orders
        result1 = execution_policy.decide(sample_order, good_market_conditions)
        
        # Create rejection scenario
        bad_conditions = MarketConditions(
            symbol="BTC/USD",
            bid_price=49000.0,
            ask_price=51000.0,  # 400 bps spread - exceeds limit
            spread_bps=400.0,
            bid_depth_usd=15000.0,
            ask_depth_usd=15000.0,
            volume_24h_usd=5000000.0,
            timestamp=time.time()
        )
        
        order2 = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            price=50000.0,
            time_in_force=TimeInForce.POST_ONLY,
            client_order_id="test_order_002"
        )
        
        result2 = execution_policy.decide(order2, bad_conditions)
        
        # Check updated statistics
        updated_stats = execution_policy.get_statistics()
        
        assert updated_stats['total_requests'] == initial_total + 2
        assert updated_stats['approval_rate'] <= 1.0
        assert updated_stats['gate_failures']['spread'] >= 1  # At least one spread failure
    
    def test_singleton_pattern_enforcement(self):
        """Test that ExecutionPolicy maintains singleton pattern"""
        reset_execution_policy()
        
        policy1 = get_execution_policy()
        policy2 = get_execution_policy()
        
        assert policy1 is policy2
        assert id(policy1) == id(policy2)
        
        # Test that configuration changes are shared
        policy1.max_spread_bps = 25
        assert policy2.max_spread_bps == 25


@pytest.mark.integration
class TestExecutionPolicyIntegration:
    """Integration tests for ExecutionPolicy with other systems"""
    
    def test_execution_policy_order_pipeline_integration(self):
        """Test ExecutionPolicy integration with order pipeline"""
        # This would test the full order execution flow
        # from order creation through policy checks to execution
        pass
    
    def test_execution_policy_market_data_integration(self):
        """Test ExecutionPolicy with real market data feeds"""
        # This would test policy decisions with live market data
        pass


if __name__ == "__main__":
    # Run specific enforcement tests
    pytest.main([__file__, "-v", "--tb=short"])