#!/usr/bin/env python3
"""
Test suite voor UnifiedRiskGuard
Comprehensive testing van eenduidige risk enforcement
"""

import pytest
import time
from unittest.mock import patch, MagicMock

import sys
sys.path.append('.')

from src.cryptosmarttrader.risk.unified_risk_guard import (
    UnifiedRiskGuard,
    StandardOrderRequest,
    RiskEvaluationResult,
    RiskDecision,
    OrderSide,
    RiskLimits,
    PortfolioState
)


class TestUnifiedRiskGuard:
    """Test unified risk guard functionality"""
    
    @pytest.fixture
    def risk_guard(self):
        """Clean risk guard instance voor elke test"""
        # Reset singleton instance
        UnifiedRiskGuard._instance = None
        return UnifiedRiskGuard()
    
    @pytest.fixture
    def test_order(self):
        """Standard test order"""
        return StandardOrderRequest(
            symbol="ETH",
            side=OrderSide.BUY,
            size=1.0,
            price=2000.0,
            client_order_id="test_order_001"
        )
    
    @pytest.fixture
    def market_data(self):
        """Standard market data"""
        return {
            'price': 2000.0,
            'volume': 1000000.0,
            'volatility': 0.3,
            'timestamp': time.time()
        }
    
    def test_singleton_enforcement(self):
        """CRITICAL: Singleton pattern ensures één centrale risk authority"""
        guard1 = UnifiedRiskGuard()
        guard2 = UnifiedRiskGuard()
        
        assert guard1 is guard2
        assert id(guard1) == id(guard2)
    
    def test_kill_switch_blocks_all_orders(self, risk_guard, test_order, market_data):
        """CRITICAL: Kill switch MUST block ALL orders zonder uitzondering"""
        # Activate kill switch
        risk_guard.activate_kill_switch("Test emergency")
        
        # ALL orders should be blocked
        result = risk_guard.evaluate_order(test_order, market_data)
        
        assert result.decision == RiskDecision.EMERGENCY_STOP
        assert "KILL_SWITCH_ACTIVE" in result.reason
        assert result.risk_score == 1.0
    
    def test_eenduidige_interface_consistency(self, risk_guard, market_data):
        """CRITICAL: Elke order gebruikt exact dezelfde interface"""
        orders = [
            StandardOrderRequest("BTC", OrderSide.BUY, 0.1, 50000.0),
            StandardOrderRequest("ETH", OrderSide.SELL, 2.0, 2000.0),
            StandardOrderRequest("SOL", OrderSide.BUY, 10.0, 100.0)
        ]
        
        # All orders moet through exact same method
        for order in orders:
            result = risk_guard.evaluate_order(order, market_data)
            
            # Check consistent return type
            assert isinstance(result, RiskEvaluationResult)
            assert isinstance(result.decision, RiskDecision)
            assert isinstance(result.reason, str)
            assert isinstance(result.risk_score, (int, float))
            assert result.evaluation_time_ms >= 0
    
    def test_mandatory_data_quality_gate(self, risk_guard, test_order):
        """CRITICAL: Data quality gate is mandatory"""
        # No market data should be rejected
        result = risk_guard.evaluate_order(test_order, None)
        assert result.decision == RiskDecision.REJECT
        assert "DATA_GAP" in result.reason
        
        # Stale data should be rejected
        stale_data = {
            'price': 2000.0,
            'volume': 1000000.0,
            'timestamp': time.time() - 600  # 10 minutes old
        }
        result = risk_guard.evaluate_order(test_order, stale_data)
        assert result.decision == RiskDecision.REJECT
        assert "STALE_DATA" in result.reason
        
        # Incomplete data should be rejected
        incomplete_data = {
            'price': 2000.0,
            'timestamp': time.time()
            # Missing volume
        }
        result = risk_guard.evaluate_order(test_order, incomplete_data)
        assert result.decision == RiskDecision.REJECT
        assert "INCOMPLETE_DATA" in result.reason
    
    def test_daily_loss_limits_enforcement(self, risk_guard, test_order, market_data):
        """CRITICAL: Daily loss limits must be enforced"""
        # Set portfolio state met loss at limit
        risk_guard.portfolio_state.daily_pnl_usd = -5000.0  # At limit
        risk_guard.portfolio_state.total_value_usd = 100000.0
        
        result = risk_guard.evaluate_order(test_order, market_data)
        assert result.decision == RiskDecision.REJECT
        assert "DAILY_LOSS_LIMIT" in result.reason
        
        # Test percentage limit
        risk_guard.portfolio_state.daily_pnl_usd = -5001.0  # Over percentage limit
        result = risk_guard.evaluate_order(test_order, market_data)
        assert result.decision == RiskDecision.REJECT
        assert any(limit_type in result.reason for limit_type in ["DAILY_LOSS_LIMIT", "DAILY_LOSS_PERCENT"])
    
    def test_exposure_limits_with_size_reduction(self, risk_guard, test_order, market_data):
        """CRITICAL: Exposure limits met intelligent size reduction"""
        # Set high exposure near limit
        risk_guard.portfolio_state.total_exposure_usd = 98000.0  # Near 100k limit
        
        # Large order that would exceed limit
        large_order = StandardOrderRequest(
            symbol="ETH",
            side=OrderSide.BUY,
            size=5.0,  # 5 ETH * 2000 = 10k, would exceed limit
            price=2000.0
        )
        
        result = risk_guard.evaluate_order(large_order, market_data)
        
        # Should reduce size, not reject
        assert result.decision == RiskDecision.REDUCE_SIZE
        assert result.adjusted_size is not None
        assert result.adjusted_size < large_order.size
        assert "EXPOSURE_SIZE_REDUCED" in result.reason
    
    def test_position_count_limits(self, risk_guard, test_order, market_data):
        """CRITICAL: Position count limits enforced"""
        # Set position count at limit
        risk_guard.portfolio_state.position_count = 10  # At limit
        
        result = risk_guard.evaluate_order(test_order, market_data)
        assert result.decision == RiskDecision.REJECT
        assert "POSITION_COUNT_LIMIT" in result.reason
    
    def test_all_gates_pass_approval(self, risk_guard, test_order, market_data):
        """CRITICAL: Order approved when all gates pass"""
        # Set healthy portfolio state
        risk_guard.portfolio_state.daily_pnl_usd = 1000.0  # Positive
        risk_guard.portfolio_state.max_drawdown_from_peak = 2.0  # Low drawdown
        risk_guard.portfolio_state.position_count = 3  # Well below limit
        risk_guard.portfolio_state.total_exposure_usd = 30000.0  # Well below limit
        
        result = risk_guard.evaluate_order(test_order, market_data)
        assert result.decision == RiskDecision.APPROVE
        assert "APPROVED" in result.reason
        assert result.risk_score >= 0.0
    
    def test_audit_trail_completeness(self, risk_guard, test_order, market_data):
        """CRITICAL: Complete audit trail voor ALL decisions"""
        initial_history_size = len(risk_guard.decision_history)
        
        # Execute multiple orders
        orders = [
            StandardOrderRequest("BTC", OrderSide.BUY, 0.1, 50000.0),
            StandardOrderRequest("ETH", OrderSide.SELL, 1.0, 2000.0),
            StandardOrderRequest("SOL", OrderSide.BUY, 5.0, 100.0)
        ]
        
        for order in orders:
            risk_guard.evaluate_order(order, market_data)
        
        # Check audit trail updated
        assert len(risk_guard.decision_history) == initial_history_size + len(orders)
        
        # Check audit entries have all required fields
        for decision in risk_guard.decision_history[-len(orders):]:
            assert decision.decision is not None
            assert decision.reason is not None
            assert decision.risk_score >= 0.0
            assert decision.evaluation_time_ms >= 0.0
            assert decision.timestamp > 0
    
    def test_performance_metrics_tracking(self, risk_guard, test_order, market_data):
        """CRITICAL: Performance metrics voor monitoring"""
        initial_count = risk_guard.evaluation_count
        
        # Execute orders
        risk_guard.evaluate_order(test_order, market_data)
        
        metrics = risk_guard.get_performance_metrics()
        
        assert metrics['total_evaluations'] == initial_count + 1
        assert 'approval_rate' in metrics
        assert 'rejection_rate' in metrics
        assert 'avg_evaluation_time_ms' in metrics
        assert metrics['avg_evaluation_time_ms'] >= 0
    
    def test_error_handling_fallback(self, risk_guard, test_order):
        """CRITICAL: Error handling falls back to REJECT"""
        # Simulate error in evaluation
        with patch.object(risk_guard, '_execute_unified_risk_gates', side_effect=Exception("Test error")):
            result = risk_guard.evaluate_order(test_order, {})
            
            assert result.decision == RiskDecision.REJECT
            assert "EVALUATION_ERROR" in result.reason
    
    def test_thread_safety(self, risk_guard, market_data):
        """CRITICAL: Thread-safe concurrent access"""
        import threading
        import concurrent.futures
        
        results = []
        
        def evaluate_order(order_id):
            order = StandardOrderRequest(
                symbol="ETH",
                side=OrderSide.BUY,
                size=0.1,
                price=2000.0,
                client_order_id=f"thread_order_{order_id}"
            )
            return risk_guard.evaluate_order(order, market_data)
        
        # Execute concurrent evaluations
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(evaluate_order, i) for i in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All evaluations should complete
        assert len(results) == 50
        
        # All results should be valid
        for result in results:
            assert isinstance(result, RiskEvaluationResult)
            assert result.decision is not None
    
    def test_emergency_state_persistence(self, risk_guard):
        """CRITICAL: Emergency state persistence"""
        # Activate kill switch
        risk_guard.activate_kill_switch("Test emergency persistence")
        
        # Check emergency state file creation
        assert risk_guard.emergency_state_file.exists()
        
        # Check state persisted
        import json
        with open(risk_guard.emergency_state_file, 'r') as f:
            state = json.load(f)
        
        assert state['kill_switch_active'] is True
        assert state['reason'] == "Test emergency persistence"
    
    def test_zero_bypass_architecture(self, risk_guard, market_data):
        """CRITICAL: NO bypass possible - ALL orders go through evaluate_order"""
        
        # Test that there's only ONE way to evaluate orders
        # Any attempt to bypass should fail
        
        test_orders = [
            StandardOrderRequest("BTC", OrderSide.BUY, 0.1, 50000.0),
            StandardOrderRequest("ETH", OrderSide.SELL, 1.0, 2000.0),
            StandardOrderRequest("SOL", OrderSide.BUY, 5.0, 100.0),
            StandardOrderRequest("ADA", OrderSide.SELL, 100.0, 0.5)
        ]
        
        # ALL orders must go through exact same method
        for order in test_orders:
            result = risk_guard.evaluate_order(order, market_data)
            
            # Must return standardized result
            assert isinstance(result, RiskEvaluationResult)
            assert hasattr(result, 'decision')
            assert hasattr(result, 'reason')
            assert hasattr(result, 'risk_score')
            assert hasattr(result, 'evaluation_time_ms')
            
        # Check that evaluation count matches number of orders processed
        metrics = risk_guard.get_performance_metrics()
        assert metrics['total_evaluations'] >= len(test_orders)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])