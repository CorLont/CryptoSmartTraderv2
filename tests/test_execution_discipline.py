#!/usr/bin/env python3
"""
Critical Tests for Execution Discipline System
Ensures harde ExecutionPolicy enforcement vóór ELKE order
"""

import pytest
import time
from decimal import Decimal
from unittest.mock import Mock, patch

import sys
sys.path.append('.')

from src.cryptosmarttrader.execution.execution_discipline_system import (
    ExecutionDisciplineSystem,
    OrderRequest,
    OrderSide,
    TimeInForce,
    ExecutionDecision,
    MarketConditions
)


class TestExecutionDisciplineSystem:
    """Critical tests voor Execution Discipline enforcement"""
    
    @pytest.fixture
    def execution_system(self):
        """Fresh execution discipline system"""
        return ExecutionDisciplineSystem()
    
    @pytest.fixture
    def test_order(self):
        """Standard test order"""
        return OrderRequest(
            symbol="ETH",
            side=OrderSide.BUY,
            size=1.0,
            limit_price=2000.0,
            time_in_force=TimeInForce.POST_ONLY,
            max_slippage_bps=10.0
        )
    
    @pytest.fixture
    def good_market_conditions(self):
        """Good market conditions voor testing"""
        return MarketConditions(
            spread_bps=5.0,  # Tight spread
            bid_depth_usd=50000.0,  # Good depth
            ask_depth_usd=50000.0,
            volume_1m_usd=100000.0,  # Good volume
            last_price=2000.0,
            bid_price=1999.0,
            ask_price=2001.0,
            timestamp=time.time()
        )
    
    def test_idempotency_protection(self, execution_system, test_order):
        """CRITICAL: Idempotency protection tegen duplicate orders"""
        # First order should be approved
        decision1, reason1, details1 = execution_system.evaluate_order(test_order)
        
        # Same order (same client_order_id) should be rejected
        decision2, reason2, details2 = execution_system.evaluate_order(test_order)
        
        assert decision1 == ExecutionDecision.APPROVE
        assert decision2 == ExecutionDecision.REJECT
        assert "DUPLICATE_ORDER" in reason2
    
    def test_spread_gate_enforcement(self, execution_system, test_order):
        """CRITICAL: Spread gate moet wide spreads blokkeren"""
        # Wide spread market conditions
        wide_spread_conditions = MarketConditions(
            spread_bps=150.0,  # 1.5% spread - too wide
            bid_depth_usd=10000.0,
            ask_depth_usd=10000.0,
            volume_1m_usd=50000.0,
            last_price=2000.0,
            bid_price=1985.0,  # Wide spread
            ask_price=2015.0,
            timestamp=time.time()
        )
        
        decision, reason, details = execution_system.evaluate_order(test_order, wide_spread_conditions)
        
        assert decision == ExecutionDecision.REJECT
        assert "SPREAD_TOO_WIDE" in reason
    
    def test_liquidity_depth_gate(self, execution_system, test_order):
        """CRITICAL: Liquidity depth gate enforcement"""
        # Shallow market conditions
        shallow_conditions = MarketConditions(
            spread_bps=5.0,
            bid_depth_usd=500.0,  # Too shallow for 2000 USD order
            ask_depth_usd=500.0,
            volume_1m_usd=5000.0,
            last_price=2000.0,
            bid_price=1999.0,
            ask_price=2001.0,
            timestamp=time.time()
        )
        
        decision, reason, details = execution_system.evaluate_order(test_order, shallow_conditions)
        
        assert decision == ExecutionDecision.REJECT
        assert "INSUFFICIENT_DEPTH" in reason
    
    def test_volume_gate_enforcement(self, execution_system, test_order):
        """CRITICAL: Volume gate voor market activity validation"""
        # Low volume conditions
        low_volume_conditions = MarketConditions(
            spread_bps=5.0,
            bid_depth_usd=50000.0,
            ask_depth_usd=50000.0,
            volume_1m_usd=1000.0,  # Very low volume
            last_price=2000.0,
            bid_price=1999.0,
            ask_price=2001.0,
            timestamp=time.time()
        )
        
        decision, reason, details = execution_system.evaluate_order(test_order, low_volume_conditions)
        
        assert decision == ExecutionDecision.REJECT
        assert "INSUFFICIENT_VOLUME" in reason
    
    def test_slippage_budget_enforcement(self, execution_system, test_order):
        """CRITICAL: Slippage budget enforcement"""
        # Market with price movement that would exceed slippage budget
        test_order.max_slippage_bps = 5.0  # Very tight slippage budget
        
        high_impact_conditions = MarketConditions(
            spread_bps=8.0,  # Spread alone exceeds slippage budget
            bid_depth_usd=50000.0,
            ask_depth_usd=50000.0,
            volume_1m_usd=100000.0,
            last_price=2000.0,
            bid_price=1996.0,  # Wide spread
            ask_price=2004.0,
            timestamp=time.time()
        )
        
        decision, reason, details = execution_system.evaluate_order(test_order, high_impact_conditions)
        
        assert decision == ExecutionDecision.REJECT
        assert "SLIPPAGE_BUDGET_EXCEEDED" in reason
    
    def test_time_in_force_validation(self, execution_system, good_market_conditions):
        """CRITICAL: Time-in-Force validation"""
        # Test POST_ONLY validation
        post_only_order = OrderRequest(
            symbol="ETH",
            side=OrderSide.BUY,
            size=1.0,
            limit_price=2001.0,  # Would cross spread (take liquidity)
            time_in_force=TimeInForce.POST_ONLY
        )
        
        decision, reason, details = execution_system.evaluate_order(post_only_order, good_market_conditions)
        
        assert decision == ExecutionDecision.REJECT
        assert "POST_ONLY_WOULD_CROSS" in reason
    
    def test_price_validation_gate(self, execution_system, test_order, good_market_conditions):
        """CRITICAL: Price validation against market"""
        # Order with price too far from market
        test_order.limit_price = 1500.0  # 25% below market price
        
        decision, reason, details = execution_system.evaluate_order(test_order, good_market_conditions)
        
        assert decision == ExecutionDecision.REJECT
        assert "PRICE_TOO_AGGRESSIVE" in reason
    
    def test_approved_order_conditions(self, execution_system, test_order, good_market_conditions):
        """Test conditions waar order approved wordt"""
        decision, reason, details = execution_system.evaluate_order(test_order, good_market_conditions)
        
        assert decision == ExecutionDecision.APPROVE
        assert "APPROVED" in reason
        assert details is not None
        assert details.get('estimated_slippage_bps') is not None
    
    def test_client_order_id_generation(self, execution_system):
        """CRITICAL: Client Order ID generation voor idempotency"""
        order1 = OrderRequest(
            symbol="ETH",
            side=OrderSide.BUY,
            size=1.0,
            strategy_id="test_strategy"
        )
        
        order2 = OrderRequest(
            symbol="ETH",
            side=OrderSide.BUY,
            size=1.0,
            strategy_id="test_strategy"
        )
        
        # Same parameters in same minute should generate same ID
        assert order1.client_order_id == order2.client_order_id
        
        # Different parameters should generate different ID
        order3 = OrderRequest(
            symbol="BTC",  # Different symbol
            side=OrderSide.BUY,
            size=1.0,
            strategy_id="test_strategy"
        )
        
        assert order1.client_order_id != order3.client_order_id
    
    def test_execution_timing_tracking(self, execution_system, test_order, good_market_conditions):
        """Test execution timing tracking"""
        start_time = time.time()
        
        decision, reason, details = execution_system.evaluate_order(test_order, good_market_conditions)
        
        # Verify timing is tracked
        assert 'evaluation_time_ms' in details
        assert details['evaluation_time_ms'] > 0
        assert details['evaluation_time_ms'] < 1000  # Should be fast < 1 second
    
    def test_historical_decision_tracking(self, execution_system, test_order, good_market_conditions):
        """Test historical decision tracking"""
        initial_count = len(execution_system.decision_history)
        
        # Execute multiple orders
        for i in range(3):
            test_order.client_order_id = f"test_order_{i}"
            execution_system.evaluate_order(test_order, good_market_conditions)
        
        # Verify decisions are tracked
        assert len(execution_system.decision_history) == initial_count + 3
        
        # Verify decision details
        last_decision = execution_system.decision_history[-1]
        assert 'timestamp' in last_decision
        assert 'symbol' in last_decision
        assert 'decision' in last_decision
    
    def test_concurrent_order_evaluation(self, execution_system, good_market_conditions):
        """CRITICAL: Thread safety voor concurrent order evaluation"""
        import threading
        
        results = []
        errors = []
        
        def evaluate_order_thread(order_id):
            try:
                test_order = OrderRequest(
                    symbol="ETH",
                    side=OrderSide.BUY,
                    size=1.0,
                    client_order_id=f"concurrent_test_{order_id}"
                )
                decision, reason, details = execution_system.evaluate_order(test_order, good_market_conditions)
                results.append((decision, reason, details))
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads with different orders
        threads = []
        for i in range(10):
            thread = threading.Thread(target=evaluate_order_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify no errors and all processed
        assert len(errors) == 0
        assert len(results) == 10
        
        # All should be approved (different client_order_ids)
        for decision, reason, details in results:
            assert decision == ExecutionDecision.APPROVE


class TestMarketConditionsValidation:
    """Test market conditions validation logic"""
    
    def test_stale_market_data_rejection(self):
        """Test rejection of stale market data"""
        execution_system = ExecutionDisciplineSystem()
        
        test_order = OrderRequest(
            symbol="ETH",
            side=OrderSide.BUY,
            size=1.0
        )
        
        # Stale market conditions (older than 30 seconds)
        stale_conditions = MarketConditions(
            spread_bps=5.0,
            bid_depth_usd=50000.0,
            ask_depth_usd=50000.0,
            volume_1m_usd=100000.0,
            last_price=2000.0,
            bid_price=1999.0,
            ask_price=2001.0,
            timestamp=time.time() - 60  # 1 minute old
        )
        
        decision, reason, details = execution_system.evaluate_order(test_order, stale_conditions)
        
        assert decision == ExecutionDecision.REJECT
        assert "STALE_MARKET_DATA" in reason
    
    def test_missing_market_data_handling(self):
        """Test handling of missing market data"""
        execution_system = ExecutionDisciplineSystem()
        
        test_order = OrderRequest(
            symbol="ETH",
            side=OrderSide.BUY,
            size=1.0
        )
        
        # No market conditions provided
        decision, reason, details = execution_system.evaluate_order(test_order, None)
        
        assert decision == ExecutionDecision.DEFER
        assert "NO_MARKET_DATA" in reason


class TestExecutionPolicyConfiguration:
    """Test execution policy configuration"""
    
    def test_configurable_thresholds(self):
        """Test configurable policy thresholds"""
        custom_config = {
            'max_spread_bps': 200.0,  # Allow wider spreads
            'min_depth_ratio': 2.0,   # Require 2x order size in depth
            'min_volume_ratio': 5.0,  # Require 5x order size in volume
            'max_slippage_bps': 50.0, # Allow higher slippage
            'max_price_deviation': 0.1  # 10% max price deviation
        }
        
        execution_system = ExecutionDisciplineSystem(config=custom_config)
        
        # Verify custom thresholds are applied
        assert execution_system.config['max_spread_bps'] == 200.0
        assert execution_system.config['min_depth_ratio'] == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])